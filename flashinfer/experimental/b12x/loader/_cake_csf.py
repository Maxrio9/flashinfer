"""Lossless Cake producer encoding of MXFP4 scale planes, counted and filled on the CPU.

The producer format is what SM103 Cake MegaMoE pipelines consume on chip with
``weight_scales``. Each scale plane is in Cake's physical row order and is
divided into tiles of 128 rows by four scale columns (512 bytes, one Cake
scale word per row). Every row has one base byte, chosen in ``0..252`` so
that ``base + 3`` cannot carry into a neighbouring byte. A 144-byte tile holds
one selector byte per row (four 2-bit offsets from the row base), a 32-bit
header and three 32-bit exception records. The header is the number of
exception records, or ``RAW_FLAG | index`` for a tile with more than three
exceptions, whose 512 original bytes are stored verbatim in a raw pool. A
record packs the row (bits 0-6), the column within the group (bits 8-9) and
the exact byte (bits 16-23). Encoding and decoding use integer operations
only, and every byte value is exact.

``count_csf_scales`` reads CPU ``CsfSource`` planes, in either checkpoint CSF
encoding and with any valid logical view, and returns the exact numeric layout.
Callers allocate it once and ``fill`` it; the result owns no source plane,
weight or expanded scale copy.
"""

from dataclasses import dataclass
import numpy as np
from . import _csf_blocks as _blocks
from ._csf_source import CsfSource
from ._cake_scale_operand import (
    CsfScaleSpec,
    CsfScaleBuffers,
    PreparedCsfScales,
    RAW_FLAG,
)

TILE_BYTES = 144


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cake_rows(values, fc1):
    """Reorder one logical [rows, columns] plane into Cake's physical row order.

    Each 128-row group stores logical row ``p // 4 + 32 * (p % 4)`` at physical
    row ``p``. FC1 sources stack up then gate; Cake groups eight gate rows,
    then the eight up rows with the same channels.
    """
    rows, columns = values.shape
    require(
        rows % 128 == 0 and columns % 4 == 0,
        "Cake scale planes require groups of 128 rows and four columns",
    )
    physical = np.arange(rows)
    logical = physical // 128 * 128 + physical % 4 * 32 + physical % 128 // 4
    if fc1:
        logical = (
            logical // 16 * 8 + logical % 8 + np.where(logical % 16 < 8, rows // 2, 0)
        )
    require(len(np.unique(logical)) == rows, "Cake row permutation is not bijective")
    return np.ascontiguousarray(values[logical])


def encode_producer(values, raw_base=0):
    """Encode one physical plane as (tiles, bases, raw tiles, statistics).

    ``raw_base`` is the raw-pool index of this plane's first raw tile. The
    tile, header and record layouts are described in the module docstring.
    """
    rows, columns = values.shape
    require(rows % 128 == 0 and columns % 4 == 0, "unsupported producer geometry")
    # Starts 0..252 ensure byte-wise base+offset never carries into a neighbour.
    labels = np.arange(rows, dtype=np.int32)[:, None] * 256 + values.astype(np.int32)
    histogram = np.bincount(labels.reshape(-1), minlength=rows * 256).reshape(rows, 256)
    bases = (
        (
            histogram[:, :253]
            + histogram[:, 1:254]
            + histogram[:, 2:255]
            + histogram[:, 3:256]
        )
        .argmax(axis=1)
        .astype(np.uint8)
    )
    raw_values = values.reshape(rows // 128, 128, columns // 4, 4).transpose(0, 2, 1, 3)
    offsets = raw_values.astype(np.int16) - bases.reshape(
        rows // 128, 1, 128, 1
    ).astype(np.int16)
    exceptions = (offsets < 0) | (offsets > 3)
    fields = np.where(exceptions, 0, offsets).astype(np.uint8)
    selectors = (
        fields[..., 0]
        | (fields[..., 1] << 2)
        | (fields[..., 2] << 4)
        | (fields[..., 3] << 6)
    )
    tile_count = selectors.shape[0] * selectors.shape[1]
    tiles = np.zeros((tile_count, TILE_BYTES), dtype=np.uint8)
    tiles[:, :128] = selectors.reshape(tile_count, 128)
    counts = exceptions.sum(axis=(2, 3)).reshape(-1)
    heavy = counts > 3
    raw = np.ascontiguousarray(raw_values.reshape(tile_count, 128, 4)[heavy])
    require(
        raw_base + len(raw) < RAW_FLAG, "raw fallback index exceeds the 31-bit header"
    )
    headers = counts.astype(np.uint32)
    headers[heavy] = RAW_FLAG | (raw_base + np.arange(len(raw), dtype=np.uint32))
    tiles[:, 128:132] = headers.astype("<u4").view(np.uint8).reshape(tile_count, 4)
    tile_ids, coordinates = np.nonzero(
        exceptions.reshape(tile_count, 512) & ~heavy[:, None]
    )
    if len(tile_ids):
        _, first, per_tile = np.unique(tile_ids, return_index=True, return_counts=True)
        ranks = np.arange(len(tile_ids)) - np.repeat(first, per_tile)
        require(
            int(ranks.max()) < 3, "nonraw producer tile exceeds three exception records"
        )
        exact = raw_values.reshape(tile_count, 512)[tile_ids, coordinates].astype(
            np.uint32
        )
        records = (
            (coordinates // 4).astype(np.uint32)
            | ((coordinates % 4).astype(np.uint32) << 8)
            | (exact << 16)
        )
        packed_records = np.zeros((tile_count, 3), dtype="<u4")
        packed_records[tile_ids, ranks] = records
        tiles[:, 132:] = packed_records.view(np.uint8).reshape(tile_count, 12)
    stats = dict(
        tiles=tile_count,
        raw_tiles=len(raw),
        exception_tiles=int(np.count_nonzero((counts > 0) & ~heavy)),
        record_exceptions=int(counts[~heavy].sum()),
        producer_tile_bytes=tiles.nbytes,
        producer_base_bytes=bases.nbytes,
        producer_raw_bytes=raw.nbytes,
        decoded_bytes=values.nbytes,
        byte255=int(np.count_nonzero(values == 255)),
    )
    return tiles, bases, raw, stats


class LogicalCsfReader:
    """Validate CPU checkpoint components and gather one logical expert view."""

    def __init__(self, source):
        source.validate()
        if source.device.type != "cpu":
            raise ValueError(
                "scale-only preparation requires CPU CsfSource components; CUDA sources are not staged implicitly"
            )
        self.blocks = _blocks
        self.source = source
        self.parts = tuple(tensor.detach().numpy() for tensor in source.components)
        self.starts = source.starts.detach().numpy()
        records = self.parts[-1].size
        if (
            self.starts[0] != 0
            or self.starts[-1] != records
            or np.any(self.starts < 0)
            or np.any(self.starts[1:] < self.starts[:-1])
        ):
            raise ValueError("CSF starts must cover one consecutive exception arena")

    def plane(self, segment):
        """Return one existing FI CPU block plane, validating its complete payload."""
        s = self.source
        if type(segment) is not int or not 0 <= segment < s.segments:
            raise ValueError("CSF segment index is outside the source")
        start, end = map(int, self.starts[segment : segment + 2])
        if s.encoding == "csf":
            fixed = self.parts[0][segment]
            bits = 4 if s.quantization == "nvfp4" else 1
            tail = s.source_columns * bits % 8
            # The GPU source validator/reader ignore unused final code bits.
            # export_csf_blocks requires canonical zeros there. Clear only those
            # ignored bits in a bounded temporary, leaving source bytes intact.
            if tail:
                fixed = fixed.copy()
                codes = fixed[:, 16:].reshape(s.source_rows, -1)
                codes[:, -1] &= np.uint8((1 << tail) - 1)
                fixed[:, 16:] = codes.reshape(s.source_rows // 16, -1)
            plane = self.blocks.export_csf_blocks(
                fixed,
                self.parts[-1][start:end],
                rows=s.source_rows,
                columns=s.source_columns,
                quantization=s.quantization,
            )
        else:
            plane = self.blocks.CsfBlockPlane(
                s.source_rows,
                s.source_columns,
                s.quantization,
                *(part[segment] for part in self.parts[:-1]),
                self.parts[-1][start:end],
            )
        plane.validate()
        return plane

    def validate_payloads(self):
        """Validate all segments with bounded per-segment temporary storage."""
        for segment in range(self.source.segments):
            self.plane(segment)
        return self

    def expert(self, expert):
        """Read the exact joined/cropped/padded uint8 grid for one expert."""
        s = self.source
        if type(expert) is not int or not 0 <= expert < s.num_experts:
            raise ValueError("CSF expert index is outside the source")
        r0, r1 = s.row_range or (0, s.source_rows)
        c0, c1 = s.column_range or (0, s.source_columns)
        per_segment = s.rows // s.join
        result = np.full((s.rows, s.columns), s.padding_base, dtype=np.uint8)
        for section in range(s.join):
            plane = self.plane(expert * s.join + section)
            bases, fields, positions, replacements = self.blocks.logical_csf_components(
                plane
            )
            values = (
                bases[r0:r1, None].astype(np.uint16) + fields[r0:r1, c0:c1]
            ).astype(np.uint8)
            rr, cc = positions // s.source_columns, positions % s.source_columns
            selected = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
            values[rr[selected] - r0, cc[selected] - c0] = replacements[selected]
            first = section * per_segment
            result[first : first + r1 - r0, : c1 - c0] = values
        return result


@dataclass(frozen=True)
class CsfScalePreparation:
    """Exact native sizes and CPU source views for one bounded fill pass."""

    spec: CsfScaleSpec
    fc1: object
    fc2: object
    raw_counts: tuple

    @property
    def raw_tiles(self):
        return sum(sum(pair) for pair in self.raw_counts)

    @property
    def numeric_layout(self):
        names = ("l1_tiles", "l1_bases", "l2_tiles", "l2_bases", "raw_pool")
        return tuple(
            dict(name=name, shape=shape, dtype=dtype, nbytes=int(np.prod(shape)) * size)
            for name, shape, dtype, size in zip(
                names,
                (*self.spec.shapes, (max(self.raw_tiles, 1), 128)),
                ("uint8", "uint8", "uint8", "uint8", "uint32"),
                (1, 1, 1, 1, 4),
                strict=True,
            )
        )

    @property
    def nbytes(self):
        return sum(item["nbytes"] for item in self.numeric_layout)

    def fill(self, out):
        """Fill fresh unbound, source-disjoint numeric owners.

        No prepared plan may use out during fill. A failed fill returns no
        operand; its unbound outputs may be partially written and must not be
        bound. CPU checkpoint sources must remain immutable from count to fill.
        Packed weights are never accepted or transformed here.
        """
        import torch

        if not isinstance(out, CsfScaleBuffers):
            raise TypeError("out requires caller-allocated CsfScaleBuffers")
        out.validate(self.spec, self.raw_tiles)
        for output in out.owners:
            first, last = (
                output.data_ptr(),
                output.data_ptr() + output.numel() * output.element_size(),
            )
            for source in (self.fc1, self.fc2):
                for tensor in (*source.components, source.starts):
                    if tensor.device != output.device:
                        continue
                    begin, end = (
                        tensor.data_ptr(),
                        tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                    )
                    if max(first, begin) < min(last, end):
                        raise ValueError(
                            "output scale owners must not overlap CPU checkpoint components or starts"
                        )
        cursor = 0
        readers = (LogicalCsfReader(self.fc1), LogicalCsfReader(self.fc2))
        for reader in readers:
            reader.validate_payloads()
        for expert, expected_pair in enumerate(self.raw_counts):
            for matrix, (reader, expected_raw) in enumerate(
                zip(readers, expected_pair, strict=True)
            ):
                if not 0 <= cursor < RAW_FLAG or cursor + expected_raw >= RAW_FLAG:
                    raise ValueError("raw indices exceed the low 31-bit header field")
                physical = cake_rows(reader.expert(expert), matrix == 0)
                tiles, bases, raw, _ = encode_producer(physical, raw_base=cursor)
                if len(raw) != expected_raw:
                    raise ValueError(
                        "immutable scale source changed between count and fill"
                    )
                shape = self.spec.shapes[matrix * 2]
                target_tiles = (out.l1_tiles, out.l2_tiles)[matrix][expert]
                target_bases = (out.l1_bases, out.l2_bases)[matrix][expert]
                target_tiles.copy_(torch.from_numpy(tiles.reshape(shape[1:])))
                target_bases.copy_(torch.from_numpy(bases))
                if len(raw):
                    out.raw_pool[cursor : cursor + len(raw)].copy_(
                        torch.from_numpy(raw.view("<u4").reshape(-1, 128))
                    )
                cursor += len(raw)
        if cursor != self.raw_tiles:
            raise ValueError("encoded raw extent differs from counted layout")
        if cursor == 0:
            out.raw_pool.zero_()  # Declared sentinel, never a logical raw index.
        return PreparedCsfScales._prepared(self.spec, out, cursor)


def count_csf_scales(fc1, fc2):
    """Count lossless Cake producer bytes from CPU logical CsfSource operands.

    FC1 joins up then gate; FC2 is unjoined. Inputs can use either checkpoint
    encoding and valid cropped/padded views. Both final scale rectangles must
    match complete H/I divisible by 256 Cake destinations. No CUDA source is
    copied implicitly and no weight tensor is accepted by this API.
    """
    for source in (fc1, fc2):
        if not isinstance(source, CsfSource):
            raise TypeError("scale-only preparation requires CsfSource operands")
        source.validate()
        if source.device.type != "cpu":
            raise ValueError(
                "scale-only preparation requires CPU CsfSource components; CUDA sources are not staged implicitly"
            )
    if (
        fc1.quantization != "mxfp4"
        or fc2.quantization != "mxfp4"
        or fc1.join != 2
        or fc2.join != 1
    ):
        raise ValueError(
            "Cake UE8M0 scales require joined MXFP4 up/gate and unjoined down sources"
        )
    spec = CsfScaleSpec(fc1.num_experts, fc2.rows, fc1.rows // 2)
    if (
        fc2.num_experts != spec.experts
        or fc1.columns != spec.hidden // 32
        or fc2.columns != spec.intermediate // 32
    ):
        raise ValueError("logical scale sources disagree on expert/H/I geometry")
    readers = (LogicalCsfReader(fc1), LogicalCsfReader(fc2))
    for reader in readers:
        reader.validate_payloads()
    counts = []
    total = 0
    for expert in range(spec.experts):
        pair = []
        for matrix, reader in enumerate(readers):
            _, _, raw, _ = encode_producer(
                cake_rows(reader.expert(expert), matrix == 0)
            )
            pair.append(len(raw))
            total += len(raw)
            if total >= RAW_FLAG:
                raise ValueError("raw indices exceed the low 31-bit header field")
        counts.append(tuple(pair))
    return CsfScalePreparation(spec, fc1, fc2, tuple(counts))
