"""Cake producer scale tiles reproduce every UE8M0 byte of their CSF sources."""

import numpy as np
import pytest
import torch

from b12x.loader import (
    CsfScaleBuffers,
    CsfScaleSpec,
    PreparedCsfScales,
    count_csf_scales,
)
from b12x.loader._cake_csf import TILE_BYTES, cake_rows, encode_producer
from b12x.loader._cake_scale_operand import RAW_FLAG
from b12x.loader._csf_source import stage_csf_sources


def decode_tiles(tiles, bases, raw_pool, columns, raw_base=0):
    """Scalar tile traversal: selectors plus row bases, then records or raw bytes."""
    rows = len(bases)
    groups = columns // 4
    result = np.empty((rows, columns), dtype=np.uint8)
    for tile_index, tile in enumerate(tiles.reshape(-1, TILE_BYTES)):
        block, group = divmod(tile_index, groups)
        header = int(tile[128:132].view("<u4")[0])
        values = np.empty((128, 4), dtype=np.uint8)
        for row in range(128):
            for column in range(4):
                offset = (int(tile[row]) >> (2 * column)) & 3
                values[row, column] = (int(bases[block * 128 + row]) + offset) & 255
        if header & RAW_FLAG:
            values = raw_pool[(header & ~RAW_FLAG) - raw_base]
        else:
            for record in tile[132:].view("<u4")[:header]:
                record = int(record)
                values[record & 127, (record >> 8) & 3] = record >> 16
        result[block * 128 : (block + 1) * 128, group * 4 : (group + 1) * 4] = values
    return result


def _plane(rows, columns, seed, *, dense_rows=()):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 253, size=(rows, 1))
    values = (base + rng.integers(0, 4, size=(rows, columns))).astype(np.uint8)
    sparse = rng.random((rows, columns)) < 0.004
    values[sparse] = rng.integers(0, 256, size=int(sparse.sum()), dtype=np.uint8)
    for row in dense_rows:
        values[row] = np.arange(columns, dtype=np.uint8) * 37 + row
    return values


@pytest.mark.parametrize("rows,columns", [(128, 8), (512, 8), (4608, 160), (5120, 72)])
def test_producer_tiles_round_trip_palette_record_and_raw_bytes(rows, columns):
    values = _plane(rows, columns, rows + columns, dense_rows=range(0, 128, 9))
    values[3, :4] = (255, 0, 254, 1)
    tiles, bases, raw, stats = encode_producer(values, raw_base=17)
    assert (
        tiles.shape == (rows // 128 * columns // 4, TILE_BYTES)
        and bases.dtype == np.uint8
    )
    assert int(bases.max()) <= 252
    assert stats["raw_tiles"] == len(raw) > 0
    pool = np.concatenate((np.zeros((17, 128, 4), dtype=np.uint8), raw))
    np.testing.assert_array_equal(decode_tiles(tiles, bases, pool, columns), values)


def test_every_byte_value_survives_in_palette_record_and_raw_tiles():
    all_bytes = np.tile(np.arange(256, dtype=np.uint8), 4).reshape(128, 8)
    single = np.full((128, 8), 100, dtype=np.uint8)
    single[0, 0] = 255
    for values in (all_bytes, single, np.full((128, 8), 255, dtype=np.uint8)):
        tiles, bases, raw, stats = encode_producer(values)
        np.testing.assert_array_equal(decode_tiles(tiles, bases, raw, 8), values)
    assert encode_producer(single)[3]["record_exceptions"] == 1
    # Each row base covers its first four columns; the second column group falls back to raw.
    assert encode_producer(all_bytes)[3]["raw_tiles"] == 1


def _source(planes, *, join):
    """Slab CSF planes (row base, one code bit, verbatim exception records)."""
    encoded = []
    for values in planes:
        rows, columns = values.shape
        bases = np.minimum(values.min(axis=1), 254).astype(np.uint8)
        codes = values == bases[:, None].astype(np.int16) + 1
        ordinary = bases[:, None].astype(np.int16) + codes
        positions = np.flatnonzero(values.reshape(-1) != ordinary.reshape(-1)).astype(
            np.uint32
        )
        records = positions | (values.reshape(-1)[positions].astype(np.uint32) << 24)
        packed = np.packbits(codes, axis=1, bitorder="little")
        fixed = np.concatenate(
            (bases.reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), axis=1
        )
        encoded.append((torch.from_numpy(fixed), torch.from_numpy(records)))
    return stage_csf_sources(
        encoded,
        rows=rows,
        columns=columns,
        quantization="mxfp4",
        device="cpu",
        encoding="csf",
        join=join,
    )


def _bank(experts=3, hidden=256, intermediate=256):
    up_gate = [
        _plane(intermediate, hidden // 32, 7 * e + p, dense_rows=(5 * e,))
        for e in range(experts)
        for p in range(2)
    ]
    down = [
        _plane(hidden, intermediate // 32, 101 + e, dense_rows=(e,))
        for e in range(experts)
    ]
    return up_gate, down, _source(up_gate, join=2), _source(down, join=1)


def _allocate(preparation):
    kinds = {"uint8": torch.uint8, "uint32": torch.uint32}
    return CsfScaleBuffers(
        **{
            item["name"]: torch.empty(item["shape"], dtype=kinds[item["dtype"]])
            for item in preparation.numeric_layout
        }
    )


def test_count_and_fill_reproduce_each_expert_in_cake_row_order():
    up_gate, down, fc1, fc2 = _bank()
    preparation = count_csf_scales(fc1, fc2)
    assert preparation.spec == CsfScaleSpec(3, 256, 256)
    prepared = preparation.fill(_allocate(preparation))
    assert (
        isinstance(prepared, PreparedCsfScales)
        and prepared.raw_tiles == preparation.raw_tiles > 0
    )
    assert prepared.nbytes == preparation.nbytes
    buffers = prepared.buffers
    pool = buffers.raw_pool.numpy().view(np.uint8).reshape(-1, 128, 4)
    for expert in range(3):
        for tiles, bases, logical, fc1_rows in (
            (
                buffers.l1_tiles,
                buffers.l1_bases,
                np.concatenate(up_gate[2 * expert : 2 * expert + 2]),
                True,
            ),
            (buffers.l2_tiles, buffers.l2_bases, down[expert], False),
        ):
            decoded = decode_tiles(
                tiles[expert].numpy(), bases[expert].numpy(), pool, logical.shape[1]
            )
            np.testing.assert_array_equal(decoded, cake_rows(logical, fc1_rows))


def test_prepared_scales_are_immutable_typed_operands():
    _, _, fc1, fc2 = _bank(experts=1)
    preparation = count_csf_scales(fc1, fc2)
    buffers = _allocate(preparation)
    with pytest.raises(TypeError, match="count_csf_scales"):
        PreparedCsfScales()
    with pytest.raises(ValueError, match="exact contiguous aligned shapes"):
        preparation.fill(
            CsfScaleBuffers(
                *buffers.owners[:4], torch.empty((1, 64), dtype=torch.uint32)
            )
        )
    prepared = preparation.fill(buffers)
    spec = prepared.spec
    values13 = torch.empty(
        spec.experts * 2 * spec.intermediate * spec.hidden // 2, dtype=torch.uint8
    )
    values2 = torch.empty(
        spec.experts * spec.hidden * spec.intermediate // 2, dtype=torch.uint8
    )
    bound = prepared.stage_bindings(dict(B1=values13, B2=values2))
    assert bound["SFB1"] is buffers.raw_pool and bound["SFB2"] is buffers.raw_pool
    assert (
        bound["csf_l1_tiles"] is buffers.l1_tiles
        and bound["csf_l2_bases"] is buffers.l2_bases
    )
    with pytest.raises(ValueError, match="omit raw scale bindings"):
        prepared.stage_bindings(dict(B1=values13, B2=values2, SFB1=buffers.raw_pool))
    with pytest.raises(ValueError, match="packed weight bank"):
        prepared.stage_bindings(dict(B1=values13[:-2], B2=values2))
    with pytest.raises(ValueError, match="must not overlap"):
        prepared.stage_bindings(
            dict(B1=values13, B2=values2, scratch=buffers.l1_bases.view(-1))
        )
    with pytest.raises(ValueError, match="1..384 experts"):
        CsfScaleSpec(0, 256, 256)
