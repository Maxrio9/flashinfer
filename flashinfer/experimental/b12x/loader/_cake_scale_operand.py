"""Typed immutable producer scale operands for prepared SM103 Cake pipeline calls.

``CsfScaleSpec`` names the producer encoding, the Cake scale layout, the
numerical recipe and the bank geometry. ``CsfScaleBuffers`` are the final
numeric owners: FC1 and FC2 tiles and row bases, and the shared raw pool.
``PreparedCsfScales`` binds filled owners to a specification; it is created by
``count_csf_scales(...).fill(out)`` and is never modified afterwards.
"""

from dataclasses import dataclass

ENCODING = "csf_2bit_tile144_rowbase_raw512"
LAYOUT = "cake_n128_k128_ue8m0"
MXFP4_UE8M0_A8 = "mxfp4_ue8m0_a8"
RAW_FLAG = 1 << 31


@dataclass(frozen=True)
class CsfScaleSpec:
    """Producer format and real bank geometry, independent of the selected schedule."""

    experts: int
    hidden: int
    intermediate: int
    encoding: str = ENCODING
    layout: str = LAYOUT
    numerical_recipe: str = MXFP4_UE8M0_A8

    def __post_init__(self):
        if (
            type(self.experts) is not int
            or not 1 <= self.experts <= 384
            or any(
                type(value) is not int or value <= 0 or value % 256
                for value in (self.hidden, self.intermediate)
            )
        ):
            raise ValueError(
                "Cake scale geometry requires 1..384 experts and H/I divisible by 256"
            )
        if self.numerical_recipe != MXFP4_UE8M0_A8:
            raise ValueError(
                "Cake compressed scales require the MXFP4/UE8M0 A8 numerical recipe"
            )
        if self.encoding != ENCODING or self.layout != LAYOUT:
            raise ValueError("unsupported Cake compressed scale encoding or layout")

    def identity(self):
        return dict(
            encoding=self.encoding,
            layout=self.layout,
            numerical_recipe=self.numerical_recipe,
            experts=self.experts,
            hidden=self.hidden,
            intermediate=self.intermediate,
        )

    def validate_args(self, surface, args):
        if (
            surface != "pipeline"
            or args["routed_weight_dtype"] != "fp4"
            or args["hidden"] != self.hidden
            or args["intermediate"] != self.intermediate
            or not self.experts <= args["num_experts"] <= 384
        ):
            raise ValueError(
                "compressed scale geometry differs from the selected FP4 pipeline"
            )

    @property
    def shapes(self):
        e, h, i = self.experts, self.hidden, self.intermediate
        return (
            (e, 2 * i // 128, h // 128, 144),
            (e, 2 * i),
            (e, h // 128, i // 128, 144),
            (e, h),
        )


@dataclass(frozen=True)
class CsfScaleBuffers:
    """Caller-allocated final numeric storage; no weight or source owner is copied."""

    l1_tiles: object
    l1_bases: object
    l2_tiles: object
    l2_bases: object
    raw_pool: object

    @property
    def owners(self):
        return self.l1_tiles, self.l1_bases, self.l2_tiles, self.l2_bases, self.raw_pool

    def validate(self, spec, raw_tiles, *, require_cuda=False):
        import torch

        if type(raw_tiles) is not int or not 0 <= raw_tiles < RAW_FLAG:
            raise ValueError("raw tile extent must fit the low 31-bit header field")
        if not all(isinstance(value, torch.Tensor) for value in self.owners):
            raise TypeError("scale buffers require numeric Tensor owners")
        device = self.l1_tiles.device
        if device.type not in ("cpu", "cuda") or (
            require_cuda and device.type != "cuda"
        ):
            raise ValueError(
                "prepared pipeline scale binding requires CUDA numeric owners"
            )
        extents = []
        for tensor, shape, dtype in zip(
            self.owners,
            (*spec.shapes, (max(raw_tiles, 1), 128)),
            (torch.uint8, torch.uint8, torch.uint8, torch.uint8, torch.uint32),
            strict=True,
        ):
            if (
                tensor.device != device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "scale buffers require exact contiguous aligned shapes/dtypes on one device"
                )
            extents.append(
                (
                    tensor.data_ptr(),
                    tensor.data_ptr() + tensor.numel() * tensor.element_size(),
                )
            )
        for index, (first, end) in enumerate(extents):
            if any(
                max(first, other) < min(end, stop) for other, stop in extents[:index]
            ):
                raise ValueError("distinct scale numeric owners must not overlap")
        return self


@dataclass(frozen=True, init=False)
class PreparedCsfScales:
    """Validated producer bytes, held immutable until every bound plan is released.

    Construct with count_csf_scales(...).fill(out). Mutating its numeric owners
    after preparation invalidates every bound plan; update_scales is unsupported.
    """

    spec: CsfScaleSpec
    buffers: CsfScaleBuffers
    raw_tiles: int

    def __init__(self):
        raise TypeError(
            "construct prepared scales with count_csf_scales(...).fill(out)"
        )

    @classmethod
    def _prepared(cls, spec, buffers, raw_tiles):
        buffers.validate(spec, raw_tiles)
        value = object.__new__(cls)
        for name, item in (
            ("spec", spec),
            ("buffers", buffers),
            ("raw_tiles", raw_tiles),
        ):
            object.__setattr__(value, name, item)
        return value

    @property
    def owners(self):
        return self.buffers.owners

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in self.owners)

    def stage_bindings(self, bindings):
        """Borrow packed B1/B2; add only compressed scale operands and raw maps."""
        self.buffers.validate(self.spec, self.raw_tiles)
        forbidden = {
            "SFB1",
            "SFB2",
            "csf_l1_tiles",
            "csf_l1_bases",
            "csf_l2_tiles",
            "csf_l2_bases",
        }
        if forbidden.intersection(bindings):
            raise ValueError(
                "pass compressed scales through weight_scales; omit raw scale bindings"
            )
        spec = self.spec
        for name, count in (
            ("B1", spec.experts * 2 * spec.intermediate * spec.hidden // 2),
            ("B2", spec.experts * spec.hidden * spec.intermediate // 2),
        ):
            tensor = bindings[name]
            import torch

            if (
                tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
                or tensor.numel() != count
                or tensor.device != self.buffers.l1_tiles.device
            ):
                raise ValueError(
                    "existing packed weight bank differs from compressed scale geometry"
                )
        for value in bindings.values():
            if (
                not isinstance(value, torch.Tensor)
                or value.device != self.buffers.l1_tiles.device
            ):
                continue
            begin, end = (
                value.data_ptr(),
                value.data_ptr() + value.numel() * value.element_size(),
            )
            for immutable in self.owners:
                first, last = (
                    immutable.data_ptr(),
                    immutable.data_ptr() + immutable.numel() * immutable.element_size(),
                )
                if max(first, begin) < min(last, end):
                    raise ValueError(
                        "immutable scale owners must not overlap weights or caller workspace"
                    )
        result = dict(bindings)
        result.update(
            csf_l1_tiles=self.buffers.l1_tiles,
            csf_l1_bases=self.buffers.l1_bases,
            csf_l2_tiles=self.buffers.l2_tiles,
            csf_l2_bases=self.buffers.l2_bases,
            SFB1=self.buffers.raw_pool,
            SFB2=self.buffers.raw_pool,
        )
        return result


def scale_spec(value):
    if isinstance(value, PreparedCsfScales):
        return value.spec
    if isinstance(value, CsfScaleSpec):
        return value
    raise TypeError("weight_scales requires a CsfScaleSpec or PreparedCsfScales")
