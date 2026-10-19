"""Integer-only UE8M0 preparation; includes every byte, without FP arithmetic."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8 as RECIPE
from b12x.moe.fused_moe import _impl
from b12x.moe.fused_moe._tuning import MoeDecodeConfig


def _grid(rows, columns, device="cpu"):
    return (
        torch.arange(rows * columns, dtype=torch.int64, device=device)
        .remainder(256)
        .to(torch.uint8)
        .reshape(1, rows, columns)
    )


def _unpack(sfb, rows, columns):
    return (
        sfb.view(torch.uint8)
        .reshape(1, rows // 256, columns // 4, 32, 8, 4)
        .permute(0, 1, 3, 4, 2, 5)
        .reshape(1, rows, columns)
    )


@pytest.mark.parametrize("recipe", (None, RECIPE))
def test_cpu_integer_repack_preserves_all_bytes_only_for_explicit_recipe(recipe):
    source = _grid(256, 8)
    expected = source.clone() if recipe else source.clamp(max=247)
    packed = _impl._e8m0_scale_to_w4a8_sfb_inplace(
        source.clone(), weight_E=1, rows=256, k_dim=256, numerical_recipe=recipe
    )
    assert torch.equal(_unpack(packed, 256, 8), expected)


def test_weight_plan_and_execution_cannot_mix_scale_recipes():
    plan = _impl.plan_b12x_fp4_moe_weights(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=1,
        hidden_size=256,
        intermediate_size=128,
        numerical_recipe=RECIPE,
    )
    kwargs = dict(
        max_tokens=17,
        num_topk=3,
        route_num_experts=384,
        device="cpu",
        weight_plan=plan,
        quant_mode="w4a8_mx",
        deterministic_output=True,
        swiglu_limit=10.0,
        decode_config=MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=128,
            dynamic_route_mode="grouped",
        ),
    )
    caps = _impl.TPMoEScratchCaps(**kwargs, numerical_recipe=RECIPE)
    assert caps.weight_plan.numerical_recipe == caps.numerical_recipe
    with pytest.raises(ValueError, match="match the prepared weight plan"):
        _impl.TPMoEScratchCaps(**kwargs)
    kwargs["weight_plan"] = replace(plan, numerical_recipe=None)
    with pytest.raises(ValueError, match="match the prepared weight plan"):
        _impl.TPMoEScratchCaps(**kwargs, numerical_recipe=RECIPE)


@pytest.mark.parametrize(
    "changes",
    (
        {"quant_modes": "w4a16"},
        {"source_format": "modelopt_nvfp4"},
        {"params_dtype": torch.float16},
        {"activation": "squared_relu"},
        {"hidden_size": 128},
        {"intermediate_size": 192},
    ),
)
def test_weight_recipe_rejects_unsupported_representation(changes):
    kwargs = dict(
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=1,
        hidden_size=256,
        intermediate_size=128,
        numerical_recipe=RECIPE,
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        _impl.plan_b12x_fp4_moe_weights(**kwargs)


def _source(scales, join):
    """Valid row-base/one-bit streams, with arbitrary bytes as U24 exceptions."""
    from b12x.loader._csf_source import CsfSource

    _, rows, columns = scales.shape
    source_rows = rows // join
    fixed = torch.zeros(
        (join, source_rows // 16, 16 * (1 + (columns + 7) // 8)), dtype=torch.uint8
    )
    starts, records = [0], []
    for segment in scales.cpu().reshape(join, source_rows, columns):
        for position, value in enumerate(segment.flatten().tolist()):
            if value:
                records.append((value << 24) | position)
        starts.append(len(records))
    result = CsfSource(
        (
            fixed.to(scales.device),
            torch.tensor(records, dtype=torch.uint32, device=scales.device),
        ),
        torch.tensor(starts, dtype=torch.int64, device=scales.device),
        source_rows,
        columns,
        "mxfp4",
        join,
        "csf",
        join=join,
    )
    result.validate()
    return result


@pytest.mark.parametrize("compressed", (False, True))
def test_sm120_raw_and_compressed_preparation_preserve_all_256_bytes(compressed):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    s13, s2 = _grid(256, 8, "cuda"), _grid(256, 4, "cuda")
    expected13, expected2 = s13.clone(), s2.clone()
    w13 = torch.zeros((1, 256, 128), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((1, 256, 64), dtype=torch.uint8, device="cuda")
    plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=10.0
        ),
        geometry=moe.MoEGeometry(num_experts=1, hidden_size=256, intermediate_size=128),
        numerical_recipe=RECIPE,
    )
    if compressed:
        weights = moe.Mxfp4CsfWeights(
            w13,
            w2,
            _source(s13, 2),
            _source(s2, 1),
            torch.empty_like(s13),
            torch.empty_like(s2),
        )
    else:
        unit = torch.ones(1, dtype=torch.float32, device="cuda")
        weights = moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=s13,
            w2_block_scales=s2,
            w13_global_scales=unit,
            w2_global_scales=unit,
        )
    experts = moe.prepare_weights(plan=plan, weights=weights)
    assert (
        experts.plan.numerical_recipe == experts._impl.plan.numerical_recipe == RECIPE
    )

    def check():
        assert torch.equal(_unpack(experts._impl.w1_blockscale, 256, 8), expected13)
        assert torch.equal(_unpack(experts._impl.w2_blockscale, 256, 4), expected2)

    check()
    if compressed:
        assert moe.expand_scales(experts)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.expand_scales(experts)
        for poison in (0xA5, 0x17, 0xCC):
            experts._impl.w1_blockscale.view(torch.uint8).fill_(poison)
            experts._impl.w2_blockscale.view(torch.uint8).fill_(poison)
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            check()
