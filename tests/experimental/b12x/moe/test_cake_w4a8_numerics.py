"""Host-only eligibility, lowering and compiler identity for Cake W4A8 numerics."""

from dataclasses import replace
from inspect import signature
from types import SimpleNamespace

import pytest
import torch

from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8 as RECIPE
from b12x.moe.fused_moe import _impl
from b12x.moe.fused_moe._preparation import (
    _dynamic_program_arguments,
    _lower_caps,
    _weight_payload,
    _decode_query,
)
from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    _default_config,
    _validate_query,
    validate_moe_decode_config,
    TUNING,
)
from b12x.preparation import DeviceIdentity, FrozenMapping

DEVICE = DeviceIdentity("nvidia", (12, 0), 170, "RTX SM120")


def _query(**changes):
    query = MoeDecodeQuery(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        top_k=2,
        num_tokens=129,
        routed_rows=258,
        route_num_experts=None,
        route_logits_dtype=None,
        apply_router_weight_on_input=False,
        collect_activation_amax=False,
        deterministic_output=True,
        swiglu_limit=10.0,
        swiglu_alpha=None,
        swiglu_beta=None,
        w13_layout="w13",
        weight_layouts=("qmma_repacked",),
        w4a16_weight_layout=None,
        w4a16_scale_format=None,
        w4a16_block_size_m=None,
        fast_math=True,
        numerical_recipe=RECIPE,
        controls=FrozenMapping({"dynamic_work_source": "materialized_queue"}),
        shared_input_scales=True,
    )
    return replace(query, **changes)


def _config(**changes):
    return replace(
        MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=128,
            dynamic_route_mode="grouped",
        ),
        **changes,
    )


def _weight_plan():
    return _impl.plan_b12x_fp4_moe_weights(
        numerical_recipe=RECIPE,
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
    )


def test_recipe_rejects_unknown_strings_before_lowering():
    with pytest.raises(ValueError, match="numerical_recipe"):
        _validate_query(_query(numerical_recipe="unimplemented"), DEVICE)


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "micro", "dynamic_tile_m": None, "dynamic_route_mode": None},
        {"dynamic_tile_m": 16},
        {"dynamic_tile_m": 32},
        {"dynamic_route_mode": "direct"},
        {"route_planner": "triton"},
    ],
)
def test_recipe_rejects_tactics_that_do_not_implement_boundary(changes):
    with pytest.raises(ValueError, match="weighted MXFP8"):
        validate_moe_decode_config(_query(), _config(**changes), DEVICE)


@pytest.mark.parametrize(
    "changes",
    [
        {"quant_mode": "w4a16"},
        {"source_format": "modelopt_nvfp4"},
        {"io_dtype": "float16"},
        {"activation": "situ"},
        {"swiglu_limit": None},
        {"swiglu_limit": 8.0},
        {"swiglu_alpha": 1.5},
        {"swiglu_beta": 1.0},
        {"apply_router_weight_on_input": True},
        {"collect_activation_amax": True},
        {"deterministic_output": False},
        {"fast_math": False},
        {"hidden_size": 128},
        {"intermediate_size": 192},
        {"controls": FrozenMapping({"dynamic_work_source": "ready_queue"})},
    ],
)
def test_recipe_rejects_unsupported_numerical_contract(changes):
    with pytest.raises(ValueError, match="weighted MXFP8"):
        validate_moe_decode_config(_query(**changes), _config(), DEVICE)


def test_recipe_rejects_other_architecture():
    with pytest.raises(ValueError, match="weighted MXFP8"):
        validate_moe_decode_config(
            _query(), _config(), DeviceIdentity("nvidia", (10, 3), 148, "B300")
        )


@pytest.mark.parametrize("tile_m", (64, 128))
def test_recipe_is_preserved_by_caps_execution_and_compiler_lowering(tile_m):
    q = _query()
    config = _config(dynamic_tile_m=tile_m)
    validate_moe_decode_config(q, config, DEVICE)
    caps = _lower_caps(q, config, _weight_plan(), torch.device("cpu"))
    plan = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
    assert caps.numerical_recipe == plan.launch_plan.numerical_recipe == RECIPE
    args = _dynamic_program_arguments(plan.launch_plan, caps)
    assert args["share_input_across_experts"]
    assert not args["direct_routing"] and not args["external_route_plan"]
    assert args["planned_tile_m"] == tile_m
    assert plan.scratch_specs()[0].shape[0] > 16


def test_recipe_survives_actual_tuning_compiler_job_weight_payload():
    # Ordinary tuning reconstructs the weight plan in its compiler process,
    # unlike direct binding tests that retain the original Python owner.
    declared = _weight_plan()
    payload = _weight_payload(SimpleNamespace(plan=SimpleNamespace(_impl=declared)))
    args = dict(payload)
    args["params_dtype"] = getattr(torch, args["params_dtype"])
    restored = _impl.plan_b12x_fp4_moe_weights(**args)
    query = _decode_query(TUNING.encode_query(_query()))
    caps = _lower_caps(query, _config(), restored, torch.device("cpu"))
    assert restored == declared
    assert caps.numerical_recipe == restored.numerical_recipe == RECIPE
    stock = _weight_payload(
        SimpleNamespace(
            plan=SimpleNamespace(_impl=replace(declared, numerical_recipe=None))
        )
    )
    assert "numerical_recipe" not in stock


def test_explicit_recipe_has_eligible_default_without_global_switches():
    config = _default_config(_query(), DEVICE)
    assert config == _config()


@pytest.mark.parametrize("tile_m", (64, 128))
def test_recipe_survives_existing_dispatcher_abi(tile_m):
    kwargs = dict(
        volatile_launch_state=True,
        external_route_plan_requested=False,
        policy_max_active_clusters=37,
        planned_tile_m=tile_m,
        planned_direct_routing=False,
        w4a8_n64_repacked=False,
    )
    stock = _impl._encode_dynamic_launch_policy(**kwargs)
    weighted = _impl._encode_dynamic_launch_policy(**kwargs, numerical_recipe=RECIPE)
    assert stock >= 0 and weighted < 0
    assert _impl._decode_dynamic_launch_policy(
        weighted
    ) == _impl._decode_dynamic_launch_policy(stock)
    assert len(signature(_impl._tp_moe_dynamic_launch_fake).parameters) == 64


def test_compiler_and_cache_do_not_alias_stock_numerics(monkeypatch):
    # Fake compiler results belong to this test, outside the serving-program cache.
    monkeypatch.setattr(_impl, "_DYNAMIC_KERNEL_CACHE", {})
    monkeypatch.setattr(_impl, "_current_compute_capability", lambda: (12, 0))
    monkeypatch.setattr(_impl, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(
        _impl, "raise_if_kernel_resolution_frozen", lambda *a, **k: None
    )
    seen = []

    def compile_only(launch, *args, **kwargs):
        seen.append((launch._kernel, kwargs["compile_spec"]))
        return object()

    monkeypatch.setattr(_impl, "b12x_compile", compile_only)
    kwargs = dict(
        topk_ids_dtype=torch.int32,
        fast_math=True,
        activation="silu",
        quant_mode="w4a8_mx",
        w4a8_repacked=True,
        mac_override=1,
        deterministic_output=True,
        swiglu_limit=10.0,
        planned_tile_m=128,
        share_input_across_experts=True,
    )
    stock, _ = _impl._get_dynamic_kernel(8, 129, 256, 128, 2, 129, **kwargs)
    weighted, _ = _impl._get_dynamic_kernel(
        8, 129, 256, 128, 2, 129, **kwargs, numerical_recipe=RECIPE
    )
    again, _ = _impl._get_dynamic_kernel(
        8, 129, 256, 128, 2, 129, **kwargs, numerical_recipe=RECIPE
    )
    assert stock is not weighted and again is weighted
    assert len(seen) == 2 and seen[0][1] != seen[1][1]
    kernel = seen[1][0]
    assert kernel.w4a8_split_materialized
    assert kernel.materialized_phase1_kernel.weighted_mxfp8
    assert kernel.materialized_phase2_kernel.weighted_mxfp8


def test_mx_input_packing_is_independent_of_per_expert_global_scales():
    # W4A8 quantizes BF16 rows before applying expert scales in FC1. Unlike
    # NVFP4 it does not need a uniform-input-scale proof to share those rows.
    validate_moe_decode_config(_query(shared_input_scales=False), _config(), DEVICE)


@pytest.mark.parametrize("tile", ((16, 128), (32, 128)))
def test_direct_kernel_construction_rejects_unsupported_recipe_tactics(tile):
    from b12x.moe._shared.kernels.dynamic.kernel import MoEDynamicKernelBackend

    with pytest.raises(ValueError, match="weighted MXFP8"):
        MoEDynamicKernelBackend(
            16,
            tile,
            quant_recipe="w4a8_mx",
            w4a8_repacked=True,
            share_input_across_experts=True,
            materialize_intermediate=True,
            deterministic_output=True,
            fast_math=True,
            swiglu_limit=10.0,
            numerical_recipe=RECIPE,
        )
