"""Host eligibility of real physical banks without changing the router domain."""

from dataclasses import replace

import pytest

from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8 as RECIPE
from b12x.moe.fused_moe._tuning import (
    MoeDecodeConfig,
    MoeDecodeQuery,
    TUNING,
    _materialize_tuning,
    validate_moe_decode_config,
)
from b12x.preparation import DeviceIdentity, FrozenMapping

DEVICE = DeviceIdentity("nvidia", (12, 0), 170, "RTX SM120")
CONFIG = MoeDecodeConfig(
    backend="dynamic",
    route_planner="internal",
    max_active_clusters=None,
    dynamic_tile_m=128,
    dynamic_route_mode="grouped",
)


def _query(**changes):
    q = MoeDecodeQuery(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=1,
        hidden_size=5120,
        intermediate_size=2304,
        top_k=6,
        num_tokens=17,
        routed_rows=102,
        route_num_experts=384,
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
    )
    return replace(q, **changes)


@pytest.mark.parametrize(
    "local_experts,topk,global_experts",
    (
        (1, 3, 128),
        (1, 6, 384),
        (2, 6, 384),
        (384, 6, 384),
    ),
)
def test_masked_bank_retains_original_router_topk(local_experts, topk, global_experts):
    q = _query(
        num_experts=local_experts,
        top_k=topk,
        routed_rows=17 * topk,
        route_num_experts=global_experts,
    )
    result = _materialize_tuning(q, DEVICE, TUNING.encode_config(CONFIG))
    assert result == CONFIG
    assert q.top_k == topk and q.num_experts == local_experts
    assert q.route_num_experts == global_experts


@pytest.mark.parametrize(
    "changes",
    (
        {"numerical_recipe": None},
        {"route_num_experts": None},
        {"route_num_experts": 2},
        {"top_k": 9, "routed_rows": 153},
    ),
)
def test_small_bank_eligibility_is_scoped_to_validated_masked_recipe(changes):
    with pytest.raises(ValueError):
        _materialize_tuning(_query(**changes), DEVICE, TUNING.encode_config(CONFIG))


def test_stock_full_bank_contract_is_unchanged():
    q = _query(numerical_recipe=None, num_experts=8, top_k=6, route_num_experts=None)
    validate_moe_decode_config(q, CONFIG, DEVICE)
    assert _materialize_tuning(q, DEVICE, TUNING.encode_config(CONFIG)) == CONFIG
