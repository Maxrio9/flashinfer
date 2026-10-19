"""CPU validation of retained route eligibility, identity, storage, and binding."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8
from b12x.moe.fused_moe import _impl
from b12x.moe.fused_moe._preparation import (
    _FusedMoeState,
    _decode_query,
    _lower_caps,
    _query,
)
from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation._cache import SelectionCache
from b12x.preparation.session import PreparationJob


DEVICE = DeviceIdentity("nvidia", (12, 0), 170, "SM120")


def _declaration(
    tokens=1,
    *,
    hidden=2560,
    intermediate=640,
    top_k=10,
    local=128,
    global_experts=512,
    mode="a4",
    recipe=None,
    output_layout="token_and_routes",
    deterministic=None,
):
    weights = moe.plan_weights(
        source=moe.PackedSource(
            format="fp4_e8m0_k32" if recipe else "modelopt_nvfp4", w13_layout="w13"
        ),
        activation=moe.ActivationSpec(
            mode=mode,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0 if recipe else None,
        ),
        geometry=moe.MoEGeometry(
            num_experts=local, hidden_size=hidden, intermediate_size=intermediate
        ),
        numerical_recipe=recipe,
    )
    experts = SimpleNamespace(
        plan=weights,
        device=torch.device("cpu"),
        num_experts=local,
        hidden_size=hidden,
        intermediate_size=intermediate,
        _impl=SimpleNamespace(can_share_input=lambda **kwargs: True),
    )
    query = _query(
        experts,
        moe.ExecutionCapacity(
            max_tokens=tokens, top_k=top_k, route_num_experts=global_experts
        ),
        tokens,
        moe.RoutingSpec(
            output_layout=output_layout, deterministic_output=deterministic
        ),
        FrozenMapping(),
        FrozenMapping(),
    )
    return query, weights


@pytest.fixture(autouse=True)
def no_cuda_initialization():
    yield
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("value", ["routes", "", None, False])
def test_unknown_output_layout_is_rejected(value):
    with pytest.raises(ValueError, match="output_layout"):
        moe.RoutingSpec(output_layout=value)


def test_retained_routes_resolve_determinism_and_reject_contradictions():
    assert moe.RoutingSpec().output_layout == "token_sum"
    assert moe.RoutingSpec().deterministic_output is None
    assert (
        moe.RoutingSpec(output_layout="token_and_routes").deterministic_output is True
    )
    with pytest.raises(ValueError, match="deterministic_output=True"):
        moe.RoutingSpec(output_layout="token_and_routes", deterministic_output=False)
    assert moe.RoutingSpec(deterministic_output=False).deterministic_output is False


@pytest.mark.parametrize("tokens", [1, 17, 129, 4096])
@pytest.mark.parametrize("mode", ["a4"])
def test_qwen_route_contract_retains_all_eligible_dynamic_choices(tokens, mode):
    query, _ = _declaration(tokens, mode=mode)
    configuration = TUNING.configure(query, device=DEVICE)
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert configuration.pinned is None and configs
    assert configuration.default.backend == "dynamic"
    assert all(
        config.backend == "dynamic"
        and config.dynamic_route_mode == "grouped"
        and not config.nvfp4_materialize_intermediate
        for config in configs
    )
    assert {config.dynamic_tile_m for config in configs} >= {16, 64, 128}
    # The contract filters the ordinary tuning space; it introduces no tactic pin.
    ordinary = replace(query, output_layout="token_sum")
    all_configs = [
        config for _, config in TUNING.eligible_plan(ordinary, DEVICE).candidates
    ]
    assert configs == [config for config in all_configs if config.backend == "dynamic"]


def test_retained_routes_are_not_selected_by_model_geometry():
    query, _ = _declaration(
        9, hidden=512, intermediate=256, top_k=4, local=8, global_experts=16
    )
    assert TUNING.eligible_plan(query, DEVICE).candidates


def test_token_sum_keeps_micro_and_unchanged_default_numerics():
    query, _ = _declaration(output_layout="token_sum", deterministic=True)
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    assert any(config.backend == "micro" for config in configs)
    assert query.numerical_recipe is None
    encoded = dict(TUNING.encode_query(query))
    encoded.pop("output_layout")
    assert _decode_query(encoded) == query


@pytest.mark.parametrize(
    "changes",
    [
        {"output_layout": "unsupported"},
        {"deterministic_output": False},
        {"io_dtype": "float16"},
    ],
)
def test_query_rejects_invalid_retained_route_contract(changes):
    query, _ = _declaration()
    with pytest.raises(ValueError, match="output_layout|deterministic BF16"):
        TUNING.configure(replace(query, **changes), device=DEVICE)


@pytest.mark.parametrize("backend", ["micro", "w4a16"])
def test_override_cannot_select_a_token_only_tactic(backend):
    query, _ = _declaration(mode="a4")
    config = moe.MoeDecodeConfig(
        backend=backend,
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode="packed" if backend == "w4a16" else None,
    )
    with pytest.raises(ValueError, match="retaining original route rows"):
        TUNING.configure(query, device=DEVICE, override=config)


def test_output_layout_has_distinct_query_and_compiler_payload_identity():
    query, _ = _declaration()
    ordinary = replace(query, output_layout="token_sum")
    retained = TUNING.configure(query, device=DEVICE)
    reduced = TUNING.configure(ordinary, device=DEVICE)
    assert retained.encoded_query != reduced.encoded_query
    assert retained.encoded_query["output_layout"] == "token_and_routes"
    assert _decode_query(retained.encoded_query) == query
    assert _decode_query(reduced.encoded_query) == ordinary
    assert "output_layout" in TUNING.query_fields


def test_token_only_selection_cache_cannot_satisfy_retained_route_request(tmp_path):
    query, _ = _declaration()
    keys = []
    for layout in ("token_sum", "token_and_routes"):
        configuration = TUNING.configure(
            replace(query, output_layout=layout), device=DEVICE
        )
        obligation = SimpleNamespace(
            configuration=configuration,
            candidate_program_identity="candidate-programs",
            request=SimpleNamespace(
                dependencies=(),
                plan=SimpleNamespace(contract=TUNING, invocation=FrozenMapping()),
            ),
        )
        keys.append(PreparationJob._choice_key(None, obligation, {}))
    assert keys[0] != keys[1]
    cache = SelectionCache(tmp_path, {"schema_version": 23, "tuning_cache_version": 1})
    assignment, config = TUNING.eligible_plan(
        replace(query, output_layout="token_sum"), DEVICE
    ).candidates[0]
    cache.save(
        keys[0],
        assignment=assignment,
        config=TUNING.config_payload(config),
        coverage={
            "cartesian_count": 2,
            "legal_count": 2,
            "effective_count": 2,
            "measured_count": 2,
        },
        programs=(),
    )
    loaded = SelectionCache(tmp_path, cache.identity)
    assert loaded.get(keys[0]) is not None
    assert loaded.get(keys[1]) is None


def test_scratch_lowering_reserves_original_bf16_rows_and_rejects_wrong_tactics():
    query, weights = _declaration(17)
    config = TUNING.configure(query, device=DEVICE).default
    caps = _lower_caps(query, config, weights._impl, torch.device("cpu"))
    plan = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
    assert plan.launch_plan.implementation == "dynamic"
    spec = next(
        spec
        for spec in plan._core_workspace_plan.tensor_specs
        if spec.name == "route_output"
    )
    assert spec.dtype == torch.bfloat16
    assert spec.shape == (query.num_tokens * query.top_k, query.hidden_size)
    with pytest.raises(ValueError, match="retaining original route rows"):
        _lower_caps(
            query, replace(config, backend="micro"), weights._impl, torch.device("cpu")
        )


def test_weighted_mxfp8_recipe_keeps_its_declared_numerical_contract():
    query, _ = _declaration(
        17,
        hidden=5120,
        intermediate=2304,
        top_k=6,
        local=77,
        global_experts=384,
        mode="a8",
        recipe=BF16_FC1_WEIGHTED_MXFP8,
    )
    configs = [config for _, config in TUNING.eligible_plan(query, DEVICE).candidates]
    ordinary = replace(query, output_layout="token_sum")
    assert configs == [
        config for _, config in TUNING.eligible_plan(ordinary, DEVICE).candidates
    ]
    assert {config.dynamic_tile_m for config in configs} == {64, 128}
    assert query.numerical_recipe == BF16_FC1_WEIGHTED_MXFP8


@dataclass
class _Binding:
    a: torch.Tensor
    topk_ids: torch.Tensor
    route_output: object
    implementation: str = "dynamic"
    deterministic_output: bool = True
    compact_launches: object = None


def _state(binding, output_layout="token_and_routes"):
    _, weights = _declaration(hidden=128, intermediate=128, top_k=3, local=8)
    return _FusedMoeState(
        experts=SimpleNamespace(plan=weights, _impl=object()),
        scratch=SimpleNamespace(
            caps=SimpleNamespace(w4a16_fast_math=True), bind=lambda **kwargs: binding
        ),
        config=moe.MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=16,
            dynamic_route_mode="grouped",
        ),
        route_query=None,
        launchers=(),
        route_launcher=None,
        output_layout=output_layout,
    )


def test_binding_retains_full_original_slot_storage_without_a_copy():
    source = torch.empty((2, 128), dtype=torch.bfloat16)
    ids = torch.tensor([[7, 1, 4], [2, -1, 0]], dtype=torch.int32)
    rows = torch.empty((8, 128), dtype=torch.bfloat16)
    binding = _state(_Binding(source, ids, rows)).bind()
    assert binding.route_output is rows
    assert binding.a is source and binding.topk_ids is ids


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "dtype",
        "short",
        "width",
        "strided",
        "device",
        "nondeterministic",
        "implementation",
    ],
)
def test_binding_fails_closed_when_retained_storage_contract_is_missing(problem):
    binding = _Binding(
        torch.empty((2, 128), dtype=torch.bfloat16),
        torch.empty((2, 3), dtype=torch.int32),
        torch.empty((6, 128), dtype=torch.bfloat16),
    )
    if problem == "missing":
        binding.route_output = None
    elif problem == "dtype":
        binding.route_output = binding.route_output.float()
    elif problem == "short":
        binding.route_output = binding.route_output[:5]
    elif problem == "width":
        binding.route_output = binding.route_output[:, :127].contiguous()
    elif problem == "strided":
        binding.route_output = torch.empty((6, 256), dtype=torch.bfloat16)[:, ::2]
    elif problem == "device":
        binding.route_output = torch.empty(
            (6, 128), dtype=torch.bfloat16, device="meta"
        )
    elif problem == "nondeterministic":
        binding.deterministic_output = False
    else:
        binding.implementation = "micro"
    with pytest.raises(ValueError, match="weighted BF16 original route rows"):
        _state(binding).bind()


def test_token_sum_binding_does_not_require_retained_storage():
    binding = _Binding(
        torch.empty((2, 128), dtype=torch.bfloat16),
        torch.empty((2, 3), dtype=torch.int32),
        None,
        implementation="micro",
    )
    assert _state(binding, "token_sum").bind().route_output is None
