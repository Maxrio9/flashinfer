"""SM103 Cake pipeline programs that consume producer scales on chip: source and ABI contract."""

import json

import pytest

from flashinfer.experimental.mega_moe_v3 import _csf_producer_source as producer
from flashinfer.experimental.mega_moe_v3 import _csf_program, _masked_source, runtime
from flashinfer.experimental.mega_moe_v3.scales import CsfScaleSpec

ARCH = "sm_103a"


def _route(capacity):
    args = dict(
        num_tokens=capacity,
        num_experts=384,
        top_k=6,
        hidden=5120,
        intermediate=2304,
        routed_weight_dtype="fp4",
        activation_clamp=10.0,
    )
    return args, runtime._route(ARCH, "pipeline", args)


def _sources(name):
    from flashinfer.jit import env

    record = dict(runtime.catalog().program(ARCH, name))
    paths = [
        env.FLASHINFER_CSRC_DIR / path.removeprefix("csrc/")
        for path in record["sources"]
    ]
    return record, paths[0].read_text(), paths[1].read_text()


@pytest.mark.parametrize("capacity", sorted(producer.CAPACITIES))
@pytest.mark.parametrize("experts", [1, 307, 384])
def test_every_catalogued_capacity_has_an_on_chip_variant(capacity, experts):
    _, route = _route(capacity)
    name = runtime.stage_program(route["stages"][0], ARCH)
    assert name == producer.program(capacity)
    record, kernel, binding = _sources(name)
    transformed, variant, _ = _csf_program.source_variant(
        ARCH,
        route["num_sms"][ARCH],
        name,
        record,
        kernel,
        binding,
        CsfScaleSpec(experts, 5120, 2304),
    )
    assert f"#define CAKE_LOCAL_EXPERTS {experts}u" in transformed["kernel"]
    assert (
        transformed["dynamic_smem_bytes"]
        == transformed["original_smem_bytes"] + producer.EXTRA_SMEM
    )
    for operand in producer.OPERANDS:
        assert (
            f"arg_{operand}" in transformed["binding"]
            and operand in transformed["kernel"]
        )
    # The four scale operands precede the grid scalars; every other argument keeps its order.
    plan = [tuple(item) for item in variant["arg_plan"]]
    grid = plan.index(("grid", "grid_x"))
    assert plan[grid - 4 : grid] == [
        ("buffer", operand) for operand in producer.OPERANDS
    ]
    assert [item for item in plan if item[1] not in producer.OPERANDS] == [
        tuple(i) for i in record["arg_plan"]
    ]
    assert kernel.count("mbarrier_wait(") == transformed["baseline_kernel"].count(
        "mbarrier_wait("
    )


def test_scale_routes_carry_the_scale_identity_and_leave_raw_routes_unchanged():
    args, raw = _route(16)
    spec = CsfScaleSpec(12, 5120, 2304)
    prepared = runtime._route(ARCH, "pipeline", args, weight_scales=spec)
    assert prepared["metadata"]["weight_scales"] == spec.identity()
    assert (
        json.loads(prepared["metadata"]["prepared_key"])[1]["weight_scales"]
        == spec.identity()
    )
    assert (
        runtime._route(ARCH, "pipeline", args) is raw
        and "weight_scales" not in raw["metadata"]
    )
    assert runtime.key("pipeline", args) != runtime.key(
        "pipeline", args, weight_scales=spec
    )
    with pytest.raises(ValueError, match="geometry differs"):
        runtime.key("pipeline", dict(args, hidden=4096), weight_scales=spec)
    with pytest.raises(ValueError, match="1..384 experts"):
        CsfScaleSpec(385, 5120, 2304)


@pytest.mark.parametrize("name", sorted(_masked_source._PROGRAMS))
def test_masked_routes_keep_barrier_phases_and_reject_changed_sources(name):
    _, kernel, binding = _sources(name)
    patched, wrapper = _masked_source.patch_masked_sources(
        kernel, binding, program=name, num_local_experts=128
    )
    assert wrapper == binding
    assert patched.count("mbarrier_wait(") == kernel.count("mbarrier_wait(")
    assert patched.count("mbarrier_arrive_expect_tx(") == kernel.count(
        "mbarrier_arrive_expect_tx("
    )
    assert (
        "((int)row_3 >= 0) ? 2560u : 0u" in patched
        and "if (masked_row == -1) continue;" in patched
    )
    assert "token_to_permuted[idx] = (e_h == -1) ? -1 : -2;" in patched
    with pytest.raises(ValueError, match="already been extended"):
        _masked_source.patch_masked_sources(
            patched, wrapper, program=name, num_local_experts=128
        )
    changed = kernel.replace(
        "int e_h = topk_idx_i32[idx * 2];", "int e_h = topk_idx_i32[2 * idx];"
    )
    with pytest.raises(ValueError, match="dispatch local-ID guard"):
        _masked_source.patch_masked_sources(
            changed, binding, program=name, num_local_experts=8
        )
    for extent in (0, 385, True):
        with pytest.raises(ValueError, match="1..384 real local experts"):
            _masked_source.patch_masked_sources(
                kernel, binding, program=name, num_local_experts=extent
            )
