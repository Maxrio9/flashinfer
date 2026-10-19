"""Actual DSV4.1-Flash expert geometry, physical banks and native route results.

Synthetic checkpoint bytes use H=5120/I=2304 with the real MXFP4 layout. The
sparse analytical fixture keeps a bit-tight per-route oracle practical even
at 4096 tokens. This is a component qualification, not a model quality eval.
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics.reference import _fp4_encode_nibbles
from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8 as RECIPE
from b12x.moe.fused_moe import _impl
from b12x.moe.fused_moe._tuning import MoeDecodeConfig

HIDDEN = 5120
INTERMEDIATE = 2304
CAPACITY = 4096
LIVE_COUNTS = (1, 17, 128, 4096, 17, 1)


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")


def _pack(values):
    nibbles = _fp4_encode_nibbles(values)
    return (nibbles[..., ::2] | (nibbles[..., 1::2] << 4)).contiguous()


def _quantize(values):
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    amax = blocks.abs().amax(-1)
    bits = amax.contiguous().view(torch.int32).to(torch.int64)
    scales = (((bits + 0x1FFFFF) >> 23).clamp(min=113) - 8).to(torch.uint8)
    inverse = torch.exp2(127.0 - scales.float())
    payload = (blocks * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = payload.float() * torch.exp2(scales.float() - 127)[..., None]
    return payload.view(torch.uint8).reshape_as(values), scales, dequant.flatten(-2)


def _experts(bank_experts):
    device = "cuda"
    e, h, n = bank_experts, HIDDEN, INTERMEDIATE
    values = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
        device=device,
    )
    rows = torch.arange(n, device=device)
    lanes = torch.arange(4, device=device)
    expert_ids = torch.arange(e, device=device)
    gate = values[
        (expert_ids[:, None, None] * 7 + lanes[None, :, None] * 3 + rows[None, None, :])
        % 12
    ]
    up = values[
        (
            expert_ids[:, None, None] * 5
            + lanes[None, :, None]
            + rows[None, None, :] * 5
            + 1
        )
        % 12
    ]
    gate_scales = torch.exp2(((rows + 1) % 5 - 2).float())
    up_scales = torch.exp2((rows % 5 - 2).float())
    w13 = torch.zeros((e, 2 * n, h), device=device)
    w13[:, :n, :4] = up.transpose(1, 2)
    w13[:, n:, :4] = gate.transpose(1, 2)
    w13_scales = torch.empty((e, 2 * n, h // 32), dtype=torch.uint8, device=device)
    w13_scales[:, :n] = (rows % 5 + 125)[None, :, None]
    w13_scales[:, n:] = ((rows + 1) % 5 + 125)[None, :, None]
    down_rows = torch.arange(h, device=device)
    down_values = values[(expert_ids[:, None] * 3 + down_rows[None, :] * 7 + 2) % 12]
    down_scales = torch.exp2((down_rows % 3 - 1).float())
    w2 = torch.zeros((e, h, n), device=device)
    w2[:, down_rows, down_rows % n] = down_values
    w2_scales = (
        (down_rows % 3 + 126)
        .to(torch.uint8)[None, :, None]
        .expand(e, h, n // 32)
        .contiguous()
    )
    # Plain MXFP4 stores all weight scaling in exact UE8M0 block bytes.
    alpha = torch.ones(e, device=device)
    unit = torch.ones(1, device=device)
    plan = _impl.plan_b12x_fp4_moe_weights(
        numerical_recipe=RECIPE,
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=e,
        hidden_size=h,
        intermediate_size=n,
    )
    prepared = _impl.prepare_b12x_fp4_moe_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=_pack(w13),
        w2_fp4=_pack(w2),
        w1_blockscale=w13_scales,
        w2_blockscale=w2_scales,
        w1_global_scale=alpha,
        w2_global_scale=torch.ones(e, device=device),
        a1_gscale=unit,
        a2_gscale=unit,
        immutable_input_scales=True,
    )
    assert prepared.num_experts == e  # No fake experts or expanded expert bank.
    return (
        prepared,
        gate * gate_scales,
        up * up_scales,
        alpha,
        down_values * down_scales,
    )


def _live_values(count, topk, bank_experts, mode):
    device = "cuda"
    rows = torch.arange(count, device=device)
    slots = torch.arange(topk, device=device)
    position = (rows + (1 if mode == "recovered" else 0)) % 4
    amplitudes = torch.tensor([0.0, 2.0**-30, 0.25, 0.5, 1.0, 2.0, 4.0], device=device)
    amp = amplitudes[(rows + (2 if mode == "recovered" else 0)) % len(amplitudes)]
    x = torch.zeros((count, HIDDEN), dtype=torch.bfloat16, device=device)
    x[rows, position] = amp.to(torch.bfloat16)
    # Keep original token/slot positions and original finite FP32 weights. A
    # local bank owns at most E distinct slots; all others remain inactive.
    ids = torch.full((count, topk), -1, dtype=torch.int32, device=device)
    for expert in range(bank_experts):
        route_slot = (rows + expert + (1 if mode == "recovered" else 0)) % topk
        ids[rows, route_slot] = expert
    choices = torch.tensor([0.33333334, 0.20012345, 0.77703125, 0.9999], device=device)
    weights = choices[(rows[:, None] + slots[None, :]) % 4]
    if mode in ("mixed", "recovered"):
        ids[::4] = -1
        ids[1::5, 0] = bank_experts + 7  # Positive out-of-bank IDs are inactive too.
    if mode == "empty":
        ids.fill_(-1)
    if mode == "recovered":
        weights = weights * 0.73123455
    return x, ids, weights, position, amp


def _expected(ids, weights, position, amp, gate, up, alpha, down):
    e, _, n = gate.shape
    safe_ids = ids.long().clamp(0, e - 1)
    active = (ids >= 0) & (ids < e)
    raw_g = (
        gate[safe_ids, position[:, None]] * amp[:, None, None] * alpha[safe_ids, None]
    )
    raw_u = up[safe_ids, position[:, None]] * amp[:, None, None] * alpha[safe_ids, None]
    # Match the +0 initial tensor-core accumulator for all-zero dot products,
    # rather than the sign of one selected zero product. Later zeros retain
    # their exact activation/route-weight sign.
    zero_input = amp[:, None, None] == 0
    raw_g = torch.where(zero_input, torch.zeros_like(raw_g), raw_g)
    raw_u = torch.where(zero_input, torch.zeros_like(raw_u), raw_u)
    g = raw_g.to(torch.bfloat16).float().clamp(max=10.0)
    u = raw_u.to(torch.bfloat16).float().clamp(-10.0, 10.0)
    weighted = g * torch.sigmoid(g) * u * weights[..., None]
    weighted = torch.where(active[..., None], weighted, torch.zeros_like(weighted))
    payload, scales, quantized = _quantize(weighted)
    selected = quantized[..., torch.arange(HIDDEN, device=ids.device) % n]
    route = (selected * down[safe_ids]).to(torch.bfloat16)
    route = torch.where(active[..., None], route, torch.zeros_like(route))
    output = route.float().sum(1).to(torch.bfloat16)
    return payload, scales, route, output


def _check_intermediate(binding, ids, payload, scales, bank_experts):
    n = INTERMEDIATE
    rows = binding.physical_tiles_capacity * 128
    storage = binding.materialized_intermediate.view(torch.uint8).flatten()
    actual = storage[: rows * n].reshape(rows, n)
    actual_scales = (
        storage[rows * n : rows * (n + n // 32)]
        .reshape(n // 128, rows, 4)
        .permute(1, 0, 2)
        .reshape(rows, n // 32)
    )
    routed_pairs = ids.numel()
    covered = torch.zeros(routed_pairs, dtype=torch.int32, device=ids.device)
    for expert in range(bank_experts):
        count = int(binding.row_counts[expert])
        first = int(binding.expert_tile_base[expert]) * 128
        physical = torch.arange(first, first + count, device=ids.device)
        route = binding.token_map[physical].long()
        assert torch.equal(ids.flatten()[route], torch.full_like(route, expert))
        assert torch.equal(
            binding.token_weights[physical], binding.topk_weights.flatten()[route]
        )
        assert torch.equal(actual[physical], payload.reshape(-1, n)[route])
        assert torch.equal(actual_scales[physical], scales.reshape(-1, n // 32)[route])
        covered.scatter_add_(0, route, torch.ones_like(route, dtype=torch.int32))
    assert torch.equal(
        covered, ((ids.flatten() >= 0) & (ids.flatten() < bank_experts)).to(torch.int32)
    )


@pytest.mark.parametrize("bank_experts,topk", ((1, 3), (2, 6)))
def test_ds41_physical_banks_capacity_transitions_and_per_route_output(
    bank_experts, topk
):
    _require_sm120()
    experts, gate, up, alpha, down = _experts(bank_experts)
    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=128,
        dynamic_route_mode="grouped",
    )
    caps = _impl.TPMoEScratchCaps(
        max_tokens=CAPACITY,
        num_topk=topk,
        device=torch.device("cuda", torch.cuda.current_device()),
        weight_plan=experts.plan,
        quant_mode="w4a8_mx",
        decode_config=config,
        deterministic_output=True,
        swiglu_limit=10.0,
        numerical_recipe=RECIPE,
    )
    plan = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
    scratch = torch.empty(plan.layout.total_nbytes, dtype=torch.uint8, device="cuda")
    x = torch.empty((CAPACITY, HIDDEN), dtype=torch.bfloat16, device="cuda")
    ids = torch.empty((CAPACITY, topk), dtype=torch.int32, device="cuda")
    weights = torch.empty((CAPACITY, topk), dtype=torch.float32, device="cuda")
    output = torch.empty_like(x)
    # This is the native worker's return contract: weighted BF16 rows in the
    # original token/slot order, separate from a rounded per-bank token sum.
    results = torch.empty((CAPACITY, topk, HIDDEN), dtype=torch.bfloat16, device="cuda")
    graphs, bindings = {}, {}
    for count in sorted(set(LIVE_COUNTS)):
        values = _live_values(count, topk, bank_experts, "active")
        x[:count].copy_(values[0])
        ids[:count].copy_(values[1])
        weights[:count].copy_(values[2])
        binding = plan.bind(
            scratch=scratch,
            a=x[:count],
            experts=experts,
            topk_ids=ids[:count],
            topk_weights=weights[:count],
            output=output[:count],
            input_scales_static=True,
            fast_math=True,
        )
        _impl.b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        native_routes = binding.route_output[: count * topk].view(count, topk, HIDDEN)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _impl.b12x_moe_fp4(binding=binding)
            results[:count].copy_(native_routes)
        graphs[count], bindings[count] = graph, binding
    for count in LIVE_COUNTS:
        for mode in ("active", "mixed", "empty", "recovered"):
            new_x, new_ids, new_weights, position, amp = _live_values(
                count, topk, bank_experts, mode
            )
            x[:count].copy_(new_x)
            ids[:count].copy_(new_ids)
            weights[:count].copy_(new_weights)
            payload, scales, routes, expected_output = _expected(
                new_ids, new_weights, position, amp, gate, up, alpha, down
            )
            scratch.fill_(0xA5)
            results[:count].fill_(float("nan"))
            output[:count].fill_(float("nan"))
            before = torch.cuda.memory_allocated()
            graphs[count].replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            torch.testing.assert_close(
                results[:count],
                routes,
                rtol=0,
                atol=0,
                msg=lambda message: f"{count=}, {mode=}, {topk=}: {message}",
            )
            torch.testing.assert_close(output[:count], expected_output, rtol=0, atol=0)
            _check_intermediate(bindings[count], new_ids, payload, scales, bank_experts)
            # Input packing is only required for tokens with an active local
            # route; an empty bank may leave unconsumed input scratch untouched.
            active = ((new_ids >= 0) & (new_ids < bank_experts)).any(1)
            input_bytes, input_scales, _ = _quantize(new_x.float())
            actual_input = (
                bindings[count].packed_a_flat[: count * HIDDEN].view(count, HIDDEN)
            )
            actual_scales = (
                bindings[count]
                .scale_flat[: count * (HIDDEN // 32)]
                .view(count, HIDDEN // 32)
            )
            assert torch.equal(actual_input[active], input_bytes[active])
            assert torch.equal(actual_scales[active], input_scales[active])
