"""SM120 byte-level producer and prepared replay checks for Cake W4A8 numerics.

Run only on an isolated SM120 device. No model checkpoint is required.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics.math import fabs_f32
from b12x._lib.intrinsics.reference import _fp4_encode_nibbles
from b12x.moe._shared.kernels.w4a8_phase1 import (
    W4A8MaterializedPhase1Kernel,
    quantize_block_fp8_cake,
)
from b12x.moe._shared.numerics import BF16_FC1_WEIGHTED_MXFP8 as RECIPE
from b12x.moe.fused_moe import _impl
from b12x.moe.fused_moe._tuning import MoeDecodeConfig


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")


def _tensor(tensor, dtype):
    result = from_dlpack(tensor, assumed_align=16)
    result.element_type = dtype
    return result


def _quantize_reference(values):
    blocks = values.reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    amax = blocks.abs().amax(-1)
    bits = amax.contiguous().view(torch.int32).to(torch.int64)
    exponent = (((bits + 0x1FFFFF) >> 23).clamp(min=113) - 8).to(torch.uint8)
    inverse = torch.exp2(127.0 - exponent.float())
    payload = (blocks * inverse[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.view(torch.uint8).reshape_as(values), exponent


def _dequantize(payload, exponent):
    blocks = payload.view(torch.float8_e4m3fn).float().reshape(*exponent.shape, 32)
    return (blocks * torch.exp2(exponent.float() - 127)[..., None]).flatten(-2)


class _BoundaryProbe:
    """Use the production activation and quantizer without an MMA oracle error."""

    def __init__(self, activate):
        self.activate = bool(activate)
        self.phase = W4A8MaterializedPhase1Kernel(
            fast_math=True,
            source_tile_m=128,
            deterministic_output=True,
            num_topk=2,
            swiglu_limit=10.0,
            numerical_recipe=RECIPE,
        )

    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        alpha: cute.Tensor,
        weights: cute.Tensor,
        weighted: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(gate, up, alpha, weights, weighted, payload, scales).launch(
            grid=(cute.ceil_div(gate.shape[0], 32), 1, 1),
            block=[32, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gate: cute.Tensor,
        up: cute.Tensor,
        alpha: cute.Tensor,
        weights: cute.Tensor,
        weighted: cute.Tensor,
        payload: cute.Tensor,
        scales: cute.Tensor,
    ):
        row = cute.arch.block_idx()[0] * 32 + cute.arch.thread_idx()[0]
        if row < gate.shape[0]:
            values = cute.make_rmem_tensor((32,), cutlass.Float32)
            amax = cutlass.Float32(0.0)
            for col in cutlass.range_constexpr(32):
                value = gate[row, col]
                if cutlass.const_expr(self.activate):
                    value = (
                        self.phase._activated_value(value, up[row, col], alpha[row])
                        * weights[row]
                    )
                values[col] = value
                weighted[row, col] = value
                amax = cutlass.max(amax, fabs_f32(value))
            words, exponent = quantize_block_fp8_cake(values, amax)
            for word in cutlass.range_constexpr(8):
                payload[row, word] = words[word]
            scales[row] = exponent.to(cutlass.Uint8)


@pytest.mark.parametrize("activate", (False, True))
def test_cake_producer_exact_bytes_and_changed_input_graph(activate):
    _require_sm120()
    if activate:
        # Just above/below BF16 halfway points, both clamp boundaries, an
        # unbounded negative gate, original non-BF16 routing weights, zero and
        # floor-limited tiny weighted activations.
        gate_row = torch.tensor(
            [
                0.0,
                1.00390,
                1.00392,
                -1.00390,
                -1.00392,
                9.9687,
                9.9688,
                10.0313,
                -12.0,
                24.0,
                -24.0,
                0.125,
                -0.125,
                2.0,
                -2.0,
                4.0,
            ],
            device="cuda",
        )
        gate = gate_row.repeat(2).repeat(8, 1)
        up = torch.linspace(-12, 12, 32, device="cuda").repeat(8, 1)
        alpha = torch.tensor(
            [1.0, 1.001, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0], device="cuda"
        )
        weights = torch.tensor(
            [0.33333334, 0.20012345, 0.77703125, 0.9999, 0.0, 2.0**-32, 2.0**-25, 1.0],
            device="cuda",
        )
    else:
        # The exact exponent transition is the integer bit rule, including
        # the zero/tiny floor and values surrounding its rounding boundary.
        max_bits = torch.tensor(
            [
                0,
                1,
                0x33000000,
                0x37DFFFFF,
                0x37E00000,
                0x37E00001,
                0x3FDFFFFF,
                0x3FE00000,
                0x3FE00001,
                0x42C80000,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        gate = max_bits.view(torch.float32)[:, None] * torch.linspace(
            -1, 1, 32, device="cuda"
        )
        up = torch.zeros_like(gate)
        alpha = torch.ones(len(gate), device="cuda")
        weights = torch.ones_like(alpha)
    weighted = torch.empty_like(gate)
    payload = torch.empty((len(gate), 8), dtype=torch.int32, device="cuda")
    scales = torch.empty(len(gate), dtype=torch.uint8, device="cuda")
    args = (
        _tensor(gate, cutlass.Float32),
        _tensor(up, cutlass.Float32),
        _tensor(alpha, cutlass.Float32),
        _tensor(weights, cutlass.Float32),
        _tensor(weighted, cutlass.Float32),
        _tensor(payload, cutlass.Uint32),
        _tensor(scales, cutlass.Uint8),
    )
    compiled = cute.compile(
        _BoundaryProbe(activate),
        *args,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    def run():
        # Capture changes the current stream. Resolve it for every launch,
        # rather than retaining the preparation stream in the argument tuple.
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for changed in (False, True):
        if changed:
            gate.mul_(-0.5)
            weights.mul_(0.73123455)
        weighted.fill_(float("nan"))
        payload.fill_(-1)
        scales.zero_()
        before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        if activate:
            g = (gate * alpha[:, None]).to(torch.bfloat16).float().clamp(max=10.0)
            u = (up * alpha[:, None]).to(torch.bfloat16).float().clamp(-10.0, 10.0)
            expected = g * torch.sigmoid(g) * u * weights[:, None]
            torch.testing.assert_close(weighted, expected, rtol=2e-6, atol=1e-8)
        else:
            torch.testing.assert_close(weighted, gate, rtol=0, atol=0)
        # Quantization is bit-tight against the actual FP32 producer output;
        # this does not hide a BF16 round: the FP32 boundary is checked above.
        expected_bytes, expected_scales = _quantize_reference(weighted)
        assert torch.equal(payload.view(torch.uint8), expected_bytes)
        assert torch.equal(scales, expected_scales.flatten())
        if activate:
            assert torch.equal(scales[4:6], torch.full_like(scales[4:6], 105))


def _pack(values):
    nibbles = _fp4_encode_nibbles(values)
    return (nibbles[..., ::2] | (nibbles[..., 1::2] << 4)).contiguous()


def _fixture(m):
    E, K, N, TOPK = 2, 256, 128, 2
    device = "cuda"
    gate_values = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
        device=device,
    )
    gate = gate_values[torch.arange(N, device=device) % len(gate_values)].repeat(E, 1)
    gate[1] = gate[1].roll(3)
    up = gate_values[
        (torch.arange(N, device=device) * 5 + 1) % len(gate_values)
    ].repeat(E, 1)
    up[1] = up[1].roll(7)
    w13 = torch.zeros((E, 2 * N, K), device=device)
    # Native w13 is [up, gate]; checkpoint w31 is normalized by preparation.
    w13[:, :N, 0] = up
    w13[:, N:, 0] = gate
    w2 = torch.zeros((E, K, N), device=device)
    w2[:, torch.arange(K, device=device), torch.arange(K, device=device) % N] = 1.0
    # Plain MXFP4 checkpoint preparation requires unit global weight scales.
    alpha = torch.ones(E, device=device)
    ones = torch.ones(E, device=device)
    plan = _impl.plan_b12x_fp4_moe_weights(
        numerical_recipe=RECIPE,
        quant_modes="w4a8_mx",
        source_format="fp4_e8m0_k32",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=E,
        hidden_size=K,
        intermediate_size=N,
    )
    experts = _impl.prepare_b12x_fp4_moe_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=_pack(w13),
        w2_fp4=_pack(w2),
        w1_blockscale=torch.full(
            (E, 2 * N, K // 32), 127, dtype=torch.uint8, device=device
        ),
        w2_blockscale=torch.full(
            (E, K, N // 32), 127, dtype=torch.uint8, device=device
        ),
        w1_global_scale=alpha,
        w2_global_scale=ones,
        a1_gscale=ones,
        a2_gscale=ones,
        immutable_input_scales=True,
    )
    x = torch.zeros((m, K), dtype=torch.bfloat16, device=device)
    amplitudes = torch.tensor([0.0, 2.0**-30, 0.25, 0.5, 1.0, 2.0, 4.0], device=device)
    x[:, 0] = amplitudes[torch.arange(m, device=device) % len(amplitudes)]
    ids = (
        torch.arange(TOPK, device=device)[None, :]
        + torch.arange(m, device=device)[:, None]
    ) % E
    weights = torch.tensor([0.33333334, 0.20012345], device=device).repeat(m, 1)
    weights[::7] = torch.tensor([0.0, 2.0**-30], device=device)
    return experts, x, ids.to(torch.int32), weights, gate, up, alpha


def _expected(x, ids, weights, gate, up, alpha):
    E, N = gate.shape
    raw_gate = x[:, :1, None].float() * gate[None] * alpha[None, :, None]
    raw_up = x[:, :1, None].float() * up[None] * alpha[None, :, None]
    # The analytical one-hot dot product must preserve MMA accumulation's
    # initial +0. An entirely zero input sums signed-zero products into +0;
    # multiplying a selected negative weight by zero alone would invent -0.
    # Do not normalize later activation/route-weight zeros: their signs matter.
    zero_input = x[:, :1, None] == 0
    raw_gate = torch.where(zero_input, torch.zeros_like(raw_gate), raw_gate)
    raw_up = torch.where(zero_input, torch.zeros_like(raw_up), raw_up)
    all_gate = raw_gate.to(torch.bfloat16).float().clamp(max=10.0)
    all_up = raw_up.to(torch.bfloat16).float().clamp(-10.0, 10.0)
    all_act = all_gate * torch.sigmoid(all_gate) * all_up
    token = torch.arange(len(x), device=x.device)[:, None]
    weighted = all_act[token, ids.long()] * weights[..., None]
    payload, scales = _quantize_reference(weighted)
    route_output = (
        _dequantize(payload, scales).repeat(1, 1, x.shape[1] // N).to(torch.bfloat16)
    )
    output = route_output.float().sum(dim=1).to(torch.bfloat16)
    return payload, scales, output


def _check_materialized_bytes(binding, ids, payload, scales, tile_m, expected_output):
    rows = binding.physical_tiles_capacity * tile_m
    n = payload.shape[-1]
    storage = binding.materialized_intermediate.view(torch.uint8).flatten()
    actual_payload = storage[: rows * n].reshape(rows, n)
    actual_scales = (
        storage[rows * n : rows * (n + n // 32)]
        .reshape(n // 128, rows, 4)
        .permute(1, 0, 2)
        .reshape(rows, n // 32)
    )
    # token_map stores original token-major route IDs under deterministic output.
    for expert in range(2):
        count = int(binding.row_counts[expert])
        first = int(binding.expert_tile_base[expert]) * tile_m
        physical = torch.arange(first, first + count, device=ids.device)
        route = binding.token_map[physical].long()
        assert torch.equal(ids.flatten()[route], torch.full_like(route, expert))
        assert torch.equal(
            binding.token_weights[physical], binding.topk_weights.flatten()[route]
        )
        observed = actual_payload[physical]
        expected = payload.reshape(-1, n)[route]
        expected_sf = scales.reshape(-1, n // 32)[route]
        if not torch.equal(observed, expected):
            different = observed != expected
            zero_sign = different & ((observed & 0x7F) == 0) & ((expected & 0x7F) == 0)
            raise AssertionError(
                {
                    "expert": expert,
                    "different_payload_bytes": int(different.sum()),
                    "signed_zero_only_bytes": int(zero_sign.sum()),
                    "other_payload_bytes": int((different & ~zero_sign).sum()),
                    "different_scale_bytes": int(
                        (actual_scales[physical] != expected_sf).sum()
                    ),
                    "first_original_routes": route[different.any(1)][:8].tolist(),
                    "final_output_max_abs_error": float(
                        (binding.output.float() - expected_output.float()).abs().max()
                    ),
                }
            )
        assert torch.equal(actual_scales[physical], expected_sf)


@pytest.mark.parametrize("tile_m", (64, 128))
def test_prepared_cake_recipe_exact_intermediate_and_replay(tile_m):
    _require_sm120()
    # More than one tile per expert exercises source halves and persistent
    # task scheduling; this also prevents a single-token fixture from hiding
    # wrong physical-row routing weights.
    experts, x, ids, weights, gate, up, alpha = _fixture(129)
    config = MoeDecodeConfig(
        backend="dynamic",
        route_planner="internal",
        max_active_clusters=None,
        dynamic_tile_m=tile_m,
        dynamic_route_mode="grouped",
    )
    caps = _impl.TPMoEScratchCaps(
        max_tokens=len(x),
        num_topk=2,
        device=x.device,
        weight_plan=experts.plan,
        quant_mode="w4a8_mx",
        decode_config=config,
        swiglu_limit=10.0,
        deterministic_output=True,
        numerical_recipe=RECIPE,
    )
    plan = _impl.plan_tp_moe_scratch(caps, prewarm_launches=False)
    scratch = torch.empty(plan.layout.total_nbytes, dtype=torch.uint8, device=x.device)
    output = torch.empty_like(x)
    binding = plan.bind(
        scratch=scratch,
        a=x,
        experts=experts,
        topk_ids=ids,
        topk_weights=weights,
        output=output,
        input_scales_static=True,
        fast_math=True,
    )
    _impl.b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _impl.b12x_moe_fp4(binding=binding)
    for changed in (False, True):
        if changed:
            x[:, 0].copy_(x[:, 0].roll(1))
            weights.mul_(0.71312345)
        expected_payload, expected_scales, expected_output = _expected(
            x, ids, weights, gate, up, alpha
        )
        scratch.fill_(0xA5)
        output.fill_(float("nan"))
        before = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        # Check the actual public source-input producer too: zero-only and
        # tiny BF16 K32 blocks carry exponent 105 rather than the stock zero.
        input_payload, input_scales = _quantize_reference(x.float())
        assert torch.equal(
            binding.packed_a_flat[: x.numel()].reshape_as(x), input_payload
        )
        assert torch.equal(
            binding.scale_flat[: input_scales.numel()].reshape_as(input_scales),
            input_scales,
        )
        _check_materialized_bytes(
            binding, ids, expected_payload, expected_scales, tile_m, expected_output
        )
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
