"""Autotuned NVFP4 original-slot rows survive changed-route CUDA graph replay."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_nvfp4_csf import compress_fixture
from .test_nvfp4_phase_kernels import _build_domain
from .test_prepared_nvfp4_split import _experts
from ..conftest import require_b12x


@pytest.mark.parametrize(
    "tokens,compressed", [(1, False), (17, False), (129, False), (17, True)]
)
def test_autotuned_weighted_original_slots_and_poisoned_replay(tokens, compressed):
    device = require_b12x()
    e, h, n, k = 8, 256, 128, 4
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=k, seed=1401)
    experts, packed = _experts(domain)
    source, ids = domain["x"], domain["topk_ids"]
    probabilities = domain["topk_weights"]
    probabilities.fill_(1.0)
    original_ids = ids.clone()
    original_source = source.clone()
    expansion, scale_buffers = None, ()
    if compressed:
        s13 = packed.w13_block_scales.view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
        s2 = packed.w2_block_scales.view(torch.float8_e4m3fn).view(e, h, n // 16)
        scale_buffers = (torch.empty_like(s13), torch.empty_like(s2))
        experts = moe.prepare_weights(
            plan=experts.plan,
            weights=moe.Nvfp4CsfWeights(
                packed=replace(
                    packed,
                    w13_block_scales=scale_buffers[0],
                    w2_block_scales=scale_buffers[1],
                ),
                w13_scales=compress_fixture(s13, 2 * n, h // 16),
                w2_scales=compress_fixture(s2, h, n // 16),
            ),
        )
        expansion = moe.plan_scale_expansion(experts, routes=tokens * k)
    plan = moe.plan_execution(
        experts=experts,
        capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=k),
        routing=moe.RoutingSpec(output_layout="token_and_routes"),
        invocation={"fast_math": False},
        scale_expansion=expansion,
    )

    def produce():
        source.copy_(original_source)
        ids.copy_(original_ids)
        probabilities.fill_(1.0)

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding),
            produce=produce,
            output=output,
            owners=(scratch, binding, original_source, original_ids),
        )

    def prepare_expansion(state):
        active = torch.empty(e, dtype=torch.int32, device=device)
        binding = state.bind(
            ids=ids, out13=scale_buffers[0], out2=scale_buffers[1], scratch=active
        )
        return PreparedCall(run=binding.run, produce=produce, owners=(active, binding))

    with PreparationSession(device=device, autotune=True, compile_workers=0) as session:
        requests = []
        if expansion is not None:
            requests.append(
                expansion.request(
                    name="scale-expansion",
                    prepare_call=prepare_expansion,
                    benchmark_call=prepare_expansion,
                )
            )
        requests.append(
            plan.request(
                name="retained-routes",
                prepare_call=prepare,
                benchmark_call=prepare,
                dependencies=("scale-expansion",) if expansion is not None else (),
            )
        )
        session.prepare(tuple(requests))
        assert plan.prepared.selection.config.backend == "dynamic"
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in plan.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = moe.bind(
            plan,
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        assert binding.route_output.dtype == torch.bfloat16
        rows = binding.route_output[: tokens * k].view(tokens, k, h)
        assert rows.is_contiguous()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.run(binding=binding)

        def replay():
            binding.route_output.fill_(float("nan"))
            output.fill_(float("nan"))
            addresses = tuple(t.data_ptr() for t in (*scratch, output))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            assert tuple(t.data_ptr() for t in (*scratch, output)) == addresses
            assert torch.isfinite(rows).all() and torch.isfinite(output).all()
            # BF16 route rows are accumulated in original slot order in FP32.
            expected = torch.zeros_like(source, dtype=torch.float32)
            for slot in range(k):
                expected.add_(rows[:, slot].float())
            torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
            return rows.clone()

        baseline = replay()
        assert baseline.abs().sum() > 0
        probabilities[:, 0] = 0.5
        weighted = replay()
        torch.testing.assert_close(weighted[:, 0], baseline[:, 0] * 0.5, rtol=0, atol=0)
        torch.testing.assert_close(weighted[:, 1:], baseline[:, 1:], rtol=0, atol=0)
        probabilities.fill_(1.0)

        # Expert-major storage must be scattered back to caller top-k slots.
        permutation = torch.tensor([2, 0, 3, 1], device=device)
        ids.copy_(original_ids[:, permutation])
        permuted = replay()
        torch.testing.assert_close(permuted, baseline[:, permutation], rtol=0, atol=0)

        ids[:, 1] = -1
        masked = replay()
        assert torch.count_nonzero(masked[:, 1]) == 0
        torch.testing.assert_close(
            masked[:, (0, 2, 3)], permuted[:, (0, 2, 3)], rtol=0, atol=0
        )
        ids.fill_(-1)
        assert torch.count_nonzero(replay()) == 0
        assert torch.count_nonzero(output) == 0

        ids.copy_(original_ids)
        source.neg_()
        changed = replay()
        assert not torch.equal(changed, baseline)
        source.copy_(original_source)
        torch.testing.assert_close(replay(), baseline, rtol=0, atol=0)
        graph.reset()
