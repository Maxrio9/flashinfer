"""On-chip producer scales give the same SM103 Cake pipeline bytes as raw UE8M0 scales."""

import numpy as np
import pytest
import torch

H, I, E, K = 5120, 2304, 8, 6


@pytest.fixture(scope="module")
def bank():
    from flashinfer.experimental.mega_moe_v3 import runtime

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    try:
        arch = runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))
    if arch != "sm_103a":
        pytest.skip("on-chip producer scales are generated for SM103")
    generator = torch.Generator(device="cuda").manual_seed(1034)
    w1 = torch.randint(
        0,
        256,
        (E, 2 * I, H // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    rng = np.random.default_rng(1034)
    s1 = rng.integers(118, 122, (E, 2 * I, H // 32), dtype=np.uint8)
    s2 = rng.integers(115, 119, (E, H, I // 32), dtype=np.uint8)
    # Isolated outliers become exception records; rows spanning 26 exponents force raw-pool
    # tiles. Exponents stay finite so outputs can be compared as numbers.
    s1.reshape(-1)[::997] = 125
    s2.reshape(-1)[::1009] = 108
    s1[:, 5] = 100 + np.arange(H // 32, dtype=np.uint8) * 7 % 26
    s2[:, 130] = 96 + np.arange(I // 32, dtype=np.uint8) * 11 % 26
    return w1, w2, s1, s2


def _source(planes, *, join):
    from b12x.loader._csf_source import stage_csf_sources

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


def _producer(s1, s2):
    from b12x.loader import CsfScaleBuffers, count_csf_scales

    # The producer's FC1 source joins up then gate; FlashInfer's w1 stacks gate then up.
    fc1 = _source([plane for e in range(E) for plane in (s1[e, I:], s1[e, :I])], join=2)
    fc2 = _source(list(s2), join=1)
    preparation = count_csf_scales(fc1, fc2)
    kinds = {"uint8": torch.uint8, "uint32": torch.uint32}
    buffers = CsfScaleBuffers(
        **{
            item["name"]: torch.empty(
                item["shape"], dtype=kinds[item["dtype"]], device="cuda"
            )
            for item in preparation.numeric_layout
        }
    )
    return preparation.fill(buffers)


def _run(bank, ids, weights, x, sf, *, weight_scales=None):
    from flashinfer.experimental.mega_moe_v3 import runtime
    from flashinfer.experimental.mega_moe_v3.preparation import (
        prepare_pipeline_bindings,
    )

    w1, w2, s1, s2 = bank
    tokens = ids.shape[0]
    capacity = next(c for c in (1, 16, 128, 512, 1024, 4096) if tokens <= c)
    as_float = lambda b: torch.pow(2.0, torch.from_numpy(b).cuda().float() - 127)
    xp = torch.zeros((capacity, H), dtype=x.dtype, device="cuda")
    sp = torch.zeros((capacity, H // 128), dtype=sf.dtype, device="cuda")
    ip = torch.arange(K, device="cuda").repeat(capacity, 1).long()
    wp = torch.zeros((capacity, K), device="cuda")
    (
        xp[:tokens].copy_(x),
        sp[:tokens].copy_(sf),
        ip[:tokens].copy_(ids),
        wp[:tokens].copy_(weights),
    )
    args = dict(
        num_tokens=capacity,
        num_experts=384,
        top_k=K,
        hidden=H,
        intermediate=I,
        routed_weight_dtype="fp4",
        activation_clamp=10.0,
        x_fp8_packed=xp,
        x_sf_packed=sp,
        topk_idx=ip,
        topk_weights=wp,
        w1_fp4=w1,
        w2_fp4=w2,
        w1_sf=as_float(s1),
        w2_sf=as_float(s2),
    )
    route = runtime._route("sm_103a", "pipeline", args)
    data = prepare_pipeline_bindings(args, route["num_sms"]["sm_103a"])
    bindings = dict(data["bindings"], num_tokens=tokens)
    if weight_scales is not None:
        del bindings["SFB1"], bindings["SFB2"]
    plan = runtime.bind_prepared(
        "pipeline",
        args,
        {"pipeline": bindings},
        data["output"],
        reset_storage=data["reset_storage"],
        reset_buffers=data["reset_buffers"],
        owners=(args, data),
        weight_scales=weight_scales,
    )
    results = []
    for _ in range(2):
        data["output"].fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        results.append(data["output"][:tokens].clone())
    assert torch.equal(results[0], results[1])
    return results[0]


@pytest.mark.parametrize("tokens", [1, 6, 16, 100])
def test_on_chip_scales_match_raw_scales(bank, tokens):
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    source = torch.randn((tokens, H), device="cuda", generator=generator) * 0.3
    grouped = source.reshape(tokens, H // 32, 32)
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448.0))
    )
    x = (grouped / scales[..., None]).reshape(tokens, H).to(torch.float8_e4m3fn)
    sf = (
        ((scales.view(torch.int32) >> 23) & 255)
        .to(torch.uint8)
        .contiguous()
        .view(torch.int32)
    )
    ids = (
        torch.rand((tokens, E), device="cuda", generator=generator).topk(K, -1).indices
    )
    weights = torch.rand((tokens, K), device="cuda", generator=generator).softmax(-1)
    prepared = _producer(bank[2], bank[3])
    assert prepared.raw_tiles > 0
    expected = _run(bank, ids, weights, x, sf)
    actual = _run(bank, ids, weights, x, sf, weight_scales=prepared)
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected), (
        (actual.float() - expected.float()).abs().max().item()
    )
