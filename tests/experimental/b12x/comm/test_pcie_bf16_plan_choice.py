"""TP4 plan routing preserves small, fused, aliased, and non-BF16 operations."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_allreduce as manager
from b12x.comm.pcie import pcie_oneshot as one
from b12x.comm.pcie import pcie_twoshot_bf16 as two


class Channel:
    rank = 0
    world_size = 4
    device = torch.device("cpu")
    _closed = False
    row_elems = 8
    max_rows = 1 << 20
    accepts = two.PCIeTwoShotBF16.accepts

    def _prepared_channel_for_stream(self, *_):
        return self

    def close(self):
        self._closed = True


def owner():
    return manager.PCIeAllReduce(
        Channel(), "oneshot", algorithm_override="auto", bf16_twoshot=Channel()
    )


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((8, 5120), False),
        ((32, 5120), True),
        ((64, 2560), True),
        ((131072,), True),
        ((131040,), False),
        ((131073,), False),
    ],
)
def test_plan_size_and_flat_quarter_geometry(monkeypatch, shape, expected):
    from b12x.comm.pcie import _preparation

    runtime = owner()
    source = torch.empty(shape, dtype=torch.bfloat16)
    output = torch.empty_like(source)
    calls = []
    monkeypatch.setattr(
        _preparation,
        "query_from_runtime",
        lambda target, **kwargs: calls.append((target, kwargs)) or kwargs,
    )
    monkeypatch.setattr(_preparation, "plan", lambda query, **_: query)
    plan = runtime.plan(source, out=output)
    assert (plan["surface"] == "PCIeTwoShotBF16.all_reduce") is expected
    assert calls[0][0] is (runtime._bf16_twoshot if expected else runtime._runtime)
    fused = runtime.plan(source, operation="all_reduce_fused_add_rms_norm", out=output)
    assert fused["surface"] == "OneshotAllReduce.all_reduce_fused_add_rms_norm"


@pytest.mark.parametrize(
    "case",
    [
        "fp32",
        "strided_source",
        "strided_output",
        "unaligned",
        "alias",
        "overlap",
        "no_output",
        "channel",
    ],
)
def test_ineligible_contract_stays_on_oneshot(monkeypatch, case):
    from b12x.comm.pcie import _preparation

    runtime = owner()
    source = torch.empty(
        (32, 5120), dtype=torch.float32 if case == "fp32" else torch.bfloat16
    )
    output = torch.empty_like(source)
    if case == "strided_source":
        source = torch.empty((32, 10240), dtype=torch.bfloat16)[:, ::2]
    if case == "strided_output":
        output = torch.empty((32, 10240), dtype=torch.bfloat16)[:, ::2]
    if case == "unaligned":
        source = torch.empty(source.numel() + 1, dtype=torch.bfloat16)[1:].view_as(
            source
        )
    if case == "alias":
        output = source
    if case == "overlap":
        storage = torch.empty(source.numel() + 32, dtype=torch.bfloat16)
        source, output = storage[:-32].view_as(source), storage[32:].view_as(output)
    if case == "no_output":
        output = None
    monkeypatch.setattr(
        _preparation, "query_from_runtime", lambda target, **kwargs: kwargs
    )
    monkeypatch.setattr(_preparation, "plan", lambda query, **_: query)
    plan = runtime.plan(
        source, out=output, **({"channel_id": "other"} if case == "channel" else {})
    )
    assert plan["surface"] == "OneshotAllReduce.all_reduce"


@pytest.mark.parametrize(
    "world,single,capacity,enabled",
    [
        (4, True, 327680, True),
        (4, True, 327696, True),
        (4, True, 65536, False),
        (4, False, 327680, False),
        (2, True, 327680, False),
    ],
)
def test_factory_eagerly_forwards_same_ipc(
    monkeypatch, world, single, capacity, enabled
):
    ipc, group = object(), object()
    calls = []
    channel = Channel()
    channel.world_size = world
    monkeypatch.setattr(manager.dist, "get_world_size", lambda **_: world)
    monkeypatch.setattr(manager, "_algorithm_override", lambda: "auto")
    monkeypatch.setattr(manager, "_bf16_twoshot_supported", lambda **_: True)
    monkeypatch.setattr(manager.PCIeAllReduce, "_maybe_island_rs", lambda **_: None)
    monkeypatch.setattr(
        manager.PCIeOneshotAllReducePool,
        "from_exchange_group",
        lambda **kwargs: calls.append(("one", kwargs)) or channel,
    )
    monkeypatch.setattr(
        manager.PCIeTwoShotBF16,
        "from_exchange_group",
        lambda **kwargs: calls.append(("two", kwargs)) or Channel(),
    )
    runtime = manager.PCIeAllReduce.from_exchange_group(
        exchange_group=group,
        device="cuda:0",
        max_size=capacity,
        single_channel=single,
        ipc=ipc,
    )
    assert [name for name, _ in calls] == (["one", "two"] if enabled else ["one"])
    assert all(kwargs["ipc"] is ipc for _, kwargs in calls)
    if enabled:
        kwargs = calls[1][1]
        assert kwargs["row_elems"] == 8 and kwargs["max_rows"] % 4 == 0
        assert capacity <= kwargs["max_rows"] * 16 < capacity + 64
    assert (runtime._bf16_twoshot is not None) is enabled


def capability_factory(monkeypatch, counts, *, rank=0, failure=None):
    channel, auxiliary_calls = Channel(), []
    monkeypatch.delenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", raising=False)
    monkeypatch.setattr(manager.dist, "get_world_size", lambda **_: 4)
    monkeypatch.setattr(manager.dist, "get_rank", lambda **_: rank)
    monkeypatch.setattr(manager, "_algorithm_override", lambda: "auto")
    monkeypatch.setattr(manager.PCIeAllReduce, "_maybe_island_rs", lambda **_: None)
    monkeypatch.setattr(
        manager.PCIeOneshotAllReducePool, "from_exchange_group", lambda **_: channel
    )

    def properties(_):
        if failure == "local_query":
            raise RuntimeError("device properties failed")
        return SimpleNamespace(multi_processor_count=counts[rank])

    def gather(value, group):
        if isinstance(value, tuple):
            if failure == "status_exchange":
                raise RuntimeError("setup status exchange failed")
            if failure == "peer_query":
                return [(), ("RuntimeError: peer properties failed",), (), ()]
            return [value] * 4
        if failure == "count_exchange":
            raise RuntimeError("SM count exchange failed")
        if failure == "invalid_count":
            return [64, None, 64, 64]
        return counts

    def auxiliary(**kwargs):
        auxiliary_calls.append(kwargs)
        if failure == "setup":
            raise RuntimeError("two-shot allocation failed")
        return Channel()

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(one, "_broadcast_gather_object", gather)
    monkeypatch.setattr(manager, "_broadcast_gather_object", gather)
    monkeypatch.setattr(manager.PCIeTwoShotBF16, "from_exchange_group", auxiliary)
    return channel, auxiliary_calls


@pytest.mark.parametrize("sms", [36, 48, 63, 64])
def test_factory_retains_oneshot_below_twoshot_residency(monkeypatch, sms):
    channel, calls = capability_factory(monkeypatch, [sms] * 4)
    runtime = manager.PCIeAllReduce.from_exchange_group(
        exchange_group=object(), device="cuda:0", max_size=327680, single_channel=True
    )
    assert runtime._runtime is channel and not channel._closed
    assert (runtime._bf16_twoshot is not None) is (sms >= 64)
    assert len(calls) == int(sms >= 64)


@pytest.mark.parametrize("rank", range(4))
def test_mixed_sm_counts_make_the_same_factory_choice(monkeypatch, rank):
    channel, calls = capability_factory(monkeypatch, [80, 48, 80, 80], rank=rank)
    runtime = manager.PCIeAllReduce.from_exchange_group(
        exchange_group=object(), device="cuda:0", max_size=327680, single_channel=True
    )
    assert runtime._runtime is channel and runtime._bf16_twoshot is None
    assert not channel._closed and calls == []


@pytest.mark.parametrize(
    "failure,match",
    [
        ("local_query", "device properties failed"),
        ("peer_query", "peer properties failed"),
        ("status_exchange", "failed to exchange peer.*status"),
        ("count_exchange", "SM count exchange failed"),
        ("invalid_count", "invalid peer.*SM count"),
        ("setup", "two-shot allocation failed"),
    ],
)
def test_capability_and_setup_errors_close_owner_and_propagate(
    monkeypatch, failure, match
):
    channel, calls = capability_factory(monkeypatch, [80] * 4, failure=failure)
    with pytest.raises(RuntimeError, match=match):
        manager.PCIeAllReduce.from_exchange_group(
            exchange_group=object(),
            device="cuda:0",
            max_size=327680,
            single_channel=True,
        )
    assert channel._closed
    assert len(calls) == int(failure == "setup")


@pytest.mark.parametrize("cap,expected", [("0", 80), ("48", 48), ("96", 80)])
def test_visible_sm_query_preserves_test_slice_semantics(monkeypatch, cap, expected):
    monkeypatch.setenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", cap)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=80),
    )
    assert one._visible_sm_count(torch.device("cuda:0")) == expected
    if expected < 64:
        with pytest.raises(RuntimeError, match="64 visible SMs"):
            one._require_full_grid_residency(
                owner="two-shot",
                required_sms=64,
                device=torch.device("cuda:0"),
                exchange_group=None,
            )
    else:
        one._require_full_grid_residency(
            owner="two-shot",
            required_sms=64,
            device=torch.device("cuda:0"),
            exchange_group=None,
        )


def test_factory_honors_visible_sm_test_slice_limit(monkeypatch):
    channel, calls = capability_factory(monkeypatch, [80] * 4)
    monkeypatch.setenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", "48")
    monkeypatch.setattr(
        manager, "_broadcast_gather_object", lambda value, group: [value] * 4
    )
    runtime = manager.PCIeAllReduce.from_exchange_group(
        exchange_group=object(), device="cuda:0", max_size=327680, single_channel=True
    )
    assert runtime._runtime is channel and runtime._bf16_twoshot is None
    assert not channel._closed and calls == []


def test_twoshot_factory_uses_supplied_allocator(monkeypatch):
    calls = []
    ipc = SimpleNamespace(cudaSetDevice=lambda device: calls.append(("device", device)))
    shared = SimpleNamespace(peer_ptrs=(4096, 8192, 12288, 16384))
    monkeypatch.setattr(two.dist, "get_rank", lambda **_: 0)
    monkeypatch.setattr(two.dist, "get_world_size", lambda **_: 4)
    monkeypatch.setattr(two, "_normalize_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(
        two, "_run_collective_preallocation_setup", lambda **kwargs: kwargs["setup"]()
    )
    monkeypatch.setattr(two, "_require_full_grid_residency", lambda **_: None)
    monkeypatch.setattr(two, "_require_collective_contract", lambda **_: None)
    monkeypatch.setattr(
        two, "CudaRTLibrary", lambda: pytest.fail("provided allocator was replaced")
    )
    monkeypatch.setattr(
        two.PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        lambda *args, **kwargs: calls.append(("allocate", kwargs["ipc"])) or shared,
    )
    monkeypatch.setattr(
        two.PCIeTwoShotBF16,
        "_from_prepared_factory",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        two,
        "_finish_collective_runtime_setup",
        lambda **kwargs: calls.append(("finish", kwargs["ipc"])),
    )
    runtime = two.PCIeTwoShotBF16.from_exchange_group(
        exchange_group=object(), device="cuda:0", max_rows=16, row_elems=8, ipc=ipc
    )
    assert runtime.ipc is ipc
    assert calls == [("device", 0), ("allocate", ipc), ("finish", ipc)]


def test_manager_capture_and_close_retain_both_serial_owners():
    events = []

    class Owner(Channel):
        def __init__(self, name):
            self.name = name

        @contextmanager
        def capture(self, **kwargs):
            events.append((self.name, "enter", kwargs))
            yield self
            events.append((self.name, "leave"))

        def close(self):
            events.append((self.name, "close"))

    one, second = Owner("one"), Owner("two")
    runtime = manager.PCIeAllReduce(
        one, "oneshot", algorithm_override="auto", bf16_twoshot=second
    )
    with runtime.capture(stream="stream") as captured:
        assert captured is runtime
    assert events == [
        ("one", "enter", dict(stream="stream", channel_id=None)),
        ("two", "enter", {}),
        ("two", "leave"),
        ("one", "leave"),
    ]
    runtime.close()
    assert events[-2:] == [("two", "close"), ("one", "close")]


def test_prepared_launch_dispatch_checks_actual_owner(monkeypatch):
    runtime = owner()
    source = torch.empty((32, 5120), dtype=torch.bfloat16)
    output = torch.empty_like(source)
    plan = object()
    state = SimpleNamespace(
        query=SimpleNamespace(surface="PCIeTwoShotBF16.all_reduce"),
        runtime=runtime._bf16_twoshot,
    )
    calls = []
    runtime._bf16_twoshot.all_reduce = lambda inp, **kwargs: (
        calls.append((inp, kwargs)) or output
    )
    monkeypatch.setattr(manager, "require_prepared", lambda *_: state)
    assert runtime.all_reduce(source, out=output, plan=plan) is output
    assert calls == [(source, dict(plan=plan, out=output))]
    with pytest.raises(ValueError, match="without launch overrides"):
        runtime.all_reduce(source, out=output, plan=plan, blocks=1)
    state.runtime = Channel()
    with pytest.raises(ValueError, match="another all-reduce manager"):
        runtime.all_reduce(source, out=output, plan=plan)


def test_multishape_capture_context_rejects_overlap():
    runtime = SimpleNamespace(_capture_context_depth=0)
    with two.PCIeTwoShotBF16.capture(runtime):
        assert runtime._capture_context_depth == 1
        with pytest.raises(RuntimeError, match="overlapping"):
            with two.PCIeTwoShotBF16.capture(runtime):
                pass
    assert runtime._capture_context_depth == 0
