"""Prefetched bounds and row bases preserve CPU offsets and decoded scale bytes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import pytest
import numpy as np
import torch
from cutlass.cutlass_dsl import Int32, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.csf_inline import CsfInlineReader
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from .test_csf_blocks import planes_and_native
from .test_csf_native_preparation import logical_grid
from ..conftest import require_b12x


class ReadColumnBounds:
    def __init__(self, geometry, prefetched, decode):
        self.reader = CsfInlineReader(geometry)
        self.prefetched = prefetched
        self.decode = decode
        self.row_groups = (self.reader.plane.storage.rows + 127) // 128
        self.blocks = self.reader.plane.storage.experts * self.row_groups

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        observed: cute.Pointer,
        poison: cute.Pointer,
        stream,
    ):
        self.kernel(source, output, observed, poison).launch(
            grid=(self.blocks, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, source, output, observed, poison):
        lane, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        source_expert = Int32(block) // Int32(self.row_groups)
        row_group = Int32(block) % Int32(self.row_groups)
        columns = self.reader.plane.storage.column_groups
        groups = self.reader.plane.storage.groups
        sections = self.reader.plane.sections
        expert = source_expert // Int32(sections)
        section = source_expert % Int32(sections)
        base_word = self.reader.prefetch_row_bases(
            source, source_expert, row_group, Int32(lane)
        )
        bases_out = cute.make_tensor(
            observed, cute.make_layout(self.blocks * columns * 4 + self.blocks * 32)
        )
        bases_out[self.blocks * columns * 4 + Int32(block) * 32 + Int32(lane)] = (
            base_word.to(Int32)
        )
        cached_bounds = None
        if cutlass.const_expr(self.prefetched):
            cached_bounds = self.reader.prefetch_column_bounds(
                source, source_expert, row_group
            )
        shared = cutlass.utils.SmemAllocator().allocate_tensor(
            Uint32, cute.make_layout(128), byte_alignment=16
        )
        target = cute.recast_ptr(shared.iterator, dtype=cutlass.Uint8)
        poison_value = cute.make_tensor(poison, cute.make_layout(1))[0]
        for column in range(columns):
            # Reuse one staging slot across column groups and graph replays.
            for i in cutlass.range_constexpr(4):
                shared[Int32(lane) + Int32(i * 32)] = poison_value
            cute.arch.sync_warp()
            local_group = row_group * Int32(columns) + Int32(column)
            group = section * Int32(groups) + local_group
            direct_first, direct_end = self.reader.plane.group_record_bounds(
                source, source_expert, local_group
            )
            first, end = direct_first, direct_end
            if cutlass.const_expr(self.prefetched):
                first, end = Uint32(0), Uint32(0)
                for chunk in cutlass.range_constexpr(
                    cute.size(cached_bounds, mode=[0])
                ):
                    selected_first = cute.arch.shuffle_sync(
                        cached_bounds[chunk, 0], Int32(column) % Int32(32)
                    )
                    selected_end = cute.arch.shuffle_sync(
                        cached_bounds[chunk, 1], Int32(column) % Int32(32)
                    )
                    if Int32(column) // Int32(32) == Int32(chunk):
                        first, end = selected_first, selected_end
            index = Int32(block) * Int32(columns) + Int32(column)
            if lane == 0:
                bounds_out = cute.make_tensor(
                    observed, cute.make_layout(self.blocks * columns * 4)
                )
                bounds_out[index * 4] = direct_first
                bounds_out[index * 4 + 1] = direct_end
                bounds_out[index * 4 + 2] = first
                bounds_out[index * 4 + 3] = end
            if cutlass.const_expr(self.decode):
                if cutlass.const_expr(self.prefetched):
                    self.reader.stage_compressed(
                        source,
                        expert,
                        group,
                        target,
                        Int32(lane),
                        (first, end),
                        None,
                        base_word,
                    )
                else:
                    self.reader.stage_compressed(
                        source, expert, group, target, Int32(lane)
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_warp()
                self.reader.expand_compressed(source, group, target, Int32(lane))
                nr = cutlass.min(
                    Int32(128), Int32(self.reader.plane.storage.rows) - row_group * 128
                )
                nc = cutlass.min(
                    Int32(4), Int32(self.reader.plane.columns) - Int32(column) * 4
                )
                out = cute.make_tensor(
                    cute.recast_ptr(output, dtype=Uint32),
                    cute.make_layout(self.blocks * columns * 128),
                )
                for quarter in cutlass.range_constexpr(4):
                    position = Int32(lane) * 16 + Int32(quarter * 4)
                    if position < nr * nc:
                        out[index * 128 + position // 4] = shared[position // 4]
                cute.arch.sync_warp()


def _exercise(quantization, rows, columns, sections, exceptions, decode):
    device = require_b12x()
    planes, native = planes_and_native(
        rows,
        columns,
        2,
        quantization=quantization,
        dense=exceptions == "dense",
        empty=exceptions == "empty",
    )
    logical = logical_grid(native, rows, columns)
    payload = upload_csf_block_payload(
        planes,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=3,
        group_rows=rows // sections,
    )
    plane = Nvfp4CompactBlockPlane(payload)
    # The asynchronous slot decoder requires aligned expert code planes.
    assert plane.scales.geometry.code_bytes % 16 == 0
    row_groups = (rows // sections + 127) // 128
    column_groups = (columns + 3) // 4
    count = 2 * sections * row_groups * column_groups
    expected = torch.full((count, 512), 0xD3, dtype=torch.uint8)
    index = 0
    for expert in range(2):
        for section in range(sections):
            for r in range(row_groups):
                nr = min(128, rows // sections - r * 128)
                for c in range(column_groups):
                    nc = min(4, columns - c * 4)
                    values = logical[
                        expert,
                        section * (rows // sections) + r * 128 : section
                        * (rows // sections)
                        + r * 128
                        + nr,
                        c * 4 : c * 4 + nc,
                    ]
                    expected[index, : nr * nc] = values.reshape(-1)
                    index += 1
    outputs = [torch.empty_like(expected, device=device) for _ in range(2)]
    observed = [
        torch.empty(
            count * 4 + 2 * sections * row_groups * 32, dtype=torch.int32, device=device
        )
        for _ in range(2)
    ]
    poison = torch.zeros(1, dtype=torch.int32, device=device)
    source_before = payload.storage.clone()
    # Independently read the serialized CPU page-base/prefix tables. Their
    # terminal prefix per 32-group page is part of the storage contract.
    layout = plane.scales.geometry
    raw = source_before.cpu().numpy()
    pages = np.frombuffer(
        raw, dtype="<u4", count=layout.experts * layout.pages, offset=layout.page_bases
    ).reshape(layout.experts, layout.pages)
    prefixes = np.frombuffer(
        raw, dtype="<u2", count=layout.experts * layout.prefixes, offset=layout.offsets
    ).reshape(layout.experts, layout.prefixes)
    cpu_bounds = []
    for expert in range(layout.experts):
        for group in range(layout.groups):
            page = group // 32
            prefix = group + page
            first = int(pages[expert, page]) + int(prefixes[expert, prefix])
            end = int(pages[expert, page]) + int(prefixes[expert, prefix + 1])
            cpu_bounds.append((first, end, first, end))
    cpu_bounds = torch.tensor(cpu_bounds, dtype=torch.int32).reshape(-1)
    base_bytes = raw[
        layout.bases : layout.bases + layout.experts * layout.rows
    ].reshape(layout.experts, layout.rows)
    cpu_bases = np.zeros((layout.experts, row_groups, 128), dtype=np.uint8)
    for expert in range(layout.experts):
        for group in range(row_groups):
            nr = min(128, layout.rows - group * 128)
            cpu_bases[expert, group, :nr] = base_bytes[
                expert, group * 128 : group * 128 + nr
            ]
    cpu_observed = torch.cat(
        (cpu_bounds, torch.from_numpy(cpu_bases.reshape(-1).view(np.int32)).clone())
    )
    ptr = make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16)
    i32ptr = make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16)
    programs = [
        b12x_compile(
            ReadColumnBounds(plane.geometry, cached, decode),
            ptr,
            ptr,
            i32ptr,
            i32ptr,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "test.csf_column_bounds_and_row_bases",
                1,
                (plane.geometry, cached, decode),
            ),
        )
        for cached in (False, True)
    ]
    source = make_ptr(
        cutlass.Uint8,
        payload.storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    control = make_ptr(
        cutlass.Int32, poison.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    graphs = []
    try:
        for program, out, bounds in zip(programs, outputs, observed, strict=True):
            args = (
                source,
                make_ptr(
                    cutlass.Uint8,
                    out.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Int32,
                    bounds.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                control,
            )
            program(*args, current_cuda_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                program(*args, current_cuda_stream())
            graphs.append(graph)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for value in (0x12345678, -1, 0):
            poison.fill_(value)
            for graph, out, bounds in zip(graphs, outputs, observed, strict=True):
                out.fill_(0xD3)
                bounds.fill_(-1)
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(outputs[0], outputs[1])
            assert torch.equal(observed[0], observed[1])
            for out, bounds in zip(outputs, observed, strict=True):
                assert torch.equal(bounds.cpu(), cpu_observed)
                if decode:
                    assert torch.equal(out.cpu(), expected)
                assert torch.equal(
                    bounds[: count * 4].reshape(count, 4)[:, :2],
                    bounds[: count * 4].reshape(count, 4)[:, 2:],
                )
            assert torch.equal(payload.storage, source_before)
            assert torch.cuda.memory_allocated() == allocated
    finally:
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("quantization", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize(
    "rows,columns,sections",
    [(128, 4, 1), (384, 130, 2), (512, 131, 2), (512, 161, 2), (1152, 160, 2)],
)
@pytest.mark.parametrize("exceptions", ["empty", "sparse", "dense"])
def test_prefetched_column_bounds_and_cpu_bytes(
    quantization, rows, columns, sections, exceptions
):
    # W4A8 K64-native operands have two/four scale columns per final group.
    # Odd-column cases qualify the generic bounds helper independently of the
    # shared decoder's two/four-column consumer restriction.
    _exercise(quantization, rows, columns, sections, exceptions, columns % 2 == 0)


@pytest.mark.parametrize("quantization", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("columns", [131, 161])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Shared destination3 decoder assumes two/four scale columns; qualify odd tails after its base-byte mapping is generalized",
)
def test_shared_decoder_odd_tail_cpu_bytes(quantization, columns):
    # Keep the independent byte invariant visible; do not certify odd-tail
    # decoding merely because prefetched and ordinary consumers agree.
    _exercise(quantization, 512, columns, 2, "empty", True)
