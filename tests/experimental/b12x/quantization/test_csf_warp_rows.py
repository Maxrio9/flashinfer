"""Warp-local consumers reproduce padded scale words from canonical CSF slots."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
import pytest
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


class ReadWords:
    def __init__(self, geometry, warp_rows):
        self.reader = CsfInlineReader(geometry)
        self.warp_rows = warp_rows
        self.threads = 128 if warp_rows else 32
        self.init_rounds = 256 // self.threads
        self.blocks = (
            self.reader.plane.storage.experts * self.reader.plane.storage.groups
        )

    @cute.jit
    def __call__(self, source: cute.Pointer, output: cute.Pointer, stream):
        self.kernel(source, output).launch(
            grid=(self.blocks, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, source, output):
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = Int32(tid) % Int32(32), Int32(tid) // Int32(32)
        block, _, _ = cute.arch.block_idx()
        groups = self.reader.plane.storage.groups * self.reader.plane.sections
        expert, group = Int32(block) // Int32(groups), Int32(block) % Int32(groups)
        shared = cutlass.utils.SmemAllocator().allocate_tensor(
            cutlass.Uint32, cute.make_layout(256), byte_alignment=16
        )
        target = cute.recast_ptr(shared.iterator, dtype=cutlass.Uint8)
        for i in cutlass.range_constexpr(self.init_rounds):
            shared[Int32(tid) + Int32(i * self.threads)] = Uint32(0xA5A5A5A5)
        cute.arch.sync_threads()
        if warp == Int32(0):
            self.reader.stage_compressed(source, expert, group, target, lane)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if cutlass.const_expr(self.warp_rows):
            self.reader.expand_warp_rows(
                source,
                target.toint().to(Int32),
                target.toint().to(Int32) + Int32(512),
                warp,
                lane,
            )
        else:
            self.reader.expand_compressed(source, group, target, lane)
            self.reader.pad_rows(group, target, lane)
        out = cute.make_tensor(
            cute.recast_ptr(output, dtype=Uint32), cute.make_layout(self.blocks * 128)
        )
        for i in cutlass.range_constexpr(4):
            row = lane + Int32(i * 32)
            if cutlass.const_expr(self.warp_rows):
                row = warp * Int32(32) + lane // Int32(4) + Int32(i * 8)
                value = shared[Int32(128) + row]
                if lane % Int32(4) == Int32(0):
                    out[Int32(block) * Int32(128) + row] = value
            else:
                out[Int32(block) * Int32(128) + row] = shared[row]


@pytest.mark.parametrize("pointer_dtype", [cutlass.Uint8, cutlass.Uint32])
@pytest.mark.parametrize("quantization", ["mxfp4"])
@pytest.mark.parametrize(
    "rows,columns,sections", [(128, 4, 1), (1152, 160, 2), (5120, 18, 1), (192, 6, 1)]
)
@pytest.mark.parametrize("exceptions", ["empty", "sparse", "dense"])
def test_warp_scale_words_exact(
    pointer_dtype, quantization, rows, columns, sections, exceptions
):
    device = require_b12x()
    planes, native = planes_and_native(
        rows,
        columns,
        2,
        quantization=quantization,
        dense=exceptions == "dense",
        empty=exceptions == "empty",
    )
    payload = upload_csf_block_payload(
        planes,
        device,
        native_codes=True,
        group_record_bytes=3,
        destination=3,
        group_rows=rows // sections,
    )
    plane = Nvfp4CompactBlockPlane(payload)
    size = plane.scales.geometry.experts * plane.scales.geometry.groups * 128
    # Independently assemble padded row words from the native CPU byte grid.
    logical = logical_grid(native, rows, columns)
    expected_bytes = torch.zeros((size // 128, 128, 4), dtype=torch.uint8)
    block = 0
    section_rows = rows // sections
    for expert in range(2):
        for section in range(sections):
            for row_group in range((section_rows + 127) // 128):
                nr = min(128, section_rows - row_group * 128)
                start = section * section_rows + row_group * 128
                for column in range((columns + 3) // 4):
                    nc = min(4, columns - column * 4)
                    expected_bytes[block, :nr, :nc] = logical[
                        expert, start : start + nr, column * 4 : column * 4 + nc
                    ]
                    block += 1
    expected = expected_bytes.reshape(-1).view(torch.int32)
    outputs = [torch.empty(size, dtype=torch.int32, device=device) for _ in range(2)]
    source_before = payload.storage.clone()
    pointer = make_ptr(pointer_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    programs = [
        b12x_compile(
            ReadWords(plane.geometry, registers),
            pointer,
            make_ptr(cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "test.csf_warp_rows", 1, (plane.geometry, registers, str(pointer_dtype))
            ),
        )
        for registers in (False, True)
    ]
    source = make_ptr(
        pointer_dtype,
        payload.storage.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    pointers = [
        make_ptr(cutlass.Uint8, o.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for o in outputs
    ]
    graphs = []
    for program, output in zip(programs, pointers, strict=True):
        program(source, output, current_cuda_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            program(source, output, current_cuda_stream())
        graphs.append(graph)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    for poison in (0x12345678, -1, 0):
        for graph, output in zip(graphs, outputs, strict=True):
            output.fill_(poison)
            graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(outputs[0], output) for output in outputs[1:])
        assert all(torch.equal(output.cpu(), expected) for output in outputs)
        assert torch.equal(payload.storage, source_before)
        assert torch.cuda.memory_allocated() == allocated


@pytest.mark.parametrize("pointer_dtype", [cutlass.Uint8, cutlass.Uint32])
@pytest.mark.parametrize("record_count", [6, 7, 8, 31, 32, 33, 212, 256])
def test_global_record_tail_pointer_stride_boundary(
    pointer_dtype, record_count, monkeypatch
):
    """The last replacement crosses staging capacity through either pointer view."""
    import numpy as np
    import sys
    from b12x.loader._csf_blocks import export_csf_blocks

    rows, columns = 128, 4
    bases = np.full(rows, 120, dtype=np.uint8)
    codes = np.zeros(rows * ((columns + 7) // 8), dtype=np.uint8)
    fixed = np.concatenate(
        (bases.reshape(-1, 16), codes.reshape(rows // 16, -1)), axis=1
    )
    tail_position = 104 * columns + 2
    positions = [
        position for position in range(rows * columns) if position != tail_position
    ][: record_count - 1]
    positions.append(tail_position)
    assert len(positions) == len(set(positions)) == record_count
    records = np.array(positions, dtype=np.uint32) | (np.uint32(128) << np.uint32(24))
    plane = export_csf_blocks(
        fixed, records, rows=rows, columns=columns, quantization="mxfp4"
    )
    assert plane.exceptions.size == record_count
    logical = np.full((2, rows, columns), 120, dtype=np.uint8)
    for position in positions:
        logical[:, position // columns, position % columns] = 128
    # logical_grid reverses the native [row_group,column_group,row32,row_block,col] packing.
    native = torch.from_numpy(
        logical.reshape(2, 1, 4, 32, 1, 4)
        .transpose(0, 1, 4, 3, 2, 5)
        .copy()
        .reshape(2, rows, columns)
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "planes_and_native",
        lambda *args, **kwargs: ((plane, plane), native),
    )
    test_warp_scale_words_exact(pointer_dtype, "mxfp4", rows, columns, 1, "sparse")
