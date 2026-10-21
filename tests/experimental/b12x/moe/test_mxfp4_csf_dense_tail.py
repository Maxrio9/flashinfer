"""Inline-CSF W4A8 dynamic MoE with exception tails: groups carry more than 32 exception records.

Reuses test_mxfp4_csf.py's comparison of inline scales against natively expanded scales under
poisoned graph replay, with 16-row dense overrides so the staged exception list overflows into the
resident record tail.
"""

import numpy as np
import pytest
import torch

from . import test_mxfp4_csf as base


def _dense_planes(rows, columns, *, experts, seed):
    from b12x.loader._csf_blocks import export_csf_blocks
    from b12x.moe import fused_moe as moe

    device = base.require_b12x()
    rng = np.random.default_rng(seed + rows + columns)
    fixed, exceptions, grids = [], [], []
    for _ in range(experts):
        bases = rng.integers(120, 126, rows).astype(np.uint8)
        grid = bases[:, None] + rng.integers(0, 2, (rows, columns)).astype(np.uint8)
        grid.reshape(-1)[::97] = rng.integers(116, 130, grid.reshape(-1)[::97].shape)
        for row in rng.choice(rows - 16, 3, replace=False):
            grid[row : row + 16] = rng.integers(114, 132, (16, columns))
        selected = grid == bases[:, None] + 1
        outside = (grid != bases[:, None]) & ~selected
        selectors = np.packbits(selected, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        column_groups = (columns + 3) // 4
        group = position // columns // 128 * column_groups + position % columns // 4
        assert np.bincount(group).max() > 32
        exceptions.append(
            torch.from_numpy(
                position | (grid.reshape(-1)[position].astype(np.uint32) << 24)
            )
        )
        grids.append(grid)
    batch = moe.CsfBlockPlanes(
        tuple(
            export_csf_blocks(f, x, rows=rows, columns=columns, quantization="mxfp4")
            for f, x in zip(fixed, exceptions, strict=True)
        )
    )
    return batch, torch.from_numpy(np.stack(grids)).to(device)


@pytest.mark.parametrize(
    "h,n,tokens,tile_m", [(5120, 576, 128, 16), (512, 256, 32, 32)]
)
@pytest.mark.parametrize("layout", ["w13", "w31"])
def test_dense_exception_tail_inline_poisoned_graph(
    h, n, tokens, tile_m, layout, monkeypatch
):
    monkeypatch.setattr(base, "_csf_planes", _dense_planes)
    base.test_inline_poisoned_graph(
        h, n, tokens, "dynamic", tile_m, layout, monkeypatch
    )
