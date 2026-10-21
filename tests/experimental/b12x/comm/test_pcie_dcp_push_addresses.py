"""Check peer staging arithmetic without importing CUDA or compiling kernels.

The arithmetic coverage includes wider rank counts; GPU synchronization and
native replay require separate multi-GPU tests.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


_KERNEL = (
    Path(__file__).resolve().parents[4]
    / "flashinfer/experimental/b12x/comm/pcie/_dcp_a2a_cute.py"
)


def _staging_expressions():
    tree = ast.parse(_KERNEL.read_text())
    launch = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_AllGatherHeadsLaunch"
    )
    kernel = next(
        node
        for node in launch.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    )
    expressions = [
        node.value
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "staged_pack"
            for target in node.targets
        )
    ]
    assert len(expressions) == 2
    return [compile(ast.Expression(node), str(_KERNEL), "eval") for node in expressions]


@pytest.mark.parametrize("world", (2, 4, 8, 9, 10, 12, 16))
def test_push_staging_addresses_match_rank_order_and_capacity(world):
    write, read = _staging_expressions()
    for batches in (1, 2, 13, 24, 32, 40, 49, 64, 192):
        for local_packs in (4, 8, 24, 32, 33, 72, 1560, 4040):
            for receiver in range(world):
                sources = [rank for rank in range(world) if rank != receiver]
                for slot, source in enumerate(sources):
                    for batch in sorted({0, batches - 1}):
                        for pack in sorted(
                            {
                                0,
                                local_packs - 1,
                                min(31, local_packs - 1),
                                min(32, local_packs - 1),
                            }
                        ):
                            common = dict(
                                Int64=int,
                                batch_index=batch,
                                peers=world - 1,
                                local_packs=local_packs,
                                pack=pack,
                            )
                            written = eval(
                                write,
                                {},
                                dict(
                                    common,
                                    self=SimpleNamespace(_rank=source),
                                    peer=receiver,
                                ),
                            )
                            consumed = eval(
                                read,
                                {},
                                dict(
                                    common,
                                    self=SimpleNamespace(_rank=receiver),
                                    source=source,
                                ),
                            )
                            expected = (batch * (world - 1) + slot) * local_packs + pack
                            assert written == consumed == expected
                            assert 0 <= written < batches * (world - 1) * local_packs
                            # The caller's allocation covers every rank, including self.
                            assert (
                                batches * (world - 1) * local_packs
                                < batches * world * local_packs
                            )
                            if world == 2:
                                assert written == batch * local_packs + pack
