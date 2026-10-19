"""Local-bank guards and masked combine semantics for generated Cake sources.

The catalogued pipeline programs schedule a fixed 384-expert domain and read
every route ID as a valid expert. The patched source keeps that domain, guards
every route-ID read with a compile-time bank extent, treats -1 as a masked
route that performs no work and contributes nothing, and makes a token's output
NaN for any other ID outside the bank. Every anchor must occur exactly once.
"""

import re
from pathlib import Path

_PROGRAM_PREFIX = "deepgemm_mega_moe_v3_pipeline_fp4_"
_PROGRAMS = frozenset(
    _PROGRAM_PREFIX + suffix
    for suffix in ("t1", "t16_sm103a", "t128_sm103a", "t512", "t1024", "t4096")
)
_LOCAL = "CAKE_LOCAL_EXPERTS"
_MARKER = "// Cake local-bank masked-route extension"


def _replace_once(source, anchor, replacement, description):
    count = source.count(anchor)
    if count != 1:
        raise ValueError(
            f"Cake source mismatch at {description}: expected one anchor, found {count}"
        )
    return source.replace(anchor, replacement, 1)


def _one_match(pattern, source, description, *, flags=0):
    matches = list(re.finditer(pattern, source, flags))
    if len(matches) != 1:
        raise ValueError(
            f"Cake source mismatch at {description}: expected one anchor, found {len(matches)}"
        )
    return matches[0]


def patch_masked_sources(kernel, binding, *, program, num_local_experts):
    """Return patched source strings; no compilation, file writes or GPU use.

    The fixed 384-entry scheduling arrays remain intact. A compile-time local
    bank extent guards every raw-ID read before it can index one of them.
    -1 is a mask; any other out-of-range ID produces NaN in that token's output
    instead of indexing arbitrary memory or silently dropping an invalid route.
    """
    if program not in _PROGRAMS:
        raise ValueError(f"unsupported SM103 Cake FP4 program: {program}")
    if type(num_local_experts) is not int or not 1 <= num_local_experts <= 384:
        raise ValueError("a masked Cake bank must contain 1..384 real local experts")
    if _MARKER in kernel or _LOCAL in kernel:
        raise ValueError("Cake source has already been extended for masked routes")
    if f"kernel_{program}(" not in kernel or f"kernel_{program}(" not in binding:
        raise ValueError(
            "Cake kernel and binding symbols differ from the selected program"
        )
    one_token = program == _PROGRAM_PREFIX + "t1"
    declaration = "                int e_h = topk_idx_i32[idx * 2];\n"
    # t1 builds the same shared histogram on every CTA. Only CTA zero writes
    # masked combine entries. Later programs distribute each route once.
    store = "if (bid == 0) " if one_token else ""
    guarded = (
        declaration
        + f"""                if ((unsigned int)e_h >= {_LOCAL}) {{
                    {store}token_to_permuted[idx] = (e_h == -1) ? -1 : -2;
                    continue;
                }}
"""
    )
    kernel = _replace_once(kernel, declaration, guarded, "dispatch local-ID guard")
    if one_token:
        declaration = "                int e_s = topk_idx_i32[pair_idx * 2];\n"
        kernel = _replace_once(
            kernel,
            declaration,
            declaration
            + f"""                if ((unsigned int)e_s >= {_LOCAL}) continue;
""",
            "t1 direct pull guard",
        )
    else:
        # Masked slots do not appear in the histogram. The physical pair loop
        # must terminate at the last live expert instead of walking past it.
        old = """                while (expert_end <= pair_idx) {
                    pull_expert += 1;
                    expert_start = expert_end;
                    expert_end += expert_counts[pull_expert];
                }
"""
        new = f"""                while (expert_end <= pair_idx && pull_expert + 1 < {_LOCAL}) {{
                    pull_expert += 1;
                    expert_start = expert_end;
                    expert_end += expert_counts[pull_expert];
                }}
                if (pair_idx >= expert_end) break;
"""
        kernel = _replace_once(kernel, old, new, "bounded histogram pull")

    # Keep every combine barrier's phase advancing even when the corresponding
    # route is masked. Skipping arrive/wait would hang when masks change across
    # tokens or launches. No fake expert or persistent zero output row is needed.
    copy = _one_match(
        r"(?P<indent> +)(?P<call>cp_async_bulk_gmem2smem\(combine_smem_addr[^\n]+expert_output[^\n]+;)",
        kernel,
        "combine row copy",
    )
    replacement = (
        copy["indent"]
        + "if ((int)row_3 >= 0) {\n"
        + copy["indent"]
        + "    "
        + copy["call"]
        + "\n"
        + copy["indent"]
        + "}"
    )
    kernel = kernel[: copy.start()] + replacement + kernel[copy.end() :]
    arrive = _one_match(
        r"mbarrier_arrive_expect_tx\(combine_barriers_addr \+ \(epi_warp_2 \* 6 \+ (?P<slot>i_\d+)\) \* 8, 2560\);",
        kernel,
        "combine barrier arrival",
    )
    kernel = (
        kernel[: arrive.start()]
        + arrive[0].replace(", 2560);", ", ((int)row_3 >= 0) ? 2560u : 0u);")
        + kernel[arrive.end() :]
    )
    wait = _one_match(
        r"(?P<indent> +)mbarrier_wait\(combine_barriers_addr \+ \(epi_warp_2 \* 6 \+ (?P<slot>i_\d+)\) \* 8, phase_1\);\n",
        kernel,
        "combine barrier wait",
    )
    indent, slot = wait["indent"], wait["slot"]
    guard = f"""{indent}int masked_row = (int)__shfl_sync(0xFFFFFFFF, stored_row, {slot});
{indent}if (masked_row == -1) continue;
{indent}if (masked_row < -1) {{
{indent}    #pragma unroll
{indent}    for (int masked_column = 0; masked_column < 40; ++masked_column)
{indent}        reduced[masked_column] = __int_as_float(0x7fc00000);
{indent}    continue;
{indent}}}
"""
    kernel = kernel[: wait.end()] + guard + kernel[wait.end() :]
    kernel = f"{_MARKER}\n#define {_LOCAL} {num_local_experts}u\n" + kernel
    return kernel, binding


def _renamed_sources(kernel, binding, original, name):
    old, new = "kernel_" + original, "kernel_" + name
    if kernel.count(old + "(") != 1 or binding.count(old + "(") != 1:
        raise ValueError(
            "Cake kernel declaration count differs from the direct FFI ABI"
        )
    kernel, binding = kernel.replace(old, new), binding.replace(old, new)
    namespace = _one_match(
        r"namespace (?P<name>\w+) \{", binding, "direct FFI host namespace"
    )["name"]
    binding = re.sub(r"\b" + re.escape(namespace) + r"\b", name + "_host", binding)
    return kernel, binding


def _store_source(path, source):
    """Write complete content under an immutable key, tolerating shared builders."""
    from tempfile import NamedTemporaryFile

    data = source.encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(
                f"masked Cake cache contains different source bytes: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".cake-", suffix=".cu", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def masked_cuda_flags(record, arch, num_sms):
    """Preserve catalog flags and specialize only its declared dynamic macros.

    t16/t128 are fixed-architecture sources and declare no definitions. Adding
    NUM_CTAS to those sources can clobber a fixed declaration; the other four
    exported schedules explicitly request that compile-line specialization.
    """
    definitions = tuple(record.get("definitions", ()))
    if definitions not in ((), ("NUM_CTAS",)):
        raise ValueError(
            "masked Cake supports only the catalogued NUM_CTAS specialization"
        )
    flags = record["compile_flags"]
    flags = flags[arch] if isinstance(flags, dict) else flags
    if any(not isinstance(flag, str) for flag in flags):
        raise ValueError("Cake compile flags must be the catalogued strings")
    dynamic = [f"-DNUM_CTAS={num_sms}"] if definitions else []
    return [*flags, *dynamic, "--device-entity-has-hidden-visibility=false"]
