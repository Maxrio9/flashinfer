"""Compile an exact scale-consumption variant of the selected Cake source pair."""

from functools import cache
import hashlib
import json
import re
from . import _csf_producer_source as producer
from . import _masked_source as masked
from .scales import CsfScaleSpec


def source_variant(arch, num_sms, name, record, kernel, binding, spec):
    """CPU-only source/ABI closure; fail closed on unsupported source schedules."""
    if arch != "sm_103a" or type(num_sms) is not int or num_sms <= 0:
        raise ValueError("compressed Cake scales require a selected SM103 schedule")
    if not isinstance(spec, CsfScaleSpec):
        raise TypeError(
            "compressed Cake source selection requires a typed scale specification"
        )
    candidates = [
        capacity
        for capacity in producer.CAPACITIES
        if producer.program(capacity) == name
    ]
    if (
        len(candidates) != 1
        or record.get("ffi_prepare_entry")
        or record["tma_workspace_bytes"]
    ):
        raise ValueError(
            "selected Cake program has no direct compressed-scale source contract"
        )
    if (spec.hidden, spec.intermediate) != (5120, 2304):
        raise ValueError(
            "scale geometry differs from the selected generated source contract"
        )
    fixed_sms = re.search(r"^#define NUM_CTAS (\d+)$", kernel, re.M)
    if not record.get("definitions") and (
        fixed_sms is None or int(fixed_sms[1]) != num_sms
    ):
        raise ValueError("selected fixed source uses a different SM count")
    flags = masked.masked_cuda_flags(record, arch, num_sms)
    transformed = producer.transform(
        kernel,
        binding,
        capacity=candidates[0],
        local_experts=spec.experts,
        masked=masked,
    )
    plan = [list(item) for item in record["arg_plan"]]
    position = plan.index(["grid", "grid_x"])
    plan[position:position] = [["buffer", name] for name in producer.OPERANDS]
    return transformed, {**record, "arg_plan": plan}, flags


def program_identity(arch, num_sms, record, flags, spec, kernel, binding):
    """Bind program cache identity to source, ABI, flags and numeric scale contract."""
    identity = json.dumps(
        dict(
            arch=arch,
            num_sms=num_sms,
            record=record,
            flags=flags,
            scale_spec=spec.identity(),
            kernel=kernel,
            binding=binding,
        ),
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


@cache
def _load(arch, num_sms, name, record_json, kernel, binding, scale_identity):
    from ..deepgemm_common import nvcc_flags
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = json.loads(record_json)
    spec = CsfScaleSpec(**json.loads(scale_identity))
    transformed, record, recorded_flags = source_variant(
        arch, num_sms, name, record, kernel, binding, spec
    )
    flags = [*nvcc_flags(arch), *recorded_flags]
    digest = program_identity(
        arch,
        num_sms,
        record,
        flags,
        spec,
        transformed["kernel"],
        transformed["binding"],
    )
    variant = "cake_csf2b_" + digest[:24]
    previous = transformed["program_name"]
    kernel, binding = masked._renamed_sources(
        transformed["kernel"], transformed["binding"], previous, variant
    )
    directory = env.FLASHINFER_GEN_SRC_DIR / variant
    paths = (directory / "kernel.cu", directory / "binding.cu")
    for path, source in zip(paths, (kernel, binding), strict=True):
        masked._store_source(path, source)
    jit = gen_jit_spec(
        name=variant,
        sources=list(paths),
        extra_cuda_cflags=flags,
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )
    module = jit.build_and_load()
    return module, {
        **record,
        "sources": [str(path) for path in paths],
        "scale_spec": spec.identity(),
        "closure_sha256": digest,
        "source_program": variant,
        "library_path": str(jit.get_library_path()),
    }


def load_csf_program(catalog, arch, name, num_sms, spec):
    """Resolve/compile during preparation; no catalog mutation or raw fallback."""
    from flashinfer.jit import env

    if num_sms not in catalog.supported_num_sms(arch):
        raise ValueError("SM count differs from the selected catalog")
    record = dict(catalog.program(arch, name))
    if len(record["sources"]) != 2:
        raise ValueError(
            "compressed Cake requires the selected kernel/direct-binding source pair"
        )
    paths = [
        env.FLASHINFER_CSRC_DIR / path.removeprefix("csrc/")
        for path in record["sources"]
    ]
    if not paths[0].name.endswith("_kernel.cu") or not paths[1].name.endswith(
        "_binding.cu"
    ):
        raise ValueError("selected Cake source order differs from its direct ABI")
    return _load(
        arch,
        num_sms,
        name,
        json.dumps(record, sort_keys=True),
        paths[0].read_text(),
        paths[1].read_text(),
        json.dumps(spec.identity(), sort_keys=True),
    )
