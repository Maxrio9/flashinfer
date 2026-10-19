"""Source transform that makes a selected Cake pipeline consume producer scales on chip.

The load-B role of the masked local-bank program stages each task's 144-byte
producer tiles and row bases into shared memory with cp.async, expands them
into Cake's UE8M0 scale words beside the weight TMA, applies exception records,
and loads raw-fallback tiles by TMA from the raw pool bound as SFB1. The direct
FFI entry gains four tensor operands (FC1/FC2 tiles and bases) that it
validates against the bank extent. Scheduling, the argument order of every
other operand and the self-cleaning counters are unchanged.
"""

import hashlib
import re

TILE, RING, PAIRS, EXTRA_SMEM = 144, 8, 4, 1408
OPERANDS = ("csf_l1_tiles", "csf_l1_bases", "csf_l2_tiles", "csf_l2_bases")
SIG_OLD = "uint8_t* __restrict__ PrivateSF2Bytes)"
SIG_NEW = "uint8_t* __restrict__ PrivateSF2Bytes, const uint4* __restrict__ csf_l1_tiles, const unsigned int* __restrict__ csf_l1_bases, const uint4* __restrict__ csf_l2_tiles, const unsigned int* __restrict__ csf_l2_bases)"
HELPERS = '\n// Exact two-bit scale producer: eight immutable 144-byte tiles and two base blocks.\n__device__ __forceinline__ void csf_cp_async_16(int dst, const void* src) {\n    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\\n" :: "r"(dst), "l"(src) : "memory");\n}\n__device__ __forceinline__ void csf_cp_async_commit() {\n    asm volatile("cp.async.commit_group;\\n" ::: "memory");\n}\ntemplate <int N> __device__ __forceinline__ void csf_cp_async_wait() {\n    asm volatile("cp.async.wait_group %0;\\n" :: "n"(N) : "memory");\n}\n__device__ __forceinline__ void csf_fence_proxy_async() {\n    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n}\n__device__ __forceinline__ unsigned int csf_spread2(unsigned int b) {\n    return (b | (b << 6) | (b << 12) | (b << 18)) & 0x03030303u;\n}\n'


def once(text, anchor, replacement, description):
    if text.count(anchor) != 1:
        raise ValueError(
            f"{description}: expected one source anchor, got {text.count(anchor)}"
        )
    return text.replace(anchor, replacement, 1)


def paired_loop(stages):
    def stage(index):
        return f"""                    const unsigned int csf_s{index} = b_stage;
                    mbarrier_wait(empty_addr + b_stage * 8, _phase_empty_1);
                    if (elect_sync()) {{
                        tma_2d_gmem2smem_cta2(smem_b_addr + b_stage * 16384, selected_weight_map, (k_1 + {index}) * 128, task_1[1] * task_1[6] + n_offset, ((full_addr + b_stage * 8) & 0xFEFFFFFF));
                    }}
                    __syncwarp();
                    b_stage += 1;
                    if (b_stage == {stages}) {{ b_stage = 0; _phase_empty_1 ^= 1; }}
"""

    def words(index):
        return "".join(
            f"""                    csf_w{index}.{field} = __byte_perm(csf_bases, 0, 0x{row:04x}u) + csf_spread2(__byte_perm(csf_sel{index}, 0, 0x{selector:04x}u));
"""
            for field, row, selector in zip(
                "xyzw",
                (0, 0x1111, 0x2222, 0x3333),
                (0x4440, 0x4441, 0x4442, 0x4443),
                strict=True,
            )
        )

    def records(index):
        return f"""                        if (lane < csf_hdr{index} && !(csf_hdr{index} & 0x80000000u)) {{
                            const unsigned int r = reinterpret_cast<const unsigned int*>(csf_tile{index})[33 + lane];
                            reinterpret_cast<unsigned char*>(smem_sfb + csf_s{index} * 128)[(r & 0x7fu) * 4 + ((r >> 8) & 3u)] = (unsigned char)(r >> 16);
                        }}
"""

    return f"""                const void* selected_weight_map = ((task_1[0] == 1) ? (&B1) : (&B2));
                const unsigned int n_offset = (task_1[3] * 2 + (unsigned int)cta_rank) * 128;
                const unsigned int csf_k_tiles = task_1[7] / 128;
                const unsigned int csf_block_row = task_1[1] * (task_1[6] / 128) + task_1[3] * 2 + (unsigned int)cta_rank;
                const uint4* csf_task_tiles = ((task_1[0] == 1) ? csf_l1_tiles : csf_l2_tiles) + (unsigned long long)csf_block_row * csf_k_tiles * 9;
                const unsigned int* csf_task_bases = ((task_1[0] == 1) ? csf_l1_bases : csf_l2_bases) + (unsigned long long)csf_block_row * 32;
                const unsigned int csf_bslot = csf_task_count & 1u;
                const unsigned int csf_t0 = csf_tile_count;
                csf_task_count += 1;
                csf_tile_count += csf_k_tiles;
                if (lane < 8)
                    csf_cp_async_16(csf_bases_addr + (int)(csf_bslot * 128 + lane * 16), reinterpret_cast<const uint4*>(csf_task_bases) + lane);
                csf_cp_async_commit();
                #pragma unroll
                for (unsigned int u = 0; u < {PAIRS}; ++u) {{
                    if (2 * u < csf_k_tiles && lane < 18)
                        csf_cp_async_16(csf_tiles_addr + (int)(((csf_t0 / 2 + u) % {PAIRS}) * 288 + lane * 16), csf_task_tiles + u * 18 + lane);
                    csf_cp_async_commit();
                }}
                csf_cp_async_wait<{PAIRS - 1}>();
                __syncwarp();
                const unsigned int csf_bases = csf_bases_ptr[csf_bslot * 32 + lane];
                #pragma unroll 1
                for (int k_1 = 0; k_1 < task_1[7] / 128; k_1 += 2) {{
{stage(0)}{stage(1)}                    const unsigned int csf_pslot = ((csf_t0 + (unsigned int)k_1) / 2) % {PAIRS};
                    csf_cp_async_wait<{PAIRS - 1}>();
                    __syncwarp();
                    const unsigned char* csf_tile0 = csf_tiles_ptr + csf_pslot * 288;
                    const unsigned char* csf_tile1 = csf_tile0 + 144;
                    const unsigned int csf_hdr0 = reinterpret_cast<const unsigned int*>(csf_tile0)[32];
                    const unsigned int csf_hdr1 = reinterpret_cast<const unsigned int*>(csf_tile1)[32];
                    const unsigned int csf_sel0 = reinterpret_cast<const unsigned int*>(csf_tile0)[lane];
                    const unsigned int csf_sel1 = reinterpret_cast<const unsigned int*>(csf_tile1)[lane];
                    uint4 csf_w0, csf_w1;
{words(0)}{words(1)}                    reinterpret_cast<uint4*>(smem_sfb + csf_s0 * 128)[lane] = csf_w0;
                    reinterpret_cast<uint4*>(smem_sfb + csf_s1 * 128)[lane] = csf_w1;
                    unsigned int csf_raw0 = 0, csf_raw1 = 0;
                    if ((csf_hdr0 | csf_hdr1) != 0) {{
                        __syncwarp();
                        if (csf_hdr0 & 0x80000000u) csf_raw0 = 512;
                        if (csf_hdr1 & 0x80000000u) csf_raw1 = 512;
{records(0)}{records(1)}                        if ((csf_raw0 | csf_raw1) != 0) {{
                            csf_fence_proxy_async();
                            __syncwarp();
                            if (elect_sync()) {{
                                if (csf_raw0) tma_2d_gmem2smem_cta2(smem_sfb_addr + csf_s0 * 512, (&SFB1), 0, csf_hdr0 & 0x7fffffffu, ((full_addr + csf_s0 * 8) & 0xFEFFFFFF));
                                if (csf_raw1) tma_2d_gmem2smem_cta2(smem_sfb_addr + csf_s1 * 512, (&SFB1), 0, csf_hdr1 & 0x7fffffffu, ((full_addr + csf_s1 * 8) & 0xFEFFFFFF));
                            }}
                        }}
                    }}
                    csf_fence_proxy_async();
                    __syncwarp();
                    if ((unsigned int)k_1 + {RING} < csf_k_tiles && lane < 18)
                        csf_cp_async_16(csf_tiles_addr + (int)(csf_pslot * 288 + lane * 16), csf_task_tiles + (unsigned long long)(k_1 + {RING}) * 9 + lane);
                    csf_cp_async_commit();
                    if (elect_sync()) {{
                        if (cta_rank == 0) {{
                            mbarrier_arrive_expect_tx(full_addr + csf_s0 * 8, 16384u + csf_raw0);
                            mbarrier_arrive_expect_tx(full_addr + csf_s1 * 8, 16384u + csf_raw1);
                        }} else {{
                            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((full_addr + csf_s0 * 8) & 0xFEFFFFFF), "r"(csf_raw0) : "memory");
                            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((full_addr + csf_s1 * 8) & 0xFEFFFFFF), "r"(csf_raw1) : "memory");
                        }}
                    }}
                    __syncwarp();
                }}"""


def binding_validation(local_experts, capacity):
    text = f"""  // The scheduler keeps its original E384 domain; weights/scales contain only real local experts.
  TVM_FFI_CHECK(arg_num_experts == 384 && arg_num_tokens >= 1 && arg_num_tokens <= {capacity} && arg_top_k == 6, ValueError)
      << "Cake T{capacity} with on-chip scales requires the E384 scheduling domain, 1..{capacity} tokens, and six physical slots";
  TVM_FFI_CHECK(arg_N1 == 4608 && arg_K1 == 5120 && arg_N2 == 5120 && arg_K2 == 2304, ValueError)
      << "Cake T{capacity} source geometry differs from its generated H5120/I2304 contract";
  TVM_FFI_CHECK(arg_B1.numel() == {local_experts}LL * arg_N1 * arg_K1 / 2 &&
                arg_B2.numel() == {local_experts}LL * arg_N2 * arg_K2 / 2, ValueError)
      << "packed weight bank differs from the local expert extent";
"""
    for name, size in zip(
        OPERANDS,
        (
            f"{local_experts}LL * (arg_N1 / 128) * (arg_K1 / 128) * 144",
            f"{local_experts}LL * arg_N1",
            f"{local_experts}LL * (arg_N2 / 128) * (arg_K2 / 128) * 144",
            f"{local_experts}LL * arg_N2",
        ),
        strict=True,
    ):
        text += f'''  check_cuda_tensor(arg_{name}, "{name}");
  check_dtype(arg_{name}, DLDataType{{kDLUInt, 8, 1}}, "{name}");
  check_contiguous(arg_{name}, "{name}");
  check_same_device(arg_{name}, arg_topk_idx_i32, "{name}", "topk_idx_i32");
  TVM_FFI_CHECK(arg_{name}.numel() == {size} &&
                reinterpret_cast<uintptr_t>(arg_{name}.data_ptr()) % 16 == 0, ValueError)
      << "{name} requires its exact byte extent and 16-byte alignment";
'''
    text += """  check_contiguous(arg_SFB1, "raw scale pool");
  check_contiguous(arg_SFB2, "raw scale pool alias");
  TVM_FFI_CHECK(arg_SFB1.ndim() == 2 && arg_SFB1.size(0) >= 1 && arg_SFB1.size(1) == 128 &&
                arg_SFB2.ndim() == 2 && arg_SFB2.size(0) == arg_SFB1.size(0) && arg_SFB2.size(1) == 128 &&
                arg_SFB1.data_ptr() == arg_SFB2.data_ptr() &&
                reinterpret_cast<uintptr_t>(arg_SFB1.data_ptr()) % 16 == 0, ValueError)
      << "SFB1/SFB2 must alias one uint32[at_least_one,128] raw pool; empty logical pools require a 512-byte sentinel";
"""
    return text


CAPACITIES = {
    1: ("t1", 11, 230400),
    16: ("t16_sm103a", 11, 230400),
    128: ("t128_sm103a", 11, 230400),
    512: ("t512", 11, 230400),
    1024: ("t1024", 10, 226560),
    4096: ("t4096", 7, 219904),
}


def program(capacity):
    return "deepgemm_mega_moe_v3_pipeline_fp4_" + CAPACITIES[capacity][0]


def validation(local_experts, capacity):
    return binding_validation(local_experts, capacity)


def state_source(smem):
    return f"""            unsigned int csf_tile_count = 0, csf_task_count = 0;
            const int csf_tiles_addr = smem + {smem};
            const int csf_bases_addr = smem + {smem + RING * TILE};
            const unsigned char* csf_tiles_ptr = reinterpret_cast<const unsigned char*>(smem_raw + {smem});
            const unsigned int* csf_bases_ptr = reinterpret_cast<const unsigned int*>(smem_raw + {smem + RING * TILE});
"""


def transform(kernel, binding, *, capacity, local_experts, masked):
    if type(capacity) is not int or capacity not in CAPACITIES:
        raise ValueError("capacity must be one of 1,16,128,512,1024,4096")
    if type(local_experts) is not int or not 1 <= local_experts <= 384:
        raise ValueError("local expert count must be 1..384")
    selected = program(capacity)
    baseline_kernel, baseline_binding = masked.patch_masked_sources(
        kernel, binding, program=selected, num_local_experts=local_experts
    )
    kernel, binding = baseline_kernel, baseline_binding
    stages = int(re.search(r"#define NUM_PIPE_STAGES (\d+)", kernel)[1])
    smem = int(re.search(r"dynamicSmemBytes = (\d+)u;", binding)[1])
    if (stages, smem) != CAPACITIES[capacity][1:]:
        raise ValueError("generated stage/shared-memory geometry has changed")
    if smem + EXTRA_SMEM > 232448:
        raise ValueError(
            "compressed-scale staging exceeds the supported shared-memory limit"
        )
    kernel = once(kernel, SIG_OLD, SIG_NEW, "kernel ABI")
    kernel = once(
        kernel,
        "\n#include <cuda_bf16.h>\n",
        "\n#include <cuda_bf16.h>\n" + HELPERS,
        "device helpers",
    )
    begin, end = (
        kernel.index("// ---- Role: load_b ----"),
        kernel.index("// ---- Role: mma ----"),
    )
    role = kernel[begin:end]
    role = once(
        role,
        "            unsigned int task_1[8];",
        state_source(smem) + "            unsigned int task_1[8];",
        "load-B private state",
    )
    weight_anchor = (
        "                const void* selected_weight_map"
        if "                const void* selected_weight_map" in role
        else "                unsigned int n_offset"
    )
    start = role.index(weight_anchor)
    finish = f"                    if (b_stage == {stages}) {{ b_stage = 0; _phase_empty_1 ^= 1; }}\n                }}"
    stop = role.index(finish, start) + len(finish)
    role = role[:start] + paired_loop(stages) + role[stop:]
    kernel = kernel[:begin] + role + kernel[end:]
    binding = once(binding, SIG_OLD, SIG_NEW, "host kernel declaration")
    binding = once(
        binding,
        "TensorView arg_PrivateSF2Bytes, int64_t grid_x",
        "TensorView arg_PrivateSF2Bytes, "
        + ", ".join("TensorView arg_" + name for name in OPERANDS)
        + ", int64_t grid_x",
        "direct FFI operands",
    )
    binding = once(
        binding,
        "  cudaStream_t stream = ",
        validation(local_experts, capacity) + "\n  cudaStream_t stream = ",
        "operand validation",
    )
    anchor = "  uint8_t* p_PrivateSF2Bytes = static_cast<uint8_t*>(arg_PrivateSF2Bytes.data_ptr());\n"
    binding = once(
        binding,
        anchor,
        anchor
        + "".join(f"  void* p_{name} = arg_{name}.data_ptr();\n" for name in OPERANDS),
        "kernel pointer arguments",
    )
    binding = once(
        binding,
        "&p_PrivateSF2Bytes};",
        "&p_PrivateSF2Bytes, " + ", ".join("&p_" + name for name in OPERANDS) + "};",
        "kernel argument order",
    )
    if binding.count(str(smem)) != 2:
        raise ValueError("shared-memory launch anchors differ")
    binding = binding.replace(str(smem), str(smem + EXTRA_SMEM))
    digest = hashlib.sha256((kernel + binding).encode()).hexdigest()[:20]
    name = f"cake_csf2b_t{capacity}_e{local_experts}_{digest}"
    kernel, binding = masked._renamed_sources(kernel, binding, selected, name)
    return dict(
        kernel=kernel,
        binding=binding,
        baseline_kernel=baseline_kernel,
        baseline_binding=baseline_binding,
        program_name=name,
        original_smem_bytes=smem,
        dynamic_smem_bytes=smem + EXTRA_SMEM,
        pipeline_stages=stages,
    )
