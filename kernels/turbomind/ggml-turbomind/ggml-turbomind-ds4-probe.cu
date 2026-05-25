// DS4/V100 fixed-shape TurboMind probe kernels.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"

namespace tmg = turbomind::gemm;

namespace ds4_probe_kernels {
__global__ void ds4_reduce6_half_to_float_kernel(
    const half* __restrict__ down_routes,
    float* __restrict__ route_out,
    const int* __restrict__ sorted_pairs,
    const float* __restrict__ route_weights,
    int n_routes,
    int hidden)
{
    const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (col >= hidden) {
        return;
    }

    float acc = 0.0f;
#pragma unroll
    for (int row = 0; row < 6; ++row) {
        const int pair = __ldg(sorted_pairs + row);
        const int tok = pair / n_routes;
        if (tok == 0) {
            const float w = __ldg(route_weights + row);
            acc += __half2float(__ldg(down_routes + (int64_t)row * hidden + col)) * w;
        }
    }
    route_out[col] = acc;
}
}  // namespace ds4_probe_kernels

namespace {

using namespace turbomind::gemm;
using namespace turbomind::gemm::sm70_s884;

using ProbeConfigM16 = Config_MXF4<kColMajor, 0>::Type<
    16, 128, 32, 1, 4, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Stream,
    2, true, 1, 32>;
using ProbeConfigM64 = Config_MXF4<kColMajor, 0>::Type<
    64, 128, 32, 1, 4, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Stream,
    2, true, 1, 32, 32, 128>;
using ProbeConfigM64S3 = Config_MXF4<kColMajor, 0>::Type<
    64, 128, 32, 1, 4, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Stream,
    3, true, 1, 32, 32, 128>;
using ProbeConfigM64S4 = Config_MXF4<kColMajor, 0>::Type<
    64, 128, 32, 1, 4, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Stream,
    4, true, 1, 32, 32, 128>;
using ProbeConfigM64N256 = Config_MXF4<kColMajor, 0>::Type<
    64, 256, 32, 1, 4, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Stream,
    2, true, 1, 32, 64, 128>;
using ProbeConfigM128 = Config_MXF4<kColMajor, 0>::Type<
    128, 128, 16, 2, 2, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Default,
    2, true, 1, 32, 64, 128>;
using ProbeConfigM128S3 = Config_MXF4<kColMajor, 0>::Type<
    128, 128, 16, 2, 2, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Default,
    3, true, 1, 32, 64, 128>;
using ProbeConfigM128S4 = Config_MXF4<kColMajor, 0>::Type<
    128, 128, 16, 2, 2, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Default,
    4, true, 1, 32, 64, 128>;

using ProbeKernelM16 = KernelImpl<typename ProbeConfigM16::Kernel>;
using ProbeKernelM64 = KernelImpl<typename ProbeConfigM64::Kernel>;
using ProbeKernelM64S3 = KernelImpl<typename ProbeConfigM64S3::Kernel>;
using ProbeKernelM64S4 = KernelImpl<typename ProbeConfigM64S4::Kernel>;
using ProbeKernelM64N256 = KernelImpl<typename ProbeConfigM64N256::Kernel>;
using ProbeKernelM128 = KernelImpl<typename ProbeConfigM128::Kernel>;
using ProbeKernelM128S3 = KernelImpl<typename ProbeConfigM128S3::Kernel>;
using ProbeKernelM128S4 = KernelImpl<typename ProbeConfigM128S4::Kernel>;

ProbeKernelM16 *probe_kernel_m16()
{
    static ProbeKernelM16 *kernel = new ProbeKernelM16();
    return kernel;
}

ProbeKernelM64 *probe_kernel_m64()
{
    static ProbeKernelM64 *kernel = new ProbeKernelM64();
    return kernel;
}

ProbeKernelM64S3 *probe_kernel_m64_s3()
{
    static ProbeKernelM64S3 *kernel = new ProbeKernelM64S3();
    return kernel;
}

ProbeKernelM64S4 *probe_kernel_m64_s4()
{
    static ProbeKernelM64S4 *kernel = new ProbeKernelM64S4();
    return kernel;
}

ProbeKernelM64N256 *probe_kernel_m64n256()
{
    static ProbeKernelM64N256 *kernel = new ProbeKernelM64N256();
    return kernel;
}

ProbeKernelM128 *probe_kernel_m128()
{
    static ProbeKernelM128 *kernel = new ProbeKernelM128();
    return kernel;
}

ProbeKernelM128S3 *probe_kernel_m128_s3()
{
    static ProbeKernelM128S3 *kernel = new ProbeKernelM128S3();
    return kernel;
}

ProbeKernelM128S4 *probe_kernel_m128_s4()
{
    static ProbeKernelM128S4 *kernel = new ProbeKernelM128S4();
    return kernel;
}

template<class Kernel>
int launch_ds4_mxfp4_matmul(
    Kernel*             kernel,
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    int                n,
    int                k,
    bool               gated_silu,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (!A || !expert_offsets || !weights_packed || !scales_packed || !D) return 1;
    if (!kernel || num_experts <= 0 || total_tokens <= 0) return 2;

    constexpr int group_size = 32;

    cudaStream_t stream = (cudaStream_t)stream_v;

    tmg::Workspace workspace{};
    workspace.barriers        = barriers;
    workspace.barriers_size   = barriers_size;
    workspace.partials        = partials;
    workspace.partials_size   = partials_size;
    workspace.tensormaps      = nullptr;
    workspace.tensormaps_size = 0;
    workspace.flags           = flags;

    tmg::Operation op{};
    op.dispatch  = tmg::DispatchPolicy::kDefault;
    op.epilogue  = gated_silu ? tmg::Epilogue::kGatedSilu : tmg::Epilogue::kNone;
    op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
    op.quant_b   = tmg::QuantDesc{tmg::QuantType::kK, group_size};
    op.batch_dim = 0;

    tmg::MatrixLayout Adesc{};
    Adesc.type    = turbomind::kHalf;
    Adesc.order   = tmg::Order::kRowMajor;
    Adesc.rows    = total_tokens;
    Adesc.cols    = k;
    Adesc.ld      = k;
    Adesc.pack    = 0;
    Adesc.num     = num_experts;
    Adesc.offsets = const_cast<int*>(expert_offsets);
    Adesc.idxs    = nullptr;

    auto convs = tmg::GetConverters(
        turbomind::kHalf,
        turbomind::kFloat4_e2m1,
        turbomind::kHalf,
        false,
        70);
    const tmg::LayoutConverter* conv_w = convs[0];
    const tmg::LayoutConverter* conv_s = convs[1];
    if (!conv_w || !conv_s) return 3;

    tmg::MatrixLayout Bdesc{};
    Bdesc.type    = turbomind::kFloat4_e2m1;
    Bdesc.order   = conv_w->order;
    Bdesc.rows    = n;
    Bdesc.cols    = k;
    Bdesc.ld      = 0;
    Bdesc.pack    = (tmg::Pack)((uint32_t)k_pack_value & 0xFFFu);
    Bdesc.num     = num_experts;
    if (tmg::get_operand_tag(conv_w->pack) != tmg::OPERAND_A) {
        std::swap(Bdesc.rows, Bdesc.cols);
        Bdesc.order = ~Bdesc.order;
    }

    uint32_t v_pack_raw = ((uint32_t)k_pack_value >> 12) & 0xFFFu;
    if (v_pack_raw == 0) {
        v_pack_raw = ((uint32_t)k_pack_value & 0xF0Fu) | (uint32_t)tmg::OPERAND_V;
    }

    tmg::MatrixLayout Vdesc{};
    Vdesc.type    = turbomind::kUint8;
    Vdesc.order   = conv_s->order;
    Vdesc.rows    = n;
    Vdesc.cols    = k / group_size;
    Vdesc.ld      = 0;
    Vdesc.pack    = (tmg::Pack)v_pack_raw;
    Vdesc.num     = num_experts;
    if (tmg::get_operand_tag(conv_s->pack) != tmg::OPERAND_U) {
        std::swap(Vdesc.rows, Vdesc.cols);
        Vdesc.order = ~Vdesc.order;
    }

    tmg::MatrixLayout Ddesc{};
    Ddesc.type    = turbomind::kHalf;
    Ddesc.order   = tmg::Order::kRowMajor;
    Ddesc.rows    = total_tokens;
    Ddesc.cols    = n;
    Ddesc.ld      = gated_silu ? n / 2 : n;
    Ddesc.pack    = 0;
    Ddesc.num     = num_experts;
    Ddesc.offsets = const_cast<int*>(expert_offsets);
    Ddesc.idxs    = nullptr;

    tmg::MatrixLayout Cdesc = Ddesc;
    tmg::MatrixLayout Udesc{};

    return kernel->Launch(
        op,
        1.0f,
        A,
        Adesc,
        nullptr,
        Udesc,
        weights_packed,
        Bdesc,
        scales_packed,
        Vdesc,
        0.0f,
        nullptr,
        Cdesc,
        D,
        Ddesc,
        0,
        1,
        workspace,
        stream);
}

template<class Gemm>
int launch_ds4_mxfp4_down_reduce_epilogue(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    float*             route_out,
    const int*         sorted_pairs,
    const float*       route_weights,
    int                n_routes,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (!A || !expert_offsets || !weights_packed || !scales_packed || !route_out ||
        !sorted_pairs || !route_weights) {
        return 1;
    }
    if (num_experts != 6 ||
        (total_tokens != 6 && total_tokens != 768 && total_tokens != 1536) ||
        n_routes <= 0) {
        return 2;
    }

    constexpr int n = 4096;
    constexpr int k = 2048;
    constexpr int group_size = 32;

    cudaStream_t stream = (cudaStream_t)stream_v;

    tmg::MatrixLayout Adesc{};
    Adesc.type    = turbomind::kHalf;
    Adesc.order   = tmg::Order::kRowMajor;
    Adesc.rows    = total_tokens;
    Adesc.cols    = k;
    Adesc.ld      = k;
    Adesc.pack    = 0;
    Adesc.num     = num_experts;
    Adesc.offsets = const_cast<int*>(expert_offsets);
    Adesc.idxs    = nullptr;

    auto convs = tmg::GetConverters(
        turbomind::kHalf,
        turbomind::kFloat4_e2m1,
        turbomind::kHalf,
        false,
        70);
    const tmg::LayoutConverter* conv_w = convs[0];
    const tmg::LayoutConverter* conv_s = convs[1];
    if (!conv_w || !conv_s) return 3;

    tmg::MatrixLayout Bdesc{};
    Bdesc.type    = turbomind::kFloat4_e2m1;
    Bdesc.order   = conv_w->order;
    Bdesc.rows    = n;
    Bdesc.cols    = k;
    Bdesc.ld      = 0;
    Bdesc.pack    = (tmg::Pack)((uint32_t)k_pack_value & 0xFFFu);
    Bdesc.num     = num_experts;
    if (tmg::get_operand_tag(conv_w->pack) != tmg::OPERAND_A) {
        std::swap(Bdesc.rows, Bdesc.cols);
        Bdesc.order = ~Bdesc.order;
    }

    uint32_t v_pack_raw = ((uint32_t)k_pack_value >> 12) & 0xFFFu;
    if (v_pack_raw == 0) {
        v_pack_raw = ((uint32_t)k_pack_value & 0xF0Fu) | (uint32_t)tmg::OPERAND_V;
    }

    tmg::MatrixLayout Vdesc{};
    Vdesc.type    = turbomind::kUint8;
    Vdesc.order   = conv_s->order;
    Vdesc.rows    = n;
    Vdesc.cols    = k / group_size;
    Vdesc.ld      = 0;
    Vdesc.pack    = (tmg::Pack)v_pack_raw;
    Vdesc.num     = num_experts;
    if (tmg::get_operand_tag(conv_s->pack) != tmg::OPERAND_U) {
        std::swap(Vdesc.rows, Vdesc.cols);
        Vdesc.order = ~Vdesc.order;
    }

    tmg::MatrixLayout Ddesc{};
    Ddesc.type    = turbomind::kHalf;
    Ddesc.order   = tmg::Order::kRowMajor;
    Ddesc.rows    = total_tokens;
    Ddesc.cols    = n;
    Ddesc.ld      = n;
    Ddesc.pack    = 0;
    Ddesc.num     = num_experts;
    Ddesc.offsets = const_cast<int*>(expert_offsets);
    Ddesc.idxs    = nullptr;

    using Sched = typename Gemm::Scheduler;
    Sched sched{{Ddesc.rows, Ddesc.cols, Adesc.cols, std::max(1, Ddesc.num)}, 1, 1};
    sched.offsets_ = Ddesc.offsets;

    tmg::MatrixLayout Pdesc = Ddesc;
    Pdesc.ld = tmg::mk2cs<Gemm::kOrderC>(Pdesc.rows, Pdesc.cols).x;

    tmg::MatrixCombination_v3 combin_mat{
        tmg::to_param(nullptr, Ddesc),
        1.0f,
        0.0f,
    };

    tmg::EpilogueParam epilogue{
        tmg::to_param(route_out, Ddesc),
        tmg::to_param(partials, Pdesc),
        (int*)barriers,
        combin_mat,
        false,
        false,
        true,
        route_out,
        sorted_pairs,
        route_weights,
        n_routes,
        n,
    };

    tmg::GemmParam param{
        tmg::to_param((void*)A, Adesc),
        tmg::to_param((void*)weights_packed, Bdesc),
        tmg::MatrixParam{},
        tmg::to_param((void*)scales_packed, Vdesc),
    };

    constexpr size_t smem_size = sizeof(typename Gemm::SharedStorage);
    const auto grid = sched.get_grid_shape();
    const auto block = Gemm::Impl::WARPS * WARP_SIZE;
    tmg::gemm_kernel<Gemm><<<grid, block, smem_size, stream>>>(param, epilogue, sched);
    return 0;
}

template<class Kernel>
int launch_ds4_mxfp4_gated_silu(
    Kernel*             kernel,
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    return launch_ds4_mxfp4_matmul(kernel,
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   4096,
                                   true,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

}  // namespace

int ggml_turbomind_ds4_reduce6_half_to_float_launch(
    const void*  down_routes_half,
    float*       route_out,
    const int*   sorted_pairs,
    const float* route_weights,
    int          n_routes,
    int          hidden,
    void*        stream_v)
{
    if (!down_routes_half || !route_out || !sorted_pairs || !route_weights) return 1;
    if (n_routes != 6 || hidden != 4096) return 2;
    cudaStream_t stream = (cudaStream_t)stream_v;
    ds4_probe_kernels::ds4_reduce6_half_to_float_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(
        (const half*)down_routes_half,
        route_out,
        sorted_pairs,
        route_weights,
        n_routes,
        hidden);
    return 0;
}

// Per-request 6-route decode shape: the production per-step async pipeline
// calls the routed FFN one slot at a time, so each active expert receives a
// single token (total_tokens == num_experts == 6). Reuses the SM70 M16 MXFP4
// probe kernel; M16 handles the partial 6-row M tile via predication.
int ggml_turbomind_ds4_mxfp4_gated_silu_6_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 6) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m16(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_96_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 96) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m16(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m64_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m64_s4_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64_s4(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m64_s3_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64_s3(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_s3_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128_s3(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_s4_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128_s4(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m64_s3_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64_s3(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m64_s4_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64_s4(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_s3_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128_s3(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_s4_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128_s4(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m64n256_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m64n256(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_768_m128_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   2048,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_768_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 768) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gate_up_768_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 768) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   4096,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_768_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 768) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   2048,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_1536_m128_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   2048,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_gated_silu_1536_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_gated_silu(probe_kernel_m128(),
                                       A,
                                       expert_offsets,
                                       num_experts,
                                       total_tokens,
                                       weights_packed,
                                       scales_packed,
                                       k_pack_value,
                                       D,
                                       barriers,
                                       barriers_size,
                                       partials,
                                       partials_size,
                                       flags,
                                       stream_v);
}

int ggml_turbomind_ds4_mxfp4_gate_up_1536_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   4096,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_1536_m128_group_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (num_experts != 1 || total_tokens != 1536) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m128(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   2048,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_768_m64n256_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    void*              D,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    return launch_ds4_mxfp4_matmul(probe_kernel_m64n256(),
                                   A,
                                   expert_offsets,
                                   num_experts,
                                   total_tokens,
                                   weights_packed,
                                   scales_packed,
                                   k_pack_value,
                                   4096,
                                   2048,
                                   false,
                                   D,
                                   barriers,
                                   barriers_size,
                                   partials,
                                   partials_size,
                                   flags,
                                   stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_768_m128_reduce_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    float*             route_out,
    const int*         sorted_pairs,
    const float*       route_weights,
    int                n_routes,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 768) return 2;
    (void)probe_kernel_m128();
    return launch_ds4_mxfp4_down_reduce_epilogue<typename ProbeConfigM128::Kernel>(
        A,
        expert_offsets,
        num_experts,
        total_tokens,
        weights_packed,
        scales_packed,
        k_pack_value,
        route_out,
        sorted_pairs,
        route_weights,
        n_routes,
        barriers,
        barriers_size,
        partials,
        partials_size,
        flags,
        stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_6_m16_reduce_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    float*             route_out,
    const int*         sorted_pairs,
    const float*       route_weights,
    int                n_routes,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 6) return 2;
    (void)probe_kernel_m16();
    return launch_ds4_mxfp4_down_reduce_epilogue<typename ProbeConfigM16::Kernel>(
        A,
        expert_offsets,
        num_experts,
        total_tokens,
        weights_packed,
        scales_packed,
        k_pack_value,
        route_out,
        sorted_pairs,
        route_weights,
        n_routes,
        barriers,
        barriers_size,
        partials,
        partials_size,
        flags,
        stream_v);
}

int ggml_turbomind_ds4_mxfp4_down_1536_m128_reduce_launch(
    const void*        A,
    const int*         expert_offsets,
    int                num_experts,
    int                total_tokens,
    const void* const* weights_packed,
    const void* const* scales_packed,
    int                k_pack_value,
    float*             route_out,
    const int*         sorted_pairs,
    const float*       route_weights,
    int                n_routes,
    void*              barriers,
    size_t             barriers_size,
    void*              partials,
    size_t             partials_size,
    int*               flags,
    void*              stream_v)
{
    if (total_tokens != 1536) return 2;
    (void)probe_kernel_m128();
    return launch_ds4_mxfp4_down_reduce_epilogue<typename ProbeConfigM128::Kernel>(
        A,
        expert_offsets,
        num_experts,
        total_tokens,
        weights_packed,
        scales_packed,
        k_pack_value,
        route_out,
        sorted_pairs,
        route_weights,
        n_routes,
        barriers,
        barriers_size,
        partials,
        partials_size,
        flags,
        stream_v);
}
