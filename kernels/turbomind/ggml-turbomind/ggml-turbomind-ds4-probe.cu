// DS4/V100 fixed-shape TurboMind probe kernels.

#include <cuda_runtime.h>

#include <algorithm>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/kernel_impl.h"

namespace tmg = turbomind::gemm;

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
using ProbeConfigM128 = Config_MXF4<kColMajor, 0>::Type<
    128, 128, 16, 2, 2, 1,
    turbomind::cache_policy::Default,
    turbomind::cache_policy::Default,
    2, true, 1, 32, 64, 128>;

using ProbeKernelM16 = KernelImpl<typename ProbeConfigM16::Kernel>;
using ProbeKernelM64 = KernelImpl<typename ProbeConfigM64::Kernel>;
using ProbeKernelM128 = KernelImpl<typename ProbeConfigM128::Kernel>;

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

ProbeKernelM128 *probe_kernel_m128()
{
    static ProbeKernelM128 *kernel = new ProbeKernelM128();
    return kernel;
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
    if (!A || !expert_offsets || !weights_packed || !scales_packed || !D) return 1;
    if (!kernel || num_experts != 6 || total_tokens <= 0) return 2;

    constexpr int K = 4096;
    constexpr int N = 4096;
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
    op.epilogue  = tmg::Epilogue::kGatedSilu;
    op.quant_a   = tmg::QuantDesc{tmg::QuantType::kNone, 0};
    op.quant_b   = tmg::QuantDesc{tmg::QuantType::kK, group_size};
    op.batch_dim = 0;

    tmg::MatrixLayout Adesc{};
    Adesc.type    = turbomind::kHalf;
    Adesc.order   = tmg::Order::kRowMajor;
    Adesc.rows    = total_tokens;
    Adesc.cols    = K;
    Adesc.ld      = K;
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
    Bdesc.rows    = N;
    Bdesc.cols    = K;
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
    Vdesc.rows    = N;
    Vdesc.cols    = K / group_size;
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
    Ddesc.cols    = N;
    Ddesc.ld      = N / 2;
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

}  // namespace

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
