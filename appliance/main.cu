#define _FILE_OFFSET_BITS 64

#include "ds4_v100_tp_runtime.h"
#include "kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h"
extern "C" {
#include "ds4.h"
}

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <dlfcn.h>
#include <mma.h>
#if __has_include(<nccl.h>)
#include <nccl.h>
#else
#include "third_party/nccl_compat/nccl.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace {

#include "engine/runtime_types.cuh"
#include "engine/runtime_options.cuh"

#include "appliance/options.cu"
#include "engine/runtime_profiler.cu"

#include "kernels/v100/common.cuh"
#include "kernels/v100/dense.cuh"
#include "kernels/v100/hc_mix.cuh"
#include "kernels/v100/hc_shards.cuh"
#include "kernels/v100/norm.cuh"
#include "kernels/v100/compose.cuh"
#include "kernels/v100/router.cuh"
#include "kernels/v100/diagnostics.cuh"
#include "kernels/v100/fill_pack.cuh"
#include "kernels/v100/attention.cuh"


#include "engine/runtime_pack.cu"
#include "engine/router_step.cu"
#include "engine/hc_final.cu"
#include "engine/hc_current.cu"
#include "engine/output_head.cu"
#include "engine/ep_dense.cu"
#include "engine/turbomind_bindings.cu"
#include "engine/ep_executor.cu"
#include "engine/diagnostics_support.cu"
#include "engine/router_plan.cu"
#include "engine/runtime_resources.cu"
#include "engine/ep_compose.cu"
#include "engine/attention_projection.cu"
#include "engine/compressed_kv_step.cu"
#include "engine/attention_read.cu"
#include "engine/attention_output.cu"
#include "engine/post_attention_ffn.cu"
#include "engine/decode_loop.cu"

} // namespace

#include "engine/layer_decode.cu"
#include "engine/layer_runner.cu"
#include "engine/token_major_loop.cu"
#include "appliance/http_server.cu"
#include "engine/appliance_runtime.cu"
#include "appliance/entrypoint.cu"
