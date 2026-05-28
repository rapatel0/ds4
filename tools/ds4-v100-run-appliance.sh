#!/usr/bin/env bash
set -eu

env_file=""
mode="run"
allow_missing=0

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-run-appliance.sh [options]

Options:
  --env FILE        load deployment environment file
  --check           validate config and exit without starting the server
  --print-command   validate config, print the resolved command, and exit
  --allow-missing   allow missing model/GPU files during local config checks
  --help            show this help

The launcher reads DS4_V100_* variables from the environment file, validates
the deployment contract, then execs tools/ds4-v100-replay --serve.
USAGE
}

fail() {
    echo "ds4-v100-run-appliance: $*" >&2
    exit 1
}

warn() {
    echo "ds4-v100-run-appliance: warning: $*" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --env)
            [ "$#" -ge 2 ] || fail "--env requires a value"
            env_file="$2"
            shift 2
            ;;
        --check)
            mode="check"
            shift
            ;;
        --print-command)
            mode="print"
            shift
            ;;
        --allow-missing)
            allow_missing=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

if [ -n "$env_file" ]; then
    [ -f "$env_file" ] || fail "missing env file $env_file"
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
fi

: "${DS4_V100_BIN:=./tools/ds4-v100-replay}"
: "${DS4_V100_MODEL:=/models/DSv4-Flash-256e-fixed.gguf}"
: "${DS4_V100_MTP_MODEL:=/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}"
: "${DS4_V100_APPLIANCE_DIR:=}"
: "${DS4_V100_PACK_INDEX:=docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv}"
: "${DS4_V100_CTX:=1048576}"
: "${DS4_V100_SLOTS:=1}"
: "${DS4_V100_ACTIVE_MICROBATCH:=1}"
: "${DS4_V100_MICROBATCH_WAIT_US:=auto}"
: "${DS4_V100_QUEUE_POLICY:=reject-busy}"
: "${DS4_V100_TOKENS:=2}"
: "${DS4_V100_ASYNC_PIPELINE_MODE:=off}"
: "${DS4_V100_ASYNC_HANDOFF:=0}"
: "${DS4_V100_ASYNC_EVENT_HANDOFF:=auto}"
: "${DS4_V100_ASYNC_SLOT_CHUNK:=}"
: "${DS4_V100_ASYNC_FFN_WAVEFRONT:=0}"
: "${DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK:=2}"
: "${DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE:=0}"
: "${DS4_V100_STARTUP_WARMUP:=auto}"
: "${DS4_V100_CUDA_LIB_DIR:=auto}"
: "${DS4_V100_CUDA_PROFILER_WINDOW:=0}"
: "${DS4_V100_CUDA_TENSOR_POOL:=auto}"
: "${DS4_V100_CUDA_TENSOR_POOL_MAX_MIB:=2048}"
: "${DS4_V100_CUDA_F8_ROWPAIR:=1}"
: "${DS4_V100_CUDA_F8_ROW4:=0}"
: "${DS4_V100_CUDA_F8_WARP_SCALE:=0}"
: "${DS4_V100_CUDA_F8_GROUPED_DS4_FAST:=1}"
: "${DS4_V100_CUDA_F8_HMMA_SHARED_DOWN:=0}"
: "${DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU:=1}"
: "${DS4_V100_CUDA_F8_HMMA_ATTN_BATCH:=1}"
: "${DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH:=0}"
: "${DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE:=0}"
: "${DS4_V100_CUDA_F8_HMMA_SINGLE:=0}"
: "${DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE:=0}"
: "${DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2:=0}"
: "${DS4_V100_F8_SHARED_DOWN_ADD:=0}"
: "${DS4_V100_ENABLE_BATCH_ATTN_PROJ:=1}"
: "${DS4_V100_BATCH_ATTN_OUTPUT_A:=0}"
: "${DS4_V100_BATCH_ATTN_OUTPUT_B:=0}"
: "${DS4_V100_ENABLE_OUTPUT_HEAD_BATCH:=0}"
: "${DS4_V100_BATCH_SHARED_F8:=1}"
: "${DS4_V100_FFN_DIRECT_DELTA:=0}"
: "${DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A:=0}"
: "${DS4_V100_SINGLE_SLOT_ATTN_SCRATCH:=1}"
: "${DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA:=0}"
: "${DS4_V100_TURBOMIND_ROUTED_FFN:=0}"
: "${DS4_V100_TURBOMIND_STRICT:=0}"
: "${DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS:=1}"
: "${DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC:=0}"
: "${DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD:=0}"
: "${DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE:=0}"
: "${DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2:=0}"
: "${DS4_V100_TURBOMIND_INDEXED_A:=0}"
: "${DS4_V100_TURBOMIND_FUSED_GATE_UP:=1}"
: "${DS4_V100_TURBOMIND_GATED_SILU:=1}"
: "${DS4_V100_TURBOMIND_COMPACT_SCHEDULE:=1}"
: "${DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC:=0}"
: "${DS4_V100_TURBOMIND_ROUTED_EXECUTOR:=fused6_reduce}"
: "${DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE:=0}"
: "${DS4_V100_TURBOMIND_GATE_UP_PROBE:=auto}"
: "${DS4_V100_TURBOMIND_DOWN_PROBE:=off}"
: "${DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE:=0}"
: "${DS4_V100_TURBOMIND_DISPATCH_POLICY:=default}"
: "${DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE:=0}"
: "${DS4_V100_TURBOMIND_GROUP_PIPELINE:=0}"
: "${DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS:=8}"
: "${DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS:=0}"
: "${DS4_V100_TURBOMIND_GRAPH:=1}"
: "${DS4_V100_TURBOMIND_GRAPH_VERBOSE:=0}"
: "${DS4_V100_TURBOMIND_PROFILE:=0}"
: "${DS4_V100_TP_EP_ROUTED_FFN:=0}"
: "${DS4_V100_TP_EP_LAYER_FIRST:=}"
: "${DS4_V100_TP_EP_LAYER_COUNT:=1}"
: "${DS4_V100_TP_EP_PEER:=}"
: "${DS4_V100_TP_EP_SHARD_DIR:=}"
: "${DS4_V100_TP_EP_ASYNC_INPUT:=0}"
: "${DS4_V100_TP_EP_PARALLEL_HALVES:=0}"
: "${DS4_V100_TP_EP_COPY_EVENT_COMPOSE:=1}"
: "${DS4_V100_TP_EP_RETURN_FP16:=0}"
: "${DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE:=1}"
: "${DS4_V100_TP_EP_COMPACT_MOE_DECODE:=1}"
: "${DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE:=auto}"
: "${DS4_V100_TP_EP_FUSED_GATED_SILU:=0}"
: "${DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD:=1}"
: "${DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY:=1}"
: "${DS4_V100_TP_EP_DECODE_CUDAGRAPH:=0}"
: "${DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC:=0}"
: "${DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC:=0}"
: "${DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC:=}"
if [ "${DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE+x}" != x ]; then
    DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE=
fi
: "${DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT:=0}"
: "${DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM:=0}"
: "${DS4_V100_TP_EP_HC_FINAL_EXPAND:=1}"
: "${DS4_V100_TP_EP_HC_CURRENT_INPUT:=1}"
: "${DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER:=0}"
: "${DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER:=1}"
: "${DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE:=1}"
: "${DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY:=0}"
: "${DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC:=1}"
: "${DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK:=0}"
: "${DS4_V100_TP_EP_PEER_ACCOUNTING:=0}"
: "${DS4_V100_TP_EP_PEER_REJECT_SYS:=0}"
: "${DS4_V100_TP_EP_HC_PERSIST_STATE:=1}"
: "${DS4_V100_TP_EP_MODEL_ROUTER_ROUTES:=1}"
: "${DS4_V100_TP_EP_ROUTER_CUBLAS:=0}"
: "${DS4_V100_TP_EP_ROUTER_HASH_FAST:=0}"
: "${DS4_V100_TP_EP_GPU_ROUTE_PLAN:=1}"
: "${DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD:=1}"
: "${DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT:=1}"
: "${DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT:=0}"
: "${DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS:=0}"
: "${DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS:=1}"
: "${DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN:=1}"
: "${DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC:=0}"
: "${DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM:=0}"
: "${DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM:=0}"
: "${DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY:=0}"
: "${DS4_V100_TP_EP_TRUE_SHARED_FFN:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS:=auto}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC:=1}"
: "${DS4_V100_TP_EP_EXTRA_ARGS:=}"
: "${DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE:=1}"
: "${DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB:=1024}"
: "${DS4_V100_TP_EP_DEFER_NCCL_INIT:=1}"
: "${DS4_V100_NCCL_TOPOLOGY_POLICY:=no-sys}"
: "${DS4_V100_NCCL_NO_SYS_RING:=0 3 2 1 5 7 6 4}"
: "${DS4_V100_NCCL_ALLOW_VISIBLE_REMAP:=0}"
: "${DS4_V100_NCCL_ALGO:=auto}"
: "${DS4_V100_NCCL_PROTO:=auto}"
: "${DS4_V100_TP_EP_FP8_E5M2_KV:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND:=0}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM:=1}"
: "${DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND:=0}"
: "${DS4_V100_TP_EP_REFERENCE_HC_REDUCE:=0}"
: "${DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD:=0}"
: "${DS4_V100_TP_EP_KV_ALL_SLOTS:=0}"
: "${DS4_V100_TP_EP_VRAM_REPORT:=0}"
: "${DS4_V100_TP_EP_VRAM_MIN_FREE_MIB:=64}"
: "${DS4_V100_TP_EP_NCCL_MIN_FREE_MIB:=}"
: "${DS4_V100_TP_EP_VERBOSE:=0}"
: "${DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD:=1}"
: "${DS4_V100_TP_EP_BIN:=./appliance/ds4-v100-tp-ep-appliance}"
: "${DS4_V100_TP_EP_CONTRACT:=/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv}"
: "${DS4_V100_TP_EP_TM_INDEX:=}"
: "${DS4_V100_TP_EP_TOKENIZER_MODEL:=$DS4_V100_MODEL}"
: "${DS4_V100_TP_EP_TOP_K:=6}"
: "${DS4_V100_TP_EP_KV_SLOT:=7}"
: "${DS4_V100_TP_EP_POSITION:=100000}"
: "${DS4_V100_TURBOMIND_LIB:=./build/turbomind-v100/libggml-turbomind.so}"
: "${DS4_V100_HOST:=127.0.0.1}"
: "${DS4_V100_PORT:=18080}"
: "${DS4_V100_ALLOW_NONLOCAL_HOST:=0}"
: "${DS4_V100_CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${DS4_V100_REQUIRE_GPUS:=8}"
: "${DS4_V100_RESERVE_MIB:=4096}"
: "${DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP:=}"
: "${DS4_V100_MAX_REQUESTS:=0}"
: "${DS4_V100_LOG_DIR:=logs/v100-appliance}"
: "${DS4_LOCK_FILE:=$DS4_V100_LOG_DIR/ds4.lock}"
: "${DS4_V100_SERVE_MODE:=base}"
if [ "${DS4_V100_SERVE_MODE_LOCK:-}" = "tp-ep" ] ||
   [ "${DS4_V100_SERVE_MODE_LOCK:-}" = "base" ]; then
    DS4_V100_SERVE_MODE="$DS4_V100_SERVE_MODE_LOCK"
elif [ -n "${DS4_V100_SERVE_MODE_LOCK:-}" ]; then
    fail "DS4_V100_SERVE_MODE_LOCK must be base or tp-ep"
fi
: "${DS4_V100_MTP_SERVING:=off}"
: "${DS4_V100_MTP_TOP_K:=5}"
: "${DS4_V100_MTP_GPU:=7}"

is_uint() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

require_file() {
    local label="$1"
    local path="$2"
    if [ -f "$path" ]; then
        return 0
    fi
    if [ "$allow_missing" -eq 1 ]; then
        warn "missing $label $path"
        return 0
    fi
    fail "missing $label $path"
}

require_dir() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        return 0
    fi
    if [ "$allow_missing" -eq 1 ]; then
        warn "missing $label $path"
        return 0
    fi
    fail "missing $label $path"
}

require_exec() {
    local path="$1"
    if [ -x "$path" ]; then
        return 0
    fi
    if [ "$allow_missing" -eq 1 ]; then
        warn "missing executable $path"
        return 0
    fi
    fail "missing executable $path"
}

visible_gpu_count() {
    if [ -z "$DS4_V100_CUDA_VISIBLE_DEVICES" ] || [ "$DS4_V100_CUDA_VISIBLE_DEVICES" = "all" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
        else
            echo 0
        fi
        return
    fi
    printf '%s\n' "$DS4_V100_CUDA_VISIBLE_DEVICES" | awk -F, '
        {
            n = 0
            for (i = 1; i <= NF; i++) if ($i != "") n++
            print n
        }'
}

check_gpu_reserve() {
    [ "$DS4_V100_REQUIRE_GPUS" -eq 0 ] && return 0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [ "$allow_missing" -eq 1 ]; then
            warn "nvidia-smi not found; skipping GPU checks"
            return 0
        fi
        fail "nvidia-smi not found"
    fi

    local count
    count="$(visible_gpu_count)"
    if [ "$count" -lt "$DS4_V100_REQUIRE_GPUS" ]; then
        fail "requires $DS4_V100_REQUIRE_GPUS visible GPUs, got $count"
    fi

    [ "$DS4_V100_RESERVE_MIB" -eq 0 ] && return 0
    local free_values
    free_values="$(CUDA_VISIBLE_DEVICES="$DS4_V100_CUDA_VISIBLE_DEVICES" \
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "$free_values" ]; then
        if [ "$allow_missing" -eq 1 ]; then
            warn "could not query GPU free memory"
            return 0
        fi
        fail "could not query GPU free memory"
    fi
    local checked=0
    local free_mib
    while IFS= read -r free_mib; do
        free_mib="$(printf '%s' "$free_mib" | tr -dc '0-9')"
        [ -z "$free_mib" ] && continue
        checked=$((checked + 1))
        if [ "$free_mib" -lt "$DS4_V100_RESERVE_MIB" ]; then
            fail "GPU reserve check failed: ${free_mib} MiB free < ${DS4_V100_RESERVE_MIB} MiB"
        fi
        [ "$checked" -ge "$DS4_V100_REQUIRE_GPUS" ] && break
    done <<EOF
$free_values
EOF
    if [ "$checked" -lt "$DS4_V100_REQUIRE_GPUS" ]; then
        fail "GPU reserve check saw $checked GPUs, expected $DS4_V100_REQUIRE_GPUS"
    fi
}

case "$DS4_V100_SERVE_MODE" in
    base|tp-ep) ;;
    *) fail "DS4_V100_SERVE_MODE must be base or tp-ep" ;;
esac
mtp_serving_enabled=0
case "$DS4_V100_MTP_SERVING" in
    off|false|0) ;;
    verify|commit) mtp_serving_enabled=1 ;;
    *) fail "DS4_V100_MTP_SERVING must be off, verify, or commit" ;;
esac

is_uint "$DS4_V100_CTX" || fail "DS4_V100_CTX must be a positive integer"
is_uint "$DS4_V100_SLOTS" || fail "DS4_V100_SLOTS must be a positive integer"
is_uint "$DS4_V100_ACTIVE_MICROBATCH" || fail "DS4_V100_ACTIVE_MICROBATCH must be a positive integer"
if [ "$DS4_V100_MICROBATCH_WAIT_US" != "auto" ]; then
    is_uint "$DS4_V100_MICROBATCH_WAIT_US" || fail "DS4_V100_MICROBATCH_WAIT_US must be auto or an integer"
fi
if [ -n "$DS4_V100_ASYNC_SLOT_CHUNK" ]; then
    is_uint "$DS4_V100_ASYNC_SLOT_CHUNK" || fail "DS4_V100_ASYNC_SLOT_CHUNK must be empty or a positive integer"
fi
is_uint "$DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK" || fail "DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK must be a positive integer"
is_uint "$DS4_V100_TOKENS" || fail "DS4_V100_TOKENS must be a positive integer"
is_uint "$DS4_V100_PORT" || fail "DS4_V100_PORT must be a positive integer"
is_uint "$DS4_V100_ALLOW_NONLOCAL_HOST" || fail "DS4_V100_ALLOW_NONLOCAL_HOST must be an integer"
is_uint "$DS4_V100_REQUIRE_GPUS" || fail "DS4_V100_REQUIRE_GPUS must be an integer"
is_uint "$DS4_V100_RESERVE_MIB" || fail "DS4_V100_RESERVE_MIB must be an integer"
if [ -n "$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP" ]; then
    is_uint "$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP" || fail "DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP must be empty or an integer"
fi
is_uint "$DS4_V100_MAX_REQUESTS" || fail "DS4_V100_MAX_REQUESTS must be an integer"
is_uint "$DS4_V100_MTP_TOP_K" || fail "DS4_V100_MTP_TOP_K must be an integer"
is_uint "$DS4_V100_MTP_GPU" || fail "DS4_V100_MTP_GPU must be an integer"
is_uint "$DS4_V100_TP_EP_LAYER_COUNT" || fail "DS4_V100_TP_EP_LAYER_COUNT must be an integer"
is_uint "$DS4_V100_TP_EP_TOP_K" || fail "DS4_V100_TP_EP_TOP_K must be an integer"
is_uint "$DS4_V100_TP_EP_KV_SLOT" || fail "DS4_V100_TP_EP_KV_SLOT must be an integer"
is_uint "$DS4_V100_TP_EP_POSITION" || fail "DS4_V100_TP_EP_POSITION must be an integer"
if [ -n "$DS4_V100_TP_EP_LAYER_FIRST" ]; then
    is_uint "$DS4_V100_TP_EP_LAYER_FIRST" || fail "DS4_V100_TP_EP_LAYER_FIRST must be empty or an integer"
fi
if [ -n "$DS4_V100_TP_EP_PEER" ]; then
    is_uint "$DS4_V100_TP_EP_PEER" || fail "DS4_V100_TP_EP_PEER must be empty or an integer"
fi

[ "$DS4_V100_CTX" -ge 1 ] || fail "DS4_V100_CTX must be positive"
[ "$DS4_V100_SLOTS" -ge 1 ] && [ "$DS4_V100_SLOTS" -le 256 ] || fail "DS4_V100_SLOTS must be between 1 and 256"
[ "$DS4_V100_ACTIVE_MICROBATCH" -ge 1 ] || fail "DS4_V100_ACTIVE_MICROBATCH must be positive"
[ "$DS4_V100_ACTIVE_MICROBATCH" -le "$DS4_V100_SLOTS" ] || fail "DS4_V100_ACTIVE_MICROBATCH must be in [1,DS4_V100_SLOTS]"
ctx_cap_startup_warmup=0
case "$DS4_V100_STARTUP_WARMUP" in
    auto)
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            ctx_cap_startup_warmup=1
        fi
        ;;
    0|false|off) ctx_cap_startup_warmup=0 ;;
    1|true|on) ctx_cap_startup_warmup=1 ;;
    *) fail "DS4_V100_STARTUP_WARMUP must be auto, 0, or 1" ;;
esac
ctx_slot_cap=256
if [ "$DS4_V100_CTX" -gt 524288 ]; then
    ctx_slot_cap=7
elif [ "$DS4_V100_CTX" -gt 262144 ]; then
    ctx_slot_cap=14
elif [ "$DS4_V100_CTX" -gt 131072 ]; then
    ctx_slot_cap=32
elif [ "$DS4_V100_CTX" -gt 65536 ]; then
    ctx_slot_cap=32
elif [ "$DS4_V100_CTX" -gt 32768 ]; then
    ctx_slot_cap=64
elif [ "$DS4_V100_CTX" -gt 16384 ]; then
    ctx_slot_cap=128
fi
if [ -n "$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP" ]; then
    [ "$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP" -ge 1 ] || fail "DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP must be positive"
    [ "$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP" -le 256 ] || fail "DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP must be <= 256"
    ctx_slot_cap="$DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP"
    warn "using experimental ctx slot cap override: $ctx_slot_cap"
fi
[ "$DS4_V100_SLOTS" -le "$ctx_slot_cap" ] || fail "DS4_V100_SLOTS=$DS4_V100_SLOTS exceeds ctx=$DS4_V100_CTX admission cap $ctx_slot_cap"
if [ "$DS4_V100_MICROBATCH_WAIT_US" != "auto" ]; then
    [ "$DS4_V100_MICROBATCH_WAIT_US" -le 1000000 ] || fail "DS4_V100_MICROBATCH_WAIT_US must be <= 1000000"
fi
if [ -n "$DS4_V100_ASYNC_SLOT_CHUNK" ]; then
    [ "$DS4_V100_ASYNC_SLOT_CHUNK" -ge 1 ] || fail "DS4_V100_ASYNC_SLOT_CHUNK must be positive"
    [ "$DS4_V100_ASYNC_SLOT_CHUNK" -le 256 ] || fail "DS4_V100_ASYNC_SLOT_CHUNK must be <= 256"
fi
case "$DS4_V100_ASYNC_FFN_WAVEFRONT" in
    0|false|off) DS4_V100_ASYNC_FFN_WAVEFRONT=0 ;;
    1|true|on) DS4_V100_ASYNC_FFN_WAVEFRONT=1 ;;
    *) fail "DS4_V100_ASYNC_FFN_WAVEFRONT must be 0 or 1" ;;
esac
case "$DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE" in
    0|false|off) DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE=0 ;;
    1|true|on) DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE=1 ;;
    *) fail "DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE must be 0 or 1" ;;
esac
[ "$DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK" -ge 1 ] || fail "DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK must be positive"
[ "$DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK" -le 256 ] || fail "DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK must be <= 256"
is_uint "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be an integer"
[ "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" -ge 64 ] || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be >= 64"
[ "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" -le 8192 ] || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be <= 8192"
[ "$DS4_V100_TOKENS" -ge 1 ] || fail "DS4_V100_TOKENS must be positive"
[ "$DS4_V100_TOKENS" -le 64 ] || fail "DS4_V100_TOKENS must be <= 64"
[ "$DS4_V100_MTP_TOP_K" -ge 2 ] && [ "$DS4_V100_MTP_TOP_K" -le 16 ] || fail "DS4_V100_MTP_TOP_K must be between 2 and 16"
[ "$DS4_V100_TP_EP_LAYER_COUNT" -ge 1 ] || fail "DS4_V100_TP_EP_LAYER_COUNT must be positive"
[ "$DS4_V100_TP_EP_TOP_K" -ge 1 ] && [ "$DS4_V100_TP_EP_TOP_K" -le 16 ] || fail "DS4_V100_TP_EP_TOP_K must be in [1,16]"
[ "$DS4_V100_TP_EP_KV_SLOT" -ge 0 ] || fail "DS4_V100_TP_EP_KV_SLOT must be non-negative"
if [ -n "$DS4_V100_TP_EP_LAYER_FIRST" ]; then
    [ "$DS4_V100_TP_EP_LAYER_FIRST" -le 42 ] || fail "DS4_V100_TP_EP_LAYER_FIRST must be in [0,42]"
    [ $((DS4_V100_TP_EP_LAYER_FIRST + DS4_V100_TP_EP_LAYER_COUNT)) -le 43 ] || fail "DS4_V100_TP_EP layer span exceeds [0,42]"
fi
if [ -n "$DS4_V100_TP_EP_PEER" ]; then
    [ "$DS4_V100_TP_EP_PEER" -le 7 ] || fail "DS4_V100_TP_EP_PEER must be in [0,7]"
fi
[ "$DS4_V100_PORT" -ge 1 ] && [ "$DS4_V100_PORT" -le 65535 ] || fail "DS4_V100_PORT out of range"
[ -n "$DS4_V100_HOST" ] || fail "DS4_V100_HOST must not be empty"
case "$DS4_V100_HOST" in
    127.*|localhost) ;;
    *)
        [ "$DS4_V100_ALLOW_NONLOCAL_HOST" -eq 1 ] ||
            fail "non-loopback bind requires DS4_V100_ALLOW_NONLOCAL_HOST=1; got DS4_V100_HOST=$DS4_V100_HOST"
        ;;
esac
case "$DS4_V100_QUEUE_POLICY" in
    reject-busy|sequential) ;;
    *) fail "DS4_V100_QUEUE_POLICY must be reject-busy or sequential" ;;
esac
case "$DS4_V100_ASYNC_PIPELINE_MODE" in
    off|auto|per-step|per_step|persistent|mailbox|mbox) ;;
    *) fail "DS4_V100_ASYNC_PIPELINE_MODE must be off, auto, per-step, persistent, or mailbox" ;;
esac
case "$DS4_V100_ASYNC_HANDOFF" in
    0|false|off) async_handoff=0 ;;
    1|true|on) async_handoff=1 ;;
    *) fail "DS4_V100_ASYNC_HANDOFF must be 0 or 1" ;;
esac
case "$DS4_V100_ASYNC_EVENT_HANDOFF" in
    0|false|off|auto) ;;
    1|true|on) ;;
    *) fail "DS4_V100_ASYNC_EVENT_HANDOFF must be auto, 0, or 1" ;;
esac
case "$DS4_V100_STARTUP_WARMUP" in
    auto)
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            startup_warmup=1
        else
            startup_warmup=0
        fi
        ;;
    0|false|off) startup_warmup=0 ;;
    1|true|on) startup_warmup=1 ;;
    *) fail "DS4_V100_STARTUP_WARMUP must be auto, 0, or 1" ;;
esac
cuda_lib_dir=""
case "$DS4_V100_CUDA_LIB_DIR" in
    auto)
        if [ -d /localpool/ds4/cuda-12.2-link/lib64 ]; then
            cuda_lib_dir=/localpool/ds4/cuda-12.2-link/lib64
        elif [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib64" ]; then
            cuda_lib_dir="$CUDA_HOME/lib64"
        fi
        ;;
    "")
        cuda_lib_dir=""
        ;;
    *)
        [ -d "$DS4_V100_CUDA_LIB_DIR" ] || fail "DS4_V100_CUDA_LIB_DIR does not exist: $DS4_V100_CUDA_LIB_DIR"
        cuda_lib_dir="$DS4_V100_CUDA_LIB_DIR"
        ;;
esac
case "$DS4_V100_CUDA_PROFILER_WINDOW" in
    0|false|off) cuda_profiler_window=0 ;;
    1|true|on) cuda_profiler_window=1 ;;
    *) fail "DS4_V100_CUDA_PROFILER_WINDOW must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_TENSOR_POOL" in
    auto)
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            cuda_tensor_pool=1
        else
            cuda_tensor_pool=0
        fi
        ;;
    0|false|off) cuda_tensor_pool=0 ;;
    1|true|on) cuda_tensor_pool=1 ;;
    *) fail "DS4_V100_CUDA_TENSOR_POOL must be auto, 0, or 1" ;;
esac
microbatch_wait_us="$DS4_V100_MICROBATCH_WAIT_US"
case "$microbatch_wait_us" in
    auto)
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -ge 16 ]; then
            microbatch_wait_us=200000
        elif [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            microbatch_wait_us=50000
        else
            microbatch_wait_us=0
        fi
        ;;
esac
case "$DS4_V100_TURBOMIND_ROUTED_FFN" in
    0|false|off) DS4_V100_TURBOMIND_ROUTED_FFN=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ROUTED_FFN=1 ;;
    *) fail "DS4_V100_TURBOMIND_ROUTED_FFN must be 0 or 1" ;;
esac
case "$DS4_V100_FFN_DIRECT_DELTA" in
    0|false|off) DS4_V100_FFN_DIRECT_DELTA=0 ;;
    1|true|on) DS4_V100_FFN_DIRECT_DELTA=1 ;;
    *) fail "DS4_V100_FFN_DIRECT_DELTA must be 0 or 1" ;;
esac
case "$DS4_V100_F8_SHARED_DOWN_ADD" in
    0|false|off) DS4_V100_F8_SHARED_DOWN_ADD=0 ;;
    1|true|on) DS4_V100_F8_SHARED_DOWN_ADD=1 ;;
    *) fail "DS4_V100_F8_SHARED_DOWN_ADD must be 0 or 1" ;;
esac
case "$DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A" in
    0|false|off) DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=0 ;;
    1|true|on) DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=1 ;;
    *) fail "DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A must be 0 or 1" ;;
esac
case "$DS4_V100_BATCH_ATTN_OUTPUT_B" in
    0|false|off) DS4_V100_BATCH_ATTN_OUTPUT_B=0 ;;
    1|true|on) DS4_V100_BATCH_ATTN_OUTPUT_B=1 ;;
    *) fail "DS4_V100_BATCH_ATTN_OUTPUT_B must be 0 or 1" ;;
esac
case "$DS4_V100_BATCH_ATTN_OUTPUT_A" in
    0|false|off) DS4_V100_BATCH_ATTN_OUTPUT_A=0 ;;
    1|true|on) DS4_V100_BATCH_ATTN_OUTPUT_A=1 ;;
    *) fail "DS4_V100_BATCH_ATTN_OUTPUT_A must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_ROWPAIR" in
    0|false|off) DS4_V100_CUDA_F8_ROWPAIR=0 ;;
    1|true|on) DS4_V100_CUDA_F8_ROWPAIR=1 ;;
    *) fail "DS4_V100_CUDA_F8_ROWPAIR must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_ROW4" in
    0|false|off) DS4_V100_CUDA_F8_ROW4=0 ;;
    1|true|on) DS4_V100_CUDA_F8_ROW4=1 ;;
    *) fail "DS4_V100_CUDA_F8_ROW4 must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_WARP_SCALE" in
    0|false|off) DS4_V100_CUDA_F8_WARP_SCALE=0 ;;
    1|true|on) DS4_V100_CUDA_F8_WARP_SCALE=1 ;;
    *) fail "DS4_V100_CUDA_F8_WARP_SCALE must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_GROUPED_DS4_FAST" in
    0|false|off) DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0 ;;
    1|true|on) DS4_V100_CUDA_F8_GROUPED_DS4_FAST=1 ;;
    *) fail "DS4_V100_CUDA_F8_GROUPED_DS4_FAST must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH" in
    0|false|off) DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=0 ;;
    1|true|on) DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1 ;;
    *) fail "DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH must be 0 or 1" ;;
esac
case "$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE" in
    0|false|off) DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE=0 ;;
    1|true|on) DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE=1 ;;
    *) fail "DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE must be 0 or 1" ;;
esac
case "$DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA" in
    0|false|off) DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA=0 ;;
    1|true|on) DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA=1 ;;
    *) fail "DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_STRICT" in
    0|false|off) DS4_V100_TURBOMIND_STRICT=0 ;;
    1|true|on) DS4_V100_TURBOMIND_STRICT=1 ;;
    *) fail "DS4_V100_TURBOMIND_STRICT must be 0 or 1" ;;
esac
case "$DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS" in
    0|false|off) DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=0 ;;
    1|true|on) DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1 ;;
    *) fail "DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC" in
    0|false|off) DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=1 ;;
    *) fail "DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD" in
    0|false|off) DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0 ;;
    1|true|on) DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1 ;;
    *) fail "DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE" in
    0|false|off) DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1 ;;
    *) fail "DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2" in
    0|false|off) DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=1 ;;
    *) fail "DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2 must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_INDEXED_A" in
    0|false|off) DS4_V100_TURBOMIND_INDEXED_A=0 ;;
    1|true|on) DS4_V100_TURBOMIND_INDEXED_A=1 ;;
    *) fail "DS4_V100_TURBOMIND_INDEXED_A must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_FUSED_GATE_UP" in
    0|false|off) DS4_V100_TURBOMIND_FUSED_GATE_UP=0 ;;
    1|true|on) DS4_V100_TURBOMIND_FUSED_GATE_UP=1 ;;
    *) fail "DS4_V100_TURBOMIND_FUSED_GATE_UP must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_GATED_SILU" in
    0|false|off) DS4_V100_TURBOMIND_GATED_SILU=0 ;;
    1|true|on) DS4_V100_TURBOMIND_GATED_SILU=1 ;;
    *) fail "DS4_V100_TURBOMIND_GATED_SILU must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_COMPACT_SCHEDULE" in
    0|false|off) DS4_V100_TURBOMIND_COMPACT_SCHEDULE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 ;;
    *) fail "DS4_V100_TURBOMIND_COMPACT_SCHEDULE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_ROUTED_EXECUTOR" in
    0|false|off|none) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off ;;
    1|true|on|auto) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=auto ;;
    fixed96|chain96|ffn96|96) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed96 ;;
    fixed768|chain768|ffn768|768) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed768 ;;
    fixed6|chain6|ffn6|6) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed6 ;;
    fused6|fused_6|ffn_fused6|unexpanded6|indexed6) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6 ;;
    fused6_reduce|fused_6_reduce|ffn_fused6_reduce|unexpanded6_reduce|indexed6_reduce|reduce6) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce ;;
    fused6_split_reduce|fused_6_split_reduce|ffn_fused6_split_reduce|split_reduce6|materialized6_reduce) DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_split_reduce ;;
    *) fail "DS4_V100_TURBOMIND_ROUTED_EXECUTOR must be off, auto, fixed96, fixed768, fixed6, fused6, fused6_reduce, or fused6_split_reduce" ;;
esac
case "$DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE" in
    0|false|off) DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=1 ;;
    *) fail "DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_GATE_UP_PROBE" in
    0|false|off|none) DS4_V100_TURBOMIND_GATE_UP_PROBE=off ;;
    1|true|on|auto) DS4_V100_TURBOMIND_GATE_UP_PROBE=auto ;;
    m64|m128|m64n256|n256|m64_s3|m64s3|m64_s4|m64s4|m128_s3|m128s3|m128_s4|m128s4|m128_1536|1536_m128|m128_s3_1536|m128s3_1536|1536_m128_s3|1536_m128s3|m128_s4_1536|m128s4_1536|1536_m128_s4|1536_m128s4|m64_s3_1536|m64s3_1536|1536_m64_s3|1536_m64s3|m64_s4_1536|m64s4_1536|1536_m64_s4|1536_m64s4) ;;
    *) fail "DS4_V100_TURBOMIND_GATE_UP_PROBE must be off, auto, m64, m128, m64n256, or an explicit stage-count/1536 probe" ;;
esac
case "$DS4_V100_TURBOMIND_DOWN_PROBE" in
    0|false|off|none) DS4_V100_TURBOMIND_DOWN_PROBE=off ;;
    1|true|on|auto|m128) DS4_V100_TURBOMIND_DOWN_PROBE=auto ;;
    m64n256|n256) DS4_V100_TURBOMIND_DOWN_PROBE=m64n256 ;;
    *) fail "DS4_V100_TURBOMIND_DOWN_PROBE must be off, auto, or m64n256" ;;
esac
case "$DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE" in
    0|false|off) DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1 ;;
    *) fail "DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE" in
    0|false|off) DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1 ;;
    *) fail "DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_DISPATCH_POLICY" in
    default|reuse) ;;
    measure|append)
        if [ "$DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE" -ne 1 ]; then
            fail "DS4_V100_TURBOMIND_DISPATCH_POLICY=$DS4_V100_TURBOMIND_DISPATCH_POLICY requires DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1"
        fi
        ;;
    *) fail "DS4_V100_TURBOMIND_DISPATCH_POLICY must be default, measure, reuse, or append" ;;
esac
case "$DS4_V100_TURBOMIND_GROUP_PIPELINE" in
    0|false|off) DS4_V100_TURBOMIND_GROUP_PIPELINE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_GROUP_PIPELINE=1 ;;
    *) fail "DS4_V100_TURBOMIND_GROUP_PIPELINE must be 0 or 1" ;;
esac
is_uint "$DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS" ||
    fail "DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS must be an integer"
[ "$DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS" -ge 1 ] &&
    [ "$DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS" -le 8 ] ||
    fail "DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS must be in [1,8]"
case "$DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS" in
    0|false|off) DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=0 ;;
    1|true|on) DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=1 ;;
    *) fail "DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC" in
    0|false|off) DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=0 ;;
    1|true|on) DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=1 ;;
    *) fail "DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_SINGLE_SLOT_ATTN_SCRATCH" in
    0|false|off) DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=0 ;;
    1|true|on) DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=1 ;;
    *) fail "DS4_V100_SINGLE_SLOT_ATTN_SCRATCH must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_GRAPH" in
    0|false|off) DS4_V100_TURBOMIND_GRAPH=0 ;;
    1|true|on) DS4_V100_TURBOMIND_GRAPH=1 ;;
    *) fail "DS4_V100_TURBOMIND_GRAPH must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_GRAPH_VERBOSE" in
    0|false|off) DS4_V100_TURBOMIND_GRAPH_VERBOSE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_GRAPH_VERBOSE=1 ;;
    *) fail "DS4_V100_TURBOMIND_GRAPH_VERBOSE must be 0 or 1" ;;
esac
case "$DS4_V100_TURBOMIND_PROFILE" in
    0|false|off) DS4_V100_TURBOMIND_PROFILE=0 ;;
    1|true|on) DS4_V100_TURBOMIND_PROFILE=1 ;;
    *) fail "DS4_V100_TURBOMIND_PROFILE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_ROUTED_FFN" in
    0|false|off|none) DS4_V100_TP_EP_ROUTED_FFN=0 ;;
    1|true|on|layer3|span) DS4_V100_TP_EP_ROUTED_FFN=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTED_FFN must be off, on, layer3, or span" ;;
esac
case "$DS4_V100_TP_EP_ASYNC_INPUT" in
    0|false|off) DS4_V100_TP_EP_ASYNC_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_ASYNC_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_ASYNC_INPUT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DECODE_CUDAGRAPH" in
    0|false|off) DS4_V100_TP_EP_DECODE_CUDAGRAPH=0 ;;
    1|true|on) DS4_V100_TP_EP_DECODE_CUDAGRAPH=1 ;;
    *) fail "DS4_V100_TP_EP_DECODE_CUDAGRAPH must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC" in
    0|false|off) DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC=0 ;;
    1|true|on) DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC=1 ;;
    *) fail "DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC" in
    0|false|off) DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC=0 ;;
    1|true|on) DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC=1 ;;
    *) fail "DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT" in
    0|false|off) DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT=0 ;;
    1|true|on) DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT=1 ;;
    *) fail "DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM" in
    0|false|off) DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM=0 ;;
    1|true|on) DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM=1 ;;
    *) fail "DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT" -eq 1 ]; then
    DS4_V100_TP_EP_DECODE_CUDAGRAPH=1
fi
if [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC" -eq 1 ]; then
    DS4_V100_TP_EP_DECODE_CUDAGRAPH=1
fi
if [ -n "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC" ]; then
    DS4_V100_TP_EP_DECODE_CUDAGRAPH=1
fi
case "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE" in
    ""|routed_ffn|dense|compose|final_hc|compose_eager_final_hc) ;;
    *) fail "DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE must be empty, routed_ffn, dense, compose, final_hc, or compose_eager_final_hc" ;;
esac
if [ -n "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE" ]; then
    DS4_V100_TP_EP_DECODE_CUDAGRAPH=1
fi
case "$DS4_V100_TP_EP_PARALLEL_HALVES" in
    0|false|off) DS4_V100_TP_EP_PARALLEL_HALVES=0 ;;
    1|true|on) DS4_V100_TP_EP_PARALLEL_HALVES=1 ;;
    *) fail "DS4_V100_TP_EP_PARALLEL_HALVES must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_COPY_EVENT_COMPOSE" in
    0|false|off) DS4_V100_TP_EP_COPY_EVENT_COMPOSE=0 ;;
    1|true|on) DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1 ;;
    *) fail "DS4_V100_TP_EP_COPY_EVENT_COMPOSE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_RETURN_FP16" in
    0|false|off) DS4_V100_TP_EP_RETURN_FP16=0 ;;
    1|true|on) DS4_V100_TP_EP_RETURN_FP16=1 ;;
    *) fail "DS4_V100_TP_EP_RETURN_FP16 must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE" in
    0|false|off) DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=0 ;;
    1|true|on) DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1 ;;
    *) fail "DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_COMPACT_MOE_DECODE" in
    0|false|off) DS4_V100_TP_EP_COMPACT_MOE_DECODE=0 ;;
    1|true|on) DS4_V100_TP_EP_COMPACT_MOE_DECODE=1 ;;
    *) fail "DS4_V100_TP_EP_COMPACT_MOE_DECODE must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_COMPACT_MOE_DECODE" -eq 1 ]; then
    DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1
fi
case "$DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE" in
    0|false|off) DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=0 ;;
    1|true|on) DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=1 ;;
    auto) DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=auto ;;
    *) fail "DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE must be 0, 1, or auto" ;;
esac
case "$DS4_V100_TP_EP_FUSED_GATED_SILU" in
    0|false|off) DS4_V100_TP_EP_FUSED_GATED_SILU=0 ;;
    1|true|on) DS4_V100_TP_EP_FUSED_GATED_SILU=1 ;;
    *) fail "DS4_V100_TP_EP_FUSED_GATED_SILU must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD" in
    0|false|off) DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=0 ;;
    1|true|on) DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=1 ;;
    *) fail "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY" in
    0|false|off) DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY=0 ;;
    1|true|on) DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY=1 ;;
    *) fail "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY" -eq 1 ]; then
    DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=1
fi
case "$DS4_V100_TP_EP_HC_FINAL_EXPAND" in
    0|false|off) DS4_V100_TP_EP_HC_FINAL_EXPAND=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_FINAL_EXPAND=1 ;;
    *) fail "DS4_V100_TP_EP_HC_FINAL_EXPAND must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_INPUT" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_INPUT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_PEER_ACCOUNTING" in
    0|false|off) DS4_V100_TP_EP_PEER_ACCOUNTING=0 ;;
    1|true|on) DS4_V100_TP_EP_PEER_ACCOUNTING=1 ;;
    *) fail "DS4_V100_TP_EP_PEER_ACCOUNTING must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_PEER_REJECT_SYS" in
    0|false|off) DS4_V100_TP_EP_PEER_REJECT_SYS=0 ;;
    1|true|on) DS4_V100_TP_EP_PEER_REJECT_SYS=1 ;;
    *) fail "DS4_V100_TP_EP_PEER_REJECT_SYS must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_PEER_REJECT_SYS" -eq 1 ]; then
    DS4_V100_TP_EP_PEER_ACCOUNTING=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_HC_PERSIST_STATE" in
    0|false|off) DS4_V100_TP_EP_HC_PERSIST_STATE=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_PERSIST_STATE=1 ;;
    *) fail "DS4_V100_TP_EP_HC_PERSIST_STATE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_MODEL_ROUTER_ROUTES" in
    0|false|off) DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=0 ;;
    1|true|on) DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1 ;;
    *) fail "DS4_V100_TP_EP_MODEL_ROUTER_ROUTES must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_ROUTER_CUBLAS" in
    0|false|off) DS4_V100_TP_EP_ROUTER_CUBLAS=0 ;;
    1|true|on) DS4_V100_TP_EP_ROUTER_CUBLAS=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTER_CUBLAS must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_ROUTER_HASH_FAST" in
    0|false|off) DS4_V100_TP_EP_ROUTER_HASH_FAST=0 ;;
    1|true|on) DS4_V100_TP_EP_ROUTER_HASH_FAST=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTER_HASH_FAST must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_GPU_ROUTE_PLAN" in
    0|false|off) DS4_V100_TP_EP_GPU_ROUTE_PLAN=0 ;;
    1|true|on) DS4_V100_TP_EP_GPU_ROUTE_PLAN=1 ;;
    *) fail "DS4_V100_TP_EP_GPU_ROUTE_PLAN must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD" in
    0|false|off) DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=0 ;;
    1|true|on) DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD" in
    0|false|off) DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=0 ;;
    1|true|on) DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD=1 ;;
    *) fail "DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_ROUTER_CUBLAS" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
fi
if [ "$DS4_V100_TP_EP_ROUTER_HASH_FAST" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
fi
if [ "$DS4_V100_TP_EP_GPU_ROUTE_PLAN" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_COMPACT_MOE_DECODE=1
    DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1
fi
if [ "$DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_COMPACT_MOE_DECODE=1
    DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1
fi
if [ "$DS4_V100_TP_EP_MODEL_ROUTER_ROUTES" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
    [ "$DS4_V100_TP_EP_TOP_K" -eq 6 ] || fail "DS4_V100_TP_EP_MODEL_ROUTER_ROUTES requires DS4_V100_TP_EP_TOP_K=6"
    if [ "$DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE" -eq 1 ] &&
       [ "$DS4_V100_TP_EP_COMPACT_MOE_DECODE" -ne 1 ]; then
        fail "DS4_V100_TP_EP_MODEL_ROUTER_ROUTES with compact route compose requires DS4_V100_TP_EP_COMPACT_MOE_DECODE=1"
    fi
fi
case "$DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT" in
    0|false|off) DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT" in
    0|false|off) DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
    DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1
fi
case "$DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS" in
    0|false|off) DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=0 ;;
    1|true|on) DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=1 ;;
    *) fail "DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS" in
    0|false|off) DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=0 ;;
    1|true|on) DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=1 ;;
    *) fail "DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS" -eq 1 ] && \
   [ "$DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS" -eq 1 ]; then
    fail "DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS and DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS are mutually exclusive"
fi
if [ "$DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=1
    DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
    DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
    DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1
fi
if [ "$DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS" -eq 1 ]; then
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_SHARED_FFN" in
    0|false|off) DS4_V100_TP_EP_TRUE_SHARED_FFN=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_SHARED_FFN=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_SHARED_FFN must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_SHARED_FFN" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN" in
    0|false|off) DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=0 ;;
    1|true|on) DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1 ;;
    *) fail "DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC" in
    0|false|off) DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC=0 ;;
    1|true|on) DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC=1 ;;
    *) fail "DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM" in
    0|false|off) DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM=0 ;;
    1|true|on) DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM=1 ;;
    *) fail "DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM" in
    0|false|off) DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=0 ;;
    1|true|on) DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=1 ;;
    *) fail "DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY" in
    0|false|off) DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY=0 ;;
    1|true|on) DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY=1 ;;
    *) fail "DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC" -eq 1 ]; then
    DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
fi
if [ "$DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY" -eq 1 ]; then
    DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=1
fi
if [ "$DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=1
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_COMPACT_MOE_DECODE=1
    DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
if [ "$DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE" = auto ]; then
    if [ "$DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE" -eq 0 ] &&
       [ "$DS4_V100_TP_EP_RETURN_FP16" -eq 0 ]; then
        DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=1
    else
        DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=0
    fi
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_TRUE_SHARED_FFN=1
    DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1
    DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1
    DS4_V100_TP_EP_ROUTE_PLAN_ASYNC_UPLOAD=0
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS" in
    auto) ;;
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS must be auto, 0, or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS" = "auto" ]; then
    if [ "$DS4_V100_SERVE_MODE" = "tp-ep" ] &&
       [ "$DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT" -eq 1 ]; then
        DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=1
    else
        DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=0
    fi
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE" in
    0|false|off) DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE=0 ;;
    1|true|on) DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE=1 ;;
    *) fail "DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE must be 0 or 1" ;;
esac
is_uint "$DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB" || fail "DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB must be an integer"
[ "$DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB" -ge 64 ] &&
    [ "$DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB" -le 4096 ] ||
    fail "DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB must be between 64 and 4096"
case "$DS4_V100_TP_EP_DEFER_NCCL_INIT" in
    0|false|off) DS4_V100_TP_EP_DEFER_NCCL_INIT=0 ;;
    1|true|on) DS4_V100_TP_EP_DEFER_NCCL_INIT=1 ;;
    *) fail "DS4_V100_TP_EP_DEFER_NCCL_INIT must be 0 or 1" ;;
esac
case "$DS4_V100_NCCL_TOPOLOGY_POLICY" in
    no-sys|off) ;;
    *) fail "DS4_V100_NCCL_TOPOLOGY_POLICY must be no-sys or off" ;;
esac
case "$DS4_V100_NCCL_ALLOW_VISIBLE_REMAP" in
    0|false|off) DS4_V100_NCCL_ALLOW_VISIBLE_REMAP=0 ;;
    1|true|on) DS4_V100_NCCL_ALLOW_VISIBLE_REMAP=1 ;;
    *) fail "DS4_V100_NCCL_ALLOW_VISIBLE_REMAP must be 0 or 1" ;;
esac
case "$DS4_V100_NCCL_ALGO" in
    auto|Ring|Tree|CollNetDirect|CollNetChain|NVLS|NVLSTree|PAT) ;;
    *) fail "DS4_V100_NCCL_ALGO must be auto or a valid NCCL_ALGO value" ;;
esac
case "$DS4_V100_NCCL_PROTO" in
    auto|LL|LL128|Simple) ;;
    *) fail "DS4_V100_NCCL_PROTO must be auto, LL, LL128, or Simple" ;;
esac
if [ "$DS4_V100_NCCL_TOPOLOGY_POLICY" = "no-sys" ]; then
    if [ "$DS4_V100_CUDA_VISIBLE_DEVICES" != "0,1,2,3,4,5,6,7" ] &&
       [ "$DS4_V100_NCCL_ALLOW_VISIBLE_REMAP" -ne 1 ]; then
        fail "DS4_V100_NCCL_TOPOLOGY_POLICY=no-sys requires DS4_V100_CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; set DS4_V100_NCCL_ALLOW_VISIBLE_REMAP=1 only for diagnostics"
    fi
    NCCL_RINGS="$DS4_V100_NCCL_NO_SYS_RING"
    NCCL_P2P_LEVEL=NVL
fi
if [ "$DS4_V100_NCCL_ALGO" != "auto" ]; then
    NCCL_ALGO="$DS4_V100_NCCL_ALGO"
else
    unset NCCL_ALGO
fi
if [ "$DS4_V100_NCCL_PROTO" != "auto" ]; then
    NCCL_PROTO="$DS4_V100_NCCL_PROTO"
else
    unset NCCL_PROTO
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_FP8_E5M2_KV" in
    0|false|off) DS4_V100_TP_EP_FP8_E5M2_KV=0 ;;
    1|true|on) DS4_V100_TP_EP_FP8_E5M2_KV=1 ;;
    *) fail "DS4_V100_TP_EP_FP8_E5M2_KV must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND" in
    0|false|off) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND=0 ;;
    1|true|on) DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND=1 ;;
    *) fail "DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_INPUT_FILL" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DIRECT_INPUT_FILL" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ATTN_INPUT_FILL" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_ROPE_ROUND" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM_ROPE_ROUND" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
fi
if [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE" -eq 1 ] ||
   [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE" -eq 1 ] ||
   [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE" -eq 1 ]; then
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_INDEXER=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_COMPRESSED=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_RAW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1
    DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RESIDENCY=1
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_REFERENCE_HC_REDUCE" in
    0|false|off) DS4_V100_TP_EP_REFERENCE_HC_REDUCE=0 ;;
    1|true|on) DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1 ;;
    *) fail "DS4_V100_TP_EP_REFERENCE_HC_REDUCE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD" in
    0|false|off) DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=0 ;;
    1|true|on) DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=1 ;;
    *) fail "DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY" in
    0|false|off) DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY=0 ;;
    1|true|on) DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY=1 ;;
    *) fail "DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
if [ "$DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
if [ "$DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD" -eq 1 ]; then
    DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1
fi
if [ "$DS4_V100_TP_EP_REFERENCE_HC_REDUCE" -eq 1 ]; then
    DS4_V100_TP_EP_HC_CURRENT_INPUT=1
    DS4_V100_TP_EP_HC_FINAL_EXPAND=1
fi
case "$DS4_V100_TP_EP_KV_ALL_SLOTS" in
    0|false|off) DS4_V100_TP_EP_KV_ALL_SLOTS=0 ;;
    1|true|on) DS4_V100_TP_EP_KV_ALL_SLOTS=1 ;;
    *) fail "DS4_V100_TP_EP_KV_ALL_SLOTS must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_VRAM_REPORT" in
    0|false|off) DS4_V100_TP_EP_VRAM_REPORT=0 ;;
    1|true|on) DS4_V100_TP_EP_VRAM_REPORT=1 ;;
    *) fail "DS4_V100_TP_EP_VRAM_REPORT must be 0 or 1" ;;
esac
is_uint "$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB" || fail "DS4_V100_TP_EP_VRAM_MIN_FREE_MIB must be an integer"
tp_ep_nccl_gate_active=0
if [ "$DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE" -eq 1 ] ||
   [ "$DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER" -eq 1 ] ||
   [ "$DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE" -eq 1 ] ||
   [ "$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER" -eq 1 ]; then
    tp_ep_nccl_gate_active=1
fi
if [ -z "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB" ]; then
    if [ "$tp_ep_nccl_gate_active" -eq 1 ]; then
        DS4_V100_TP_EP_NCCL_MIN_FREE_MIB=1536
    else
        DS4_V100_TP_EP_NCCL_MIN_FREE_MIB=0
    fi
fi
is_uint "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB" || fail "DS4_V100_TP_EP_NCCL_MIN_FREE_MIB must be an integer"
case "$DS4_V100_TP_EP_VERBOSE" in
    0|false|off) DS4_V100_TP_EP_VERBOSE=0 ;;
    1|true|on) DS4_V100_TP_EP_VERBOSE=1 ;;
    *) fail "DS4_V100_TP_EP_VERBOSE must be 0 or 1" ;;
esac
if [ "$DS4_V100_TP_EP_ROUTED_FFN" -eq 1 ]; then
    [ -n "$DS4_V100_TP_EP_LAYER_FIRST" ] || fail "DS4_V100_TP_EP_ROUTED_FFN requires DS4_V100_TP_EP_LAYER_FIRST"
    [ -n "$DS4_V100_TP_EP_SHARD_DIR" ] || fail "DS4_V100_TP_EP_ROUTED_FFN requires DS4_V100_TP_EP_SHARD_DIR"
fi
if [ "$DS4_V100_SERVE_MODE" = "tp-ep" ]; then
    [ "$DS4_V100_CTX" -eq 262144 ] || fail "DS4_V100_SERVE_MODE=tp-ep currently requires DS4_V100_CTX=262144"
    [ "$DS4_V100_SLOTS" -le 32 ] || fail "DS4_V100_SERVE_MODE=tp-ep currently supports DS4_V100_SLOTS<=32"
    [ "$DS4_V100_ACTIVE_MICROBATCH" -eq "$DS4_V100_SLOTS" ] || fail "DS4_V100_SERVE_MODE=tp-ep requires active_microbatch == slots"
    if [ "$mtp_serving_enabled" -eq 1 ]; then
        fail "DS4_V100_SERVE_MODE=tp-ep does not support MTP yet"
    fi
    if [ -z "$DS4_V100_APPLIANCE_DIR" ]; then
        fail "DS4_V100_SERVE_MODE=tp-ep requires DS4_V100_APPLIANCE_DIR"
    fi
    if [ -z "$DS4_V100_TP_EP_TM_INDEX" ]; then
        DS4_V100_TP_EP_TM_INDEX="$DS4_V100_APPLIANCE_DIR/turbomind-pack-index.tsv"
    fi
fi

async_pipeline_mode="$DS4_V100_ASYNC_PIPELINE_MODE"
case "$async_pipeline_mode" in
    per_step) async_pipeline_mode="per-step" ;;
    mbox) async_pipeline_mode="mailbox" ;;
    auto)
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            async_pipeline_mode="per-step"
        else
            async_pipeline_mode="off"
        fi
        ;;
esac
case "$DS4_V100_ASYNC_EVENT_HANDOFF" in
    auto)
        if [ "$async_pipeline_mode" = "per-step" ] && [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
            async_event_handoff=1
        else
            async_event_handoff=0
        fi
        ;;
    0|false|off) async_event_handoff=0 ;;
    1|true|on) async_event_handoff=1 ;;
esac
if [ "$async_event_handoff" -eq 1 ] && [ "$async_pipeline_mode" != "per-step" ]; then
    fail "DS4_V100_ASYNC_EVENT_HANDOFF requires resolved async pipeline mode per-step"
fi
if [ "$DS4_V100_ASYNC_FFN_WAVEFRONT" -eq 1 ] && [ "$async_pipeline_mode" != "per-step" ]; then
    fail "DS4_V100_ASYNC_FFN_WAVEFRONT requires resolved async pipeline mode per-step"
fi

if [ "$DS4_V100_SERVE_MODE" = "tp-ep" ]; then
    require_exec "$DS4_V100_TP_EP_BIN"
else
    require_exec "$DS4_V100_BIN"
    require_file "model" "$DS4_V100_MODEL"
fi
if [ -n "$DS4_V100_APPLIANCE_DIR" ]; then
    require_dir "appliance directory" "$DS4_V100_APPLIANCE_DIR"
    require_file "appliance pack index" "$DS4_V100_APPLIANCE_DIR/pack-index.tsv"
    require_file "appliance TurboMind index" "$DS4_V100_APPLIANCE_DIR/turbomind-pack-index.tsv"
    for gpu in 0 1 2 3 4 5 6 7; do
        require_file "appliance gpu${gpu} shard" "$DS4_V100_APPLIANCE_DIR/gpu${gpu}.weights"
    done
else
    require_file "pack index" "$DS4_V100_PACK_INDEX"
fi
if [ "$mtp_serving_enabled" -eq 1 ] && [ -z "$DS4_V100_MTP_MODEL" ]; then
    fail "DS4_V100_MTP_MODEL is required when DS4_V100_MTP_SERVING=$DS4_V100_MTP_SERVING"
fi
if [ "$DS4_V100_SERVE_MODE" != "tp-ep" ] &&
   { [ "$mtp_serving_enabled" -eq 1 ] || [ -n "$DS4_V100_MTP_MODEL" ]; }; then
    require_file "MTP model" "$DS4_V100_MTP_MODEL"
fi
if [ "$DS4_V100_SERVE_MODE" = "tp-ep" ]; then
    require_file "TP/EP contract" "$DS4_V100_TP_EP_CONTRACT"
    require_file "TP/EP TurboMind index" "$DS4_V100_TP_EP_TM_INDEX"
    require_file "TurboMind library" "$DS4_V100_TURBOMIND_LIB"
    if [ -n "$DS4_V100_TP_EP_TOKENIZER_MODEL" ]; then
        require_file "TP/EP tokenizer model" "$DS4_V100_TP_EP_TOKENIZER_MODEL"
    fi
fi
check_gpu_reserve

if [ "$DS4_V100_SERVE_MODE" = "tp-ep" ]; then
    cmd=(
        "$DS4_V100_TP_EP_BIN"
        --serve-http
        --pack-dir "$DS4_V100_APPLIANCE_DIR"
        --contract "$DS4_V100_TP_EP_CONTRACT"
        --tm-index "$DS4_V100_TP_EP_TM_INDEX"
        --lib "$DS4_V100_TURBOMIND_LIB"
        --slots "$DS4_V100_SLOTS"
        --top-k "$DS4_V100_TP_EP_TOP_K"
        --kv-slot "$DS4_V100_TP_EP_KV_SLOT"
        --position "$DS4_V100_TP_EP_POSITION"
        --warmup 0
        --iters 1
        --decode-steps "$DS4_V100_TOKENS"
        --host "$DS4_V100_HOST"
        --port "$DS4_V100_PORT"
        --microbatch-wait-us "$microbatch_wait_us"
    )
    if [ -n "$DS4_V100_TP_EP_TOKENIZER_MODEL" ]; then
        cmd+=(--tokenizer-model "$DS4_V100_TP_EP_TOKENIZER_MODEL")
    fi
    if [ "$DS4_V100_TP_EP_VRAM_REPORT" -eq 1 ]; then
        cmd+=(--vram-report)
    fi
    if [ "$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB" -gt 0 ]; then
        cmd+=(--vram-min-free-mib "$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB")
    fi
    if [ "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB" -gt 0 ]; then
        cmd+=(--nccl-min-free-mib "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB")
    fi
    if [ "$cuda_profiler_window" -eq 1 ]; then
        cmd+=(--cuda-profiler-window)
    fi
    if [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT" -eq 1 ]; then
        cmd+=(--decode-cudagraph-persistent-replay-gate)
    elif [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH" -eq 1 ]; then
        cmd+=(--decode-cudagraph-gate)
    fi
    if [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC" -eq 1 ]; then
        cmd+=(--decode-cudagraph-output-sync-gate)
    fi
    if [ "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC" -eq 1 ]; then
        cmd+=(--decode-cudagraph-hc-current-sync-gate)
    fi
    if [ -n "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC" ]; then
        cmd+=(--decode-cudagraph-stage-sync-gate "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC")
    fi
    if [ -n "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE" ]; then
        cmd+=(--decode-cudagraph-suffix-stage-gate "$DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE")
    fi
    if [ "$DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM" -eq 1 ]; then
        cmd+=(--decode-stage-checksum-gate)
    fi
    if [ -n "$DS4_V100_TP_EP_EXTRA_ARGS" ]; then
        while IFS= read -r extra_arg; do
            [ -n "$extra_arg" ] || continue
            cmd+=("$extra_arg")
        done <<< "$DS4_V100_TP_EP_EXTRA_ARGS"
    fi
else
    cmd=(
        "$DS4_V100_BIN"
        --serve
        --model "$DS4_V100_MODEL"
        --ctx "$DS4_V100_CTX"
        --slots "$DS4_V100_SLOTS"
        --active-microbatch "$DS4_V100_ACTIVE_MICROBATCH"
        --microbatch-wait-us "$microbatch_wait_us"
        --queue-policy "$DS4_V100_QUEUE_POLICY"
        --tokens "$DS4_V100_TOKENS"
        --host "$DS4_V100_HOST"
        --port "$DS4_V100_PORT"
    )
    if [ -n "$DS4_V100_APPLIANCE_DIR" ]; then
        cmd+=(--appliance-dir "$DS4_V100_APPLIANCE_DIR")
    else
        cmd+=(--index "$DS4_V100_PACK_INDEX")
    fi
fi
if [ "$DS4_V100_MAX_REQUESTS" -gt 0 ]; then
    cmd+=(--max-requests "$DS4_V100_MAX_REQUESTS")
fi
if [ "$DS4_V100_SERVE_MODE" != "tp-ep" ]; then
    if [ "$async_pipeline_mode" != "off" ]; then
        cmd+=(--async-pipeline-mode "$async_pipeline_mode")
    fi
    if [ "$async_handoff" -eq 1 ]; then
        cmd+=(--async-handoff)
    fi
    if [ "$async_event_handoff" -eq 1 ]; then
        cmd+=(--async-event-handoff)
    fi
    if [ "$startup_warmup" -eq 1 ]; then
        cmd+=(--startup-warmup)
    fi
    if [ "$cuda_profiler_window" -eq 1 ]; then
        cmd+=(--cuda-profiler-window)
    fi
    if [ "$mtp_serving_enabled" -eq 1 ]; then
        cmd+=(
            --mtp-model "$DS4_V100_MTP_MODEL"
            --mtp-serving "$DS4_V100_MTP_SERVING"
            --mtp-top-k "$DS4_V100_MTP_TOP_K"
            --mtp-gpu "$DS4_V100_MTP_GPU"
            --mtp-reserve-mib "$DS4_V100_RESERVE_MIB"
        )
    fi
fi

print_resolved() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$DS4_V100_CUDA_VISIBLE_DEVICES"
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

if [ "$mode" = "check" ]; then
    echo "ds4-v100-run-appliance: config ok mode=$DS4_V100_SERVE_MODE mtp=$DS4_V100_MTP_SERVING host=$DS4_V100_HOST port=$DS4_V100_PORT ctx=$DS4_V100_CTX slots=$DS4_V100_SLOTS active_microbatch=$DS4_V100_ACTIVE_MICROBATCH microbatch_wait_us=$microbatch_wait_us tokens=$DS4_V100_TOKENS async_pipeline_mode=$async_pipeline_mode async_handoff=$async_handoff async_event_handoff=$async_event_handoff async_slot_chunk=${DS4_V100_ASYNC_SLOT_CHUNK:-default} async_ffn_wavefront=$DS4_V100_ASYNC_FFN_WAVEFRONT async_ffn_wavefront_chunk=$DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK async_ffn_wavefront_verbose=$DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE startup_warmup=$startup_warmup cuda_profiler_window=$cuda_profiler_window cuda_tensor_pool=$cuda_tensor_pool cuda_tensor_pool_max_mib=$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB cuda_f8_rowpair=$DS4_V100_CUDA_F8_ROWPAIR cuda_f8_row4=$DS4_V100_CUDA_F8_ROW4 cuda_f8_warp_scale=$DS4_V100_CUDA_F8_WARP_SCALE cuda_f8_grouped_ds4_fast=$DS4_V100_CUDA_F8_GROUPED_DS4_FAST cuda_f8_hmma_shared_down=$DS4_V100_CUDA_F8_HMMA_SHARED_DOWN cuda_f8_hmma_pair_swiglu=$DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU cuda_f8_hmma_attn_batch=$DS4_V100_CUDA_F8_HMMA_ATTN_BATCH cuda_f8_hmma_grouped_attn_o_batch=$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH cuda_f8_hmma_grouped_attn_o_single=$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE cuda_f8_hmma_single=$DS4_V100_CUDA_F8_HMMA_SINGLE cuda_f8_pair_swiglu_single=$DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE cuda_f8_pair_swiglu_single_rows2=$DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2 f8_shared_down_add=$DS4_V100_F8_SHARED_DOWN_ADD batch_attn_proj=$DS4_V100_ENABLE_BATCH_ATTN_PROJ batch_attn_output_a=$DS4_V100_BATCH_ATTN_OUTPUT_A batch_attn_output_b=$DS4_V100_BATCH_ATTN_OUTPUT_B batch_shared_f8=$DS4_V100_BATCH_SHARED_F8 ffn_direct_delta=$DS4_V100_FFN_DIRECT_DELTA disable_grouped_attn_output_a=$DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A single_slot_attn_scratch=$DS4_V100_SINGLE_SLOT_ATTN_SCRATCH single_slot_attn_output_a_hmma=$DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA appliance_dir=${DS4_V100_APPLIANCE_DIR:-none} turbomind_routed_ffn=$DS4_V100_TURBOMIND_ROUTED_FFN disable_turbomind_total_tokens=$DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS turbomind_route_validate_sync=$DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC turbomind_small_route_build=$DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD turbomind_route_row_reduce=$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE turbomind_route_row_reduce_h2=$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2 turbomind_indexed_a=$DS4_V100_TURBOMIND_INDEXED_A turbomind_fused_gate_up=$DS4_V100_TURBOMIND_FUSED_GATE_UP turbomind_gated_silu=$DS4_V100_TURBOMIND_GATED_SILU turbomind_compact_schedule=$DS4_V100_TURBOMIND_COMPACT_SCHEDULE turbomind_compact_no_host_sync=$DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC turbomind_routed_executor=$DS4_V100_TURBOMIND_ROUTED_EXECUTOR turbomind_routed_executor_verbose=$DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE turbomind_gate_up_probe=$DS4_V100_TURBOMIND_GATE_UP_PROBE turbomind_down_probe=$DS4_V100_TURBOMIND_DOWN_PROBE turbomind_down_reduce_epilogue=$DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE turbomind_dispatch_policy=$DS4_V100_TURBOMIND_DISPATCH_POLICY turbomind_allow_unsafe_measure=$DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE turbomind_group_pipeline=$DS4_V100_TURBOMIND_GROUP_PIPELINE turbomind_group_pipeline_streams=$DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS turbomind_group_pipeline_auto_groups=$DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS turbomind_graph=$DS4_V100_TURBOMIND_GRAPH turbomind_graph_verbose=$DS4_V100_TURBOMIND_GRAPH_VERBOSE turbomind_profile=$DS4_V100_TURBOMIND_PROFILE tp_ep_routed_ffn=$DS4_V100_TP_EP_ROUTED_FFN tp_ep_layer_first=${DS4_V100_TP_EP_LAYER_FIRST:-none} tp_ep_layer_count=$DS4_V100_TP_EP_LAYER_COUNT tp_ep_peer=${DS4_V100_TP_EP_PEER:-auto} tp_ep_peer_accounting=$DS4_V100_TP_EP_PEER_ACCOUNTING tp_ep_peer_reject_sys=$DS4_V100_TP_EP_PEER_REJECT_SYS tp_ep_async_input=$DS4_V100_TP_EP_ASYNC_INPUT tp_ep_parallel_halves=$DS4_V100_TP_EP_PARALLEL_HALVES tp_ep_parallel_expert_load=$DS4_V100_TP_EP_PARALLEL_EXPERT_LOAD tp_ep_copy_event_compose=$DS4_V100_TP_EP_COPY_EVENT_COMPOSE tp_ep_return_fp16=$DS4_V100_TP_EP_RETURN_FP16 tp_ep_compact_route_compose=$DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE tp_ep_compact_moe_decode=$DS4_V100_TP_EP_COMPACT_MOE_DECODE tp_ep_nccl_reduce_scatter_compose=$DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE tp_ep_hc_final_expand=$DS4_V100_TP_EP_HC_FINAL_EXPAND tp_ep_hc_current_input=$DS4_V100_TP_EP_HC_CURRENT_INPUT tp_ep_model_router_routes=$DS4_V100_TP_EP_MODEL_ROUTER_ROUTES tp_ep_router_cublas=$DS4_V100_TP_EP_ROUTER_CUBLAS tp_ep_gpu_route_plan=$DS4_V100_TP_EP_GPU_ROUTE_PLAN tp_ep_routed_ffn_norm_input=$DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT tp_ep_true_shared_ffn=$DS4_V100_TP_EP_TRUE_SHARED_FFN tp_ep_verbose=$DS4_V100_TP_EP_VERBOSE"
    echo "ds4-v100-run-appliance: nccl topology policy=$DS4_V100_NCCL_TOPOLOGY_POLICY cuda_visible_devices=$DS4_V100_CUDA_VISIBLE_DEVICES nccl_algo=${NCCL_ALGO:-auto} nccl_proto=${NCCL_PROTO:-auto} nccl_rings=${NCCL_RINGS:-} nccl_p2p_level=${NCCL_P2P_LEVEL:-}"
    exit 0
fi
if [ "$mode" = "print" ]; then
    print_resolved
    exit 0
fi

mkdir -p "$DS4_V100_LOG_DIR"
{
    echo "DS4_V100_MODEL=$DS4_V100_MODEL"
    echo "DS4_V100_MTP_MODEL=$DS4_V100_MTP_MODEL"
    echo "DS4_V100_APPLIANCE_DIR=$DS4_V100_APPLIANCE_DIR"
    echo "DS4_V100_PACK_INDEX=$DS4_V100_PACK_INDEX"
    echo "DS4_V100_CTX=$DS4_V100_CTX"
    echo "DS4_V100_SLOTS=$DS4_V100_SLOTS"
    echo "DS4_V100_ACTIVE_MICROBATCH=$DS4_V100_ACTIVE_MICROBATCH"
    echo "DS4_V100_MICROBATCH_WAIT_US=$DS4_V100_MICROBATCH_WAIT_US"
    echo "DS4_V100_MICROBATCH_WAIT_US_RESOLVED=$microbatch_wait_us"
    echo "DS4_V100_QUEUE_POLICY=$DS4_V100_QUEUE_POLICY"
    echo "DS4_V100_TOKENS=$DS4_V100_TOKENS"
    echo "DS4_V100_ASYNC_PIPELINE_MODE=$DS4_V100_ASYNC_PIPELINE_MODE"
    echo "DS4_V100_ASYNC_PIPELINE_MODE_RESOLVED=$async_pipeline_mode"
    echo "DS4_V100_ASYNC_HANDOFF=$DS4_V100_ASYNC_HANDOFF"
    echo "DS4_V100_ASYNC_HANDOFF_RESOLVED=$async_handoff"
    echo "DS4_V100_ASYNC_EVENT_HANDOFF=$DS4_V100_ASYNC_EVENT_HANDOFF"
    echo "DS4_V100_ASYNC_EVENT_HANDOFF_RESOLVED=$async_event_handoff"
    echo "DS4_V100_ASYNC_SLOT_CHUNK=$DS4_V100_ASYNC_SLOT_CHUNK"
    echo "DS4_V100_ASYNC_FFN_WAVEFRONT=$DS4_V100_ASYNC_FFN_WAVEFRONT"
    echo "DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK=$DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK"
    echo "DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE=$DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE"
    echo "DS4_V100_STARTUP_WARMUP=$DS4_V100_STARTUP_WARMUP"
    echo "DS4_V100_STARTUP_WARMUP_RESOLVED=$startup_warmup"
    echo "DS4_V100_CUDA_PROFILER_WINDOW=$DS4_V100_CUDA_PROFILER_WINDOW"
    echo "DS4_V100_CUDA_PROFILER_WINDOW_RESOLVED=$cuda_profiler_window"
    echo "DS4_V100_CUDA_TENSOR_POOL=$DS4_V100_CUDA_TENSOR_POOL"
    echo "DS4_V100_CUDA_TENSOR_POOL_RESOLVED=$cuda_tensor_pool"
    echo "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB=$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB"
    echo "DS4_V100_CUDA_F8_ROWPAIR=$DS4_V100_CUDA_F8_ROWPAIR"
    echo "DS4_V100_CUDA_F8_ROW4=$DS4_V100_CUDA_F8_ROW4"
    echo "DS4_V100_CUDA_F8_WARP_SCALE=$DS4_V100_CUDA_F8_WARP_SCALE"
    echo "DS4_V100_CUDA_F8_GROUPED_DS4_FAST=$DS4_V100_CUDA_F8_GROUPED_DS4_FAST"
    echo "DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=$DS4_V100_CUDA_F8_HMMA_SHARED_DOWN"
    echo "DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=$DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU"
    echo "DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=$DS4_V100_CUDA_F8_HMMA_ATTN_BATCH"
    echo "DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH"
    echo "DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE=$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE"
    echo "DS4_V100_CUDA_F8_HMMA_SINGLE=$DS4_V100_CUDA_F8_HMMA_SINGLE"
    echo "DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=$DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE"
    echo "DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=$DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2"
    echo "DS4_V100_F8_SHARED_DOWN_ADD=$DS4_V100_F8_SHARED_DOWN_ADD"
    echo "DS4_V100_ENABLE_BATCH_ATTN_PROJ=$DS4_V100_ENABLE_BATCH_ATTN_PROJ"
    echo "DS4_V100_BATCH_ATTN_OUTPUT_A=$DS4_V100_BATCH_ATTN_OUTPUT_A"
    echo "DS4_V100_BATCH_ATTN_OUTPUT_B=$DS4_V100_BATCH_ATTN_OUTPUT_B"
    echo "DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=$DS4_V100_ENABLE_OUTPUT_HEAD_BATCH"
    echo "DS4_V100_BATCH_SHARED_F8=$DS4_V100_BATCH_SHARED_F8"
    echo "DS4_V100_FFN_DIRECT_DELTA=$DS4_V100_FFN_DIRECT_DELTA"
    echo "DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=$DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A"
    echo "DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA=$DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA"
    echo "DS4_V100_TURBOMIND_ROUTED_FFN=$DS4_V100_TURBOMIND_ROUTED_FFN"
    echo "DS4_V100_TURBOMIND_STRICT=$DS4_V100_TURBOMIND_STRICT"
    echo "DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=$DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS"
    echo "DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=$DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC"
    echo "DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=$DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD"
    echo "DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE"
    echo "DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=$DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2"
    echo "DS4_V100_TURBOMIND_INDEXED_A=$DS4_V100_TURBOMIND_INDEXED_A"
    echo "DS4_V100_TURBOMIND_FUSED_GATE_UP=$DS4_V100_TURBOMIND_FUSED_GATE_UP"
    echo "DS4_V100_TURBOMIND_GATED_SILU=$DS4_V100_TURBOMIND_GATED_SILU"
    echo "DS4_V100_TURBOMIND_COMPACT_SCHEDULE=$DS4_V100_TURBOMIND_COMPACT_SCHEDULE"
    echo "DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC=$DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC"
    echo "DS4_V100_SINGLE_SLOT_ATTN_SCRATCH=$DS4_V100_SINGLE_SLOT_ATTN_SCRATCH"
    echo "DS4_V100_TURBOMIND_ROUTED_EXECUTOR=$DS4_V100_TURBOMIND_ROUTED_EXECUTOR"
    echo "DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE=$DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE"
    echo "DS4_V100_TURBOMIND_GATE_UP_PROBE=$DS4_V100_TURBOMIND_GATE_UP_PROBE"
    echo "DS4_V100_TURBOMIND_DOWN_PROBE=$DS4_V100_TURBOMIND_DOWN_PROBE"
    echo "DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=$DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE"
    echo "DS4_V100_TURBOMIND_DISPATCH_POLICY=$DS4_V100_TURBOMIND_DISPATCH_POLICY"
    echo "DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=$DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE"
    echo "DS4_V100_TURBOMIND_GROUP_PIPELINE=$DS4_V100_TURBOMIND_GROUP_PIPELINE"
    echo "DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS=$DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS"
    echo "DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS=$DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS"
    echo "DS4_V100_TURBOMIND_GRAPH=$DS4_V100_TURBOMIND_GRAPH"
    echo "DS4_V100_TURBOMIND_GRAPH_VERBOSE=$DS4_V100_TURBOMIND_GRAPH_VERBOSE"
    echo "DS4_V100_TURBOMIND_PROFILE=$DS4_V100_TURBOMIND_PROFILE"
    echo "DS4_V100_TP_EP_ROUTED_FFN=$DS4_V100_TP_EP_ROUTED_FFN"
    echo "DS4_V100_TP_EP_LAYER_FIRST=$DS4_V100_TP_EP_LAYER_FIRST"
    echo "DS4_V100_TP_EP_LAYER_COUNT=$DS4_V100_TP_EP_LAYER_COUNT"
    echo "DS4_V100_TP_EP_PEER=$DS4_V100_TP_EP_PEER"
    echo "DS4_V100_TP_EP_SHARD_DIR=$DS4_V100_TP_EP_SHARD_DIR"
    echo "DS4_V100_TP_EP_ASYNC_INPUT=$DS4_V100_TP_EP_ASYNC_INPUT"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH=$DS4_V100_TP_EP_DECODE_CUDAGRAPH"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC=$DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC=$DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC=$DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE=$DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE"
    echo "DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT=$DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT"
    echo "DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM=$DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM"
    echo "DS4_V100_TP_EP_PARALLEL_HALVES=$DS4_V100_TP_EP_PARALLEL_HALVES"
    echo "DS4_V100_TP_EP_COPY_EVENT_COMPOSE=$DS4_V100_TP_EP_COPY_EVENT_COMPOSE"
    echo "DS4_V100_TP_EP_RETURN_FP16=$DS4_V100_TP_EP_RETURN_FP16"
    echo "DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=$DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE"
    echo "DS4_V100_TP_EP_COMPACT_MOE_DECODE=$DS4_V100_TP_EP_COMPACT_MOE_DECODE"
    echo "DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE=$DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE"
    echo "DS4_V100_TP_EP_FUSED_GATED_SILU=$DS4_V100_TP_EP_FUSED_GATED_SILU"
    echo "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=$DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD"
    echo "DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY=$DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY"
    echo "DS4_V100_TP_EP_HC_FINAL_EXPAND=$DS4_V100_TP_EP_HC_FINAL_EXPAND"
    echo "DS4_V100_TP_EP_HC_CURRENT_INPUT=$DS4_V100_TP_EP_HC_CURRENT_INPUT"
    echo "DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=$DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER"
    echo "DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=$DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER"
    echo "DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=$DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE"
    echo "DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY=$DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY"
    echo "DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=$DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC"
    echo "DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK=$DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK"
    echo "DS4_V100_TP_EP_PEER_ACCOUNTING=$DS4_V100_TP_EP_PEER_ACCOUNTING"
    echo "DS4_V100_TP_EP_PEER_REJECT_SYS=$DS4_V100_TP_EP_PEER_REJECT_SYS"
    echo "DS4_V100_TP_EP_HC_PERSIST_STATE=$DS4_V100_TP_EP_HC_PERSIST_STATE"
    echo "DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=$DS4_V100_TP_EP_MODEL_ROUTER_ROUTES"
    echo "DS4_V100_TP_EP_ROUTER_CUBLAS=$DS4_V100_TP_EP_ROUTER_CUBLAS"
    echo "DS4_V100_TP_EP_ROUTER_HASH_FAST=$DS4_V100_TP_EP_ROUTER_HASH_FAST"
    echo "DS4_V100_TP_EP_GPU_ROUTE_PLAN=$DS4_V100_TP_EP_GPU_ROUTE_PLAN"
    echo "DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=$DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT"
    echo "DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT=$DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT"
    echo "DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS=$DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS"
    echo "DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=$DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS"
    echo "DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN=$DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN"
    echo "DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC=$DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC"
    echo "DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM=$DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM"
    echo "DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM=$DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM"
    echo "DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY=$DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY"
    echo "DS4_V100_TP_EP_TRUE_SHARED_FFN=$DS4_V100_TP_EP_TRUE_SHARED_FFN"
    echo "DS4_V100_TP_EP_REFERENCE_HC_REDUCE=$DS4_V100_TP_EP_REFERENCE_HC_REDUCE"
    echo "DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=$DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD"
    echo "DS4_V100_TP_EP_KV_ALL_SLOTS=$DS4_V100_TP_EP_KV_ALL_SLOTS"
    echo "DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB=$DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB"
    echo "DS4_V100_TP_EP_DEFER_NCCL_INIT=$DS4_V100_TP_EP_DEFER_NCCL_INIT"
    echo "DS4_V100_NCCL_TOPOLOGY_POLICY=$DS4_V100_NCCL_TOPOLOGY_POLICY"
    echo "DS4_V100_NCCL_NO_SYS_RING=$DS4_V100_NCCL_NO_SYS_RING"
    echo "DS4_V100_NCCL_ALLOW_VISIBLE_REMAP=$DS4_V100_NCCL_ALLOW_VISIBLE_REMAP"
    echo "DS4_V100_NCCL_ALGO=$DS4_V100_NCCL_ALGO"
    echo "DS4_V100_NCCL_PROTO=$DS4_V100_NCCL_PROTO"
    echo "NCCL_ALGO=${NCCL_ALGO:-auto}"
    echo "NCCL_PROTO=${NCCL_PROTO:-auto}"
    echo "NCCL_RINGS=${NCCL_RINGS:-}"
    echo "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-}"
    echo "DS4_V100_TP_EP_FP8_E5M2_KV=$DS4_V100_TP_EP_FP8_E5M2_KV"
    echo "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT=$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_LOCAL_INPUT"
    echo "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT=$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION_RANK_MAJOR_INPUT"
    echo "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT=$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT"
    echo "DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER=$DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER"
    echo "DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT=$DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT"
    echo "DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS=$DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS"
    echo "DS4_V100_TP_EP_VRAM_REPORT=$DS4_V100_TP_EP_VRAM_REPORT"
    echo "DS4_V100_TP_EP_VRAM_MIN_FREE_MIB=$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB"
    echo "DS4_V100_TP_EP_NCCL_MIN_FREE_MIB=$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB"
    echo "DS4_V100_TP_EP_VERBOSE=$DS4_V100_TP_EP_VERBOSE"
    echo "DS4_V100_TP_EP_BIN=$DS4_V100_TP_EP_BIN"
    echo "DS4_V100_TP_EP_CONTRACT=$DS4_V100_TP_EP_CONTRACT"
    echo "DS4_V100_TP_EP_TM_INDEX=$DS4_V100_TP_EP_TM_INDEX"
    echo "DS4_V100_TP_EP_TOP_K=$DS4_V100_TP_EP_TOP_K"
    echo "DS4_V100_TP_EP_KV_SLOT=$DS4_V100_TP_EP_KV_SLOT"
    echo "DS4_V100_TP_EP_POSITION=$DS4_V100_TP_EP_POSITION"
    echo "DS4_V100_TURBOMIND_LIB=$DS4_V100_TURBOMIND_LIB"
    echo "DS4_V100_HOST=$DS4_V100_HOST"
    echo "DS4_V100_PORT=$DS4_V100_PORT"
    echo "DS4_V100_ALLOW_NONLOCAL_HOST=$DS4_V100_ALLOW_NONLOCAL_HOST"
    echo "DS4_V100_CUDA_VISIBLE_DEVICES=$DS4_V100_CUDA_VISIBLE_DEVICES"
    echo "DS4_V100_CUDA_LIB_DIR=$DS4_V100_CUDA_LIB_DIR"
    echo "DS4_V100_CUDA_LIB_DIR_RESOLVED=$cuda_lib_dir"
    echo "DS4_V100_REQUIRE_GPUS=$DS4_V100_REQUIRE_GPUS"
	    echo "DS4_V100_RESERVE_MIB=$DS4_V100_RESERVE_MIB"
	    echo "DS4_LOCK_FILE=$DS4_LOCK_FILE"
	    echo "DS4_V100_SERVE_MODE=$DS4_V100_SERVE_MODE"
	    echo "DS4_V100_MTP_SERVING=$DS4_V100_MTP_SERVING"
	    echo "DS4_V100_MTP_TOP_K=$DS4_V100_MTP_TOP_K"
    echo "DS4_V100_MTP_GPU=$DS4_V100_MTP_GPU"
} >"$DS4_V100_LOG_DIR/startup.env"
print_resolved >"$DS4_V100_LOG_DIR/command.txt"

export CUDA_VISIBLE_DEVICES="$DS4_V100_CUDA_VISIBLE_DEVICES"
if [ -n "$cuda_lib_dir" ]; then
    export LD_LIBRARY_PATH="$cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export DS4_V100_ASYNC_SLOT_CHUNK
export DS4_V100_ASYNC_FFN_WAVEFRONT
export DS4_V100_ASYNC_FFN_WAVEFRONT_CHUNK
export DS4_V100_ASYNC_FFN_WAVEFRONT_VERBOSE
export DS4_V100_BATCH_SHARED_F8
export DS4_V100_FFN_DIRECT_DELTA
export DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A
export DS4_V100_SINGLE_SLOT_ATTN_SCRATCH
export DS4_V100_SINGLE_SLOT_ATTN_OUTPUT_A_HMMA
export DS4_V100_TURBOMIND_ROUTED_FFN
export DS4_V100_TURBOMIND_STRICT
export DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS
export DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC
export DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD
export DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE
export DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2
export DS4_V100_TURBOMIND_INDEXED_A
export DS4_V100_TURBOMIND_FUSED_GATE_UP
export DS4_V100_TURBOMIND_GATED_SILU
export DS4_V100_TURBOMIND_COMPACT_SCHEDULE
export DS4_V100_TURBOMIND_COMPACT_NO_HOST_SYNC
export DS4_V100_TURBOMIND_ROUTED_EXECUTOR
export DS4_V100_TURBOMIND_ROUTED_EXECUTOR_VERBOSE
export DS4_V100_TURBOMIND_GATE_UP_PROBE
export DS4_V100_TURBOMIND_DOWN_PROBE
export DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE
export DS4_V100_TURBOMIND_DISPATCH_POLICY
export DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE
export DS4_V100_TURBOMIND_GROUP_PIPELINE
export DS4_V100_TURBOMIND_GROUP_PIPELINE_STREAMS
export DS4_V100_TURBOMIND_GROUP_PIPELINE_AUTO_GROUPS
export DS4_V100_TURBOMIND_GRAPH
export DS4_V100_TURBOMIND_GRAPH_VERBOSE
export DS4_V100_TURBOMIND_PROFILE
export DS4_V100_TP_EP_ROUTED_FFN
export DS4_V100_TP_EP_LAYER_FIRST
export DS4_V100_TP_EP_LAYER_COUNT
export DS4_V100_TP_EP_PEER
export DS4_V100_TP_EP_SHARD_DIR
export DS4_V100_TP_EP_ASYNC_INPUT
export DS4_V100_TP_EP_DECODE_CUDAGRAPH
export DS4_V100_TP_EP_DECODE_CUDAGRAPH_OUTPUT_SYNC
export DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC
export DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC
export DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE
export DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT
export DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM
export DS4_V100_TP_EP_PARALLEL_HALVES
export DS4_V100_TP_EP_COPY_EVENT_COMPOSE
export DS4_V100_TP_EP_RETURN_FP16
export DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE
export DS4_V100_TP_EP_COMPACT_MOE_DECODE
export DS4_V100_TP_EP_NCCL_REDUCE_SCATTER_COMPOSE
export DS4_V100_TP_EP_FUSED_GATED_SILU
export DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD
export DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY
export DS4_V100_TP_EP_HC_FINAL_EXPAND
export DS4_V100_TP_EP_HC_CURRENT_INPUT
export DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER
export DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER
export DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE
export DS4_V100_TP_EP_HC_CURRENT_FULL_PARITY
export DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC
export DS4_V100_TP_EP_HC_CURRENT_INPUT_FUSED_FILL_PACK
export DS4_V100_TP_EP_PEER_ACCOUNTING
export DS4_V100_TP_EP_PEER_REJECT_SYS
export DS4_V100_TP_EP_HC_PERSIST_STATE
export DS4_V100_TP_EP_MODEL_ROUTER_ROUTES
export DS4_V100_TP_EP_ROUTER_CUBLAS
export DS4_V100_TP_EP_ROUTER_HASH_FAST
export DS4_V100_TP_EP_GPU_ROUTE_PLAN
export DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT
export DS4_V100_TP_EP_ROUTED_FFN_RANK_MAJOR_INPUT
export DS4_V100_TP_EP_MODEL_ROUTER_RANK_MAJOR_LOGITS
export DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS
export DS4_V100_TP_EP_POST_ATTENTION_FIXED_CAPACITY_ROUTE_PLAN
export DS4_V100_TP_EP_POST_ATTENTION_DEVICE_ACTUAL_ROUTE_SYNC
export DS4_V100_TP_EP_POST_ATTENTION_SLOT_MAJOR_FFN_NORM
export DS4_V100_TP_EP_POST_ATTENTION_SKIP_SLOT_MAJOR_FFN_NORM
export DS4_V100_TP_EP_POST_ATTENTION_MASKED_COMPACT_COPY
export DS4_V100_TP_EP_TRUE_SHARED_FFN
export DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT
export DS4_V100_TP_EP_TRUE_DS4_ATTENTION_OUTPUT_NCCL_ALLGATHER
export DS4_V100_TP_EP_TRUE_DS4_POST_ATTENTION_FFN_INPUT
export DS4_V100_TP_EP_TRUE_DS4_SEMANTIC_SKIP_STATS
export DS4_V100_TP_EP_REFERENCE_HC_REDUCE
export DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD
export DS4_V100_TP_EP_KV_ALL_SLOTS
export DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB
export DS4_V100_TP_EP_DEFER_NCCL_INIT
export DS4_V100_NCCL_TOPOLOGY_POLICY
export DS4_V100_NCCL_NO_SYS_RING
export DS4_V100_NCCL_ALLOW_VISIBLE_REMAP
export DS4_V100_NCCL_ALGO
export DS4_V100_NCCL_PROTO
export NCCL_ALGO
export NCCL_PROTO
export NCCL_RINGS
export NCCL_P2P_LEVEL
export DS4_V100_TP_EP_FP8_E5M2_KV
export DS4_V100_TP_EP_VRAM_REPORT
export DS4_V100_TP_EP_VRAM_MIN_FREE_MIB
export DS4_V100_TP_EP_NCCL_MIN_FREE_MIB
export DS4_V100_TP_EP_VERBOSE
export DS4_LOCK_FILE
export DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE
export DS4_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2="$DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2"
export DS4_V100_F8_SHARED_DOWN_ADD
export DS4_V100_TURBOMIND_LIB
export DS4_CUDA_TENSOR_POOL="$cuda_tensor_pool"
export DS4_CUDA_TENSOR_POOL_MAX_MIB="$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB"
export DS4_CUDA_F8_ROWPAIR="$DS4_V100_CUDA_F8_ROWPAIR"
export DS4_CUDA_F8_ROW4="$DS4_V100_CUDA_F8_ROW4"
export DS4_CUDA_F8_WARP_SCALE="$DS4_V100_CUDA_F8_WARP_SCALE"
export DS4_CUDA_F8_GROUPED_DS4_FAST="$DS4_V100_CUDA_F8_GROUPED_DS4_FAST"
export DS4_CUDA_F8_HMMA_SHARED_DOWN="$DS4_V100_CUDA_F8_HMMA_SHARED_DOWN"
export DS4_CUDA_F8_HMMA_PAIR_SWIGLU="$DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU"
export DS4_CUDA_F8_HMMA_ATTN_BATCH="$DS4_V100_CUDA_F8_HMMA_ATTN_BATCH"
export DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH="$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH"
export DS4_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE="$DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_SINGLE"
export DS4_CUDA_F8_HMMA_SINGLE="$DS4_V100_CUDA_F8_HMMA_SINGLE"
export DS4_CUDA_F8_SHARED_DOWN_ADD="$DS4_V100_F8_SHARED_DOWN_ADD"
export DS4_V100_ENABLE_BATCH_ATTN_PROJ
export DS4_V100_BATCH_ATTN_OUTPUT_A
export DS4_V100_BATCH_ATTN_OUTPUT_B
exec "${cmd[@]}"
