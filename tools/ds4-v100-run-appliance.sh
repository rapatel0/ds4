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
: "${DS4_V100_ASYNC_EVENT_HANDOFF:=0}"
: "${DS4_V100_STARTUP_WARMUP:=auto}"
: "${DS4_V100_CUDA_PROFILER_WINDOW:=0}"
: "${DS4_V100_CUDA_TENSOR_POOL:=auto}"
: "${DS4_V100_CUDA_TENSOR_POOL_MAX_MIB:=2048}"
: "${DS4_V100_ENABLE_OUTPUT_HEAD_BATCH:=0}"
: "${DS4_V100_BATCH_SHARED_F8:=1}"
: "${DS4_V100_TURBOMIND_ROUTED_FFN:=0}"
: "${DS4_V100_TURBOMIND_STRICT:=0}"
: "${DS4_V100_TURBOMIND_LIB:=./build/turbomind-v100/libggml-turbomind.so}"
: "${DS4_V100_HOST:=127.0.0.1}"
: "${DS4_V100_PORT:=18080}"
: "${DS4_V100_CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${DS4_V100_REQUIRE_GPUS:=8}"
: "${DS4_V100_RESERVE_MIB:=4096}"
: "${DS4_V100_MAX_REQUESTS:=0}"
: "${DS4_V100_LOG_DIR:=logs/v100-appliance}"
: "${DS4_V100_SERVE_MODE:=base}"
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
    base) ;;
    *) fail "only DS4_V100_SERVE_MODE=base is supported by this deployment package" ;;
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
is_uint "$DS4_V100_TOKENS" || fail "DS4_V100_TOKENS must be a positive integer"
is_uint "$DS4_V100_PORT" || fail "DS4_V100_PORT must be a positive integer"
is_uint "$DS4_V100_REQUIRE_GPUS" || fail "DS4_V100_REQUIRE_GPUS must be an integer"
is_uint "$DS4_V100_RESERVE_MIB" || fail "DS4_V100_RESERVE_MIB must be an integer"
is_uint "$DS4_V100_MAX_REQUESTS" || fail "DS4_V100_MAX_REQUESTS must be an integer"
is_uint "$DS4_V100_MTP_TOP_K" || fail "DS4_V100_MTP_TOP_K must be an integer"
is_uint "$DS4_V100_MTP_GPU" || fail "DS4_V100_MTP_GPU must be an integer"

[ "$DS4_V100_CTX" -ge 1 ] || fail "DS4_V100_CTX must be positive"
[ "$DS4_V100_SLOTS" -ge 1 ] && [ "$DS4_V100_SLOTS" -le 8 ] || fail "DS4_V100_SLOTS must be between 1 and 8"
[ "$DS4_V100_ACTIVE_MICROBATCH" -ge 1 ] || fail "DS4_V100_ACTIVE_MICROBATCH must be positive"
[ "$DS4_V100_ACTIVE_MICROBATCH" -le "$DS4_V100_SLOTS" ] || fail "DS4_V100_ACTIVE_MICROBATCH must be in [1,DS4_V100_SLOTS]"
if [ "$DS4_V100_MICROBATCH_WAIT_US" != "auto" ]; then
    [ "$DS4_V100_MICROBATCH_WAIT_US" -le 1000000 ] || fail "DS4_V100_MICROBATCH_WAIT_US must be <= 1000000"
fi
is_uint "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be an integer"
[ "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" -ge 64 ] || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be >= 64"
[ "$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB" -le 8192 ] || fail "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB must be <= 8192"
[ "$DS4_V100_TOKENS" -ge 1 ] || fail "DS4_V100_TOKENS must be positive"
[ "$DS4_V100_TOKENS" -le 64 ] || fail "DS4_V100_TOKENS must be <= 64"
[ "$DS4_V100_MTP_TOP_K" -ge 2 ] && [ "$DS4_V100_MTP_TOP_K" -le 16 ] || fail "DS4_V100_MTP_TOP_K must be between 2 and 16"
[ "$DS4_V100_PORT" -ge 1 ] && [ "$DS4_V100_PORT" -le 65535 ] || fail "DS4_V100_PORT out of range"
[ -n "$DS4_V100_HOST" ] || fail "DS4_V100_HOST must not be empty"
case "$DS4_V100_HOST" in
    127.*|localhost) ;;
    *) fail "default deployment must bind loopback only; got DS4_V100_HOST=$DS4_V100_HOST" ;;
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
    0|false|off) async_event_handoff=0 ;;
    1|true|on) async_event_handoff=1 ;;
    *) fail "DS4_V100_ASYNC_EVENT_HANDOFF must be 0 or 1" ;;
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
        if [ "$DS4_V100_ACTIVE_MICROBATCH" -gt 1 ]; then
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
case "$DS4_V100_TURBOMIND_STRICT" in
    0|false|off) DS4_V100_TURBOMIND_STRICT=0 ;;
    1|true|on) DS4_V100_TURBOMIND_STRICT=1 ;;
    *) fail "DS4_V100_TURBOMIND_STRICT must be 0 or 1" ;;
esac

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
if [ "$async_event_handoff" -eq 1 ] && [ "$async_pipeline_mode" != "per-step" ]; then
    fail "DS4_V100_ASYNC_EVENT_HANDOFF requires resolved async pipeline mode per-step"
fi

require_exec "$DS4_V100_BIN"
require_file "model" "$DS4_V100_MODEL"
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
if [ "$mtp_serving_enabled" -eq 1 ] || [ -n "$DS4_V100_MTP_MODEL" ]; then
    require_file "MTP model" "$DS4_V100_MTP_MODEL"
fi
check_gpu_reserve

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
if [ "$DS4_V100_MAX_REQUESTS" -gt 0 ]; then
    cmd+=(--max-requests "$DS4_V100_MAX_REQUESTS")
fi
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

print_resolved() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$DS4_V100_CUDA_VISIBLE_DEVICES"
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

if [ "$mode" = "check" ]; then
    echo "ds4-v100-run-appliance: config ok mode=$DS4_V100_SERVE_MODE mtp=$DS4_V100_MTP_SERVING host=$DS4_V100_HOST port=$DS4_V100_PORT ctx=$DS4_V100_CTX slots=$DS4_V100_SLOTS active_microbatch=$DS4_V100_ACTIVE_MICROBATCH microbatch_wait_us=$microbatch_wait_us tokens=$DS4_V100_TOKENS async_pipeline_mode=$async_pipeline_mode async_handoff=$async_handoff async_event_handoff=$async_event_handoff startup_warmup=$startup_warmup cuda_profiler_window=$cuda_profiler_window cuda_tensor_pool=$cuda_tensor_pool cuda_tensor_pool_max_mib=$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB batch_shared_f8=$DS4_V100_BATCH_SHARED_F8 appliance_dir=${DS4_V100_APPLIANCE_DIR:-none} turbomind_routed_ffn=$DS4_V100_TURBOMIND_ROUTED_FFN"
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
    echo "DS4_V100_STARTUP_WARMUP=$DS4_V100_STARTUP_WARMUP"
    echo "DS4_V100_STARTUP_WARMUP_RESOLVED=$startup_warmup"
    echo "DS4_V100_CUDA_PROFILER_WINDOW=$DS4_V100_CUDA_PROFILER_WINDOW"
    echo "DS4_V100_CUDA_PROFILER_WINDOW_RESOLVED=$cuda_profiler_window"
    echo "DS4_V100_CUDA_TENSOR_POOL=$DS4_V100_CUDA_TENSOR_POOL"
    echo "DS4_V100_CUDA_TENSOR_POOL_RESOLVED=$cuda_tensor_pool"
    echo "DS4_V100_CUDA_TENSOR_POOL_MAX_MIB=$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB"
    echo "DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=$DS4_V100_ENABLE_OUTPUT_HEAD_BATCH"
    echo "DS4_V100_BATCH_SHARED_F8=$DS4_V100_BATCH_SHARED_F8"
    echo "DS4_V100_TURBOMIND_ROUTED_FFN=$DS4_V100_TURBOMIND_ROUTED_FFN"
    echo "DS4_V100_TURBOMIND_STRICT=$DS4_V100_TURBOMIND_STRICT"
    echo "DS4_V100_TURBOMIND_LIB=$DS4_V100_TURBOMIND_LIB"
    echo "DS4_V100_HOST=$DS4_V100_HOST"
    echo "DS4_V100_PORT=$DS4_V100_PORT"
    echo "DS4_V100_CUDA_VISIBLE_DEVICES=$DS4_V100_CUDA_VISIBLE_DEVICES"
    echo "DS4_V100_REQUIRE_GPUS=$DS4_V100_REQUIRE_GPUS"
    echo "DS4_V100_RESERVE_MIB=$DS4_V100_RESERVE_MIB"
    echo "DS4_V100_SERVE_MODE=$DS4_V100_SERVE_MODE"
    echo "DS4_V100_MTP_SERVING=$DS4_V100_MTP_SERVING"
    echo "DS4_V100_MTP_TOP_K=$DS4_V100_MTP_TOP_K"
    echo "DS4_V100_MTP_GPU=$DS4_V100_MTP_GPU"
} >"$DS4_V100_LOG_DIR/startup.env"
print_resolved >"$DS4_V100_LOG_DIR/command.txt"

export CUDA_VISIBLE_DEVICES="$DS4_V100_CUDA_VISIBLE_DEVICES"
export DS4_V100_BATCH_SHARED_F8
export DS4_V100_TURBOMIND_ROUTED_FFN
export DS4_V100_TURBOMIND_STRICT
export DS4_V100_TURBOMIND_LIB
export DS4_CUDA_TENSOR_POOL="$cuda_tensor_pool"
export DS4_CUDA_TENSOR_POOL_MAX_MIB="$DS4_V100_CUDA_TENSOR_POOL_MAX_MIB"
exec "${cmd[@]}"
