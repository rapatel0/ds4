#!/usr/bin/env bash
set -eu

env_file=""
mode="run"
allow_missing=0

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-run-tp-ep-appliance.sh [options]

Options:
  --env FILE        load deployment environment file
  --check           validate config and exit without starting the server
  --print-command   validate config, print the resolved command, and exit
  --allow-missing   allow missing model/GPU files during local config checks
  --help            show this help

This launcher is TP/EP-only and execs appliance/ds4-v100-tp-ep-appliance.
MTP serving is intentionally unsupported on the TP/EP path.
USAGE
}

fail() {
    echo "ds4-v100-run-tp-ep-appliance: $*" >&2
    exit 1
}

warn() {
    echo "ds4-v100-run-tp-ep-appliance: warning: $*" >&2
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

: "${DS4_V100_MODEL:=/models/DSv4-Flash-256e-fixed.gguf}"
: "${DS4_V100_APPLIANCE_DIR:=}"
: "${DS4_V100_CTX:=262144}"
: "${DS4_V100_SLOTS:=32}"
: "${DS4_V100_ACTIVE_MICROBATCH:=$DS4_V100_SLOTS}"
: "${DS4_V100_MICROBATCH_WAIT_US:=auto}"
: "${DS4_V100_TOKENS:=2}"
# Decode graph mode. Promoted default is no-suffix full capture (Sprint 580):
#   full   = no-suffix full-capture persistent replay (default; ~1.2x wall / ~1.5x
#            decode vs suffix at 32 slots / 256K, parity within the determinism
#            floor after the Sprint 579 dense<->rank barrier fix)
#   suffix = suffix-stage replay (the prior promoted default; opt-out)
#   eager  = no decode graph
# Back-compat: DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY, if set, overrides the mode
# (1/true/on -> suffix, 0/false/off -> eager). Leaving it unset selects the
# promoted full-capture default.
: "${DS4_V100_TP_EP_DECODE_GRAPH_MODE:=full}"
: "${DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY:=}"
: "${DS4_V100_TP_EP_EXTRA_ARGS:=}"
: "${DS4_V100_TP_EP_VRAM_MIN_FREE_MIB:=64}"
: "${DS4_V100_TP_EP_NCCL_MIN_FREE_MIB:=0}"
: "${DS4_V100_TP_EP_BIN:=./appliance/ds4-v100-tp-ep-appliance}"
: "${DS4_V100_TP_EP_CONTRACT:=/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv}"
: "${DS4_V100_TP_EP_TM_INDEX:=}"
: "${DS4_V100_TP_EP_TOKENIZER_MODEL:=$DS4_V100_MODEL}"
: "${DS4_V100_TP_EP_POSITION:=100000}"
: "${DS4_V100_TURBOMIND_LIB:=./build/turbomind-v100/libggml-turbomind.so}"
: "${DS4_V100_HOST:=127.0.0.1}"
: "${DS4_V100_PORT:=18080}"
: "${DS4_V100_ALLOW_NONLOCAL_HOST:=0}"
: "${DS4_V100_CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${DS4_V100_REQUIRE_GPUS:=8}"
: "${DS4_V100_RESERVE_MIB:=4096}"
: "${DS4_V100_MAX_REQUESTS:=0}"
: "${DS4_V100_LOG_DIR:=logs/v100-tp-ep-appliance}"
: "${DS4_LOCK_FILE:=$DS4_V100_LOG_DIR/ds4.lock}"
: "${DS4_V100_CUDA_LIB_DIR:=auto}"
: "${DS4_V100_NCCL_TOPOLOGY_POLICY:=no-sys}"
: "${DS4_V100_NCCL_NO_SYS_RING:=0 3 2 1 5 7 6 4}"
: "${DS4_V100_NCCL_ALLOW_VISIBLE_REMAP:=0}"
: "${DS4_V100_NCCL_ALGO:=auto}"
: "${DS4_V100_NCCL_PROTO:=auto}"
: "${DS4_V100_MTP_SERVING:=off}"
# Sprint 597 Phase 2: EP sub-stage profiler (default off). When 1, the
# appliance arms per-rank CUDA-event markers at the EP sub-stage boundaries
# and emits tp_ep_ep_stage_profile TSV lines; flag-off is byte-identical.
: "${DS4_V100_TP_EP_EP_STAGE_PROFILE:=0}"
# Sprint 598 B2-C: EP-return transport for the full-capture graph branch.
#   nccl = grouped per-source NCCL broadcasts captured in-graph (promoted
#          default after the Sprint 598 reference gate: EP return
#          6.92 -> 0.61 ms/layer, decode-domain 71.2 -> 162.1 tok/s (2.28x),
#          tolerance bit-exact vs the s597 control)
#   copy = the prior per-pair copy_f32 remote-load path (rollback flag)
: "${DS4_V100_TP_EP_EP_RETURN_TRANSPORT:=nccl}"

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

for name in \
    DS4_V100_CTX DS4_V100_SLOTS DS4_V100_ACTIVE_MICROBATCH \
    DS4_V100_TOKENS DS4_V100_PORT DS4_V100_ALLOW_NONLOCAL_HOST \
    DS4_V100_REQUIRE_GPUS DS4_V100_RESERVE_MIB DS4_V100_MAX_REQUESTS \
    DS4_V100_TP_EP_POSITION DS4_V100_TP_EP_VRAM_MIN_FREE_MIB \
    DS4_V100_TP_EP_NCCL_MIN_FREE_MIB; do
    eval "value=\${$name}"
    is_uint "$value" || fail "$name must be an integer"
done
if [ "$DS4_V100_MICROBATCH_WAIT_US" != "auto" ]; then
    is_uint "$DS4_V100_MICROBATCH_WAIT_US" || fail "DS4_V100_MICROBATCH_WAIT_US must be auto or an integer"
    [ "$DS4_V100_MICROBATCH_WAIT_US" -le 1000000 ] || fail "DS4_V100_MICROBATCH_WAIT_US must be <= 1000000"
fi

[ "$DS4_V100_CTX" -eq 262144 ] || fail "TP/EP currently requires DS4_V100_CTX=262144"
[ "$DS4_V100_SLOTS" -ge 1 ] && [ "$DS4_V100_SLOTS" -le 32 ] || fail "TP/EP supports DS4_V100_SLOTS in [1,32]"
[ "$DS4_V100_ACTIVE_MICROBATCH" -eq "$DS4_V100_SLOTS" ] || fail "TP/EP requires active_microbatch == slots"
[ "$DS4_V100_TOKENS" -ge 1 ] || fail "DS4_V100_TOKENS must be positive"
[ "$DS4_V100_PORT" -ge 1 ] && [ "$DS4_V100_PORT" -le 65535 ] || fail "DS4_V100_PORT must be in [1,65535]"
case "$DS4_V100_ALLOW_NONLOCAL_HOST" in
    0|1) ;;
    *) fail "DS4_V100_ALLOW_NONLOCAL_HOST must be 0 or 1" ;;
esac
if [ "$DS4_V100_ALLOW_NONLOCAL_HOST" -ne 1 ]; then
    case "$DS4_V100_HOST" in
        127.*|localhost|::1) ;;
        *) fail "refusing non-local host $DS4_V100_HOST without DS4_V100_ALLOW_NONLOCAL_HOST=1" ;;
    esac
fi
case "$DS4_V100_MTP_SERVING" in
    off|false|0) ;;
    *) fail "TP/EP does not support MTP yet; set DS4_V100_MTP_SERVING=off" ;;
esac
case "$DS4_V100_TP_EP_EP_STAGE_PROFILE" in
    0|1) ;;
    *) fail "DS4_V100_TP_EP_EP_STAGE_PROFILE must be 0 or 1" ;;
esac
case "$DS4_V100_TP_EP_EP_RETURN_TRANSPORT" in
    copy|nccl) ;;
    *) fail "DS4_V100_TP_EP_EP_RETURN_TRANSPORT must be copy or nccl" ;;
esac
# Back-compat: GRAPH_SUFFIX_REPLAY, if explicitly set, overrides the graph mode.
if [ -n "$DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY" ]; then
    case "$DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY" in
        1|true|on) DS4_V100_TP_EP_DECODE_GRAPH_MODE=suffix ;;
        0|false|off) DS4_V100_TP_EP_DECODE_GRAPH_MODE=eager ;;
        *) fail "DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY must be 0/1, true/false, or on/off" ;;
    esac
fi
case "$DS4_V100_TP_EP_DECODE_GRAPH_MODE" in
    full|suffix|eager) ;;
    *) fail "DS4_V100_TP_EP_DECODE_GRAPH_MODE must be full, suffix, or eager" ;;
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

case "$DS4_V100_NCCL_ALGO" in
    auto) unset NCCL_ALGO ;;
    "") ;;
    *) export NCCL_ALGO="$DS4_V100_NCCL_ALGO" ;;
esac
case "$DS4_V100_NCCL_PROTO" in
    auto) unset NCCL_PROTO ;;
    "") ;;
    *) export NCCL_PROTO="$DS4_V100_NCCL_PROTO" ;;
esac
case "$DS4_V100_NCCL_TOPOLOGY_POLICY" in
    no-sys)
        export NCCL_P2P_LEVEL=NVL
        if [ "$DS4_V100_NCCL_ALLOW_VISIBLE_REMAP" = "1" ]; then
            export NCCL_RINGS="$DS4_V100_NCCL_NO_SYS_RING"
        fi
        ;;
    default|auto) ;;
    *) fail "DS4_V100_NCCL_TOPOLOGY_POLICY must be no-sys, default, or auto" ;;
esac

if [ -z "$DS4_V100_TP_EP_TM_INDEX" ] && [ -n "$DS4_V100_APPLIANCE_DIR" ]; then
    DS4_V100_TP_EP_TM_INDEX="$DS4_V100_APPLIANCE_DIR/turbomind-pack-index.tsv"
fi

require_exec "$DS4_V100_TP_EP_BIN"
require_dir "appliance directory" "$DS4_V100_APPLIANCE_DIR"
require_file "appliance pack index" "$DS4_V100_APPLIANCE_DIR/pack-index.tsv"
require_file "TP/EP contract" "$DS4_V100_TP_EP_CONTRACT"
require_file "TP/EP TurboMind index" "$DS4_V100_TP_EP_TM_INDEX"
require_file "TurboMind library" "$DS4_V100_TURBOMIND_LIB"
if [ -n "$DS4_V100_TP_EP_TOKENIZER_MODEL" ]; then
    require_file "TP/EP tokenizer model" "$DS4_V100_TP_EP_TOKENIZER_MODEL"
fi
for gpu in 0 1 2 3 4 5 6 7; do
    require_file "appliance gpu${gpu} shard" "$DS4_V100_APPLIANCE_DIR/gpu${gpu}.weights"
done
check_gpu_reserve

cmd=(
    "$DS4_V100_TP_EP_BIN"
    --serve-http
    --pack-dir "$DS4_V100_APPLIANCE_DIR"
    --contract "$DS4_V100_TP_EP_CONTRACT"
    --tm-index "$DS4_V100_TP_EP_TM_INDEX"
    --lib "$DS4_V100_TURBOMIND_LIB"
    --slots "$DS4_V100_SLOTS"
    --position "$DS4_V100_TP_EP_POSITION"
    --decode-steps "$DS4_V100_TOKENS"
    --host "$DS4_V100_HOST"
    --port "$DS4_V100_PORT"
    --microbatch-wait-us "$microbatch_wait_us"
)
if [ -n "$DS4_V100_TP_EP_TOKENIZER_MODEL" ]; then
    cmd+=(--tokenizer-model "$DS4_V100_TP_EP_TOKENIZER_MODEL")
fi
if [ "$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB" -gt 0 ]; then
    cmd+=(--vram-min-free-mib "$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB")
fi
if [ "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB" -gt 0 ]; then
    cmd+=(--nccl-min-free-mib "$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB")
fi
if [ "$DS4_V100_MAX_REQUESTS" -gt 0 ]; then
    cmd+=(--max-requests "$DS4_V100_MAX_REQUESTS")
fi
case "$DS4_V100_TP_EP_DECODE_GRAPH_MODE" in
    suffix)
        cmd+=(
            --decode-cudagraph-gate
            --decode-cudagraph-replay-probe-gate
            --decode-cudagraph-persistent-replay-gate
            --decode-cudagraph-suffix-stage
            compose_eager_final_hc
        )
        ;;
    full)
        cmd+=(
            --decode-cudagraph-gate
            --decode-cudagraph-replay-probe-gate
            --decode-cudagraph-persistent-replay-gate
        )
        ;;
    eager) ;;
esac
if [ -n "$DS4_V100_TP_EP_EXTRA_ARGS" ]; then
    while IFS= read -r extra_arg; do
        [ -n "$extra_arg" ] || continue
        cmd+=("$extra_arg")
    done <<< "$DS4_V100_TP_EP_EXTRA_ARGS"
fi

print_resolved() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$DS4_V100_CUDA_VISIBLE_DEVICES"
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

if [ "$mode" = "check" ]; then
    echo "ds4-v100-run-tp-ep-appliance: config ok host=$DS4_V100_HOST port=$DS4_V100_PORT ctx=$DS4_V100_CTX slots=$DS4_V100_SLOTS microbatch_wait_us=$microbatch_wait_us tokens=$DS4_V100_TOKENS decode_graph_mode=$DS4_V100_TP_EP_DECODE_GRAPH_MODE ep_stage_profile=$DS4_V100_TP_EP_EP_STAGE_PROFILE ep_return_transport=$DS4_V100_TP_EP_EP_RETURN_TRANSPORT tp_ep_bin=$DS4_V100_TP_EP_BIN tp_ep_contract=$DS4_V100_TP_EP_CONTRACT tp_ep_tm_index=$DS4_V100_TP_EP_TM_INDEX mtp=off"
    exit 0
fi
if [ "$mode" = "print" ]; then
    print_resolved
    exit 0
fi

mkdir -p "$DS4_V100_LOG_DIR"
{
    echo "DS4_V100_APPLIANCE_DIR=$DS4_V100_APPLIANCE_DIR"
    echo "DS4_V100_CTX=$DS4_V100_CTX"
    echo "DS4_V100_SLOTS=$DS4_V100_SLOTS"
    echo "DS4_V100_ACTIVE_MICROBATCH=$DS4_V100_ACTIVE_MICROBATCH"
    echo "DS4_V100_MICROBATCH_WAIT_US=$DS4_V100_MICROBATCH_WAIT_US"
    echo "DS4_V100_MICROBATCH_WAIT_US_RESOLVED=$microbatch_wait_us"
    echo "DS4_V100_TOKENS=$DS4_V100_TOKENS"
    echo "DS4_V100_TP_EP_DECODE_GRAPH_MODE=$DS4_V100_TP_EP_DECODE_GRAPH_MODE"
    echo "DS4_V100_TP_EP_EP_STAGE_PROFILE=$DS4_V100_TP_EP_EP_STAGE_PROFILE"
    echo "DS4_V100_TP_EP_EP_RETURN_TRANSPORT=$DS4_V100_TP_EP_EP_RETURN_TRANSPORT"
    echo "DS4_V100_TP_EP_EXTRA_ARGS=$DS4_V100_TP_EP_EXTRA_ARGS"
    echo "DS4_V100_TP_EP_VRAM_MIN_FREE_MIB=$DS4_V100_TP_EP_VRAM_MIN_FREE_MIB"
    echo "DS4_V100_TP_EP_NCCL_MIN_FREE_MIB=$DS4_V100_TP_EP_NCCL_MIN_FREE_MIB"
    echo "DS4_V100_TP_EP_BIN=$DS4_V100_TP_EP_BIN"
    echo "DS4_V100_TP_EP_CONTRACT=$DS4_V100_TP_EP_CONTRACT"
    echo "DS4_V100_TP_EP_TM_INDEX=$DS4_V100_TP_EP_TM_INDEX"
    echo "DS4_V100_TP_EP_TOKENIZER_MODEL=$DS4_V100_TP_EP_TOKENIZER_MODEL"
    echo "DS4_V100_TP_EP_POSITION=$DS4_V100_TP_EP_POSITION"
    echo "DS4_V100_TURBOMIND_LIB=$DS4_V100_TURBOMIND_LIB"
    echo "DS4_V100_HOST=$DS4_V100_HOST"
    echo "DS4_V100_PORT=$DS4_V100_PORT"
    echo "DS4_V100_CUDA_VISIBLE_DEVICES=$DS4_V100_CUDA_VISIBLE_DEVICES"
    echo "DS4_V100_CUDA_LIB_DIR=$DS4_V100_CUDA_LIB_DIR"
    echo "DS4_V100_CUDA_LIB_DIR_RESOLVED=$cuda_lib_dir"
    echo "DS4_V100_REQUIRE_GPUS=$DS4_V100_REQUIRE_GPUS"
    echo "DS4_V100_RESERVE_MIB=$DS4_V100_RESERVE_MIB"
    echo "DS4_LOCK_FILE=$DS4_LOCK_FILE"
    echo "DS4_V100_SERVE_MODE=tp-ep"
    echo "DS4_V100_MTP_SERVING=off"
} >"$DS4_V100_LOG_DIR/startup.env"
print_resolved >"$DS4_V100_LOG_DIR/command.txt"

export CUDA_VISIBLE_DEVICES="$DS4_V100_CUDA_VISIBLE_DEVICES"
if [ -n "$cuda_lib_dir" ]; then
    export LD_LIBRARY_PATH="$cuda_lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export DS4_LOCK_FILE
export DS4_V100_TURBOMIND_LIB
export DS4_V100_TP_EP_EP_STAGE_PROFILE
export DS4_V100_TP_EP_EP_RETURN_TRANSPORT
export DS4_V100_NCCL_TOPOLOGY_POLICY
export DS4_V100_NCCL_NO_SYS_RING
export DS4_V100_NCCL_ALLOW_VISIBLE_REMAP
export DS4_V100_NCCL_ALGO
export DS4_V100_NCCL_PROTO

exec "${cmd[@]}"
