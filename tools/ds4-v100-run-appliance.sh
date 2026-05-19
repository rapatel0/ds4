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
: "${DS4_V100_PACK_INDEX:=docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv}"
: "${DS4_V100_CTX:=1048576}"
: "${DS4_V100_SLOTS:=1}"
: "${DS4_V100_TOKENS:=2}"
: "${DS4_V100_HOST:=127.0.0.1}"
: "${DS4_V100_PORT:=18080}"
: "${DS4_V100_CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${DS4_V100_REQUIRE_GPUS:=8}"
: "${DS4_V100_RESERVE_MIB:=4096}"
: "${DS4_V100_MAX_REQUESTS:=0}"
: "${DS4_V100_LOG_DIR:=logs/v100-appliance}"
: "${DS4_V100_SERVE_MODE:=base}"
: "${DS4_V100_MTP_SERVING:=off}"

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
case "$DS4_V100_MTP_SERVING" in
    off|false|0) ;;
    *) fail "MTP speculative serving is not exposed yet; set DS4_V100_MTP_SERVING=off" ;;
esac

is_uint "$DS4_V100_CTX" || fail "DS4_V100_CTX must be a positive integer"
is_uint "$DS4_V100_SLOTS" || fail "DS4_V100_SLOTS must be a positive integer"
is_uint "$DS4_V100_TOKENS" || fail "DS4_V100_TOKENS must be a positive integer"
is_uint "$DS4_V100_PORT" || fail "DS4_V100_PORT must be a positive integer"
is_uint "$DS4_V100_REQUIRE_GPUS" || fail "DS4_V100_REQUIRE_GPUS must be an integer"
is_uint "$DS4_V100_RESERVE_MIB" || fail "DS4_V100_RESERVE_MIB must be an integer"
is_uint "$DS4_V100_MAX_REQUESTS" || fail "DS4_V100_MAX_REQUESTS must be an integer"

[ "$DS4_V100_SLOTS" -eq 1 ] || fail "only DS4_V100_SLOTS=1 is supported"
[ "$DS4_V100_CTX" -ge 1 ] || fail "DS4_V100_CTX must be positive"
[ "$DS4_V100_TOKENS" -ge 1 ] || fail "DS4_V100_TOKENS must be positive"
[ "$DS4_V100_TOKENS" -le 64 ] || fail "DS4_V100_TOKENS must be <= 64"
[ "$DS4_V100_PORT" -ge 1 ] && [ "$DS4_V100_PORT" -le 65535 ] || fail "DS4_V100_PORT out of range"
[ -n "$DS4_V100_HOST" ] || fail "DS4_V100_HOST must not be empty"
case "$DS4_V100_HOST" in
    127.*|localhost) ;;
    *) fail "default deployment must bind loopback only; got DS4_V100_HOST=$DS4_V100_HOST" ;;
esac

require_exec "$DS4_V100_BIN"
require_file "model" "$DS4_V100_MODEL"
require_file "pack index" "$DS4_V100_PACK_INDEX"
if [ -n "$DS4_V100_MTP_MODEL" ]; then
    require_file "MTP model" "$DS4_V100_MTP_MODEL"
fi
check_gpu_reserve

cmd=(
    "$DS4_V100_BIN"
    --serve
    --model "$DS4_V100_MODEL"
    --index "$DS4_V100_PACK_INDEX"
    --ctx "$DS4_V100_CTX"
    --tokens "$DS4_V100_TOKENS"
    --host "$DS4_V100_HOST"
    --port "$DS4_V100_PORT"
)
if [ "$DS4_V100_MAX_REQUESTS" -gt 0 ]; then
    cmd+=(--max-requests "$DS4_V100_MAX_REQUESTS")
fi

print_resolved() {
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$DS4_V100_CUDA_VISIBLE_DEVICES"
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

if [ "$mode" = "check" ]; then
    echo "ds4-v100-run-appliance: config ok mode=$DS4_V100_SERVE_MODE host=$DS4_V100_HOST port=$DS4_V100_PORT ctx=$DS4_V100_CTX slots=$DS4_V100_SLOTS tokens=$DS4_V100_TOKENS"
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
    echo "DS4_V100_PACK_INDEX=$DS4_V100_PACK_INDEX"
    echo "DS4_V100_CTX=$DS4_V100_CTX"
    echo "DS4_V100_SLOTS=$DS4_V100_SLOTS"
    echo "DS4_V100_TOKENS=$DS4_V100_TOKENS"
    echo "DS4_V100_HOST=$DS4_V100_HOST"
    echo "DS4_V100_PORT=$DS4_V100_PORT"
    echo "DS4_V100_CUDA_VISIBLE_DEVICES=$DS4_V100_CUDA_VISIBLE_DEVICES"
    echo "DS4_V100_REQUIRE_GPUS=$DS4_V100_REQUIRE_GPUS"
    echo "DS4_V100_RESERVE_MIB=$DS4_V100_RESERVE_MIB"
    echo "DS4_V100_SERVE_MODE=$DS4_V100_SERVE_MODE"
    echo "DS4_V100_MTP_SERVING=$DS4_V100_MTP_SERVING"
} >"$DS4_V100_LOG_DIR/startup.env"
print_resolved >"$DS4_V100_LOG_DIR/command.txt"

export CUDA_VISIBLE_DEVICES="$DS4_V100_CUDA_VISIBLE_DEVICES"
exec "${cmd[@]}"
