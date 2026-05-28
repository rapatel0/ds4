#!/usr/bin/env bash
set -u

appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
log_dir="/workspace/logs/sprint215-practical-serving-matrix"
port_base="18600"
tokens="64"
quick="0"
run_forced_256k32="0"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-practical-serving-matrix.sh [options]

Options:
  --appliance-dir DIR       production appliance pack directory
  --model FILE              DS4 Flash source GGUF
  --mtp-model FILE          MTP sidecar GGUF
  --log-dir DIR             output log directory
  --port-base N             first server port
  --tokens N                tokens/request for base cases, default 64
  --quick                   smaller request counts for smoke/debug runs
  --run-forced-256k-32      actually run 32-slot/256K with the sustained bench
  --help                    show this help

By default the 32-slot/256K case records the launcher admission failure instead
of bypassing the production cap. Use --run-forced-256k-32 only for an explicit
VRAM experiment after checking the node is idle.
USAGE
}

fail() {
    echo "ds4-v100-practical-serving-matrix: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --appliance-dir)
            [ "$#" -ge 2 ] || fail "--appliance-dir requires a value"
            appliance_dir="$2"
            shift 2
            ;;
        --model)
            [ "$#" -ge 2 ] || fail "--model requires a value"
            model="$2"
            shift 2
            ;;
        --mtp-model)
            [ "$#" -ge 2 ] || fail "--mtp-model requires a value"
            mtp_model="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || fail "--log-dir requires a value"
            log_dir="$2"
            shift 2
            ;;
        --port-base)
            [ "$#" -ge 2 ] || fail "--port-base requires a value"
            port_base="$2"
            shift 2
            ;;
        --tokens)
            [ "$#" -ge 2 ] || fail "--tokens requires a value"
            tokens="$2"
            shift 2
            ;;
        --quick)
            quick="1"
            shift
            ;;
        --run-forced-256k-32)
            run_forced_256k32="1"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown option: $1"
            ;;
    esac
done

[ -d "$appliance_dir" ] || fail "missing appliance dir $appliance_dir"
[ -f "$appliance_dir/pack-index.tsv" ] || fail "missing pack index $appliance_dir/pack-index.tsv"
[ -f "$appliance_dir/turbomind-pack-index.tsv" ] || fail "missing TurboMind index $appliance_dir/turbomind-pack-index.tsv"
[ -f "$model" ] || fail "missing model $model"
[ -x ./tools/ds4-v100-replay ] || fail "missing ./tools/ds4-v100-replay"
[ -x ./tools/ds4-v100-sustained-decode-bench.sh ] || fail "missing sustained decode bench"
case "$tokens" in ''|0|1|*[!0-9]*) fail "--tokens must be an integer >= 2" ;; esac
case "$port_base" in ''|0|*[!0-9]*) fail "--port-base must be a positive integer" ;; esac

mkdir -p "$log_dir"
summary="$log_dir/matrix.tsv"
: >"$summary"
printf "case\tctx\tslots\tactive_microbatch\tmtp\tstatus\tlog_dir\n" >>"$summary"

base_requests_16="16"
base_requests_32="32"
mtp_verify_tokens="4"
mtp_commit_tokens="16"
mtp_commit_requests="1"
warmup="1"
if [ "$quick" -eq 1 ]; then
    base_requests_16="2"
    base_requests_32="2"
    tokens="4"
    mtp_verify_tokens="2"
    mtp_commit_tokens="4"
    warmup="0"
fi

run_sustained_case() {
    local name="$1"
    local ctx="$2"
    local slots="$3"
    local reqs="$4"
    local case_tokens="$5"
    local mtp="$6"
    local port="$7"
    local case_dir="$log_dir/$name"
    mkdir -p "$case_dir"
    {
        echo "case=$name"
        echo "ctx=$ctx"
        echo "slots=$slots"
        echo "requests=$reqs"
        echo "tokens=$case_tokens"
        echo "mtp=$mtp"
        date -u +"started_utc=%Y-%m-%dT%H:%M:%SZ"
    } >"$case_dir/command.txt"
    local rc=0
    DS4_V100_CUDA_TENSOR_POOL=1 \
    DS4_CUDA_TENSOR_POOL=1 \
    DS4_CUDA_TENSOR_POOL_MAX_MIB="${DS4_CUDA_TENSOR_POOL_MAX_MIB:-2048}" \
    DS4_CUDA_F8_ROWPAIR=1 \
    DS4_CUDA_F8_GROUPED_DS4_FAST=1 \
    DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1 \
    DS4_CUDA_F8_HMMA_ATTN_BATCH=1 \
    DS4_V100_ENABLE_BATCH_ATTN_PROJ=1 \
    DS4_V100_BATCH_SHARED_F8=1 \
    DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1 \
    DS4_V100_TURBOMIND_FUSED_GATE_UP=1 \
    DS4_V100_TURBOMIND_GATED_SILU=1 \
    DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1 \
    DS4_V100_TURBOMIND_ROUTED_EXECUTOR="${DS4_V100_TURBOMIND_ROUTED_EXECUTOR:-fused6_reduce}" \
    DS4_V100_TURBOMIND_GRAPH=1 \
    DS4_V100_TURBOMIND_LIB="${DS4_V100_TURBOMIND_LIB:-./build/turbomind-v100/libggml-turbomind.so}" \
    ./tools/ds4-v100-sustained-decode-bench.sh \
        --model "$model" \
        --mtp-model "$mtp_model" \
        --appliance-dir "$appliance_dir" \
        --ctx-tiers "$ctx" \
        --slot-tiers "$slots" \
        --tokens "$case_tokens" \
        --requests "$reqs" \
        --warmup-requests "$warmup" \
        --port-base "$port" \
        --microbatch-wait-us 200000 \
        --async-pipeline-mode per-step \
        --async-event-handoff \
        --mtp-serving "$mtp" \
        --log-dir "$case_dir" \
        >"$case_dir/stdout.log" 2>"$case_dir/stderr.log" || rc=$?
    echo "$rc" >"$case_dir/exit_code.txt"
    date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ" >>"$case_dir/command.txt"
    if [ "$rc" -eq 0 ]; then
        printf "%s\t%s\t%s\t%s\t%s\tpass\t%s\n" "$name" "$ctx" "$slots" "$slots" "$mtp" "$case_dir" >>"$summary"
    else
        printf "%s\t%s\t%s\t%s\t%s\tfail:%s\t%s\n" "$name" "$ctx" "$slots" "$slots" "$mtp" "$rc" "$case_dir" >>"$summary"
    fi
    return "$rc"
}

record_256k32_admission() {
    local case_dir="$log_dir/forced-256k-32-admission"
    mkdir -p "$case_dir"
    local rc=0
    DS4_V100_CTX=262144 \
    DS4_V100_SLOTS=32 \
    DS4_V100_ACTIVE_MICROBATCH=32 \
    DS4_V100_BIN=/bin/true \
    DS4_V100_MODEL="$model" \
    DS4_V100_APPLIANCE_DIR="$appliance_dir" \
    DS4_V100_LOG_DIR="$case_dir/runtime" \
    DS4_V100_PORT=18699 \
    bash ./tools/ds4-v100-run-pp-appliance.sh \
        >"$case_dir/stdout.log" 2>"$case_dir/stderr.log" || rc=$?
    echo "$rc" >"$case_dir/exit_code.txt"
    if [ "$rc" -eq 0 ]; then
        printf "forced-256k-32-admission\t262144\t32\t32\toff\tunexpected-pass\t%s\n" "$case_dir" >>"$summary"
    else
        printf "forced-256k-32-admission\t262144\t32\t32\toff\tfail-closed:%s\t%s\n" "$rc" "$case_dir" >>"$summary"
    fi
}

run_sustained_case "production-baseline-256k-16" 262144 16 "$base_requests_16" "$tokens" off "$port_base" || true
run_sustained_case "long-throughput-128k-32" 131072 32 "$base_requests_32" "$tokens" off "$((port_base + 10))" || true

if [ "$run_forced_256k32" -eq 1 ]; then
    run_sustained_case "forced-256k-32" 262144 32 "$base_requests_32" "$tokens" off "$((port_base + 20))" || true
else
    record_256k32_admission
fi

if [ -f "$mtp_model" ]; then
    run_sustained_case "mtp-verify-256k-16" 262144 16 "$base_requests_16" "$mtp_verify_tokens" verify "$((port_base + 30))" || true
    run_sustained_case "mtp-commit-256k-1" 262144 1 "$mtp_commit_requests" "$mtp_commit_tokens" commit "$((port_base + 40))" || true
else
    printf "mtp-verify-256k-16\t262144\t16\t16\tverify\tskipped-missing-mtp-model\t%s\n" "$log_dir" >>"$summary"
    printf "mtp-commit-256k-1\t262144\t1\t1\tcommit\tskipped-missing-mtp-model\t%s\n" "$log_dir" >>"$summary"
fi

echo "matrix_summary=$summary"
cat "$summary"
