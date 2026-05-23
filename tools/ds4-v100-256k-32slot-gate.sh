#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
ctx="262144"
slots="32"
tokens="64"
requests="32"
warmup_requests="0"
port_base="19100"
log_dir="/workspace/logs/sprint217-256k-32slot-gate"
bench_bin="./tools/ds4-v100-sustained-decode-bench.sh"
launcher_bin="./tools/ds4-v100-run-appliance.sh"
memory_promote_max_mib="30000"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-256k-32slot-gate.sh [options]

Runs the focused 256K/32-slot admission gate using the existing experimental
launcher cap override, then writes a promotion verdict.

Options:
  --model FILE                 base DS4 model
  --mtp-model FILE             MTP sidecar model path, passed through for bench compatibility
  --appliance-dir DIR          appliance pack dir
  --ctx N                      context tokens, default 262144
  --slots N                    slots, default 32
  --tokens N                   generated tokens per request, default 64
  --requests N                 timed requests, default 32
  --warmup-requests N          warmup requests, default 0
  --port-base N                first port, default 19100
  --log-dir DIR                output dir
  --bench-bin FILE             sustained decode bench path
  --launcher-bin FILE          appliance launcher path
  --memory-promote-max-mib N   max observed memory threshold, default 30000
  --help                       show this help
USAGE
}

fail() {
    echo "ds4-v100-256k-32slot-gate: $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --mtp-model) mtp_model="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --slots) slots="$2"; shift 2 ;;
        --tokens) tokens="$2"; shift 2 ;;
        --requests) requests="$2"; shift 2 ;;
        --warmup-requests) warmup_requests="$2"; shift 2 ;;
        --port-base) port_base="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --bench-bin) bench_bin="$2"; shift 2 ;;
        --launcher-bin) launcher_bin="$2"; shift 2 ;;
        --memory-promote-max-mib) memory_promote_max_mib="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done

for n in "$ctx" "$slots" "$tokens" "$requests" "$warmup_requests" "$port_base" "$memory_promote_max_mib"; do
    case "$n" in ''|*[!0-9]*) fail "numeric option expected, got '$n'" ;; esac
done
[ "$tokens" -ge 2 ] || fail "--tokens must be >= 2"
[ "$slots" -ge 1 ] || fail "--slots must be >= 1"
[ -x "$bench_bin" ] || fail "missing executable bench script $bench_bin"
[ -x "$launcher_bin" ] || fail "missing executable launcher $launcher_bin"
[ -f "$model" ] || fail "missing model $model"
[ -d "$appliance_dir" ] || fail "missing appliance dir $appliance_dir"

mkdir -p "$log_dir"

export DS4_V100_CUDA_TENSOR_POOL="${DS4_V100_CUDA_TENSOR_POOL:-1}"
export DS4_CUDA_TENSOR_POOL="${DS4_CUDA_TENSOR_POOL:-1}"
export DS4_CUDA_F8_ROWPAIR="${DS4_CUDA_F8_ROWPAIR:-1}"
export DS4_CUDA_F8_GROUPED_DS4_FAST="${DS4_CUDA_F8_GROUPED_DS4_FAST:-1}"
export DS4_CUDA_F8_HMMA_PAIR_SWIGLU="${DS4_CUDA_F8_HMMA_PAIR_SWIGLU:-1}"
export DS4_CUDA_F8_HMMA_ATTN_BATCH="${DS4_CUDA_F8_HMMA_ATTN_BATCH:-1}"
export DS4_V100_ENABLE_BATCH_ATTN_PROJ="${DS4_V100_ENABLE_BATCH_ATTN_PROJ:-1}"
export DS4_V100_BATCH_SHARED_F8="${DS4_V100_BATCH_SHARED_F8:-1}"
export DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS="${DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS:-1}"
export DS4_V100_TURBOMIND_FUSED_GATE_UP="${DS4_V100_TURBOMIND_FUSED_GATE_UP:-1}"
export DS4_V100_TURBOMIND_GATED_SILU="${DS4_V100_TURBOMIND_GATED_SILU:-1}"
export DS4_V100_TURBOMIND_COMPACT_SCHEDULE="${DS4_V100_TURBOMIND_COMPACT_SCHEDULE:-1}"
export DS4_V100_TURBOMIND_ROUTED_EXECUTOR="${DS4_V100_TURBOMIND_ROUTED_EXECUTOR:-fused6_reduce}"
export DS4_V100_TURBOMIND_GRAPH="${DS4_V100_TURBOMIND_GRAPH:-1}"
export DS4_V100_TURBOMIND_LIB="${DS4_V100_TURBOMIND_LIB:-./build/turbomind-v100/libggml-turbomind.so}"

check_dir="$log_dir/launcher-check"
case_dir="$log_dir/probe"
mkdir -p "$check_dir" "$case_dir"

set +e
DS4_V100_MODEL="$model" \
DS4_V100_MTP_MODEL="$mtp_model" \
DS4_V100_APPLIANCE_DIR="$appliance_dir" \
DS4_V100_CTX="$ctx" \
DS4_V100_SLOTS="$slots" \
DS4_V100_ACTIVE_MICROBATCH="$slots" \
DS4_V100_TOKENS="$tokens" \
DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP="$slots" \
DS4_V100_ASYNC_PIPELINE_MODE=per-step \
DS4_V100_ASYNC_EVENT_HANDOFF=1 \
DS4_V100_MTP_SERVING=off \
"$launcher_bin" --check >"$check_dir/stdout.log" 2>"$check_dir/stderr.log"
check_rc=$?
set -e
printf '%s\n' "$check_rc" >"$check_dir/exit_code.txt"

bench_rc=99
if [ "$check_rc" -eq 0 ]; then
    cmd=(
        "$bench_bin"
        --model "$model"
        --mtp-model "$mtp_model"
        --appliance-dir "$appliance_dir"
        --ctx-tiers "$ctx"
        --slot-tiers "$slots"
        --tokens "$tokens"
        --requests "$requests"
        --warmup-requests "$warmup_requests"
        --port-base "$port_base"
        --microbatch-wait-us 200000
        --async-pipeline-mode per-step
        --async-event-handoff
        --mtp-serving off
        --log-dir "$case_dir"
    )
    printf '%q ' "${cmd[@]}" >"$case_dir/command.txt"
    printf '\n' >>"$case_dir/command.txt"
    set +e
    "${cmd[@]}" >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
    bench_rc=$?
    set -e
fi
printf '%s\n' "$bench_rc" >"$case_dir/exit_code.txt"

python3 - "$log_dir" "$check_rc" "$bench_rc" "$memory_promote_max_mib" "$slots" "$tokens" <<'PY'
import json
import os
import sys

root, check_rc, bench_rc, mem_limit, slots, tokens = sys.argv[1:]
check_rc = int(check_rc)
bench_rc = int(bench_rc)
mem_limit = float(mem_limit)
slots = int(slots)
tokens = int(tokens)

summary = {
    "schema": "ds4_v100_256k_32slot_gate.v1",
    "launcher_check_exit_code": check_rc,
    "bench_exit_code": bench_rc,
    "memory_promote_max_mib": mem_limit,
    "decision": "fail_launcher_check" if check_rc else "fail_bench",
}

case_paths = [
    os.path.join(root, "probe", "sustained_decode.json"),
    os.path.join(
        root,
        "probe",
        "cases",
        f"case_1_ctx262144_s{slots}_sequential_mtpoff_tok{tokens}",
        "result.json",
    ),
]
for case_path in case_paths:
    if not os.path.exists(case_path):
        continue
    with open(case_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "cases" in data:
        cases = data.get("cases", [])
        if not cases:
            continue
        c = cases[0]
    else:
        c = data
    if c:
        gpu = c.get("gpu_utilization", {}) if isinstance(c.get("gpu_utilization", {}), dict) else {}
        max_mem = float(gpu.get("max_memory_used_mib", 0.0) or 0.0)
        token_match = int(c.get("token_match", 0) or 0)
        token_mismatch = int(c.get("token_mismatch", 0) or 0)
        status_other = int(c.get("status_other", 0) or 0)
        summary.update({
            "ctx": 262144,
            "slots": slots,
            "status_200": int(c.get("status_200", 0) or 0),
            "status_other": status_other,
            "token_match": token_match,
            "token_mismatch": token_mismatch,
            "generated_tokens_per_second": float(c.get("aggregate_generated_tokens_per_second", 0.0) or 0.0),
            "prompt_tokens_per_second": float(c.get("aggregate_prompt_tokens_per_second", 0.0) or 0.0),
            "continuation_tokens_per_second": float(c.get("aggregate_continuation_tokens_per_second", 0.0) or 0.0),
            "avg_gpu_util_percent": float(gpu.get("avg_gpu_util_percent", 0.0) or 0.0),
            "max_gpu_util_percent": float(gpu.get("max_gpu_util_percent", 0.0) or 0.0),
            "max_memory_used_mib": max_mem,
        })
        if bench_rc == 0 and token_match == slots and token_mismatch == 0 and status_other == 0 and max_mem < mem_limit:
            summary["decision"] = "pass_promote_256k_32slot"
        elif bench_rc == 0:
            summary["decision"] = "pass_keep_cap_due_to_reserve_or_correctness"
        break

json_path = os.path.join(root, "slot_gate_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True, indent=2)
    f.write("\n")

md_path = os.path.join(root, "slot_gate_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# DS4 V100 256K {slots}-Slot Gate\n\n")
    f.write(f"Decision: `{summary['decision']}`\n\n")
    f.write(f"Launcher check exit code: `{check_rc}`\n\n")
    f.write(f"Benchmark exit code: `{bench_rc}`\n\n")
    if "generated_tokens_per_second" in summary:
        f.write("| Ctx | Slots | Generated tok/s | Prompt tok/s | Continuation tok/s | Match | Avg GPU util | Max GPU util | Max memory MiB |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        f.write(
            f"| {summary['ctx']} | {summary['slots']} | "
            f"{summary['generated_tokens_per_second']:.6f} | "
            f"{summary['prompt_tokens_per_second']:.6f} | "
            f"{summary['continuation_tokens_per_second']:.6f} | "
            f"{summary['token_match']}/{summary['token_match'] + summary['token_mismatch']} | "
            f"{summary['avg_gpu_util_percent']:.3f}% | "
            f"{summary['max_gpu_util_percent']:.3f}% | "
            f"{summary['max_memory_used_mib']:.1f} |\n"
        )

print(json_path)
print(md_path)
PY

cat "$log_dir/slot_gate_summary.md"
case "$(python3 - "$log_dir/slot_gate_summary.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("decision", ""))
PY
)" in
    fail_launcher_check|fail_bench)
        exit 1
        ;;
    *)
        exit 0
        ;;
esac
