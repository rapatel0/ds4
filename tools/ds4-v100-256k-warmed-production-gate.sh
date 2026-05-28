#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
ctx="262144"
slots="32"
tokens="64"
requests="64"
warmup_requests="0"
port="19420"
log_dir="/workspace/logs/sprint219-256k-32slot-warmed-gate"
soak_bin="./tools/ds4-v100-appliance-soak.sh"
launcher_bin="./tools/ds4-v100-run-pp-appliance.sh"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-256k-warmed-production-gate.sh [options]

Runs the production launcher path for the validated warmed DS4 V100 serving
target: 32 slots at 256K context. The gate also proves the cold path still
fails closed without an experimental cap.

Options:
  --model FILE            base DS4 model
  --appliance-dir DIR     appliance pack dir
  --ctx N                 context tokens, default 262144
  --slots N               slots and active microbatch, default 32
  --tokens N              generated tokens per request, default 64
  --requests N            timed requests, default 64
  --warmup-requests N     request warmups after server starts, default 0
  --port N                server port, default 19420
  --log-dir DIR           output dir
  --soak-bin FILE         appliance soak script
  --launcher-bin FILE     appliance launcher script
  --help                  show this help
USAGE
}

fail() {
    echo "ds4-v100-256k-warmed-production-gate: $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --slots) slots="$2"; shift 2 ;;
        --tokens) tokens="$2"; shift 2 ;;
        --requests) requests="$2"; shift 2 ;;
        --warmup-requests) warmup_requests="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --soak-bin) soak_bin="$2"; shift 2 ;;
        --launcher-bin) launcher_bin="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done

for n in "$ctx" "$slots" "$tokens" "$requests" "$warmup_requests" "$port"; do
    case "$n" in ''|*[!0-9]*) fail "numeric option expected, got '$n'" ;; esac
done
[ "$ctx" -eq 262144 ] || fail "this gate is scoped to --ctx 262144"
[ "$slots" -ge 17 ] || fail "this gate is scoped to warmed active slots above 16"
[ "$requests" -ge 64 ] || fail "--requests must be >= 64 for the Sprint 219 gate"
[ "$tokens" -eq 64 ] || fail "--tokens must be 64 for the production gate"
[ -x "$soak_bin" ] || fail "missing executable soak script $soak_bin"
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

negative_dir="$log_dir/cold-negative-check"
mkdir -p "$negative_dir"
set +e
DS4_V100_MODEL="$model" \
DS4_V100_APPLIANCE_DIR="$appliance_dir" \
DS4_V100_CTX="$ctx" \
DS4_V100_SLOTS="$slots" \
DS4_V100_ACTIVE_MICROBATCH="$slots" \
DS4_V100_TOKENS="$tokens" \
DS4_V100_ASYNC_PIPELINE_MODE=per-step \
DS4_V100_ASYNC_EVENT_HANDOFF=1 \
DS4_V100_STARTUP_WARMUP=0 \
DS4_V100_MTP_SERVING=off \
"$launcher_bin" --check >"$negative_dir/stdout.log" 2>"$negative_dir/stderr.log"
negative_rc=$?
set -e
printf '%s\n' "$negative_rc" >"$negative_dir/exit_code.txt"
if [ "$negative_rc" -eq 0 ]; then
    fail "cold negative check unexpectedly passed"
fi
if ! grep -q "exceeds ctx=262144 admission cap 16" "$negative_dir/stderr.log"; then
    cat "$negative_dir/stderr.log" >&2
    fail "cold negative check failed for an unexpected reason"
fi

soak_dir="$log_dir/warmed-soak"
cmd=(
    "$soak_bin"
    --model "$model"
    --appliance-dir "$appliance_dir"
    --ctx "$ctx"
    --slots "$slots"
    --active-microbatch "$slots"
    --tokens "$tokens"
    --requests "$requests"
    --warmup-requests "$warmup_requests"
    --port "$port"
    --microbatch-wait-us 200000
    --async-pipeline-mode per-step
    --async-event-handoff 1
    --log-dir "$soak_dir"
)
printf '%q ' "${cmd[@]}" >"$log_dir/warmed-soak-command.txt"
printf '\n' >>"$log_dir/warmed-soak-command.txt"
DS4_V100_STARTUP_WARMUP=1 "${cmd[@]}" >"$log_dir/warmed-soak.stdout.log" 2>"$log_dir/warmed-soak.stderr.log"

python3 - "$log_dir" "$soak_dir" "$ctx" "$slots" "$tokens" "$requests" <<'PY'
import csv
import json
import os
import sys

root, soak_dir, ctx, slots, tokens, requests = sys.argv[1:]
summary_path = os.path.join(soak_dir, "summary.json")
status_before_path = os.path.join(soak_dir, "status_before.json")
metrics_before_path = os.path.join(soak_dir, "metrics_before.txt")
gpu_path = os.path.join(soak_dir, "gpu_util.csv")

with open(summary_path, "r", encoding="utf-8") as f:
    soak = json.load(f)
with open(status_before_path, "r", encoding="utf-8") as f:
    status_before = json.load(f)
metrics_before = ""
if os.path.exists(metrics_before_path):
    with open(metrics_before_path, "r", encoding="utf-8", errors="replace") as f:
        metrics_before = f.read()

max_mem = 0.0
max_util = 0.0
avg_util_vals = []
if os.path.exists(gpu_path):
    with open(gpu_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                util = float(row[2].strip())
                mem = float(row[3].strip())
            except Exception:
                continue
            avg_util_vals.append(util)
            max_util = max(max_util, util)
            max_mem = max(max_mem, mem)
avg_util = sum(avg_util_vals) / len(avg_util_vals) if avg_util_vals else 0.0

requests_i = int(requests)
token_match = int(soak.get("token_match", 0) or 0)
status_200 = int(soak.get("status_200", 0) or 0)
errors = int(soak.get("errors", 0) or 0)
warmed_ready = bool(status_before.get("warmed_ready"))
warmup_required = bool(status_before.get("warmup_required"))
metric_ready = "ds4_v100_warmed_ready 1" in metrics_before
metric_required = "ds4_v100_warmup_required 1" in metrics_before

decision = "pass"
if status_200 != requests_i or token_match != requests_i or errors != 0:
    decision = "fail_correctness"
elif not warmed_ready or not warmup_required or not metric_ready or not metric_required:
    decision = "fail_readiness_contract"

out = {
    "schema": "ds4_v100_256k_warmed_production_gate.v1",
    "decision": decision,
    "ctx": int(ctx),
    "slots": int(slots),
    "active_microbatch": int(slots),
    "tokens": int(tokens),
    "requests": requests_i,
    "status_200": status_200,
    "token_match": token_match,
    "errors": errors,
    "generated_tokens_per_second": float(soak.get("aggregate_generated_tokens_per_second", 0.0) or 0.0),
    "prompt_tokens_per_second": float(soak.get("aggregate_prompt_tokens_per_second", 0.0) or 0.0),
    "continuation_tokens_per_second": float(soak.get("aggregate_continuation_tokens_per_second", 0.0) or 0.0),
    "latency_ms_avg": float(soak.get("latency_ms_avg", 0.0) or 0.0),
    "prefill_prompt_replay_ms_avg": float(soak.get("prefill_prompt_replay_ms_avg", 0.0) or 0.0),
    "continuation_decode_ms_avg": float(soak.get("continuation_decode_ms_avg", 0.0) or 0.0),
    "warmed_ready": warmed_ready,
    "warmup_required": warmup_required,
    "metric_warmed_ready": metric_ready,
    "metric_warmup_required": metric_required,
    "max_gpu_util_percent": max_util,
    "avg_gpu_util_percent": avg_util,
    "max_memory_used_mib": max_mem,
}

json_path = os.path.join(root, "warmed_production_gate_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
    f.write("\n")

md_path = os.path.join(root, "warmed_production_gate_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# DS4 V100 256K Warmed Production Gate\n\n")
    f.write(f"Decision: `{decision}`\n\n")
    f.write("| Ctx | Slots | Requests | Generated tok/s | Prompt tok/s | Continuation tok/s | Match | Avg GPU util | Max GPU util | Max memory MiB |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(
        f"| {ctx} | {slots} | {requests} | "
        f"{out['generated_tokens_per_second']:.6f} | "
        f"{out['prompt_tokens_per_second']:.6f} | "
        f"{out['continuation_tokens_per_second']:.6f} | "
        f"{token_match}/{requests_i} | "
        f"{avg_util:.3f}% | {max_util:.3f}% | {max_mem:.1f} |\n\n"
    )
    f.write(f"- `warmup_required`: `{str(warmup_required).lower()}`\n")
    f.write(f"- `warmed_ready`: `{str(warmed_ready).lower()}`\n")
    f.write(f"- metric `ds4_v100_warmup_required 1`: `{str(metric_required).lower()}`\n")
    f.write(f"- metric `ds4_v100_warmed_ready 1`: `{str(metric_ready).lower()}`\n")

print(json_path)
print(md_path)
PY

cat "$log_dir/warmed_production_gate_summary.md"
decision="$(python3 - "$log_dir/warmed_production_gate_summary.json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("decision", ""))
PY
)"
[ "$decision" = "pass" ] || exit 1
