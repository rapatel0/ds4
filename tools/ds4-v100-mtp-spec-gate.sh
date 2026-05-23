#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
ctx="262144"
tokens="16"
requests="1"
warmup_requests="0"
port_base="19040"
log_dir="/workspace/logs/sprint216-mtp-spec-gate"
bench_bin="./tools/ds4-v100-sustained-decode-bench.sh"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-mtp-spec-gate.sh [options]

Runs a focused one-slot MTP speculative accounting gate:
  1. normal target generation with MTP off;
  2. current MTP commit mode;
  3. a report comparing target forwards, effective output tokens, and saves.

Options:
  --model FILE          base DS4 model, default /models/DSv4-Flash-256e-fixed.gguf
  --mtp-model FILE      MTP sidecar model
  --appliance-dir DIR   appliance pack dir
  --ctx N               context tokens, default 262144
  --tokens N            generated tokens, default 16
  --requests N          timed requests, default 1
  --warmup-requests N   warmup requests, default 0
  --port-base N         first port, default 19040
  --log-dir DIR         output dir, default /workspace/logs/sprint216-mtp-spec-gate
  --bench-bin FILE      sustained decode bench path
  --help                show this help
USAGE
}

fail() {
    echo "ds4-v100-mtp-spec-gate: $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --mtp-model) mtp_model="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --tokens) tokens="$2"; shift 2 ;;
        --requests) requests="$2"; shift 2 ;;
        --warmup-requests) warmup_requests="$2"; shift 2 ;;
        --port-base) port_base="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --bench-bin) bench_bin="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done

for n in "$ctx" "$tokens" "$requests" "$warmup_requests" "$port_base"; do
    case "$n" in ''|*[!0-9]*) fail "numeric option expected, got '$n'" ;; esac
done
[ "$tokens" -ge 2 ] || fail "--tokens must be >= 2"
[ -x "$bench_bin" ] || fail "missing executable bench script $bench_bin"
[ -f "$model" ] || fail "missing model $model"
[ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
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

run_case() {
    local name="$1"
    local mode="$2"
    local port="$3"
    local out_dir="$log_dir/$name"
    mkdir -p "$out_dir"
    local cmd=(
        "$bench_bin"
        --model "$model"
        --mtp-model "$mtp_model"
        --appliance-dir "$appliance_dir"
        --ctx-tiers "$ctx"
        --slot-tiers 1
        --tokens "$tokens"
        --requests "$requests"
        --warmup-requests "$warmup_requests"
        --port-base "$port"
        --microbatch-wait-us 200000
        --async-pipeline-mode per-step
        --async-event-handoff
        --mtp-serving "$mode"
        --log-dir "$out_dir"
    )
    printf '%q ' "${cmd[@]}" >"$out_dir/command.txt"
    printf '\n' >>"$out_dir/command.txt"
    "${cmd[@]}" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
}

run_case baseline-off off "$port_base"
run_case mtp-commit commit "$((port_base + 1))"

python3 - "$log_dir" <<'PY'
import json
import os
import sys

root = sys.argv[1]

def load_case(name):
    path = os.path.join(root, name, "sustained_decode.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if not cases:
        raise SystemExit(f"missing case in {path}")
    return cases[0]

base = load_case("baseline-off")
commit = load_case("mtp-commit")
mtp = commit.get("mtp", {}) if isinstance(commit.get("mtp", {}), dict) else {}

summary = {
    "schema": "ds4_v100_mtp_spec_gate.v1",
    "baseline": {
        "generated_tokens_per_second": base.get("aggregate_generated_tokens_per_second", 0.0),
        "continuation_tokens_per_second": base.get("aggregate_continuation_tokens_per_second", 0.0),
        "token_match": base.get("token_match", 0),
        "token_mismatch": base.get("token_mismatch", 0),
    },
    "mtp_commit": {
        "generated_tokens_per_second": commit.get("aggregate_generated_tokens_per_second", 0.0),
        "continuation_tokens_per_second": commit.get("aggregate_continuation_tokens_per_second", 0.0),
        "token_match": commit.get("token_match", 0),
        "token_mismatch": commit.get("token_mismatch", 0),
        "attempted": mtp.get("attempted", 0),
        "accepted": mtp.get("accepted", 0),
        "rejected": mtp.get("rejected", 0),
        "committed": mtp.get("committed", 0),
        "draft_tokens_proposed": mtp.get("draft_tokens_proposed", 0),
        "draft_tokens_accepted": mtp.get("draft_tokens_accepted", 0),
        "accepted_prefix_len_max": mtp.get("accepted_prefix_len_max", 0),
        "target_tokens_verified": mtp.get("target_tokens_verified", 0),
        "target_forwards": mtp.get("target_forwards", 0),
        "effective_output_tokens": mtp.get("effective_output_tokens", 0),
        "speculative_saves": mtp.get("speculative_saves", 0),
        "draft_ms_total": mtp.get("draft_ms_total", 0.0),
    },
}
summary["decision"] = (
    "pass_true_speculative_gate"
    if summary["mtp_commit"]["speculative_saves"] > 0 and summary["mtp_commit"]["token_mismatch"] == 0
    else "fail_serial_target_replay"
)

json_path = os.path.join(root, "mtp_spec_gate.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True, indent=2)
    f.write("\n")

md_path = os.path.join(root, "mtp_spec_gate.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# DS4 V100 MTP Speculative Gate\n\n")
    f.write(f"Decision: `{summary['decision']}`\n\n")
    f.write("| Mode | Generated tok/s | Continuation tok/s | Match | Target forwards | Effective output tokens | Spec saves |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    f.write(
        "| baseline off | "
        f"{summary['baseline']['generated_tokens_per_second']:.6f} | "
        f"{summary['baseline']['continuation_tokens_per_second']:.6f} | "
        f"{summary['baseline']['token_match']}/{summary['baseline']['token_match'] + summary['baseline']['token_mismatch']} | "
        "n/a | n/a | n/a |\n"
    )
    f.write(
        "| mtp commit | "
        f"{summary['mtp_commit']['generated_tokens_per_second']:.6f} | "
        f"{summary['mtp_commit']['continuation_tokens_per_second']:.6f} | "
        f"{summary['mtp_commit']['token_match']}/{summary['mtp_commit']['token_match'] + summary['mtp_commit']['token_mismatch']} | "
        f"{summary['mtp_commit']['target_forwards']} | "
        f"{summary['mtp_commit']['effective_output_tokens']} | "
        f"{summary['mtp_commit']['speculative_saves']} |\n"
    )
    f.write("\n")
    f.write(
        "Current commit mode is a real speedup candidate only when "
        "`speculative_saves > 0`; accepted or committed draft counts alone do not qualify.\n"
    )

print(json_path)
print(md_path)
PY

echo "ds4-v100-mtp-spec-gate: PASS log_dir=$log_dir"
