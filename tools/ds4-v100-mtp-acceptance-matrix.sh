#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
appliance_dir=""
ctx="262144"
prompts="tests/test-vectors/prompts/short_reasoning_plain.txt"
block_sizes="2,4,8"
tokens="16"
log_dir=""
expected_hex=""
replay_bin="${DS4_V100_REPLAY_BIN:-./tools/ds4-v100-replay}"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-mtp-acceptance-matrix.sh --appliance-dir DIR [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          DeepSeek-V4 Flash MTP sidecar GGUF
  --appliance-dir DIR       appliance dir containing pack-index.tsv,
                            turbomind-pack-index.tsv, and gpuN.weights shards
  --ctx N                   context tokens, default 262144
  --prompts CSV             comma list of prompt files
  --block-sizes CSV         comma list of MTP draft block sizes, default 2,4,8
  --tokens N                replay token budget, default 16
  --expected-token-hex HEX  optional first-token byte check for every prompt
  --log-dir DIR             output directory; default logs/mtp-acceptance-matrix
  --help                    show this help

Environment:
  DS4_V100_REPLAY_BIN       replay executable, default ./tools/ds4-v100-replay

The harness runs one-slot --mtp-draft-block-smoke diagnostics and writes:
  mtp_acceptance_matrix.tsv
  mtp_acceptance_summary.md
  cases/<case>.stdout.log
  cases/<case>.stderr.log
USAGE
}

fail() {
    echo "ds4-v100-mtp-acceptance-matrix: $*" >&2
    exit 1
}

is_uint() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

require_file() {
    local label="$1"
    local path="$2"
    [ -f "$path" ] || fail "missing $label $path"
}

require_dir() {
    local label="$1"
    local path="$2"
    [ -d "$path" ] || fail "missing $label $path"
}

parse_csv_numbers() {
    local csv="$1"
    local label="$2"
    local item
    IFS=',' read -r -a _items <<<"$csv"
    [ "${#_items[@]}" -gt 0 ] || fail "empty $label"
    for item in "${_items[@]}"; do
        is_uint "$item" || fail "invalid numeric value '$item' in $label"
        [ "$item" -gt 0 ] || fail "$label values must be positive"
        [ "$item" -lt "$tokens" ] || fail "$label value $item must be less than --tokens $tokens"
    done
}

case_name() {
    local prompt="$1"
    local block="$2"
    local base
    base="$(basename "$prompt")"
    base="${base%.*}"
    printf '%s_block%s' "$base" "$block"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
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
        --appliance-dir)
            [ "$#" -ge 2 ] || fail "--appliance-dir requires a value"
            appliance_dir="$2"
            shift 2
            ;;
        --ctx)
            [ "$#" -ge 2 ] || fail "--ctx requires a value"
            ctx="$2"
            shift 2
            ;;
        --prompts)
            [ "$#" -ge 2 ] || fail "--prompts requires a value"
            prompts="$2"
            shift 2
            ;;
        --block-sizes)
            [ "$#" -ge 2 ] || fail "--block-sizes requires a value"
            block_sizes="$2"
            shift 2
            ;;
        --tokens)
            [ "$#" -ge 2 ] || fail "--tokens requires a value"
            tokens="$2"
            shift 2
            ;;
        --expected-token-hex)
            [ "$#" -ge 2 ] || fail "--expected-token-hex requires a value"
            expected_hex="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || fail "--log-dir requires a value"
            log_dir="$2"
            shift 2
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

[ -n "$appliance_dir" ] || fail "--appliance-dir is required"
[ -n "$log_dir" ] || log_dir="logs/mtp-acceptance-matrix"
is_uint "$ctx" || fail "--ctx must be a positive integer"
[ "$ctx" -gt 0 ] || fail "--ctx must be positive"
is_uint "$tokens" || fail "--tokens must be a positive integer"
[ "$tokens" -gt 1 ] || fail "--tokens must be greater than 1"
parse_csv_numbers "$block_sizes" "--block-sizes"

require_file "replay executable" "$replay_bin"
require_file "model" "$model"
require_file "MTP model" "$mtp_model"
require_dir "appliance dir" "$appliance_dir"
require_file "pack index" "$appliance_dir/pack-index.tsv"
require_file "TurboMind pack index" "$appliance_dir/turbomind-pack-index.tsv"

IFS=',' read -r -a prompt_items <<<"$prompts"
[ "${#prompt_items[@]}" -gt 0 ] || fail "empty --prompts"
for prompt in "${prompt_items[@]}"; do
    require_file "prompt" "$prompt"
done

mkdir -p "$log_dir/cases"
tsv="$log_dir/mtp_acceptance_matrix.tsv"
summary="$log_dir/mtp_acceptance_summary.md"

printf 'case\tprompt\tblock_tokens\tstatus\tprompt_tokens\tfirst_token\tfirst_hex\tdraft_tokens\ttarget_tokens\taccepted_prefix_len\ttarget_forwards\ttarget_tokens_verified\teffective_output_tokens\tspeculative_saves\tsnapshot_bytes\tmtp_ms\tverify_ms\tmtp_raw_row\tmtp_n_raw\n' >"$tsv"

export DS4_V100_CUDA_TENSOR_POOL="${DS4_V100_CUDA_TENSOR_POOL:-auto}"
export DS4_V100_CUDA_TENSOR_POOL_MAX_MIB="${DS4_V100_CUDA_TENSOR_POOL_MAX_MIB:-2048}"
export DS4_V100_CUDA_F8_ROWPAIR="${DS4_V100_CUDA_F8_ROWPAIR:-1}"
export DS4_V100_CUDA_F8_ROW4="${DS4_V100_CUDA_F8_ROW4:-0}"
export DS4_V100_CUDA_F8_WARP_SCALE="${DS4_V100_CUDA_F8_WARP_SCALE:-0}"
export DS4_V100_CUDA_F8_GROUPED_DS4_FAST="${DS4_V100_CUDA_F8_GROUPED_DS4_FAST:-1}"
export DS4_V100_CUDA_F8_HMMA_SHARED_DOWN="${DS4_V100_CUDA_F8_HMMA_SHARED_DOWN:-0}"
export DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU="${DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU:-1}"
export DS4_V100_CUDA_F8_HMMA_ATTN_BATCH="${DS4_V100_CUDA_F8_HMMA_ATTN_BATCH:-1}"
export DS4_V100_ENABLE_BATCH_ATTN_PROJ="${DS4_V100_ENABLE_BATCH_ATTN_PROJ:-1}"
export DS4_V100_BATCH_SHARED_F8="${DS4_V100_BATCH_SHARED_F8:-1}"
export DS4_V100_SINGLE_SLOT_ATTN_SCRATCH="${DS4_V100_SINGLE_SLOT_ATTN_SCRATCH:-1}"
export DS4_V100_TURBOMIND_FUSED_GATE_UP="${DS4_V100_TURBOMIND_FUSED_GATE_UP:-1}"
export DS4_V100_TURBOMIND_GATED_SILU="${DS4_V100_TURBOMIND_GATED_SILU:-1}"
export DS4_V100_TURBOMIND_COMPACT_SCHEDULE="${DS4_V100_TURBOMIND_COMPACT_SCHEDULE:-1}"
export DS4_V100_TURBOMIND_ROUTED_EXECUTOR="${DS4_V100_TURBOMIND_ROUTED_EXECUTOR:-fused6_reduce}"
export DS4_V100_TURBOMIND_GATE_UP_PROBE="${DS4_V100_TURBOMIND_GATE_UP_PROBE:-auto}"
export DS4_V100_TURBOMIND_GRAPH="${DS4_V100_TURBOMIND_GRAPH:-1}"
export DS4_V100_TURBOMIND_LIB="${DS4_V100_TURBOMIND_LIB:-./build/turbomind-v100/libggml-turbomind.so}"

failures=0
total_cases=0
for prompt in "${prompt_items[@]}"; do
    IFS=',' read -r -a block_items <<<"$block_sizes"
    for block in "${block_items[@]}"; do
        total_cases=$((total_cases + 1))
        name="$(case_name "$prompt" "$block")"
        stdout_log="$log_dir/cases/$name.stdout.log"
        stderr_log="$log_dir/cases/$name.stderr.log"
        cmd=(
            "$replay_bin"
            --model "$model"
            --mtp-model "$mtp_model"
            --appliance-dir "$appliance_dir"
            --prompt-file "$prompt"
            --ctx "$ctx"
            --slots 1
            --active-microbatch 1
            --tokens "$tokens"
            --mtp-draft-block-smoke "$block"
            --json
        )
        if [ -n "$expected_hex" ]; then
            cmd+=(--expected-token-hex "$expected_hex")
        fi
        echo "case $name"
        if "${cmd[@]}" >"$stdout_log" 2>"$stderr_log"; then
            python3 - "$stdout_log" "$tsv" "$name" "$prompt" "$block" <<'PY'
import json
import sys

stdout_path, tsv_path, case_name, prompt, block = sys.argv[1:6]
payload = None
with open(stdout_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            payload = json.loads(line)
if payload is None:
    raise SystemExit("no JSON payload found")

def arr(name):
    return ",".join(str(x) for x in payload.get(name, []))

fields = [
    case_name,
    prompt,
    block,
    "ok",
    payload.get("prompt_tokens", ""),
    payload.get("first_token", ""),
    payload.get("first_hex", ""),
    arr("draft_tokens"),
    arr("target_tokens"),
    payload.get("accepted_prefix_len", ""),
    payload.get("target_forwards", ""),
    payload.get("target_tokens_verified", ""),
    payload.get("effective_output_tokens", ""),
    payload.get("speculative_saves", ""),
    payload.get("snapshot_bytes", ""),
    payload.get("mtp_ms", ""),
    payload.get("verify_ms", ""),
    payload.get("mtp_raw_row", ""),
    payload.get("mtp_n_raw", ""),
]
with open(tsv_path, "a", encoding="utf-8") as out:
    out.write("\t".join(str(x) for x in fields) + "\n")
PY
        else
            failures=$((failures + 1))
            printf '%s\t%s\t%s\tfail\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n' \
                "$name" "$prompt" "$block" >>"$tsv"
        fi
    done
done

python3 - "$tsv" "$summary" "$total_cases" "$failures" <<'PY'
import csv
import sys
from collections import Counter

tsv_path, summary_path, total_cases, failures = sys.argv[1:5]
rows = []
with open(tsv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

ok_rows = [r for r in rows if r.get("status") == "ok"]
accepted = [int(r["accepted_prefix_len"]) for r in ok_rows if r.get("accepted_prefix_len")]
spec_saves = [int(r["speculative_saves"]) for r in ok_rows if r.get("speculative_saves")]
block_counts = Counter()
block_ge2 = Counter()
for r in ok_rows:
    block = r.get("block_tokens", "")
    block_counts[block] += 1
    try:
        if int(r.get("accepted_prefix_len") or 0) >= 2:
            block_ge2[block] += 1
    except ValueError:
        pass

total_ok = len(ok_rows)
avg_accept = (sum(accepted) / len(accepted)) if accepted else 0.0
max_accept = max(accepted) if accepted else 0
total_saves = sum(spec_saves) if spec_saves else 0
ge2 = sum(1 for v in accepted if v >= 2)
decision = "continue-mtp-evaluation" if total_ok and ge2 >= max(1, total_ok // 3) else "pivot-away-from-mtp-throughput"

with open(summary_path, "w", encoding="utf-8") as out:
    out.write("# MTP Acceptance Matrix Summary\n\n")
    out.write(f"- cases: {total_cases}\n")
    out.write(f"- ok_cases: {total_ok}\n")
    out.write(f"- failed_cases: {failures}\n")
    out.write(f"- average_accepted_prefix: {avg_accept:.3f}\n")
    out.write(f"- max_accepted_prefix: {max_accept}\n")
    out.write(f"- cases_with_accepted_prefix_ge_2: {ge2}\n")
    out.write(f"- total_speculative_saves: {total_saves}\n")
    out.write(f"- decision: {decision}\n\n")
    out.write("| Block | Cases | Accepted Prefix >= 2 |\n")
    out.write("|---:|---:|---:|\n")
    for block in sorted(block_counts, key=lambda x: int(x) if x.isdigit() else x):
        out.write(f"| {block} | {block_counts[block]} | {block_ge2[block]} |\n")
    out.write("\n## Cases\n\n")
    out.write("| Case | Prompt | Block | Accepted | Effective | Target Forwards | Spec Saves |\n")
    out.write("|---|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        out.write(
            f"| {r.get('case','')} | {r.get('prompt','')} | {r.get('block_tokens','')} | "
            f"{r.get('accepted_prefix_len','')} | {r.get('effective_output_tokens','')} | "
            f"{r.get('target_forwards','')} | {r.get('speculative_saves','')} |\n"
        )
PY

cat "$summary"
[ "$failures" -eq 0 ] || fail "$failures case(s) failed; see $log_dir/cases"
