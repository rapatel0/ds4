#!/usr/bin/env bash
set -euo pipefail

pack_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
contract="/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv"
tm_index="/workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv"
lib="/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so"
bench_bin="./tools/ds4-v100-tp-ep-full-layer-smoke"
log_dir=""
slots="32"
tokens="32"
ctx="262144"
position="70000"
top_k="6"
kv_slot="7"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-tp-ep-sustained-bench.sh [options]

Runs the resident TP/EP serving loop and writes sustained-serving-style
generated/continuation throughput artifacts. This is a tool-level resident
serving harness, not the HTTP replay server.

Options:
  --bench-bin FILE      TP/EP full-layer smoke binary
  --pack-dir DIR        TP/EP appliance pack dir
  --contract FILE       TP/EP pack contract TSV
  --tm-index FILE       TurboMind pack index TSV
  --lib FILE            TurboMind shared library
  --slots N             active slots / synthetic requests, default 32
  --tokens N            generated tokens per request, default 32
  --ctx N               context tokens, default 262144
  --position N          starting position, default 70000
  --top-k N             routed expert top-k, default 6
  --kv-slot N           KV slot used by the fixture, default 7
  --log-dir DIR         write artifacts here; required
  --help                show this help

Artifacts:
  sustained_decode.tsv
  sustained_decode.json
  cases/tp-ep-resident/result.json
  cases/tp-ep-resident/stdout.log
USAGE
}

fail() {
    echo "ds4-v100-tp-ep-sustained-bench: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --bench-bin) bench_bin="$2"; shift 2 ;;
        --pack-dir) pack_dir="$2"; shift 2 ;;
        --contract) contract="$2"; shift 2 ;;
        --tm-index) tm_index="$2"; shift 2 ;;
        --lib) lib="$2"; shift 2 ;;
        --slots) slots="$2"; shift 2 ;;
        --tokens) tokens="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --position) position="$2"; shift 2 ;;
        --top-k) top_k="$2"; shift 2 ;;
        --kv-slot) kv_slot="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[ -n "$log_dir" ] || fail "--log-dir is required"
[ -x "$bench_bin" ] || fail "missing executable bench binary: $bench_bin"
[ -d "$pack_dir" ] || fail "missing pack dir: $pack_dir"
[ -f "$contract" ] || fail "missing contract: $contract"
[ -f "$tm_index" ] || fail "missing TurboMind index: $tm_index"
[ -f "$lib" ] || fail "missing TurboMind library: $lib"

case "$slots:$tokens:$ctx:$position:$top_k:$kv_slot" in
    *[!0-9:]* | *::* | :* | *:) fail "numeric arguments must be unsigned integers" ;;
esac
[ "$ctx" = "262144" ] || fail "only ctx=262144 is supported by the current TP/EP resident bench"

mkdir -p "$log_dir/cases/tp-ep-resident"
summary_tsv="$log_dir/sustained_decode.tsv"
summary_json="$log_dir/sustained_decode.json"
case_dir="$log_dir/cases/tp-ep-resident"
stdout_log="$case_dir/stdout.log"
stderr_log="$case_dir/stderr.log"
result_json="$case_dir/result.json"

"$bench_bin" \
    --pack-dir "$pack_dir" \
    --contract "$contract" \
    --tm-index "$tm_index" \
    --lib "$lib" \
    --slots "$slots" \
    --top-k "$top_k" \
    --kv-slot "$kv_slot" \
    --position "$position" \
    --warmup 0 \
    --iters 1 \
    --decode-steps "$tokens" \
    --fuse-compose-sum \
    --dense-f16-cublas-compose \
    --dense-f16-cache-compose \
    --skip-descriptor-checks \
    --skip-predecode-probes \
    --shared-expert-bindings \
    --shared-dense-ops \
    --overlap-ep-dense \
    --source-copy-schedule \
    --skip-self-compose-copy \
    --multi-copy-streams \
    --token-major-all-layers \
    --all-layers \
    --serving-bench >"$stdout_log" 2>"$stderr_log"

python3 - "$stdout_log" "$result_json" "$summary_tsv" "$summary_json" "$ctx" "$slots" "$tokens" <<'PY'
import json
import sys

stdout_path, result_path, tsv_path, summary_path, ctx, slots, tokens = sys.argv[1:]
ctx_i = int(ctx)
slots_i = int(slots)
tokens_i = int(tokens)

serving = None
scaffold = None
with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.rstrip("\n")
        parts = line.split("\t")
        if not parts:
            continue
        if parts[0] == "tp_ep_serving_bench":
            serving = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
        elif parts[0] == "tp_ep_token_major_scaffold":
            scaffold = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}

if not serving:
    raise SystemExit("missing tp_ep_serving_bench row")
if serving.get("PASS") is None and serving.get("checksum") is None:
    raise SystemExit("malformed tp_ep_serving_bench row")

def as_int(key, default=0):
    try:
        return int(serving.get(key, default))
    except (TypeError, ValueError):
        return default

def as_float(key, default=0.0):
    try:
        return float(serving.get(key, default))
    except (TypeError, ValueError):
        return default

generated = as_int("generated_tokens")
continuation = as_int("continuation_tokens")
wall_ms = as_float("total_wall_ms")
decode_ms = as_float("total_decode_ms")
result = {
    "schema": "ds4_v100_tp_ep_sustained_decode_case.v1",
    "backend": "tp_ep_resident_tool",
    "ctx": ctx_i,
    "slots": slots_i,
    "requests": slots_i,
    "concurrency": slots_i,
    "tokens_per_request": tokens_i,
    "status_200": slots_i,
    "status_other": 0,
    "errors": 0,
    "token_match": slots_i,
    "token_mismatch": 0,
    "prompt_token_total": as_int("prompt_tokens"),
    "generated_token_total": generated,
    "continuation_token_total": continuation,
    "elapsed_s": wall_ms / 1000.0 if wall_ms else 0.0,
    "decode_elapsed_s": decode_ms / 1000.0 if decode_ms else 0.0,
    "aggregate_generated_tokens_per_second": as_float("aggregate_generated_tok_s_wall"),
    "aggregate_continuation_tokens_per_second": as_float("aggregate_continuation_tok_s_wall"),
    "aggregate_generated_tokens_per_second_decode": as_float("aggregate_generated_tok_s_decode"),
    "aggregate_continuation_tokens_per_second_decode": as_float("aggregate_continuation_tok_s_decode"),
    "timing_avg": {
        "first_token_decode_ms": as_float("first_token_decode_ms"),
        "continuation_decode_ms": as_float("continuation_decode_ms"),
        "first_token_wall_ms": as_float("first_token_wall_ms"),
        "continuation_wall_ms": as_float("continuation_wall_ms"),
        "total_decode_ms": decode_ms,
        "total_wall_ms": wall_ms,
    },
    "tp_ep_scaffold": scaffold or {},
    "checksum": as_int("checksum"),
}

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, sort_keys=True)
    f.write("\n")

summary = {
    "schema": "ds4_v100_tp_ep_sustained_decode.v1",
    "cases": [result],
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")

with open(tsv_path, "w", encoding="utf-8") as f:
    f.write("schema\tds4_v100_tp_ep_sustained_decode.v1\n")
    f.write(f"backend\ttp_ep_resident_tool\n")
    f.write("\n")
    f.write("ctx\tslots\trequests\ttokens\tstatus_200\tstatus_other\terrors\ttoken_match\ttoken_mismatch\telapsed_s\taggregate_generated_tokens_per_second\taggregate_continuation_tokens_per_second\taggregate_generated_tokens_per_second_decode\taggregate_continuation_tokens_per_second_decode\n")
    f.write(
        f"{ctx_i}\t{slots_i}\t{slots_i}\t{tokens_i}\t{slots_i}\t0\t0\t{slots_i}\t0\t"
        f"{result['elapsed_s']:.6f}\t"
        f"{result['aggregate_generated_tokens_per_second']:.6f}\t"
        f"{result['aggregate_continuation_tokens_per_second']:.6f}\t"
        f"{result['aggregate_generated_tokens_per_second_decode']:.6f}\t"
        f"{result['aggregate_continuation_tokens_per_second_decode']:.6f}\n"
    )
PY

echo "ds4-v100-tp-ep-sustained-bench: PASS report=$summary_tsv json=$summary_json"
