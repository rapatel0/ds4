#!/usr/bin/env bash
set -euo pipefail

log_dir=""
tokens_cases="32,64"
port_base="18100"
slots="32"
ctx="262144"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
contract="/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv"
tm_index=""
tp_ep_bin="./tools/ds4-v100-tp-ep-full-layer-smoke"
turbomind_lib="/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so"
run_appliance="./tools/ds4-v100-run-appliance.sh"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-tp-ep-http-bench.sh --log-dir DIR [options]

Starts the TP/EP appliance launcher once per token case, drives the HTTP
surface with Python stdlib, and writes a sustained HTTP matrix.

Options:
  --log-dir DIR       output directory; required
  --tokens-cases CSV  generated tokens per request, default 32,64
  --port-base N       first port to use, default 18100
  --slots N           active slots, default 32
  --ctx N             context, default 262144
  --appliance-dir DIR production appliance pack
  --contract FILE     TP/EP pack contract TSV
  --tm-index FILE     TurboMind pack index; default appliance dir index
  --tp-ep-bin FILE    TP/EP HTTP server binary
  --turbomind-lib FILE
  --run-appliance FILE
  --help
USAGE
}

fail() {
    echo "ds4-v100-tp-ep-http-bench: $*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log-dir) log_dir="$2"; shift 2 ;;
        --tokens-cases) tokens_cases="$2"; shift 2 ;;
        --port-base) port_base="$2"; shift 2 ;;
        --slots) slots="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --contract) contract="$2"; shift 2 ;;
        --tm-index) tm_index="$2"; shift 2 ;;
        --tp-ep-bin) tp_ep_bin="$2"; shift 2 ;;
        --turbomind-lib) turbomind_lib="$2"; shift 2 ;;
        --run-appliance) run_appliance="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[ -n "$log_dir" ] || fail "--log-dir is required"
[ "$ctx" = "262144" ] || fail "current TP/EP HTTP bench requires ctx=262144"
[ "$slots" = "32" ] || fail "current TP/EP HTTP bench requires slots=32"
[ -x "$run_appliance" ] || fail "missing launcher: $run_appliance"
[ -x "$tp_ep_bin" ] || fail "missing TP/EP binary: $tp_ep_bin"
[ -d "$appliance_dir" ] || fail "missing appliance dir: $appliance_dir"
[ -f "$contract" ] || fail "missing TP/EP contract: $contract"
[ -f "$turbomind_lib" ] || fail "missing TurboMind library: $turbomind_lib"
if [ -z "$tm_index" ]; then
    tm_index="$appliance_dir/turbomind-pack-index.tsv"
fi
[ -f "$tm_index" ] || fail "missing TurboMind index: $tm_index"

case "$tokens_cases:$port_base" in
    *[!0-9,:]* | *::* | :* | *:) fail "tokens cases and port base must be numeric" ;;
esac

mkdir -p "$log_dir/cases"
summary_tsv="$log_dir/sustained_http.tsv"
summary_json="$log_dir/sustained_http.json"
printf 'schema\tds4_v100_tp_ep_sustained_http.v1\n' >"$summary_tsv"
printf 'backend\ttp_ep_launcher_http\n\n' >>"$summary_tsv"
printf 'tokens\tctx\tslots\tstatus_200\tgenerated_tokens\tcontinuation_tokens\telapsed_s\tgenerated_tok_s\tcontinuation_tok_s\tgenerated_tok_s_decode\tcontinuation_tok_s_decode\n' >>"$summary_tsv"

case_jsons=()
case_index=0
IFS=',' read -r -a token_values <<<"$tokens_cases"
for tokens in "${token_values[@]}"; do
    [ -n "$tokens" ] || continue
    case "$tokens" in *[!0-9]*) fail "bad token case: $tokens" ;; esac
    [ "$tokens" -ge 1 ] && [ "$tokens" -le 64 ] || fail "token case must be in [1,64]: $tokens"
    port=$((port_base + case_index))
    case_dir="$log_dir/cases/case_${case_index}_ctx${ctx}_s${slots}_tok${tokens}"
    mkdir -p "$case_dir/runtime"
    server_log="$case_dir/server.log"
    server_err="$case_dir/server.err"

    DS4_V100_SERVE_MODE=tp-ep \
    DS4_V100_TP_EP_BIN="$tp_ep_bin" \
    DS4_V100_APPLIANCE_DIR="$appliance_dir" \
    DS4_V100_TP_EP_CONTRACT="$contract" \
    DS4_V100_TP_EP_TM_INDEX="$tm_index" \
    DS4_V100_TURBOMIND_LIB="$turbomind_lib" \
    DS4_V100_CTX="$ctx" \
    DS4_V100_SLOTS="$slots" \
    DS4_V100_ACTIVE_MICROBATCH="$slots" \
    DS4_V100_TOKENS="$tokens" \
    DS4_V100_HOST=127.0.0.1 \
    DS4_V100_PORT="$port" \
    DS4_V100_MAX_REQUESTS=4 \
    DS4_V100_LOG_DIR="$case_dir/runtime" \
    "$run_appliance" >"$server_log" 2>"$server_err" &
    server_pid=$!

    for _ in $(seq 1 180); do
        if grep -q "tp_ep_http_serving" "$server_log"; then break; fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            cat "$server_err" >&2 || true
            fail "server exited before listening for token case $tokens"
        fi
        sleep 1
    done
    grep -q "tp_ep_http_serving" "$server_log" || fail "server did not listen for token case $tokens"

    python3 - "$case_dir" "$port" "$tokens" <<'PY'
import json
import sys
import urllib.request

case_dir, port, tokens = sys.argv[1:]
base = f"http://127.0.0.1:{port}"
requests = [
    ("health", "/health", None, "json"),
    ("status_before", "/v100/status", None, "json"),
    ("response", "/v100/selected-token", json.dumps({"max_tokens": int(tokens)}).encode(), "json"),
    ("metrics", "/metrics", None, "txt"),
]
for name, path, data, suffix in requests:
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    with open(f"{case_dir}/{name}.{suffix}", "wb") as f:
        f.write(body)
PY
    wait "$server_pid"

    python3 - "$case_dir" "$tokens" "$ctx" "$slots" "$summary_tsv" <<'PY'
import json
import sys

case_dir, tokens, ctx, slots, summary_tsv = sys.argv[1:]
with open(f"{case_dir}/response.json", "r", encoding="utf-8") as f:
    result = json.load(f)
timing = result["timing_ms"]
row = {
    "schema": "ds4_v100_tp_ep_sustained_http_case.v1",
    "backend": "tp_ep_launcher_http",
    "tokens_per_request": int(tokens),
    "ctx": int(ctx),
    "slots": int(slots),
    "status_200": int(slots),
    "generated_tokens": result["generated_tokens"],
    "continuation_tokens": result["continuation_tokens"],
    "elapsed_s": timing["total_wall"] / 1000.0,
    "generated_tok_s": timing["generated_tokens_per_second"],
    "continuation_tok_s": timing["continuation_tokens_per_second"],
    "generated_tok_s_decode": timing["generated_tokens_per_second_decode"],
    "continuation_tok_s_decode": timing["continuation_tokens_per_second_decode"],
    "token_match": result["token_match"],
    "token_mismatch": result["token_mismatch"],
}
with open(f"{case_dir}/result.json", "w", encoding="utf-8") as f:
    json.dump(row, f, sort_keys=True)
    f.write("\n")
with open(summary_tsv, "a", encoding="utf-8") as f:
    f.write(
        f"{row['tokens_per_request']}\t{row['ctx']}\t{row['slots']}\t"
        f"{row['status_200']}\t{row['generated_tokens']}\t{row['continuation_tokens']}\t"
        f"{row['elapsed_s']:.6f}\t{row['generated_tok_s']:.6f}\t"
        f"{row['continuation_tok_s']:.6f}\t{row['generated_tok_s_decode']:.6f}\t"
        f"{row['continuation_tok_s_decode']:.6f}\n"
    )
PY
    case_jsons+=("$case_dir/result.json")
    case_index=$((case_index + 1))
done

python3 - "$summary_json" "${case_jsons[@]}" <<'PY'
import json
import sys

summary_path = sys.argv[1]
cases = []
for path in sys.argv[2:]:
    with open(path, "r", encoding="utf-8") as f:
        cases.append(json.load(f))
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({"schema": "ds4_v100_tp_ep_sustained_http.v1", "cases": cases}, f, sort_keys=True)
    f.write("\n")
PY

echo "ds4-v100-tp-ep-http-bench: PASS report=$summary_tsv json=$summary_json"
