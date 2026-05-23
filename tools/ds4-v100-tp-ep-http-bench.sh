#!/usr/bin/env bash
set -euo pipefail

log_dir=""
tokens_cases="32,64"
generation_requests="1"
port_base="18100"
slots="32"
ctx="262144"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
contract="/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv"
tm_index=""
tp_ep_bin="./tools/ds4-v100-tp-ep-full-layer-smoke"
turbomind_lib="/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so"
run_appliance="./tools/ds4-v100-run-appliance.sh"
copy_event_compose="1"
ep_return_fp16="0"
compact_route_compose="1"
diagnostic_output_head="0"
concurrent_requests="1"
request_token_pattern=""
endpoint="selected-token"
hc_final_expand="0"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-tp-ep-http-bench.sh --log-dir DIR [options]

Starts the TP/EP appliance launcher once per token case, drives the HTTP
surface with Python stdlib, and writes a sustained HTTP matrix.

Options:
  --log-dir DIR       output directory; required
  --tokens-cases CSV  generated tokens per request, default 32,64
  --requests N        generation requests per resident server, default 1
  --port-base N       first port to use, default 18100
  --slots N           active slots, default 32
  --ctx N             context, default 262144
  --appliance-dir DIR production appliance pack
  --contract FILE     TP/EP pack contract TSV
  --tm-index FILE     TurboMind pack index; default appliance dir index
  --tp-ep-bin FILE    TP/EP HTTP server binary
  --turbomind-lib FILE
  --run-appliance FILE
  --copy-event-compose
  --no-copy-event-compose
  --ep-return-fp16
  --compact-route-compose
  --no-compact-route-compose
  --hc-final-expand
  --diagnostic-output-head
  --concurrent-requests
  --sequential-requests
  --request-token-pattern CSV
  --endpoint selected-token|completions
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
        --requests) generation_requests="$2"; shift 2 ;;
        --port-base) port_base="$2"; shift 2 ;;
        --slots) slots="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --contract) contract="$2"; shift 2 ;;
        --tm-index) tm_index="$2"; shift 2 ;;
        --tp-ep-bin) tp_ep_bin="$2"; shift 2 ;;
        --turbomind-lib) turbomind_lib="$2"; shift 2 ;;
        --run-appliance) run_appliance="$2"; shift 2 ;;
        --copy-event-compose) copy_event_compose="1"; shift ;;
        --no-copy-event-compose) copy_event_compose="0"; shift ;;
        --ep-return-fp16) ep_return_fp16="1"; shift ;;
        --compact-route-compose) compact_route_compose="1"; shift ;;
        --no-compact-route-compose) compact_route_compose="0"; shift ;;
        --hc-final-expand) hc_final_expand="1"; shift ;;
        --diagnostic-output-head) diagnostic_output_head="1"; shift ;;
        --concurrent-requests) concurrent_requests="1"; shift ;;
        --sequential-requests) concurrent_requests="0"; shift ;;
        --request-token-pattern) request_token_pattern="$2"; shift 2 ;;
        --endpoint) endpoint="$2"; shift 2 ;;
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
case "$generation_requests" in
    *[!0-9]* | "") fail "--requests must be numeric" ;;
esac
[ "$generation_requests" -ge 1 ] && [ "$generation_requests" -le 128 ] || fail "--requests must be in [1,128]"
if [ -n "$request_token_pattern" ]; then
    case "$request_token_pattern" in
        *[!0-9,]* | *::* | :* | *:) fail "--request-token-pattern must be numeric CSV" ;;
    esac
fi
case "$endpoint" in
    selected-token|completions) ;;
    *) fail "--endpoint must be selected-token or completions" ;;
esac

mkdir -p "$log_dir/cases"
summary_tsv="$log_dir/sustained_http.tsv"
summary_json="$log_dir/sustained_http.json"
printf 'schema\tds4_v100_tp_ep_sustained_http.v4\n' >"$summary_tsv"
printf 'backend\ttp_ep_launcher_http\n\n' >>"$summary_tsv"
printf 'endpoint\ttokens\trequest_token_pattern\tctx\tslots\tgeneration_requests\tcoalesced_batches\tcoalesced_batch_max\tstatus_200\tgenerated_tokens\tcontinuation_tokens\telapsed_s\tgenerated_tok_s\tcontinuation_tok_s\tgenerated_tok_s_decode\tcontinuation_tok_s_decode\tep_ms\tdense_ms\tcompose_ms\tcompose_reduce_ms\tcompose_copy_ms\tcompose_final_ms\tgpu_util_avg\tgpu_util_max\tgpu_mem_used_max_mib\n' >>"$summary_tsv"

case_jsons=()
case_index=0
IFS=',' read -r -a token_values <<<"$tokens_cases"
for tokens in "${token_values[@]}"; do
    [ -n "$tokens" ] || continue
    case "$tokens" in *[!0-9]*) fail "bad token case: $tokens" ;; esac
    [ "$tokens" -ge 1 ] && [ "$tokens" -le 64 ] || fail "token case must be in [1,64]: $tokens"
    case_suffix="tok${tokens}"
    if [ -n "$request_token_pattern" ]; then
        case_suffix="${case_suffix}_pat${request_token_pattern//,/x}"
    fi
    case_suffix="${case_suffix}_${endpoint}"
    port=$((port_base + case_index))
    case_dir="$log_dir/cases/case_${case_index}_ctx${ctx}_s${slots}_${case_suffix}"
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
    DS4_V100_MAX_REQUESTS=$((generation_requests + 4)) \
    DS4_V100_TP_EP_COPY_EVENT_COMPOSE="$copy_event_compose" \
    DS4_V100_TP_EP_RETURN_FP16="$ep_return_fp16" \
    DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE="$compact_route_compose" \
    DS4_V100_TP_EP_HC_FINAL_EXPAND="$hc_final_expand" \
    DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD="$diagnostic_output_head" \
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

    python3 - "$case_dir" "$port" "$tokens" "$generation_requests" "$concurrent_requests" "$request_token_pattern" "$endpoint" <<'PY'
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

case_dir, port, tokens, generation_requests, concurrent_requests, token_pattern, endpoint = sys.argv[1:]
tokens = int(tokens)
generation_requests = int(generation_requests)
concurrent_requests = int(concurrent_requests)
if token_pattern:
    pattern = [int(x) for x in token_pattern.split(",") if x]
else:
    pattern = [tokens]
request_tokens = [pattern[i % len(pattern)] for i in range(generation_requests)]
base = f"http://127.0.0.1:{port}"
post_path = "/v1/completions" if endpoint == "completions" else "/v100/selected-token"

def fetch(name, path, data=None, suffix="json"):
    req = urllib.request.Request(
        base + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    with open(f"{case_dir}/{name}.{suffix}", "wb") as f:
        f.write(body)
    return body

fetch("health", "/health")
fetch("status_before", "/v100/status")

stop = threading.Event()
gpu_csv = f"{case_dir}/gpu_util.csv"
def sample_gpu():
    if not shutil.which("nvidia-smi"):
        with open(gpu_csv, "w", encoding="utf-8") as f:
            f.write("timestamp,index,utilization.gpu,memory.used,memory.total\n")
        return
    with open(gpu_csv, "w", encoding="utf-8") as f:
        f.write("timestamp,index,utilization.gpu,memory.used,memory.total\n")
        while not stop.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ], text=True, stderr=subprocess.DEVNULL)
                f.write(out)
                f.flush()
            except Exception as exc:
                f.write(f"sample_error,-1,0,0,0 # {exc}\n")
                f.flush()
            stop.wait(0.2)

thread = threading.Thread(target=sample_gpu, daemon=True)
thread.start()
responses = []
errors = []
try:
    if concurrent_requests:
        responses = [None] * generation_requests
        def post_one(i):
            try:
                if endpoint == "completions":
                    payload = {
                        "model": "ds4-v100-tp-ep-diagnostic",
                        "prompt": f"diagnostic request {i}",
                        "max_tokens": request_tokens[i],
                        "request_index": i,
                    }
                else:
                    payload = {"max_tokens": request_tokens[i], "request_index": i}
                body = fetch(
                    f"response_{i:03d}",
                    post_path,
                    json.dumps(payload).encode(),
                )
                responses[i] = json.loads(body.decode("utf-8"))
            except Exception as exc:
                errors.append(f"request {i}: {exc}")
        workers = [threading.Thread(target=post_one, args=(i,)) for i in range(generation_requests)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        responses = [r for r in responses if r is not None]
    else:
        for i in range(generation_requests):
            if endpoint == "completions":
                payload = {
                    "model": "ds4-v100-tp-ep-diagnostic",
                    "prompt": f"diagnostic request {i}",
                    "max_tokens": request_tokens[i],
                    "request_index": i,
                }
            else:
                payload = {"max_tokens": request_tokens[i], "request_index": i}
            body = fetch(
                f"response_{i:03d}",
                post_path,
                json.dumps(payload).encode(),
            )
            responses.append(json.loads(body.decode("utf-8")))
    if errors:
        raise RuntimeError("; ".join(errors))
finally:
    stop.set()
    thread.join(timeout=2.0)

if responses:
    with open(f"{case_dir}/response.json", "w", encoding="utf-8") as f:
        json.dump(responses[-1], f, sort_keys=True)
        f.write("\n")
with open(f"{case_dir}/responses.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, sort_keys=True)
    f.write("\n")
fetch("status_after", "/v100/status")
fetch("metrics", "/metrics", None, "txt")
PY
    wait "$server_pid"

    python3 - "$case_dir" "$tokens" "$request_token_pattern" "$ctx" "$slots" "$generation_requests" "$summary_tsv" "$endpoint" <<'PY'
import json
import sys

case_dir, tokens, request_token_pattern, ctx, slots, generation_requests, summary_tsv, endpoint = sys.argv[1:]
with open(f"{case_dir}/responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)
if len(responses) != int(generation_requests):
    raise SystemExit(f"expected {generation_requests} responses, found {len(responses)}")
metas = [r.get("ds4_v100", r) for r in responses]
total_generated = sum(r["generated_tokens"] for r in metas)
total_continuation = sum(r["continuation_tokens"] for r in metas)
batch_rows = {}
for i, r in enumerate(metas):
    batch_rows.setdefault(r.get("coalesced_batch_id", i + 1), r)
unique_batches = list(batch_rows.values())
total_wall_ms = sum(r["timing_ms"]["total_wall"] for r in unique_batches)
total_decode_ms = sum(r["timing_ms"]["total_decode"] for r in unique_batches)
total_cont_wall_ms = sum(r["timing_ms"]["continuation_wall"] for r in unique_batches)
total_cont_decode_ms = sum(r["timing_ms"]["continuation_decode"] for r in unique_batches)
total_ep_ms = sum(r["timing_ms"].get("ep", 0.0) for r in unique_batches)
total_dense_ms = sum(r["timing_ms"].get("dense", 0.0) for r in unique_batches)
total_compose_ms = sum(r["timing_ms"].get("compose", 0.0) for r in unique_batches)
total_compose_reduce_ms = sum(r["timing_ms"].get("compose_reduce", 0.0) for r in unique_batches)
total_compose_copy_ms = sum(r["timing_ms"].get("compose_copy", 0.0) for r in unique_batches)
total_compose_final_ms = sum(r["timing_ms"].get("compose_final", 0.0) for r in unique_batches)
coalesced_batches = len(unique_batches)
coalesced_batch_max = max((int(r.get("coalesced_batch_size", 1)) for r in metas), default=0)
token_match = sum(r["token_match"] for r in metas)
token_mismatch = sum(r["token_mismatch"] for r in metas)
gpu_utils = []
gpu_mem_used = []
try:
    with open(f"{case_dir}/gpu_util.csv", "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = int(parts[1])
                util = float(parts[2])
                mem = float(parts[3])
            except ValueError:
                continue
            if idx >= 0:
                gpu_utils.append(util)
                gpu_mem_used.append(mem)
except FileNotFoundError:
    pass
gpu_util_avg = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
gpu_util_max = max(gpu_utils) if gpu_utils else 0.0
gpu_mem_used_max = max(gpu_mem_used) if gpu_mem_used else 0.0
row = {
    "schema": "ds4_v100_tp_ep_sustained_http_case.v4",
    "backend": "tp_ep_launcher_http",
    "endpoint": endpoint,
    "tokens_per_request": int(tokens),
    "request_token_pattern": request_token_pattern or str(tokens),
    "ctx": int(ctx),
    "slots": int(slots),
    "generation_requests": int(generation_requests),
    "coalesced_batches": coalesced_batches,
    "coalesced_batch_max": coalesced_batch_max,
    "status_200": token_match,
    "generated_tokens": total_generated,
    "continuation_tokens": total_continuation,
    "elapsed_s": total_wall_ms / 1000.0,
    "generated_tok_s": (total_generated * 1000.0 / total_wall_ms) if total_wall_ms > 0 else 0.0,
    "continuation_tok_s": (total_continuation * 1000.0 / total_cont_wall_ms) if total_cont_wall_ms > 0 else 0.0,
    "generated_tok_s_decode": (total_generated * 1000.0 / total_decode_ms) if total_decode_ms > 0 else 0.0,
    "continuation_tok_s_decode": (total_continuation * 1000.0 / total_cont_decode_ms) if total_cont_decode_ms > 0 else 0.0,
    "ep_ms": total_ep_ms,
    "dense_ms": total_dense_ms,
    "compose_ms": total_compose_ms,
    "compose_reduce_ms": total_compose_reduce_ms,
    "compose_copy_ms": total_compose_copy_ms,
    "compose_final_ms": total_compose_final_ms,
    "token_match": token_match,
    "token_mismatch": token_mismatch,
    "gpu_util_avg": gpu_util_avg,
    "gpu_util_max": gpu_util_max,
    "gpu_mem_used_max_mib": gpu_mem_used_max,
}
with open(f"{case_dir}/result.json", "w", encoding="utf-8") as f:
    json.dump(row, f, sort_keys=True)
    f.write("\n")
with open(summary_tsv, "a", encoding="utf-8") as f:
    f.write(
        f"{row['endpoint']}\t"
        f"{row['tokens_per_request']}\t{row['request_token_pattern']}\t"
        f"{row['ctx']}\t{row['slots']}\t"
        f"{row['generation_requests']}\t{row['coalesced_batches']}\t"
        f"{row['coalesced_batch_max']}\t{row['status_200']}\t"
        f"{row['generated_tokens']}\t{row['continuation_tokens']}\t"
        f"{row['elapsed_s']:.6f}\t{row['generated_tok_s']:.6f}\t"
        f"{row['continuation_tok_s']:.6f}\t{row['generated_tok_s_decode']:.6f}\t"
        f"{row['continuation_tok_s_decode']:.6f}\t"
        f"{row['ep_ms']:.6f}\t{row['dense_ms']:.6f}\t"
        f"{row['compose_ms']:.6f}\t{row['compose_reduce_ms']:.6f}\t"
        f"{row['compose_copy_ms']:.6f}\t{row['compose_final_ms']:.6f}\t"
        f"{row['gpu_util_avg']:.6f}\t{row['gpu_util_max']:.6f}\t"
        f"{row['gpu_mem_used_max_mib']:.6f}\n"
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
    json.dump({"schema": "ds4_v100_tp_ep_sustained_http.v4", "cases": cases}, f, sort_keys=True)
    f.write("\n")
PY

echo "ds4-v100-tp-ep-http-bench: PASS report=$summary_tsv json=$summary_json"
