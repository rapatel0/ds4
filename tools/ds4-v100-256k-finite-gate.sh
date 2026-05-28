#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
appliance_dir="/workspace/packs/ds4-appliance-full-tm-gated-s181"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
ctx="262144"
slots="18"
tokens="16"
requests="18"
host="127.0.0.1"
port="19220"
log_dir="/workspace/logs/sprint218-256k-finite-source"
launcher_bin="./tools/ds4-v100-run-pp-appliance.sh"
replay_bin="./tools/ds4-v100-replay"
sample_seconds="1"
layer_checks="${DS4_V100_DEBUG_HC_FINITE_LAYER_CHECKS:-1}"
pre_output_check="${DS4_V100_DEBUG_HC_FINITE_PRE_OUTPUT:-1}"
startup_warmup="${DS4_V100_STARTUP_WARMUP:-0}"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-256k-finite-gate.sh [options]

Runs a focused 256K active-batch finite diagnostic with
DS4_V100_DEBUG_HC_FINITE=1. The default shape is the nearest known failing
Sprint 217 case: ctx=262144, slots=18, active_microbatch=18.

Options:
  --model FILE             base DS4 model
  --mtp-model FILE         MTP model path, checked for launcher compatibility
  --appliance-dir DIR      appliance pack dir
  --prompt-file FILE       prompt file
  --ctx N                  context tokens, default 262144
  --slots N                slots and active microbatch, default 18
  --tokens N               generated tokens per request, default 16
  --requests N             concurrent requests, default slots
  --host ADDR              bind/probe address, default 127.0.0.1
  --port N                 server port, default 19220
  --log-dir DIR            output dir
  --launcher-bin FILE      appliance launcher path
  --replay-bin FILE        replay server executable
  --sample-seconds N       nvidia-smi sample interval, default 1
  --layer-checks 0|1       check HC after every layer, default 1
  --pre-output-check 0|1   check HC before output head, default 1
  --startup-warmup 0|1     run launcher startup warmup, default 0
  --help                   show this help
USAGE
}

fail() {
    echo "ds4-v100-256k-finite-gate: $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) model="$2"; shift 2 ;;
        --mtp-model) mtp_model="$2"; shift 2 ;;
        --appliance-dir) appliance_dir="$2"; shift 2 ;;
        --prompt-file) prompt_file="$2"; shift 2 ;;
        --ctx) ctx="$2"; shift 2 ;;
        --slots) slots="$2"; requests="$2"; shift 2 ;;
        --tokens) tokens="$2"; shift 2 ;;
        --requests) requests="$2"; shift 2 ;;
        --host) host="$2"; shift 2 ;;
        --port) port="$2"; shift 2 ;;
        --log-dir) log_dir="$2"; shift 2 ;;
        --launcher-bin) launcher_bin="$2"; shift 2 ;;
        --replay-bin) replay_bin="$2"; shift 2 ;;
        --sample-seconds) sample_seconds="$2"; shift 2 ;;
        --layer-checks) layer_checks="$2"; shift 2 ;;
        --pre-output-check) pre_output_check="$2"; shift 2 ;;
        --startup-warmup) startup_warmup="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; fail "unknown option: $1" ;;
    esac
done

for n in "$ctx" "$slots" "$tokens" "$requests" "$port" "$sample_seconds"; do
    case "$n" in ''|*[!0-9]*) fail "numeric option expected, got '$n'" ;; esac
done
case "$layer_checks" in 0|1) ;; *) fail "--layer-checks must be 0 or 1" ;; esac
case "$pre_output_check" in 0|1) ;; *) fail "--pre-output-check must be 0 or 1" ;; esac
case "$startup_warmup" in 0|1) ;; *) fail "--startup-warmup must be 0 or 1" ;; esac
[ "$slots" -ge 1 ] || fail "--slots must be >= 1"
[ "$requests" -ge 1 ] || fail "--requests must be >= 1"
[ "$tokens" -ge 1 ] || fail "--tokens must be >= 1"
[ -x "$launcher_bin" ] || fail "missing executable launcher $launcher_bin"
[ -x "$replay_bin" ] || fail "missing executable replay binary $replay_bin"
[ -f "$model" ] || fail "missing model $model"
[ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
[ -d "$appliance_dir" ] || fail "missing appliance dir $appliance_dir"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"

mkdir -p "$log_dir"

request_json="$log_dir/request.json"
server_log="$log_dir/server.log"
client_log="$log_dir/client.log"
responses_json="$log_dir/responses.json"
summary_json="$log_dir/finite_gate_summary.json"
summary_md="$log_dir/finite_gate_summary.md"
gpu_csv="$log_dir/gpu_util.csv"
gpu_err="$log_dir/gpu_util.err"
check_dir="$log_dir/launcher-check"
mkdir -p "$check_dir"

python3 - "$prompt_file" "$tokens" >"$request_json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    prompt = f.read()
print(json.dumps({"prompt": prompt, "tokens": int(sys.argv[2])}))
PY

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
export DS4_V100_DEBUG_HC_FINITE=1
export DS4_V100_DEBUG_HC_FINITE_LAYER_CHECKS="$layer_checks"
export DS4_V100_DEBUG_HC_FINITE_PRE_OUTPUT="$pre_output_check"
export DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP="$slots"
export DS4_V100_STARTUP_WARMUP="$startup_warmup"

set +e
DS4_V100_BIN="$replay_bin" \
DS4_V100_MODEL="$model" \
DS4_V100_MTP_MODEL="$mtp_model" \
DS4_V100_APPLIANCE_DIR="$appliance_dir" \
DS4_V100_CTX="$ctx" \
DS4_V100_SLOTS="$slots" \
DS4_V100_ACTIVE_MICROBATCH="$slots" \
DS4_V100_TOKENS="$tokens" \
DS4_V100_ASYNC_PIPELINE_MODE=per-step \
DS4_V100_ASYNC_EVENT_HANDOFF=1 \
DS4_V100_STARTUP_WARMUP="$startup_warmup" \
DS4_V100_MTP_SERVING=off \
"$launcher_bin" --check >"$check_dir/stdout.log" 2>"$check_dir/stderr.log"
check_rc=$?
set -e
printf '%s\n' "$check_rc" >"$check_dir/exit_code.txt"
[ "$check_rc" -eq 0 ] || fail "launcher check failed; see $check_dir"

server_pid=""
gpu_pid=""
cleanup() {
    if [ -n "$server_pid" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "$gpu_pid" ]; then
        kill "$gpu_pid" >/dev/null 2>&1 || true
        wait "$gpu_pid" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
    (
        echo "timestamp,index,utilization.gpu [%],memory.used [MiB],memory.free [MiB]"
        while true; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.free \
                --format=csv,noheader,nounits || true
            sleep "$sample_seconds"
        done
    ) >"$gpu_csv" 2>"$gpu_err" &
    gpu_pid="$!"
fi

(
    export DS4_V100_BIN="$replay_bin"
    export DS4_V100_MODEL="$model"
    export DS4_V100_MTP_MODEL="$mtp_model"
    export DS4_V100_APPLIANCE_DIR="$appliance_dir"
    export DS4_V100_CTX="$ctx"
    export DS4_V100_SLOTS="$slots"
    export DS4_V100_ACTIVE_MICROBATCH="$slots"
    export DS4_V100_MICROBATCH_WAIT_US=200000
    export DS4_V100_QUEUE_POLICY=sequential
    export DS4_V100_TOKENS="$tokens"
    export DS4_V100_ASYNC_PIPELINE_MODE=per-step
    export DS4_V100_ASYNC_EVENT_HANDOFF=1
    export DS4_V100_STARTUP_WARMUP="$startup_warmup"
    export DS4_V100_HOST="$host"
    export DS4_V100_PORT="$port"
    export DS4_V100_REQUIRE_GPUS=8
    export DS4_V100_RESERVE_MIB="${DS4_V100_RESERVE_MIB:-2048}"
    export DS4_V100_MAX_REQUESTS=$((requests + 8))
    export DS4_V100_LOG_DIR="$log_dir/runtime"
    export DS4_V100_MTP_SERVING=off
    exec "$launcher_bin"
) >"$server_log" 2>&1 &
server_pid="$!"

for _ in $(seq 1 420); do
    if grep -q "serving http://" "$server_log"; then
        break
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        cat "$server_log" >&2
        fail "appliance exited before listening"
    fi
    sleep 1
done
grep -q "serving http://" "$server_log" || { cat "$server_log" >&2; fail "appliance did not listen"; }

python3 - "$host" "$port" "$request_json" "$requests" "$log_dir" >"$client_log" 2>&1 <<'PY'
import http.client
import json
import os
import sys
import threading
import time

host = sys.argv[1]
port = int(sys.argv[2])
request_json = sys.argv[3]
requests = int(sys.argv[4])
log_dir = sys.argv[5]
with open(request_json, "rb") as f:
    payload = f.read()

headers = {
    "Content-Type": "application/json",
    "Content-Length": str(len(payload)),
    "Connection": "close",
}
rows = []
lock = threading.Lock()

def send_one(i):
    t0 = time.perf_counter()
    status = -1
    body = b""
    err = ""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=900)
        conn.request("POST", "/v100/selected-token", body=payload, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        body = resp.read()
        conn.close()
    except Exception as exc:
        err = repr(exc)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    body_text = body.decode("utf-8", errors="replace")
    with open(os.path.join(log_dir, f"response_{i:03d}.body"), "w", encoding="utf-8") as f:
        f.write(body_text)
    row = {"index": i, "status": status, "elapsed_ms": elapsed_ms, "error": err, "body": body_text}
    with lock:
        rows.append(row)

threads = []
for i in range(requests):
    t = threading.Thread(target=send_one, args=(i,), daemon=True)
    t.start()
    threads.append(t)
for t in threads:
    t.join()

rows.sort(key=lambda r: r["index"])
with open(os.path.join(log_dir, "responses.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps({"requests": requests, "status_200": sum(1 for r in rows if r["status"] == 200), "status_other": sum(1 for r in rows if r["status"] != 200)}, sort_keys=True))
PY

if kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
    server_pid=""
fi

python3 - "$responses_json" "$server_log" "$gpu_csv" "$summary_json" "$summary_md" "$ctx" "$slots" "$tokens" "$requests" <<'PY'
import csv
import json
import os
import re
import sys

responses_path, server_log, gpu_csv, summary_json, summary_md, ctx, slots, tokens, requests = sys.argv[1:]
with open(responses_path, "r", encoding="utf-8") as f:
    responses = json.load(f)
status_200 = sum(1 for r in responses if int(r.get("status", -1)) == 200)
status_other = len(responses) - status_200
first_error_body = ""
for r in responses:
    if int(r.get("status", -1)) != 200:
        first_error_body = str(r.get("body", ""))
        break

server_text = ""
if os.path.exists(server_log):
    with open(server_log, "r", encoding="utf-8", errors="replace") as f:
        server_text = f.read()
first_hc = ""
for line in server_text.splitlines():
    if "HC non-finite:" in line:
        first_hc = line.strip()
        break

max_mem = 0.0
max_util = 0.0
if os.path.exists(gpu_csv):
    with open(gpu_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                max_util = max(max_util, float(row[2].strip()))
                max_mem = max(max_mem, float(row[3].strip()))
            except Exception:
                pass

decision = "hc_nonfinite_localized" if first_hc else "no_hc_nonfinite_observed"
if status_200 == len(responses):
    decision = "unexpected_pass"
elif status_other and not first_hc:
    decision = "failed_without_hc_nonfinite"

summary = {
    "schema": "ds4_v100_256k_finite_gate.v1",
    "ctx": int(ctx),
    "slots": int(slots),
    "active_microbatch": int(slots),
    "tokens": int(tokens),
    "requests": int(requests),
    "layer_checks": os.environ.get("DS4_V100_DEBUG_HC_FINITE_LAYER_CHECKS", ""),
    "pre_output_check": os.environ.get("DS4_V100_DEBUG_HC_FINITE_PRE_OUTPUT", ""),
    "startup_warmup": os.environ.get("DS4_V100_STARTUP_WARMUP", ""),
    "status_200": status_200,
    "status_other": status_other,
    "decision": decision,
    "first_error_body": first_error_body,
    "first_hc_nonfinite": first_hc,
    "max_gpu_util_percent": max_util,
    "max_memory_used_mib": max_mem,
}
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
    f.write("\n")
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# DS4 V100 256K Finite Gate\n\n")
    f.write(f"Decision: `{decision}`\n\n")
    f.write("| Ctx | Slots | Requests | Status 200 | Status other | Max GPU util | Max memory MiB |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
    f.write(f"| {ctx} | {slots} | {requests} | {status_200} | {status_other} | {max_util:.3f}% | {max_mem:.1f} |\n\n")
    if first_hc:
        f.write("First HC non-finite:\n\n")
        f.write("```text\n")
        f.write(first_hc + "\n")
        f.write("```\n\n")
    if first_error_body:
        f.write("First non-200 body:\n\n")
        f.write("```json\n")
        f.write(first_error_body[:2000] + "\n")
        f.write("```\n")
print(summary_json)
print(summary_md)
PY

cat "$summary_md"

case "$(python3 - "$summary_json" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("decision", ""))
PY
)" in
    hc_nonfinite_localized|failed_without_hc_nonfinite|unexpected_pass|no_hc_nonfinite_observed)
        exit 0
        ;;
    *)
        exit 1
        ;;
esac
