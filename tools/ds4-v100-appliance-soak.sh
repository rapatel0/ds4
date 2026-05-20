#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
pack_index="docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
ctx="1048576"
slots="4"
active_microbatch="4"
queue_policy="sequential"
tokens="16"
requests="4"
host="127.0.0.1"
port="18420"
async_pipeline_mode="auto"
sample_ms="500"
log_dir=""
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
require_gpus="8"
reserve_mib="4096"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-appliance-soak.sh --log-dir DIR [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          MTP sidecar GGUF path for launcher validation
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx N                   KV context tokens, default 1048576
  --slots N                 configured slots, default 4
  --active-microbatch N     active decode slots, default slots
  --queue-policy MODE       sequential or reject-busy, default sequential
  --tokens N                generated tokens per request, default 16
  --requests N              timed requests, default 4
  --host ADDR               bind/probe address, default 127.0.0.1
  --port N                  server port, default 18420
  --async-pipeline-mode M   off, auto, per-step, or persistent, default auto
  --sample-ms N             nvidia-smi sample interval, default 500
  --cuda-visible-devices L  CUDA_VISIBLE_DEVICES list, default 0..7
  --require-gpus N          required visible GPU count, default 8
  --reserve-mib N           required free memory per GPU, default 4096
  --log-dir DIR             artifact directory, required
  --help                    show this help
USAGE
}

fail() {
    echo "ds4-v100-appliance-soak: $*" >&2
    exit 1
}

need_value() {
    local opt="$1"
    [ "$#" -ge 2 ] || fail "$opt requires a value"
    printf '%s' "$2"
}

is_uint() {
    case "${1:-}" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model) model="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --mtp-model) mtp_model="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --pack-index|--index) pack_index="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --prompt-file) prompt_file="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --expected-token-hex) expected_hex="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --ctx) ctx="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --slots) slots="$(need_value "$1" "${2:-}")"; active_microbatch="$slots"; shift 2 ;;
        --active-microbatch) active_microbatch="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --queue-policy) queue_policy="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --tokens) tokens="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --requests) requests="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --host) host="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --port) port="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --async-pipeline-mode) async_pipeline_mode="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --sample-ms) sample_ms="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --cuda-visible-devices) cuda_visible_devices="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --require-gpus) require_gpus="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --reserve-mib) reserve_mib="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --log-dir) log_dir="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) fail "unknown option: $1" ;;
    esac
done

[ -n "$log_dir" ] || { usage >&2; exit 2; }
[ -x ./tools/ds4-v100-run-appliance.sh ] || fail "missing ./tools/ds4-v100-run-appliance.sh"
[ -x ./tools/ds4-v100-replay ] || fail "missing ./tools/ds4-v100-replay"
[ -f "$model" ] || fail "missing model $model"
[ -f "$pack_index" ] || fail "missing pack index $pack_index"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"
[ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
for v in "$ctx" "$slots" "$active_microbatch" "$tokens" "$requests" "$port" "$sample_ms" "$require_gpus" "$reserve_mib"; do
    is_uint "$v" || fail "numeric option expected, got $v"
done
[ "$slots" -ge 1 ] && [ "$slots" -le 8 ] || fail "--slots must be in [1,8]"
[ "$active_microbatch" -ge 1 ] && [ "$active_microbatch" -le "$slots" ] || fail "--active-microbatch must be in [1,slots]"
case "$queue_policy" in sequential|reject-busy) ;; *) fail "--queue-policy must be sequential or reject-busy" ;; esac
case "$async_pipeline_mode" in off|auto|per-step|per_step|persistent) ;; *) fail "--async-pipeline-mode must be off, auto, per-step, or persistent" ;; esac

rm -rf "$log_dir"
mkdir -p "$log_dir/runtime"

request_json="$log_dir/request.json"
status_before="$log_dir/status_before.json"
status_after="$log_dir/status_after.json"
health_json="$log_dir/health.json"
metrics_before="$log_dir/metrics_before.txt"
metrics_after="$log_dir/metrics_after.txt"
responses_json="$log_dir/responses.json"
summary_json="$log_dir/summary.json"
server_log="$log_dir/server.log"
gpu_csv="$log_dir/gpu_util.csv"
gpu_err="$log_dir/gpu_util.err"

python3 - "$prompt_file" "$tokens" >"$request_json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8", errors="ignore") as f:
    prompt = f.read()
print(json.dumps({"prompt": prompt, "tokens": int(sys.argv[2])}))
PY

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
        while :; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
                --format=csv,noheader,nounits >>"$gpu_csv" 2>>"$gpu_err" || true
            sleep "$(python3 - "$sample_ms" <<'PY'
import sys
print(max(0.1, int(sys.argv[1]) / 1000.0))
PY
)"
        done
    ) &
    gpu_pid="$!"
fi

(
    export DS4_V100_BIN=./tools/ds4-v100-replay
    export DS4_V100_MODEL="$model"
    export DS4_V100_MTP_MODEL="$mtp_model"
    export DS4_V100_PACK_INDEX="$pack_index"
    export DS4_V100_CTX="$ctx"
    export DS4_V100_SLOTS="$slots"
    export DS4_V100_ACTIVE_MICROBATCH="$active_microbatch"
    export DS4_V100_QUEUE_POLICY="$queue_policy"
    export DS4_V100_TOKENS="$tokens"
    export DS4_V100_ASYNC_PIPELINE_MODE="$async_pipeline_mode"
    export DS4_V100_HOST="$host"
    export DS4_V100_PORT="$port"
    export DS4_V100_CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    export DS4_V100_REQUIRE_GPUS="$require_gpus"
    export DS4_V100_RESERVE_MIB="$reserve_mib"
    export DS4_V100_MAX_REQUESTS=$((requests + 64))
    export DS4_V100_LOG_DIR="$log_dir/runtime"
    export DS4_V100_SERVE_MODE=base
    export DS4_V100_MTP_SERVING=off
    exec ./tools/ds4-v100-run-appliance.sh
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

python3 - "$host" "$port" "$request_json" "$requests" "$expected_hex" "$health_json" "$status_before" "$metrics_before" "$responses_json" "$status_after" "$metrics_after" "$summary_json" <<'PY'
import http.client
import json
import statistics
import sys
import threading
import time

host, port_s, request_path, requests_s, expected_hex = sys.argv[1:6]
health_path, status_before_path, metrics_before_path, responses_path, status_after_path, metrics_after_path, summary_path = sys.argv[6:]
port = int(port_s)
n_requests = int(requests_s)
expected_hex = expected_hex.lower()

with open(request_path, "rb") as f:
    body = f.read()
headers = {"content-type": "application/json"}

def fetch(method, path, body_bytes=None, headers_in=None, timeout=900):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request(method, path, body_bytes, headers_in or {})
    resp = conn.getresponse()
    payload = resp.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    conn.close()
    return resp.status, payload.decode("utf-8", "replace"), elapsed_ms

status, payload, _ = fetch("GET", "/health", timeout=60)
assert status == 200, status
with open(health_path, "w", encoding="utf-8") as f:
    f.write(payload)
    f.write("\n")

status, payload, _ = fetch("GET", "/v100/status", timeout=60)
assert status == 200, status
status_before = json.loads(payload)
with open(status_before_path, "w", encoding="utf-8") as f:
    json.dump(status_before, f, sort_keys=True)
    f.write("\n")

status, payload, _ = fetch("GET", "/metrics", timeout=60)
assert status == 200, status
with open(metrics_before_path, "w", encoding="utf-8") as f:
    f.write(payload)

rows = []
lock = threading.Lock()
def run_one(i):
    status, payload, elapsed_ms = fetch("POST", "/v100/selected-token", body, headers)
    row = {"index": i, "status": status, "elapsed_ms": elapsed_ms}
    try:
        row["body"] = json.loads(payload)
    except Exception as exc:
        row["error"] = repr(exc)
        row["raw"] = payload
    with lock:
        rows.append(row)

threads = [threading.Thread(target=run_one, args=(i,)) for i in range(n_requests)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed_s = max(0.0, time.perf_counter() - t0)
rows.sort(key=lambda r: r["index"])

status, payload, _ = fetch("GET", "/v100/status", timeout=60)
assert status == 200, status
status_after = json.loads(payload)
with open(status_after_path, "w", encoding="utf-8") as f:
    json.dump(status_after, f, sort_keys=True)
    f.write("\n")

status, payload, _ = fetch("GET", "/metrics", timeout=60)
assert status == 200, status
with open(metrics_after_path, "w", encoding="utf-8") as f:
    f.write(payload)

matches = 0
errors = 0
latencies = []
generated = 0
continuation = 0
for row in rows:
    latencies.append(float(row.get("elapsed_ms", 0.0)))
    body_obj = row.get("body")
    if row.get("status") != 200 or not isinstance(body_obj, dict):
        errors += 1
        continue
    toks = body_obj.get("tokens") or []
    first_hex = str(toks[0].get("text_hex", "")).lower() if toks else ""
    if first_hex == expected_hex and "async_pipeline" in body_obj.get("timing_ms", {}):
        matches += 1
    else:
        errors += 1
    n_generated = int(body_obj.get("generated_tokens", 0))
    generated += n_generated
    continuation += max(0, n_generated - 1)

with open(responses_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, sort_keys=True)
    f.write("\n")

summary = {
    "schema": "ds4_v100_appliance_soak.v1",
    "requests": n_requests,
    "elapsed_s": elapsed_s,
    "status_200": sum(1 for r in rows if r.get("status") == 200),
    "errors": errors,
    "token_match": matches,
    "generated_tokens": generated,
    "continuation_tokens": continuation,
    "aggregate_generated_tokens_per_second": generated / elapsed_s if elapsed_s > 0 else 0.0,
    "aggregate_continuation_tokens_per_second": continuation / elapsed_s if elapsed_s > 0 else 0.0,
    "latency_ms_avg": statistics.fmean(latencies) if latencies else 0.0,
    "async_pipeline_mode": status_before.get("async_pipeline_mode"),
    "async_pipeline_decode": bool(status_before.get("async_pipeline_decode")),
}
assert matches == n_requests, summary
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")
print(json.dumps(summary, sort_keys=True))
PY

if [ -n "$gpu_pid" ]; then
    kill "$gpu_pid" >/dev/null 2>&1 || true
    wait "$gpu_pid" >/dev/null 2>&1 || true
    gpu_pid=""
fi

cat "$summary_json"
