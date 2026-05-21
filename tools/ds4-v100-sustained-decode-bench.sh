#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
pack_index=""
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
ctx_tiers="1048576"
slot_tiers="1"
queue_policies="sequential"
tokens="16"
requests="8"
warmup_requests="1"
host="127.0.0.1"
port_base="18220"
sample_ms="500"
log_dir=""
profile_decode="0"
wavefront_decode="0"
async_pipeline_decode="0"
async_pipeline_mode="off"
async_handoff="0"
async_event_handoff="0"
mtp_serving="off"
mtp_top_k="5"
mtp_gpu="7"
mtp_reserve_mib="4096"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-sustained-decode-bench.sh --pack-index FILE [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          DeepSeek-V4 Flash MTP sidecar GGUF
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx-tiers LIST          comma list of ctx tiers, default 1048576
  --slot-tiers LIST         comma list of slot tiers, default 1
  --queue-policies LIST     comma list: sequential,reject-busy, default sequential
  --tokens N                generated tokens per request, default 16
  --requests N              timed requests per case, default 8
  --warmup-requests N       sequential warmup requests per case, default 1
  --host ADDR               bind/probe address, default 127.0.0.1
  --port-base N             base port for case runs, default 18220
  --sample-ms N             nvidia-smi sample period in ms, default 500
  --log-dir DIR             write benchmark artifacts
  --profile-decode          pass --profile-decode to the replay server and
                            preserve averaged stage_profile timing
  --wavefront-decode        pass --wavefront-decode to the replay server
  --async-pipeline-decode   pass preferred async pipeline mode to the server
  --async-pipeline-mode M   off, persistent, per-step, or mailbox
  --async-pipeline-per-step pass --async-pipeline-mode per-step
  --async-handoff           queue HC peer handoff copies on the destination stream
  --async-event-handoff     use CUDA events for per-step stage handoff ordering
  --mtp-serving MODE        off, verify, or commit, default off
  --mtp-top-k N             MTP draft candidates to report, default 5
  --mtp-gpu N               MTP sidecar GPU, default 7
  --mtp-reserve-mib N       MTP free-memory reserve, default 4096
  --help                    show this help

Each case starts one resident replay server with:
  active_microbatch = slots
  concurrency        = slots

The benchmark writes:
  sustained_decode.tsv
  sustained_decode.json
  cases/<case>/result.json
  cases/<case>/server.log
  cases/<case>/gpu_util.csv when nvidia-smi is available

The report separates aggregate generated tok/s from continuation tok/s so
multi-token decode can be evaluated without treating first-token prompt replay
as steady-state decode.
USAGE
}

fail() {
    echo "ds4-v100-sustained-decode-bench: $*" >&2
    exit 1
}

parse_csv_numbers() {
    local csv="$1"
    local kind="$2"
    local item=""
    IFS=',' read -r -a _items <<<"$csv"
    if [ "${#_items[@]}" -eq 0 ]; then
        fail "empty $kind list"
    fi
    for item in "${_items[@]}"; do
        case "$item" in
            ''|*[!0-9]*)
                fail "invalid numeric value '$item' in $kind list"
                ;;
        esac
    done
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
        --pack-index|--index)
            [ "$#" -ge 2 ] || fail "--pack-index requires a value"
            pack_index="$2"
            shift 2
            ;;
        --prompt-file)
            [ "$#" -ge 2 ] || fail "--prompt-file requires a value"
            prompt_file="$2"
            shift 2
            ;;
        --expected-token-hex)
            [ "$#" -ge 2 ] || fail "--expected-token-hex requires a value"
            expected_hex="$2"
            shift 2
            ;;
        --ctx-tiers)
            [ "$#" -ge 2 ] || fail "--ctx-tiers requires a value"
            ctx_tiers="$2"
            shift 2
            ;;
        --slot-tiers)
            [ "$#" -ge 2 ] || fail "--slot-tiers requires a value"
            slot_tiers="$2"
            shift 2
            ;;
        --queue-policies)
            [ "$#" -ge 2 ] || fail "--queue-policies requires a value"
            queue_policies="$2"
            shift 2
            ;;
        --tokens)
            [ "$#" -ge 2 ] || fail "--tokens requires a value"
            tokens="$2"
            shift 2
            ;;
        --requests)
            [ "$#" -ge 2 ] || fail "--requests requires a value"
            requests="$2"
            shift 2
            ;;
        --warmup-requests)
            [ "$#" -ge 2 ] || fail "--warmup-requests requires a value"
            warmup_requests="$2"
            shift 2
            ;;
        --host)
            [ "$#" -ge 2 ] || fail "--host requires a value"
            host="$2"
            shift 2
            ;;
        --port-base)
            [ "$#" -ge 2 ] || fail "--port-base requires a value"
            port_base="$2"
            shift 2
            ;;
        --sample-ms)
            [ "$#" -ge 2 ] || fail "--sample-ms requires a value"
            sample_ms="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || fail "--log-dir requires a value"
            log_dir="$2"
            shift 2
            ;;
        --profile-decode)
            profile_decode="1"
            shift
            ;;
        --wavefront-decode)
            wavefront_decode="1"
            shift
            ;;
        --async-pipeline-decode)
            async_pipeline_decode="1"
            async_pipeline_mode="per-step"
            shift
            ;;
        --async-pipeline-per-step)
            async_pipeline_decode="1"
            async_pipeline_mode="per-step"
            shift
            ;;
        --async-handoff)
            async_handoff="1"
            shift
            ;;
        --async-event-handoff)
            async_pipeline_decode="1"
            async_pipeline_mode="per-step"
            async_event_handoff="1"
            shift
            ;;
        --async-pipeline-mode)
            [ "$#" -ge 2 ] || fail "--async-pipeline-mode requires a value"
            case "$2" in
                off|false|0)
                    async_pipeline_decode="0"
                    async_pipeline_mode="off"
                    ;;
                persistent|on|true|1)
                    async_pipeline_decode="1"
                    async_pipeline_mode="persistent"
                    ;;
                per-step|per_step|step)
                    async_pipeline_decode="1"
                    async_pipeline_mode="per-step"
                    ;;
                mailbox|mbox)
                    async_pipeline_decode="1"
                    async_pipeline_mode="mailbox"
                    ;;
                *)
                    fail "--async-pipeline-mode must be off, persistent, per-step, or mailbox"
                    ;;
            esac
            shift 2
            ;;
        --mtp-serving)
            [ "$#" -ge 2 ] || fail "--mtp-serving requires a value"
            case "$2" in
                off|false|0) mtp_serving="off" ;;
                verify) mtp_serving="verify" ;;
                commit) mtp_serving="commit" ;;
                *) fail "--mtp-serving must be off, verify, or commit" ;;
            esac
            shift 2
            ;;
        --mtp-top-k)
            [ "$#" -ge 2 ] || fail "--mtp-top-k requires a value"
            mtp_top_k="$2"
            shift 2
            ;;
        --mtp-gpu)
            [ "$#" -ge 2 ] || fail "--mtp-gpu requires a value"
            mtp_gpu="$2"
            shift 2
            ;;
        --mtp-reserve-mib)
            [ "$#" -ge 2 ] || fail "--mtp-reserve-mib requires a value"
            mtp_reserve_mib="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ds4-v100-sustained-decode-bench: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[ -n "$pack_index" ] || { usage >&2; exit 2; }
[ -x ./tools/ds4-v100-replay ] || fail "missing executable ./tools/ds4-v100-replay"
[ -f "$model" ] || fail "missing model $model"
[ -f "$pack_index" ] || fail "missing pack index $pack_index"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"
if [ "$mtp_serving" != "off" ]; then
    [ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
fi

case "$tokens" in ''|0|1|*[!0-9]*) fail "--tokens must be an integer >= 2" ;; esac
case "$requests" in ''|0|*[!0-9]*) fail "--requests must be a positive integer" ;; esac
case "$warmup_requests" in ''|*[!0-9]*) fail "--warmup-requests must be a non-negative integer" ;; esac
case "$port_base" in ''|0|*[!0-9]*) fail "--port-base must be a positive integer" ;; esac
case "$sample_ms" in ''|0|*[!0-9]*) fail "--sample-ms must be a positive integer" ;; esac
case "$mtp_top_k" in ''|0|1|*[!0-9]*) fail "--mtp-top-k must be an integer >= 2" ;; esac
case "$mtp_gpu" in ''|*[!0-9]*) fail "--mtp-gpu must be an integer" ;; esac
case "$mtp_reserve_mib" in ''|*[!0-9]*) fail "--mtp-reserve-mib must be an integer" ;; esac
[ "$mtp_top_k" -le 16 ] || fail "--mtp-top-k must be <= 16"

parse_csv_numbers "$ctx_tiers" "--ctx-tiers"
parse_csv_numbers "$slot_tiers" "--slot-tiers"

IFS=',' read -r -a policy_list <<<"$queue_policies"
if [ "${#policy_list[@]}" -eq 0 ]; then
    fail "empty --queue-policies list"
fi
for policy in "${policy_list[@]}"; do
    case "$policy" in
        sequential|reject-busy)
            ;;
        *)
            fail "--queue-policies must contain only sequential,reject-busy"
            ;;
    esac
done

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-sustained-decode.XXXXXX)"
else
    mkdir -p "$work_dir" || exit 2
fi

request_json="$work_dir/request.json"
cases_dir="$work_dir/cases"
summary_tsv="$work_dir/sustained_decode.tsv"
summary_json="$work_dir/sustained_decode.json"
mkdir -p "$cases_dir"

cleanup() {
    if [ -n "${server_pid:-}" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "${gpu_pid:-}" ]; then
        kill "$gpu_pid" >/dev/null 2>&1 || true
        wait "$gpu_pid" >/dev/null 2>&1 || true
    fi
    if [ -z "$log_dir" ]; then
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

python3 - "$prompt_file" "$tokens" >"$request_json" <<'PY'
import json
import sys

prompt_path = sys.argv[1]
tokens = int(sys.argv[2])
with open(prompt_path, "r", encoding="utf-8", errors="ignore") as f:
    prompt = f.read()
print(json.dumps({"prompt": prompt, "tokens": tokens}))
PY

run_load_client() {
    local port="$1"
    local slots="$2"
    local result_json="$3"
    python3 - "$host" "$port" "$request_json" "$requests" "$slots" "$tokens" "$expected_hex" "$warmup_requests" "$result_json" <<'PY'
import http.client
import json
import math
import statistics
import sys
import threading
import time

host = sys.argv[1]
port = int(sys.argv[2])
request_json_path = sys.argv[3]
n_requests = int(sys.argv[4])
concurrency = int(sys.argv[5])
expected_tokens = int(sys.argv[6])
expected_hex = sys.argv[7].strip().lower()
warmup_requests = int(sys.argv[8])
out_path = sys.argv[9]

with open(request_json_path, "rb") as f:
    payload = f.read()

headers = {
    "Content-Type": "application/json",
    "Content-Length": str(len(payload)),
    "Connection": "close",
}

def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (len(sorted_vals) - 1) * p
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)

def send_once():
    t0 = time.perf_counter()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=600)
        conn.request("POST", "/v100/selected-token", body=payload, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        body = resp.read()
        conn.close()
    except Exception as exc:
        return {
            "status": -1,
            "latency_ms": 0.0,
            "error": repr(exc),
            "generated": 0,
            "first_hex": "",
            "timing": {},
            "mtp": {},
        }
    dt_ms = (time.perf_counter() - t0) * 1000.0
    timing = {}
    generated = 0
    first_hex = ""
    parse_error = ""
    if status == 200:
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
            generated = int(data.get("generated_tokens", 0))
            toks = data.get("tokens", [])
            if toks:
                first_hex = str(toks[0].get("text_hex", "")).lower()
            timing = data.get("timing_ms", {}) if isinstance(data, dict) else {}
            mtp = data.get("mtp", {}) if isinstance(data, dict) else {}
        except Exception as exc:
            parse_error = repr(exc)
            mtp = {}
    else:
        mtp = {}
    return {
        "status": status,
        "latency_ms": dt_ms,
        "error": parse_error,
        "generated": generated,
        "first_hex": first_hex,
        "timing": timing if isinstance(timing, dict) else {},
        "mtp": mtp if isinstance(mtp, dict) else {},
    }

def add_timing(acc, resp):
    timing = resp.get("timing", {})
    for key in (
        "prompt_replay",
        "continuation_decode",
        "output_head",
        "token_text",
        "total",
        "prompt_tokens_per_second",
        "continuation_tokens_per_second",
        "generated_tokens_per_second",
    ):
        try:
            acc.setdefault(key, []).append(float(timing.get(key, 0.0)))
        except Exception:
            acc.setdefault(key, []).append(0.0)
    for key in ("stage_decode", "handoff"):
        vals = timing.get(key, [])
        if not isinstance(vals, list):
            vals = []
        acc.setdefault(key, []).append([float(v) for v in vals])
    async_pipeline = timing.get("async_pipeline", {})
    if not isinstance(async_pipeline, dict):
        async_pipeline = {}
    for key in ("total", "setup", "host_wait", "complete"):
        try:
            acc.setdefault("async_pipeline_" + key, []).append(float(async_pipeline.get(key, 0.0)))
        except Exception:
            acc.setdefault("async_pipeline_" + key, []).append(0.0)
    for key in ("wait_prev", "handoff", "device_sync"):
        vals = async_pipeline.get(key, [])
        if not isinstance(vals, list):
            vals = []
        acc.setdefault("async_pipeline_" + key, []).append([float(v) for v in vals])
    stage_profile = timing.get("stage_profile", {})
    if not isinstance(stage_profile, dict):
        stage_profile = {}
    for key in ("hc_attn", "attention", "hc_ffn", "ffn", "hc_final", "total"):
        vals = stage_profile.get(key, [])
        if not isinstance(vals, list):
            vals = []
        acc.setdefault("stage_profile_" + key, []).append([float(v) for v in vals])

def avg(vals):
    return statistics.fmean(vals) if vals else 0.0

def avg_arrays(rows):
    if not rows:
        return []
    width = max((len(r) for r in rows), default=0)
    out = []
    for i in range(width):
        vals = [r[i] for r in rows if i < len(r)]
        out.append(avg(vals))
    return out

warmup = {
    "requests": warmup_requests,
    "status_200": 0,
    "errors": 0,
    "token_match": 0,
    "token_mismatch": 0,
}
for _ in range(warmup_requests):
    r = send_once()
    if r["status"] == 200:
        warmup["status_200"] += 1
    if r.get("error"):
        warmup["errors"] += 1
    if r["status"] == 200 and r["generated"] == expected_tokens and r["first_hex"] == expected_hex:
        warmup["token_match"] += 1
    else:
        warmup["token_mismatch"] += 1

stats = {
    "status_200": 0,
    "status_other": 0,
    "status_429": 0,
    "status_413": 0,
    "errors": 0,
    "token_match": 0,
    "token_mismatch": 0,
    "generated_token_total": 0,
    "continuation_token_total": 0,
    "mtp_attempted": 0,
    "mtp_accepted": 0,
    "mtp_rejected": 0,
    "mtp_committed": 0,
    "mtp_skipped": 0,
}
latencies_ms = []
timing_acc = {}
mtp_draft_ms = []
next_request = [0]
next_lock = threading.Lock()
stats_lock = threading.Lock()

def worker():
    while True:
        with next_lock:
            if next_request[0] >= n_requests:
                return
            next_request[0] += 1
        r = send_once()
        with stats_lock:
            if r["latency_ms"] > 0:
                latencies_ms.append(r["latency_ms"])
            if r["status"] == 200:
                stats["status_200"] += 1
            else:
                stats["status_other"] += 1
                if r["status"] == 429:
                    stats["status_429"] += 1
                if r["status"] == 413:
                    stats["status_413"] += 1
            if r.get("error"):
                stats["errors"] += 1
            generated = int(r.get("generated", 0))
            stats["generated_token_total"] += generated
            stats["continuation_token_total"] += max(0, generated - 1)
            if r["status"] == 200 and generated == expected_tokens and r["first_hex"] == expected_hex:
                stats["token_match"] += 1
            else:
                stats["token_mismatch"] += 1
            if r["status"] == 200:
                add_timing(timing_acc, r)
                mtp = r.get("mtp", {})
                if isinstance(mtp, dict) and mtp.get("enabled"):
                    if mtp.get("attempted"):
                        stats["mtp_attempted"] += int(mtp.get("attempts", 1) or 1)
                    stats["mtp_accepted"] += int(mtp.get("accepted_count", 1 if mtp.get("accepted") else 0) or 0)
                    stats["mtp_rejected"] += int(mtp.get("rejected_count", 0) or 0)
                    stats["mtp_committed"] += int(mtp.get("commit_count", 0) or 0)
                    if mtp.get("skipped"):
                        stats["mtp_skipped"] += 1
                    try:
                        mtp_draft_ms.append(float(mtp.get("draft_total_ms", mtp.get("draft_ms", 0.0)) or 0.0))
                    except Exception:
                        mtp_draft_ms.append(0.0)

threads = []
t0 = time.perf_counter()
for _ in range(max(1, concurrency)):
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    threads.append(t)
for t in threads:
    t.join()
elapsed_s = max(0.0, time.perf_counter() - t0)

sorted_lat = sorted(latencies_ms)
timing_avg = {
    "prompt_replay_ms": avg(timing_acc.get("prompt_replay", [])),
    "continuation_decode_ms": avg(timing_acc.get("continuation_decode", [])),
    "output_head_ms": avg(timing_acc.get("output_head", [])),
    "token_text_ms": avg(timing_acc.get("token_text", [])),
    "total_ms": avg(timing_acc.get("total", [])),
    "prompt_tokens_per_second": avg(timing_acc.get("prompt_tokens_per_second", [])),
    "continuation_tokens_per_second": avg(timing_acc.get("continuation_tokens_per_second", [])),
    "generated_tokens_per_second": avg(timing_acc.get("generated_tokens_per_second", [])),
    "stage_decode_ms": avg_arrays(timing_acc.get("stage_decode", [])),
    "handoff_ms": avg_arrays(timing_acc.get("handoff", [])),
    "stage_profile_ms": {
        "hc_attn": avg_arrays(timing_acc.get("stage_profile_hc_attn", [])),
        "attention": avg_arrays(timing_acc.get("stage_profile_attention", [])),
        "hc_ffn": avg_arrays(timing_acc.get("stage_profile_hc_ffn", [])),
        "ffn": avg_arrays(timing_acc.get("stage_profile_ffn", [])),
        "hc_final": avg_arrays(timing_acc.get("stage_profile_hc_final", [])),
        "total": avg_arrays(timing_acc.get("stage_profile_total", [])),
    },
    "async_pipeline_ms": {
        "total": avg(timing_acc.get("async_pipeline_total", [])),
        "setup": avg(timing_acc.get("async_pipeline_setup", [])),
        "host_wait": avg(timing_acc.get("async_pipeline_host_wait", [])),
        "complete": avg(timing_acc.get("async_pipeline_complete", [])),
        "wait_prev": avg_arrays(timing_acc.get("async_pipeline_wait_prev", [])),
        "handoff": avg_arrays(timing_acc.get("async_pipeline_handoff", [])),
        "device_sync": avg_arrays(timing_acc.get("async_pipeline_device_sync", [])),
    },
}

summary = {
    "schema": "ds4_v100_sustained_decode_case.v1",
    "host": host,
    "port": port,
    "requests": n_requests,
    "warmup": warmup,
    "concurrency": concurrency,
    "tokens_per_request": expected_tokens,
    "elapsed_s": elapsed_s,
    "status_200": stats["status_200"],
    "status_other": stats["status_other"],
    "status_429": stats["status_429"],
    "status_413": stats["status_413"],
    "errors": stats["errors"],
    "token_match": stats["token_match"],
    "token_mismatch": stats["token_mismatch"],
    "generated_token_total": stats["generated_token_total"],
    "continuation_token_total": stats["continuation_token_total"],
    "mtp": {
        "attempted": stats["mtp_attempted"],
        "accepted": stats["mtp_accepted"],
        "rejected": stats["mtp_rejected"],
        "committed": stats["mtp_committed"],
        "skipped": stats["mtp_skipped"],
        "draft_ms_avg": avg(mtp_draft_ms),
        "draft_ms_total": sum(mtp_draft_ms),
    },
    "latency_ms": {
        "avg": avg(latencies_ms),
        "p50": percentile(sorted_lat, 0.50),
        "p95": percentile(sorted_lat, 0.95),
        "p99": percentile(sorted_lat, 0.99),
    },
    "aggregate_generated_tokens_per_second": (
        stats["generated_token_total"] / elapsed_s if elapsed_s > 0 else 0.0
    ),
    "aggregate_continuation_tokens_per_second": (
        stats["continuation_token_total"] / elapsed_s if elapsed_s > 0 else 0.0
    ),
    "timing_avg": timing_avg,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")

ok = (
    warmup["status_200"] == warmup_requests
    and warmup["errors"] == 0
    and warmup["token_mismatch"] == 0
    and stats["status_200"] == n_requests
    and stats["status_other"] == 0
    and stats["errors"] == 0
    and stats["token_mismatch"] == 0
)
sys.exit(0 if ok else 1)
PY
}

fetch_server_status() {
    local port="$1"
    local out_path="$2"
    python3 - "$host" "$port" "$out_path" <<'PY'
import http.client
import sys

host = sys.argv[1]
port = int(sys.argv[2])
out_path = sys.argv[3]

try:
    conn = http.client.HTTPConnection(host, port, timeout=30)
    conn.request("GET", "/v100/status")
    resp = conn.getresponse()
    body = resp.read()
    conn.close()
except Exception:
    sys.exit(1)

if resp.status != 200:
    sys.exit(1)

with open(out_path, "wb") as f:
    f.write(body)
    if not body.endswith(b"\n"):
        f.write(b"\n")
PY
}

merge_server_status() {
    local result_json="$1"
    local before_json="$2"
    local after_json="$3"
    python3 - "$result_json" "$before_json" "$after_json" <<'PY'
import json
import os
import sys

result_path, before_path, after_path = sys.argv[1:4]
with open(result_path, "r", encoding="utf-8") as f:
    result = json.load(f)

def load_status(path):
    if not path or not os.path.exists(path):
        return {"available": False, "reason": "missing_status_snapshot"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}

result["server_status_before"] = load_status(before_path)
result["server_status_after"] = load_status(after_path)

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, sort_keys=True)
    f.write("\n")
PY
}

merge_gpu_utilization() {
    local result_json="$1"
    local gpu_csv="$2"
    local gpu_err="$3"
    python3 - "$result_json" "$gpu_csv" "$gpu_err" <<'PY'
import json
import math
import os
import sys

result_path, csv_path, err_path = sys.argv[1:4]
with open(result_path, "r", encoding="utf-8") as f:
    result = json.load(f)

def to_float(value):
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v

samples = []
if os.path.exists(csv_path):
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\n").split(",")]
            if len(parts) < 7:
                continue
            gpu = to_float(parts[2])
            mem = to_float(parts[3])
            used = to_float(parts[4])
            total = to_float(parts[5])
            power = to_float(parts[6])
            if gpu is None:
                continue
            samples.append({
                "index": int(to_float(parts[1]) or 0),
                "gpu_util": gpu,
                "memory_util": mem,
                "memory_used_mib": used,
                "memory_total_mib": total,
                "power_draw_w": power,
            })

if samples:
    gpu_vals = [s["gpu_util"] for s in samples if s["gpu_util"] is not None]
    mem_vals = [s["memory_util"] for s in samples if s["memory_util"] is not None]
    used_vals = [s["memory_used_mib"] for s in samples if s["memory_used_mib"] is not None]
    power_vals = [s["power_draw_w"] for s in samples if s["power_draw_w"] is not None]
    per_gpu = []
    for idx in sorted({s["index"] for s in samples}):
        rows = [s for s in samples if s["index"] == idx]
        row_gpu = [s["gpu_util"] for s in rows if s["gpu_util"] is not None]
        row_mem = [s["memory_util"] for s in rows if s["memory_util"] is not None]
        row_used = [s["memory_used_mib"] for s in rows if s["memory_used_mib"] is not None]
        row_power = [s["power_draw_w"] for s in rows if s["power_draw_w"] is not None]
        per_gpu.append({
            "index": idx,
            "samples": len(rows),
            "avg_gpu_util_percent": sum(row_gpu) / len(row_gpu) if row_gpu else 0.0,
            "max_gpu_util_percent": max(row_gpu) if row_gpu else 0.0,
            "avg_memory_util_percent": sum(row_mem) / len(row_mem) if row_mem else 0.0,
            "max_memory_used_mib": max(row_used) if row_used else 0.0,
            "avg_power_draw_w": sum(row_power) / len(row_power) if row_power else 0.0,
        })
    result["gpu_utilization"] = {
        "available": True,
        "samples": len(samples),
        "avg_gpu_util_percent": sum(gpu_vals) / len(gpu_vals) if gpu_vals else 0.0,
        "max_gpu_util_percent": max(gpu_vals) if gpu_vals else 0.0,
        "avg_memory_util_percent": sum(mem_vals) / len(mem_vals) if mem_vals else 0.0,
        "max_memory_used_mib": max(used_vals) if used_vals else 0.0,
        "avg_power_draw_w": sum(power_vals) / len(power_vals) if power_vals else 0.0,
        "per_gpu": per_gpu,
    }
else:
    reason = "nvidia_smi_unavailable_or_no_samples"
    if os.path.exists(err_path):
        with open(err_path, "r", encoding="utf-8", errors="replace") as f:
            err = f.read().strip()
        if err:
            reason = err[:240]
    result["gpu_utilization"] = {
        "available": False,
        "samples": 0,
        "reason": reason,
    }

with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, sort_keys=True)
    f.write("\n")
PY
}

load_case() {
    local case_dir="$1"
    local case_name="$2"
    local ctx="$3"
    local slots="$4"
    local policy="$5"
    local port="$6"

    local server_log="$case_dir/server.log"
    local result_json="$case_dir/result.json"
    local status_before_json="$case_dir/server_status_before.json"
    local status_after_json="$case_dir/server_status_after.json"
    local gpu_csv="$case_dir/gpu_util.csv"
    local gpu_err="$case_dir/gpu_util.err"
    local server_max_requests
    server_max_requests=$((requests + warmup_requests + 32))
    if [ "$mtp_serving" != "off" ] && [ "$slots" -ne 1 ]; then
        fail "MTP sustained decode benchmark currently requires slot tier 1"
    fi

    local cmd=(
        ./tools/ds4-v100-replay
        --serve
        --model "$model"
        --index "$pack_index"
        --ctx "$ctx"
        --slots "$slots"
        --active-microbatch "$slots"
        --queue-policy "$policy"
        --tokens "$tokens"
        --host "$host"
        --port "$port"
        --max-requests "$server_max_requests"
    )
    if [ "$profile_decode" = "1" ]; then
        cmd+=(--profile-decode)
    fi
    if [ "$wavefront_decode" = "1" ]; then
        cmd+=(--wavefront-decode)
    fi
    if [ "$async_pipeline_decode" = "1" ]; then
        cmd+=(--async-pipeline-mode "$async_pipeline_mode")
    fi
    if [ "$async_handoff" = "1" ]; then
        cmd+=(--async-handoff)
    fi
    if [ "$async_event_handoff" = "1" ]; then
        cmd+=(--async-event-handoff)
    fi
    if [ "$mtp_serving" != "off" ]; then
        cmd+=(
            --mtp-model "$mtp_model"
            --mtp-serving "$mtp_serving"
            --mtp-top-k "$mtp_top_k"
            --mtp-gpu "$mtp_gpu"
            --mtp-reserve-mib "$mtp_reserve_mib"
        )
    fi

    DS4_LOCK_FILE="$case_dir/ds4.lock" "${cmd[@]}" >"$server_log" 2>&1 &
    server_pid="$!"

    for _ in $(seq 1 420); do
        if ! kill -0 "$server_pid" >/dev/null 2>&1; then
            cat "$server_log" >&2
            fail "case $case_name server exited before listening"
        fi
        if grep -q "serving http://" "$server_log"; then
            break
        fi
        sleep 1
    done
    if ! grep -q "serving http://" "$server_log"; then
        cat "$server_log" >&2
        fail "case $case_name server did not start listening in time"
    fi
    fetch_server_status "$port" "$status_before_json" || true

    gpu_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits \
            -lms "$sample_ms" >"$gpu_csv" 2>"$gpu_err" &
        gpu_pid="$!"
    else
        printf '%s\n' "nvidia-smi command not found" >"$gpu_err"
    fi

    local py_rc=0
    run_load_client "$port" "$slots" "$result_json" || py_rc=$?
    fetch_server_status "$port" "$status_after_json" || true

    if [ -n "$gpu_pid" ]; then
        kill "$gpu_pid" >/dev/null 2>&1 || true
        wait "$gpu_pid" >/dev/null 2>&1 || true
        gpu_pid=""
    fi

    if [ -n "$server_pid" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
        server_pid=""
    fi

    if [ -f "$result_json" ]; then
        merge_server_status "$result_json" "$status_before_json" "$status_after_json"
        merge_gpu_utilization "$result_json" "$gpu_csv" "$gpu_err"
    fi
    return "$py_rc"
}

expected_lower="$(printf '%s' "$expected_hex" | tr 'A-F' 'a-f')"
printf 'schema\tds4_v100_sustained_decode.v1\n' >"$summary_tsv"
printf 'model\t%s\n' "$model" >>"$summary_tsv"
printf 'pack_index\t%s\n' "$pack_index" >>"$summary_tsv"
printf 'expected_token_hex\t%s\n' "$expected_lower" >>"$summary_tsv"
printf 'tokens_per_request\t%s\n' "$tokens" >>"$summary_tsv"
printf 'requests_per_case\t%s\n' "$requests" >>"$summary_tsv"
printf 'warmup_requests\t%s\n' "$warmup_requests" >>"$summary_tsv"
printf 'profile_decode\t%s\n' "$profile_decode" >>"$summary_tsv"
printf 'wavefront_decode\t%s\n' "$wavefront_decode" >>"$summary_tsv"
printf 'async_pipeline_decode\t%s\n' "$async_pipeline_decode" >>"$summary_tsv"
printf 'async_pipeline_mode\t%s\n' "$async_pipeline_mode" >>"$summary_tsv"
printf 'async_event_handoff\t%s\n' "$async_event_handoff" >>"$summary_tsv"
printf 'async_handoff\t%s\n' "$async_handoff" >>"$summary_tsv"
printf 'mtp_serving\t%s\n' "$mtp_serving" >>"$summary_tsv"
printf 'mtp_top_k\t%s\n' "$mtp_top_k" >>"$summary_tsv"
printf '\nctx\tslots\tpolicy\tmtp_serving\tstatus_200\tstatus_other\terrors\ttoken_match\ttoken_mismatch\tlatency_avg_ms\tlatency_p50_ms\tlatency_p95_ms\tlatency_p99_ms\telapsed_s\taggregate_generated_tokens_per_second\taggregate_continuation_tokens_per_second\tavg_continuation_response_tokens_per_second\tavg_gpu_util_percent\tmax_gpu_util_percent\tmtp_attempted\tmtp_accepted\tmtp_rejected\tmtp_committed\tmtp_draft_ms_avg\tmtp_draft_ms_total\tavg_async_total_ms\tavg_async_setup_ms\tavg_async_host_wait_ms\tavg_async_complete_ms\tavg_async_wait_prev_sum_ms\tavg_async_handoff_sum_ms\tavg_async_device_sync_sum_ms\n' >>"$summary_tsv"

case_index=0
case_paths=()
IFS=',' read -r -a ctx_list <<<"$ctx_tiers"
IFS=',' read -r -a slots_list <<<"$slot_tiers"
for ctx in "${ctx_list[@]}"; do
    for slots in "${slots_list[@]}"; do
        if [ "$slots" -lt 1 ] || [ "$slots" -gt 16 ]; then
            fail "slot tier $slots must be in [1,16]"
        fi
        for policy in "${policy_list[@]}"; do
            case_index=$((case_index + 1))
            port=$((port_base + case_index - 1))
            case_name="case_${case_index}_ctx${ctx}_s${slots}_${policy}_mtp${mtp_serving}_tok${tokens}"
            case_dir="$cases_dir/$case_name"
            mkdir -p "$case_dir"
            if ! load_case "$case_dir" "$case_name" "$ctx" "$slots" "$policy" "$port"; then
                if [ -f "$case_dir/result.json" ]; then
                    cat "$case_dir/result.json" >&2
                fi
                if [ -f "$case_dir/server.log" ]; then
                    cat "$case_dir/server.log" >&2
                fi
                fail "sustained decode case failed: $case_name"
            fi
            case_paths+=("$case_dir/result.json")
            row="$(python3 - "$case_dir/result.json" "$ctx" "$slots" "$policy" "$mtp_serving" <<'PY'
import json
import sys

rpath, ctx, slots, policy, mtp_serving = sys.argv[1:]
with open(rpath, "r", encoding="utf-8") as f:
    d = json.load(f)
lat = d.get("latency_ms", {})
timing = d.get("timing_avg", {})
async_timing = timing.get("async_pipeline_ms", {})
if not isinstance(async_timing, dict):
    async_timing = {}
gpu = d.get("gpu_utilization", {})
def sum_array(name):
    vals = async_timing.get(name, [])
    if not isinstance(vals, list):
        return 0.0
    total = 0.0
    for v in vals:
        try:
            total += float(v)
        except Exception:
            pass
    return total
mtp = d.get("mtp", {})
if not isinstance(mtp, dict):
    mtp = {}
print("\t".join([
    ctx,
    slots,
    policy,
    mtp_serving,
    str(d.get("status_200", 0)),
    str(d.get("status_other", 0)),
    str(d.get("errors", 0)),
    str(d.get("token_match", 0)),
    str(d.get("token_mismatch", 0)),
    f"{float(lat.get('avg', 0.0)):.3f}",
    f"{float(lat.get('p50', 0.0)):.3f}",
    f"{float(lat.get('p95', 0.0)):.3f}",
    f"{float(lat.get('p99', 0.0)):.3f}",
    f"{float(d.get('elapsed_s', 0.0)):.6f}",
    f"{float(d.get('aggregate_generated_tokens_per_second', 0.0)):.6f}",
    f"{float(d.get('aggregate_continuation_tokens_per_second', 0.0)):.6f}",
    f"{float(timing.get('continuation_tokens_per_second', 0.0)):.6f}",
    f"{float(gpu.get('avg_gpu_util_percent', 0.0)):.3f}",
    f"{float(gpu.get('max_gpu_util_percent', 0.0)):.3f}",
    str(mtp.get("attempted", 0)),
    str(mtp.get("accepted", 0)),
    str(mtp.get("rejected", 0)),
    str(mtp.get("committed", 0)),
    f"{float(mtp.get('draft_ms_avg', 0.0)):.3f}",
    f"{float(mtp.get('draft_ms_total', 0.0)):.3f}",
    f"{float(async_timing.get('total', 0.0)):.3f}",
    f"{float(async_timing.get('setup', 0.0)):.3f}",
    f"{float(async_timing.get('host_wait', 0.0)):.3f}",
    f"{float(async_timing.get('complete', 0.0)):.3f}",
    f"{sum_array('wait_prev'):.3f}",
    f"{sum_array('handoff'):.3f}",
    f"{sum_array('device_sync'):.3f}",
]))
PY
)"
            printf '%s\n' "$row" >>"$summary_tsv"
            echo "ds4-v100-sustained-decode-bench: PASS $case_name"
        done
    done
done

python3 - "$summary_json" "$mtp_serving" "$mtp_top_k" "$async_handoff" "$async_event_handoff" "${case_paths[@]}" <<'PY'
import json
import sys

out = sys.argv[1]
mtp_serving = sys.argv[2]
mtp_top_k = int(sys.argv[3])
async_handoff = bool(int(sys.argv[4]))
async_event_handoff = bool(int(sys.argv[5]))
paths = sys.argv[6:]
rows = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        rows.append(json.load(f))
summary = {
    "schema": "ds4_v100_sustained_decode.v1",
    "mtp_serving": mtp_serving,
    "mtp_top_k": mtp_top_k,
    "async_handoff": async_handoff,
    "async_event_handoff": async_event_handoff,
    "cases": rows,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")
PY

cat "$summary_tsv"
echo "ds4-v100-sustained-decode-bench: PASS cases=$case_index report=$summary_tsv json=$summary_json"
