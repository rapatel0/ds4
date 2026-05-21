#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model=""
pack_index=""
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
ctx_tiers="262144,1048576"
slot_tiers="2,4"
queue_policies="sequential"
mtp_mode="off"
tokens="1"
requests="16"
host="127.0.0.1"
port_base="18120"
top_k="5"
mtp_gpu="7"
reserve_mib="4096"
log_dir=""

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-aggregate-throughput.sh --pack-index FILE [options]

Options:
  --model FILE               source-layout GGUF model
  --mtp-model FILE           DeepSeek-V4 Flash MTP sidecar GGUF (required for --mtp-mode verify/both)
  --pack-index FILE          V100 pack-index.tsv
  --prompt-file FILE         prompt file
  --expected-token-hex HEX   expected first response token bytes, default 3136
  --ctx-tiers LIST           comma list of ctx tiers, default 262144,1048576
  --slot-tiers LIST          comma list of slot tiers, default 2,4
  --queue-policies LIST      comma list: sequential,reject-busy, default sequential
  --mtp-mode MODE            off, verify, or both (default off)
  --tokens N                 generated tokens per request, default 1
  --requests N               requests per tier run, default 16
  --host ADDR                bind/probe address, default 127.0.0.1
  --port-base N              base port for tier runs, default 18120
  --top-k N                  MTP draft top-k for verify mode, default 5
  --mtp-gpu N                MTP sidecar GPU for verify mode, default 7
  --reserve-mib N            required MTP reserve MiB for verify mode, default 4096
  --log-dir DIR              write benchmark artifacts
  --help                     show this help

Each tier run starts one resident replay server with:
  active_microbatch = slots
  concurrency        = slots

For each case the script emits:
  - success/failure counters
  - p50/p95/p99 request latency
  - aggregate tok/s across successful requests
  - (verify mode) MTP attempted/accepted counts
USAGE
}

fail() {
    echo "ds4-v100-aggregate-throughput: $*" >&2
    exit 1
}

parse_csv() {
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
        --mtp-mode)
            [ "$#" -ge 2 ] || fail "--mtp-mode requires a value"
            mtp_mode="$2"
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
        --top-k)
            [ "$#" -ge 2 ] || fail "--top-k requires a value"
            top_k="$2"
            shift 2
            ;;
        --mtp-gpu)
            [ "$#" -ge 2 ] || fail "--mtp-gpu requires a value"
            mtp_gpu="$2"
            shift 2
            ;;
        --reserve-mib)
            [ "$#" -ge 2 ] || fail "--reserve-mib requires a value"
            reserve_mib="$2"
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
            echo "ds4-v100-aggregate-throughput: unknown option: $1" >&2
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

case "$tokens" in ''|0|*[!0-9]*) fail "--tokens must be a positive integer" ;; esac
case "$requests" in ''|0|*[!0-9]*) fail "--requests must be a positive integer" ;; esac
case "$port_base" in ''|0|*[!0-9]*) fail "--port-base must be a positive integer" ;; esac
case "$top_k" in ''|0|1|*[!0-9]*) fail "--top-k must be an integer >= 2" ;; esac
case "$mtp_gpu" in ''|*[!0-9]*) fail "--mtp-gpu must be an integer" ;; esac
case "$reserve_mib" in ''|*[!0-9]*) fail "--reserve-mib must be an integer" ;; esac

parse_csv "$ctx_tiers" "--ctx-tiers"
parse_csv "$slot_tiers" "--slot-tiers"

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

case "$mtp_mode" in
    off|verify|both)
        ;;
    *)
        fail "--mtp-mode must be off, verify, or both"
        ;;
esac

if [ "$mtp_mode" != "off" ] && [ -z "$mtp_model" ]; then
    fail "--mtp-model is required for --mtp-mode $mtp_mode"
fi
if [ -n "$mtp_model" ] && [ ! -f "$mtp_model" ]; then
    fail "missing MTP model $mtp_model"
fi
if [ "$mtp_mode" != "off" ] && [ "$tokens" -lt 2 ]; then
    fail "--tokens must be >= 2 when --mtp-mode is verify or both"
fi

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-aggregate-throughput.XXXXXX)"
else
    mkdir -p "$work_dir" || exit 2
fi

request_json="$work_dir/request.json"
cases_dir="$work_dir/cases"
summary_tsv="$work_dir/aggregate_throughput.tsv"
summary_json="$work_dir/aggregate_throughput.json"
mkdir -p "$cases_dir"

cleanup() {
    if [ -n "${server_pid:-}" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
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

load_case() {
    local case_dir="$1"
    local case_name="$2"
    local mode="$3"
    local ctx="$4"
    local slots="$5"
    local policy="$6"
    local port="$7"

    local server_log="$case_dir/server.log"
    local result_json="$case_dir/result.json"
    local server_max_requests
    server_max_requests=$((requests + 32))

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
    if [ "$mode" = "verify" ]; then
        cmd+=(
            --mtp-model "$mtp_model"
            --mtp-serving verify
            --mtp-top-k "$top_k"
            --mtp-gpu "$mtp_gpu"
            --mtp-reserve-mib "$reserve_mib"
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

    python3 - "$host" "$port" "$request_json" "$requests" "$slots" "$tokens" "$expected_hex" "$mode" "$result_json" <<'PY'
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
mode = sys.argv[8].strip()
out_path = sys.argv[9]

with open(request_json_path, "rb") as f:
    payload = f.read()

headers = {
    "Content-Type": "application/json",
    "Content-Length": str(len(payload)),
    "Connection": "close",
}

latencies_ms = []
stats = {
    "status_200": 0,
    "status_other": 0,
    "status_429": 0,
    "status_413": 0,
    "token_match": 0,
    "token_mismatch": 0,
    "prompt_token_total": 0,
    "generated_token_total": 0,
    "continuation_token_total": 0,
    "mtp_attempted": 0,
    "mtp_accepted": 0,
    "errors": 0,
}

next_request = [0]
next_lock = threading.Lock()
stats_lock = threading.Lock()

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

def worker():
    while True:
        with next_lock:
            if next_request[0] >= n_requests:
                return
            next_request[0] += 1
        t0 = time.perf_counter()
        status = -1
        body = b""
        try:
            conn = http.client.HTTPConnection(host, port, timeout=300)
            conn.request("POST", "/v100/selected-token", body=payload, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            body = resp.read()
            conn.close()
        except Exception:
            with stats_lock:
                stats["errors"] += 1
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        with stats_lock:
            latencies_ms.append(dt_ms)
        if status != 200:
            with stats_lock:
                stats["status_other"] += 1
                if status == 429:
                    stats["status_429"] += 1
                if status == 413:
                    stats["status_413"] += 1
            continue
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except Exception:
            with stats_lock:
                stats["errors"] += 1
            continue
        first_hex = ""
        try:
            toks = data.get("tokens", [])
            if toks:
                first_hex = str(toks[0].get("text_hex", "")).lower()
        except Exception:
            first_hex = ""
        generated = int(data.get("generated_tokens", 0))
        prompt_tokens = int(data.get("prompt_tokens", 0))
        mtp = data.get("mtp") if isinstance(data, dict) else None
        with stats_lock:
            stats["status_200"] += 1
            stats["prompt_token_total"] += prompt_tokens
            stats["generated_token_total"] += generated
            stats["continuation_token_total"] += max(0, generated - 1)
            if generated == expected_tokens and first_hex == expected_hex:
                stats["token_match"] += 1
            else:
                stats["token_mismatch"] += 1
            if isinstance(mtp, dict):
                if bool(mtp.get("attempted", False)):
                    stats["mtp_attempted"] += 1
                if bool(mtp.get("accepted", False)):
                    stats["mtp_accepted"] += 1

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
p50 = percentile(sorted_lat, 0.50)
p95 = percentile(sorted_lat, 0.95)
p99 = percentile(sorted_lat, 0.99)
avg = statistics.fmean(sorted_lat) if sorted_lat else 0.0
tok_s = (stats["generated_token_total"] / elapsed_s) if elapsed_s > 0 else 0.0
prompt_tok_s = (stats["prompt_token_total"] / elapsed_s) if elapsed_s > 0 else 0.0
continuation_tok_s = (stats["continuation_token_total"] / elapsed_s) if elapsed_s > 0 else 0.0

summary = {
    "schema": "ds4_v100_aggregate_throughput_case.v1",
    "host": host,
    "port": port,
    "mode": mode,
    "requests": n_requests,
    "concurrency": concurrency,
    "elapsed_s": elapsed_s,
    "status_200": stats["status_200"],
    "status_other": stats["status_other"],
    "status_429": stats["status_429"],
    "status_413": stats["status_413"],
    "errors": stats["errors"],
    "token_match": stats["token_match"],
    "token_mismatch": stats["token_mismatch"],
    "prompt_token_total": stats["prompt_token_total"],
    "generated_token_total": stats["generated_token_total"],
    "continuation_token_total": stats["continuation_token_total"],
    "mtpa_attempted": stats["mtp_attempted"],
    "mtpa_accepted": stats["mtp_accepted"],
    "latency_ms": {
        "avg": avg,
        "p50": p50,
        "p95": p95,
        "p99": p99,
    },
    "aggregate_tokens_per_second": tok_s,
    "aggregate_prompt_tokens_per_second": prompt_tok_s,
    "aggregate_continuation_tokens_per_second": continuation_tok_s,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")

ok = (
    stats["status_200"] == n_requests
    and stats["status_other"] == 0
    and stats["errors"] == 0
    and stats["token_mismatch"] == 0
)
if mode == "verify":
    ok = ok and (stats["mtp_attempted"] == n_requests) and (stats["mtp_accepted"] == n_requests)
sys.exit(0 if ok else 1)
PY

    local py_rc=$?
    if [ -n "$server_pid" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
        server_pid=""
    fi
    return "$py_rc"
}

expected_lower="$(printf '%s' "$expected_hex" | tr 'A-F' 'a-f')"
printf 'schema\tds4_v100_aggregate_throughput.v1\n' >"$summary_tsv"
printf 'model\t%s\n' "$model" >>"$summary_tsv"
printf 'pack_index\t%s\n' "$pack_index" >>"$summary_tsv"
printf 'expected_token_hex\t%s\n' "$expected_lower" >>"$summary_tsv"
printf 'requests_per_case\t%s\n' "$requests" >>"$summary_tsv"
printf '\nctx\tslots\tpolicy\tmode\tstatus_200\tstatus_other\terrors\ttoken_match\ttoken_mismatch\tmtp_attempted\tmtp_accepted\tlatency_avg_ms\tlatency_p50_ms\tlatency_p95_ms\tlatency_p99_ms\telapsed_s\taggregate_prompt_tokens_per_second\taggregate_tokens_per_second\taggregate_continuation_tokens_per_second\n' >>"$summary_tsv"

mode_list=("off")
if [ "$mtp_mode" = "verify" ]; then
    mode_list=("verify")
elif [ "$mtp_mode" = "both" ]; then
    mode_list=("off" "verify")
fi

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
            for mode in "${mode_list[@]}"; do
                case_index=$((case_index + 1))
                port=$((port_base + case_index - 1))
                case_name="case_${case_index}_ctx${ctx}_s${slots}_${policy}_${mode}"
                case_dir="$cases_dir/$case_name"
                mkdir -p "$case_dir"
                if ! load_case "$case_dir" "$case_name" "$mode" "$ctx" "$slots" "$policy" "$port"; then
                    if [ -f "$case_dir/result.json" ]; then
                        cat "$case_dir/result.json" >&2
                    fi
                    if [ -f "$case_dir/server.log" ]; then
                        cat "$case_dir/server.log" >&2
                    fi
                    fail "load case failed: $case_name"
                fi
                case_paths+=("$case_dir/result.json")
                row="$(python3 - "$case_dir/result.json" "$ctx" "$slots" "$policy" "$mode" <<'PY'
import json
import sys
rpath, ctx, slots, policy, mode = sys.argv[1:]
with open(rpath, "r", encoding="utf-8") as f:
    d = json.load(f)
lat = d.get("latency_ms", {})
print("\t".join([
    ctx,
    slots,
    policy,
    mode,
    str(d.get("status_200", 0)),
    str(d.get("status_other", 0)),
    str(d.get("errors", 0)),
    str(d.get("token_match", 0)),
    str(d.get("token_mismatch", 0)),
    str(d.get("mtpa_attempted", 0)),
    str(d.get("mtpa_accepted", 0)),
    f"{float(lat.get('avg', 0.0)):.3f}",
    f"{float(lat.get('p50', 0.0)):.3f}",
    f"{float(lat.get('p95', 0.0)):.3f}",
    f"{float(lat.get('p99', 0.0)):.3f}",
    f"{float(d.get('elapsed_s', 0.0)):.6f}",
    f"{float(d.get('aggregate_prompt_tokens_per_second', 0.0)):.6f}",
    f"{float(d.get('aggregate_tokens_per_second', 0.0)):.6f}",
    f"{float(d.get('aggregate_continuation_tokens_per_second', 0.0)):.6f}",
]))
PY
)"
                printf '%s\n' "$row" >>"$summary_tsv"
                echo "ds4-v100-aggregate-throughput: PASS $case_name"
            done
        done
    done
done

python3 - "$summary_json" "${case_paths[@]}" <<'PY'
import json
import sys

out = sys.argv[1]
paths = sys.argv[2:]
rows = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        rows.append(json.load(f))
summary = {
    "schema": "ds4_v100_aggregate_throughput.v1",
    "cases": rows,
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, sort_keys=True)
    f.write("\n")
PY

cat "$summary_tsv"
echo "ds4-v100-aggregate-throughput: PASS cases=$case_index report=$summary_tsv json=$summary_json"
