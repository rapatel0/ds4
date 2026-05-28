#!/usr/bin/env bash
set -euo pipefail

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
appliance_dir=""
pack_index="docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
ctx="1048576"
slots="4"
active_microbatch="4"
microbatch_wait_us="auto"
queue_policy="sequential"
tokens="16"
requests="4"
warmup_requests="1"
host="127.0.0.1"
port="18420"
async_pipeline_mode="auto"
async_handoff="0"
async_event_handoff="${DS4_V100_ASYNC_EVENT_HANDOFF:-auto}"
sample_ms="500"
log_dir=""
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
require_gpus="8"
reserve_mib="4096"
replay_bin="${DS4_V100_BIN:-./tools/ds4-v100-replay}"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-appliance-soak.sh --log-dir DIR [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          MTP sidecar GGUF path for launcher validation
  --appliance-dir DIR       prepacked V100 appliance directory
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx N                   KV context tokens, default 1048576
  --slots N                 configured slots, default 4
  --active-microbatch N     active decode slots, default slots
  --microbatch-wait-us N    max request coalescing wait, default auto
  --queue-policy MODE       sequential or reject-busy, default sequential
  --tokens N                generated tokens per request, default 16
  --requests N              timed requests, default 4
  --warmup-requests N       untimed warmup requests, default 1
  --host ADDR               bind/probe address, default 127.0.0.1
  --port N                  server port, default 18420
  --async-pipeline-mode M   off, auto, per-step, persistent, or mailbox, default auto
  --async-handoff           queue HC peer handoff copies on the destination stream
  --async-event-handoff M   auto, 0, or 1, default auto
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
        --appliance-dir) appliance_dir="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --pack-index|--index) pack_index="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --prompt-file) prompt_file="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --expected-token-hex) expected_hex="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --ctx) ctx="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --slots) slots="$(need_value "$1" "${2:-}")"; active_microbatch="$slots"; shift 2 ;;
        --active-microbatch) active_microbatch="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --microbatch-wait-us) microbatch_wait_us="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --queue-policy) queue_policy="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --tokens) tokens="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --requests) requests="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --warmup-requests) warmup_requests="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --host) host="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --port) port="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --async-pipeline-mode) async_pipeline_mode="$(need_value "$1" "${2:-}")"; shift 2 ;;
        --async-handoff) async_handoff="1"; shift ;;
        --async-event-handoff) async_event_handoff="$(need_value "$1" "${2:-}")"; shift 2 ;;
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
[ -x ./tools/ds4-v100-run-pp-appliance.sh ] || fail "missing ./tools/ds4-v100-run-pp-appliance.sh"
[ -x ./tools/ds4-v100-replay ] || fail "missing ./tools/ds4-v100-replay"
[ -f "$model" ] || fail "missing model $model"
if [ -n "$appliance_dir" ]; then
    [ -d "$appliance_dir" ] || fail "missing appliance directory $appliance_dir"
    [ -f "$appliance_dir/pack-index.tsv" ] || fail "missing appliance pack index $appliance_dir/pack-index.tsv"
    [ -f "$appliance_dir/turbomind-pack-index.tsv" ] || fail "missing appliance TurboMind index $appliance_dir/turbomind-pack-index.tsv"
    for gpu in 0 1 2 3 4 5 6 7; do
        [ -f "$appliance_dir/gpu${gpu}.weights" ] || fail "missing appliance shard $appliance_dir/gpu${gpu}.weights"
    done
else
    [ -f "$pack_index" ] || fail "missing pack index $pack_index"
fi
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"
[ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
for v in "$ctx" "$slots" "$active_microbatch" "$tokens" "$requests" "$warmup_requests" "$port" "$sample_ms" "$require_gpus" "$reserve_mib"; do
    is_uint "$v" || fail "numeric option expected, got $v"
done
if [ "$microbatch_wait_us" != "auto" ]; then
    is_uint "$microbatch_wait_us" || fail "--microbatch-wait-us must be auto or an integer"
    [ "$microbatch_wait_us" -le 1000000 ] || fail "--microbatch-wait-us must be <= 1000000"
fi
[ "$slots" -ge 1 ] && [ "$slots" -le 256 ] || fail "--slots must be in [1,256]"
[ "$active_microbatch" -ge 1 ] && [ "$active_microbatch" -le "$slots" ] || fail "--active-microbatch must be in [1,slots]"
case "$queue_policy" in sequential|reject-busy) ;; *) fail "--queue-policy must be sequential or reject-busy" ;; esac
case "$async_pipeline_mode" in
    off|auto|per-step|per_step|persistent|mailbox|mbox) ;;
    *) fail "--async-pipeline-mode must be off, auto, per-step, persistent, or mailbox" ;;
esac
case "$async_handoff" in 0|1) ;; *) fail "--async-handoff must be 0 or 1" ;; esac
case "$async_event_handoff" in
    auto|0|1|false|true|off|on) ;;
    *) fail "--async-event-handoff must be auto, 0, or 1" ;;
esac

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

awk -v tokens="$tokens" '
BEGIN {
    printf "{\"prompt\":\""
}
{
    if (NR > 1) printf "\\n"
    for (i = 1; i <= length($0); i++) {
        c = substr($0, i, 1)
        if (c == "\\") {
            printf "\\\\"
        } else if (c == "\"") {
            printf "\\\""
        } else if (c == "\t") {
            printf "\\t"
        } else {
            printf "%s", c
        }
    }
}
END {
    printf "\",\"tokens\":%s}\n", tokens
}
' "$prompt_file" >"$request_json"

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
    sample_s="$(awk -v ms="$sample_ms" 'BEGIN { s = ms / 1000.0; if (s < 0.1) s = 0.1; printf "%.3f", s }')"
    (
        while :; do
            nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
                --format=csv,noheader,nounits >>"$gpu_csv" 2>>"$gpu_err" || true
            sleep "$sample_s"
        done
    ) &
    gpu_pid="$!"
fi

(
    export DS4_V100_BIN="$replay_bin"
    export DS4_V100_MODEL="$model"
    export DS4_V100_MTP_MODEL="$mtp_model"
    export DS4_V100_APPLIANCE_DIR="$appliance_dir"
    export DS4_V100_PACK_INDEX="$pack_index"
    export DS4_V100_CTX="$ctx"
    export DS4_V100_SLOTS="$slots"
    export DS4_V100_ACTIVE_MICROBATCH="$active_microbatch"
    export DS4_V100_MICROBATCH_WAIT_US="$microbatch_wait_us"
    export DS4_V100_QUEUE_POLICY="$queue_policy"
    export DS4_V100_TOKENS="$tokens"
    export DS4_V100_ASYNC_PIPELINE_MODE="$async_pipeline_mode"
    export DS4_V100_ASYNC_HANDOFF="$async_handoff"
    export DS4_V100_ASYNC_EVENT_HANDOFF="$async_event_handoff"
    export DS4_V100_HOST="$host"
    export DS4_V100_PORT="$port"
    export DS4_V100_CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
    export DS4_V100_REQUIRE_GPUS="$require_gpus"
    export DS4_V100_RESERVE_MIB="$reserve_mib"
    export DS4_V100_MAX_REQUESTS=$((requests + warmup_requests + 64))
    export DS4_V100_LOG_DIR="$log_dir/runtime"
    export DS4_V100_MTP_SERVING=off
    exec ./tools/ds4-v100-run-pp-appliance.sh
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

http_body() {
    sed -n '/^\r\{0,1\}$/,$p' "$1" | sed '1{/^\r\{0,1\}$/d;}' >"$2"
}

now_s() {
    perl -MTime::HiRes=time -e 'printf "%.6f\n", time'
}

http_get() {
    local path="$1"
    local out_http="$2"
    local out_body="$3"
    if ! exec 3<>"/dev/tcp/$host/$port"; then
        echo "ds4-v100-appliance-soak: GET $path failed" >&2
        cat "$server_log" >&2
        return 1
    fi
    {
        printf 'GET %s HTTP/1.1\r\n' "$path"
        printf 'Host: %s:%s\r\n' "$host" "$port"
        printf 'Connection: close\r\n'
        printf '\r\n'
    } >&3
    cat <&3 >"$out_http"
    exec 3<&-
    exec 3>&-
    local status_line
    status_line="$(sed -n '1p' "$out_http")"
    case "$status_line" in
        *" 200 "*) ;;
        *)
            echo "ds4-v100-appliance-soak: GET $path non-200 response: $status_line" >&2
            cat "$out_http" >&2
            return 1
            ;;
    esac
    http_body "$out_http" "$out_body"
}

body_len="$(wc -c <"$request_json" | tr -d ' ')"
expected_lower="$(printf '%s' "$expected_hex" | tr 'A-F' 'a-f')"
health_http="$log_dir/health.http"
status_before_http="$log_dir/status_before.http"
status_after_http="$log_dir/status_after.http"
metrics_before_http="$log_dir/metrics_before.http"
metrics_after_http="$log_dir/metrics_after.http"

http_get "/health" "$health_http" "$health_json" || fail "health request failed"
grep -q '"status":"ok"' "$health_json" || fail "bad /health response"
http_get "/v100/status" "$status_before_http" "$status_before" || fail "status request failed"
http_get "/metrics" "$metrics_before_http" "$metrics_before" || fail "metrics request failed"
async_pipeline_decode_before="$(sed -n 's/.*"async_pipeline_decode":\([^,}]*\).*/\1/p' "$status_before" | sed -n '1p')"
require_async_response=0
case "${async_pipeline_decode_before:-false}" in
    true) require_async_response=1 ;;
esac

post_request() {
    local request_id="$1"
    local response_prefix="${2:-response}"
    local require_async="${3:-1}"
    local response_http_i="$log_dir/${response_prefix}_${request_id}.http"
    local response_json_i="$log_dir/${response_prefix}_${request_id}.json"
    local response_row_i="$log_dir/${response_prefix}_${request_id}.row.json"
    local start_s end_s elapsed_ms status_line got generated prompt_tokens
    local prompt_replay_ms prompt_tps continuation_ms continuation_tps generated_tps

    start_s="$(now_s)"
    if ! exec 3<>"/dev/tcp/$host/$port"; then
        echo "ds4-v100-appliance-soak: request $request_id failed" >&2
        cat "$server_log" >&2
        return 1
    fi
    {
        printf 'POST /v100/selected-token HTTP/1.1\r\n'
        printf 'Host: %s:%s\r\n' "$host" "$port"
        printf 'Content-Type: application/json\r\n'
        printf 'Content-Length: %s\r\n' "$body_len"
        printf 'Connection: close\r\n'
        printf '\r\n'
        cat "$request_json"
    } >&3
    cat <&3 >"$response_http_i"
    exec 3<&-
    exec 3>&-
    end_s="$(now_s)"
    elapsed_ms="$(awk -v a="$start_s" -v b="$end_s" 'BEGIN { printf "%.3f", (b - a) * 1000.0 }')"

    status_line="$(sed -n '1p' "$response_http_i")"
    case "$status_line" in
        *" 200 "*) ;;
        *)
            echo "ds4-v100-appliance-soak: request $request_id non-200 response: $status_line" >&2
            cat "$response_http_i" >&2
            return 1
            ;;
    esac
    http_body "$response_http_i" "$response_json_i"
    got="$(grep -o '"text_hex":"[^"]*"' "$response_json_i" | sed -n '1{s/^"text_hex":"//;s/"$//;p;}')"
    generated="$(sed -n 's/.*"generated_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    prompt_tokens="$(sed -n 's/.*"prompt_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    prompt_replay_ms="$(sed -n 's/.*"prompt_replay":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    prompt_tps="$(sed -n 's/.*"prompt_tokens_per_second":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    continuation_ms="$(sed -n 's/.*"continuation_decode":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    continuation_tps="$(sed -n 's/.*"continuation_tokens_per_second":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    generated_tps="$(sed -n 's/.*"generated_tokens_per_second":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    if [ "$got" != "$expected_lower" ]; then
        echo "ds4-v100-appliance-soak: request $request_id expected $expected_hex, got ${got:-none}" >&2
        cat "$response_json_i" >&2
        return 1
    fi
    if ! is_uint "${generated:-}"; then
        echo "ds4-v100-appliance-soak: request $request_id missing generated_tokens" >&2
        cat "$response_json_i" >&2
        return 1
    fi
    if ! is_uint "${prompt_tokens:-}"; then
        prompt_tokens=0
    fi
    if [ "$require_async" -eq 1 ] && ! grep -q '"async_pipeline":' "$response_json_i"; then
        echo "ds4-v100-appliance-soak: request $request_id missing async_pipeline timing" >&2
        cat "$response_json_i" >&2
        return 1
    fi
    printf '{"index":%s,"status":200,"elapsed_ms":%s,"first_hex":"%s","prompt_tokens":%s,"prompt_replay_ms":%s,"prompt_tokens_per_second":%s,"generated_tokens":%s,"generated_tokens_per_second":%s,"continuation_ms":%s,"continuation_tokens_per_second":%s}\n' \
        "$request_id" "$elapsed_ms" "$got" "$prompt_tokens" "${prompt_replay_ms:-0}" \
        "${prompt_tps:-0}" "$generated" "${generated_tps:-0}" "${continuation_ms:-0}" \
        "${continuation_tps:-0}" >"$response_row_i"
}

for warmup_id in $(seq 1 "$warmup_requests"); do
    post_request "$warmup_id" "warmup" 0 || fail "warmup request $warmup_id failed"
done

request_start_s="$(now_s)"
pids=""
for request_id in $(seq 1 "$requests"); do
    post_request "$request_id" "response" "$require_async_response" &
    pids="$pids $!"
done
failed=0
for pid in $pids; do
    if ! wait "$pid"; then
        failed=1
    fi
done
request_end_s="$(now_s)"
[ "$failed" -eq 0 ] || fail "one or more requests failed"

http_get "/v100/status" "$status_after_http" "$status_after" || fail "status after request failed"
http_get "/metrics" "$metrics_after_http" "$metrics_after" || fail "metrics after request failed"

elapsed_s="$(awk -v a="$request_start_s" -v b="$request_end_s" 'BEGIN { printf "%.6f", b - a }')"
generated_total="$(awk '
    match($0, /"generated_tokens":[0-9]+/) {
        v = substr($0, RSTART + 19, RLENGTH - 19)
        sum += v
    }
    END { printf "%d", sum }
' "$log_dir"/response_*.row.json)"
continuation_total="$(awk '
    match($0, /"generated_tokens":[0-9]+/) {
        v = substr($0, RSTART + 19, RLENGTH - 19)
        if (v > 1) sum += v - 1
    }
    END { printf "%d", sum }
' "$log_dir"/response_*.row.json)"
prompt_total="$(awk '
    match($0, /"prompt_tokens":[0-9]+/) {
        v = substr($0, RSTART + 16, RLENGTH - 16)
        sum += v
    }
    END { printf "%d", sum }
' "$log_dir"/response_*.row.json)"
aggregate_generated_tps="$(awk -v toks="$generated_total" -v elapsed="$elapsed_s" 'BEGIN { v = 0.0; if (elapsed > 0) v = toks / elapsed; printf "%.6f", v }')"
aggregate_continuation_tps="$(awk -v toks="$continuation_total" -v elapsed="$elapsed_s" 'BEGIN { v = 0.0; if (elapsed > 0) v = toks / elapsed; printf "%.6f", v }')"
aggregate_prompt_tps="$(awk -v toks="$prompt_total" -v elapsed="$elapsed_s" 'BEGIN { v = 0.0; if (elapsed > 0) v = toks / elapsed; printf "%.6f", v }')"
latency_ms_avg="$(awk '
    match($0, /"elapsed_ms":[0-9.]+/) {
        v = substr($0, RSTART + 13, RLENGTH - 13)
        sum += v
        n++
    }
    END { printf "%.3f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
prompt_replay_ms_avg="$(awk '
    match($0, /"prompt_replay_ms":[0-9.]+/) {
        v = substr($0, RSTART + 19, RLENGTH - 19)
        sum += v
        n++
    }
    END { printf "%.3f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
continuation_ms_avg="$(awk '
    match($0, /"continuation_ms":[0-9.]+/) {
        v = substr($0, RSTART + 18, RLENGTH - 18)
        sum += v
        n++
    }
    END { printf "%.3f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
prompt_response_tps_avg="$(awk '
    match($0, /"prompt_tokens_per_second":[0-9.]+/) {
        v = substr($0, RSTART + 27, RLENGTH - 27)
        sum += v
        n++
    }
    END { printf "%.6f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
continuation_response_tps_avg="$(awk '
    match($0, /"continuation_tokens_per_second":[0-9.]+/) {
        v = substr($0, RSTART + 33, RLENGTH - 33)
        sum += v
        n++
    }
    END { printf "%.6f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
generated_response_tps_avg="$(awk '
    match($0, /"generated_tokens_per_second":[0-9.]+/) {
        v = substr($0, RSTART + 30, RLENGTH - 30)
        sum += v
        n++
    }
    END { printf "%.6f", n ? sum / n : 0.0 }
' "$log_dir"/response_*.row.json)"
async_pipeline_mode="$(sed -n 's/.*"async_pipeline_mode":"\([^"]*\)".*/\1/p' "$status_before" | sed -n '1p')"
async_pipeline_decode="$(sed -n 's/.*"async_pipeline_decode":\([^,}]*\).*/\1/p' "$status_before" | sed -n '1p')"
async_handoff_status="$(sed -n 's/.*"async_handoff":\([^,}]*\).*/\1/p' "$status_before" | sed -n '1p')"
async_event_handoff_status="$(sed -n 's/.*"async_event_handoff":\([^,}]*\).*/\1/p' "$status_before" | sed -n '1p')"

{
    printf '['
    sep=""
    for request_id in $(seq 1 "$requests"); do
        row="$log_dir/response_${request_id}.row.json"
        [ -f "$row" ] || fail "missing response row $row"
        printf '%s' "$sep"
        cat "$row"
        sep=","
    done
    printf ']\n'
} >"$responses_json"

printf '{"aggregate_continuation_tokens_per_second":%s,"aggregate_generated_tokens_per_second":%s,"aggregate_prompt_tokens_per_second":%s,"async_event_handoff":%s,"async_handoff":%s,"async_pipeline_decode":%s,"async_pipeline_mode":"%s","continuation_decode_ms_avg":%s,"continuation_response_tokens_per_second_avg":%s,"continuation_tokens":%s,"elapsed_s":%s,"errors":0,"generated_response_tokens_per_second_avg":%s,"generated_tokens":%s,"latency_ms_avg":%s,"prefill_prompt_replay_ms_avg":%s,"prompt_response_tokens_per_second_avg":%s,"prompt_tokens":%s,"requests":%s,"schema":"ds4_v100_appliance_soak.v1","status_200":%s,"token_match":%s,"warmup_requests":%s}\n' \
    "$aggregate_continuation_tps" \
    "$aggregate_generated_tps" \
    "$aggregate_prompt_tps" \
    "${async_event_handoff_status:-false}" \
    "${async_handoff_status:-false}" \
    "${async_pipeline_decode:-false}" \
    "${async_pipeline_mode:-unknown}" \
    "$continuation_ms_avg" \
    "$continuation_response_tps_avg" \
    "$continuation_total" \
    "$elapsed_s" \
    "$generated_response_tps_avg" \
    "$generated_total" \
    "$latency_ms_avg" \
    "$prompt_replay_ms_avg" \
    "$prompt_response_tps_avg" \
    "$prompt_total" \
    "$requests" \
    "$requests" \
    "$requests" \
    "$warmup_requests" >"$summary_json"

if [ -n "$gpu_pid" ]; then
    kill "$gpu_pid" >/dev/null 2>&1 || true
    wait "$gpu_pid" >/dev/null 2>&1 || true
    gpu_pid=""
fi

cat "$summary_json"
