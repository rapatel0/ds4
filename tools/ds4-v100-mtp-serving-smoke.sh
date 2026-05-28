#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
pack_index=""
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
host="127.0.0.1"
port="18083"
ctx="1048576"
tokens="2"
requests="1"
top_k="5"
mtp_gpu="7"
reserve_mib="4096"
mode="verify"
log_dir=""

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-mtp-serving-smoke.sh --pack-index FILE [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          DeepSeek-V4 Flash MTP sidecar GGUF
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx N                   KV context tokens, default 1048576
  --tokens N                generated tokens to request, default 2
  --requests N              HTTP requests to send after one upload, default 1
  --mode MODE               MTP serving mode: verify or commit, default verify
  --top-k N                 MTP draft candidates to report, default 5
  --mtp-gpu N               MTP sidecar GPU, default 7
  --reserve-mib N           required MTP free-memory reserve, default 4096
  --host ADDR               bind/probe address, default 127.0.0.1
  --port N                  bind/probe port, default 18083
  --log-dir DIR             write server/request/response logs
  --help                    show this help

The smoke starts tools/ds4-v100-replay with --mtp-serving MODE and validates
status, metrics, first-token bytes, and exact MTP top-1 acceptance. Commit mode
also validates that an accepted draft is counted as committed.
USAGE
}

fail() {
    echo "ds4-v100-mtp-serving-smoke: $*" >&2
    if [ -n "${server_pid:-}" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    exit 1
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
        --ctx)
            [ "$#" -ge 2 ] || fail "--ctx requires a value"
            ctx="$2"
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
        --mode)
            [ "$#" -ge 2 ] || fail "--mode requires a value"
            mode="$2"
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
        --host)
            [ "$#" -ge 2 ] || fail "--host requires a value"
            host="$2"
            shift 2
            ;;
        --port)
            [ "$#" -ge 2 ] || fail "--port requires a value"
            port="$2"
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
            echo "ds4-v100-mtp-serving-smoke: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$requests" in ''|0|*[!0-9]*) fail "--requests must be a positive integer" ;; esac
case "$tokens" in ''|0|1|*[!0-9]*) fail "--tokens must be an integer >= 2" ;; esac
case "$ctx" in ''|0|*[!0-9]*) fail "--ctx must be a positive integer" ;; esac
case "$top_k" in ''|0|1|*[!0-9]*) fail "--top-k must be an integer >= 2" ;; esac
case "$mtp_gpu" in ''|*[!0-9]*) fail "--mtp-gpu must be an integer" ;; esac
case "$reserve_mib" in ''|*[!0-9]*) fail "--reserve-mib must be an integer" ;; esac
case "$mode" in verify|commit) ;; *) fail "--mode must be verify or commit" ;; esac

[ -n "$pack_index" ] || { usage >&2; exit 2; }
[ -x ./tools/ds4-v100-replay ] || fail "missing executable ./tools/ds4-v100-replay"
[ -f "$model" ] || fail "missing model $model"
[ -f "$mtp_model" ] || fail "missing MTP model $mtp_model"
[ -f "$pack_index" ] || fail "missing pack index $pack_index"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"
[ "$top_k" -le 16 ] || fail "--top-k must be <= 16"

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-mtp-serving-smoke.XXXXXX)"
else
    mkdir -p "$work_dir" || exit 2
fi

server_log="$work_dir/mtp_serving_server.log"
request_json="$work_dir/mtp_serving_request.json"
health_http="$work_dir/mtp_serving_health.http"
health_json="$work_dir/mtp_serving_health.json"
status_http="$work_dir/mtp_serving_status.http"
status_json="$work_dir/mtp_serving_status.json"
metrics_http="$work_dir/mtp_serving_metrics.http"
metrics_txt="$work_dir/mtp_serving_metrics.txt"
final_status_http="$work_dir/mtp_serving_final_status.http"
final_status_json="$work_dir/mtp_serving_final_status.json"
final_metrics_http="$work_dir/mtp_serving_final_metrics.http"
final_metrics_txt="$work_dir/mtp_serving_final_metrics.txt"

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

server_max_requests=$((requests + 5))

DS4_LOCK_FILE="$work_dir/ds4.lock" ./tools/ds4-v100-replay \
    --serve \
    --model "$model" \
    --mtp-model "$mtp_model" \
    --index "$pack_index" \
    --ctx "$ctx" \
    --tokens "$tokens" \
    --host "$host" \
    --port "$port" \
    --max-requests "$server_max_requests" \
    --mtp-serving "$mode" \
    --mtp-top-k "$top_k" \
    --mtp-gpu "$mtp_gpu" \
    --mtp-reserve-mib "$reserve_mib" \
    >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 420); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        cat "$server_log" >&2
        fail "server exited before listening"
    fi
    if grep -q "serving http://" "$server_log"; then
        break
    fi
    sleep 1
done
if ! grep -q "serving http://" "$server_log"; then
    cat "$server_log" >&2
    fail "server did not start listening in time"
fi

http_body() {
    sed -n '/^\r\{0,1\}$/,$p' "$1" | sed '1{/^\r\{0,1\}$/d;}' >"$2"
}

http_get() {
    path="$1"
    out_http="$2"
    out_body="$3"
    if ! exec 3<>"/dev/tcp/$host/$port"; then
        cat "$server_log" >&2
        fail "GET $path failed"
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
    status_line="$(sed -n '1p' "$out_http")"
    case "$status_line" in
        *" 200 "*) ;;
        *) cat "$out_http" >&2; fail "GET $path non-200 response: $status_line" ;;
    esac
    http_body "$out_http" "$out_body"
}

http_get "/health" "$health_http" "$health_json"
grep -q '"status":"ok"' "$health_json" || fail "bad /health response"

http_get "/v100/status" "$status_http" "$status_json"
grep -q '"service":"ds4-v100-replay"' "$status_json" || fail "missing service in status"
if [ "$mode" = "commit" ]; then
    grep -q '"mode":"mtp_commit_one_slot"' "$status_json" || fail "missing mtp_commit_one_slot mode"
else
    grep -q '"mode":"mtp_verify_one_slot"' "$status_json" || fail "missing mtp_verify_one_slot mode"
fi
grep -q '"readiness_level":3' "$status_json" || fail "missing readiness_level=3"
grep -q '"mtp_enabled":true' "$status_json" || fail "status should report mtp_enabled=true"
grep -q '"speculative_serving":true' "$status_json" || fail "status should report speculative_serving=true"
grep -q "\"serving_mode\":\"$mode\"" "$status_json" || fail "status missing MTP $mode mode"
grep -q "\"top_k\":$top_k" "$status_json" || fail "status MTP top_k mismatch"

http_get "/metrics" "$metrics_http" "$metrics_txt"
grep -q '^ds4_readiness_level 3$' "$metrics_txt" || fail "metrics missing readiness level 3"
grep -q '^ds4_mtp_enabled 1$' "$metrics_txt" || fail "metrics should report mtp enabled"
grep -q '^ds4_mtp_drafts_total 0$' "$metrics_txt" || fail "initial MTP drafts should be 0"

awk -v tokens="$tokens" '
BEGIN {
    printf "{\"prompt\":\""
}
{
    if (NR > 1) printf "\\n"
    for (i = 1; i <= length($0); i++) {
        c = substr($0, i, 1)
        if (c == "\\") printf "\\\\"
        else if (c == "\"") printf "\\\""
        else if (c == "\t") printf "\\t"
        else printf "%s", c
    }
}
END {
    printf "\",\"tokens\":%s}\n", tokens
}
' "$prompt_file" >"$request_json"

body_len="$(wc -c <"$request_json" | tr -d ' ')"
expected_lower="$(printf '%s' "$expected_hex" | tr 'A-F' 'a-f')"
last_prompt_tokens=""
last_generated_tokens=""
last_first_token=""
last_hex=""
last_draft_ms=""

for request_id in $(seq 1 "$requests"); do
    response_http_i="$work_dir/mtp_serving_response_${request_id}.http"
    response_json_i="$work_dir/mtp_serving_response_${request_id}.json"
    if ! exec 3<>"/dev/tcp/$host/$port"; then
        cat "$server_log" >&2
        fail "request $request_id failed"
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

    status_line="$(sed -n '1p' "$response_http_i")"
    case "$status_line" in
        *" 200 "*) ;;
        *) cat "$response_http_i" >&2; fail "request $request_id non-200 response: $status_line" ;;
    esac
    http_body "$response_http_i" "$response_json_i"
    got="$(grep -o '"text_hex":"[^"]*"' "$response_json_i" | sed -n '1{s/^"text_hex":"//;s/"$//;p;}')"
    if [ "$got" != "$expected_lower" ]; then
        cat "$response_json_i" >&2
        fail "request $request_id expected $expected_hex, got ${got:-none}"
    fi
    grep -q '"mtp":{"enabled":true' "$response_json_i" || fail "response missing enabled MTP object"
    if [ "$mode" = "commit" ]; then
        grep -q '"commit_mode":true' "$response_json_i" || fail "response missing commit_mode=true"
        grep -q '"commit_applied":true' "$response_json_i" || fail "response missing commit_applied=true"
        grep -q '"commit_count":1' "$response_json_i" || fail "response commit_count mismatch"
    else
        grep -q '"commit_mode":false' "$response_json_i" || fail "response should report commit_mode=false"
        grep -q '"commit_applied":false' "$response_json_i" || fail "response should report commit_applied=false"
    fi
    grep -q '"attempted":true' "$response_json_i" || fail "response missing MTP attempted=true"
    grep -q '"accepted":true' "$response_json_i" || fail "response missing MTP accepted=true"
    grep -q '"attempts":1' "$response_json_i" || fail "response MTP attempt count mismatch"
    grep -q '"accepted_count":1' "$response_json_i" || fail "response MTP accepted_count mismatch"
    grep -q '"rejected_count":0' "$response_json_i" || fail "response MTP rejected_count mismatch"
    grep -q '"committed_token":926' "$response_json_i" || fail "response committed token mismatch"
    grep -q '"target_token":1' "$response_json_i" || fail "response target token mismatch"
    grep -q '"draft_token":1' "$response_json_i" || fail "response draft token mismatch"
    grep -q '"scratch_device_bytes":' "$response_json_i" || fail "response missing MTP scratch device bytes"
    grep -q '"scratch_host_bytes":' "$response_json_i" || fail "response missing MTP scratch host bytes"
    grep -q "\"forward_run_count\":$request_id" "$response_json_i" || fail "response MTP forward run counter mismatch"
    grep -q '"draft_topk":\[' "$response_json_i" || fail "response missing draft_topk"

    last_prompt_tokens="$(sed -n 's/.*"prompt_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_generated_tokens="$(sed -n 's/.*"generated_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_first_token="$(sed -n 's/.*"tokens":\[{"id":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_draft_ms="$(sed -n 's/.*"draft_ms":\([0-9.][0-9.]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    if [ "$last_generated_tokens" != "$tokens" ]; then
        cat "$response_json_i" >&2
        fail "request $request_id expected generated_tokens=$tokens, got ${last_generated_tokens:-unknown}"
    fi
    if ! awk -v v="${last_draft_ms:-0}" 'BEGIN { exit !((v + 0.0) > 0.0) }'; then
        cat "$response_json_i" >&2
        fail "request $request_id expected draft_ms > 0"
    fi
    last_hex="$got"
    echo "ds4-v100-mtp-serving-smoke: request=$request_id prompt_tokens=${last_prompt_tokens:-unknown} generated_tokens=${last_generated_tokens:-unknown} first_token=${last_first_token:-unknown} first_hex=$last_hex mtp_draft_ms=${last_draft_ms:-unknown} accepted=true ok"
done

http_get "/v100/status" "$final_status_http" "$final_status_json"
grep -q "\"drafts\":$requests" "$final_status_json" || fail "final status drafts counter mismatch"
grep -q "\"accepted\":$requests" "$final_status_json" || fail "final status accepted counter mismatch"
if [ "$mode" = "commit" ]; then
    grep -q "\"committed\":$requests" "$final_status_json" || fail "final status committed counter mismatch"
else
    grep -q '"committed":0' "$final_status_json" || fail "final status committed counter should be 0"
fi

http_get "/metrics" "$final_metrics_http" "$final_metrics_txt"
grep -q "^ds4_mtp_drafts_total $requests$" "$final_metrics_txt" || fail "final metrics drafts counter mismatch"
grep -q "^ds4_mtp_accepted_total $requests$" "$final_metrics_txt" || fail "final metrics accepted counter mismatch"
grep -q '^ds4_mtp_rejected_total 0$' "$final_metrics_txt" || fail "final metrics rejected counter mismatch"
if [ "$mode" = "commit" ]; then
    grep -q "^ds4_mtp_committed_total $requests$" "$final_metrics_txt" || fail "final metrics committed counter mismatch"
else
    grep -q '^ds4_mtp_committed_total 0$' "$final_metrics_txt" || fail "final metrics committed counter should be 0"
fi

wait "$server_pid"
server_pid=""
echo "ds4-v100-mtp-serving-smoke: mode=$mode health=ok status=ok metrics=ok requests=$requests prompt_tokens=${last_prompt_tokens:-unknown} generated_tokens=${last_generated_tokens:-unknown} first_token=${last_first_token:-unknown} first_hex=$last_hex mtp_accepted=$requests ok"
