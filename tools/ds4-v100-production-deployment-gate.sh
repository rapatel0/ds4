#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model="/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
pack_index="docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
host="127.0.0.1"
port="18082"
ctx="1048576"
tokens="2"
requests="1"
reserve_mib="4096"
require_gpus="8"
visible_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
log_dir=""

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-production-deployment-gate.sh --pack-index FILE [options]

Options:
  --model FILE              source-layout GGUF model
  --mtp-model FILE          MTP sidecar GGUF model used by the readiness gate
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx N                   KV context tokens, default 1048576
  --tokens N                generated tokens to request, default 2
  --requests N              generation requests to send, default 1
  --host ADDR               bind/probe address, default 127.0.0.1
  --port N                  bind/probe port, default 18082
  --reserve-mib N           minimum pre-start free memory per GPU, default 4096
  --require-gpus N          required visible GPUs, default 8
  --visible-devices LIST    CUDA_VISIBLE_DEVICES for the launched service
  --log-dir DIR             write env, server, request, and response logs
  --help                    show this help

This gate validates the production launcher path, health/status/metrics, and a
bounded generation request without changing the model runtime.
USAGE
}

fail() {
    echo "ds4-v100-production-deployment-gate: $*" >&2
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
        --reserve-mib)
            [ "$#" -ge 2 ] || fail "--reserve-mib requires a value"
            reserve_mib="$2"
            shift 2
            ;;
        --require-gpus)
            [ "$#" -ge 2 ] || fail "--require-gpus requires a value"
            require_gpus="$2"
            shift 2
            ;;
        --visible-devices)
            [ "$#" -ge 2 ] || fail "--visible-devices requires a value"
            visible_devices="$2"
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

case "$requests" in ''|0|*[!0-9]*) fail "--requests must be a positive integer" ;; esac
case "$tokens" in ''|0|*[!0-9]*) fail "--tokens must be a positive integer" ;; esac
case "$ctx" in ''|0|*[!0-9]*) fail "--ctx must be a positive integer" ;; esac
case "$port" in ''|0|*[!0-9]*) fail "--port must be a positive integer" ;; esac

[ -x ./tools/ds4-v100-run-appliance.sh ] || fail "missing executable ./tools/ds4-v100-run-appliance.sh"
[ -x ./tools/ds4-v100-replay ] || fail "missing executable ./tools/ds4-v100-replay"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-production-deployment.XXXXXX)"
else
    mkdir -p "$work_dir" || exit 2
fi

env_path="$work_dir/ds4-v100-appliance.env"
launcher_check_log="$work_dir/launcher_check.log"
server_log="$work_dir/appliance_server.log"
request_json="$work_dir/appliance_request.json"
health_http="$work_dir/appliance_health.http"
health_json="$work_dir/appliance_health.json"
status_http="$work_dir/appliance_status.http"
status_json="$work_dir/appliance_status.json"
metrics_http="$work_dir/appliance_metrics.http"
metrics_txt="$work_dir/appliance_metrics.txt"
final_status_http="$work_dir/appliance_final_status.http"
final_status_json="$work_dir/appliance_final_status.json"

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

max_requests=$((requests + 4))
cat >"$env_path" <<EOF
DS4_V100_BIN=./tools/ds4-v100-replay
DS4_V100_MODEL=$model
DS4_V100_MTP_MODEL=$mtp_model
DS4_V100_PACK_INDEX=$pack_index
DS4_V100_CTX=$ctx
DS4_V100_SLOTS=1
DS4_V100_TOKENS=$tokens
DS4_V100_HOST=$host
DS4_V100_PORT=$port
DS4_V100_CUDA_VISIBLE_DEVICES=$visible_devices
DS4_V100_REQUIRE_GPUS=$require_gpus
DS4_V100_RESERVE_MIB=$reserve_mib
DS4_V100_MAX_REQUESTS=$max_requests
DS4_V100_LOG_DIR=$work_dir/runtime
DS4_V100_SERVE_MODE=base
DS4_V100_MTP_SERVING=off
EOF

if ! ./tools/ds4-v100-run-appliance.sh --env "$env_path" --check >"$launcher_check_log" 2>&1; then
    cat "$launcher_check_log" >&2
    fail "launcher config check failed"
fi

./tools/ds4-v100-run-appliance.sh --env "$env_path" >"$server_log" 2>&1 &
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
grep -q '"mode":"base_one_slot"' "$status_json" || fail "missing base_one_slot mode in status"
grep -q '"readiness_level":2' "$status_json" || fail "missing readiness_level=2 in status"
grep -q '"mtp_enabled":false' "$status_json" || fail "status should report mtp_enabled=false"
grep -q "\"ctx_tokens\":$ctx" "$status_json" || fail "status ctx_tokens mismatch"
grep -q '"max_tokens":64' "$status_json" || fail "status max_tokens mismatch"
grep -q '"slots":1' "$status_json" || fail "status missing slots=1 limit"
initial_served="$(sed -n 's/.*"served_requests":\([0-9][0-9]*\).*/\1/p' "$status_json" | sed -n '1p')"

http_get "/metrics" "$metrics_http" "$metrics_txt"
grep -q '^ds4_v100_readiness_level 2$' "$metrics_txt" || fail "metrics missing readiness level"
grep -q "^ds4_v100_ctx_tokens $ctx$" "$metrics_txt" || fail "metrics ctx_tokens mismatch"
grep -q '^ds4_v100_mtp_enabled 0$' "$metrics_txt" || fail "metrics should report mtp disabled"

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

for request_id in $(seq 1 "$requests"); do
    response_http_i="$work_dir/appliance_response_${request_id}.http"
    response_json_i="$work_dir/appliance_response_${request_id}.json"
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
    last_prompt_tokens="$(sed -n 's/.*"prompt_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_generated_tokens="$(sed -n 's/.*"generated_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_first_token="$(sed -n 's/.*"tokens":\[{"id":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    if [ "$last_generated_tokens" != "$tokens" ]; then
        cat "$response_json_i" >&2
        fail "request $request_id expected generated_tokens=$tokens, got ${last_generated_tokens:-unknown}"
    fi
    last_hex="$got"
    echo "ds4-v100-production-deployment-gate: request=$request_id prompt_tokens=${last_prompt_tokens:-unknown} generated_tokens=${last_generated_tokens:-unknown} first_token=${last_first_token:-unknown} first_hex=$last_hex ok"
done

http_get "/v100/status" "$final_status_http" "$final_status_json"
final_served="$(sed -n 's/.*"served_requests":\([0-9][0-9]*\).*/\1/p' "$final_status_json" | sed -n '1p')"
if [ -n "$initial_served" ] && [ -n "$final_served" ] && [ "$final_served" -le "$initial_served" ]; then
    fail "served_requests did not advance: initial=$initial_served final=$final_served"
fi

wait "$server_pid"
server_pid=""
echo "ds4-v100-production-deployment-gate: launcher=ok health=ok status=ok metrics=ok requests=$requests prompt_tokens=${last_prompt_tokens:-unknown} generated_tokens=${last_generated_tokens:-unknown} first_token=${last_first_token:-unknown} first_hex=$last_hex ok"
