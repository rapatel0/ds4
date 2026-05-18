#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
pack_index=""
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
host="127.0.0.1"
port="18080"
tokens="1"
requests="2"
log_dir=""

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-appliance-smoke.sh --pack-index FILE [options]

Options:
  --model FILE              source-layout GGUF model
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --tokens N                generated tokens to request, default 1
  --requests N              HTTP requests to send after one upload, default 2
  --host ADDR               bind/probe address, default 127.0.0.1
  --port N                  bind/probe port, default 18080
  --log-dir DIR             write server/request/response logs
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --model requires a value" >&2; exit 2; }
            model="$2"
            shift 2
            ;;
        --pack-index|--index)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --pack-index requires a value" >&2; exit 2; }
            pack_index="$2"
            shift 2
            ;;
        --prompt-file)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --prompt-file requires a value" >&2; exit 2; }
            prompt_file="$2"
            shift 2
            ;;
        --expected-token-hex)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --expected-token-hex requires a value" >&2; exit 2; }
            expected_hex="$2"
            shift 2
            ;;
        --tokens)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --tokens requires a value" >&2; exit 2; }
            tokens="$2"
            shift 2
            ;;
        --requests)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --requests requires a value" >&2; exit 2; }
            requests="$2"
            shift 2
            ;;
        --host)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --host requires a value" >&2; exit 2; }
            host="$2"
            shift 2
            ;;
        --port)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --port requires a value" >&2; exit 2; }
            port="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || { echo "ds4-v100-appliance-smoke: --log-dir requires a value" >&2; exit 2; }
            log_dir="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ds4-v100-appliance-smoke: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$pack_index" ]; then
    usage >&2
    exit 2
fi
if [ ! -x ./tools/ds4-v100-replay ]; then
    echo "ds4-v100-appliance-smoke: missing executable ./tools/ds4-v100-replay" >&2
    exit 1
fi
if [ ! -f "$prompt_file" ]; then
    echo "ds4-v100-appliance-smoke: missing prompt file $prompt_file" >&2
    exit 1
fi
case "$requests" in
    ''|0|*[!0-9]*)
        echo "ds4-v100-appliance-smoke: --requests must be a positive integer" >&2
        exit 2
        ;;
esac

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-appliance-smoke.XXXXXX)"
else
    mkdir -p "$work_dir"
fi
server_log="$work_dir/appliance_server.log"
request_json="$work_dir/appliance_request.json"
response_http="$work_dir/appliance_response.http"
response_json="$work_dir/appliance_response.json"

cleanup() {
    if [ "${server_pid:-}" ]; then
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    if [ -z "$log_dir" ]; then
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

./tools/ds4-v100-replay \
    --serve \
    --model "$model" \
    --index "$pack_index" \
    --host "$host" \
    --port "$port" \
    --tokens "$tokens" \
    --max-requests "$requests" \
    >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 420); do
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
        echo "ds4-v100-appliance-smoke: server exited before listening" >&2
        cat "$server_log" >&2
        exit 1
    fi
    if grep -q "serving http://" "$server_log"; then
        break
    fi
    sleep 1
done
if ! grep -q "serving http://" "$server_log"; then
    echo "ds4-v100-appliance-smoke: server did not start listening in time" >&2
    cat "$server_log" >&2
    exit 1
fi

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
        echo "ds4-v100-appliance-smoke: request $request_id failed" >&2
        cat "$server_log" >&2
        exit 1
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
        *)
            echo "ds4-v100-appliance-smoke: request $request_id non-200 response: $status_line" >&2
            cat "$response_http_i" >&2
            exit 1
            ;;
    esac
    sed -n '/^\r\{0,1\}$/,$p' "$response_http_i" | sed '1{/^\r\{0,1\}$/d;}' >"$response_json_i"
    got="$(sed -n 's/.*"text_hex":"\([^"]*\)".*/\1/p' "$response_json_i" | sed -n '1p')"
    if [ "$got" != "$expected_lower" ]; then
        echo "ds4-v100-appliance-smoke: request $request_id expected $expected_hex, got ${got:-none}" >&2
        cat "$response_json_i" >&2
        exit 1
    fi
    last_prompt_tokens="$(sed -n 's/.*"prompt_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_generated_tokens="$(sed -n 's/.*"generated_tokens":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_first_token="$(sed -n 's/.*"tokens":\[{"id":\([0-9][0-9]*\).*/\1/p' "$response_json_i" | sed -n '1p')"
    last_hex="$got"
    cp "$response_http_i" "$response_http"
    cp "$response_json_i" "$response_json"
done
wait "$server_pid"
server_pid=""
echo "ds4-v100-appliance-smoke: requests=$requests prompt_tokens=${last_prompt_tokens:-unknown} generated_tokens=${last_generated_tokens:-unknown} first_token=${last_first_token:-unknown} first_hex=$last_hex ok"
