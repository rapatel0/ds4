#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
pack_index="docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
planner_ctx="1048576"
plan_slots="8"
smoke_ctx="1048576"
smoke_slots="1"
active_microbatch="1"
queue_policy="reject-busy"
host="127.0.0.1"
port="18084"
reject_ctx="32"
reject_tokens="64"
requests="4"
log_dir=""

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-slot-context-envelope.sh --pack-index FILE [options]

Options:
  --model FILE                    source-layout GGUF model
  --pack-index FILE               V100 pack-index.tsv
  --prompt-file FILE              prompt file
  --expected-token-hex HEX         expected first response token bytes, default 3136
  --planner-ctx N                 context used by planner JSON matrix, default 1048576
  --plan-slots N                  configured slots used for planned envelope, default 8
  --smoke-slots N                 configured slots for appliance smoke, default 1
  --smoke-ctx N                   context for appliance smoke, default 1048576
  --active-microbatch N           active decode requests, default 1
  --queue-policy MODE             reject-busy or sequential, default reject-busy
  --host ADDR                     loopback host, default 127.0.0.1
  --port N                        loopback port, default 18084
  --reject-ctx N                  short-context rejection smoke, default 32
  --reject-tokens N               rejection smoke token count, default 64
  --requests N                    appliance requests per smoke, default 4
  --log-dir DIR                   write artifacts
  --help                          show this help

This script runs three checks:
  - planner envelope JSON/TSV for the configured envelope assumptions,
  - one-loopback appliance smoke in conservative single-slot mode,
  - one intentional context-overrun request and verifies HTTP 413 context_exceeded.
USAGE
}

fail() {
    echo "ds4-v100-slot-context-envelope: $*" >&2
    if [ -n "${replay_pid:-}" ]; then
        kill "$replay_pid" >/dev/null 2>&1 || true
        wait "$replay_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "${reject_pid:-}" ]; then
        kill "$reject_pid" >/dev/null 2>&1 || true
        wait "$reject_pid" >/dev/null 2>&1 || true
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
        --planner-ctx)
            [ "$#" -ge 2 ] || fail "--planner-ctx requires a value"
            planner_ctx="$2"
            shift 2
            ;;
        --plan-slots)
            [ "$#" -ge 2 ] || fail "--plan-slots requires a value"
            plan_slots="$2"
            shift 2
            ;;
        --smoke-slots)
            [ "$#" -ge 2 ] || fail "--smoke-slots requires a value"
            smoke_slots="$2"
            shift 2
            ;;
        --smoke-ctx)
            [ "$#" -ge 2 ] || fail "--smoke-ctx requires a value"
            smoke_ctx="$2"
            shift 2
            ;;
        --active-microbatch)
            [ "$#" -ge 2 ] || fail "--active-microbatch requires a value"
            active_microbatch="$2"
            shift 2
            ;;
        --queue-policy)
            [ "$#" -ge 2 ] || fail "--queue-policy requires a value"
            queue_policy="$2"
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
        --reject-ctx)
            [ "$#" -ge 2 ] || fail "--reject-ctx requires a value"
            reject_ctx="$2"
            shift 2
            ;;
        --reject-tokens)
            [ "$#" -ge 2 ] || fail "--reject-tokens requires a value"
            reject_tokens="$2"
            shift 2
            ;;
        --requests)
            [ "$#" -ge 2 ] || fail "--requests requires a value"
            requests="$2"
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
            echo "ds4-v100-slot-context-envelope: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$plan_slots" in ''|0|*[!0-9]*) fail "--plan-slots must be in [1,8]" ;; esac
case "$smoke_slots" in ''|0|*[!0-9]*) fail "--smoke-slots must be in [1,8]" ;; esac
case "$active_microbatch" in ''|0|*[!0-9]*) fail "--active-microbatch must be in [1,slots]" ;; esac
if [ "$active_microbatch" -gt "$smoke_slots" ]; then
    fail "--active-microbatch must be <= --smoke-slots"
fi
case "$planner_ctx" in ''|0|*[!0-9]*) fail "--planner-ctx must be a positive integer" ;; esac
case "$smoke_ctx" in ''|0|*[!0-9]*) fail "--smoke-ctx must be a positive integer" ;; esac
case "$reject_ctx" in ''|0|*[!0-9]*) fail "--reject-ctx must be a positive integer" ;; esac
case "$reject_tokens" in ''|0|*[!0-9]*) fail "--reject-tokens must be a positive integer" ;; esac
case "$requests" in ''|0|*[!0-9]*) fail "--requests must be a positive integer" ;; esac
case "$port" in ''|0|*[!0-9]*) fail "--port must be an integer" ;; esac

if [ -n "$log_dir" ]; then
    mkdir -p "$log_dir" || exit 2
fi

[ -x ./tools/ds4-v100-replay ] || fail "missing executable ./tools/ds4-v100-replay"
[ -x ./tools/ds4-v100-plan ] || fail "missing executable ./tools/ds4-v100-plan"
[ -x ./tools/ds4-v100-appliance-smoke.sh ] || fail "missing executable ./tools/ds4-v100-appliance-smoke.sh"
[ -f "$model" ] || fail "missing model $model"
[ -f "$pack_index" ] || fail "missing pack index $pack_index"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"

case "$queue_policy" in
    reject|busy)
        queue_policy="reject-busy"
        ;;
    queue)
        queue_policy="sequential"
        ;;
    reject-busy|sequential)
        ;;
    *)
        fail "--queue-policy must be reject-busy or sequential"
        ;;
esac

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-slot-context-envelope.XXXXXX)"
else
    work_dir="$(printf '%s' "$work_dir")"
fi

cleanup() {
    if [ -n "${replay_pid:-}" ]; then
        kill "$replay_pid" >/dev/null 2>&1 || true
        wait "$replay_pid" >/dev/null 2>&1 || true
    fi
    if [ -n "${reject_pid:-}" ]; then
        kill "$reject_pid" >/dev/null 2>&1 || true
        wait "$reject_pid" >/dev/null 2>&1 || true
    fi
    if [ -z "$log_dir" ]; then
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

plan_json="$work_dir/slot_context_envelope_plan.json"
plan_tsv="$work_dir/slot_context_envelope_plan.tsv"
plan_log="$work_dir/slot_context_envelope_plan.log"
smoke_log_dir="$work_dir/appliance_smoke"
rejection_log="$work_dir/rejection.log"
report="$work_dir/slot_context_envelope.report"

planner_args=(--json --ctx "$planner_ctx" --slots "$plan_slots" --gpus 8 --device-total-bytes 34359738368)
if ! ./tools/ds4-v100-plan "${planner_args[@]}" >"$plan_json" 2>"$plan_log"; then
    cat "$plan_log" >&2
    fail "planner envelope generation failed"
fi

if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 required for slot/context report rendering"
fi

python3 - "$plan_json" "$plan_tsv" <<'PY'
import json
import sys

in_json, out_tsv = sys.argv[1], sys.argv[2]
with open(in_json, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = data.get("target_matrix", [])
with open(out_tsv, "w", encoding="utf-8") as f:
    f.write("ctx_tokens\tslots\tfits\tadmitted_by_tier\tworst_gpu\tworst_total_bytes\n")
    for row in rows:
        f.write(f"{row.get('ctx_tokens')}\t{row.get('slots')}\t{row.get('fits')}\t{row.get('admitted_by_tier')}\t{row.get('worst_gpu')}\t{row.get('worst_total_bytes')}\n")
PY

run_smoke() {
    local smoke_port="$1"
    if ! ./tools/ds4-v100-appliance-smoke.sh \
        --model "$model" \
        --index "$pack_index" \
        --prompt-file "$prompt_file" \
        --ctx "$smoke_ctx" \
        --slots "$smoke_slots" \
        --active-microbatch "$active_microbatch" \
        --queue-policy "$queue_policy" \
        --tokens 1 \
        --requests "$requests" \
        --expected-token-hex "$expected_hex" \
        --host "$host" \
        --port "$smoke_port" \
        --log-dir "$smoke_log_dir"; then
        return 1
    fi
    return 0
}

if [ "$smoke_slots" -gt 8 ]; then
    fail "--smoke-slots must be <= 8"
fi

run_smoke "$port" || fail "appliance smoke failed"

rejection_port=$((port + 7))
rejection_server_log="$work_dir/reject_server.log"
rejection_request="$work_dir/reject_request.json"
rejection_response="$work_dir/reject_response.http"

python3 - "$prompt_file" "$reject_tokens" >"$rejection_request" <<'PY'
import json
import sys

prompt_path = sys.argv[1]
tokens = int(sys.argv[2])
with open(prompt_path, "r", encoding="utf-8", errors="ignore") as f:
    prompt = f.read()
print(json.dumps({"prompt": prompt, "tokens": tokens}))
PY

echo "ds4-v100-replay --serve --model $model --index $pack_index --ctx $reject_ctx --slots $smoke_slots --active-microbatch $active_microbatch --queue-policy $queue_policy --tokens 1 --host $host --port $rejection_port --max-requests 16 >$rejection_server_log 2>&1" >"$rejection_log"

DS4_LOCK_FILE="$work_dir/reject_ds4.lock" ./tools/ds4-v100-replay \
    --serve \
    --model "$model" \
    --index "$pack_index" \
    --ctx "$reject_ctx" \
    --slots "$smoke_slots" \
    --active-microbatch "$active_microbatch" \
    --queue-policy "$queue_policy" \
    --tokens 1 \
    --host "$host" \
    --port "$rejection_port" \
    --max-requests 16 \
    >"$rejection_server_log" 2>&1 &
reject_pid="$!"

for _ in $(seq 1 420); do
    if ! kill -0 "$reject_pid" >/dev/null 2>&1; then
        cat "$rejection_server_log" >&2
        fail "rejection server exited before listening"
    fi
    if grep -q "serving http://" "$rejection_server_log"; then
        break
    fi
    sleep 1
done
if ! grep -q "serving http://" "$rejection_server_log"; then
    cat "$rejection_server_log" >&2
    fail "rejection server did not start in time"
fi

if ! exec 3<>"/dev/tcp/$host/$rejection_port"; then
    cat "$rejection_server_log" >&2
    fail "could not connect to rejection server"
fi
{
    printf 'POST /v100/selected-token HTTP/1.1\r\n'
    printf 'Host: %s:%s\r\n' "$host" "$rejection_port"
    printf 'Content-Type: application/json\r\n'
    body_len="$(wc -c <"$rejection_request")"
    printf 'Content-Length: %s\r\n' "$body_len"
    printf 'Connection: close\r\n'
    printf '\r\n'
    cat "$rejection_request"
} >&3
cat <&3 >"$rejection_response"
exec 3<&-
exec 3>&-

rejection_status_line="$(sed -n '1p' "$rejection_response")"
if ! printf '%s\n' "$rejection_status_line" | grep -q '413'; then
    cat "$rejection_response" >&2
    fail "expected 413 context_exceeded for over-context request, got: $rejection_status_line"
fi
if ! grep -q '"error":"context_exceeded"' "$rejection_response"; then
    cat "$rejection_response" >&2
    fail "expected context_exceeded error body"
fi

cat >"$report" <<EOF
schema\tds4_v100_slot_context_envelope.v1
model\t$model
pack_index\t$pack_index
planner_ctx\t$planner_ctx
planner_slots\t$plan_slots
smoke_slots\t$smoke_slots
smoke_ctx\t$smoke_ctx
active_microbatch\t$active_microbatch
queue_policy\t$queue_policy
requests\t$requests
reject_ctx\t$reject_ctx
reject_tokens\t$reject_tokens
status\tPASS
EOF

echo "schema\tds4_v100_slot_context_envelope.v1" >"$work_dir/slot_context_envelope_artifacts.tsv"
cat "$plan_tsv" >>"$work_dir/slot_context_envelope_artifacts.tsv"

echo "ds4-v100-slot-context-envelope: PASS model=$model plan_slots=$plan_slots smoke_slots=$smoke_slots reject_ctx=$reject_ctx reject_tokens=$reject_tokens"
