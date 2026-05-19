#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
pack_index="docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv"
prompt_file="tests/test-vectors/prompts/short_reasoning_plain.txt"
expected_hex="3136"
ctx="1048576"
tokens="2"
log_dir=""
min_speedup="1.10"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-throughput-bench.sh --pack-index FILE [options]

Options:
  --model FILE              source-layout GGUF model
  --pack-index FILE         V100 pack-index.tsv
  --prompt-file FILE        prompt file for correctness replay
  --expected-token-hex HEX  expected first response token bytes, default 3136
  --ctx N                   KV context tokens, default 1048576
  --tokens N                generated tokens for correctness replay, default 2
  --min-speedup N           required serial/parallel open speedup, default 1.10
  --log-dir DIR             write benchmark artifacts
  --help                    show this help

Runs serial open-only, parallel open-only, and one normal replay. The benchmark
passes only if parallel open is faster by the requested threshold and the replay
keeps the expected first token.
USAGE
}

fail() {
    echo "ds4-v100-throughput-bench: $*" >&2
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
        --min-speedup)
            [ "$#" -ge 2 ] || fail "--min-speedup requires a value"
            min_speedup="$2"
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

[ -x ./tools/ds4-v100-replay ] || fail "missing executable ./tools/ds4-v100-replay"
[ -f "$model" ] || fail "missing model $model"
[ -f "$pack_index" ] || fail "missing pack index $pack_index"
[ -f "$prompt_file" ] || fail "missing prompt file $prompt_file"

case "$ctx" in ''|0|*[!0-9]*) fail "--ctx must be a positive integer" ;; esac
case "$tokens" in ''|0|*[!0-9]*) fail "--tokens must be a positive integer" ;; esac

work_dir="$log_dir"
if [ -z "$work_dir" ]; then
    work_dir="$(mktemp -d -t ds4-v100-throughput-bench.XXXXXX)"
else
    mkdir -p "$work_dir" || exit 2
fi

serial_json="$work_dir/serial_open.json"
serial_log="$work_dir/serial_open.log"
parallel_json="$work_dir/parallel_open.json"
parallel_log="$work_dir/parallel_open.log"
replay_json="$work_dir/replay.json"
replay_log="$work_dir/replay.log"
report="$work_dir/throughput_optimization.report"
summary_json="$work_dir/throughput_optimization.json"

cleanup() {
    if [ -z "$log_dir" ]; then
        rm -rf "$work_dir"
    fi
}
trap cleanup EXIT

extract_number() {
    key="$1"
    file="$2"
    sed -n "s/.*\"$key\":\\([0-9.][0-9.]*\\).*/\\1/p" "$file" | sed -n '1p'
}

extract_array() {
    key="$1"
    file="$2"
    sed -n "s/.*\"$key\":\\(\\[[^]]*\\]\\).*/\\1/p" "$file" | sed -n '1p'
}

extract_hex() {
    file="$1"
    grep -o '"text_hex":"[^"]*"' "$file" | sed -n '1{s/^"text_hex":"//;s/"$//;p;}'
}

run_open() {
    mode="$1"
    out_json="$2"
    out_log="$3"
    args=(--model "$model" --index "$pack_index" --ctx "$ctx" --open-only --json)
    if [ "$mode" = "serial" ]; then
        args+=(--serial-open)
    fi
    if ! ./tools/ds4-v100-replay "${args[@]}" >"$out_json" 2>"$out_log"; then
        cat "$out_log" >&2
        fail "$mode open-only failed"
    fi
}

run_open serial "$serial_json" "$serial_log"
run_open parallel "$parallel_json" "$parallel_log"

if ! ./tools/ds4-v100-replay \
    --model "$model" \
    --index "$pack_index" \
    --ctx "$ctx" \
    --prompt-file "$prompt_file" \
    --tokens "$tokens" \
    --expected-token-hex "$expected_hex" \
    --json >"$replay_json" 2>"$replay_log"; then
    cat "$replay_log" >&2
    fail "correctness replay failed"
fi

serial_open_ms="$(extract_number open_total "$serial_json")"
parallel_open_ms="$(extract_number open_total "$parallel_json")"
serial_open_stage="$(extract_array open_stage "$serial_json")"
parallel_open_stage="$(extract_array open_stage "$parallel_json")"
replay_open_ms="$(extract_number open_total "$replay_json")"
prompt_replay_ms="$(extract_number prompt_replay "$replay_json")"
continuation_decode_ms="$(extract_number continuation_decode "$replay_json")"
continuation_tps="$(extract_number continuation_tokens_per_second "$replay_json")"
generated_tps="$(extract_number generated_tokens_per_second "$replay_json")"
first_hex="$(extract_hex "$replay_json")"

[ -n "$serial_open_ms" ] || fail "failed to parse serial open timing"
[ -n "$parallel_open_ms" ] || fail "failed to parse parallel open timing"
[ -n "$serial_open_stage" ] || fail "failed to parse serial per-stage open timing"
[ -n "$parallel_open_stage" ] || fail "failed to parse parallel per-stage open timing"
[ -n "$first_hex" ] || fail "failed to parse replay first token"

speedup="$(awk -v s="$serial_open_ms" -v p="$parallel_open_ms" 'BEGIN {
    if ((p + 0.0) <= 0.0) print "0";
    else printf "%.6f", (s + 0.0) / (p + 0.0);
}')"
verdict="PASS"
if ! awk -v v="$speedup" -v min="$min_speedup" 'BEGIN { exit !((v + 0.0) >= (min + 0.0)) }'; then
    verdict="FAIL"
fi
expected_lower="$(printf '%s' "$expected_hex" | tr 'A-F' 'a-f')"
if [ "$first_hex" != "$expected_lower" ]; then
    verdict="FAIL"
fi

{
    printf 'schema\tds4_v100_throughput_optimization.v1\n'
    printf 'model\t%s\n' "$model"
    printf 'pack_index\t%s\n' "$pack_index"
    printf 'ctx\t%s\n' "$ctx"
    printf 'tokens\t%s\n' "$tokens"
    printf 'serial_open_ms\t%s\n' "$serial_open_ms"
    printf 'serial_open_stage_ms\t%s\n' "$serial_open_stage"
    printf 'parallel_open_ms\t%s\n' "$parallel_open_ms"
    printf 'parallel_open_stage_ms\t%s\n' "$parallel_open_stage"
    printf 'speedup\t%s\n' "$speedup"
    printf 'min_speedup\t%s\n' "$min_speedup"
    printf 'replay_open_ms\t%s\n' "${replay_open_ms:-unknown}"
    printf 'prompt_replay_ms\t%s\n' "${prompt_replay_ms:-unknown}"
    printf 'continuation_decode_ms\t%s\n' "${continuation_decode_ms:-unknown}"
    printf 'continuation_tokens_per_second\t%s\n' "${continuation_tps:-unknown}"
    printf 'generated_tokens_per_second\t%s\n' "${generated_tps:-unknown}"
    printf 'first_token_hex\t%s\n' "$first_hex"
    printf 'expected_token_hex\t%s\n' "$expected_lower"
    printf 'verdict\t%s\n' "$verdict"
} >"$report"

cat >"$summary_json" <<EOF
{"schema":"ds4_v100_throughput_optimization.v1","model_path":"$model","pack_index_path":"$pack_index","ctx_tokens":$ctx,"requested_tokens":$tokens,"serial_open_ms":$serial_open_ms,"serial_open_stage_ms":$serial_open_stage,"parallel_open_ms":$parallel_open_ms,"parallel_open_stage_ms":$parallel_open_stage,"speedup":$speedup,"min_speedup":$min_speedup,"replay_open_ms":${replay_open_ms:-0},"prompt_replay_ms":${prompt_replay_ms:-0},"continuation_decode_ms":${continuation_decode_ms:-0},"continuation_tokens_per_second":${continuation_tps:-0},"generated_tokens_per_second":${generated_tps:-0},"first_token_hex":"$first_hex","expected_token_hex":"$expected_lower","optimization_claimed":$([ "$verdict" = "PASS" ] && echo true || echo false),"verdict":"$verdict"}
EOF

cat "$report"
if [ "$verdict" != "PASS" ]; then
    fail "throughput optimization failed verdict=$verdict speedup=$speedup min=$min_speedup first_hex=$first_hex"
fi
echo "ds4-v100-throughput-bench: serial_open_ms=$serial_open_ms parallel_open_ms=$parallel_open_ms speedup=$speedup continuation_tps=${continuation_tps:-unknown} first_hex=$first_hex PASS"
