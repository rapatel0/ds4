#!/usr/bin/env bash
set -u

model="/models/DSv4-Flash-256e-fixed.gguf"
mtp_model=""
ctx="1048576"
slots="1"
build=0
skip_model=0
log_dir=""
cuda_arch="${CUDA_ARCH:-sm_70}"
pack_index=""
descriptor_layer="2"
aggregate_profile="fast"
aggregate_ctx_tiers=""
aggregate_slot_tiers=""
aggregate_queue_policies=""
aggregate_requests=""
aggregate_tokens=""
aggregate_host="127.0.0.1"
aggregate_port_base="18120"
sustained_profile="off"
sustained_ctx_tiers=""
sustained_slot_tiers=""
sustained_queue_policies=""
sustained_requests=""
sustained_tokens=""
sustained_warmup_requests=""
sustained_host="127.0.0.1"
sustained_port_base="18220"
sustained_sample_ms="500"

usage() {
    cat <<'USAGE'
usage: tools/ds4-v100-gate.sh [options]

Options:
  --model FILE      Source-layout GGUF model path
  --mtp-model FILE  DeepSeek-V4 Flash MTP sidecar GGUF path
  --ctx N           KV context tier for the V100 context smoke (default 1048576)
  --slots N         KV slots for the V100 context smoke (default 1)
  --build           Build required targets before running the gate
  --cuda-arch ARCH  CUDA arch to pass to make when --build is used (default sm_70)
  --log-dir DIR     Write each command's output to DIR
  --pack-index FILE Validate real pack-index layer descriptors
  --descriptor-layer N
                    Layer to validate when --pack-index is supplied (default 2)
  --aggregate-profile MODE
                    Throughput matrix profile: fast or full (default fast)
  --aggregate-ctx-tiers LIST
                    Override aggregate context tiers CSV
  --aggregate-slot-tiers LIST
                    Override aggregate slot tiers CSV
  --aggregate-queue-policies LIST
                    Override aggregate queue policies CSV
  --aggregate-requests N
                    Override aggregate requests per case
  --aggregate-tokens N
                    Override aggregate generated tokens per request
  --aggregate-host ADDR
                    Host for aggregate throughput runs (default 127.0.0.1)
  --aggregate-port-base N
                    Base port for aggregate throughput runs (default 18120)
  --sustained-profile MODE
                    Sustained decode profile: off, smoke, or full (default off)
  --sustained-ctx-tiers LIST
                    Override sustained decode context tiers CSV
  --sustained-slot-tiers LIST
                    Override sustained decode slot tiers CSV
  --sustained-queue-policies LIST
                    Override sustained decode queue policies CSV
  --sustained-requests N
                    Override sustained decode timed requests per case
  --sustained-tokens N
                    Override sustained decode generated tokens per request
  --sustained-warmup-requests N
                    Override sustained decode warmup requests per case
  --sustained-host ADDR
                    Host for sustained decode runs (default 127.0.0.1)
  --sustained-port-base N
                    Base port for sustained decode runs (default 18220)
  --sustained-sample-ms N
                    nvidia-smi sample period in ms for sustained runs
  --skip-model      Skip real-model source guard check
  --help            Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --model requires a value" >&2; exit 2; }
            model="$2"
            shift 2
            ;;
        --mtp-model)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --mtp-model requires a value" >&2; exit 2; }
            mtp_model="$2"
            shift 2
            ;;
        --ctx)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --ctx requires a value" >&2; exit 2; }
            ctx="$2"
            shift 2
            ;;
        --slots)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --slots requires a value" >&2; exit 2; }
            slots="$2"
            shift 2
            ;;
        --build)
            build=1
            shift
            ;;
        --cuda-arch)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --cuda-arch requires a value" >&2; exit 2; }
            cuda_arch="$2"
            shift 2
            ;;
        --log-dir)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --log-dir requires a value" >&2; exit 2; }
            log_dir="$2"
            shift 2
            ;;
        --pack-index)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --pack-index requires a value" >&2; exit 2; }
            pack_index="$2"
            shift 2
            ;;
        --descriptor-layer)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --descriptor-layer requires a value" >&2; exit 2; }
            descriptor_layer="$2"
            shift 2
            ;;
        --aggregate-profile)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-profile requires a value" >&2; exit 2; }
            aggregate_profile="$2"
            shift 2
            ;;
        --aggregate-ctx-tiers)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-ctx-tiers requires a value" >&2; exit 2; }
            aggregate_ctx_tiers="$2"
            shift 2
            ;;
        --aggregate-slot-tiers)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-slot-tiers requires a value" >&2; exit 2; }
            aggregate_slot_tiers="$2"
            shift 2
            ;;
        --aggregate-queue-policies)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-queue-policies requires a value" >&2; exit 2; }
            aggregate_queue_policies="$2"
            shift 2
            ;;
        --aggregate-requests)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-requests requires a value" >&2; exit 2; }
            aggregate_requests="$2"
            shift 2
            ;;
        --aggregate-tokens)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-tokens requires a value" >&2; exit 2; }
            aggregate_tokens="$2"
            shift 2
            ;;
        --aggregate-host)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-host requires a value" >&2; exit 2; }
            aggregate_host="$2"
            shift 2
            ;;
        --aggregate-port-base)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --aggregate-port-base requires a value" >&2; exit 2; }
            aggregate_port_base="$2"
            shift 2
            ;;
        --sustained-profile)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-profile requires a value" >&2; exit 2; }
            sustained_profile="$2"
            shift 2
            ;;
        --sustained-ctx-tiers)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-ctx-tiers requires a value" >&2; exit 2; }
            sustained_ctx_tiers="$2"
            shift 2
            ;;
        --sustained-slot-tiers)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-slot-tiers requires a value" >&2; exit 2; }
            sustained_slot_tiers="$2"
            shift 2
            ;;
        --sustained-queue-policies)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-queue-policies requires a value" >&2; exit 2; }
            sustained_queue_policies="$2"
            shift 2
            ;;
        --sustained-requests)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-requests requires a value" >&2; exit 2; }
            sustained_requests="$2"
            shift 2
            ;;
        --sustained-tokens)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-tokens requires a value" >&2; exit 2; }
            sustained_tokens="$2"
            shift 2
            ;;
        --sustained-warmup-requests)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-warmup-requests requires a value" >&2; exit 2; }
            sustained_warmup_requests="$2"
            shift 2
            ;;
        --sustained-host)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-host requires a value" >&2; exit 2; }
            sustained_host="$2"
            shift 2
            ;;
        --sustained-port-base)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-port-base requires a value" >&2; exit 2; }
            sustained_port_base="$2"
            shift 2
            ;;
        --sustained-sample-ms)
            [ "$#" -ge 2 ] || { echo "ds4-v100-gate: --sustained-sample-ms requires a value" >&2; exit 2; }
            sustained_sample_ms="$2"
            shift 2
            ;;
        --skip-model)
            skip_model=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ds4-v100-gate: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$aggregate_profile" in
    fast)
        [ -n "$aggregate_ctx_tiers" ] || aggregate_ctx_tiers="262144,1048576"
        [ -n "$aggregate_slot_tiers" ] || aggregate_slot_tiers="2"
        [ -n "$aggregate_queue_policies" ] || aggregate_queue_policies="sequential"
        [ -n "$aggregate_requests" ] || aggregate_requests="8"
        [ -n "$aggregate_tokens" ] || aggregate_tokens="1"
        ;;
    full)
        [ -n "$aggregate_ctx_tiers" ] || aggregate_ctx_tiers="131072,262144,524288,1048576"
        [ -n "$aggregate_slot_tiers" ] || aggregate_slot_tiers="1,2,4,8"
        [ -n "$aggregate_queue_policies" ] || aggregate_queue_policies="sequential,reject-busy"
        [ -n "$aggregate_requests" ] || aggregate_requests="4"
        [ -n "$aggregate_tokens" ] || aggregate_tokens="1"
        ;;
    *)
        echo "ds4-v100-gate: --aggregate-profile must be fast or full" >&2
        exit 2
        ;;
esac

case "$aggregate_requests" in
    ''|0|*[!0-9]*)
        echo "ds4-v100-gate: --aggregate-requests must be a positive integer" >&2
        exit 2
        ;;
esac
case "$aggregate_tokens" in
    ''|0|*[!0-9]*)
        echo "ds4-v100-gate: --aggregate-tokens must be a positive integer" >&2
        exit 2
        ;;
esac
case "$aggregate_port_base" in
    ''|0|*[!0-9]*)
        echo "ds4-v100-gate: --aggregate-port-base must be a positive integer" >&2
        exit 2
        ;;
esac

case "$sustained_profile" in
    off)
        ;;
    smoke)
        [ -n "$sustained_ctx_tiers" ] || sustained_ctx_tiers="1048576"
        [ -n "$sustained_slot_tiers" ] || sustained_slot_tiers="1"
        [ -n "$sustained_queue_policies" ] || sustained_queue_policies="sequential"
        [ -n "$sustained_requests" ] || sustained_requests="2"
        [ -n "$sustained_tokens" ] || sustained_tokens="4"
        [ -n "$sustained_warmup_requests" ] || sustained_warmup_requests="0"
        ;;
    full)
        [ -n "$sustained_ctx_tiers" ] || sustained_ctx_tiers="262144,1048576"
        [ -n "$sustained_slot_tiers" ] || sustained_slot_tiers="1,2,4"
        [ -n "$sustained_queue_policies" ] || sustained_queue_policies="sequential"
        [ -n "$sustained_requests" ] || sustained_requests="8"
        [ -n "$sustained_tokens" ] || sustained_tokens="16"
        [ -n "$sustained_warmup_requests" ] || sustained_warmup_requests="1"
        ;;
    *)
        echo "ds4-v100-gate: --sustained-profile must be off, smoke, or full" >&2
        exit 2
        ;;
esac

if [ "$sustained_profile" != "off" ]; then
    case "$sustained_requests" in
        ''|0|*[!0-9]*)
            echo "ds4-v100-gate: --sustained-requests must be a positive integer" >&2
            exit 2
            ;;
    esac
    case "$sustained_tokens" in
        ''|0|1|*[!0-9]*)
            echo "ds4-v100-gate: --sustained-tokens must be an integer >= 2" >&2
            exit 2
            ;;
    esac
    case "$sustained_warmup_requests" in
        ''|*[!0-9]*)
            echo "ds4-v100-gate: --sustained-warmup-requests must be a non-negative integer" >&2
            exit 2
            ;;
    esac
    case "$sustained_port_base" in
        ''|0|*[!0-9]*)
            echo "ds4-v100-gate: --sustained-port-base must be a positive integer" >&2
            exit 2
            ;;
    esac
    case "$sustained_sample_ms" in
        ''|0|*[!0-9]*)
            echo "ds4-v100-gate: --sustained-sample-ms must be a positive integer" >&2
            exit 2
            ;;
    esac
fi

targets=(
    tools/ds4-source-oracle-vector
    tests/cuda_source_dtypes_smoke
    tests/cuda_bf16_probe
    tests/cuda_v100_context_smoke
    tests/cuda_v100_compressor_bridge_smoke
    tests/cuda_v100_prefill_kv_smoke
    tests/cuda_hc_relay_smoke
    tests/cuda_v100_projection_attention_smoke
    tests/cuda_v100_bounded_logits_smoke
    tests/cuda_v100_mxfp4_moe_smoke
)

if [ -n "$mtp_model" ]; then
    targets+=(tools/ds4-v100-mtp-sidecar-gate)
    targets+=(tools/ds4-v100-mtp-residency-smoke)
    targets+=(tools/ds4-v100-mtp-prefix-smoke)
    targets+=(tools/ds4-v100-mtp-q4k-smoke)
    targets+=(tools/ds4-v100-mtp-ffn-smoke)
    targets+=(tools/ds4-v100-mtp-attn-smoke)
    if [ -n "$pack_index" ]; then
        targets+=(tools/ds4-v100-mtp-logits-smoke)
        targets+=(tools/ds4-v100-mtp-forward-smoke)
        targets+=(tools/ds4-v100-mtp-verify-smoke)
    fi
fi

if [ -n "$pack_index" ]; then
    targets+=(tools/ds4-v100-plan)
    targets+=(tools/ds4-v100-layer-descriptor-gate)
    targets+=(tests/v100_layer_binding_smoke)
    targets+=(tests/v100_layer_state_smoke)
    targets+=(tests/cuda_v100_descriptor_bound_attention_smoke)
    targets+=(tests/cuda_v100_descriptor_bound_ffn_smoke)
    targets+=(tests/cuda_v100_integrated_layer_smoke)
    targets+=(tests/cuda_v100_stage_scheduler_smoke)
    targets+=(tests/cuda_v100_two_stage_scheduler_smoke)
    targets+=(tests/cuda_v100_full_scheduler_smoke)
    targets+=(tests/cuda_v100_output_head_parity_smoke)
    targets+=(tests/cuda_v100_selected_token_smoke)
    targets+=(tests/cuda_v100_scheduler_checkpoint_parity_smoke)
    targets+=(tests/cuda_v100_scheduler_snapshot_smoke)
    targets+=(tools/ds4-v100-replay)
fi

if [ -n "$log_dir" ]; then
    mkdir -p "$log_dir" || exit 2
fi

if [ "$build" -eq 1 ]; then
    echo "gate	build	CUDA_ARCH=$cuda_arch targets=${targets[*]}"
    if ! CUDA_ARCH="$cuda_arch" make "${targets[@]}"; then
        echo "gate	build	FAIL"
        exit 1
    fi
    echo "gate	build	PASS"
fi

echo "gate	aggregate_profile	profile=$aggregate_profile ctx_tiers=$aggregate_ctx_tiers slot_tiers=$aggregate_slot_tiers queue_policies=$aggregate_queue_policies requests=$aggregate_requests tokens=$aggregate_tokens host=$aggregate_host port_base=$aggregate_port_base"
if [ "$sustained_profile" != "off" ]; then
    echo "gate	sustained_profile	profile=$sustained_profile ctx_tiers=$sustained_ctx_tiers slot_tiers=$sustained_slot_tiers queue_policies=$sustained_queue_policies requests=$sustained_requests tokens=$sustained_tokens warmup_requests=$sustained_warmup_requests host=$sustained_host port_base=$sustained_port_base sample_ms=$sustained_sample_ms"
else
    echo "gate	sustained_profile	profile=off"
fi

failures=0
full_scheduler_ready=0
selected_token_ready=0
public_serving_ready=0
base_usability_ready=0
throughput_ready=0
throughput_optimization_ready=0
production_deployment_ready=0
slot_context_admission_ready=0
active_microbatch_scheduler_ready=0
aggregate_slot_context_throughput_ready=0
sustained_decode_ready=0
mtp_sidecar_ready=0
mtp_residency_ready=0
mtp_prefix_ready=0
mtp_q4k_ready=0
mtp_ffn_ready=0
mtp_attn_ready=0
mtp_logits_ready=0
mtp_forward_ready=0
mtp_rollback_ready=0
mtp_verify_ready=0
mtp_speculative_serving_ready=0

run_gate() {
    local name="$1"
    shift
    local log_path=""
    local tmp=""
    if [ -n "$log_dir" ]; then
        log_path="$log_dir/${name}.log"
        tmp="$log_path"
    else
        tmp="$(mktemp -t ds4-v100-gate-${name}.XXXXXX)"
    fi

    if "$@" >"$tmp" 2>&1; then
        echo "gate	${name}	PASS	command=$*"
        if [ -z "$log_dir" ]; then
            rm -f "$tmp"
        fi
        return 0
    fi

    echo "gate	${name}	FAIL	command=$*"
    cat "$tmp"
    if [ -z "$log_dir" ]; then
        rm -f "$tmp"
    else
        echo "gate	${name}	LOG	$log_path"
    fi
    failures=$((failures + 1))
    return 1
}

if command -v nvidia-smi >/dev/null 2>&1; then
    run_gate "nvidia_smi" nvidia-smi -L || true
else
    echo "gate	nvidia_smi	SKIP	command_not_found"
fi

if [ "$skip_model" -eq 0 ]; then
    if [ ! -f "$model" ]; then
        echo "gate	source_guards	FAIL	missing_model=$model"
        failures=$((failures + 1))
    else
        run_gate "source_guards" ./tools/ds4-source-oracle-vector --model "$model" --guards-only || true
    fi
else
    echo "gate	source_guards	SKIP	--skip-model"
fi

if [ -n "$mtp_model" ]; then
    if [ ! -f "$mtp_model" ]; then
        echo "gate	mtp_sidecar	FAIL	missing_mtp_model=$mtp_model"
        failures=$((failures + 1))
    else
        if run_gate "mtp_sidecar" ./tools/ds4-v100-mtp-sidecar-gate --mtp-model "$mtp_model"; then
            mtp_sidecar_ready=1
        fi
        if run_gate "mtp_residency" ./tools/ds4-v100-mtp-residency-smoke \
            --mtp-model "$mtp_model" --gpu 7 --require-gpus 8 --reserve-mib 4096; then
            mtp_residency_ready=1
        fi
        mtp_prefix_args=(
            --mtp-model "$mtp_model"
            --gpu 7
            --require-gpus 8
            --reserve-mib 4096
        )
        if [ -n "$log_dir" ]; then
            mtp_prefix_args+=(--report "$log_dir/mtp_prefix.report")
        fi
        if run_gate "mtp_prefix" ./tools/ds4-v100-mtp-prefix-smoke "${mtp_prefix_args[@]}"; then
            mtp_prefix_ready=1
        fi
        mtp_q4k_args=(
            --mtp-model "$mtp_model"
            --gpu 7
            --require-gpus 8
            --reserve-mib 4096
        )
        if [ -n "$log_dir" ]; then
            mtp_q4k_args+=(--report "$log_dir/mtp_q4k.report")
        fi
        if run_gate "mtp_q4k" ./tools/ds4-v100-mtp-q4k-smoke "${mtp_q4k_args[@]}"; then
            mtp_q4k_ready=1
        fi
        mtp_ffn_args=(
            --mtp-model "$mtp_model"
            --gpu 7
            --require-gpus 8
            --reserve-mib 4096
        )
        if [ -n "$log_dir" ]; then
            mtp_ffn_args+=(--report "$log_dir/mtp_ffn.report")
        fi
        if run_gate "mtp_ffn" ./tools/ds4-v100-mtp-ffn-smoke "${mtp_ffn_args[@]}"; then
            mtp_ffn_ready=1
        fi
        mtp_attn_args=(
            --mtp-model "$mtp_model"
            --gpu 7
            --require-gpus 8
            --reserve-mib 4096
        )
        if [ -n "$log_dir" ]; then
            mtp_attn_args+=(--report "$log_dir/mtp_attn.report")
        fi
        if run_gate "mtp_attn" ./tools/ds4-v100-mtp-attn-smoke "${mtp_attn_args[@]}"; then
            mtp_attn_ready=1
        fi
        if [ -n "$pack_index" ]; then
            mtp_logits_args=(
                --model "$model"
                --mtp-model "$mtp_model"
                --pack-index "$pack_index"
                --gpu 7
                --require-gpus 8
                --reserve-mib 4096
            )
            if [ -n "$log_dir" ]; then
                mtp_logits_args+=(--report "$log_dir/mtp_logits.report")
            fi
            if run_gate "mtp_logits" ./tools/ds4-v100-mtp-logits-smoke "${mtp_logits_args[@]}"; then
                mtp_logits_ready=1
            fi
            mtp_forward_args=(
                --model "$model"
                --mtp-model "$mtp_model"
                --pack-index "$pack_index"
                --gpu 7
                --require-gpus 8
                --reserve-mib 4096
            )
            if [ -n "$log_dir" ]; then
                mtp_forward_args+=(--report "$log_dir/mtp_forward.report")
            fi
            if run_gate "mtp_forward" ./tools/ds4-v100-mtp-forward-smoke "${mtp_forward_args[@]}"; then
                mtp_forward_ready=1
            fi
            mtp_verify_args=(
                --model "$model"
                --mtp-model "$mtp_model"
                --pack-index "$pack_index"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --gpu 7
                --require-gpus 8
                --reserve-mib 4096
                --ctx "$ctx"
            )
            if [ -n "$log_dir" ]; then
                mtp_verify_args+=(--report "$log_dir/mtp_verify.report")
            fi
            if run_gate "mtp_verify" ./tools/ds4-v100-mtp-verify-smoke "${mtp_verify_args[@]}"; then
                mtp_rollback_ready=1
                mtp_verify_ready=1
            fi
        else
            echo "gate	mtp_logits	SKIP	no_pack_index"
            echo "gate	mtp_forward	SKIP	no_pack_index"
            echo "gate	mtp_rollback	SKIP	no_pack_index"
            echo "gate	mtp_verify	SKIP	no_pack_index"
        fi
    fi
else
    echo "gate	mtp_sidecar	SKIP	no_mtp_model"
    echo "gate	mtp_residency	SKIP	no_mtp_model"
    echo "gate	mtp_prefix	SKIP	no_mtp_model"
    echo "gate	mtp_q4k	SKIP	no_mtp_model"
    echo "gate	mtp_ffn	SKIP	no_mtp_model"
    echo "gate	mtp_attn	SKIP	no_mtp_model"
    echo "gate	mtp_logits	SKIP	no_mtp_model"
    echo "gate	mtp_forward	SKIP	no_mtp_model"
    echo "gate	mtp_rollback	SKIP	no_mtp_model"
    echo "gate	mtp_verify	SKIP	no_mtp_model"
    echo "gate	mtp_speculative_serving	SKIP	no_mtp_model"
fi

run_gate "source_dtypes" ./tests/cuda_source_dtypes_smoke || true
run_gate "bf16_probe" ./tests/cuda_bf16_probe || true
run_gate "context_kv" ./tests/cuda_v100_context_smoke --production --kv-ctx "$ctx" --kv-slots "$slots" || true
run_gate "compressor_bridge" ./tests/cuda_v100_compressor_bridge_smoke || true
run_gate "prefill_kv" ./tests/cuda_v100_prefill_kv_smoke || true
run_gate "hc_relay" ./tests/cuda_hc_relay_smoke || true
run_gate "projection_attention" ./tests/cuda_v100_projection_attention_smoke || true
run_gate "bounded_logits" ./tests/cuda_v100_bounded_logits_smoke || true
run_gate "mxfp4_moe" ./tests/cuda_v100_mxfp4_moe_smoke || true
if [ -n "$pack_index" ]; then
    if [ ! -f "$pack_index" ]; then
        echo "gate	layer_descriptors	FAIL	missing_pack_index=$pack_index"
        failures=$((failures + 1))
    else
        run_gate "layer_descriptors" ./tools/ds4-v100-layer-descriptor-gate --index "$pack_index" --layer "$descriptor_layer" --gpus 8 || true
        run_gate "layer_bindings" ./tests/v100_layer_binding_smoke --index "$pack_index" --layer "$descriptor_layer" || true
        run_gate "layer_state" ./tests/v100_layer_state_smoke --index "$pack_index" --layer "$descriptor_layer" || true
        if [ "$skip_model" -eq 0 ] && [ -f "$model" ]; then
            run_gate "descriptor_bound_attention" ./tests/cuda_v100_descriptor_bound_attention_smoke --index "$pack_index" --model "$model" --layer "$descriptor_layer" || true
            run_gate "descriptor_bound_ffn" ./tests/cuda_v100_descriptor_bound_ffn_smoke --index "$pack_index" --model "$model" --layer "$descriptor_layer" --router-token 16 || true
            run_gate "integrated_layer" ./tests/cuda_v100_integrated_layer_smoke --index "$pack_index" --model "$model" --layer "$descriptor_layer" --router-token 16 --position 16 || true
            run_gate "integrated_layer_bias" ./tests/cuda_v100_integrated_layer_smoke --index "$pack_index" --model "$model" --layer 3 --router-token 16 --position 16 || true
            run_gate "stage_scheduler" ./tests/cuda_v100_stage_scheduler_smoke --index "$pack_index" --model "$model" --stage 0 --token 16 --position 16 || true
            run_gate "two_stage_scheduler" ./tests/cuda_v100_two_stage_scheduler_smoke --index "$pack_index" --model "$model" --token 16 --position 16 || true
            if run_gate "full_scheduler" ./tests/cuda_v100_full_scheduler_smoke --index "$pack_index" --model "$model" --token 16 --position 16; then
                full_scheduler_ready=1
            fi
            if run_gate "active_microbatch_scheduler" ./tests/cuda_v100_full_scheduler_smoke --index "$pack_index" --model "$model" --token 16 --position 16 --slots 2; then
                active_microbatch_scheduler_ready=1
            fi
            run_gate "scheduler_checkpoint_parity" ./tests/cuda_v100_scheduler_checkpoint_parity_smoke --index "$pack_index" --model "$model" --layers -1,0,1,2,3,a4 --ctx 4096 --prompt-tokens 1 || true
            run_gate "scheduler_snapshot" ./tests/cuda_v100_scheduler_snapshot_smoke --index "$pack_index" --model "$model" --ctx 4096 --steps 8 || true
            run_gate "output_head_parity" ./tests/cuda_v100_output_head_parity_smoke --index "$pack_index" --model "$model" || true
            if run_gate "scheduler_output_head" ./tests/cuda_v100_selected_token_smoke --index "$pack_index" --model "$model" --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --expected-token-hex 3136; then
                selected_token_ready=1
            fi
            if run_gate "v100_replay_tool" ./tools/ds4-v100-replay --index "$pack_index" --model "$model" --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --tokens 2 --expected-token-hex 3136 --json; then
                throughput_ready=1
            fi
            throughput_optimization_args=(
                --index "$pack_index"
                --model "$model"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --ctx "$ctx"
                --tokens 2
                --expected-token-hex 3136
                --min-speedup 1.05
            )
            if [ -n "$log_dir" ]; then
                throughput_optimization_args+=(--log-dir "$log_dir/throughput_optimization")
            fi
            if run_gate "throughput_optimization" ./tools/ds4-v100-throughput-bench.sh "${throughput_optimization_args[@]}"; then
                throughput_optimization_ready=1
            fi
            appliance_args=(
                --index "$pack_index"
                --model "$model"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --tokens 1
                --requests 2
                --expected-token-hex 3136
                --host 127.0.0.1
                --port 18080
            )
            if [ -n "$log_dir" ]; then
                appliance_args+=(--log-dir "$log_dir/v100_appliance_http")
            fi
            if run_gate "v100_appliance_http" ./tools/ds4-v100-appliance-smoke.sh "${appliance_args[@]}"; then
                public_serving_ready=1
            fi
            appliance_long_args=(
                --index "$pack_index"
                --model "$model"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --tokens 2
                --requests 2
                --expected-token-hex 3136
                --host 127.0.0.1
                --port 18081
            )
            if [ -n "$log_dir" ]; then
                appliance_long_args+=(--log-dir "$log_dir/v100_appliance_http_long")
            fi
            if run_gate "v100_appliance_http_long" ./tools/ds4-v100-appliance-smoke.sh "${appliance_long_args[@]}"; then
                base_usability_ready=1
            fi
            production_args=(
                --index "$pack_index"
                --model "$model"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --slots "$slots"
                --ctx "$ctx"
                --tokens 2
                --requests 1
                --expected-token-hex 3136
                --host 127.0.0.1
                --port 18082
                --reserve-mib 4096
                --require-gpus 8
            )
            if [ -n "$mtp_model" ]; then
                production_args+=(--mtp-model "$mtp_model")
            fi
            if [ -n "$log_dir" ]; then
                production_args+=(--log-dir "$log_dir/production_deployment")
            fi
            if run_gate "production_deployment" ./tools/ds4-v100-production-deployment-gate.sh "${production_args[@]}"; then
                production_deployment_ready=1
            fi
            slot_context_args=(
                --pack-index "$pack_index"
                --model "$model"
                --plan-slots 8
                --smoke-slots "$slots"
                --smoke-ctx "$ctx"
                --requests 2
            )
            if [ -n "$log_dir" ]; then
                slot_context_args+=(--log-dir "$log_dir/slot_context_envelope")
            fi
            if run_gate "slot_context_admission" bash ./tools/ds4-v100-slot-context-envelope.sh "${slot_context_args[@]}"; then
                slot_context_admission_ready=1
            fi
            aggregate_throughput_args=(
                --pack-index "$pack_index"
                --model "$model"
                --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                --expected-token-hex 3136
                --ctx-tiers "$aggregate_ctx_tiers"
                --slot-tiers "$aggregate_slot_tiers"
                --queue-policies "$aggregate_queue_policies"
                --requests "$aggregate_requests"
                --tokens "$aggregate_tokens"
                --host "$aggregate_host"
                --port-base "$aggregate_port_base"
            )
            if [ -n "$log_dir" ]; then
                aggregate_throughput_args+=(--log-dir "$log_dir/aggregate_slot_context_throughput")
            fi
            if run_gate "aggregate_slot_context_throughput" bash ./tools/ds4-v100-aggregate-throughput.sh "${aggregate_throughput_args[@]}"; then
                aggregate_slot_context_throughput_ready=1
            fi
            if [ "$sustained_profile" != "off" ]; then
                sustained_decode_args=(
                    --pack-index "$pack_index"
                    --model "$model"
                    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                    --expected-token-hex 3136
                    --ctx-tiers "$sustained_ctx_tiers"
                    --slot-tiers "$sustained_slot_tiers"
                    --queue-policies "$sustained_queue_policies"
                    --requests "$sustained_requests"
                    --tokens "$sustained_tokens"
                    --warmup-requests "$sustained_warmup_requests"
                    --host "$sustained_host"
                    --port-base "$sustained_port_base"
                    --sample-ms "$sustained_sample_ms"
                )
                if [ -n "$log_dir" ]; then
                    sustained_decode_args+=(--log-dir "$log_dir/sustained_decode")
                fi
                if run_gate "sustained_decode" bash ./tools/ds4-v100-sustained-decode-bench.sh "${sustained_decode_args[@]}"; then
                    sustained_decode_ready=1
                fi
            else
                echo "gate	sustained_decode	SKIP	profile=off"
            fi
            if [ -n "$mtp_model" ]; then
                mtp_serving_args=(
                    --index "$pack_index"
                    --model "$model"
                    --mtp-model "$mtp_model"
                    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
                    --ctx "$ctx"
                    --tokens 2
                    --requests 1
                    --expected-token-hex 3136
                    --host 127.0.0.1
                    --port 18083
                    --top-k 5
                    --mtp-gpu 7
                    --reserve-mib 4096
                )
                if [ -n "$log_dir" ]; then
                    mtp_serving_args+=(--log-dir "$log_dir/mtp_speculative_serving")
                fi
                if run_gate "mtp_speculative_serving" ./tools/ds4-v100-mtp-serving-smoke.sh "${mtp_serving_args[@]}"; then
                    mtp_speculative_serving_ready=1
                fi
            fi
        else
            echo "gate	descriptor_bound_attention	SKIP	no_model"
            echo "gate	descriptor_bound_ffn	SKIP	no_model"
            echo "gate	integrated_layer	SKIP	no_model"
            echo "gate	integrated_layer_bias	SKIP	no_model"
            echo "gate	stage_scheduler	SKIP	no_model"
            echo "gate	two_stage_scheduler	SKIP	no_model"
            echo "gate	full_scheduler	SKIP	no_model"
            echo "gate	active_microbatch_scheduler	SKIP	no_model"
            echo "gate	scheduler_checkpoint_parity	SKIP	no_model"
            echo "gate	scheduler_snapshot	SKIP	no_model"
            echo "gate	output_head_parity	SKIP	no_model"
            echo "gate	scheduler_output_head	SKIP	no_model"
            echo "gate	v100_replay_tool	SKIP	no_model"
            echo "gate	throughput_optimization	SKIP	no_model"
            echo "gate	v100_appliance_http	SKIP	no_model"
            echo "gate	v100_appliance_http_long	SKIP	no_model"
            echo "gate	production_deployment	SKIP	no_model"
            echo "gate	slot_context_admission	SKIP	no_model"
            echo "gate	aggregate_slot_context_throughput	SKIP	no_model"
            if [ "$sustained_profile" != "off" ]; then
                echo "gate	sustained_decode	SKIP	no_model"
            fi
            echo "gate	mtp_speculative_serving	SKIP	no_model"
        fi
    fi
else
    echo "gate	layer_descriptors	SKIP	no_pack_index"
    echo "gate	layer_bindings	SKIP	no_pack_index"
    echo "gate	layer_state	SKIP	no_pack_index"
    echo "gate	descriptor_bound_attention	SKIP	no_pack_index"
    echo "gate	descriptor_bound_ffn	SKIP	no_pack_index"
    echo "gate	integrated_layer	SKIP	no_pack_index"
    echo "gate	integrated_layer_bias	SKIP	no_pack_index"
    echo "gate	stage_scheduler	SKIP	no_pack_index"
    echo "gate	two_stage_scheduler	SKIP	no_pack_index"
    echo "gate	full_scheduler	SKIP	no_pack_index"
    echo "gate	active_microbatch_scheduler	SKIP	no_pack_index"
    echo "gate	scheduler_checkpoint_parity	SKIP	no_pack_index"
    echo "gate	scheduler_snapshot	SKIP	no_pack_index"
    echo "gate	output_head_parity	SKIP	no_pack_index"
    echo "gate	scheduler_output_head	SKIP	no_pack_index"
    echo "gate	v100_replay_tool	SKIP	no_pack_index"
    echo "gate	throughput_optimization	SKIP	no_pack_index"
    echo "gate	v100_appliance_http	SKIP	no_pack_index"
    echo "gate	v100_appliance_http_long	SKIP	no_pack_index"
    echo "gate	production_deployment	SKIP	no_pack_index"
    echo "gate	slot_context_admission	SKIP	no_pack_index"
    echo "gate	aggregate_slot_context_throughput	SKIP	no_pack_index"
    if [ "$sustained_profile" != "off" ]; then
        echo "gate	sustained_decode	SKIP	no_pack_index"
    fi
    echo "gate	mtp_speculative_serving	SKIP	no_pack_index"
fi

if [ "$failures" -ne 0 ]; then
    echo "gate	summary	FAIL	failures=$failures ready=false"
    exit 1
fi

missing=""
add_missing() {
    if [ -z "$missing" ]; then
        missing="$1"
    else
        missing="$missing,$1"
    fi
}

if [ "$full_scheduler_ready" -eq 0 ]; then
    add_missing "full_43_layer_scheduler"
fi
if [ "$selected_token_ready" -eq 0 ]; then
    add_missing "real_model_selected_token"
fi
if [ "$public_serving_ready" -eq 0 ]; then
    add_missing "public_serving"
fi
if [ "$base_usability_ready" -eq 0 ]; then
    add_missing "base_appliance_usability"
fi
if [ -n "$mtp_model" ]; then
    if [ "$mtp_sidecar_ready" -eq 0 ]; then
        add_missing "mtp_sidecar"
    elif [ "$mtp_residency_ready" -eq 0 ]; then
        add_missing "mtp_residency"
    elif [ "$mtp_prefix_ready" -eq 0 ]; then
        add_missing "mtp_prefix"
    elif [ "$mtp_q4k_ready" -eq 0 ]; then
        add_missing "mtp_q4k"
    elif [ "$mtp_ffn_ready" -eq 0 ]; then
        add_missing "mtp_ffn"
    elif [ "$mtp_attn_ready" -eq 0 ]; then
        add_missing "mtp_attn"
    elif [ "$mtp_logits_ready" -eq 0 ]; then
        add_missing "mtp_logits"
    elif [ "$mtp_forward_ready" -eq 0 ]; then
        add_missing "mtp_forward"
    elif [ "$mtp_rollback_ready" -eq 0 ]; then
        add_missing "mtp_rollback"
    elif [ "$mtp_verify_ready" -eq 0 ]; then
        add_missing "mtp_verify"
    elif [ "$mtp_speculative_serving_ready" -eq 0 ]; then
        add_missing "mtp_speculative_serving"
    fi
else
    add_missing "mtp"
fi
if [ "$throughput_ready" -eq 0 ]; then
    add_missing "throughput_benchmark"
fi
if [ "$production_deployment_ready" -eq 0 ]; then
    add_missing "production_deployment"
fi
if [ "$throughput_optimization_ready" -eq 0 ]; then
    add_missing "throughput_optimization"
fi
if [ "$slot_context_admission_ready" -eq 0 ]; then
    add_missing "slot_context_admission"
fi
if [ "$active_microbatch_scheduler_ready" -eq 0 ]; then
    add_missing "active_microbatch_scheduler"
fi
if [ "$aggregate_slot_context_throughput_ready" -eq 0 ]; then
    add_missing "aggregate_slot_context_throughput"
fi
if [ "$sustained_profile" != "off" ] && [ "$sustained_decode_ready" -eq 0 ]; then
    add_missing "sustained_decode"
fi

if [ -n "$missing" ]; then
    echo "gate	readiness	NOT_READY	missing=$missing"
    echo "gate	summary	PASS	failures=0 ready=false"
    exit 0
fi

echo "gate	readiness	READY	missing="
echo "gate	summary	PASS	failures=0 ready=true"
exit 0
