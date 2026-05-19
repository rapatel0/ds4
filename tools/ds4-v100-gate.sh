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
fi

if [ -n "$pack_index" ]; then
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

failures=0
full_scheduler_ready=0
selected_token_ready=0
public_serving_ready=0
base_usability_ready=0
throughput_ready=0
mtp_sidecar_ready=0
mtp_residency_ready=0
mtp_prefix_ready=0

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
    fi
else
    echo "gate	mtp_sidecar	SKIP	no_mtp_model"
    echo "gate	mtp_residency	SKIP	no_mtp_model"
    echo "gate	mtp_prefix	SKIP	no_mtp_model"
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
            run_gate "scheduler_checkpoint_parity" ./tests/cuda_v100_scheduler_checkpoint_parity_smoke --index "$pack_index" --model "$model" --layers -1,0,1,2,3,a4 --ctx 4096 --prompt-tokens 1 || true
            run_gate "output_head_parity" ./tests/cuda_v100_output_head_parity_smoke --index "$pack_index" --model "$model" || true
            if run_gate "scheduler_output_head" ./tests/cuda_v100_selected_token_smoke --index "$pack_index" --model "$model" --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --expected-token-hex 3136; then
                selected_token_ready=1
            fi
            if run_gate "v100_replay_tool" ./tools/ds4-v100-replay --index "$pack_index" --model "$model" --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --tokens 2 --expected-token-hex 3136 --json; then
                throughput_ready=1
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
        else
            echo "gate	descriptor_bound_attention	SKIP	no_model"
            echo "gate	descriptor_bound_ffn	SKIP	no_model"
            echo "gate	integrated_layer	SKIP	no_model"
            echo "gate	integrated_layer_bias	SKIP	no_model"
            echo "gate	stage_scheduler	SKIP	no_model"
            echo "gate	two_stage_scheduler	SKIP	no_model"
            echo "gate	full_scheduler	SKIP	no_model"
            echo "gate	scheduler_checkpoint_parity	SKIP	no_model"
            echo "gate	output_head_parity	SKIP	no_model"
            echo "gate	scheduler_output_head	SKIP	no_model"
            echo "gate	v100_replay_tool	SKIP	no_model"
            echo "gate	v100_appliance_http	SKIP	no_model"
            echo "gate	v100_appliance_http_long	SKIP	no_model"
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
    echo "gate	scheduler_checkpoint_parity	SKIP	no_pack_index"
    echo "gate	output_head_parity	SKIP	no_pack_index"
    echo "gate	scheduler_output_head	SKIP	no_pack_index"
    echo "gate	v100_replay_tool	SKIP	no_pack_index"
    echo "gate	v100_appliance_http	SKIP	no_pack_index"
    echo "gate	v100_appliance_http_long	SKIP	no_pack_index"
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
    else
        add_missing "mtp_forward"
    fi
else
    add_missing "mtp"
fi
if [ "$throughput_ready" -eq 0 ]; then
    add_missing "throughput_benchmark"
fi
echo "gate	readiness	NOT_READY	missing=$missing"
echo "gate	summary	PASS	failures=0 ready=false"
exit 0
