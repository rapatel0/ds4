# TEMP Status Report 416: Persistent TP/EP CUDA Graph Replay

Date: 2026-05-26

## Scope

Sprint execution resumed after the connectivity interruption. The focus stayed on the hard-cut TP/EP path only: no PP/layer-split work.

Primary objective: replace per-token CUDA graph recapture with a persistent per-layer graph exec cache for the TP/EP token-major decode path.

## Implementation

- Added a per-layer TP CUDA graph exec cache in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added `--decode-cudagraph-persistent-replay-gate`.
- First token/layer invocation captures and instantiates a graph.
- Later token/layer invocations replay the cached `cudaGraphExec_t`.
- Cache is owned by `SharedRankBuffers`, so it survives across token-major decode steps and is destroyed with the shared rank buffers.
- Added permanent profile harness support in `tools/ds4-v100-tp-ep-profile.py` via `--persistent-decode-cudagraph`.

## Cluster Validation

Build:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS on gpu-01.

Run artifact:

```text
/localpool/ds4/workspace/logs/spike-b-c-capture/persistent-replay-slot8-steps4/
```

Configuration:

```text
slots=8
ctx=262144
decode_steps=4
layers=43
token-major all-layers
TP/EP resident shared state
diagnostic output head enabled
persistent CUDA graph replay enabled
```

## Topline Result

```text
PASS
capture_attempted=43
capture_succeeded=43
replay_attempted=172
replay_succeeded=172
capture_error=cudaSuccess
replay_error=cudaSuccess
capture_eligible=1
blocker=none
```

Performance from the run:

```text
first_token_decode_ms=125.271039
continuation_decode_ms=271.106144
total_decode_ms=396.377183
generated_tokens=32
continuation_tokens=24
aggregate_generated_tok_s_decode=80.731186
aggregate_continuation_tok_s_decode=88.526212
total_wall_ms=1562.212280
aggregate_generated_tok_s_wall=20.483772
```

## Interpretation

This resolves the previous Spike B/C graph failure mode:

```text
store_f32_device_to_f8_kv_rows_kernel:
operation would make the legacy stream depend on a capturing blocking stream
```

That failure came from recapturing across token-major steps. With persistent graph execs, step 0 captures once per layer and subsequent steps replay cached graphs.

The result is not yet a throughput win. It is an operational milestone: persistent replay now works end-to-end for the TP/EP token-major path, which gives us a stable base for CUDA graph replay, NCCL wiring, kernel selection, and later MTP.

## Next Work

- Run the same persistent replay path through the permanent profile harness with GPU utilization sampling.
- Add a longer continuation benchmark now that recapture is gone.
- Combine persistent replay with the NCCL current-HC/allgather and compact MoE gates.
- Use nsight/nvprof on the replay window to identify the dominant kernels inside the cached graph.
