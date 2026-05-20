# Sprint 092: Appliance Multi-Slot Async Soak

## Goal

Benchmark the full TurboMind appliance directory through the operator-facing
service path under multi-slot async load.

## Implementation Plan

- Add `--appliance-dir` support to `tools/ds4-v100-appliance-soak.sh`.
- Launch the existing full appliance artifact from k8s-local storage:
  `/workspace/ds4-appliance-full-tm-s090`.
- Run a 4-slot, active-microbatch-4 HTTP soak at 1M context with async pipeline
  mode `auto`.
- Capture correctness, aggregate tok/s, request latency, startup behavior, and
  sampled GPU utilization.

## Definition Of Done

- [x] Soak harness accepts a prepacked appliance directory.
- [x] Cluster soak returns expected first token hex `3136` for all requests.
- [x] Cluster soak reports aggregate generated and continuation tok/s.
- [x] GPU utilization sample is captured.
- [x] Cluster log is committed.

## Result

Sprint 092 benchmarked the full Sprint 090 TurboMind appliance through the
operator-facing launcher path under 4-slot async load.

Harness changes:

- Added `--appliance-dir` to `tools/ds4-v100-appliance-soak.sh`.
- Removed the harness dependency on Python so it runs inside the current
  cluster pod, which only has Bash, Perl, awk, and CUDA tools.
- Added one untimed warmup request by default. This matters because four
  concurrent cold first requests all failed with
  `layer 7 decode failed: TurboMind routed FFN failed` while the runtime was
  lazily loading model tensors into the CUDA cache. A single warmup request
  loads the resident appliance path first; the timed concurrent requests then
  pass.

Cluster validation:

```text
appliance_dir=/workspace/ds4-appliance-full-tm-s090
ctx=1048576
slots=4
active_microbatch=4
queue_policy=sequential
async_pipeline_mode=per-step
warmup_requests=1
timed_requests=4
generated_tokens=64
continuation_tokens=60
token_match=4/4
errors=0
elapsed_s=5.685832
latency_ms_avg=5641.968
aggregate_generated_tokens_per_second=11.256048
aggregate_continuation_tokens_per_second=10.552545
```

The service status confirms the timed batch used the tensor-batched slot path:

```text
tensor_batched_groups=1
tensor_batched_requests=4
tensor_batched_tokens=64
async_pipeline_decode=true
async_pipeline_mode=per-step
```

The result is correct and appliance-resident, but it is still far below the
practical serving target. The next optimization work should focus on why a
4-slot tensor-batched group only reaches ~11 generated tok/s after warmup:
stage scheduling, per-token request-state handling, TurboMind routed FFN
occupancy at small effective M, and whether the current batching path is still
serializing too much work between slots.

## Stop Conditions

- Stop if the appliance server cannot launch from
  `/workspace/ds4-appliance-full-tm-s090`.
- Stop if any request returns the wrong first token.
- Stop if the cluster reports insufficient free VRAM before serving starts.
