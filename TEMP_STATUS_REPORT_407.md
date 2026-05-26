# TEMP_STATUS_REPORT_407

Date: 2026-05-26

## Current Focus

Sprint 407 moved lazy output-head from direct diagnostics into the HTTP serving
loop so the prototype appliance can answer chat requests without keeping the
output head resident at startup.

## Implementation

Changed `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

- lazy output-head now opens when `serving_result` is requested, not only for
  direct `--serving-bench`;
- HTTP prefill disables `diagnostic_output_head` and
  `diagnostic_output_head_lazy_gate`, so prefill updates state without logits;
- resident output-head behavior is unchanged when lazy mode is off.

## V100 Results

Artifacts:

- `logs/from-cluster/sprint407-http-lazy-output-head/http-lazy-control/`
- `logs/from-cluster/sprint407-http-lazy-output-head/http-lazy-hc-nccl/`

| Case | HTTP 200 | First token | Server decode tok/s | Continuation decode tok/s | Client tok/s | Min free VRAM | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| HTTP lazy control | 32/32 | 83480 | 108.683003 | 108.261807 | 5.031959 | 1018 MiB | Prototype serving works |
| HTTP lazy + HC-current NCCL | 32/32 | 83480 | 110.879994 | 109.438988 | 5.594779 | 386 MiB | Serves, but reserve fails |

Response 0 in both runs generated `[83480, 79768]`, returned
`diagnostic_output_head=1`, and advanced cache position with resident KV/HC
metadata.

## Decision

Promote HTTP lazy output-head as the prototype serving path.

Keep HC-current NCCL diagnostic-only. It serves correctly at the target shape,
but `nccl_after_lazy_output_head` has only `386 MiB` free on GPU0 and all
eight GPUs fail the `1536 MiB` reserve.

## Next

Continue NCCL production admission:

- reduce output-head peak memory;
- consider streaming/vocab-chunked output-head projection or quantized logits;
- rerun HTTP readiness/parity with GPU sampling once reserve is admitted;
- only then promote HC-current NCCL default behavior.
