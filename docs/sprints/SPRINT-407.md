# Sprint 407: HTTP Lazy Output Head

Date: 2026-05-26

## Overview

Sprint 406 made HC-current NCCL operational at the direct `32` slot / `256K`
target shape by compacting attention compressed-KV state. The remaining gap is
that lazy diagnostic output-head only works for direct serving-bench runs.
HTTP serving still passes no output head when lazy mode is enabled, so request
generation cannot return selected tokens through the lazy path.

This sprint makes lazy/on-demand output-head usable in the HTTP serving loop.

## Constraints

- TP/EP only. No PP/layer-split work.
- Keep resident output-head behavior unchanged when lazy mode is off.
- Do not run output head during HTTP prefill steps.
- Preserve HTTP response generation and selected-token output.
- Keep HC-current NCCL diagnostic-only until VRAM reserve is admitted.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`

Changes:

1. Extend the lazy output-head condition in `run_token_major_serving_loop` so
   it can open a temporary output head whenever `serving_result` is requested,
   not only when `opt.serving_bench` is set.
2. Disable `diagnostic_output_head` and `diagnostic_output_head_lazy_gate` on
   HTTP prefill calls, because prefill updates KV/state but does not need
   logits.
3. Keep the existing resident output-head path unchanged.

## Validation

Local:

```text
git diff --check
```

V100:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run HTTP target-shape probes:

1. lazy output-head, non-NCCL, `32` requests / `32` slots / `256K`
2. lazy output-head + HC-current NCCL, `32` requests / `32` slots / `256K`

Record:

- HTTP status count
- first selected token
- server generated/continuation decode tok/s
- VRAM checkpoints
- whether `/status` reports ready and cache metadata

Artifacts:

- `logs/from-cluster/sprint407-http-lazy-output-head/http-lazy-control/`
- `logs/from-cluster/sprint407-http-lazy-output-head/http-lazy-hc-nccl/`

Results:

| Case | HTTP 200 | First token | Server decode tok/s | Continuation decode tok/s | Client tok/s | Min free VRAM | Reserve |
|---|---:|---:|---:|---:|---:|---:|---|
| HTTP lazy control | 32/32 | 83480 | 108.683003 | 108.261807 | 5.031959 | 1018 MiB | pass at 64 MiB |
| HTTP lazy + HC-current NCCL | 32/32 | 83480 | 110.879994 | 109.438988 | 5.594779 | 386 MiB | fails 1536 MiB NCCL reserve |

Both cases returned chat-completion responses with generated token sequence
`[83480, 79768]` for response 0, `diagnostic_output_head=1`, resident KV/HC
metadata, token input seeding, and cache position advancement. The lazy
output-head rows appear in `server.out`, proving that HTTP decode steps now
open and close the output head on demand.

The HC-current NCCL HTTP case is now operational at the target shape, but it
is still diagnostic-only because `nccl_after_lazy_output_head` leaves only
`386 MiB` free on GPU0 and all eight GPUs fail the `1536 MiB` reserve.

## Definition of Done

- Lazy output-head produces selected tokens in HTTP serving.
- Prefill does not invoke output-head.
- V100 build passes.
- HTTP lazy non-NCCL run succeeds.
- HTTP lazy + HC-current NCCL run is attempted and recorded.
- Sprint doc, status, vision, temporary report, and cluster artifacts are
  committed explicitly.

## Decision Gate

Promote HTTP lazy output-head as the diagnostic/prototype serving path if it
returns valid selected tokens and responses. Do not promote HC-current NCCL as
production default until the `1536 MiB` NCCL reserve passes at the target
shape.

## Decision

Promote HTTP lazy output-head as the prototype serving path for target-shape
TP/EP work.

Do not promote HC-current NCCL as production default yet. It serves correctly
and is slightly faster in this short HTTP probe, but it still lacks safe VRAM
reserve. Next work should reduce the output-head peak or lower output-head
residency cost so NCCL can pass admission with margin.
