# Sprint 243 - TP/EP Dense HMMA Compose Gate

Date: 2026-05-23
Status: Complete

## Overview

Sprint 242 reduced TP/EP compose synchronization by fusing the FP32 EP
remote-sum into next-hidden compose. The representative resident layer loop is
now dominated by dense F8 work plus the remaining compose/peer boundary:

```text
32 slots / 256K / 50 resident steps / MTP off
  EP:      0.317783 ms/step
  Dense:   0.755056 ms/step
  Compose: 0.568906 ms/step
  Total:   1.641832 ms/step
```

The current dense F8 implementation in the separate TP/EP smoke is a scalar
CUDA dot-product reference. That is useful for correctness, but it does not
represent the intended V100 execution path. The next gate is to use the prior
HMMA/CUTLASS-style lesson from the PP-era kernels: keep F8 bytes resident,
expand/dequantize into FP16 fragments inside the GPU, and feed Volta HMMA with
FP16 inputs and FP32 accumulation.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in HMMA dense path to the separate TP/EP full-layer smoke.
- Target the dense composition tensors used in the resident loop:
  - `blk.2.attn_output_b.weight`
  - `blk.2.ffn_down_shexp.weight`
- Preserve the default scalar path for A/B.
- Run the same 32-slot / 256K / 50-step resident loop with:
  - scalar dense + fused compose/sum control;
  - HMMA dense + fused compose/sum candidate.
- Report whether the HMMA dense path is selected and record stage timings.

## Non-Goals

- No PP scheduler edits.
- No `ds4_v100_scheduler.*` changes.
- No MTP.
- No server/API integration.
- No claim of final DS4 logits equivalence.
- No expanded-FP16 resident weight format as the production answer.

## Design

Add a bounded HMMA kernel for the TP/EP smoke:

```text
F8 packed row bytes in VRAM
  -> per-tile F8 decode + scale load
  -> FP16 fragments in shared/registers
  -> WMMA/HMMA FP16xFP16 -> FP32 accumulation
  -> FP32 output shard for the current smoke
```

The kernel should support the current `32`-slot target by tiling the token
dimension in `16`-token blocks. This avoids the old 16-token HMMA limitation
while preserving the same row-sharded TP output layout.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Opt-in HMMA dense compose path and reporting |
| `docs/sprints/SPRINT-243.md` | Plan and evidence |
| `docs/sprints/STATUS.md` | Status update |
| `docs/sprints/VISION.md` | Outcome update |
| `logs/from-cluster/sprint243-tp-ep-dense-hmma/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan is committed before implementation evidence.
- [x] Implementation stays in the separate TP/EP codepath.
- [x] No PP scheduler files are modified.
- [x] HMMA dense option builds on the V100 pod.
- [x] Scalar dense + fused compose/sum control still passes.
- [x] HMMA dense + fused compose/sum candidate passes finite/repeat checks.
- [x] A/B evidence records `ms_per_step`, `slot_step_tok_s`, dense stage time,
      compose stage time, and checksum.
- [x] Evidence is copied to
      `logs/from-cluster/sprint243-tp-ep-dense-hmma/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Risks

- HMMA accumulation changes numeric output versus the scalar FP32 reference.
  This sprint gates finite/repeat behavior and throughput, not final logits
  equivalence.
- The first HMMA kernel may improve dense time but expose another boundary as
  dominant.
- The kernel is a bounded layer-2 composition path, not the final all-layer
  dense implementation.

## Evidence

V100 pod: `llm/llamacpp-build-8gpu`

Command shape:

```text
slots=32
ctx=262144
top_k=6
layer=2
decode_steps=50
MTP=off
fuse_compose_sum=on
dense_compute_all=on
compose_next_hidden=on
```

Logs:

- `logs/from-cluster/sprint243-tp-ep-dense-hmma/layer2-decode-loop-scalar-dense-fused-compose-32slot-256k-50steps.log`
- `logs/from-cluster/sprint243-tp-ep-dense-hmma/layer2-decode-loop-hmma-dense-fused-compose-32slot-256k-50steps.log`

| Mode | Dense HMMA | ms/step | Slot-step tok/s | EP ms/step | Dense ms/step | Compose ms/step | Checksum | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Scalar dense + fused compose | 0 | 1.620386 | 19748.386791 | 0.309798 | 0.753941 | 0.556571 | 2382924023 | PASS |
| HMMA dense + fused compose | 1 | 3.533215 | 9056.907248 | 0.306812 | 2.667910 | 0.558428 | 442383 | PASS |

The one-shot compose gate also shows the HMMA dense candidate is slower:

| Mode | Dense HMMA | Attn dense ms | Shared dense ms | Compose ms | Result |
|---|---:|---:|---:|---:|---|
| Scalar dense + fused compose | 0 | 0.555418 | 0.153702 | 2.274831 | PASS |
| HMMA dense + fused compose | 1 | 1.762714 | 0.487322 | 2.111211 | PASS |

## Decision

Reject this naive HMMA dense candidate as a default and keep it as an explicit
diagnostic only. It is finite and repeat-stable, but it is much slower than
the scalar F8 dense reference at the target `32`-slot shape. The likely issue
is not HMMA itself; it is this first implementation's per-tile F8 decode and
shared-memory staging pattern, which reloads and converts packed weights too
often for the current row-sharded composition tensors.

The next dense optimization should not tune this kernel blindly. It should use
the older production HMMA evidence more directly: either adapt the existing
`ds4_cuda.cu` F8 HMMA paths that already encode shape-specific lessons, or use
a prepacked/streaming low-bit dense kernel where F8 decode is amortized and
software-pipelined across K tiles.
