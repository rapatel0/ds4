# Sprint 241 - TP/EP FP16 EP Return A/B

Date: 2026-05-23
Status: Complete

## Overview

Sprint 240 introduced a resident TP/EP layer-loop benchmark and showed the
representative loop is dominated by scalar dense kernels plus compose/peer
synchronization, not TurboMind EP alone. The easiest isolated communication
optimization is the EP return path: Sprint 239/240 returned expert
contributions as FP32 for observability, even though the inputs and TurboMind
outputs are half.

Sprint 241 adds an opt-in FP16 EP return path and A/B benchmarks it against the
current FP32 return path at `32` slots / `256K`, MTP off.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add `--ep-return-fp16` to the separate TP/EP full-layer smoke.
- Preserve FP32 local accumulation before the peer boundary.
- Convert each source/destination EP contribution shard to FP16 before peer
  copy.
- Peer-copy FP16 shards across all eight GPUs.
- Re-expand FP16 remote shards to FP32 when summing on destination ranks.
- Report EP return dtype and byte count in the one-shot composition and
  resident decode-loop summaries.
- Run A/B on the V100 pod:
  - FP32 return baseline;
  - FP16 return candidate.
- Record correctness and throughput deltas.

## Non-Goals

- No PP scheduler edits.
- No changes to `ds4_v100_scheduler.*`.
- No server/API integration.
- No MTP.
- No logits equivalence claim.
- No dense HMMA/CUTLASS work in this sprint.
- No quantization below FP16 for EP return.

## Architecture

The current Sprint 240 return path is:

```text
d_down half
  -> FP32 local contribution buffer [source][dest][slot][hidden_shard]
  -> FP32 peer copy to destination
  -> FP32 sum
  -> FP32 next-hidden compose
```

The Sprint 241 candidate path is:

```text
d_down half
  -> FP32 local contribution buffer
  -> FP16 cast buffer [source][dest][slot][hidden_shard]
  -> FP16 peer copy to destination
  -> FP32 sum after half->float expansion
  -> FP32 next-hidden compose
```

This halves the EP return peer payload at `32` slots:

```text
FP32 aggregate return = 8 * 8 * 32 * 512 * 4 B = 4194304 B
FP16 aggregate return = 8 * 8 * 32 * 512 * 2 B = 2097152 B
```

The expected speedup may be modest because Sprint 240 suggests synchronization
and kernel boundaries matter more than raw NVLink bandwidth. The point of this
sprint is to measure that directly and make FP16 return available if it is
neutral or better.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | FP16 EP return option and benchmark reporting |
| `docs/sprints/SPRINT-241.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint241-tp-ep-fp16-return/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Implementation stays in the separate TP/EP codepath.
- [x] No PP scheduler files are modified.
- [x] `--ep-return-fp16` builds on the V100 pod.
- [x] FP32 return baseline still passes.
- [x] FP16 return candidate passes finite/checksum checks.
- [x] FP16 run reports half EP return bytes.
- [x] A/B evidence records `ms_per_step`, `slot_step_tok_s`, and stage timings.
- [x] Evidence is copied to
      `logs/from-cluster/sprint241-tp-ep-fp16-return/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Risks

- FP16 return may not improve wall time if synchronization dominates.
- FP16 return changes the checksum and may need a different correctness
  expectation than exact FP32 return.
- Additional cast kernels may offset the lower peer-copy payload.

## Decision

Complete. `--ep-return-fp16` is implemented and validated in the separate
TP/EP path, but it should stay opt-in rather than becoming the default.

Same-binary A/B on the V100 pod at `32` slots / `256K`, MTP off, `50`
resident decode-loop steps:

| Mode | Return bytes | ms/step | Slot-step tok/s | EP ms/step | Dense ms/step | Compose ms/step | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| FP32 return | 4194304 | 1.788149 | 17895.603225 | 0.321324 | 0.752902 | 0.713836 | PASS |
| FP16 return | 2097152 | 1.937399 | 16516.992775 | 0.317810 | 0.759782 | 0.859697 | PASS |

FP16 return halves the peer payload and preserves finite/checksum validation,
but the standalone cast-to-half and half-to-float expansion kernels make the
resident loop slower by about `8.3%` in this implementation. The result
confirms the Sprint 240 diagnosis: at this message size, synchronization and
extra kernel boundaries matter more than raw NVLink payload bytes.

Decision: keep FP32 return as the default for this smoke. Keep
`--ep-return-fp16` as a diagnostic and revisit only if the conversion is fused
into the EP reduction or next-hidden compose kernel.

Evidence:

- `logs/from-cluster/sprint241-tp-ep-fp16-return/layer2-decode-loop-fp32-return-32slot-256k-50steps.log`
- `logs/from-cluster/sprint241-tp-ep-fp16-return/layer2-decode-loop-fp16-return-32slot-256k-50steps.log`
