# Sprint 242 - TP/EP Fused Remote-Sum Compose A/B

Date: 2026-05-23
Status: Complete

## Overview

Sprint 241 showed that reducing EP return payload from FP32 to FP16 is correct
but slower as a standalone optimization because extra cast/expand kernels cost
more than the saved peer-copy bytes. The next measured bottleneck is
compose/synchronization. Sprint 240's resident loop spends about
`0.71 ms/step` in compose for FP32 return.

Sprint 242 removes the standalone destination `ep_sum` zero/add passes by
fusing remote contribution summation directly into the next-hidden compose
kernel.

## Goals

- Keep the hard cut: no PP/layer-split variant work.
- Add an opt-in `--fuse-compose-sum` mode to the separate TP/EP full-layer
  smoke.
- Preserve the current FP32 return default.
- In fused mode, compute:

  ```text
  next_hidden = residual + attn + shared + 0.125 * sum(remote_ep_contrib[src])
  ```

  in one CUDA kernel per destination rank.
- Support the FP32 return path first.
- Report whether fused compose/sum is enabled.
- A/B benchmark at `32` slots / `256K`, MTP off, `50` resident steps:
  - baseline FP32 return, unfused compose;
  - FP32 return, fused compose/sum.

## Non-Goals

- No PP scheduler edits.
- No changes to `ds4_v100_scheduler.*`.
- No server/API integration.
- No MTP.
- No logits equivalence claim.
- No dense HMMA/CUTLASS work in this sprint.
- No FP16 return fusion unless trivial after the FP32 fused path works.

## Architecture

Current compose path:

```text
zero ep_sum
for src in 0..7:
  ep_sum += remote[src]
next = residual + attn + shared + 0.125 * ep_sum
```

Fused path:

```text
next[i] =
  residual(i) +
  attn[i] +
  shared[i] +
  0.125 * (
    remote0[i] + remote1[i] + ... + remote7[i]
  )
```

This removes one zero kernel and eight add kernels per destination rank per
step. It should reduce launch/sync pressure even if the raw memory traffic is
similar.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-full-layer-smoke.cu` | Fused compose/sum option and reporting |
| `docs/sprints/SPRINT-242.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint242-tp-ep-fused-compose/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] Implementation stays in the separate TP/EP codepath.
- [x] No PP scheduler files are modified.
- [x] `--fuse-compose-sum` builds on the V100 pod.
- [x] Baseline unfused FP32 return still passes.
- [x] Fused FP32 return candidate passes finite/checksum checks.
- [x] A/B evidence records `ms_per_step`, `slot_step_tok_s`, and stage timings.
- [x] Evidence is copied to
      `logs/from-cluster/sprint242-tp-ep-fused-compose/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Risks

- Fusing sum into compose may not improve if peer-copy synchronization dominates.
- The fused kernel currently hardcodes eight remote sources, matching TP8.
- If the fused path changes floating-point summation order, checksum may differ
  while finite validation still passes.

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
dense_compute_all=on
compose_next_hidden=on
```

Logs:

- `logs/from-cluster/sprint242-tp-ep-fused-compose/layer2-decode-loop-unfused-compose-32slot-256k-50steps.log`
- `logs/from-cluster/sprint242-tp-ep-fused-compose/layer2-decode-loop-fused-compose-32slot-256k-50steps.log`

| Mode | Fused sum | ms/step | Slot-step tok/s | EP ms/step | Dense ms/step | Compose ms/step | Checksum | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FP32 return baseline | 0 | 1.784008 | 17937.138290 | 0.316384 | 0.753850 | 0.713663 | 2382924023 | PASS |
| FP32 return fused compose/sum | 1 | 1.641832 | 19490.418145 | 0.317783 | 0.755056 | 0.568906 | 2382924023 | PASS |

The one-shot compose gate also improved:

| Mode | Compose ms | Checksum | Result |
|---|---:|---:|---|
| Unfused | 2.578263 | 4112649481 | PASS |
| Fused compose/sum | 2.168121 | 4112649481 | PASS |

## Decision

Promote `--fuse-compose-sum` as the default direction for the TP/EP smoke
path, while keeping it explicit until server integration. It removes one zero
kernel and eight add kernels per destination rank in the FP32 EP return path,
preserves checksum and finite validation, reduces compose time by about
`20.3%`, and improves the representative resident layer-loop metric by about
`8.7%`.

This confirms Sprint 241's diagnosis: the useful lever is not standalone
payload quantization; it is removing extra synchronization and fusing the
peer-return reduction into the next hidden compose boundary.
