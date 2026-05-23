# Sprint 242 - TP/EP Fused Remote-Sum Compose A/B

Date: 2026-05-23
Status: Planned

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

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] Implementation stays in the separate TP/EP codepath.
- [ ] No PP scheduler files are modified.
- [ ] `--fuse-compose-sum` builds on the V100 pod.
- [ ] Baseline unfused FP32 return still passes.
- [ ] Fused FP32 return candidate passes finite/checksum checks.
- [ ] A/B evidence records `ms_per_step`, `slot_step_tok_s`, and stage timings.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint242-tp-ep-fused-compose/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- Fusing sum into compose may not improve if peer-copy synchronization dominates.
- The fused kernel currently hardcodes eight remote sources, matching TP8.
- If the fused path changes floating-point summation order, checksum may differ
  while finite validation still passes.

## Decision

Pending.
