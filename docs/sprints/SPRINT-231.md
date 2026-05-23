# Sprint 231 - TP/EP Routed Expert Slice

Date: 2026-05-23
Status: Planned

## Overview

Sprint 230 proved that the separate TP runtime can own all eight GPUs and
update sharded DS4 KV rows at the target `32` slot / `256K` shape. Sprint 231
adds the next missing bounded primitive: expert-parallel routed FFN execution
using the real low-bit TurboMind MXFP4 kernels, still outside the frozen
PP/layer scheduler.

This sprint is a TP/EP runtime gate, not a serving integration. It should make
the EP route distribution, local expert ownership, dispatch/return byte model,
and kernel behavior concrete at the production slot target.

## Goals

- Add a new TP/EP-only routed expert smoke tool.
- Own all eight V100s in one process and enable peer access.
- Model EP8 ownership as `32` experts per GPU from `256` total experts.
- Model `32` active slots with `top_k=6`, or `192` aggregate routes.
- Distribute routes across GPUs and report:
  - routes per GPU;
  - active local experts per GPU;
  - max routes per local expert;
  - worst-rank imbalance;
  - aggregate dispatch bytes and return bytes.
- Run the real TurboMind MXFP4 grouped gated-SiLU and down kernels on every
  GPU for the local routed rows.
- Validate finite output and deterministic repeat output on all GPUs.
- Measure per-rank gate/up latency, down latency, total local EP latency, and
  worst-rank latency.

## Non-Goals

- No PP scheduler changes.
- No generic PP/TP scheduler abstraction.
- No serving integration.
- No full DS4 layer correctness against logits.
- No dense TP attention implementation.
- No MTP.
- No production throughput claim.

## Implementation

1. Add `tools/ds4-v100-tp-ep-expert-smoke.cu`.
2. Reuse the public TurboMind C ABI from
   `kernels/turbomind/ggml-turbomind/include/ggml-turbomind-api.h`.
3. Generate deterministic MXFP4 fixtures for local experts.
4. Use `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens` for fused
   gate/up.
5. Use `ggml_turbomind_mul_mat_grouped_total_tokens` for down projection.
6. Use a fixed EP route plan by default:
   - `256` global experts;
   - `8` GPUs;
   - `32` experts per GPU;
   - `32` slots;
   - `top_k=6`;
   - balanced aggregate routes but non-uniform local expert density.
7. Add a Makefile target for the new tool.
8. Build on the V100 pod with `CUDA_ARCH=sm_70`.
9. Run the smoke against `./libggml-turbomind.so` for:
   - `32` slots / `top_k=6`;
   - a denser diagnostic case if time allows.
10. Copy evidence to
    `logs/from-cluster/sprint231-tp-ep-expert-slice/`.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp-ep-expert-smoke.cu` | TP/EP routed expert slice smoke |
| `Makefile` | build target |
| `docs/sprints/SPRINT-231.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint231-tp-ep-expert-slice/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] New smoke tool is TP/EP-only and does not modify PP scheduler files.
- [ ] Smoke uses the real TurboMind MXFP4 grouped gated-SiLU ABI.
- [ ] Smoke uses the real TurboMind MXFP4 grouped down ABI.
- [ ] Smoke reports route distribution and imbalance across all eight GPUs.
- [ ] Smoke reports dispatch bytes and return bytes for `32` slots / `top_k=6`.
- [ ] V100 build passes with `CUDA_ARCH=sm_70`.
- [ ] V100 smoke passes finite and deterministic repeat checks.
- [ ] Smoke reports per-rank and worst-rank latency.
- [ ] Evidence is copied to
      `logs/from-cluster/sprint231-tp-ep-expert-slice/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Risks

- The current TurboMind ABI is process-global enough that repeated `init` /
  `shutdown` semantics may not be safe across all devices. If that appears,
  this sprint should record the exact failure and narrow the next work to
  making the ABI multi-device safe.
- This validates EP kernel behavior and route density, not complete TP/EP
  serving. Full correctness still requires a one-layer TP/EP gate.
- A deterministic repeat check is weaker than a DS4 logits comparison, but it
  is enough for this bounded kernel/lifecycle sprint.

## Decision

Pending.
