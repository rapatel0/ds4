# Sprint 379: Fused Gated-SiLU Gate

## Overview

Implement the next Vision gate after Sprint 378:
`--fused-gated-silu-gate`.

Sprint 378 promoted compact MoE return composition for real model-router
routes. The next measured bottleneck is the routed-FFN activation boundary:
the correctness path currently runs routed gate/up as a grouped MXFP4 GEMM,
writes a full fp16 gate/up intermediate, then launches a separate DS4
clamp/SwiGLU kernel before the routed down GEMM.

This sprint tests whether that boundary can be removed by using a fused
TurboMind gated-SiLU epilogue path with DS4-compatible clamp semantics, while
keeping the source quantized expert layout and the promoted compact-MoE
serving baseline.

## Scope

- Add a default-off TP/EP CLI gate:

```text
--fused-gated-silu-gate
```

- Add launcher/profile plumbing:

```text
DS4_V100_TP_EP_FUSED_GATED_SILU=1
tools/ds4-v100-tp-ep-profile.py --fused-gated-silu
```

- Extend or select a TurboMind routed gate/up path that produces the down-GEMM
  input directly:

```text
current:   MXFP4 gate/up GEMM -> fp16 gate_up -> clamp/SwiGLU kernel -> fp16 gated
candidate: MXFP4 gate/up GEMM + DS4 clamp/SwiGLU epilogue -> fp16 gated
```

- Preserve the current default path unchanged.
- Use Sprint 378's model-router compact-MoE serving path as the candidate
  baseline.
- Validate on the V100 pod with same-binary direct and HTTP A/B.

## Out Of Scope

- No PP/layer-split work.
- No generic scheduler abstraction.
- No TP-sharded expert topology rewrite.
- No MTP in this sprint.
- No broad dtype conversion or offline repack requirement beyond the existing
  fused gate/up expert layout.
- No CUDA graph or P2P transport rewrite.

## Architecture

The existing accurate routed path is:

```text
rank.d_a [routes, 4096] fp16
  -> api.mmgt(... ffn_gate_up_exps.weight ...)
rank.d_gate_up [routes, 4096] fp16
  -> routed_fused_gate_up_swiglu_clamp_kernel(clamp=10.0, optional route scale)
rank.d_gated [routes, 2048] fp16
  -> api.mmgt(... ffn_down_exps.weight ...)
rank.d_down [routes, 4096] fp16
```

The desired candidate path is:

```text
rank.d_a [routes, 4096] fp16
  -> TurboMind MXFP4 grouped GEMM with DS4 clamp/SwiGLU epilogue
rank.d_gated [routes, 2048] fp16
  -> existing routed down GEMM
rank.d_down [routes, 4096] fp16
```

The important correctness detail is DS4's routed SwiGLU clamp:

```text
gate = min(gate, 10.0)
up   = clamp(up, -10.0, 10.0)
out  = silu(gate) * up
```

When the reference path applies per-route input rescale through
`route_inv_scale`, the fused candidate must either support that scale in the
epilogue or stay disabled for that mode. The sprint should not silently use the
unclamped generic gated-SiLU path if it changes tokens or checksums.

## Implementation Plan

### Phase 1: Gate And Harness Plumbing

- Add `Options::fused_gated_silu_gate`.
- Parse `--fused-gated-silu-gate`.
- Add `DS4_V100_TP_EP_FUSED_GATED_SILU`.
- Add `tools/ds4-v100-tp-ep-profile.py --fused-gated-silu`.
- Emit the gate in scaffold/profile summaries.
- Keep default behavior unchanged.

### Phase 2: TurboMind ABI Probe

- Inspect the existing TurboMind gated-SiLU ABI and DS4 probe kernels.
- Prefer adding a narrowly named DS4 clamped epilogue ABI instead of changing
  generic `ggml_turbomind_mul_mat_grouped_gated_silu_total_tokens` semantics.
- Candidate ABI should accept the same grouped expert inputs plus:
  - clamp value;
  - optional per-route inverse scale pointer, or a documented rejection when
    route scaling is active.
- Fall back to the current two-step accurate path on any non-zero return.

### Phase 3: Direct Correctness And Timing

- Build the TurboMind library and TP/EP full-layer smoke on gpu-01.
- Run direct-token-major A/B:
  - `32` slots;
  - `256K` context;
  - `position=262080`;
  - model-router routes;
  - compact-MoE decode enabled;
  - control: current clamped path;
  - candidate: fused gated-SiLU gate.
- Require first token and all-layer checksum parity.
- Record gate/up, down, EP, compose, and decode timing.

### Phase 4: HTTP Serving A/B

- Run `/v1/chat/completions` A/B at:
  - `32` concurrent requests;
  - `32` configured slots;
  - `256K` context;
  - `position=262080`;
  - `32` generated tokens/request;
  - GPU sampling enabled.
- Compare response token streams, client tok/s, server decode tok/s, average
  and max GPU utilization, routed FFN timing, and compose timing.

### Phase 5: Record Decision

- Update `TEMP_STATUS_REPORT_379.md`.
- Update `docs/sprints/STATUS.md`.
- Update `docs/sprints/VISION.md`.
- Record explicit `PROMOTE`, `KEEP-OPT-IN`, or `REJECT`.
- Commit the code, sprint doc, and status artifacts.

## Definition Of Done

- `--fused-gated-silu-gate` builds and defaults off.
- Launcher/profile plumbing exists and defaults off.
- Candidate path never changes the default serving behavior unless explicitly
  enabled.
- Candidate direct V100 A/B preserves first token and all-layer checksum, or
  the sprint records a concrete correctness blocker.
- Candidate HTTP A/B preserves response token streams, or the sprint records a
  concrete correctness blocker.
- Report includes client tok/s, server decode tok/s, stage timing, GPU
  utilization, and whether the standalone clamp/SwiGLU launch was removed.
- Decision is recorded in this sprint doc, `TEMP_STATUS_REPORT_379.md`,
  `docs/sprints/STATUS.md`, and `docs/sprints/VISION.md`.

## Decision Rule

Promote if the fused candidate preserves token/checksum parity and improves
server decode tok/s or average GPU utilization at the real `32` slot / `256K`
model-router compact-MoE serving shape.

Keep opt-in if correctness holds but performance is flat/noisy.

Reject if the candidate changes token/checksum behavior, cannot preserve DS4
clamp/route-scale semantics, or regresses serving throughput.

## Risks

- Generic TurboMind gated-SiLU may not implement DS4 clamp semantics today.
- Route input rescale may require an epilogue input that the existing
  TurboMind operation contract does not expose.
- Removing one launch may be too small to move HTTP topline if the routed down
  GEMM or all-to-all remains dominant.
- The real serving shape may have route counts that do not match the fastest
  fixed-shape TurboMind probes.
