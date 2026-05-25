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

## Progress

Phase 1 gate and harness plumbing is implemented.

Added:

- `--fused-gated-silu-gate`
- `DS4_V100_TP_EP_FUSED_GATED_SILU=1`
- `tools/ds4-v100-tp-ep-profile.py --fused-gated-silu`
- `tools/ds4-v100-tp-ep-profile.py --routed-ffn-norm-input`
- scaffold fields for `fused_gated_silu_gate`,
  `routed_ffn_norm_input_gate`, and `routed_gate_standalone_swiglu`

Local validation passed:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100 build passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct V100 evidence at `32` slots / `256K` / `position=262080` /
`1` generated token:

| Branch | Mode | First token | Decode tok/s | EP ms | Standalone SwiGLU |
|---|---|---:|---:|---:|---:|
| current serving-shaped | control | 54639 | 68.313313 | 16.873766 | 0 |
| current serving-shaped | fused flag | 54639 | 67.795224 | 17.999301 | 0 |
| routed-normalized | clamped control | 41432 | 45.368432 | 34.189967 | 1 |
| routed-normalized | generic fused epilogue | 54639 | 57.367413 | 27.038617 | 0 |

Current finding: the production-shaped model-router compact-MoE branch already
has no standalone routed SwiGLU launch, so the fused flag is a parity-clean
no-op there. The routed-normalized branch does contain the standalone clamped
SwiGLU launch, and the generic TurboMind fused epilogue removes it and improves
the direct proxy, but changes the first token. The generic epilogue is not
DS4-equivalent.

A narrowly scoped DS4-clamped TurboMind ABI was then implemented:

```text
ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens
```

It preserves the existing generic ABI and adds an opt-in epilogue bit for:

```text
gate = min(gate, 10)
up   = clamp(up, -10, 10)
out  = silu(gate) * up
```

V100 validation after the ABI change:

```text
cmake --build build/turbomind-v100 --target ggml-turbomind -j80
nm -D build/turbomind-v100/libggml-turbomind.so | grep ggml_turbomind_mul_mat_grouped_gated_silu_clamped_total_tokens
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

The exported symbol exists and the full-layer smoke harness rebuilds.

Layer-0 EP-only V100 evidence at `32` slots / `256K` / `position=262080` /
model-router routes / routed-normalized input:

| Mode | Worst gate ms | Worst down ms | Worst EP ms | Result |
|---|---:|---:|---:|---|
| two-step clamped gate | 4.102144 | 0.223232 | 4.254720 | PASS |
| fused DS4-clamped gate | 0.622592 | 0.181248 | 0.803840 | PASS |

That proves the clamped fused ABI launches and removes the expensive
standalone clamped SwiGLU boundary in isolation.

Resident serving-shaped A/B did not complete. With
`--routed-ffn-norm-input --fused-gated-silu`, both candidate attempts failed at
layer 0 before the routed gate executed:

```text
returncode: 4
tp_ep_token_major_item step 0 layer 0 rc 4 FAIL
scaffold_pass_invocations: 0
```

The failure is the resident `ds4_v100_tp_runtime_dense_kv_slice` precheck
returning non-zero `max_abs` before `run_decode_loop()` reaches
`run_gate_selected()`. A same-binary routed-normalized control rerun passes
with first token `41432` and `59.381346` direct generated tok/s. The
serving-shaped fused flag without routed-normalized input also passes with
first token `54639` and `68.824485` direct generated tok/s.

## Outcome

Decision: keep `--fused-gated-silu-gate` default-off and diagnostic-only; do
not promote.

The generic fused epilogue is rejected for correctness. The DS4-clamped
TurboMind ABI is promising in the EP-only shape, but Sprint 379 cannot claim
serving correctness or throughput because the resident direct A/B fails before
the candidate gate executes. No HTTP A/B was run after that blocker because
direct serving parity is the prerequisite.

Follow-up: diagnose the resident dense-KV precheck interaction under
`routed-normalized + fused-gated-silu`, or add a deterministic fused-gate
output parity harness before attempting another serving promotion.
