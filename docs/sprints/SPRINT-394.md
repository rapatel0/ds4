# Sprint 394: Fast Hash-Router Gate

## Overview

Add a default-off TP/EP router-selection gate for the DS4 model-router hash
case.

The current model-router selection kernel computes softplus/sqrt probabilities
for all `256` global experts per active slot, then, when a router hash table is
present, keeps only the six hash-row experts. At the target decode shape this
is part of the measured HC-current/router boundary that remains around
`27-28 ms` per all-layer step after route-upload packing. DS4 Flash uses the
hash-row path in the production-shaped model-router run, so we can test a
narrow equivalent kernel that evaluates only the six selected hash experts.

## Scope

- Add `--router-hash-fast-gate` / `DS4_V100_TP_EP_ROUTER_HASH_FAST=1`.
- Keep the existing router kernel as the default and fallback.
- Use the fast kernel only when `hash`, `tokens`, and `hash_rows` are present.
- Preserve current hash semantics:
  - inactive slots emit `-1` selected experts and zero weights.
  - token rows outside the hash table map to row `0`.
  - selected expert IDs come directly from the hash row.
  - weights are `sqrt(softplus(logit[e]))`, normalized and scaled by `1.5`.
  - router bias is ignored on the hash path, matching the current kernel.
- Wire the gate through the profile harness and emitted scaffold metadata.
- Run same-binary V100 A/B at the target real-router compact-MoE shape if the
  cluster is available.

## Out Of Scope

- No PP/layer-split work.
- No top-k parallel kernel for the non-hash router fallback.
- No GPU route-plan promotion.
- No MTP work.
- No default promotion unless same-binary A/B preserves response parity and
  improves the serving topline or the measured router boundary.

## Definition Of Done

- The new CLI/env gate is implemented and default-off.
- The profile harness can enable the gate and gives the run a distinct suffix.
- The emitted summary includes the gate state.
- The binary builds for `sm_70` on the V100 pod.
- A direct or HTTP V100 A/B records first-token/checksum or response parity.
- Sprint docs record promote/reject with numbers.

## Risks

- If the production model-router path does not always have a hash row, the gate
  may only cover a subset of layers. The implementation must fall back to the
  existing kernel without changing semantics.
- This removes router selection math, but the broader path may still be
  dominated by dense logits, D2H route planning, or compose.

## Execution Plan

1. Implement the fast hash-router kernel and gate wiring.
2. Add profile harness support.
3. Build on the V100 pod.
4. Run same-binary target-shape A/B with readiness and response parity.
5. Promote only if parity holds and performance improves.

## Outcome

Complete. Added the default-off `--router-hash-fast-gate` /
`DS4_V100_TP_EP_ROUTER_HASH_FAST=1` path.

Implementation:

- Added `router_select_hash_fast_rows_kernel`, used only when router hash rows,
  router tokens, and `hash_rows > 0` are present.
- Preserved current hash-router semantics exactly: inactive-slot handling,
  token-row clamp to `0`, hash-row expert selection, `sqrt(softplus(logit))`
  weighting, normalization, and `1.5` scaling.
- Kept the original router select kernel as the default and fallback.
- Wired the gate through:
  - `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  - `tools/ds4-v100-run-appliance.sh`
  - `tools/ds4-v100-tp-ep-profile.py`
  - `tools/ds4-v100-tp-ep-active-slot-matrix.py`

## Validation

Local syntax/config:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py \
  tools/ds4-v100-tp-ep-active-slot-matrix.py \
  tools/ds4-v100-http-readiness-check.py

DS4_V100_TP_EP_ROUTER_HASH_FAST=1 ... tools/ds4-v100-run-appliance.sh \
  --print-command --allow-missing
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

Build passed with only existing unused-function warnings.

Same-binary V100 HTTP A/B:

Shape:

```text
32 requests / 32 slots / 256K ctx / position 262080 / 32 generated tokens
model-router routes / compact MoE / prompt-file soak / VRAM report
```

| Metric | Control | Router hash fast |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Response parity | `32/32` | `32/32` |
| Readiness | `true` | `true` |
| First token | `83484` | `83484` |
| Server decode tok/s | `106.900859` | `107.274556` |
| Client generated tok/s | `37.231411` | `38.262372` |
| Avg GPU util | `9.296875%` | `9.441489%` |
| Max GPU util | `50%` | `50%` |
| Router select ms | `27.766750` | `27.683134` |
| Route upload ms | `6.607606` | `6.709833` |
| HC-current FFN/router ms | `36.211953` | `36.287395` |
| Scaffold decode ms | `289.821429` | `293.484520` |
| Projected slot-step tok/s | `110.412816` | `109.034712` |
| Compressed-KV sum ms | `3285.935154` | `3317.395070` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

Permanent validators:

```text
response parity: match=true, matched_pairs=32, failed_pairs=0
control readiness: ready=true, failure_count=0
candidate readiness: ready=true, failure_count=0
```

## Decision

Do not promote `router-hash-fast` as a default. It is correct and keeps
serving readiness clean, but it does not materially move the intended router
boundary. The tiny server/client topline improvement is within the noise of
this harness, while scaffold decode and compressed-KV totals regress.

Keep the gate as an opt-in diagnostic. The result is useful: the expensive
router boundary is not caused by evaluating non-hash expert probabilities in
the select kernel. The next performance sprint should target the broader
HC-current/router scheduling boundary or remove the host route-planning path,
not micro-optimize hash selection.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint394-router-hash-fast/http-control`
  - `/workspace/logs/sprint394-router-hash-fast/http-candidate-router-hash-fast`
  - `/workspace/logs/sprint394-router-hash-fast/http-parity-summary.json`
- Local:
  - `logs/from-cluster/sprint394-router-hash-fast`
