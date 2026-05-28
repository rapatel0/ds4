# Sprint 416: Rank-Local Attention Projection Input

Date: 2026-05-27

## Objective

Reduce graph-captured full-hidden copy work in the TP/EP persistent graph path
without reopening peer-copy capture. The first target is attention projection
input: keep the canonical device-0 normalized hidden for downstream semantic
paths, but let each rank normalize and consume its local full-hidden replica
directly for `attn_q_a` and `attn_kv_latent` input packing.

This sprint remains TP/EP-only. No PP/layer-split variants.

## Rationale

`TEMP_STATUS_REPORT_418.md` rejected `cudaMemcpyPeerAsync` inside CUDA graph
capture:

```text
operation not permitted when stream is capturing
```

The next useful path is therefore reducing graph-safe copy kernels, not
replacing them with peer copies. Attention projection currently computes
`hc->d_attn_normed` on device 0, then graph-copies that full hidden tensor to
every rank before filling dense inputs. When the HC-current path has already
materialized `r.d_current_full` on each rank, those copies can be avoided by
doing the RMS norm locally on each rank and filling the dense inputs from that
rank-local normalized buffer.

## Implementation

1. Add a default-off gate:

```text
--true-ds4-attention-projection-rank-local-input-gate
```

2. Replicate the small `attn_norm.weight` tensor to each rank during
`SharedHcControls` initialization when the gate is enabled.

3. In `run_true_attention_projection`:
   - keep the existing device-0 `hc->d_attn_normed` computation for downstream
     compressed-KV / semantic consumers
   - under the new gate, run the same RMS norm on each rank using
     `r.d_current_full`
   - fill `attn_q_a` and `attn_kv_latent` inputs from that rank-local normalized
     buffer
   - avoid the graph-safe full-hidden copy from `hc->d_attn_normed` to every
     rank for this projection family

4. Wire the gate through:
   - `tools/ds4-v100-run-appliance.sh`
   - `tools/ds4-v100-tp-ep-profile.py`
   - `deploy/v100/ds4-v100-appliance.env.example`

## Definition of Done

- Local syntax checks pass.
- V100 build passes.
- Resident layer-2 graph run passes with the new gate.
- Compare baseline vs rank-local gate on resident layer 2 for:
  - first token/checksum or existing resident PASS invariants
  - `decode_ms_per_step`
  - graph capture/replay success
- If resident layer passes, run the full all-layer deferred-NCCL 8-slot direct
  decode shape and record whether it passes or where it fails.
- Update `TEMP_STATUS_REPORT_419.md` and `docs/sprints/VISION.md` with the
  measured result.

## Risks

- Rank-local normalization is redundant compute versus one device-0 norm, but
  it removes full-hidden graph copies. The gate should remain default-off until
  measured.
- Downstream compressed-KV still uses the canonical `hc->d_attn_normed`; this
  sprint intentionally attacks only one copy family.
- If the gate changes tokens or checksums, keep it diagnostic-only and use the
  result to choose the next rank-local family.

## Outcome

Status: complete for direct-decode validation; HTTP serving promotion remains
follow-up work.

Implemented and wired:

- `--true-ds4-attention-projection-rank-local-input-gate`
- launcher/profile/env exposure
- per-rank `attn_norm.weight` replication
- rank-local RMS norm plus `attn_q_a` / `attn_kv_latent` input fill

Validation:

- V100 build passed.
- Resident layer 2 passed with unchanged checksum `8290057485`.
- Resident layer 2 improved from `2.476288` ms/step to `2.304768` ms/step
  with graph replay success unchanged.
- All-layer direct decode passed with unchanged checksum `4335215310`.
- Clean same-binary all-layer A/B at `slots=8`, `ctx=262144`,
  `decode_steps=4`, scratch `256 MiB`, deferred NCCL, and persistent graph
  replay improved generated decode from `84.072506` to `92.702737` tok/s.
- Continuation decode improved from `94.326524` to `105.428529` tok/s.
- Capture/replay stayed at `43/43` and `172/172`.

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/resident-layer2-baseline/
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/resident-layer2-ranklocal-rebuilt/
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/ab-clean-baseline-slot8-tokens4-scratch256/
/localpool/ds4/workspace/logs/sprint416-rank-local-attn-proj/ab-clean-ranklocal-slot8-tokens4-scratch256/
```

Decision:

- Keep direct remote-source attention projection fill rejected; it reduced graph
  nodes but regressed the all-layer path.
- Treat rank-local attention projection input as the next HTTP serving
  candidate, not yet a production default.
- Record the expert-residency headroom issue: current all-layer shared expert
  packing can report `147169738752` aggregate bytes and the scratch-512 control
  OOMed during expert allocation. This is independent of the rank-local gate
  and should be handled by the next memory/planner sprint.

Detailed report:

```text
TEMP_STATUS_REPORT_420.md
```
