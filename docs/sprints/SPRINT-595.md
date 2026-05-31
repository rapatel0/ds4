# Sprint 595 - MTP raw-SWA attention CPU oracle

Date: 2026-05-30

## Why This Sprint Exists

Sprint 594 showed the MTP layer-43 activation path is live: prologue output,
raw-SWA rows, attention heads, post-attention state, next hidden, and final HC
are finite and input-responsive. Acceptance is still `0/71`, and the raw-row
frontier looked coherent, so another token-level or stats-only probe is unlikely
to localize the bug.

The next useful boundary is the exact raw-SWA attention computation. Upstream
`research/ds4/ds4.c` computes sink-aware attention as:

1. score each visible raw row with `dot(q, kv) / sqrt(head_dim)`.
2. include `attn_sinks[h]` in the softmax denominator but not in the value sum.
3. return the weighted sum of the same raw KV rows.

The V100 path should produce the same `d_attn_heads` for layer 43, slot 0, and
rank-local heads, modulo float reduction/order noise. A same-logical-point CPU
oracle can prove or reject this boundary without needing a full Metal oracle.

## Scope

1. Add temporary MTP-only diagnostic code after `run_layer(43)` to copy, for
   rank 0 / slot 0:
   - `ops->attn_q_b.d_out[0]` for the rank-local Q heads.
   - `ranks[0].d_attn_raw_swa` for the visible raw window.
   - `ranks[0].d_attn_sinks` for rank-local sink logits.
   - `ranks[0].d_attn_heads` for the GPU result.
   - `position`, physical `raw_row`, and `valid_rows`.
2. Compute the CPU reference using the upstream sink-aware formula and the same
   visible raw row order expected by `attention_raw_swa_window_kernel`.
3. Emit a compact mismatch line with `max_abs`, `mean_abs`, first bad index,
   head/element coordinates, and a small value sample.
4. Run the deterministic MTP acceptance harness once with the diagnostic build.
5. If the oracle fails, inspect the mismatch pattern and test one minimal
   temporary candidate only if the pattern clearly identifies it:
   - raw window order / wrap order.
   - sink sign or sink placement in denominator.
   - scale factor.
6. Remove all temporary oracle/probe code before commit unless a minimal durable
   correctness fix is found.

## Non-Goals

- Do not add a permanent diagnostic flag.
- Do not run throughput A/B; acceptance is still zero.
- Do not build the K-wide verifier until the MTP draft has nonzero acceptance.
- Do not re-run dense F8, output-head HC slicing, HC-current, or raw-frontier
  probes unless this oracle contradicts their inputs.
- Do not modify or delete the MTP smoke cluster in this sprint.

## Definition of Done

- Temporary raw-SWA attention oracle build passes on the pod.
- The deterministic harness emits oracle evidence at layer 43.
- The sprint records whether raw-SWA attention matches the CPU reference or
  identifies the first concrete semantic mismatch.
- Any temporary candidate has deterministic acceptance evidence and is either
  promoted as a narrow durable fix or removed.
- Temporary diagnostic code is removed before commit unless promoted.
- Clean source is recopied to `/workspace/s573-continuation-instrument` and the
  clean pod build passes.
- `VISION.md` and `SPIKE_B_STEERING.md` record the result and the next target.

## Result

Status: COMPLETE. No code was promoted.

The first oracle attempt copied `ranks[0].d_attn_sinks` after `run_layer(43)`.
That failed with `max_abs` roughly `0.4-1.0`, but the failure was diagnostic:
the per-rank sink staging buffer can be overwritten by later rank broadcasts
after rank 0 has already used its own slice. The oracle was corrected to copy
rank-0 sinks directly from `shared_hc_controls->d_attn_sinks[43]`.

With the corrected source, the raw-SWA attention oracle passed at every sampled
frontier in the deterministic harness:

```text
position 0  valid_rows 1   max_abs 1.90734863e-06  mean_abs 9.0171218e-08   PASS
position 1  valid_rows 2   max_abs 9.53674316e-07  mean_abs 1.27611884e-07  PASS
position 2  valid_rows 3   max_abs 1.90734863e-06  mean_abs 2.37144739e-07  PASS
position 3  valid_rows 4   max_abs 2.74181366e-06  mean_abs 1.8414454e-07   PASS
position 7  valid_rows 8   max_abs 3.57627869e-06  mean_abs 3.17674989e-07  PASS
position 15 valid_rows 16  max_abs 1.90734863e-06  mean_abs 1.80809423e-07  PASS
position 31 valid_rows 32  max_abs 1.25169754e-06  mean_abs 1.89170294e-07  PASS
```

The acceptance harness remained rejected:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12]  [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
```

Conclusion: layer-43 rank-local raw-SWA attention math, sink placement, scale,
window order, and raw row selection match the CPU reference to float noise for
the sampled MTP draft points. Do not re-test raw-SWA attention unless its inputs
change. The remaining likely body-level targets are attention-output projection
and the post-attention/FFN handoff.

Temporary oracle code was removed from `engine/token_major_loop.cu`. Clean source
was recopied to `/workspace/s573-continuation-instrument`, and the clean pod
build passed:

```text
BUILD_EXIT=0
make appliance/ds4-v100-tp-ep-appliance
```
