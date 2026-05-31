# Sprint 596 - MTP attention-output and post-attention handoff oracle

Date: 2026-05-30

## Why This Sprint Exists

Sprint 595 cleared layer-43 raw-SWA attention: the GPU `d_attn_heads` matched a
same-point CPU sink-aware raw-window reference to float noise at sampled MTP
frontiers. Acceptance remains `0/71`, so the zero-acceptance blocker is
downstream of raw-SWA attention and upstream of the already-cleared MTP output
head.

The next boundary is the attention-output handoff:

```text
d_attn_heads
  -> attn_output_a grouped dense
  -> allgather / full low-rank vector
  -> attn_output_b dense shard
  -> add current shard => d_post_attn_shard
  -> FFN norm / router / routed+shared FFN
```

This sprint should determine whether attention-output projection and the
post-attention residual add are semantically correct for layer 43 before moving
deeper into routed-FFN activation order.

## Scope

1. Add a temporary MTP-only same-point diagnostic after `run_layer(43)` that
   inspects rank 0 / slot 0:
   - `ranks[0].d_attn_heads`.
   - `ops->attn_output_a.d_out[rank]` for all ranks, reassembled in rank-major
     order.
   - `ops->attn.d_out[0]` for the rank-0 attention-output shard.
   - `ranks[0].d_current_shard` and `ranks[0].d_post_attn_shard`.
2. Prefer reusing the existing F8 dense CPU oracle helper from the Sprint 592
   dense-pack work if still available in the tree or easy to revive as a
   temporary local helper. Otherwise, first validate the handoff invariants that
   do not require dequantizing F8:
   - `d_post_attn_shard == d_current_shard + ops->attn.d_out[0]`.
   - `attn_output_a` reassembly order matches the input expected by
     `attn_output_b`.
   - finite/value-scale sanity for each boundary using real value samples.
3. If the residual add or reassembly invariant fails, test only the narrow
   corresponding fix.
4. If attention-output and residual handoff pass, record that the next sprint is
   routed-FFN activation sequence / router semantics.
5. Remove all temporary diagnostic/probe code before commit unless a minimal
   durable correctness fix is found.

## Non-Goals

- Do not add a permanent diagnostic flag.
- Do not run throughput A/B; MTP acceptance is still zero.
- Do not build the K-wide verifier until MTP acceptance is nonzero.
- Do not re-test raw-SWA attention, dense F8 pack/orientation, HC-current, or
  output-head HC slicing unless their inputs change.

## Definition of Done

- Temporary attention-output/post-attention diagnostic build passes on the pod.
- The deterministic harness emits same-point evidence for layer 43.
- The sprint records whether:
  - raw attention heads are handed to `attn_output_a` correctly.
  - `attn_output_a` rank-major reassembly order is correct.
  - `attn_output_b` rank-0 shard is finite and scale-plausible or CPU-oracle
    checked.
  - `d_post_attn_shard == d_current_shard + attention_output_shard`.
- Any temporary fix candidate has deterministic acceptance evidence and is
  either promoted as a narrow durable fix or removed.
- Temporary diagnostic code is removed before commit unless promoted.
- Clean source is recopied to `/workspace/s573-continuation-instrument` and the
  clean pod build passes.
- `VISION.md` and `SPIKE_B_STEERING.md` record the result and next target.

## Result

Status: ABORTED / PUNTED by user decision.

The first temporary residual-handoff diagnostic was implemented and built on the
pod, but the run was intentionally interrupted before it produced usable
evidence. No diagnostic code was promoted. The temporary changes were removed
from `engine/post_attention_ffn.cu`, and the clean local source tree has no
engine diff.

The broader MTP finding is now recorded in `MTP_IMPLEMENTATION.md`: the V100
TP/EP MTP draft path currently does not work. The integrated draft executes and
does not corrupt the main token stream, but deterministic draft acceptance stays
at `0/71`, so MTP is not a performance feature. The remaining MTP body
localization work is punted until explicitly reopened.
