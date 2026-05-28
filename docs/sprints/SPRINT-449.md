# Sprint 449: Combined Rank-Major Recheck

## Objective

Stay TP/EP only and rerun the combined rank-major HTTP A/B that failed in
Sprint 445, now that Sprint 448 made attention rank-local input correctness
clean.

The question is whether the previously token-changing combined rank-major path
is now a usable serving bundle when attention rank-local input and routed-FFN
rank-major input are enabled together.

## Implementation Plan

1. Use the rebuilt `sm_70` binary from Sprint 448.
2. Run a same-binary HTTP A/B at the reduced isolation shape:

   ```text
   slots=8
   ctx=262144
   requests=4
   tokens=2
   position=100000
   ```

3. Keep the control on the known-clean semantic TP/EP path:

   ```text
   HC-current NCCL
   post-attention FFN input
   fixed-capacity post-attention route plan
   compact MoE decode
   lazy output head
   ```

4. Enable these candidate gates together:

   ```text
   --candidate-attention-projection-rank-local-input
   --candidate-routed-ffn-rank-major-input
   ```

5. Record response parity, first token, server decode, continuation decode,
   client generated tok/s, GPU utilization, and VRAM margin.

## Validation

- Remote HTTP A/B completes both legs.
- Response parity artifacts are present.
- No stale DS4 GPU process remains after the run.

## Decision Rule

- If parity fails, isolate the interaction between attention-rank-local input
  and routed-FFN rank-major input before adding any router rank-major work.
- If parity passes but throughput regresses, keep the bundle default-off and
  inspect per-layer timers to find the added sync/copy cost.
- If parity passes and throughput improves, run the same candidate at a larger
  serving shape before launcher promotion.

## Execution Notes

The first attempt at
`/localpool/ds4/workspace/logs/s449-combined-rankmajor-attn-ffn` failed before
producing a model result because an unrelated queued router-rank-major isolate
started on the same port base and killed the control server with `rc=-15`.

After clearing that queued job, the clean rerun used port base `18800`:

```text
/localpool/ds4/workspace/logs/s449-combined-rankmajor-attn-ffn-rerun
```

## Outcome

Clean rerun summary:

| Leg | Server generated decode tok/s | Server continuation decode tok/s | Client generated tok/s | GPU util avg | First token |
|---|---:|---:|---:|---:|---:|
| Control | 20.573850 | 20.485800 | 0.742192 | 9.651316% | 71302 |
| Attention rank-local + routed FFN rank-major | 20.372120 | 20.233079 | 0.745437 | 10.302632% | 71302 |

Response parity matched `4/4`:

```text
matched_pairs=4
failed_pairs=0
match=true
```

Both legs returned `4/4` HTTP 200 responses, had `vram_failures=0`, and kept
the same `5698 MiB` minimum free VRAM. Readiness was marked false only because
this reduced two-token run uses the strict `client_generated_tok_s >= 1`
threshold; the model-serving checks and parity artifacts were valid.

## Decision

The Sprint 445 combined rank-major token-change blocker is fixed. Attention
rank-local input and routed-FFN rank-major input can now run together without
changing response tokens at the reduced 8-slot / 256K shape.

Do not promote the combined bundle yet. The candidate regressed server
generated decode by about `0.98%` and continuation decode by about `1.23%`,
while the GPU-util/client improvements were too small to justify a launcher
default. The next performance sprint should inspect the per-layer timing delta
and look for a bundle that keeps the FFN rank-major win without paying extra
attention-rank-local overhead.
