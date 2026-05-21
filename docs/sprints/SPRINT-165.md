# Sprint 165 - TP2 Async Input Overlap Gate

Date: 2026-05-21

## Objective

Test whether Sprint 164's one-layer TP2 scheduler regression is mainly caused
by artificial host-side serialization of peer input copies before owner-half
compute.

Sprint 164 proved the TP2 overlay is correct and fits, but the production
scheduler helper currently copies `x`, `selected`, and `weights` to the peer
with synchronous peer copies before launching the owner half. The earlier TP
proxy's positive result assumed the owner and peer halves can overlap. This
sprint removes only that obvious serialization point and measures the real
effect.

## Scope

- Add an explicit TP2 async-input mode:
  - `DS4_V100_TP2_ASYNC_INPUT=1`
  - alias: `DS4_V100_TP_ROUTED_FFN_ASYNC_INPUT=1`
- Keep the TP2 overlay default-off.
- Keep synchronous peer output copy and owner add for safety; do not introduce
  double-buffered peer output yet.
- Do not promote TP2 as a production default unless V100 validation shows a
  material same-path improvement.

## Implementation

1. Add a narrow helper in `ds4_v100_layer_execute.c` to choose async TP2 input
   copies.
2. In `execute_turbomind_tp2_routed()`:
   - when async input is enabled, enqueue `x`, `selected`, and `weights` copies
     with `ds4_gpu_tensor_copy_async`
   - then launch owner-half compute on the owner GPU
   - then launch peer-half compute on the peer GPU, relying on peer default
     stream ordering after the enqueued input copies
   - keep the peer output copy synchronous before the owner add
3. Leave existing synchronous behavior available as the control path.
4. Record the result in `docs/sprints/VISION.md` and a cluster log.

## Validation

Build on V100:

```bash
make -j80 CUDA_ARCH=sm_70 \
  tests/cuda_v100_stage_scheduler_smoke \
  tools/ds4-v100-replay
```

Run focused stage-0 profile at `ctx=262144`, `slots=16`:

- non-TP control
- TP2 sync-input overlay
- TP2 async-input overlay

Required gates:

- TP2 async-input scheduler smoke passes with `tp2_layers=1`.
- Negative TP2 layer selection still fails closed.
- Async-input stage profile must beat the Sprint 164 sync-input TP2 overlay
  before any full selected-token decode is worth running.
- If async-input still regresses versus no-TP, keep TP2 diagnostic-only and
  move the next sprint to either a broader persistent TP/EP boundary or a
  non-TP fused routed-FFN boundary.

## Definition of Done

- [x] Async TP2 input-copy mode is implemented behind explicit env gates.
- [x] Default TP2 behavior remains unchanged when async env is unset.
- [x] V100 build passes.
- [x] Scheduler TP2 async-input smoke passes and reports `tp2_layers=1`.
- [x] Stage-profile A/B/C is captured:
  - no TP
  - TP2 synchronous input
  - TP2 async input
- [x] Result is recorded in the vision and cluster logs.
- [x] Changes are committed.

## Risks

- `ds4_gpu_tensor_copy_async` uses stream 0. This sprint relies only on same-peer
  default-stream ordering from input copies to peer-half compute, which is
  conservative enough for correctness.
- The peer output copy remains synchronous to avoid overwriting reusable peer
  scratch before an async owner-side copy has completed.
- If V100 default-stream semantics or CUDA peer-copy behavior are more
  synchronizing than expected, this may show no gain. That is still useful
  evidence against the overlay path.

## Results

Built on the V100 pod:

```bash
make -j80 CUDA_ARCH=sm_70 \
  tests/cuda_v100_stage_scheduler_smoke \
  tools/ds4-v100-replay
```

The explicit async-input mode passed scheduler smoke with `tp2_layers=1`, and
the negative layer-selection gate still failed closed:

```text
cuda_v100_stage_scheduler_smoke: TP2 routed FFN layer 2 has no TP2 bindings
rc=1
```

Stage-0 profile at `ctx=262144`, `slots=16`:

| Run | TP2 mode | tp2_layers | ffn_ms | total_ms |
|---|---|---:|---:|---:|
| cold A/B/C | off | 0 | `148.462` | `246.462` |
| cold A/B/C | sync input | 1 | `212.429` | `317.593` |
| cold A/B/C | async input | 1 | `136.294` | `235.591` |
| warm repeat | off | 0 | `84.978` | `166.453` |
| warm repeat | async input | 1 | `145.917` | `245.107` |

Conclusion:

Async input copies remove a large fraction of the synchronous overlay penalty,
but they do not make one-layer TP2 competitive with the normal production path.
The TP2 overlay remains diagnostic-only. The next sprint should stop optimizing
the per-layer overlay and move to either a broader persistent TP/EP scheduler
boundary or a non-TP fused routed-FFN boundary.
