# Sprint 164 - Guarded Scheduler TP Routed Layer

Date: 2026-05-21

## Objective

Move the Sprint 163 TP routed-FFN primitive one step closer to serving by
wiring it into the stage scheduler for exactly one layer behind explicit
runtime controls.

The sprint should prove that scheduler-owned layer execution can select the TP2
path for a named layer, use owner and peer overlay arenas for the TP split
weights, and still produce the same selected-token output as the current
single-GPU routed FFN path.

## Strategy

Do not repack the full appliance yet. Use an overlay:

- normal appliance shard dir: current production fused appliance
- normal TurboMind index: current production fused appliance
- TP overlay dir: layer-local `--emit-tp-split` pack from Sprint 163
- combined TurboMind index: production index plus TP rows for the selected layer

This keeps the full serving appliance intact while allowing the one selected
layer to read its TP split descriptors from separate owner/peer arenas.

## Implementation

1. Add explicit opt-in configuration:
   - `DS4_V100_TP2_LAYER`
   - `DS4_V100_TP2_SHARD_DIR`
   - default off
2. Extend scheduler state with:
   - selected TP layer id
   - TP owner overlay arena
   - TP peer overlay arena
   - reusable owner/peer TP tensors sized by active slots
3. Extend layer execute config with optional TP2 pointers.
4. In `execute_ffn_delta()` and `execute_ffn_delta_batch()`, select TP only
   when all conditions hold:
   - env/config enabled
   - current layer matches
   - layer state has TP2 descriptors
   - fused gate/up is enabled
   - owner and peer overlay arenas are present
   - TP scratch tensors are large enough
5. Preserve fallback behavior:
   - if TP is off, missing, or not the selected layer, run the existing path
   - if TP is selected but setup is invalid, fail closed with a clear error
6. Add report counters/timing sufficient to see whether the selected layer
   actually took the TP path.

## Validation

- Build:
  - `make -j80 CUDA_ARCH=sm_70 tests/cuda_v100_stage_scheduler_smoke tools/ds4-v100-replay`
- Unit/primitive control:
  - rerun `tests/cuda_v100_tp_routed_ffn_smoke`
- Scheduler correctness:
  - generate a combined TM index from production fused TM rows plus Sprint 163
    TP rows
  - run a single-stage or full-scheduler selected-token smoke with
    `DS4_V100_TP2_LAYER=3`
  - compare selected-token output against the current non-TP control
- Performance evidence:
  - run a focused one-stage layer/profile smoke with TP off/on for layer 3
  - only claim layer-local timing, not HTTP serving throughput

## Definition of Done

- [x] TP2 scheduler integration is default-off and gated by explicit env/config.
- [x] A selected layer can execute routed FFN through owner+peer TP overlay arenas.
- [x] Non-TP scheduler controls still pass.
- [x] TP-enabled scheduler correctness passes on the V100 cluster.
- [x] Logs capture whether the TP path was selected and the layer-local timing.
- [x] Vision is updated with the result.
- [x] Changes are committed.

## Risks

- The current scheduler owns one stage arena. The TP overlay must not disturb
  normal stage-resident weights or KV state.
- The first integration may only validate one stage/layer. Full HTTP serving
  should wait for a correctness gate and timing evidence.
- If one-slot TP remains only ~1.06x at the layer level, HTTP throughput may not
  move without slot batching or multiple TP-enabled layers.

## Results

Built on the V100 pod:

```bash
make -j80 CUDA_ARCH=sm_70 \
  tests/cuda_v100_stage_scheduler_smoke \
  tests/cuda_v100_tp_routed_ffn_smoke \
  tools/ds4-v100-replay
```

Generated combined descriptor index:

```text
/workspace/ds4-tp2-combined-s164/turbomind-pack-index.tsv
```

Validation:

- Primitive TP smoke using the Sprint 163 split pack passed at layer 3,
  GPU0/GPU3, `tokens=16`, `routes=96`:
  - `max_abs=1.34401e-06`
  - `rel=0.000278022`
  - `bad=0`
  - `ref_ms=1.9739`
  - `total_ms=1.2204`
  - `speedup=1.617x`
- Scheduler non-TP control passed with `tp2_layers=0`.
- Scheduler TP overlay passed with `tp2_layers=1` for both `slots=1` and
  `slots=16`.
- Negative gate passed: selecting layer 2 with the layer-3 TP descriptor index
  fails open with `TP2 routed FFN layer 2 has no TP2 bindings`.
- Full 8-stage serial open with the overlay fits in 32 GiB V100 memory:
  - observed during open: GPU0 about `26.7 GiB`, GPU3 about `22.5 GiB`
  - `open_total=257153.285 ms`
- Full selected-token decode with overlay also completed.

Performance read:

| Run | slots | TP2 layers | stage0 decode | prompt replay | generated tok/s |
|---|---:|---:|---:|---:|---:|
| no TP full decode | 16 | 0 | `996.590 ms` | `1560.589 ms` | `0.639435` |
| one-layer TP overlay full decode | 16 | 1 | `1361.390 ms` | `1953.716 ms` | `0.511053` |
| stage0 profile no TP | 16 | 0 | `total_ms=178.819`, `ffn_ms=88.577` | n/a | n/a |
| stage0 profile TP overlay | 16 | 1 | `total_ms=216.829`, `ffn_ms=118.585` | n/a | n/a |

Conclusion:

The TP2 scheduler integration is correct, default-off, and useful as a
diagnostic topology primitive, but this one-layer overlay must not be promoted
as a performance default. The synchronous peer input/route copies plus peer
output copy dominate the benefit when applied to one layer. The next TP attempt
should either use persistent peer ownership with overlap across a larger layer
group, or move to an explicit TP/EP scheduler where peer payloads are native to
the execution topology instead of an overlay bolted onto layer-parallel
execution.
