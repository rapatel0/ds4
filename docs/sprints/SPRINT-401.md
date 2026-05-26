# Sprint 401: NCCL HC-Current AllGather Gate

Date: 2026-05-26

## Overview

Stay on the TP/EP path and test NCCL on a broader hidden-state serving
boundary than Sprint 400. The current HC-current input stage computes a
512-wide current shard on each rank, then either rebuilds the full 4096-wide
hidden state on GPU0 and broadcasts it, or uses peer-gather kernels to rebuild
the full hidden state on every rank.

Sprint 399 showed NCCL is materially faster than peer-copy doubling for F16
hidden collectives in proxy form. Sprint 400 showed a narrow attention-output
NCCL allgather is correct but not memory-admitted at the target shape because
the communicator costs about `0.6-0.7 GiB/GPU`. This sprint attaches NCCL to
the HC-current hidden allgather instead: a true TP hidden-state boundary used
by every layer before dense attention/shared FFN and routed expert input
staging.

## Constraints

- TP/EP only. No PP/layer-split work.
- No generic scheduler abstraction.
- Default off until same-binary V100 A/B proves correctness and performance.
- Preserve first-token evidence.
- Keep the production measurement shape: `32` slots, `256K` context,
  `position=262080`, model-router compact-MoE.
- Do not combine with Sprint 400 attention-output NCCL in the same candidate;
  measure one NCCL serving boundary at a time.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`

Planned changes:

1. Add `--tp-hc-current-input-nccl-allgather-gate`.
2. Add launcher/profile wiring:
   `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER`.
3. Reuse the existing per-rank NCCL communicator lifecycle when the gate is
   active.
4. After each rank computes `d_current_shard`, allgather the rank-local
   `[slots,512]` FP32 shards into a rank-major buffer on every GPU.
5. Convert rank-major `[rank][slot][512]` to slot-major `[slot][4096]` in each
   rank's `d_current_full`.
6. Reuse the existing downstream dense input fill, shared FFN input fill, and
   routed route-pack path from `d_current_full`.
7. Leave the existing GPU0 gather/broadcast and peer-gather paths as fallback
   diagnostics.

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS on the V100 pod. Only existing unused-function warnings were
emitted.

Direct same-binary A/B:

- `32` slots
- `256K` context / `position=262080`
- `tokens=2`
- model-router compact-MoE
- route-plan async upload enabled
- HC-current stream-sync enabled
- candidate adds only HC-current NCCL allgather

Record:

- first token
- generated/continuation decode tok/s
- `sum_hc_current_gather_ms`
- `sum_hc_current_fill_pack_ms`
- total decode ms
- VRAM admission and communicator memory delta

If the target shape fails memory admission, run a `16` slot functional
diagnostic to distinguish implementation correctness from target-shape
admission failure.

## Results

Target `32` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | HC gather ms | HC fill/pack ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 54639 | 85.897762 | 99.733266 | 6.986851 | 28.665986 | 745.071798 | 1746 MiB |
| NCCL HC-current allgather | 2 | n/a | n/a | n/a | partial | partial | partial | 1114 MiB |

The first candidate run exposed a CUDA device-context bug after the NCCL loop:
the current device was left on the last rank and the next GPU0 control-stream
kernel failed with `invalid resource handle`. The implementation now resets
the current device to GPU0 before the control-stream FFN norm. After that fix,
the candidate executed through layer 28 and then failed on raw-SWA allocation:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:9804: out of memory
```

The memory delta matches Sprint 400: NCCL communicator initialization raises
rank-buffer max used from `2317` to `2979` MiB and lowers post-output-head
minimum free VRAM from `1746` to `1114` MiB. The target production shape is
not admitted with this additional communicator footprint.

Functional `16` slot / `256K` direct A/B:

| Mode | RC | First token | Generated decode tok/s | Continuation decode tok/s | HC gather ms | HC fill/pack ms | Decode ms | Min free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 0 | 54639 | 65.078267 | 73.863020 | 5.532507 | 17.896066 | 491.715616 | 4454 MiB |
| NCCL HC-current allgather | 0 | 54639 | 61.918746 | 69.068787 | 15.830067 | 8.442391 | 516.806332 | 3820 MiB |

The `16` slot run proves functional correctness: the first token matches
`54639`. It is not a performance win. The NCCL allgather plus rank-major
transpose makes the HC gather bucket about `2.86x` slower, and the smaller
fill/pack bucket does not recover the loss. Total decode regresses by about
`5.1%`.

Artifacts:

- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-control/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-candidate/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-control-16/`
- `logs/from-cluster/sprint401-hc-current-nccl-allgather/direct-candidate-16/`

## Decision

REJECT as a default and keep diagnostic-only.

This confirms the current HC-current payload is not the right NCCL serving
boundary in isolation. NCCL adds the same `~0.6-0.7 GiB/GPU` communicator
footprint seen in Sprint 400, fails target-shape memory admission, and is
slower even when the smaller `16` slot shape fits. Future NCCL work should be
admitted as a shared topology resource and should replace a larger fused
TP/expert boundary, not a narrow per-layer hidden allgather.

## Definition of Done

- Gate exists in binary, launcher, and profile harness.
- V100 build passes.
- Same-binary direct A/B records correctness evidence and HC-current timing.
- Sprint doc, temporary status report, status, and vision are updated with an
  explicit PROMOTE or REJECT decision.
- Commit all kept artifacts explicitly.

## Risks

- NCCL communicator memory may again erase the target-shape headroom.
- The current default GPU0 gather/broadcast may be cheaper than NCCL at this
  small FP32 payload because downstream GPU0 router work still serializes the
  stage.
- Rank-major to slot-major conversion may offset collective savings.
