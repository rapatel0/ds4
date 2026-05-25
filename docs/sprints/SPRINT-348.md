---
sprint: 348
title: TP/EP HC Current Peer Gather
status: completed
started: 2026-05-25
completed: 2026-05-25
branch: claude-takeover
---

# Sprint 348 - TP/EP HC Current Peer Gather

## Overview

Sprint 347 made direct TP/EP profiling reliable and showed the dominant
serving-stage timer is `sum_hc_current_input_ms`. The current
`run_shared_hc_current_input` path computes rank-local current shards, gathers
them into a full current vector on GPU0, then broadcasts that full vector back
to every rank before filling dense inputs and routed inputs.

This sprint adds an opt-in TP/EP-only peer-gather variant. Each rank builds its
own full `[slots, hidden]` current input directly from the eight rank-local
current shards using peer reads. GPU0 still uses its full copy for FFN norm and
router/control work, but the central gather-plus-broadcast staging path is no
longer required for dense input fill.

No PP/layer-split work. No MTP.

## Implementation

1. Add `--tp-hc-current-input-peer-gather-gate`.
2. Add a CUDA kernel that gathers eight current shards into one destination
   rank's full current vector.
3. In `run_shared_hc_current_input`, when the gate is enabled:
   - build `r.d_current_full` on every rank directly from the eight
     `d_current_shard` buffers;
   - use rank 0's gathered full current for FFN norm/router control work;
   - skip the old GPU0 full-current broadcast back to each rank.
4. Add launcher/env and profiler-harness switches so the path can be exercised
   through both direct profiler runs and HTTP serving A/B.

## Verification

Local:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100:

```text
direct-token-major control, 32 slots, 256K, 2 decode steps
direct-token-major peer-gather candidate, same shape
HTTP sanity or A/B if direct candidate is correct
```

## Definition of Done

- [x] Peer-gather gate is implemented in the TP/EP binary.
- [x] Launcher and profiler harness can enable the gate.
- [x] V100 build passes.
- [x] Direct control and candidate both pass with finite output head.
- [x] Candidate result is recorded as promoted, rejected, or follow-up.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and
      `TEMP_STATUS_REPORT_060.md` are updated.
- [x] Cluster artifacts are copied into `logs/from-cluster/`.
- [x] Sprint artifacts are committed.

## Outcome

Implemented the opt-in gate:

```text
--tp-hc-current-input-peer-gather-gate
DS4_V100_TP_EP_HC_CURRENT_INPUT_PEER_GATHER=1
tools/ds4-v100-tp-ep-profile.py --hc-current-peer-gather
```

The gate adds `gather_current_shards_to_full8_kernel`, which lets each TP rank
build its own full current vector directly from all eight current shards. GPU0
uses its gathered full vector for FFN norm/router control work, and the old
GPU0 full-current broadcast is skipped.

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Direct 32-slot / 256K / 2-step A/B:

| Case | Generated tok/s decode | Continuation tok/s decode | sum decode ms | HC-current ms | Output finite |
|---|---:|---:|---:|---:|---:|
| Control | `87.263615` | `100.446187` | `733.409911` | `596.248809` | `0` bad |
| Peer gather | `67.495350` | `80.223389` | `948.213473` | `801.525057` | `0` bad |

## Decision

Reject peer gather as a promoted serving default. It preserves output
correctness for the tested window, but it makes the hot stage slower:

```text
HC-current input: 596.248809 ms -> 801.525057 ms
generated tok/s:  87.263615 -> 67.495350
```

The old GPU0 gather-plus-broadcast is not the main removable cost in this
shape. The next optimization should target the actual HC control computation
and synchronization structure: avoid the repeated central dense/control
synchronizations or fuse the HC current split/norm/fill chain, rather than
spreading the same gather work across every rank.

Artifacts:

```text
logs/from-cluster/sprint348-hc-peer-gather/cluster/
```
