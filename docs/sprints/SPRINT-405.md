# Sprint 405: Lazy Diagnostic Output Head

Date: 2026-05-26

## Overview

Sprint 404 showed that HC-current NCCL at `32` slots / `256K` is short by up
to `422 MiB/GPU` against the `1536 MiB` NCCL reserve. The first necessary
memory-reclaim step is output-head residency: it costs `130-134 MiB/GPU` and
is not needed during the 43-layer decode loop.

This sprint adds an opt-in lazy diagnostic output-head mode. The resident
startup path no longer opens the output head before layer decode when the gate
is active. Instead, direct serving-bench runs open the output head after the
layer loop, run top-1 selection, record metrics, and close it before returning.

## Constraints

- TP/EP only. No PP/layer-split work.
- Default behavior unchanged.
- This is the first half of the S404 paired memory plan; it is expected to be
  insufficient alone for GPU0.
- Preserve first-token correctness for non-NCCL control.

## Implementation

Files:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-run-appliance.sh`

Add:

- binary flag `--diagnostic-output-head-lazy-gate`
- profile flag `--lazy-output-head`
- launcher env `DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD_LAZY`

The lazy path is scoped to direct `--serving-bench` runs in this sprint. HTTP
serving still uses the existing resident output-head object.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

V100 probes:

1. `lazy-output-head` control at `32` slots / `256K`, direct token-major,
   model-router compact-MoE.
2. `hc-current NCCL + lazy-output-head` at `32` slots / `256K`, direct
   token-major, model-router compact-MoE.

Artifacts:

- `logs/from-cluster/sprint405-lazy-output-head/lazy-control/`
- `logs/from-cluster/sprint405-lazy-output-head/lazy-hc-nccl/`

Results:

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | Key checkpoint |
|---|---:|---:|---:|---:|---:|---|
| lazy control | 0 | 54639 | 97.034724 | 105.686032 | 68 MiB | `after_hc_controls=1880 MiB`, `after_lazy_output_head=68 MiB` |
| lazy + HC-current NCCL | 2 | n/a | n/a | n/a | 1248 MiB | CUDA OOM at compressed KV state allocation, layer 5 |

The lazy control preserves the expected first token and comparable non-NCCL
decode throughput, but it is not a safe default: opening the output head after
the layer loop leaves only `68 MiB` free on GPU0 because decode-time
allocations are still resident.

The lazy + NCCL case gets past resident output-head startup, but still fails
before first-token completion. NCCL rank buffers plus HC controls leave only
`1248 MiB` free after `after_hc_controls`; the run then OOMs at
`tools/ds4-v100-tp-ep-full-layer-smoke.cu:9869` while allocating compressed
KV state for layer 5. This confirms output-head laziness alone is
insufficient.

## Definition of Done

- Lazy output-head flag exists in binary, launcher, and profile harness.
- Local checks pass.
- V100 build passes.
- V100 artifacts quantify the memory delta and first-token behavior.
- Sprint doc, status, vision, and temporary status report are updated.
- Commit all kept artifacts explicitly.

## Decision Gate

Do not promote lazy output-head as a default unless it preserves first token
and does not regress the non-NCCL target shape materially. Even if it works, do
not consider NCCL solved until the paired HC-control residency reduction also
lands and HC-current NCCL is admitted at `32` slots / `256K`.

## Decision

Keep lazy diagnostic output head default-off and diagnostic-only.

Promote the harness flag because it exposes the real memory issue cleanly, but
do not promote it into the production serving path. The next NCCL memory sprint
must reduce or stream GPU0 HC-control residency and/or compressed-KV transient
residency before opening additional resident resources. Lazy output-head alone
moves the peak; it does not reduce the target-shape peak enough.
