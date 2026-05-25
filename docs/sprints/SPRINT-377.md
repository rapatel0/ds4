# Sprint 377: Batched Paged Attention Gate

## Overview

Implement the next throughput-prompt gate after Sprint 376:
`--batched-paged-attn-gate`.

Sprint 376 rejected CUDA graph replay as a promotion path because the current
TP/EP decode step uses pervasive `cudaMemcpyPeerAsync` transport that this
V100/CUDA stack rejects during stream capture. The next best throughput lever
is therefore reducing steady decode launch count and fragmented attention/KV
work without depending on CUDA graphs.

This sprint focuses on the typed-KV attention boundary. The current path has
many per-slot and per-row-family row store/load launches around raw-SWA,
compressed attention, and ratio-4 indexer history. Sprint 377 should collapse
the hot attention/KV row work into batched, block-table-indexed kernels behind
a default-off gate, then A/B it against the current serving default at the
real `32` slot / `256K` shape.

## Scope

- Add a default-off CLI gate:

```text
--batched-paged-attn-gate
```

- Add launcher/profile plumbing:

```text
DS4_V100_TP_EP_BATCHED_PAGED_ATTN=1
tools/ds4-v100-tp-ep-profile.py --batched-paged-attn
```

- Extend, rather than replace, the existing typed-KV batch-row path:

```text
--true-ds4-attention-typed-kv-batch-rows-gate
```

- Implement a first production-shaped batched attention/KV row path for:
  - raw-SWA rows
  - compressed attention rows
  - ratio-4 indexer rows
- Keep the current default path unchanged.
- Validate on the V100 pod with same-binary A/B.

## Out Of Scope

- No PP/layer-split work.
- No generic PP/TP scheduler abstraction.
- No MTP in this sprint.
- No broad dtype conversion.
- No P2P transport rewrite for CUDA graphs.
- No TP-sharded expert topology rewrite.

## Architecture

The gate should introduce a small paged-attention work plan that is built once
per layer step and consumed by a small number of kernels per row family.

Initial data model:

```text
block_table[slot][family][visible_block]
seq_lens[slot][family]
row_positions[slot][family][visible_row]
row_family = raw_swa | compressed_attn | indexer
```

The first implementation can use the existing bounded compressed-row metadata:

```text
kBoundedCompRows
attn_comp_row_loaded_layers
attn_comp_row_loaded_position_layers
index_comp_row_loaded_layers
index_comp_row_loaded_position_layers
```

The important change is kernel granularity: avoid separate launches for each
slot/family row where a single batched kernel can process all active slots and
their visible rows.

## Implementation Plan

### Phase 1: Baseline Metrology

Run a read-only baseline before changing behavior:

```text
tools/ds4-v100-tp-ep-active-slot-matrix.py
tools/ds4-v100-tp-ep-http-ab.py
```

Target shape:

```text
32 configured slots
32 active requests
256K context
position 262080
32 generated tokens/request
```

Record:

- server decode tok/s
- client tok/s
- average and max GPU utilization
- attention/raw-read/compressed-KV stage timing
- kernel-count evidence where available
- first token
- decode checksum

### Phase 2: Gate Plumbing

Add:

- CLI flag parsing in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- env plumbing in `tools/ds4-v100-run-appliance.sh`
- profile flag plumbing in `tools/ds4-v100-tp-ep-profile.py`
- status/metrics visibility if the server exposes gate state there

The gate must default off everywhere.

### Phase 3: Batched Row Plan

Create a compact per-layer, per-step row plan for all active slots:

- raw-SWA visible rows per slot
- compressed-attention visible rows per slot
- ratio-4 indexer visible rows per slot
- source physical row or bounded-cache row index
- row count per slot

Use fixed-size arrays for the first implementation. Avoid dynamic allocation
inside the decode step.

### Phase 4: Batched Kernels

Implement the minimum kernel set needed to prove the launch-count thesis:

- one batched raw-SWA row gather/load kernel
- one batched compressed-attention row gather/load kernel
- one batched ratio-4 indexer row gather/load or score-prep kernel

Use SM70-friendly constraints:

- `head_dim = 512`
- online fp32 max/sum for softmax where attention math is included
- clamp exponent input to `[-80, 0]`
- floor softmax denominator at `1e-24`
- choose tile shapes that respect the V100 96 KiB shared-memory ceiling

The first implementation may be a staging/launch-count reducer before it is a
fully fused FlashAttention replacement. It still must preserve first-token and
checksum parity.

### Phase 5: V100 A/B

Build on the V100 pod:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run same-binary A/B:

- direct token-major at `32` slots / `256K`
- HTTP serving at `32` active requests / `32` slots / `256K` /
  `32` generated tokens/request
- GPU utilization sampling enabled

Artifacts go under:

```text
logs/from-cluster/sprint377-batched-paged-attn
```

## Definition Of Done

- `--batched-paged-attn-gate` builds and defaults off.
- Launcher/profile plumbing exists and defaults off.
- Baseline V100 metrics are recorded before candidate runs.
- Batched row plan emits debug/audit counters for row families and active slots.
- Candidate V100 direct run preserves first token and all-layer decode checksum.
- Candidate V100 HTTP A/B reports client tok/s, server decode tok/s, stage
  timing, and GPU utilization.
- Sprint doc records an explicit PROMOTE, KEEP-OPT-IN, or REJECT decision.
- `TEMP_STATUS_REPORT_377.md` summarizes the result.
- `docs/sprints/VISION.md` and `docs/sprints/STATUS.md` are updated.
- Changes are committed.

## Decision Rule

Promote only if the gate preserves first token/checksum and improves server
decode tok/s or average GPU utilization at the real `32` slot / `256K` serving
shape.

Keep opt-in if correctness holds but performance is flat or noisy.

Reject if it changes first token/checksum, increases attention/KV stage time
enough to regress serving throughput, or introduces new long-context/session
cache risk.

## Risks

- A staging-only batched path may reduce launch count but increase memory
  traffic; the A/B must decide.
- DS4's compressed/indexer row semantics are easy to perturb. Preserve the
  existing compact-reference and typed-history checks while developing.
- A full fused paged attention kernel may be too large for this sprint. If so,
  stop at the smallest batched row-family kernel that gives a clean A/B answer.
- GPU0-heavy control staging may remain the bottleneck even if row-family
  launches are reduced.

## Follow-On Candidates

- `--compact-moe-decode-gate`
- `--fused-gated-silu-gate`
- TP-sharded expert A/B
- P2P kernel transport plan if CUDA graphs are revisited

## Progress

### Baseline Metrology

Ran the required read-only V100 baseline before candidate implementation.

Command shape:

```text
32 active chat requests
32 configured slots
256K context
position 262080
32 generated tokens/request
GPU sampling interval 250 ms
```

Artifact path:

```text
logs/from-cluster/sprint377-batched-paged-attn/baseline-matrix
```

| Metric | Value |
|---|---:|
| HTTP 200 | `32/32` |
| Coalesced batch size | `32` |
| First token | `89340` |
| Client generated tok/s | `40.157540` |
| Server generated tok/s | `74.895420` |
| Server generated tok/s decode | `88.372350` |
| Server continuation tok/s decode | `88.329223` |
| Scaffold projected slot-step tok/s | `56.990488` |
| Avg GPU util | `7.972222%` |
| Max GPU util | `38%` |
| Max GPU memory used | `32398 MiB` |
| Compressed-KV sum | `5436.764269 ms` |

Next implementation step: add default-off gate plumbing and the first
fixed-size row-family plan.

### Gate Plumbing

Added default-off gate plumbing:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu --batched-paged-attn-gate`
- `DS4_V100_TP_EP_BATCHED_PAGED_ATTN=1`
- `tools/ds4-v100-tp-ep-profile.py --batched-paged-attn`
- active-slot matrix artifact suffix support

Validation:

| Check | Result |
|---|---|
| Python profile/matrix compile | pass |
| launcher shell syntax | pass |
| V100 build | pass |
| launcher `--print-command` | includes `--batched-paged-attn-gate` |

No-op direct smoke:

```text
logs/from-cluster/sprint377-batched-paged-attn/gate-plumbing-smoke/none-direct-batched-paged-attn
```

| Metric | Value |
|---|---:|
| Return code | `0` |
| First token | `54639` |
| Output finite bad | `0` |
| Generated tok/s decode | `77.855330` |

Next implementation step: fixed-size row-family plan and the first batched row
kernel.

### Row-Family Plan

Added a fixed-size row-family plan audit behind `--batched-paged-attn-gate`.
The audit logs only when a layer's compressed/indexer row signature changes,
with a per-layer cap, so it can be used in serving-shaped runs without turning
stdout into the benchmark.

Artifacts:

```text
logs/from-cluster/sprint377-batched-paged-attn/row-plan-smoke/none-direct-batched-paged-attn
logs/from-cluster/sprint377-batched-paged-attn/row-plan-change-smoke/none-direct-batched-paged-attn
```

Validation:

| Check | Result |
|---|---|
| V100 build | pass |
| 1-token direct row-plan smoke | pass, `43` plan rows, first token `54639` |
| 8-token direct row-plan smoke | pass, `127` plan rows, first token `98751` |

8-token topline:

| Metric | Value |
|---|---:|
| Generated decode tok/s | `96.553089` |
| Continuation decode tok/s | `99.794998` |
| Compressed rows emitted | `42` |
| Compressed-KV sum | `813.233407 ms` |
| Attention projection sum | `479.943118 ms` |
| Attention state sum | `339.001691 ms` |
| Raw-read sum | `124.838245 ms` |
| Typed-history sum | `30.807917 ms` |
| EP sum | `208.598725 ms` |
| Compose sum | `145.634186 ms` |

Representative plan samples:

```text
layer 2 position 262083 raw_valid_rows 4 visible_attn_rows 1 visible_indexer_rows 1 target_family_kernels 3
layer 4 position 262087 raw_valid_rows 8 visible_attn_rows 2 visible_indexer_rows 2 target_family_kernels 3
```

Finding: the row plan works, but the narrow typed-history load target is not
the current hot path. Pending typed-history reloads are `0` in the observed
compressed/indexer samples because the current skip-load/cache path already
avoids the reload storm. The first real S-C kernel should therefore fuse more
of the raw+compressed attention computation, or this sprint should close with
the evidence and move to compact MoE.

## Decision

Decision: **REJECT narrow S-C typed-history load replacement as the next
implementation target; keep the row planner diagnostic-only.**

Reason:

- The `--batched-paged-attn-gate` plumbing and row-family planner are correct
  and validated on V100.
- The observed row-family plan does not show the assumed per-slot typed-history
  load storm at the target serving shape.
- In the 8-token direct run, typed-history is `30.807917 ms` of
  `2651.391081 ms` summed decode, while compressed-KV, attention projection,
  attention state, EP, and compose are materially larger.
- Building a load-only batched row kernel would likely optimize a cold path and
  would not satisfy the Vision requirement to move practical serving
  throughput.

Outcome:

- Keep `--batched-paged-attn-gate` and its row-plan audit as opt-in diagnostic
  infrastructure.
- Do not promote the gate as a serving default.
- Move the next sprint to `--compact-moe-decode-gate`, the next item in
  `docs/sprints/VISION.md`.
