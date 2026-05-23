# Sprint 227 - TP8 Collective Workbench

Date: 2026-05-23
Status: Complete - TP8 Boundary Characterized

## Overview

Sprint 226 made the TP8/EP8 memory and topology contract explicit. The next
risk is not whether the 32-slot/256K shape fits in VRAM; it is whether the
8-GPU boundary can move hidden state and expert outputs cheaply enough to make
TP/EP serving plausible.

This sprint builds and runs a TP-only collective workbench. It does not touch
the PP/layer scheduler and does not add a generic scheduler abstraction.

Target shape:

```text
devices: 8x V100
tokens: 32 first, then 64 and 128 as density controls
hidden: 4096
dtype: fp16 payloads
layers: 43
collectives/layer: 2 for hidden TP proxy
EP routes: slots * top_k = slots * 6
```

## Goals

- Add a TP8-only collective workbench tool that can measure:
  - hidden all-reduce;
  - reduce-scatter;
  - all-gather;
  - reduce-scatter followed by all-gather;
  - expert-output reduction proxy.
- Keep all work in new TP-specific files or existing TP-only tools.
- Measure 32, 64, and 128 token payloads on the V100 pod.
- Wrap at least the main 32-token run with the existing NVLink snapshot helper.
- Report timing, effective wire bandwidth, overhead-only tok/s, and
  correctness for each mode.
- Store cluster evidence under
  `logs/from-cluster/sprint227-tp8-collective-workbench/`.

## Non-Goals

- No PP/layer-split runtime work.
- No production serving integration.
- No MTP work.
- No TP/EP pack format changes.
- No NCCL dependency unless the existing environment already provides it and
  the tool can still build with the project Makefile.
- No claim that TP serving is operational. This sprint only characterizes the
  boundary needed before the TP runtime skeleton.

## Implementation

1. Add `tools/ds4-v100-tp8-collective-workbench.cu`.
2. Support CLI options:
   - `--mode allreduce|reduce-scatter|allgather|rs-ag|ep-reduce`;
   - `--algo root|doubling` where applicable;
   - `--tokens`;
   - `--hidden`;
   - `--layers`;
   - `--collectives-per-layer`;
   - `--warmup`;
   - `--iters`;
   - `--devices`.
3. Reuse the existing peer-access and stream patterns from the TP8 collective
   tools, but keep the workbench standalone.
4. Add a Makefile target for the new tool.
5. Local validation:
   - check source formatting with `git diff --check`;
   - verify the target is present in `make help`/Makefile context as needed.
6. V100 validation:
   - build with `make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp8-collective-workbench`;
   - run all modes at 32 tokens;
   - run density controls at 64 and 128 tokens;
   - run the 32-token suite inside `tools/ds4-v100-nvlink-snapshot.sh`.
7. Update this sprint document, `docs/sprints/STATUS.md`, and
   `docs/sprints/VISION.md` with the result.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp8-collective-workbench.cu` | TP8 collective and EP-boundary microbench |
| `Makefile` | build target |
| `docs/sprints/SPRINT-227.md` | plan and evidence |
| `docs/sprints/STATUS.md` | status update |
| `docs/sprints/VISION.md` | outcome update |
| `logs/from-cluster/sprint227-tp8-collective-workbench/` | V100 evidence |

## Definition Of Done

- [x] Sprint plan exists and is committed before implementation evidence.
- [x] New workbench builds on the V100 pod for `sm_70`.
- [x] Workbench supports all five modes:
      `allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, `ep-reduce`.
- [x] Workbench reports latency, per-layer latency, per-collective latency,
      effective wire bandwidth, overhead-only tok/s, and correctness.
- [x] 32-token runs pass correctness for every mode.
- [x] 64-token and 128-token density controls pass for the key modes.
- [x] NVLink snapshot evidence is recorded for the 32-token suite.
- [x] Results are copied into
      `logs/from-cluster/sprint227-tp8-collective-workbench/`.
- [x] Status and vision docs are updated with the decision.
- [x] Changes are committed with explicit `git add` paths.

## Execution Evidence

Local validation:

```text
make tools/ds4-v100-tp8-collective-workbench
  -> Darwin guard reports CUDA build required
git diff --check
```

V100 build:

```text
cd /workspace/ds4-sprint181
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp8-collective-workbench
```

The 32-token suite was run through:

```text
tools/ds4-v100-nvlink-snapshot.sh \
  /workspace/logs/sprint227-tp8-collective-workbench/nvlink-32-suite \
  -- bash -lc '<32-token suite>'
```

32-token results:

| Mode | Algo | Collectives/layer | Avg ms | Per collective | Effective wire | Overhead-only tok/s | Verify |
|---|---|---:|---:|---:|---:|---:|---|
| allreduce | doubling | 2 | `26.904544` | `0.312844` | `13.407 GB/s` | `1189.390` | ok |
| reduce-scatter | root | 1 | `11.945595` | `0.277805` | `7.431 GB/s` | `2678.812` | ok |
| allgather | root/direct | 1 | `18.319043` | `0.426024` | `4.307 GB/s` | `1746.816` | ok |
| rs-ag | root/direct | 1 | `32.361613` | `0.752596` | `5.181 GB/s` | `988.826` | ok |
| ep-reduce | doubling | 2 | `27.436756` | `0.319032` | `13.147 GB/s` | `1166.319` | ok |

Density controls:

| Mode | Tokens | Avg ms | Per collective | Effective wire | Overhead-only tok/s | Verify |
|---|---:|---:|---:|---:|---:|---|
| allreduce | 64 | `30.189999` | `0.351047` | `23.896 GB/s` | `2119.907` | ok |
| allreduce | 128 | `38.412399` | `0.446656` | `37.562 GB/s` | `3332.257` | ok |
| rs-ag | 64 | `39.088974` | `0.909046` | `8.579 GB/s` | `1637.290` | ok |
| rs-ag | 128 | `52.946127` | `1.231305` | `12.668 GB/s` | `2417.552` | ok |
| ep-reduce | 64 | `36.672923` | `0.426429` | `19.672 GB/s` | `1745.157` | ok |
| ep-reduce | 128 | `39.337174` | `0.457409` | `36.679 GB/s` | `3253.920` | ok |

The NVLink status snapshot confirms all V100 NVLink links report
`25.781 GB/s`. Per-link byte counters remain unavailable in the pod:
`nvidia-smi nvlink -gt d` reports `Data Tx: N/A` and `Data Rx: N/A`.

Evidence is stored in
`logs/from-cluster/sprint227-tp8-collective-workbench/`.

## Decision

Sprint 227 keeps the TP8/EP8 path viable and clears the collective workbench
gate. The important signal is density: the 43-layer, two-collective hidden
all-reduce proxy improves from `1189` overhead-only tok/s at 32 tokens to
`3332` at 128 tokens. The EP output-reduce proxy follows the same shape,
reaching `3254` overhead-only tok/s at 128 tokens.

The root/direct reduce-scatter plus all-gather implementation is correct but
not attractive as written: at 32 tokens it is slower than the doubling
all-reduce proxy (`32.36 ms` versus `26.90 ms`) and should not become the first
runtime boundary. It remains useful as a correctness/shape tool.

Next sprint should move to the TP/EP pack contract. The runtime should begin
with the doubling-style hidden reduction boundary and avoid designing around
root/direct RS+AG unless a later NCCL-grade implementation changes the result.
