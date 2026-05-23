# Sprint 227 - TP8 Collective Workbench

Date: 2026-05-23
Status: Planned

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

- [ ] Sprint plan exists and is committed before implementation evidence.
- [ ] New workbench builds on the V100 pod for `sm_70`.
- [ ] Workbench supports all five modes:
      `allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, `ep-reduce`.
- [ ] Workbench reports latency, per-layer latency, per-collective latency,
      effective wire bandwidth, overhead-only tok/s, and correctness.
- [ ] 32-token runs pass correctness for every mode.
- [ ] 64-token and 128-token density controls pass for the key modes.
- [ ] NVLink snapshot evidence is recorded for the 32-token suite.
- [ ] Results are copied into
      `logs/from-cluster/sprint227-tp8-collective-workbench/`.
- [ ] Status and vision docs are updated with the decision.
- [ ] Changes are committed with explicit `git add` paths.

## Decision

Pending.
