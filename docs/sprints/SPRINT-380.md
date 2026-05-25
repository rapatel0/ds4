# Sprint 380: TP-Sharded Expert A/B Gate

## Overview

Implement `--tp-experts-ab-gate` as the next TP/EP-only topology measurement
after Sprint 379.

The question is not "can we invent another scheduler." The question is whether
TP-sharded routed experts can beat the current EP8 all-to-all/compose path at
the real `32` slot / `256K` serving shape strongly enough to justify a serving
integration sprint.

Prior evidence:

- Sprint 211 rejected TP8 MXFP4 expert execution at `mid_shard=256` because it
  produced NaNs and simple reduction erased compute speedup.
- Sprint 211's TP4 control remained correct and showed useful compute speedup
  at the same route shapes.
- Sprint 378 promoted compact MoE for the current EP8 model-router path.
- Sprint 379 found no serving promotion for fused gated-SiLU; S-F is now the
  next Vision item.

## Scope

- Add a permanent, default-off measurement driver:

```text
tools/ds4-v100-tp-experts-ab.py
```

- Compare three expert paths at matched route-count tiers:
  - current resident EP8 direct serving path;
  - TP4 TurboMind MXFP4 expert workbench;
  - TP8 TurboMind MXFP4 expert workbench.
- Reuse existing tools rather than adding a generic scheduler:
  - `tools/ds4-v100-tp-ep-profile.py`
  - `tools/ds4-v100-tp8-turbomind-ffn-smoke`
- Record route shape, correctness, compute ms, reduction ms, total ms, direct
  serving tok/s, and decision notes.
- Keep the path separate from PP/layer-split code and from launcher defaults.

## Out Of Scope

- No PP/layer-split work.
- No generic scheduler abstraction.
- No serving integration of TP-sharded experts in this sprint.
- No MTP.
- No offline pack format change.
- No attempt to fix Sprint 211's TP8 `mid_shard=256` correctness inside this
  measurement sprint unless the failure disappears with current kernels.

## Architecture

The driver should produce one artifact directory with:

```text
summary.json
summary.tsv
ep8-direct/
tp4-turbomind/
tp8-turbomind/
```

EP8 serving control:

```text
tools/ds4-v100-tp-ep-profile.py \
  --run-mode direct-token-major \
  --tokens 1 \
  --position 262080 \
  --slots 32 \
  --model-router-routes \
  --compact-moe-decode
```

TP workbench:

```text
tools/ds4-v100-tp8-turbomind-ffn-smoke \
  --tokens-per-active {16,32,64} \
  --warmup 5 \
  --iters 50
```

The repo already contains separate TP4 and TP8 TurboMind workbenches. Sprint
380 wires both into the permanent driver instead of relying on historical TP4
logs.

## Implementation Plan

### Phase 1: Permanent Measurement Driver

- Add `tools/ds4-v100-tp-experts-ab.py`.
- Accept paths for:
  - pack directory;
  - contract;
  - TurboMind library;
  - artifact directory;
  - tokens-per-active cases.
- Run EP8 direct control once.
- Run TP expert workbench for each configured route tier.
- Parse known output lines:
  - `latency_ms full=... tp8_compute=... tp8_reduce=...`
  - `correctness routes=... max_abs=... bad=...`
  - EP8 direct `summary.json` fields.
- Write `summary.json` and `summary.tsv`.

### Phase 2: TP4 Visibility

- Inspect the existing TP4 workbench:
  `tools/ds4-v100-tp4-turbomind-layer-smoke.cu`.
- Add driver support for rerunning TP4 and TP8 with shared summary parsing.

### Phase 3: V100 Measurement

- Build the required tools on gpu-01:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp8-turbomind-ffn-smoke \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

- Run the driver at:
  - `32` slots;
  - `256K` context;
  - `position=262080`;
  - route tiers `16,32,64` tokens/active expert;
  - model-router compact-MoE EP8 control.

### Phase 4: Decision

Record one of:

- **Proceed to TP expert serving integration** if TP4 or TP8 is correct and
  total expert time plus reduction has a credible path to beat EP8 serving
  compose/all-to-all.
- **Defer TP expert integration** if TP8 still fails correctness and TP4 total
  time is erased by reduction/copy.
- **Run a focused kernel sprint** if the only blocker is a specific TurboMind
  shard shape such as TP8 `mid_shard=256`.

Update:

- `TEMP_STATUS_REPORT_380.md`
- `docs/sprints/STATUS.md`
- `docs/sprints/VISION.md`

## Definition Of Done

- `tools/ds4-v100-tp-experts-ab.py` exists and is disabled by default.
- Local Python syntax validation passes.
- V100 build passes for required CUDA tools.
- V100 driver run writes `summary.json` and `summary.tsv`.
- The report includes EP8 direct tok/s, TP workbench correctness, TP compute
  ms, TP reduction ms, total speedup, and route tiers.
- Decision is recorded in this sprint doc, status, vision, and
  `TEMP_STATUS_REPORT_380.md`.

## Risks

- Existing TP8 MXFP4 correctness may still fail exactly as Sprint 211 showed.
- Existing TP4 data may not be easily rerunnable without editing the old
  workbench.
- Synthetic expert fixtures do not equal serving integration, so this sprint
  must not overclaim.
- EP8 direct serving time includes attention/dense/control work, while TP
  workbench time is expert-only. The report must label this clearly.

## Security

No new network service. No model weights copied into logs. Artifact summaries
must contain timings, route counts, and correctness metrics only.

## Progress

Phase 1 and Phase 2 are implemented.

Added:

```text
tools/ds4-v100-tp-experts-ab.py
```

The driver now runs:

- EP8 direct serving control;
- TP4 TurboMind MXFP4 workbench;
- TP8 TurboMind MXFP4 workbench.

Local validation passed:

```text
python3 -m py_compile tools/ds4-v100-tp-experts-ab.py
git diff --check
```

V100 build validation passed:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp4-turbomind-layer-smoke \
  tools/ds4-v100-tp8-turbomind-ffn-smoke \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

First V100 driver smoke artifact:

```text
/workspace/logs/sprint380-tp-experts-ab/smoke
```

| Path | Return | Result |
|---|---:|---|
| EP8 direct serving | 0 | first token `54639`, direct decode `66.569095` tok/s, EP `18.220610` ms, compose `22.522762` ms |
| TP8 workbench, tokens/active 16 | 1 | correctness `FAIL`, routes `96`, NaNs `378153`, TP8 compute `0.087654` ms, TP8 total `0.581414` ms |

Full TP4/TP8 matrix artifact:

```text
/workspace/logs/sprint380-tp-experts-ab/tp4-tp8-matrix-parsed
```

| Path | Tokens/active | Routes | Correctness | NaNs | Full ms | TP compute ms | TP reduce ms | TP total ms | Total speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 96 | PASS | 0 | 0.292434 | 0.123945 | 0.153265 | 0.277210 | 1.055x |
| TP4 | 32 | 192 | PASS | 0 | 0.331469 | 0.132833 | 0.239310 | 0.372143 | 0.891x |
| TP4 | 64 | 384 | PASS | 0 | 0.575406 | 0.161587 | 0.459294 | 0.620881 | 0.927x |
| TP8 | 16 | 96 | FAIL | 378153 | 0.250614 | 0.071660 | 0.482224 | 0.553884 | 0.452x |
| TP8 | 32 | 192 | FAIL | 756305 | 0.295444 | 0.080056 | 0.888402 | 0.968458 | 0.305x |
| TP8 | 64 | 384 | FAIL | 1512469 | 0.604365 | 0.141844 | 1.664169 | 1.806014 | 0.335x |

## Outcome

Decision: do not integrate TP-sharded experts into serving yet.

TP8 remains rejected: the current TurboMind `mid_shard=256` MXFP4 path is
numerically invalid and total speedup is below `1.0x`.

TP4 is numerically valid and has real compute speedup, but the simple output
reduction/compose boundary erases the win at the larger route tiers. It only
beats the full reference at the smallest `96` route tier. A serving integration
sprint would be premature unless it first prototypes a better fused TP4
reduction/compose boundary.
