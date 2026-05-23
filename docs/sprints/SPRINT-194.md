# Sprint 194 - TP/EP Topology Cost Contract

Date: 2026-05-23
Status: Completed

## Objective

Turn the tensor-parallel discussion into an executable appliance contract. Add a
planner tool that estimates per-token communication, activation residency, and
collective pressure for the current layer-split appliance, the rejected
routed-only TP2 overlay, and candidate full TP/EP topologies.

## Context

Recent sprint evidence is clear:

- Single-kernel substitutions are exhausted. Sprint 192's single-slot attention
  output HMMA path preserved IDs but dropped len-1024 continuation throughput
  from `14.577239` to `8.747221` tok/s.
- The easy resident F8->F16 cache shortcut is unsafe. Sprint 193 changed output
  IDs from `3955, 361` to `201, 5`.
- Routed-only TP2 overlays are correct but slower. Sprint 178's two-layer
  parallel-halves path reached `65.348426` continuation tok/s versus
  `70.185744` for the no-TP control.
- The TP2 code copies full F32 hidden/route tensors to the peer, runs two
  routed halves, copies a full F32 partial output back, and reduces every layer.
  That is not persistent tensor parallelism.

The next implementation decision is no longer whether a wrapper kernel can be
slightly faster. It is whether the appliance should pivot to full TP/EP
ownership, and if so which communication envelope is plausible on 8x V100.

## Scope

- Add `tools/ds4-v100-tp-estimate.c`.
- Build it as `tools/ds4-v100-tp-estimate`.
- Model these modes:
  - current 8-stage layer split;
  - existing routed-only TP2 overlay;
  - TP=2/4/8 with PP=1 full-layer tensor parallelism;
  - TP=4, PP=2 hybrid tensor/pipeline fallback.
- Report:
  - per-token payload for hidden handoff, routed activation dispatch, all-reduce,
    and output gather;
  - minimum NVLink transfer time at configurable GB/s;
  - whether the topology is expected to be compute-shape-positive,
    communication-positive, or likely dominated by collectives;
  - qualitative next implementation target.
- Keep it independent from runtime so it can run locally and in the V100 pod.
- Update the vision/status documents with the resulting decision rule.

## Non-Goals

- No production TP runtime in this sprint.
- No NCCL/CUDA-IPC collective implementation yet.
- No change to default serving flags.
- No claim that the model is performance optimized until V100 A/B proves it.

## Implementation Plan

1. Implement a standalone C estimator using DS4-Flash dimensions:
   `hidden=4096`, `hc=4`, `routes=6`, `layers=43`, `gpus=8`.
2. Accept `--slots`, `--ctx`, `--active-microbatch`, `--topology`,
   `--nvlink-gbps`, and `--json`.
3. Print a text table by default and machine-readable JSON with `--json`.
4. Add a local smoke path:

   ```text
   make tools/ds4-v100-tp-estimate
   ./tools/ds4-v100-tp-estimate --slots 16 --ctx 262144 --active-microbatch 16
   ```

5. Use the output to update `docs/sprints/VISION.md`:
   - routed-only TP2 remains rejected;
   - full TP/EP is only worth implementing if dense attention/shared paths are
     natively TP-owned too;
   - the next code sprint should choose either a full TP/EP stage prototype or a
     monolithic routed-FFN kernel, not another per-layer overlay.

## Definition Of Done

- [x] `tools/ds4-v100-tp-estimate` builds locally.
- [x] Text output covers layer, routed-only TP2, TP=2, TP=4, TP=8, and TP4/PP2.
- [x] JSON output is syntactically valid.
- [x] Sprint outcome records the numeric topology comparison.
- [x] Vision/status docs are updated.
- [x] Changes are committed.

## Implementation

Added `tools/ds4-v100-tp-estimate.c` and the corresponding Makefile target.
The tool is deliberately runtime-independent so it can run locally, in the V100
pod, or inside CI without requiring CUDA devices.

It uses DS4-Flash dimensions:

- layers: `43`
- hidden: `4096`
- HC lanes: `4`
- routed experts per token: `6`
- default active microbatch: `16`
- default context: `262144`
- default effective NVLink budget: `150 GB/s`

The estimates are not a substitute for V100 A/B, but they make the topology
trade explicit before another TP implementation sprint.

## Validation

Local build:

```text
make tools/ds4-v100-tp-estimate
```

Text run:

```text
./tools/ds4-v100-tp-estimate --slots 16 --active-microbatch 16 --ctx 262144
```

JSON validation:

```text
./tools/ds4-v100-tp-estimate --slots 16 --active-microbatch 16 \
  --ctx 262144 --json | python3 -m json.tool
```

Static check:

```text
git diff --check
```

All passed.

## Results

For the current practical 16-slot / 256K tier with `active_microbatch=16`:

| Topology | TP | PP | Total wire/token | Minimum transfer time at 150 GB/s | Decision |
|---|---:|---:|---:|---:|---|
| current layer split | 1 | 8 | `7.000 MiB` | `0.049 ms` | baseline |
| routed-only TP2 overlay | 2 | 8 | `21.531 MiB` | `0.151 ms` | rejected |
| full TP2/PP1 | 2 | 1 | `75.250 MiB` | `0.526 ms` | possible probe |
| full TP4/PP1 | 4 | 1 | `112.875 MiB` | `0.789 ms` | strong candidate |
| full TP8/PP1 | 8 | 1 | `131.688 MiB` | `0.921 ms` | high-risk candidate |
| TP4/PP2 hybrid | 4 | 2 | `113.875 MiB` | `0.796 ms` | fallback candidate |

The key result is directional, not the exact transfer time: routed-only TP2 is
structurally inferior because it adds traffic without changing the dense
attention/shared execution shape. Full TP/EP intentionally spends more
communication, but it only makes sense if dense attention, shared FFN, routed
experts, and output ownership are all native to the topology. A per-layer
routed-FFN overlay cannot get there.

## Outcome

The next implementation sprint should not expand the existing TP2 overlay.
Sprint 194 turns the decision into a concrete rule:

- if pursuing TP next, build a bounded full-layer TP4/PP1 prototype first;
- include dense attention projection/output and shared FFN ownership, not only
  routed experts;
- treat TP8/PP1 as a later production target after TP4 proves the collective
  and memory layout;
- use TP4/PP2 only as a memory fallback, not the first latency target.
