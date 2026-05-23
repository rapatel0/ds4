# Sprint 208 - Separate TP8 Investigation Path

Date: 2026-05-23
Status: Completed

## Overview

Create a separate tensor-parallel investigation path for DS4 on the 8x V100
host, with `PP1/TP8` as the primary target and `PP2/TP4` as a fallback/control.

This sprint is intentionally not a production scheduler integration sprint. Its
job is to answer whether full TP8 is worth implementing as a separate runtime
family for the 32-slot, 128K-256K practical-serving goal.

## Rationale

The existing appliance is a pipeline-parallel/layer-sharded runtime. It owns
contiguous layer ranges on each GPU and relays hidden context between stages.
That architecture is clear and working, but it serializes each token through
the GPU stages.

Full TP8 changes the ownership model:

- all eight GPUs participate in every layer;
- weights, KV/cache, and intermediate activations are sharded inside the TP
  group;
- hidden and partial outputs are reduced inside the layer;
- output head and future MTP load can be distributed across all GPUs;
- pipeline stage relay disappears.

Because that changes scheduler semantics, KV ownership, pack layout, runtime
entry points, and failure modes, the TP path must use separate files and
separate investigation artifacts. The PP/layer scheduler remains the baseline,
not the abstraction to stretch.

The current planner envelope for 32 slots, 256K, F8 KV, and KV sharding says
full TP8 fits:

| Topology | Config fits | Worst GPU | Max slots @256K | TP wire / step |
|---|---:|---:|---:|---:|
| PP8/TP1 layer split | yes | 28.91 GiB | 57 | 0.00 MiB |
| PP4/TP2 | yes | 27.27 GiB | 71 | 21.50 MiB |
| PP2/TP4 | yes | 26.95 GiB | 81 | 32.25 MiB |
| PP1/TP8 | yes | 26.84 GiB | 80 | 37.62 MiB |

The byte volume is not the obvious problem. The uncertainty is whether the
runtime can absorb many small synchronization points while keeping all GPUs
resident and balanced.

## Use Cases

1. As an engineer, I can run a TP-specific planner without touching the
   PP/layer planner and see whether 32-slot/128K and 32-slot/256K fit.
2. As an engineer, I can measure 8-GPU hidden-reduction costs at the actual
   32-slot target rather than extrapolating from TP4.
3. As an engineer, I can see whether replicated KV fails the memory gate and
   sharded KV passes it.
4. As an engineer, I can decide whether to build a TP8 scheduler based on
   V100 evidence, not topology preference.

## Architecture

### Separation Rule

TP work must use new files by default.

| Area | PP/layer path | TP path |
|---|---|---|
| Planner | `tools/ds4-v100-plan.c` | `tools/ds4-v100-plan-tp.c` |
| Scheduler | `ds4_v100_scheduler.*` | none in this sprint; future TP-only scheduler files if gates pass |
| Runtime entry | current replay/appliance | TP-specific runner or explicit TP entry |
| Pack layout | per-GPU layer shards | TP shard descriptors with split axes |
| KV/cache | layer-owned KV | TP-sharded KV/cache |
| Tests | PP scheduler smokes | TP-specific collective/KV/layer smokes |

Reuse is allowed only below the ownership boundary: source tensor readers,
pack helpers, stable TurboMind kernels, CUDA utility wrappers, and measurement
patterns.

### Topologies To Compare

| Topology | Meaning | Role |
|---|---|---|
| `PP8/TP1` | current layer split | baseline/control |
| `PP4/TP2` | four pipeline stages, two GPUs per stage | conservative TP control |
| `PP2/TP4` | two pipeline stages, four GPUs per stage | fallback if TP8 sync is costly |
| `PP1/TP8` | all GPUs participate in every layer | primary investigation target |

### TP Wire

`TP wire` is per-step inter-GPU traffic from tensor-parallel reductions. For
32 slots and hidden size 4096 in FP16, one hidden payload is about 256 KiB.
Even with about two reductions per layer across 43 layers, the total byte
volume is tens of MiB per decode step. The risk is not raw bandwidth; the risk
is many small rendezvous points.

### KV Sharding

KV sharding is required for 32-slot/256K TP4/TP8.

| 32 slots / 256K / F8 KV | KV sharded | KV replicated |
|---|---:|---:|
| PP2/TP4 | about 26.95 GiB, fits | about 37.10 GiB, over budget |
| PP1/TP8 | about 26.84 GiB, fits | about 50.63 GiB, over budget |

Replicated KV can be used only as a diagnostic at small context. It must not be
used as evidence that TP8 satisfies the target configuration.

## Implementation

1. **Harden the TP planner**
   - Keep `tools/ds4-v100-plan-tp.c` separate from the PP planner.
   - Add or refine output for `PP8/TP1`, `PP4/TP2`, `PP2/TP4`, and `PP1/TP8`.
   - Ensure `--kv-dtype f16|f8|q8_0` and `--kv-sharding on|off` are supported.
   - Print admission for 128K, 256K, 512K, and 1M.
   - Print per-GPU breakdown and per-step TP wire estimate.

2. **Add TP8 collective probe**
   - Create a new 8-GPU tool, not an extension hidden inside the PP scheduler.
   - Suggested file: `tools/ds4-v100-tp8-collective-smoke.cu`.
   - Measure 32, 64, and 128 active-token hidden payloads.
   - Measure root, ring/doubling, and any existing CUDA/NCCL-grade option if
     available.
   - Record timing and effective wire bandwidth.

3. **Add TP8 resident-boundary probe**
   - Suggested file: `tools/ds4-v100-tp8-layer-proxy.cu`.
   - Run 43-layer-style repeated reductions with resident GPU work between
     collectives.
   - Include at least a no-op/local-op mode and a synthetic GEMM or TurboMind
     routed-work mode.
   - Compare against the existing TP4 resident-boundary evidence.

4. **Add NVLink traffic capture**
   - Add a wrapper or script that snapshots NVLink counters before and after
     TP8 probes when `nvidia-smi nvlink` counters are available.
   - If counters are unavailable in the pod, record the command failure and
     rely on CUDA timing/effective wire bytes for this sprint.

5. **Document the decision**
   - Update `docs/architecture/DS4-V100-TP8-INVESTIGATION.md` with measured
     results.
   - Update `docs/sprints/STATUS.md` and `docs/sprints/VISION.md` with the
     decision: continue to TP8 one-layer prototype, fall back to PP2/TP4, or
     pause TP work.

## Files Summary

Expected new or updated files:

| File | Purpose |
|---|---|
| `docs/architecture/DS4-V100-TP8-INVESTIGATION.md` | Current TP8 insight and gates |
| `tools/ds4-v100-plan-tp.c` | Separate TP memory/topology planner |
| `tools/ds4-v100-tp8-collective-smoke.cu` | New 8-GPU collective benchmark |
| `tools/ds4-v100-tp8-layer-proxy.cu` | New resident TP8 boundary benchmark |
| `tools/ds4-v100-nvlink-snapshot.sh` | Optional NVLink counter wrapper |
| `Makefile` | Build targets for new tools |
| `docs/sprints/STATUS.md` | Topline result after execution |
| `docs/sprints/VISION.md` | Vision update after execution |

## Definition Of Done

- [x] `tools/ds4-v100-plan-tp` builds locally.
- [x] Planner output covers all four target topologies.
- [x] Planner output explicitly demonstrates 32-slot/256K with sharded KV and
      the replicated-KV failure case.
- [x] TP8 collective probe builds on the V100 node.
- [x] TP8 collective probe runs on all 8 GPUs for 32, 64, and 128 active-token
      payloads.
- [x] TP8 resident-boundary probe builds on the V100 node.
- [x] TP8 resident-boundary probe runs with resident work between reductions.
- [x] NVLink counter or effective-wire evidence is captured.
- [x] Results are copied into `logs/from-cluster/sprint208-tp8/`.
- [x] Architecture/status/vision docs record the decision.
- [x] Changes are committed with explicit `git add` paths.

## Decision Gate

Continue toward a TP8 one-layer runtime prototype only if:

- 32-slot/256K F8 or Q8 KV sharded planner envelope has at least about 3 GiB
  post-reserve headroom on the worst GPU;
- TP8 collective latency at 32 active tokens is not a clear non-starter versus
  the PP/layer baseline;
- resident-boundary timing improves materially at 64 or 128 active tokens, or
  is close enough at 32 tokens that full-layer compute could plausibly pay for
  synchronization;
- no evidence suggests replicated hidden/KV movement is accidentally being
  used in place of the intended sharded design.

If TP8 fails the 32-slot boundary gate, fall back to PP2/TP4 as a smaller
sync-domain control. If both fail, pause TP scheduling and return to local
kernel dataflow/MTP work.

## Execution

Sprint 208 added three TP-specific execution artifacts without touching the
PP/layer scheduler:

- `tools/ds4-v100-plan-tp.c`
- `tools/ds4-v100-tp8-collective-smoke.cu`
- `tools/ds4-v100-tp8-layer-proxy.cu`

The TP8 probes use all eight V100s, FP16 hidden payloads, peer access, and two
hand-rolled collective algorithms:

- `root`: gather to GPU0, reduce, broadcast.
- `doubling`: recursive-doubling peer exchange with local add at each step.

`tools/ds4-v100-nvlink-snapshot.sh` records `nvidia-smi nvlink --status`,
`nvidia-smi nvlink -gt d`, and `nvidia-smi topo -m` before and after each
benchmark command. In this pod, per-link byte counters report `N/A`, so the
usable traffic evidence is the benchmark's effective-wire calculation plus the
recorded NVLink link status/topology.

## Validation

Local build passed:

```text
make tools/ds4-v100-plan-tp
```

V100 build passed:

```text
make -j80 tools/ds4-v100-plan-tp \
  tools/ds4-v100-tp8-collective-smoke \
  tools/ds4-v100-tp8-layer-proxy CUDA_ARCH=sm_70
```

Planner gate at 32 slots, 256K, F8 KV:

| Topology | KV mode | Fits | Worst GPU |
|---|---|---:|---:|
| PP1/TP8 | sharded | yes | `26.84 GiB` |
| PP1/TP8 | replicated | no | `50.63 GiB` |
| PP2/TP4 | sharded | yes | `26.95 GiB` |
| PP2/TP4 | replicated | no | `37.10 GiB` |

TP8 collective smoke:

| Algo | Tokens | Avg latency | Effective wire | Correctness |
|---|---:|---:|---:|---|
| root | 32 | `0.339747 ms` | `10.802 GB/s` | ok |
| root | 64 | `0.579366 ms` | `12.669 GB/s` | ok |
| root | 128 | `1.046316 ms` | `14.030 GB/s` | ok |
| doubling | 32 | `0.322599 ms` | `13.002 GB/s` | ok |
| doubling | 64 | `0.372364 ms` | `22.528 GB/s` | ok |
| doubling | 128 | `0.436299 ms` | `38.454 GB/s` | ok |

TP8 43-layer resident-boundary proxy, 2 collectives/layer:

| Algo | Tokens | Resident work | Boundary avg | Overhead-only tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| root | 32 | 0 | `33.470735 ms` | `956.059` | ok |
| root | 64 | 0 | `52.316409 ms` | `1223.326` | ok |
| root | 128 | 0 | `93.079371 ms` | `1375.170` | ok |
| doubling | 32 | 0 | `29.381000 ms` | `1089.139` | ok |
| doubling | 64 | 0 | `32.605223 ms` | `1962.876` | ok |
| doubling | 128 | 0 | `37.994584 ms` | `3368.901` | ok |
| doubling | 32 | 64 repeats | `30.028386 ms` | `1065.658` | ok |
| doubling | 64 | 64 repeats | `32.493648 ms` | `1969.616` | ok |
| doubling | 128 | 64 repeats | `38.268012 ms` | `3344.830` | ok |

Evidence:

```text
logs/from-cluster/sprint208-tp8/
```

## Decision

Continue TP8 investigation to a bounded one-layer TP8 prototype in new TP-only
files.

The Sprint 208 evidence does not prove TP8 serving will win, but it clears the
first topology gate:

- PP1/TP8 with F8 KV sharding fits 32-slot/256K with about `5.16 GiB`
  post-reserve headroom on the worst GPU.
- Replicated KV correctly fails the same target, confirming sharded KV is a
  hard requirement.
- TP8 recursive-doubling reductions are materially better than root, especially
  at 64 and 128 tokens.
- The 32-token 43-layer boundary is not free at `29.381 ms`, but its
  overhead-only ceiling is around `1089 tok/s`, which is high enough to justify
  the next bounded TP8 layer experiment.

The next sprint should not integrate a scheduler. It should build a TP-only
one-layer prototype that keeps dense/routed work and sharded-KV ownership inside
the TP8 boundary, then compares that bounded layer against the PP/layer
baseline.

## Risks

- **Collective latency**: raw NVLink bandwidth is ample, but many small
  reductions can still dominate decode.
- **KV ownership**: sharded KV is required for target fit and changes attention
  implementation.
- **False positives from synthetic probes**: pure collectives may look good
  while resident compute imbalance breaks real runtime behavior.
- **Abstraction drift**: modifying or generalizing the PP scheduler for TP would
  create a fragile mixed runtime. This sprint must not implement a generic
  scheduler and must not add TP modes to the PP scheduler.
- **Cluster counter access**: NVLink counters may be unavailable in the pod;
  record this explicitly if blocked.

## Security

No external service exposure is planned. Cluster commands should run in the
existing V100 build pod or direct node workflow using localpool/k8s-local
storage, not the host mirror disk. Do not copy model weights into docs or logs.

## Dependencies

- V100 node access and CUDA build environment.
- Existing TurboMind copied tree and SM70 build path.
- Existing TP4 probes as references.
- Current appliance pack for baseline comparisons.

## Open Questions

- Should the first TP8 resident-boundary probe use hand-rolled peer copies or
  introduce an NCCL-grade option immediately?
- Does TP8 need a mocked sharded attention/KV gate before the first one-layer
  prototype, or can this sprint stop at planner plus boundary evidence?
- What is the minimum resident compute needed between collectives to avoid
  overfitting to a communication-only benchmark?
