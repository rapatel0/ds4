# DS4 V100 TP8 Investigation Note

Date: 2026-05-23
Status: Investigational

## Decision

Treat tensor parallelism as a separate runtime family, not as another mode of
the current layer-sharded appliance scheduler.

The existing runtime is effectively a pipeline-parallel/layer-sharded path:
each GPU owns a contiguous layer range, keeps that layer range's weights and KV
resident, and sends hidden context across stage boundaries. That is the right
shape for the current production appliance, but it is the wrong abstraction for
full tensor parallelism.

The TP path should therefore use new files by default:

| Area | Existing PP/layer path | TP investigation path |
|---|---|---|
| Planner | `tools/ds4-v100-plan.c` | `tools/ds4-v100-plan-tp.c` and successors |
| Scheduler | `ds4_v100_scheduler.*`, replay path | no scheduler in first investigation; future TP-only scheduler files if gates pass |
| Runtime entry | current appliance launcher/replay | TP-specific runner or explicit TP mode entry |
| Pack manifest | per-GPU layer shards | TP shard descriptors with split-axis metadata |
| KV/cache | layer-owned KV on one GPU | TP-sharded KV/cache inside the TP group |
| Collectives | hidden-context relay at PP boundaries | hidden/partial reductions inside layers |
| Tests | current scheduler/replay smokes | TP-specific planner, collective, KV, and layer smokes |

Reuse should happen only below the ownership boundary: GGUF/source readers,
pack helpers, stable TurboMind kernels, CUDA utilities, logging helpers, and
measurement harness patterns. The current PP scheduler should not become a
polymorphic scheduler, and Sprint 208 should not add any generic scheduler
implementation. If TP earns runtime work, it should start as a completely
separate TP-only code path.

## Why TP8 Is Back On The Table

Earlier TP4 experiments were scoped around smaller route shapes and partial
routed-FFN overlays. Those tests showed useful facts:

- routed MXFP4 expert compute can scale across GPUs when the shape is large
  enough;
- routed-only copy-in/copy-out erases much of the compute win;
- small-payload collectives can dominate when measured as many tiny barriers;
- full-layer resident TP is more plausible than a routed-only overlay.

The practical target has shifted toward 32 active slots and 128K-256K context.
That changes the decision:

- 32 slots create a denser decode shape than the six-route single-token path;
- the per-step TP wire volume is still small relative to V100 NVLink bandwidth;
- full `PP1/TP8` removes pipeline stage relay and lets all GPUs work on every
  layer;
- quantized, sharded KV makes 32-slot/256K VRAM fit plausible.

The current planner envelope for 32 slots, 256K, F8 KV, KV sharded:

| Topology | Fits | Worst GPU | Max slots @256K | Estimated TP wire / decode step |
|---|---:|---:|---:|---:|
| PP8/TP1 layer split | yes | 28.91 GiB | 57 | 0.00 MiB |
| PP4/TP2 | yes | 27.27 GiB | 71 | 21.50 MiB |
| PP2/TP4 | yes | 26.95 GiB | 81 | 32.25 MiB |
| PP1/TP8 | yes | 26.84 GiB | 80 | 37.62 MiB |

For MTP-on with the same assumptions, `PP1/TP8` remains plausible in the
planner envelope at about 27.29 GiB worst GPU because the output/MTP burden is
split across all eight devices.

## What TP Wire Means

`TP wire` is estimated inter-GPU traffic from tensor-parallel reductions during
one decode step. It is not resident memory.

For 32 slots and hidden size 4096 in FP16:

```text
one hidden payload = 32 * 4096 * 2 bytes = 256 KiB
```

If the implementation needs roughly two hidden reductions per layer across 43
layers, the total traffic is tens of MiB per decode step. Against a V100 SXM2
NVLink budget around 150 GB/s bidirectional per card and about 900 GB/s
aggregate, this byte volume is not the obvious blocker.

The real risk is synchronization:

```text
43 layers * ~2 reductions/layer = ~86 rendezvous points per decode step
```

So the key TP experiment is not "can NVLink move 37 MiB?" It is "can the
runtime keep the 8 GPUs resident, balanced, and synchronized without turning
many small collectives into a latency wall?"

## KV Sharding Requirement

For 32 slots at 256K context, TP with replicated KV is not viable.

Planner sensitivity:

| 32 slots / 256K / F8 KV | KV sharded | KV replicated |
|---|---:|---:|
| PP2/TP4 | about 26.95 GiB, fits | about 37.10 GiB, over budget |
| PP1/TP8 | about 26.84 GiB, fits | about 50.63 GiB, over budget |

Therefore full TP8 requires sharded KV/cache ownership. Each GPU should own a
slice of each TP layer's KV/cache state and attention must be implemented so
the partial result can be reduced or gathered at a planned boundary. Replicated
KV can remain a diagnostic fallback for small contexts, but it cannot be the
production design for 32-slot/256K.

## Current TP8 Hypothesis

For the target serving shape, the preferred investigation order is:

1. `PP1/TP8` as the primary target.
2. `PP2/TP4` as fallback/control.
3. Current `PP8/TP1` layer split as production baseline.

The reason to prefer `PP1/TP8` is not only memory fit. It also changes the
execution shape:

- all GPUs participate in every layer;
- pipeline stage latency disappears;
- output head and optional MTP load can be split across all GPUs;
- per-GPU expert/KV working sets shrink;
- dense, shared, and routed computation can be fused within one full-layer TP
  boundary instead of crossing the PP scheduler boundary.

The likely downside is not raw NVLink bandwidth. The likely downside is
collective latency, route imbalance, and the engineering cost of sharded KV and
new scheduler semantics.

## Required Gates Before Production TP Work

1. **TP planner gate**
   - `tools/ds4-v100-plan-tp.c` must report 32-slot/128K and 32-slot/256K
     admission for `PP1/TP8` with F8/Q8 KV sharding.
   - It must also print the replicated-KV failure case so the design does not
     accidentally drift into a non-fitting topology.

2. **TP8 boundary gate**
   - Measure 8-GPU hidden reductions at 32, 64, and 128 active-token payloads
     over 43 layers.
   - Capture both timing and NVLink traffic/counter evidence where available.

3. **TP8 resident-compute gate**
   - Add enough resident GPU work between reductions to mimic a full-layer
     schedule, not a pure communication microbenchmark.
   - Compare against current PP/layer baseline timing assumptions.

4. **TP8 sharded-KV gate**
   - Implement a bounded KV/cache ownership smoke where each GPU owns only its
     shard.
   - Prove index/shape correctness at 128K and 256K planning sizes without
     allocating a replicated cache.

5. **TP8 one-layer execution gate**
   - Future work only after Sprint 208: build a bounded TP8 layer slice with
     sharded KV placeholders or a real attention shard, routed FFN shard, and
     final hidden reduction.
   - This must live in new TP files and must not modify the production PP
     scheduler or introduce a generic scheduler abstraction.

## Non-Goals For The First TP8 Sprint

- Do not retrofit the current layer scheduler into a TP scheduler.
- Do not make TP8 the default serving path.
- Do not implement full model generation in TP8 before the boundary/KV gates.
- Do not use replicated KV as evidence that 32-slot/256K is viable.
- Do not combine this with MTP token commit; MTP remains a later acceleration
  once the TP base forward is coherent.

## Practical Conclusion

There is no clear raw-bandwidth downside to full TP8 for the 32-slot target.
The material unknown is whether many small per-layer collectives can be made
cheap enough when resident compute is present and whether sharded KV can be
implemented without fighting the existing PP abstractions.

That is a strong reason to investigate TP8, but also a strong reason to keep it
as a separate code path until it earns production integration.

## Sprint 208 Evidence

Sprint 208 built the first separate TP8 investigation tools and ran them on the
8x V100 pod:

- `tools/ds4-v100-plan-tp`
- `tools/ds4-v100-tp8-collective-smoke`
- `tools/ds4-v100-tp8-layer-proxy`
- `tools/ds4-v100-nvlink-snapshot.sh`

Planner result at 32 slots, 256K, F8 KV:

| Topology | KV mode | Fits | Worst GPU |
|---|---|---:|---:|
| PP1/TP8 | sharded | yes | `26.84 GiB` |
| PP1/TP8 | replicated | no | `50.63 GiB` |
| PP2/TP4 | sharded | yes | `26.95 GiB` |
| PP2/TP4 | replicated | no | `37.10 GiB` |

TP8 collective result:

| Algo | Tokens | Avg latency | Effective wire | Correctness |
|---|---:|---:|---:|---|
| root | 32 | `0.339747 ms` | `10.802 GB/s` | ok |
| root | 64 | `0.579366 ms` | `12.669 GB/s` | ok |
| root | 128 | `1.046316 ms` | `14.030 GB/s` | ok |
| doubling | 32 | `0.322599 ms` | `13.002 GB/s` | ok |
| doubling | 64 | `0.372364 ms` | `22.528 GB/s` | ok |
| doubling | 128 | `0.436299 ms` | `38.454 GB/s` | ok |

TP8 43-layer resident-boundary result, with two collectives per layer:

| Algo | Tokens | Resident work | Boundary avg | Overhead-only tok/s |
|---|---:|---:|---:|---:|
| root | 32 | 0 | `33.470735 ms` | `956.059` |
| root | 64 | 0 | `52.316409 ms` | `1223.326` |
| root | 128 | 0 | `93.079371 ms` | `1375.170` |
| doubling | 32 | 0 | `29.381000 ms` | `1089.139` |
| doubling | 64 | 0 | `32.605223 ms` | `1962.876` |
| doubling | 128 | 0 | `37.994584 ms` | `3368.901` |
| doubling | 32 | 64 repeats | `30.028386 ms` | `1065.658` |
| doubling | 64 | 64 repeats | `32.493648 ms` | `1969.616` |
| doubling | 128 | 64 repeats | `38.268012 ms` | `3344.830` |

The NVLink snapshot wrapper recorded link status and topology. Per-link byte
counters reported `N/A` inside the pod, so the evidence for traffic volume is
the benchmark's effective-wire calculation rather than hardware byte counters.

Decision: continue TP8 to a bounded one-layer prototype in new TP-only files.
The boundary is not free, but it is not a clear non-starter at 32 slots and
improves substantially at 64 and 128 tokens. The next gate must keep actual
dense/routed work and sharded-KV ownership inside the TP8 boundary.
