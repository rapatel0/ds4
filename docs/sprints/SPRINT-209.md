# Sprint 209 - Bounded TP8 One-Layer Prototype

Date: 2026-05-23
Status: Completed

## Overview

Build a bounded one-layer TP8 prototype in completely separate TP-only files.
This sprint should answer whether the Sprint 208 TP8 boundary remains plausible
when a layer-like resident compute body and sharded-KV ownership are inside the
same TP8 boundary.

This is not a scheduler sprint.

## Rationale

Sprint 208 showed that full `PP1/TP8` is not disqualified by memory or raw
collective timing:

| Gate | Result |
|---|---:|
| 32-slot/256K PP1/TP8 with F8 KV sharded | `26.84 GiB`, fits |
| 32-slot/256K PP1/TP8 with replicated KV | `50.63 GiB`, fails |
| TP8 doubling collective, 32 tokens | `0.322599 ms` |
| TP8 doubling 43-layer boundary, 32 tokens | `29.381000 ms` |
| TP8 doubling 43-layer boundary, 128 tokens | `37.994584 ms` |

That clears the first gate but does not prove a real TP layer. The next useful
question is whether an actual TP8 layer-shaped prototype can keep work resident,
avoid replicated KV, and preserve the hidden-state reduction contract.

The prototype must not modify or abstract the existing PP/layer scheduler.

## Scope

1. Add new TP-only files for the bounded one-layer prototype.
2. Define a small TP8 layer ownership model local to the prototype:
   - eight participants;
   - one DS4 layer id;
   - hidden shard/reduction policy;
   - KV shard descriptor;
   - compute body selection.
3. Add a sharded-KV allocation/descriptor smoke for DS4 ratio-4 and ratio-128
   layer shapes at 128K and 256K planning contexts.
4. Add a resident compute body between TP8 reductions:
   - preferred first body: synthetic DS4-shaped projection and routed-like
     split/reduce using FP16 hidden payloads;
   - stretch body: TurboMind-backed routed FFN shard if it can be adapted
     without touching PP runtime files.
5. Run 32, 64, and 128 token shapes on all eight V100s.
6. Compare timing against Sprint 208 boundary-only evidence.
7. Record the decision for Sprint 210.

## Non-Goals

- No generic scheduler.
- No PP scheduler changes.
- No launcher default changes.
- No full-model TP serving.
- No MTP integration.
- No replicated-KV success claim.
- No attempt to clean up or commit unrelated Sprint 207 kernel/runtime edits.

## Architecture

Suggested new files:

| File | Purpose |
|---|---|
| `tools/ds4-v100-tp8-layer-smoke.cu` | Main bounded one-layer TP8 executable |
| `tools/ds4-v100-tp8-kv-shard-smoke.c` | Host-side KV shard descriptor/admission smoke, if useful |
| `docs/sprints/SPRINT-209.md` | Sprint plan and outcome |
| `logs/from-cluster/sprint209-tp8-layer/` | V100 evidence |

If shared helpers are needed, they should be local to new TP8 files first.
Extract only after repetition proves the boundary is stable.

### KV Shard Contract

For a layer with `kv_bytes(layer, ctx, dtype)`, each GPU in TP8 owns:

```text
ceil(kv_bytes / 8)
```

The smoke should report per-GPU shard bytes for:

- ratio-4 layer at 128K and 256K;
- ratio-128 layer at 128K and 256K;
- F8/Q8 KV first, F16 as a comparison.

It should explicitly reject replicated allocation for the 32-slot/256K TP8
target.

### One-Layer Compute Contract

The first prototype does not need to reproduce full DS4 numerics. It must prove
the TP8 runtime shape:

```text
resident hidden shards
  -> layer-like local compute
  -> TP8 hidden reduction
  -> optional second layer-like local compute
  -> TP8 hidden reduction
  -> identical reduced hidden on all participants
```

The timing output should include:

- total layer time;
- reduction time if separable;
- resident compute time if separable;
- effective wire GB/s;
- overhead-only tok/s;
- correctness result.

## Definition Of Done

- [x] Sprint plan exists.
- [x] New TP-only one-layer prototype file exists.
- [x] No PP scheduler files are modified for this sprint.
- [x] KV shard descriptor/admission smoke reports 128K and 256K per-GPU bytes.
- [x] V100 build passes with `CUDA_ARCH=sm_70`.
- [x] 32, 64, and 128 token TP8 one-layer runs pass correctness.
- [x] Results are copied to `logs/from-cluster/sprint209-tp8-layer/`.
- [x] Sprint 209 document records validation and decision.
- [x] Status/Vision documents are updated.
- [x] Changes are committed with explicit `git add` paths.

## Execution

Implemented `tools/ds4-v100-tp8-layer-smoke.cu` as a standalone TP-only
prototype. It does not call or modify the PP/layer scheduler. The executable:

- enables peer access across eight V100s;
- allocates hidden buffers on every participant;
- allocates a per-GPU TP8 KV shard from the DS4 ratio/dtype/context/slot
  descriptor, not replicated logical KV;
- runs two resident synthetic layer compute phases;
- runs two TP8 hidden reductions using root or recursive doubling, with
  recursive doubling as the default;
- reports total, compute, reduction, per-reduction, effective-wire, and token
  rate metrics;
- verifies identical reduced hidden state across all participants.

The existing `tools/ds4-v100-tp8-kv-shard-smoke.c` was also kept as the
host-side descriptor/admission check and used for 128K and 256K evidence.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp8-layer-smoke CUDA_ARCH=sm_70
```

Cluster evidence is in `logs/from-cluster/sprint209-tp8-layer/`.

### KV Shard Admission

At 32 slots:

| Context | Layer class | F8 per-layer/slot | F8 replicated/layer | F8 TP8 shard/layer/GPU | Coverage |
|---:|---|---:|---:|---:|---|
| 128K | ratio-4 | `20.219 MiB` | `0.632 GiB` | `0.079 GiB` | ok |
| 128K | ratio-128 | `0.567 MiB` | `0.018 GiB` | `0.002 GiB` | ok |
| 256K | ratio-4 | `40.375 MiB` | `1.262 GiB` | `0.158 GiB` | ok |
| 256K | ratio-128 | `1.071 MiB` | `0.033 GiB` | `0.004 GiB` | ok |

The descriptor smoke also reports Q8 and F16 comparison rows. The key
admission result is that TP8 shard descriptors cover the logical KV without
allocating a replicated copy on every GPU.

### TP8 One-Layer Smoke

Configuration:

```text
algo=doubling
devices=0,1,2,3,4,5,6,7
hidden=4096
ctx=262144
slots=32
ratio=4
kv_dtype=f8_e4m3_b128
kv_shard_bytes=169347072
compute_repeats=64
warmup=3
iters=20
```

| Tokens | Total avg | Compute avg | Reduce avg | Per reduction | Effective wire | Prototype tok/s | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | `0.739408 ms` | `0.104600 ms` | `0.634680 ms` | `0.317340 ms` | `19.826 GB/s` | `43277.895` | ok |
| 64 | `0.876011 ms` | `0.157262 ms` | `0.718601 ms` | `0.359300 ms` | `35.021 GB/s` | `73058.460` | ok |
| 128 | `1.098461 ms` | `0.257736 ms` | `0.840586 ms` | `0.420293 ms` | `59.877 GB/s` | `116526.647` | ok |

The prototype uses synthetic resident compute, so the `tok/s` column is not a
serving throughput claim. It is a boundary-shape measurement.

## Decision

Continue the TP8 branch, but keep it in separate TP-only files.

The Sprint 209 gate passes:

- sharded KV ownership is represented without accidental replication;
- 32/64/128 token TP8 one-layer runs pass correctness on all eight V100s;
- the one-layer resident boundary stays far below the Sprint 208 43-layer
  envelope;
- the timing improves with larger token shapes, so it is not purely dominated
  by synchronization.

This does not justify scheduler integration yet. The next sprint should replace
the synthetic resident body with a real TP-only DS4 layer slice: attention/KV
descriptor flow plus routed/shared FFN shard execution in new TP files. Do not
add a generic scheduler and do not retrofit TP into `ds4_v100_scheduler.*`.

## Decision Gate

Continue to a TP8 runtime branch only if:

- the bounded one-layer prototype remains within the Sprint 208 timing envelope
  plus the cost of useful resident compute;
- sharded-KV ownership is represented without accidental replication;
- correctness passes at 32, 64, and 128 tokens;
- the result plausibly beats or complements the current PP/layer serving path
  at 32-slot/128K-256K.

If the one-layer prototype is dominated by synchronization at 32 tokens and
does not improve materially at 64/128, pause TP8 runtime work and plan a
different high-throughput lever.

## Risks

- Synthetic compute may underrepresent real DS4 attention and routed FFN costs.
- A TurboMind-backed body may be too large for this sprint if it requires pack
  changes.
- Sharded KV descriptor correctness does not equal full sharded attention
  correctness.
- The prototype may look good but still fail once routing imbalance is real.

## Security

No service exposure. Run on the existing V100 build pod or direct node workflow.
Do not copy model weights into logs.

## Dependencies

- Sprint 208 TP8 planner and collective probes.
- V100 build pod.
- Existing TurboMind probe code as optional reference only.
