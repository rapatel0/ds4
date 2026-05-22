# TEMP Status Report 006

Date: 2026-05-22

## Topline

We are not done with the high-throughput serving objective yet. The appliance is
real and runnable on the 8x V100 pod with the persistent production pack at:

```text
/workspace/packs/ds4-appliance-full-tm-gated-s181
```

Current best comparable metrics:

| Mode | Metric | Result | Notes |
|---|---:|---:|---|
| Persistent production pack, 16-slot/256K sustained | generated tok/s | `48.163685` | Sprint 181, `16/16` match |
| Persistent production pack, 16-slot/256K sustained | continuation tok/s | `47.411127` | Sprint 181 |
| Persistent production pack, 1-slot/256K | generated tok/s | `10.357728` | Sprint 181 |
| Persistent production pack, 1-slot/256K | continuation tok/s | `10.195888` | Sprint 181 |
| Direct synthetic filled-context, len-1024/ctx-256K | prompt tok/s | `14.381306` | Sprint 187 patched profile |
| Direct synthetic filled-context, len-1024/ctx-256K | continuation tok/s | `14.282227` | Sprint 187 patched profile |
| Direct synthetic filled-context, len-4096/ctx-256K | prompt tok/s | `14.217155` | Sprint 186 |
| Direct synthetic filled-context, len-4096/ctx-256K | continuation tok/s | `13.354373` | Sprint 186 |

Older served-path short-generation tests reached about `70-71` generated tok/s
on 16-slot/256K, but the current persistent-pack baseline after pod recycle is
Sprint 181's `48.16` generated / `47.41` continuation tok/s. Treat those as
different harness regimes until we rerun the same served benchmark on the
persistent pack.

## What Changed Since The Last Report

Sprint 187 repaired direct synthetic profiling. The first len-1024
`--profile-decode` run completed but reported zero `stage_profile` buckets
because the single-slot layer executor did not populate timing fields. The
batch path did, the direct single path did not.

Patch:

```text
ds4_v100_layer_execute.c
```

The single-slot HC decode path now records the same profile buckets as the
batch path:

- HC attention prep
- attention
- HC FFN prep
- FFN
- HC final expansion

Validation:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed on `llm/llamacpp-build-8gpu`.

## Sprint 187 Profile Signal

Patched direct synthetic len-1024 / ctx-262144 profile:

```text
prompt replay:        71203.549 ms
prompt tok/s:         14.381306
continuation tok/s:   14.282227
output ids:           926, 926
```

Profile bucket sums:

| Bucket | Sum ms | Share |
|---|---:|---:|
| Attention | `37779.266` | `56.8%` |
| FFN | `21473.437` | `32.3%` |
| HC FFN prep | `3754.986` | `5.6%` |
| HC attention prep | `3069.656` | `4.6%` |
| HC final | `443.923` | `0.7%` |
| Total profiled | `66521.268` | `100.0%` |

Summed handoff was only `219.707 ms`, so inter-stage transfer is not the
current filled-context bottleneck in this direct profile. The hot path is
attention/KV first, FFN second.

Evidence:

```text
logs/from-cluster/sprint187-synthetic-prompt-profile/
```

## Techniques Explored So Far

Explored and mostly exhausted as topline levers:

- Wider slot admission at short context.
- Slot chunking to expose denser routed shapes.
- Fixed-shape TurboMind routed executors for served shapes.
- Six-route fused routed wrappers.
- Down-reduce epilogues.
- Route-row reduce variants.
- Stream-per-expert host software pipelining.
- Scheduler-side FFN microbatch wavefront.
- One-layer and two-layer TP/EP overlays with copy-back/reduce.
- Host-thread parallel TP halves.
- Wrapper-level CUDA graph attempts.
- Online single-token attention gate.

Useful but not production-promoted:

- MTP verify works for active microbatch, but true MTP speedup is not shipped
  because verify still recomputes the base target path.
- Tensor-parallel primitives and pack descriptors exist, but the current
  overlay returns/reduces full hidden state per layer and regresses served
  throughput.
- Online attention showed a material short A/B gain but diverged by token 6 in
  an 8-token direct compare, so it remains default-off.

## Current Interpretation

For `>=256K` practical use, the latest filled-context evidence points away from
more handoff or wrapper scheduling work. The two credible next implementation
directions are:

1. Attention/KV execution: make the long-context decode path cheaper without
   changing quality.
2. Broader persistent TP/EP or fused execution boundary: avoid the current
   per-layer copy-back/reduce penalty if we revisit tensor/expert parallelism.

The immediate next sprint should be selected from those two directions. Given
Sprint 187's profile, attention/KV is now the more direct evidence-backed
target for filled-context behavior.
