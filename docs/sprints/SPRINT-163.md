# Sprint 163 - One-Layer TP Routed FFN Executor

Date: 2026-05-21

## Objective

Build the first production-shaped tensor-parallel routed FFN primitive for the
V100 appliance. This sprint should prove that a real `--emit-tp-split`
TurboMind appliance pack can be consumed by runtime bindings and executed across
one NV2 GPU pair, with output compared against the existing single-GPU routed
FFN path.

This is deliberately narrower than serving integration. The goal is to retire
the main uncertainty from Sprint 162: whether the DS4 runtime can bind and
execute real TP split descriptors, not just synthetic TP fixtures.

## Scope

- One routed layer only.
- One TP2 pair only, initially `gpu0 <-> gpu3`.
- Fused interleaved gate/up and down TP descriptors only:
  - `ffn_gate_up_exps.tp0.weight`
  - `ffn_gate_up_exps.tp1.weight`
  - `ffn_down_exps.tp0.weight`
  - `ffn_down_exps.tp1.weight`
- Explicit smoke/probe path only. Do not enable in HTTP serving defaults.
- Compare routed FFN output before shared expert and before layer residual
  expansion.

## Implementation

1. Extend layer-state binding so TP split TurboMind descriptors are visible to
   runtime code without changing the existing single-GPU path.
2. Treat `_tp2` gate/up layouts as interleaved in the GPU view flags.
3. Add a focused CUDA smoke that:
   - opens a normal single-GPU TurboMind pack and a matching TP split pack;
   - binds layer state for both;
   - uploads owner and peer sidecar shards to separate arenas;
   - executes the current single-GPU routed FFN as reference;
   - executes owner half and peer half from TP descriptors;
   - peer-copies the peer partial back;
   - sums partials on the owner GPU;
   - compares against the reference output.
4. Record per-step timing for reference, owner half, peer half, copy, and sum
   when the smoke runs on the V100 cluster.

## Definition of Done

- `ds4_v100_layer_state` exposes TP split routed descriptors for a layer when
  present.
- Existing non-TP layer-state tests still pass.
- A new TP smoke builds locally and on the V100 cluster.
- The TP smoke passes correctness on a real TP split pack on an NV2 pair.
- Sprint artifacts record the exact command, correctness envelope, and timing.
- `docs/sprints/VISION.md` is updated with the Sprint 163 result.
- Changes are committed.

## Risks

- TP split packs may use bounded expert subsets from earlier experiments. The
  smoke must either use all 256 experts or route only to experts present in the
  pack and state that limitation explicitly.
- The current public GPU API uses default streams and one arena per call. The
  first implementation may prove correctness with sequential half execution
  before adding concurrent streams.
- TP down halves produce partial F32 outputs. The first executor should sum
  partials explicitly rather than trying to use the existing down-reduce
  epilogue.

## Non-Goals

- No full 8-GPU TP scheduler rewrite.
- No serving default change.
- No HTTP throughput claim until the one-layer primitive passes correctness.

## Result

Implemented the bounded runtime primitive and validated it on the V100 cluster.

Code shipped in this sprint:

- `ds4_v100_layer_state` now binds optional TP2 TurboMind routed descriptors
  when present.
- `_tp2` interleaved gate/up layouts now set the same GPU view flag as the
  normal interleaved gate/up pack.
- `ds4_gpu_enable_peer_access()` exposes a small production helper for enabling
  owner/peer copies.
- `tests/cuda_v100_tp_routed_ffn_smoke` opens real owner and peer arenas,
  executes the single-GPU routed FFN reference, executes TP owner and peer
  halves, copies the peer partial back, sums on owner, and compares output.

The Sprint 153 TP pack only contained six experts, so Sprint 163 generated a
new layer-3 TP split pack with all 256 experts:

```text
/workspace/ds4-tp-split-pack-s163
gpu0.weights = 5,133,828,096 bytes
gpu3.weights = 1,711,276,032 bytes
tm_rows = 6
experts = 256/256
```

Validation on `gpu0 <-> gpu3`:

| Shape | Ref single-GPU | Owner half | Peer half | Total TP | Speedup | Correctness |
|---|---:|---:|---:|---:|---:|---|
| 16 tokens / 96 routes | `2.1191 ms` | `1.1647 ms` | `1.1656 ms` | `1.2330 ms` | `1.719x` | PASS |
| 1 token / 6 routes | `0.2071 ms` | `0.1454 ms` | `0.1452 ms` | `0.1946 ms` | `1.064x` | PASS |

Copy and sum timing:

| Shape | Input copy | Peer output copy | Owner sum |
|---|---:|---:|---:|
| 16 tokens / 96 routes | `0.0260 ms` | `0.0102 ms` | `0.0164 ms` |
| 1 token / 6 routes | `0.0150 ms` | `0.0057 ms` | `0.0126 ms` |

Correctness envelope:

```text
tokens=16: max_abs=1.34401e-06 rel=0.000278022 bad=0 nan=0
tokens=1:  max_abs=9.16421e-07 rel=0.000276278 bad=0 nan=0
```

## Interpretation

TP is now validated as a descriptor-bound runtime primitive. The 96-route shape
is materially positive, but the one-slot 6-route shape is only modestly
positive once real descriptor execution and synchronous copies are included.

That means the next serving work should not blindly rewrite the whole scheduler
for TP. The practical next step is a guarded one-layer scheduler integration
that can run a selected layer through this TP primitive and report layer-local
timing while preserving the current layer-parallel serving path as fallback.

## Artifacts

- `logs/from-cluster/sprint163-tp-routed-ffn-smoke/run.log`
