# Sprint 195 - TP4 Hidden Collective Smoke

Date: 2026-05-23
Status: Completed

## Objective

Implement and validate the first missing runtime primitive for a real TP4/PP1
prototype: a four-GPU hidden-state collective smoke/benchmark for DS4's
`active_microbatch x hidden` payload.

## Context

Sprint 194 showed that the rejected routed-only TP2 overlay should not be
expanded. It copies full F32 hidden state into and out of a routed-only peer
path while the dense attention/shared execution shape remains unchanged.

The next TP path must be full-layer ownership. Before modifying layer execution,
we need a measured, repeatable primitive for the communication payload that a
TP4 layer will require. The model-level estimator says TP4/PP1 at 16-slot /
256K has a communication envelope around `112.875 MiB` per token, but that
number is only useful if we can measure the underlying hidden collectives on the
actual V100 NVLink topology.

## Scope

- Add `tools/ds4-v100-tp4-collective-smoke.cu`.
- Build it as `tools/ds4-v100-tp4-collective-smoke` on CUDA hosts.
- Keep Darwin/non-CUDA hosts fail-closed with a clear Makefile message.
- Implement a root-based four-GPU all-reduce smoke:
  - allocate one `tokens x hidden` F32 tensor per participant;
  - gather peer tensors to the root through CUDA peer copies;
  - reduce on the root with a CUDA kernel;
  - broadcast the reduced tensor back to all participants;
  - verify all participants receive the same reduced values.
- Report average/min/max latency and effective wire GB/s.
- Default shape: `devices=0,1,2,3`, `tokens=16`, `hidden=4096`.

## Non-Goals

- No NCCL dependency.
- No production collective scheduler integration.
- No optimized ring/tree all-reduce in this sprint.
- No full TP layer execution yet.

## Implementation Plan

1. Create a standalone CUDA tool with CLI flags:
   - `--devices 0,1,2,3`
   - `--tokens N`
   - `--hidden N`
   - `--warmup N`
   - `--iters N`
2. Enable peer access between the selected devices when possible.
3. Allocate and initialize deterministic F32 input tensors on each GPU.
4. Run warmup iterations, then timed iterations using CUDA events on the root
   stream.
5. Verify the final broadcast on every device against the expected sum.
6. Add Makefile build/clean wiring.
7. Run local build behavior and, if the V100 pod is reachable, build and run the
   tool there.

## Definition Of Done

- [x] CUDA target is wired into the Makefile.
- [x] Tool validates arguments and visible device count.
- [x] Tool verifies reduced output correctness.
- [x] Tool reports latency and effective wire bandwidth.
- [x] V100 build/run evidence is recorded when the cluster is reachable.
- [x] Vision/status docs are updated with the result.
- [x] Changes are committed.

## Validation

Local non-CUDA behavior:

```text
$ make tools/ds4-v100-tp4-collective-smoke
tools/ds4-v100-tp4-collective-smoke requires a CUDA build
make: *** [tools/ds4-v100-tp4-collective-smoke] Error 2
```

Static check:

```text
$ git diff --check
```

V100 build:

```text
$ make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp4-collective-smoke
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -arch=sm_70 ...
```

Topology:

```text
GPU0-3 are one NVLink island with NV1/NV2 links.
GPU4-7 are one NVLink island with NV1/NV2 links.
Cross-island paths include SYS links.
```

TP4 collective results:

| Devices | Tokens | Hidden | Bytes/tensor | Avg ms | Min ms | Max ms | Effective wire GB/s | Verify |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0,1,2,3 | 16 | 4096 | 262144 | 0.114019 | 0.105408 | 0.117024 | 13.795 | ok |
| 4,5,6,7 | 16 | 4096 | 262144 | 0.109620 | 0.108160 | 0.116576 | 14.348 | ok |
| 0,1,2,3 | 64 | 4096 | 1048576 | 0.269475 | 0.257792 | 0.280480 | 23.347 | ok |
| 0,1,2,3 | 128 | 4096 | 2097152 | 0.502929 | 0.491648 | 0.517984 | 25.019 | ok |
| 0,1,2,3 | 256 | 4096 | 4194304 | 0.965824 | 0.958688 | 0.982240 | 26.056 | ok |
| 0,1,2,3 | 512 | 4096 | 8388608 | 1.856035 | 1.848352 | 1.866688 | 27.118 | ok |
| 4,5,6,7 | 512 | 4096 | 8388608 | 1.855423 | 1.849728 | 1.863936 | 27.127 | ok |
| 0,1,2,3 | 1024 | 4096 | 16777216 | 3.671276 | 3.661920 | 3.677344 | 27.419 | ok |

## Decision

The TP4 hidden collective is functionally correct on both local NVLink islands,
but the naive root gather/reduce/broadcast implementation is not the production
collective we want. It pays roughly `0.11 ms` even for the 16-token hidden
payload and tops out near `27 GB/s` effective wire bandwidth at larger payloads.

This does not reject full TP4/PP1. It says that the first full-layer TP4
prototype should stay inside one four-GPU NVLink island and should use a better
collective plan before promotion:

- NCCL all-reduce if dependency policy allows it;
- or a repo-owned ring/tree all-reduce using peer copies and device-side reduce;
- or a fused collective boundary that keeps partial hidden state resident and
  avoids materializing full F32 hidden tensors between substeps.

Do not base production TP4 on this root collective. Use it as the correctness
and measurement floor.
