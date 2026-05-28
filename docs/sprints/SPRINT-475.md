# Sprint 475: TP/EP No-SYS Fabric Guard

## Overview

This sprint makes SYS avoidance an explicit TP/EP serving invariant. The
current appliance can remap `CUDA_VISIBLE_DEVICES`, but that alone does not
prove NCCL selected a no-SYS transport path. The sprint adds harness support
for a V100 no-SYS NCCL policy, runs the topology-aware ring at the target
serving shape, and records enough NCCL/DCGM evidence to decide whether
topology is still a performance blocker.

No PP/layer-split work is in scope.

## Target Policy

Physical V100 device order stays natural:

```text
0,1,2,3,4,5,6,7
```

Default NCCL rank ring:

```text
0 -> 3 -> 2 -> 1 -> 5 -> 7 -> 6 -> 4 -> 0
```

Every ring edge is direct NVLink on the current V100 topology:

```text
0-3 NV2
3-2 NV2
2-1 NV2
1-5 NV2
5-7 NV1
7-6 NV2
6-4 NV1
4-0 NV2
```

Expected policy environment:

```text
NCCL_ALGO=Ring
NCCL_RINGS="0 3 2 1 5 7 6 4"
NCCL_P2P_LEVEL=NVL
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
NCCL_TOPO_DUMP_FILE=<artifact>/nccl-topology.xml
NCCL_GRAPH_DUMP_FILE=<artifact>/nccl-graph.xml
```

## Implementation Tasks

1. Add first-class profile/A-B flags for the no-SYS NCCL policy:
   `--nccl-no-sys-ring`, `--nccl-rings`, `--nccl-p2p-level`,
   `--nccl-debug`, and related dump controls.
2. Archive the exact NCCL environment in each profile artifact.
3. Summarize NCCL log/XML evidence in `summary.json`, including SYS/NET/ring
   mention counts.
4. Run the updated topology-aware ring at the TP/EP target diagnostic shape:
   `32` slots, `256K` context, `32` requests, `4` generated tokens.
5. Capture request-window telemetry with DCGM or dmon. Prefer DCGM fields
   `1005,1009,1010,1011,1012` for DRAM, PCIe, and NVLink bytes when available.
6. Record whether the no-SYS policy improved, regressed, or failed correctness
   relative to the existing natural-order baseline.

## Definition of Done

- The harness can launch the topology-aware no-SYS policy without ad hoc shell
  environment.
- Artifacts include `nccl-env.txt`, `server.err`, NCCL topology/graph dumps when
  the installed NCCL supports them, and GPU fabric counters.
- The sprint report states whether NCCL selected SYS/NET paths in the hot
  collective path, using concrete artifact evidence.
- If SYS cannot be ruled out from external evidence, the next action is an
  internal peer-copy accounting wrapper rather than another visible-order A/B.

## Stop Conditions

- Do not promote a no-SYS ring if response parity fails.
- Do not infer topology success from `CUDA_VISIBLE_DEVICES` alone.
- Do not start a new cluster run while the global TP/EP HTTP A/B lock is held.

## Outcome

Implemented first-class NCCL fabric controls in the TP/EP profile and A/B
harness:

- `--nccl-no-sys-ring`
- `--nccl-rings`
- `--nccl-p2p-level`
- `--nccl-debug`
- `--nccl-debug-subsys`
- `--nccl-topo-dump-file`
- `--nccl-graph-dump-file`

Each profile artifact now records `nccl-env.txt`, NCCL dump paths, and summary
fields for graph channel count, graph edge count, NV1/NV2 edge count, SYS edge
count, and SYS edge list when `nccl-graph.xml` is present.

Two target-shape runs were completed on `gpu-01`:

| Case | Visible devices | NCCL ring | HTTP | Decode tok/s | Continuation tok/s | NCCL graph | SYS graph edges | Min free |
|---|---|---|---:|---:|---:|---|---:|---:|
| Existing natural control | `0,1,2,3,4,5,6,7` | default | 32/32 | 24.933568 | 24.953894 | not captured | n/a | 1984 MiB |
| Visible-order no-SYS | `0,3,2,1,5,7,6,4` | `0 1 2 3 4 5 6 7` | 32/32 | 36.926941 | 36.838943 | captured | 0 mentions | 2086 MiB |
| Natural-order forced ring | `0,1,2,3,4,5,6,7` | `0 3 2 1 5 7 6 4` | 32/32 | 36.780967 | 36.786596 | 12 channels / 96 edges | 0 | 2086 MiB |
| Default no-SYS smoke | `0,1,2,3,4,5,6,7` | `0 3 2 1 5 7 6 4` | 8/8 | 38.441693 | n/a | 12 channels / 96 edges | 0 | 2972 MiB |

The natural-order forced ring is the preferred policy shape. It preserves the
runtime's physical rank/shard mapping while asking NCCL to use the no-SYS
physical route:

```text
0 -> 3 -> 2 -> 1 -> 5 -> 7 -> 6 -> 4 -> 0
```

The captured NCCL graph for that run classified all 96 channel edges as direct
NVLink:

```text
NV1 edges: 32
NV2 edges: 64
SYS edges: 0
```

Response parity against the older natural control was not byte-identical
(`0/32` full matches), but the mismatch is not enough to reject the topology
policy by itself: both runs served concurrent four-token decode, and the first
tokens matched while later sampled tokens drifted. Treat this as "fabric proof
passed; exact deterministic parity not established." The next same-run A/B
must either force deterministic decode or compare only deterministic logits /
first-token checksums before using response parity as a promotion gate.

## Decision

Do not use visible-device remapping as the production topology policy. It can
change runtime rank semantics and makes the pack/KV mapping harder to reason
about.

Keep natural `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and make the no-SYS NCCL
policy the default for the TP/EP launcher and profiling harness:

```text
DS4_V100_NCCL_TOPOLOGY_POLICY=no-sys
NCCL_ALGO=Ring
NCCL_RINGS="0 3 2 1 5 7 6 4"
NCCL_P2P_LEVEL=NVL
```

The default smoke at `/localpool/ds4/workspace/s475-default-nosys-smoke-s32-t1`
confirmed these values are active without passing ad hoc NCCL flags.

The production guardrail should be stronger than launch environment:

1. Parse NCCL graph dumps and fail diagnostics if any hot graph edge is `SYS`.
2. Add internal transfer accounting for every direct peer copy:
   `src`, `dst`, `bytes`, `op`, and topology class.
3. Reject direct `cudaMemcpyPeerAsync` over `SYS` in TP/EP hot paths.
4. Route unavoidable point-to-point transfers through a precomputed NVLink path
   or replace them with symmetric NCCL collectives.
