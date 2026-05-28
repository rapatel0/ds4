# TEMP Status Report 475: No-SYS NCCL Topology Guard

## Current Status

The TP/EP appliance now treats SYS avoidance as a default fabric invariant, not
as an optional experiment.

Default production shape:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_ALGO=Ring
NCCL_RINGS="0 3 2 1 5 7 6 4"
NCCL_P2P_LEVEL=NVL
```

This keeps the pack/KV rank mapping stable and moves topology control into
NCCL. Visible-device remapping is now diagnostic-only.

## Implemented

- `tools/ds4-v100-run-appliance.sh` defaults to
  `DS4_V100_NCCL_TOPOLOGY_POLICY=no-sys`.
- The launcher exports the forced no-SYS NCCL ring and NVLink-only P2P.
- The launcher rejects visible-device remapping under the no-SYS policy unless
  `DS4_V100_NCCL_ALLOW_VISIBLE_REMAP=1` is explicitly set.
- `deploy/v100/ds4-v100-appliance.env.example` and the k8s config include the
  no-SYS defaults.
- `tools/ds4-v100-tp-ep-profile.py` defaults to the same natural-order no-SYS
  NCCL policy and writes NCCL environment/artifact summaries.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py` forwards the no-SYS policy by default.

## Cluster Evidence

Artifact:

```text
/localpool/ds4/workspace/s475-nccl-natural-rank-ring-s32-t4
```

Shape:

```text
ctx=262144
slots=32
requests=32
tokens=4
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | 32/32 |
| Server generated decode tok/s | 36.780967 |
| Server continuation decode tok/s | 36.786596 |
| Client generated tok/s | 5.000133 |
| Min free VRAM | 2086 MiB |

NCCL graph summary:

| Metric | Value |
|---|---:|
| Channels | 12 |
| Graph edges | 96 |
| NV2 edges | 64 |
| NV1 edges | 32 |
| SYS edges | 0 |

The graph proof is the key result: all hot NCCL ring edges were direct NVLink
edges. No SYS edges were found.

Default smoke artifact:

```text
/localpool/ds4/workspace/s475-default-nosys-smoke-s32-t1
```

This run did not pass any ad hoc NCCL topology flags. It used the new harness
defaults and confirmed:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NCCL_ALGO=Ring
NCCL_RINGS="0 3 2 1 5 7 6 4"
NCCL_P2P_LEVEL=NVL
```

Default-smoke result:

| Metric | Value |
|---|---:|
| HTTP 200 | 8/8 |
| Server generated decode tok/s | 38.441693 |
| NCCL graph channels | 12 |
| NCCL graph edges | 96 |
| NV2 edges | 64 |
| NV1 edges | 32 |
| SYS edges | 0 |
| Min free VRAM | 2972 MiB |

## Caveat

Response parity against an older natural-order control was not byte-identical:
`0/32` full matches. The first generated token matched, while later sampled
tokens drifted under concurrent four-token decode. I am treating that as
insufficient for a deterministic correctness rejection of the topology policy.

Future promotion A/Bs should either force deterministic decode or compare
deterministic logits/first-token checksums before using full response text as
the topology gate.

## Next Work

1. Add internal peer-copy accounting in the TP/EP runtime:
   `src`, `dst`, `bytes`, `op`, topology class.
2. Fail diagnostics if a hot direct peer copy attempts a `SYS` edge.
3. Route unavoidable point-to-point transfers through a precomputed NVLink path
   or replace them with NCCL collectives.
4. Rerun the locked steady-state profile with the default no-SYS launcher path
   and compare against the Sprint 474 baseline.
