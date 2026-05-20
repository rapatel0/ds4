# Sprint 072: MTP Commit Throughput Decision Gate

## Status

Complete.

## Overview

Sprint 071 shipped exact-verified one-slot MTP commit. That proves safe state
mutation, but it does not prove practical speed because exact verification
still computes the target verifier token.

Sprint 072 turns this into measured serving evidence. Extend the sustained
decode benchmark so it can run `off`, `verify`, and `commit` MTP modes with the
same request shape, record MTP counters/timing, run the comparison on the V100
cluster, and update the vision with the next throughput decision.

## Goals

1. Add MTP mode support to `tools/ds4-v100-sustained-decode-bench.sh`.
2. Preserve the existing non-MTP benchmark behavior as the default.
3. Record per-case MTP stats from response JSON and server status:
   - requests;
   - drafts;
   - accepted;
   - rejected;
   - committed;
   - average draft ms;
   - total draft ms.
4. Guard MTP benchmark cases to one slot until the runtime has per-slot MTP
   scratch/raw state.
5. Run V100 comparison for `off`, `verify`, and `commit` on the same fixture.
6. Decide whether exact commit should lead to skip-verify/recursive MTP work or
   whether practical throughput should pivot back to stage/kernel scheduling.

## Non-Goals

- Implementing unsafe skip-verify.
- Recursive MTP drafting.
- Multi-slot MTP.
- Changing MTP kernels.
- Changing the practical 4-slot default appliance profile.

## Implementation

1. Add benchmark flags:
   - `--mtp-model FILE`;
   - `--mtp-serving off|verify|commit`;
   - `--mtp-top-k N`;
   - `--mtp-gpu N`;
   - `--mtp-reserve-mib N`.
2. Forward MTP flags to `tools/ds4-v100-replay` when enabled.
3. Extract MTP fields from successful response JSON in the load client.
4. Merge server `mtp` status into each case result.
5. Add MTP columns to the TSV summary while keeping existing columns stable.
6. Run cluster benchmarks with the same model, prompt, context, request count,
   and token count for `off`, `verify`, and `commit`.
7. Add Sprint 072 report and update `docs/sprints/VISION.md`.

## Definition of Done

- [x] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [x] Local object compile is unaffected.
- [x] `git diff --check` passes.
- [x] V100 build passes for `tools/ds4-v100-replay`.
- [x] Non-MTP sustained bench still runs with the default `--mtp-serving off`.
- [x] V100 verify and commit sustained bench cases run with one slot and report
  MTP counters.
- [x] TSV/JSON artifacts compare off, verify, and commit throughput.
- [x] Sprint report records the measured decision.
- [x] Vision document is updated.

## Outcome

`SHIP`, with a throughput decision to pivot away from more exact-verify MTP
work for the next optimization sprint.

The sustained benchmark now supports `--mtp-serving off|verify|commit`, forwards
the sidecar model and placement flags, records client-side MTP attempts,
accepts, rejects, commits, draft timing, and merges server status snapshots into
case JSON.

Cluster evidence on the V100 pod used the same 1M-context, one-slot,
two-token, four-request fixture for `off`, `verify`, and `commit`:

| Mode | Generated tok/s | Continuation tok/s | Avg latency ms | MTP attempted | MTP accepted | MTP committed |
|---|---:|---:|---:|---:|---:|---:|
| off | `0.788607` | `0.394304` | `2535.818` | `0` | `0` | `0` |
| verify | `0.774126` | `0.387063` | `2583.175` | `4` | `4` | `0` |
| commit | `0.777308` | `0.388654` | `2572.592` | `4` | `4` | `4` |

Exact commit is correctness-positive: every measured draft was accepted and
committed, and the one-slot commit path mutates serving state safely. It is not
throughput-positive on this fixture because the target verifier token is still
computed and the MTP sidecar adds work. The next high-throughput sprint should
return to stage/kernel throughput before pursuing recursive or skip-verify MTP.

Artifacts:

- `logs/from-cluster/sprint072-mtp-off`
- `logs/from-cluster/sprint072-mtp-verify`
- `logs/from-cluster/sprint072-mtp-commit`
- `logs/from-cluster/sprint072-mtp-default-off`
- `logs/from-cluster/sprint072-mtp-comparison`

## Risks

- The short fixture reaches EOS quickly, so this benchmark is a decision gate
  for exact commit overhead, not a full long-generation MTP acceptance study.
- Exact commit is expected to be slower than `off` because it adds MTP work
  without removing target verification.
- A future recursive MTP sprint will need a different benchmark prompt set and
  acceptance analysis.

## Security

No new serving surface. Benchmark support only forwards existing loopback MTP
serving modes.
