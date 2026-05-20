# Sprint 072: MTP Commit Throughput Decision Gate

## Status

Planned.

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

- [ ] `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- [ ] Local object compile is unaffected.
- [ ] `git diff --check` passes.
- [ ] V100 build passes for `tools/ds4-v100-replay`.
- [ ] Non-MTP sustained bench still runs with the default `--mtp-serving off`.
- [ ] V100 verify and commit sustained bench cases run with one slot and report
  MTP counters.
- [ ] TSV/JSON artifacts compare off, verify, and commit throughput.
- [ ] Sprint report records the measured decision.
- [ ] Vision document is updated.

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
