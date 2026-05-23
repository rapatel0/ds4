# Sprint 215 - Practical Serving Matrix And MTP Viability

Date: 2026-05-23
Status: Planned

## Overview

Stop treating isolated routed-FFN microbenchmarks as the main path to practical
serving. Sprint 214 rejected the first tile-local candidate, and Sprint 182
already showed that 256K serving has visible attention/KV and host/stage wait
costs. Sprint 215 builds and runs a practical serving matrix against the
persistent production appliance pack, then uses the evidence to choose the next
implementation lever.

This is a production-serving sprint, not another low-level kernel sprint.

## Goals

- Establish the best deployable long-context throughput mode available today.
- Test the 32-slot target at `128K` and explicitly attempt or fail-close
  `32-slot/256K`.
- Measure MTP verify and one-slot commit as actual serving modes, with
  acceptance/commit counters separated from base throughput.
- Produce one status artifact that tells an operator what to run today and what
  remains before the higher-throughput vision is realized.

## Non-Goals

- No generic scheduler abstraction.
- No TP/PP runtime integration.
- No routed-FFN reducer-only kernel work.
- No claim that MTP is a speedup unless effective-token evidence proves it.
- No production default change without same-binary correctness and throughput
  evidence.

## Implementation

1. Add a repeatable benchmark wrapper, for example
   `tools/ds4-v100-practical-serving-matrix.sh`.
2. The wrapper should use the localpool production pack by default:
   `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
3. The wrapper should run or fail-close these cases:

| Case | Context | Slots | Active microbatch | MTP | Purpose |
|---|---:|---:|---:|---|---|
| production-baseline | 256K | 16 | 16 | off | Current long-context baseline |
| long-throughput | 128K | 32 | 32 | off | User target: 32 slots with long context |
| forced-256k-32 | 256K | 32 | 32 | off | Determine whether cap or real VRAM blocks this |
| mtp-verify | 256K | 16 | 16 | verify | Confirm MTP sidecar works with production pack |
| mtp-commit | 256K | 1 | 1 | commit | Measure one-slot commit semantics and counters |

4. Record generated tok/s, continuation tok/s, prompt/prefill tok/s, token
   match, status JSON, MTP attempted/accepted/committed counters, and GPU memory
   headroom where available.
5. If `32-slot/256K` is launcher-capped, record the exact fail-closed message.
   If it can be forced safely, run it only after a planner/reserve check.
6. Update the runbook/status docs with the best current practical mode and the
   measured gap to the desired `~1k-2k aggregate tok/s` target.

## Parallel Work Lanes

| Lane | Work | Write scope |
|---|---|---|
| A | Benchmark wrapper and launcher validation | `tools/ds4-v100-practical-serving-matrix.sh` |
| B | V100 execution and log capture | `logs/from-cluster/sprint215-practical-serving-matrix/` |
| C | MTP counter interpretation | `tools/ds4-v100-replay.c` only if reporting is wrong |
| D | Operator/status synthesis | docs after evidence exists |

Workers are not alone in the codebase. Do not revert unrelated edits. Keep this
sprint separate from TP-only files and PP scheduler files.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-practical-serving-matrix.sh` | repeatable production serving matrix |
| `tools/ds4-v100-run-appliance.sh` | optional fail-closed admission/reporting fixes only |
| `tools/ds4-v100-replay.c` | optional MTP counter/reporting fixes only |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator mode recommendation |
| `docs/sprints/STATUS.md` | topline status |
| `docs/sprints/VISION.md` | sprint outcome and next lever |
| `logs/from-cluster/sprint215-practical-serving-matrix/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Practical serving matrix wrapper exists and passes `bash -n`.
- [ ] V100 build is run if any C/CUDA/replay code changes.
- [ ] 16-slot/256K base production-pack run passes token match.
- [ ] 32-slot/128K base production-pack run passes token match.
- [ ] 32-slot/256K is either run with reserve evidence or fails closed with
      the exact admission reason recorded.
- [ ] MTP verify production-pack run records attempted/accepted counters.
- [ ] MTP commit one-slot run records attempted/accepted/committed counters and
      states whether it produces real effective-token speedup.
- [ ] Prompt/prefill, generated, and continuation/decode tok/s are recorded
      separately.
- [ ] Best current practical operating mode is documented.
- [ ] Next implementation lever is selected from evidence.
- [ ] Logs are copied to
      `logs/from-cluster/sprint215-practical-serving-matrix/`.
- [ ] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Decision Gates

Promote a serving mode only if:

- token match passes;
- status counters show the intended slot and microbatch shape;
- continuation/decode tok/s improves materially over the current long-context
  baseline or the mode satisfies a distinct operational need;
- VRAM reserve remains acceptable on all eight 32GB V100s.

Select MTP as the next implementation lever only if:

- acceptance is non-trivial on the production prompts;
- the current commit path can be changed into true speculative verification
  without breaking base token correctness;
- evidence shows the expected speedup is larger than the observed MTP overhead.

Otherwise select attention/KV or a true Tensor Core fused/persistent routed-FFN
boundary as the next sprint.

## Risks

- The 32-slot/256K target may be blocked by real KV memory, not just launcher
  admission.
- MTP commit may only be accounting/observability today and may not save target
  forwards.
- Served results can vary with prompt shape and request coalescing; the wrapper
  must record enough status to interpret the run.

## Security

No external exposure. No model weights in logs. Use local cluster access and
the existing read-only model mount.

## Dependencies

- Persistent production pack from Sprint 181:
  `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- V100 build pod `llm/llamacpp-build-8gpu`.
- Sprint 214 decision to stop reducer-only routed-FFN work.
