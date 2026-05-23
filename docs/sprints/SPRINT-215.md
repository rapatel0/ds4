# Sprint 215 - Practical Serving Matrix And MTP Viability

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists and is committed before execution evidence is staged.
- [x] Practical serving matrix wrapper exists and passes `bash -n`.
- [x] V100 build is run if any C/CUDA/replay code changes.
- [x] 16-slot/256K base production-pack run passes token match.
- [x] 32-slot/128K base production-pack run passes token match.
- [x] 32-slot/256K is either run with reserve evidence or fails closed with
      the exact admission reason recorded.
- [x] MTP verify production-pack run records attempted/accepted counters.
- [x] MTP commit one-slot run records attempted/accepted/committed counters and
      states whether it produces real effective-token speedup.
- [x] Prompt/prefill, generated, and continuation/decode tok/s are recorded
      separately.
- [x] Best current practical operating mode is documented.
- [x] Next implementation lever is selected from evidence.
- [x] Logs are copied to
      `logs/from-cluster/sprint215-practical-serving-matrix/`.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated.
- [x] Changes are committed with explicit `git add` paths.

## Execution

Added `tools/ds4-v100-practical-serving-matrix.sh`, a production-serving
wrapper around the sustained decode harness. The script runs the restored
localpool appliance pack by default and exports the same TurboMind/F8 flags
needed for the interleaved gated pack, including `fused6_reduce` and CUDA graph
selection.

Also widened `tools/ds4-v100-sustained-decode-bench.sh` so benchmark slot tiers
can cover the existing served admission range up to 256 slots instead of
failing at 16. The launcher still owns production admission caps.

## V100 Evidence

Cluster target: `llm/llamacpp-build-8gpu` on `gpu-01`.

Command:

```text
cd /workspace/ds4-sprint181
./tools/ds4-v100-practical-serving-matrix.sh \
  --log-dir /workspace/logs/sprint215-practical-serving-matrix \
  --port-base 18900
```

Topline:

| Case | Ctx | Slots | MTP | Generated tok/s | Continuation tok/s | Prompt tok/s | Match | Notes |
|---|---:|---:|---|---:|---:|---:|---:|---|
| production-baseline | 256K | 16 | off | `62.602937` | `61.624766` | `17.607076` | 16/16 | current long-context baseline |
| long-throughput | 128K | 32 | off | `69.488893` | `68.403129` | `19.543751` | 32/32 | best current practical long-context mode |
| forced-256k-32 admission | 256K | 32 | off | n/a | n/a | n/a | n/a | fail-closed: `DS4_V100_SLOTS=32 exceeds ctx=262144 admission cap 16` |
| mtp-verify | 256K | 16 | verify | `16.373227` | `12.279921` | `73.679524` | 16/16 | `attempted=16`, `accepted=0`, `rejected=16` |
| mtp-commit | 256K | 1 | commit | `8.369430` | `7.846341` | `9.415609` | 1/1 | `attempted=15`, `accepted=8`, `committed=8` |

GPU utilization:

| Case | Avg GPU util | Max GPU util | Max memory used |
|---|---:|---:|---:|
| 16-slot/256K base | `36.30%` | `65%` | `24070 MiB` |
| 32-slot/128K base | `45.88%` | `88%` | `24124 MiB` |
| 16-slot/256K MTP verify | `30.92%` | `65%` | `24070 MiB` |
| 1-slot/256K MTP commit | `8.74%` | `16%` | `24020 MiB` |

Evidence:

```text
logs/from-cluster/sprint215-practical-serving-matrix/
```

## Decision

The best deployable practical serving mode today is:

```text
ctx=131072
slots=32
active_microbatch=32
async_pipeline_mode=per-step
async_event_handoff=on
TurboMind fused6_reduce + graph
MTP off
```

This mode keeps a long context tier (`128K`) and improves continuation decode
throughput over the 16-slot/256K mode by about `11.0%`
(`68.403129` vs `61.624766` tok/s). It is still far below the desired
`~1k-2k` aggregate tok/s serving target, so the high-throughput vision is not
realized yet.

Do not promote MTP as a speedup yet. MTP verify is operationally valid but slow
because it runs after base generation. MTP commit accepts drafts (`8/15`) and
reports committed tokens, but the current one-slot path still does serial base
work and achieves only `7.846341` continuation tok/s. The next MTP sprint must
be a true speculative verifier that batches target verification over drafted
tokens; otherwise MTP remains observability, not throughput.

The next implementation lever should be MTP true speculative verification or a
256K attention/KV execution-boundary change. Another routed-FFN reducer wrapper
is explicitly not indicated by the evidence.

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
