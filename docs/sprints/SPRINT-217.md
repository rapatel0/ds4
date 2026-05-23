# Sprint 217 - 32-Slot 256K Admission Probe

Date: 2026-05-23
Status: Planned

## Overview

Sprint 215 found the best current practical long-context mode at `32` slots and
`128K`, but `32` slots at `256K` was never measured because the launcher cap
failed closed at `16`. Sprint 216 then rejected current MTP commit as a real
speculative speedup, so the next practical lever is to test whether the
`256K`/`32`-slot cap is conservative.

Sprint 217 builds a focused admission probe using the existing experimental
slot-cap override. Production defaults stay fail-closed until real V100
evidence proves fit, correctness, and acceptable reserve.

## Goals

- Run a real V100 `ctx=262144`, `slots=32`, `active_microbatch=32` serving
  probe against the production TurboMind pack.
- Use production F8/TurboMind flags and the explicit
  `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=32` override.
- Record correctness, prompt/prefill tok/s, continuation/decode tok/s,
  generated tok/s, GPU utilization, and max memory.
- Promote the launcher cap from `16` to `32` for `256K` only if evidence is
  clean and memory reserve remains acceptable.

## Non-Goals

- No TP/PP scheduler work.
- No MTP work.
- No routed-FFN kernel tuning.
- No permanent cap increase without real V100 evidence.
- No broad admission changes for `512K` or `1M`.

## Implementation

1. Add a focused gate wrapper, for example
   `tools/ds4-v100-256k-32slot-gate.sh`.
2. The wrapper should:
   - use `/workspace/packs/ds4-appliance-full-tm-gated-s181` by default;
   - export the same production TurboMind/F8 flags as Sprint 215;
   - run a launcher `--check` with
     `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=32`;
   - run `tools/ds4-v100-sustained-decode-bench.sh` at `256K`/`32` slots;
   - write a summary JSON/Markdown verdict.
3. Run the wrapper on the V100 pod.
4. If the result passes with acceptable memory headroom, update
   `tools/ds4-v100-run-appliance.sh` and docs so `ctx=262144` admits `32`
   slots by default.
5. If the result fails or headroom is weak, leave the cap at `16` and document
   the exact blocker.

## Parallel Work Lanes

| Lane | Work | Write scope |
|---|---|---|
| A | Gate wrapper and launcher check path | `tools/ds4-v100-256k-32slot-gate.sh` |
| B | V100 execution and evidence capture | `logs/from-cluster/sprint217-256k-32slot-gate/` |
| C | Conditional launcher promotion | `tools/ds4-v100-run-appliance.sh` only if evidence passes |
| D | Operator/status synthesis | docs after evidence exists |

Workers are not alone in the codebase. Do not revert unrelated edits. This
sprint intentionally stays on the current PP/layer serving path.

## Files In Scope

| File | Purpose |
|---|---|
| `tools/ds4-v100-256k-32slot-gate.sh` | repeatable focused admission probe |
| `tools/ds4-v100-run-appliance.sh` | conditional cap promotion only |
| `tools/ds4-v100-sustained-decode-bench.sh` | use only; reporting fix if needed |
| `docs/operations/DS4-V100-APPLIANCE.md` | operator recommendation |
| `docs/sprints/STATUS.md` | topline status |
| `docs/sprints/VISION.md` | vision progress and next lever |
| `logs/from-cluster/sprint217-256k-32slot-gate/` | V100 evidence |

## Definition Of Done

- [ ] Sprint plan exists and is committed before execution evidence is staged.
- [ ] Focused `256K`/`32`-slot gate wrapper exists and passes `bash -n`.
- [ ] Launcher `--check` with the experimental cap override passes or records
      a clear fail-closed reason.
- [ ] V100 focused sustained decode run completes or fails with the exact
      allocator/OOM/admission error recorded.
- [ ] If the run completes, token match, generated tok/s, continuation tok/s,
      prompt tok/s, GPU utilization, and max memory are recorded.
- [ ] If worst-GPU memory reserve is acceptable, `ctx=262144` default admission
      is promoted to `32` slots and validated with launcher `--check` without
      the override.
- [ ] If reserve is unacceptable or the run fails, the cap remains `16` and the
      blocker is documented.
- [ ] Logs are copied to
      `logs/from-cluster/sprint217-256k-32slot-gate/`.
- [ ] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated.
- [ ] Changes are committed with explicit `git add` paths.

## Verification Strategy

V100 target:

```text
pod: llm/llamacpp-build-8gpu
workspace: /workspace/ds4-sprint181
pack: /workspace/packs/ds4-appliance-full-tm-gated-s181
base model: /models/DSv4-Flash-256e-fixed.gguf
```

Primary probe:

```text
ctx=262144
slots=32
active_microbatch=32
tokens=64
requests=32
warmup_requests=0
MTP=off
```

Promotion gate:

- token match must be `32/32`;
- `status_other=0`;
- worst observed GPU memory should leave at least a practical reserve, with
  `30 GiB` max used treated as the default warning line on 32GB V100s;
- continuation tok/s should be compared against Sprint 215's `16`-slot/`256K`
  and `32`-slot/`128K` baselines.

## Decision Gates

Promote `256K`/`32` only if:

- correctness passes;
- no allocator/OOM failure occurs;
- worst-GPU memory remains below the reserve threshold;
- aggregate continuation throughput is operationally useful.

Keep the `16`-slot cap if:

- the run fails;
- memory headroom is too small for production variance;
- throughput regresses enough that `32` slots at `128K` remains the better
  practical mode.

## Risks

- The cap may be hiding real KV/scratch memory pressure.
- A single short run may pass while longer runs fragment or grow memory.
- Wider `256K` concurrency may fit but not improve throughput if attention/KV
  bandwidth dominates.

## Security

No external exposure. No model weights in logs. Use the existing cluster pod,
local model mounts, and persistent appliance pack.

## Dependencies

- Sprint 215 production serving matrix.
- Sprint 216 MTP decision, which moves the next practical lever back to slot
  admission/KV/attention.
- V100 build pod `llm/llamacpp-build-8gpu`.
