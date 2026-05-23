# Sprint 217 - 32-Slot 256K Admission Probe

Date: 2026-05-23
Status: Complete

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

- [x] Sprint plan exists and is committed before execution evidence is staged.
- [x] Focused `256K`/`32`-slot gate wrapper exists and passes `bash -n`.
- [x] Launcher `--check` with the experimental cap override passes or records
      a clear fail-closed reason.
- [x] V100 focused sustained decode run completes or fails with the exact
      allocator/OOM/admission error recorded.
- [x] If the run completes, token match, generated tok/s, continuation tok/s,
      prompt tok/s, GPU utilization, and max memory are recorded.
- [x] If worst-GPU memory reserve is acceptable, `ctx=262144` default admission
      is promoted to `32` slots and validated with launcher `--check` without
      the override.
- [x] If reserve is unacceptable or the run fails, the cap remains `16` and the
      blocker is documented.
- [x] Logs are copied to
      `logs/from-cluster/sprint217-256k-32slot-gate/`.
- [x] `docs/sprints/STATUS.md`, `docs/sprints/VISION.md`, and the appliance
      runbook are updated.
- [x] Changes are committed with explicit `git add` paths.

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

## Execution

Added `tools/ds4-v100-256k-32slot-gate.sh`, a focused admission probe that:

- runs launcher `--check` with `DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=<slots>`;
- exports the production TurboMind/F8 flags;
- runs the sustained decode harness at `ctx=262144`;
- writes `slot_gate_summary.json` and `slot_gate_summary.md`;
- exits non-zero when the benchmark fails.

Local validation:

```text
bash -n tools/ds4-v100-256k-32slot-gate.sh
```

## V100 Evidence

The primary `32`-slot probe used:

```text
cd /workspace/ds4-sprint181
./tools/ds4-v100-256k-32slot-gate.sh \
  --log-dir /workspace/logs/sprint217-256k-32slot-gate \
  --port-base 19100 \
  --requests 32 \
  --warmup-requests 0 \
  --tokens 64 \
  --ctx 262144 \
  --slots 32
```

Launcher check with the experimental cap override passed:

```text
DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP=32
DS4_V100_CTX=262144
DS4_V100_SLOTS=32
DS4_V100_ACTIVE_MICROBATCH=32
```

The actual benchmark failed before any successful generation request:

| Active slots | Status 200 | Status other | Token match | Max GPU util | Max memory used | Result |
|---:|---:|---:|---:|---:|---:|---|
| 18 | 0 | 18 | 0/18 | `68%` | `24076 MiB` | fail |
| 20 | 0 | 20 | 0/20 | `76%` | `24084 MiB` | fail |
| 24 | 0 | 24 | 0/24 | `81%` | `24096 MiB` | fail |
| 32 | 0 | 32 | 0/32 | `90%` | `24124 MiB` | fail |

The 32-slot failure is not a VRAM fit failure. Worst observed memory stayed
near Sprint 215's successful `32`-slot/`128K` run. A manual single-request
server at `slots=32`, `active_microbatch=32`, `ctx=262144` returned HTTP 200,
which proves the resident pack can open and a single request can generate.

The failing shape is the active batch above 16 at `256K`. A 32-concurrent
manual reproduction with the production flags returned HTTP 500 for all
requests:

```text
{"error":"output-head fast selected-token sequence failed"}
```

Disabling the output-head fastpath changed the error to:

```text
{"error":"output-head logits contained non-finite values"}
```

That makes the blocker upstream hidden-state/logit non-finites at the
`256K`, active-batch-greater-than-16 shape, not the output-head fastpath alone.

Evidence:

```text
logs/from-cluster/sprint217-256k-32slot-gate/
logs/from-cluster/sprint217-256k-24slot-gate/
logs/from-cluster/sprint217-256k-20slot-gate/
logs/from-cluster/sprint217-256k-18slot-gate/
```

## Decision

Do not promote `ctx=262144` admission above `16` slots. The cap is protecting a
real correctness boundary, not only conservative VRAM accounting.

The next practical-serving sprint should isolate the non-finite source for
`256K` active batches above 16. The likely target is the long-context
attention/KV path or a batch-width assumption in a fused projection/FFN stage,
because single-request `slots=32` works and memory headroom remains adequate.
