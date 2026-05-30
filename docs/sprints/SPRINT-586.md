# Sprint 586 — MTP per-slot raw-window state repair

Date: 2026-05-30

## Why This Sprint Exists

Sprint 585 built the EP MTP draft path, but draft acceptance stayed near zero.
The second-opinion review found that the current appliance MTP scaffold feeds
layer 43 a raw-SWA window derived from the local `run_token_major_serving_loop`
step. HTTP serving calls that loop one decode step at a time, so the MTP layer
usually sees `valid_rows=1` even after prompt prefill and continuation.

Upstream `ds4.c` keeps an MTP-owned raw-cache frontier (`mtp_n_raw`) and passes
`mtp_n_raw + 1` into the MTP block before incrementing the frontier. The V100
appliance needs the same state, but per HTTP cache slot.

## Scope

1. Add per-cache-slot MTP raw row state to the HTTP session table.
2. Reset the state on cache miss, eviction, or prompt mismatch.
3. Thread a per-slot raw row vector into `run_token_major_serving_loop`.
4. For layer 43, derive `true_ds4_attention_raw_valid_rows` from the active
   slots' MTP raw counts rather than local loop step.
5. Increment active-slot MTP raw counts after each successful MTP layer-43 run,
   including prompt prefill where MTP executes without an output head.

## Non-Goals

- Do not implement the K-wide verifier in this sprint.
- Do not delete the LP sidecar.
- Do not add another runtime flag or permanent smoke.
- Do not chase a full tensor oracle until this state repair is tested.

## Validation

- Build the appliance locally/pod-side.
- Copy the modified files into the pod tree and rebuild.
- Run the existing Sprint 585 acceptance harness to determine whether the MTP
  draft acceptance moves from the prior near-zero result.
- Main serving must remain byte-identical between MTP off and MTP on for the
  served main-token stream.

## Definition of Done

- MTP raw-window state is represented per cache slot and survives across
  request continuations.
- Cache lifecycle resets that state correctly.
- The MTP layer uses the per-slot state for raw-SWA visibility.
- Validation result is recorded, including promotion/rejection decision for the
  state repair.
- Steering and vision documents are updated if the result changes the B1 plan.

## Execution Result

Implemented the per-slot MTP raw row counter in the HTTP session table and
threaded it into `run_token_major_serving_loop`. Layer 43 now uses
`min(active_slot_mtp_raw_rows) + 1` for its raw-SWA window and increments active
slots after each finite MTP layer-43 execution. Cache miss, prompt mismatch, and
slot eviction reset the counter.

Pod build:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

Validation:

- Existing Sprint 585 acceptance harness:
  `/workspace/s585_accept2.sh` -> `ACCEPT_EXIT=0`.
- MTP raw window now grows mechanically: logs show
  `mtp_raw_valid_rows` progressing from `1` through prompt/continuation rows
  instead of staying at `1` for every HTTP loop call.
- Acceptance is still rejected: `pairs 71`, `same_index_match 0`, and
  `next_index_match 0`.
- MTP-off baseline harness still runs: `/workspace/s585_base.sh` ->
  `BASE_EXIT=0`.

Decision: keep the per-slot raw-window repair as canonical state plumbing, but
do **not** promote B1/MTP. This repair rules out "MTP only sees one raw row" as
the sole cause. The next sprint should numerically localize the MTP body/head
with same-logical-point instrumentation: first raw ring row contents and
prologue output, then attention output, FFN output, and MTP head logits.
