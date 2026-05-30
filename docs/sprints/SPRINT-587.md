# Sprint 587 - MTP same-logical-point numerical localization

Date: 2026-05-30

## Why This Sprint Exists

Sprint 586 repaired the MTP raw-window frontier so layer 43 now sees an
advancing raw-SWA window across prompt prefill and continuation. The MTP draft
still has `0/71` same-index and next-index acceptance against the main model.
That rules out "valid_rows is pinned at 1" as the sole cause, but it does not
identify which MTP stage first becomes numerically wrong.

This sprint localizes the first broken stage at the same logical decode point.
The prior C1 work showed broad tensor comparisons can be timing artifacts; this
sprint therefore compares only deterministic request-level outputs and
stage-local diagnostics captured at the exact MTP draft point.

## Scope

1. Reuse the latest promoted/committed MTP path as the control state. Do not
   rerun broad baselines unless a code-path change invalidates them.
2. Add only temporary or tightly-scoped diagnostics needed to observe layer-43
   MTP stages:
   - prologue token/hidden input and prologue output;
   - layer-43 raw row/state update and raw-window visibility;
   - layer-43 attention output;
   - post-attention FFN input/output;
   - final layer-43 hidden before the MTP head;
   - MTP head top token/logit.
3. Use deterministic serving requests (`temperature=0`, `top_p=1`) with
   warmup separated from measured/localization windows.
4. If the first broken stage is identified and the fix is mechanical, apply the
   fix and rebuild. If it requires a larger oracle or design change, document
   the blocker and stop before adding permanent surface area.

## Non-Goals

- Do not implement K-wide speculative verification.
- Do not add permanent runtime flags for debugging.
- Do not accumulate another long-lived smoke harness.
- Do not use `research/` as repo code; it remains informational reference only.
- Do not promote B1/MTP unless request-level main-serving parity still holds
  and the draft acceptance signal becomes correct enough to justify promotion.

## Validation

- Pod build of `appliance/ds4-v100-tp-ep-appliance` after any real code change.
- Deterministic MTP-on serving request with startup isolated from the
  localization window.
- Existing acceptance harness may be reused only after a candidate fix or
  decisive diagnostic change; the previous Sprint 586 rejection is otherwise
  the current baseline.
- MTP-off/main-serving parity must remain unchanged if any production fix
  lands.

## Definition of Done

- The first suspicious MTP stage is identified with same-logical-point evidence,
  or a specific missing oracle/blocker is documented.
- Any production fix is built on the pod and validated with deterministic
  acceptance/parity checks.
- Temporary diagnostics are removed or kept strictly local to the pod if no
  production fix is committed.
- Steering/vision are updated only if the result changes the B1 plan.

## Execution Result

Built and tested two same-logical-point candidate fixes. Neither is promoted.

### Candidate 1: condition the MTP prologue on the main selected token

Second-opinion review ranked the top hypothesis as an ordering/conditioning
bug: the scaffold runs MTP before the normal serving output head, so the
prologue receives `decode_input_tokens` (`x_t`) rather than the just-selected
main token (`x_{t+1}`). A temporary build computed the main output head before
MTP, passed its selected token vector into `run_mtp_prologue`, and restored the
main hidden before the normal serving head path.

Evidence:

- Pod build passed:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

- The same-point log confirmed the candidate actually changed conditioning:
  examples included `input_token0=128822`, `main_token0=53022`, and the draft
  consumed the main-token vector.
- The deterministic acceptance harness still rejected:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12]  [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [13105, 13428, 91481, 82318, 82318, 84941, 44, 84941, 95125, 85948, 8762, 115702]
ACCEPT_EXIT=0
```

Decision: reject. The change is plausibly canonical for a future K-wide
speculative driver, but it does not improve the current one-token acceptance
signal and adds extra output-head work to the MTP path.

### Candidate 2: condition on main token and run layer 43 at `position + 1`

The next semantic hypothesis was that if the prologue is conditioned on
`x_{t+1}`, the MTP layer should run in the `t+1` position frame. A temporary
build paired Candidate 1 with `mtp_opt.position = opt.position + step + 1`.

Evidence:

- Pod build passed:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

- The deterministic acceptance harness still rejected:

```text
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12]  [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [13105, 13428, 91481, 82318, 82318, 108300, 24, 89431, 95125, 72697, 15807, 115702]
ACCEPT_EXIT=0
```

Decision: reject. The candidate did not move same-index or next-index
acceptance and made the draft distribution more repetitive in the same-point
logs.

### Diagnostic lesson

A temporary tensor-stat pass that enabled `true_ds4_attention_saturation_audit`
and disabled semantic stats skipping for layer 43 crashed during CUDA graph
capture:

```text
cuda error ./engine/diagnostics_support.cu:38: operation not permitted when stream is capturing
tp_ep_decode_cudagraph_replay_probe_start layer 43
```

Do not collect host-side tensor stats inside the captured MTP layer path. The
next localization pass should either run MTP layer 43 with graph capture
disabled in a one-off diagnostic build, or use device-side checks that are safe
under capture.

### Final state

Both candidates were removed from local and remote code. The remote tree was
restored from local files and rebuilt:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

No production code is promoted by this sprint. B1/MTP remains blocked on a
numerical localization of the layer-43 body/head. The next sprint should focus
on layer-43 raw-ring isolation and MTP-body correctness with graph capture
disabled for the diagnostic path, avoiding host tensor collection inside CUDA
capture.
