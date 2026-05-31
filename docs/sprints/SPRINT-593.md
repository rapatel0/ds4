# Sprint 593 - MTP HC/control semantic localization

Date: 2026-05-30

## Why This Sprint Exists

Sprint 590 cleared the MTP output-head HC slicing. Sprint 591 rejected the
layer-43-only inverse attention-head RoPE candidate. Sprint 592 cleared the
MTP layer-43 F8 dense pack/orientation path and fixed the MTP `pack_dir`
diagnostic inheritance bug. Acceptance remains `0/71`, so the next useful
boundary is the non-dense body semantics: HC controls, HC reduction mode, and
raw-SWA/body state.

The goal is not to add another permanent flag. The goal is to run small
one-off MTP-only probes, promote only a real correctness fix, and otherwise
record exactly which semantic branch is cleared.

## Scope

1. Add temporary MTP-only probes in the serving draft path:
   - centralized HC-current path (`tp_hc_current_allreduce_gate = false`) to
     test whether rank-sharded HC all-reduce is the acceptance blocker.
   - reference HC reduction (`reference_hc_reduce_gate = true`) to test whether
     the promoted clamped HC reducer is wrong for layer 43.
   - HC-current full parity (`tp_hc_current_full_parity_gate = true`) to emit
     the existing all-reduce-vs-full mix diff for layer 43.
2. Build each probe on the pod using the clean Sprint 592 code as the base.
3. Run the deterministic MTP acceptance harness (`temperature=0`, `top_p=1`).
4. If a probe produces nonzero acceptance, isolate the minimal semantic fix and
   leave only that durable change.
5. If all probes remain `0/71`, remove the temporary probes and move the next
   sprint to raw-SWA row activation capture and post-attention/FFN boundary
   checks.

## Non-Goals

- Do not build the K-wide verifier until acceptance is nonzero.
- Do not run throughput A/B; these are correctness probes only.
- Do not add permanent CLI/env flags for the probes.
- Do not re-test MTP dense F8 pack/orientation unless the pack inputs change.
- Do not change the promoted main 0-42 path.

## Definition of Done

- Pod builds pass for the temporary MTP-only probes.
- Each probe has deterministic acceptance evidence.
- Any probe that changes the draft tokens is recorded, even if acceptance stays
  zero.
- Temporary probe code is removed before commit unless it becomes the minimal
  durable correctness fix.
- `VISION.md` and `SPIKE_B_STEERING.md` record the result so the next sprint
  does not duplicate cleared work.

## Result

Status: COMPLETE. The HC/control probes did not clear MTP acceptance, but they
removed the largest remaining HC-control ambiguity from the B1 search space.

Probe A: MTP-only centralized HC-current path
(`mtp_opt.tp_hc_current_allreduce_gate = false`).

- Pod build: `BUILD_EXIT=0`.
- Deterministic harness: `pairs 71 same_index_match 0 (0.00) next_index_match 0`.
- Main token stream stayed identical to the accepted control.
- First draft tokens were unchanged from the baseline:
  `[112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]`.

Conclusion: rank-sharded HC-current all-reduce is not the obvious zero-acceptance
cause.

Probe B: MTP-only reference HC reduction
(`mtp_opt.reference_hc_reduce_gate = true`).

- Pod build: `BUILD_EXIT=0`.
- Deterministic harness: `pairs 71 same_index_match 0 (0.00) next_index_match 0`.
- The probe was active and changed draft tokens:
  `[112865, 15387, 8373, 13151, 80355, 84941, 5626, 124211, 91481, 21859, 32634, 67132]`.

Conclusion: the promoted clamped HC reducer affects the MTP draft, but it is not
the root zero-acceptance blocker by itself.

Probe C: MTP-only HC-current full parity
(`mtp_opt.tp_hc_current_full_parity_gate = true`).

- Pod build: `BUILD_EXIT=0`.
- Deterministic harness: `pairs 71 same_index_match 0 (0.00) next_index_match 0`.
- Built-in layer-43 mix parity passed for all sampled rows. Tail evidence:
  `max_abs_diff` ranged about `1.4e-6` to `3.3e-6`, `diff_bad = 0`, `PASS`.
- First draft tokens matched baseline.

Conclusion: layer-43 HC all-reduce mix agrees with the centralized full path at
float noise. Do not re-run HC all-reduce/full-parity probes unless the HC
control loading or rank slicing changes.

All temporary probe code was removed. The clean pod build passed:

```text
BUILD_EXIT=0
make appliance/ds4-v100-tp-ep-appliance
```

Next sprint should move below HC controls into actual activation/state capture:
raw-SWA row contents, attention output after raw-SWA, post-attention handoff,
and FFN output. The current evidence says the bug is not dense F8
pack/orientation, not output-head HC slicing, not layer-43-only inverse-head
RoPE, and not HC-current all-reduce/rank slicing.
