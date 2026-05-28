# Code Cleanup — snapshot, then aggressively delete the flag matrix

## Why

`tools/ds4-v100-tp-ep-full-layer-smoke.cu` has accumulated ~80 sprints of
default-off feature gates. Function bodies are now decision trees over an
exponential flag space. **Which code actually runs depends on the gate
combination, not on any single flag.** Concrete cost of this just this week:
the A6 investigation took multiple read passes to discover that the *real*
rank-major attention norm (`fill_two_hidden_inputs_half_from_rank_major_norm_kernel`,
lines 2220–2278) is fully written and correct — but its dispatch site is
dead because line 13448 hardcodes `const bool rank_major_input = false;`.
Correct working code is hiding next to broken vestigial code under the same
gate names. That has to stop.

This sprint:
1. **Snapshots** the current state so we have a recoverable baseline.
2. **Aggressively deletes deprecated paths.** No incremental conservation —
   if it's classified as gone, it's gone in this sprint.
3. **Revives** the small set of paths we already verified are implemented +
   correct + disabled (A6 PATH 4 below).
4. Permits **rewriting tangled sections from scratch** when in-place cleanup
   would leave the structure worse than the rewrite.

This is **parity-preserving** by construction: anything removed is dead
relative to the validated promoted serving binary.

## Step 0 — Snapshot and push (do this FIRST)

Before any deletion or rewrite:

1. `git add -u && git add <any new files>` — stage every locally-modified file.
2. Commit with message **"Pre-cleanup snapshot: state before TEMP_CODE_CLEANUP_PROMPT"**.
   Include the SHA in the sprint's status report.
3. **Push to `origin/claude-takeover`.** This is the recoverable baseline; the
   cleanup is destructive and we need a verified checkpoint.
4. Tag this commit `pre-cleanup-snapshot` for fast lookup.

Only after the snapshot is pushed do you start deleting.

## Categorization framework (six buckets)

| Bucket | Definition | Action |
|---|---|---|
| **Promoted** | flag default-on in the validated serving binary | **delete the flag and the else-branch** |
| **Rejected — terminal** | confirmed worse + not on the roadmap to revisit | **delete the flag and the if-branch** |
| **Dormant — revive** | implementation exists, is correct, has been disabled by a hardcoded false or stale gate; verified valuable | **enable** (un-hardcode or re-gate properly), then promote and clean up the surrounding dead siblings |
| **Diagnostic / audit** | flag enables a parity log, peer-accounting counter, or debug emission | **delete unless the audit is still open**; if open, move behind a `DS4_DIAGNOSTICS=1` umbrella, off the hot path |
| **Configurable runtime knob** | a real choice an operator might want at runtime | **keep, document inline** at the flag definition with a one-line "why this exists" |
| **Experimental — alive** | still being judged in a current/imminent sprint | **keep, but rename to include the sprint number it's owned by** (e.g., `--sprint-485-x-gate`) with a sunset comment |

Bias toward deletion. If you can't justify keeping a flag in <1 sentence, it's
out.

## Concrete targets

### Dormant → revive (do this with the A6 cleanup specifically)

1. **Re-enable A6 PATH 4: the rank-major attention norm.** At line 13448 of
   `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:
   ```cpp
   // before:
   const bool rank_major_input = false;
   // after:
   const bool rank_major_input = opt.tp_hc_current_input_nccl_allgather_gate;
   ```
   (or add a dedicated `--attn-projection-rank-major-norm-gate` for
   decoupling). The kernel `fill_two_hidden_inputs_half_from_rank_major_norm_kernel`
   (lines 2220–2278) is correct and unchanged; the runtime guard
   `&& r.d_current_full_rank_major` at line 13509 is the safety net.

   - Audit buffer lifetime: `r.d_current_full_rank_major` must remain
     populated until the kernel reads it. The transpose to slot-major at
     ~7508 must not free it.
   - Expected parity: **bit-exact 256/256.** Math is identical to control
     (same fp32 reduction). If parity fails, the dispatch is misrouted.
   - Expected perf: the GPU0-serial norm step + the broadcast disappear for
     this sublayer, replaced by parallel local norm on each rank. Small per
     layer, but ×43 layers per sublayer.

2. **Then delete the broken / no-op siblings of A6.** Once PATH 4 is the
   path, PATH 2 (broadcast sub-branch, lines 13498–13508) is a no-op clone of
   control, and PATH 3 (`else` sub-branch, lines 13517–13533) is broken and
   useless. Delete both. The dispatch becomes one branch: rank-major if
   buffer available, control otherwise.

### Promoted → delete the flag and the else-branch

3. **`ds4_peer_copy_async` else-branches everywhere.** Sprint 479 promoted
   NCCL on the hot path. Every `if (graph_event_order) {…} else if (rank ==
   0) cudaMemcpyAsync … else ds4_peer_copy_async …` block has the `else`
   branch dead. Replace each with the unconditional NCCL path.
4. **`enqueue_graph_f32_copy_from_device0` and
   `enqueue_graph_f32_copy_between_devices` wrappers.** Same — non-NCCL
   fallbacks dead post-479.
5. **`--decode-cudagraph-peer-copy-gate`.** Already retired unconditionally
   per 479's status. Remove the gate, the parser entry, all references.

### Rejected — terminal → delete the flag and the if-branch

6. **`--true-ds4-attention-projection-direct-input-fill-gate` and the PATH 3
   `else` sub-branch (lines 13517–13533)** — see item #2. The data-consistency
   bug isn't worth fixing because the path has no perf upside even at 1.0
   agreement.
7. **`--true-ds4-attention-projection-rank-local-input-gate`.** After items
   #1 and #2, the flag itself is meaningless — PATH 2 was identical to
   control, PATH 3 is gone, PATH 4 is now the default when the rank-major
   buffer is available. Remove the flag.
8. **`--routed-ffn-rank-major-input-parity-gate` and every `*-parity-gate`
   in the file.** Audit-time tools whose audit is closed. Delete.
9. **Any other rejected-and-not-coming-back path the executor identifies**
   during the categorization sweep — bias toward delete.

### Diagnostic → delete or move under `DS4_DIAGNOSTICS`

10. Every flag whose only effect is to print a comparison log, emit a tensor
    diff, or count peer ops. Decide per flag: if the audit is closed, delete.
    Otherwise move behind `DS4_DIAGNOSTICS=1` so they don't appear in the
    hot-path control flow.
11. **Peer-accounting counters** (`peer_copy_ops`, `peer_copy_sys_bytes`):
    keep, cluster under a single `diagnostics::peer_accounting` section.

### Configurable runtime knob → keep, document

12. `--nccl-reduce-scatter-compose-gate` — real operator choice (FP32 vs
    compact-route). Keep, one-line documentation.
13. Slot/context/req shapes, `--ep-return-fp16`, etc. — real config. Keep.

### Experimental — alive → tag with sprint owner

14. `--tp-hc-current-allreduce-gate` (A2), `--model-router-allreduce-logits-gate`
    (A3) — pending promotion under tolerance gate. Rename or tag with the
    docket item number from `TEMP_POST_SWEEP_DOCKET.md`.

## Methodology — including the rewrite-from-scratch option

Per flag:

1. **Identify.** `git log -S "<flag_name>"` to find the introduction sprint
   and the intended outcome.
2. **Classify** into one of the six buckets. **Bias toward elimination.**
3. **Apply the action.**
4. **Validate.** Selected-token **256/256** parity at the reference shape
   (32 slots / 256K / 256 req / 64 tok), `peer_copy_sys_bytes = 0`. Parity
   miss → back out + reclassify (the branch wasn't actually dead).
5. Commit with a message that names the flag and the bucket.

When in-place cleanup is too tangled — i.e., the if-else tree is so
interleaved that excising dead branches leaves an unreadable structure — you
have **explicit permission to rewrite the function (or the whole file)
section from scratch**:

- Copy the function's signature and external interface.
- Write the body containing only the surviving paths, in clear sequential
  form, with named helper functions for sub-steps.
- Delete the old function in the same commit.
- Validate against the strict bit-exact gate as usual.
- For very large rewrites, write to a **new file** (e.g.,
  `tools/ds4-v100-tp-ep-full-layer-smoke-v2.cu`), move the build target,
  and delete the old file once the new one is parity-clean. The git diff
  will record the rewrite cleanly even if line-for-line diff doesn't.

The file is currently ~19k lines. If a clean rewrite of the most-tangled
regions (attention-projection prefix, HC-current step, EP compose) drops
that meaningfully, that's a win in itself.

## Order

1. **Step 0 (snapshot + push).** Required first.
2. **Item #1 (A6 PATH 4 revive).** Small, well-understood change; lands a
   real win and serves as a forcing function for the surrounding cleanup
   (items #2, #6, #7).
3. **Items #3–#5 (transport dead-branch deletion).** Lowest-risk pure
   deletion; reclaims the most lines.
4. **Items #6–#7 (A6 sibling deletion).** Cleans up after #1.
5. **Items #8 + #10 (parity-gate and diagnostic deletion).**
6. **Items #11 + #12–#13 (diagnostic isolation, knob documentation).**
7. **Item #14 (tag the surviving experimental gates).**
8. **(Optional)** rewrite-from-scratch of any function/region still tangled
   after the above. Most-tangled candidates: attention-projection prefix
   (~13440–13550), HC-current step (~7240–7560), EP compose (~12860–12914).

## Gate

**Bit-exact selected-token parity 256/256** at the reference shape after every
commit, plus zero `peer_copy_sys_bytes`. Strict. No tolerance. The whole point
of cleanup is no behavior change in the promoted serving binary.

If a parity miss appears, a branch you classified as dead is actually
reachable. Back out the commit, reclassify, try again. Do not relax the gate
to accommodate a surprise.

## Out of scope

- **No new optimizations during cleanup.** Don't take the opportunity to
  "improve" anything other than items explicitly listed under
  "Dormant → revive" (A6 is in scope precisely because it's parity-bit-exact;
  it's a revive, not an optimization).
- **No restructuring of headers / build system / Python tooling** unless
  the cleanup of `.cu` mandates it.
- **No renaming or restyling for aesthetics.** Deletion and revives only.

## Going-forward discipline (commit to `docs/sprints/VISION.md` at end of sprint)

1. Every new flag introduced in a future sprint must include a **sunset
   criterion** in its introduction commit ("delete if not promoted by sprint
   Y" or "diagnostic — delete when audit Z is closed").
2. **Promotion commits remove the flag and the dead else-branch in the same
   commit.** No "we'll clean up later."
3. **Rejection commits remove the flag and the dead if-branch in the same
   commit.** Same rule.
4. Flags older than 5 sprints that aren't config knobs are technical debt by
   definition and accumulate on a rolling cleanup target list reviewed each
   sprint.

## Reporting

Per commit:
- Flag eliminated / revived / moved.
- Bucket it was in.
- Lines deleted (or net delta).
- Reference-shape parity 256/256 confirmed.
- `peer_copy_sys_bytes = 0` confirmed.

End-of-sprint summary:
- Pre-cleanup snapshot SHA + tag.
- Flag count before / after.
- File line count before / after.
- Function length for the worst offenders (attention-projection prefix,
  HC-current step, EP compose) before / after.
- Inventory of surviving flags by bucket.
- Any function/file rewritten from scratch, with the diff strategy noted.

## One-line summary

Snapshot and push first; then enable A6 PATH 4 (one line) and aggressively
delete the four dead buckets — including rewriting tangled sections from
scratch when in-place cleanup would leave structure worse than a rewrite —
all under strict bit-exact parity.
