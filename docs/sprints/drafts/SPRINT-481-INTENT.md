# Sprint 481 Intent - Code Cleanup After Pattern-A Promotion

## Seed Prompt

`docs/sprints/archive/TEMP_CODE_CLEANUP_PROMPT.md`

Snapshot and push the current state, then aggressively delete dead feature-gate
branches in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`, revive the verified A6
rank-major attention-norm path, and keep only promoted paths, true runtime
knobs, active experiments, and diagnostics that still have an open audit.

## Orientation Summary

- Current branch is `claude-takeover`; the working tree is dirty from recent
  TP/EP transport, Pattern-A promotion, and sprint/status documentation work.
- Recent work promoted no-SYS NCCL transport, direct peer-copy retirement,
  HC-current all-reduce, A3 router all-reduce under relaxed agreement policy,
  and non-compact FP32 EP compose ReduceScatter.
- Primary code target is `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
  (`23155` lines). Supporting surfaces include `tools/ds4-v100-run-appliance.sh`,
  `tools/ds4-v100-tp-ep-profile.py`, `tools/ds4-v100-tp-ep-nccl-http-ab.py`,
  and `deploy/v100/ds4-v100-appliance.env.example`.
- The cleanup must preserve the promoted serving binary behavior. Strict
  selected-token parity and `peer_copy_sys_bytes=0` are the gate after cleanup
  commits.
- `docs/sprints/VISION.md` exists and already tracks the recent Sprint 479/480
  promotion context. Sprint 481 should add the going-forward flag discipline.

## Relevant Code Areas

- A6 attention projection path:
  `true_ds4_attention_projection_*`, `rank_major_input`, and
  `fill_two_hidden_inputs_half_from_rank_major_norm_kernel`.
- Retired peer-copy graph helpers:
  `enqueue_graph_f32_copy_from_device0`,
  `enqueue_graph_f32_copy_between_devices`,
  `decode_cudagraph_peer_copy_gate`, and `ds4_peer_copy_async` fallbacks.
- Closed parity diagnostics:
  `*_parity_gate` fields, parser switches, scaffold logging, profile/harness
  summary fields.
- Launcher/env defaults for promoted or retained runtime knobs.

## Constraints

- Step 0 from the prompt is mandatory before destructive edits:
  stage all current work, commit
  `Pre-cleanup snapshot: state before TEMP_CODE_CLEANUP_PROMPT`, push to
  `origin/claude-takeover`, and tag `pre-cleanup-snapshot`.
- Use explicit `git add` commands per repo guidance.
- Do not relax the cleanup gate to tolerance; this sprint is parity-preserving.
- Do not introduce new optimizations beyond the A6 PATH 4 revive specified in
  the prompt.
- If a branch proves reachable by parity failure, back it out and reclassify.

## Success Criteria

- Snapshot commit SHA and tag are recorded in the status report.
- A6 PATH 4 rank-major attention norm is enabled through the promoted
  rank-major current buffer when available.
- Broken/no-op A6 siblings are removed or made unreachable.
- Retired peer-copy graph fallbacks and closed parity diagnostics are removed
  from the hot-path control flow.
- Surviving flags are inventoried by bucket with one-line justification.
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu` line count and flag count are
  lower than the pre-cleanup snapshot.
- Reference-shape strict parity is confirmed after cleanup commits:
  `32` slots / `256K` / `256` requests / `64` tokens, bit-exact selected-token
  parity, and `peer_copy_sys_bytes=0`.

## Verification Strategy

- Before destructive edits: snapshot commit, push, tag.
- Fast local checks after source edits:
  `python3 -m py_compile` for touched Python harnesses, `bash -n` for launcher,
  and targeted V100 build of `tools/ds4-v100-tp-ep-full-layer-smoke`.
- Promotion gate after each cleanup commit:
  strict selected-token parity `256/256` at the reference shape and
  `peer_copy_sys_bytes=0`.
- End-of-sprint report:
  snapshot SHA/tag, line/flag deltas, surviving flag inventory, validation
  commands, and artifact paths.

## Uncertainty

- Correctness: High. The prompt claims A6 PATH 4 is bit-exact, but the
  execution must prove that at the reference shape.
- Scope: High. The prompt is intentionally aggressive and touches a 23k-line
  CUDA serving file with many historical gates.
- Architecture: Medium. The desired end state is clear, but some cleanup may be
  better as local rewrites than surgical branch deletion.

## Open Questions

- Whether to split cleanup into multiple commits by bucket even if the same
  validation gate is expensive. The prompt prefers per-flag commits; execution
  should keep commits coherent and validation-backed.
- Whether any diagnostic flags are still tied to an open audit. Default bias is
  deletion unless the current sprint/status docs identify an active owner.

## Vision Context

The VISION ledger has reached Sprint 480 with A3 promoted, non-compact FP32
ReduceScatter aligned, A2 retained, and A6 rejected. Sprint 481 sits after that
promotion wave as technical-debt reduction: promotion commits should remove
their old gates and branches rather than carrying a growing flag matrix.
