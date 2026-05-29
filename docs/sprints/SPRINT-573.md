# Sprint 573 - C1 Same-Logical-Point Continuation Instrumentation and Repair

Date: 2026-05-29

## Goal

Identify the exact request/replay state that diverges at no-suffix full-capture
continuation step `0 -> 1`, then attempt the narrowest repair if the
instrumentation cleanly fingers a single cause.

Sprint 570 rejected the default flip (`128/128` diverged). Sprint 571 localized
the failure to early continuation, not a long-generation threshold. Sprint 572
rejected the cache-miss capture-as-served-result repair: divergence persisted and
only shifted the `s569-shape` first diff from offset `1` to offset `2`. Two blind
repairs have now failed, so this sprint instruments first and repairs only on
evidence.

## Code analysis carried into this sprint

- **RoPE / pure-kernel position consumers are not the cause.** They read the
  decode position by dereferencing the device buffer
  (`decode_position_u32_dev(const uint64_t *decode_position)` in
  `kernels/v100/attention.cuh`), and `update_device_decode_position()`
  (`engine/decode_loop.cu:157`) writes the live position before every replay,
  including the cache-hit launch (`attempt_capture_probe` entry at
  `decode_loop.cu:1421`). Cross-position replay therefore gets the correct
  RoPE. The Sprint 544-552 "make position dynamic" arc landed.
- **The cache-hit replay path rebases only the HC shard.**
  `prepare_full_capture_replay_hc_buffers` (`decode_loop.cu:1572-1608`) copies
  `d_final_hc_shard` into the captured graph input buffer and nothing else.
  Compressed/indexer host row bookkeeping is mirrored separately by
  `apply_full_capture_replay_compressed_kv_host_state`
  (`decode_loop.cu:480-530`), but only at emit boundaries
  (`(opt.position + 1) % ratio == 0`). At a non-emit `0 -> 1` step the only
  carried state is HC (rebased) plus the device-dynamic raw window, so a simple
  "emit row not mirrored" explanation does not cover the every-response offset-1
  divergence.
- **Two distinct failure shapes, matching Sprint 571's two offsets:**
  - `s569-shape` diverges at offset `1`: token `0` is the eager-served result
    and matches; token `1` is the first cache-hit replay and differs. This is a
    cross-position replay-state question.
  - `s570-prompt` diverges at offset `0`: even the first continuation token
    differs, and the full-capture leg reported `batch_prompt_tokens=32` versus
    control `317` for the same request. That `32` is a prompt-cache-hit
    signature (only the tail is prefilled). This points at prompt
    prefill/coalescing state, possibly a comparison confound across the
    warmup and measured batches rather than a replay-correctness bug.

## New hypothesis to test first (cheap, no rebuild)

The offset-0 `batch_prompt_tokens=32` signature suggests the measured leg may be
reusing a cached prompt prefix populated during warmup, and the two legs may
populate or reuse that cache differently. If so, part of the "divergence" is a
warmup-vs-measured prompt-cache coalescing confound, not full-capture replay
incorrectness.

Discriminator: rerun the `s569-shape` and `s570-prompt-32` A/B with prompt-cache
reuse defeated (unique per-request content so no two requests share a cacheable
prefix, and no shared prefix between warmup and measured batches). If the legs
then match, the divergence is a cache-coalescing comparison artifact and the C1
conclusion changes. If they still diverge, the bug is in cross-position replay
state and the instrumentation below localizes it.

## Plan

1. Reuse the promoted tree (HEAD, no code change since Sprint 570) shipped to the
   V100 node via `git archive HEAD`. Rebuild the appliance in-pod.
2. Cache-discriminator A/B (no instrumentation, promoted binary): `s569-shape`
   and `s570-prompt-32`, cache-busted unique content, suffix-control versus
   opt-in no-suffix full capture. Compare `generated_token_sequence` by index.
3. If divergence persists, add same-logical-point instrumentation emitted on
   both legs at identical logical points:
   - prompt admission: `prompt_prefill_tokens`, `batch_prompt_tokens`,
     `cache_pos_in/out`, `coalesced_batch_id`, `coalesced_slot_index`,
     `cache_slot` (already in response metadata; capture per request);
   - continuation step `0` and step `1`: per-slot decode input token id,
     selected token id, device decode position, and the capture-vs-replay host
     emit/row counters (`attn_comp_rows_written_layers`, row position arrays)
     read at the same logical point in eager and replay.
4. Run the instrumented A/B and identify the first state that differs.
5. If a single cause is cleanly identified, implement the narrowest repair and
   re-validate. Otherwise reject with evidence and record the next target.

## Validation

Correctness only (per `VALIDATION_CONTROL_POLICY.md`; no perf opt-in). Oracle is
the request-level deterministic `generated_token_sequence`. Server-log counts are
used only to confirm graph replay was active and taken at the same logical point.

## Definition of Done

- Remote V100 build matches the current tree or is rebuilt.
- Cache-discriminator A/B artifacts recorded.
- If instrumentation is added, instrumented A/B artifacts recorded and the first
  diverging state named with evidence.
- A repair is implemented and validated, or rejected with direct evidence.
- Steering and vision updated with the result.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results

Remote tree: `/workspace/s573-continuation-instrument` (HEAD `7d5a9342`, shipped
via `git archive`). Build passed (`make appliance/ds4-v100-tp-ep-appliance`,
`BUILD_EXIT=0`).

### Cache-discriminator A/B (inconclusive, confounded)

Artifacts: `/workspace/s573-continuation-instrument-artifacts`.

The unique-prefix cache-bust was ineffective: with the forced decode position
`250000`, the synthetic 250K context dominates output and a ~15-token unique
prefix is negligible. Control collapsed `32` unique-prompt requests to only `10`
distinct sequences (`2` distinct first tokens), so the prompt-cache-confound
question could not be answered from this cell. Do not draw conclusions from it.

### Determinism baseline A/B (decisive)

Artifacts: `/workspace/s573-determinism-artifacts`. One cell, Sprint 569 shared
prompt, `32` slots, `32` tokens, `position 250000`, one warmup + one measured
batch. Four legs from the same binary: `eager` (no graph), `control-A` and
`control-B` (promoted suffix replay, identical config, separate processes),
`full` (opt-in no-suffix full capture).

| Comparison | Mismatch /32 | First-diff offsets |
| --- | ---: | --- |
| `control-A` vs `control-B` (determinism floor) | `3` | all `0` |
| `eager` vs `control-A` | `3` | all `0` |
| `eager` vs `full` | `12` | mix of `0` and `28` |
| `control-A` vs `full` | `12` | mix of `0` and `28` |

Distinct sequences for `32` identical-prompt requests (deterministic engine would
give `1`): `eager 6`, `control-A 6`, `control-B 5`, `full 8`.

Noise-vs-signal separation (per-request offsets):

- Determinism floor `control-A`!=`control-B` at requests `{1,7,14}`, all offset
  `0`. `eager`!=`control-A` at `{3,14,16}`, all offset `0`.
  **Noise-baseline maximum offset is `0`.**
- `eager`!=`full` at `{1,3,8,9,10,11,15,16,17,18,19,30}`. The offset-`0` members
  `{1,3,16}` are exactly noise requests. The members not explained by noise are
  `{8,9,10,11,17,18,19}` — **all at offset `28`** — plus `{15,30}` at offset `0`.

### Findings

1. **The serving decode path is nondeterministic at the first token.** Pure
   eager produces `6` distinct continuations for `32` identical prompts, and two
   identical promoted-control runs differ on `3/32` requests. All nondeterminism
   is at offset `0` (first decode step), consistent with batch/slot-dependent
   reduction order (MoE routing / NCCL all-reduce / slot-position compute). It
   never propagates a *new* first-diff beyond offset `0`.
2. **Promoted suffix-control matches eager within the noise floor** (`3/32`,
   identical to `control-A` vs `control-B`). The promoted path is sound; using it
   as the reference is fine *if* the noise floor is accounted for.
3. **Full capture has a real divergence above the noise floor**, isolated for the
   first time to a clean cluster of `7` requests `{8,9,10,11,17,18,19}` at
   **offset `28`**, a position that appears in no noise comparison. Its offset-`0`
   mismatches are just nondeterminism.
4. **This reframes Sprints 570-572.** The exact-sequence-equality oracle against a
   nondeterministic reference can never reach `0` mismatches, so "`128/128`
   diverged" conflated ~9% per-request first-token noise (compounded over
   sequence length) with the real bug. The real bug is **late-position**, not the
   "step `0 -> 1`" early continuation that 571/572 instrumented and 572 tried to
   repair — which is why the 572 repair only shuffled offset-`0`/`1`/`2` noise
   without fixing parity.
5. **Mechanistic target.** Even (ratio-4) layers emit a compressed-KV row at
   `(position+1) % 4 == 0`; with `position 250000 ≡ 0 (mod 4)` the emit boundaries
   fall at generation offsets `3,7,...,27`, and offset `28` is the first token
   generated after the offset-`27` emit. The offset-`28` cluster is consistent
   with compressed-KV emit/eviction replay state under cross-position full-capture
   reuse — the surface `apply_full_capture_replay_compressed_kv_host_state`
   (`engine/decode_loop.cu:480-530`) mirrors only at emit boundaries and only the
   host row bookkeeping. The exact diverging state (ring-row index baked in the
   captured graph vs live, or device emit/load not re-driven on cache-hit replay)
   is the next instrumentation target.

## Decision

No code change to the promoted tree this sprint. The repair was *not* attempted
because the instrumentation relocated the bug: the prior "early continuation"
target was largely nondeterminism noise, and the real signal is a late-position
(offset `28`) compressed-KV emit-replay divergence. Attempting the early-
continuation repair (the user-approved scope) would have repeated the Sprint 572
mistake of perturbing noise.

Two durable methodology corrections come out of this sprint:

- **Validation must carry a determinism floor.** Future full-capture gates run an
  identical-config `control-A` vs `control-B` pair and judge full capture against
  that floor (and against `eager`), not against exact equality with one control
  run.
- **The C1 full-capture bug is late-position compressed-KV emit-replay state**,
  not early-continuation handoff.

Next sprint (574): targeted same-logical-point instrumentation at the ratio-4
compressed-KV emit boundary around generation offset `27 -> 28` (comp ring-row
index, load/store decision, row-position metadata at capture vs cache-hit
replay), then the narrow repair, validated against the determinism floor.
