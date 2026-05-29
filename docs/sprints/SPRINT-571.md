# Sprint 571 - C1 Long-Generation Full-Capture Divergence Localization

Date: 2026-05-29

## Goal

Explain why Sprint 569's warmed `32` token serving gate matched while Sprint
570's longer promotion gate diverged for every measured response.

This is a diagnostic sprint. Do not add permanent flags and do not promote
no-suffix full capture. The output is a concrete first failing variable and the
next code target.

## Context

Sprint 570 preserved the performance signal but failed correctness:

- `128/128` generated token sequences diverged under the `64` token / `128`
  request measured gate.
- Both legs replayed persistent graphs in the measured window.
- Peer-copy/SYS transport stayed clean.
- Response metadata showed prompt-prefill/cache state differences between
  control and full capture, for example measured response `0` had
  `batch_prompt_tokens=203` on suffix-control and `32` on full capture.

Open hypotheses:

1. Generation length: no-suffix full capture may match at `32` generated tokens
   and diverge only after step `32`.
2. Prompt length/prefill: Sprint 570 used a longer prompt than Sprint 569, and
   prompt prefill/cache metadata differed across legs.
3. Warmup/cache state: two full-slot warmup batches before measurement may put
   full capture into a replay state not covered by Sprint 569.
4. Batch/coalescing state: repeated full-slot batches may assign slots or
   prompt-cache state differently across graph modes.

## Plan

Run a small diagnostic matrix on the current committed tree. Each cell compares
promoted suffix-control against opt-in no-suffix full capture with deterministic
generation and artifact parsing from `ds4_v100.generated_token_sequence`.

Matrix:

1. `s569-shape`: Sprint 569 prompt, `32` tokens, one warmup batch, one measured
   batch. This confirms the previous passing condition still passes.
2. `s570-prompt-32`: Sprint 570 prompt, `32` tokens, one warmup batch, one
   measured batch. This isolates prompt length/prefill from generation length.
3. `s570-prompt-64`: Sprint 570 prompt, `64` tokens, one warmup batch, one
   measured batch. This isolates generation length from the two-warmup/four-
   measured promotion shape.
4. If needed, `s570-prompt-64-two-warmups`: Sprint 570 prompt, `64` tokens, two
   warmup batches, one measured batch. This isolates warmup/cache state.

For each matrix cell, record:

- HTTP 200 counts.
- Generated sequence mismatch count by request index.
- First mismatch token offset.
- Prompt/cache metadata for representative requests:
  `prompt_prefill_tokens`, `batch_prompt_tokens`, `cache_pos_in`,
  `cache_pos_out`, `coalesced_batch_id`, `coalesced_slot_index`, and
  `cache_slot`.
- Graph replay counters and peer/SYS counters.

## Definition of Done

- Remote V100 build is reused only if it matches the current tree; otherwise
  rebuild.
- The diagnostic matrix artifacts are recorded.
- At least the first three matrix cells are run.
- The sprint identifies the first failing variable with evidence.
- Steering and vision are updated with the localization result.
- All repo changes from this sprint are committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results

Remote tree: `/workspace/s570-full-capture-promotion`

Artifacts: `/workspace/s571-full-capture-localize-artifacts`

Build:

- Reused the Sprint 570 remote build. No code changed after Sprint 570; only
  sprint documents changed locally.

Matrix cells completed:

| Cell | Prompt | Tokens | Warmup batches | Measured batches | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `s569-shape` | Sprint 569 prompt | `32` | `1` | `1` | `32/32` sequence mismatch; first diff offset `1` |
| `s570-prompt-32` | Sprint 570 prompt | `32` | `1` | `1` | `32/32` sequence mismatch; first diff offset `0` |
| `s570-prompt-64` | Sprint 570 prompt | `64` | `1` | `1` | `15/32` sequence mismatch; first diff offsets include `0` and `7` |

The optional fourth cell was stopped after the first three cells completed,
because the minimum matrix had already disproved a pure `64` token length
explanation.

Representative response-level evidence:

| Cell | Leg | HTTP 200 | Continuation tok/s wall | Sample selected token | Sample batch prompt tokens | Graph replay lines | Peer/SYS hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s569-shape` | suffix-control | `32/32` | `11.930177` | `32974` | `132` | `7009` | `0/0` |
| `s569-shape` | full-capture | `32/32` | `16.024793` | `32974` | `332` | `7009` | `0/0` |
| `s570-prompt-32` | suffix-control | `32/32` | `11.460702` | `32079` | `317` | `7611` | `0/0` |
| `s570-prompt-32` | full-capture | `32/32` | `29.708495` | `120180` | `32` | `5160` | `0/0` |
| `s570-prompt-64` | suffix-control | `32/32` | `15.498096` | `101202` | `260` | `10363` | `0/0` |
| `s570-prompt-64` | full-capture | `32/32` | `20.841490` | `101202` | `431` | `10363` | `0/0` |

Key observations:

- The Sprint 569 serving pass is not reproducible under the new diagnostic
  harness, even at the same prompt family, token count, position, and warmup
  count.
- In the recreated `s569-shape` cell, the first selected/generated token matches
  for the representative request (`32974`), then all measured responses diverge
  at continuation offset `1`.
- The longer Sprint 570 prompt can diverge immediately at offset `0`, so prompt
  prefill/cache state is part of the risk surface.
- The `64` token condition is not the first failing variable. It is a larger
  exposure window, not the root cause by itself.
- All completed cells show graph replay activity and zero peer-copy/SYS hits, so
  this is not a transport regression.
- The launcher commands for Sprint 569 and the `s569-shape` full-capture
  candidate are equivalent except for port and `--max-requests` (`80` in Sprint
  569, `96` in Sprint 571). That small harness difference should not be enough
  to assert production readiness.

Timing caveat:

- Avoid tensor/log timing conclusions here. The evidence above is request-level
  generated-token metadata and generated text, which are comparable response
  outputs. Log counts are used only to confirm that graph replay was active.

## Decision

Sprint 571 localizes the blocker to early continuation replay/state consistency,
not to a pure long-generation threshold.

No-suffix full capture remains opt-in diagnostic only. The next sprint should
instrument a controlled same-batch replay comparison around continuation step
0 -> 1, using request-level generated-token metadata as the semantic oracle and
capturing comparable prompt-cache/coalescing state. Do not use broad tensor
checksums unless they are taken at the same logical point.
