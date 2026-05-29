# Sprint 574 - C1 Full-Capture Divergence Position-Dependence

Date: 2026-05-29

## Goal

Localize the late-position (offset `28`) full-capture divergence Sprint 573
isolated, then repair it, validated against the determinism floor.

## Result summary

The repair was not reached. Two diagnostic findings changed the picture and the
instrumentation path:

1. The divergence is **strongly position-dependent**, and the position every
   gate since Sprint 569 used (`250000`) is a comparatively benign one.
2. The compressed-KV emit hypothesis is **wrong** for this served path (emit is
   off), and the per-layer stage-checksum tool is **misaligned with the serving
   loop**, so the exact diverging layer is not yet localized.

## Work performed (no promoted-tree code change)

Reused the Sprint 573 build (`/workspace/s573-continuation-instrument`, HEAD
`7d5a9342`). All experiments correctness-only, request-level generated token
sequences as the oracle, judged against the Sprint 573 determinism floor.

### Position-shift experiment (decisive)

`eager` vs `full` (no-suffix full capture) at two start positions, Sprint 569
shared prompt, `32` slots, `32` tokens, one warmup + one measured batch.

| Position | eager vs full mismatch | First-diff offsets | full distinct seqs |
| ---: | ---: | --- | ---: |
| `250000` | `12/32` (7 real at offset `28`, rest noise) | offset `28` cluster | `8` |
| `250064` | `32/32` | almost all offset `1` | `32` |

At `250064` full capture diverges immediately at the first cache-hit replay
(offset `1`) for every request and produces `32` distinct sequences for `32`
identical prompts. So the offset-`28`/7-slot behavior at `250000` is not the
worst case; it is an unusually clean position. The bug is governed by absolute
decode position, which means the captured graph carries a position-derived value
that is only correct near the capture position, and whose error magnitude scales
with how far replay drifts from it.

### Ruled out

- **Compressed-KV emit.** No `tp_ep_compressed_kv_projection` activity in the
  served logs; `true_ds4_compressed_kv_gate` is off on this path. The offset-`28`
  alignment with the ratio-4 emit boundary was a coincidence, not the cause.
- **RoPE and raw-window row selection.** Both read the decode position by
  dereferencing the live device buffer (`decode_position_u32_dev`,
  `decode_raw_row_dev` in `kernels/v100/attention.cuh`); the raw-window
  `valid_rows` is a constant. These are device-dynamic and correct across
  cross-position replay.

### Instrumentation dead-end (documented)

`--decode-stage-checksum-gate` is already CLI-wired and needs no rebuild, but in
the **serving** path the engine decodes one token per call (`decode_steps == 1`),
so every generated token logs as `step 0`. The checksum `step` field is the
intra-call index, not the generation offset, so per-offset layer localization is
impossible from serving logs, and a cross-process step-`0` diff is dominated by
unaligned last-token state. The eager leg produced only `step 0` checksum keys
(`10320`, = layers x stages x tensors x ranks for one step); steps `1`/`2` were
empty, confirming this.

## Decision

No promoted-tree code change. Do not attempt a repair until the diverging layer
is localized with a step-meaningful tool — repairing blind is the Sprint 572
mistake the determinism work corrected.

Next (Sprint 575): localize with a **multi-step capture/replay probe** (not the
HTTP serving path), where `decode_steps > 1` gives meaningful per-offset step
indices, run at the catastrophic position `250064` with
`--decode-stage-checksum-gate`, and diff eager-step-`k` vs full-replay-step-`k`
per `(layer, stage, tensor)` to find the first diverging computation. The strong
position-dependence points the search at the captured graph's position-derived
state that survives the HC rebase (everything other than HC and the
device-dynamic RoPE/raw-window). Then the narrow repair, validated against the
determinism floor at `250064` and `250000`.

## Definition of Done

- Position-dependence finding recorded with artifacts
  (`/workspace/s573-shift-artifacts`, `/workspace/s573-stageck-artifacts`).
- Compressed-KV and RoPE/raw-window hypotheses recorded as ruled out.
- The serving stage-checksum misalignment recorded so it is not retried.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.
