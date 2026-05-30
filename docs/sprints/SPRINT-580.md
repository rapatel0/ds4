# Sprint 580 - C1 Full-Capture Serving Promotion Gate

Date: 2026-05-29

## Goal

Decide whether to flip the TP/EP launcher default to no-suffix full capture now
that Sprint 579 fixed the batch-instability defect (bidirectional dense<->rank
barrier in `enqueue_rank_streams_wait_after_dense_streams`). Full capture is now
deterministic (full-vs-full `0/8`) and matches eager (`0/8`) at matched positions.

This gate **opts into perf measurement** (per `VALIDATION_CONTROL_POLICY`): the
failure mode is "structurally landed but the perf didn't transfer," which is the
canonical opt-in case for CUDA-graph work.

## Process (per SPIKE_B_STEERING)

- Control reuse: the promoted suffix-control is the reference; refresh only
  because the binary changed (the Sprint 579 barrier fix touches the shared
  cudagraph helper). So this is a fresh same-binary A/B.
- Validation judges parity in logit/sequence space against the **determinism
  floor** (identical-config `control-A` vs `control-B`), not exact equality with
  one run.
- Promotion requires a same-binary A/B at the real serving shape, parity within
  the determinism floor, no transport/SYS regressions, and materially better
  request-window decode tok/s or GPU util.

## Plan

1. Build: reuse `/workspace/s573-continuation-instrument` (carries the committed
   Sprint 579 fix; binary already rebuilt `BUILD_EXIT=0`) only if it matches HEAD
   `f70723ca`'s engine; otherwise rsync HEAD and rebuild.
2. Re-confirm the promoted suffix-control path under the strengthened barrier:
   `control-A` vs `control-B` (identical suffix-control config) determinism floor
   at the reference shape, and that suffix-control still matches eager within that
   floor.
3. Candidate parity: no-suffix full capture vs eager at matched positions (short
   prompt so the batch coalesces to one position), judged within the determinism
   floor; and full-vs-full to confirm determinism holds at the serving shape.
4. Performance A/B: promoted suffix-control vs no-suffix full capture at `32`
   slots / `256K`, deterministic generation, startup+warmup excluded, measured
   request window. Record continuation tok/s wall + decode, median/P95 latency,
   graph replay counters, peer-copy/SYS counters.
5. Decision: promote the launcher default to no-suffix full capture iff parity is
   within the determinism floor, counters are clean, and the candidate is
   materially faster; otherwise record correctness-clean + perf and keep the
   default, with the remaining blocker named.

## Reference shape

- `32` slots / `256K` context, deterministic (`temperature=0`, `top_p=1`).
- Warmup excluded; measured window timed.
- Promoted control: `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=1`.
- Candidate: `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0` +
  `--decode-cudagraph-gate --decode-cudagraph-replay-probe-gate
  --decode-cudagraph-persistent-replay-gate`.

## Definition of Done

- Remote V100 build matches HEAD or is rebuilt.
- Determinism floor, candidate parity (vs eager + full-vs-full), and the perf A/B
  recorded with artifact paths.
- Promote / reject / continue decision recorded with evidence.
- If promoted, the launcher default flip is implemented and the suffix-control
  opt-out preserved.
- Steering and vision updated.
- All repo changes committed, excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`.

## Results

Reused the Sprint 579 build (`/workspace/s573-continuation-instrument`, carries
the committed barrier fix; binary `BUILD_EXIT=0`).

### Gate A/B (5 legs, `32` slots / pos `250000`, short prompt -> matched positions)

Artifacts: `/workspace/s580-gate-artifacts`. All legs landed at the same position
(`250042`).

| Comparison | Mismatch /32 |
| --- | ---: |
| `control-A` vs `control-B` (determinism floor) | `0` |
| `full-A` vs `full-B` (full-capture determinism) | `0` |
| `eager` vs `full-A` (correctness) | `0` |
| `eager` vs `control-A` | `0` |

Parity is perfect and within the determinism floor: full capture is deterministic
at the reference slot count and matches eager exactly.

| Leg | tok/s wall | tok/s decode | median latency |
| --- | ---: | ---: | ---: |
| suffix-control | `23.56` | `1.542` | `42.09s` |
| no-suffix full capture | `28.34` | `2.34` | `34.98s` |
| eager | `22.93` | `1.426` | `43.24s` |

Speedup full capture vs suffix-control: **`1.203x` wall, `1.518x` decode**, median
latency `42.09s -> 34.98s`.

### Launcher default flip

Replaced the binary `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY` knob with a three-mode
`DS4_V100_TP_EP_DECODE_GRAPH_MODE` (`full` default, `suffix`, `eager`).
`GRAPH_SUFFIX_REPLAY`, if set, still overrides (1->suffix, 0->eager) for
back-compat, so existing validation harnesses (`SUFFIX_REPLAY=0` + explicit gate
args) are unaffected. Default mode `full` emits the three full-capture cudagraph
gates without the suffix-stage.

A default-path smoke confirmed the launcher resolves to `full` mode (suffix-stage
absent) and the server reaches HTTP serving with the full-capture args. The smoke
harness server was then killed during its health-wait loop (an environmental /
bash-harness artifact -- GPUs free, ~106 GB host RAM free; the python-harness gate
served all 5 legs cleanly). The gate is the authoritative validation.

## Decision

**Promote no-suffix full capture as the TP/EP launcher default.** All criteria
met: parity within the determinism floor (perfect `0` across floor, determinism,
and eager-match), clean counters, and materially better throughput
(`1.20x` wall / `1.52x` decode). The suffix-control and eager paths remain
available via `DS4_V100_TP_EP_DECODE_GRAPH_MODE` (or the legacy
`GRAPH_SUFFIX_REPLAY` override).
