# Sprint 597 Follow-Ups

Items discovered during execution (distinct from the planning-time
`SPRINT-597-DEFERRED.md`).

## 1. HC-current eager cost regression (1.10 → 5.55 ms/layer-step)

- **What**: The eager attribution shows HC-current input at `5.552` ms vs
  Sprint 581's `1.096` ms — now the clear #2 decode cost after EP. Localize
  what grew (suspect: post-MTP-churn structural leftovers in the HC-current
  fill/pack path, `engine/hc_current.cu`), and whether the full-capture leg
  pays it too (the graph leg's EP window is 83% of the 10.24 ms replay; the
  prefix remainder includes HC-current).
- **Why**: Found in Phase 0 Leg B; exactly the anchor-drift risk the sprint
  flagged. Not in 597 scope (measurement only).
- **Severity**: Important (second-largest decode cost; caps the post-B2-C
  ceiling).
- **Suggested sprint**: 599, or fold into 598's perf gate analysis if B2-C
  lands fast.
- **Files**: `engine/hc_current.cu`, `engine/decode_loop.cu` (HC stage),
  attribution: `logs/from-cluster/sprint597-phase234/`.

## 2. HTTP server listen backlog (16) starves burst connects

- **What**: `appliance/http_server.cu:415` uses a listen backlog of 16; 128
  simultaneous connects produce Errno-110 timeouts for ~80 requests. This is
  what depressed the historical `26.8` tok/s wall anchor. Raise the backlog
  (e.g. 256) or document wave-submission as the bench contract; upstream the
  pod harness fixes (wave-of-32 submission, UTF-8 decode-replace, 900 s
  cold-load listen wait) into `tools/ds4-v100-tp-ep-http-bench.sh`.
- **Why**: First Phase 0 Leg A run failed; root-caused to the backlog. The
  harness fixes currently live only in the pod copy and COMMANDS.md.
- **Severity**: Important (affects every future benchmark's comparability;
  the repo harness as-committed cannot reproduce the clean runs).
- **Suggested sprint**: 598 (small, alongside B2-C's bench runs).
- **Files**: `appliance/http_server.cu`, `tools/ds4-v100-tp-ep-http-bench.sh`.

## 3. Re-verify flag-off byte-identity on the final profiler binary

- **What**: The flag-off tolerance/identity run used the binary one
  flag-on-only collector edit before final (the `ep_window` emitter landed
  after). The flag-off object path is argued unchanged, but the DoD-grade
  statement should be re-proven on the exact committed source with one
  tolerance run vs `/workspace/s597-phase01-artifacts/phase0-full-control/`.
- **Why**: Execution deviation #4 in SPRINT-597-REPORT.md.
- **Severity**: Nice-to-have (low risk; the edit is provably flag-on-only).
- **Suggested sprint**: 598 warm-up (one run, ~15 min).
- **Files**: `engine/runtime_profiler.cu`.

## 4. Split route-plan vs routed-input-pack in the profiler TSV

- **What**: The TSV reports them as one combined stage because the boundary
  lives in `engine/post_attention_ffn.cu`, outside 597's allowed edit
  surface. nsys splits them (0.099 / 0.062 ms). Add the boundary mark if the
  combined stage ever matters to a decision.
- **Why**: Execution deviation #2.
- **Severity**: Nice-to-have (both components are small).
- **Suggested sprint**: only on demand.
- **Files**: `engine/post_attention_ffn.cu`, `engine/runtime_profiler.cu`.

## 5. Per-pair SYS costs are congestion-coupled distributions

- **What**: In-situ SYS copy times vary 90 µs - 3.6 ms per pair depending on
  concurrent load (isolated microbench: only 2-3.5x NVLink). Any B2-C
  evaluation must measure end-to-end step time, not per-copy means, and the
  one-hop relay schedule should be checked for self-congestion on the relay
  links.
- **Why**: Phase 1/3 finding; caveat recorded in PHASE1-FINDING.md.
- **Severity**: Important (B2-C design input, not a defect).
- **Suggested sprint**: 598 (design constraint).
- **Files**: `logs/from-cluster/sprint597-phase01/phase1-nsys-insitu-attribution.txt`.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| HC-current 5.55 ms regression localization | Important | 599 (or 598 analysis) | engine/hc_current.cu |
| Listen backlog + harness upstreaming | Important | 598 | appliance/http_server.cu, tools/ds4-v100-tp-ep-http-bench.sh |
| Flag-off identity re-proof on final binary | Nice-to-have | 598 warm-up | engine/runtime_profiler.cu |
| route-plan/input-pack TSV split | Nice-to-have | on demand | engine/post_attention_ffn.cu |
| SYS congestion-coupling as B2-C design constraint | Important | 598 | (analysis artifact) |
