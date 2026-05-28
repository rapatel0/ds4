# Audit — state through Sprint 482 (in-flight)

Cumulative status of the optimization program as of 2026-05-28. Captured while
Sprint 482 (A6 PATH 4 failure capture) is in flight. Supersedes the prior
audit that ran through Sprint 480.

## Done — confirmed against sprint artifacts

| Sprint | Item | Status | Evidence |
|---|---|---|---|
| 478 | **A2 mix/RMS all-reduce** | ✅ default-on | `DS4_V100_TP_EP_HC_CURRENT_ALLREDUCE=1`; survived 481 sanity run 256/256, 16384 tokens, peer_copy_ops=0 |
| 479 | **SYS transport sweep** | ✅ complete | per-site Direct SYS = 0, 256/256 HTTP, peer_copy_sys_bytes=0 |
| 480 | **A3 router all-reduce** | ✅ promoted | `DS4_V100_TP_EP_MODEL_ROUTER_ALLREDUCE_LOGITS=1`; s480 artifact agreement 1.0/1.0, rel-err 0.025 (advisory) |
| 480 | **EP-compose ReduceScatter (non-compact FP32)** | ✅ default `auto` | s480 artifact agreement 1.0/1.0, rel-err 7e-5; auto = enabled for non-compact FP32, off for compact-route serving |
| 480 | **Tolerance harness updated** | ✅ | `tools/ds4-v100-http-response-tolerance.py` gates on agreement; rel-err is JSON-advisory only |
| 481 | **Code cleanup — first wave** | ✅ done | `pre-cleanup-snapshot` tag pushed; TEMP_STATUS_REPORT_001..475 archived (191 files); superseded TEMP_<topic>.md archived; `--decode-cudagraph-peer-copy-gate` removed; rejected TP/EP cleanup gates removed |
| 481 | **Repo-clutter** | ✅ done | root is markdown-clean (0 TEMP_*.md at root); status reports archived; cleanup discipline added to VISION |

Items #1 (A2), #2 (A3), #4 (EP-compose for FP32) and the repo-clutter sweep
from `TEMP_POST_SWEEP_DOCKET.md` are all landed.

## In flight

**Sprint 482 — A6 PATH 4 failure capture.** Sprint 481 attempted the A6 PATH 4
revive (the one-line `rank_major_input = false` removal previously diagnosed)
and **it returned 0/256 HTTP 200 responses**. Backed out. The "one-line fix"
wasn't, and the buffer-lifetime concern raised in the original prompt proved
warranted (or there's some other interaction).

Sprint 482's scope is **observability, not promotion**:

- Add a diagnostic-only flag
  `--true-ds4-attention-projection-rank-major-input-gate`.
- Give the profile harness an early-failure summary capture so a server that
  crashes before readiness leaves a usable post-mortem.
- Route PATH 4 only when the new flag *and* the prereq HC-current NCCL
  allgather gate are both on, and the rank-major buffer exists.
- Leave the old broken rank-local sibling untouched for historical comparison.
- No broad A/B re-runs until the failure mode is captured.

The promoted serving binary (A2 + A3 + EP-compose-RS-auto + 479 transport
sweep) is unchanged during 482.

## A6 reclassified

A6 is **not** in the same Pattern-A "drift but quality preserved" category as
A2/A3. The 0/256 failure is a real serving break, not gate strictness. Three
plausible root causes 482 needs to differentiate (in order of suspicion):

1. **Buffer lifetime.** `r.d_current_full_rank_major` is freed or overwritten
   before the norm kernel reads it.
2. **Race / sync.** The allgather is in-flight when the norm starts.
3. **Real arithmetic mismatch.** Something in the kernel's reduction structure
   was missed in the prior code reading.

## Updated docket status (vs `TEMP_POST_SWEEP_DOCKET.md`)

| # | Item | Status now |
|---|---|---|
| 1 | A2 re-promote | ✅ DONE (since 478, confirmed surviving 481) |
| 2 | A3 router all-reduce | ✅ DONE (promoted 480) |
| 3 | A1-attn / A1-ffn norms | ⏸️ OPEN — Pattern A template; cheapest next-up |
| 4 | EP-compose ReduceScatter | ⚠️ PARTIAL — FP32 non-compact promoted; **compact-route variant still open** (needs variable-shape collective) |
| 5 | Final RMS-norm | ⏸️ OPEN — Pattern A template |
| 6 | LM-head argmax | ⏸️ OPEN |
| 7 | Indexer top-K | ⏸️ OPEN |
| 8 | A4b narrow-attn row-parallel | ⏸️ OPEN — needs FP8 dequant kernel re-tune at K=512 |
| 9 | Piecewise graph capture (C1) | ⏸️ OPEN — **now unblocked** (every hot op is NCCL-capturable post-479) |
| 10 | MTP (B1) | ⏸️ OPEN |
| 11 | TP-experts vs EP (B3) | ⏸️ OPEN |
| — | **A6 rank-major norm** | ⚠️ UNDER INVESTIGATION in 482 |
| — | Code cleanup, repo clutter | ✅ DONE in 481 |

## Concrete next moves

1. **Let 482 finish** — get the A6 PATH 4 failure-capture data. Don't gate
   that with a quality A/B; the goal is observability.
2. **In parallel** (since 482 is observability-only and shouldn't block the
   docket): **start docket #3 — A1-attn + A1-ffn rank-local norms.** Same
   A2 template; cheapest remaining Pattern-A item; expected +0.5–1 %
   combined. Promotable under the relaxed agreement gate. Doesn't touch the
   attention staging path 482 is debugging.
3. **After 482's failure capture lands:** decide A6's path forward — if it's
   a fixable buffer-lifetime/sync issue, fix and re-promote; if the
   rank-major norm has a real arithmetic issue, skip A6 and move on. Don't
   sink more than one sprint on A6 before re-evaluating against the docket.
4. **EP-compose-RS for compact-route (the missing half of #4)** — the +8.9 %
   short-run measurement is sitting in a gate the served path doesn't take.
   Either extend RS to handle variable per-pair shapes (AlltoAllv-then-reduce
   or per-route-group RS) or decide it isn't worth the implementation cost.

## Cumulative banked perf

Net of all promoted optimizations since the last audit:

- A2 mix/RMS all-reduce: +2.7 % (measured per sprint 478)
- A3 router all-reduce: +2–3 % (estimated; not measured in isolation post-promotion)
- EP-compose RS (FP32 non-compact only): up to +8.9 % on the path that uses
  it; **not applicable in the promoted compact-route serving default**, so its
  contribution to served decode tok/s is ~0
- 479 transport sweep: ~0 perf on its own (overhead-routing fix, prereq for
  the others)

Expected combined uplift in promoted serving: **~+5 %** vs the pre-478
baseline of 35.8 tok/s — i.e., roughly ~37.5 tok/s at the reference shape if
A3's standalone effect lands as estimated. A fresh reference-shape benchmark
of the current promoted binary would confirm.
