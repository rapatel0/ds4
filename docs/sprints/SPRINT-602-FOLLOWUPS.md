# Sprint 602 Follow-Ups

## 1. Join reclaim (~1.5 ms/layer) — the 603 lead

- **What**: The zero-NCCL stack's 16 all-rank rank-stream joins/layer cost
  ~1.5 ms/layer and mask the relay+batched gains (153 vs the 208
  demonstrated). Replace with per-collective producer-consumer edges (each
  consumer waits only the producers of the buffers it reads), gated by the
  s602 census methodology (token-race-zero must hold; the racy pairwise
  attempt is the cautionary tale — 30x divergence-mass, late-step bias).
- **Severity**: Critical (the largest single step-time lever; also what
  makes the correctness default cost-free).
- **Suggested sprint**: 603 lead.
- **Files**: `engine/runtime_pack.cu` (s602 sync sites), `engine/decode_loop.cu`.

## 2. Checksum-only late-step flicker (ours, instrumentable)

- **What**: Under Simple-stress, build5 shows ~1.7 checksum-only events/256
  steps (late-step bias, onsets 6/59/63); LL serving regime shows 0.17/run,
  zero token impact. Lives in the s602 site sync. Hunt with the
  full-barrier control (n must be >1 this time; the n=1 zero had P≈0.18 of
  being luck) and per-site jitter bisect.
- **Severity**: Important (checksum-only today; could become token-level if
  pacing tightens in 603).
- **Suggested sprint**: 603 (alongside #1 — same code).
- **Files**: `engine/runtime_pack.cu` s602 joins.

## 3. NO_SYS_RING documentation is wrong (never exported)

- **What**: `deploy/v100/ds4-v100-appliance.env.example` documents
  `DS4_V100_NCCL_NO_SYS_RING="0 3 2 1 5 7 6 4"` as the production fabric
  policy, but the reference config never exports NCCL_RINGS; the real auto
  ch0 ring is `0 3 2 1 5 6 7 4`. With NCCL leaving the captured graph this
  is mostly historical, but the env example + steering references should be
  corrected to avoid misleading future transport work (it misled the s602
  spec's ring premise).
- **Severity**: Nice-to-have (doc debt).
- **Files**: `deploy/v100/ds4-v100-appliance.env.example`, SPIKE_B_STEERING.

## 4. Binary defaults still select the racing config

- **What**: Options-struct defaults remain nccl/copy/nccl; only the
  launcher flips to the zero-NCCL stack. Non-launcher invocations (smokes,
  probes) silently run the token-corrupting config. After a 603 soak, flip
  the binary defaults too (carries the s598 follow-up #3 forward).
- **Severity**: Important (correctness footgun for diagnostics).
- **Suggested sprint**: 603/604 after soak.
- **Files**: `engine/runtime_options.cuh`.

## 5. NVIDIA escalation update

- **What**: The s600 escalation package now has a stronger close: exact
  accumulation-order reverse-engineering, the any-captured-NCCL ⇒
  token-events counter-proof, and a working NCCL-free replacement. Worth
  filing/updating (user action; outward-facing).
- **Severity**: Nice-to-have.

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Join reclaim ~1.5 ms/layer | Critical | 603 lead | engine/runtime_pack.cu |
| Late-step checksum flicker hunt | Important | 603 | engine/runtime_pack.cu |
| NO_SYS_RING doc correction | Nice-to-have | any | deploy/v100 env example |
| Binary defaults still racing | Important | 603/604 | engine/runtime_options.cuh |
| NVIDIA escalation update | Nice-to-have | user | s600/s602 artifacts |
