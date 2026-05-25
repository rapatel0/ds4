---
sprint: 342
title: TP/EP Typed KV Diagnostic Overhead Isolation
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 342 - TP/EP Typed KV Diagnostic Overhead Isolation

## Goal

Measure and reduce the remaining TP/EP typed-KV serving overhead without
changing typed KV semantics.

## Scope

TP/EP only. No PP/layer-split work. No MTP. This sprint keeps production typed
KV stores, typed history visibility, resident session state, 32 slots, and
256K context. Diagnostic-only store suppression variants remain available, but
the primary candidate is a production-compatible quiet mode that suppresses
per-layer typed KV PASS logging in the hot serving loop.

## Definition of Done

- [x] Add a typed-KV quiet gate to the TP/EP binary.
- [x] Add appliance env and launcher support for the quiet gate.
- [x] Extend the HTTP A/B harness so it can compare typed baseline vs typed
      quiet mode at the same 32-request / 32-slot / 256K / 8-token shape.
- [x] Build on the V100 pod.
- [x] Run the same-shape HTTP A/B on the V100 pod.
- [x] Record server tok/s, decode tok/s, typed evidence counters, final GPU
      state, and the decision for the next sprint.

## Outcome

Added the production-compatible quiet gate:

- binary flag: `--true-ds4-attention-typed-kv-quiet-gate`
- appliance env: `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET=1`

Quiet mode suppresses the four per-layer typed KV PASS log families while
preserving typed KV raw-SWA, compressed-attention, ratio-4 indexer, and history
semantics:

- `tp_ep_true_attention_typed_kv_raw`
- `tp_ep_true_attention_typed_kv_compressed`
- `tp_ep_true_attention_typed_kv_indexer`
- `tp_ep_true_attention_typed_kv_history`

The HTTP A/B harness now has `--typed-quiet-variant quiet` and records
`typed_quiet_meta`.

## Validation

Local:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-http-ab.py
git diff --check
```

Result: PASS.

V100 build:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

Same-shape measurement:

```text
endpoint: /v1/chat/completions
requests: 32 concurrent
slots: 32
ctx: 262144
tokens/request: 8
diagnostic output head: on
HC persistent state: on
typed candidates: skip-current-load on
```

Summary:

```text
case           server tok/s  decode tok/s  typed lines  quiet
control        309.202473    730.769885    0            0
typed-history   73.427107     85.479279    2058         0
typed-quiet     75.284862     87.627420    0            1
```

The verbose typed baseline emitted `946` raw, `105` compressed, `105`
indexer, and `902` history lines. The quiet candidate emitted zero typed PASS
lines and reported `typed_quiet_meta=1`.

Final idle GPU recheck:

```text
0, 0, 32495, 0
1, 0, 32495, 0
2, 0, 32495, 0
3, 0, 32495, 0
4, 0, 32495, 0
5, 0, 32495, 0
6, 0, 32495, 0
7, 0, 32495, 0
```

## Decision

If quiet mode materially improves typed-history throughput, stdout/log
formatting is a real hot-path cost and the serving default should move toward
aggregate counters rather than per-row PASS lines. If quiet mode is flat, the
next sprint should target typed-row synchronization and bookkeeping rather than
logging.

Quiet mode was effectively flat: `73.427107 -> 75.284862` server tok/s
(`+2.5%`) and `85.479279 -> 87.627420` decode tok/s (`+2.5%`). Per-row stdout
formatting is not the main typed-KV regression.

The next sprint should target the typed row API synchronization/bookkeeping
shape directly. The current device APIs launch one store/load kernel per
slot/family/rank and the serving path wraps those calls with broad device
synchronizations. That per-row API shape is now the best candidate for the
remaining `~4.1x` wall throughput gap to control.

## Artifacts

- `logs/from-cluster/sprint342-typed-kv-quiet-overhead/cluster/summary.tsv`
- `logs/from-cluster/sprint342-typed-kv-quiet-overhead/cluster/final-gpu-recheck.csv`
