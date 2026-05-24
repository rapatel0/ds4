---
sprint: 341
title: TP/EP Typed KV Store Family Cost
status: completed
started: 2026-05-24
completed: 2026-05-24
branch: claude-takeover
---

# Sprint 341 - TP/EP Typed KV Store Family Cost

## Goal

Identify which production typed KV store family dominates the remaining
typed-history serving overhead: raw-SWA, compressed-attention, or ratio-4
indexer.

## Scope

TP/EP only. No PP/layer-split work. No MTP. Store suppression modes are
diagnostic-only and not production-correct; they keep f32 staging alive so the
serving shape can be measured while one or more typed persistence stores are
bypassed.

## Definition of Done

- [x] Add diagnostic binary gates for skipping raw, compressed, and indexer
      typed stores independently.
- [x] Add appliance env toggles and launcher command emission for those gates.
- [x] Extend the HTTP A/B harness to run typed store-family variants.
- [x] Build on the V100 pod.
- [x] Run a same-shape `32` request / `32` slot / `256K` / `8` token HTTP
      measurement for control, typed baseline, and store-family variants.
- [x] Record server tok/s, decode tok/s, typed store/load counters, final GPU
      state, and the decision for the next optimization sprint.

## Outcome

Added diagnostic-only binary flags:

- `--true-ds4-attention-typed-kv-skip-raw-store-gate`
- `--true-ds4-attention-typed-kv-skip-compressed-store-gate`
- `--true-ds4-attention-typed-kv-skip-indexer-store-gate`

Added matching appliance env toggles:

- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_RAW_STORE`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_COMPRESSED_STORE`
- `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_INDEXER_STORE`

The HTTP A/B harness can now run `baseline`, `no-raw`, `no-compressed`,
`no-indexer`, and `no-stores` typed variants.

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
case                       server tok/s  decode tok/s  typed stores
control                    308.223158    722.800920    0
typed-history               75.577828     87.938497    1156
typed-no-raw-store          77.304656     91.097009     210
typed-no-compressed-store   74.992652     87.648516    1051
typed-no-indexer-store      73.921469     86.162083    1051
typed-no-stores             79.039985     93.242875       0
```

The first full run completed control and typed baseline, then the first
store-family variant hit a CUDA allocation OOM during server startup after
rapid serial process teardown. The harness was hardened with GPU-idle cooldown
and the remaining variants were resumed successfully.

Final GPU state was idle: all devices reported `0 MiB` used.

## Decision

Typed KV store overhead is not the main remaining bottleneck. Removing all
typed stores improves the typed candidate only from `75.577828` to
`79.039985` server tok/s. The gap to control remains roughly `3.9x` by wall
tok/s and `7.7x` by decode tok/s.

The next sprint should profile/measure the typed instrumentation overhead that
remains even when stores and current loads are skipped. The likely sources are
per-layer device synchronizations, verbose typed PASS logging, and extra typed
history/indexer bookkeeping in the hot serving loop.

## Artifacts

- `logs/from-cluster/sprint341-typed-store-family-cost/cluster/combined-summary.tsv`
- `logs/from-cluster/sprint341-typed-store-family-cost/cluster/`
- `logs/from-cluster/sprint341-typed-store-family-cost-resume/cluster/`
