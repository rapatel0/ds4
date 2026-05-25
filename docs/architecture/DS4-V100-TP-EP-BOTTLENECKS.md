# DS4 V100 TP/EP Bottleneck Map

Last updated: 2026-05-25

This document summarizes the current TP/EP bottleneck picture for the
DeepSeek V4 Flash V100 appliance. It answers three questions:

1. What are the measured bottlenecks?
2. Where do they appear layer by layer?
3. What has already been tried?

The scope is the TP/EP path only. PP/layer-split work is historical baseline
only and is not an optimization direction.

## Executive Summary

The current TP/EP serving path is operational, but not performance-complete.
The main bottleneck is no longer "missing tensor-core dispatch" in the broad
sense. The current limiter is the true-attention/compressed-KV prefix,
especially compressed/indexer dense projection and the staging/state work
around it.

At `32` slots and `256K` context, active request coalescing works, but server
decode throughput and GPU utilization stay nearly flat from `1` to `32`
active requests. That means the full 32-slot step is dominated by fixed
per-layer/per-step work, not simply by lack of active slots.

The strongest current evidence:

| Evidence | Result | Interpretation |
|---|---:|---|
| Sprint 371 active-slot matrix | server decode stays `97.4-100.0 tok/s` from `1` to `32` active requests | coalescing works, but full-step cost is fixed and bottlenecked elsewhere |
| Sprint 371 GPU utilization | average GPU util stays `9.8-10.3%`; max around `39-41%` | low utilization remains at full 32-slot target shape |
| Sprint 351 prefix split | compressed KV `228.813 ms`, attention projection `170.866 ms`, attention state `105.655 ms` out of the measured prefix | true-attention/compressed-KV prefix dominates |
| Sprint 352 compressed-KV internals | indexer dense `36.616 ms`, attention dense `24.659 ms`, attention state/emit `24.363 ms` at emitted-row boundary | compressed/indexer dense and state/emit are the local hot path |
| Sprint 372 skip dense host stats | direct scaffold `100.740 -> 117.464 tok/s`, compressed-KV sum `3141.768 -> 1789.795 ms` | host stats/sync were real overhead; remaining path is still dense/staging heavy |
| Sprint 373 INT8 audit | BF16 attention compressor shapes are `M=32, N=128/64, K=4096` and save `232.5 MiB` if packed INT8+scale | best next kernel-format workbench target |
| Sprint 374 INT8 workbench | best copied tc-grid INT8 is `4.2-4.6x` slower than cuBLAS FP16 for `M=32,N=128/64,K=4096` | do not wire tc-grid INT8; use TurboMind small-M or fuse the boundary |

## Bottleneck Ranking

| Rank | Bottleneck | Where It Appears | Current Status |
|---:|---|---|---|
| 1 | Compressed/indexer dense projection | ratio-4 layers and ratio-128 compressed layers | primary measured hot path; tc-grid INT8 rejected; TurboMind small-M or fusion next |
| 2 | Compressed-KV state/emit boundaries | ratio-4 and ratio-128 layers when compressed rows emit | some fusions tried; pool+norm promoted, others rejected |
| 3 | Attention projection/state | all layers, with extra cost on compressed layers | measured significant prefix cost; not yet deeply optimized |
| 4 | GPU0-heavy orchestration/output-head/harness work | serving harness and selected-token/output path | visible as GPU0 higher util than peers; not final production balance |
| 5 | Active-slot inefficiency at low/moderate occupancy | scheduler/coalescing path | useful practical work, but not the 32-slot topline limiter |
| 6 | F8/BF16/FP4 conversion on V100 | all low-bit source-weight families | unavoidable on V100; must happen inside GPU kernels or bounded scratch |
| 7 | MTP not yet enabled | decode loop | deferred until TP/EP path is correct and benchmarkable |

## DType Reality

V100 does not natively execute BF16, FP8, or Blackwell-style FP4 tensor-core
MMA. On this hardware:

- BF16 source tensors must be converted to FP16 for HMMA or handled by custom
  kernels that expand inside the GPU.
- FP8/F8_E4M3_B128 source tensors are compressed storage plus scale metadata;
  they must be decoded into FP16/HMMA or custom low-bit paths.
- MXFP4 routed experts are source-compressed; TurboMind-style kernels unpack
  and compute through FP16 tensor-core paths.
- INT8 is attractive only where the resulting kernel shape and scale layout
  reduce data movement or improve tensor-core utilization enough to offset
  quantization risk.

Sprint 373 corrected an important assumption: the compressed/indexer dense hot
set is mostly BF16 in the current pack, not FP8.

| Tensor Family | Source DType | TP Serving Shape | INT8+Scale Decision |
|---|---|---|---|
| `attn_compress_{kv,gate}.weight` | BF16 | `M=32, N=128/64, K=4096` | tc-grid INT8 rejected; TurboMind small-M or fusion target |
| `indexer.compress_{kv,gate}.weight` | BF16 | `M=32, N=32, K=4096` | possible fused-indexer target |
| `indexer.proj.weight` | BF16 | `M=32, N=8, K=4096` | too small for standalone GEMM |
| `indexer.attn_q_b.weight` | F8 E4M3 B128 | `M=32, N=1024, K=1024` | compute-only candidate; INT8+scale is larger |

## Layer-By-Layer Bottleneck Map

Layer classes:

- Layers `0-1`: SWA-only. No compressed-KV/indexer path.
- Even layers `2,4,...,42`: ratio-4 compressed KV plus indexer. These are the
  hottest attention/cache layers.
- Odd layers `3,5,...,41`: ratio-128 compressed KV, no indexer. These still
  pay compressed attention/compressor cost, but much less long-context KV work.

The table below describes the expected hot path at the current `32` slot /
`256K` serving target.

| Layer | Class | Compression | Indexer | Dominant Current Bottleneck | Candidate Kernel/Format Work |
|---:|---|---:|---|---|---|
| 0 | SWA-only | none | no | attention projection/state, routed FFN, GPU0/control overhead | FP8/F16 attention dense tuning; routed expert path already uses TurboMind-style work |
| 1 | SWA-only | none | no | attention projection/state, routed FFN | FP8/F16 attention dense tuning |
| 2 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 3 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 4 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 5 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 6 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 7 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 8 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 9 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 10 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 11 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 12 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 13 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 14 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 15 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 16 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 17 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 18 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 19 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 20 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 21 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 22 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 23 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 24 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 25 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 26 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 27 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 28 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 29 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 30 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 31 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 32 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 33 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 34 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 35 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 36 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 37 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 38 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 39 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 40 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 41 | ratio-128 | 128 | no | attention compressor BF16 dense, compressed state/emit | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion |
| 42 | ratio-4 | 4 | yes | attention compressor BF16 dense, indexer dense/state, compressed-row emit, output-head interaction after final layer | tc-grid INT8 rejected; TurboMind small-M or compressor-state fusion; output-head rank-balance later |

## What We Have Tried

| Sprint | Attempt | Result | Decision |
|---|---|---|---|
| 347 | Permanent direct profile mode plus nvprof window | Confirmed TurboMind, CUTLASS, BF16/F8 unpack, gather, dense-fill, cast, compressor kernels are active | bottleneck is not simply missing kernel dispatch |
| 348 | Peer-gather current input across ranks | Correct but slower: decode `87.264 -> 67.495 tok/s` | rejected |
| 349 | Stream-scoped synchronization for HC/current input | Improved direct decode `74.842 -> 81.191 tok/s`; promoted | default |
| 350 | Split HC-current timer into substages | Showed old label included true-attention/compressed-KV prefix | redirected optimization target |
| 351 | Split true-attention/compressed-KV prefix | Measured compressed KV, attention projection, attention state as major prefix costs | focus moved to compressed projection/store |
| 352 | Split compressed-KV internals; suppress typed stores | Store suppression flat: `81.647 -> 81.734 tok/s`; indexer/attention dense identified | typed stores not next lever |
| 353 | Fused compressed/indexer input fill | Correct; small direct improvement `79.012 -> 80.535 tok/s`, stage reduction under `1 ms` | diagnostic only |
| 354 | Fused RoPE + F16 round for emitted rows | Correct; total decode regressed within noise | rejected |
| 355 | Fused compressed pool + norm | Correct; compressed-KV sum `130.511 -> 127.737 ms`; small topline win | later promoted after confirmation |
| 356 | Launcher/profile exposure for compressed fusion gates | Made gates testable through serving harness | infrastructure |
| 357 | Selected-token HTTP emitted-row profile mode | Avoided prompt-position ambiguity for compressed-row tests | infrastructure |
| 358 | Longer selected-token HTTP A/B for fusions | Combined input-fill + pool-norm not promotable; pool-only interesting | pool-only kept |
| 359 | Direct confirmation of pool+norm | Decode `95.852 -> 97.619 tok/s`; compressed-KV `3521.094 -> 3458.470 ms` | promoted pool+norm default |
| 360 | Launcher default validation for pool+norm | Default command and serving run confirmed | default validated |
| 361 | Full chat endpoint check for pool+norm | Stable but short-chat topline flat/slightly slower | decode-path win, not short-chat proof |
| 362 | Profile harness aligned with launcher defaults | Default/disable behavior made reliable | infrastructure |
| 363 | Wider pool+norm+RoPE+round fusion | Correct but direct decode regressed `95.908 -> 95.463 tok/s` | rejected |
| 364 | Direct compressed input fill from `hc->d_attn_normed` | Correct but doubled one-step compressed-KV cost `126.725 -> 260.366 ms` | rejected |
| 365 | Local fused attention input fill | Correct; HTTP regressed `72.886 -> 70.674 tok/s` despite slight direct positive | diagnostic only |
| 366 | CUDA event waits instead of host sync between compressed fill/dense | HTTP improved `71.834 -> 74.432 tok/s`; compressed-KV `3437.636 -> 3137.755 ms` | promoted default |
| 367 | Chat validation for event-wait default | Chat improved `50.648 -> 52.023` client tok/s, server decode `96.117 -> 99.522` | default validated |
| 368 | Context admission check | Invalid long-context chat shape now returns HTTP 400; valid shape remains OK | correctness fix |
| 369 | Permanent GPU utilization sampler | Showed low util and GPU0-heavy imbalance in normal artifacts | metrology |
| 370 | Active-slot matrix driver | Smoke verified coalescing | metrology |
| 371 | Full active-slot matrix | Server decode/util flat from `1` to `32` active requests | full-occupancy kernel/state work is next |
| 372 | Skip compressed dense host stats | Direct decode `100.740 -> 117.464 tok/s`; chat server decode `99.748 -> 117.341`; selected-token semantic parity clean | production candidate, still default-off pending deterministic chat parity |
| 373 | INT8 candidate audit | Scoped INT8+scale candidate set saves `280.6 MB`; BF16 attention compressor is best target | next workbench target |
| 374 | INT8 compressor workbench | cuBLAS FP16 `0.009250/0.008803 ms` beats best tc-grid INT8 `0.042721/0.036673 ms` for `N=128/64` | reject copied tc-grid INT8 for compressor path |

## What Has Not Been Proven Yet

| Item | Status |
|---|---|
| Full production-quality chat parity with skip-stats default-on | not proven; selected-token parity is clean |
| INT8 compressor kernel performance at `M=32,N=128/64,K=4096` | measured in Sprint 374; copied tc-grid INT8 is slower than FP16 baseline |
| Production offline INT8 pack conversion | not implemented yet |
| Fully balanced output-head/rank-0 serving path | not implemented yet |
| MTP decode uplift | deferred until TP/EP serving path is stable and benchmarkable |
| 1M context with 32 slots | not feasible under current KV budget; 32 slots target is 256K |

## Current Recommendation

The Sprint 374 workbench rejected the copied tc-grid INT8 path for the BF16
attention compressor family. The next implementation step should change the
execution boundary instead of only changing dtype:

1. Adapt the vLLM/TurboMind SM70 small-M GEMM registry for the exact compressor
   shape and compare it against the same cuBLAS FP16 baseline.
2. Or fuse compressor dense output with adjacent compressed state/emit work so
   the runtime removes launches, staging, and format traffic.

Do not add an offline INT8 pack variant for `attn_compress_{kv,gate}.weight`
until a kernel path beats the current FP16 tensor-op baseline on V100.

## Reference Artifacts

- [STATUS.md](../sprints/STATUS.md)
- [SPRINT-371.md](../sprints/SPRINT-371.md)
- [SPRINT-372.md](../sprints/SPRINT-372.md)
- [SPRINT-373.md](../sprints/SPRINT-373.md)
- [SPRINT-374.md](../sprints/SPRINT-374.md)
- [TEMP_STATUS_REPORT_084.md](../../TEMP_STATUS_REPORT_084.md)
- [TEMP_STATUS_REPORT_085.md](../../TEMP_STATUS_REPORT_085.md)
- [INT8_CANDIDATE_AUDIT.md](../../logs/from-cluster/sprint373-int8-candidate-audit/INT8_CANDIDATE_AUDIT.md)
- [INT8_COMPRESSOR_WORKBENCH.md](../../logs/from-cluster/sprint374-int8-compressor-workbench/INT8_COMPRESSOR_WORKBENCH.md)
