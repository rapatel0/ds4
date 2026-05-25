# TEMP Status Report 377: Batched Paged Attention Gate

Date: 2026-05-25

## Current Focus

Sprint 377 is the S-C gate from `TEMP_THROUGHPUT_PROMPT.md`:
`--batched-paged-attn-gate`.

Sprint 376 rejected CUDA graph replay because the current TP/EP decode step
uses stream-capture-incompatible `cudaMemcpyPeerAsync` transport. Sprint 377
therefore targets launch-count and staging reduction in the attention/KV row
path without depending on CUDA graphs.

## Baseline V100 Run

Command shape:

```text
32 active chat requests
32 configured slots
256K context
position 262080
32 generated tokens/request
GPU sampling interval 250 ms
```

Artifact path:

```text
logs/from-cluster/sprint377-batched-paged-attn/baseline-matrix
```

Topline:

| Metric | Value |
|---|---:|
| HTTP 200 | `32/32` |
| Coalesced batch size | `32` |
| First token | `89340` |
| Client generated tok/s | `40.157540` |
| Server generated tok/s | `74.895420` |
| Server generated tok/s decode | `88.372350` |
| Server continuation tok/s decode | `88.329223` |
| Scaffold projected slot-step tok/s | `56.990488` |
| Avg GPU util | `7.972222%` |
| Max GPU util | `38%` |
| Max GPU memory used | `32398 MiB` |
| Compressed-KV sum | `5436.764269 ms` |

Stage evidence from the first scaffold row:

| Stage | ms |
|---|---:|
| `sum_decode_ms` | `561.497209` |
| `sum_pre_ep_attention_projection_ms` | `87.215132` |
| `sum_pre_ep_attention_state_ms` | `103.187385` |
| `sum_pre_ep_compressed_kv_ms` | `162.085447` |
| `sum_pre_ep_raw_read_ms` | `40.072128` |
| `sum_pre_ep_typed_history_ms` | `15.094877` |
| `sum_hc_current_input_ms` | `457.899720` |
| `sum_ep_ms` | `31.515356` |
| `sum_compose_ms` | `42.200063` |

Interpretation: baseline utilization is still low and GPU0-heavy. The
attention/KV prefix remains a large part of the decode step, with compressed
KV and attention state/projection the most visible targets for this sprint.

## Next Work

1. Add default-off `--batched-paged-attn-gate` plumbing.
2. Add launcher/profile env plumbing.
3. Implement a fixed-size per-layer row-family plan for raw-SWA, compressed
   attention, and ratio-4 indexer rows.
4. Add the smallest batched row-family kernel that can reduce launch count
   while preserving first-token/checksum parity.
5. Build and run direct + HTTP V100 A/B.

## Gate Plumbing Smoke

Implemented default-off plumbing for:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu --batched-paged-attn-gate`
- `DS4_V100_TP_EP_BATCHED_PAGED_ATTN=1`
- `tools/ds4-v100-tp-ep-profile.py --batched-paged-attn`
- active-slot matrix artifact suffixing for `--batched-paged-attn`

Validation:

| Check | Result |
|---|---|
| `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-active-slot-matrix.py` | pass |
| `bash -n tools/ds4-v100-run-appliance.sh` | pass |
| V100 `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke` | pass |
| launcher `--print-command` with `DS4_V100_TP_EP_BATCHED_PAGED_ATTN=1` | emits `--batched-paged-attn-gate` |

No-op direct smoke artifact:

```text
logs/from-cluster/sprint377-batched-paged-attn/gate-plumbing-smoke/none-direct-batched-paged-attn
```

Smoke result:

| Metric | Value |
|---|---:|
| Return code | `0` |
| First token | `54639` |
| Output finite bad | `0` |
| Generated tok/s decode | `77.855330` |
| Scaffold checksum path | pass |

Interpretation: the gate is now safely wired and default-off. It is currently
a no-op except for enabling existing typed batch-row state; the next step is
the fixed-size row-family plan and first batched row kernel.

## Row-Family Plan Smoke

Implemented the fixed-size row-family plan audit behind
`--batched-paged-attn-gate`.

Validation artifacts:

```text
logs/from-cluster/sprint377-batched-paged-attn/row-plan-smoke/none-direct-batched-paged-attn
logs/from-cluster/sprint377-batched-paged-attn/row-plan-change-smoke/none-direct-batched-paged-attn
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: pass.

The first 1-token direct smoke emitted `43` plan rows and preserved first token
`54639`, but all compressed/indexer counts were zero because the first step
was raw-SWA only.

The 8-token direct smoke is the useful row-plan result:

| Metric | Value |
|---|---:|
| Return code | `0` |
| First token | `98751` |
| Output finite bad | `0` |
| Plan rows emitted | `127` |
| Generated tok/s decode | `96.553089` |
| Continuation tok/s decode | `99.794998` |
| Compressed rows emitted | `42` |
| Compressed-KV sum | `813.233407 ms` |
| Attention projection sum | `479.943118 ms` |
| Attention state sum | `339.001691 ms` |
| Raw-read sum | `124.838245 ms` |
| Typed-history sum | `30.807917 ms` |
| EP sum | `208.598725 ms` |
| Compose sum | `145.634186 ms` |

Representative ratio-4 row-plan line once compressed/indexer rows appear:

```text
layer 2 position 262083 raw_valid_rows 4 visible_attn_rows 1 visible_indexer_rows 1 target_family_kernels 3
layer 4 position 262087 raw_valid_rows 8 visible_attn_rows 2 visible_indexer_rows 2 target_family_kernels 3
```

Interpretation: the row-family planner is working, but it also weakens the
original S-C bottleneck assumption. At this served shape, pending typed-history
reloads are `0` in the observed compressed/indexer samples because
skip-current-load and the bounded reload cache are already avoiding the reload
storm. Typed-history is only `30.807917 ms` of `2651.391081 ms` summed decode
in the 8-token direct run. The larger measured costs are still compressed KV
projection/state, attention projection/state, then EP/compose.

Practical consequence: a narrow S-C kernel that only replaces typed-history
row loads is unlikely to move topline throughput. The next useful S-C work must
either fuse more of raw+compressed attention itself or Sprint 377 should close
with this evidence and move to `--compact-moe-decode-gate`.
