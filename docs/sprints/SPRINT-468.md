# Sprint 468: TP/EP Typed-History Final Barrier

## Objective

Fix TP/EP graph-event-order response parity without diagnostic stage sync by
adding the missing graph-safe boundary at the end of typed KV history.

## Rationale

Sprint 467 found that `DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC=typed_history`
restores HTTP parity, while `raw_read` alone does not. Reinspection of
`run_true_ds4_attention_typed_kv_history_load()` shows a likely precise cause:
for ratio-4 layers, graph mode copies `d_indexer_topk` from rank 0 to the other
ranks at the end of typed history, but does not add a final graph-order barrier
after those copies. A host stage sync waits for the copies; the earlier
`sync_typed_kv_boundary()` event barrier did not because it ran before this
final top-k copy.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add a final graph-safe typed-history boundary after indexer top-k broadcast.
- Remove the failed store-side `__threadfence_system()` experiment from Sprint
  467 because it did not restore parity and adds avoidable store overhead.
- Validate with HTTP response parity at `8` slots / `256K` without diagnostic
  stage sync.

## Definition of Done

- V100 build succeeds.
- Graph-event-order candidate runs without
  `DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC`.
- HTTP response parity matches eager control at `8` requests / `8` slots /
  `256K` / `1` token.
- If parity passes, rerun at `3` tokens to ensure the fix is not first-token
  only.
- Sprint and status docs record artifacts, tokens, throughput, and next
  decision.

## Implementation

- Added a final `sync_typed_kv_boundary(opt, ranks)` at the end of typed KV
  history after the ratio-4 indexer top-k copy/broadcast path.
- Kept the Sprint 467 graph-safe event barrier implementation of
  `sync_typed_kv_boundary()` for graph mode.
- Removed the failed Sprint 467 store-side `__threadfence_system()` experiment
  from typed KV F8 stores. It did not restore parity and would add avoidable
  store overhead.

## Validation

Remote build:

```bash
cd /localpool/ds4/workspace/ds4-sprint181
sudo env PATH=/localpool/ds4/cuda-12.2-link/bin:$PATH \
  CUDA_HOME=/localpool/ds4/cuda-12.2-link CUDA_ARCH=sm_70 \
  NVCCFLAGS="-I/localpool/ds4/cuda-12.2-link/include -gencode arch=compute_70,code=sm_70 -O3 --use_fast_math -Xcompiler -fPIC" \
  CUDA_LDLIBS="-L/localpool/ds4/cuda-12.2-link/lib64 -lcudart -lcublas -lcuda -lnccl" \
  make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Build completed successfully with only the expected unused-kernel warnings.

### HTTP A/B: 8 slots, 256K, 1 token

Artifacts:

`/localpool/ds4/workspace/logs/s468-typed-history-final-barrier-s8-t1`

Result:

| Metric | Eager control | Graph candidate |
|---|---:|---:|
| Response parity | - | `8/8` |
| Server generated decode tok/s | `20.331259` | `8.478366` |
| Client generated tok/s | `0.831886` | `0.303515` |
| Request-window GPU util avg | `11.35%` | `4.59%` |
| Graph captures | `0` | `43` |
| Graph replays | `0` | `0` |
| Event barrier calls | `0` | `215` |
| Min free VRAM | `5092 MiB` | `5086 MiB` |

The candidate matched parity without
`DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC`. This proves the missing
typed-history final boundary was the correctness bug for the non-persistent
graph-event-order path.

### HTTP A/B: 8 slots, 256K, 3 tokens

Artifacts:

`/localpool/ds4/workspace/logs/s468-typed-history-final-barrier-s8-t3`

Result:

| Metric | Eager control | Graph candidate |
|---|---:|---:|
| Response parity | - | `8/8` |
| Server generated decode tok/s | `20.333332` | `7.522808` |
| Server continuation tok/s | `20.240114` | `7.551176` |
| Client generated tok/s | `2.073092` | `0.710383` |
| Request-window GPU util avg | `10.32%` | `3.56%` |
| Graph captures | `0` | `43` |
| Graph replays | `0` | `0` |
| Event barrier calls | `0` | `215` |
| Min free VRAM | `5092 MiB` | `5086 MiB` |

The 3-token run confirms the correctness fix is not first-token-only. It also
confirms the current `--decode-cudagraph` serving path is capture-only for this
shape; it does not replay captured graphs, so it is expectedly slower than eager.

## Decision

Correctness: promote the typed-history final graph boundary.

Performance: do not promote non-persistent graph serving as a throughput path.
It is now correctness-clean, but capture-only execution is slower than eager.

Next work should move from one-shot graph capture to persistent graph replay:
reuse the now-correct event ordering, keep dynamic decode state device-updated,
and validate replay at the same `8` slot / `256K` shape before returning to the
larger `32` slot target.
