# Sprint 079 Report: Routed MXFP4 Row-Pair Occupancy Probe

## Outcome

`SHIP_OPT_IN_ONLY`.

Sprint 079 added an opt-in row-pair variant for the grouped routed MXFP4
gate/up/SwiGLU and down-sum kernels. The path is correct on focused and
real-model V100 smokes, but it regressed the paired 1M/4-slot sustained
benchmark by `0.22%`, so the default appliance path remains unchanged.

## Changes

- Added `DS4_CUDA_MXFP4_ROUTE_ROWS2=1` as an internal CUDA kernel-selection
  switch.
- Added row-pair grouped routed MXFP4 gate/up/SwiGLU kernels for contiguous and
  pointer-input batched execution.
- Added a row-pair grouped routed MXFP4 down-sum kernel.
- Kept existing public GPU APIs unchanged.
- Extended `tests/cuda_v100_mxfp4_moe_smoke.c` to exercise the row-pair
  pointer-input batch path.

## Validation

Local:

```bash
make tests/cuda_v100_mxfp4_moe_smoke.o
git diff --check
```

Local CUDA compilation was unavailable in this shell because the CUDA compiler
resolved as `c`; the actual `sm_70` build was performed on the V100 pod.

V100 build:

```bash
CUDA_ARCH=sm_70 make ds4_cuda.o tests/cuda_v100_mxfp4_moe_smoke \
  tests/cuda_v100_selected_token_smoke tests/cuda_v100_full_scheduler_smoke \
  tools/ds4-v100-replay
```

V100 smokes:

```bash
env -u DS4_CUDA_MXFP4_ROUTE_ROWS2 ./tests/cuda_v100_mxfp4_moe_smoke
DS4_CUDA_MXFP4_ROUTE_ROWS2=1 ./tests/cuda_v100_mxfp4_moe_smoke
DS4_CUDA_MXFP4_ROUTE_ROWS2=1 ./tests/cuda_v100_full_scheduler_smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --token 16 --position 16 --slots 4
DS4_CUDA_MXFP4_ROUTE_ROWS2=1 ./tests/cuda_v100_selected_token_smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --expected-token-hex 3136 --top-k 1
```

Results:

- default focused MXFP4 smoke: `ok`
- row-pair focused MXFP4 smoke: `ok`
- row-pair full scheduler smoke: `stages=8 ... uploaded_tensors=1328 ... ok`
- row-pair selected-token smoke:
  `selected=926 logit=35.250481 expected=3136 topk=926:35.250481:3136 ok`

## Throughput Evidence

Fixture:

- model: `/models/DSv4-Flash-256e-fixed.gguf`
- pack index: `docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv`
- context: `1048576`
- slots: `4`
- tokens/request: `16`
- measured requests: `4`
- warmup requests: `1`
- async pipeline mode: `per-step`
- expected first token hex: `3136`

| Path | Generated tok/s | Continuation tok/s | Avg latency ms | Async total ms | Handoff sum ms | Device sync sum ms | Avg GPU util | Token match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default routed MXFP4 | `9.055694` | `8.489713` | `7065.558` | `6923.854` | `259.981` | `7.062` | `19.911%` | `4/4` |
| `DS4_CUDA_MXFP4_ROUTE_ROWS2=1` | `9.035946` | `8.471200` | `7081.063` | `6939.079` | `280.830` | `7.118` | `19.790%` | `4/4` |

Generated tok/s changed by `-0.22%`, below the `3%` default threshold and in the
wrong direction.

## Decision

Keep the row-pair kernels opt-in behind `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`.

The experiment is useful evidence: reducing CTA count by pairing adjacent rows
does not solve the routed MXFP4 utilization problem. The next kernel sprint
should target a more substantial execution-shape change, such as route/expert
tiling, a packed integer dot-product path, or a persistent grouped expert
kernel, rather than more row-level CTA consolidation.

## Artifacts

- `logs/from-cluster/sprint079-smokes`
- `logs/from-cluster/sprint079-rows1-default`
- `logs/from-cluster/sprint079-rows2-optin`
