# SPRINT-019 Report: V100 Integrated Single-Layer Runtime Slice

## Verdict

`SHIP`.

Sprint 019 moved the V100 path from isolated descriptor-bound attention and FFN
smokes to a reusable hidden-vector layer execution surface. The new executor
produces a bounded next-hidden vector for layer 2 by composing semantic
raw/compressed attention inputs, grouped attention output, residual, FFN
pre-norm, real hash-router selected MXFP4 experts, shared F8 expert, and final
residual.

## What Changed

- Added `ds4_v100_layer_execute.h` and `ds4_v100_layer_execute.c`.
- Added `ds4_v100_layer_execute_decode`, a callable layer-state based executor.
- Added grouped descriptor-bound F8 attention output:
  8 groups of `[4096 -> 1024]` from `attn_output_a`, then
  `[8192 -> 4096]` through `attn_output_b`.
- Added RoPE/head-RMS attention sequencing before attention and inverse RoPE
  after attention.
- Added semantic attention over explicit raw plus compressed KV rows with sink
  logits and compressed-row masking.
- Added router-selected descriptor-bound FFN inside the executor, using
  `ffn_norm` as the FFN input.
- Added `tests/cuda_v100_integrated_layer_smoke.c`.
- Added the integrated smoke to `tools/ds4-v100-gate.sh`.
- Updated gate readiness from `full_layer_scheduler,attention_residual_norm,...`
  to `full_43_layer_scheduler,real_model_selected_token,...`.

## Validation

Local build checks:

```sh
make ds4_v100_layer_execute.o tests/cuda_v100_integrated_layer_smoke.o tests/v100_layer_state_smoke
git diff --check
```

V100 one-card integrated smoke:

```text
cuda_v100_integrated_layer_smoke: layer=2 token=16 pos=16 expert0=84 gpu=0 arena_bytes=11784434944 hidden=4096 raw=3 comp=3 ok
```

Full V100 appliance gate:

```text
gate	integrated_layer	PASS	command=./tests/cuda_v100_integrated_layer_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --layer 2 --router-token 16 --position 16
gate	readiness	NOT_READY	missing=full_43_layer_scheduler,real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Logs:

- `docs/sprints/drafts/SPRINT-019-INTEGRATED-LAYER-1GPU.log`
- `docs/sprints/drafts/SPRINT-019-GATE-CLUSTER.log`
- `docs/sprints/drafts/SPRINT-019-GATE-CLUSTER/`

## Important Limits

- The executor accepts raw and compressed KV tensors as explicit inputs. It does
  not yet bind and execute real attention compressor/indexer descriptors.
- The slice operates on one hidden vector, not the full DS4 four-row HC state
  with HC pre/post composition.
- The result is a layer-2 next-hidden correctness gate, not a full 43-layer
  selected-token decode.
- No MTP, serving endpoint, slot wavefront, or throughput benchmark shipped in
  this sprint.

## Next Sprint

The next useful sprint is to bind compressor/indexer descriptors into
`ds4_v100_layer_state`, wrap the executor with HC pre/post state handling, and
start walking the scheduler across layer classes toward a real selected-token
gate.
