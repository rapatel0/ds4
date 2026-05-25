# TEMP Status Report 069 - TP/EP Emitted-Row HTTP Profile

Date: 2026-05-25

## Current Focus

TP/EP only. Sprint 357 closed the gap from direct-only compressed-fusion
experiments to HTTP-serving-visible emitted-row profiling. PP/layer-split work
remains frozen. MTP remains deferred until TP/EP serving correctness and
performance are stronger.

## What Changed

- `tools/ds4-v100-tp-ep-profile.py` now supports:
  - `--http-endpoint chat`
  - `--http-endpoint selected-token`
- `chat` remains the default path.
- `selected-token` posts directly to `/v100/selected-token` with no prompt
  prefill, so `DS4_V100_TP_EP_POSITION=262143` reliably exercises emitted
  compressed rows.
- HTTP summaries now parse TP/EP timing lines from `server.out`, matching the
  useful direct-profile compressed-KV timing fields.

## Latest V100 Evidence

Shape:

```text
32 slots
256K context
position=262143
1 token/request
32 concurrent selected-token HTTP requests
HC current stream sync enabled
```

Results:

| Variant | HTTP 200 | Emitted layers | Fused input layers | Fused pool layers | Client tok/s | Compressed-KV sum ms |
|---|---:|---:|---:|---:|---:|---:|
| control | 32/32 | 41 | 0 | 0 | 19.739916 | 127.697384 |
| input-fill + pool-norm | 32/32 | 41 | 20 | 40 | 19.719484 | 123.651985 |

Interpretation:

- The fused path is active through the resident HTTP serving path.
- Parsed compressed-KV stage time improves by `4.045399 ms`.
- One-token selected-token client tok/s is not a useful promotion metric
  because HTTP orchestration dominates the tiny request.
- The compressed-fusion gates should remain opt-in until a longer amortized
  serving A/B shows a real topline win.

## Tests Run

- Local:
  - `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`
  - `git diff --check`
  - `make test` failed due missing local `ds4flash.gguf` fixture.
  - Model-free smokes passed:
    - `pack_index_smoke`
    - `gpu_arena_smoke`
    - `bf16_probe_smoke`
    - `v100_context_smoke`
    - `source_dtypes_smoke`
    - `ds4-v100-tp8-kv-shard-smoke`
- V100 pod:
  - profiler harness py_compile passed.
  - `make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
    passed.
  - Direct 32-slot / 256K / emitted-row smoke passed with token `54639` and
    `81.102929` projected generated decode tok/s.

## Next Best Step

Do not add more HTTP wrappers just for their own sake. Either:

1. Run longer selected-token/chat serving A/B with enough tokens/request to
   amortize HTTP overhead and decide whether fused pool+norm or combined
   fusions should become defaults, or
2. Continue implementation on compressed-KV state/emit fragmentation, where
   the stage timer still shows substantial work.

The practical goal remains an end-to-end TP/EP DS4-V100 serving appliance at
32 slots / 256K context, with production typed KV semantics and benchmarked
throughput.
