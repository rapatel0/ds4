# Sprint 181 - Persistent Production Appliance Pack

Date: 2026-05-22

## Objective

Restore the real optimized TurboMind appliance benchmark after the build pod
recycle by moving pack generation and benchmark artifacts off the host root
mirror and onto persistent `localpool` storage.

Sprint 180 validated MTP verify microbatching only against the source-layout
pack because the previous optimized interleaved TurboMind pack was lost with
the disposable `/workspace`. This sprint makes the production-pack path
repeatable and then reruns the 16-slot/256K throughput gate against the
optimized pack.

## Scope

- Add a localpool-backed 8-GPU build pod manifest.
- Add an operator helper to recreate the build pod with `/workspace` backed by
  `/localpool/ds4/workspace`.
- Document the pack/build/benchmark workflow in the appliance runbook.
- Build TurboMind and `tools/ds4-v100-appliance-pack` on the V100 pod.
- Generate the interleaved gated-SiLU TurboMind appliance pack under
  `/workspace/packs/`.
- Run the production-pack selected-token smoke and sustained decode gate.
- Record prompt/prefill, generated, and continuation/decode tok/s separately.

## Non-Goals

- No new kernel optimization in this sprint.
- No tensor-parallel topology change.
- No promotion of MTP commit mode.

## Definition of Done

- [x] `/workspace` inside `llm/llamacpp-build-8gpu` is backed by localpool, not
  `/dev/md0`.
- [x] TurboMind `.so`, appliance packer, and replay binary build on the pod.
- [x] Production interleaved TurboMind appliance pack exists under
  `/workspace/packs/`.
- [x] Selected-token smoke passes against the appliance pack.
- [x] 16-slot/256K sustained decode records generated and continuation tok/s.
- [x] Results are copied into `logs/from-cluster/`.
- [x] Vision/runbook status is updated.
- [x] Changes are committed.

## Planned Commands

```bash
tools/ds4-v100-localpool-build-pod.sh ensure

cmake -S kernels/turbomind/ggml-turbomind \
  -B build/turbomind-v100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DFETCHCONTENT_SOURCE_DIR_FMT=/workspace/deps/fmt \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/workspace/deps/cutlass \
  -DFETCHCONTENT_SOURCE_DIR_CONCURRENTQUEUE=/workspace/deps/concurrentqueue

cmake --build build/turbomind-v100 \
  --target ggml-turbomind test_ggml_turbomind_grouped_gate_up_fusion -j80

CUDA_ARCH=sm_70 make -j80 tools/ds4-v100-appliance-pack tools/ds4-v100-replay

tools/ds4-v100-appliance-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /workspace/packs/ds4-appliance-full-tm-gated \
  --pack-gpu 0 \
  --fuse-gate-up-interleaved \
  --lib build/turbomind-v100/libggml-turbomind.so

tools/ds4-v100-sustained-decode-bench.sh \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated \
  --ctx-tiers 262144 \
  --slot-tiers 16 \
  --tokens 64 \
  --requests 16 \
  --warmup-requests 1 \
  --async-pipeline-mode per-step \
  --async-event-handoff \
  --microbatch-wait-us 200000 \
  --log-dir /workspace/logs/sprint181-production-pack-256k-16slot
```

## Outcome

Completed on `gpu-01` in pod `llm/llamacpp-build-8gpu`.

Storage and build:

- Replaced the disposable build pod with
  `deploy/v100/ds4-v100-build-localpool.pod.yaml` via
  `tools/ds4-v100-localpool-build-pod.sh ensure`.
- Verified `/workspace` inside the pod is backed by `localpool` with `2.5T`
  available, while `/models` remains the read-only model PVC.
- The pod cannot resolve GitHub, so the build used local
  `FETCHCONTENT_SOURCE_DIR_*` overrides for `fmt`, CUTLASS, and
  `concurrentqueue`.
- TurboMind `.so`, `tools/ds4-v100-appliance-pack`, and
  `tools/ds4-v100-replay` built successfully for `sm_70`.
- TurboMind gated-SiLU microbench passed. The rebuilt library measured:
  - 6 routes: `0.2464 ms` separate, `0.1679 ms` fused,
    `0.1657 ms` gated.
  - 24 routes: `0.2483 ms` separate, `0.1687 ms` fused,
    `0.1676 ms` gated.
  - 48 routes: `0.2554 ms` separate, `0.1668 ms` fused,
    `0.1673 ms` gated.

Pack:

```text
/workspace/packs/ds4-appliance-full-tm-gated-s181
total size: 142G
source_rows=1199
tm_rows=86
skipped_rows=43
source_bytes=8973123932
tm_weight_bytes=138512695296
tm_scale_bytes=8657043456
```

Per-GPU shard sizes:

| GPU | Bytes |
|---:|---:|
| 0 | `22524134668` |
| 1 | `21494393612` |
| 2 | `21494393612` |
| 3 | `21494393612` |
| 4 | `21494393612` |
| 5 | `17922654732` |
| 6 | `17901334540` |
| 7 | `11817197824` |

Validation:

| Run | Context | Slots | Tokens/request | Requests | Generated tok/s | Continuation tok/s | Match |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke | 256K | 16 | 1 | 2 | n/a | n/a | 2/2 |
| Sustained base | 256K | 1 | 64 | 1 | `10.357728` | `10.195888` | 1/1 |
| Sustained base | 256K | 16 | 64 | 16 | `48.163685` | `47.411127` | 16/16 |
| MTP verify | 256K | 16 | 2 | 16 | `7.013779` | `3.506890` | 16/16, MTP 16/16 |

The MTP verify run proves production-pack compatibility for the active
microbatch MTP surface. It is not a speedup result because verify mode still
computes the base target token and only checks the draft.

Evidence:

```text
logs/from-cluster/sprint181-production-pack/
```

## Decision

The optimized production appliance baseline is restored and now survives pod
recycles because the pack lives on localpool. Future performance sprints should
use `/workspace/packs/ds4-appliance-full-tm-gated-s181` instead of falling back
to source-layout validation when the pod is recreated.

The next material optimization should be measured against this pack. The last
wrapper-level six-route probes are exhausted; remaining plausible performance
work is either a true persistent routed-FFN boundary that removes intermediate
global-memory handoffs, or a broader persistent TP/EP ownership model that
avoids per-layer copy-back/reduce overhead.
