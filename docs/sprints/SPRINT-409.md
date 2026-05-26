# Sprint 409: Skip Unused TP Runtime Comp-State Arena

Date: 2026-05-26

## Overview

Sprint 408 proved the remaining HC-current NCCL blocker is not temporary lazy
output-head residency. Post-close free VRAM is still only `520-522 MiB`
against the `1536 MiB` NCCL reserve.

The next concrete memory target is the TP runtime compressed-state arena. At
the target `32` slot / `256K` shape, startup reports:

```text
kv_bytes_per_gpu         3707940864
comp_state_bytes_per_gpu 1803550720
scratch_bytes_per_gpu    1610612736
```

In the current `ds4_v100_tp_runtime.cu` implementation, `gpu_state.comp_state`
is allocated, reported, and freed, but no row store/load path reads or writes
it. The active compressed attention state is still held in `RankState`
mirrors. Skipping this unused arena should reclaim about `1.68 GiB/GPU`,
which is larger than the measured `~1.0 GiB` NCCL reserve deficit.

## Constraints

- TP/EP only. No PP/layer-split work.
- Keep the change behind an explicit gate until V100 target-shape validation.
- Preserve the default runtime allocation behavior until promotion.
- Preserve typed KV semantics and selected-token output.

## Implementation

Files:

- `ds4_v100_tp_runtime.h`
- `ds4_v100_tp_runtime.cu`
- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`
- `deploy/v100/ds4-v100-appliance.env.example`

Changes:

1. Add `allocate_comp_state` to `ds4_v100_tp_runtime_config`.
2. Keep `allocate_comp_state=1` in `ds4_v100_tp_runtime_default_config`.
3. When disabled, skip `cudaMalloc(gpu_state.comp_state)` and report
   `comp_state_bytes=0`.
4. Add full-layer smoke gate:
   `--tp-runtime-skip-unused-comp-state-gate`.
5. Add launcher/profile plumbing:
   `DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE=1` and
   `tools/ds4-v100-tp-ep-profile.py --skip-tp-runtime-comp-state`.

## Validation

Local:

```text
git diff --check
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
```

V100:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

Run target-shape probes at `32` slots / `256K` / `position=262080`:

1. Direct lazy + HC-current NCCL + skip unused comp-state.
2. HTTP lazy + HC-current NCCL + skip unused comp-state.

Record:

- first token / HTTP 200 count;
- response-0 generated token sequence;
- `tp_ep_all_layer_tp_runtime_shared` comp-state bytes;
- `after_tp_runtime`, `after_hc_controls`, `after_lazy_output_head_close`;
- `nccl_after_lazy_output_head_close` reserve failures;
- generated and continuation decode tok/s.

## Results

Artifacts:

- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/direct-lazy-hc-nccl-skip-comp-state/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-lazy-hc-nccl-skip-comp-state/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-lazy-hc-nccl-skip-comp-state-sampled/`
- `logs/from-cluster/sprint409-skip-unused-tp-comp-state/http-readiness-sampled.json`

V100 build passed at `sm_70`.

| Case | Result | First token | Decode tok/s | Continuation tok/s | TP runtime comp-state | NCCL post-close free | NCCL reserve |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct lazy + HC-current NCCL + skip comp-state | returncode 0 | 54639 | 95.402649 | 106.596995 | 0 B/GPU | 2242 MiB | pass, 0 failures |
| HTTP lazy + HC-current NCCL + skip comp-state | 32/32 HTTP 200 | 83480 | 113.117381 | 114.092661 | 0 B/GPU | 2240 MiB | pass, 0 failures |
| HTTP sampled repeat | 32/32 HTTP 200 | 83480 | 114.199600 | 113.663353 | 0 B/GPU | 2240 MiB | pass, 0 failures |

HTTP response 0 preserved generated token sequence `[83480, 79768]`.

The sampled repeat passed `tools/ds4-v100-http-readiness-check.py` with
`32/32` responses, resident KV metadata, typed KV metadata, compact MoE,
checksums, GPU samples, `vram_failures=0`, and `2106 MiB` minimum free VRAM.

Skipping the unused TP-runtime comp-state arena moved:

- `comp_state_bytes_per_gpu`: `1803550720` -> `0`
- `after_tp_runtime` min free: `22720 MiB` -> `24440 MiB`
- `after_hc_controls` min free: `1248 MiB` -> `2968 MiB`
- `after_lazy_output_head_close` min free: `520-522 MiB` -> `2240-2242 MiB`

## Definition of Done

- Gate is implemented and disableable.
- Local checks pass.
- V100 build passes.
- Direct target-shape NCCL run passes correctness and records reserve.
- HTTP target-shape NCCL run passes readiness and records reserve.
- Sprint doc, status, vision, temporary report, and artifacts are committed.

## Decision Gate

Promote the skip gate only if target-shape HTTP serving preserves selected
tokens and passes the `1536 MiB` NCCL reserve with meaningful margin. If the
reserve still fails, keep the gate diagnostic and move next to per-layer
RankState mirror residency.

## Decision

Promote skipping the unused TP-runtime comp-state arena as the launcher/profile
default. Keep the low-level binary behavior gate-controlled so diagnostics can
still allocate the old arena if needed.

HC-current NCCL is now memory-admitted at the target `32` slot / `256K` shape.
The next NCCL work should move from admission to throughput: measure the
serving NCCL path with Nsight/gpu samples and decide whether the HC-current
allgather boundary should become a default or be replaced by a broader TP/EP
collective.
