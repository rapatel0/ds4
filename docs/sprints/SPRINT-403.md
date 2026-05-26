# Sprint 403: NCCL With Quantized KV Matrix

Date: 2026-05-26

## Overview

Sprints 400-402 established the current NCCL state:

- NCCL collectives are faster in synthetic TP8 boundary workbenches.
- Narrow serving-path NCCL gates are correct at smaller shapes.
- At the production target `32` slots / `256K`, narrow NCCL gates fall below
  the new `1536 MiB` NCCL VRAM reserve after communicator allocation.

The next useful NCCL question is whether the existing quantized KV path buys
back enough memory to admit NCCL at the target shape without changing model
quality. Sprint 403 creates a reusable matrix harness and runs the direct V100
matrix for control, FP8 KV, HC-current NCCL, and FP8 KV + HC-current NCCL.

## Constraints

- TP/EP only. No PP/layer-split work.
- Do not promote any default without target-shape evidence.
- Keep NCCL gates default-off.
- Preserve first-token correctness as the minimum direct-run correctness gate.
- Use `32` slots and `256K` context as the target shape.

## Implementation

Add:

- `tools/ds4-v100-tp-ep-nccl-kv-matrix.py`

The harness runs `tools/ds4-v100-tp-ep-profile.py` for a fixed set of
direct-token-major cases:

| Case | Flags |
|---|---|
| `control` | real-router compact-MoE, no FP8 KV, no NCCL |
| `fp8-kv` | `--fp8-e5m2-kv` |
| `hc-nccl` | `--hc-current-nccl-allgather` |
| `fp8-kv-hc-nccl` | `--fp8-e5m2-kv --hc-current-nccl-allgather` |

Each case uses:

```text
--ctx 262144
--slots 32
--position 262080
--requests 32
--max-requests 80
--tokens 2
--model-router-routes
--compact-moe-decode
--hc-current-stream-sync
--vram-report
--vram-min-free-mib 64
--nccl-min-free-mib 1536
```

The harness writes:

- per-case command/stdout/stderr artifacts
- profile harness artifacts
- `matrix-summary.json`
- `matrix-summary.md`

## Validation

Local checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-nccl-kv-matrix.py
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100 build:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

V100 run:

```text
python3 tools/ds4-v100-tp-ep-nccl-kv-matrix.py \
  --artifact-dir /workspace/logs/sprint403-nccl-kv-matrix \
  --ctx 262144 \
  --slots 32 \
  --position 262080 \
  --requests 32 \
  --max-requests 80 \
  --tokens 2
```

## Results

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-nccl-kv-matrix.py
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
git diff --check
```

V100 build/preconditions passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-nccl-kv-matrix.py
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

The matrix ran at `32` slots / `256K` / `position=262080` / `2` generated
tokens:

| Case | Return | First token | Generated decode tok/s | Continuation decode tok/s | Min free VRAM | NCCL threshold | NCCL failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Control | `0` | `54639` | `98.076858` | `107.106917` | `1746 MiB` | n/a | `0` |
| FP8 E5M2 KV | `0` | `54639` | `93.927351` | `103.344304` | `1746 MiB` | n/a | `0` |
| HC-current NCCL | `14` | n/a | n/a | n/a | `1114 MiB` | `1536 MiB` | `5` |
| FP8 E5M2 KV + HC-current NCCL | `14` | n/a | n/a | n/a | `1114 MiB` | `1536 MiB` | `5` |

Artifacts:

- `logs/from-cluster/sprint403-nccl-kv-matrix/matrix-summary.json`
- `logs/from-cluster/sprint403-nccl-kv-matrix/matrix-summary.md`
- `logs/from-cluster/sprint403-nccl-kv-matrix/control/`
- `logs/from-cluster/sprint403-nccl-kv-matrix/fp8-kv/`
- `logs/from-cluster/sprint403-nccl-kv-matrix/hc-nccl/`
- `logs/from-cluster/sprint403-nccl-kv-matrix/fp8-kv-hc-nccl/`

## Interpretation

`--fp8-e5m2-kv` is not an F16-to-FP8 memory reclamation switch in the current
TP/EP path. The default TP runtime config already uses FP8 E4M3 block-128 KV;
the gate changes the FP8 flavor to E5M2. That explains why both FP8 E5M2 cases
had the same minimum free VRAM as their matching non-E5M2 cases.

The combined FP8 E5M2 + HC-current NCCL target still fails at
`nccl_after_output_head` with the same `1114 MiB` free against the `1536 MiB`
NCCL reserve. So quantized-KV flavor switching does not make narrow NCCL
serving gates viable at `32` slots / `256K`.

## Decision

Do not promote `--fp8-e5m2-kv` or the HC-current NCCL allgather for the target
shape. Keep both diagnostic-only.

The next NCCL work should not assume additional KV quantization headroom from
`--fp8-e5m2-kv`. To make NCCL viable at `32` slots / `256K`, we need either:

- remove/reschedule at least about `422 MiB/GPU` of resident allocations before
  the NCCL post-output-head checkpoint, or
- replace a broader peer-copy boundary so NCCL's communicator overhead is paid
  for a material topology change, not for a single narrow gather.

## Definition of Done

- Matrix harness exists and is reusable.
- Local syntax/checks pass.
- V100 build passes.
- Matrix artifacts are pulled into `logs/from-cluster/sprint403-nccl-kv-matrix`.
- Sprint results answer whether FP8 KV admits HC-current NCCL at
  `32` slots / `256K`.
- `STATUS.md`, `VISION.md`, and a temporary status report are updated.
- Commit all kept artifacts explicitly.

## Decision Gate

Promote nothing unless the `fp8-kv-hc-nccl` target case:

1. returns `0`,
2. preserves the control first token,
3. satisfies the `1536 MiB` NCCL VRAM reserve,
4. improves generated or continuation decode throughput versus the matching
   FP8 KV control.

If it is admitted but slower, keep NCCL diagnostic-only and use the result to
guide broader NCCL boundary design. If it is not admitted, the next NCCL work
must reduce resident memory before adding more collectives.
