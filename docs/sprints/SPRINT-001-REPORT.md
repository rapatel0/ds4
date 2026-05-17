---
sprint: 001
title: Baseline DS4 V100 Appliance Planner And Source Inventory
status: ship
date: 2026-05-17
architecture: ../architecture/DS4-V100-LAYOUT.md
planner: ../../tools/ds4-v100-plan.c
---

# SPRINT-001 Report

## Verdict

`SHIP`: the exact source model was identified, the tensor inventory was
captured, the architecture document was reconciled against the actual GGUF, and
the static V100 planner shows the baseline layer-sharded appliance layout fits
inside 8x 32 GiB V100 VRAM with a 4 GiB reserve per GPU.

This sprint did not attempt full decode. That remains correctly deferred until
loader/type-table support, pack manifest generation, and per-device CUDA
ownership are implemented.

## Source Model

| Field | Value |
|---|---|
| Path | `/models/DSv4-Flash-256e-fixed.gguf` |
| Size | `156148189504` bytes, 145.42 GiB |
| SHA-256 | `4fc794b38e9767260228fec42e4d4572e60b9c6c9df99c14f81d8e76c32f1599` |
| GGUF | v3, 45 metadata keys, 1328 tensors |
| Tensor bytes described | 145.42 GiB |
| Inventory TSV | `docs/sprints/drafts/SPRINT-001-TENSOR-INVENTORY.tsv` |
| Inventory summary | `docs/sprints/drafts/SPRINT-001-INVENTORY.txt` |

## Source Tensor Inventory

| GGML Type ID | Type | Count | Bytes |
|---:|---|---:|---:|
| 0 | F32 | 684 | 0.30 GiB |
| 26 | I32 | 3 | 0.01 GiB |
| 30 | BF16 | 147 | 2.55 GiB |
| 39 | MXFP4 | 129 | 137.06 GiB |
| 42 | F8_E4M3_B128 | 365 | 5.50 GiB |

| Tensor Family | Count | Bytes |
|---|---:|---:|
| global | 2 | 0.99 GiB |
| control | 43 | 0.00 GiB |
| HC | 261 | 0.13 GiB |
| attention | 387 | 4.32 GiB |
| compressor | 164 | 0.49 GiB |
| indexer | 126 | 0.26 GiB |
| router | 86 | 0.18 GiB |
| routed expert | 129 | 137.06 GiB |
| shared expert | 129 | 1.02 GiB |
| output head | 1 | 0.99 GiB |

No unsupported GGML type IDs were found in the target source model.

## Architecture Deltas

Planning baseline:

```text
docs/architecture/DS4-V100-LAYOUT.md
sha256: 27e4b012643c9bfc664e54f294deb3d0e93e0968205975d07d838fc13e75f519
```

Inventory confirmed the broad architecture and corrected specific source
details. The architecture document was updated for:

- BF16 token embedding and BF16 output head.
- F32 HC control tensors, including `hc_attn_fn`, `hc_ffn_fn`, and `hc_head_fn`.
- `attn_kv_latent.weight` as the source KV projection name.
- `attn_compress_*` and `indexer.compress_*` source names.
- BF16 compressor/indexer KV and gate tensors.
- F8 `indexer.attn_q_b.weight`.
- F32 `ffn_gate_inp.weight`, `ffn_norm.weight`, and `exp_probs_b`.
- Shared experts as F8_E4M3_B128.
- Output head first runtime layout as BF16 source-faithful, not FP8/Q8 by default.

These are implementation deltas, not feasibility blockers.

## Planner Output

Planner artifact:

```bash
make tools/ds4-v100-plan
./tools/ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off
```

Saved outputs:

- `docs/sprints/drafts/SPRINT-001-PLAN-256K-S4.txt`
- `docs/sprints/drafts/SPRINT-001-PLAN-1M-S1-MTP.txt`

Baseline layer map:

| GPU | Layers | Layer Mix | Est. Weights |
|---:|---|---|---:|
| gpu0 | 0-5 | 2 SWA, 2 ratio-4, 2 ratio-128 | 19.99 GiB |
| gpu1 | 6-11 | 3 ratio-4, 3 ratio-128 | 20.02 GiB |
| gpu2 | 12-17 | 3 ratio-4, 3 ratio-128 | 20.02 GiB |
| gpu3 | 18-23 | 3 ratio-4, 3 ratio-128 | 20.02 GiB |
| gpu4 | 24-29 | 3 ratio-4, 3 ratio-128 | 20.02 GiB |
| gpu5 | 30-34 | 3 ratio-4, 2 ratio-128 | 16.69 GiB |
| gpu6 | 35-39 | 2 ratio-4, 3 ratio-128 | 16.67 GiB |
| gpu7 | 40-42 | 2 ratio-4, 1 ratio-128 | 10.02 GiB |

Configured plan for 4 slots at 256K context, MTP off, F16 KV, 1 GiB scratch/GPU,
and 4 GiB reserve/GPU:

| GPU | Weights | KV | Comp State | Scratch | Globals | Reserve | Planned Total | Headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu0 | 19.99 | 0.64 | 0.10 | 1.00 | 0.99 | 4.00 | 26.72 | 5.28 |
| gpu1 | 20.02 | 0.96 | 0.15 | 1.00 | 0.00 | 4.00 | 26.13 | 5.87 |
| gpu2 | 20.02 | 0.96 | 0.15 | 1.00 | 0.00 | 4.00 | 26.13 | 5.87 |
| gpu3 | 20.02 | 0.96 | 0.15 | 1.00 | 0.00 | 4.00 | 26.13 | 5.87 |
| gpu4 | 20.02 | 0.96 | 0.15 | 1.00 | 0.00 | 4.00 | 26.13 | 5.87 |
| gpu5 | 16.69 | 0.96 | 0.12 | 1.00 | 0.00 | 4.00 | 22.77 | 9.23 |
| gpu6 | 16.67 | 0.65 | 0.12 | 1.00 | 0.00 | 4.00 | 22.44 | 9.56 |
| gpu7 | 10.02 | 0.63 | 0.07 | 1.00 | 0.99 | 4.00 | 16.71 | 15.29 |

Memory-only slot admission with F16 KV, 1 GiB scratch/GPU, and 4 GiB reserve/GPU:

| Context | Max Admitted Slots | Worst-GPU Planned Total At Max |
|---:|---:|---:|
| 128K | 50 | 31.98 GiB |
| 256K | 25 | 31.96 GiB |
| 512K | 12 | 31.67 GiB |
| 1M | 6 | 31.67 GiB |

These are memory-admission numbers only. They do not claim throughput or
latency feasibility. Active microbatch, bandwidth, and scheduler behavior still
need runtime validation.

The 1M single-slot, MTP-on diagnostic remains comfortable:

```text
./tools/ds4-v100-plan --ctx 1048576 --slots 1 --gpus 8 --mtp on
Worst configured GPU total including reserve: 26.72 GiB / 32.00 GiB
gpu7 includes about 3.60 GiB rough MTP bytes and still plans at 20.31 GiB.
```

## Format And Kernel Policy

Confirmed first policy:

| Tensor Family | Source Dtype | First Runtime Layout | First Kernel Family |
|---|---|---|---|
| attention dense Q/KV/output | F8_E4M3_B128 | source FP8 blocked pack | FP8 dequant plus FP16 HMMA dense |
| routed experts | MXFP4 | source MXFP4 grouped pack | TurboMind sm70 grouped MXFP4 or owned grouped low-bit |
| shared expert | F8_E4M3_B128 | source FP8 dense pack | safe dense/shared-expert path |
| router/norms/HC/control | F32/BF16/I32 | source-faithful small tensors | DS4 control kernels |
| output head | BF16 | BF16 source-faithful first | BF16/F16 output projection, vocab TP later |
| KV cache | cache state | F16 first | DS4 compressed KV/attention kernels |

Blanket INT8 routed experts are not viable as the default layout. The planner
shows that expanding all routed expert tensors to INT8 would push the worst GPU
to about 43.59 GiB for the 4-slot 256K configured plan. INT8 remains a
per-family candidate only where a pack-specific memory and quality gate passes.

## Pack Manifest Contract

Sprint 002 should use this manifest shape:

```text
semantic_tensor_id
source_name
source_dtype
source_shape
runtime_layout
owning_gpu
layer_id
kernel_family
byte_offset
byte_length
scale_offset
checksum
```

The planner currently uses static DS4 constants. It is structured so a later
revision can consume this manifest or the inventory TSV.

## Cluster Commands

The V100 pod was created using the documented handoff:

```bash
kubectl apply -f /Users/ravi/repos/deepseek/manifests/llamacpp-build-8gpu.yaml
kubectl -n llm wait --for=condition=Ready pod/llamacpp-build-8gpu --timeout=10m
kubectl -n llm exec llamacpp-build-8gpu -- nvidia-smi -L
```

The pod reported 8x `Tesla V100-SXM2-32GB`.

Inventory command:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
MODEL=/models/DSv4-Flash-256e-fixed.gguf
stat -c "model_size_bytes=%s model_path=%n" "$MODEL"
/workspace/ds4-tools/ds4-v100-plan \
  --ctx 262144 --slots 4 --gpus 8 --mtp off \
  --inventory "$MODEL" \
  --inventory-tsv /workspace/ds4-tools/SPRINT-001-TENSOR-INVENTORY.tsv
sha256sum "$MODEL"
'
```

## Validation

Passed:

```bash
make tools/ds4-v100-plan
./tools/ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off
./tools/ds4-v100-plan --ctx 1048576 --slots 1 --gpus 8 --mtp on
make cpu
```

Cluster validation passed:

```bash
cc -O3 -Wall -Wextra -std=c99 -D_FILE_OFFSET_BITS=64 \
  -o /workspace/ds4-tools/ds4-v100-plan \
  /workspace/ds4-tools/ds4-v100-plan.c
/workspace/ds4-tools/ds4-v100-plan --inventory /models/DSv4-Flash-256e-fixed.gguf
sha256sum /models/DSv4-Flash-256e-fixed.gguf
```

Validation gap:

```bash
make test
```

`make test` compiled `ds4_test` but failed at runtime because local
`ds4flash.gguf` is not present:

```text
ds4: cannot open model 'ds4flash.gguf': No such file or directory
```

This is an environment/model availability gap, not a planner build failure.

## Next Implementation Surface

Recommended Sprint 002 order:

1. Loader/type-table support for GGML type IDs 39 and 42, plus BF16 output and
   the measured source tensor names.
2. Inventory-backed pack manifest generation.
3. Per-device CUDA ownership metadata and layer-device plan plumbing.
4. First execution target: routed expert MXFP4 grouped path, because it
   dominates bytes and already has V100 kernel evidence.
5. Dense FP8 attention/shared/output-head paths after source layout loading is
   strict and testable.

Minimal loader deltas:

- Accept `GGML_TYPE_MXFP4 = 39`.
- Accept `GGML_TYPE_F8_E4M3_B128 = 42`.
- Accept BF16 global/output tensors.
- Map `attn_kv_latent.weight`.
- Map `attn_compress_*` and `indexer.compress_*`.
- Keep fixed DS4 shape validation strict.
- Do not introduce arbitrary GGUF loading.

## Definition Of Done

- [x] `DS4-V100-LAYOUT.md` is explicitly cited in this report as the planning baseline.
- [x] The target source model is identified by path, size, and SHA-256.
- [x] Tensor inventory includes names, dimensions, GGML type IDs, source dtype,
      grouped tensor family, and byte estimates.
- [x] Inventory is reconciled against `DS4-V100-LAYOUT.md`; architecture deltas
      are documented and patched.
- [x] Planner prints the baseline 8-GPU layer map from the architecture doc.
- [x] Planner prints per-GPU weight, KV, scratch, relay, global, reserve, and
      headroom estimates.
- [x] Planner reports admitted slots for 128K, 256K, 512K, and 1M context tiers.
- [x] Planner distinguishes source dtype from runtime layout and marks INT8 as
      candidate, not assumed default.
- [x] Planner rejects configurations that overfill 32 GiB V100 VRAM after reserve.
- [x] Report records whether the baseline closes `SHIP`, `EXTEND`, or `STOP`.
- [x] Next-sprint implementation surface is concrete enough to begin coding
      without reopening the topology/dtype discussion.
