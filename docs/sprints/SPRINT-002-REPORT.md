# SPRINT-002 Report: Source Loader And Pack Manifest Baseline

Date: 2026-05-17

## Verdict

`EXTEND`: the native DS4-Flash source GGUF can now be recognized, validated, and
manifested, but runtime decode remains intentionally blocked until the V100
FP8/MXFP4 execution kernels are wired.

## Implemented

- Added GGUF tensor type support for native source dtypes:
  - `GGML_TYPE_MXFP4 = 39`
  - `GGML_TYPE_F8_E4M3_B128 = 42`
  - `BF16 = 30`
- Added measured source tensor-name binding for:
  - `hc_head_*`
  - `blk.N.attn_kv_latent.weight`
  - `blk.N.attn_compress_*`
  - `blk.N.indexer.compress_*`
  - source HC tensors without `.weight`
  - source router tensors `exp_probs_b` and `ffn_gate_tid2eid`
- Added a source-specific metadata validator for the high-intelligence
  `/models/DSv4-Flash-256e-fixed.gguf` header. The source GGUF does not carry
  the older converted-model keys such as `attention.output_lora_rank`,
  `hyper_connection.*`, `attention.compress_ratios`, or RoPE scaling metadata.
- Added an inspect-only guard: source-native GGUF loads for `--inspect`, but
  generation exits before inference with a clear message because the V100
  source-format kernels are not connected yet.
- Extended `tools/ds4-v100-plan` with:
  - absolute GGUF tensor offsets
  - `--manifest FILE`
  - per-tensor owner GPU
  - source dtype, runtime layout, kernel family, byte length, and checksum
    placeholder fields

## Artifacts

- Pack manifest:
  `docs/sprints/drafts/SPRINT-002-PACK-MANIFEST.tsv`
- Planner/inventory log:
  `docs/sprints/drafts/SPRINT-002-PLAN-MANIFEST.log`

The manifest has 1328 tensor rows plus a header. It records absolute offsets
from the GGUF file start and labels those offsets as `absolute_gguf_file`.

## Validation

Local:

```bash
make cpu
cc -O2 -Wall -Wextra -std=c99 -o /tmp/ds4-v100-plan tools/ds4-v100-plan.c
./tools/ds4-v100-plan --ctx 262144 --slots 4 --gpus 8 --mtp off
```

Cluster on `llamacpp-build-8gpu`:

```bash
make cpu
./ds4 --inspect --cpu -m /models/DSv4-Flash-256e-fixed.gguf
./ds4 --cpu -m /models/DSv4-Flash-256e-fixed.gguf -p test -n 1
/tmp/ds4-v100-plan \
  --ctx 262144 \
  --slots 4 \
  --gpus 8 \
  --mtp off \
  --inventory /models/DSv4-Flash-256e-fixed.gguf \
  --inventory-tsv /tmp/ds4-source-inventory.tsv \
  --manifest /tmp/ds4-pack-manifest.tsv
```

Results:

- Inspect succeeds and reports 1328 tensors:
  - F32: 684 tensors, 0.30 GiB
  - I32: 3 tensors, 0.01 GiB
  - BF16: 147 tensors, 2.55 GiB
  - MXFP4: 129 tensors, 137.06 GiB
  - F8_E4M3_B128: 365 tensors, 5.50 GiB
- Generation fails fast by design with:
  `native DS4-Flash source layout is recognized, but V100 FP8/MXFP4 execution kernels are not wired into runtime yet`
- Manifest generation succeeds with 1329 TSV lines.

## Not Done

- No V100 CUDA decode path has been implemented for source FP8/MXFP4 tensors.
- No MTP path has been enabled for the source model.
- No correctness, first-token, or throughput benchmark exists yet for runtime
  decode, because runtime decode is still blocked.
- No per-GPU packed shard files are emitted yet; this sprint emits the manifest
  contract those shards should consume.

