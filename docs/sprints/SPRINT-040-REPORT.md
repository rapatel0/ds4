# Sprint 040 Report: Resident One-Token MTP Forward Composition

## Summary

Sprint 040 shipped a resident gpu7 MTP forward composition smoke and wired it
into the V100 appliance gate.

The new smoke composes the previously validated MTP primitives into one
deterministic one-token path:

1. MTP prefix from deterministic embedding plus previous HC.
2. Integrated MTP attention with raw-cache store and grouped output.
3. MTP FFN with bias router, Q4_K routed experts, and Q8_0 shared expert.
4. MTP HC-head collapse, output norm, base BF16 output projection, and top-k.

The full V100 gate now includes `mtp_forward PASS` and advances readiness to
`missing=mtp_verify`.

## Code Changes

- Added `tools/ds4-v100-mtp-forward-smoke.c`.
- Added Makefile build/clean rules for `tools/ds4-v100-mtp-forward-smoke`.
- Added `mtp_forward` build/run/readiness wiring to
  `tools/ds4-v100-gate.sh`.

## Validation

Local compile:

```bash
make tools/ds4-v100-mtp-forward-smoke.o ds4_v100_mtp.o ds4_v100_context.o ds4_gpu_arena_stub.o ds4_cpu.o ds4_source_formats.o
```

Cluster focused build:

```bash
CUDA_ARCH=sm_70 make tools/ds4-v100-mtp-forward-smoke
```

Focused V100 smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 ./tools/ds4-v100-mtp-forward-smoke \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-040-MTP-FORWARD/mtp_forward.report
```

Result:

```text
mtp_forward_smoke: cpu_top1=101365 gpu_top1=101365 boundary_max_abs=0.959003448 logit_max_abs=0.0884904861 PASS
```

Full gate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-040-GATE-CLUSTER-8GPU
```

Gate result:

```text
gate mtp_forward PASS
gate readiness NOT_READY missing=mtp_verify
gate summary PASS failures=0 ready=false
```

Post-run GPU memory returned to 0 MiB on all eight V100s.

## Focused Smoke Evidence

From `docs/sprints/drafts/SPRINT-040-MTP-FORWARD/mtp_forward.report`:

| Check | Max Abs | Tolerance | Result |
|---|---:|---:|---|
| `prefix_hc` | 0.000127315521 | 0.05 | PASS |
| `attn_next_hc` | 0.327054977 | 0.75 | PASS |
| `ffn_next_hc` | 0.959003448 | 1.25 | PASS |
| selected logits | 0.0884904861 | 0.10 | PASS |

Top-k candidates matched exactly:

| Rank | CPU Token | GPU Token | CPU Logit | GPU Logit | Delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 101365 | 101365 | 15.6609735 | 15.7485046 | 0.0875310898 |
| 2 | 40810 | 40810 | 15.6245451 | 15.7130356 | 0.0884904861 |
| 3 | 102216 | 102216 | 15.5625267 | 15.5519838 | 0.0105428696 |
| 4 | 7178 | 7178 | 15.0537891 | 15.0295925 | 0.0241966248 |
| 5 | 112542 | 112542 | 14.92519 | 15.0134554 | 0.088265419 |

Memory evidence:

- MTP sidecar arena: `3,807,601,408` bytes.
- MTP sidecar uploaded bytes: `3,807,600,108`.
- Base `output.weight`: `1,059,061,760` bytes.
- Free after MTP upload: `29,937,369,088` bytes.
- Free after output-head upload: `28,878,307,328` bytes.
- Required reserve: `4,294,967,296` bytes.

## Remaining Blocker

The next readiness rung is `mtp_verify`: prove that a native prompt-token MTP
draft can be verified against the target model and that accept/reject/rollback
does not corrupt target-model or MTP raw-cache state.
