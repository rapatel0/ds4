---
sprint: 007
title: Source-Layout Single-Slot Decode Oracle
status: completed
date: 2026-05-18
verdict: SHIP
---

# SPRINT-007 Report: Source-Layout Single-Slot Decode Oracle

## Verdict

`SHIP`

Sprint 007 produced the guarded source-layout oracle needed before production
V100 kernels are trusted. The source model still fails closed for normal
generation, while the diagnostic oracle can run a bounded official-vector
comparison and now matches the expected first token exactly.

## What Shipped

- Shared source-format helpers for BF16, F8_E4M3_B128, MXFP4, and related
  scalar formats.
- A model-less source dtype smoke test covering source decode behavior and
  malformed row/span handling.
- Source-aware CPU reference dispatch for the oracle path so source F32,
  F8_E4M3_B128, BF16, and MXFP4 tensors are not silently interpreted through
  legacy F16 or Q8_0 helpers.
- A diagnostic source-layout oracle gate that remains CPU-only and rejects
  normal generation.
- A narrower session unlock for `--dump-logprobs`, allowing official-vector
  evidence without exposing source-layout oracle mode as a normal serving path.
- A correction to MXFP4 row semantics: GGML `block_mxfp4` stores low nibbles in
  positions 0-15 and high nibbles in positions 16-31. The earlier interleaved
  assumption selected the wrong token; the corrected layout selects the
  official expected token.
- A source-oracle KV correction: the source model uses the default F16 KV
  contract. The legacy CPU reference E4M3 KV round-trip remains available for
  older paths, but source-layout oracle runs skip it by default.

## Official Vector Evidence

Fixture: `tests/test-vectors/prompts/short_reasoning_plain.txt`

Prompt:

```text
Answer with only the number: 2048 divided by 128 is
```

Expected first generated token from the official fixture:

```text
text: 16
bytes: 3136
```

Cluster oracle result:

```text
selected id: 926
selected text: 16
selected bytes: [49,54]
top logprob: -0.000390697416
prompt tokens: 18
ctx: 4096
```

Artifact:

- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-oracle-official-short-reasoning.json`

## Validation

Local validation:

```text
make cpu tests/source_dtypes_smoke
./tests/source_dtypes_smoke
git diff --check
```

Cluster validation on `llamacpp-build-8gpu`:

```text
make clean
make cpu tests/source_dtypes_smoke CUDA_ARCH=sm_70
./tests/source_dtypes_smoke
./ds4 --cpu --source-layout-oracle -t 80 \
  -m /models/DSv4-Flash-256e-fixed.gguf \
  -sys "" --nothink \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  -n 1 -c 4096 \
  --dump-logprobs /tmp/ds4-source-oracle-official-short-reasoning-final.json \
  --logprobs-top-k 20
```

Archived artifacts:

- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-cluster-build.log`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-cluster-dtype-smoke.log`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-inspect.log`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-normal-guard.log`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-oracle-generation-guard.log`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-oracle-official-short-reasoning.json`
- `docs/sprints/drafts/SPRINT-007-cluster-logs/SPRINT-007-source-oracle-official-short-reasoning.stderr`

The inspect artifact confirms the source model layout:

```text
model: DeepSeek-V4-Flash-256e-fixed
layers: 43
file size: 145.42 GiB
tensor types:
  f32        684 tensors, 0.30 GiB
  i32          3 tensors, 0.01 GiB
  bf16       147 tensors, 2.55 GiB
  mxfp4      129 tensors, 137.06 GiB
  f8_e4m3_b128   365 tensors, 5.50 GiB
```

## V100 Precision Policy

This sprint does not change the production precision plan:

- BF16 source tensors are not native BF16 compute on V100. They are decoded or
  converted explicitly to FP16 runtime storage or FP16 scratch tiles; production
  dense math should use FP16 tensor cores with FP32 accumulation where
  applicable.
- F8_E4M3_B128 and MXFP4 are source/runtime packed inputs. The CPU oracle
  dequantizes them for correctness; later V100 kernels should minimize casts by
  consuming the packed layout directly or by dequantizing into FP16 HMMA tiles.
- FP32 is acceptable for scalar control, reductions, correctness diagnostics,
  and the CPU oracle. It is not the production default for large GEMMs.
- KV starts as F16 for source-layout correctness. F8 KV is a later optimization
  gate, not the baseline correctness mode.

## Deviations

- The sprint plan described a separate unlock-token field and dedicated oracle
  tool. The shipped implementation uses the existing CLI plus a diagnostic
  `--dump-logprobs` session unlock. Normal generation and normal source-model
  open remain guarded.
- The official-vector proof is first-token selected-token equality, not a
  full-logit parity artifact. Full-logit capture remains deferred.
- The oracle is CPU/host-side and correctness-oriented. It does not prove
  device-side throughput, long-context KV behavior, MTP, or production V100
  kernel scheduling.
- A prior auxiliary first-token log is retained in the draft log directory, but
  the `SHIP` verdict relies on the final official `short_reasoning_plain`
  JSON artifact after the MXFP4 layout fix.

## Remaining Scope

The project is not deployed or performance optimized yet. The next sprint needs
to move from source-layout oracle correctness to the first production-relevant
execution surfaces:

- prompt prefill and DS4 compressed KV/indexer state;
- device-side source-format kernel anchors for F8_E4M3_B128 and MXFP4;
- validation against the Sprint 007 oracle;
- careful memory accounting for long context and slot admission;
- continued fail-closed behavior for normal source serving until correctness is
  proven beyond the one-token oracle.

See `docs/sprints/SPRINT-007-DEFERRED.md` and
`docs/sprints/SPRINT-007-FOLLOWUPS.md`.
