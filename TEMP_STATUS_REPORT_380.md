# TEMP Status Report 380: TP-Sharded Expert A/B

Date: 2026-05-25

## Current Focus

Sprint 380 starts S-F from the Vision: `--tp-experts-ab-gate`.

The goal is to measure whether TP-sharded routed experts are worth integrating
into the TP/EP serving path, not to rewrite serving topology yet.

## Implemented

- Added `docs/sprints/SPRINT-380.md`.
- Added permanent measurement driver:

```text
tools/ds4-v100-tp-experts-ab.py
```

The driver writes:

```text
summary.json
summary.tsv
ep8-direct/
tp8-turbomind/
```

It can run:

- EP8 direct serving control through `tools/ds4-v100-tp-ep-profile.py`
- TP8 TurboMind MXFP4 expert workbench through
  `tools/ds4-v100-tp8-turbomind-ffn-smoke`

Local validation:

```text
python3 -m py_compile tools/ds4-v100-tp-experts-ab.py
git diff --check
```

V100 build validation:

```text
make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 \
  tools/ds4-v100-tp8-turbomind-ffn-smoke \
  tools/ds4-v100-tp-ep-full-layer-smoke
```

## V100 Smoke Evidence

Artifact root:

```text
/workspace/logs/sprint380-tp-experts-ab
```

### Driver Smoke

Command shape:

```text
32 slots
256K context
position 262080
1 generated token
model-router routes
compact-MoE decode
tokens_per_active=16
warmup=1
iters=5
```

Result:

| Path | Return | Key metric |
|---|---:|---|
| EP8 direct serving | 0 | first token `54639`, direct decode `66.569095` tok/s, EP `18.220610` ms, compose `22.522762` ms |
| TP8 workbench | 1 | correctness `FAIL`, routes `96`, NaNs `378153`, TP8 compute `0.087654` ms, TP8 total `0.581414` ms |

### TP4/TP8 Matrix

Command shape:

```text
skip EP8
tokens_per_active=16,32,64
warmup=5
tokens_per_active=16,32,64
warmup=5
iters=50
```

Artifact:

```text
/workspace/logs/sprint380-tp-experts-ab/tp4-tp8-matrix-parsed
```

| Path | Tokens/active | Routes | Correctness | NaNs | Full ms | TP compute ms | TP reduce ms | TP total ms | Total speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| TP4 | 16 | 96 | PASS | 0 | 0.292434 | 0.123945 | 0.153265 | 0.277210 | 1.055x |
| TP4 | 32 | 192 | PASS | 0 | 0.331469 | 0.132833 | 0.239310 | 0.372143 | 0.891x |
| TP4 | 64 | 384 | PASS | 0 | 0.575406 | 0.161587 | 0.459294 | 0.620881 | 0.927x |
| TP8 | 16 | 96 | FAIL | 378153 | 0.250614 | 0.071660 | 0.482224 | 0.553884 | 0.452x |
| TP8 | 32 | 192 | FAIL | 756305 | 0.295444 | 0.080056 | 0.888402 | 0.968458 | 0.305x |
| TP8 | 64 | 384 | FAIL | 1512469 | 0.604365 | 0.141844 | 1.664169 | 1.806014 | 0.335x |

This reconfirms and sharpens the Sprint 211 conclusion with current kernels:

- TP8 MXFP4 compute is fast, but `mid_shard=256` remains numerically invalid
  and total speedup is below `1.0x`.
- TP4 MXFP4 is numerically valid and has real compute speedup, but the simple
  output reduction erases most of the win. It only beats the full reference at
  the smallest `96` route tier, not at `192` or `384` routes.

## Current Interpretation

TP8 expert integration should not proceed with the current TurboMind shard
shape.

Do not integrate TP8.

Do not integrate TP4 into serving with the current reduction boundary. TP4 is
the only viable TP expert branch numerically, but it needs a better
reduce/compose boundary before it can plausibly improve the EP8 serving path.

Sprint 380 can close as a topology measurement. If TP experts are revisited,
the next sprint should not be another workbench rerun; it should prototype a
fused TP4 reduction/compose boundary and compare against EP8 direct serving.
