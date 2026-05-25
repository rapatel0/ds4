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

### TP8 Matrix

Command shape:

```text
skip EP8
tokens_per_active=16,32,64
warmup=5
iters=50
```

| Tokens/active | Routes | Correctness | NaNs | Full ms | TP8 compute ms | TP8 reduce ms | TP8 total ms | Total speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | 96 | FAIL | 378153 | 0.293499 | 0.071660 | 0.489438 | 0.561097 | 0.523x |
| 32 | 192 | FAIL | 756305 | 0.347197 | 0.080814 | 0.903803 | 0.984617 | 0.353x |
| 64 | 384 | FAIL | 1512469 | 0.606597 | 0.141926 | 1.668078 | 1.810004 | 0.335x |

This reconfirms the old TP8 conclusion with current kernels: TP8 MXFP4 compute
is fast, but `mid_shard=256` remains numerically invalid and simple output
reduction erases total speedup.

## Current Interpretation

TP8 expert integration should not proceed with the current TurboMind shard
shape.

Sprint 380 is not complete yet because TP4 remains the plausible branch from
Sprint 211. The next work is to expose/rerun TP4 in the same driver or record
why TP4 cannot be rerun with current tools.
