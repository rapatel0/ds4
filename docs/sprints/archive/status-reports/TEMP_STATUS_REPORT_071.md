# TEMP Status Report 071 - Sprint 359 Pool-Norm Promotion

Date: 2026-05-25

## Current Focus

TP/EP serving performance. Sprint 359 resolved whether fused compressed
pool+norm should remain diagnostic or become a serving default.

## Direct Multi-Step A/B

Shape:

```text
run mode: direct-token-major
slots: 32
context: 256K
position: 262112
decode steps: 32
HC current stream sync: on
```

| Variant | Decode tok/s | Wall tok/s | Compressed-KV sum ms | Pre-EP compressed-KV ms | First token | Finite bad |
|---|---:|---:|---:|---:|---:|---:|
| control | 95.851552 | 74.814127 | 3521.094409 | 3533.823377 | 98751 | 0 |
| pool-norm | 97.619138 | 76.140370 | 3458.469603 | 3470.514540 | 98751 | 0 |

## Decision

Promote fused compressed pool+norm as the TP/EP serving default.

Changed:

- `tools/ds4-v100-run-appliance.sh`
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_FUSED_POOL_NORM` now defaults to
    `1`.
- `deploy/v100/ds4-v100-appliance.env.example`
  - documents the same default.

Still default-off:

- fused input-fill
- fused RoPE+round

## Validation

- V100 direct control returned `rc=0`.
- V100 direct pool-norm returned `rc=0`.
- Both runs preserved first token `98751`.
- Both runs had `output_head_finite_bad=0`.
- Local checks:
  - `python3 -m py_compile tools/ds4-v100-tp-ep-profile.py`
  - `bash -n tools/ds4-v100-run-appliance.sh`
  - `git diff --check`
- Local launcher `--print-command` proof includes:

```text
--true-ds4-compressed-kv-fused-pool-norm-gate
```

## Next Best Step

With one small default win promoted, the next useful work should be either:

1. rerun full HTTP chat/selected-token topline with the promoted default, or
2. implement the next deeper compressed state/emit fusion to keep reducing the
   true-attention/compressed-KV prefix.

Artifacts:

```text
logs/from-cluster/sprint359-direct-pool-norm-multistep/
```
