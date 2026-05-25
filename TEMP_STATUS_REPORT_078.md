# TEMP Status Report 078 - Sprint 366

Date: 2026-05-25

## Current Focus

TP/EP-only long-context serving optimization. Sprint 366 targeted the
compressed-KV projection boundary after Sprint 365 showed input-fill
micro-fusion was not a serving-visible default.

## Implemented

- Added `--true-ds4-compressed-kv-dense-event-wait-gate`.
- Added launcher env:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=1`.
- Added profile harness flags:
  `--compressed-dense-event-wait` and
  `--disable-compressed-dense-event-wait`.
- Promoted the gate to launcher default while keeping it disableable.

## Mechanism

The path no longer synchronizes each rank's local stream on the host after
compressed attention/indexer input-fill kernels. It records a per-rank CUDA
event and makes that rank's dense stream wait on it. Downstream dense
completion, gather, state/emit, typed-KV, and output semantics are unchanged.

## Results

Direct 32-step A/B, `32` slots / `256K` / `position=262112`:

| Variant | First token | Bad | Decode tok/s | Wall tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| control | 98751 | 0 | 96.214306 | 75.215206 | 3431.137744 ms |
| dense event wait | 98751 | 0 | 99.093248 | 76.897975 | 3127.236790 ms |

Selected-token HTTP A/B, same shape:

| Variant | HTTP 200 | First token | Bad | Client tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|
| control | 32/32 | 109328 | 0 | 71.833757 | 3437.636456 ms |
| dense event wait | 32/32 | 109328 | 0 | 74.432464 | 3137.755187 ms |

Default proof, `1` selected-token request at `position=262143`:

| Variant | HTTP 200 | Event rows | Fused pool rows |
|---|---:|---:|---:|
| default | 1/1 | 39 | 39 |
| explicit disable | 1/1 | 0 | 39 |

## Validation

Local checks passed:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

V100 build passed:

```text
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Decision

Promote compressed dense event waits. This is a real serving-visible win at
the target 32-slot / 256K long-context shape, and the control switch remains
available for future A/B runs.

## Artifacts

```text
logs/from-cluster/sprint366-compressed-dense-event-wait/
logs/from-cluster/sprint366-compressed-dense-event-wait-http/
logs/from-cluster/sprint366-compressed-dense-event-wait-default-proof/
```
