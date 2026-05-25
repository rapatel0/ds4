# Sprint 372: Skip Compressed Dense Host Stats

## Overview

Add and test an opt-in TP/EP serving gate that skips host-side dense-output
statistics in the compressed-KV projection path.

Sprint 371 showed full 32-slot serving is not limited by active request
count. The next target is full-occupancy overhead. In
`run_true_ds4_compressed_kv_projection_gate`, every compressed/indexer dense
projection currently synchronizes the dense stream and copies dense outputs
back to host to compute max/finite stats. That is diagnostic validation work,
not production serving work.

## Scope

- Add a default-off gate:
  `--true-ds4-compressed-kv-skip-dense-stats-gate`.
- Expose it through:
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS`
  - `tools/ds4-v100-run-appliance.sh`
  - `tools/ds4-v100-tp-ep-profile.py`
- Preserve the default validation behavior.
- A/B the gate on V100 at the long-context 32-slot chat shape.

## Definition Of Done

- Local syntax and focused tests pass.
- V100 build passes.
- Direct or HTTP A/B proves correctness at the selected-token/output-head
  level.
- Results show whether skipping host stats is a production candidate.
- Docs/status/artifacts are committed.

## Changes

- Added `--true-ds4-compressed-kv-skip-dense-stats-gate` to the TP/EP
  full-layer smoke/serving binary.
- Exposed the gate through:
  - `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS`
  - `tools/ds4-v100-run-appliance.sh`
  - `tools/ds4-v100-tp-ep-profile.py --skip-compressed-dense-stats`
- Preserved the default validation path. The gate is default-off.
- Added profiler summary counting for `skip_dense_stats` compressed-KV rows.

## Validation

Local validation:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-active-slot-matrix.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
./ds4_test --server
./ds4_test --metal-kernels
```

V100 validation:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
bash -n tools/ds4-v100-run-appliance.sh
make -B -j80 CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Results

Direct token-major, `32` slots / `256K` / `32` decode steps:

| Mode | First token | Scaffold tok/s | Compressed-KV sum | Attn dense | Indexer dense |
|---|---:|---:|---:|---:|---:|
| control | 98751 | 100.739521 | 3141.768079 ms | 782.562774 ms | 1091.318664 ms |
| skip stats | 98751 | 117.463961 | 1789.795027 ms | 352.453565 ms | 215.810354 ms |

HTTP chat, `32` requests / `32` slots / `256K` / `position=262080` /
`32` tokens/request:

| Mode | HTTP 200 | Coalesced | Client tok/s | Server wall tok/s | Server decode tok/s | Compressed-KV sum |
|---|---:|---:|---:|---:|---:|---:|
| control | 32/32 | 32 | 51.345855 | 84.085278 | 99.748339 | 4817.184988 ms |
| skip stats | 32/32 | 32 | 58.923892 | 96.153537 | 117.340768 | 2645.572812 ms |

Selected-token, `32` requests / `32` slots / `256K`:

| Shape | Mode | First token | Client tok/s | Scaffold tok/s | Compressed-KV sum |
|---|---|---:|---:|---:|---:|
| 8 tokens, position 262136 | control | 36944 | 60.048430 | 103.977380 | 789.960018 ms |
| 8 tokens, position 262136 | skip stats | 36944 | 64.474617 | 122.890131 | 448.038406 ms |
| 32 tokens, position 262112 | control | 109328 | 74.522267 | 101.137018 | 3214.154721 ms |
| 32 tokens, position 262112 | skip stats | 109328 | 67.004359 | 113.130472 | 1895.528614 ms |

The 32-token selected-token response bodies are semantically identical after
excluding timing fields: `0/32` differences. Raw file hashes differ because
the endpoint embeds timing counters in each response.

## Decision

Skipping compressed dense host stats is a real production candidate, but it
is not promoted as a default in this sprint.

The direct and selected-token runs show the expected effect: the gate removes
host-side diagnostic copies/synchronization from the compressed/indexer dense
projection path and cuts compressed-KV parsed time materially while preserving
the selected output token. The full chat A/B also improves server decode
throughput, but the normal chat response-text comparison still needs a more
deterministic comparator before default promotion.

Leave the gate opt-in:

```text
DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1
```

Next work should either promote this after deterministic chat/token parity, or
move to the next true hot path: compressed/indexer dense projection format and
kernel selection, including an offline INT8+scale pack variant for the current
FP8 source tensors.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint372-skip-dense-stats-direct-ab`
  - `/workspace/logs/sprint372-skip-dense-stats-http-ab`
  - `/workspace/logs/sprint372-skip-dense-stats-selected-ab`
  - `/workspace/logs/sprint372-skip-dense-stats-selected32-ab`
- Local:
  - `logs/from-cluster/sprint372-skip-dense-stats-direct-ab`
  - `logs/from-cluster/sprint372-skip-dense-stats-http-ab`
  - `logs/from-cluster/sprint372-skip-dense-stats-selected-ab`
  - `logs/from-cluster/sprint372-skip-dense-stats-selected32-ab`
