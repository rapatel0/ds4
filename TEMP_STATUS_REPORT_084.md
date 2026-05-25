# TEMP Status Report 084

Date: 2026-05-25

## Current Focus

TP/EP compressed-KV hot path reduction and format planning.

## Sprint 372 Result

Implemented a default-off TP/EP serving gate:

```text
DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1
--true-ds4-compressed-kv-skip-dense-stats-gate
```

This skips host-side dense-output statistics after compressed/indexer dense
projection. It does not change model math or dtypes; it removes diagnostic
host copies/synchronization from the serving-visible hot path.

| Test | Mode | Main result |
|---|---|---|
| Direct token-major, 32 slots / 256K / 32 steps | control | first token `98751`, scaffold `100.739521` tok/s, compressed-KV `3141.768079` ms |
| Direct token-major, 32 slots / 256K / 32 steps | skip stats | first token `98751`, scaffold `117.463961` tok/s, compressed-KV `1789.795027` ms |
| HTTP chat, 32 requests / 32 slots / 256K / 32 tokens | control | `32/32` HTTP 200, client `51.345855` tok/s, server decode `99.748339` tok/s |
| HTTP chat, 32 requests / 32 slots / 256K / 32 tokens | skip stats | `32/32` HTTP 200, client `58.923892` tok/s, server decode `117.340768` tok/s |
| Selected-token, 32 requests / 32 slots / 256K / 8 tokens | control | first token `36944`, client `60.048430` tok/s |
| Selected-token, 32 requests / 32 slots / 256K / 8 tokens | skip stats | first token `36944`, client `64.474617` tok/s |
| Selected-token, 32 requests / 32 slots / 256K / 32 tokens | control | first token `109328`, scaffold `101.137018` tok/s, compressed-KV `3214.154721` ms |
| Selected-token, 32 requests / 32 slots / 256K / 32 tokens | skip stats | first token `109328`, scaffold `113.130472` tok/s, compressed-KV `1895.528614` ms |

The selected-token 32-token response bodies have `0/32` semantic differences
after excluding timing fields. Raw files differ because timing counters are
embedded in responses.

## Decision

Skip-stats is a production candidate but remains default-off for now. The
direct and selected-token evidence is strong; normal chat text parity still
needs a deterministic comparator before promotion.

## Current Bottleneck

The observed `GPU0 ~40%` and peer GPUs `<10%` pattern is real and expected
for the current prototype. TP/EP dense compressed/indexer projection runs
across ranks, but orchestration, output-head ownership, selected-token
response work, and some synchronization/measurement paths are still GPU0-heavy.
The active-slot matrix showed that coalescing works, but server decode and
average GPU utilization remain flat at full 32-slot occupancy.

The current hot area is:

- compressed/indexer dense projection
- surrounding compressed-KV staging/state boundaries
- GPU0-heavy harness/control/output-head work

## INT8 Direction

Offline INT8+scale conversion is worth testing, scoped to the FP8 source
compressed/indexer dense tensors first. It should be implemented as a pack
variant, not a global model conversion:

- preserve source tensor dtype/scale metadata in the manifest
- create INT8+scale packed shards for the candidate tensors
- run an INT8 kernel that expands/dequantizes inside the GPU and emits the
  same FP32 downstream output contract
- A/B per-layer diffs and selected-token/chat parity before enabling serving

This is attractive because V100 lacks native FP8/FP4 tensor cores, while our
existing integer/TurboMind/CUTLASS work may make INT8 storage plus fused
dequant/MMA more efficient than the current FP8-to-FP16 staging path.

## Artifacts

- `logs/from-cluster/sprint372-skip-dense-stats-direct-ab`
- `logs/from-cluster/sprint372-skip-dense-stats-http-ab`
- `logs/from-cluster/sprint372-skip-dense-stats-selected-ab`
- `logs/from-cluster/sprint372-skip-dense-stats-selected32-ab`
