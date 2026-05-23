# Sprint 291 - TP/EP Final-HC Carry Scaffold

Date: 2026-05-23

## Goal

Add a TP/EP-only final-HC carry scaffold so the token-major loop has an
explicit place to materialize DS4 output-head input shape.

This sprint does not claim real text serving. The current carry kernel creates
a proxy HC shard from the per-rank hidden shard. It proves ownership, memory
layout, timing, and finite dataflow for `[slots,4,4096]`, but the true DS4 HC
row semantics still need to be ported from the layer executor.

## Implementation

- Added `--final-hc-carry-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Added per-rank `d_final_hc_shard`:

```text
rank p owns [slots][4][512] float
all ranks together represent [slots][4][4096]
```

- Added `expand_hidden_to_proxy_hc_shard_kernel`.
- The kernel is invoked after the hidden compose stage when the gate is
  enabled.
- The decode loop now reports `final_hc_ms_per_step`.
- The token-major all-layer summary now reports:
  - `final_hc_carry_gate`
  - `sum_final_hc_ms`
- The gate reads the sharded HC buffers back for finite checks and checksum so
  failures are visible even when normal decode checksum skipping is enabled.

## Definition of Done

- [x] The TP/EP full-layer smoke builds on the V100 pod.
- [x] The 1-token all-layer gate passes with `--final-hc-carry-gate`.
- [x] The matching 1-token control run passes without the gate.
- [x] A short 4-token continuation gate passes with the carry enabled.
- [x] Sprint status and vision are updated with the outcome.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

1-token carry gate:

```text
./tools/ds4-v100-tp-ep-full-layer-smoke \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --decode-steps 1 \
  --fuse-compose-sum --dense-f16-cublas-compose --dense-f16-cache-compose \
  --skip-descriptor-checks --skip-predecode-probes \
  --shared-expert-bindings --shared-dense-ops --overlap-ep-dense \
  --source-copy-schedule --skip-self-compose-copy --multi-copy-streams \
  --copy-event-compose --compact-route-compose \
  --token-major-all-layers --all-layers --serving-bench \
  --final-hc-carry-gate
```

Results:

| Run | Steps | Slots | Pass invocations | Decode ms | Wall ms | Sum final-HC ms | Decode tok/s | Wall tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Control | 1 | 32 | 43 | 70.923652 | 84.892843 | 0.000000 | 451.189400 | 376.945793 | PASS |
| Carry | 1 | 32 | 43 | 75.554825 | 140.486733 | 2.100054 | 423.533507 | 227.779516 | PASS |
| Carry | 4 | 32 | 172 | 179.526855 | 424.273807 | 8.113938 | 712.985252 | 301.691968 | PASS |

The 4-token carry run reports continuation decode throughput of `960.823272`
tok/s and continuation wall throughput of `345.413817` tok/s. The final-HC
carry cost is about `2.03 ms/token` across all 43 layers, or about
`0.047 ms/layer`.

Evidence:

```text
logs/from-cluster/sprint291-tp-ep-final-hc-carry/cluster/
```

## Decision

Keep the sharded HC carry buffer shape:

```text
per GPU: [slots][4][512] f32
logical: [slots][4][4096] f32
```

The overhead is small enough to keep moving toward output-head integration.
However, the proxy expansion is not semantically correct DS4 HC. The next
sprint should replace the proxy expansion with the real HC row update
semantics, or add a clearly labeled bridge that gathers these shards into the
resident output-head primitive while keeping the endpoint diagnostic.

## Remaining Gap

`/v1/completions` still cannot emit real model tokens. The missing work is:

- preserve or reconstruct true DS4 HC rows through the TP/EP layer loop;
- feed the resident vocab-sharded output head from that HC state;
- return selected token IDs from the HTTP path;
- then add tokenizer text, prompt prefill, stop handling, and MTP.
