# Sprint 290 - TP/EP Resident Output Head Gate

Date: 2026-05-23

## Goal

Turn the Sprint 289 cold output-head diagnostic into a resident TP/EP
output-head timing gate.

This sprint stays TP/EP-only. It does not touch PP/layer-split code and does
not claim real text serving yet, because the TP/EP token-major loop still needs
to carry final DS4 HC into the output head.

## Context

Sprint 289 proved the TP/EP output-head tensor layout:

```text
synthetic HC [slots,4,4096]
  -> output HC collapse
  -> output_norm
  -> BF16 output.weight vocab shards on 8 GPUs
  -> global top-1
```

That gate was cold and serial: it reloaded output shards and read full logits
back to host. Its useful signal was that the scalar BF16 projection kernel was
about `7.6 ms` at 32 slots, but the cold total was not representative of the
serving path.

## Implementation

- Added `--output-head-resident-gate` to
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Preloads real BF16 `output.weight` vocab shards once across all 8 V100s.
- Keeps output-head scratch resident for repeated measured iterations.
- Separates timing for:
  - HC prep/collapse on GPU0
  - embedding broadcast to all vocab-shard owners
  - vocab projection wall time
  - worst per-GPU projection kernel time
  - top-1 reduction/readback
- Added `shard_top1_kernel` so each GPU reduces its own vocab shard on device.
  The host now receives only `8 * slots` token/logit candidates instead of the
  full logits tensor.

## Definition of Done

- [x] The TP/EP full-layer smoke builds on the V100 pod.
- [x] Resident output-head gate runs against the real production pack and
  contract.
- [x] 16, 32, and 64 slot gates pass with finite logits and deterministic
  selected-token output.
- [x] GPU-side shard top-1 is measured against the previous full-logit
  host-readback gate.
- [x] Sprint status and vision are updated with the outcome.

## Validation

Build:

```text
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Main 32-slot command:

```text
./tools/ds4-v100-tp-ep-full-layer-smoke \
  --output-head-resident-gate \
  --pack-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --contract /workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv \
  --tm-index /workspace/packs/ds4-appliance-full-tm-gated-s181/turbomind-pack-index.tsv \
  --lib /workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so \
  --slots 32 --warmup 3 --iters 10
```

Results with full-logit host readback before GPU top-1:

| Slots | Avg total ms | Projection wall ms | Worst kernel ms | Host readback/reduce ms | Output-head tok/s | Result |
|---:|---:|---:|---:|---:|---:|---|
| 16 | 8.225363 | 3.767571 | 3.721120 | 3.901915 | 1945.202905 | PASS |
| 32 | 15.980438 | 7.478751 | 7.429891 | 7.637466 | 2002.448256 | PASS |
| 64 | 30.895625 | 14.727130 | 14.679424 | 14.708078 | 2071.490725 | PASS |

Results after GPU-side per-shard top-1:

| Slots | Avg total ms | HC prep ms | Broadcast ms | Projection wall ms | Worst kernel ms | Device top-1/readback ms | Output-head tok/s | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 4.489646 | 0.067640 | 0.455966 | 3.764156 | 3.718890 | 0.201784 | 3563.755123 | PASS |
| 32 | 8.528343 | 0.079832 | 0.762456 | 7.474198 | 7.427597 | 0.211761 | 3752.194257 | PASS |
| 64 | 16.505764 | 0.082531 | 1.344104 | 14.874445 | 14.831216 | 0.204582 | 3877.433386 | PASS |

All runs select token `26803` for slot 0 with finite logit
`3608.255126953`.

Evidence:

```text
logs/from-cluster/sprint290-tp-ep-output-head-resident-gate/cluster/
```

## Decision

Promote GPU-side shard top-1 as the TP/EP output-head reduction shape. Full
logit readback is rejected for serving because it roughly doubles resident
output-head latency at the 32-slot target.

The remaining output-head bottleneck is now the BF16 scalar projection itself:
about `7.4 ms` at 32 slots and `14.8 ms` at 64 slots. That is acceptable as a
first real-token path, but it is not the final optimized path. Once final HC is
available in the token-major loop, we should wire this resident output-head
primitive into `/v1/completions`, then revisit projection kernel selection.

## Remaining Gap

The TP/EP serving loop still carries per-rank hidden shards, not final DS4 HC
`[slots,4,4096]`. The next sprint should add a TP/EP HC carry contract for the
token-major loop and feed the resident output-head primitive from that state.
