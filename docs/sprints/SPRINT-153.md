# Sprint 153 - Bounded TP Pack Contract

Date: 2026-05-21

## Objective

Turn the 2-way routed-FFN tensor-parallel idea from a benchmark-only result
into a bounded appliance-format contract that can be loaded and reasoned about
by the V100 context code.

This is not a production scheduler rewrite. It is the smallest useful
checkpoint before deciding whether to build a one-stage TP routed-FFN executor
for the 128-slot/32K tier.

## Changes

- Added `--emit-tp-split` to `tools/ds4-v100-appliance-pack`.
- For each selected MXFP4 gate tensor, the packer can now emit experimental
  two-way middle-dimension splits:
  - `blk.N.ffn_gate_up_exps.tp0.weight`
  - `blk.N.ffn_gate_up_exps.tp1.weight`
  - `blk.N.ffn_down_exps.tp0.weight`
  - `blk.N.ffn_down_exps.tp1.weight`
- The split keeps half 0 on the layer owner GPU and places half 1 on a fixed
  NVLink peer:
  - `0<->3`, `1<->2`, `4<->7`, `5<->6`
- Added context binding support for TP-routed expert IDs, allowing their owner
  GPU to differ from the layer stage while keeping normal TurboMind tensors
  fail-closed on layer ownership.
- Extended `tools/ds4-v100-context-smoke` with:
  - `--tm-index`
  - `--allow-partial`

## Pack Validation

Command shape:

```text
tools/ds4-v100-appliance-pack \
  --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --source /models/DSv4-Flash-256e-fixed.gguf \
  --out-dir /workspace/ds4-tp-split-pack-s153 \
  --layer 3 \
  --expert-limit 6 \
  --fuse-gate-up-interleaved \
  --keep-separate-gate-up \
  --emit-tp-split \
  --skip-non-experts \
  --lib build/turbomind-v100-s127/libggml-turbomind.so
```

Result:

| Artifact | Result |
|---|---:|
| TurboMind descriptor rows | `8` |
| Normal pack rows | header only, by design for bounded `--skip-non-experts` |
| `gpu0.weights` | `173,801,472` bytes |
| `gpu3.weights` | `40,108,032` bytes |
| Other GPU shards | `0` bytes |
| Context bind | PASS with `turbomind_tensor_count=8` |

The TP rows use these dimensions:

| Tensor | GPU | Shape | Packed experts | Weight bytes/expert | Scale bytes/expert |
|---|---:|---|---:|---:|---:|
| `ffn_gate_up_exps.tp0` | 0 | `[4096x2048x256]` | 6 | `4,194,304` | `262,144` |
| `ffn_down_exps.tp0` | 0 | `[1024x4096x256]` | 6 | `2,097,152` | `131,072` |
| `ffn_gate_up_exps.tp1` | 3 | `[4096x2048x256]` | 6 | `4,194,304` | `262,144` |
| `ffn_down_exps.tp1` | 3 | `[1024x4096x256]` | 6 | `2,097,152` | `131,072` |

## Kernel And Topology Results

Standalone one-GPU TP split fixture:

| Routes | Full routed | Half 0 | Half 1 | Ideal compute speedup | Sequential speedup | Reduce payload |
|---:|---:|---:|---:|---:|---:|---:|
| 768 | `1.0298 ms` | `0.5281 ms` | `0.5279 ms` | `1.950x` | `0.976x` | `6 MiB F16 / 12 MiB F32` |
| 1536 | `1.3823 ms` | `0.9028 ms` | `0.9047 ms` | `1.528x` | `0.770x` | `12 MiB F16 / 24 MiB F32` |

Peer-copy timing:

| Payload | NV2 | NV1 | SYS |
|---:|---:|---:|---:|
| 6 MiB F16, 768 routes | `0.131 ms` | `0.261 ms` | `0.656 ms` |
| 12 MiB F16, 1536 routes | `0.261 ms` | `0.520 ms` | `1.303 ms` |

Real 2-GPU TP proxy on NV2 pair `0,3`:

| Routes | Full one-GPU | Concurrent half compute | Compute speedup | Total with copies | Total speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 768 | `0.9769 ms` | `0.5612 ms` | `1.741x` | `0.8446 ms` | `1.157x` | PASS |
| 1536 | `1.3002 ms` | `0.8817 ms` | `1.475x` | `1.4264 ms` | `0.912x` | PASS |

The 768-route result is positive but smaller than the earlier Sprint 150
`~1.28x` best case. The conclusion is unchanged: 2-way TP may be worth a
bounded 128-slot/32K prototype on NV2 pairs, but 1536 routes are already
copy-limited unless the scheduler keeps activations replicated or overlaps the
payload movement better.

## Decision

Do not rewrite the production topology broadly yet.

Proceed only with a narrow one-stage TP executor if the next sprint can keep
scope to:

- one routed layer;
- NV2 peer pair placement;
- 128-slot/32K, 768-route compact served shape;
- explicit opt-in runtime path;
- correctness against the existing full one-GPU routed FFN.

The default layer-sharded appliance remains the production path.

## Artifacts

- `logs/from-cluster/sprint153-tp-pack-contract/pack.log`
- `logs/from-cluster/sprint153-tp-pack-contract/context-smoke.log`
- `logs/from-cluster/sprint153-tp-pack-contract/tp-split-kernel-fixture.log`
- `logs/from-cluster/sprint153-tp-pack-contract/tp-split-kernel-fixture-1536.log`
- `logs/from-cluster/sprint153-tp-pack-contract/tp-split-2gpu-0-3-128.log`
- `logs/from-cluster/sprint153-tp-pack-contract/tp-split-2gpu-0-3-256.log`
- `logs/from-cluster/sprint153-tp-pack-contract/p2p-768-f16.log`
- `logs/from-cluster/sprint153-tp-pack-contract/p2p-1536-f16.log`

## Validation

- `make -C /workspace/ds4 -j80 CUDA_ARCH=sm_70 tools/ds4-v100-appliance-pack tools/ds4-v100-context-smoke`
- `tools/ds4-v100-appliance-pack --emit-tp-split ...`
- `tools/ds4-v100-context-smoke --tm-index /workspace/ds4-tp-split-pack-s153/turbomind-pack-index.tsv --allow-partial`
- `test_ggml_turbomind_grouped_gate_up_fusion` with `DS4_TURBOMIND_GATE_UP_TP_SPLIT=1`
- `test_ggml_turbomind_p2p_reduce_proxy`
- `test_ggml_turbomind_tp_split_2gpu` on pair `0,3`
