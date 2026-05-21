# Sprint 162 - TP Route-Shape Gate

Date: 2026-05-21

## Objective

Decide whether the next implementation should be tensor-parallel routed FFN
work or another scheduler-local batching attempt.

Sprint 161 rejected small-route fused executor variants. The remaining
question was whether TP only helps at the old 128-slot/32K diagnostic shape, or
whether it also helps the practical 256K serving shapes:

```text
one-slot served shape:     1 slot  * 6 routes = 6 routes
16-slot practical shape:  16 slots * 6 routes = 96 routes
```

## Result

The existing two-GPU TurboMind TP proxy was run on clean NV2 pairs for both
practical route counts. It includes conservative input activation copy to the
peer GPU and partial output copy back.

| Shape | Pair | Full one-GPU | Concurrent halves | Total with copies | Total speedup | Correctness |
|---|---|---:|---:|---:|---:|---|
| 6 routes | `0,3` | `0.1457 ms` | `0.0850 ms` | `0.1157 ms` | `1.260x` | PASS |
| 96 routes | `0,3` | `0.2923 ms` | `0.1593 ms` | `0.2201 ms` | `1.328x` | PASS |
| 96 routes | `4,7` | `0.2920 ms` | `0.1596 ms` | `0.2203 ms` | `1.325x` | PASS |

Correctness tolerance matched the prior TP gates:

```text
rel ~= 2.47e-04
bad = 0
nan = 0
```

## Interpretation

TP is now a credible implementation target at the actual 256K serving shapes.
This changes the next-step priority:

- It is not necessary to wait for 768-route/128-slot batches to make TP
  compute-positive.
- TP can help even when the current per-step pipeline presents one request at a
  time, because the 6-route proxy still shows `1.26x` after copies.
- The engineering blocker is no longer TP math. It is runtime integration:
  loading TP split descriptors into layer state, dispatching the TP half on an
  NV2 peer, and summing the partial outputs without breaking the current
  stage pipeline.

## Next Implementation Sprint

Sprint 163 should build a bounded one-layer TP routed-FFN executor, not another
proxy:

1. Generate or reuse a bounded TP split pack for one routed layer on pair
   `0,3`.
2. Bind the four TP descriptors:
   - `ffn_gate_up_exps.tp0.weight`
   - `ffn_gate_up_exps.tp1.weight`
   - `ffn_down_exps.tp0.weight`
   - `ffn_down_exps.tp1.weight`
3. Add an explicit opt-in runtime path for that one layer only.
4. For the selected layer, run:
   - owner half on the layer GPU;
   - peer half on the NV2 peer;
   - peer partial copy back;
   - FP32 sum into the normal routed output buffer.
5. Compare the one-layer TP output against the current one-GPU routed FFN
   output before any served-mode test.

Keep the first implementation bounded to one layer and one NV2 pair. Do not
attempt a broad 8-GPU TP scheduler rewrite until the one-layer executor passes
correctness inside the appliance runtime.

## Artifacts

- `logs/from-cluster/sprint162-tp-96-proxy/tp-6-0-3-timeout.log`
- `logs/from-cluster/sprint162-tp-96-proxy/tp-96-0-3.log`
- `logs/from-cluster/sprint162-tp-96-proxy/tp-96-4-7.log`

## Validation

Cluster command shape:

```text
cd /workspace/ds4/build/turbomind-v100-s127
DS4_TP_SPLIT_CASES=1  DS4_TP_SPLIT_GPU0=0 DS4_TP_SPLIT_GPU1=3 ./test_ggml_turbomind_tp_split_2gpu
DS4_TP_SPLIT_CASES=16 DS4_TP_SPLIT_GPU0=0 DS4_TP_SPLIT_GPU1=3 ./test_ggml_turbomind_tp_split_2gpu
DS4_TP_SPLIT_CASES=16 DS4_TP_SPLIT_GPU0=4 DS4_TP_SPLIT_GPU1=7 ./test_ggml_turbomind_tp_split_2gpu
```
