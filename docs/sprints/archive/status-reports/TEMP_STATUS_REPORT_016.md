# TEMP Status Report 016 - Sprint 202 TP4 Routed-FFN Compute

Date: 2026-05-23

## Topline

Sprint 202 measured the compute side of TP4 using real TurboMind MXFP4 routed
FFN kernels. It also found and fixed a benchmark lifecycle bug where the full
reference and shard 0 briefly crossed streams on GPU0 and shared one TurboMind
workspace. The corrected result is clear:

- TP4 expert compute scales well.
- Full-hidden copy-in/copy-out erases the win for practical route counts.
- The next TP implementation must be full-layer TP4/EP, not routed-only TP.

## Current Best Served Baseline

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| `fused6_reduce + graph` | 256K | 16 | `67.886268` | `66.825545` | `16/16` |

## New Sprint 202 Data

V100 target:

```text
test_ggml_turbomind_tp_split_4gpu
```

Measured on GPUs `0,1,2,3`:

| Tokens/active expert | Total routes | Full 1-GPU | TP4 concurrent compute | Compute speedup | Total with copies | Copy-inclusive speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---:|---|
| `1` | `6` | `0.1457 ms` | `0.0543 ms` | `2.686x` | `0.1479 ms` | `0.986x` | PASS |
| `16` | `96` | `0.2920 ms` | `0.1242 ms` | `2.350x` | `0.3729 ms` | `0.783x` | PASS |
| `128` | `768` | `1.1553 ms` | `0.3178 ms` | `3.636x` | `1.6936 ms` | `0.682x` | PASS |

Evidence:

```text
logs/from-cluster/sprint202-tp4-routed-ffn/
```

## Interpretation

The compute-only TP4 result is good enough to keep TP4 alive:

- `96` routes: `2.350x`.
- `768` routes: `3.636x`.

But copy-inclusive routing is bad at the same shapes:

- `96` routes: `0.783x`.
- `768` routes: `0.682x`.

So the project should not implement a wider routed-only TP overlay. The only
rational TP path is full-layer TP4/EP, where hidden state stays resident inside
the TP boundary and the routed expert compute speedup can pay for collectives.

## Next

Plan Sprint 203 as a bounded full-layer TP4/EP slice over a small layer span, or
return to the alternate serious branch: a persistent fused routed-FFN kernel
with CUTLASS/TurboMind-style software pipelining.
