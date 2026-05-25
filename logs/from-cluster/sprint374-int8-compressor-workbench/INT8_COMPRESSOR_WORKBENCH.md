# INT8 Compressor Workbench

Target shapes: `M=32,N=128/64,K=4096`.

| M | N | K | Kernel | ms | TFLOP/s | GB/s | max abs | p99 abs | mean abs | OK |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 4096 | `cublas-f16-tensorop` | 0.009250 | 3.627 | 143.469 | 0.000109196 | 8.84533e-05 | 3.0773e-05 | 1 |
| 32 | 128 | 4096 | `tc-grid-v12s-ks8+zero` | 0.042721 | 0.785 | 25.695 | 0.0152217 | 0.0125701 | 0.00382764 | 1 |
| 32 | 128 | 4096 | `tc-grid-v13-rf-v6` | 0.282150 | 0.119 | 3.891 | 0.0134012 | 0.011393 | 0.00375203 | 1 |
| 32 | 64 | 4096 | `cublas-f16-tensorop` | 0.008803 | 1.906 | 90.268 | 0.000109196 | 8.92878e-05 | 3.09974e-05 | 1 |
| 32 | 64 | 4096 | `tc-grid-v12s-ks8+zero` | 0.036673 | 0.457 | 22.115 | 0.0152217 | 0.0125701 | 0.00382366 | 1 |
| 32 | 64 | 4096 | `tc-grid-v13-rf-v6` | 0.260751 | 0.064 | 3.110 | 0.0134012 | 0.011393 | 0.00376133 | 1 |

Notes:

- `tc-grid-v12s-ks8+zero` includes the required output zeroing for split-K atomic accumulation.
- The cuBLAS baseline uses FP16 tensor-op inputs and FP32 output as the BF16-on-V100 proxy.
- INT8 inputs use the tc-grid contract: FP32 activations, INT8 weights, FP16 per-row/per-32K scales.
