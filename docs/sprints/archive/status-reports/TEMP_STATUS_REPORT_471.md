# TEMP Status Report 471: Corrected V100 DCGMI Profiling

## Current Status

The TP/EP profile harness now has a V100-correct `dcgmi` sampler backend, and
the HTTP A/B wrapper forwards `--gpu-sampler dcgmi` plus `--dcgmi-fields`.

The updated profiling rule is:

- `nvidia-smi dmon`: cheap NVML health only.
- `dcgmi dmon`: short, targeted hardware-counter windows.
- Default DCGMI fields:
  `203,252,155,150,1002,1003,1005,1009,1010,1001,1011,1012`.
- Tensor activity uses a separate pass with `1004`; do not mix it with
  `1002/1003`.

## Clean Serving Runs

Default zero-multiplex pass:

```text
/localpool/ds4/workspace/logs/s470-dcgmi-serving-s8-t8-default/...
8 slots, 256K context, 8 requests, 8 tokens/request
responses_complete: yes
http_200: 8/8
client_generated_tok_s: 3.667873
gpu_sample_source: dcgmi
gpu_sample_count: 1968
gpu_steady_sample_count: 280
gpu_steady_util_avg: 8.871429%
gpu_steady_sm_active_avg: 0.016914
gpu_steady_sm_occupancy_avg: 0.005125
gpu_steady_dram_active_avg: 0.005089
gpu_steady_gr_engine_active_avg: 0.049082
gpu_steady_nvlink_tx_bytes_avg: 60513147.99
gpu_steady_nvlink_rx_bytes_avg: 60517512.20
```

Tensor-only pass:

```text
/localpool/ds4/workspace/logs/s470-dcgmi-serving-s8-t8-tensor/...
8 slots, 256K context, 8 requests, 8 tokens/request
responses_complete: yes
http_200: 8/8
client_generated_tok_s: 3.465720
gpu_sample_source: dcgmi
gpu_sample_count: 2024
gpu_steady_sample_count: 288
gpu_steady_tensor_active_avg: 0.000694
gpu_steady_tensor_active_max: 0.002
```

## Interpretation

The corrected counters point to the same practical bottleneck, but with better
evidence: the current serving path is not saturating tensor cores. Request-window
SM activity and occupancy are low, and tensor activity is effectively zero.

That makes the next optimization gate sharper: changes should be judged by
whether they increase request-window `tensor_active`, `sm_occupancy`, and
throughput together. NVML utilization alone is not enough.

## Next

Use the corrected DCGMI harness while attacking the actual utilization problem:
reduce graph/event fragmentation, remove rank-major staging that serializes
work, and move the hot routed/dense shapes into HMMA-friendly fused kernels.
