# Sprint 130 - Routed FFN Software-Pipeline Targeting

Date: 2026-05-21

## Objective

Answer whether kernel fusion with software pipelining is a useful path for the
DS4 V100 appliance, using the copied TurboMind SM70 templates and local tc-grid
kernels as evidence, then test the closest existing runtime analogue on the
actual 8x V100 node.

## Finding

Yes, but only at the routed FFN packed-GEMM boundary.

The useful shape is not a giant fused layer kernel. The useful shape is a
fixed DS4/V100 routed-expert kernel that keeps the hot packed-weight path inside
one software-pipelined loop:

```text
LDG packed MXFP4 weights/scales
  -> stage raw packed bytes/scales
  -> register dequant to half fragments
  -> SM70 HMMA m8n8k4
  -> fused routed epilogue
```

TurboMind's SM70 mainloop already follows this model: it stages A/B and quant
metadata through register/shared-memory buffers, transforms packed B fragments
with `Transform_HMMA_SIMT_B`, and feeds Volta HMMA. The local tc-grid v13
kernels independently found the same pattern for INT8: store raw quantized B,
dequant in registers immediately before MMA, and schedule the next LDS/dequant
work ahead of the current MMA.

That is the kernel family that can plausibly move GPU utilization. Small
launch-boundary fusions around the FFN tail are not enough.

## V100 Check

I reran the closest existing epilogue-fusion analogue on `gpu-01`, using the
current fused production appliance and compact TurboMind schedule:

```text
appliance: /workspace/ds4-appliance-full-tm-fused-s111
lib:       /workspace/ds4/build/turbomind-v100-s127/libggml-turbomind.so
ctx:       262144
slots:     16
tokens:    16
requests:  16
```

| Mode | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---|
| compact fused control | `45.837745` | `42.972886` | 16/16 token match |
| compact fused + route-row-reduce | `45.660765` | `42.806967` | 16/16 token match |

Decision: keep `DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=0` for the production
fused appliance. Route-row-reduce remains a correct diagnostic, but the
current data confirms that final scatter/reduce fusion is not the missing
throughput lever.

## Implementation Direction

The next real implementation slice should be a guarded DS4-only routed-FFN
kernel probe, not another generic TurboMind dispatch knob.

Start with the current compact active-expert route schedule and the interleaved
gate/up format:

- fixed V100 shapes first: `CTA_M=8` or `16`, `CTA_N=128`, `CTA_K=32` or `64`;
- input A remains half row-major for the active routed rows;
- B consumes the existing persistent TurboMind-packed MXFP4 expert weights;
- dequant happens in registers immediately before SM70 HMMA;
- gate/up uses the interleaved gated-SiLU epilogue shape;
- down and final reduce can stay on the existing path until the gate/up probe
  shows a measurable per-layer win.

Promotion gate:

- standalone routed gate/up probe must beat TurboMind gated-SiLU for the same
  compact route schedule;
- full 43-layer smoke must pass with the new branch enabled;
- served 16-slot/256K A/B must clear the current `45.84` tok/s control by more
  than run noise before becoming a default.

## Decision

Proceed with a DS4-specific packed MXFP4 gate/up software-pipeline probe.
Do not spend another sprint on route metadata launches, dispatch-policy
selection, or final scatter fusion unless profiling changes materially.

