# TEMP Current Report

Date: 2026-05-21

## Latest Update

After this report was first written, I tested a true stage-count software
pipeline variant on the fused MXFP4 gate/up+gated-SiLU kernel. The 768-route
`m128_s4` probe improved the isolated benchmark (`0.5811 ms` vs `0.6033 ms`
for `m128`) and passed full 43-layer smoke, but served A/B was only
`60.049057` generated / `56.295991` decode tok/s versus `59.865668` /
`56.124063` control. A targeted NCU pass then showed essentially identical
kernel time, SM throughput, DRAM throughput, and HMMA instruction count between
`m128` and `m128_s4`, so the stage-4 path should stay opt-in.

I also added a TP split probe. Splitting the DS4 routed-FFN middle dimension
from `2048` into two `1024` halves gives an ideal 2-way compute speedup of
`1.858x` at 768 routes and `1.468x` at 1536 routes before communication.
Peer-copy payload timing on the node shows 12 MiB hidden-state payloads move in
about `0.26 ms` over NV2, `0.52 ms` over NV1, and `1.29-1.31 ms` over SYS.
That makes 2-way TP worth a bounded prototype on NV2 pairs, but it is not a
credible direct path from `~61 tok/s` to `1k+ tok/s` by itself.

## Short Answer

We have a correct, deployed 8x V100 DS4-Flash appliance path, but we are still
far from the practical throughput objective. The best observed served
throughput is now about `61` aggregate generated tok/s, with decode separated
at about `57` continuation tok/s. The target discussed in the vision is
roughly `1k-2k` aggregate tok/s for practical high-throughput serving, so the
remaining gap is large.

## Current Best Performance

| Mode | Context | Slots | Generated tok/s | Decode/continuation tok/s | Notes |
|---|---:|---:|---:|---:|---|
| Best short-context aggregate | 16K | 256 | `61.223893` | `57.397400` | Sprint 146 control repeat, 256/256 token match |
| Best 32K aggregate | 32K | 128 | `60.130047` | `56.371919` | Sprint 139, 128/128 token match |
| Best 64K aggregate | 64K | 64 | `57.322945` | `53.740261` | Sprint 136 |
| Best 128K aggregate | 128K | 32 | `52.840889` | `49.538334` | Sprint 135 |
| Best 256K aggregate | 256K | 16 | `46.394722` | `43.495052` | Sprint 128 best opt-in stack |
| Current 256K production-auto repeat | 256K | 16 | `43.534061` | `40.813182` | Sprint 122 production-auto |
| Best 1M long-context aggregate | 1M | 4 | `21.771077` | `20.410385` | Sprint 119 |
| Best observed single-slot sustained | 1M | 1 | `3.600787` | `3.375738` | Older Sprint 061 log; not recently rebenchmarked after later high-slot work |

Important caveat: the latest benchmark harness now separates prompt/prefill,
generated, and continuation/decode. Older results often reported generated
throughput only, so for current decisions I am weighting continuation/decode
more heavily.

## How Far We Are

If the objective is "correct appliance that serves DS4-Flash on the V100
cluster", we are there.

If the objective is "performance-optimized practical serving", we are not
there. Current best aggregate throughput is `~61 tok/s`, which is roughly:

- `~16x` below a `1k tok/s` lower practical target.
- `~33x` below a `2k tok/s` target.
- Much farther below any synthetic/high-batch hero target.

The fresh 256-slot routed-FFN profile says the main V100 bottleneck remains
inside the routed expert path:

| Bucket | Approx share of profiled routed FFN time at 256-slot/16K |
|---|---:|
| gate/up MXFP4 GEMM | `~61%` |
| down MXFP4 GEMM | `~31%` |
| route build | `~3%` |
| gather | `~2%` |
| scatter/reduce | `~2-3%` |

That means small host-side routing tweaks are mostly exhausted. The next real
lever has to change the expert GEMM dataflow or the way expert batches are
formed.

## Techniques Already Explored

Kernel fusion:

- Fused TurboMind gate/up appliance shipped. This was a real win: 8-slot/256K
  improved from `31.312694` to `33.430971` generated tok/s in the same-binary
  A/B.
- Gated-SiLU epilogue and interleaved gate/up packing work correctly, but the
  served improvement was small.
- Shared F8 gate/up SwiGLU HMMA and batched attention-projection F8 HMMA
  shipped as defaults.
- Route-row reduce, half2 reduce, down-reduce epilogue, fixed-shape 768-route
  gate/down probes, m64n256 tile probes, and 1536-route fixed-shape probes were
  tested. Most were correct; most were neutral or worse in served A/B.

Batched decode / multislots:

- Yes. We tested active-slot/microbatch scaling extensively:
  8 -> 16 -> 32 -> 64 -> 128 -> 192 -> 256 slots.
- It helped materially at first, but the curve flattened:
  128-slot/16K was `59.860493`, 192-slot/16K was `60.700926`, and 256-slot/16K
  was `61.065087` in Sprint 145.
- Async event handoff shipped and helped preserve stage overlap.
- Chunking slots to feed wider kernels was tested and regressed badly because
  it lost stage overlap.

Software pipelining:

- Tested as a stage-count variant inside the fused TurboMind MXFP4
  gate/up+gated-SiLU kernel.
- The isolated 768-route `m128_s4` path improved, but served throughput and NCU
  counters did not show a material end-to-end gain.
- TurboMind's SM70 MXFP4 kernels already use the relevant style internally:
  packed low-bit load, dequant/staging, and Volta HMMA.
- We have not yet written a new end-to-end DS4-only persistent routed-FFN kernel
  that software-pipelines gate/up, activation, down, and reduce as one larger
  unit.
- Sprint 147 extended the down+weighted-reduce epilogue to the
  1536-route/256-slot shape and passed full-scheduler smoke, but served A/B was
  deferred after the strategy pivot.

Tensor parallel variants:

- Not yet implemented in the production scheduler, but now measured with a
  standalone TP split proxy.
- Current runtime is layer-scheduled across the 8 V100s: each GPU owns
  contiguous layers and KV for those layers, and only HC state crosses device
  boundaries.
- The TP split proxy shows ideal 2-way FFN compute speedups of `1.858x` at
  768 routes and `1.468x` at 1536 routes before communication.
- The P2P proxy shows placement matters: NV2 moves 12 MiB in about `0.26 ms`,
  NV1 in about `0.52 ms`, and SYS in about `1.3 ms`.
- My current read is that 2-way TP is worth prototyping on NV2 pairs, while
  8-way expert parallelism is probably underfilled because the compact served
  shape currently has only 6 active expert groups.

## What The Experiments Show So Far

1. Memory fit is not the bottleneck for DS4-Flash on 8x32GB V100, within the
   guarded context/slot modes.
2. Decode throughput is dominated by routed expert execution, especially gate/up
   and down MXFP4 GEMMs.
3. Simple slot widening helps but is now mostly exhausted.
4. Small fixed-shape probes can win microbenchmarks and still fail to move
   served throughput.
5. Kernel fusion helps only when it removes a large enough boundary. Fusing a
   tail kernel or bypassing generic dispatch is too small.
6. The next serious optimization needs either:
   - a persistent/software-pipelined routed-FFN executor, or
   - a tensor/expert-parallel scheduling redesign that creates larger expert
     batches without losing the current layer pipeline overlap.

## Time Estimate

For a credible next performance step, such as a bounded 2-GPU routed-FFN TP
prototype or a DS4-only persistent routed-FFN prototype: I would budget
1-2 focused weeks for a serious prototype and V100 A/B cycle, assuming no
major correctness surprises.

For the original practical target of `1k-2k` aggregate tok/s: I would budget
4-8+ weeks of focused kernel and scheduler work, and I do not think it is
guaranteed on this architecture without a larger tensor/expert-parallel or
persistent-kernel redesign. The current path is correct and increasingly well
measured, but it is still over an order of magnitude away from that target.

## Immediate Next Step

Prototype the smallest production-relevant 2-way TP routed-FFN path:

- constrain placement to NV2 pairs first;
- split the `2048` intermediate dimension across two GPUs;
- overlap half-FFN compute with the hidden payload exchange/reduce;
- validate against the current layer-owned FFN output for one stage before
  attempting a scheduler-wide change.
