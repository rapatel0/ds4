# TEMP Current Report

Date: 2026-05-21

## Latest Update

After this report was first written, I tested a true stage-count software
pipeline variant on the fused MXFP4 gate/up+gated-SiLU kernel. The 768-route
`m128_s4` probe improved the isolated benchmark (`0.5811 ms` vs `0.6033 ms`
for `m128`) and passed full 43-layer smoke, but served A/B was only
`60.049057` generated / `56.295991` decode tok/s versus `59.865668` /
`56.124063` control. The full-scheduler profile did not show a reliable
gate/up bucket reduction. I would keep it opt-in and move the main effort to a
larger routed-FFN executor boundary or a TP/EP microbenchmark.

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

- Partly explored, but not fully implemented as a new DS4 persistent executor.
- TurboMind's SM70 MXFP4 kernels already use the relevant style internally:
  packed low-bit load, dequant/staging, and Volta HMMA.
- We have not yet written a new end-to-end DS4-only persistent routed-FFN kernel
  that software-pipelines gate/up, activation, down, and reduce as one larger
  unit.
- Sprint 147 is currently starting a smaller step in that direction: extending
  the down+weighted-reduce epilogue to the 1536-route/256-slot shape. This can
  remove a large `down_routes` materialization plus follow-up scatter read, but
  it is still not the full persistent executor.

Tensor parallel variants:

- Not yet implemented or benchmarked.
- Current runtime is layer-scheduled across the 8 V100s: each GPU owns
  contiguous layers and KV for those layers, and only HC state crosses device
  boundaries.
- We have discussed tensor/expert-parallel layouts, but no true tensor-parallel
  runtime variant has been built or tested yet.
- My current read is that tensor/expert parallelism may be necessary for a
  large jump, but it is also a larger architecture change than the current
  appliance path.

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

For the current Sprint 147 down-reduce extension: likely hours to one day to
finish correctness and served A/B.

For a credible next performance step, such as a DS4-only persistent routed-FFN
prototype: I would budget 1-2 focused weeks for a serious prototype and V100
A/B cycle, assuming no major correctness surprises.

For the original practical target of `1k-2k` aggregate tok/s: I would budget
4-8+ weeks of focused kernel and scheduler work, and I do not think it is
guaranteed on this architecture without a larger tensor/expert-parallel or
persistent-kernel redesign. The current path is correct and increasingly well
measured, but it is still over an order of magnitude away from that target.

## Immediate Next Step

Finish Sprint 147:

- validate the 1536-route down-reduce epilogue on the full 43-layer scheduler;
- run served 256-slot/16K A/B with prefill and decode split;
- keep it opt-in unless decode throughput moves materially;
- if it is neutral, stop tuning individual GEMM/epilogue variants and move to a
  larger persistent routed-FFN or tensor/expert-parallel experiment.
