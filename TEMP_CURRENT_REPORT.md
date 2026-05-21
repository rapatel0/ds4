# TEMP Current Report

Date: 2026-05-21

## Latest Update

Sprint 154 fully tested the currently implemented fused routed-FFN boundary:
TurboMind fused gate/up + gated-SiLU plus the down-projection route-weighted
reduce epilogue. It is correct, but not a material speedup. At 128-slot/32K,
the down-reduce epilogue was run-noise flat at `59.509317` generated tok/s
versus `59.502747` control. At 256-slot/16K, it was slightly slower at
`60.642962` versus `60.671924` control. Continuation/decode moved the same
way: `55.789985` versus `55.783825` at 128-slot/32K, and `56.852777` versus
`56.879929` at 256-slot/16K.

A synchronized 128-slot full-scheduler profile confirmed why this did not
move the topline: gate/up and down GEMMs still dominate, while the final
scatter/reduce tail is too small to matter at served level. The practical
conclusion is that stage-count tuning and epilogue-only fusion are now both
exhausted. The next implementation needs to change the routed expert execution
model itself, either with a persistent/grouped routed-FFN executor or a narrow
one-layer 2-way TP prototype for the 128-slot/32K NV2 case.

I also moved the tensor-parallel idea one step closer to a real appliance
contract. `tools/ds4-v100-appliance-pack --emit-tp-split` now emits bounded
2-way MXFP4 routed-FFN splits for gate/up and down, and the V100 context binder
accepts those TP expert descriptors while still enforcing normal layer-owner
rules for non-TP tensors. A layer-3, six-expert bounded pack emitted 8
TurboMind descriptors, split `tp1` onto GPU3 from GPU0, and passed context
binding with `turbomind_tensor_count=8`.

The fresh TP evidence is narrow but useful. On the real 2-GPU proxy using NV2
pair `0,3`, 768 routes measured `0.9769 ms` full one-GPU, `0.5612 ms`
concurrent half compute, and `0.8446 ms` total with conservative input/output
copies, for `1.157x` total speedup. At 1536 routes, total-with-copy was slower:
`1.4264 ms` versus `1.3002 ms` full one-GPU. This keeps 2-way TP as a bounded
128-slot/32K candidate only; it is not a broad 256-slot/16K topology answer.

I broadened the fused MXFP4 gate/up+gated-SiLU software-pipeline test into a
2/3/4-stage sweep. The result is now clear: deeper staging inside this single
fused GEMM is not a material lever. At 768 routed rows, `m128`, `m128_s3`, and
`m128_s4` measured `0.5809 ms`, `0.5863 ms`, and `0.5794 ms`. At 1536 routed
rows, `m128_1536`, `m128_s3_1536`, and `m128_s4_1536` measured `0.8743 ms`,
`0.8821 ms`, and `0.8774 ms`. NCU all-GEMM profiling of the 768-route fixed
probe launch also stayed flat: `690.18 us`, `695.30 us`, and `688.77 us`, all
with `50,331,648` HMMA instructions. Stage-count variants should remain
explicit diagnostics only; the next fused-kernel attempt needs to change the
larger routed-FFN boundary.

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

I then built the bounded 2-GPU TP proxy. On clean NV2 pairs, 768-route
concurrent half-FFN compute is about `1.87x` faster and remains about `1.28x`
faster after conservative input/output payload copies. At 1536 routes, compute
speedup is only `1.29-1.46x` and total-with-copy is slower (`0.85-0.94x`).
This narrows TP to an opt-in 128-slot/32K candidate unless we keep hidden state
replicated across TP ranks or overlap the payloads better.

The 2-GPU TP proxy now has a correctness gate. With finite synthetic MXFP4
fixtures, `full_down` matches `half0_down + half1_down` on both clean NV2
pairs for 768 and 1536 routes (`rel ~= 2.46e-04`, `bad=0`, max abs
`6.1035e-05`). The decomposition is correct; the remaining issue is whether a
production scheduler can keep the 768-route payload movement overlapped enough
to preserve the measured speedup.

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
- Sprint 154 completed the missing served A/B for the down-reduce epilogue at
  both 768-route and 1536-route high-slot shapes. It was flat at 128-slot/32K
  and slightly slower at 256-slot/16K, so it stays off by default.

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
- Expanded to a 2/3/4-stage sweep. The 768-route `m128`, `m128_s3`, and
  `m128_s4` fixed probes measured `0.5809 ms`, `0.5863 ms`, and `0.5794 ms`;
  the 1536-route `m128_1536`, `m128_s3_1536`, and `m128_s4_1536` probes
  measured `0.8743 ms`, `0.8821 ms`, and `0.8774 ms`.
- NCU fixed-probe counters were also neutral: about `690-695 us`, `40%` SM
  throughput, `11-12%` DRAM throughput, and identical HMMA instruction count.
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
  standalone TP split proxy and represented in a bounded appliance pack
  contract.
- Current runtime is layer-scheduled across the 8 V100s: each GPU owns
  contiguous layers and KV for those layers, and only HC state crosses device
  boundaries.
- The TP split proxy shows ideal 2-way FFN compute speedups of `1.858x` at
  768 routes and `1.468x` at 1536 routes before communication.
- The P2P proxy shows placement matters: NV2 moves 12 MiB in about `0.26 ms`,
  NV1 in about `0.52 ms`, and SYS in about `1.3 ms`.
- The real 2-GPU proxy shows 768-route total-with-copy speedup of about
  `1.16x-1.28x` on NV2 pairs depending on run, but 1536-route total-with-copy
  is neutral to slower.
- The TP split correctness gate passes for 768 and 1536 routes on clean NV2
  pairs, so the split itself is not the blocker.
- The bounded TP appliance pack emits `ffn_gate_up_exps.tp{0,1}` and
  `ffn_down_exps.tp{0,1}` rows and passes partial context binding.
- My current read is that 2-way TP is worth prototyping only for the
  128-slot/32K route shape first, while 8-way expert parallelism is probably
  underfilled because the compact served shape currently has only 6 active
  expert groups.

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

Do not keep tuning stage count inside the current gate/up fused GEMM. The next
implementation should choose one larger boundary:

- a true routed-FFN fused/persistent executor that combines gate/up, activation,
  down, and route-weighted reduce; or
- the smallest production-relevant 2-way TP routed-FFN path, constrained to one
  routed layer on NV2 pairs and the 128-slot/32K tier first.
