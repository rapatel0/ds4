# DS4 V100 Experiment Status

Last updated: 2026-05-21

## Topline

The appliance is correct and served on the 8x V100 node, but it is not yet in
the practical throughput range from the vision. The current 8-slot default is
the Sprint 111 fused TurboMind gate/up appliance plus the Sprint 115 shared
gate/up SwiGLU F8 HMMA path, Sprint 116 batched attention-projection F8 HMMA
path for active 4/8-slot batches, and Sprint 119 event-ordered handoff for
multi-slot per-step serving. Sprint 121 adds an admitted 16-slot 256K
throughput mode. Sprint 122 stabilizes 16-slot request coalescing by resolving
launcher `auto` microbatch wait to 200 ms at high active-slot counts.
Sprint 123 tested production-path shared-FFN fusion candidates. They were
correct, but stayed opt-in because the best measured candidate did not clear
the promotion bar. Sprint 124 tested a TurboMind route-row reduce that removes
the packed output clear plus atomic scatter-add; it is correct, but also stayed
opt-in because repeat A/B stayed inside run noise. Sprint 125 tested batched
grouped attention output-A; the rows2 variant was correct and slightly faster,
but the gain was about `0.3%`, and the HMMA variant regressed, so defaults are
unchanged. Sprint 126 added a default-off production routed-expert stage
profiler. It confirmed the current binary still serves correctly at
`43.453309` generated tok/s with profiling disabled, and showed that the
separate SwiGLU stage is only about `3.2%` of profiled routed-FFN time.
Sprint 127 added an opt-in TurboMind gated-SiLU ABI and an interleaved
gate/up appliance pack. The path is correct, removes the standalone SwiGLU
profile bucket, and measured `43.933293` generated tok/s, but it is still a
small end-to-end change rather than the persistent expert pipeline needed for
the vision target. Sprint 128 added compact active-expert scheduling for the
packed TurboMind path, promoted it as the launcher default, and raised the
existing fused appliance default to `45.888778` generated tok/s. The best
Sprint 128 opt-in stack, interleaved gated appliance plus compact schedule plus
route-row-reduce, reached `46.394722` generated tok/s. Sprint 129 exposed
TurboMind dispatch policy selection and tested the safe `reuse` policy against
the default compact path. `reuse` was neutral at `45.813841` vs `45.840691`
generated tok/s, and TurboMind `measure` hit a full-appliance measurer fatal,
so dispatch-policy tuning is not the next throughput lever. Sprint 130 reran
the route-row-reduce tail fusion against the current compact fused appliance:
control was `45.837745` generated tok/s and route-row-reduce was `45.660765`,
both with 16/16 token match. Sprint 131 then added an opt-in TurboMind
indexed-A path that uses compact token indices for gate/up and avoids
route-expanded activation scratch. It was correct, but measured `45.789937`
vs `45.663281` control, so it remains opt-in. Sprint 132 extended the
standalone TurboMind gate/up benchmark to the production 96-route shape and
measured the interleaved gated path at `0.1776 ms` vs `0.2889 ms` for separate
gate+up, a `1.626x` isolated speedup. Sprint 133 then corrected the benchmark
to use compact active-expert grouping like the served runtime; at that topology
gated-SiLU was `0.1740 ms` vs `0.1895 ms` separate gate+up, only `1.089x`.
Sprint 134 added a fixed-shape DS4 ABI that directly launches the matching SM70
MXFP4 gated kernel; it was bit-identical and neutral at `0.1746 ms` vs
`0.1746 ms` generic gated. Sprint 135 then widened the short-context served
scheduling shape instead of changing another wrapper boundary: 32-slot 128K
passed full scheduler smoke and served at `52.840889` generated tok/s versus
`45.780913` for the same-context 16-slot control. Sprint 136 then admitted
64 slots at 64K and reached `57.322945` generated tok/s versus `52.884400` for
the same-context 32-slot control. Sprint 137 admitted 128 slots at 32K and
reached `59.598172` generated tok/s versus `57.170428` for the same-context
64-slot control. These results keep pointing the next implementation at
lower-level packed MXFP4 dataflow rather than another launch-boundary,
dispatch, wrapper data-movement tweak, or simple admission-width change.
Sprint 138 widened the compact TurboMind gate/up benchmark defaults to include
192/384/768 routed-row shapes. At 768 routes, the current fused gate_up
baseline is `0.6379 ms` and gated-SiLU is `0.6481 ms`. Sprint 139 added a
fixed-shape 768-route m128 gated-SiLU probe, wired it into the appliance under
exact guards, and validated the interleaved gated appliance at `60.130047`
generated tok/s for 128-slot/32K. The isolated kernel result improved to
`0.5999 ms`, but the served probe-off control was `60.061899`, so the
production effect is tiny. Sprint 140 applied the same fixed-shape strategy to
the 768-route down projection. It improved the isolated down benchmark
(`0.3026 ms` vs `0.3272 ms`) and passed full 43-layer smoke, but served A/B was
slower with the probe enabled (`60.038469` vs `60.129772`), so it remains
default-off. Sprint 141 added a half2-vectorized route-row-reduce tail variant.
It passed full 43-layer 128-slot smoke, but 128-slot served A/B stayed neutral:
control `60.108232`, scalar route-row reduce `60.112248`, and half2 route-row
reduce `60.104512`, all with `128/128` token match. Sprint 142 moved the
weighted route reduction into the TurboMind down GEMM epilogue for the exact
768-route high-slot shape. It passed full 43-layer 128-slot smoke and served
correctly at `60.041003` generated tok/s versus `59.987105` same-binary
control, so it remains default-off because the result is only run-noise
positive. Sprint 143 added explicit prefill versus decode metrics to the soak,
sustained decode, and aggregate throughput harnesses so future experiments can
separate prompt replay, continuation decode, and aggregate generated rates.

| Track | Context | Slots | Best Generated tok/s | Current Default Generated tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Short-context max-throughput target | 32,768 | 128 | `60.130047` | opt-in | 128/128 token match |
| Short-context high-throughput target | 65,536 | 64 | `57.322945` | opt-in | 64/64 token match |
| Short-context throughput target | 131,072 | 32 | `52.840889` | opt-in | 32/32 token match |
| Throughput serving target | 262,144 | 16 | `46.394722` | `45.888778` | 16/16 token match |
| 8-slot compatibility target | 262,144 | 8 | `34.689964` | `34.490294` | 8/8 token match |
| Long-context target | 1,048,576 | 4 | `21.771077` | `21.771077` | 4/4 token match |

The older `20.249531` long-context result used the Sprint 108 small-route build
path, but that path is not the default because the 8-slot/256K A/B was neutral
to slightly worse.

## Tested

- Full 8-GPU resident appliance pack on `gpu-01` using k8s-local `/workspace`.
- Full 43-layer scheduler over the TurboMind appliance pack.
- Selected-token oracle for the official short prompt, expected text hex
  `3136`, selected token id `926`.
- HTTP served soak benchmarks at:
  - `ctx=32768`, `slots=128`, `active_microbatch=128`, 16 generated tokens.
  - `ctx=65536`, `slots=64`, `active_microbatch=64`, 16 generated tokens.
  - `ctx=131072`, `slots=32`, `active_microbatch=32`, 16 generated tokens.
  - `ctx=262144`, `slots=8`, `active_microbatch=8`, 16 generated tokens.
  - `ctx=262144`, `slots=16`, `active_microbatch=16`, 16 generated tokens.
  - `ctx=1048576`, `slots=4`, `active_microbatch=4`, 16 generated tokens.
- MTP exact commit path was previously validated, but it is not the current
  throughput default because exact verification did not improve tok/s.
- Copied TurboMind MXFP4 kernels are in the production appliance path for
  routed experts; copied tc-grid INT8 kernels remain proof artifacts, not the
  selected source-quality path.

## Recent Experiment Results

| Sprint | Change | Result | Decision |
|---|---|---|---|
| 103 | Exact-bit E4M3 F8 decode | Raised 8-slot/256K to `30.862791` | Shipped |
| 104 | Warp reductions for F8 arena kernels | Raised 8-slot/256K repeat to `31.451185` | Shipped |
| 105 | BF16/F32 warp reductions | Correct but no gain | Rejected |
| 106 | Warm served `nvprof` profile | F8 rows2/grouped rows2 ~51% GPU time; TurboMind ~25% | Used for targeting |
| 107 | DS4 grouped F8 attention-output kernel | Best 8-slot/256K `31.811137` | Shipped/default |
| 108 | TurboMind small-route build fusion | Correct; `31.759013` opt-in vs `31.794180` rollback on repeat | Kept opt-in |
| 109 | F8 row4 CTA probe | Correct; `30.998275` row4 vs `31.380225` control at 8-slot/256K | Rejected as default |
| 110 | TurboMind fused gate/up grouped-GEMM probe | Correct; fused gate_up was `1.46x-1.53x` faster than separate gate and up calls | Proceed to appliance implementation |
| 111 | Production fused TurboMind gate_up appliance | Correct; `33.430971` fused vs `31.312694` same-binary separate control at 8-slot/256K; `21.403909` at 4-slot/1M | Shipped/default for fused packs |
| 112 | Fused appliance profile and F8 warp-scale probe | F8 row-pair/grouped kernels were `54.58%` GPU time; warp-scale was correct but `29.009399` vs `33.484099` control at 8-slot/256K | Kept opt-in/off |
| 113 | Direct FFN delta accumulation | Correct; `33.360404` direct delta vs `33.589285` control at 8-slot/256K | Kept opt-in/off |
| 114 | Shared-down F8 HMMA batch kernel | Correct; `33.550415` HMMA vs `33.397763` control at 8-slot/256K, and `21.396331` vs `21.365610` at 4-slot/1M | Kept opt-in/off |
| 115 | Shared gate/up SwiGLU F8 HMMA batch kernel | Correct; `33.578236` HMMA vs `33.292541` control at 8-slot/256K, and `21.455638` vs `21.430420` at 4-slot/1M | Shipped/default |
| 116 | Batched attention projection F8 HMMA kernel | Correct; `33.697698` HMMA batch vs `33.380614` control at 8-slot/256K, and `21.469010` vs `21.333447` at 4-slot/1M | Shipped/default for 4/8-slot batches |
| 117 | F8 shape trace, async chunk probe, and per-slot shared gate/up/SwiGLU fusion | Trace showed the fast served path is per-slot stage-pipelined; `DS4_V100_ASYNC_SLOT_CHUNK=4` was correct but only `11.483646`; single shared pair-SwiGLU was correct at `33.562643` vs `33.697698` default | Keep opt-in/off; next fusion must be software-pipelined/HMMA, not just scalar launch reduction |
| 118 | Single-token HMMA for the hot `4096 x 8192` F8 projection | Correct and traced as `plain/hmma_single`, but `16.083451` vs `33.502249` same-binary control at 8-slot/256K | Keep opt-in/off; do not broaden single-token WMMA |
| 119 | Event-ordered stage handoff for per-step multi-slot serving | Correct; `34.433252` vs `33.379839` at 8-slot/256K and `21.771077` vs `21.566859` at 4-slot/1M | Shipped/default as `DS4_V100_ASYNC_EVENT_HANDOFF=auto` |
| 120 | Single-token shared gate/up/SwiGLU row-pair fusion | Correct; current default repeat was `34.490294`, scalar single fusion was `34.689964`, and row-pair single fusion was `34.380968` at 8-slot/256K | Keep opt-in/off; row-pair compaction is not the missing kernel lever |
| 121 | 16-slot 256K throughput mode | Correct; `43.659461` at 16-slot/256K vs `34.445844` same-binary 8-slot control | Keep as admitted 256K throughput mode; reject unsafe 16-slot long-context configs |
| 122 | 16-slot profile, HMMA admission, async chunk probes, and 16-slot rendezvous policy | Correct; best `43.730215`, production-auto `43.534061`, one 16-request tensor batch after 200 ms auto wait; chunked tensor scheduling regressed (`28.876459` at chunk 2, `18.447169` at chunk 4, `13.315378` at chunk 16) | Ship 16-slot auto rendezvous; keep chunk/output-B probes opt-in/off |
| 123 | Production-path shared FFN fusion A/B | Correct; scalar shared-pair fusion reached `43.887206`, fused shared-down-add reached `43.539555`, and combined scalar+down-add reached `43.812630` at 16-slot/256K | Keep opt-in/off; small shared-FFN launch/epilogue fusion is not enough |
| 124 | TurboMind route-row reduce | Correct; first candidate reached `43.822500`, but repeat was `42.998450` vs `43.517862` control repeat at 16-slot/256K | Keep opt-in/off; final routed scatter fusion is not enough |
| 125 | Batched grouped attention output-A | Correct; rows2 output-A batching reached `43.640921`, rows2 A+B reached `43.619996`, and HMMA A+B reached `43.245208` vs `43.503005` control at 16-slot/256K | Keep opt-in/off; single projection batching is too small |
| 126 | Routed-expert stage profiler | Correct; full 43-layer profile showed gate/up `47.0%`, down `23.4%`, route build `16.8%`, gather `3.7%`, SwiGLU `3.2%`, scatter `4.6%` of profiled routed-FFN time; no-profile served sanity was `43.453309` | Ship default-off diagnostic; use it to target larger TurboMind/persistent expert work |
| 127 | TurboMind gated-SiLU interleaved pack | Correct; standalone gated grouped path was `1.47x-1.55x` faster than separate gate/up, full 43-layer gated profile removed standalone SwiGLU and reduced profiled routed-FFN total from `28.242 ms` to `26.734 ms`; served A/B was `43.933293` vs `43.691032` control | Keep opt-in/off; confirms format-aware epilogue fusion, but does not materially change the topline |
| 128 | TurboMind compact active-expert schedule | Correct; full smokes passed on both the interleaved gated and existing fused appliances, compact served A/B was `46.328184` vs `43.879880`, compact+route-row-reduce reached `46.394722`, and the fused-appliance launcher default reached `45.888778` | Ship/default as `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`; keep gated-SiLU and route-row-reduce opt-in |
| 129 | TurboMind dispatch policy probe | Correct for `default` and `reuse`; `reuse` served at `45.813841` vs `45.840691` default, while unsafe `measure` aborted the full scheduler in TurboMind's measurer | Keep `default`; block `measure`/`append` unless `DS4_V100_TURBOMIND_ALLOW_UNSAFE_MEASURE=1`; move to persistent routed-FFN work |
| 130 | Routed FFN software-pipeline targeting | Correct; route-row-reduce repeated at `45.660765` vs `45.837745` compact fused control, and TurboMind/tc-grid review points to packed MXFP4 load/dequant/HMMA pipelining as the useful fusion boundary | Keep route-row-reduce opt-in; implement a guarded DS4-specific routed gate/up software-pipeline probe |
| 131 | TurboMind indexed-A routed activation probe | Correct; avoids route-expanded activation materialization for gate/up, passed full 43-layer smokes, and served at `45.789937` vs `45.663281` control | Keep indexed-A opt-in; wrapper-level activation compaction is not enough |
| 132 | Production-shaped TurboMind gate/up benchmark | Correct; added env-selectable cases and the served-profile 96-route case, where gated-SiLU measured `0.1776 ms` vs `0.2889 ms` separate gate+up | Use as the 1-GPU V100 acceptance harness for lower-level routed-expert kernel work |
| 133 | Compact-group gate/up benchmark correction | Correct; compact 96-route gated-SiLU measured `0.1740 ms`, while compact separate gate+up was `0.1895 ms`; sparse256 overstated fusion benefit at `1.534x` | Use compact mode as the acceptance baseline; sparse-group wins do not predict served default wins |
| 134 | Fixed-shape compact gate/up ABI probe | Correct; direct fixed SM70 launch was bit-identical to generic gated and measured `0.1746 ms` vs `0.1746 ms` | Do not promote; dispatch bypass is not the missing lever |
| 135 | 32-slot 128K throughput admission | Correct; full 43-layer smoke passed, and 32-slot 128K served at `52.840889` vs `45.780913` same-context 16-slot control | Ship as explicit short-context throughput mode; evaluate 64-slot short context and deeper software-pipelined expert kernels next |
| 136 | 64-slot 64K throughput admission | Correct; full 43-layer smoke passed, and 64-slot 64K served at `57.322945` vs `52.884400` same-context 32-slot control | Ship as explicit short-context throughput mode; slot scaling helps but shows diminishing returns |
| 137 | 128-slot 32K throughput admission | Correct; full 43-layer smoke passed, status/metrics confirmed 128-slot serving, and 128-slot 32K served at `59.598172` vs `57.170428` same-context 64-slot control | Ship as explicit short-context throughput mode; simple slot widening is now mostly exhausted |
| 138 | Wide compact TurboMind gate/up benchmark | Correct; 192/384/768 route compact cases pass, with 768-route fused gate_up at `0.6379 ms` and gated-SiLU at `0.6481 ms` | Use as the software-pipelined MXFP4 expert acceptance baseline |
| 139 | Fixed-shape 128-slot gate/up probe | Correct; m128 measured `0.5999 ms` vs `0.6480 ms` generic gated in isolation, full 43-layer 128-slot smoke passed, and served gated A/B was `60.130047` vs `60.061899` probe-off | Keep exact-guard auto path; move next work to a larger routed-FFN boundary |
| 140 | Fixed-shape 128-slot down probe | Correct; fixed down measured `0.3026 ms` vs `0.3272 ms` generic and full 43-layer smoke passed, but served A/B was `60.038469` vs `60.129772` down-probe-off | Keep off by default; target down epilogue plus weighted reduce or persistent routed FFN next |
| 141 | Half2 route-row reduce tail probe | Correct; full 43-layer 128-slot smoke passed, but half2 route-row reduce was `60.104512` vs scalar route-row reduce `60.112248` and control `60.108232` | Keep off by default; separate tail vectorization does not move served throughput |
| 142 | TurboMind down-epilogue reduce probe | Correct; full 43-layer 128-slot smoke passed and served A/B was `60.041003` vs `59.987105` same-binary control | Keep off by default; the atomic epilogue reduce is correct but not a material throughput win |
| 143 | Prefill/decode metric split | Correct; one-request V100 smoke emitted aggregate prompt, generated, and continuation tok/s plus response-local prompt/decode rates | Ship benchmark visibility change; use split metrics in future A/B decisions |

## Remaining

- Close the throughput gap. The current best `~60` tok/s aggregate is far below the
  `~1k-2k` practical target discussed in the vision.
- Improve GPU utilization. The latest profile says the bottleneck is device
  kernel shape/occupancy, not disk, host RAM, or bulk PCIe/NVLink traffic.
- Attack larger hot-path buckets instead of small host-side route plumbing:
  - TurboMind MXFP4 expert occupancy and route-expanded activation layout.
  - Persistent/grouped expert execution beyond the shipped Sprint 111 fused
    gate_up launch reduction, Sprint 127 gated-SiLU epilogue fusion, and
    Sprint 128 compact active-expert scheduling. Sprint 128 is the first recent
    routed-expert scheduler change to clear the run band, but the topline is
    still two orders of magnitude below the vision target.
  - A larger software-pipelined F8/attention-output/FFN rewrite. Sprint 117
    showed scalar per-slot shared-FFN fusion removes calls but does not improve
    throughput, and Sprint 118 showed naive single-token WMMA is much slower.
    Sprint 123 showed shared-down-add epilogue fusion is also only a small
    opt-in gain, and Sprint 124 showed the packed TurboMind route-row reduce is
    correct but not a promoted throughput win. Sprint 125 showed batched
    grouped attention output-A is also correct but below the promotion bar.
    Sprint 129 showed TurboMind dispatch policy tuning is either neutral
    (`reuse`) or unsafe in the full appliance path (`measure`).
    Sprint 130 repeated the current route-row-reduce tail fusion and found it
    slightly slower than the compact fused control. Sprint 131 reduced
    gate/up activation materialization with TurboMind indexed-A, but remained
    inside run noise. Sprint 132 showed the existing interleaved gated gate/up
    primitive is `1.626x` faster than separate gate/up at the sparse 96-route
    standalone shape, but Sprint 133 showed the served compact topology shrinks
    that to `1.089x`. Sprint 134 showed direct fixed-shape dispatch is neutral.
    The useful version therefore needs either a lower-level packed
    decode/activation staging/MMA/epilogue specialization that beats the compact
    `0.1740 ms` baseline, or a scheduler that keeps expert work larger than the
    current compact microshape. Sprints 135-136 confirm that wider served
    scheduling does help: 32-slot 128K reached `52.840889`, 64-slot 64K
    reached `57.322945`, and 128-slot 32K reached `59.598172`, but the
    marginal gain is shrinking and all remain far below the practical vision
    target. Sprint 138 set the next kernel acceptance target: beat the compact
    768-route fused gate_up baseline of about `0.638 ms`. Sprint 139 beat that
    target in isolation with a fixed m128 gated-SiLU probe at `0.5999 ms`, but
    served A/B only moved from `60.061899` to `60.130047`, so gate/up-only
    specialization is not enough. Sprint 140 repeated the experiment for down:
    the isolated fixed down probe improved to `0.3026 ms`, but served A/B
    regressed from `60.129772` to `60.038469`. Sprint 141 vectorized the
    separate route-row reduce tail with half2, but the 128-slot A/B stayed in
    the same band as control. Sprint 142 moved that reduce into the down GEMM
    epilogue, but the served result was only `60.041003` vs `59.987105`
    control. Even a correct weighted-reduce epilogue is too small in this
    atomic form.
    Sprint 122 further showed that merely chunking slots to feed wider kernels
    loses too much stage overlap, so the fusion target must match the per-slot
    served topology or replace it with an overlapped scheduler.
- Decide whether the next production step is a deeper TurboMind adapter change
  or a lower-level CUTLASS/TurboMind-inspired persistent kernel probe.

## Operator Status

The default launcher now keeps `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=0`,
`DS4_V100_CUDA_F8_ROW4=0`, `DS4_V100_CUDA_F8_WARP_SCALE=0`, and
`DS4_V100_FFN_DIRECT_DELTA=0`, while
`DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`,
`DS4_V100_TURBOMIND_GATE_UP_PROBE=auto`,
`DS4_V100_TURBOMIND_DOWN_PROBE=off`,
`DS4_V100_CUDA_F8_HMMA_PAIR_SWIGLU=1`,
`DS4_V100_ENABLE_BATCH_ATTN_PROJ=1`, and
`DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=1` are default.
`DS4_V100_ASYNC_EVENT_HANDOFF=auto` enables event-ordered handoff for
multi-slot per-step serving and resolves off for one-slot latency configs.
`ctx=32768` can now admit 128 slots, `ctx=65536` can admit 64 slots,
`ctx=131072` can admit 32 slots, and `ctx=262144` remains capped at 16 slots;
the launcher rejects over-cap configs before allocation.
`DS4_V100_MICROBATCH_WAIT_US=auto` resolves to 200 ms when
`DS4_V100_ACTIVE_MICROBATCH >= 16` so bursty high-slot clients form one tensor
batch. The opt-in diagnostic paths can be enabled, or defaults rolled back,
with:

```text
DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD=1
DS4_V100_CUDA_F8_ROW4=1
DS4_V100_CUDA_F8_WARP_SCALE=1
DS4_V100_CUDA_F8_HMMA_SHARED_DOWN=1
DS4_V100_ENABLE_BATCH_ATTN_PROJ=0
DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=0
DS4_V100_FFN_DIRECT_DELTA=1
DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE=1
DS4_V100_CUDA_F8_HMMA_SINGLE=1
DS4_V100_TURBOMIND_FUSED_GATE_UP=0
DS4_V100_ASYNC_EVENT_HANDOFF=0
DS4_V100_CUDA_F8_PAIR_SWIGLU_SINGLE_ROWS2=1
DS4_V100_F8_SHARED_DOWN_ADD=1
DS4_V100_BATCH_ATTN_OUTPUT_B=1
DS4_V100_ASYNC_SLOT_CHUNK=2
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE=1
DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=1
DS4_V100_TURBOMIND_INDEXED_A=1
DS4_V100_BATCH_ATTN_OUTPUT_A=1
DS4_V100_CUDA_F8_HMMA_GROUPED_ATTN_O_BATCH=1
DS4_V100_TURBOMIND_PROFILE=1
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=0
DS4_V100_TURBOMIND_GATE_UP_PROBE=off
DS4_V100_TURBOMIND_DOWN_PROBE=auto
DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1
```

The fused gate/up path is default-enabled for appliances that contain fused
`ffn_gate_up_exps.weight` tensors. Set `DS4_V100_TURBOMIND_FUSED_GATE_UP=0`
only when the appliance also contains separate gate/up tensors. The gated-SiLU
path additionally requires an appliance packed with `--fuse-gate-up-interleaved`
and the Sprint 127 TurboMind ABI; do not enable it against the Sprint 111
`[all gate][all up]` fused pack.
Compact scheduling is default-on after Sprint 128; set
`DS4_V100_TURBOMIND_COMPACT_SCHEDULE=0` only to roll back to the old 256-expert
grouped schedule.
The fixed-shape m128 gate/up probe is default-auto after Sprint 139, but only
selects on the exact interleaved gated 768-route compact shape. Set
`DS4_V100_TURBOMIND_GATE_UP_PROBE=off` to force the generic TurboMind gated
path.
The fixed-shape m128 down probe is available after Sprint 140, but stays
default-off because served A/B was slower than generic TurboMind. Set
`DS4_V100_TURBOMIND_DOWN_PROBE=auto` only for focused diagnostics.
`DS4_V100_TURBOMIND_ROUTE_ROW_REDUCE_H2=1` is available after Sprint 141, but
stays default-off because served A/B was neutral against both scalar
route-row-reduce and control.
`DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` is available after Sprint 142, but
stays default-off because the fused atomic epilogue reduce was only run-noise
positive in served A/B.
