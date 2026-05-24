# DS4 V100 Experiment Status

Last updated: 2026-05-23

## Topline

Current active implementation track is TP8/EP8 only. By Sprint 303 the
diagnostic `/v1/completions` path accepts tokenized prompts, performs
correctness-oriented prompt prefill, runs multi-token autoregressive
output-head/sample/feed, persists resident session state, and returns
`generated_token_sequence` plus `slot_position`. The latest V100 smoke at
`32` slots / `256K`, `prompt_tokens=[31,32,33]`, `max_tokens=3`, returns
`[127885,57114,78026]`, advances the resident cursor to `100005`, and reports
`214.100724` wall tok/s / `353.667490` decode tok/s for the generated section.
Sprint 304 adds the same token-ID serving contract to `/v1/chat/completions`;
the V100 smoke returns `object=chat.completion` with matching
`choices[0].token_ids` and `ds4_v100.generated_token_sequence` of
`[0,57085,104170]`, at `210.355981` wall tok/s / `350.653125` decode tok/s.
Tokenizer text I/O, active-slot-only decode, optimized batched prefill, exact
DS4 HC parity, and MTP remain open.

Current promoted serving baseline is Sprint 199's graph-backed
`fused6_reduce` production pack at 16-slot/256K: `67.886268` generated tok/s
and `66.825545` continuation tok/s with `16/16` token match. Sprint 201 adds a
bounded full-layer TP4 boundary measurement: the 43-layer, 4-collective/layer
proxy costs `22-24 ms` at 16 active tokens (`655-724 tok/s` overhead-only),
`34.830881 ms` at 64 tokens (`1837 tok/s` overhead-only), and `51.026125 ms`
at 128 tokens (`2509 tok/s` overhead-only). TP4 is still worth exploring only
as a broad full-layer TP/EP topology, not as another routed-only overlay.
Sprint 202 confirms that interpretation from the compute side, after fixing a
benchmark warmup bug where the full reference and shard 0 shared the GPU0
TurboMind workspace on different streams. Corrected real TurboMind MXFP4 TP4
routed-FFN compute reaches `2.350x-3.636x` speedup at 96-768 routes, but
conservative full-hidden copies reduce those same cases to `0.783x-0.682x`.
Sprint 203 implements the first resident TP4 layer-slice gate. It is correct,
but the naive resident boundary is not fast enough: `96 routes x 43 layers`
measured `0.825x` versus one GPU, and `768 routes x 43 layers` measured
`0.589x`. Production TP4 scheduler work should wait for a real concurrent
collective or fused reduction boundary.
Sprint 204 added that first concurrent boundary using per-device async
doubling. It helps larger shapes (`1.071x` at `768 routes x 43 layers`) but
does not reliably clear the production decode shape (`0.896x` on the longer
`96 routes x 43 layers` repeat).
Sprint 205 tested async root gather/reduce/broadcast and rejected it: the
96-route 43-layer gate was `0.860x`.

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
Sprint 144 added explicit SM70 MXFP4 `m64n256` probes for the 768-route
gate/up and down shapes. Both were correct, but served 128-slot/32K A/B
regressed versus control, so they remain opt-in test hooks only. Sprint 145
added a guarded 256-slot/16K admission tier. It passed planner, full 43-layer
smoke, and served correctness, reaching `61.065087` generated tok/s and
`57.248519` continuation/decode tok/s. The lift over the 128-slot/16K control
was only about 2%, so the practical ceiling is now clearly a routed-expert
execution problem rather than a simple slot-count problem. Sprint 146 added
explicit 1536-route fixed-shape gate/up and down probes for the 256-slot
compact routed shape. The gate probe was correct and slightly faster in
isolation (`0.9435 ms` vs `0.9651 ms` generic gated), but served A/B was flat
to slightly worse: `61.204203` generated tok/s and `57.378940`
continuation/decode tok/s versus `61.223893` and `57.397400` control. The
1536-route probes therefore remain explicit opt-ins and are not selected by
`auto`. Sprint 147 extended the down-reduce epilogue to that 1536-route shape
and proved full-scheduler correctness, but served A/B was deferred after the
strategy pivot to larger fused-kernel work. Sprint 148 tested deeper SM70
software pipelining in the fused MXFP4 gate/up+gated-SiLU kernel. The
768-route `m128_s4` probe improved the standalone probe (`0.5811 ms` vs
`0.6033 ms` for `m128`) and passed full 43-layer smoke, but served A/B was
only `60.049057` generated / `56.295991` continuation tok/s versus
`59.865668` / `56.124063` control. The profile did not show a reliable
gate/up bucket reduction, so stage-4 probes remain explicit opt-ins.
Sprint 149 then measured the first TP topology proxy. Splitting the routed-FFN
middle dimension from `2048` into two `1024` halves shows ideal 2-way compute
speedups of `1.858x` at 768 routes and `1.468x` at 1536 routes before
communication. P2P payload timing shows placement matters: a 12 MiB hidden
payload moves in about `0.26 ms` over NV2, `0.52 ms` over NV1, and
`1.29-1.31 ms` over SYS. This justifies a bounded 2-GPU TP prototype on NV2
pairs, but not an immediate 8-way scheduler rewrite. Sprint 150 built that
2-GPU proxy. Clean NV2 pairs delivered about `1.87x` concurrent compute
speedup and `1.28x` total-with-copy speedup at 768 routes, but the 1536-route
shape was neutral to slower after copies (`0.85-0.94x`). TP is therefore a
targeted 128-slot/32K candidate, not a blanket fix for the current 256-slot
ceiling. Sprint 151 added full-vs-split correctness to the same proxy. With
finite MXFP4 fixtures, both clean NV2 pairs pass at 768 and 1536 routes with
`rel ~= 2.46e-04`, `bad=0`, and max absolute difference `6.1035e-05`.
Sprint 152 completed the 2/3/4-stage fused gate/up software-pipeline sweep and
found stage count neutral. Sprint 153 then added a bounded TP appliance-pack
contract: `--emit-tp-split` emits split gate/up and down TurboMind descriptors,
the context binder accepts those TP expert rows, and a layer-3 GPU0/GPU3
bounded pack passed partial context smoke. The real two-GPU NV2 proxy measured
`1.157x` total-with-copy speedup at 768 routes but `0.912x` at 1536 routes, so
TP remains a narrow 128-slot/32K candidate only. Sprint 154 closed the served
A/B for the fused down-reduce boundary at both current high-slot shapes:
128-slot/32K was run-noise flat (`59.509317` vs `59.502747` generated tok/s),
and 256-slot/16K was slightly slower (`60.642962` vs `60.671924`). A
synchronized 128-slot profile still showed gate/up and down GEMMs dominating,
so epilogue-only down-reduce fusion remains explicit opt-in only. Sprint 155
implemented an opt-in stream-per-expert routed-FFN pipeline for the actual
non-interleaved fused gate/up appliance pack. It proved active on V100 with
`group_pipeline_calls=6` in a profiled stage smoke, but served throughput
regressed: `59.125703` vs `59.394915` generated tok/s at 128-slot/32K and
`60.308689` vs `60.648138` at 256-slot/16K. Sprint 156 found that the manual
exact six-group version is slightly positive in the deterministic served
benchmark, but it is not production-safe because arbitrary traffic can activate
more groups. A safe auto-group implementation passed full scheduler smoke but
regressed served throughput due to host active-group readback. The group
pipeline remains diagnostic-only. Sprint 157 added an opt-in CUDA Graph replay
probe around the TurboMind routed-FFN core. It built and passed full scheduler
smoke, but served graph capture failed before replay on the current
legacy-default-stream kernel path. The 128-slot/32K graph probe was correct but
slower than control (`59.450666` / `55.734999` generated/decode tok/s with
global capture, `59.367233` / `55.656781` with thread-local capture, both with
zero captures). Graph replay remains diagnostic-only unless the routed-FFN
executor is rewritten around an explicit stream.
Sprint 158 added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR` with a guarded fixed96
gate_up executor. The full 43-layer 16-slot/256K scheduler smoke selected the
fixed kernel at `total_routes=96`, but the HTTP served path reached routed FFN
as `total_routes=6` per request, so fixed96 did not fire in served A/B. The
guarded opt-in was neutral (`46.167311` generated / `43.281854` continuation
tok/s) versus control (`46.113721` / `43.231614`) with `16/16` token match.
This keeps fixed96 as an executor probe and shifts the next experiment toward
served batch formation at `>=256K` or TP/EP topology that creates denser
executor shapes.

| Track | Context | Slots | Best Generated tok/s | Current Default Generated tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Short-context ceiling target | 16,384 | 256 | `61.223893` | opt-in | 256/256 token match |
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
  - `ctx=16384`, `slots=128/192/256`, matching active microbatch, 16 generated tokens.
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
| 205 | Async root resident TP4 reduction | Correct on V100; root_async speedups were `0.970x` at 96 routes x 4 layers, `0.866x` at 768 routes x 4 layers, and `0.860x` at 96 routes x 43 layers | Reject root_async and pause the TP4 production decode branch; pivot to persistent fused routed-FFN |
| 204 | Concurrent resident TP4 reduction | Correct on V100 with `doubling_async`; 43-layer 768-route speedup was `1.071x`, while the longer 43-layer 96-route repeat was `0.896x` | TP4 remains plausible for larger batched/prefill shapes, but not ready for production decode scheduler integration |
| 203 | Resident TP4 layer-slice gate | Correct on V100 at 6/96/768 routes; 43-layer root speedups were `0.825x` at 96 routes and `0.589x` at 768 routes; hand-rolled doubling was slower than root in 4-layer tests | Do not wire this TP4 boundary into production; next TP work needs a real concurrent collective/fused reduction, otherwise pivot back to persistent fused routed-FFN |
| 202 | TP4 routed-FFN compute envelope | Correct on V100 after fixing a benchmark stream/workspace overlap; corrected 6/96/768-route compute-only speedups were `2.686x`, `2.350x`, `3.636x`; copy-inclusive speedups were `0.986x`, `0.783x`, `0.682x` | TP4 expert compute is worth pursuing only inside a full-layer resident TP/EP topology; reject routed-only TP overlay expansion |
| 201 | TP4 full-layer boundary proxy | Correct on V100; 16-token default verified at `22.113369 ms` root and `24.414061 ms` doubling for the full 43-layer boundary; larger doubling cases reached `1837` overhead-only tok/s at 64 tokens and `2509` at 128 tokens | Use only for a full-layer TP4/EP prototype that keeps dense+routed compute inside the boundary; do not expand routed-only TP overlays |
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
| 144 | SM70 MXFP4 m64n256 tile probe | Correct; standalone down improved slightly (`0.2896 ms` vs `0.2936 ms`), but served A/B regressed: control `59.993301`, down `m64n256` `59.791839`, gate `m64n256` `59.797232` | Keep opt-in only; do not promote individual tile tweaks without served wins |
| 145 | 256-slot 16K short-context admission | Correct; planner worst GPU was `29.07 GiB / 32.00 GiB` including reserve, full 43-layer smoke passed, and served 16K runs reached `59.860493` at 128 slots, `60.700926` at 192 slots, and `61.065087` at 256 slots | Ship guarded 256-slot admission for `ctx <= 16K`; simple slot widening is now mostly exhausted |
| 146 | 1536-route fixed-shape gate/up and down probes | Correct; standalone gate `m128_1536` improved to `0.9435 ms` vs `0.9651 ms`, but served A/B was `61.204203` vs `61.223893` control and continuation/decode was `57.378940` vs `57.397400` | Keep explicit opt-in only; do not select 1536 probes from `auto` |
| 147 | 1536-route down-reduce epilogue | Correct; full 43-layer 256-slot smoke passed with `DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` | Keep explicit opt-in only; served A/B deferred for larger fused-kernel work |
| 148 | SM70 stage-4 fused gate/up software-pipeline probe | Correct; `m128_s4` improved the isolated 768-route probe to `0.5811 ms` vs `0.6033 ms`, but served A/B was only `60.049057` vs `59.865668` control and profile stayed neutral | Keep explicit opt-in only; stage count alone is not the material fused-kernel lever |
| 149 | TP split and P2P topology probe | Correct; ideal 2-way FFN compute speedup was `1.858x` at 768 routes and `1.468x` at 1536 routes, with 12 MiB NV2 payloads around `0.26 ms` | Prototype only on NV2 pairs; do not start with 8-way TP/EP |
| 150 | Two-GPU TP split timing proxy | Correct; 768-route NV2 total-with-copy speedup was about `1.28x`, while 1536 routes were neutral to slower (`0.85-0.94x`) | Candidate only for 128-slot/32K first |
| 151 | Two-GPU TP correctness gate | Correct; full one-GPU routed-FFN output matches FP32 sum of TP partial outputs with `rel ~= 2.46e-04`, `bad=0` | Split math is valid; remaining risk is production scheduling and payload overlap |
| 152 | 2/3/4-stage fused gate/up software-pipeline sweep | Correct; 768-route `m128/m128_s3/m128_s4` measured `0.5809/0.5863/0.5794 ms`, 1536-route `m128_1536/m128_s3_1536/m128_s4_1536` measured `0.8743/0.8821/0.8774 ms`, and NCU fixed-probe HMMA counts were identical | Do not spend more effort on gate/up stage count; next fusion must cover a larger routed-FFN boundary or use bounded TP |
| 153 | Bounded TP pack contract | Correct; `--emit-tp-split` emitted `ffn_gate_up_exps.tp{0,1}` and `ffn_down_exps.tp{0,1}` rows across GPU0/GPU3, partial context binding passed, and the real two-GPU NV2 proxy was `1.157x` total-with-copy at 768 routes but `0.912x` at 1536 routes | Keep TP scoped to a one-layer 128-slot/32K prototype; default runtime remains layer-sharded |
| 154 | Fused routed-FFN boundary validation | Correct; down-reduce epilogue served at `59.509317` vs `59.502747` control for 128-slot/32K and `60.642962` vs `60.671924` control for 256-slot/16K | Keep off by default; epilogue-only fusion is too small, and the next lever must change routed expert execution |
| 155 | Host stream-per-expert routed-FFN pipeline | Correct and active, but served A/B regressed at both 128-slot/32K and 256-slot/16K | Keep diagnostic-only |
| 156 | Exact six-group and safe auto-group pipeline validation | Exact six-group diagnostic was slightly positive; safe auto-group was correct but slower due to host readback | Keep diagnostic-only; do not promote unsafe hardcoded group count |
| 157 | Routed-FFN CUDA Graph replay probe | Builds and passes scheduler smoke, but served capture fails with zero graph captures; 128-slot/32K graph candidate was slower than control | Keep diagnostic-only; explicit stream plumbing or persistent FFN executor needed |

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
    reached `57.322945`, 128-slot 32K reached `59.598172`, and 256-slot 16K
    reached `61.065087`, but the marginal gain is shrinking and all remain far
    below the practical vision target. Sprint 138 set the next kernel
    acceptance target: beat the compact
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
    atomic form. Sprint 145 confirms that widening to 256 slots at 16K is
    memory-safe and correct, but only moves continuation/decode throughput by
    about 2% versus the 128-slot/16K control. Sprint 146 then tested the
    matching 1536-route fixed gate/down probes for that 256-slot shape; the
    microbenchmark improved slightly, but served decode was neutral to
    slightly worse, so fixed-shape specialization alone is not enough.
    Sprint 148 then tested the stage-count version of software pipelining on
    the fused gate/up kernel. It improved the isolated 768-route probe but did
    not reduce the routed-FFN gate/up bucket reliably in the full scheduler.
    Sprint 152 broadened that into a 2/3/4-stage sweep and found the fixed
    probes neutral at both 768 and 1536 routes, with identical HMMA counts in
    NCU. Stage count inside the existing fused gate/up kernel is therefore
    exhausted. Sprint 153 added a bounded TP pack contract and proved the split
    descriptors can be emitted and bound, but the latest two-GPU proxy remains
    positive only for the 768-route shape. Sprint 154 completed the missing
    served A/B for down-reduce epilogue fusion at 768 and 1536 routed rows;
    the 128-slot run was flat and the 256-slot run was slightly slower, so
    epilogue-only fusion is also exhausted.
    Sprint 122 further showed that merely chunking slots to feed wider kernels
    loses too much stage overlap, so the fusion target must match the per-slot
    served topology or replace it with an overlapped scheduler.
- Decide whether the next production step is a true routed-FFN fused/persistent
  executor or the bounded 2-way TP routed-FFN prototype for 128-slot/32K.

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
`DS4_V100_TURBOMIND_DOWN_REDUCE_EPILOGUE=1` is available after Sprints 142 and
154, but stays default-off because the 128-slot/32K served result was flat and
the 256-slot/16K result was slightly slower.
The 1536-route fixed gate/up and down probes are available after Sprint 146
for 256-slot/16K diagnostics. They require explicit `m128_1536` or
`1536_m128` modes and are intentionally not selected by `auto`.
Stage-4 fused gate/up probes are available after Sprint 148 for diagnostics
with explicit modes such as `m128_s4`, `m64_s4`, and `m128_s4_1536`. They are
not selected by `auto` because served and profile results stayed inside the
run band.
Stage-3 fused gate/up probes are available after Sprint 152 for diagnostics
with explicit modes such as `m128_s3`, `m64_s3`, and `m128_s3_1536`. They are
not selected by `auto` because the 2/3/4-stage sweep stayed neutral.
The TP split pack path is available after Sprint 153 through
`tools/ds4-v100-appliance-pack --emit-tp-split`. It is a bounded prototype
format only; no launcher or production scheduler path selects TP expert rows by
default.

The TP4 hidden collective smoke is available after Sprint 195 through
`tools/ds4-v100-tp4-collective-smoke`. It is a measurement and correctness
primitive only. On the V100 node, the root gather/reduce/broadcast version
verifies on both four-GPU NVLink islands, but measures about `0.11 ms` for the
16-token/4096-hidden payload and tops out near `27 GB/s` effective wire
bandwidth at larger payloads. Do not use it as the production TP4 collective;
the next TP step needs NCCL, ring/tree peer-copy all-reduce, or a fused resident
boundary that avoids repeated full-hidden materialization.
After Sprint 196, the same tool supports `--algo doubling`, a recursive
pairwise all-reduce. It verifies on V100 and reaches `81.065 GB/s` effective
wire bandwidth at 1024 tokens, but it is slower than root at the current
16-token decode payload (`0.133761 ms` versus `0.110762 ms`). Treat this as a
prefill/batched-TP primitive, not a direct decode-serving win.
After Sprint 197, `DS4_V100_TURBOMIND_PROFILE=1` reports routed-FFN liveness
counters. The current `fused6_reduce` path has `compact_a_calls == calls`,
`down_reduce_epilogue_calls == calls`, `down_routes_calls == 0`, and
`mid_half_calls == calls`. The six-route `mid_half` allocation is `24576` bytes
per call, which means the next routed-FFN optimization needs to reduce the
gate/up and down execution boundary rather than only shrinking scratch memory.
After Sprint 198, `DS4_V100_TURBOMIND_GRAPH=1` can capture and replay the
current `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce` path. The graph key
includes routed-executor mode and capture skips the compact active-expert host
readback. Short direct replay on V100 matched output IDs
`201,200,84921,200,18,90,926,14` and improved continuation from `16.022442` to
`17.980888` tok/s, with `43` captures, `129` launches, and `0` failures. Keep
graph replay default-off until the same-binary 16-slot/256K served A/B passes;
the older Sprint 169 graph path regressed served throughput.
After Sprint 199, the served gate has passed for the current production pack.
At 16-slot/256K, `fused6_reduce` graph off measured `54.725463` generated /
`53.870377` continuation tok/s, while `fused6_reduce` graph on measured
`67.886268` / `66.825545`, both with `16/16` token match. The graph server log
reported `43` captures, `129` launches, and `0` launch/capture failures. A
separate routed-executor-off production control measured `56.719099` generated
/ `55.832863` continuation tok/s, so the promoted stack is about `+19.7%`
continuation versus the prior production control in this harness.
After Sprint 200, the focused six-route TurboMind bench covers the exact
production routed-FFN shape. It measured generic gated-SiLU at `0.0946 ms`,
fixed `m16_6` gated-SiLU at `0.1196 ms`, generic down at `0.0512 ms`, output
clear at `0.0022 ms`, and six-route down-reduce with clear at `0.0650 ms`.
The fixed six-route probe is slower than generic, and the clear boundary is too
small to justify a new wrapper ABI. This pushes the next material work toward
bounded full-layer TP4/EP or a real non-atomic route-reduce epilogue rewrite.
After Sprint 201, `tools/ds4-v100-tp4-layer-proxy` measures the full 43-layer
TP4 boundary rather than an isolated collective. The default 16-token shape
verifies and costs `22-24 ms` before compute, while 64-token and 128-token
shapes improve to `1837` and `2509` overhead-only tok/s. That result does not
justify another partial TP overlay; it supports either a bounded full-layer
TP4/EP slice or a return to a persistent fused routed-FFN kernel.
After Sprint 202, `test_ggml_turbomind_tp_split_4gpu` proves real four-way
TurboMind MXFP4 routed-FFN compute can scale, but only if the runtime avoids
routed-only hidden-state copies. The sprint also caught and fixed a benchmark
lifecycle bug where full-reference and shard work crossed streams on GPU0. At
96/768 routes, corrected compute-only TP4 is `2.350x/3.636x`;
copy-inclusive TP4 is `0.783x/0.682x`. This pushes the next TP implementation toward a
bounded full-layer resident TP4/EP slice.
After Sprint 203, that resident slice exists as
`test_ggml_turbomind_tp4_resident_layer_slice`. It keeps hidden state resident
across a multi-layer TP4 routed-FFN loop and passes correctness, but root
reduction is still slower than the one-GPU reference at both the production
96-route shape and the larger 768-route shape. The simple doubling variant is
also slower because it is sequential, not a true concurrent collective. Do not
move TP4 into the scheduler until the collective boundary is materially better.
After Sprint 204, `doubling_async` proves concurrency matters: the same
resident slice becomes positive for 768-route high-batch shapes. The 96-route
production decode shape remains negative on repeat, so the project should not
spend the next sprint wiring TP4 into serving. Either build a stronger
collective gate or pivot back to persistent fused routed-FFN work.
After Sprint 205, the stronger small-payload root alternative is rejected.
The next sprint should not continue TP4 decode integration; it should start the
persistent fused routed-FFN path unless the explicit goal changes to
larger-batch/prefill TP4.
