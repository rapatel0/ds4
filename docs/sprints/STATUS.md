# DS4 V100 Appliance Status

Last updated: 2026-05-23

## Topline

Current promoted serving baseline is Sprint 199's graph-backed
`fused6_reduce` production pack at 16-slot/256K: `67.886268` generated tok/s
and `66.825545` continuation tok/s with `16/16` token match. Sprint 200 then
rejected the easy exact six-route kernel cut-ins, and Sprint 201 measured the
first full-layer TP4 boundary envelope. A 43-layer, 4-collective/layer TP4
proxy costs `22-24 ms` at 16 active tokens, or about `655-724 tok/s`
overhead-only before DS4 compute. The same boundary improves to
`1837 tok/s` overhead-only at 64 active tokens and `2509 tok/s` at 128 active
tokens. Sprint 202 then measured real TurboMind MXFP4 routed-FFN TP4 compute,
after fixing a benchmark lifecycle bug where the full reference and shard 0
crossed streams on GPU0 and shared one TurboMind workspace. Corrected
compute-only speedup reaches `2.350x` at 96 routes and `3.636x` at 768 routes,
but conservative copy-inclusive timing regresses to `0.783x` and `0.682x`.
This keeps full-layer TP4/EP viable only if dense and routed work stay inside
the TP boundary; routed-only overlays remain rejected. Sprint 203 then built
that first resident TP4 layer-slice gate. It is correct, but the naive
resident root boundary is still slower than a one-GPU full-width routed-FFN
reference: `0.825x` at `96 routes x 43 layers` and `0.589x` at
`768 routes x 43 layers`. This blocks production TP4 scheduler integration
until a real concurrent collective or fused reduction boundary exists.
Sprint 204 added a concurrent `doubling_async` resident reduction. It improves
the resident slice and is positive at larger 768-route shapes (`1.071x` over
43 layers), but it does not reliably clear the production 96-route decode gate:
the first 43-layer run was `1.006x`, while the longer repeat was `0.896x`.
TP4 should remain a larger-batch/prefill candidate unless the collective gets a
fused/NCCL-grade implementation.
Sprint 205 tested the missing async root variant. It is correct, but slower:
`0.970x` at `96 routes x 4 layers`, `0.866x` at `768 routes x 4 layers`, and
`0.860x` at `96 routes x 43 layers`. This closes the current TP4 production
decode branch; next implementation should pivot to persistent fused routed-FFN
work.
Sprint 208 re-opened topology investigation for the 32-slot target with a
separate TP8 path, not a PP scheduler abstraction. The new TP planner shows
32-slot/256K `PP1/TP8` fits with F8 KV sharding at `26.84 GiB` worst GPU and
fails with replicated KV at `50.63 GiB`, making KV sharding mandatory. The
8-GPU FP16 collective smoke passed at 32/64/128 tokens; recursive doubling beat
root and measured `0.322599 ms`, `0.372364 ms`, and `0.436299 ms`
respectively. The 43-layer, 2-reduction/layer resident-boundary proxy measured
`29.381000 ms` at 32 tokens (`1089.139` overhead-only tok/s), `32.605223 ms`
at 64 tokens (`1962.876` tok/s), and `37.994584 ms` at 128 tokens
(`3368.901` tok/s). This clears the first TP8 investigation gate but does not
prove serving; the next TP sprint should build a bounded one-layer TP8
prototype in new TP-only files with sharded KV ownership inside the boundary.
Sprint 209 built that bounded prototype without touching the PP scheduler. At
32 slots / 256K / ratio-4 F8 KV, each GPU allocates a `169347072` byte KV shard
for the layer instead of a replicated `1.262 GiB` logical KV allocation. The
one-layer TP8 smoke passed correctness at 32/64/128 tokens with total average
latencies of `0.739408 ms`, `0.876011 ms`, and `1.098461 ms`; reduction time
was `0.634680 ms`, `0.718601 ms`, and `0.840586 ms`. This keeps TP8 alive, but
still only as a separate TP branch: the synthetic compute body must be replaced
with a real TP-only DS4 layer slice before any serving integration.
Sprint 210 replaced the synthetic body with a TP-only resident FFN fixture that
uses cuBLAS FP16 Tensor Core GEMMs for column-parallel gate/up, gated SiLU,
row-parallel down, and TP8 hidden reduction. At 32 slots / 256K / ratio-4 F8
KV, the `mid_shard=1024` gate passed correctness at 32/64/128 tokens with
total latencies `0.614750 ms`, `0.709350 ms`, and `0.796927 ms`. The denser
`mid_shard=2048` sweep also passed and reached `62.956` fixture TFLOP/s at 128
tokens. This confirms TP8 can put useful resident GEMM work inside the boundary,
but it is still an FP16 fixture; the next TP sprint should adapt the low-bit
TurboMind MXFP4 expert path to the separate TP8 codepath before any serving
integration.
Sprint 211 did that low-bit TP8 gate and rejected the current TP8 MXFP4 shard
shape. The new separate TP-only `ds4-v100-tp8-turbomind-ffn-smoke` runs through
the public TurboMind ABI with synthetic MXFP4 fixtures, but TP8
`mid_shard=256` produces invalid partial sums: 96/192/384-route runs all fail
correctness with large NaN counts. The compute-only timing is attractive
(`3.927x-4.189x` versus full reference), but the simple output gather/reduce is
already slower than the reference (`0.524x`, `0.368x`, `0.317x` total speedup).
The existing TP4 TurboMind control remains correct at the same route shapes and
shows `2.333x-3.676x` compute speedup, although copy-inclusive timing is still
below `1.0x`. Next work should pivot to TP4/PP1 low-bit layer ownership with a
better reduction boundary, or explicitly design a TP8 MXFP4 shard-256 kernel
before returning to TP8.
Sprint 212 executed that TP4/PP1 low-bit layer-body gate in a separate TP-only
tool. Correctness passes at 96/192/384 routes after fixing two benchmark bugs
(synthetic fixture overflow and a missing `cudaSetDevice()` before TurboMind
calls). Compute-only TP4 speedup is still strong (`2.335x/2.597x/3.707x`), but
the resident root reduction only wins at 96 routes (`1.078x`) and regresses at
192/384 routes (`0.932x/0.967x`). This rejects TP4/PP1 runtime ownership as
the next implementation step; TP should stay parked for prefill/larger-batch
work or until a fused/NCCL-grade collective exists. The next sprint should
return to a monolithic/persistent low-bit routed-FFN executor.
Sprint 213 closed the existing `fused6_split_reduce` materialized reducer
branch. The opt-in path is correct, builds on V100, passes full scheduler smoke
with `tm_layers=43`, and preserves CUDA graph capture (`43` captures, `129`
launches, `0` failures). The focused six-route FFN sequence improved from
`0.1391 ms` atomic to `0.1290 ms` materialized, but served 16-slot/256K
continuation only moved from `60.236036` to `60.655009` tok/s (`+0.7%`), below
the promotion gate. Defaults stay on `fused6_reduce + graph`; the next sprint
should stop reducer-wrapper tuning and build a true tile-local/persistent
routed-FFN workbench.
Sprint 214 built that standalone workbench and rejected the first tile-local
candidate. The tool compares `gated_silu -> down_reduce`,
`gated_silu -> down -> split_reduce`, and a diagnostic raw-MXFP4
down+route-reduce kernel for the exact six-route production decode shape. On
the V100 pod, the finite fixture passed baseline and candidate correctness
(`split_bad=0/4096`, `candidate_bad=0/4096`), but timing was decisively below
the gate: atomic sequence `0.184721 ms`, split sequence `0.165159 ms`,
candidate sequence `0.370901 ms`, and candidate down-only `0.276122 ms`
(`0.445x` versus best baseline). The branch is rejected because replacing the
Tensor Core down GEMM with SIMT F32 accumulation loses more than it saves by
removing the materialized down-route buffer. The next sprint should either move
to practical serving levers such as MTP/continuous batching or build a real
Tensor Core fused/persistent routed-FFN kernel rather than another reducer-only
wrapper.
Sprint 215 added a repeatable practical-serving matrix and ran it against the
persistent production TurboMind pack. The best current practical long-context
mode is now `32` slots at `128K`: `69.488893` generated tok/s and
`68.403129` continuation tok/s with `32/32` token match, `45.88%` average GPU
utilization, and `24124 MiB` max observed memory. The current `16`-slot/`256K`
baseline remains valid at `62.602937` generated tok/s and `61.624766`
continuation tok/s with `16/16` token match. `32` slots at `256K` still fails
closed at the production launcher cap: `DS4_V100_SLOTS=32 exceeds ctx=262144
admission cap 16`. MTP verify is compatible but not a speedup
(`attempted=16`, `accepted=0`, `16.373227` generated tok/s), and one-slot MTP
commit accepted `8/15` drafts but only reached `8.369430` generated tok/s /
`7.846341` continuation tok/s. MTP is therefore not shipped as a speedup; the
next MTP step must be true speculative target verification over drafted tokens.
Sprint 216 built that focused MTP speculative gate and rejected the current
commit path as a throughput feature. The new replay and sustained-bench
accounting reports draft proposals, accepted drafts, target tokens verified,
target forwards, effective output tokens, and speculative saves. On the V100
pod, one-slot `256K` MTP commit again accepted `8/15` drafts, but the decisive
fields were `target_forwards=16`, `effective_output_tokens=16`, and
`speculative_saves=0`; continuation throughput was `4.276211` tok/s versus
`4.644949` for the same one-slot baseline. The API gap is now explicit:
current replay batching is across slots at the same decode step, while MTP
needs a one-slot multi-position target verification/state-advance primitive to
save target forwards. MTP remains default-off for production throughput.
Sprint 217 tested whether the `256K` slot cap was merely conservative and
found a real cold-path failure above 16 active slots. Sprint 218 localized that
failure: without launcher startup warmup, an `18`-slot/`256K` run first reports
NaN HC at `stage=1`, `gpu=1`, `layer=6`, `slot=0`, `token=0`, `position=0`.
Pre-output-head checking shows NaN HC reaches the output stage before logits,
so the output-head fastpath is not the first source. With launcher startup
warmup enabled, the same warmed appliance path passes the full target shape.
Sprint 219 made that warmed-readiness contract explicit in `/v100/status` and
metrics, then ran the longer production gate at `64` requests: `64/64`
matches, max memory `24124 MiB`, max GPU util `88%`, `58.241743` generated
tok/s, `16.380490` prompt tok/s, and `57.331715` continuation tok/s. The
launcher admits `ctx=262144`, `slots=32` only when startup warmup resolves
enabled; the cold `DS4_V100_STARTUP_WARMUP=0` path still fails closed at cap
`16`. Sprint 220 then aligned the operator deployment path with that result:
the env example now defaults to the production appliance dir, `ctx=262144`,
`slots=32`, `active_microbatch=32`, and `DS4_V100_STARTUP_WARMUP=auto`; the
production deployment smoke now accepts `--appliance-dir`, checks warmed
readiness in status and metrics, and passed on the V100 pod with two bounded
generation requests returning `3136`. Sprint 221 added the first explicit
MTP target-block verification primitive. Replay now has all-stage target
snapshot wrappers and a one-slot forced-block verification API, with
`tools/ds4-v100-replay --target-block-smoke N` as the diagnostic gate. The
V100 production-pack smoke passed for a 4-token block: first token bytes
`3136`, snapshot bytes `30107648`, `target_forwards=4`,
`accepted_prefix_len=4`, `target_tokens_verified=4`,
`effective_output_tokens=4`, and `speculative_saves=0`; a negative guard fails
closed for multi-slot use. This is the right MTP boundary, but not yet an MTP
speedup because the block body is still serial target execution. Sprint 222
then connected real chained MTP draft blocks to that boundary. The helper can
now return MTP next-HC, and `--mtp-draft-block-smoke 4` produced a real draft
sequence without target forwards between draft steps. The V100 production-pack
fixture reported `draft_tokens=1,0,1,0`, `target_tokens=1,380,5,380`,
`accepted_prefix_len=1`, `target_forwards=4`, `target_tokens_verified=4`,
`effective_output_tokens=2`, `speculative_saves=0`, `mtp_ms=18.081`, and
`verify_ms=231.553`; the multi-slot guard fails closed. This ships the
diagnostic but does not promote MTP throughput. The next MTP decision requires
a broader acceptance matrix; if accepted prefixes stay low, the practical
serving branch should pivot back to attention/KV or persistent low-bit
execution. Sprint 223 ran that matrix on five prompt fixtures and block sizes
`2,4,8` after fixing real-prompt replay compressed-cache cap sizing. The full
V100 production-pack matrix passed `15/15` cases with average accepted prefix
`1.533`, max accepted prefix `2`, `10/15` cases at accepted prefix `>=2`, and
total speculative saves `4`. The decision is to continue MTP only in the
block-2 shape: block-2 accepted both drafted tokens in `4/5` cases and reported
`speculative_saves=1`, while block-4 and block-8 never accepted more than two
tokens and mostly add verifier work. The next MTP sprint should build and
measure a block-2 exact speculative commit/verify path, not longer draft
blocks. Sprint 224 built that exact block-2 path and measured it on the V100
production pack. The path is token-correct and faster than same-process
baseline on `4/5` prompts: ok-case average `3.663043` block2 tok/s versus
`2.032918` baseline tok/s (`1.801865x`). It reports `target_forwards=7`,
`effective_output_tokens=8`, and `speculative_saves=1` for 8-token runs. It is
not ready for serving integration because `long_memory_archive` failed token
parity at token 1 (`baseline=16`, `got=8773`), and a follow-up
`--target-block-smoke 2` on that same prompt also failed target replay reset
determinism (`got=32085 want=10220`). The next sprint should fix long-prompt
replay reset/snapshot determinism, then rerun the block-2 gate.
Sprint 225 cleared that reset/snapshot blocker and tightened throughput
methodology. `tools/ds4-v100-replay` now has `--reset-parity-smoke` and
`--prompt-token-limit`; full `long_memory_archive` reset parity passes with
`prompt_tokens=3353`, `generated_tokens=1`, `first_token=32085`, and
`match=true`. Full `--target-block-smoke 2` also passes on that prompt with
`snapshot_bytes=907214848`. Bounded MTP block-2 checks pass through the 1024
token prefix with `token_match=true` and `speculative_saves=1`, but the full
single-slot MTP run was stopped after the methodology review and is not
promotion evidence. Practical throughput claims are now guarded:
`tools/ds4-v100-sustained-decode-bench.sh` defaults to `32` slots at `256K`,
`active_microbatch=slots`, startup warmup, `200000 us` microbatch wait, and
per-step async event handoff; slot tier `1` requires
`--allow-single-slot-diagnostic`. The current Sprint 225 practical serving
repeat measured `50.434232` generated tok/s and `47.282093` continuation tok/s
at `32` slots / `256K`, with `64/64` token match, average GPU utilization
`47.076%`, and max GPU utilization `96%`. TP remains prototype-only and is not
operational in production serving.

Current maximum-context production mode is now the Sprint 219 warmed
32-slot/256K appliance result. Sprint 137 adds an explicit
128-slot/32K short-context throughput mode. Sprint 139 raises the best observed
gated-appliance 128-slot/32K run to `60.130047` generated tok/s, while showing
the fixed-shape gate/up probe itself only contributes about `0.1%` end-to-end.
The runtime now reliably coalesces
high-slot concurrent requests into one tensor batch by resolving launcher
`auto` microbatch wait to 200 ms at `active_microbatch >= 16`. The best current
served result is Sprint 146's `61.223893` generated tok/s at 256-slot/16K;
the best 32K result remains `60.130047`, and the current 256K production-auto
repeat remains `43.534061` generated tok/s. Sprint 123 found
correct opt-in shared-FFN fusions up to `43.887206`. Sprint 124 added a
correct opt-in TurboMind route-row reduce path and measured up to `43.822500`.
Sprint 125 added a correct grouped-batch attention output-A probe and measured
up to `43.640921`. Sprint 126 added a default-off production routed-expert
stage profiler and confirmed the current binary still serves at `43.453309`
generated tok/s with `16/16` token match. Sprint 127 added an opt-in
TurboMind gated-SiLU path with interleaved fused gate/up packs. It removed the
standalone SwiGLU bucket from the routed-expert profile and measured
`43.933293` generated tok/s with `16/16` token match. Sprint 128 compacted the
packed TurboMind grouped schedule from 256 experts to at most `total_routes`
groups and promoted that path as the launcher default after same-binary A/B
reached `45.888778` generated tok/s on the existing fused appliance and
`46.394722` on the interleaved gated appliance with route-row-reduce opt-in.
Sprint 129 exposed TurboMind dispatch policy selection, rejected unsafe
`measure` after a full-scheduler measurer fatal, and found safe `reuse`
neutral at `45.813841` vs `45.840691` default. Sprint 130 reran the closest
existing routed-FFN epilogue-fusion analogue on the current fused appliance:
compact control was `45.837745`, while compact plus route-row-reduce was
`45.660765`, so route-row-reduce remains opt-in. Sprint 131 added a correct
opt-in TurboMind indexed-A path that avoids route-expanded FP16 activations for
gate/up GEMMs, but served A/B was only `45.789937` vs `45.663281` control, so
it also remains opt-in. Sprint 132 extended the standalone TurboMind gate/up
benchmark to the production 96-route shape from the served profile; the
interleaved gated path passed at `0.1776 ms` vs `0.2889 ms` for separate
gate+up, a `1.626x` isolated speedup. Sprint 133 corrected that benchmark to
also use the served compact active-expert topology; compact 96-route gated-SiLU
is `0.1740 ms` vs `0.1895 ms` separate gate+up, only `1.089x`. Sprint 134
added a fixed-shape DS4 ABI probe that bypasses generic dispatch and directly
launches the matching SM70 MXFP4 gated kernel; it was bit-identical and exactly
neutral at `0.1746 ms` vs `0.1746 ms` generic gated. Sprint 135 raised the
admitted short-context throughput tier to 32 slots at 128K, while keeping 256K
capped at 16 slots. The 32-slot 128K appliance passed full scheduler smoke and
served correctness, reaching `52.840889` generated tok/s versus `45.780913`
for a same-context 16-slot control. The next target is therefore not dispatch
bypass or gate/up launch fusion; it must change kernel math/dataflow or widen
the served scheduling shape further. Sprint 136 widened the short-context tier
again to 64 slots at 64K, passed full scheduler smoke, and reached `57.322945`
generated tok/s versus `52.884400` for a same-context 32-slot control.
Sprint 137 admitted 128 slots at 32K, passed full scheduler smoke, and reached
`59.598172` generated tok/s versus `57.170428` for a same-context 64-slot
control. The slot-width sweep remains positive but is clearly diminishing.
Sprint 138 widened the standalone TurboMind compact gate/up benchmark defaults
to cover 192/384/768 routed-row shapes. The 768-route compact baseline is
`0.6379 ms` for fused gate_up and `0.6481 ms` for gated-SiLU. Sprint 139 added
a fixed-shape 768-route m128 gated-SiLU probe and wired it into the appliance
under exact production guards. It beat the isolated generic gated path
(`0.5999 ms` vs `0.6480 ms`) and served correctly at `60.130047` generated
tok/s on the 128-slot/32K gated appliance, but same-binary probe-off was
`60.061899`, so the end-to-end gain is only about `0.1%`. Sprint 140 repeated
that fixed-shape strategy for the 768-route down projection. The down probe was
correct and faster in isolation (`0.3026 ms` vs `0.3272 ms`), but served A/B
was slower with it enabled (`60.038469` vs `60.129772`), so it remains opt-in
and default-off. Sprint 141 added an opt-in half2-vectorized variant of the
route-row reduce tail. It passed the full 43-layer 128-slot smoke, but served
A/B stayed neutral: control was `60.108232`, scalar route-row reduce was
`60.112248`, and half2 route-row reduce was `60.104512`, all with `128/128`
token match. Tail-kernel vectorization is therefore not the missing throughput
lever. Sprint 142 moved that idea into the TurboMind down GEMM epilogue for
the exact 768-route high-slot shape. The fused epilogue reduce path passed the
full 43-layer 128-slot smoke and served correctly at `60.041003` generated
tok/s versus `59.987105` same-binary control, so it remains opt-in/off because
the effect is only run-noise positive. Sprint 143 added first-class prefill
versus decode metrics to the soak, sustained decode, and aggregate throughput
harnesses so future A/B runs show prompt replay, continuation decode, and
aggregate generated rates separately. Sprint 144 added explicit SM70 MXFP4
`m64n256` tile probes for the 768-route gate/up and down shapes. Both passed
full 43-layer smoke; the standalone down probe was slightly faster
(`0.2896 ms` vs `0.2936 ms`), but served A/B regressed to `59.791839`
generated tok/s versus `59.993301` control, and gate `m64n256` also regressed
to `59.797232`. The probes remain explicit opt-ins only. Sprint 145 admitted a
guarded 256-slot/16K short-context tier after planner and full-scheduler
validation. It served correctly at `61.065087` generated tok/s and
`57.248519` continuation/decode tok/s with `256/256` token match, but the
decode gain over the 128-slot/16K control was only about 2%, so slot widening
is now a ceiling-expansion tactic rather than the main throughput lever.
Sprint 146 added explicit 1536-route fixed-shape gate/up and down probes for
the 256-slot compact routed shape. The gate probe was correct and slightly
faster in isolation (`0.9435 ms` vs `0.9651 ms` generic gated), but served A/B
was flat to slightly worse (`61.204203` generated tok/s and `57.378940`
continuation/decode tok/s versus `61.223893` and `57.397400` control), so the
1536-route probes remain explicit opt-ins and are not selected by `auto`.
Sprint 147 extended the down-reduce epilogue to the 1536-route shape and
validated it with a full 43-layer 256-slot smoke, but served A/B was deferred
after the strategy pivot to larger fused-kernel work. Sprint 148 tested a real
stage-4 SM70 software-pipeline variant of the fused MXFP4 gate/up+gated-SiLU
kernel. The 768-route `m128_s4` probe improved the isolated gate/up benchmark
(`0.5811 ms` vs `0.6033 ms` for `m128`) and passed full 43-layer smoke, but
served A/B was only run-noise positive (`60.049057` generated and `56.295991`
continuation/decode tok/s versus `59.865668` and `56.124063` control), and the
profile did not show a reliable gate/up bucket reduction. Stage-4 probes remain
explicit opt-ins. Sprint 149 added a TP split benchmark and a P2P reduce-payload
proxy. The 2-way FFN middle-dimension split shows ideal compute speedups of
`1.858x` at 768 routes and `1.468x` at 1536 routes before communication.
Peer-copy measurements show a 12 MiB hidden payload takes about `0.26 ms` over
NV2, `0.52 ms` over NV1, and `1.29-1.31 ms` over SYS, so a TP prototype should
start with NV2 pairs and stay bounded before any scheduler-wide rewrite.
Sprint 150 built that bounded 2-GPU TP proxy. On clean NV2 pairs, the
768-route shape shows about `1.87x` concurrent compute speedup and about
`1.28x` total speedup after conservative input/output copies. The 1536-route
shape is neutral to slower after copies (`0.85-0.94x`), so TP is a candidate
for the 128-slot/32K tier first, not a broad replacement for the 256-slot/16K
ceiling. Sprint 151 added the missing correctness gate: finite MXFP4 fixtures
now compare full one-GPU down output against the sum of the two TP partials.
Both clean NV2 pairs pass at 768 and 1536 routes with `rel ~= 2.46e-04`,
`bad=0`, and max absolute difference `6.1035e-05`. Sprint 152 completed the
fused gate/up software-pipeline sweep by adding 3-stage variants and comparing
2/3/4-stage fixed probes. At 768 routes, `m128`, `m128_s3`, and `m128_s4`
measured `0.5809 ms`, `0.5863 ms`, and `0.5794 ms`; at 1536 routes,
`m128_1536`, `m128_s3_1536`, and `m128_s4_1536` measured `0.8743 ms`,
`0.8821 ms`, and `0.8774 ms`. NCU fixed-probe counters were also neutral, with
identical HMMA instruction counts. Stage-count tuning inside the existing
fused gate/up GEMM is now exhausted as a material lever. Sprint 153 added a
bounded 2-way TP appliance-pack contract and context binding for split
TurboMind expert descriptors. A layer-3, six-expert pack emitted 8 TurboMind
rows across GPU0 and GPU3 and passed partial context binding. The real 2-GPU
NV2 proxy remains positive only at the 768-route shape (`1.157x`
total-with-copy on pair `0,3` in this run) and slower at 1536 routes
(`0.912x`), so TP remains a narrow 128-slot/32K prototype candidate rather
than a broad topology replacement. Sprint 154 closed the served A/B gap for
the largest currently implemented fused routed-FFN boundary: fused gate/up plus
gated-SiLU plus the down-projection route-weighted reduce epilogue. The
128-slot/32K result was flat (`59.509317` generated / `55.789985`
continuation tok/s versus `59.502747` / `55.783825` control), and the
256-slot/16K result was slightly slower (`60.642962` / `56.852777` versus
`60.671924` / `56.879929` control). A synchronized profile kept gate/up at
about `58-61%` and down at about `25-29%` of profiled routed-FFN time, so
epilogue-only fusion is not a material software-pipeline lever. Sprint 155
then implemented a true opt-in stream-per-expert routed-FFN pipeline for the
current non-interleaved fused gate/up pack. The path proved active on V100
(`group_pipeline_calls=6` in a profiled stage smoke), but served throughput
regressed: 128-slot/32K was `59.125703` generated / `55.430346` continuation
tok/s versus `59.394915` / `55.682733` control, and 256-slot/16K was
`60.308689` / `56.539396` versus `60.648138` / `56.857630` control. The flag
remains diagnostic-only; the next material FFN path must remove launches/stream
joins with a persistent fused boundary or continue the bounded 2-way TP
prototype. Sprint 156 retested that path with the exact observed six active
expert groups. The manual six-group diagnostic was slightly positive at
128-slot/32K (`59.645848` generated / `55.917982` continuation tok/s versus
`59.516392` / `55.796618` control) and at 256-slot/16K (`60.675527` /
`56.883307` versus `60.442968` / `56.665283` control), but hardcoding six
groups is not safe for arbitrary serving traffic. A safe auto-group mode was
implemented and passed full scheduler smoke, but its host active-group readback
regressed served throughput. The group pipeline therefore remains diagnostic;
the next material path is still a persistent/larger fused routed-FFN executor.
Sprint 157 added an opt-in CUDA Graph replay probe around the TurboMind
routed-FFN core. It built and passed full 43-layer graph-off and graph-on
scheduler smokes, but served 128-slot/32K capture failed in the current
legacy-default-stream kernel path. Graph-disabled control was `59.607704`
generated / `55.882222` continuation tok/s; graph enabled with stable scratch
was correct but measured `59.450666` / `55.734999` with zero captures, and the
thread-local capture variant measured `59.367233` / `55.656781` with zero
captures. The graph flag remains diagnostic-only; real graph replay would
require threading an explicit stream through the routed-FFN executor.
Sprint 158 added `DS4_V100_TURBOMIND_ROUTED_EXECUTOR` and a guarded fixed96
routed gate_up executor for the 16-slot/256K product shape. Full 43-layer
scheduler smoke proved the intended fused-kernel shape is legal
(`total_routes=96`, six compact active experts, 16 routes per expert) and
selected the fixed gate_up kernel. Served 16-slot/256K A/B was correct but did
not select fixed96 because the HTTP path is currently reaching the routed FFN
as one request at a time (`total_routes=6`). The final guard avoids overhead on
that served shape: control measured `46.113721` generated / `43.231614`
continuation tok/s and guarded fixed96 measured `46.167311` / `43.281854`.
The flag remains explicit opt-in. The next material issue is served batch
formation for `>=256K`, or a topology path that makes the executor dense
without relying on current HTTP coalescing.

The default stack still uses the Sprint 111 fused TurboMind gate/up appliance,
Sprint 115 shared gate/up SwiGLU F8 HMMA, Sprint 116 batched
attention-projection F8 HMMA for active 4/8-slot batches, and Sprint 119
event-ordered handoff for multi-slot per-step serving. Sprint 128 adds compact
TurboMind expert scheduling as a default routed-FFN optimization. Sprint 122
confirms that chunking slots to expose wider batch kernels is slower in the
current topology because it gives up too much stage overlap.

| Mode | Context | Slots | Generated tok/s | Continuation tok/s | Correctness |
|---|---:|---:|---:|---:|---|
| Sprint 146 256-slot 16K control repeat | 16,384 | 256 | `61.223893` | `57.397400` | 256/256 token match |
| Sprint 146 gate `m128_1536` opt-in | 16,384 | 256 | `61.204203` | `57.378940` | 256/256 token match |
| Sprint 145 256-slot 16K throughput ceiling | 16,384 | 256 | `61.065087` | `57.248519` | 256/256 token match |
| Sprint 156 six-group pipeline diagnostic | 16,384 | 256 | `60.675527` | `56.883307` | 256/256 token match |
| Sprint 156 same-binary control | 16,384 | 256 | `60.442968` | `56.665283` | 256/256 token match |
| Sprint 156 safe auto-group diagnostic | 16,384 | 256 | `60.232265` | `56.467748` | 256/256 token match |
| Sprint 154 256-slot 16K down-reduce control | 16,384 | 256 | `60.671924` | `56.879929` | 256/256 token match |
| Sprint 154 256-slot 16K down-reduce opt-in | 16,384 | 256 | `60.642962` | `56.852777` | 256/256 token match |
| Sprint 155 active group-pipeline opt-in | 16,384 | 256 | `60.308689` | `56.539396` | 256/256 token match |
| Sprint 145 192-slot 16K midpoint | 16,384 | 192 | `60.700926` | `56.907118` | 192/192 token match |
| Sprint 145 128-slot 16K control | 16,384 | 128 | `59.860493` | `56.119213` | 128/128 token match |
| Sprint 139 gated m128 auto probe | 32,768 | 128 | `60.130047` | `56.371919` | 128/128 token match |
| Sprint 157 graph-disabled control | 32,768 | 128 | `59.607704` | `55.882222` | 128/128 token match |
| Sprint 157 graph stable-scratch global capture probe | 32,768 | 128 | `59.450666` | `55.734999` | 128/128 token match, 0 captures |
| Sprint 157 graph stable-scratch thread-local capture probe | 32,768 | 128 | `59.367233` | `55.656781` | 128/128 token match, 0 captures |
| Sprint 156 six-group pipeline diagnostic | 32,768 | 128 | `59.645848` | `55.917982` | 128/128 token match |
| Sprint 156 same-binary control | 32,768 | 128 | `59.516392` | `55.796618` | 128/128 token match |
| Sprint 156 safe auto-group diagnostic | 32,768 | 128 | `58.988662` | `55.301871` | 128/128 token match |
| Sprint 154 128-slot 32K down-reduce opt-in | 32,768 | 128 | `59.509317` | `55.789985` | 128/128 token match |
| Sprint 154 128-slot 32K down-reduce control | 32,768 | 128 | `59.502747` | `55.783825` | 128/128 token match |
| Sprint 148 gate `m128_s4` opt-in | 32,768 | 128 | `60.049057` | `56.295991` | 128/128 token match |
| Sprint 148 same-binary control | 32,768 | 128 | `59.865668` | `56.124063` | 128/128 token match |
| Sprint 140 gated down-probe-off control | 32,768 | 128 | `60.129772` | `56.371661` | 128/128 token match |
| Sprint 141 scalar route-row reduce repeat | 32,768 | 128 | `60.112248` | `56.355232` | 128/128 token match |
| Sprint 141 control repeat | 32,768 | 128 | `60.108232` | `56.351468` | 128/128 token match |
| Sprint 141 half2 route-row reduce | 32,768 | 128 | `60.104512` | `56.347980` | 128/128 token match |
| Sprint 141 indexed-A repeat | 32,768 | 128 | `60.056960` | `56.303400` | 128/128 token match |
| Sprint 142 down-reduce epilogue opt-in | 32,768 | 128 | `60.041003` | `56.288440` | 128/128 token match |
| Sprint 141 route-row reduce earlier repeat | 32,768 | 128 | `60.022743` | `56.271322` | 128/128 token match |
| Sprint 144 control with split metrics | 32,768 | 128 | `59.993301` | `56.243719` | 128/128 token match |
| Sprint 144 gate m64n256 probe | 32,768 | 128 | `59.797232` | `56.059905` | 128/128 token match |
| Sprint 144 down m64n256 probe | 32,768 | 128 | `59.791839` | `56.054849` | 128/128 token match |
| Sprint 140 gated down-probe-auto candidate | 32,768 | 128 | `60.038469` | `56.286064` | 128/128 token match |
| Sprint 142 down-reduce epilogue control | 32,768 | 128 | `59.987105` | `56.237910` | 128/128 token match |
| Sprint 139 gated probe-off control | 32,768 | 128 | `60.061899` | `56.308030` | 128/128 token match |
| Sprint 137 128-slot 32K throughput mode | 32,768 | 128 | `59.598172` | `55.873286` | 128/128 token match |
| Sprint 137 same-context control | 32,768 | 64 | `57.170428` | `53.597276` | 64/64 token match |
| Sprint 136 64-slot 64K throughput mode | 65,536 | 64 | `57.322945` | `53.740261` | 64/64 token match |
| Sprint 136 same-context control | 65,536 | 32 | `52.884400` | `49.579125` | 32/32 token match |
| Sprint 135 32-slot 128K throughput mode | 131,072 | 32 | `52.840889` | `49.538334` | 32/32 token match |
| Sprint 135 same-context control | 131,072 | 16 | `45.780913` | `42.919606` | 16/16 token match |
| Sprint 128 gated compact + route-row-reduce opt-in | 262,144 | 16 | `46.394722` | `43.495052` | 16/16 token match |
| Sprint 128 gated compact opt-in | 262,144 | 16 | `46.328184` | `43.432672` | 16/16 token match |
| Sprint 158 guarded fixed96 served path | 262,144 | 16 | `46.167311` | `43.281854` | 16/16 token match, fixed96 not selected in HTTP path |
| Sprint 158 same-binary control | 262,144 | 16 | `46.113721` | `43.231614` | 16/16 token match |
| Sprint 128 compact launcher default on fused appliance | 262,144 | 16 | `45.888778` | `43.020729` | 16/16 token match |
| Sprint 129 default dispatch control | 262,144 | 16 | `45.840691` | `42.975648` | 16/16 token match |
| Sprint 129 reuse dispatch probe | 262,144 | 16 | `45.813841` | `42.950476` | 16/16 token match |
| Sprint 130 compact fused control repeat | 262,144 | 16 | `45.837745` | `42.972886` | 16/16 token match |
| Sprint 131 compact fused indexed-A opt-in | 262,144 | 16 | `45.789937` | `42.928066` | 16/16 token match |
| Sprint 130 compact fused route-row-reduce repeat | 262,144 | 16 | `45.660765` | `42.806967` | 16/16 token match |
| Sprint 131 compact fused control repeat | 262,144 | 16 | `45.663281` | `42.809326` | 16/16 token match |
| Sprint 128 compact explicit on fused appliance | 262,144 | 16 | `45.747461` | `42.888244` | 16/16 token match |
| Sprint 128 gated compact-off same-binary control | 262,144 | 16 | `43.879880` | `41.137387` | 16/16 token match |
| Sprint 127 interleaved gated-SiLU opt-in | 262,144 | 16 | `43.933293` | `41.187462` | 16/16 token match |
| Sprint 123 best opt-in shared FFN fusion | 262,144 | 16 | `43.887206` | `41.144256` | 16/16 token match |
| Sprint 127 same-binary fused gate/up control | 262,144 | 16 | `43.691032` | `40.960343` | 16/16 token match |
| Sprint 126 no-profile same-binary sanity | 262,144 | 16 | `43.453309` | `40.737477` | 16/16 token match |
| Sprint 124 route-row reduce opt-in | 262,144 | 16 | `43.822500` | `41.083593` | 16/16 token match |
| Sprint 125 output-A rows2 batch opt-in | 262,144 | 16 | `43.640921` | `40.913364` | 16/16 token match |
| Sprint 125 output-A HMMA plus output-B batch opt-in | 262,144 | 16 | `43.245208` | `40.542383` | 16/16 token match |
| Sprint 124 same-binary control repeat | 262,144 | 16 | `43.517862` | `40.797995` | 16/16 token match |
| Sprint 123 shared-down-add plus scalar shared-pair fusion | 262,144 | 16 | `43.812630` | `41.074340` | 16/16 token match |
| Sprint 123 same-binary fused-add control | 262,144 | 16 | `43.070728` | `40.378807` | 16/16 token match |
| Sprint 122 production-auto 16-slot throughput mode | 262,144 | 16 | `43.534061` | `40.813182` | 16/16 token match |
| Sprint 122 best observed 16-slot candidate | 262,144 | 16 | `43.730215` | `40.997076` | 16/16 token match |
| Sprint 121 16-slot throughput mode | 262,144 | 16 | `43.659461` | `40.930745` | 16/16 token match |
| Sprint 121 same-binary 8-slot control | 262,144 | 8 | `34.445844` | `32.292979` | 8/8 token match |
| Sprint 120 current default repeat | 262,144 | 8 | `34.490294` | `32.334651` | 8/8 token match |
| Single scalar fusion opt-in repeat | 262,144 | 8 | `34.689964` | `32.521841` | 8/8 token match |
| Single row-pair fusion opt-in | 262,144 | 8 | `34.380968` | `32.232157` | 8/8 token match |
| Event-ordered handoff default | 262,144 | 8 | `34.433252` | `32.281173` | 8/8 token match |
| Event-ordered handoff default | 1,048,576 | 4 | `21.771077` | `20.410385` | 4/4 token match |
| Batched attention projection F8 HMMA default | 262,144 | 8 | `33.697698` | `31.591592` | 8/8 token match |
| Single-token HMMA opt-in | 262,144 | 8 | `16.083451` | `15.078235` | 8/8 token match |
| Sprint 118 same-binary control | 262,144 | 8 | `33.502249` | `31.408359` | 8/8 token match |
| Per-slot shared pair-SwiGLU opt-in | 262,144 | 8 | `33.562643` | `31.464978` | 8/8 token match |
| Async slot chunk 4 opt-in | 262,144 | 8 | `11.483646` | `10.765918` | 8/8 token match |
| Promoted launcher default repeat | 262,144 | 8 | `33.540586` | `31.444300` | 8/8 token match |
| Pair-SwiGLU F8 HMMA default | 262,144 | 8 | `33.578236` | `31.479596` | 8/8 token match |
| Pair+down F8 HMMA opt-in | 262,144 | 8 | `33.674684` | `31.570016` | 8/8 token match |
| Production fused gate_up appliance | 262,144 | 8 | `33.589285` | `31.489955` | 8/8 token match |
| Shared-down F8 HMMA opt-in | 262,144 | 8 | `33.550415` | `31.453514` | 8/8 token match |
| Direct FFN delta opt-in | 262,144 | 8 | `33.360404` | `31.275379` | 8/8 token match |
| Same-binary separate gate/up control | 262,144 | 8 | `31.312694` | `29.355651` | 8/8 token match |
| Batched attention projection F8 HMMA default | 1,048,576 | 4 | `21.469010` | `20.127197` | 4/4 token match |
| Pair-SwiGLU F8 HMMA default | 1,048,576 | 4 | `21.455638` | `20.114660` | 4/4 token match |
| Production fused gate_up appliance | 1,048,576 | 4 | `21.403909` | `20.066165` | 4/4 token match |
| Shared-down F8 HMMA opt-in | 1,048,576 | 4 | `21.396331` | `20.059061` | 4/4 token match |
| Small-route opt-in | 1,048,576 | 4 | `20.249531` | `18.983935` | 4/4 token match |
| Pair+down F8 HMMA opt-in | 1,048,576 | 4 | `21.370925` | `20.035242` | 4/4 token match |

The last pre-Sprint107 committed baseline was Sprint 104 at `31.451185`
generated tok/s for 8-slot/256K and `20.026385` for 4-slot/1M.

## Recent Experiments

| Sprint | Experiment | Result | Decision |
|---|---|---|---|
| 213 | Routed FFN materialized split-reduce gate | V100 build passed; symbol exported; focused split-reduce correctness passed and improved the focused FFN sequence `0.1391 ms -> 0.1290 ms`; full scheduler smoke passed; served A/B was `60.655009` vs `60.236036` continuation tok/s, `16/16` token match, `43` graph captures and `129` launches with `0` failures | Keep as diagnostic/default-off; do not promote; next sprint should build a tile-local/persistent routed-FFN workbench |
| 212 | TP4/PP1 low-bit layer body pivot | New separate TP-only `tools/ds4-v100-tp4-turbomind-layer-smoke` built on V100; correctness passed at 96/192/384 routes; TP4 compute speedups were `2.335x`, `2.597x`, `3.707x`; resident-reduce total speedups were `1.078x`, `0.932x`, `0.967x` | Do not build TP4/PP1 runtime ownership next; return to monolithic/persistent low-bit routed-FFN or a better collective |
| 211 | TP8 TurboMind MXFP4 expert body | Separate TP-only TP8 low-bit smoke ran the public TurboMind ABI, but `mid_shard=256` failed correctness at 96/192/384 routes with NaNs despite `3.927x-4.189x` compute-only speedup | Reject current TP8 MXFP4 shard shape; pivot to TP4 control or design a shard-256 kernel |
| 210 | TP8 real layer-body fixture | Separate TP-only FP16 Tensor Core layer body passed 32/64/128 token gates with `0.614750/0.709350/0.796927 ms` total latency | Continue TP8 only by replacing the FP16 fixture with real low-bit expert work |
| 209 | TP8 one-layer prototype | Separate TP-only one-layer smoke passed 32/64/128 token gates with sharded 32-slot/256K F8 KV and total latencies `0.739408/0.876011/1.098461 ms` | Continue TP8 in new TP-only files; no PP scheduler integration |
| 208 | Separate TP8 investigation path | TP8 planner/probes showed 32-slot/256K fits with sharded KV at `26.84 GiB` worst GPU and the 43-layer boundary proxy measured `29.381/32.605/37.995 ms` at 32/64/128 tokens | Continue bounded TP8 prototypes in separate files |
| 205 | Async root resident TP4 reduction | V100 build passed; `root_async` correctness passed; speedups were `0.970x` at 96 routes x 4 layers, `0.866x` at 768 routes x 4 layers, and `0.860x` at 96 routes x 43 layers | Reject root_async; pause TP4 production decode branch and pivot to persistent fused routed-FFN |
| 204 | Concurrent resident TP4 reduction | V100 build passed; `doubling_async` correctness passed at 96/768 routes; 43-layer 768-route speedup was `1.071x`, but 43-layer 96-route repeat was `0.896x` | Keep TP4 for larger batched/prefill investigation only; do not integrate production decode scheduler without a fused/NCCL-grade collective |
| 203 | Resident TP4 layer-slice gate | V100 build passed; resident TP4 correctness passed at 6/96/768 routes; 43-layer root speedup was `0.825x` at 96 routes and `0.589x` at 768 routes; hand-rolled doubling was slower than root in 4-layer tests | Do not wire this TP4 boundary into production; next TP work needs a real concurrent collective/fused reduction, otherwise pivot back to persistent fused routed-FFN |
| 202 | TP4 routed-FFN compute envelope | V100 build passed; fixed a GPU0 stream/workspace overlap in the benchmark warmup; real TurboMind MXFP4 TP4 split correctness passed at 6/96/768 routes; corrected compute-only speedup was `2.686x`, `2.350x`, `3.636x`; copy-inclusive speedup was `0.986x`, `0.783x`, `0.682x` | TP4 compute is strong enough for full-layer TP/EP, but routed-only full-hidden copy overlays are rejected |
| 201 | TP4 full-layer boundary proxy | V100 build passed; 16-token/43-layer/4-collective boundary measured `22.113369 ms` root and `24.414061 ms` doubling, both verified; 64-token doubling measured `34.830881 ms`, 128-token doubling measured `51.026125 ms` | Full-layer TP4/EP remains plausible only as a broad topology that keeps dense+routed compute inside the boundary; do not expand routed-only TP overlays |
| 103 | Exact-bit E4M3 F8 decode replacing `ldexpf()` | Improved 8-slot/256K to `30.862791` generated tok/s and 4-slot/1M to `19.733742` | Shipped |
| 104 | Warp-shuffle reductions for hot F8 arena kernels | Improved 8-slot/256K repeat to `31.451185`; 4-slot/1M to `20.026385` | Shipped; current baseline |
| 105 | Extend warp reductions to BF16/F32 matmuls | Correct, but repeat result was inside Sprint 104 band | Rejected and reverted |
| 106 | Warm served `nvprof` profile of Sprint 104 | F8 rows2/grouped rows2 were ~51% GPU time; TurboMind SM70 MXFP4 was ~25%; GPU memcpy traffic was small | Use profile to choose next kernel target |
| 107 | DS4-specific grouped F8 rows2 attention-output-A kernel | Correct and faster for 8-slot/256K; neutral for 4-slot/1M | Shipped |
| 108 | TurboMind small-route count/prefix/scatter fusion | Correct; 8-slot repeat was `31.759013` opt-in vs `31.794180` rollback, while 4-slot/1M was `20.249531` opt-in vs `20.081695` rollback | Kept opt-in |
| 109 | F8 four-output-row CTA probe | Correct; regressed 8-slot/256K to `30.998275` vs `31.380225` control and 4-slot/1M to `19.898462` vs `20.041787` control | Rejected as default; opt-in only |
| 110 | TurboMind fused gate+up grouped-GEMM probe | Correct; `1.504x`, `1.532x`, and `1.462x` faster at 6, 24, and 48 total routes | Proceed to appliance implementation |
| 111 | Production fused TurboMind gate_up appliance | Correct; 8-slot/256K improved to `33.430971` from `31.312694` same-binary separate control | Shipped/default for fused packs |
| 112 | Fused appliance profile plus F8 warp-scale probe | Profile showed F8 row-pair/grouped kernels at `54.58%` GPU time; warp-scale was correct but regressed 8-slot/256K to `29.009399` vs `33.484099` control | Kept opt-in/off |
| 113 | Direct FFN delta accumulation and cached FFN input ptr table | Correct, but `33.360404` vs `33.589285` control at 8-slot/256K | Kept opt-in/off |
| 114 | DS4-shaped shared-down F8 HMMA batch kernel | Correct; `33.550415` vs `33.397763` control at 8-slot/256K and `21.396331` vs `21.365610` at 4-slot/1M | Kept opt-in/off |
| 115 | DS4-shaped shared gate/up SwiGLU F8 HMMA kernel | Correct; `33.578236` vs `33.292541` control at 8-slot/256K and `21.455638` vs `21.430420` at 4-slot/1M | Shipped/default |
| 116 | DS4-shaped attention projection F8 HMMA batch kernel | Correct; `33.697698` vs `33.380614` control at 8-slot/256K and `21.469010` vs `21.333447` at 4-slot/1M | Shipped/default for active 4/8-slot batches |
| 117 | F8 wrapper shape trace and per-slot shared gate/up/SwiGLU fusion | Correct; trace showed the fast path is per-slot stage-pipelined, chunk-4 batching dropped to `11.483646`, and scalar shared-pair fusion reached `33.562643` | Kept opt-in/off; next target should be software-pipelined/Tensor-Core fusion |
| 118 | Single-token HMMA for the hot `4096 x 8192` F8 projection | Correct and traced, but regressed to `16.083451` vs `33.502249` same-binary control | Kept opt-in/off; naive n=1 WMMA is not viable |
| 119 | Event-ordered stage handoff | Correct; `34.433252` vs `33.379839` at 8-slot/256K and `21.771077` vs `21.566859` at 4-slot/1M | Shipped/default as `DS4_V100_ASYNC_EVENT_HANDOFF=auto` |
| 120 | Single shared gate/up/SwiGLU row-pair probe | Correct; `34.380968` row-pair vs `34.490294` default and `34.689964` scalar single-fusion at 8-slot/256K | Kept opt-in/off; row-pair compaction does not beat the default |
| 121 | 16-slot 256K throughput mode | Correct; `43.659461` at 16-slot/256K vs `34.445844` same-binary 8-slot control | Shipped as admitted 256K mode with context-aware launcher guard |
| 122 | 16-slot profile, 16-token HMMA admission, async chunk probes, and rendezvous stabilization | Correct; best `43.730215`, production-auto `43.534061`, one 16-request tensor batch after 200 ms auto wait; chunked tensor scheduling regressed (`28.876459` at chunk 2, `18.447169` at chunk 4, `13.315378` at chunk 16) | Shipped 16-slot auto rendezvous; kept chunk/output-B/shared-down probes opt-in/off |
| 123 | Production-path shared FFN fusion A/B | Correct; scalar shared-pair fusion reached `43.887206`, fused shared-down-add reached `43.539555`, and combined scalar+down-add reached `43.812630` at 16-slot/256K | Kept opt-in/off; launch/epilogue fusion alone is not enough |
| 124 | TurboMind route-row reduce replacing packed output clear plus atomic scatter-add | Correct; first candidate reached `43.822500`, but the repeat was `42.998450` vs `43.517862` control repeat at 16-slot/256K | Kept opt-in/off; routed-FFN tail fusion alone is not enough |
| 125 | Batched grouped attention output-A probe | Correct; output-A rows2 batching reached `43.640921`, rows2 A+B reached `43.619996`, and HMMA A+B reached `43.245208` vs `43.503005` control at 16-slot/256K | Kept opt-in/off; another single projection boundary is too small |
| 126 | Production routed-expert stage profiler | Correct; full 43-layer profile showed fused gate/up at `47.0%`, down at `23.4%`, route build at `16.8%`, and SwiGLU at only `3.2%` of profiled routed-FFN time; no-profile served sanity was `43.453309` generated tok/s | Shipped default-off diagnostic; next target should be TurboMind gated epilogue/interleaved pack or deeper persistent routed-expert pipeline |
| 127 | TurboMind gated-SiLU epilogue with interleaved fused gate/up appliance pack | Correct; standalone grouped test showed `1.47x-1.55x` speedup vs separate gate/up, full 43-layer gated profile removed standalone SwiGLU and dropped profiled routed-FFN total from `28.242 ms` to `26.734 ms`, served A/B was `43.933293` vs `43.691032` control | Keep opt-in/off; format and epilogue fusion are valid, but the next material step is a persistent routed-expert pipeline |
| 128 | TurboMind compact active-expert schedule | Correct; compact schedule passed full 43-layer smokes on both the interleaved gated and existing fused appliances, improved served A/B from `43.879880` to `46.328184` on the gated appliance, and the launcher-default fused appliance reached `45.888778` | Shipped/default as `DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1`; keep gated-SiLU and route-row-reduce opt-in |
| 129 | TurboMind dispatch policy probe | Correct for `default` and `reuse`; full scheduler `measure` aborted inside TurboMind's measurer, while served `reuse` was `45.813841` vs `45.840691` default | Keep default dispatch; guard unsafe measure/append; move to DS4-specific persistent routed-FFN |
| 130 | Routed FFN software-pipeline targeting | Correctness held; compact fused route-row-reduce repeated at `45.660765` vs `45.837745` control, confirming final scatter/reduce fusion is not the lever | Keep route-row-reduce opt-in; next code should target the packed MXFP4 gate/up mainloop with DS4-specific software pipelining |
| 131 | TurboMind indexed-A routed activation probe | Correct; full 43-layer smokes passed with indexed-A off/on, and served A/B was `45.789937` vs `45.663281` control | Keep indexed-A opt-in; wrapper-level activation compaction is correct but not a promotion-level win |
| 132 | Production-shaped TurboMind gate/up benchmark | Correct; historical 6/24/48-route cases still pass, and the 96-route served-profile case shows gated-SiLU at `0.1776 ms` vs `0.2889 ms` separate gate+up | Use this as the benchmark harness for any lower-level SM70 mainloop probe; no appliance default change |
| 133 | Compact-group gate/up benchmark correction | Correct; at 96 routes, sparse256 gated is `0.2128 ms` while compact gated is `0.1740 ms`, and compact separate gate+up is already `0.1895 ms` | Future probes must beat compact gated, not sparse grouped overhead |
| 134 | Fixed-shape compact gate/up ABI probe | Correct; direct fixed SM70 launch was bit-identical and `0.1746 ms` vs `0.1746 ms` generic gated | Do not promote; generic TurboMind already selects this effective path |
| 135 | 32-slot 128K throughput admission | Correct; full 43-layer smoke passed, and 32-slot 128K served at `52.840889` vs `45.780913` same-context 16-slot control | Ship as explicit short-context throughput mode; test wider short-context admission and lower-level software-pipelined kernels next |
| 136 | 64-slot 64K throughput admission | Correct; full 43-layer smoke passed, and 64-slot 64K served at `57.322945` vs `52.884400` same-context 32-slot control | Ship as explicit short-context throughput mode; diminishing slot-width returns make software-pipelined expert kernels the next major lever |
| 137 | 128-slot 32K throughput admission | Correct; full 43-layer smoke and status/metrics confirmed 128 slots, and served throughput reached `59.598172` vs `57.170428` same-context 64-slot control | Ship as explicit short-context throughput mode; stop treating admission width as the main lever and move to software-pipelined expert kernels |
| 138 | Wide compact TurboMind gate/up benchmark | Correct; default compact benchmark now covers up to 768 routed rows, where fused gate_up is `0.6379 ms` and gated-SiLU is `0.6481 ms` | Use `0.638 ms` as the acceptance target for the next packed MXFP4 software-pipelined kernel probe |
| 139 | Fixed-shape 128-slot gate/up probe | Correct; the 768-route m128 probe measured `0.5999 ms` vs `0.6480 ms` generic gated in isolation, passed full 43-layer 128-slot smoke, and served at `60.130047` vs `60.061899` probe-off | Keep guarded auto selection, but do not treat gate/up-only fusion as the remaining major lever |
| 140 | Fixed-shape 128-slot down probe | Correct; down m128 measured `0.3026 ms` vs `0.3272 ms` generic in isolation and full 43-layer smoke passed, but served A/B was `60.038469` vs `60.129772` down-probe-off | Keep down probe opt-in/off; move to down epilogue plus weighted reduce or a persistent routed-FFN executor |
| 141 | Half2 route-row reduce tail probe | Correct; full 43-layer 128-slot smoke passed, but 128-slot served A/B was neutral: half2 route-row reduce `60.104512`, scalar route-row reduce `60.112248`, control `60.108232` | Keep half2 reduce opt-in/off; separate tail-kernel vectorization is not enough |
| 142 | TurboMind down-epilogue reduce probe | Correct; full 43-layer 128-slot smoke passed and served A/B was `60.041003` vs `59.987105` control | Keep off by default; atomic epilogue fusion proves the integration boundary but is not a material throughput win |
| 143 | Prefill/decode metric split | Correct; V100 one-request smoke reported aggregate prompt `6.841274`, generated `0.760142`, continuation `0.380071`, and response-local prompt/decode rates | Ship benchmark visibility change; no runtime default change |
| 144 | SM70 MXFP4 m64n256 tile probe | Correct; full 43-layer smoke passed, but served 128-slot/32K A/B regressed: control `59.993301`, down `m64n256` `59.791839`, gate `m64n256` `59.797232` | Keep explicit opt-in only; larger routed-FFN executor work is still the next lever |
| 145 | 256-slot 16K short-context admission | Correct; planner worst GPU was `29.07 GiB / 32.00 GiB` including reserve, full 43-layer smoke passed, and served 16K runs reached `59.860493` at 128 slots, `60.700926` at 192 slots, and `61.065087` at 256 slots | Ship guarded 256-slot admission for `ctx <= 16K`; simple slot widening is now mostly exhausted |
| 146 | 1536-route fixed-shape gate/up and down probes | Correct; standalone gate `m128_1536` improved to `0.9435 ms` vs `0.9651 ms`, but served A/B was `61.204203` vs `61.223893` control and continuation/decode was `57.378940` vs `57.397400` | Keep explicit opt-in only; do not select 1536 probes from `auto` |
| 147 | 1536-route down-reduce correctness checkpoint | Correct; the down GEMM route-weighted F32 accumulation epilogue covers 1536 routes and passed full 43-layer 256-slot smoke | Keep opt-in pending served A/B |
| 148 | Stage-4 fused gate/up software-pipeline probe | Correct; isolated `m128_s4` improved the 768-route probe but served A/B was only `60.049057` vs `59.865668`, inside the run band | Keep stage-count variants explicit opt-ins |
| 149 | TP split and P2P topology probe | Correct; ideal 2-way FFN split was positive before communication and NV2 payloads were fast enough for a bounded prototype | Prototype only on clean NV2 pairs |
| 150 | Two-GPU TP split proxy | Correct; 768 routes were positive after copies, but 1536 routes were neutral to slower | Scope TP to 128-slot/32K first |
| 151 | Two-GPU TP correctness gate | Correct; full one-GPU output matched the sum of two TP partials at 768 and 1536 routes on NV2 pairs | Split math accepted; scheduler remains the risk |
| 152 | Fused gate/up stage-count sweep | Correct; 2/3/4-stage variants were neutral at 768 and 1536 routes, and NCU counters were flat | Stop tuning gate/up stage count |
| 153 | Bounded TP pack contract | Correct; `--emit-tp-split` emits split gate/up and down rows and context binding accepts TP descriptors. Real 2-GPU NV2 proxy was `1.157x` at 768 routes and `0.912x` at 1536 routes | Keep TP to a one-layer 128-slot/32K prototype |
| 154 | Fused routed-FFN boundary validation | Correct; down-reduce epilogue served A/B was flat at 128-slot/32K and slightly slower at 256-slot/16K, while profile kept gate/up at `~58-61%` and down at `~25-29%` | Keep down-reduce opt-in; next work must change gate/up+down execution, not only the epilogue |

## Sprint 106 Profile Takeaway

The profile does not point at disk or host RAM as the decode bottleneck.
`cudaMemcpy` API accounting is noisy, but GPU memcpy time was tiny. The main
device buckets were:

- F8 rows2 arena matmul: `38.97%`
- F8 grouped rows2 arena matmul: `12.39%`
- TurboMind SM70 MXFP4 grouped GEMM: `25.42%`

That makes the practical next targets F8 execution shape and TurboMind routed
expert scheduling.

## Current Shipped Change

Sprint 107 adds a guarded DS4-specialized CUDA kernel for the fixed grouped
attention-output-A shape:

- groups: `8`
- rows per group: `1024`
- columns per group: `4096`
- fallback: existing generic grouped rows2 kernel
- rollback knob: `DS4_V100_CUDA_F8_GROUPED_DS4_FAST=0`

Validation already completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- `cuda_v100_selected_token_smoke --expected-token-hex 3136`: passed

Throughput completed:

- 8-slot/256K fast: `31.811137` generated tok/s, `8/8`
- 8-slot/256K fast repeat: `31.630774` generated tok/s, `8/8`
- 8-slot/256K rollback: `31.098630` generated tok/s, `8/8`
- 4-slot/1M fast: `20.095510` generated tok/s, `4/4`
- 4-slot/1M rollback: `20.105807` generated tok/s, `4/4`

Remaining optional validation:

- Focused profile to confirm whether grouped F8 kernel time moved.

## Current Opt-In Probe

Sprint 108 adds a guarded small-route TurboMind route builder:

- combines route count, prefix, and scatter into one one-block kernel for the
  production small-route shape;
- is controlled by `DS4_V100_TURBOMIND_SMALL_ROUTE_BUILD`;
- remains disabled by default because it did not improve the 8-slot/256K
  practical target.

Validation completed on the cluster:

- `cuda_v100_turbomind_adapter_smoke`: passed
- `cuda_v100_stage_scheduler_smoke --stage 0 --slots 4`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with small-route on: passed, token id `926`, hex `3136`
- selected-token smoke with small-route off: passed, token id `926`, hex `3136`
- rebuilt default check: `turbomind_small_route_build=0`, selected-token passed

Sprint 109 adds a guarded F8 row4 CTA probe:

- computes four large F8 output rows per CTA for the ungrouped and DS4 grouped
  attention-output paths;
- is controlled by `DS4_V100_CUDA_F8_ROW4`;
- remains disabled by default because it reduced throughput in both measured
  serving tiers.

Validation completed on the cluster:

- `cuda_source_dtypes_smoke`: passed
- `cuda_v100_projection_attention_smoke`: passed
- `cuda_v100_full_scheduler_smoke --slots 8`: passed
- selected-token smoke with row4 on: passed, token id `926`, hex `3136`

Sprint 110 adds a standalone TurboMind fused gate/up benchmark:

- shape: `K=4096`, `N=2048`, fused `N=4096`, 256 experts;
- route set: six sparse active experts;
- result: fused gate_up is `1.46x-1.53x` faster than separate gate and up
  grouped calls;
- correctness: exact output match for both halves of the fused tensor.

Sprint 111 ships that fused gate/up result into the appliance:

- packer emits `blk.N.ffn_gate_up_exps.weight` with `--fuse-gate-up`;
- runtime defaults to the fused path with `DS4_V100_TURBOMIND_FUSED_GATE_UP=1`;
- selected-token smoke passed with token id `926`, hex `3136`;
- full scheduler smoke passed with `tm_layers=43`;
- 8-slot/256K served A/B improved from `31.312694` to `33.430971`
  generated tok/s;
- 4-slot/1M fused sanity passed at `21.403909` generated tok/s.

Sprint 112 profiles the fused appliance and tests a narrow F8 scale-hoist
variant:

- fused 8-slot/256K profile reached `33.972205` generated tok/s under the
  profiler harness and preserved `8/8` token matches;
- F8 row-pair plus DS4 grouped attention-output kernels were `54.58%` of GPU
  time after Sprint 111;
- warp-broadcast E8M0 scale loading passed source/projection, scheduler, and
  selected-token correctness;
- same-binary 8-slot/256K A/B regressed from `33.484099` to `29.009399`
  generated tok/s, so `DS4_V100_CUDA_F8_WARP_SCALE=0` remains the default.

Sprint 113 tests direct FFN delta accumulation:

- batch scratch now exposes contiguous FFN norm/delta tensors with stable
  per-slot views;
- TurboMind routed FFN wrappers can consume an existing device pointer table;
- TurboMind routed FFN wrappers can accumulate into an existing output tensor;
- selected-token correctness passed with `DS4_V100_FFN_DIRECT_DELTA=1`;
- same-binary 8-slot/256K A/B was `33.360404` generated tok/s with direct delta
  versus `33.589285` control, so `DS4_V100_FFN_DIRECT_DELTA=0` remains the
  default.

Sprint 114 tests a DS4-shaped shared-down F8 HMMA batch kernel:

- the kernel is guarded by `DS4_CUDA_F8_HMMA_SHARED_DOWN=1`;
- it only dispatches for `rows=4096`, `cols=2048`, and `n_tokens=4/8`;
- focused target-shape smoke passed against the existing scalar F8 path;
- full scheduler and selected-token smokes passed with the fused appliance;
- same-binary A/B showed small positive deltas, but not enough to promote:
  `33.550415` vs `33.397763` at 8-slot/256K and `21.396331` vs `21.365610`
  at 4-slot/1M.

Sprint 115 ships a DS4-shaped shared gate/up SwiGLU F8 HMMA batch kernel:

- the kernel is guarded by `DS4_CUDA_F8_HMMA_PAIR_SWIGLU=1`;
- it only dispatches for `rows=2048`, `cols=4096`, and `n_tokens=4/8`;
- focused pair-SwiGLU smoke, full scheduler, and selected-token smokes passed;
- same-binary A/B improved both measured tiers:
  `33.578236` vs `33.292541` at 8-slot/256K and `21.455638` vs `21.430420`
  at 4-slot/1M;
- the combined pair+shared-down HMMA path reached `33.674684` at 8-slot/256K
  but regressed 4-slot/1M to `21.370925`, so only pair-SwiGLU HMMA is default.

Sprint 116 ships a DS4-shaped batched attention projection F8 HMMA path:

- the remaining profile after Sprint 115 still showed ungrouped F8 row-pair
  matmuls as the largest device bucket: `41.65%` GPU time and `12,341` calls;
- the new kernels cover `attn_q_a` (`1024 x 4096`), `attn_kv_latent`
  (`512 x 4096`), and `attn_q_b` (`32768 x 1024`) for active 4/8-slot batches;
- `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` and
  `DS4_V100_CUDA_F8_HMMA_ATTN_BATCH=1` are now launcher defaults, while
  non-4/8-slot batches stay on the per-slot projection path unless
  `DS4_V100_ENABLE_BATCH_ATTN_PROJ_ANY=1` is set;
- focused CUDA smoke, full scheduler, and selected-token oracle all passed;
- same-binary A/B improved both measured tiers:
  `33.697698` vs `33.380614` at 8-slot/256K and `21.469010` vs `21.333447`
  at 4-slot/1M.

## Next Target

The next target still needs to change a larger execution boundary. Sprints
123-154 show that per-slot shared-FFN fusion, route-row tail fusion, dispatch
bypass, tile probes, fixed-shape 768/1536-route probes, stage-count tuning,
epilogue-only down-reduce fusion, bounded TP probes, and simple slot widening
are correct but too small to close the practical serving gap. Aggregate
throughput is still about `61` tok/s at 256-slot/16K and about `46-71` tok/s at
16-slot/256K depending on async-pipeline era and test harness, far below the
practical serving target. Sprint 194 adds a topology estimator and changes the
TP decision rule: the existing routed-only TP2 overlay should not be expanded,
because at 16-slot/256K it moves about `21.531 MiB` per token without changing
the dense execution shape, versus `7.000 MiB` for the current layer split. Full
TP/EP moves more wire bytes (`112.875 MiB` for TP4/PP1 at the same tier), so it
is only worth implementing if attention, shared FFN, routed experts, and output
ownership all become native to the topology. Sprint 195 then measured the first
TP4 primitive directly: a root gather/reduce/broadcast hidden collective is
correct on the V100 NVLink islands, but costs about `0.11 ms` for the
16-token/4096-hidden decode payload and only reaches about `27 GB/s` effective
wire bandwidth at larger payloads. The next sprint should not build production
TP4 on that root collective. It should either implement a real ring/tree/NCCL
TP4 collective inside one four-GPU NVLink island, or return to the monolithic
routed-FFN kernel that removes the global `mid_half` boundary.
Sprint 196 implemented the repo-owned tree-style option as recursive doubling.
It is correct and materially faster for larger payloads (`1.656 ms` vs
`3.676 ms` root at 1024 x 4096), but it is slower at the actual 16-token decode
payload (`0.134 ms` vs `0.111 ms`). That narrows the next production path:
direct TP4 collectives are not the decode lever unless fused into a larger
persistent boundary. The next sprint should prioritize the monolithic
routed-FFN or persistent layer boundary, while keeping the doubling collective
as the baseline for later TP4/prefill work.
Sprint 197 added the missing liveness contract to the TurboMind profile. The
current `fused6_reduce` production-shaped path now proves compact activation
staging is selected, `down_routes` is elided, and `mid_half` remains
materialized on every routed FFN call. The remaining `mid_half` buffer is only
`24576` bytes per six-route call, so a pure buffer-elision sprint is unlikely to
move serving throughput. The next implementation should target a persistent or
tile-level gate/up+down executor that reduces launch/GEMM boundary cost, or
use the TP4 doubling collective for larger batched/prefill shapes where it
actually wins.
Sprint 198 reopened CUDA graph replay for the current `fused6` /
`fused6_reduce` routed executor path. Direct replay with `fused6_reduce`
matched output IDs and improved continuation throughput from `16.022442` to
`17.980888` tok/s, with `43` graph captures, `129` launches, and `0` failures.
This is not promoted: Sprint 169 showed graph replay can help direct replay
while regressing served throughput. The required next gate is a same-binary
16-slot/256K served A/B before treating graph replay as a practical serving
optimization.
Sprint 199 ran that missing served gate and promoted the combined
`fused6_reduce + graph replay` stack for the Sprint 181+ production V100
appliance pack. Same-binary 16-slot/256K serving improved from `54.725463`
generated / `53.870377` continuation tok/s with `fused6_reduce` graph off to
`67.886268` / `66.825545` with graph on, with `16/16` token match. Against the
routed-executor-off production control (`56.719099` / `55.832863`), the
promoted stack is about `+19.7%` continuation. The launcher and V100 env
template now default to `DS4_V100_TURBOMIND_GATED_SILU=1`,
`DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fused6_reduce`, and
`DS4_V100_TURBOMIND_GRAPH=1` for the production pack.
Sprint 200 added the missing exact six-route focused TurboMind bench and
triggered the persistent-kernel stop condition. The fixed `m16_6` gated-SiLU
probe measured `0.1196 ms` versus `0.0946 ms` for the generic gated-SiLU path,
which confirms the Sprint 199 promoted stack should not use the fixed six-route
gate/up probe. The down-reduce output clear measured only `0.0022 ms`, while
six-route down-reduce with clear measured `0.0650 ms`; a clear-only fusion is
therefore not material. The next sprint should pivot to bounded full-layer
TP4/EP rather than adding another six-route wrapper ABI.
Sprint 201 added the bounded TP4 layer-boundary proxy and measured the full
43-layer communication envelope directly on the V100 pod. The 16-token target
shape costs `22-24 ms` before any DS4 compute, which is acceptable only if TP4
changes the whole layer execution shape. Larger active-token runs improve the
overhead-only envelope to `1837 tok/s` at 64 tokens and `2509 tok/s` at
128 tokens, so TP4 is more promising for high-batch throughput/prefill than
low-batch decode latency.
Sprint 202 measured the matching routed-expert compute side with a four-GPU
TurboMind MXFP4 split. It also fixed a benchmark warmup bug where the full
reference and shard 0 shared the GPU0 workspace on different streams. TP4
compute itself is strong (`2.35x-3.64x` at practical route counts), but
conservative routed-only input/output copies erase it (`0.68x-0.78x`). This
confirms that the next TP sprint must be a full-layer resident boundary or not
TP at all.
Sprint 203 implemented that resident boundary as a benchmark slice. Correctness
passes, but the naive root all-reduce boundary is still slower than the
one-GPU reference (`0.825x` at 96 routes over 43 layers, `0.589x` at 768
routes), and the simple doubling variant is worse in this implementation. TP4
production work should not continue into the scheduler until the collective is
made concurrent/fused; otherwise the next practical serving sprint should
return to a persistent fused routed-FFN executor.
Sprint 204 made that boundary concurrent with per-device async doubling. The
larger 768-route 43-layer shape improved to `1.071x`, but the production
96-route 43-layer repeat was `0.896x`. This keeps TP4 alive for larger
batched/prefill work, not for immediate production decode integration.
Sprint 205 tested async root gather/reduce/broadcast for small-payload decode.
It was correct but slower than the one-GPU reference and slower than
`doubling_async`, with `0.860x` at the 96-route 43-layer gate. The current TP4
decode branch is now blocked; the next sprint should return to persistent fused
routed-FFN work.

The concise current status is also tracked in
`docs/sprints/EXPERIMENT-STATUS.md`.
