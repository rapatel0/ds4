---
created: 2026-05-17
last_updated: 2026-05-21
last_updated_by: sprint-execute
revision: 152
---

# Vision: DS4 V100 Appliance

## North Star

Build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB cluster that runs
the high-intelligence source quantized model from pure device-resident packs by
default, preserves model quality, and reaches a verified deployed serving path
before broad throughput tuning.

The sprint sequence should keep format, topology, sharding, and scheduling
decisions explicit. The project is not trying to become a generic GGUF runner;
it is a narrow DS4 runtime tuned for this hardware.

With readiness now closed, the next north star is practical serving: keep the
source-model quality and device-resident appliance contract, but move from a
correctness-first runtime to a high-utilization decode runtime with sustained
multi-token benchmarking, continuous batching, MTP draft commit where safe, and
optimized V100 low-bit expert kernels in the actual hot path.

## Current State

- Sprints 001-004 proved the memory and pack-residency foundation: the source
  model is inventoried, manifested, packed into per-GPU shards, reconciled, and
  uploaded to all 8 V100s as CUDA device memory.
- The source model generation guard is still active by design for normal
  serving. Sprint 007 added only a bounded CPU diagnostic oracle path for
  source-layout correctness evidence.
- Sprint 005 proved a first resident source-dtype diagnostic: native BF16
  `token_embd.weight` bytes can be gathered from V100 device memory and expanded
  to F32 bit-exactly. This is not native BF16 math; V100 production GEMMs must
  target FP16 tensor cores or the selected low-bit/integer kernels.
- The tightest observed residency case still leaves more than the planned 3 GiB
  reserve on a 32 GB V100. Weight VRAM fit is no longer the primary blocker.
- The readiness ladder is now closed through Level 6. The main remaining risk
  is practical-use performance: sustained decode throughput, GPU utilization,
  multi-token batching, true MTP draft commit, and hot-path low-bit kernel
  integration.
- Sprint 111 shipped the DS4-shaped fused TurboMind gate/up path into the
  appliance. The fused pack preserves the same per-GPU shard sizes, binds
  `43/43` routed layers through TurboMind metadata, passes selected-token
  correctness, and improves the 8-slot/256K served target from `31.312694` to
  `33.430971` generated tok/s in a same-binary A/B. The fused 4-slot/1M sanity
  run reached `21.403909` generated tok/s.
- Sprints 112-116 tested hot-path cleanup around the fused appliance.
  F8 warp-scale hoisting and direct FFN-delta accumulation both passed
  correctness but regressed the primary 8-slot/256K A/B, so they remain opt-in.
  The Sprint 114 shared-down F8 HMMA path also passed correctness but remains
  opt-in because the pair+down combination regressed 4-slot/1M. Sprint 115
  shipped shared gate/up SwiGLU F8 HMMA as the new launcher default, improving
  same-binary 8-slot/256K to `33.578236` and 4-slot/1M to `21.455638`. The
  combined pair+down opt-in reached `33.674684`, but it is not the default.
  Sprint 116 then shipped batched attention-projection F8 HMMA for active
  4/8-slot batches, improving same-binary 8-slot/256K to `33.697698` and
  4-slot/1M to `21.469010`. Sprints 117-118 showed scalar single-slot fusion
  and naive single-token WMMA are not viable throughput levers. Sprint 119
  shipped event-ordered handoff as the multi-slot per-step default, raising the
  measured 8-slot/256K appliance target to `34.433252` and 4-slot/1M to
  `21.771077`. Sprint 120 implemented a row-pair per-slot shared
  gate/up/SwiGLU probe, but it did not beat the default. Sprint 121 raised the
  active-slot ceiling to 16 for the 256K tier and measured `43.659461`
  generated tok/s with `16/16` token matches, versus `34.445844` for the
  same-binary 8-slot control. Sprint 122 stabilized that mode by making launcher
  `auto` rendezvous wait 200 ms at 16 active slots; production-auto now reaches
  `43.534061` generated tok/s with one 16-request tensor batch, and the best
  observed candidate reached `43.730215`. Sprint 122 also showed that chunking
  slots to feed wider batch kernels regresses because it loses stage overlap.
  Sprint 123 added an opt-in fused shared-down-add F8 epilogue and re-tested
  per-slot shared gate/up/SwiGLU fusion at 16 slots. The path stayed correct
  and reached `43.887206` generated tok/s in the best opt-in run, but the gain
  did not clear the promotion bar, so production defaults are unchanged.
  Sprint 124 then added a correct opt-in TurboMind route-row reduce that
  replaces the packed output clear plus atomic scatter-add; it reached
  `43.822500` in the first candidate run but repeated at `42.998450` versus a
  `43.517862` control repeat, so it also remains opt-in. Sprint 125 added
  batched grouped attention output-A. The rows2 path was correct and reached
  `43.640921` versus a `43.503005` control, while the fixed HMMA grouped
  output-A path regressed to `43.245208`; defaults remain unchanged. Sprint 126
  added a default-off routed-expert stage profiler and revalidated the current
  binary at `43.453309` generated tok/s with `16/16` token match. The 43-layer
  profile showed fused gate/up at `47.0%`, down at `23.4%`, route build at
  `16.8%`, and standalone SwiGLU at only `3.2%` of profiled routed-FFN time.
  Sprint 127 then implemented that bounded TurboMind gated-SiLU step with an
  interleaved fused gate/up appliance pack. It passed standalone, stage, full
  43-layer, profile, and served A/B validation; the routed profile dropped from
  `28.242 ms` to `26.734 ms` by removing standalone SwiGLU, and the served
  A/B moved from `43.691032` to `43.933293` generated tok/s. Sprint 128 then
  compacted the packed TurboMind grouped schedule so the gate/up and down GEMMs
  see at most `total_routes` groups instead of the full 256-expert schedule.
  That is now a launcher default: the existing fused appliance reached
  `45.888778` generated tok/s, and the interleaved gated appliance with compact
  plus route-row-reduce opt-in reached `46.394722`. Sprint 129 then exposed
  TurboMind dispatch policy selection. The safe `reuse` policy was neutral at
  `45.813841` versus a `45.840691` default control, and unsafe `measure`
  aborted the full scheduler inside TurboMind's measurer. Sprint 130 repeated
  the closest existing FFN epilogue-fusion analogue on the current compact
  fused appliance: route-row-reduce was `45.660765` versus a `45.837745`
  control, both correct. Sprint 131 then added a correct opt-in TurboMind
  indexed-A route that avoids route-expanded gate/up activation materialization,
  but it measured `45.789937` versus `45.663281` control. Sprint 132 extended
  the standalone TurboMind gate/up benchmark to the production 96-route shape:
  interleaved gated gate/up passed at `0.1776 ms` versus `0.2889 ms` separate
  gate+up, a `1.626x` isolated speedup. Sprint 133 corrected that benchmark to
  the served compact active-expert topology: compact gated-SiLU was
  `0.1740 ms` versus `0.1895 ms` separate gate+up, only `1.089x`.
  Sprint 134 then added a fixed-shape DS4 ABI that bypasses generic dispatch
  and directly launches the matching SM70 MXFP4 gated kernel; it was
  bit-identical and neutral at `0.1746 ms` versus `0.1746 ms` generic gated.
  Sprint 135 then widened the served scheduling shape where memory allowed it:
  32-slot 128K full scheduler smoke passed and the served appliance reached
  `52.840889` generated tok/s versus `45.780913` for the same-context 16-slot
  control. Sprint 136 added the next short-context tier: 64-slot 64K full
  scheduler smoke passed and served throughput reached `57.322945` generated
  tok/s versus `52.884400` for the same-context 32-slot control. Sprint 137
  added 128-slot 32K admission; full scheduler smoke passed, status/metrics
  confirmed the 128-slot binary, and served throughput reached `59.598172`
  generated tok/s versus `57.170428` for the same-context 64-slot control.
  256K remains capped at 16 slots to avoid overfilling 32 GB V100s. Sprint 138
  then updated the compact TurboMind gate/up benchmark to cover the high-slot
  served route shapes. The 768-route compact baseline is `0.6379 ms` for fused
  gate_up and `0.6481 ms` for gated-SiLU, giving the next software-pipelined
  MXFP4 kernel a concrete acceptance target. Sprint 139 added a fixed-shape
  768-route m128 gated-SiLU probe and wired it into the production appliance
  under exact guards. It beat the isolated target at `0.5999 ms`, passed full
  43-layer 128-slot smoke, and served at `60.130047` generated tok/s on the
  interleaved gated appliance, but the same-binary probe-off control was
  `60.061899`, so gate/up-only specialization is not a material topline lever.
  Sprint 140 repeated the fixed-shape approach for the 768-route down
  projection. The isolated down probe was correct and faster (`0.3026 ms` vs
  `0.3272 ms`), but served A/B was slower with it enabled (`60.038469` vs
  `60.129772`), so it remains opt-in and default-off. Sprint 141 added a
  half2-vectorized route-row-reduce tail variant. It passed full 43-layer
  128-slot smoke, but served A/B stayed neutral: control `60.108232`, scalar
  route-row reduce `60.112248`, and half2 route-row reduce `60.104512`.
  Sprint 142 then moved the weighted route reduction into the TurboMind down
  GEMM epilogue for the exact 768-route high-slot shape. It passed full
  43-layer smoke and served correctly at `60.041003` generated tok/s versus
  `59.987105` same-binary control, but the improvement is only run-noise
  positive, so it also remains opt-in and default-off. Sprint 143 added
  explicit prompt/prefill versus continuation/decode metrics to the benchmark
  harnesses. Sprint 144 tested a wider-N SM70 MXFP4 tile and kept it opt-in
  after served 128-slot/32K A/B regressed. Sprint 145 widened the
  short-context admission ceiling to 256 slots at 16K after planner and
  full-scheduler validation. It served correctly at `61.065087` generated
  tok/s and `57.248519` continuation/decode tok/s, but only improved decode by
  about 2% over the 128-slot/16K control. Sprint 146 tested the matching
  1536-route fixed-shape gate/up and down probes for the 256-slot compact
  routed shape. The gate probe improved in isolation (`0.9435 ms` vs
  `0.9651 ms` generic gated), but served A/B was flat to slightly worse:
  `61.204203` generated tok/s and `57.378940` continuation/decode tok/s versus
  `61.223893` and `57.397400` control. The 1536-route probes stay explicit
  opt-ins and are not selected by `auto`. Sprint 147 extended the down-reduce
  epilogue to the 1536-route shape and passed full-scheduler smoke. Sprint 148
  tested a deeper stage-count software-pipeline variant of the fused MXFP4
  gate/up+gated-SiLU kernel; it improved the isolated 768-route probe but did
  not move served throughput or NCU counters materially. Sprint 149 measured a
  2-way TP split proxy for the routed FFN: the ideal compute speedup is
  `1.858x` at 768 routes and `1.468x` at 1536 routes before communication, and
  a 12 MiB hidden payload takes about `0.26 ms` over NV2, `0.52 ms` over NV1,
  and `1.29-1.31 ms` over SYS. Sprint 150 then ran a real two-GPU TP proxy:
  clean NV2 pairs show about `1.28x` total-with-copy speedup at 768 routes but
  `0.85-0.94x` at 1536 routes, so TP is a targeted 128-slot/32K candidate, not
  a broad 256-slot/16K solution yet. Sprint 151 added the missing correctness
  gate: full one-GPU down output matches the sum of the two TP partials at 768
  and 1536 routes on clean NV2 pairs, with `rel ~= 2.46e-04` and `bad=0`.
  Dispatch-policy tuning, dispatch bypass, final scatter fusion, wrapper-level
  activation compaction, separate tail-vectorization, atomic epilogue reduce,
  simple slot widening, fixed-shape gate/down probes, and basic gate/up launch
  fusion are therefore not the missing throughput lever. The project remains
  far below the practical
  serving target, so the next meaningful step is still larger
  execution-boundary work:
  a narrow DS4-only persistent grouped routed-expert pipeline that
  software-pipelines packed MXFP4 dequant, gate/up HMMA, gated activation, down
  HMMA, and weighted scatter/reduce for the current compact routed shape, or a
  bounded one-stage 2-GPU TP routed-FFN prototype on NV2 pairs for the
  128-slot/32K tier.
- Sprint 006 has shipped that context/skeleton contract. The project now has a
  verified 8-GPU V100 topology check, descriptor policy, HC relay smoke, and
  no-math layer walk over the real pack index, while source-layout generation
  remains guarded.
- Sprint 007 shipped a guarded CPU-only source-layout oracle. The official
  `short_reasoning_plain` fixture now selects the expected first token exactly
  on the cluster. The sprint corrected MXFP4 row layout to match GGML's
  `block_mxfp4` low-half/high-half nibble ordering and reset source-layout KV
  correctness to the default F16 cache contract.
- Sprint 008 shipped the bridge from CPU source oracle to first V100 source
  anchors: automated official-vector validation, source-layout guard checks,
  exact F16 KV admission by stage/context/slot, MXFP4 parity hardening, and a
  bounded CUDA F8_E4M3_B128 source-format row-decode probe on `sm_70`. Full
  V100 source-layout prefill execution is now the next runtime sprint.
- Sprint 009 shipped the first bounded V100 prefill/KV execution surface:
  deterministic F16 KV arena planning/allocation, V100 context allocation for
  256K and 1M single-slot tiers, guarded source-layout validation on the real
  model, and a CUDA `sm_70` diagnostic smoke that bridges F8 source rows into
  raw SWA, compressed KV, ratio-4 indexer KV, and compression-state surfaces.
- Sprint 010 shipped the stage-owned KV integration gate: deterministic
  per-layer KV/state subviews inside each GPU's `kv_arena`, V100 diagnostic
  writes through those subviews for ratio-4 and ratio-128 layers, real
  compressor recurrence smokes for attention and indexer-shaped paths, CPU
  references, and real-model guard validation. It did not ship dense
  projection, MoE, output-head logits, selected-token decode, or serving.
- Sprint 011 shipped the bounded source projection/attention gate: device
  source-F8 projection diagnostics from resident arenas, executable BF16/F32
  V100 policy checks, projection-fed ratio-128 and ratio-4
  attention/compressor smokes, and device-resident writes into stage-owned KV
  views. It did not ship full layer output, MoE, output-head logits,
  selected-token decode, or serving.
- Sprint 012 shipped the bounded source-BF16 output-head/logits gate and a
  runnable V100 appliance gate. The gate passes real-model source guards and
  all implemented CUDA smokes on the 8x V100 pod, but reports `ready=false`
  because full layer/MoE execution, full selected-token decode, public serving,
  MTP, and throughput benchmarks remain missing.
- Sprint 013 shipped the first source-MXFP4 routed expert execution surface and
  a bounded MoE/logits selected-token smoke. The gate now validates router
  selection, MXFP4 gate/up/down expert matmuls, SwiGLU accumulation, BF16
  output-head logits, and selected-token comparison on V100. It still reports
  `ready=false` because the path is synthetic and not yet bound to real
  pack-index layer descriptors or the full layer scheduler.
- Sprint 014 shipped the real pack-index layer descriptor gate. The appliance
  gate now validates layer-2 attention, compressor/indexer, router,
  routed/shared expert, HC control, and output-head descriptors from the real
  pack index on the 8x V100 pod. It still reports `ready=false` because
  descriptors are not yet materialized as runtime bindings and no real
  descriptor-bound layer compute has shipped.
- Sprint 015 shipped runtime tensor bindings and the first descriptor-bound
  real-byte FFN compute gate. The V100 pod now runs layer-2 routed MXFP4 plus
  shared F8 FFN bytes from the source GGUF at real pack offsets and compares
  the output against CPU source-format references. The gate still reports
  `ready=false` because real router scheduling, full attention/residual/norm
  layer execution, selected-token decode, serving, MTP, and throughput remain
  incomplete.
- Sprint 016 shipped descriptor-bound real router scheduling for the layer-2
  FFN slice. The V100 pod now computes router logits from real
  `ffn_gate_inp.weight` bytes, selects experts through the real
  `ffn_gate_tid2eid` hash table, executes all six selected MXFP4 routed experts
  plus the shared F8 expert, and compares the result against CPU source-format
  references. The gate still reports `ready=false` because scheduler-owned full
  layer execution, attention/residual/norm integration, real-model selected
  token decode, serving, MTP, and throughput remain incomplete.
- Sprint 017 shipped the scheduler-owned layer-state surface for the
  descriptor-bound router/FFN slice. The state binds real layer descriptors
  once, validates dimensions and router kind, carries layer/stage/KV metadata,
  exposes source row views, constructs selected routed expert matrices, and
  sizes the FFN arena span. The gate now includes `layer_state` and still
  reports `ready=false` because full layer output, real selected-token decode,
  serving, MTP, and throughput remain incomplete.
- Sprint 018 shipped the first descriptor-bound attention projection,
  residual, and norm gate from real source bytes. The layer state now owns
  attention FP8/control descriptors and the V100 pod validates real layer-2
  q/kv/output projection surfaces, residual add, and FFN pre-norm against CPU
  source-format references. This still is not full attention semantics: softmax
  over raw/compressed KV, semantic layer output, real selected-token decode,
  serving, MTP, and throughput remain incomplete.
- Sprint 019 shipped the first reusable hidden-vector layer execution surface.
  The V100 pod now validates layer-2 semantic attention over explicit raw plus
  compressed KV rows with sinks, grouped F8 attention output, residual,
  FFN pre-norm, real hash-router selected MXFP4 experts, shared F8 expert, and
  final next-hidden residual. The gate now includes `integrated_layer` and
  reports `ready=false` because full HC pre/post scheduling, real compressor/
  indexer descriptor binding, full 43-layer selected-token decode, serving,
  MTP, and throughput remain incomplete.
- Sprint 020 extended the runtime layer bridge with real compressor/indexer
  descriptor ownership and an executable DS4 HC-state layer entrypoint. The
  V100 pod now validates layer-2 `[4 x 4096]` HC attention pre/post and FFN
  pre/post around the hidden-vector body, and the full 8-GPU gate passes with
  `ready=false`. The remaining critical gap is executor-owned compressed-row
  generation and indexed ratio-4 compressed attention.
- Sprint 021 shipped executor-owned compressed-row generation for the
  representative ratio-4 layer. The V100 pod now validates mutable raw KV,
  attention compressor recurrence, emitted attention compressed rows, ratio-4
  indexer recurrence, emitted indexer rows, forced top-k visibility, indexed
  mixed attention, the existing HC layer entrypoint, and the full appliance
  gate. The next critical gap is the full 43-layer single-slot scheduler that
  produces a real selected token.
- Sprint 022 shipped the first resident multi-layer scheduler surface. The
  executor now supports both hash and bias router layers, the V100 gate
  validates a real ratio-128 bias-router layer, and the stage scheduler uploads
  the complete gpu0 shard and executes layers 0-5 from a token embedding seed.
  The next critical gap is cross-GPU HC relay through stages 1-7 and the final
  output-head selected-token gate.
- Sprint 023 shipped the first cross-GPU scheduler handoff. The V100 pod now
  executes layers 0-5 on gpu0, peer-copies HC, and executes layers 6-11 on
  gpu1 with resident arenas. It also fixed CUDA model-range caching so cached
  control tensors are device-local instead of being reused across GPUs. The
  next critical gap is extending the stage chain through gpu7, then attaching
  output-head selected-token validation.
- Sprint 024 shipped the full 8-stage scheduler chain. The V100 pod now
  executes all 43 layers across gpu0-gpu7 with resident arenas and peer HC
  handoffs, producing a finite nonzero final HC state on gpu7. The full gate no
  longer lists `full_43_layer_scheduler`; the next critical gap is collapsing
  final HC through the output head and comparing a selected token against the
  source oracle.
- Sprint 025 extended the scheduler with gpu7 output-head selected-token
  execution. The V100 pod can now replay the short official prompt through all
  43 layers, run HC-head collapse, output norm, BF16 output projection, and
  select top-1. The explicit oracle check fails today: expected token bytes
  `3136`, got `0a0a` at token id 271. The next critical gap is localizing the
  numerical divergence across the 43-layer body.
- Sprint 026 localized the first selected-token failure away from the
  output-head adapter. A deterministic HC parity smoke matches CPU and V100
  output-head top-5 on gpu7, while the prompt replay top-k remains dominated
  by punctuation/newline-like tokens. The next critical gap is finding the
  first divergent layer or stage in the 43-layer scheduler body.
- Sprint 027 shipped the selected-token correctness fix and checkpoint
  diagnostics. The V100 scheduler now matches the official short-prompt
  expected token bytes `3136`; checkpoint replay proves the seed, early layers,
  and layer-4 after-attention match the CPU source-layout oracle, while
  layer-4 final HC still shows FFN numeric drift. The current readiness
  blockers are now public serving, MTP, and throughput benchmarking.
- Sprint 028 extracted the selected-token path into a reusable one-shot V100
  replay runtime and `tools/ds4-v100-replay`. The tool loads all eight resident
  stages, replays prompt tokens, generates greedy continuations, verifies token
  bytes, and emits timing/memory JSON. Throughput/timing evidence now exists;
  the remaining readiness blockers are public serving and MTP.
- Sprint 029 shipped the first resident HTTP appliance surface. The replay
  runtime can now reset all eight stage schedulers between independent one-slot
  loopback requests, `tools/ds4-v100-replay --serve` returns the expected token
  bytes `3136`, and the full gate now reports readiness with only `mtp`
  missing.
- Sprint 030 shipped MTP sidecar readiness. The appliance gate now validates
  the real `deepseek4_mtp_support` companion GGUF, reports its exact
  F32/Q8_0/Q4_K tensor contract, keeps selected-token and HTTP readiness green,
  and narrows the remaining blocker from generic `mtp` to `mtp_runtime`.
- Sprint 031 shipped the first MTP runtime bridge. The appliance now exposes a
  typed MTP tensor inventory, uploads all 32 sidecar tensors into a compact
  gpu7 device arena, spot-checks resident bytes, and narrows readiness from
  `missing=mtp_runtime` to `missing=mtp_forward`.
- Sprint 032 shipped the Level 2 base appliance usability gate. The one-slot
  non-MTP loopback service now exposes health/status, proves two sequential
  two-token HTTP requests from one resident process, documents operator limits,
  and leaves overall readiness blocked only on `missing=mtp_forward`.
- Sprint 033 shipped the first resident MTP compute primitive. The gpu7 MTP
  sidecar arena now feeds Q8_0 projection matmuls directly, and the gate proves
  `mtp.0.e_proj.weight` plus `mtp.0.h_proj.weight` parity against the existing
  Q8_0 CUDA path with `max_abs=0`; readiness remains blocked on full
  `missing=mtp_forward`.
- Sprint 034 shipped resident MTP prefix composition. The gpu7 sidecar arena
  now feeds F32 `enorm`/`hnorm` weights and Q8_0 `e_proj`/`h_proj` weights into
  the native prefix chain, producing `mtp_input_hc` with F32 norms matching CPU
  within `4.5e-08` and the full CPU-reference prefix chain within the explicit
  Q8 accumulation tolerance. Readiness remains blocked on full
  `missing=mtp_forward`.
- Sprint 035 shipped resident MTP Q4_K routed expert execution. The gpu7
  sidecar arena now feeds `mtp.0.ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps` directly into the V100 Q4_K decode kernels, matching the
  focused CPU reference with `max_abs=1.43051147e-06`. The full V100 gate now
  includes `mtp_q4k` and passes with `failures=0 ready=false
  missing=mtp_forward`.
- Sprint 036 shipped the resident MTP FFN slice. The gpu7 sidecar arena now
  feeds HC FFN control, FFN RMS norm, bias-router selection, Q4_K routed
  experts, Q8_0 shared experts, routed+shared accumulation, and HC expansion
  to `next_hc`, matching a CPU sidecar-byte reference with `next_hc
  max_abs=2.38418579e-06`. The full V100 gate now includes `mtp_ffn` and still
  honestly reports `missing=mtp_forward`.
- Sprint 037 shipped resident MTP raw/SWA attention. The gpu7 sidecar arena now
  feeds `mtp.0.attn_sinks.weight` directly into an arena-backed attention
  decoder, the focused V100 smoke validates production FP8-plus-F16 raw KV
  store and 128-row ring-cache wrap semantics with `global_max_abs=1.27183739e-08`,
  and the full V100 gate now includes `mtp_attn` while still honestly reporting
  `missing=mtp_forward`.
- Sprint 038 shipped the resident integrated MTP attention slice. The gpu7
  sidecar arena now composes HC attention control, attention norm, Q/KV
  projections and norms, production raw KV store, sink-aware attention, grouped
  Q8_0 attention output, and HC expansion from real sidecar bytes. The focused
  smoke validates the integrated slice against a CPU sidecar-byte reference
  with `q_heads max_abs=2.14576721e-06`, `kv_row max_abs=0.000867605209`,
  `heads max_abs=2.14576721e-06`, `attn_out max_abs=0.258209229`, and
  `next_hc max_abs=0.19461441`. The full V100 gate still passes with
  `missing=mtp_forward`.
- Sprint 039 shipped resident MTP logits/top-k parity. The gpu7 sidecar arena
  now runs MTP-specific HC-head collapse from `mtp.0.hc_head_*`, applies
  `mtp.0.norm.weight`, projects through the base BF16 `output.weight`, and
  selects top-k draft candidates. The focused V100 smoke matches CPU top-5
  tokens exactly with `top1=65615` and `max_abs=9.53674316e-07`; the full gate
  now includes `mtp_logits PASS` and still correctly reports
  `missing=mtp_forward`.
- Sprint 040 shipped resident one-token MTP forward composition. The gpu7
  sidecar arena now composes deterministic prefix, integrated attention, FFN,
  MTP output HC collapse, output norm, base BF16 vocabulary projection, and
  top-k in one continuous CUDA path. The focused V100 smoke matches CPU/GPU
  top-5 tokens exactly with `top1=101365`, `boundary_max_abs=0.959003448`, and
  `logit_max_abs=0.0884904861`; the full gate now includes `mtp_forward PASS`
  and correctly advances readiness to `missing=mtp_verify`.
- Sprint 041 shipped the rollback/state-safety primitive behind the
  `mtp_rollback` gate. The target scheduler can snapshot and restore mutable
  HC, raw KV, compressed KV/state, indexer KV/state/top-k, and counters after
  real prompt replay. The focused V100 snapshot smoke captures
  `30064724` bytes after eight decode positions, proves HC mutation
  (`hc_mutate_delta=68.5005646`), restores exactly, and replays
  deterministically. The MTP rollback smoke keeps the real 3.807600108 GB
  MTP sidecar resident on gpu7, captures `30107648` bytes of target state,
  rejects token `16` against target top-1 `1`, restores target and MTP raw
  visibility exactly, and the full gate passes with `missing=mtp_verify`.
- Sprint 042 shipped native prompt-token MTP verify. The real MTP sidecar now
  drafts from the committed target token embedding and gpu7 post-commit target
  HC; on the short fixture it produces `mtp_top1=1`, matching target top-1
  `1` exactly after committed token `926` at position `18`. Forced reject
  rollback still restores exactly, and the full gate now passes with
  `missing=production_deployment`.
- Sprint 043 shipped the production deployment package. The appliance now has
  an operator launcher, env config, systemd and Kubernetes templates,
  `/metrics`, richer status limits, a production deployment smoke, and runbook
  coverage. The full 8-GPU gate passes with `production_deployment PASS` and
  reports `production_deployment PASS`.
- Sprint 044 shipped the first throughput optimization: parallel stage
  open/upload for the replay runtime, with a serial fallback and benchmark
  gate. Focused cluster timing improved cold open from `343989.990 ms` to
  `63032.135 ms` (`5.457375x`), the full gate's optimization benchmark
  improved from `227418.984 ms` to `59449.323 ms` (`3.825426x`), and the full
  gate now reports `missing=mtp_speculative_serving`.
- Sprint 045 shipped production MTP verify serving. The resident HTTP appliance
  now supports explicit `DS4_V100_MTP_SERVING=verify`, reports
  `mode=mtp_verify_one_slot`, returns MTP draft/verify diagnostics, exposes MTP
  counters in `/metrics`, and passes the full gate with
  `missing=aggregate_slot_context_envelope`.
- Sprint 046 shipped the slot/context admission contract and envelope gate.
  Planner-driven 1/2/4/8 slot tiers, context admission rejection behavior, and
  status/metrics limit reporting are now wired into the full gate as
  `slot_context_admission`.
- Sprint 047 shipped the active-microbatch scheduler core primitives.
  `ds4_v100_stage_scheduler` now owns per-slot KV/HC state and exposes
  `decode_token_batch`, `decode_hc_batch`, and `handoff_batch`, with
  multi-slot scheduler smokes and gate rung `active_microbatch_scheduler`.
- Sprint 048 shipped request-loop active microbatch integration for the base
  appliance path. `tools/ds4-v100-replay --serve` now batches pending
  non-MTP one-token requests through the scheduler batch APIs while preserving
  explicit queue/admission behavior. The remaining gap is cluster throughput
  and latency evidence across slot/context tiers.
- Sprint 049 shipped the first cluster-backed aggregate throughput evidence
  harness. The runtime now has a dedicated concurrent load script and recorded
  p50/p95/p99 plus aggregate tok/s evidence on `gpu-01` for
  1/2/4/8-slot coverage at 256K, 1/8-slot extremes at 1M, and a focused
  MTP on/off comparison.
- Sprint 050 closed the remaining proof gap and hardened the gate path. The
  full 8-GPU gate now runs with `failures=0` and reports
  `gate readiness READY` / `ready=true`.
- Sprint 051 added explicit aggregate matrix profiles to the gate execution
  path (`fast` and `full`) plus per-axis CLI overrides, then executed the full
  32-case matrix on `gpu-01` with all requests/token checks passing and
  artifacts captured under `logs/from-cluster/sprint051-full-profile`.
- Practical-use performance is not yet optimized. The current full-profile
  aggregate gate intentionally uses a one-token request shape and reports only
  generated tokens over full request latency; it measured `0.320543-0.382304`
  aggregate tok/s. The focused continuation decode benchmark measured
  `6.912252` continuation tok/s at `ctx=1048576`, `tokens=2`. Low observed GPU
  utilization is consistent with the current runtime shape: first-token
  batching only, `tensor_batched_slots=false`, per-request reset/prompt replay
  in the served path, no true MTP draft commit, and no persistent grouped
  expert hot path.
- Sprint 052 replaced the one-token aggregate gate as the practical performance
  reference with sustained multi-token decode evidence. The first cluster
  baseline at `ctx=1048576`, `slots=1`, `tokens=16`, `requests=4` measured
  `3.304551` aggregate generated tok/s, `3.098017` aggregate continuation
  tok/s, `6.869750` average per-response continuation tok/s, `10.804%`
  average GPU utilization, and `22.000%` max GPU utilization. This confirms
  that the next blocker is not measurement shape anymore; it is keeping active
  multi-token work resident and batched.
- Sprint 053 shipped same-length token-step microbatching for non-MTP HTTP
  batches and added explicit tensor-batch counters/status snapshots to the
  sustained benchmark. The cluster comparison at `ctx=1048576`, `tokens=16`,
  and `requests=4` measured `3.291466` generated tok/s for one slot and
  `3.371659` generated tok/s for two slots. The two-slot run proved the batch
  path executed (`tensor_batched_groups=1`, `tensor_batched_requests=2`,
  `tensor_batched_tokens=32`), but GPU utilization stayed low at `11.133%`
  average / `22.000%` max. The practical blocker has moved from request-loop
  wiring to hot-path kernel occupancy, routed expert batching, and persistent
  scheduling.
- Sprint 054 shipped the first real hot-path source-MXFP4 fusion in the
  scheduler: routed expert gate+up+SwiGLU now runs as one CUDA primitive per
  selected route. V100 correctness still selects token hex `3136`, and the
  focused MXFP4 smoke compares the fused primitive against the old separate
  gate/up/SwiGLU path. The sustained 1M comparison improved one-slot generated
  tok/s from `3.291466` to `3.384749` and two-slot generated tok/s from
  `3.371659` to `3.486851`, but average GPU utilization remains about `11%`.
  This validates the fusion direction while showing that larger route/down
  grouping is still required.
- Sprint 055 shipped the second routed-MXFP4 fusion: down projection and route
  accumulation now run as one CUDA primitive. Correctness still selects token
  hex `3136`, and the focused MXFP4 smoke compares fused down+add against the
  previous separate sequence. Sustained 1M generated tok/s improved from
  `3.384749` to `3.410425` at one slot and from `3.486851` to `3.503283` at
  two slots. The gain is now below 1%, so further one-route launch cleanup is
  not enough; the next jump needs grouped selected-route execution or a real
  layer-executor batch path.
- Sprint 056 shipped grouped selected-route MXFP4 execution in the main FFN
  path. The executor now processes all six routed experts through a grouped
  gate/up/SwiGLU kernel plus grouped down-sum kernel, while preserving source
  MXFP4 residency and selected-token hex `3136`. Sustained 1M generated tok/s
  improved from `3.410425` to `3.552642` at one slot and from `3.503283` to
  `3.676873` at two slots. Average GPU utilization remains about `11%`, and
  the two-slot benchmark still reported `tensor_batched_groups=0`, so the next
  practical-use blocker is deterministic token-step coalescing and true
  batched layer execution across active slots.
- Sprint 057 shipped deterministic server-side token-step coalescing. The
  default two-slot sustained benchmark now reports `tensor_batched_groups=2`,
  `tensor_batched_requests=4`, and `tensor_batched_tokens=64` while preserving
  token hex `3136`. Generated tok/s is essentially unchanged
  (`3.662490` at two slots), confirming that request coalescing is now honest
  but not sufficient. An opt-in batched FFN layer slice was implemented behind
  `DS4_V100_BATCH_LAYER_FFN`, but it regressed two-slot generated tok/s to
  `3.630558`, so it remains disabled by default.
- Sprint 058 removed replay-only router selected-expert/weight readbacks from
  the hot path while preserving direct diagnostic defaults. Correctness still
  selects token hex `3136`, and sustained 1M generated tok/s improved to
  `3.583987` at one slot and `3.704572` at two slots. This confirms the
  readback was real synchronization overhead, but the small gain and continued
  `~11%` average GPU utilization keep the main blocker on copy-free or
  persistent MoE/layer batching.
- Sprint 059 added scheduler-owned persistent layer batch scratch and enabled
  the multi-slot layer batch path by default after V100 evidence showed it
  faster. Correctness still selects token hex `3136`, and sustained 1M two-slot
  generated tok/s improved to `3.862932` with `3.621499` continuation tok/s.
  The remaining hot-path gap is the per-slot FFN input copy and low-occupancy
  routed MXFP4 execution; average GPU utilization is still about `11%`.
- Sprint 060 removed the routed FFN per-slot input copy by adding a
  pointer-input grouped MXFP4 batch primitive. Correctness still selects token
  hex `3136`, and sustained 1M two-slot generated tok/s improved to `3.915266`
  with `3.670562` continuation tok/s. GPU utilization remains low at about
  `12%`, so the next blocker is shared expert batching, remaining CPU/view
  overhead, or higher-slot scaling rather than routed input staging.
- Sprint 061 shipped the shared F8 batch primitive as an opt-in path and
  removed remaining per-layer routed-output view allocation from the default
  batch path. The shared batch path is correct but not default-fast on V100:
  the best opt-in two-slot 1M run measured `3.884237` generated tok/s, below
  Sprint 060's `3.915266`. The 4-slot 256K run measured `3.834046`, proving
  that simply adding active slots under the current layer-synchronous schedule
  does not raise aggregate throughput.
- Sprint 062 shipped an explicit decode profiling switch for the replay server
  and benchmark harness, then captured profiled sustained evidence on the V100
  pod. The profiled matrix measured `3.767204` generated tok/s at 1M/2 slots,
  `3.732457` at 1M/4 slots, `3.781844` at 256K/2 slots, and `3.747405` at
  256K/4 slots. Stage-profile totals nearly equal stage-decode totals, and
  four-slot cases roughly double serialized stage time without improving
  aggregate tok/s. The next practical-use sprint should prove opt-in
  stage-wavefront scheduling before more MTP commit or kernel rewrite work.
- Sprint 063 shipped the scheduler mechanics needed for stage wavefronting:
  slot-addressable decode-token, decode-HC, handoff, and HC-read entrypoints,
  plus per-device CUDA temp scratch. The new two-stage V100 smoke advances two
  independent slot lanes in wavefront order and matches the serial reference
  exactly with `max_abs_slot0=0` and `max_abs_slot1=0`. The next sprint should
  wire this into the served same-length non-MTP batch path behind an opt-in
  flag and compare against the Sprint 062 `~3.7-3.8` tok/s baseline.
- Sprint 064 shipped the opt-in served wavefront decode path and benchmark
  plumbing. Correctness passed through the HTTP sustained benchmark with token
  hex `3136`, but the paired V100 serial control was faster in every tested
  case: wavefront measured `3.703159` generated tok/s at 1M/2 slots versus
  serial `3.855080`, and `3.694816` at 256K/4 slots versus serial `3.839727`.
  The single-threaded diagonal schedule should stay diagnostic only; the next
  practical-use sprint needs true asynchronous stage workers, MTP commit, or a
  persistent low-bit kernel change rather than this wavefront ordering alone.
- Sprint 065 shipped the first opt-in true async stage pipeline. Same-length
  non-MTP batches now use one host worker per V100 stage, so different GPUs can
  overlap across active slots. The paired V100 matrix improved from serial
  `3.852906` to async `5.571149` generated tok/s at 1M/2 slots, and from
  serial `3.813005` to async `8.668248` at 1M/4 slots. The path remains
  opt-in because workers are still created per token-step batch; the next
  practical-use sprint should make stage workers persistent and then retest
  default-readiness.
- Sprint 066 converted the opt-in async path to replay-owned persistent stage
  workers and preserved correctness, but it did not improve over Sprint 065.
  The paired V100 matrix measured persistent async at `5.132695` generated
  tok/s for 1M/2 slots and `7.942345` for 1M/4 slots, versus same-build serial
  `3.851964` and `3.788708`. Because those numbers are `7-15%` below Sprint
  065's per-step worker path, async remains opt-in and the next practical-use
  sprint should profile dispatch/handoff synchronization before defaulting it.
- Sprint 067 added async pipeline timing counters and a same-binary A/B mode.
  The result confirms that per-step async is the preferred opt-in path:
  `5.576155` generated tok/s at 1M/2 slots and `8.617368` at 1M/4 slots,
  compared with persistent async `5.106227` and `7.909776` and same-build
  serial `3.853443` and `3.822580`. The bare `--async-pipeline-decode` flag now
  selects per-step; persistent remains available as `--async-pipeline-mode
  persistent`.
- Sprint 068 wired that preferred async path into the operator-facing appliance
  launcher. `DS4_V100_ASYNC_PIPELINE_MODE=auto` now resolves to `per-step` for
  multi-slot configs and to `off` for one-slot latency configs. The deployment
  examples now use a practical 4-slot sequential profile, and the V100 launcher
  smoke proved `/v100/status` reports `async_pipeline_mode="per-step"` while
  generation still returns token hex `3136`.
- Sprint 069 added a reusable appliance soak harness and ran the practical
  4-slot, 1M-context launcher profile. The launched appliance returned `4/4`
  token matches with `async_pipeline_mode="per-step"`, `7.518610` aggregate
  generated tok/s, and `7.048697` continuation tok/s. The deployment path is now
  repeatably validated; the next throughput gain must come from MTP draft commit
  or lower-overhead inter-stage handoff rather than more launcher plumbing.
- Sprint 070 made the MTP forward runtime persistent and reusable. The focused
  V100 MTP serving smoke accepted `3/3` drafts with `forward_run_count=1..3`,
  `scratch_device_bytes=1848592`, `scratch_host_bytes=517120`, and draft timing
  of `4.800`, `4.560`, and `4.562 ms`. Because that is effectively flat versus
  the Sprint 045 `~4.6 ms` baseline, Sprint 071 should implement true one-slot
  MTP commit rather than continue allocation cleanup.
- Sprint 071 shipped exact-verified one-slot MTP commit serving. The V100
  commit smoke accepted `2/2` drafts, reported `mode="mtp_commit_one_slot"`,
  `mtp.serving_mode="commit"`, and `mtp.committed=2`, and matched the
  verify-mode target baseline token sequence `[926, 1]`. This proves safe
  state mutation for accepted MTP drafts, but it is not yet a throughput win
  because target verification still runs.
- Sprint 072 measured the exact-commit throughput gate on V100. On the same
  1M-context, one-slot, two-token fixture, `off` measured `0.788607` generated
  tok/s, `verify` measured `0.774126`, and `commit` measured `0.777308` while
  committing `4/4` accepted drafts. Exact commit is correct and observable, but
  it is not throughput-positive because target verification still computes the
  verifier token; the next practical lever should return to stage/kernel
  throughput before recursive or skip-verify MTP.
- Sprint 073 shipped an opt-in persistent mailbox async mode. It is correct and
  improves old persistent 1M/4-slot throughput from `7.865004` to `8.053284`
  generated tok/s, but remains slower than per-step at `8.649395` generated
  tok/s. Appliance `auto` should stay on `per-step`; the next performance
  lever should move below pthread scheduling to CUDA event/stream handoff,
  peer-copy overlap, or kernel-side execution work.
- Sprint 074 shipped an opt-in async HC handoff path. It improved per-step
  1M/4-slot throughput from `8.605744` to `8.738546` generated tok/s
  (`+1.543%`) while preserving token correctness, but stayed below the `3%`
  threshold for changing the appliance default. Keep async handoff opt-in; the
  next lever should be explicit CUDA stream/event handoff or kernel-side work.
- Sprint 075 tested a device-resident output-head top-1 path. The CUDA top-1
  primitive and persistent gpu7 scratch are correct, but the one-thread device
  reducer regressed output-head timing from `346.461 ms` to `423.818 ms`.
  Generated tok/s moved only from `8.659254` to `8.697510`, so the path stays
  opt-in behind `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1` and the host-logit
  output-head path remains default.
- Sprint 076 replaced that serial scan with a deterministic parallel F32
  top-1 reducer. On the same 1M/4-slot per-step fixture, generated tok/s
  improved from `8.656498` to `9.031197` and output-head timing dropped from
  `324.953 ms` to `134.510 ms`, so greedy `k == 1` output selection now uses
  the device top-1 path by default. `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1`
  remains the rollback path.
- Sprint 077 tested batching output-head selection across active slots. The
  primitive is correct and opt-in, but the paired V100 run regressed generated
  tok/s from `9.028544` to `8.616841` and output-head timing from `135.080 ms`
  to `139.750 ms`, so practical serving keeps the Sprint 076 per-slot device
  top-1 default. `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1` is now an experiment
  switch, not a production default.
- Sprint 078 tested CUDA event-ordered stage handoff for the per-step async
  pipeline. It is correct and opt-in, removes the explicit per-stage device-sync
  bucket, and reduces handoff timing from `248.432 ms` to `193.909 ms`, but
  generated tok/s only moves from `9.147418` to `9.158602`. The next useful
  sprint should pivot to kernel-side work, especially routed MXFP4 occupancy.
- Sprint 079 tested row-pair routed MXFP4 gate/up/SwiGLU and down-sum kernels.
  The path is correct and opt-in, but paired V100 evidence regressed generated
  tok/s from `9.055694` to `9.035946` at 1M/4 slots. Halving CTA row count is
  not enough; the next kernel sprint needs route/expert tiling, packed low-bit
  dot products, or a persistent grouped expert shape.
- Sprint 080 changed the kernel policy from "use prior deepseek work as design
  evidence" to "copy candidate kernel source into this repo and prove it here."
  The copied tc-grid V100 INT8 `v13_rf_v6` path builds and runs from `ds4`,
  reaching `46.391 TFLOP/s` on `M=2048,N=7168,K=7168`, but only `7.223
  TFLOP/s` on `M=128,N=2048,K=4096`. Keep it as a proof/benchmark path and
  prioritize copied TurboMind MXFP4 grouped GEMM for production routed experts.
- Sprint 081 copied the TurboMind C ABI wrapper and required lmdeploy
  `turbomind` source into `ds4`. The copied build produces
  `libggml-turbomind.so` on V100 and grouped MXFP4 compare passes on DS4
  gate/up and down shapes. This is now the preferred routed-expert adapter
  target because it keeps source MXFP4 rather than expanding experts to INT8.
- Sprint 082 shipped the first DS4 routed-expert adapter smoke for copied
  TurboMind. Source MXFP4 expert bytes are packed through the TurboMind C ABI,
  selected route rows are grouped by expert, grouped gate/up/down GEMMs run on
  V100, DS4 SwiGLU and route weights are applied, and the final routed output
  matches the existing source-MXFP4 arena reference with `max_abs=0.00129318`
  and `rel=0.000258549`. This makes opt-in runtime integration the next
  kernel-side step.
- Sprint 083 shipped that opt-in runtime integration. Setting
  `DS4_V100_TURBOMIND_ROUTED_FFN=1` routes DS4's MXFP4 FFN wrapper through
  copied TurboMind with device-built expert grouping and strict/fallback modes.
  The V100 smoke matches the source-MXFP4 arena reference with
  `max_abs=0.00129318`, `rel=0.000258549`, and `host_ms=43.298` for the
  bounded eight-expert fixture. It stays off by default because it transiently
  repacks expert weights during the call; the production path needs offline
  TurboMind expert packs or a memory-planner-admitted cache.
- Sprint 084 shipped the first offline TurboMind expert sidecar packer.
  `tools/ds4-v100-turbomind-pack` reads the existing V100 pack index and real
  DS4 Flash GGUF, packs MXFP4 gate/up/down experts through copied TurboMind,
  and writes `gpuN.turbomind` plus `turbomind-pack-index.tsv`. The V100
  bounded validation packed layer 0 gate/up/down with two experts each,
  recording `k_pack=0x341321` and a `26,738,688` byte sidecar. Runtime loading
  is now validated in a bounded smoke.
- Sprint 085 shipped the first persistent TurboMind sidecar loader smoke.
  `ds4_turbomind_pack.{h,c}` parses the derived sidecar index, and
  `tests/cuda_v100_turbomind_sidecar_smoke` uploads `gpuN.turbomind` once,
  rebuilds TurboMind `StridedPtrH` tables from sidecar offsets, and runs
  gate/up/down from persistent packed buffers. The V100 smoke matches the
  source-MXFP4 arena reference with `max_abs=5.91128e-07`, `rel=0.000493098`,
  and `host_ms=0.265` for the bounded layer-0/two-expert fixture.
- Sprint 086 added explicit VRAM admission for TurboMind sidecars.
  `tools/ds4-v100-turbomind-admit` reports source arena bytes, source expert
  payload bytes, sidecar bytes, duplicate totals, and replacement-style totals
  per GPU. On the bounded sidecar, duplicate residency fits; the report also
  shows GPU 0 already carries `19.125 GiB` of source expert payload, so full
  production sidecars should replace source expert residency or be admitted as
  a bounded cache instead of silently duplicating all experts.
- Sprint 087 moved TurboMind experts into the single appliance pack shape.
  `tools/ds4-v100-appliance-pack` writes TurboMind-packed routed experts into
  `gpuN.weights` and describes them with `turbomind-pack-index.tsv`; non-expert
  tensors remain in `pack-index.tsv`. The new
  `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32` CUDA API runs
  directly from prepacked resident spans without transient repacking. V100
  bounded validation against the appliance-shaped `gpu0.weights` reports
  `packed_api max_abs=5.91128e-07`, `rel=0.000493098`, and `PASS`.
- Sprints 088-090 promoted that format into the scheduler and then into a full
  8-GPU appliance. The V100 cluster now has a production-shaped appliance
  directory on k8s-local storage at `/workspace/ds4-appliance-full-tm-s090`.
  It contains `gpu0.weights` through `gpu7.weights`, replaces routed expert
  residency with TurboMind-packed experts inside those shard files, runs all
  43 layers with `tm_layers=43`, and preserves the official first token
  `3136` in replay.
- Sprint 091 wired that appliance directory into the operator-facing launcher
  and HTTP smoke path. `DS4_V100_APPLIANCE_DIR` now validates all shard files
  and runs replay with `--appliance-dir`; the served smoke returned first token
  `3136` while uploading only 8 appliance shard tensors.
- Sprint 092 ran the first warm-started, 4-slot async soak from the full
  appliance directory. The timed batch produced `64` tokens across 4 requests
  with `token_match=4/4`, `tensor_batched_groups=1`,
  `tensor_batched_tokens=64`, `11.256048` generated tok/s, and `10.552545`
  continuation tok/s. A cold concurrent first-request attempt failed in the
  TurboMind routed FFN path while CUDA tensor caches were being lazily loaded,
  so production launch must warm the appliance before admitting concurrent
  traffic.
- Sprint 093 moved that warmup into the appliance server/launcher. With
  `DS4_V100_STARTUP_WARMUP=auto`, 4-slot serving resolves to
  `--startup-warmup`, runs one internal generation before listening, exposes
  `startup_warmup=true` in status/metrics, and passes the same 4-request soak
  with `warmup_requests=0`. The measured rate stayed flat at `11.241074`
  generated tok/s, so the operational race is fixed but decode throughput
  still needs kernel/control-path work. The first decode-window profiler pass
  shows F8 dense/projection matmul (`42.32%`), residual HtoD control copies
  (`31.68%`), and TurboMind MXFP4 GEMM (`13.90%`) as the largest GPU buckets.
- Sprint 094 made two hot-path changes in the production appliance path:
  TurboMind packed expert pointer tables are now cached per resident arena, and
  the multi-slot FFN executor sends all active slots through one TurboMind
  routed expert call per layer instead of one call per slot. The launcher now
  enables shared F8 FFN batching by default for multi-slot appliance serving.
  The 1M/4-slot no-client-warmup soak remains correct and improves to
  `12.634955` generated tok/s and `11.845270` continuation tok/s.
- Sprints 095-100 moved the appliance from low-teens tok/s to the current
  production-default baseline. Request rendezvous made 8-slot 256K serving
  deterministic, the CUDA tensor pool removed request-window allocator churn,
  grouped F8 attention output cut F8 launch count, and TurboMind route
  validation readback is now debug-only. The current best production-default
  evidence is the 8-slot/256K soak at about `26.4` generated tok/s with
  `token_match=8/8`.
- Sprint 101 repaired the opt-in batch attention projection semantics so it
  matches the single-slot path's attention RMS norm and compressed-KV prep
  ordering. V100 evidence says it should remain opt-in: 8-slot/256K was flat
  (`26.432087` opt-in vs `26.402101` default generated tok/s) and 4-slot/1M
  regressed (`17.503345` opt-in vs `18.102742` default).
- Sprint 102 shipped a broader F8 arena matmul kernel-shape change. Row-pair
  F8 kernels compute two output rows per CTA across the single, batch,
  pointer-table, and grouped arena F8 APIs. `DS4_V100_CUDA_F8_ROWPAIR=1` is now
  the launcher default after V100 A/B improved 8-slot/256K serving to
  `27.049799` generated tok/s and 4-slot/1M serving to `18.500281` generated
  tok/s with token-match correctness.
- Sprint 103 removed the per-weight `ldexpf()` path from E4M3 F8 decode by
  constructing exact F32 bit patterns directly. The production appliance now
  measures `30.862791` generated tok/s at 256K/8 slots and `19.733742`
  generated tok/s at 1M/4 slots, with token-match correctness preserved.
- Sprint 104 replaced the hot F8 arena shared-memory reduction trees with
  warp-shuffle block reductions. The production appliance now measures
  `31.383579` and repeat `31.451185` generated tok/s at 256K/8 slots, and
  `20.026385` generated tok/s at 1M/4 slots, with token-match correctness
  preserved. A served-path F8-to-F16 cache/cuBLAS experiment was rejected
  because it was too slow to satisfy the practical-use path.
- Sprint 105 tested extending the same warp-reduction pattern to BF16/F32
  output/control arena matmuls. Correctness passed and one 8-slot run reached
  `31.612471` generated tok/s, but the repeat fell back to `31.479378`, too
  close to the Sprint 104 band to justify shipping the reduction-order change.
  The code was reverted and Sprint 104 remains the committed baseline.
- `docs/architecture/DS4-V100-LAYOUT.md` is the architecture anchor for
  sharding, memory layout, kernel selection, tensor-parallel alternatives, and
  context/slot assumptions. Sprint plans should reference it instead of
  re-deriving the topology.

## Readiness Ladder

The gate's `ready=false` status is intentionally conservative. It should not
mean "nothing works"; it should say which rung has not been proven yet.

| Level | Name | Meaning | Required Evidence | Current Status |
|---|---|---|---|---|
| 0 | Fit and residency | The source model can be mapped, packed, and held in 8x V100 VRAM with reserve. | Pack inventory, per-GPU memory plan, resident upload smoke, source-layout guards. | Complete through Sprints 001-006. |
| 1 | Single-prompt base correctness | The base model path can run one known prompt through all 43 layers and select the expected first token. | Full 8-stage scheduler, output-head parity, selected-token hex `3136`, one-shot replay. | Complete through Sprints 024-028. |
| 2 | Minimal usable base appliance | A human/operator can start the base model service and use it for short non-MTP one-slot generation with documented limits. | Longer prompt/decode smoke, repeat HTTP requests, failure logs, run command, health check, 1-slot timing report. | Complete through Sprint 032, with explicit limits: one slot, sequential loopback HTTP, no MTP, no streaming, no production supervisor. |
| 3 | MTP-assisted correctness | The resident MTP sidecar can produce a K=1 draft token that matches a trusted oracle and does not corrupt target-model state. | MTP sidecar residency, Q8_0/Q4_K kernel parity, MTP forward logits/top-k, draft/verify/rollback tests, served verify diagnostics. | Complete through Sprint 045. The focused and full 8-GPU gates produce a native prompt-token draft from committed token `926` and post-commit target HC, match target top-1 token `1` exactly, expose the same verify path through HTTP as `mtp_verify_one_slot`, and keep rollback/off mode available. |
| 4 | Production deployment | The appliance can be left running on the cluster with operational confidence. | Supervised service, config files, restart behavior, health/metrics endpoint, deployment/runbook, known rollback path. | Complete through Sprint 043 for the base one-slot service. The full gate includes `production_deployment PASS` with launcher, config, `/metrics`, supervisor templates, runbook, and rollback mode. |
| 5 | Startup and one-slot throughput optimization | Fresh-process startup/upload is measured and improved, and the one-slot decode path has gate-owned timing evidence. | Serial-vs-parallel startup benchmark, default replay timing, first-token correctness, full-gate `throughput_optimization PASS`. | Complete through Sprint 044. Focused timing improved cold open from `343989.990 ms` to `63032.135 ms`; full-gate timing improved from `227418.984 ms` to `59449.323 ms`; first token remains `3136`. |
| 6 | Aggregate slot/context operating envelope | The appliance can admit and schedule multiple slots/context tiers and report aggregate tok/s. | 1/2/4/8-slot admission, context-tier benchmarks, active microbatch scheduling, queueing or rejection semantics, and MTP-aware throughput comparison. | Complete through Sprint 051: admission, scheduler batch primitives, request-loop batching, cluster slot/context load evidence (including queue-policy and focused MTP comparison), full gate closure with `ready=true`, and explicit fast/full aggregate gate profiles are proven. |

MTP is not required for the first minimally usable base appliance. It is
required for the intended performance path if speculative decoding proves
correct and beneficial on V100. With Sprint 045 complete, "ready to use" means
the checked base appliance can run either the default base one-slot service or
the explicit one-token MTP verify service as a loopback cluster process. It
does not yet mean multi-slot batching, true draft commit without target
recompute, streaming, or externally exposed serving.

## Practical Use Optimization

The back-of-envelope roofline for DeepSeek-V4-Flash on 8x V100 is useful only
as a ceiling: 13B active parameters imply roughly 26 GFLOP/token, and 8x V100
has about 1 PFLOP/s FP16 tensor peak. That produces a theoretical compute bound
near 38k tok/s, but it assumes perfectly batched dense tensor-core GEMMs with no
routing, no FP4/FP8 unpack/dequant, no KV traffic, no inter-GPU scheduling, and
no expert imbalance.

The practical target should be staged from current evidence, not from roofline:

| Runtime state | Expected aggregate decode range | Confidence | Qualification |
|---|---:|---|---|
| Sprint 051 one-token aggregate gate | `~0.3` tok/s | Measured | Correctness-first one-token request shape; useful for admission/correctness but not practical serving throughput. |
| Sprint 052 sustained one-slot baseline | `3.30` generated tok/s, `3.10` continuation tok/s | Measured | Multi-token requests at 1M context prove low utilization remains real: average GPU utilization `10.804%`, max `22.000%`. |
| Sprint 053 same-length token-step batching | `3.37` generated tok/s, `3.16` continuation tok/s | Measured | Two-slot serving proves the batch branch is used (`1` group / `2` requests / `32` tokens), but only improves aggregate generated tok/s by about `2.4%`; average GPU utilization remains about `11%`. |
| Sprint 054 fused MXFP4 gate/up/SwiGLU | `3.49` generated tok/s, `3.27` continuation tok/s | Measured | First hot-path fusion is correct and improves 1M sustained two-slot generated tok/s by about `3.4%` over Sprint 053, but utilization remains about `11%`. |
| Sprint 055 fused MXFP4 down+accum | `3.50` generated tok/s, `3.28` continuation tok/s | Measured | Removes another routed route launch, but improves two-slot generated tok/s by only about `0.5%` over Sprint 054; one-route cleanup is reaching diminishing returns. |
| Sprint 056 grouped selected MXFP4 routes | `3.68` generated tok/s, `3.45` continuation tok/s | Measured | Groups all selected routes in the single-slot routed FFN path and improves two-slot generated tok/s by about `5.0%` over Sprint 055, but average GPU utilization remains about `11%` and the benchmark did not coalesce token-step groups. |
| Sprint 057 deterministic token-step coalescing | `3.66` generated tok/s, `3.43` continuation tok/s | Measured | Server-side rendezvous makes the two-slot benchmark reliably enter the batch path (`2` groups / `4` requests / `64` tokens), but throughput is essentially flat; opt-in batched FFN regressed and remains disabled. |
| Sprint 058 replay router readback suppression | `3.70` generated tok/s, `3.47` continuation tok/s | Measured | Removes appliance hot-path selected-expert/route-weight CPU readbacks and improves two-slot generated tok/s by about `1.15%` over Sprint 057, but utilization remains about `11%`; the next gain needs copy-free or persistent MoE batching. |
| Sprint 059 persistent layer batch scratch | `3.86` generated tok/s, `3.62` continuation tok/s | Measured | Reuses scheduler-owned scratch across multi-slot layer batches and enables the path by default, improving two-slot generated tok/s by about `4.27%` over Sprint 058; utilization remains about `11%`. |
| Sprint 060 pointer-input routed FFN batch | `3.92` generated tok/s, `3.67` continuation tok/s | Measured | Removes the routed FFN per-slot input copy by passing per-slot input tensor pointers into the grouped MXFP4 batch kernel; two-slot generated tok/s improves another `1.35%`, but utilization remains about `12%`. |
| Sprint 061 shared F8 batch and 4-slot retest | `3.86` generated tok/s at 1M/2 slots, `3.83` at 256K/4 slots | Measured | Shared F8 batching is correct but remains opt-in because it did not beat the per-slot shared path. Persistent output views remove minor allocation churn. Four active slots do not improve aggregate tok/s, so the next gain requires a larger execution-shape change. |
| Sprint 062 decode timing matrix | `3.77` generated tok/s at 1M/2 slots, `3.75` at 256K/4 slots | Measured | Opt-in synchronized profiling confirms the stage-synchronous execution shape is the dominant practical blocker: summed stage-profile time matches summed stage-decode time, while 4 slots increase latency without raising aggregate throughput. |
| Sprint 063 wavefront lane proof | Correctness proof, not a throughput run | Measured | Slot-addressable stage scheduler APIs and per-device CUDA scratch now support two-stage wavefront lane mechanics. The V100 smoke matches serial HC exactly, so the next measurement should be an opt-in served wavefront benchmark. |
| Sprint 064 opt-in served wavefront | `3.70` generated tok/s at 1M/2 slots, `3.69` at 256K/4 slots | Measured | Served wavefront decode is correct but slower than the paired serial control (`3.86` and `3.84` generated tok/s respectively). The current single-threaded diagonal scheduler does not create useful overlap and should remain non-default. |
| Sprint 065 opt-in async stage pipeline | `5.57` generated tok/s at 1M/2 slots, `8.67` at 1M/4 slots | Measured | One host worker per stage creates real cross-GPU overlap while preserving token hex `3136`. Four slots finally scale, with avg GPU utilization rising to about `20%`; keep opt-in until workers are persistent across token steps. |
| Sprint 066 persistent async workers | `5.13` generated tok/s at 1M/2 slots, `7.94` at 1M/4 slots | Measured | Replay-owned persistent stage workers are correct and still faster than serial (`3.85` and `3.79` generated tok/s), but they are `7-15%` slower than Sprint 065's per-step worker path. Keep async opt-in and profile dispatch/handoff synchronization before defaulting. |
| Sprint 067 async A/B dispatch | `5.58` generated tok/s at 1M/2 slots, `8.62` at 1M/4 slots | Measured | Same-binary A/B confirms per-step async beats persistent by `7-9%`. Timing counters show persistent saves setup but loses more in wait-for-previous-slot accumulation, so `--async-pipeline-decode` now selects per-step. |
| Sprint 068 appliance async serving profile | 4-slot per-step async deployment path | Measured | The launcher now resolves `DS4_V100_ASYNC_PIPELINE_MODE=auto` to per-step for multi-slot practical serving. V100 loopback smoke through `ds4-v100-run-appliance.sh` reports `async_pipeline_mode=per-step` and returns token hex `3136`. |
| Sprint 069 appliance launcher soak | `7.52` generated tok/s, `7.05` continuation tok/s at 1M/4 slots | Measured | Reusable launcher soak validates health/status/metrics and concurrent generation through `ds4-v100-run-appliance.sh`, with `4/4` token matches and async timing present in responses. |
| Sprint 070 persistent MTP forward runtime | `4.56-4.80 ms` MTP draft time, `3/3` accepted drafts | Measured | MTP forward scratch is now resident and reused across requests, with scratch/run counters exposed in serving JSON. Timing stayed flat versus Sprint 045, so the next practical MTP gain requires true draft commit. |
| Sprint 071 exact MTP commit serving | `2/2` accepted and committed drafts, sequence `[926, 1]` matches verify baseline | Measured | Commit mode now mutates the generation path by emitting accepted MTP drafts after exact target verification. This closes the state-contract gap but does not yet improve throughput because verification still computes the target token. |
| Sprint 072 MTP commit throughput gate | `0.789` off, `0.774` verify, `0.777` commit generated tok/s at 1M/1 slot | Measured | Commit accepted and committed `4/4` drafts, but exact commit was `1.43%` slower than off because the target verifier token still runs. MTP remains correct; near-term throughput work should pivot back to stage/kernel scheduling. |
| Sprint 073 persistent mailbox workers | `8.05` generated tok/s at 1M/4 slots | Measured | Mailbox workers are correct and beat old persistent by `2.39%`, but stay `6.89%` slower than per-step. Keep `per-step` as the appliance `auto` default and pivot below pthread scheduling. |
| Sprint 074 async HC handoff | `8.74` generated tok/s at 1M/4 slots | Measured | Queued peer handoff is correct and improves per-step throughput by `1.54%`, but remains below the default-change threshold. Keep opt-in and pursue explicit stream/event handoff or kernel-side work. |
| Sprint 075 output-head top-1 candidate | `8.70` generated tok/s at 1M/4 slots | Measured | Device top-1 is correct but not default-worthy: generated tok/s improved only `0.44%` while output-head timing regressed `22.33%` (`423.818 ms` vs `346.461 ms`). Keep it opt-in and either build a real parallel reducer or return to larger stage/kernel costs. |
| Sprint 076 parallel output-head top-1 | `9.03` generated tok/s at 1M/4 slots | Measured | Parallel device top-1 clears the default-change gate: generated tok/s improves `4.33%`, continuation tok/s improves `4.33%`, and output-head timing drops `58.61%` (`134.510 ms` vs `324.953 ms`). Greedy `k == 1` now defaults to device top-1 with an env rollback. |
| Sprint 077 batched output-head selection | `9.01` generated tok/s default, `8.62` with batch opt-in at 1M/4 slots | Measured | Batched row projection/top-1 is correct but slower than the per-slot device top-1 control: generated tok/s regressed `4.56%` and output-head timing rose `3.46%` (`139.750 ms` vs `135.080 ms`). Keep `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1` opt-in and pivot back to stage/kernel costs. |
| Sprint 078 event-ordered stage handoff | `9.16` generated tok/s at 1M/4 slots | Measured | CUDA events removed the explicit device-sync bucket and reduced handoff sum by `21.95%`, but generated tok/s improved only `0.12%` (`9.158602` vs `9.147418`). Keep `DS4_V100_ASYNC_EVENT_HANDOFF=1` opt-in and pivot to kernel-side routed MXFP4 occupancy. |
| Sprint 079 routed MXFP4 row-pair probe | `9.06` default, `9.04` row-pair opt-in at 1M/4 slots | Measured | Row-pair MXFP4 gate/up/SwiGLU and down-sum kernels are correct, but generated tok/s regressed `0.22%` (`9.035946` vs `9.055694`). Keep `DS4_CUDA_MXFP4_ROUTE_ROWS2=1` opt-in and target a larger route/expert tiling or packed low-bit kernel rewrite next. |
| Sprint 080 copied tc-grid INT8 V100 kernel proof | `7.223 TFLOP/s` at `M=128,N=2048,K=4096`; `46.391 TFLOP/s` at `M=2048,N=7168,K=7168` | Measured | Copied tc-grid `v13_rf_v6` source now builds and runs from `ds4`. It confirms high-M V100 low-bit HMMA can work, but low-M routed decode remains underfilled and INT8 would expand source MXFP4 experts. Use this as a benchmark/proof path; prioritize TurboMind MXFP4 grouped GEMM for hot-path integration. |
| Sprint 081 copied TurboMind MXFP4 grouped GEMM proof | `0.1037-0.1454 ms` grouped DS4-shape expert GEMMs | Measured | Copied TurboMind source builds from `ds4` and grouped MXFP4 compare passes for DS4 gate/up/down shapes. Down grouped is `1.23-1.26x` faster than six single calls; gate/up grouped is roughly neutral/slower at tiny token counts. This is the preferred source-format-preserving hot-path adapter target. |
| Sprint 082 TurboMind routed expert adapter smoke | `max_abs=0.00129318`, `rel=0.000258549` versus DS4 arena reference | Measured | The adapter now packs source MXFP4 bytes through copied TurboMind, groups selected route rows by expert, runs grouped gate/up/down, applies DS4 SwiGLU and route weights, and matches the existing routed-output reference on V100. Next step is an opt-in runtime path with 256-expert packing and sustained throughput comparison. |
| Sprint 083 opt-in TurboMind runtime bridge | `max_abs=0.00129318`, `rel=0.000258549`, `host_ms=43.298` on bounded runtime-wrapper smoke | Measured | The DS4 CUDA wrapper can now route MXFP4 FFN through copied TurboMind behind `DS4_V100_TURBOMIND_ROUTED_FFN=1`. This proves runtime semantics and fallback, but transient per-call packing makes it a validation bridge rather than a throughput default. |
| Sprint 084 offline TurboMind sidecar packer | layer-0 gate/up/down, `2/256` experts each, `26,738,688` byte sidecar | Measured | The new packer reads real source GGUF bytes through the existing pack index and emits `gpuN.turbomind` plus `turbomind-pack-index.tsv`. This starts the production format path; runtime sidecar loading and full memory admission are still pending. |
| Sprint 085 persistent TurboMind sidecar smoke | `max_abs=5.91128e-07`, `rel=0.000493098`, `host_ms=0.265` | Measured | The new sidecar parser and CUDA smoke load `gpuN.turbomind` once, reconstruct TurboMind pointer tables from offsets/strides, and run grouped gate/up/down from persistent packed buffers against the source-MXFP4 arena reference. This removes the transient repack tax at the adapter boundary; full memory admission and scheduler selection remain. |
| Sprint 086 TurboMind sidecar admission | GPU0 bounded duplicate total `27.002 GiB`; source expert payload `19.125 GiB` | Measured | The admission tool proves the accounting path and shows the production constraint: bounded sidecars fit as duplicate cache, but full sidecars should be replacement/admitted artifacts because source experts already occupy most per-GPU payload. |
| Sprint 087 appliance-packed TurboMind experts | `packed_api max_abs=5.91128e-07`, `rel=0.000493098`, `PASS` | Measured | TurboMind experts now live in the appliance `gpuN.weights` file rather than a separate sidecar file, and the DS4 CUDA API can execute directly from those prepacked resident spans without repacking. Scheduler binding is the next step. |
| Sprint 089 appliance-backed scheduler smoke | stage-0 `gpu0.weights` `22,524,134,668` bytes; scheduler `tm_layers=1` | Measured | The scheduler now opens a bounded appliance directory without a source GGUF model map, uploads `gpu0.weights`, executes layers 0-5, and positively reports that one routed layer used the no-repack TurboMind appliance path. This is not a throughput result; full 8-GPU appliance generation and sustained decode benchmarking remain next. |
| Sprint 090 full appliance replay | full appliance `142G`; `tm_layers=43`; replay `0.620997` generated tok/s, `9.491896` continuation tok/s | Measured | A full 8-GPU appliance was generated on k8s-local storage, all shards fit in 32 GB V100 VRAM, full scheduler smoke executed all 43 layers from `gpuN.weights`, and replay returned first token hex `3136`. This proves the format and residency contract; launcher integration and multi-slot async benchmarking remain. |
| Sprint 091 appliance launcher smoke | served first token `3136`; `uploaded_tensors=8`; open `65.033s` | Measured | `DS4_V100_APPLIANCE_DIR` now drives the operator launcher and HTTP smoke through `--appliance-dir`, so the production service path no longer needs the source pack index for scheduler residency. Multi-slot async and MTP appliance benchmarks remain. |
| Sprint 092 appliance 4-slot async soak | `11.256048` generated tok/s, `10.552545` continuation tok/s at 1M/4 slots | Measured | The full appliance path is correct under warm-started 4-request tensor batching (`token_match=4/4`, `tensor_batched_groups=1`, `tensor_batched_tokens=64`). Cold concurrent first requests failed during lazy CUDA tensor-cache load in the TurboMind routed FFN path, so launch must include a warmup before serving traffic. Performance is still in the low baseline range, not practical-use optimized. |
| Sprint 093 server startup warmup + profile | `11.241074` generated tok/s, `10.538507` continuation tok/s at 1M/4 slots with `warmup_requests=0` | Measured | Server-side startup warmup now prevents cold concurrent first traffic from racing CUDA tensor-cache loading. Decode-window profiling shows F8 matmul `42.32%`, HtoD control copies `31.68%`, and TurboMind GEMM `13.90%`; targeted NCU reports one F8 matmul launch at `58.71%` SM throughput and `12.96%` DRAM throughput. Next work should reduce HtoD/control churn and dense F8 launch overhead before more TurboMind tuning. |
| Sprint 094 grouped TurboMind and shared F8 default | `12.634955` generated tok/s, `11.845270` continuation tok/s at 1M/4 slots with `warmup_requests=0` | Measured | Caches TurboMind per-expert device pointer tables, batches routed TurboMind experts across active slots, and defaults `DS4_V100_BATCH_SHARED_F8=1`. Correctness remains `token_match=4/4`; generated tok/s improves `12.40%` over Sprint 093. HtoD copy count in the one-shot profiler drops from `801` to `153`, but large first-generation model-cache copies and F8 projection launches remain dominant. |
| Sprint 095 request rendezvous and F8 cache probe | `12.597711` generated tok/s at 1M/4 slots; `17.052974` at 256K/8 slots | Measured | Adds production `DS4_V100_MICROBATCH_WAIT_US=auto`, resolving to a 50 ms coalescing window for multi-slot serving. This fixes the split-batch behavior that previously made 8-slot probes unreliable: `token_match=8/8` now passes at 256K. An opt-in F8-to-F16 arena cache is correct but flat (`12.614479` generated tok/s), so it stays experimental. Decode-window profiling remains led by F8 arena matmul `42.52%`, HtoD `31.48%`, and TurboMind GEMM `13.82%`. |
| Sprint 096 served decode profiler window | Served HTTP batch profile: F8 arena matmul `61.64%`, TurboMind `20.15%`, HtoD `0.14%` | Measured | Extends `--cuda-profiler-window` to the HTTP appliance path and adds launcher env `DS4_V100_CUDA_PROFILER_WINDOW=1` for diagnostic runs. This proves the warmed served path is not HtoD-bound after startup warmup; the next blocker is F8 projection/shared matmul plus allocator churn (`cudaFree`/`cudaMalloc` dominate API time). |
| Sprint 097 CUDA tensor pool default | `17.532887` generated tok/s at 1M/4 slots; `25.232220` at 256K/8 slots | Measured | Adds a bounded per-device scratch tensor pool and defaults `DS4_V100_CUDA_TENSOR_POOL=auto` on for multi-slot serving. Same-binary paired fixtures improved from `11.902776` to `16.881653` generated tok/s at 1M/4 slots and from `17.193119` to `25.212896` at 256K/8 slots. The warmed profile removes `cudaMalloc` from the request window and reduces `cudaFree` to `9.18 ms` / `37` calls; F8 arena matmul is now the clear next GPU target. |
| Sprint 098 grouped F8 attention output | `17.904697` generated tok/s at 1M/4 slots; `26.206100` at 256K/8 slots | Measured | Replaces eight attention output-A F8 matmul launches per layer/slot with one grouped launch and adds `DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=1` rollback. Same-binary controls preserve correctness and show grouped beating rollback (`16.897788` and `25.456942`). Served profile reduces single F8 matmul calls from `11880` to `5544` and total CUDA kernel launches from `39684` to `34140`. |
| Sprint 099 batch attention projection probe | `17.742637` generated tok/s opt-in vs `17.764257` rollback at 1M/4 slots | Measured | Adds an explicit `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` probe for batching Q-A, Q-B, and KV F8 projections across active slots. Correctness passes at 4-slot/1M and 8-slot/256K, but same-binary controls are flat/slightly faster without the probe (`26.149613` rollback vs `26.128571` opt-in at 8 slots). Keep it off by default and stop pursuing projection-only batching in this shape. |
| Sprint 100 TurboMind sync readback A/B | `26.372672` generated tok/s at 256K/8 slots production default | Measured | Adds an opt-in TurboMind grouped GEMM ABI that accepts host-known routed row count and makes packed route validation readback debug-only. V100 evidence showed the no-row-count-readback ABI moves wait time into `cudaDeviceSynchronize` and is slower, so production defaults to the old ABI with route validation sync off (`DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1`, `DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=0`). |
| Sprint 101 batch attention semantic repair | `26.402101` default vs `26.432087` opt-in at 256K/8 slots; `18.102742` default vs `17.503345` opt-in at 1M/4 slots | Measured | Repairs the opt-in batch projection path to use attention RMS-normalized rows and the same compressed-KV preparation input as the single-slot path. Correctness passes, but the 8-slot result is noise-level and the 1M/4-slot result regresses, so `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` remains off by default. |
| Sprint 102 F8 row-pair default | `27.049799` generated tok/s at 256K/8 slots; `18.500281` at 1M/4 slots | Measured | Computes two F8 arena output rows per CTA across the hot F8 APIs. Same-binary A/B improved 8-slot/256K from `26.447308` to `27.037514` and 4-slot/1M from `17.821073` to `18.500281`, with launcher-default validation at `27.049799`. `DS4_V100_CUDA_F8_ROWPAIR=1` is now the appliance default with rollback to `0`. |
| Sprint 103 exact-bit F8 decode | `30.862791` generated tok/s at 256K/8 slots; `19.733742` at 1M/4 slots | Measured | Replaces per-weight `ldexpf()` E4M3 decode with exact F32 bit construction. Selected-token correctness remains `3136`; production soaks preserve `8/8` and `4/4` token matches. This improves the Sprint 102 launcher default by about `14.1%` at 256K/8 slots and `6.7%` at 1M/4 slots. |
| Sprint 104 F8 warp-reduction kernels | `31.383579` and repeat `31.451185` generated tok/s at 256K/8 slots; `20.026385` at 1M/4 slots | Measured | Replaces F8 arena shared-memory tree reductions with warp-shuffle block reductions in the hot F8 matmul and shared F8 pair-SwiGLU paths. Correctness remains `3136`; production soaks preserve `8/8` and `4/4` token matches. The gain over Sprint 103 is modest but repeatable and adds no VRAM pressure. |
| Sprint 105 BF16/F32 warp-reduction probe | rejected | Measured | Correctness passed and the first 8-slot run reached `31.612471`, but the repeat was `31.479378`, effectively inside the Sprint 104 band. The code was reverted; do not retry this as a default without a stronger profile reason. |
| Sprint 106 served decode baseline profile | F8 rows2/grouped rows2 about `51%` GPU time; TurboMind about `25%` | Measured | Warmed served profiling shows the request window is kernel-shape limited, not disk, host RAM, or bulk copy limited. The next useful work must change F8 execution shape or TurboMind expert occupancy. |
| Sprint 107 DS4 grouped F8 attention output | best `31.811137` generated tok/s at 256K/8 slots | Measured | Adds a DS4-specialized grouped rows2 attention-output kernel. It improved the main 8-slot/256K target and is the previous best observed served throughput before Sprint 111. |
| Sprint 108 TurboMind small-route build fusion | `31.759013` opt-in vs `31.794180` rollback at 256K/8 slots | Measured | Small route metadata fusion is correct but too small to matter. It remains opt-in. |
| Sprint 109 F8 row4 CTA probe | `30.998275` row4 vs `31.380225` row2 control at 256K/8 slots | Measured | Four-output-row CTAs preserve correctness but lose throughput, likely from register pressure and occupancy loss. Row4 remains off by default. |
| Sprint 110 TurboMind fused gate+up probe | `1.46x-1.53x` faster than separate gate/up grouped calls | Measured | A DS4-shaped fused `N=4096` gate_up MXFP4 grouped GEMM exactly matches separate gate/up outputs and clears the production implementation gate. |
| Sprint 111 production fused TurboMind gate_up | `33.430971` fused vs `31.312694` separate control at 256K/8 slots; `21.403909` at 1M/4 slots | Measured | Adds offline fused `ffn_gate_up_exps.weight` packing and a fused CUDA routed-FFN path. Full scheduler correctness passes with `tm_layers=43`; selected-token smoke returns token id `926`, hex `3136`. This is the current best observed served throughput. |
| Sprint 112 F8 warp-scale probe | `29.009399` opt-in vs `33.484099` control at 256K/8 slots | Measured | Warp-broadcast E8M0 scale loading is correct but substantially slower in the fused appliance path. Keep it off. |
| Sprint 113 direct FFN delta accumulation | `33.360404` opt-in vs `33.589285` control at 256K/8 slots | Measured | Direct shared/routed accumulation into the final FFN delta buffer is correct but slightly slower. Keep it off. |
| Sprint 114 shared-down F8 HMMA probe | `33.550415` opt-in vs `33.397763` control at 256K/8 slots; `21.396331` vs `21.365610` at 1M/4 slots | Measured | A DS4-shaped Volta WMMA kernel for `tokens x 2048` by `4096 x 2048` shared-down F8 is correct and marginally faster in-run, but the delta is too small and below the default best. Keep it opt-in and use the result to justify larger fused F8 FFN work. |
| Sprint 115 shared gate/up SwiGLU F8 HMMA | `33.578236` default vs `33.292541` control at 256K/8 slots; `21.455638` vs `21.430420` at 1M/4 slots | Measured | A DS4-shaped Volta WMMA kernel computes shared gate and up projections from the same activation tile, then applies SwiGLU. This path is now the default. Combined with Sprint 114 shared-down HMMA, it reaches `33.674684` at 256K/8 slots but regresses 1M/4 slots, so shared-down remains opt-in. |
| Sprint 121 16-slot 256K throughput mode | `43.659461` at 256K/16 slots vs `34.445844` same-binary 8-slot control | Measured | Raising the active-slot ceiling fills more Volta HMMA token lanes and increases TurboMind route counts. The launcher admits 16 slots only for 256K-class contexts and rejects unsafe 16-slot 1M configs before allocation. |
| Sprint 135 32-slot 128K throughput mode | `52.840889` at 128K/32 slots vs `45.780913` same-context 16-slot control | Measured | Raising the active-slot ceiling again improves aggregate throughput when the context cap is lowered enough to stay inside the 32 GB V100 memory budget. 256K remains capped at 16 slots. |
| Sprint 136 64-slot 64K throughput mode | `57.322945` at 64K/64 slots vs `52.884400` same-context 32-slot control | Measured | Slot-width scaling continues to help, but the marginal gain shrank to about `8.4%`. The short-context tier is now better utilized, while long-context tiers remain capped for memory safety. |
| Sprint 137 128-slot 32K throughput mode | `59.598172` at 32K/128 slots vs `57.170428` same-context 64-slot control | Measured | Slot-width scaling is still positive but now only about `4.2%`. This is useful as an explicit short-context mode, but it strongly suggests the easy scheduler-width sweep is nearly exhausted. |
| Sprint 138 wide compact gate/up baseline | `0.6379 ms` fused gate_up at 768 routed rows | Measured | The benchmark now covers the 192/384/768-route compact shapes produced by high-slot serving. The next kernel sprint should beat this larger-route baseline, not the older 96-route case. |
| Sustained benchmark without major kernel changes | `~20-60` tok/s | Medium | Current evidence reaches the low end of the continuous-batching band at 256K/16 slots and improves to `59.598172` at 32K/128 slots; more slots help when they fill existing HMMA/TurboMind tile shapes, but this is still far below the practical target. |
| Continuous token-step batching, 8-128 active slots | `~40-240` tok/s | Medium-low | Sprint 137 reached `59.598172` tok/s at 128 slots/32K. Further progress requires persistent per-slot state, no per-request reset, multi-token batching, wider safe admission where memory allows it, and useful queue depth. |
| Optimized MoE/expert batching with fused low-bit kernels | `~300-1,200` tok/s | Low until proven | Requires routed expert grouping, fused unpack/dequant plus HMMA/DP4A-style kernels, fewer launches, and hot-path kernel selection. |
| Hero synthetic benchmark | `~1,000-3,000+` tok/s | Speculative | Requires short context, high concurrency, persistent grouped kernels, MTP commit, and excellent load balance. |

The user-provided `~1k-2k` aggregate tok/s target is a reasonable aspirational
serving objective for this hardware only after continuous batching and expert
kernel work land. It should not be treated as the baseline for the current
runtime. The next sprints should first make the benchmark honest, then raise
GPU utilization with architectural changes, and only then compare against the
1k+ tok/s target.

## Sprint Sequence

### Sprint 001 - Baseline DS4 V100 Appliance Planner And Source Inventory [complete]

- **Goal**: Prove the model inventory, source dtype mix, memory envelope, and
  baseline layer-sharded V100 topology.
- **Rationale**: The project first needed to know whether the high-intelligence
  DS4 Flash source model could fit and how it should be mapped before runtime
  work could be safe.
- **Outcome**: `SHIP`. The canonical source model was identified, exact tensor
  inventory was recorded, and the planner showed 8x V100 pure-residency
  feasibility for the baseline topology.

### Sprint 002 - Source Loader And Pack Manifest Baseline [complete]

- **Goal**: Teach the runtime tools to recognize the native source GGUF layout
  and emit an inventory-backed pack manifest.
- **Rationale**: The source layout differs from the older q2/q4 DS4 GGUF family,
  so loader and manifest work had to precede any real execution path.
- **Outcome**: `EXTEND`. The source model is recognized, validated, and
  manifested, while generation remains intentionally guarded until V100
  FP8/MXFP4 execution paths exist.

### Sprint 003 - Manifest-Driven Packer Baseline [complete]

- **Goal**: Convert the Sprint 002 manifest into deterministic per-GPU shard
  offsets and a pack index.
- **Rationale**: Runtime residency needs immutable GPU-owned weight shards, not
  ad hoc GGUF offsets or persistent dequantized copies.
- **Outcome**: `SHIP`. The packer writes deterministic `gpuN.weights` layouts
  and validates source ranges; full real-model shard emission was deferred to
  persistent cluster scratch.

### Sprint 004 - Runtime Pack Loading And V100 Device Residency Smoke [complete]

- **Goal**: Prove that the runtime can reconcile pack metadata and upload all
  source-faithful packed bytes to the 8 V100s.
- **Rationale**: Device residency needed to be proven before compute scheduling,
  KV allocation, or source-model decode could be credible.
- **Outcome**: `SHIP`. Full real-model shards were emitted, both GGUF and shard
  providers loaded all 1328 tensors into CUDA device arenas, spot checks and a
  cross-provider check passed, and all GPUs retained the required reserve.

### Sprint 005 - First Resident BF16 Gather/Expand Probe [complete]

- **Goal**: Execute a diagnostic BF16 row-gather/expand probe from resident
  `ds4_gpu_arena` bytes, using `token_embd.weight` as the first source-format
  tensor family and returning host F32 samples for exact verification.
- **Rationale**: BF16 embedding is the lowest-risk useful source-dtype proof
  after residency. It validates arena pointers, descriptor bounds, dtype
  expansion, and CUDA launch semantics before Sprint 006 introduces production
  multi-GPU execution context.
- **Outcome**: `SHIP`. Host-stub and CUDA probes passed, GGUF and shard
  provider `token_embd.weight` probes passed on the 8x V100 pod, and source
  model generation remains guarded.

### Sprint 006 - Multi-GPU Execution Context And Layer Skeleton [complete]

- **Goal**: Introduce the production 8-GPU execution context and a layer-owned
  no-math layer skeleton with hidden-context relay boundaries and explicit
  fail-closed V100 execution-format policy.
- **Rationale**: Full decode requires streams, handles, scratch, tensor
  descriptors, device ownership, and boundary transfer semantics that are
  shaped by the first resident tensor probe. V100 has no native BF16, FP8, or
  FP4 tensor-core path, so Sprint 006 must encode BF16 as source/probe only,
  FP8/MXFP4 as packed inputs to later registered kernels, FP16 HMMA with FP32
  accumulation as the dense production target, and FP32 as control/debug math
  rather than a broad GEMM fallback.
- **Outcome**: `SHIP`. The sidecar context, descriptor policy, real pack
  skeleton walk, production 8x V100 topology check, CUDA resource ownership, and
  HC relay smoke shipped. Generation remains guarded.

### Sprint 007 - Source-Layout Single-Slot Decode Oracle [complete]

- **Goal**: Build a guarded CPU-only source-layout oracle that proves exact
  BF16/F32/I32/F8_E4M3_B128/MXFP4 semantics and matches at least one short
  official first token before production V100 kernels are trusted.
- **Rationale**: Source dtype and V100 runtime dtype must stay separate; a
  fail-closed correctness oracle must come before prefill, long context,
  multi-slot scheduling, MTP, or server deployment.
- **Outcome**: `SHIP`. The guarded oracle selected the official expected token
  `16` for `short_reasoning_plain`, normal source generation remains guarded,
  MXFP4 row semantics were corrected, and source-layout KV defaults to F16
  before Sprint 008 device-kernel work.

### Sprint 008 - Source Oracle Harness And V100 KV Admission Anchors [complete]

- **Goal**: Turn Sprint 007's manual source-oracle proof into automated
  official-vector validation, add source-layout guard regressions, expose exact
  F16 KV admission/reporting for the layer-owned V100 topology, and land one
  bounded CUDA source-format anchor.
- **Rationale**: Full V100 source-layout prefill should not begin until the
  oracle, guard, memory-admission, and source-format device-anchor contracts are
  executable and tested.
- **Outcome**: `SHIP`. The source oracle runner selected
  `short_reasoning_plain` token bytes `3136`, guard checks passed on the source
  model, exact F16 KV admission is reported by stage and fails closed on
  over-budget slots, MXFP4 source layout parity is hardened, and a bounded CUDA
  F8_E4M3_B128 row-decode anchor passed on V100 `sm_70`.

### Sprint 009 - V100 Prefill And Compressed KV Execution [complete]

- **Goal**: Implement the first layer-owned V100 source-layout prompt prefill
  and compressed KV/indexer state update path, validated against the Sprint 007
  oracle and Sprint 008 KV admission contract.
- **Rationale**: This turns the planning/anchor surfaces into usable prompt
  handling while preserving fail-closed normal serving until correctness is
  demonstrated.
- **Outcome**: `SHIP`. The diagnostic path now allocates derived F16 KV arenas,
  validates V100 allocation/admission at 256K and 1M single-slot tiers, runs
  source-layout guards on the real model, and passes a bounded CUDA prefill/KV
  smoke covering ratio-128 and ratio-4/indexer state updates on `sm_70`.

### Sprint 010 - V100 Single-Slot Decode Integration [complete]

- **Goal**: Wire the bounded Sprint 009 KV surfaces into a real layer-owned
  single-slot V100 prefill/decode slice that consumes projection/compressor
  outputs and compares a bounded result against the source oracle.
- **Rationale**: Sprint 009 proved diagnostic KV allocation and row/state
  updates, but deployment should wait until V100 layer execution reaches
  selected-token or bounded-logit correctness.
- **Outcome**: `SHIP`. Stage-owned KV subviews and updates now pass on V100,
  and real compressor recurrence is validated against CPU references for
  ratio-128, ratio-4 attention, and ratio-4 indexer-shaped paths. Full
  source-format dense projection, MoE, logits, selected-token decode, and
  serving remain deferred.

### Sprint 011 - V100 Source Projection And Attention Slice [complete]

- **Goal**: Prove bounded source F8/BF16 projection boundaries and feed
  projection-equivalent device tensors through ratio-4 and ratio-128
  attention/compressor slices on V100.
- **Rationale**: Sprint 010 proved KV ownership and compressor recurrence, but
  the next untrusted surface is source-format projection math. Full logits
  should wait until this path is correct.
- **Outcome**: `SHIP`. Source-F8 projection diagnostics now run from
  device-resident arenas into device tensors, BF16/F32 policy is executable,
  ratio-128 and ratio-4 attention/compressor slices compare against CPU/source
  references, and stage-owned KV writes can consume device-resident projection
  rows.

### Sprint 012 - V100 Appliance Gate And Bounded Output-Head Logits [complete]

- **Goal**: Add a bounded source-BF16 output-head/logits primitive on V100 and
  a runnable appliance readiness gate that validates real-model guards,
  existing V100 smokes, and the new logits/top-k smoke.
- **Rationale**: Deployment should wait for a coherent logits-producing V100
  path. Sprint 011 proves projection-fed attention/compressor slices; Sprint
  012 fills the output-head/logits surface and makes readiness status explicit
  without pretending full MoE or serving are complete.
- **Outcome**: `SHIP`. Source-BF16 output-head rows can now be reduced into
  bounded logits on V100 and top-k compared against a CPU reference. The
  appliance gate passes implemented checks and correctly reports `ready=false`
  until full MoE/selected-token serving exists.

### Sprint 013 - V100 Source MXFP4 MoE And Selected-Token Gate [complete]

- **Goal**: Add a bounded source-MXFP4 routed expert primitive and a
  single-token router/MoE/output-head fixture that produces a selected-token
  comparison on V100.
- **Rationale**: Sprint 012 proves the output-head/logits surface but the gate
  still reports not-ready. The next concrete blocker is the routed expert path:
  DS4 Flash stores routed gate/up/down experts as MXFP4 source tensors, and
  deployment should wait until MoE and selected-token evidence exist.
- **Outcome**: `SHIP`. MXFP4 source expert matmuls and a bounded router/MoE/
  output-head selected-token smoke pass on V100. The remaining readiness gap is
  real pack-index layer integration and shared-expert/full scheduler wiring.

### Sprint 014 - V100 Real Pack-Index Layer Descriptor Gate [complete]

- **Goal**: Add a fail-closed descriptor gate that validates the real pack-index
  rows needed by a source-layout layer, including attention, compressor/indexer,
  router, routed/shared experts, HC controls, and output head.
- **Rationale**: Sprint 013 proves synthetic MoE composition. Deployment should
  wait until the same kernel surfaces consume real model descriptors. A strict
  descriptor contract is the next integration step before real layer compute.
- **Outcome**: `SHIP`. The descriptor gate validates 35 real layer-2/global
  descriptors, fails closed on missing required rows, and is wired into the
  V100 appliance gate behind `--pack-index`.

### Sprint 015 - V100 Descriptor-Bound FFN Compute Gate [complete]

- **Goal**: Materialize validated pack-index descriptors into runtime bindings
  and consume real source-model bytes at real pack offsets in a
  descriptor-bound FFN compute path.
- **Rationale**: Descriptor validation is necessary but not sufficient; the
  next readiness jump is executing real model bytes through the bounded kernel
  surfaces, including the shared expert path.
- **Outcome**: `SHIP`. Runtime tensor bindings landed, layer-2 binding
  validation passes locally and on the pod, and a descriptor-bound V100 FFN
  smoke executes real routed MXFP4 plus shared F8 bytes from the source GGUF at
  real pack offsets.

### Sprint 016 - V100 Descriptor-Bound Router FFN Gate [complete]

- **Goal**: Upgrade descriptor-bound FFN compute from fixed expert to
  model-selected routed experts using real `ffn_gate_inp.weight` and
  `ffn_gate_tid2eid` descriptors.
- **Rationale**: Sprint 015 proves real-byte FFN compute, but serving requires
  real router scheduling before a coherent layer state machine can be trusted.
- **Outcome**: `SHIP`. Source-F32 arena matmul landed, the descriptor-bound FFN
  smoke computes router logits from real bytes, selects experts through the real
  hash-router table, executes all six selected routed experts plus the shared
  expert, and passes the full V100 appliance gate.

### Sprint 017 - V100 Scheduler-Owned Layer State Gate [complete]

- **Goal**: Introduce a reusable scheduler-owned layer execution state that
  binds real descriptors once and owns the router/FFN scratch needed by later
  attention, residual, norm, and selected-token integration.
- **Rationale**: Sprint 016 still proves router-selected FFN as a standalone
  smoke. The next readiness gap is making descriptor-bound execution a runtime
  surface the appliance scheduler can call instead of a test-local composition.
- **Outcome**: `SHIP`. `ds4_v100_layer_state` now owns descriptor-bound
  router/FFN metadata, route matrix construction, source row views, and FFN
  arena-span sizing. The descriptor-bound FFN smoke uses it, and the V100
  appliance gate includes and passes `layer_state`.

### Sprint 018 - V100 Descriptor-Bound Attention Projection Residual Norm Gate [complete]

- **Goal**: Extend the scheduler-owned layer state from router/FFN ownership to
  descriptor-bound attention projection/control ownership, then run real
  source-byte attention projection, residual add, and norm work on V100.
- **Rationale**: Serving is still blocked by the lack of full layer output.
  Sprint 017 created the state surface; Sprint 018 should bridge existing
  synthetic attention kernels to real descriptor-bound attention source bytes
  without claiming full softmax/compressed-KV layer correctness.
- **Outcome**: `SHIP`. Attention/control descriptors are part of
  `ds4_v100_layer_state`, the layer-state smoke validates real attention
  dimensions and arena span, and the new descriptor-bound attention smoke runs
  real source-byte q/kv/output projection, residual add, and FFN pre-norm
  surfaces on V100 against CPU source-format references.

### Sprint 019 - V100 Integrated Single-Layer Runtime Slice [complete]

- **Goal**: Replace Sprint 018's bounded attention-output proxy with a
  scheduler-owned single-layer executor that produces a real next-hidden vector
  for a representative ratio-4 layer by composing semantic attention,
  residual/norm, and router-selected FFN.
- **Rationale**: The appliance still cannot produce the next hidden state from
  a real layer. Sprint 018 proved real projection/control surfaces; Sprint 019
  should ship a reusable runtime slice instead of another isolated primitive.
- **Outcome**: `SHIP`. `ds4_v100_layer_execute` now composes real
  descriptor-bound projection bytes, semantic raw/compressed attention inputs,
  grouped F8 attention output, residual, FFN pre-norm, router-selected MXFP4
  routed experts, shared F8 expert, and final next-hidden residual. The
  integrated smoke passes on one V100 and in the full appliance gate. Real
  compressor/indexer descriptor binding and HC pre/post scheduling remain the
  next blockers.

### Sprint 020 - V100 Compressor/Indexer And HC Scheduler Bridge [extended]

- **Goal**: Bind real compressor/indexer descriptors into layer state, execute
  compressed row generation/selection inside the layer executor, and wrap the
  hidden-vector body with DS4 HC pre/post scheduling.
- **Rationale**: Sprint 019 proves the hidden-vector body of one layer, but the
  appliance still needs real compressed-KV production, ratio-4 indexer
  visibility, and HC state handling before a full 43-layer selected-token path
  is credible.
- **Outcome**: `EXTEND`. `ds4_v100_layer_state` now binds real
  compressor/indexer descriptors and the executor has an HC-state entrypoint
  that runs DS4 attention and FFN HC pre/post around the hidden-vector body.
  The full 8-GPU V100 appliance gate passes with `ready=false`. Executor-owned
  compressed-row generation and indexed ratio-4 compressed attention move to
  Sprint 021.

### Sprint 021 - Executor-Owned Compressor/Indexer Decode Rows [complete]

- **Goal**: Move attention compressor rows, ratio-4 indexer compressor rows,
  indexer scoring/top-k, and indexed compressed attention into the executor
  instead of passing test-built compressed KV rows.
- **Rationale**: Sprint 020 proved the descriptors and HC layer surface. The
  next correctness blocker is making compressed KV production part of the real
  scheduler-owned layer path.
- **Outcome**: `SHIP`. `ds4_v100_layer_execute` now accepts mutable
  decode-cache state, generates raw KV, attention compressed rows, ratio-4
  indexer rows, indexer top-k visibility, and indexed mixed attention from real
  descriptors. The integrated smoke forces `indexer_top_k=1` to reach indexed
  attention in eight decode steps, while production default remains 512. The
  full V100 gate passes and remains `ready=false` pending full scheduler,
  selected-token decode, serving, MTP, and throughput.

### Sprint 022 - Bias Router And Resident Stage Scheduler [complete]

- **Goal**: Remove the hash-router-only execution limit and introduce a
  scheduler-owned resident stage walk over real pack bytes.
- **Rationale**: The model cannot reach full selected-token decode if the
  executor stops at layer 2 or if scheduling remains a single-layer test
  fixture. Stage 0 is the right first target because it owns token embedding
  and includes SWA-only, ratio-4, and ratio-128 layers.
- **Outcome**: `SHIP`. The layer executor now supports hash and bias routers,
  layer 3 ratio-128 bias routing passes on V100, and
  `ds4_v100_stage_scheduler` uploads the full gpu0 shard into a resident arena
  and executes layers 0-5 from a token embedding seed. The full V100 gate
  passes and remains `ready=false` pending cross-GPU 43-layer scheduling,
  selected-token decode, serving, MTP, and throughput.

### Sprint 023 - Cross-GPU Two-Stage Scheduler Handoff [complete]

- **Goal**: Prove the first real scheduler handoff between resident stage
  owners.
- **Rationale**: Stage-local scheduling is insufficient for a layer-sharded
  appliance. The next risk is whether HC can move between GPUs and whether the
  CUDA backend can safely run the same source-model helpers on more than one
  device in one process.
- **Outcome**: `SHIP`. The scheduler now runs layers 0-5 on gpu0, copies HC to
  gpu1 with `cudaMemcpyPeer`, and runs layers 6-11 on gpu1. CUDA tensor
  allocation/copy paths now track device ownership, and model-range caches are
  device-local to avoid cross-GPU pointer reuse. The full V100 gate passes and
  remains `ready=false` pending full 43-layer scheduling, selected-token
  decode, serving, MTP, and throughput.

### Sprint 024 - Full 8-Stage Scheduler Chain [complete]

- **Goal**: Generalize the scheduler handoff from two stages to the full
  8-GPU, 43-layer model body.
- **Rationale**: Output-head correctness is not meaningful until final HC is
  produced by the real layer-sharded body, not a partial stage fixture.
- **Outcome**: `SHIP`. The full scheduler smoke opens all eight resident stage
  arenas, executes layers 0-42, handoffs HC across every stage boundary, and
  verifies finite nonzero final HC on gpu7. The V100 gate now removes
  `full_43_layer_scheduler` from readiness when this check passes and remains
  `ready=false` pending selected-token decode, serving, MTP, and throughput.

### Sprint 025 - Scheduler Output-Head Selected Token Surface [complete]

- **Goal**: Attach gpu7 output-head selected-token execution to the resident
  scheduler.
- **Rationale**: Full-body traversal is necessary but not sufficient; the
  appliance needs final HC collapse, output normalization, vocab projection,
  and a top-1 token surface before selected-token correctness can be debugged.
- **Outcome**: `EXTEND`. The output-head path runs on V100 and produces a
  finite selected token after replaying the `short_reasoning_plain` prompt, but
  the selected token does not match the official/source oracle. Readiness still
  blocks on `real_model_selected_token`.

### Sprint 026 - Output-Head Divergence Localization [complete]

- **Goal**: Prove or eliminate the gpu7 output-head adapter as the cause of the
  selected-token mismatch.
- **Rationale**: Sprint 025 proved that the scheduler can produce logits, but
  not whether the mismatch comes from final HC collapse/vocab projection or
  from earlier layer execution.
- **Outcome**: `SHIP`. The deterministic HC parity smoke matches CPU and V100
  output-head top-5 exactly enough for the diagnostic tolerance, and the prompt
  top-k diagnostic records the remaining oracle mismatch. The next blocker is
  stage/layer HC divergence localization inside the 43-layer body.

### Sprint 027 - V100 Selected-Token Correctness And HC Checkpoints [complete]

- **Goal**: Localize scheduler-body divergence and make the official
  selected-token oracle pass on V100.
- **Rationale**: Output-head parity passed in Sprint 026, so the next useful
  implementation was checkpoint visibility through the actual 43-layer body
  rather than more output-head work.
- **Outcome**: `SHIP`. The scheduler now decodes native BF16 token embeddings
  correctly, defaults KV/cache mutation to the F16 source-layout contract, and
  passes the selected-token gate for expected bytes `3136`. The gate remains
  `ready=false` only for public serving, MTP, and throughput.

### Sprint 028 - V100 Replay Runtime And Timing Tool [complete]

- **Goal**: Move selected-token replay out of a smoke test and into a reusable
  appliance runtime/tool with timing counters.
- **Rationale**: Correctness alone was not a usable surface. The project needed
  a commandable path that loads the resident 8-stage scheduler, emits tokens,
  and measures where time is going.
- **Outcome**: `SHIP`. `tools/ds4-v100-replay` generates tokens through the V100
  scheduler, verifies expected bytes `3136`, and emits JSON timing/memory data.
  The gate should now remove `throughput_benchmark`; public serving and MTP
  remain open.

### Sprint 029 - V100 Resident HTTP Appliance Smoke [complete]

- **Goal**: Keep the V100 replay runtime resident behind a minimal loopback
  HTTP endpoint and prove selected-token correctness through the served path.
- **Rationale**: A CLI replay tool is useful for measurement, but the appliance
  needs a long-running process that keeps all eight stage schedulers resident
  and handles independent requests without reuploading weights each time.
- **Outcome**: `SHIP`. `tools/ds4-v100-replay --serve` exposes
  `/v100/selected-token`, resets scheduler KV/HC state per request, returns
  expected bytes `3136`, and the full V100 gate now reports
  `missing=mtp`.

### Sprint 030 - V100 MTP Sidecar Readiness Gate [complete]

- **Goal**: Validate the actual DeepSeek-V4 Flash MTP sidecar format in the
  appliance gate without prematurely enabling speculative decode.
- **Rationale**: Prior work showed MTP loading is feasible, but the risky part
  is Q4_K/Q8_0 MTP forward parity and draft/verify state. The gate needed to
  distinguish a valid sidecar from the missing runtime path.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-sidecar-gate` validates the
  `deepseek4_mtp_support` GGUF, reports 32 tensors and 3.807600108 GB of
  described sidecar tensor bytes, and the full V100 gate now reports
  `missing=mtp_runtime`.

### Sprint 031 - V100 MTP Resident Sidecar Runtime Bridge [complete]

- **Goal**: Turn the validated MTP sidecar into a V100-resident runtime asset
  without enabling speculative decode prematurely.
- **Rationale**: MTP needs its own Q8_0/Q4_K sidecar path. Reusing the main
  source-layout MXFP4/F8 V100 layer-state binder would hide dtype and residency
  mistakes.
- **Outcome**: `SHIP`. `ds4_v100_mtp_sidecar_open` maps the sidecar, allocates
  a compact gpu7 device arena, uploads all 32 MTP tensors, spot-checks resident
  bytes, and the full V100 gate now reports `missing=mtp_forward`.

### Sprint 032 - V100 Level 2 Base Appliance Usability Gate [complete]

- **Goal**: Make the non-MTP one-slot V100 appliance usable enough for
  operator-driven short generation with health/status, repeated HTTP request
  evidence, longer decode evidence, and a runbook.
- **Rationale**: The readiness ladder needed a practical base appliance rung
  before MTP forward work and throughput claims. Without this, `ready=false`
  could hide the fact that the base model path was already usable within clear
  limits.
- **Outcome**: `SHIP`. The replay server exposes health/status, the smoke
  proves two sequential two-token HTTP requests from one resident process, the
  full V100 gate passes with `failures=0`, Level 2 no longer appears as a
  readiness blocker, and the remaining blocker is `missing=mtp_forward`.

### Sprint 033 - V100 Resident MTP Q8 Projection Probe [complete]

- **Goal**: Prove the first resident MTP sidecar compute primitive on V100 by
  running Q8_0 projection matmuls directly from the compact gpu7 arena.
- **Rationale**: MTP residency alone did not prove that the sidecar's native
  Q8_0/Q4_K tensor families could execute from `resident_offset` layout. The
  safest first step was matching the prefix projection tensors against the
  existing Q8_0 CUDA kernel path before enabling full MTP forward or
  speculative serving.
- **Outcome**: `SHIP`. `ds4_gpu_arena_q8_0_matmul_f32` now reuses the V100
  Q8_0 prequantized DP4A kernels against arena-resident sidecar bytes,
  `tools/ds4-v100-mtp-prefix-smoke` validates `e_proj` and `h_proj` with
  `max_abs=0`, and the full V100 gate passes with `missing=mtp_forward`.

### Sprint 034 - V100 Resident MTP Prefix Composition Probe [complete]

- **Goal**: Extend the resident MTP compute proof from standalone Q8_0
  projection parity to the full native prefix composition chain.
- **Rationale**: The MTP block consumes `mtp_input_hc`, not isolated projection
  outputs. The runtime needed resident F32 norm-weight access and HC
  composition before dense MTP block execution or draft logits could be
  meaningful.
- **Outcome**: `SHIP`. `ds4_gpu_arena_f32_rms_norm_f32` and
  `ds4_v100_mtp_sidecar_f32_vector_view` now support resident F32 prefix norms,
  `tools/ds4-v100-mtp-prefix-smoke` validates `enorm`, `e_proj`, HC repeat,
  `hnorm`, `h_proj`, and `mtp_input_hc`, and the full V100 gate passes with
  `failures=0 ready=false missing=mtp_forward`.

### Sprint 035 - V100 Resident MTP Q4_K Routed Expert Execution [complete]

- **Goal**: Prove the MTP sidecar's Q4_K routed expert tensors execute directly
  from the gpu7 resident arena on V100.
- **Rationale**: Prefix composition alone does not exercise the dominant MTP
  FFN tensor family. The next correctness surface needed to bypass the generic
  model-map cache and run `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps` through the decode Q4_K kernels from resident offsets.
- **Outcome**: `SHIP`. `ds4_gpu_arena_q4_k_routed_moe_one_f32` reuses the V100
  Q4_K gate/up and direct six-expert down kernels against arena-resident
  sidecar bytes, `tools/ds4-v100-mtp-q4k-smoke` validates the output against a
  selected-slice CPU Q4_K reference with `max_abs=1.43051147e-06`, and the full
  V100 gate now includes `mtp_q4k` while still honestly reporting
  `missing=mtp_forward`.

### Sprint 036 - V100 Resident MTP FFN Slice [complete]

- **Goal**: Assemble the resident MTP FFN block slice from sidecar-resident HC
  control, router, routed Q4_K experts, shared Q8_0 experts, and HC expansion.
- **Rationale**: Sprint 035 proved the dominant Q4_K routed expert primitive,
  but full MTP forward still needed the surrounding native FFN structure:
  bias-router selection, shared expert execution, routed+shared accumulation,
  and return to `[4 x 4096]` HC state.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-ffn-smoke` now starts from
  deterministic `after_attn_hc`, executes resident HC FFN control, FFN norm,
  bias router, Q4_K routed experts, Q8_0 shared experts, and HC expand to
  `next_hc`. The focused V100 smoke matches the CPU sidecar-byte reference
  with `next_hc max_abs=2.38418579e-06`, and the full gate includes
  `mtp_ffn` while still correctly reporting `missing=mtp_forward`.

### Sprint 037 - V100 Resident MTP Raw Attention [complete]

- **Goal**: Add resident MTP raw/SWA attention and cache-update evidence using
  sidecar-resident attention sinks and the native 128-row MTP raw cache.
- **Rationale**: Sprint 036 proved the FFN half of the MTP block, but native
  MTP runs `il=1` attention before FFN. The next correctness surface was raw
  cache mutation, sink-aware attention softmax, and ring wrap visibility
  without falling back to mmap-resolved sidecar tensors.
- **Outcome**: `SHIP`. `ds4_gpu_arena_attention_decode_heads_tensor` now
  launches the existing CUDA attention kernels with sinks resolved from the
  gpu7 sidecar arena, and `tools/ds4-v100-mtp-attn-smoke` validates positions
  `0,1,127,128,129` with production FP8-plus-F16 raw KV store. The focused
  smoke passes with `global_max_abs=1.27183739e-08`, the full V100 gate
  includes `mtp_attn`, and readiness remains correctly blocked on
  `missing=mtp_forward`.

### Sprint 038 - V100 Resident MTP Integrated Attention Slice [complete]

- **Goal**: Compose the resident MTP attention slice from real sidecar bytes:
  HC attention control, attention norm, Q/KV projections and norms, raw-cache
  store, sink-aware attention, grouped Q8_0 attention output, and HC expansion.
- **Rationale**: Sprint 037 proved raw attention/cache semantics with synthetic
  Q/KV. The next correctness risk was whether native MTP attention projection
  and output composition work directly from the compact gpu7 resident sidecar
  arena.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-attn-smoke` now validates both the
  raw attention/cache-wrap proof and an integrated attention projection/output
  proof against a CPU sidecar-byte reference. The focused V100 smoke passes
  with `q_heads max_abs=2.14576721e-06`, `kv_row max_abs=0.000867605209`,
  `heads max_abs=2.14576721e-06`, `attn_out max_abs=0.258209229`, and
  `next_hc max_abs=0.19461441`. The full gate includes the stronger
  `mtp_attn` proof and still correctly reports `missing=mtp_forward`.

### Sprint 039 - V100 Resident MTP Logits and Top-K Parity [complete]

- **Goal**: Prove the resident MTP sidecar can collapse an MTP HC state through
  its own output head controls, apply MTP output norm, project through the base
  model vocabulary head, and select top-k draft candidates.
- **Rationale**: Sprint 038 proved integrated MTP attention and Sprint 036
  proved the FFN slice, but `missing=mtp_forward` could not advance without a
  trusted logits/top-k proof for the final draft-candidate surface.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-logits-smoke` uploads the MTP
  sidecar plus the base BF16 `output.weight`, runs the resident MTP logits path
  on gpu7, and compares against a CPU sidecar/base-model oracle. The focused
  V100 smoke passes with exact top-5 token parity, `top1=65615`, and
  `max_abs=9.53674316e-07`. The full gate now includes `mtp_logits PASS` and
  still correctly reports `missing=mtp_forward`.

### Sprint 040 - Resident One-Token MTP Forward Composition [complete]

- **Goal**: Compose the resident MTP prefix, attention, FFN, output HC collapse,
  output norm, base vocabulary projection, and top-k into one continuous gpu7
  forward smoke.
- **Rationale**: Sprints 033-039 proved the individual MTP primitives. The next
  readiness jump required proving their boundary contract in sequence before
  attaching speculative verify/rollback semantics to the serving path.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-forward-smoke` runs a deterministic
  one-token MTP forward path from resident sidecar bytes plus the base BF16
  `output.weight`, matches CPU/GPU top-5 tokens exactly with `top1=101365`,
  and passes the full V100 gate. Readiness now correctly reports
  `missing=mtp_verify`.

### Sprint 041 - MTP Rollback State Safety [complete]

- **Goal**: Prove the target scheduler and MTP raw-visibility state can roll
  back safely after a rejected speculative token.
- **Rationale**: Sprint 040 proves a continuous deterministic forward path, but
  speculative serving is not correct until target rollback boundaries are
  explicit and gated.
- **Outcome**: `SHIP`. Scheduler snapshots now cover current HC identity and
  content, raw KV, compressed KV/state, indexer KV/state/top-k, and counters.
  The focused V100 snapshot smoke passes after eight positions with exact
  restore and deterministic replay. The `mtp_rollback` gate keeps the real MTP
  sidecar resident, proves rejected-token rollback, and the full V100 gate
  passes with `failures=0 ready=false missing=mtp_verify`.

### Sprint 042 - Native Prompt-Token MTP Verify [complete]

- **Goal**: Attach the resident MTP forward path to the actual just-committed
  target token embedding and target HC state, produce a real one-token MTP
  draft, and verify it against target top-1 by exact token equality.
- **Rationale**: Sprint 041 proves rollback safety, but the readiness ladder
  cannot advance until the draft token is produced from native prompt-token
  state instead of a deterministic or synthetic source.
- **Outcome**: `SHIP`. `tools/ds4-v100-mtp-verify-smoke` now reads
  BF16 `token_embd.weight[T]` as F32, captures gpu7 post-commit HC, runs the
  resident MTP forward path, and verifies exact target/MTP top-1 equality. On
  the 8x V100 cluster, committed token `926` at position `18` produced
  `target_top1=1` and `mtp_top1=1`, with snapshot bytes `30107648`,
  `restore_delta=0`, and `replay_delta=0`. The full gate passes with
  `failures=0 ready=false missing=production_deployment`.

### Sprint 043 - Production Deployment Package [complete]

- **Goal**: Turn the verified one-slot V100 appliance into a cluster service
  that can be started, supervised, observed, and rolled back by an operator.
- **Rationale**: The current gate proves correctness, MTP verify, HTTP loopback
  behavior, and timing diagnostics, but it still runs as smoke-test processes.
  The remaining readiness blocker is deployment packaging rather than model
  execution correctness.
- **Outcome**: `SHIP`. The appliance now has
  `tools/ds4-v100-run-appliance.sh`, an env example, systemd and Kubernetes
  deployment templates, `/metrics`, richer status limits, a production
  deployment gate, and updated operator runbook. On the 8x V100 cluster, the
  focused deployment smoke and full gate both pass; the full gate now reports
  `failures=0 ready=false missing=throughput_optimization`.

### Sprint 044 - Throughput Optimization And Operating Envelope [complete]

- **Goal**: Convert the current timing diagnostics into a credible throughput
  and operating-envelope baseline, then implement the first targeted
  optimization that improves startup or decode throughput without weakening
  correctness.
- **Rationale**: Sprint 043 makes the service deployable, but fresh-process
  startup still spends roughly 4.8-5.8 minutes opening/uploading the resident
  stages, and the current one-slot decode timings are diagnostic rather than a
  slot/context throughput claim.
- **Outcome**: `SHIP`. The replay runtime now opens/uploads all eight stage
  schedulers in parallel by default, with `--serial-open` retained for fallback
  and before/after measurement. `tools/ds4-v100-throughput-bench.sh` proves
  serial versus parallel open, preserves first-token bytes `3136`, and is wired
  into the full gate as `throughput_optimization`. On the 8x V100 cluster, the
  focused benchmark improved cold open from `343989.990 ms` to `63032.135 ms`
  (`5.457375x`), and the full gate passed with
  `failures=0 ready=false missing=mtp_speculative_serving`.

### Sprint 045 - Production MTP Speculative Serving [complete]

- **Goal**: Expose the gated one-token native MTP verify path through the
  resident HTTP appliance as a bounded speculative serving mode.
- **Rationale**: The project now has base serving, MTP correctness, deployment,
  and startup optimization. The remaining readiness blocker is that the served
  process still reports `mtp_enabled=false` and does not draft/verify/rollback
  inside the request loop.
- **Outcome**: `SHIP`. The resident HTTP appliance now supports
  `--mtp-serving verify` and launcher `DS4_V100_MTP_SERVING=verify`, opens the
  real gpu7 MTP sidecar and base output head, returns an `mtp` diagnostics
  object, and exposes MTP request/draft/accepted/rejected/skipped counters.
  Focused and full-gate smokes verify prompt token `926`, target token `1`, MTP
  draft token `1`, first token bytes `3136`, and `mtp_accepted=1`. The full
  gate now passes `mtp_speculative_serving` and reports
  `missing=aggregate_slot_context_envelope`.

### Sprint 046 - Aggregate Slot/Context Envelope [complete]

- **Goal**: Define and validate the practical slot/context operating envelope
  for the DS4 V100 appliance.
- **Rationale**: The current service is still one-slot and sequential. The next
  readiness blocker is no longer correctness, deployment, startup, or served
  MTP verify; it is whether the appliance can admit useful slot/context modes
  and report aggregate tok/s without overfilling 32 GiB V100 VRAM.
- **Outcome**: `SHIP`. Admission/queue policy and slot/context envelope tooling
  are now explicit in the runtime contract and full gate (`slot_context_admission`).
  Planner tiers, over-context rejection checks, and status/metrics limit fields
  are implemented. The remaining blocker moved from admission to active
  microbatch scheduler execution.

### Sprint 047 - Active-Microbatch Scheduler Core [complete]

- **Goal**: Implement scheduler-level multi-slot decode/handoff with per-slot
  KV/HC state so active microbatch execution has a real device-resident
  runtime surface.
- **Rationale**: Sprint 046 proved admission and policy, but execution still
  used single-slot scheduler state. The next requirement was real slot-strided
  scheduler ownership and batch decode APIs.
- **Outcome**: `SHIP`. `ds4_v100_stage_scheduler` now supports up to eight
  active slots with independent cache slices and HC cursors plus
  `decode_token_batch`, `decode_hc_batch`, and `handoff_batch` APIs. Scheduler
  and full-scheduler CUDA smokes now support `--slots N`, and the full gate
  now includes `active_microbatch_scheduler`. Service status/metrics now expose
  `scheduler_slots_ready=1` while keeping `tensor_batched_slots=false`.

### Sprint 048 - Request-Loop Active Microbatch Integration [complete]

- **Goal**: Integrate active-microbatch scheduling into the HTTP request loop
  so concurrent prompts can execute through scheduler batch APIs.
- **Rationale**: Sprint 047 enabled scheduler-level slot batching, but the
  service loop still serialized independent requests. Level 6 needed real
  request-loop integration before throughput benchmarking.
- **Outcome**: `SHIP`. `tools/ds4-v100-replay --serve` now enqueues pending
  requests and dispatches batch generation through
  `ds4_v100_replay_generate_first_token_batch` for non-MTP one-token requests,
  with fallback behavior preserved for non-batchable requests. Admission and
  queue policy remain explicit (`reject-busy`/`sequential`). Remaining work is
  cluster throughput/latency evidence and broader multi-token batching.

### Sprint 049 - Aggregate Throughput Envelope Evidence [complete]

- **Goal**: Convert Level-6 from local/runtime wiring to measured cluster
  evidence with concurrent load and explicit latency/tok/s metrics.
- **Rationale**: Sprint 048 completed request-loop integration, but readiness
  still lacked cluster-backed aggregate throughput and MTP-aware comparison data.
- **Outcome**: `SHIP`. Added
  `tools/ds4-v100-aggregate-throughput.sh` and gate rung
  `aggregate_slot_context_throughput`, and updated gate readiness so it can
  emit `READY` when no keys are missing. Ran cluster measurements on
  `llamacpp-build-8gpu` (`gpu-01`) with successful first-token correctness and
  no request errors across:
  - `ctx=262144`, `slots=1/2/4/8`, policies `sequential/reject-busy` (base mode);
  - `ctx=1048576`, `slots=1/8`, policies `sequential/reject-busy` (base mode);
  - prior `ctx=1048576`, `slots=2/4`, policy `sequential` (base mode);
  - focused MTP on/off comparison at `ctx=1048576`, `slots=2`, `tokens=2`.
  Evidence is captured under `logs/from-cluster/sprint049*`.

### Sprint 050 - Readiness Closure And Gate Hardening [complete]

- **Goal**: Close the final proof loop by running the full 8-GPU gate to
  `ready=true` and harden the gate flow against restart/lock and CLI mismatches.
- **Rationale**: Sprint 049 added broad throughput evidence, but full closure
  still depended on a clean all-rungs gate pass with no missing keys.
- **Outcome**: `SHIP`. Fixed three gate blockers:
  - added `tools/ds4-v100-plan` to build targets for slot/context admission;
  - added `--ctx` support to `ds4-v100-appliance-smoke.sh`;
  - isolated replay lock files per run/case across throughput and serving smokes.
  Updated gate readiness logic to emit `READY` when no keys are missing.
  Full cluster gate on `llamacpp-build-8gpu` now reports:
  `gate readiness READY` and `gate summary PASS failures=0 ready=true`.
  Artifacts are in `logs/from-cluster/sprint050`.

### Sprint 051 - Gate Aggregate Matrix Profiles [complete]

- **Goal**: Make broader slot/context throughput envelope runs first-class gate
  operations instead of ad hoc script edits.
- **Rationale**: Sprint 050 closed readiness and hardened the existing rung,
  but wide envelope coverage still required manual matrix edits, slowing
  iteration and increasing operator error risk.
- **Outcome**: `SHIP`. `tools/ds4-v100-gate.sh` now supports
  `--aggregate-profile fast|full` with explicit per-axis overrides
  (`ctx-tiers`, `slot-tiers`, `queue-policies`, `requests`, `tokens`,
  `host`, `port-base`) and logs the resolved matrix at runtime.
  `docs/operations/DS4-V100-APPLIANCE.md` now documents both profile defaults
  and a full-profile cluster invocation pattern. Full-profile gate execution
  on `llamacpp-build-8gpu` passed all rungs with `ready=true`, and the
  aggregate rung produced complete 32-case TSV/JSON evidence.

### Sprint 052 - Sustained Decode And Utilization Baseline [complete]

- **Goal**: Replace the one-token aggregate gate as the performance reference
  with sustained multi-token decode benchmarks, GPU utilization capture, and
  per-stage/per-kernel timing.
- **Rationale**: The current tok/s number is dominated by request shape and
  prompt replay. Before optimizing kernels, the project needs a benchmark that
  measures steady-state decode under realistic queue depth and records whether
  GPU utilization, launches, HBM, or synchronization dominate.
- **Outcome**: `SHIP`. Added
  `tools/ds4-v100-sustained-decode-bench.sh`, optional sustained profiles in
  `tools/ds4-v100-gate.sh`, runbook coverage, and cluster artifacts under
  `logs/from-cluster/sprint052-sustained-baseline`. The first sustained
  baseline measured `3.304551` aggregate generated tok/s and `10.804%` average
  GPU utilization at 1M context, one slot, and 16 generated tokens/request.

### Sprint 053 - Continuous Token-Step Microbatching [complete]

- **Goal**: Extend request-loop batching from first-token-only execution to
  multi-token token-step execution across active slots.
- **Rationale**: Moderate and high aggregate throughput require decode to keep
  multiple sequences resident and advance them together. The current
  `tensor_batched_slots=false` surface and fallback per-request generation path
  cannot feed the GPUs enough work.
- **Outcome**: `SHIP`. Added `ds4_v100_replay_generate_batch`, routed
  same-token-count non-MTP pending HTTP batches through it, exposed
  `tensor_batched_*` counters in status/metrics, and added status snapshots to
  `tools/ds4-v100-sustained-decode-bench.sh`. Cluster artifacts under
  `logs/from-cluster/sprint053-token-step-batching` prove correctness and
  batch execution. Performance improved only slightly (`3.291466` to
  `3.371659` generated tok/s at 1M, `slots=1` to `slots=2`), so the next sprint
  should focus on real hot-path kernel occupancy rather than more request-loop
  plumbing.

### Sprint 054 - Hot-Path Kernel Selection And Low-Bit Expert Integration [complete]

- **Goal**: Use Sprint 052 timing to replace the hottest routed/shared FFN and
  dense projection calls with the best available V100 low-bit kernels, then
  prove end-to-end speedup without losing selected-token correctness.
- **Rationale**: The existing MXFP4/Q8/Q4 CUDA paths prove source-format
  correctness and residency, but practical throughput needs fused unpack/
  dequant plus tensor-core or integer execution in the main decode hot path, not
  just standalone smokes.
- **Outcome**: `SHIP`. Added `ds4_gpu_arena_mxfp4_pair_swiglu_f32`, wired it
  into the routed FFN path, extended MXFP4 smoke coverage, proved real replay
  token hex `3136`, and captured cluster artifacts under
  `logs/from-cluster/sprint054-fused-mxfp4`. The change gives a small
  sustained speedup (`3.291466` to `3.384749` generated tok/s at one slot;
  `3.371659` to `3.486851` at two slots) but does not materially raise GPU
  utilization, so Sprint 055 should fuse routed down/accumulation and group
  selected routes.

### Sprint 055 - Routed Expert Batching And Persistent MoE Scheduling [complete]

- **Goal**: Batch routed experts across active slots and reduce launch overhead
  with a persistent or grouped-MoE scheduler.
- **Rationale**: V100 tensor cores need enough effective M to stay busy. The
  biggest gap between current performance and the 300+ tok/s target is likely
  expert scatter, small per-expert GEMMs, and per-route launch overhead.
- **Outcome**: `SHIP`, with scope narrowed to the next bounded launch reduction.
  Added `ds4_gpu_arena_mxfp4_matmul_add_f32`, wired routed down+accumulation
  into the scheduler, extended smoke coverage, proved real replay token hex
  `3136`, and archived cluster artifacts under
  `logs/from-cluster/sprint055-mxfp4-down-accum`. The speedup over Sprint 054
  is small (`3.384749` to `3.410425` generated tok/s at one slot;
  `3.486851` to `3.503283` at two slots), so Sprint 056 should stop
  one-route-at-a-time cleanup and group selected routes or batch layer
  execution across slots.

### Sprint 056 - Grouped MXFP4 Selected-Route Execution [complete]

- **Goal**: Collapse the six selected routed experts in the main FFN path into
  grouped MXFP4 gate/up/SwiGLU and grouped down-sum kernels.
- **Rationale**: Sprints 054 and 055 proved that route-local launch fusion is
  correct but too small. The next useful kernel primitive needed to process all
  selected routes together while preserving source MXFP4 layout.
- **Outcome**: `SHIP`. Added
  `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_f32`, wired it into
  `execute_ffn_delta`, extended the focused V100 MXFP4 smoke, proved real
  replay token hex `3136`, and archived sustained artifacts under
  `logs/from-cluster/sprint056-grouped-mxfp4-routes`. Generated tok/s improved
  by about `4.17%` at one slot and `4.96%` at two slots over Sprint 055, but
  average GPU utilization remains near `11%`.

### Sprint 057 - Deterministic Token-Step Coalescing And Layer Batching [complete]

- **Goal**: Make multi-slot sustained decode deterministically coalesce
  same-step requests and expose active slots to at least one batched layer
  executor path.
- **Rationale**: Sprint 056's two-slot benchmark did not register
  `tensor_batched_groups`, so the system still cannot rely on request-loop
  batching to feed enough work to the kernels. The next meaningful throughput
  step is scheduler shape, not another single-slot route fusion.
- **Outcome**: `SHIP`, with the batched layer slice off by default. Added a
  short server-side pending-request rendezvous so two-slot sustained decode
  reliably enters `ds4_v100_replay_generate_batch`. Added a batched MXFP4 route
  primitive and `ds4_v100_layer_execute_hc_decode_batch`, but gated scheduler
  use behind `DS4_V100_BATCH_LAYER_FFN` after the first V100 benchmark
  regressed. Default two-slot evidence now has `tensor_batched_groups=2`,
  generated tok/s `3.662490`, and token hex `3136`.

### Sprint 058 - Replay Router Readback Suppression [complete]

- **Goal**: Remove replay-only router selected-expert and route-weight CPU
  readbacks from the generation hot path while preserving diagnostic defaults.
- **Rationale**: Sprint 057 proved request coalescing was honest but flat. The
  next smallest runtime overhead was the per-layer router readback used only
  for validation/reporting, which forced host synchronization before the routed
  MXFP4 kernels consumed device-selected routes.
- **Outcome**: `SHIP`. Added replay, scheduler, and layer-executor options for
  `suppress_router_readback`, defaulted the replay appliance to suppression,
  preserved direct scheduler/layer diagnostics, proved token hex `3136`, and
  archived sustained artifacts under
  `logs/from-cluster/sprint058-router-readback-suppression`. Two-slot
  generated tok/s improved from `3.662490` to `3.704572`, so the cleanup is
  useful but not enough to change the utilization picture.

### Sprint 059 - Persistent Layer Batch Scratch [complete]

- **Goal**: Remove per-layer tensor allocation/free from the multi-slot batch
  path, then enable it by default if cluster benchmarks show a speedup.
- **Rationale**: Sprints 057 and 058 removed request-loop and readback
  synchronization gaps without materially raising throughput. The opt-in batch
  path was previously slower because it paid allocation/copy overhead before
  routed work.
- **Outcome**: `SHIP`. Added scheduler-owned `ds4_v100_layer_batch_scratch`,
  reused HC and FFN batch temporaries across layers/decode steps, preserved the
  allocation fallback for direct callers, and enabled multi-slot layer batching
  by default with `DS4_V100_BATCH_LAYER_FFN=0` as the disable escape hatch.
  Two-slot generated tok/s improved from Sprint 058's `3.704572` to
  `3.862932`, while token hex stayed `3136`.

### Sprint 060 - Pointer-Input Routed FFN Batch [complete]

- **Goal**: Remove or reduce the remaining per-slot FFN input copy before
  routed MXFP4 batch execution.
- **Rationale**: Sprint 059 made the batch path faster by removing allocation
  churn, but `execute_ffn_delta_batch` still copies every slot into
  `input_batch_t`. The next throughput step should either use a one-launch
  gather or a pointer-input routed MXFP4 primitive.
- **Outcome**: `SHIP`. Added
  `ds4_gpu_arena_mxfp4_routed_swiglu_down_sum_batch_ptrs_f32`, extended the
  focused MXFP4 smoke with separate per-slot input tensors, removed
  `ffn_input_batch` from layer scratch, and passed the full two-slot scheduler
  smoke after moving the pointer table into stage-owned scratch. Two-slot
  generated tok/s improved from `3.862932` to `3.915266`.

### Sprint 061 - Batched Shared F8 Expert Path [complete]

- **Goal**: Add and measure a batched source-F8 shared expert path, remove
  remaining FFN batch view churn, and test whether four active slots improve
  aggregate throughput.
- **Rationale**: Routed input staging is no longer the bottleneck. The next
  measured limit is still low utilization with per-slot shared expert work and
  likely small-kernel overhead.
- **Outcome**: `SHIP`, but not as a default throughput win. Added batched
  `F8_E4M3_B128` matmul and pointer-input pair-SwiGLU primitives, validated
  them on V100, exposed shared F8 batching behind `DS4_V100_BATCH_SHARED_F8=1`,
  and added persistent FFN output views to the default path. The opt-in shared
  F8 batch measured below Sprint 060, and 4-slot 256K measured only `3.834046`
  generated tok/s, so the next sprint should move to stage wavefronting,
  committed MTP, or profiler-led kernel rewrites.

### Sprint 062 - Decode Execution-Shape Profiling And MTP/Wavefront Decision [complete]

- **Goal**: Add low-overhead CUDA/event timing across the decode sections, then
  implement the highest-leverage execution-shape change: committed MTP draft
  acceptance, stage wavefronting, or a focused F8/MXFP4 kernel rewrite.
- **Rationale**: Sprint 061 proved that small FFN staging cleanup and more
  active slots do not solve utilization. The next sprint must identify where
  time is actually going and change the execution shape rather than continuing
  launch-count cleanup.
- **Outcome**: `SHIP`. Added opt-in decode profiling to replay and sustained
  benchmarking, then measured 1M/256K and 2/4-slot cases on the V100 pod.
  Stage-profile totals nearly matched stage-decode totals, confirming that the
  layer-synchronous stage schedule dominates current decode time.

### Sprint 063 - Stage Wavefront Slot-Lane Proof [complete]

- **Goal**: Prove scheduler mechanics for independent slot lanes before wiring
  wavefront ordering into the served path.
- **Rationale**: Stage wavefronting could only be considered if the scheduler
  exposed slot-addressable decode, handoff, and read APIs without corrupting
  per-slot KV/HC state.
- **Outcome**: `SHIP`. Added slot-span scheduler APIs and per-device CUDA temp
  scratch. The focused V100 smoke advances two slot lanes in wavefront order
  through two stages and matches the serial HC reference exactly.

### Sprint 064 - Opt-In Served Wavefront Decode [complete]

- **Goal**: Wire stage-wavefront ordering into same-length non-MTP served
  batches behind an opt-in flag and compare against the serial baseline.
- **Rationale**: Sprint 062 showed serialized stage time as the dominant
  blocker, and Sprint 063 proved slot-lane mechanics. The next question was
  whether served diagonal stage ordering alone improved aggregate tok/s.
- **Outcome**: `SHIP` as diagnostic, not default. Added
  `--wavefront-decode` to replay and sustained benchmarking. Correctness
  passed, but paired V100 benchmarks showed wavefront `3.70` generated tok/s
  at 1M/2 slots versus serial `3.86`, and wavefront `3.69` at 256K/4 slots
  versus serial `3.84`. Do not promote this path; move to true asynchronous
  stage workers, MTP commit, or persistent low-bit kernels.

### Sprint 065 - Async Stage Pipeline Decode [complete]

- **Goal**: Replace single-threaded stage ordering with a real opt-in
  per-stage worker pipeline for same-length non-MTP batches.
- **Rationale**: Sprint 064 proved that wavefront ordering without concurrent
  host submission regressed. The next execution-shape test had to overlap
  different V100 stages across active slots.
- **Outcome**: `SHIP` as opt-in. Added `--async-pipeline-decode` to replay and
  sustained benchmarking. The paired V100 matrix preserved token hex `3136`
  and improved generated tok/s from serial `3.85` to async `5.57` at 1M/2
  slots, and from serial `3.81` to async `8.67` at 1M/4 slots. Keep opt-in
  until the stage workers are persistent across token steps and request batches.

### Sprint 066 - Persistent Async Stage Workers [complete]

- **Goal**: Reuse one async pipeline worker per V100 stage across token steps
  instead of creating and joining workers for every token-step batch.
- **Rationale**: Sprint 065 proved per-stage host concurrency is useful but
  left worker lifetime as an obvious overhead and default-readiness question.
- **Outcome**: `SHIP` as opt-in, not default. Persistent workers preserved
  token hex `3136` and beat same-build serial at 1M/2 slots (`5.13` vs `3.85`
  generated tok/s) and 1M/4 slots (`7.94` vs `3.79`), but measured below
  Sprint 065's per-step worker path (`5.57` and `8.67`). The next sprint should
  profile persistent dispatch and handoff synchronization before defaulting or
  further extending this path.

### Sprint 067 - Async Pipeline Profiling And A/B Dispatch [complete]

- **Goal**: Add async pipeline timing counters and compare serial, persistent
  async, and per-step async in one binary.
- **Rationale**: Sprint 066 proved persistence was correct but slower than the
  prior async result; the next step was to measure the dispatch/control-plane
  difference rather than guessing.
- **Outcome**: `SHIP` as preferred opt-in. Per-step async measured `5.576155`
  generated tok/s at 1M/2 slots and `8.617368` at 1M/4 slots, beating
  persistent async by `7-9%` while preserving token hex `3136`. Timing counters
  show persistent global wakeups increase stage wait accumulation more than
  they save in thread setup. The bare `--async-pipeline-decode` flag now selects
  per-step; persistent remains available through `--async-pipeline-mode
  persistent`.

### Sprint 068 - Appliance Async Serving Profile [complete]

- **Goal**: Make the preferred async path available through the appliance
  launcher and deployment config rather than only the benchmark harness.
- **Rationale**: Sprint 067 selected per-step async as the best measured path,
  but practical use still required remembering a replay flag.
- **Outcome**: `SHIP`. Added `DS4_V100_ASYNC_PIPELINE_MODE`, an `auto` resolver,
  and practical 4-slot sequential deployment defaults. The V100 launcher smoke
  proved status reports `async_pipeline_mode=per-step` and generation returns
  token hex `3136` with async timing in the response.

### Sprint 069 - Appliance Launcher Soak Harness [complete]

- **Goal**: Turn the ad hoc launcher smoke into a reusable appliance soak
  harness and run the practical 4-slot profile through it.
- **Rationale**: Sprint 068 wired the config path, but practical use needs a
  repeatable operator-facing validation command.
- **Outcome**: `SHIP`. Added `tools/ds4-v100-appliance-soak.sh` and archived a
  V100 4-slot, 1M-context run through the launcher: `4/4` token matches,
  `async_pipeline_mode=per-step`, `7.518610` generated tok/s, and `7.048697`
  continuation tok/s.

### Sprint 070 - Persistent MTP Forward Runtime [complete]

- **Goal**: Move MTP forward scratch from per-draft allocation into the resident
  MTP forward object and expose enough counters to validate reuse.
- **Rationale**: True MTP commit needs a stateful forward runtime, and the
  project needed to know whether allocation churn was a material draft-time
  cost before committing to the next MTP speed lever.
- **Outcome**: `SHIP`. MTP serving now reuses resident scratch, reports
  scratch bytes and forward run count, and is explicitly one-slot while verify
  state is shared. The V100 serving smoke accepted `3/3` drafts with draft
  timing `4.800`, `4.560`, and `4.562 ms`, so the next lever is true one-slot
  commit rather than more allocation cleanup.

### Sprint 071 - Exact MTP Commit Serving [complete]

- **Goal**: Add an opt-in one-slot MTP commit mode that emits accepted drafts
  into the generation path after exact target verification.
- **Rationale**: Diagnostic verify proved MTP correctness, but practical
  speculative serving needs the target replay state to advance from committed
  MTP outputs.
- **Outcome**: `SHIP`. Added narrow one-slot replay feed/select hooks,
  `--mtp-serving commit`, commit counters in JSON/status/metrics, launcher
  support for `DS4_V100_MTP_SERVING=commit`, and V100 evidence that commit mode
  accepts `2/2` drafts, reports `mtp.committed=2`, and matches the verify
  baseline token sequence `[926, 1]`.

### Sprint 072 - MTP Commit Throughput Decision Gate [complete]

- **Goal**: Measure `off`, `verify`, and `commit` MTP serving modes with the
  same sustained decode fixture and decide whether exact commit should remain
  the next performance lever.
- **Rationale**: Sprint 071 proved safe state mutation, but exact verification
  still computes the target token. The project needed V100 throughput evidence
  before investing in recursive or skip-verify MTP.
- **Outcome**: `PIVOT`. The sustained benchmark now accepts MTP serving flags
  and reports MTP counters/timing. V100 evidence showed `commit` accepted and
  committed `4/4` measured drafts, but generated tok/s was `0.777308` versus
  `0.788607` for MTP off. Exact commit is correct but not throughput-positive,
  so the next sprint should return to stage/kernel throughput.

### Sprint 073 - Persistent Stage Pipeline Mailboxes [complete]

- **Goal**: Make persistent stage workers competitive by replacing the old
  global wakeup pattern with per-stage mailbox readiness.
- **Rationale**: Per-step async is the best practical path, but recreates
  workers each token-step. The old persistent path avoids that setup cost but
  regresses, so mailbox scheduling tests whether host synchronization was the
  material blocker.
- **Outcome**: `SHIP_DIAGNOSTIC`. Added `--async-pipeline-mode mailbox` across
  replay, benchmark, launcher, and soak tooling. V100 evidence proved
  correctness and showed mailbox improved old persistent at 1M/4 slots
  (`8.053284` vs `7.865004` generated tok/s), but still trailed per-step
  (`8.649395`). Keep appliance `auto` on per-step and pivot next to CUDA
  event/stream handoff, peer-copy overlap, or kernel-side execution work.

### Sprint 074 - Async Peer Handoff Probe [complete]

- **Goal**: Test whether queued HC peer-copy handoff improves the best per-step
  async pipeline without changing kernels.
- **Rationale**: Sprint 073 showed host mailbox scheduling alone is not enough,
  and handoff remains a visible cost in async timing.
- **Outcome**: `SHIP_OPT_IN`. Added `--async-handoff`,
  `DS4_V100_ASYNC_HANDOFF=1`, queued GPU tensor copy, and async scheduler
  handoff. V100 evidence showed correctness and a 1M/4-slot gain from
  `8.605744` to `8.738546` generated tok/s, but the `+1.543%` uplift is below
  the default-change rule. Keep opt-in and target explicit stream/event handoff
  or kernel-side work next.

### Sprint 075 - Output-Head Top-1 Fast Path [complete]

- **Goal**: Remove full-logit CPU readback from greedy `k == 1` output-head
  selection by adding persistent gpu7 output-head scratch plus a device top-1
  reducer.
- **Rationale**: The output-head path still allocated scratch per selection,
  copied all `129280` logits to the host, and scanned on CPU. A device-resident
  top-1 selector tested whether that synchronization/readback was a practical
  throughput blocker.
- **Outcome**: `SHIP_OPT_IN_ONLY`. The CUDA top-1 primitive and scratch reuse
  are correct, but the serial device reducer regressed output-head timing from
  `346.461 ms` to `423.818 ms` on the 1M/4-slot per-step fixture. Generated
  tok/s moved only from `8.659254` to `8.697510`, so the host-logit path stays
  default and the candidate remains opt-in with
  `DS4_V100_ENABLE_OUTPUT_HEAD_FASTPATH=1`.

### Sprint 076 - Parallel Output-Head Top-1 Reducer [complete]

- **Goal**: Convert the Sprint 075 output-head top-1 candidate from a serial
  CUDA scan into a deterministic parallel reducer and decide whether it should
  become the greedy output selector.
- **Rationale**: Sprint 075 proved the device-resident output-head path was
  correct but slow because it used one CUDA thread to scan `129280` logits.
  A real block-level reducer was the smallest way to test whether output-head
  readback/scan remained a practical throughput lever.
- **Outcome**: `SHIP_DEFAULT`. The new two-stage reducer preserves lower-token
  tie handling, carries a non-finite status flag, and keeps the public API
  stable. V100 evidence at 1M/4 slots improved generated tok/s from
  `8.656498` to `9.031197` and output-head timing from `324.953 ms` to
  `134.510 ms`, so greedy `k == 1` selection now defaults to device top-1.
  `DS4_V100_DISABLE_OUTPUT_HEAD_FASTPATH=1` remains the fallback.

### Sprint 077 - Batched Output-Head Selection [complete]

- **Goal**: Batch greedy output-head selection across active slots by running
  HC collapse, output norm, BF16 vocabulary projection, and top-1 as row-major
  slot batches on gpu7.
- **Rationale**: Sprint 076 made device top-1 fast enough to become the
  default, but replay still paid the output-head sequence once per active slot.
  Batching the output-head pass tested whether the remaining selector time was
  a useful multi-slot throughput lever.
- **Outcome**: `SHIP_OPT_IN_ONLY`. Added row-batched BF16 output projection,
  row-batched F32 top-1, scheduler batch scratch, and replay batch selection.
  V100 evidence proved correctness, but the 1M/4-slot fixture regressed
  generated tok/s from `9.028544` to `8.616841`, continuation tok/s from
  `8.464260` to `8.078288`, and output-head timing from `135.080 ms` to
  `139.750 ms`. Default serving stays on the Sprint 076 per-slot device top-1
  path; `DS4_V100_ENABLE_OUTPUT_HEAD_BATCH=1` is retained only for experiments.

### Sprint 078 - Event-Ordered Stage Handoff [complete]

- **Goal**: Replace the per-step async path's per-stage/slot device synchronize
  readiness gate with CUDA event record/wait ordering before HC peer copies.
- **Rationale**: Sprint 077 closed output-head as the next useful lever. The
  remaining async timing still showed handoff/sync cost, and an event-ordered
  path was the smallest way to test whether host-side readiness was limiting
  practical throughput.
- **Outcome**: `SHIP_OPT_IN_ONLY`. Added opaque CUDA event helpers, event-aware
  scheduler handoff, replay stage-ready events, `--async-event-handoff`, and
  deployment defaults. V100 evidence proved correctness and removed the explicit
  device-sync timing bucket, but generated tok/s improved only from `9.147418`
  to `9.158602` at 1M/4 slots. Keep `DS4_V100_ASYNC_EVENT_HANDOFF=1` opt-in and
  pivot next to routed MXFP4 occupancy or other kernel-side work.

### Sprint 079 - Routed MXFP4 Row-Pair Occupancy Probe [complete]

- **Goal**: Test whether computing two adjacent routed MXFP4 rows per CTA in
  grouped gate/up/SwiGLU and down-sum kernels improves practical V100
  utilization without changing source layout or public APIs.
- **Rationale**: Sprints 077 and 078 showed output-head batching and event
  handoff do not materially move end-to-end throughput. The routed MXFP4 expert
  path remains scalar row-reduction work with low GPU utilization, so a bounded
  row-pair kernel was the smallest kernel-side occupancy probe.
- **Outcome**: `SHIP_OPT_IN_ONLY`. Added
  `DS4_CUDA_MXFP4_ROUTE_ROWS2=1`, row-pair grouped routed MXFP4 kernels, and
  focused smoke coverage. V100 correctness passed, including selected token hex
  `3136`, but paired 1M/4-slot throughput regressed from `9.055694` to
  `9.035946` generated tok/s. The next useful kernel sprint needs a larger
  execution-shape change such as route/expert tiling, packed low-bit dot
  products, or a persistent grouped expert kernel.

### Sprint 080 - Copied tc-grid V100 INT8 Kernel Proof [complete]

- **Goal**: Copy candidate V100 low-bit kernel source from `~/repos/deepseek`
  into `ds4`, build it from this repository, and prove at least one standalone
  V100 smoke/bench.
- **Rationale**: The appliance should not rely on external working-tree kernel
  code or vague references to prior experiments. If tc-grid or TurboMind will
  inform the runtime, the relevant source needs to live here and be validated
  here.
- **Outcome**: `SHIP_PROOF_ONLY`. Copied the tc-grid headers and sm70 kernel
  headers into `kernels/tc-grid/`, added
  `tests/cuda_v100_tc_grid_int8_smoke.cu`, and built/ran the copied
  `v13_rf_v6` INT8 HMMA kernel on V100. Correctness passed with `max_abs=0`.
  Timing reached `46.391 TFLOP/s` on the large tc-grid reference shape but only
  `7.223 TFLOP/s` on `M=128,N=2048,K=4096`, so this is a useful proof and
  benchmark harness, not a default model path. The next copied-source kernel
  sprint should target TurboMind MXFP4 grouped GEMM because it better matches
  DS4's routed expert source layout and prior utilization evidence.

### Sprint 081 - Copied TurboMind MXFP4 Grouped GEMM Proof [complete]

- **Goal**: Copy the TurboMind C ABI wrapper and required lmdeploy
  `turbomind` support source into `ds4`, adapt the build to use the copied
  tree, and prove the grouped MXFP4 GEMM path on V100 for DS4 expert shapes.
- **Rationale**: Sprint 080 proved the copied-source workflow but also showed
  that INT8 tc-grid is better treated as a benchmark path unless we explicitly
  accept MXFP4-to-INT8 expansion. TurboMind is the better next hot-path target
  because it keeps DS4 routed experts in MXFP4-like source format and prior
  work indicated stronger V100 tensor-core utilization.
- **Outcome**: `SHIP_PROOF`. Copied TurboMind and required lmdeploy source
  under `kernels/turbomind/`, patched copied CMake defaults, built
  `libggml-turbomind.so` on V100, and ran grouped MXFP4 compare for DS4
  gate/up and down shapes. Correctness passed in all measured cases. Grouped
  down measured `1.23-1.26x` faster than six single-expert calls; grouped
  gate/up was roughly neutral to slightly slower at tiny token counts. The
  next implementation sprint should build the DS4 routed-expert adapter around
  this copied TurboMind source rather than continuing small scalar MXFP4 kernel
  tweaks.

### Sprint 082 - TurboMind Routed Expert Adapter Smoke [complete]

- **Goal**: Build a DS4 routed-expert adapter around the copied TurboMind C ABI
  and compare its output against the existing source-MXFP4 arena reference.
- **Rationale**: The grouped GEMM proof was not enough by itself; the runtime
  needs the exact DS4 boundary: source MXFP4 pack, selected route grouping,
  gate/up, SwiGLU, route weights, down projection, and route accumulation.
- **Outcome**: `SHIP_ADAPTER_SMOKE`. The V100 smoke matched the arena
  reference with `max_abs=0.00129318` and `rel=0.000258549` on real DS4 expert
  dimensions with a bounded expert count. The next step was to wire the same
  contract into the DS4 CUDA wrapper as an opt-in runtime path.

### Sprint 083 - Opt-In TurboMind Runtime Routed FFN Bridge [complete]

- **Goal**: Add an opt-in DS4 CUDA runtime branch that calls copied TurboMind
  for routed MXFP4 FFN execution while preserving the existing arena kernels as
  default and fallback.
- **Rationale**: This proves the scheduler-facing runtime boundary without
  committing to a memory-unsafe duplicate expert layout. The transient bridge
  packs one matrix family at a time so 32 GB V100 cards are not overfilled.
- **Outcome**: `SHIP_RUNTIME_BRIDGE`. `DS4_V100_TURBOMIND_ROUTED_FFN=1` now
  builds device-side expert offsets, gathers FP16 route rows, runs TurboMind
  grouped gate/up/down, applies DS4 SwiGLU/route weights, and scatters route
  sums back to F32 output. The V100 adapter smoke validates the wrapper with
  `max_abs=0.00129318`, `rel=0.000258549`, and `host_ms=43.298`. The bridge
  remains off by default because transient repacking is not a production
  throughput design; Sprint 084 should move TurboMind expert packs offline or
  add a planner-bounded persistent cache.

### Sprint 084 - Offline TurboMind Expert Sidecar Pack [complete]

- **Goal**: Add an offline conversion tool that derives TurboMind-ready expert
  sidecars from the normal DS4 V100 pack index and source GGUF.
- **Rationale**: The transient runtime bridge proves semantics but cannot be
  the performance path. Persistent packed experts need a separate acceleration
  artifact so runtime can avoid per-token repacking and the planner can account
  for memory explicitly.
- **Outcome**: `SHIP_SIDECAR_PACKER`. Added
  `tools/ds4-v100-turbomind-pack`, which reads real source MXFP4 expert bytes,
  packs them through copied TurboMind, and writes `gpuN.turbomind` plus
  `turbomind-pack-index.tsv`. V100 validation packed layer 0 gate/up/down with
  `2/256` experts each, `k_pack=0x341321`, and a `26,738,688` byte bounded
  sidecar. The next sprint should load a bounded sidecar into device memory,
  rebuild `StridedPtrH` tables, and run the adapter from persistent packed
  buffers instead of runtime repacking.

### Sprint 085 - Persistent TurboMind Sidecar Load [complete]

- **Goal**: Add a bounded loader and validation path for the offline
  TurboMind expert sidecar so packed experts can be uploaded once and used
  without decode-time repacking.
- **Rationale**: Sprint 084 produced the right acceleration artifact, but the
  runtime still needed proof that the derived offsets and strides can rebuild
  TurboMind's device pointer tables correctly.
- **Outcome**: `SHIP_PERSISTENT_SIDECAR_SMOKE`. Added
  `ds4_turbomind_pack.{h,c}` and
  `tests/cuda_v100_turbomind_sidecar_smoke`. The V100 smoke loads the bounded
  layer-0 `gpu0.turbomind` sidecar into one device buffer, reconstructs
  `StridedPtrH` tables, runs grouped gate/up/down from persistent packed
  buffers, and matches the source-MXFP4 arena reference with
  `max_abs=5.91128e-07`, `rel=0.000493098`, and `host_ms=0.265`. The next
  sprint should add memory admission and scheduler-side selection for resident
  TurboMind sidecars.

### Sprint 086 - TurboMind Sidecar VRAM Admission [complete]

- **Goal**: Add an admission report that prevents TurboMind sidecars from
  silently overfilling 32 GB V100s when moving beyond bounded smokes.
- **Rationale**: Persistent sidecars are the right performance path, but a full
  duplicate expert cache would be dangerous. The runtime needs explicit memory
  accounting before any broad scheduler enablement.
- **Outcome**: `SHIP_SIDECAR_ADMISSION`. Added
  `tools/ds4-v100-turbomind-admit`, which reads the source pack index and the
  TurboMind sidecar index, then reports source arena bytes, source expert
  payload, sidecar bytes, duplicate totals, and replacement-style totals per
  GPU. With 32 GiB VRAM, 4 GiB reserve, 1 GiB KV, and 1 GiB scratch, the
  bounded sidecar reports GPU0 duplicate total `27.002 GiB` and
  replacement-style total `7.877 GiB`. The key production signal is that GPU0
  already has `19.125 GiB` of source expert payload, so full sidecars should
  replace source experts or be admitted as a bounded cache.

### Sprint 087 - Single-Shard TurboMind Appliance Pack [complete]

- **Goal**: Move TurboMind-packed routed experts into the single appliance
  `gpuN.weights` layout and add a no-repack CUDA execution API for those
  resident spans.
- **Rationale**: Sidecar validation proved the packed expert format, but the
  deployed product should be one appliance directory, not a source pack plus a
  separate expert sidecar tree.
- **Outcome**: `SHIP_APPLIANCE_PACK_BOUNDARY`. Added
  `tools/ds4-v100-appliance-pack`, relaxed the TurboMind index parser to allow
  `gpuN.weights`, and added
  `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32`. Bounded V100
  validation generated `/tmp/ds4-appliance-pack-smoke/gpu0.weights` containing
  TurboMind-packed layer-0 experts, then validated both direct TurboMind and
  the DS4 no-repack CUDA API against the source-MXFP4 reference:
  `packed_api max_abs=5.91128e-07`, `rel=0.000493098`, `PASS`. The next sprint
  should wire scheduler context binding and arena upload from the appliance
  directory.

### Sprint 088 - Scheduler-Bound TurboMind Appliance Runtime [complete]

- **Goal**: Bind `turbomind-pack-index.tsv` into the scheduler/runtime path so
  the single appliance directory can drive routed expert execution.
- **Rationale**: Sprint 087 created the correct payload shape, but the runtime
  still needed explicit metadata lookup, arena sizing, shard upload, and layer
  dispatch for those prepacked spans.
- **Outcome**: `SHIP_RUNTIME_BINDING`. Added TurboMind descriptor lookup to the
  V100 context, layer-state bindings for prepacked routed experts, shard-offset
  model-map mode for appliance CPU-side control tensors, scheduler
  `shard_dir` loading for `gpuN.weights`, and layer-execute dispatch through
  `ds4_gpu_arena_turbomind_mxfp4_routed_swiglu_down_sum_f32`. Local validation
  passes `tests/v100_context_smoke`; V100 build validation passes for the
  stage/full scheduler smoke targets. V100 scheduler execution against a full
  appliance pack and throughput measurement are next.

### Sprint 089 - Appliance-Backed Scheduler Smoke [complete]

- **Goal**: Prove scheduler execution from an appliance directory on V100.
- **Rationale**: The appliance path was compiled and bound, but practical
  serving needs evidence that `gpuN.weights` can replace the source GGUF map
  during scheduler decode and still dispatch TurboMind experts.
- **Outcome**: `SHIP_APPLIANCE_SCHEDULER_SMOKE`. Added hybrid bounded
  appliance generation with `--only-gpu` and `--layer`, added
  `--appliance-dir` smoke support, added `turbomind_routed_layers_executed`
  reporting, and fixed appliance shard fd activation for CPU-side control
  tensors. V100 validation generated a stage-0 appliance with
  `gpu0.weights bytes=22524134668`; both stage and one-stage full scheduler
  smokes executed 6 layers with `tm_layers=1` and returned `ok`.

### Sprint 090 - Full Appliance Pack And Scheduler Run [complete]

- **Goal**: Generate a full 8-GPU appliance directory and run scheduler/replay
  from `gpuN.weights` shards instead of source GGUF residency.
- **Rationale**: Bounded stage-0 validation was not enough to prove the
  production artifact. The project needed exact shard sizes, full 43-layer
  execution, and first-token replay evidence from one appliance directory.
- **Outcome**: `SHIP_FULL_APPLIANCE_RUN`. Recreated the 8-GPU build pod with
  `/workspace` on `localpool/k8s-local`, generated
  `/workspace/ds4-appliance-full-tm-s090`, and validated:
  `source_rows=1199`, `tm_rows=129`, total size `142G`, largest shard
  `22524134668` bytes, scheduler `layers=43 tm_layers=43 ok`, and replay
  first token `3136` with `uploaded_tensors=8`,
  `uploaded_bytes=156142896212`, `generated_tokens_per_second=0.620997`, and
  `continuation_tokens_per_second=9.491896`.

### Sprint 091 - Appliance Directory Launcher Path [complete]

- **Goal**: Make the operator launcher and HTTP smoke use the full appliance
  directory created in Sprint 090.
- **Rationale**: A manual replay command is not enough for practical use. The
  service launcher needs a first-class appliance directory config so operators
  do not fall back to source-layout scheduler residency by accident.
- **Outcome**: `SHIP_APPLIANCE_LAUNCHER_PATH`. Added
  `DS4_V100_APPLIANCE_DIR` validation to the launcher, added
  `--appliance-dir` support to the HTTP smoke, updated the env example, and
  validated the served path on V100. The launcher print-command emits
  `--appliance-dir /workspace/ds4-appliance-full-tm-s090`, and the HTTP smoke
  returns first token `3136` with `uploaded_tensors=8`.

### Sprint 092 - Appliance Multi-Slot Async Soak [complete]

- **Goal**: Benchmark the full TurboMind appliance directory through the
  operator-facing service path under multi-slot async load.
- **Rationale**: Single-request correctness does not prove practical serving.
  The appliance needs a warm-started aggregate benchmark that uses the same
  launch contract operators will use.
- **Outcome**: `SHIP_APPLIANCE_ASYNC_SOAK`. Added `--appliance-dir` support to
  the soak harness, removed Python from the pod-side client, added a default
  warmup request, and validated a 4-slot/active-microbatch-4 run on the V100
  pod. The measured timed batch returned `token_match=4/4`,
  `generated_tokens=64`, `tensor_batched_groups=1`,
  `aggregate_generated_tokens_per_second=11.256048`, and
  `aggregate_continuation_tokens_per_second=10.552545`.

### Sprint 093 - Appliance Startup Warmup And GPU Profile [complete]

- **Goal**: Remove the cold concurrent first-request failure from the
  production appliance path and capture profiler-backed evidence for the next
  throughput optimization.
- **Rationale**: Client-side warmup is not a production contract, and kernel
  work should be chosen from actual GPU traces rather than inferred only from
  request timing.
- **Outcome**: `SHIP_STARTUP_WARMUP_PROFILE`. Added server-side
  `--startup-warmup`, launcher `DS4_V100_STARTUP_WARMUP=auto`, status/metrics
  exposure, and a CUDA profiler window for post-open decode profiling. The
  full appliance 4-slot soak passes with no client warmup:
  `token_match=4/4`, `generated_tokens=64`,
  `aggregate_generated_tokens_per_second=11.241074`. The decode-window
  profiler points next at F8 dense/projection matmul plus HtoD control traffic.

### Sprint 094 - Grouped TurboMind And Shared F8 Serving [complete]

- **Goal**: Convert measured TurboMind control churn and per-slot routed expert
  scheduling into production-path improvements.
- **Rationale**: The appliance was already using TurboMind kernels, but the
  multi-slot layer executor still called the routed expert path once per slot.
  That threw away the effective-M gain from active slots and kept avoidable
  control-table uploads in the decode window.
- **Outcome**: `SHIP_GROUPED_TM_SHARED_F8`. TurboMind packed pointer tables are
  cached per resident arena; the batched FFN executor now routes all active
  slots through one TurboMind grouped call per layer; and
  `DS4_V100_BATCH_SHARED_F8=1` is the launcher default. Cluster validation:
  `make tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke
  CUDA_ARCH=sm_70 -j8`, full scheduler smoke `--slots 4` passed, and the
  default 1M/4-slot appliance soak reports `token_match=4/4`,
  `generated_tokens=64`, `aggregate_generated_tokens_per_second=12.634955`,
  and `aggregate_continuation_tokens_per_second=11.845270`.

### Sprint 095 - Request Rendezvous And F8 Cache Probe [complete]

- **Goal**: Make multi-slot serving reliably coalesce concurrent requests and
  probe whether resident F8-to-F16 conversion helps the dense/shared path.
- **Rationale**: Split request batches hid the real 8-slot envelope, while the
  one-shot profile still pointed at F8 projection and copy overhead.
- **Outcome**: `SHIP_REQUEST_RENDEZVOUS`. Added
  `DS4_V100_MICROBATCH_WAIT_US=auto`, resolving to 50 ms for multi-slot
  serving. Correctness remains stable at `token_match=4/4` for 1M/4-slot and
  `token_match=8/8` for 256K/8-slot fixtures. The opt-in F8-to-F16 arena cache
  is correct but flat, so it stays experimental.

### Sprint 096 - Served Decode Profiling Window [complete]

- **Goal**: Profile the warmed HTTP appliance generation path rather than
  cold one-shot replay startup.
- **Rationale**: HtoD traffic in earlier profiles mixed startup cache loads
  with real served decode. Kernel work needed a profile window around timed
  generation only.
- **Outcome**: `SHIP_SERVED_PROFILE_WINDOW`. Added
  `DS4_V100_CUDA_PROFILER_WINDOW=1` and request-window
  `cudaProfilerStart/Stop` support. Served-path `nvprof` shows F8 arena matmul
  at `61.64%`, TurboMind at `20.15%`, HtoD at only `0.14%`, and CUDA API time
  dominated by allocator churn.

### Sprint 097 - CUDA Tensor Pool Default [complete]

- **Goal**: Remove allocator churn from the warmed multi-slot serving path.
- **Rationale**: Sprint 096 showed request-window API time dominated by
  repeated `cudaMalloc`/`cudaFree`, which does not change model math and should
  not be paid per token.
- **Outcome**: `SHIP_CUDA_TENSOR_POOL_DEFAULT`. Added a bounded per-device
  scratch tensor pool and launcher controls
  `DS4_V100_CUDA_TENSOR_POOL=auto|0|1` plus
  `DS4_V100_CUDA_TENSOR_POOL_MAX_MIB`. Multi-slot appliance configs now resolve
  `auto` to enabled. V100 validation reports `17.532887` generated tok/s at
  1M/4 slots and `25.232220` at 256K/8 slots, both with full token-match
  correctness. Warmed profiling removes `cudaMalloc` from the request window
  and reduces `cudaFree` to `9.18 ms`.

### Sprint 098 - Grouped F8 Attention Output [complete]

- **Goal**: Reduce F8 attention output launch count while preserving the
  existing row-wise source-format math.
- **Rationale**: Sprint 097 made allocator churn small enough that F8 arena
  matmul dominated the served profile. The attention output-A path still paid
  one F8 launch per output group per layer/slot.
- **Outcome**: `SHIP_GROUPED_F8_ATTN_OUTPUT`. Added a grouped F8 matmul helper
  for source views with grouped input slices, defaulted attention output-A to
  one grouped launch, and exposed
  `DS4_V100_DISABLE_GROUPED_ATTN_OUTPUT_A=1` as the rollback. V100 validation
  reports `17.904697` generated tok/s at 1M/4 slots and `26.206100` at
  256K/8 slots with token-match correctness. Served profiling reduces single
  F8 matmul calls from `11880` to `5544` and total CUDA kernel launches from
  `39684` to `34140`.

### Sprint 099 - Batch Attention Projection Probe [complete]

- **Goal**: Test whether projection-only batching across active slots moves the
  remaining F8 attention bottleneck.
- **Rationale**: After Sprint 098, Q-A, Q-B, and KV projection launches were
  the next obvious F8 batching candidate.
- **Outcome**: `KEEP_BATCH_ATTN_PROJ_OPT_IN`. Added reusable row-pointer table
  support and an explicit `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` probe. The path
  is correct, but same-binary V100 controls are flat/slightly faster without
  it: `17.742637` opt-in vs `17.764257` rollback at 1M/4 slots, and
  `26.128571` opt-in vs `26.149613` rollback at 256K/8 slots. Production
  default remains the Sprint 098 path; do not continue projection-only batching
  in this form.

### Sprint 100 - TurboMind Sync Readback A/B [complete]

- **Goal**: Reduce hot-path synchronization around packed TurboMind routed
  expert GEMMs without changing the appliance format or source-model math.
- **Rationale**: Sprint 098/099 profiles showed large CUDA API time around
  `cudaMemcpy` readbacks in the served request window.
- **Outcome**: `SHIP_ROUTE_VALIDATE_SYNC_OFF`. Added an optional
  `ggml_turbomind_mul_mat_grouped_total_tokens()` ABI and DS4 wrapper control,
  but kept it opt-in because V100 A/B showed the wait moves into existing
  device synchronizations and throughput regresses. Production now defaults to
  the measured faster combination: old TurboMind row-count ABI with route
  validation readback disabled
  (`DS4_V100_DISABLE_TURBOMIND_TOTAL_TOKENS=1`,
  `DS4_V100_TURBOMIND_ROUTE_VALIDATE_SYNC=0`). The production-default 8-slot
  256K soak reports `26.372672` generated tok/s with `token_match=8/8`.

### Sprint 101 - Batch Attention Projection Semantic Repair [complete]

- **Goal**: Repair the opt-in batch attention projection path so its semantics
  match the single-slot path before using it as a throughput candidate.
- **Rationale**: The Sprint 099 probe batched Q-A/Q-B/KV projections across
  active slots but skipped the attention RMS norm before projection and passed
  raw hidden rows into compressed-KV preparation. That made the path a poor
  optimization target even if the token fixture happened to pass.
- **Outcome**: `KEEP_BATCH_ATTN_PROJ_OPT_IN`. The batch path now normalizes
  each active slot before Q-A/KV projection, builds the projection pointer
  table from normalized rows, and passes the normalized row into
  `prepare_decode_cache_attention()`. It also reuses persistent attention batch
  scratch and rejects mixed shard/source offset model maps. V100 validation
  passed attention smoke, 4-slot and 8-slot full scheduler smokes, and the
  selected-token oracle. Throughput does not justify defaulting it:
  4-slot/1M regressed from `18.102742` to `17.503345` generated tok/s, while
  8-slot/256K moved only from `26.402101` to `26.432087`. The production
  default remains Sprint 100.

### Sprint 102 - F8 Row-Pair Kernel Shape Probe [complete]

- **Goal**: Change the F8 arena matmul execution shape broadly enough to move
  the warmed served decode bottleneck, while keeping a simple rollback.
- **Rationale**: F8 arena matmul remained the largest served decode GPU bucket.
  The batch-projection path alone was not enough; the next useful probe was a
  kernel-shape change applied across single, batch, pointer-table, and grouped
  F8 APIs.
- **Outcome**: `SHIP_F8_ROWPAIR_DEFAULT`. Added row-pair F8 kernels behind
  `DS4_CUDA_F8_ROWPAIR=1` and exposed the production knob as
  `DS4_V100_CUDA_F8_ROWPAIR=1`. V100 correctness passed projection attention,
  stage scheduler, full 8-slot scheduler, and selected-token oracle gates.
  Same-binary throughput improved from `26.447308` to `27.037514` generated
  tok/s at 256K/8 slots and from `17.821073` to `18.500281` at 1M/4 slots.
  Launcher-default validation reported `27.049799` generated tok/s with
  `token_match=8/8`, so row-pair is now the appliance default with rollback to
  `DS4_V100_CUDA_F8_ROWPAIR=0`.

### Sprint 103 - Exact-Bit F8 Decode [complete]

- **Goal**: Remove the remaining per-element `ldexpf()` cost from the dominant
  F8 arena decode/matmul path without changing source dtype or tensor layout.
- **Rationale**: The warmed appliance profile still pointed at F8 arena matmul
  after Sprint 102. Because E4M3 values and E8M0 scales are finite low-bit
  formats, the E4M3 value can be decoded by exact F32 exponent/mantissa bit
  construction instead of runtime `ldexpf()`.
- **Outcome**: `SHIP_EXACT_F8_DECODE`. The CUDA helper now constructs exact F32
  bits for normal and subnormal E4M3 values, preserving zero/NaN handling.
  V100 validation passed source dtype, projection attention, stage scheduler,
  full scheduler, appliance selected-token, and served throughput gates. The
  8-slot/256K production soak improved from `27.049799` to `30.862791`
  generated tok/s, and the 4-slot/1M soak improved from `18.500281` to
  `19.733742`, both with token-match correctness.

## Parking Lot

- See `docs/sprints/SPRINT-004-DEFERRED.md`: first source-format math probe,
  source-model decode correctness, MTP, production multi-GPU context, hidden
  context relay, layer scheduler, tensor-parallel variants, multi-slot
  scheduling, KV residency, JSON reports, upload optimization, and pack-only
  runtime boot.
- See `docs/sprints/SPRINT-004-FOLLOWUPS.md`: model-less default test target,
  direct CUDA arena unit target, and upload timing metrics.
- See `docs/sprints/SPRINT-005-DEFERRED.md`: HC expansion, device-output and
  stream-aware probe variants, source-layout embedding dtype cleanup, F16
  output, F32 control tensor probe, additional BF16 tensors, FP8/MXFP4 compute
  probes, and default model-less `make test` cleanup.
- See `docs/sprints/SPRINT-006-DEFERRED.md`: decode, KV population, real
  FP8/MXFP4/INT kernels, tensor-parallel exceptions, output-head math, MTP,
  serving/deployment, and host-backed or persistent dequantized runtime paths.
- See `docs/sprints/SPRINT-007-DEFERRED.md`: production V100 FP8/MXFP4 kernels,
  device-side oracle reads, prefill/KV, multi-slot scheduling, MTP, public
  CLI/server exposure, tensor-parallel exceptions, and full-logit oracle
  capture.
- See `docs/sprints/SPRINT-007-FOLLOWUPS.md`: MXFP4 parity hardening,
  official-vector automation, source-oracle guard tests, and Sprint 008
  correctness anchors.
- See `docs/sprints/SPRINT-009-FOLLOWUPS.md`: Sprint 010 integration work for
  production projection/compressor outputs, source-oracle comparison, explicit
  KV state subviews, and deployment sequencing.
- See `docs/sprints/SPRINT-010-FOLLOWUPS.md`: Sprint 011 blockers for real
  source-format projection, attention/layer output, router/expert execution,
  bounded logits/top-k comparison, and deployment re-sequencing.
- See `docs/sprints/SPRINT-011-DEFERRED.md` and
  `docs/sprints/SPRINT-011-FOLLOWUPS.md`: Sprint 012 blockers for coherent
  layer output, router/shared/routed expert correctness, output-head or
  selected-token comparison, production F8 projection kernels, and deployment
  re-sequencing.
- See `docs/sprints/SPRINT-014-FOLLOWUPS.md`: runtime descriptor table,
  descriptor-bound layer compute, layer-class descriptor coverage, shared
  expert execution, and readiness-policy cleanup.
- See `docs/sprints/SPRINT-015-FOLLOWUPS.md`: real router scheduling,
  descriptor-bound layer state, attention/residual/norm integration,
  selected-token real-model gate, and production memory reuse.
- See `docs/sprints/SPRINT-016-FOLLOWUPS.md`: scheduler-owned layer state,
  attention/residual/norm integration, real-model selected-token gate,
  production arena reuse, and representative router coverage.
- See `docs/sprints/SPRINT-017-FOLLOWUPS.md`: descriptor-bound attention,
  residual/norm/HC layer slice, real-model selected-token gate, production
  arena reuse, and bias-router coverage.
- See `docs/sprints/SPRINT-018-FOLLOWUPS.md`: full attention softmax over
  raw/compressed KV, combined attention plus FFN layer slice, real-model
  selected-token gate, and production arena reuse.
- See `docs/sprints/SPRINT-019-FOLLOWUPS.md`: compressor/indexer descriptor
  binding, HC pre/post layer scheduling, full 43-layer selected-token gate,
  production arena reuse, and timing/throughput counters.
- See `docs/sprints/SPRINT-021-FOLLOWUPS.md`: full 43-layer single-slot
  scheduler, production indexer-threshold stress, reusable scratch/timing
  counters, HC CPU reference, serving, MTP, and multi-slot throughput.
- See `docs/sprints/SPRINT-024-FOLLOWUPS.md`: output-head selected-token gate,
  full-chain failure-local reports, per-stage upload/memory timing, relay
  optimization, MTP, and throughput.
- See `docs/sprints/SPRINT-025-FOLLOWUPS.md`: selected-token divergence
  localization, top-k diagnostics, output-head CPU parity, prompt replay
  counters, and failure-preserving logs.
- See `docs/sprints/SPRINT-026-FOLLOWUPS.md`: stage/layer HC divergence
  checkpoints, per-layer execution reports, full-gate build guards, parallel
  resident uploads, and continued deferral of serving/MTP/throughput.
- See `docs/sprints/SPRINT-027-FOLLOWUPS.md`: public one-slot serving,
  throughput counters, layer-4 FFN numeric drift, explicit FP8 KV validation,
  and continued deferral of MTP/multi-slot scheduling until the single-slot
  baseline is usable.
- See `docs/sprints/SPRINT-028-FOLLOWUPS.md`: HTTP/process serving around the
  replay runtime, scheduler reset or single-session semantics, open/upload
  reduction, longer decode baselines, and continued MTP/multi-slot deferral.
- See `docs/sprints/SPRINT-029-FOLLOWUPS.md`: MTP implementation/validation,
  parallel resident stage open/upload, longer resident decode baselines,
  serving API hardening, and continued multi-slot deferral.
- See `docs/sprints/SPRINT-030-FOLLOWUPS.md`: K=1 MTP runtime parity, keeping
  Q4_K MTP separate from the MXFP4 path, MTP memory planner updates, parallel
  upload, longer decode baselines, and continued speculative-serving/multi-slot
  deferral.
- See `docs/sprints/SPRINT-031-FOLLOWUPS.md`: K=1 MTP forward probe, Q4_K
  routed expert tests, gpu7 scheduler HC access, explicit draft/verify state,
  and continued separation between resident MTP sidecar tensors and the main
  source-layout pack path.
- See `docs/sprints/SPRINT-032-FOLLOWUPS.md`: K=1 MTP forward remains the next
  runtime slice, while base-appliance hardening should focus on clearer status
  counters, startup/upload reduction, longer resident decode baselines, and
  production packaging only after request-state ownership is explicit.
- See `docs/sprints/SPRINT-033-FOLLOWUPS.md`: resident MTP F32 prefix norms and
  HC composition, resident Q4_K MTP routed experts, high-offset mmap cache
  hardening, and full MTP logits/top-k plus draft/verify/rollback tests.
- See `docs/sprints/SPRINT-034-FOLLOWUPS.md`: resident MTP Q4_K routed
  experts, dense MTP block execution, MTP logits/top-k plus draft rollback, and
  CUDA model-map cache hardening for sidecar tensor-local copies.
- See `docs/sprints/SPRINT-035-FOLLOWUPS.md`: resident MTP FFN block
  execution, MTP attention/raw cache, MTP logits/top-k plus draft rollback, and
  batched Q4_K scheduling after correctness.
- See `docs/sprints/SPRINT-036-FOLLOWUPS.md`: MTP raw/SWA attention, MTP
  logits/top-k parity, draft verify/rollback semantics, and MTP FFN
  fusion/benchmarking after correctness.
- See `docs/sprints/SPRINT-037-FOLLOWUPS.md`: integrated MTP attention
  projection/output, MTP logits/top-k parity, draft verify/rollback semantics,
  and replacing synthetic attention inputs with a native-prefix integrated
  smoke.
- See `docs/sprints/SPRINT-038-FOLLOWUPS.md`: MTP logits/top-k parity, full
  one-token resident MTP block composition, draft verify/rollback semantics,
  and sharper grouped-output tolerance calibration.
- See `docs/sprints/SPRINT-039-FOLLOWUPS.md`: full one-token resident MTP block
  composition, draft verify/rollback semantics, reusable output-head runtime
  binding, and MTP candidate margin diagnostics.
- See `docs/sprints/SPRINT-040-FOLLOWUPS.md`: MTP draft verify/rollback,
  native prompt-token MTP mode, reusable MTP composition extraction, and
  candidate-margin diagnostics.
- See `docs/sprints/SPRINT-041-FOLLOWUPS.md`: native prompt-token MTP verify,
  production snapshot boundary extraction, ratio-128 compressor rollback
  stress, and cleanup of verify/rollback tool naming.
- See `docs/sprints/SPRINT-042-FOLLOWUPS.md`: production MTP state object,
  positive-accept fixture set, and output-head arena reuse. The P0 production
  deployment package is complete through Sprint 043.
- See `docs/sprints/SPRINT-043-FOLLOWUPS.md`: throughput optimization and
  operating envelope, production MTP serving object, request surface hardening,
  and deployment manifest hardening.
- See `docs/sprints/SPRINT-044-FOLLOWUPS.md`: MTP speculative serving,
  multi-slot aggregate throughput, persistent startup strategy, and broader
  benchmark coverage.
- See `docs/sprints/SPRINT-045-FOLLOWUPS.md`: aggregate slot/context envelope,
  true MTP commit path, MTP service hardening, and startup/upload refinement.
- See `docs/sprints/SPRINT-047-FOLLOWUPS.md`: request-loop active microbatch
  integration, slot-tier throughput evidence, and tensor-batched kernel uplift.
- See `docs/sprints/SPRINT-051-FOLLOWUPS.md`: aggregate throughput regression
  thresholds and continuation decode coverage.
- See `docs/sprints/SPRINT-055-FOLLOWUPS.md` and
  `docs/sprints/SPRINT-056-FOLLOWUPS.md`: deterministic token-step coalescing,
  batched layer execution across active slots, and persistent or tensor-core
  friendly MoE kernels.
- See `docs/sprints/SPRINT-057-FOLLOWUPS.md`: persistent grouped MoE kernels,
  removing copy overhead from the opt-in batch slice, batch-level timing
  counter semantics, and a configurable rendezvous window.
- See `docs/sprints/SPRINT-058-FOLLOWUPS.md`: persistent or copy-free MoE
  batching, batch FFN scratch reuse, and batch-level timing semantics.
- See `docs/sprints/SPRINT-059-FOLLOWUPS.md`: pointer-input or gather-based
  FFN batching, persistent batch views, and higher slot-tier retesting.
- See `docs/sprints/SPRINT-060-FOLLOWUPS.md`: shared expert batching,
  persistent batch views, and higher slot-tier retesting.
- See `docs/sprints/SPRINT-066-FOLLOWUPS.md`: persistent async dispatch and
  handoff profiling, plus the decision to keep async opt-in until the Sprint
  065-to-066 regression is understood or resolved.
- See `docs/sprints/SPRINT-067-FOLLOWUPS.md`: wiring preferred per-step async
  into the appliance deployment path and replacing persistent global broadcasts
  before retrying persistent workers.
- See `docs/sprints/SPRINT-001-DEFERRED.md`: q2/q4 fallback, SSD/host-backed
  offload, INT8 default-layout questions, F8 KV mode, and broad TurboMind or
  tc-grid kernel import as conditional paths rather than default strategy.
- See `docs/sprints/SPRINT-001-FOLLOWUPS.md`,
  `docs/sprints/SPRINT-002-FOLLOWUPS.md`, and
  `docs/sprints/SPRINT-003-FOLLOWUPS.md`: earlier follow-ups that are now
  either completed by Sprints 002-004 or carried forward in the Sprint 004
  deferred/follow-up lists.

## Pivot Log

| Date | What Changed | Why | Sprints Affected |
|------|-------------|-----|-----------------|
| 2026-05-17 | Created the first DS4 V100 appliance vision after Sprint 004 residency shipped. | The project has moved from feasibility and pack-residency proof to source-format compute, correctness, deployment, and performance sequencing. | Sprint 005+ |
| 2026-05-17 | Refined Sprint 005 from a generic source-format compute probe to a BF16 resident row-gather probe on `token_embd.weight`. | Planning consensus found BF16 embedding is the smallest useful proof of arena-resident compute and avoids premature FP8/MXFP4, scheduler, or decode work. | Sprint 005-006 |
| 2026-05-17 | Corrected Sprint 005 language from BF16 compute to BF16 gather/expand and shipped the probe. | V100 has no native BF16 tensor-core execution; the useful proof is resident addressing and exact dtype expansion, while production compute must target FP16 or low-bit/integer kernels. | Sprint 005-006 |
| 2026-05-17 | Scoped Sprint 006 to sidecar V100 context, fail-closed execution policy, HC relay, memory reserve, and no-math layer skeleton. | The next risk is not another dtype probe; it is proving the appliance runtime topology without silently promoting BF16/FP8/FP4 to unsupported native V100 compute or defaulting the model to FP32 GEMMs. | Sprint 006-007 |
| 2026-05-17 | Shipped Sprint 006 and moved the next milestone to single-slot decode correctness. | The 8-GPU context, descriptor policy, peer topology, memory reserve, and HC relay contract are now verified; the next unknown is numerical correctness through actual attention, MoE, KV, and output-head math. | Sprint 007+ |
| 2026-05-18 | Refined Sprint 007 into a guarded source-layout oracle sprint. | Planning consensus found that exact FP8/MXFP4/BF16 source semantics and a narrow CPU-only diagnostic unlock are the right next gate before production V100 kernels, prefill, or deployment. | Sprint 007-008 |
| 2026-05-18 | Shipped Sprint 007 source-layout oracle, corrected MXFP4 row ordering, and restored F16 KV as the source correctness baseline. | The official vector exposed both a bad interleaved MXFP4 assumption and an unsafe forced FP8 KV round-trip; matching GGML's low-half/high-half layout and the default F16 KV contract produced the expected first token and gives Sprint 008 a real correctness anchor. | Sprint 007-008 |
| 2026-05-18 | Re-scoped Sprint 008 as oracle automation, F16 KV admission, and one CUDA source-format anchor. | Full V100 source-layout prefill combines too many unproven contracts; making oracle, guard, memory, and source-format device checks executable first reduces risk before runtime KV execution. | Sprint 008-010 |
| 2026-05-18 | Shipped Sprint 008 source oracle harness, F16 KV admission, source dtype hardening, and CUDA F8 source-format anchor. | The project now has executable correctness, memory-admission, and first device source-format contracts for the Sprint 009 V100 prefill/KV implementation. | Sprint 008-009 |
| 2026-05-18 | Shipped Sprint 009 bounded V100 prefill/KV execution and inserted a single-slot decode integration sprint before deployment. | KV arena allocation, source-layout guards, and CUDA ratio-class row/state updates now pass on V100 `sm_70`; the next risk is real projection/compressor integration and oracle comparison, not server packaging. | Sprint 009-011 |
| 2026-05-18 | Shipped Sprint 010 stage-owned KV views/updates and real compressor recurrence smokes, then moved deployment behind a logits-producing V100 source-layout gate. | The project now trusts per-layer KV/state ownership and compressor recurrence on V100, but serving still lacks real source-format dense projection, MoE, output-head logits, and selected-token correctness. | Sprint 010-012 |
| 2026-05-18 | Split Sprint 011 into a source projection and attention-slice gate before the full logits gate. | Planning showed the next concrete risk is source FP8/BF16 projection on V100; full logits remain too broad until projection and bounded attention/compressor slices are trusted. | Sprint 011-013 |
| 2026-05-18 | Shipped Sprint 011 source projection and attention/compressor slice, keeping deployment behind Sprint 012's logits gate. | V100 now has device-resident source-F8 projection diagnostics, BF16/F32 policy checks, projection-fed ratio-4/ratio-128 attention/compressor smokes, and device-row KV writes, but still lacks MoE, output head, and selected-token correctness. | Sprint 012-013 |
| 2026-05-18 | Shipped Sprint 014 real pack-index descriptor validation and moved Sprint 015 to descriptor-bound layer compute. | The appliance gate now proves the real layer-2 descriptor contract on the V100 pod, so the next risk is converting those descriptors into runtime bindings that launch compute on real resident shard bytes. | Sprint 015+ |
| 2026-05-18 | Shipped Sprint 015 descriptor-bound FFN compute from real source bytes and moved Sprint 016 to scheduler-owned layer slicing. | The appliance now proves real pack offsets can feed routed MXFP4 and shared F8 FFN compute on V100; the next blocker is real router/layer state and attention/residual/norm integration. | Sprint 016+ |
| 2026-05-18 | Shipped Sprint 016 descriptor-bound real router FFN compute and moved Sprint 017 to scheduler-owned layer state. | The appliance now proves real layer-2 router logits, hash-router selected experts, all six routed MXFP4 experts, and shared F8 FFN compute on V100; the next blocker is moving this out of a standalone smoke into a scheduler-owned runtime layer surface. | Sprint 017+ |
| 2026-05-18 | Shipped Sprint 017 scheduler-owned layer state and moved Sprint 018 to descriptor-bound attention/layer output. | Router/FFN descriptor ownership is now a reusable runtime surface instead of test-local glue; the next blocker is producing a coherent hidden state through attention, residual, norm, and HC composition. | Sprint 018+ |
| 2026-05-18 | Shipped Sprint 018 descriptor-bound attention projection/residual/norm and moved Sprint 019 to full attention/layer output. | Real source-byte q/kv/output projection, residual add, and FFN pre-norm now pass on V100 through layer state; the next blocker is semantic attention softmax over raw/compressed KV and a coherent next hidden state. | Sprint 019+ |
| 2026-05-18 | Shipped Sprint 019 integrated hidden-vector layer executor and moved Sprint 020 to compressor/indexer plus HC scheduling. | Layer 2 now produces a bounded next-hidden vector through semantic raw/compressed attention inputs and real router-selected FFN on V100; the remaining blocker is generating those compressed rows from real descriptors and running the true HC-state layer scheduler. | Sprint 020+ |
| 2026-05-18 | Extended Sprint 020 with compressor/indexer descriptor binding and a V100 HC-state layer entrypoint. | The runtime now has the true `[4 x 4096]` HC layer surface and real compressor/indexer descriptor ownership, but still needs executor-owned compressed-row generation before selected-token decode. | Sprint 021+ |
| 2026-05-18 | Shipped Sprint 021 executor-owned compressor/indexer decode rows and indexed ratio-4 attention. | The representative layer now owns raw/compressed/indexer cache mutation from real descriptors on V100; the next blocker is wiring all 43 layers into a single-slot selected-token scheduler. | Sprint 022+ |
| 2026-05-18 | Shipped Sprint 022 bias-router execution and a resident stage-0 scheduler. | The runtime now walks layers 0-5 from resident gpu0 pack bytes and validates both router families on V100; the next blocker is cross-GPU HC relay through all stages and output-head selected-token comparison. | Sprint 023+ |
| 2026-05-18 | Shipped Sprint 023 cross-GPU two-stage scheduling. | The runtime now executes layers 0-11 across gpu0 and gpu1 with a peer HC handoff and device-local CUDA model caches; the next blocker is generalizing the stage chain through gpu7. | Sprint 024+ |
| 2026-05-18 | Shipped Sprint 024 full 8-stage scheduling. | The runtime now executes all 43 layers across the 8x V100 body and removes `full_43_layer_scheduler` from gate readiness; the next blocker is output-head selected-token comparison against the source oracle. | Sprint 025+ |
| 2026-05-18 | Extended Sprint 025 with scheduler-owned output-head selected-token execution. | The output-head path now runs and produces finite logits/top-1 on V100, but official-vector comparison fails (`3136` expected, `0a0a` selected), so the next blocker is divergence localization rather than more scheduling structure. | Sprint 026+ |
| 2026-05-18 | Shipped Sprint 026 output-head divergence localization. | Deterministic CPU-vs-V100 output-head parity passes on gpu7, so the selected-token mismatch is now scoped to the 43-layer scheduler body; the next sprint should checkpoint HC after layer/stage boundaries. | Sprint 027+ |
| 2026-05-18 | Shipped Sprint 027 selected-token correctness and HC checkpoint diagnostics. | BF16 embedding decode and F16 KV/cache semantics now match the CPU source-layout oracle closely enough for the official V100 selected-token gate to pass; the next milestone moves from correctness blocking to public serving and measurement. | Sprint 028+ |
| 2026-05-18 | Shipped Sprint 028 V100 replay runtime and timing tool. | The working scheduler path is now callable outside a smoke test and emits machine-readable token/timing/memory evidence; the next milestone is keeping that runtime resident behind an HTTP or process-serving surface. | Sprint 029+ |
| 2026-05-18 | Shipped Sprint 029 resident HTTP appliance smoke. | The one-slot selected-token path is now served through a resident loopback process and `public_serving` is no longer a gate blocker; the next milestone is MTP correctness and then performance work such as parallel upload and longer resident decode baselines. | Sprint 030+ |
| 2026-05-18 | Shipped Sprint 030 MTP sidecar readiness gate. | The appliance now validates the real MTP companion GGUF and keeps baseline serving green; the remaining blocker is no longer sidecar format uncertainty but the concrete V100 MTP runtime forward/verify path. | Sprint 031+ |
| 2026-05-18 | Shipped Sprint 031 MTP resident sidecar bridge. | The appliance now uploads the real 3.807600108 GB MTP sidecar into gpu7 device memory and spot-checks residency; the remaining blocker is the K=1 MTP forward/draft path, not sidecar loading or memory fit. | Sprint 032+ |
| 2026-05-18 | Added the readiness ladder. | The vision now distinguishes base correctness, minimal usability, MTP-assisted correctness, throughput, and production deployment so `ready=false` has an actionable meaning. | Sprint 032+ |
| 2026-05-18 | Shipped Sprint 032 Level 2 base appliance usability. | The one-slot base service now has health/status, repeated two-token HTTP smoke evidence, an operator runbook, and a full gate with `missing=mtp_forward` only; the next sprint can return to MTP forward without hiding base usability gaps. | Sprint 033+ |
| 2026-05-18 | Shipped Sprint 033 resident MTP Q8_0 projection parity. | The MTP sidecar now feeds real V100 Q8_0 projection kernels from compact gpu7 resident offsets, proving the first compute step beyond residency; the next blocker is full prefix composition, MTP block execution, logits/top-k, and draft verification. | Sprint 034+ |
| 2026-05-19 | Shipped Sprint 034 resident MTP prefix composition. | The MTP sidecar now produces resident `mtp_input_hc` from F32 norms, Q8_0 projections, HC repeat, and add; the next blocker is executing the MTP block itself, especially Q4_K routed experts, then logits/top-k and draft rollback. | Sprint 035+ |
| 2026-05-19 | Shipped Sprint 035 resident MTP Q4_K routed experts. | The MTP sidecar now runs its three 1.207959552 GB Q4_K expert tensors directly from gpu7 resident offsets with V100 decode kernels; the remaining blocker is assembling the full MTP block, logits/top-k, and draft verify/rollback. | Sprint 036+ |
| 2026-05-19 | Shipped Sprint 036 resident MTP FFN slice. | The MTP sidecar now runs HC FFN control, bias router, routed Q4_K experts, shared Q8_0 experts, routed+shared accumulation, and HC expansion through `next_hc` from gpu7 resident bytes; the remaining blocker is MTP raw/SWA attention, logits/top-k, and draft verify/rollback. | Sprint 037+ |
| 2026-05-19 | Shipped Sprint 037 resident MTP raw attention. | The MTP sidecar now feeds resident attention sinks into the V100 attention decoder and validates production raw KV store plus 128-row MTP cache wrap semantics; the remaining blocker is integrated MTP attention projection/output, logits/top-k, and draft verify/rollback. | Sprint 038+ |
| 2026-05-19 | Shipped Sprint 038 resident MTP integrated attention slice. | The MTP sidecar now composes HC attention control, Q/KV projections, raw attention, grouped output projection, and HC expansion from real gpu7 resident bytes; the remaining blocker is MTP logits/top-k and draft verify/rollback. | Sprint 039+ |
| 2026-05-19 | Shipped Sprint 039 resident MTP logits/top-k parity. | The MTP sidecar now produces deterministic draft logits/top-k candidates through the MTP output head and base BF16 vocabulary head with exact top-5 CPU/GPU parity; the remaining blocker is full one-token MTP forward composition plus draft verify/rollback. | Sprint 040+ |
| 2026-05-19 | Shipped Sprint 040 resident one-token MTP forward composition. | The MTP sidecar now runs deterministic prefix, attention, FFN, output HC collapse, output norm, base BF16 vocabulary projection, and top-k as one continuous gpu7 path; the remaining blocker is native prompt-token draft verify/rollback and state safety. | Sprint 041+ |
| 2026-05-19 | Shipped Sprint 041 MTP rollback state safety. | The base target scheduler can now snapshot/restore mutable HC/KV/compressor/indexer state around a rejected speculative token while the real MTP sidecar remains resident; the remaining blocker is a real prompt-token MTP draft verified by exact target token equality. | Sprint 042+ |
| 2026-05-19 | Shipped Sprint 042 native prompt-token MTP verify. | The real MTP sidecar now drafts from committed-token embedding plus gpu7 post-commit target HC and matches target top-1 exactly on the fixture, so the remaining gate blocker is production deployment packaging. | Sprint 043+ |
| 2026-05-19 | Shipped Sprint 043 production deployment package. | The appliance now has a launcher, explicit config, supervisor templates, metrics/status probes, runbook, rollback mode, and a full-gate production deployment smoke; the remaining blocker is no longer deployment packaging but throughput optimization and operating-envelope measurement. | Sprint 044+ |
| 2026-05-19 | Shipped Sprint 044 throughput optimization. | Parallel stage open/upload cut cold replay startup to about one minute while preserving selected-token correctness; the remaining blocker is now exposing the already-gated MTP verify path as production speculative serving. | Sprint 045+ |
| 2026-05-19 | Shipped Sprint 045 production MTP verify serving. | The resident HTTP appliance now exposes explicit MTP verify diagnostics and metrics while preserving base rollback mode; the gate now advances to the aggregate slot/context envelope. | Sprint 046+ |
| 2026-05-19 | Shipped Sprint 046 slot/context admission envelope. | The runtime now has planner-driven admission tiers, queue/reject policy surfaces, and explicit slot/context gate evidence; the next blocker moved from admission policy to real active microbatch scheduler execution. | Sprint 047+ |
| 2026-05-19 | Shipped Sprint 047 active-microbatch scheduler core. | Stage schedulers now own per-slot KV/HC state and batch decode/handoff APIs with 2-slot gate evidence; the remaining blocker is request-loop integration for concurrent prompts and throughput evidence at higher slot/context tiers. | Sprint 048+ |
| 2026-05-19 | Shipped Sprint 048 request-loop active microbatch integration. | The HTTP appliance now batches pending non-MTP one-token requests through scheduler batch APIs while preserving queue/admission semantics; the remaining blocker is cluster throughput and latency evidence across slot/context tiers. | Sprint 049+ |
| 2026-05-19 | Shipped Sprint 049 aggregate throughput evidence. | The project now has cluster-backed latency/tok/s evidence across 1/2/4/8-slot 256K tiers, 1M extremes, and focused MTP on/off comparison, with a dedicated throughput harness and gate rung; remaining Level-6 work is 128K/512K load evidence, multi-token token-step batching, and full-gate `ready=true` proof. | Sprint 050+ |
| 2026-05-19 | Shipped Sprint 050 readiness closure. | Gate hardening fixes removed planner-arch mismatch, slot-context CLI mismatch, and lock-file collisions; the full 8-GPU gate now passes all rungs and reports `ready=true`. | Post-vision optimization |
| 2026-05-19 | Shipped Sprint 051 aggregate profile expansion plus full-profile cluster execution. | The full gate now has explicit `fast` and `full` aggregate throughput profiles with CLI overrides, and the 32-case full-profile matrix was executed on `gpu-01` with `ready=true` and archived artifacts. | Post-vision optimization |
| 2026-05-19 | Reframed post-readiness work around practical-use optimization. | The current low tok/s and low GPU utilization are explained by one-token benchmark shape, first-token-only batching, per-request reset/prompt replay, diagnostic MTP verify, and non-persistent grouped expert execution; the next roadmap should optimize sustained decode before using 1k+ tok/s as a target. | Sprint 052+ |
| 2026-05-19 | Shipped Sprint 052 sustained decode baseline. | The project now has a sustained multi-token benchmark with GPU utilization sampling and first cluster evidence: 1M context, one slot, 16 tokens/request, 3.304551 generated tok/s, 3.098017 continuation tok/s, and 10.804% average GPU utilization. The next blocker is continuous token-step batching. | Sprint 053+ |
| 2026-05-19 | Shipped Sprint 053 continuous token-step microbatching. | Same-length non-MTP HTTP batches now advance through the multi-token replay batch API and expose tensor-batch counters. The V100 run proved one two-request / 32-token batch at 1M context, but throughput rose only about 2.4% and GPU utilization stayed near 11%, moving the next blocker to hot-path low-bit kernels and persistent expert scheduling. | Sprint 054+ |
| 2026-05-19 | Shipped Sprint 054 fused MXFP4 routed gate/up/SwiGLU. | The first main-path source-MXFP4 fusion preserved selected-token correctness and improved sustained generated tok/s by roughly 3%, but utilization stayed near 11%. The next blocker is still launch/occupancy in the routed expert path, especially down projection, route accumulation, and grouping all selected routes. | Sprint 055+ |
| 2026-05-19 | Shipped Sprint 055 fused MXFP4 routed down accumulation. | Down projection and route accumulation are now fused in the main routed expert path, preserving selected-token correctness and adding a sub-1% sustained speedup. This is a useful diminishing-returns signal: the next sprint needs grouped route execution or layer-executor batch kernels. | Sprint 056+ |
| 2026-05-19 | Shipped Sprint 056 grouped selected-route MXFP4 execution. | The main routed FFN path now groups all selected routes and improves sustained generated tok/s by about 4-5% over Sprint 055, but GPU utilization remains near 11% and two-slot benchmark coalescing was not deterministic. The next sprint should expose active slots to the layer executor instead of continuing single-slot route fusion. | Sprint 057+ |
| 2026-05-19 | Shipped Sprint 057 deterministic token-step coalescing. | The default server now reliably forms two-slot token-step batches, but throughput stayed flat and the first batched FFN layer slice regressed when enabled. The next sprint should focus on persistent/copy-free MoE batching rather than queue plumbing. | Sprint 058+ |
| 2026-05-19 | Shipped Sprint 058 replay router readback suppression. | The appliance hot path no longer reads router-selected experts and weights back to CPU for generation, improving two-slot generated tok/s to `3.704572` while preserving token hex `3136`; utilization remains near `11%`, so the next sprint must attack persistent/copy-free MoE batching. | Sprint 059+ |
| 2026-05-19 | Shipped Sprint 059 persistent layer batch scratch. | Multi-slot layer batching is now default after scratch reuse lifted two-slot generated tok/s to `3.862932`; the next bottleneck is the remaining FFN input copy and pointer-hostile routed MXFP4 batch interface. | Sprint 060+ |
| 2026-05-19 | Shipped Sprint 060 pointer-input routed FFN batch. | The routed MXFP4 batch primitive now consumes per-slot input tensor pointers directly, lifting two-slot generated tok/s to `3.915266`; utilization remains low, so the next sprint should target shared expert batching or higher-slot scaling evidence. | Sprint 061+ |
| 2026-05-19 | Shipped Sprint 061 shared F8 batch and higher-slot evidence. | Shared F8 batching is correct but remains opt-in because it measured below the Sprint 060 default; persistent FFN output views remove minor allocation churn, and the 4-slot 256K run proves slots alone do not improve aggregate tok/s under the current layer-synchronous schedule. | Sprint 062+ |
| 2026-05-20 | Shipped Sprint 064 opt-in served wavefront decode. | The served wavefront path is correct, but the paired V100 serial control is faster by about 4% across 1M/256K and 2/4-slot cases. The next sprint should not continue the same single-threaded diagonal schedule; practical throughput needs true asynchronous stage workers, MTP commit, or persistent low-bit kernels. | Sprint 065+ |
| 2026-05-20 | Shipped Sprint 065 opt-in async stage pipeline. | True per-stage host workers raised sustained generated tok/s to `5.57` at 1M/2 slots and `8.67` at 1M/4 slots while preserving correctness. The next sprint should make the stage workers persistent and retest whether the async path is ready to become the default. | Sprint 066+ |
| 2026-05-20 | Shipped Sprint 066 persistent async workers. | Replay-owned persistent workers preserve correctness and stay faster than serial, but measured below the Sprint 065 per-step worker path. The next sprint should profile dispatch/handoff synchronization before enabling async by default. | Sprint 067+ |
| 2026-05-20 | Shipped Sprint 067 async A/B dispatch profiling. | Same-binary V100 evidence confirms per-step async is the preferred opt-in path and explains the persistent-worker regression as global wakeup/wait accumulation. The next sprint should wire the preferred mode into the appliance deployment path. | Sprint 068+ |
| 2026-05-20 | Shipped Sprint 068 appliance async serving profile. | The measured per-step async path is now selected by `DS4_V100_ASYNC_PIPELINE_MODE=auto` for multi-slot appliance deployments, and the launcher smoke proves status plus generation correctness. The next blocker is a longer launched-appliance soak and the next throughput lever: MTP commit or stream/event handoff. | Sprint 069+ |
| 2026-05-20 | Shipped Sprint 069 appliance launcher soak harness. | Practical 4-slot serving is now repeatably validated through the launcher with health/status/metrics and generation artifacts. The next throughput gain must come from MTP draft commit or lower-overhead inter-stage handoff. | Sprint 070+ |
| 2026-05-20 | Shipped Sprint 070 persistent MTP forward runtime. | MTP forward now reuses resident scratch and serving JSON proves sequential `forward_run_count`, but draft timing remains near `4.6 ms`. The next useful MTP work is true one-slot commit into target replay state, not more scratch allocation cleanup. | Sprint 071+ |
| 2026-05-20 | Shipped Sprint 071 exact MTP commit serving. | MTP can now mutate the one-slot generation path by emitting accepted drafts after exact verification, and V100 evidence proves the committed sequence matches the target baseline. The next decision is whether commit-mode throughput justifies safe skip-verify/recursive MTP work or whether optimization should return to stage/kernel scheduling. | Sprint 072+ |
| 2026-05-20 | Shipped Sprint 072 MTP commit throughput gate. | Exact commit accepted and committed all measured drafts, but the same-fixture V100 benchmark was slightly slower than MTP off because target verification still runs. Keep MTP commit as a correctness feature and pivot practical throughput back to stage/kernel scheduling before recursive or skip-verify MTP. | Sprint 073+ |
| 2026-05-20 | Shipped Sprint 073 persistent mailbox workers. | Mailbox scheduling reduces old persistent wait accounting and improves over old persistent, but it remains slower than per-step. Keep per-step as the practical default and move the next optimization below host condition-variable scheduling. | Sprint 074+ |
| 2026-05-20 | Shipped Sprint 074 async HC handoff. | Queued default-stream peer handoff is correct and slightly faster, but the gain is too small to change the appliance default. The next scheduling sprint should use explicit CUDA streams/events, or the roadmap should pivot to larger kernel-side work. | Sprint 075+ |
| 2026-05-20 | Shipped Sprint 075 output-head top-1 probe. | The device top-1 candidate is correct but slower in output-head timing because the safe serial CUDA scan is worse than the existing host scan/readback. Keep the primitive opt-in and pursue a real parallel reducer, batched output-head selection, or larger stage/kernel work. | Sprint 076+ |
| 2026-05-20 | Shipped Sprint 076 parallel output-head top-1. | Replacing the serial device scan with a deterministic parallel reducer produced a real default-worthy gain and removed most output-head selection time. The next sprint should move to a larger remaining bucket: stage synchronization/stream events, batched output-head projection, or routed MoE kernel occupancy. | Sprint 077+ |
| 2026-05-20 | Shipped Sprint 077 batched output-head selection as opt-in only. | Row-batched output projection/top-1 is correct, but it is slower than the existing per-slot parallel top-1 path at 1M/4 slots. Keep output-head batching off by default and pivot away from output selection toward stage stream/event handoff, kernel-side scheduling, or routed MoE occupancy. | Sprint 078+ |
| 2026-05-20 | Shipped Sprint 078 event-ordered stage handoff as opt-in only. | CUDA events reduce the measured handoff/sync bucket but do not materially move end-to-end throughput. Stop spending sprint budget on small host scheduling variants and move to the kernel hot path, starting with routed MXFP4 occupancy experiments. | Sprint 079+ |
| 2026-05-20 | Shipped Sprint 079 routed MXFP4 row-pair kernels as opt-in only. | Pairing adjacent rows in the routed MXFP4 kernels preserves correctness but slightly regresses sustained 1M/4-slot throughput. The next kernel attempt should stop doing row-level CTA consolidation and instead change the expert execution shape more materially. | Sprint 080+ |
| 2026-05-20 | Shipped Sprint 080 copied tc-grid V100 INT8 kernel proof. | Candidate low-bit kernels must be copied into `ds4` and proven there before runtime use. The copied tc-grid v13 path proves high-M INT8 HMMA viability but not low-M routed decode practicality; the next kernel sprint should copy/prove TurboMind MXFP4 grouped GEMM. | Sprint 081+ |
| 2026-05-20 | Shipped Sprint 081 copied TurboMind MXFP4 grouped GEMM proof. | The copied TurboMind tree builds from `ds4` and passes V100 grouped MXFP4 compare on DS4 gate/up/down expert shapes. Because it preserves source MXFP4 rather than expanding to INT8, the next implementation target should be a DS4 routed-expert adapter around TurboMind's grouped GEMM contract. | Sprint 082+ |
| 2026-05-20 | Shipped Sprint 082 TurboMind routed expert adapter smoke. | The adapter now proves the end-to-end DS4 routed expert boundary around copied TurboMind: source MXFP4 pack, expert-grouped route rows, grouped gate/up/down, DS4 SwiGLU/route weights, and output parity against the current source-MXFP4 arena reference. The next sprint should wire it into runtime behind an opt-in flag and measure sustained throughput. | Sprint 083+ |
| 2026-05-20 | Shipped Sprint 083 opt-in TurboMind runtime bridge. | The DS4 CUDA wrapper can now call copied TurboMind behind `DS4_V100_TURBOMIND_ROUTED_FFN=1`, with strict and fallback modes. Because it repacks expert matrices transiently, it is a semantic bridge rather than the performance layout. The next sprint should make TurboMind packs offline/device-resident without duplicate source expert residency. | Sprint 084+ |
| 2026-05-20 | Shipped Sprint 084 offline TurboMind expert sidecar packer. | The project now has a CUDA tool that derives TurboMind packed expert sidecars from the real DS4 GGUF and existing V100 pack index. This keeps source provenance stable while creating the format needed to remove runtime repacking. The next sprint should add a bounded sidecar loader and adapter path from persistent packed buffers. | Sprint 085+ |
| 2026-05-20 | Shipped Sprint 085 persistent TurboMind sidecar smoke. | The project can now parse the TurboMind sidecar index, upload a bounded sidecar once, reconstruct device pointer tables, and run grouped MXFP4 experts from persistent packed buffers with source-arena parity. The next sprint should make this planner-admitted and scheduler-selectable instead of a standalone smoke. | Sprint 086+ |
| 2026-05-20 | Shipped Sprint 086 TurboMind sidecar admission. | The project now reports whether sidecar layouts fit the 32 GiB V100 budget under duplicate and replacement-style residency assumptions. Use this before enabling persistent TurboMind packs broadly; full sidecars should replace source expert bytes or be admitted as bounded cache. | Sprint 087+ |
| 2026-05-20 | Shipped Sprint 087 single-shard TurboMind appliance pack. | TurboMind expert bytes can now be emitted into `gpuN.weights` and executed through a DS4 CUDA no-repack API from resident arena spans. The next step is scheduler integration: combined arena sizing/upload and layer-state bindings from `turbomind-pack-index.tsv`. | Sprint 088+ |
| 2026-05-20 | Shipped Sprint 088 scheduler-bound TurboMind appliance runtime. | Context/layer/scheduler code can now bind TurboMind metadata, map/upload appliance `gpuN.weights`, use shard offsets for control tensors, and dispatch routed experts through the no-repack TurboMind API. V100 compile validation passes for scheduler smoke targets; the next step is V100 execution and throughput measurement on a full or bounded appliance directory. | Sprint 089+ |
| 2026-05-20 | Shipped Sprint 089 appliance-backed scheduler smoke. | A bounded stage-0 appliance directory now runs scheduler decode on V100 without a source GGUF model map, using `gpu0.weights` and positively reporting one TurboMind-routed layer. The next step is generating the full 8-GPU appliance, running full 43-layer decode, then benchmarking aggregate tok/s. | Sprint 090+ |
| 2026-05-20 | Shipped Sprint 090 full appliance pack and replay. | The single-directory TurboMind appliance now exists on k8s-local storage, fits all 8 V100s, executes all 43 layers with TurboMind-routed experts, and preserves first-token correctness. Next work should wire `--appliance-dir` into the launcher/service path and benchmark multi-slot async serving from this artifact. | Sprint 091+ |
| 2026-05-20 | Shipped Sprint 091 appliance launcher path. | The operator launcher and HTTP smoke now use `DS4_V100_APPLIANCE_DIR` and pass `--appliance-dir` into replay. The next sprint should benchmark multi-slot async serving from the appliance path and compare it to the source-index baseline. | Sprint 092+ |
| 2026-05-20 | Shipped Sprint 092 appliance multi-slot async soak. | The full TurboMind appliance now has a warm-started 4-slot service benchmark: correctness passes and tensor batching is reported, but aggregate generated throughput is only `11.256048` tok/s. The next sprint should profile the timed batch and attack the remaining serialization/kernel-occupancy bottleneck before broader production serving work. | Sprint 093+ |
| 2026-05-20 | Shipped Sprint 093 startup warmup and GPU profile. | Warmup now belongs to the server/launcher, not the benchmark client, and the 4-slot soak passes with `warmup_requests=0`. Decode-window profiler evidence says the next optimization should target F8 dense/projection launch shape and residual HtoD control traffic before further TurboMind work. | Sprint 094+ |
| 2026-05-20 | Shipped Sprint 094 grouped TurboMind and shared F8 serving. | The production appliance now caches TurboMind pointer tables, batches routed experts across active slots, and defaults shared F8 batching on. This improves the 1M/4-slot appliance soak to `12.634955` generated tok/s with correctness preserved, but GPU utilization is still low enough that the next sprint should target F8 projection kernel shape and request batching/harness determinism. | Sprint 095+ |
| 2026-05-20 | Shipped Sprint 095 request rendezvous and F8 cache probe. | The appliance launcher now uses a production microbatch coalescing window for multi-slot serving. This keeps concurrent requests in the same tensor batch and validates the 8-slot/256K profile at `17.052974` generated tok/s with `8/8` token matches. The F8-to-F16 cache probe is correct but flat, so the next sprint should focus on F8 projection kernels and HtoD/control-copy reduction. | Sprint 096+ |
| 2026-05-20 | Shipped Sprint 096 served decode profiler window. | The profiler can now wrap warmed HTTP generation batches after startup warmup. Served-path `nvprof` shows HtoD is only `0.14%`; the dominant GPU bucket is F8 arena matmul at `61.64%`, followed by TurboMind at `20.15%`, while CUDA API time is dominated by allocator churn. The next sprint should reduce F8 matmul launch/shape overhead and stop repeated malloc/free in the hot path. | Sprint 097+ |
| 2026-05-20 | Shipped Sprint 097 CUDA tensor pool default. | The appliance now defaults a bounded tensor pool on for multi-slot serving. Same-binary V100 fixtures improved from `11.902776` to `16.881653` generated tok/s at 1M/4 slots and from `17.193119` to `25.212896` at 256K/8 slots; production-style `auto` runs reached `17.532887` and `25.232220`. The next sprint should focus on the remaining F8 arena matmul and `cudaMemcpy` API overhead. | Sprint 098+ |
| 2026-05-20 | Shipped Sprint 098 grouped F8 attention output. | Attention output-A now uses one grouped F8 launch instead of eight group launches per layer/slot, with a launcher rollback flag. Same-binary V100 controls show `17.904697` vs `16.897788` generated tok/s at 1M/4 slots and `26.206100` vs `25.456942` at 256K/8 slots. The next sprint should batch attention Q/KV projection work across active slots or reduce remaining `cudaMemcpy` API overhead. | Sprint 099+ |
| 2026-05-20 | Completed Sprint 099 batch attention projection probe as opt-in only. | Projection-only batching across active slots is correct but not faster, so it remains behind `DS4_V100_ENABLE_BATCH_ATTN_PROJ=1` and is not a production default. The next sprint should shift away from projection-only batching and instead attack `cudaMemcpy` API overhead or a larger attention-stage batching boundary. | Sprint 100+ |
| 2026-05-20 | Shipped Sprint 100 TurboMind sync readback A/B. | The packed TurboMind route-validation readback is now debug-only in production, while the no-row-count-readback ABI remains opt-in because V100 profiling showed the wait moved into `cudaDeviceSynchronize` and 8-slot throughput regressed. The measured production default is old ABI plus route sync off at `26.372672` generated tok/s for 256K/8 slots. The next sprint should target stage/layer synchronization or a larger batched attention boundary. | Sprint 101+ |
| 2026-05-20 | Shipped Sprint 101 batch attention semantic repair. | The opt-in batch attention projection path now matches single-slot attention RMS norm and compressed-KV prep ordering, but V100 A/B shows no production-worthy gain (`26.43` vs `26.40` at 256K/8 slots and a regression at 1M/4 slots). Keep it opt-in and move the next optimization to a larger attention-stage boundary, stage/layer synchronization, or F8 matmul shape work. | Sprint 102+ |
| 2026-05-20 | Shipped Sprint 102 F8 row-pair default. | The F8 arena matmul path now has a two-output-row CTA shape across the hot F8 APIs and is defaulted through `DS4_V100_CUDA_F8_ROWPAIR=1`. V100 A/B produced a real but modest default-worthy gain (`27.05` generated tok/s at 256K/8 slots), so the next sprint should profile the new default and pursue larger F8/TurboMind kernel occupancy or cross-layer synchronization reductions. | Sprint 103+ |
| 2026-05-20 | Shipped Sprint 103 exact-bit F8 decode. | Removing `ldexpf()` from E4M3 decode produced the first post-row-pair double-digit kernel-side gain: `30.86` generated tok/s at 256K/8 slots and `19.73` at 1M/4 slots, with selected-token correctness preserved. The next sprint should profile this new default and decide between more F8 kernel shaping, vectorized decode, or the next TurboMind occupancy step. | Sprint 104+ |
| 2026-05-20 | Shipped Sprint 104 F8 warp-reduction kernels. | Replacing F8 shared-memory tree reductions with warp-shuffle block reductions produced repeatable but modest serving gains: `31.38` and `31.45` generated tok/s at 256K/8 slots and `20.03` at 1M/4 slots. The F8-to-F16 cache/cuBLAS idea was rejected as impractically slow in the served path. The next sprint should use a fresh profile and target either larger F8 kernel tiling or TurboMind occupancy, not VRAM-heavy cache expansion. | Sprint 105+ |
| 2026-05-20 | Rejected Sprint 105 BF16/F32 warp-reduction probe. | Extending warp reductions to BF16/F32 arena matmuls preserved correctness but did not clear the default-change bar: the 8-slot repeat landed inside the Sprint 104 band. The code was reverted. Sprint 106 should start from a fresh profile and target a larger execution-shape change. | Sprint 106+ |
| 2026-05-20 | Completed Sprint 106 served decode baseline profile. | Fresh warmed HTTP `nvprof` evidence shows F8 arena rows2/grouped rows2 still dominate at about 51% of GPU time, with TurboMind SM70 MXFP4 GEMM next at about 25%. GPU memcpy traffic is tiny despite noisy API `cudaMemcpy` accounting, so the next implementation should target F8 execution shape or TurboMind route batching rather than host RAM, disk, or more BF16/F32 cleanup. | Sprint 107+ |
| 2026-05-20 | Shipped Sprint 107 DS4 grouped F8 attention-output kernel. | A DS4-specialized grouped rows2 kernel for the fixed attention-output-A shape improves 8-slot/256K serving to `31.81` generated tok/s best observed and `31.63` on repeat, while 4-slot/1M remains neutral around `20.1` tok/s. The next larger target should move to TurboMind route-build fusion rather than more small F8 shape tweaks. | Sprint 108+ |
| 2026-05-20 | Completed Sprint 108 TurboMind small-route build probe. | Fusing route count/prefix/scatter into one small-route CUDA kernel preserves correctness, but the primary 8-slot/256K A/B stayed neutral to slightly slower (`31.76` opt-in vs `31.79` rollback on repeat). Keep it opt-in and move the next optimization to larger hot-path work: F8 matmul tiling/vectorization or TurboMind expert input layout. | Sprint 109+ |
| 2026-05-20 | Completed Sprint 109 F8 row4 CTA probe. | Four-output-row CTAs preserved correctness but regressed both 8-slot/256K (`30.998` vs `31.380` control) and 4-slot/1M (`19.898` vs `20.042` control). Keep row4 off by default and move the next sprint to a software-pipelined/fused boundary that raises tensor-core occupancy, especially TurboMind gate+up expert fusion or persistent grouped experts. | Sprint 110+ |
| 2026-05-20 | Completed Sprint 110 TurboMind fused gate/up probe. | A standalone DS4-shaped MXFP4 grouped-GEMM benchmark showed fused gate_up is `1.46x-1.53x` faster than separate gate and up calls at 6, 24, and 48 routed rows, with exact output parity. Proceed to the production appliance implementation behind a rollback knob. | Sprint 111+ |
| 2026-05-20 | Shipped Sprint 111 production fused TurboMind gate_up. | The appliance packer now emits fused `ffn_gate_up_exps.weight`, the runtime selects it by default with `DS4_V100_TURBOMIND_FUSED_GATE_UP=1`, and the full fused appliance passes scheduler plus selected-token correctness. Served 8-slot/256K improved from `31.312694` to `33.430971` generated tok/s in same-binary A/B, and 4-slot/1M reached `21.403909`. The next sprint should profile the fused served path and target persistent/grouped expert execution or fused downstream scheduling. | Sprint 112+ |
| 2026-05-20 | Completed Sprint 112 fused-profile and F8 warp-scale probe. | The fused appliance profile shows F8 row-pair plus DS4 grouped attention-output kernels now dominate at `54.58%` of GPU time. A guarded warp-broadcast E8M0 scale variant preserved correctness but regressed 8-slot/256K throughput from `33.484099` to `29.009399` generated tok/s, so it remains off by default. The next sprint should avoid tiny scalar F8 tweaks and instead target a larger tiled/persistent F8 projection path or deeper TurboMind expert scheduling. | Sprint 113+ |
| 2026-05-21 | Completed Sprint 113 direct FFN delta probe. | Direct shared-FFN delta accumulation preserved correctness but measured below the fused appliance control, so it remains opt-in. Larger execution-shape work is still needed. | Sprint 114+ |
| 2026-05-21 | Completed Sprint 114 shared-down F8 HMMA probe. | The DS4-shaped shared-down HMMA kernel was correct and slightly positive alone, but pair+down regressed the 4-slot/1M tier. Keep shared-down HMMA opt-in. | Sprint 115+ |
| 2026-05-21 | Shipped Sprint 115 shared gate/up SwiGLU F8 HMMA. | The DS4-shaped shared gate/up SwiGLU HMMA batch path improved both measured tiers and is now a launcher default. Combined pair+down remains opt-in because of the long-context regression. | Sprint 116+ |
| 2026-05-21 | Shipped Sprint 116 batched attention projection F8 HMMA. | Batched q_a, kv_latent, and q_b projections now default for active 4/8-slot batches, improving the 8-slot/256K tier to `33.697698` and 4-slot/1M to `21.469010`. | Sprint 117+ |
| 2026-05-21 | Completed Sprint 117 F8 trace and scalar single-slot fusion probes. | The served path is per-slot stage-pipelined; async slot chunking and scalar shared pair-SwiGLU fusion were correct but not faster. The next fusion must be software-pipelined and tensor-core-oriented. | Sprint 118+ |
| 2026-05-21 | Completed Sprint 118 single-token HMMA probe. | The hot `4096 x 8192` single-token HMMA path was correct but regressed badly (`16.083451` vs `33.502249`), confirming that naive WMMA wastes too much of the token tile. | Sprint 119+ |
| 2026-05-21 | Shipped Sprint 119 event-ordered stage handoff. | `DS4_V100_ASYNC_EVENT_HANDOFF=auto` now enables CUDA event-ordered handoff for multi-slot per-step serving. It raised 8-slot/256K to `34.433252` generated tok/s and 4-slot/1M to `21.771077`, with token matches preserved. | Sprint 120+ |
| 2026-05-21 | Completed Sprint 120 single shared gate/up/SwiGLU row-pair probe. | The new opt-in row-pair single-fusion kernel is correct, but measured below the current default (`34.380968` vs `34.490294`). Do not promote; proceed to a real SM70 pipelined F8 mainloop or deeper TurboMind expert scheduling. | Sprint 121+ |
| 2026-05-21 | Shipped Sprint 121 16-slot 256K throughput mode. | The runtime and wrappers now admit 16 active slots for 256K serving, with context-aware guards preventing unsafe 16-slot long-context launch. Full 16-slot scheduler correctness passes at 256K, and served throughput improved to `43.659461` generated tok/s with `16/16` token matches. The next sprint should profile this mode and target a real SM70 pipelined F8 mainloop or deeper TurboMind expert scheduling. | Sprint 122+ |
| 2026-05-21 | Shipped Sprint 122 16-slot rendezvous stabilization. | The 16-slot profile confirmed the served hot path still feeds F8 wrappers as `n_tokens=1`; async slot chunking exposes wider batch kernels but regresses end-to-end throughput by losing stage overlap. The launcher now resolves `DS4_V100_MICROBATCH_WAIT_US=auto` to 200 ms at 16 active slots, producing one 16-request tensor batch and `43.534061` generated tok/s by default; best observed candidate was `43.730215`. Next work should focus on a software-pipelined hot-path kernel or TurboMind expert scheduler change that improves the per-slot topology instead of synthetic wide-batch probes. | Sprint 123+ |
| 2026-05-21 | Shipped Sprint 135 32-slot 128K throughput admission. | The scheduler and layer executor now admit 32 active slots for 128K contexts while keeping 256K capped at 16 slots. Full 32-slot scheduler smoke passed and the served appliance reached `52.840889` generated tok/s with `32/32` token matches, versus `45.780913` for the same-context 16-slot control. Next work should test 64-slot short-context fit and lower-level software-pipelined expert kernels. | Sprint 136+ |
| 2026-05-21 | Shipped Sprint 136 64-slot 64K throughput admission. | The scheduler and layer executor now admit 64 active slots for 64K contexts while keeping 128K capped at 32 slots and 256K capped at 16 slots. Full 64-slot scheduler smoke passed and the served appliance reached `57.322945` generated tok/s with `64/64` token matches, versus `52.884400` for the same-context 32-slot control. Next work should prioritize lower-level software-pipelined expert kernels or a bounded 96/128-slot occupancy map. | Sprint 137+ |
| 2026-05-21 | Shipped Sprint 137 128-slot 32K throughput admission. | The scheduler and layer executor now admit 128 active slots for 32K contexts while keeping 64K capped at 64 slots, 128K capped at 32 slots, and 256K capped at 16 slots. Full 128-slot scheduler smoke passed, status/metrics confirmed the served binary, and throughput reached `59.598172` generated tok/s with `128/128` token matches. The same-context 64-slot control was `57.170428`, so scheduler-width scaling is still positive but diminishing. Next work should shift to the software-pipelined packed MXFP4 expert kernel path. | Sprint 138+ |
| 2026-05-21 | Completed Sprint 138 wide compact gate/up baseline. | The TurboMind gate/up benchmark now defaults through `tokens_per_active=128`, covering 768 routed rows in compact mode. V100 validation passed through 192/384/768-route shapes; the 768-route fused gate_up baseline is `0.6379 ms`. Sprint 139 should implement a narrow software-pipelined packed MXFP4 kernel probe that beats that baseline or proves the current TurboMind path is already near the practical ceiling. | Sprint 139+ |
| 2026-05-21 | Completed Sprint 139 fixed-shape 128-slot gate/up probe. | A fixed m128 TurboMind MXFP4 gated-SiLU ABI beat the 768-route isolated gate/up target at `0.5999 ms`, then passed full 43-layer 128-slot smoke and served at `60.130047` generated tok/s on the interleaved gated appliance. The same-binary probe-off control was `60.061899`, so gate/up-only specialization is not a material served lever. Next work should fuse or schedule a larger routed-FFN boundary, especially down plus weighted scatter/reduce. | Sprint 140+ |
| 2026-05-21 | Completed Sprint 140 fixed-shape 128-slot down probe. | A fixed m128 TurboMind MXFP4 down ABI beat the isolated 768-route down target at `0.3026 ms` versus `0.3272 ms` generic and passed full 43-layer 128-slot smoke, but served A/B was slower with it enabled: `60.038469` versus `60.129772` down-probe-off. Keep it opt-in/default-off. The next sprint should stop tuning individual GEMMs and target down epilogue plus weighted reduce or a persistent routed-FFN executor. | Sprint 141+ |
| 2026-05-21 | Completed Sprint 141 half2 route-row reduce tail probe. | A half2-vectorized by-pair reduce tail passed full 43-layer 128-slot smoke, but served A/B was neutral: `60.104512` half2 route-row reduce, `60.112248` scalar route-row reduce, and `60.108232` control. Keep it opt-in/default-off. Separate tail-kernel vectorization is now ruled out as a material lever; the next sprint should modify the larger routed-FFN boundary, not another wrapper kernel. | Sprint 142+ |
| 2026-05-21 | Completed Sprint 142 TurboMind down-epilogue reduce probe. | The fixed-shape down GEMM epilogue can apply route weights and atomically accumulate directly into token rows, and full 43-layer 128-slot smoke passed. Served A/B was only run-noise positive (`60.041003` vs `59.987105`), so it remains opt-in/default-off. | Sprint 143+ |
| 2026-05-21 | Shipped Sprint 143 prefill/decode metric split. | The benchmark harnesses now report aggregate prompt/prefill, generated, and continuation/decode tok/s separately. This is now the required visibility layer for served A/B decisions because aggregate generated tok/s can hide whether a change helps prompt replay or decode. | Sprint 144+ |
| 2026-05-21 | Completed Sprint 144 SM70 MXFP4 m64n256 tile probe. | The wider-N tile passed standalone correctness and full 43-layer smoke, with a small isolated down improvement (`0.2896 ms` vs `0.2936 ms`), but served 128-slot/32K A/B regressed for both down `m64n256` (`59.791839`) and gate `m64n256` (`59.797232`) versus control (`59.993301`). Keep it opt-in; the next throughput step must be a larger routed-FFN executor or scheduler boundary. | Sprint 145+ |
| 2026-05-21 | Shipped Sprint 145 256-slot 16K admission. | The planner fits 256 slots at 16K with worst GPU `29.07 GiB / 32.00 GiB` including reserve, full 43-layer smoke passed, and served A/B reached `61.065087` generated tok/s / `57.248519` continuation tok/s with `256/256` token match. The 128/192/256 slot curve is nearly flat, so this is a guarded ceiling rather than the main route to practical throughput. | Sprint 146+ |
| 2026-05-21 | Completed Sprint 146 1536-route fixed-shape probe. | The 1536-route gate/up and down probes are correct and explicit opt-ins. Gate `m128_1536` improved the standalone compact 256-slot shape (`0.9435 ms` vs `0.9651 ms` generic gated), but served 256-slot/16K A/B was flat to slightly worse: `61.204203` generated / `57.378940` continuation tok/s versus `61.223893` / `57.397400` control. Keep 1536 probes out of `auto`; larger software-pipelined routed-FFN work remains the main path. | Sprint 147+ |
| 2026-05-21 | Completed Sprint 147 1536-route down-reduce checkpoint. | The down GEMM route-weighted F32 accumulation epilogue now covers the 1536-route compact shape and passed full 43-layer 256-slot smoke. Served A/B was deferred after the strategy pivot toward larger fused-kernel work, so this remains an explicit diagnostic path only. | Sprint 148+ |
| 2026-05-21 | Completed Sprint 148 stage-4 fused gate/up software-pipeline probe. | A true stage-count software-pipeline variant of the fused MXFP4 gate/up+gated-SiLU kernel was implemented and tested. The 768-route `m128_s4` probe improved the isolated benchmark (`0.5811 ms` vs `0.6033 ms` for `m128`) and passed full 43-layer smoke, but served A/B was only `60.049057` generated / `56.295991` continuation tok/s versus `59.865668` / `56.124063` control, and full-scheduler profiles did not show a reliable gate/up bucket reduction. Keep stage-4 probes opt-in; the next material path is a larger routed-FFN executor boundary or a TP/EP microbenchmark. | Sprint 149+ |
| 2026-05-21 | Completed Sprint 149 TP split and P2P topology probe. | The TurboMind harness now measures a 2-way split of the DS4 routed-FFN middle dimension and a P2P reduce-payload proxy. Ideal 2-way compute speedup is `1.858x` at 768 routes and `1.468x` at 1536 routes before communication; 12 MiB hidden payloads take about `0.26 ms` over NV2, `0.52 ms` over NV1, and `1.29-1.31 ms` over SYS. This supports a bounded 2-GPU TP prototype on NV2 pairs, not an immediate 8-way rewrite. | Sprint 150+ |
| 2026-05-21 | Completed Sprint 150 two-GPU TP split probe. | The first real two-GPU routed-FFN TP proxy runs the two `1024`-wide halves concurrently on NV2 pairs and includes conservative input/output payload copies. It is positive at 768 routes (`~1.28x` total speedup on pairs `0,3` and `4,7`) but neutral to slower at 1536 routes (`0.85-0.94x`). Next TP work should be a one-stage correctness prototype for 128-slot/32K before any scheduler-wide change. | Sprint 151+ |
| 2026-05-21 | Completed Sprint 151 two-GPU TP correctness gate. | The TP split proxy now compares full one-GPU routed-FFN output against the FP32 sum of both TP partial outputs. Finite MXFP4 fixtures pass on clean NV2 pairs at both 768 and 1536 routes with `rel ~= 2.46e-04`, `bad=0`, and max absolute difference `6.1035e-05`. The split math is valid; remaining TP risk is production scheduling and payload overlap. | Sprint 152+ |

## Open Questions

1. What reference should define correctness tolerances for mixed
   BF16/F32/F8_E4M3_B128/MXFP4 execution on V100 after MoE is included?
2. What production serving milestone should follow the loopback service:
   OpenAI-compatible API, authenticated proxy, streaming, or a narrower
   internal endpoint?
3. When should the MTP path graduate from diagnostic exact-verify mode to true
   draft commit without recomputing the base target token?
4. Should the persistent `/srv/dev/ds4-sprint004` pack become the seed
   deployment artifact, or should a formal pack release format come first?
5. Which practical-use target matters first: single-user latency, aggregate
   throughput at 8-128 slots, or synthetic maximum throughput?
