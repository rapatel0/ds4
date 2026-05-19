---
created: 2026-05-17
last_updated: 2026-05-19
last_updated_by: vision
revision: 65
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
| Sustained benchmark without major kernel changes | `~5-20` tok/s | Medium | Current evidence is at the low end; more slots will not help much until multi-token request state is batched rather than reset/serialized. |
| Continuous token-step batching, 8-32 active slots | `~40-200` tok/s | Medium-low | Requires persistent per-slot state, no per-request reset, multi-token batching, and useful queue depth. |
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

### Sprint 054 - Hot-Path Kernel Selection And Low-Bit Expert Integration [planned]

- **Goal**: Use Sprint 052 timing to replace the hottest routed/shared FFN and
  dense projection calls with the best available V100 low-bit kernels, then
  prove end-to-end speedup without losing selected-token correctness.
- **Rationale**: The existing MXFP4/Q8/Q4 CUDA paths prove source-format
  correctness and residency, but practical throughput needs fused unpack/
  dequant plus tensor-core or integer execution in the main decode hot path, not
  just standalone smokes.

### Sprint 055 - Routed Expert Batching And Persistent MoE Scheduling [tentative]

- **Goal**: Batch routed experts across active slots and reduce launch overhead
  with a persistent or grouped-MoE scheduler.
- **Rationale**: V100 tensor cores need enough effective M to stay busy. The
  biggest gap between current performance and the 300+ tok/s target is likely
  expert scatter, small per-expert GEMMs, and per-route launch overhead.

### Sprint 056 - MTP Draft Commit And Throughput Serving [tentative]

- **Goal**: Graduate MTP from one-token verify diagnostics to a committed draft
  path with rollback safety, then benchmark aggregate decode with MTP enabled.
- **Rationale**: Served MTP verify is shipped, but it still recomputes the base
  target token and reports diagnostics. Practical throughput needs accepted
  drafts to reduce target-model work, with exact rollback boundaries preserved.

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
   throughput at 8-32 slots, or synthetic maximum throughput?
