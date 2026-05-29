# Spike B Decode-Optimization Steering (updated 2026-05-29 after Sprint 562)

Steering for the next TP/EP serving-throughput phase, off the de-confounded
steady-state reference (32 slots / 256K / 256 req / 64 tok/req, ~35.9 tok/s
server decode, ~889 ms decode domain).

## Unifying diagnosis — read this first

Both dominant domains are **overhead-bound, not compute- or bandwidth-bound:**

- **EP 53% (~473 ms):** at 32 slots × top-6 = 192 activations over 256 experts,
  each GPU's 32 local experts see **<1 token** → grouped GEMM is tile-underfill;
  the cost is dispatch / pack / reduce / all-to-all orchestration, not FLOPs.
- **HC-current 40% (~357 ms):** moves ~2 MiB → the time is launch-count + sync +
  **GPU0-centralization** × 43 layers, not data.

**Consequence:** wins come from STRUCTURE — de-centralize, fuse, fewer launches,
remove host synchronization, capture static sub-regions — **not** from faster
kernels. Optimizing the expert GEMM alone moves a fraction of 53% and none of
40%. MTP remains deferred support code until the base TP/EP path is stable and
optimized.

## Current reassessment after sprints 478-548

- **A1-A3 are done.** A1 RMS-norm rank-local is rolled into A2. A2 HC mix
  row-parallel all-reduce is promoted from Sprint 478. A3 router rank-local
  all-reduce is promoted from Sprint 480.
- **A4 is complete for the served TP/EP path.** Sprint 483's "A6 PATH 4" result
  was structurally A4 for the attention-projection consumer, and Sprint 526
  finished the post-attention FFN shared/routed consumers in
  `engine/post_attention_ffn.cu`. The promoted path now reports
  `rank_major_shared_input=1`, `rank_major_route_input=1`, and
  `slot_major_ffn_norm=0`.
- **True A6 is still open.** In this document, A6 means fusing HC/current
  computation into the attention-projection prologue. It does not mean the
  Sprint 483 rank-major attention-projection consumer conversion.
- **C1 suffix replay is promoted for the TP/EP launcher.** Sprint 479 removed promoted
  hot-path direct peer-copy transport in favor of NCCL, and the structural
  extraction made the surface readable. Sprint 527 removed GPU0-centralized
  output-head prep. Sprint 528 removed output-head device-wide projection/top-1
  waits, Sprint 529 removed the attention-output eager stream-synchronization
  branches, Sprint 531 trimmed compact EP compose broadcasts while staying
  on the no-SYS NCCL broadcast path, Sprint 532 removed promoted-path
  post-attention FFN input host waits, Sprint 533 removed promoted-path
  attention-projection host waits, Sprint 534 removed promoted-path
  attention-read raw/window host waits, and Sprint 535 removed the HC-current
  final fill/pack host wait. Sprint 536 completed the preflight/control pass:
  target selected-token still passes with peer-copy/SYS zero and NCCL graph SYS
  zero. Sprint 537 restored direct suffix replay, Sprint 538 repaired serving
  parity by preventing unsafe cross-position route-shape reuse, and Sprint 539
  restored served suffix cache reuse with graph-only fixed-capacity
  post-attention route geometry. Sprint 540 then ran a warmed `32` request /
  `32` slot / `64` token selected-token gate with startup isolated; graph
  matched all generated sequences and improved request-window throughput
  `99.446247s -> 90.181067s`. The TP/EP launcher now defaults graph suffix
  replay on, with `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0` as the operational
  opt-out. Sprint 541 fixed the stale helper-host-sync audit label; the
  promoted graph path now reports `graph_audit_blocker=none`. Sprint 542
  quantified the fixed-capacity route envelope and rejected lowering static
  route caps as the next promotion lever because Sprints 434/436/437 already
  showed static rank/executor/compose caps can change tokens even when overflow
  audits pass. Remaining graph work is full-capture/sync cleanup and
  full-shape device-masked route/executor/compose efficiency tuning, not
  serving-parity repair. Sprint 549 removed the rejected static-cap,
  host-synced actual-route, and masked compact-copy experiments from active
  code so the promoted fixed-capacity route plan is the only supported
  graph-stable padding surface. Sprint 550 made the compact EP pack kernel
  route-blocked so inactive padded routes return before hidden-wide work while
  preserving the fixed graph-visible route envelope. The warmed `32` request /
  `32` slot / `64` token graph gate stayed correctness-clean and topology-clean
  but did not improve steady-state throughput versus the Sprint 540 promoted
  control (`90.181067s -> 90.528551s`, `22.709904 -> 22.622697` client
  generated tok/s). Treat that as a structural cleanup, not a performance lever.
- **Full capture is correctness-clean but still position-keyed.** Sprint 544
  disabled the promoted suffix stage and ran full-capture graph gates on the
  current surface. It matched eager response/checksum multisets with
  `graph_audit_blocker=none`, captured/replayed `43/43`, and kept peer/SYS zero,
  but had `0` persistent cache hits and `43` position invalidations. The next
  full-capture blocker is dynamic decode-position state, not helper host sync.
  Sprint 545 traced the dependency and rejected a one-scalar cache-key patch:
  position is baked into RoPE launch arguments, compressed-KV emission branches,
  typed-KV runtime row selection, compressed-row bookkeeping, and raw-window
  modulo addressing. Full-capture position reuse requires a staged
  replay-updated/device-stable design before `position` can be removed from the
  persistent cache key. Sprint 546 completed Stage 1 by adding
  `RankState::d_decode_position` and converting pure kernel consumers
  (RoPE, compressed-state store, compressed-row RoPE emit, raw SWA store/read)
  to read replay-updated device memory. Host emission branches, typed-KV row
  selection, row-position bookkeeping, and the full-capture position cache key
  remain intentionally unchanged. Sprint 547 reviewed the next topology stage
  and rejected always-launching emitted-row kernels as a standalone change: it
  would add non-emitted-position work without solving typed-KV runtime row
  selection or host row bookkeeping. Sprint 548 evaluated the larger
  post-KV suffix boundary (`attention_output` through compose, final-HC eager).
  It is replay-stable in the reduced direct scaffold, including cross-position
  cache reuse, but it is slower than the promoted `compose_eager_final_hc`
  suffix because graph size and replay cost increase. Keep it diagnostic-only;
  do not promote it as the default suffix. Sprint 551 removed the raw typed-KV
  store/load launch-argument dependency on host `opt.position` for graph mode
  by adding dynamic-position TP runtime row APIs that read
  `RankState::d_decode_position` in the store/load kernels. The targeted
  `attn_raw` smoke matched the existing static-position runtime path exactly
  at nonzero raw row `65`. This is a C1 readiness step only; full capture
  remains position-keyed because emitted compressed/indexer rows still depend
  on host `emitted`, host row counters, host row-position arrays, and static
  typed-KV runtime calls. Sprint 552 removed the static-position typed-KV
  runtime calls for graph-mode emitted compressed attention and indexer rows:
  both now use dynamic-position store/load APIs and targeted `attn`/`indexer`
  smokes matched the existing static-position row path exactly. Full capture
  remains position-keyed because emitted topology, bounded-row source/dest
  pointers, row counters, and row-position arrays are still host-selected.
  Sprint 553 removed the captured bounded-row source/dest pointers for
  graph-mode emitted compressed/indexer typed-KV store/load by computing the
  bounded compact row from `d_decode_position` inside the runtime kernels.
  Targeted smokes validated nonzero bounded row `1` for both compressed
  attention and indexer rows. Full capture remains position-keyed because
  emitted work selection, host row counters, host row-position arrays, and
  typed-history reload are still host-driven. Sprint 554 removed the
  static-position TP runtime argument from graph-mode typed-history reload:
  compressed attention and indexer history loads now derive the historical
  source position from `d_decode_position` plus bounded row inside the runtime
  kernel. Full capture remains position-keyed because emitted/non-emitted graph
  topology and host emitted-row bookkeeping are still host-selected. Sprint 555
  tested dropping the position key for no-suffix full capture when compressed KV
  is off, but rejected it: structural cache reuse worked (`43/43` replays,
  zero position invalidations), yet selected-token correctness failed versus
  eager (`29361` eager first token vs `128819` / `118235` candidates). The
  full-capture position key remains a correctness guard. Sprint 556 localized
  the immediate replay failure: no-suffix full capture without replay-probe
  preserved the same selected-token first token as eager at the exact `8x2`
  shape (`128819`), while adding replay-probe changed it to `118235`. The
  replay-probe path captures by executing the full step on live buffers and
  then immediately launches the captured full graph against those already-
  advanced buffers. That is not a valid full-capture replay parity test. Full
  capture needs a replay validation harness that snapshots/restores device
  input state, or a split cache-miss path that keeps the capture-executed result
  and tests replay only from fresh state, before any new cache-key relaxation.
  Sprint 557 promoted the first repair: no-suffix replay-probe now fails loudly
  with `full_capture_live_state_replay_requires_snapshot` instead of returning
  a misleading token. A promoted suffix replay sanity still captured/replayed
  `43/43`, so the serving suffix path remains unchanged. Sprint 558 replaced
  the guard with a real fresh-state no-suffix replay validation path: on a
  full-capture cache miss, the served request runs eager and the graph is
  captured only as an audit/cache artifact; a later same-position request can
  replay the cached full graph from fresh request state. The reduced sequential
  probe matched eager selected tokens on both requests (`128819`, `128819`) and
  replayed `43/43` cached full graphs on the second request, but the second
  request checksum still drifted (`5174931161` eager vs `5002850195` replay).
  Sprint 559 localized and fixed that checksum drift: the full-capture replay
  path was missing the host-side `d_final_hc_shard` / `d_hc_scratch_shard`
  pointer swap that eager final-HC expansion performs after enqueue. The replay
  path now mirrors that swap only for no-suffix full capture when
  `final_hc_carry_gate` and `tp_hc_final_expand_gate` are both active. The
  reduced sequential probe now matches eager selected tokens and checksums on
  both requests (`7238127778`, `5174931161`) while replaying `43/43` cached
  full graphs with zero peer/SYS transport. Treat this as full-capture
  validation repair, not no-suffix serving-default promotion. Sprint 560 added
  the next host-state mirror for no-suffix full-capture cache-hit replay:
  when compressed KV is active and the current position emits a compressed row,
  replay now mirrors eager attention/indexer compressed-row counters,
  row-position metadata, and loaded-row metadata on the host. The served
  appliance still has `true_ds4_compressed_kv_gate=false`, so Sprint 560 also
  used a one-off compressed-KV entrypoint rather than adding a permanent flag.
  That one-off validation matched compressed-KV eager selected tokens and
  decode checksums on both sequential requests (`58204`, `109597` and
  `7265791446`, `79399742586`) while replaying `43/43` cached full graphs with
  zero invalidations, zero peer/SYS, and zero NCCL graph SYS edges. Sprint 561
  then made graph-mode compressed-KV emitted/non-emitted topology stable by
  enqueueing the emitted-row topology in graph mode and masking emitted work
  from `d_decode_position` on non-emitted positions, while leaving eager compact
  topology and host row metadata tied to the real host `emitted` value. The
  one-off compressed-KV validation matched eager selected tokens/checksums at
  both an emitted ratio-4 position (`262083`: `58204`, `109597` /
  `7265791446`, `79399742586`) and an adjacent non-emitted position (`262084`:
  `109939`, `107875` / `561376577`, `2841198172`) with `43/43` cached full
  graph replays, zero invalidations, zero peer/SYS, and zero NCCL graph SYS
  edges. This is still not a compressed-KV serving promotion. Sprint 562 then
  retried full-capture cross-position cache-key relaxation. The first adjacent
  same-session replay matched, but the six-request same-session gate diverged
  on request three (`117465` / `17092309830` eager versus `2039` /
  `110810249310` replay), proving more captured position-derived graph state
  remains. The candidate was removed and full capture remains position-keyed.
- **Use previous promotions as the control.** Do not duplicate control runs
  solely because a new sprint starts. Refresh control only when the binary,
  launcher defaults, topology policy, validation harness, model path, or target
  shape changed enough to invalidate the previous promoted artifact.

## A. HC-current (40%) — de-centralize GPU0 (highest-ROI, eager-path win today)

Steps 2–6 and 9–10 run on GPU0 over the full hidden while 7 GPUs idle. Every
"global over hidden" op decomposes into rank-local partial + one tiny all-reduce:

- **A1 RMS-norm rank-local:** done as part of A2. Partial sum-of-squares on
  each `[slots,4,512]` shard → all-reduce `[slots,4]` scalar → normalize
  locally. No gather.
- **A2 HC mix as row-parallel GEMM:** done and promoted. `attn_fn[16384×24] @ hc[slots,16384]` —
  16384=4×4096 is the contraction, already sharded (2048/rank). Each rank computes
  partial `[slots,24]` → all-reduce `[slots,24]` (3 KiB). Replaces gather-to-0 +
  full-norm + mix + split + broadcast with 2 rank-local kernels + 1 tiny allreduce.
- **A3 Router rank-local:** `[slots,4096]·[4096,256]`, 4096 contraction → row-
  parallel → all-reduce `[slots,256]`. Done and promoted.
- **A4 Drop the full-current allgather (step 8):** complete for the served
  TP/EP path. Attention projection is done under the old "A6 PATH 4" name,
  router is covered by promoted A3, and Sprint 526 made post-attention FFN
  shared/route consumers rank-major by default while removing the promoted
  path's slot-major FFN norm dependency.
- **A5 Fuse the survivors:** after A1–A4, HC-current ≈ 3 rank-local kernels + 2
  tiny all-reduces/layer vs ~12 steps. Fuse norm+partial-mix; mix-apply+FFN-norm.
  Sprint 543 rejected the tempting split-apply + weighted-current fusion:
  it preserved parity but regressed the warmed graph-suffix serving gate
  (`90.181067s -> 95.164862s` and `96.046732s` for two variants). Do not retry
  that shape without a direct kernel microbenchmark proving a win.
- **A6 Fuse HC into the attention-projection prologue** (compute `current_shard`
  inside the projection kernel; drop the intermediate buffer + a launch).

Net: GPU0-serial / ~12-launch / barrier-heavy -> 8-way parallel / ~3-launch / 2
tiny collectives. **A1-A4 are complete for the served TP/EP path; the next
bankable NCCL cleanup is the model-boundary output-head A1 pattern.**

## B. EP (53%) — orchestration + sub-1-token experts

- **B1 MTP is deferred, but the research blocker is cleared.** EP is 53%
  *because* M<1/expert; MTP (verify K draft tokens) makes experts see (K+1)× tokens
  → tiles fill + fewer steps. That remains true, but TP/EP MTP is intentionally
  out of the active docket until the base TP/EP path is stable and optimized.
  The implementation record is `MTP_IMPLEMENTATION.md`.

  Current facts (verified against `research/ds4/` upstream and the pod
  2026-05-28):

  - **The sidecar runs complete canonical MTP, not a truncated probe.** Its
    32 tensors are the full MTPBlock in GGUF packing convention. Upstream
    `research/ds4/ds4.c:3068-3104` `mtp_weights_bind()` requires exactly
    these 32 tensor families. The "32 vs 1,575" gap is packing convention,
    not truncation: GGUF stacks the 256 routed experts into 3 tensors
    (`ffn_*_exps`); HF safetensors unpacks each expert as its own tensor.
  - **Upstream ds4.c has no sidecar.** It expects MTP in the *same* GGUF as
    the main model. The V100 sidecar exists because the appliance GGUF was
    produced through a pipeline that ran the HF transformers loader, which
    silently strips `mtp.*` via
    `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`. Someone
    preserved MTP in a separate Q4_K/Q8_0/F32 GGUF + a parallel runtime
    loader. **It is a packaging band-aid, not an architectural choice.**
  - **The canonical weights are local in two forms:** the sidecar's
    `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (3.6 GB, complete MTP at
    Q4_K/Q8_0/F32) and the HF safetensors cache at
    `/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`
    (complete MTP at FP8/MXFP4 with scales, same precision as the main
    path).
  - **The existing pack pipeline handles almost everything.**
    `tools/tp-ep-pack-contract.c` → `tools/appliance-pack.cu` →
    `tools/turbomind-pack.cu` → runtime mmap is already format-agnostic
    (`f8_e4m3_b128`, `mxfp4`, `bf16`, `f32`) and already shards by TP8/EP8.
    Adding MTP is **one ~200-LoC one-off safetensors→GGUF converter** plus
    a small extension to `tp-ep-pack-contract.c` for layer 43 (~50–100 LoC
    reusing existing sharding rules) plus mechanical binding additions in
    `engine/runtime_pack.cu` plus sidecar deletion.
  - **No new kernels.** MTPBlock reuses the existing Block primitives
    (`kernels/v100/norm.cuh`, `attention.cuh`, `ep_compose.cuh`,
    `hc_mix.cuh`) plus a small prologue (`e_proj` + `h_proj` + norms) and
    an HC head epilogue. The forward (Phase A) is binding only.
  - **The real B1 work is unchanged: the TP/EP speculative-decode loop**
    (Phase B / `ds4_session_eval_speculative_argmax`-equivalent),
    accepting/rejecting K draft tokens across 8 ranks with KV-consistent
    accept semantics. That is what the TP/EP launcher refuses MTP for
    today, and it does not become easier just because the sidecar is
    removed.
  - **Sequence (5–7 are MTP-specific):**
    1. C5 sync-point reduction pass 2 (in flight, sprint 529)
    2. B2 compact EP variable-size NCCL compose
    3. C1 piecewise graph capture
    4. Tuning sprint (reference-shape perf + shape envelope + NCCL
       pinning + C4 spill)
    5. MTP weight integration: converter + contract + sidecar delete
       (three sprint-sized tasks, correctness-only)
    6. MTPBlock.forward in `engine/` (one sprint, Phase A)
    7. TP/EP specdec loop (Phase B / the actual B1, multi-sprint, opts
       into perf measurement per validation policy)
- **B2 Fuse dispatch + grouped-GEMM + weighted-combine** into 1–2 kernels,
  device-side offsets only (no host sync on route counts). Template: the fork's
  `awq_moe_single_token_sm70` compact path. Make compact-route-compose the
  default. First finish the open compact-route transport half: replace the served
  path's variable-size compose movement with a topology-compatible NCCL scheme.
  Sprint 530 rejected all-pairs grouped `ncclSend`/`ncclRecv`: NCCL routed some
  pairs through SHM and failed the container `/dev/shm` budget, violating the
  no-SYS/no-SHM direction of the promoted topology policy. Future B2 transport
  work should use a ring-compatible or statically bucketed collective shape, not
  all-pairs P2P. Sprint 480's `ncclReduceScatter` evidence covers only
  non-compact FP32 and is not proof for served compact traffic. Sprint 531
  promoted the compatible near-term transport cleanup: keep NCCL broadcast,
  skip zero-route source ranks, and pack active compact rows before broadcast so
  compact return bytes follow active route counts instead of padded
  `slots * top_k` segments. The larger B2 fusion item remains open.
- **B3 TP-sharded experts vs EP A/B (the S-F question — now justified).** EP's 53%
  is dispatch/all-to-all. TP-experts have **no all-to-all** (reduce via the hidden
  all-reduce). For 13B-active/8-GPU/32-slot, test whether all-to-all overhead >
  TP-reduce cost. Strongest evidence yet to actually run it.
- **B4 Overlap shared (dense, rank-local) expert with the routed all-to-all
  dispatch** (independent until combine; re-apply sprint-261 overlap on serving).
- **B5 Correctness-preserving capacity balancing** (capacity-16 broke output in
  435; need a parity-clean fixed-shape scheme — also helps capture, see C1).

## C. Cross-cutting — and the highest-ceiling idea

- **C1 Piecewise graph capture (likely the biggest single win).** Full-serving-
  graph is blocked on parity + dynamism, but the per-layer compute region
  (HC-current → attention → EP → compose) is proven capturable in the smoke. Graph
  ONLY that static sub-region; run dynamic orchestration (request mgmt, output
  head, sampling) eager around it; feed routing via **persistent device buffers**
  (fixed worst-case `slots×top_k`; route values change per step, graph structure
  fixed). Bridges "works in smoke, blocked in serving" → could recover most of the
  2.27× without solving the whole loop. Do this after A4, output-head A1,
  sync-point reduction, and compact EP compose so the captured region is not
  polluted by avoidable host syncs or remaining non-NCCL compose movement.
  Sprint 542's route-padding audit showed the current fixed graph envelope uses
  worst-case per-rank route capacity (`192` rows/rank at the `32` slot x top-6
  shape), while logged actual route pressure had p50 max-rank `64` and max
  `132`. Do not reduce this with static caps; prior static-cap sprints changed
  tokens. The safe efficiency path is a full-shape device-masked executor and
  compose implementation that preserves graph-visible shapes while skipping
  inactive rows internally.
  Sprint 544 rechecked full capture without the suffix stage: parity is clean,
  but persistent cache reuse is still blocked by position-keyed graph captures.
  Sprint 545 showed that the next C1 code target is not a single scalar patch:
  replay-updated position must be staged across pure kernel consumers, emitted
  compressed-KV topology, typed-KV runtime row selection, host row-position
  bookkeeping, and raw-window row selection before full capture can reuse graphs
  across decode positions. Sprint 546 landed the pure-kernel stage only; the
  next blocker is compressed-KV topology because emitted-row work is still a
  host branch over `opt.position`. Sprint 547 rejected a narrow always-launch
  emitted-kernel patch because typed-KV runtime calls and host row bookkeeping
  would still make full capture position-dependent. Sprint 548 proved the
  larger post-KV suffix boundary is correct but not a promotion candidate at
  the reduced direct shape; it should remain a diagnostic boundary. Next C1
  work should reduce fixed-padding overhead inside the promoted graph-stable
  routed executor/compose path, or resume full-capture device-state work with
  a typed-KV/runtime refactor plan. Sprint 549 retired the rejected padding
  scaffolds (`device_actual_route_sync`, static rank/executor/compose caps,
  and masked compact copy); do not tune by resurrecting them. Sprint 550
  completed the obvious fixed-envelope compact-pack cleanup by making compact
  EP pack one block per route instead of a flat `routes * hidden` launch, but
  the warmed graph gate was performance-neutral. Further small padding-kernel
  rewrites need direct evidence before promotion; the next C1 sprint should
  prefer a larger grouped-GEMM/copy-shape design or resume the typed-KV/full-
  capture device-state path.
- **C2 Fix the graph-in-serving parity bug directly.** Graph mode changes the
  first token = a finite set of missing sync→event dependencies (461 fixed one).
  Diff eager vs graph dependency graph; close them all. Debuggable, not fundamental.
- **C3 Launch-count reduction is the universal eager-path lever.** Every fusion
  above (A5/A6/B2) helps the eager path that serves today, graphs or not. Given
  graph-in-serving is blocked, this is the only lever lifting the served ~12% util
  right now.
- **C4 Run the spill check** (still not done) on the HC-mix kernel, head_dim-512
  attention, and rank-major consume kernels: `-Xptxas -v` (registers/smem/spill) +
  ncu (occupancy, long-scoreboard stalls). If they spill, the rank-local rewrites
  leave perf on the floor.
- **C5 Replace hot host syncs with device events.** The hot engine path still
  contains dozens of `cudaDeviceSynchronize()` / `cudaStreamSynchronize()` calls
  across HC-current, decode-loop, attention projection/read/output,
  post-attention FFN, and EP compose. Replace structurally unnecessary host
  round-trips with `cudaEventRecord()` / `cudaStreamWaitEvent()` dependencies.
  This is both an eager-path cleanup and a graph-capture prerequisite. Sprint
  528 completed the output-head wait cleanup; Sprint 529 completed the
  attention-output projection handoff cleanup; Sprint 532 completed the
  promoted post-attention FFN input handoffs; Sprint 533 completed the
  promoted attention-projection handoffs; Sprint 534 completed the promoted
  attention-read raw/window handoffs; Sprint 535 completed the promoted
  HC-current final fill/pack handoff. Decode-loop, EP compose,
  typed-indexer/top-k, and diagnostic/control-only sync sites still need
  per-site review as part of C1/C2 ordering repair.

## D. Model-boundary NCCL cleanup

- **D1 Output-head A1 pattern.** Done in Sprint 527:
  rank-local stable reductions plus NCCL all-reduces/all-gather replaced the
  centralized prep. This is a structural/C1-prep promotion, not a direct
  throughput win at the measured shape.

## Recommended priority

| # | Idea | Domain | Why | Risk |
|---|---|---|---|---|
| Done | A4 finish rank-major consumers | HC 40% | Sprint 526 completed the remaining post-attention FFN shared/route consumers for the served path | Low |
| Done | D1 output-head A1 pattern | Model boundary | Sprint 527 removed GPU0-centralized output-head prep; timing regressed, but the capture surface is cleaner | Low |
| Done | C5 sync-point reduction pass 2 | attention output | Sprint 529 removed attention-output eager host stream waits from the promoted path | Low-Med |
| Done | B2 compact EP broadcast trim | EP 53% | Sprint 531 removed padded compact broadcast over-transfer without all-pairs SHM/P2P | Low |
| Done | C5 post-attention FFN handoffs | post-attention | Sprint 532 removed promoted-path post-attention FFN host waits with device-event ordering | Low-Med |
| Done | C5 attention-projection handoffs | attention projection | Sprint 533 removed promoted-path attention-projection host waits with device-event ordering | Low-Med |
| Done | C5 attention-read raw/window handoffs | attention read | Sprint 534 removed promoted-path attention-read raw/window host waits; typed-indexer/top-k remains separate | Low-Med |
| Done | C5 HC-current fill handoff | HC-current | Sprint 535 removed the promoted final fill/pack host wait with device-event ordering | Low-Med |
| Done | SPIKE B preflight/control | both | Sprint 536 recorded ptxas spill data, target selected-token control, sync/capture blocker counts, and reusable control artifact | Low |
| Done | C1 route-stable graph suffix replay | both | Sprints 539-540 restored cache hits, strict selected-token parity, and warmed request-window speedup; launcher default promoted with opt-out | Med |
| Done | C1 rejected padding knob cleanup | both | Sprint 549 removed static route caps, host-synced actual-route updates, and masked compact copy from active code; fixed-capacity route planning remains the supported graph-stable surface | Low |
| Done | C1 compact EP pack route-blocking | EP/compose | Sprint 550 preserved fixed graph-visible route shapes and changed compact EP pack so inactive padded routes skip hidden-wide work; the warmed gate was correctness/topology-clean but performance-neutral | Low |
| Done | C1 dynamic-position raw typed KV | full capture | Sprint 551 made graph-mode raw typed-KV store/load compute physical rows from `d_decode_position`; targeted smoke matched static row behavior exactly | Low-Med |
| Done | C1 emitted typed-KV dynamic physical row | full capture | Sprint 552 made graph-mode emitted compressed/indexer typed-KV runtime store/load compute physical rows from `d_decode_position`; targeted smokes matched static row behavior exactly | Low-Med |
| Done | C1 emitted typed-KV dynamic bounded row | full capture | Sprint 553 made graph-mode emitted compressed/indexer typed-KV runtime store/load compute the compact bounded row from `d_decode_position`; targeted smokes matched static row behavior at bounded row `1` | Low-Med |
| Done | C1 typed-history dynamic row-position load | full capture | Sprint 554 made graph-mode compressed/indexer typed-history reload derive historical source positions from `d_decode_position` plus bounded row inside the TP runtime; targeted smokes matched static row loads exactly | Low-Med |
| Rejected | C1 served full-capture position-key removal | full capture | Sprint 555 proved no-suffix full-capture cross-position reuse is still semantically unsafe even with compressed KV off; cache reuse worked but selected-token first token changed | Med-High |
| Done | C1 full-capture replay-probe localization | full capture | Sprint 556 showed plain no-suffix full capture preserves eager output, while immediate replay-probe changes it because the probe replays a captured full step on already-advanced live buffers | Med |
| Done | C1 no-suffix replay-probe guard | full capture | Sprint 557 made invalid no-suffix full-capture replay-probe fail loudly instead of replaying on already-advanced live buffers; promoted suffix replay is unaffected | Low |
| Done | C1 full-capture fresh-state replay validation | full capture | Sprint 558 split full-capture cache-miss/cache-hit validation so cache miss serves eager and captures only, while a later same-position request replays from fresh state; selected tokens matched but checksum drift remains | Med |
| Done | C1 full-capture checksum-drift localization | full capture | Sprint 559 fixed the replay-time final-HC host pointer swap mirror; same-position no-suffix full-capture replay now matches eager selected tokens and checksums at the reduced diagnostic shape | Med |
| Done | C1 emitted-row host metadata mirror | full capture | Sprint 560 mirrors compressed attention/indexer row counters, row-position arrays, and loaded-row metadata after successful no-suffix full-capture cache-hit replay; a one-off compressed-KV binary matched eager selected tokens/checksums with `43/43` replay hits | Med |
| Done | C1 emitted topology graph-stability | full capture | Sprint 561 makes graph-mode compressed-KV emitted/non-emitted topology stable by always enqueueing the emitted topology under graph capture and device-masking emitted kernels/copies from `d_decode_position`; emitted and adjacent non-emitted compressed-KV eager/replay probes matched selected tokens/checksums with `43/43` replay hits | Med |
| Rejected | C1 full-capture cross-position cache-key relaxation retry | full capture | Sprint 562 retried no-suffix full-capture cross-position reuse after Sprint 561; a two-request same-session probe matched, but a six-request same-session probe diverged on request three, so the candidate was removed and the position key remains required | Med-High |
| 1 | C1 residual captured-position state localization | full capture | Localize the request-three cross-position divergence from Sprint 562. Do not retry cache-key relaxation until remaining captured position-derived row/source arguments or host/device state are identified and made replay-dynamic. | Med-High |
| 2 | Larger executor/compose shape work | EP/compose | Sprint 550 shows the obvious compact-pack padding site is not a steady-state lever; any further padding work needs a grouped-GEMM/copy-shape design with direct evidence, not another tiny kernel rewrite. | Med-High |
| 4 | A5/A6 fusion | HC/attention | Converts rank-local structure into fewer launches | Low-Med |
| 5 | B2/B3/B4/B5 EP structural bets | EP 53% | B2 fusion, TP-expert A/B, routed/shared overlap, and correctness-preserving capacity balancing | Med |
| Deferred | B1 MTP — sidecar removal + specdec loop | EP 53% | Sidecar runs canonical MTP, not a truncation; cleanup is one ~200-LoC safetensors→GGUF converter + `tp-ep-pack-contract.c` extension + sidecar delete (3 sprints), then MTPBlock.forward (1 sprint), then TP/EP specdec loop (the actual throughput sprint). All after C5/B2/C1/tuning. | Med |

## Discipline

**Per-sprint: correctness only. Perf measurement aggregates at the tuning
sprint.** The structural changes in A4/D1/sync-reduction/EP-compact are each
clear wins over GPU0-centralized norming; per-sprint reference-shape runs add
cost (10–15 min each), noise (shape variance), and no decision value when the
expected magnitude per sprint is single-digit percent. The dedicated tuning
sprint at the end of the program measures the cumulative win.

### Per-sprint validation (minimal cost, catches real bugs)

- **Tolerance gate.** Selected-token + generated-sequence agreement ≥ `0.99`
  against the prior promoted control artifact. Reuse the artifact pointer;
  do not run a fresh control. Governed by `docs/sprints/VALIDATION_CONTROL_POLICY.md`.
- **Transport invariant.** `peer_copy_ops = 0` and `peer_copy_sys_bytes = 0`
  on the promoted path. Re-introducing a peer copy is a regression.
- **Scaffold confirmation.** Server log shows the intended path is the path
  taken (e.g., `rank_major_input=1`, `slot_major_ffn_norm=0` on every checked
  layer). Free, gives confidence the structural change actually fires.
- **Cleanup discipline.** Promotion = move into `engine/` (or
  `kernels/v100/`) and delete the flag + dead branch in the same commit. No
  drift.

### Exception: novel-risk sprints opt into perf measurement

For sprints whose **failure mode is "structurally landed but perf didn't
transfer to serving,"** opt into perf measurement inside the sprint. The
canonical case is **C1 piecewise graph capture** — sprints 415–463 had the
2.27× win in the smoke and zero transfer to serving. For that class, the
sprint plan names "decode tok/s at reference shape" as an in-scope gate
alongside the tolerance gate.

The default for everything else (A4, D1, sync-reduction, EP-compact, the
remaining Pattern A items) is correctness only.

### Tuning sprint (deferred — runs once at program completion)

Aggregates the measurement work that per-sprint validation deferred:

- Full reference-shape decode tok/s and request-window GPU util.
- Domain-table re-profile (HC-current vs EP vs other), which re-picks
  what's next.
- Shape envelope sweep — slot counts, context lengths, request counts —
  on the post-structural surface.
- `NCCL_ALGO` / `NCCL_PROTO` pinning per payload size.
- C4 spill check (`-Xptxas -v` + ncu) on the rank-local and HC-mix kernels.

### Carried-forward rules

- **No re-baselining.** Control is always the latest promoted artifact
  pointer unless a real invalidator (model / launcher / shape / flags /
  harness semantics) makes it non-comparable.
- **One-off smokes and temporary gates are evaluation-only.** A promoted
  result moves into the main path and deletes the smoke-only flag, the
  rejected branch, or the diagnostic scaffold in the same sprint unless
  there is a documented debugger-only reason to keep it default-off.

## One-line frame

With A4, D1, compact EP broadcast trim, C5 event handoffs, Sprint 536
preflight, Sprint 540 graph suffix replay promotion, Sprint 549 rejected
padding-knob cleanup, Sprint 550 compact EP pack route-blocking, Sprint 551
dynamic-position raw typed-KV, Sprint 552 dynamic-position emitted typed-KV
physical rows, Sprint 553 dynamic bounded rows, and Sprint 554 dynamic
typed-history row loads complete for the served/full-capture surface, Sprint
555 proved full-capture cross-position reuse is still unsafe without a first
divergence fix, Sprint 556 showed the immediate no-suffix replay-probe is
itself invalid because it double-applies the captured full step on live buffers,
Sprint 557 made that invalid probe fail loudly, Sprint 558 replaced the guard
with a fresh-state same-position replay validation path, Sprint 559 fixed the
final-HC host pointer swap missing from full-capture cache-hit replay, and
Sprint 560 mirrored compressed-row host metadata for same-position no-suffix
cache-hit replay and validated it with a one-off compressed-KV binary. Sprint
561 made emitted/non-emitted compressed-KV topology graph-stable and validated
both emitted and adjacent non-emitted positions against eager with exact
selected-token/checksum matches. Sprint 562 retried full-capture
cross-position cache-key relaxation and rejected it: adjacent replay matched,
but a six-request same-session probe diverged on request three, so the
candidate was removed and full capture remains position-keyed. The next ordered
work is localizing the residual captured-position state before any further
cache-key retry. MTP stays deferred until the ordered post-C1/tuning point.
