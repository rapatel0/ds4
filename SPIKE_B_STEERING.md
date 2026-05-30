# Spike B Decode-Optimization Steering (updated 2026-05-29 after Sprint 571)

Steering for the next TP/EP serving-throughput phase, off the de-confounded
steady-state reference (32 slots / 256K / 256 req / 64 tok/req, ~35.9 tok/s
server decode, ~889 ms decode domain).

**Updated baseline (Sprint 581, full-capture default promoted):** at 32 slots /
256K / 64 tok/req, the promoted no-suffix full-capture default delivers `26.8`
tok/s aggregate decode (`1.225x` over the suffix path it replaced; `2.34` vs
`1.53` per-request decode). Decode-domain gap attribution (eager, `14.4` ms/step):
**EP/MoE all-to-all = `65.2%`** (the dominant cost and the source of the
utilization "waves"; ~`0.75` tokens/expert leaves grouped-GEMM tiles nearly empty,
SMOCC ~`0.08-0.11`), attention ~`12%`, compose ~`11%`, HC-current ~`8%`, and
host-sync orchestration (route_upload/fill_pack/router_select, the PCIe/GPU0 skew)
only ~`5%`. So the throughput headroom is EP (B2 + MTP), not the host-sync/
output-head cleanup.

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

## Current reassessment after sprints 478-571

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
  Sprint 563 re-applied that failed relaxation only in a temporary remote
  diagnostic build with `--decode-stage-checksum-gate`. Sprint 566 then added
  a stronger diagnostic hash and comparable end-of-step `step_snapshot`
  records. That proved the Sprint 565 `hc_current.current_full_rank_major`
  clue was a timing artifact: eager observed the buffer immediately after
  HC-current, while replay observed it only after the graph completed. The
  comparable snapshot next pointed at `route_a`, but Sprint 567 proved that was
  scratch-state noise: graph-vs-graph control showed occurrence 1 differs only
  in full-buffer `route_a` while route totals/slots/weights, outputs, and token
  checksum all match. On occurrence 2, layer 0 still only has `route_a` scratch
  drift; layer 1 then diverges across current/output tensors. Sprint 568 found
  the actual bug: no-suffix full-capture replay recorded fixed final-HC
  input/output buffer pointers while eager advances logical HC state by swapping
  `d_final_hc_shard` / `d_hc_scratch_shard`. The replay path now rebases live
  HC contents into the captured input buffer before launch and restores host
  pointers to the captured input/output pair, allowing cross-position replay
  without stale HC reads. The six-request eager-vs-full-graph probe now matches
  selected tokens and checksums with `43` captures, `215` persistent replays,
  and zero invalidations. Sprint 569 then moved from reduced selected-token
  checks to serving metrology at `32` slots / `256K`: after a full-slot warmup,
  deterministic long-prompt measured serving output matched `32/32` generated
  token sequences between the promoted suffix-control leg and opt-in no-suffix
  full capture. The no-suffix full-capture request window improved generated
  throughput `12.603435 -> 16.807308` tok/s and median latency
  `81.205441s -> 60.873505s`. Sprint 570 extended the gate to `64` generated
  tokens/request and `128` measured requests after two full-slot warmup
  batches. Performance remained positive (`16.618822 -> 20.814267`
  continuation tok/s wall; median latency `132.083580s -> 103.958728s`), and
  both legs replayed `28724` persistent graph lines with zero peer/SYS hits.
  Correctness failed: `128/128` measured generated-token sequences diverged.
  Sprint 571 showed this is not a pure `64` token threshold. Recreating the
  Sprint 569 shape (`32` tokens, one warmup, one measured full-slot batch)
  diverged for `32/32` responses at continuation offset `1`; the longer Sprint
  570 prompt diverged immediately at offset `0` even with `32` tokens. Treat the
  Sprint 569 pass as a positive but insufficient signal, not reproducible
  promotion evidence. The next C1 target is early continuation replay/prompt-
  cache/coalescing state, compared only at request-level or same-logical-point
  diagnostics to avoid another timing-artifact chase.
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
| Done | C1 residual captured-position state localization | full capture | Sprint 563 used a temporary remote relaxed build plus `--decode-stage-checksum-gate`; first logged divergence was occurrence 1 at layer 1 `hc_current`, while layer 0 still matched | Med |
| Done | C1 layer-1 HC-current replay state guard | full capture | Sprint 564 rejected the cache-miss double-advance hypothesis and added the final-HC pointer identity to the full-capture graph cache key; position-keyed diagnostics and promoted suffix replay remain clean | Med |
| Done | C1 rank-major replay snapshot repair | full capture | Sprint 566 added stronger diagnostic hashes and comparable eager/replay `step_snapshot` records; the Sprint 565 rank-major HC-current diff is a timing artifact, not the first comparable state drift | Med |
| Done | C1 route replay boundary localization | full capture | Sprint 567 added route metadata snapshots and graph-vs-graph comparison; `route_a` full-buffer drift is scratch noise because route totals/slots/weights and outputs still match before layer-1 drift | Med |
| Done | C1 inter-layer current/HC pointer-buffer repair | full capture | Sprint 568 stores captured final-HC input/output buffer addresses and rebases live HC state into the captured input before replay; six-request eager-vs-full-graph selected-token/checksum parity passed with `215` cache-hit replays and zero invalidations | Med-High |
| Done | C1 serving parity/performance metrology | full capture | Sprint 569 ran deterministic warmed long-prompt serving at `32` slots / `256K`; no-suffix full capture matched `32/32` generated token sequences and improved request-window generated throughput `12.603435 -> 16.807308` tok/s versus the promoted suffix-control leg | Med-High |
| Rejected | C1 longer steady-state serving promotion gate | full capture | Sprint 570 kept the performance signal at `64` tokens / `128` measured requests (`16.618822 -> 20.814267` continuation tok/s wall), but generated-token sequences diverged for `128/128`; no default flip | Med-High |
| Done | C1 long-generation divergence localization | full capture | Sprint 571 showed the failure is not a pure `64` token issue: recreated `s569-shape` diverged `32/32` at continuation offset `1`, while `s570-prompt-32` diverged `32/32` at offset `0`; use request-level sequences, not timing-shifted tensor logs, as the oracle | Med-High |
| Done | C1 early continuation replay-state repair | full capture | Sprint 573 showed this was the wrong target: the early-continuation/offset-`0` divergence the prior gates chased is first-token nondeterminism (eager gives `6` distinct continuations for `32` identical prompts; identical control runs differ `3/32`, all offset `0`). The real full-capture bug is a clean offset-`28` cluster, not step `0 -> 1`. | Med-High |
| Done | C1 late-position comp-emit replay repair | full capture | Sprint 574 disproved comp-emit (emit off on served path) and found the divergence is position-dependent: benign at `250000` (`7/32` offset `28`), catastrophic at `250064` (`32/32` offset `1`). RoPE/raw-window are device-dynamic and ruled out. Serving stage-checksum is misaligned (one token/call -> all `step 0`). | Med-High |
| Done | C1 full-capture position-derived state localization | full capture | Sprint 575: no per-slot value to localize. Single-slot (`SLOTS=8`, one request) full-capture replay is bit-exact with eager at `250000` (32 tok) and `250064` (4 tok). The divergences are batch nondeterminism; `250064` floor `control-A` vs `control-B` is `32/32`. | Med-High |
| Done | C1 full-capture promotion under noise-aware gate | full capture | Sprint 576: not promotable. Logit-space floors show eager-vs-eager is bit-identical on matched tokens (`7/32` discrete router-tie flips) but full-vs-full diverges `32/32` with logit Δ up to `3.63`. Full capture is batch-unstable: a real defect, not tolerated noise. | Med |
| Done | C1 full-capture batch-instability localization | full capture | Sprint 577: full-vs-full logit floor scales with active routed tokens (1-2 active ~bit-exact, `8` -> Δ 1.21/`8` flips, `32` -> Δ 3.63/`32` flips). The 8-slot graph is constant across these, so it is not a static pointer/buffer bug; it tracks active route count -> accumulation-order nondeterminism in the graphed route/compose. compute-sanitizer OOMs before decode and no smoke reproduces the graph-replay path, so it cannot reach this bug. | Med-High |
| Done | C1 captured-compose nondeterminism: FIXED | full capture | Sprint 579: runtime per-stage diff localized the divergence to the captured `compressed_kv` stage's `attn_q_b.d_out` on `dense_stream`, caused by `enqueue_rank_streams_wait_after_dense_streams` being a dense->rank-only barrier (eager fully drains both streams). Made it bidirectional (`output_head.cu`); full-vs-full sequence mismatch went `8/8 -> 0/8`. Determinism defect fixed; correctness-preserving (ordering-only). | Med-High |
| Done | C1 full-capture serving promotion gate | full capture | Sprint 580: gate passed and **no-suffix full capture is now the promoted launcher default**. At `32` slots / pos `250000`, parity was perfect within the determinism floor (floor `0`, full-vs-full `0`, eager-vs-full `0`) and full capture was `1.203x` wall / `1.518x` decode faster than suffix-control (median latency `42.09s -> 34.98s`). Launcher uses a 3-mode `DS4_V100_TP_EP_DECODE_GRAPH_MODE` (`full` default; `suffix`/`eager` opt-outs; `GRAPH_SUFFIX_REPLAY` back-compat override). | Med |
| Done | Tuning sprint (reference-shape baseline + gap attribution) | both | Sprint 581: new baseline `26.8` tok/s agg decode at 32 slots / 256K / 64 tok/req on the full-capture default (`1.225x` over suffix). Gap attribution (eager, `14.4` ms/step): EP/MoE all-to-all `65.2%` (the waves; ~0.75 tok/expert -> empty tiles, SMOCC ~0.08-0.11), attention ~12%, compose ~11%, HC ~8%, host-sync ~5%. Throughput headroom is EP, not host-sync. Shape-envelope / NCCL-pinning / C4-spill levers deferred (situational; EP dominates today). | Med |
| In progress | MTP (B1) — the EP-fill throughput bet | EP 65% | EP is 65% of decode *because* of sub-1-token-per-expert. MTP (verify K draft tokens) makes each step see (K+1)x tokens -> fills grouped-GEMM tiles, raises occupancy, amortizes the all-to-all -> attacks the 65% at the root. **(SPRINT 584 RESULT): weight integration + MTP forward body DONE + validated; specdec loop is Sprint 585.** Sprint 584 built + validated, on the pod, the entire EP=8 MTP weight integration (steps 1-4) AND the MTP forward body executing through the multi-rank EP decode (`decode_pass=1`, EP-split experts, serving byte-intact). All weight loaders built (`open_mtp_expert_bindings`, `open_mtp_nonexpert_bindings`, `load_mtp_hc_layer43`, `load_mtp_dense_layer43`, `load_mtp_output_head`; `kLayers`->44 + guards + struct `[44]`). The ONLY remaining piece is the **TP/EP speculative-decode loop** -- a from-scratch multi-rank serve-path decode driver (draft K via the MTP forward -> verify K+1 via the existing `ds4_replay_verify_token_block` -> accept/reject), integration point `appliance/http_server.cu` generation loop, validated by a live serving run (acceptance rate + throughput). Scoped as **Sprint 585** (`docs/sprints/SPRINT-585.md`); to run as a fresh focused session. Below text is historical.

**(a) MAJOR CORRECTION (Sprint 584): the existing MTP code is LP-era, not EP=8.** The appliance had an LP=8 (layer-parallel: each layer's 256 experts on one GPU) first pass, then the current EP=8 (expert-parallel: 32 experts/rank of every layer, all-to-all). `engine/mtp_step.cu` + the sidecar are the LP MTP (Q4_K, SEPARATE gate/up, self-contained non-EP compute). The Sprint 582/583 converter (`tools/mtp-pack-fragment.c`) inherited the LP framing (separate gate/up, single-GPU emission) and is NOT the EP=8 path. The EP=8 MTP must be layer 43 of the unified EP model, reusing the per-layer EP routed-FFN dispatch (`ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32` with the FUSED `turbomind_gate_up_view`, mxfp4, EP-split). See `docs/sprints/SPRINT-584.md` for the full code-grounded EP=8 plan (converter fused-gate_up -> EP pack emission -> runtime bind -> EP forward -> sidecar delete -> specdec loop). The remaining text below (b/c and prior weight-pack notes) is retained for history but is LP-framed.

**(a-legacy, LP-era) Weight-pack: converter + EP8 CONTRACT done/validated; EP8 SHARD EMISSION still pending (Sprint 584 re-validation).** The experts are packed-FP4+E8M0 (not lossy-I8 as first assumed) -> a **lossless** re-pack to mxfp4/f8_e4m3_b128. The new artifact is the converter `tools/mtp-pack-fragment.c` (reads the safetensors, re-packs all families, stacks experts, emits a convention-compliant GGUF + Sprint-002 manifest; re-pack round-trips 0-mismatch on real weights). The pipeline handles layer 43 **generically with no code changes**: `mtp-contract2/tp-ep-pack-contract.tsv` is a correct TP8/EP8 plan -- `24` `ep_expert` rows (32 experts/rank, `expert_first` 0..224), `72` `dense_tp`, `512` `kv_shard`, `64` `replicated_control`. **But the shard emission that ran was SINGLE-GPU**: `appliance-pack`/`turbomind-pack` were driven from the standalone `owning_gpu=0` pack-index, so all 256 experts landed in `gpu0.weights` and `gpu1-7.weights` are 0 bytes. The earlier "weight-pack DONE end-to-end" claim was overstated. Remaining weight bits: **(i)** drive `appliance-pack`/`turbomind-pack` from the EP8 contract (contract -> per-rank pack-index translation) to emit 8 non-zero shards; **(ii)** `runtime_pack.cu` layer-43 bind + sidecar delete. **(b)** MTPBlock.forward in `engine/` (Phase A, kernels). **(c)** TP/EP specdec accept/reject loop (Phase B, the throughput sprint, multi-sprint, opts into perf). See `MTP_IMPLEMENTATION.md`. | High |
| 2 | B2 compact EP dispatch+combine fusion | EP 65% | Pre/parallel-to MTP: fuse dispatch+grouped-GEMM+weighted-combine into 1-2 kernels with device-side offsets; replace the variable-size compose movement with a ring-compatible / statically bucketed NCCL collective (not all-pairs P2P -> SHM budget, Sprint 530). Directly trims the 65% EP orchestration. Needs a grouped-GEMM/copy-shape design with direct evidence, not another tiny kernel rewrite (Sprint 550). | Med-High |
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
candidate was removed and full capture remains position-keyed. Sprint 563
localized the first logged divergence to layer 1 `hc_current` after layer 0
still matched under the temporary relaxed build. Sprint 564 rejected the
cache-miss capture-as-result hypothesis, added final-HC pointer identity to the
full-capture cache key, and showed a remote-only relaxed retry still diverges
on request three; the remaining blocker is not the final-HC pointer key alone.
Sprint 565 reran stage checksums after that guard and initially moved the first
sequential diff to layer 0 `hc_current.current_full_rank_major`, but Sprint 566
proved that was a mid-stage-vs-post-graph timing artifact. The comparable
end-of-step snapshot then showed `route_a` drift after an otherwise-correct
cross-position replay, but Sprint 567's graph-vs-graph control proved that
full-buffer route scratch drift is not semantic: route totals/slots/weights and
outputs still match before the next replay drifts from layer 1 onward. Sprint
568 repaired the inter-layer current/HC pointer-buffer bug by rebasing live HC
state into the captured full-graph input buffer before replay. The six-request
eager-vs-full-graph probe now matches selected tokens/checksums across positions
with `43` captures, `215` persistent replays, and zero invalidations. Sprint
569 showed a strong warmed-serving opt-in throughput signal with matching
generated token sequences, Sprint 570 rejected the default flip because a
longer steady-state gate diverged for `128/128` measured responses, and Sprint
571 localized the failure to early continuation replay state rather than a pure
64-token threshold. Sprint 572 rejected the cache-miss capture-as-served-result
repair: `s569-shape` and `s570-prompt-32` still diverged for `32/32` measured
responses. The next C1 work is same-logical-point instrumentation around
continuation step `0 -> 1` -- prompt-cache/coalescing metadata, slot order,
decode input token, selected-token handoff, and full-capture HC/current rebase
timing -- using request-level generated sequences or same-logical-point logs as
evidence. Sprint 573 ran the determinism baseline the prior gates lacked and
reframed the whole problem: the serving decode path is nondeterministic at the
first token (pure eager gives `6` distinct continuations for `32` identical
prompts; two identical promoted-control runs differ on `3/32`, all at offset
`0`), promoted suffix-control matches eager within that floor, and full capture's
real divergence is a clean cluster of `7` requests at **offset `28`** that
appears in no noise comparison. The offset-`0` full-capture mismatches were just
nondeterminism. So Sprints 570-572 conflated first-token noise (compounded over
sequence length by the exact-equality oracle) with the real bug, and the real
bug is **late-position compressed-KV emit-replay state**, not the "step `0 -> 1`"
early continuation 571/572 chased. Even (ratio-4) layers emit at
`(position+1)%4==0`; with `position 250000 ≡ 0 (mod 4)` the emit boundaries land
at generation offsets `3..27`, and offset `28` is the first token after the
offset-`27` emit. Two durable corrections: (1) every full-capture gate must run
an identical-config `control-A` vs `control-B` determinism floor and judge
against it and against `eager`, never against exact equality with one control
run; (2) the next C1 target is same-logical-point instrumentation at the ratio-4
compressed-KV emit boundary around offset `27 -> 28` (comp ring-row index,
load/store decision, row-position metadata at capture vs cache-hit replay), then
the narrow repair. Sprint 574 then disproved the comp-emit guess (compressed-KV
emit is off on the served path) and found the divergence is strongly
position-dependent: at `position 250000` full capture is benign (`7/32` at offset
`28`), but at `250064` it breaks immediately at offset `1` for `32/32` requests
(`32` distinct sequences for identical prompts). So `250000` -- used by every gate
since Sprint 569 -- is an unusually clean position that understates the bug; the
captured graph carries a position-derived value, only correct near the capture
position, whose error scales with replay drift. RoPE and raw-window row selection
are device-dynamic (read live `d_decode_position`) and ruled out. The per-layer
`--decode-stage-checksum-gate` is wired but misaligned with serving (one token per
call -> every token logs as `step 0`), so the next localization must use a
multi-step capture/replay probe (`decode_steps > 1`) at `250064`, diffing
eager-step-k vs full-replay-step-k per `(layer, stage)`. Then the narrow repair,
validated against the determinism floor at `250064` and `250000`. Sprint 575
resolved the whole question and **retracts the Sprint 574 catastrophic claim**:
single-slot testing (`SLOTS=8`, one active request) shows full-capture replay is
**per-slot bit-exact with eager** -- identical generated sequences at `250000`
(32 tokens) and `250064` (4 tokens). The `250064` "catastrophe" was measurement
instability: with the floor, `control-A` vs `control-B` (identical config) also
diverges `32/32` there (`control-B` collapsed to `5` distinct sequences). So the
C1 full-capture replay is **correct**; every divergence since Sprint 570 is
batch/concurrency nondeterminism (reduction order across concurrent slots, present
in identical-config controls: `3/32` at `250000`, `32/32` at `250064`), and the
blocker was an exact-equality oracle applied to a nondeterministic serving path.
Reframed promotion gate: (1) per-slot single-request bit-exactness (passes today);
(2) characterize whether full capture amplifies concurrent-batch variance versus
the `control-A`/`control-B` floor at matched concurrency -- if within the floor,
promote for the `1.25-1.48x` decode win; if it materially amplifies variance,
that is the real (separate, smaller) issue. Sprint 576 answered that with a
logit-space measurement (new `tp_ep_decode_top1_logit` diagnostic) and
**partly retracts Sprint 575**: it is the real issue, not a small one. With
warmup disabled so all legs share positions, identical eager runs are
**bit-identical** on matched tokens (logit Δ = 0; only `7/32` discrete
router-tie flips), but two identical **full-capture** runs diverge on `32/32`
slots with continuous logit Δ up to `3.63` (`p50 0.26`). So the inherent serving
noise is tiny and discrete (MoE router ties, `router.cuh` atomics), and full
capture adds large run-to-run **batch instability** that eager does not have.
Full capture is per-slot bit-exact (single slot) but batch-unstable under
concurrency: a real defect, **not promotable as-is**. The Sprints 570-574
divergences were this instability conflated with the small eager floor. Next:
localize the instability (EP-compose `atomicAdd` in `kernels/v100/compose.cuh`
replayed nondeterministically, route/compose scratch not deterministically
reset, or HC ping-pong) via the logit diagnostic plus a deterministic-reduction
probe; if a deterministic compose collapses full's floor toward the eager floor,
the source is confirmed. Sprint 577 localized it: the full-vs-full logit floor
**scales with active routed tokens** (1-2 active ~bit-exact; `8` -> Δ 1.21,
`8/8` flips; `32` -> Δ 3.63, `32/32`). The 8-slot graph structure is identical
across those runs, so it is **not a static pointer/buffer bug** -- it tracks
active route count, i.e. accumulation-order nondeterminism in the graphed
route/compose. A `compute-sanitizer --tool initcheck` attempt confirmed the tool
is unusable here: it initialized NCCL/8-GPU fine but **OOMed during expert load**
(shadow memory + ~24 GB model > 32 GB) before reaching decode, `0 errors` in the
load path, and no smoke test exercises the cudagraph replay path so there is no
low-memory repro. So the next step is the **deterministic-compose rebuild**
(`nccl_reduce_scatter_compose_gate` or a non-atomic combine; compile-time, needs
a rebuild): rerun full-vs-full and check the floor collapses toward eager.
Sprint 579 **fixed it**: per-stage runtime diff (matched positions via a leading
dummy leg) localized the first divergence to the captured `compressed_kv` stage's
`attn_q_b.d_out` on `dense_stream` at the first ratio-4 emit position. Root cause:
`enqueue_rank_streams_wait_after_dense_streams` (`output_head.cu`) was a
**dense->rank-only** barrier, while the eager path it substitutes for fully drains
**both** streams; the one-directional edge let `dense_stream` lap `rank_stream`
across replays and race the in-place writes to `attn_q_b.d_out`. Making the
barrier **bidirectional** (record a rank-stream event, make `dense_stream` wait on
it) took full-vs-full sequence mismatch `8/8 -> 0/8` (two identical full-capture
runs now bit-identical). The fix is ordering-only (correctness-preserving) and is
in the shared captured-region helper, so it strengthens every cudagraph path. Next
is the standard serving parity/perf promotion gate vs the eager floor (full
capture is now deterministic, so the `1.25-1.48x` decode win is on a sound path).
Sprint 580 ran that gate and **promoted no-suffix full capture to the launcher
default**: at `32` slots / pos `250000` parity was perfect within the determinism
floor (floor `0`, full-vs-full `0`, eager-vs-full `0`) and full capture was
`1.203x` wall / `1.518x` decode faster than suffix-control (median latency
`42.09s -> 34.98s`). The launcher now selects mode via
`DS4_V100_TP_EP_DECODE_GRAPH_MODE` (`full` default; `suffix`/`eager` opt-outs;
legacy `GRAPH_SUFFIX_REPLAY` still overrides). So **C1 is complete** -- both the
suffix-replay and the (now-default) full-capture graph paths are correct and
promoted. The next ordered item is the **tuning sprint** (reference-shape perf +
domain table, shape envelope, NCCL pinning, C4 spill), then the MTP bet. MTP
stays deferred until that post-C1/tuning point.
