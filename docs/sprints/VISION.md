---
created: 2026-05-17
last_updated: 2026-05-26
last_updated_by: sprint-410
revision: 423
archived_previous: docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md
---

# Vision: DS4 V100 TP/EP Appliance

## North Star

Build a DeepSeek V4 Flash appliance for the 8x V100-SXM2-32GB cluster that
runs the source quantized model from pure device-resident packs, preserves
quality, and reaches practical high-throughput serving through a native
TP/EP topology.

Hard cut: from this revision forward, no new work is spent on PP/layer-split
variants. The old layer-scheduled appliance remains only a frozen correctness
and throughput baseline. All new implementation work targets TP/EP. MTP is
deferred until TP/EP serving is operational and benchmarked.

Target topology:

```text
8x V100:
  pipeline parallel = 1
  tensor parallel   = 8
  expert parallel   = 8
  KV cache          = sharded
  slots target      = 32
  context target    = 256K minimum
  model path        = source quantized, device resident
```

Every GPU should participate in every layer. Dense paths are tensor-parallel.
Routed MoE paths are expert-parallel, using the existing low-bit TurboMind /
CUTLASS kernel work where it helps. The execution goal is to make decode look
like batched mat-mat work over active slots, not single-slot mat-vec work and
not a serial layer-chain.

## Active Performance Thesis

`TEMP_THROUGHPUT_PROMPT.md` is now the controlling throughput plan. The current
evidence says the TP/EP serving path is launch/synchronization fragmented:
server decode stays roughly `97-100` aggregate tok/s and average GPU
utilization stays around `10%` from `1` to `32` active requests at the target
`32` slot / `256K` shape. Sprint 377's fresh serving baseline at the same
shape is worse after the fully typed long-context path is enabled:
`88.372350` server decode tok/s, `40.157540` client generated tok/s, and
`7.972222%` average GPU utilization for `32/32` HTTP 200 responses. That makes
the immediate priority launch-count and sync-elimination work in the typed
attention/KV path, not another broad dtype conversion.

The performance program is intentionally isolated:

| Priority | Gate | Status | Decision |
|---:|---|---|---|
| 1 | `--async-output-gate` | Complete | Rejected as default; preserved parity but regressed server decode |
| 2 | `--decode-cudagraph-gate` | Complete | Rejected; stream capture is blocked by pervasive `cudaMemcpyPeerAsync` transport |
| 3 | `--batched-paged-attn-gate` | Complete | Rejected as narrow load target; row planner remains diagnostic-only |
| 4 | `--compact-moe-decode-gate` | Complete | Promoted for real model-router compact compose; response tokens matched and HTTP serving improved |
| 5 | `--fused-gated-silu-gate` | Complete | Not promoted; generic epilogue changes token, DS4-clamped ABI is fast in EP-only isolation but resident serving A/B fails before the gate |
| 6 | `--tp-experts-ab-gate` | Complete measurement | Do not integrate yet; TP8 fails correctness, TP4 is correct but reduction erases the win |
| 7 | `--fp8-e5m2-kv-gate` | Complete diagnostic | Correct and promising in short A/B, but not promoted pending longer parity and VRAM margin |
| 8 | `--tp-hc-current-input-fused-fill-pack-gate` | Complete diagnostic | Rejected; direct remote-load fusion preserved first token but regressed decode and HC fill/pack sharply |
| 9 | `tools/ds4-v100-tp8-layer-proxy --algo nccl` | Complete measurement | Promoted as the proxy path for true TP hidden all-reduce; serving defaults unchanged until a real TP dense/expert boundary exists |
| 10 | `--true-ds4-attention-output-nccl-allgather-gate` | Complete diagnostic | Correct at 16 slots, but rejected for target 32-slot/256K serving because NCCL communicator overhead triggers OOM |
| 11 | `--tp-hc-current-input-nccl-allgather-gate` | Complete diagnostic | Correct at 16 slots, but rejected: target 32-slot/256K OOMs and smaller shape regresses HC gather/decode |
| 12 | `--nccl-min-free-mib` / `DS4_V100_TP_EP_NCCL_MIN_FREE_MIB` | Complete harness guard | Promoted; NCCL candidates now fail early when communicator overhead leaves insufficient 32-slot/256K VRAM reserve |
| 13 | `tools/ds4-v100-tp-ep-nccl-kv-matrix.py` | Complete measurement | FP8 E5M2 KV does not reclaim VRAM; target HC-current NCCL still fails at 1114 MiB free vs 1536 MiB reserve |
| 14 | `tools/ds4-v100-tp-ep-vram-ledger.py` | Complete analysis tool | Promoted; next NCCL memory sprint must pair lazy output head with GPU0 HC-control residency reduction |
| 15 | `--diagnostic-output-head-lazy-gate` | Complete diagnostic | Correct but not promotable; lazy output-head preserves first token but leaves only 68 MiB free in control and HC-current NCCL still OOMs before first-token completion |
| 16 | Exact attention compressed-KV state layout | Complete memory fix | Promoted; target 32-slot/256K HC-current NCCL now completes with first token 54639, but reserve still fails at 386 MiB free after lazy output-head |
| 17 | HTTP lazy output-head | Complete prototype serving path | Promoted for prototype serving; 32/32 HTTP chat works at 32 slots/256K, HC-current NCCL also serves but still fails 1536 MiB reserve |
| 18 | Post-close lazy output-head VRAM checkpoint | Complete diagnostic | Keep telemetry; closing lazy output head recovers only ~136 MiB and HC-current NCCL still fails reserve at 520-522 MiB free |
| 19 | Skip unused TP-runtime comp-state arena | Complete memory fix | Promoted; HC-current NCCL now passes target 32-slot/256K reserve with 2240-2242 MiB post-close free |
| 20 | `tools/ds4-v100-tp-ep-nccl-http-ab.py` | Complete promotion harness | Promoted; target 32-slot/256K/32-token HTTP A/B passed readiness/parity and HC-current NCCL improved server decode 101.897890 -> 107.723452 tok/s |
| 21 | `--mtp-decode-gate` | Deferred multiplier | Add only after base TP/EP decode has stable metrology and launch strategy |

Promotion requires a same-binary V100 A/B at the real serving shape, unchanged
first token/checksum, and improved GPU utilization or server decode tok/s.
Rejected gates stay opt-in diagnostics. A sprint that cannot produce either a
promotion or a concrete blocker is not complete.

This sequence intentionally pauses CUDA graph work. Sprint 376 proved the
current peer-copy transport is not stream-capture-compatible on the V100 stack:
after helper sync blockers were removed, capture failed first on HC-current
`cudaMemcpyPeerAsync` and then on attention-projection peer copies. Reopening
graphs requires a separate P2P transport replacement plan; it is not the next
throughput sprint.

The near-term implementation focus is therefore:

1. Keep the 32-slot / 256K serving shape memory-admitted before optimizing it.
   Sprint 382 added startup VRAM admission/telemetry with
   `--vram-report`, `--vram-min-free-mib`, and launcher defaults
   `DS4_V100_TP_EP_VRAM_REPORT` /
   `DS4_V100_TP_EP_VRAM_MIN_FREE_MIB=64`. The V100 direct proof at
   `32` slots / `256K` recorded `vram_failures=0`, `vram_min_free_mib=1754`,
   and `vram_max_used_mib=30739`; a synthetic unsafe threshold fails cleanly
   before serving readiness. Future throughput sprints should keep this guard
   enabled so regressions show up as admission failures, not late CUDA OOMs.
   Sprint 402 added the NCCL-specific variant of this rule:
   `--nccl-min-free-mib` and `DS4_V100_TP_EP_NCCL_MIN_FREE_MIB`, defaulting to
   `1536 MiB` only when an NCCL serving gate is active. The target
   `32` slot / `256K` HC-current NCCL candidate now fails explicitly at
   `nccl_after_output_head` with `1114 MiB` free, while non-NCCL control
   remains admitted at `1746 MiB` and the smaller `16` slot NCCL diagnostic
   remains admitted at `3820 MiB`.
   Sprint 403 proved that `--fp8-e5m2-kv` does not create additional target
   headroom for NCCL: the TP runtime already defaults to FP8 E4M3 block-128 KV,
   so E5M2 is a format flavor switch, not an F16-to-FP8 allocation reduction.
   The combined E5M2 KV + HC-current NCCL case still failed at `1114 MiB`
   free against the `1536 MiB` NCCL reserve.
   Sprint 404 converted this into a per-GPU memory target: HC-current NCCL is
   short by `422 MiB` on GPU0, `74 MiB` on GPU1, `54 MiB` on GPU4, `70 MiB` on
   GPU5, and `22 MiB` on GPU6. Output-head residency costs `130-134 MiB/GPU`;
   HC controls cost `372 MiB` on GPU0 only. Therefore the next NCCL memory
   implementation should be paired: lazy/on-demand output-head residency plus
   streaming or shrinking GPU0 HC-control residency. Either change alone is
   insufficient at the target shape.
   Sprint 405 implemented the lazy diagnostic output-head half and confirmed
   that alone is insufficient. The direct non-NCCL control preserved first
   token `54639` with `97.034724` generated decode tok/s and `1880 MiB` free
   before opening the lazy head, but the lazy output-head checkpoint left only
   `68 MiB` free on GPU0. The HC-current NCCL + lazy-output-head case then
   failed before first-token completion with CUDA OOM at compressed KV state
   allocation on layer 5; NCCL plus HC controls left only `1248 MiB` free after
   `after_hc_controls`. Keep lazy output-head diagnostic-only. The next memory
   implementation must reduce/stream GPU0 HC-control residency and compressed
   KV transients, not just move the output-head allocation later.
   Sprint 406 corrected the largest compressed-KV transient waste: attention
   compressed state now uses exact per-ratio geometry instead of allocating
   every ratio layer as `128 x 1024` floats. Ratio-4 layers use `8 x 1024`;
   ratio-128 layers use `128 x 512`. This raised non-NCCL lazy-output-head
   free VRAM from `68 MiB` to `1018 MiB` and changed target HC-current NCCL
   from layer-5 OOM into a completed first-token diagnostic with token
   `54639`, generated decode `89.952595` tok/s, and continuation decode
   `100.096637` tok/s. It still fails the production NCCL reserve after lazy
   output-head: GPU0 has `386 MiB` free and all 8 GPUs are below `1536 MiB`.
   The next implementation should make lazy/on-demand output-head compatible
   with HTTP serving and continue peak-memory reduction before promoting NCCL.
   Sprint 407 completed the HTTP side of that step. Lazy output-head now works
   for HTTP decode while prefill skips logits. At `32` requests / `32` slots /
   `256K`, non-NCCL HTTP lazy serving returns `32/32` responses, first token
   `83480`, response-0 sequence `[83480, 79768]`, server generated decode
   `108.683003` tok/s, and `1018 MiB` minimum free VRAM. HTTP HC-current NCCL
   also returns `32/32` responses with the same token sequence and
   `110.879994` generated decode tok/s, but still fails production reserve:
   `386 MiB` free after lazy output-head versus the `1536 MiB` NCCL threshold.
   The appliance now has a target-shape prototype serving path; next work must
   reclaim output-head peak memory or otherwise pass the NCCL reserve before
   promoting NCCL.
   Sprint 408 split that measurement by adding a post-close lazy output-head
   checkpoint. Direct HC-current NCCL + lazy output-head at the target shape
   completed with first token `54639`, `96.275816` generated decode tok/s,
   `386 MiB` free before close, and `522 MiB` free after close. HTTP
   HC-current NCCL + lazy output-head returned `32/32` HTTP responses, first
   token `83480`, response-0 sequence `[83480, 79768]`, `112.666647` server
   generated decode tok/s, `386 MiB` free before close, and `520 MiB` free
   after close. Closing the lazy output head recovers only `134-136 MiB`, so
   output-head timing is not the primary remaining reserve lever. Keep
   HC-current NCCL diagnostic-only; the next memory implementation needs to
   reclaim persistent decode state and GPU0-heavy controls.
   Sprint 409 removed the largest concrete unused allocation from the target
   path: the TP-runtime compressed-state arena. That arena allocated
   `1803550720 B/GPU` but was not used by any current TP-runtime row store/load
   path. With `DS4_V100_TP_EP_TP_RUNTIME_SKIP_UNUSED_COMP_STATE=1`,
   `comp_state_bytes_per_gpu` reports `0`, `after_hc_controls` min free rises
   from `1248 MiB` to `2968 MiB`, and `after_lazy_output_head_close` rises
   from `520-522 MiB` to `2240-2242 MiB`. Direct HC-current NCCL preserves
   first token `54639` and passes the `1536 MiB` reserve. HTTP HC-current NCCL
   returns `32/32` responses, first token `83480`, response-0 sequence
   `[83480, 79768]`, `113.117381` server decode tok/s, and zero NCCL reserve
   failures. A sampled HTTP repeat passed readiness with GPU samples, resident
   KV, typed KV, compact MoE, checksums, and `vram_failures=0`. The launcher
   and profile harness now default this skip on.
   Sprint 410 turned the admission proof into a permanent HTTP A/B promotion
   gate. At `32` requests / `32` slots / `256K` / `32` generated
   tokens/request, control and HC-current NCCL both passed readiness and
   response parity matched `32/32` artifacts. HC-current NCCL improved server
   generated decode from `101.897890` to `107.723452` tok/s and continuation
   decode from `101.682616` to `107.545644` tok/s, with `2106 MiB` minimum
   free VRAM and zero failures. Promote
   `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1` as a launcher/env
   default. Caveat: client generated throughput regressed from `17.223947` to
   `16.627120` tok/s and sampled average GPU utilization stayed low, so this
   is not the final utilization lever.
2. Use the Sprint 383 matrix as the current before/after performance baseline.
   At `32` configured slots, `256K`, `position=262080`, and `32` generated
   chat tokens/request, active requests `1,4,8,16,32` all pass with
   `vram_failures=0` and `vram_min_free_mib=1754`. Client aggregate tok/s
   scales from `1.321769` to `43.853691`, but server decode remains flat at
   roughly `92-98` tok/s and average GPU utilization remains below `10%`.
   This confirms the next sprint should target steady-state launch/sync and
   GPU0-heavy orchestration, not admission or more active-slot batching.
3. Treat Sprint 384 as the quality-preserving serving baseline. Sprint 383
   measured launcher defaults, but DS4 intelligence requires model-router
   routing. With `--model-router-routes --compact-moe-decode`, the same
   `32` slot / `256K` matrix completes with `vram_failures=0`, server decode
   around `77-82` tok/s, and `32`-request client throughput `38.554075` tok/s.
   Future optimization should A/B against this real-router baseline unless
   the sprint is explicitly synthetic/diagnostic.
4. Keep compressed dense host stats out of the production default path. Sprint
   389 revalidated the existing skip-dense-stats gate against the current
   real-router compact-MoE TP/EP baseline at `32` slots / `256K`. Direct
   generated decode improved from `91.869507` to `102.871437` tok/s with the
   same first token `98751`; HTTP chat server decode improved from
   `89.709430` to `103.758804` tok/s, client throughput improved from
   `42.183007` to `44.592824` tok/s, first token stayed `83484`, all `32`
   generated token sequences matched, and the response checksum stayed
   `17913667583206000416`. The launcher now defaults
   `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1`, with explicit
   `=0` preserving the diagnostic path; the profile harness is aligned and
   exposes `--disable-skip-compressed-dense-stats` for future control runs.
   This is diagnostic work removal, not a dtype change.
5. Use permanent HTTP response parity for future serving A/B promotions.
   Sprint 390 added `tools/ds4-v100-http-response-parity.py` and validated it
   against Sprint 389 artifacts: `32/32` matched response pairs, same generated
   token sequences, same selected tokens, same generated text, and same DS4
   checksum. The tool also fails non-zero on a mutated generated-token fixture.
   Future HTTP A/B sprints should attach this JSON summary instead of relying
   on manual response inspection.
6. Use permanent HTTP readiness checks for future serving promotion gates.
   Sprint 393 added `tools/ds4-v100-http-readiness-check.py`, a single-case
   artifact checker for the target TP/EP serving shape. It validates response
   files, `summary.json`, `status.json`, generated-token sequence length,
   resident KV/HC metadata, typed DS4 KV gates, compact MoE, DS4 checksums,
   GPU-util samples, prompt-soak metadata, and VRAM admission. The checker
   passed on Sprint 392's real V100 multi-prompt control run at `32` requests /
   `32` slots / `256K` / `32` generated tokens with `106.390802` server decode
   tok/s, `38.912861` client generated tok/s, `9.772727%` average GPU
   utilization, first token `83484`, `vram_failures=0`, and `1746 MiB` minimum
   free VRAM. It also passed on the E5M2 candidate artifact and failed
   non-zero on a mutated token/checksum fixture. Future performance sprints
   should run this checker in addition to response parity before promoting a
   default.
7. Continue reducing the real-router route planning boundary, but shift away
   from H2D upload count after Sprint 386. Sprint 385 split the broad
   FFN/router bucket and removed unused legacy single-route-index uploads on
   the compact-MoE path. Sprint 386 then packed the compact route plan into
   one H2D upload per destination GPU. Direct real-router decode improved
   from `68.544741` to `74.838601` tok/s while route upload dropped from
   `44.079759` to `10.241125` ms. The `32` request real-router HTTP case
   preserved first token `83484`, improved server decode from `85.792845` to
   `91.778174` tok/s, and reduced route upload from `38.837019` to
   `6.796221` ms, though client aggregate tok/s moved from `42.427324` to
   `40.302457` in the single run. The remaining measured router cost is
   router dense/select, still about `27.8` ms per all-layer decode step.
   Sprint 387 proved that router dense/select can be reduced locally with a
   default-off cuBLAS SGEMM gate (`27.752540` to `4.959189` ms in the HTTP
   `32` case), but the same-binary client topline regressed (`44.579314` to
   `41.769369` tok/s) and server decode improved only about `1%`. Do not
   promote isolated router cuBLAS as a default; use it as evidence that the
   next useful boundary is broader fusion/scheduling around router, route
   planning, and HC-current/input staging.
   Sprint 388 tested that broader boundary by moving compact route-plan
   construction to GPUs. It preserved tokens but regressed direct decode
   (`76.179292` to `65.263520` tok/s) and HTTP server decode (`94.952767` to
   `87.652515` tok/s), because P2P replication, several tiny kernels,
   synchronization, and route-total readback cost more than the removed D2H.
   Sprint 394 then tested a narrower hash-router select optimization:
   `--router-hash-fast-gate` evaluates only the six DS4 hash-row experts
   instead of computing probabilities for all `256` experts before applying
   the hash row. It preserved `32/32` HTTP response parity and readiness, but
   was not promotable: server decode moved only `106.900859` to `107.274556`
   tok/s, router select only `27.766750` to `27.683134` ms, and scaffold
   decode regressed `289.821429` to `293.484520` ms. Sprint 395 then cleaned
   up the remaining CPU upload boundary with persistent pinned route-plan
   buffers and stream-ordered async H2D uploads. This was promoted:
   same-binary HTTP response parity passed `32/32`, readiness passed, server
   decode improved from `104.834948` to `107.092211` tok/s, route upload
   dropped from `6.785109` to `4.736281` ms, router D2H dropped from
   `1.016605` to `0.562918` ms, and VRAM admission stayed clean with
   `1746 MiB` minimum free. The route-upload cleanup is now done enough for
   the current architecture. Future route work should fuse planning with
   expert dispatch/compose or remove per-layer host involvement entirely; do
   not promote the naive GPU planner or isolated hash-select micro-
   optimization. Sprint 396 answered the NCCL question in the TP8 collective
   workbench: NCCL `2.19.3` is available on the V100 pod and is `2.2x-4.8x`
   faster than the current peer-copy doubling workbench across allreduce,
   reduce-scatter, allgather, rs-ag, and ep-reduce at `32` and `128` tokens,
   with exact verification. Sprint 397 then tested NCCL in the serving
   harness for the compatible non-compact FP32 EP compose boundary. It
   preserved checksum but was slower than peer-copy fused compose
   (`6.401091` ms vs `2.521989` ms), and the production compact route compose
   path correctly keeps NCCL inactive because it is route-indexed rather than
   a dense reduce-scatter. Keep NCCL diagnostic-only until a true TP hidden or
   expert collective boundary exists. Sprint 398 then tested a narrow
   HC-current fill/pack fusion that replaced explicit peer-copy plus local
   dense/half fills and route packing with one fused kernel per rank. It
   preserved first token `54639` and VRAM admission at the `32` slot / `256K`
   model-router compact-MoE shape, but generated decode regressed from
   `87.759480` to `64.310075` tok/s and HC fill/pack grew from `28.140415` to
   `320.439853` ms. This rejects direct peer/UVA remote-load fusion for this
   boundary. Future HC-current work should preserve local staging or fuse into
   downstream dense/expert consumers, not replace local reads with remote loads.
   Sprint 399 then added NCCL to the TP8 layer-boundary proxy, which is the
   correct proxy for future dense TP hidden-state all-reduce. With `43` layers,
   `2` collectives/layer, and F16 hidden payloads, NCCL beat peer-copy doubling
   at every tested shape: `32` tokens improved `29.918408` to `13.960581` ms
   (`2.14x`), `128` tokens improved `37.313934` to `17.326618` ms (`2.15x`),
   and resident-work cases with `local_op_repeats=64` still improved `2.01x`
   and `1.88x`. This keeps the architecture direction clear: use NCCL for true
   TP hidden/expert collectives, but do not attach it to compact route-indexed
   EP compose. Sprint 400 then attached NCCL to a real serving-facing TP
   attention-output allgather boundary. The implementation is functionally
   correct at `16` slots / `256K` with matching first token `45178`, but it is
   not promotable: at the target `32` slot / `256K` shape, NCCL communicator
   overhead adds roughly `+660 MiB/GPU`, reduces minimum free VRAM from
   `1746 MiB` to `1114 MiB`, and OOMs during raw-SWA allocation before a full
   token can complete. The smaller `16` slot run is also slightly slower
   overall (`29.467687` to `28.925690` generated decode tok/s). NCCL remains a
   strategic direction, but future NCCL work must amortize one shared
   communicator across broader TP hidden/expert boundaries and be admitted by
   the memory planner before promotion. Sprint 401 then tested NCCL on the
   HC-current hidden-state allgather. This is a broader and more central
   boundary than attention-output, but still not promotable. It is correct at
   `16` slots / `256K` with first token `54639`, but the target `32` slot /
   `256K` run again OOMs after communicator initialization; minimum free VRAM
   falls from `1746 MiB` to `1114 MiB`. At `16` slots, HC-current gather
   regresses from `5.532507` to `15.830067` ms and generated decode regresses
   from `65.078267` to `61.918746` tok/s. The conclusion is sharper now:
   NCCL should not be bolted onto narrow existing serving boundaries one at a
   time. Future NCCL integration needs a memory-planned shared communicator and
   a larger fused TP/expert boundary that removes enough staging and launch
   work to pay for the communicator and layout conversion.
8. Close the S-E follow-up with a narrow parity/precheck fix if we want to
   revisit fused gated-SiLU. Sprint 379 showed the current serving-shaped
   branch already has no standalone routed SwiGLU launch, the generic
   TurboMind gated-SiLU epilogue is not DS4-equivalent, and the new
   DS4-clamped ABI is fast only in EP-only isolation so far (`4.102144` ms
   two-step gate versus `0.622592` ms fused gate on layer 0). It is not a
   serving promotion candidate until the resident dense-KV precheck failure
   under `routed-normalized + fused-gated-silu` is diagnosed or a deterministic
   fused-gate parity harness proves the ABI.
9. Keep TP-sharded experts out of serving for now. Sprint 380 measured TP8 and
   TP4: TP8 is still numerically invalid, and TP4 is correct but only
   `1.055x/0.891x/0.927x` total speedup at `96/192/384` routes because
   reduction dominates. Revisit TP experts only as a focused fused TP4
   reduction/compose sprint.
10. Treat S-G E5M2 KV as a positive diagnostic, not a default yet. Sprint 381
   added `DS4_V100_TP_KV_F8_E5M2_B128` and `--fp8-e5m2-kv-gate`; V100 row
   tests passed with zero byte mismatches, direct 4-token checksum matched
   while decode improved from `70.710875` to `75.787866` tok/s, and HTTP
   selected-token client tok/s improved from `17.212677` to `22.389190`.
   However, E5M2 gives up mantissa precision and only short selected-token
   parity is proven. Sprint 391 extended this to a `32` request / `32` token
   chat A/B at `256K`: HTTP response parity passed `32/32`, server decode
   improved from `101.206458` to `107.281060` tok/s, and client throughput
   improved from `46.115999` to `47.895831` tok/s, but direct decode regressed
   slightly from `103.237368` to `102.152512` tok/s. E4M3 remains the default
   until a broader multi-prompt parity/soak accepts the precision risk.
   Sprint 392 added that multi-prompt soak harness and ran `16` varied prompts:
   response parity passed `32/32`, but server decode was effectively flat
   (`106.390802` to `106.483285` tok/s) and the layout is not a capacity win.
   Keep E5M2 diagnostic-only.
11. Add S-H MTP only after base TP/EP decode has stable metrology and a settled
   launch strategy. MTP remains the decode multiplier, but it should not hide
   kernel scheduling or topology bottlenecks.

## Current State

- The PP/layer-scheduled appliance is deployed and useful as a baseline, but
  it is no longer the optimization target.
- Sprint 225 fixed the immediate MTP reset/snapshot blocker:
  `long_memory_archive` full-prompt reset parity and target-block restore now
  pass.
- Sprint 225 also corrected the benchmark contract:
  single-slot replay is diagnostic only, while practical throughput must be
  measured with multi-slot serving and `active_microbatch == slots`.
- The current frozen production-shaped PP baseline from Sprint 225 is:
  `32` slots / `256K`, `64/64` token match, `50.434232` generated tok/s,
  `47.282093` continuation tok/s, average GPU utilization `47.076%`, max
  GPU utilization `96%`.
- The TP/EP path is now operational as a resident diagnostic text-serving
  harness. It accepts `/v1/completions` and `/v1/chat/completions`, tokenizes
  text prompts through the existing DS4 tokenizer, runs tokenized prompt
  prefill, performs multi-token autoregressive output-head/sample/feed, returns
  decoded text plus token IDs, and keeps session KV/HC cursors resident across
  requests.
- Current TP/EP text-chat metric from Sprint 306: `32` concurrent chat
  requests at `32` slots / `256K` formed one coalesced batch, tokenized each
  request to `7` prompt tokens, prefilling `6`, generated `256` total tokens,
  and returned `32/32` HTTP 200 responses. Server-side generated-section
  throughput was `214.155740` wall tok/s / `355.130754` decode tok/s.
  Client-side effective throughput including HTTP orchestration was
  `110.036538` tok/s.
- Latest TP/EP attention-correctness work from Sprint 325 added a compact
  compressed-reference diff gate and fixed a real layer-state bug in the smoke
  path. Raw-SWA, attention-compressed, and indexer-compressed buffers are now
  layer-local in the diagnostic harness. The `slots=1`, `position=100003` and
  `slots=32`, `position=262143` all-layer gates both pass their compact
  ratio-4 compressed-row/indexer-score diffs through layer `42`; the `32` slot
  diagnostic reports `39.258626` projected slot-step tok/s. This is still a
  bounded one-row diagnostic, not production long-history compressed KV.
- Sprint 326 removed that one-row diagnostic limitation. The TP/EP smoke path
  now keeps `8` bounded compressed rows per layer, tracks visible row counts,
  scores all bounded visible ratio-4 indexer rows, replicates selected indices
  across TP ranks, and reads multiple selected compressed rows in raw+compressed
  attention. The `32` slot / `256K` / `8` step all-layer attention gate passes
  with `344` layer-step invocations, `visible_compressed_rows=2`,
  `selected_compressed_rows=2`, no compact diff failures, and `20.780883`
  projected slot-step tok/s. This is still a bounded diagnostic cache, not the
  final production compressed-KV allocator.
- Latest TP/EP compressed-KV performance work through Sprint 357 has moved
  compressed fusions from direct-only experiments into serving-visible gates.
  The selected-token HTTP profiler can now force emitted compressed rows at
  `position=262143`, returns `32/32` HTTP 200 responses at `32` slots /
  `256K`, and parses compressed-KV stage timing from server output. Fused
  input-fill + pool-norm reduces parsed compressed-KV sum from `127.697384`
  to `123.651985` ms in that HTTP run, but one-token client throughput is
  flat because request overhead dominates. Keep the gates opt-in until a
  longer amortized serving A/B proves a topline win.
- Sprint 358 ran that longer amortized selected-token HTTP A/B. At
  `position=262112`, `32` tokens/request, `32` slots, and `256K`, control
  measured `71.818394` client tok/s and `3506.921796` ms compressed-KV sum.
  Combined input-fill + pool-norm is not promotable (`3509.986423` ms
  compressed-KV sum and lower scaffold decode proxy). Pool-norm only is still
  interesting (`73.052883` client tok/s and `3474.878472` ms compressed-KV
  sum), but the scaffold decode proxy regressed, so it remains opt-in pending
  repeated/direct confirmation.
- Sprint 359 supplied that direct confirmation. The non-HTTP 32-step A/B at
  the same `position=262112` long-context window improved generated decode
  tok/s from `95.851552` to `97.619138`, wall tok/s from `74.814127` to
  `76.140370`, and compressed-KV sum from `3521.094409` to `3458.469603` ms
  with the same first token. Fused compressed pool+norm is now the TP/EP
  serving default; fused input-fill and RoPE+round remain diagnostics.
- Sprint 360 validated that default through the launcher path. A
  `tools/ds4-v100-run-appliance.sh --print-command` proof includes
  `--true-ds4-compressed-kv-fused-pool-norm-gate` without an explicit
  pool-norm env override, and a launcher-started selected-token HTTP run
  returned `32/32` HTTP 200 at `32` slots / `256K` / `position=262112` /
  `32` tokens/request with `73.289956` client generated tok/s and `187`
  fused pool-norm rows.
- Sprint 361 checked the full `/v1/chat/completions` endpoint. The promoted
  default is active and stable there (`126` fused pool-norm rows, `32/32`
  HTTP 200, same first token), but the short `8` token/request chat shape is
  flat/slightly slower (`24.118711` vs `24.280060` client generated tok/s).
  Treat pool+norm as a decode-path win, not yet a proven short-chat topline
  win.
- Sprint 362 aligned the permanent TP/EP profile harness with launcher
  defaults. HTTP profile runs now inherit the pool+norm default unless
  `--disable-fused-compressed-pool-norm` is set, and V100 proof shows default
  mode emits the pool gate while disable mode does not.
- Sprint 363 implemented and rejected a wider fused compressed
  pool+norm+RoPE+round emitted-row kernel. It is correct and profiler-visible,
  but the 32-step direct 32-slot/256K gate regressed from `95.908399` to
  `95.463298` generated decode tok/s, so it remains diagnostic-only. The next
  lever should move upstream into compressed dense projection or current/gather
  staging rather than wider emitted-row scalar fusion.
- Sprint 364 implemented and rejected direct compressed input fill from
  `hc->d_attn_normed`. It preserved token correctness but doubled the
  compressed-KV one-step cost (`126.724613` to `260.365841` ms) because remote
  peer-read half-fill is far slower than explicit per-rank staging. Preserve
  local per-rank reads; future work should reduce local launch count or improve
  dense projection kernels instead.
- Sprint 365 implemented and rejected local fused attention input fill as a
  default. It preserves local per-rank staging and is correct, but the
  selected-token HTTP long-context gate regressed from `72.886325` to
  `70.674037` client tok/s even though direct 32-step decode was slightly
  positive (`94.237924` to `94.396298` tok/s). Keep the gate diagnostic-only;
  larger compressed/indexer dense projection or attention projection/state
  boundaries are the next TP/EP optimization target.
- Sprint 366 promoted compressed dense event waits as the TP/EP serving
  default. The gate preserves data layout and math, but replaces host
  synchronizes between compressed input fills and dense launches with per-rank
  CUDA event dependencies. At `32` slots / `256K`, selected-token HTTP
  improved from `71.833757` to `74.432464` client tok/s and reduced
  compressed-KV sum from `3437.636456` to `3137.755187` ms. The default is
  disableable with
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_DENSE_EVENT_WAIT=0`.
- Sprint 367 confirmed the event-wait default through `/v1/chat/completions`
  at a decode-heavy long-context shape. With `32` concurrent requests,
  `32` generated tokens/request, `32` slots, and `256K` context, the valid
  `position=262080` run returned `32/32` HTTP 200 and improved client tok/s
  from `50.648397` to `52.022782`, server wall tok/s from `81.426024` to
  `83.891024`, and server decode tok/s from `96.116667` to `99.521680`.
  Chat long-context admission must account for prompt prefill; starting at
  `262112` is invalid for a 32-token chat generation because it reaches the
  configured context boundary.
- Sprint 368 implemented that context admission. TP/EP generation requests now
  check `start_position + prompt_prefill_steps + requested_decode_steps <=
  262144` before GPU decode. The invalid Sprint 367 chat shape now returns
  HTTP 400 with `context_window_exceeded` and `final_position=262160`, while
  the valid `position=262080` 32-request chat shape still returns `32/32` HTTP
  200 and preserves first token `89340`.
- Sprint 369 made GPU utilization capture a permanent, disabled-by-default
  feature of the TP/EP profile harness. `--gpu-sample-interval-ms N` now writes
  `gpu_util.csv` and adds aggregate plus per-GPU utilization/memory summaries
  to `summary.json` for both HTTP and direct token-major profiles. The V100
  sampled smoke at `32` configured slots, `4` chat requests, `4` tokens/request,
  and `256K` context returned `4/4` HTTP 200 with `coalesced_batch_size=4`,
  server decode `99.340235` tok/s, average GPU utilization `8.412879%`, and
  max GPU utilization `39%`. This confirms the low-occupancy imbalance is now
  measurable in the main artifact path before the next scheduling/kernel
  optimization.
- Sprint 370 added a reusable active-slot matrix driver around that profile
  harness. The V100 smoke matrix at `32` configured slots, `256K` context,
  `position=100000`, `2` tokens/request, and active request cases `1,4`
  produced aggregate TSV/JSON plus per-case profile artifacts. Both cases
  passed (`1/1` and `4/4` HTTP 200) and coalescing worked, but server decode
  stayed effectively flat (`101.842964` to `101.159316` tok/s) while average
  GPU utilization stayed near `8.3%`. This is not the full matrix, but it
  validates the tool needed to characterize 1/4/8/16/32 active-slot behavior.
- Sprint 371 ran that full active-slot matrix at the target long-context chat
  shape: `32` configured slots, `256K` context, `position=262080`, and
  `32` generated tokens/request. Cases `1,4,8,16,32` all passed and coalesced
  correctly. Client aggregate tok/s scaled from `1.584552` to `50.694229`
  because the same fixed batch cost was amortized over more active responses,
  but server wall tok/s stayed `81.6-83.9`, server decode tok/s stayed
  `97.4-100.0`, and average GPU utilization stayed `9.8-10.3%`. The decision
  is: active-slot compaction helps low/moderate occupancy cost, but the 32-slot
  topline needs full-occupancy kernel/state work, especially compressed/indexer
  dense projection, attention projection/state, and GPU0-heavy staging.
- Sprint 372 implemented an opt-in production-candidate gate to skip
  host-side dense-output statistics in the compressed-KV projection path:
  `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1`. This preserves
  the current packed/runtime dtypes and removes diagnostic host copies and
  synchronization from compressed/indexer dense outputs. Direct token-major
  `32` slot / `256K` / `32` step validation kept the same first token
  (`98751`), improved scaffold decode from `100.739521` to `117.463961`
  tok/s, and reduced parsed compressed-KV time from `3141.768079` to
  `1789.795027` ms. The full chat A/B improved server decode from
  `99.748339` to `117.340768` tok/s and client throughput from `51.345855`
  to `58.923892` tok/s, but normal chat text parity still needs a cleaner
  deterministic comparator before default promotion. The current measured
  bottleneck remains compressed/indexer dense projection and surrounding
  staging, with visible GPU0-heavy harness/control imbalance.
- Next format direction: evaluate an offline INT8+scale pack variant for the
  FP8 source compressed/indexer dense tensors. This should be a scoped
  A/B path, not a whole-model conversion. The candidate must preserve source
  quantization metadata in the pack manifest, produce FP32-equivalent outputs
  for downstream state math, and prove token/logit parity before serving
  enablement.
- Sprint 373 converted that format question into a reusable TP/EP contract
  audit. The current hot compressed/indexer dense candidate set is mostly BF16,
  not FP8: BF16 attention compressor, BF16 indexer compressor, BF16 tiny
  indexer projection, plus F8 `indexer.attn_q_b`. An INT8+fp16-scale layout
  with `qk=32` would reduce the scoped candidate bytes from `0.742 GiB` to
  `0.481 GiB` aggregate (`94.977 MiB` to `61.525 MiB` per GPU). The primary
  workbench target is BF16 attention compressor GEMMs at `M=32`, `K=4096`,
  and `N=128/64`; F8 `indexer.attn_q_b` is not a memory win because INT8+scale
  is larger than F8 block-128 (`169.312 MiB` to `178.500 MiB`).
- Sprint 374 tested that candidate with a focused V100 workbench. The copied
  tc-grid INT8 kernels are numerically acceptable on the synthetic compressor
  problem, but slower than the FP16 tensor-op baseline at the actual target
  shapes: `M=32,N=128,K=4096` measured `0.009250 ms` for cuBLAS FP16 versus
  `0.042721 ms` for best tc-grid INT8, and `M=32,N=64,K=4096` measured
  `0.008803 ms` versus `0.036673 ms`. Do not wire tc-grid INT8 into the
  attention compressor production path. The next kernel direction should
  either adapt the vLLM/TurboMind SM70 small-M GEMM registry for this exact
  compressor shape or fuse the compressor dense boundary with adjacent
  state/emit work so we remove staging/launch traffic rather than only changing
  dtype.
- Sprint 375 implemented `--async-output-gate` and rejected it as a default.
  It preserves selected-token/checksum parity and reduces output-head device
  synchronizations from `26` to `0`, but the real `32` active request /
  `32` slot / `256K` HTTP A/B regressed server decode throughput from
  `99.476540` to `93.764276` tok/s and left average GPU utilization flat.
  Keep it opt-in for graph-capture investigation only.
- `TEMP_THROUGHPUT_PROMPT.md` is now the active performance steering
  document. Its core read is that the TP/EP decode path is launch/sync
  fragmented at the real serving shape: the full active-slot matrix stayed
  around `97.4-100.0` server decode tok/s and `9.8-10.3%` average GPU
  utilization from `1` to `32` active requests. The immediate work queue is
  therefore isolated gate experiments that either remove launch/sync overhead
  or prove that this thesis is wrong before more dtype or micro-kernel swaps.
  Async output and CUDA graph replay have now both been tested and rejected as
  defaults. The controlling order is now batched paged attention, compact MoE,
  fused gated-SiLU, TP-sharded expert A/B, FP8 KV, then MTP.
- Sprint 376 executed that steering document's S-A gate:
  `--decode-cudagraph-gate`. The audit removed broad in-step `sync_all` host
  waits and tracked helper-level host waits under the graph gate while
  preserving first-token/checksum parity. The graph-gated diagnostic became
  capture eligible for the target one-step non-emitted-row shape:
  helper blocker classes fell from `7` to `0`, first token stayed `54639`,
  output checksum stayed `24071637347`, and scaffold checksum stayed
  `3401922407`. Real stream capture was then rejected by the V100/CUDA stack:
  separate captures conflict with cross-stream event dependencies, root capture
  needs explicit stream joining, and joined capture fails on
  `cudaMemcpyPeerAsync`. Replacing HC-current peer copies with graph-gated
  device copy kernels moved the failure to the next peer copy in attention
  projection. Decision: reject graph replay as a promotion path and move next
  to `--batched-paged-attn-gate`.
- Sprint 377 is now the active S-C gate. The baseline V100 serving run at
  `32` active requests / `32` configured slots / `256K` context /
  `position=262080` / `32` generated tokens returned `32/32` HTTP 200 with
  first token `89340`, client generated `40.157540` tok/s, server generated
  decode `88.372350` tok/s, average GPU utilization `7.972222%`, max GPU
  utilization `38%`, and compressed-KV sum `5436.764269` ms. Gate plumbing is
  implemented and validated for `--batched-paged-attn-gate`,
  `DS4_V100_TP_EP_BATCHED_PAGED_ATTN=1`, and
  `tools/ds4-v100-tp-ep-profile.py --batched-paged-attn`; the direct no-op
  smoke preserved first token `54639` and finite output. The remaining Sprint
  377 work started with the fixed-size row-family plan. The 8-token direct
  V100 smoke emitted `127` plan rows, preserved finite output with first token
  `98751`, and showed ratio-4 layers reaching `visible_attn_rows=2` plus
  `visible_indexer_rows=2`. The key planning result is that pending
  typed-history reloads are `0` in the observed compressed/indexer samples:
  skip-current-load and the bounded reload cache already avoid the narrow
  reload storm. A load-only S-C kernel is therefore unlikely to move topline
  throughput. Decision: keep the row planner diagnostic-only, do not promote
  S-C as a serving default, and move the next sprint to compact MoE.
- Sprint 327 made the production compressed-KV memory contract executable in
  `tools/ds4-v100-plan-tp.c`. With the real TP pack and F8 KV, `32` slots at
  `256K` fits at `27.00 GiB/GPU` with `5.00 GiB` headroom after reserve;
  persistent typed KV is `3.40 GiB/GPU`. The same configuration would require
  `107.84 GiB/GPU` if KV were replicated f32, so production serving must use a
  typed TP-sharded KV arena. `1` slot at `1M` also fits at `22.56 GiB/GPU`.
- Sprint 328 proved that contract as actual V100 CUDA allocations. The new
  `tools/ds4-v100-tp-kv-arena-smoke.cu` allocates and touches the per-GPU
  resident arenas for weights, typed KV, compression state, scratch,
  collectives, and global shards. With the real pack footprint, `32` slots at
  `256K` allocated `25.001 GiB/GPU` and left `6.424 GiB/GPU` free, above the
  `2 GiB` reserve. `1` slot at `1M` allocated `20.558 GiB/GPU` and left
  `10.866 GiB/GPU` free. This removes raw VRAM fit as the immediate blocker
  for the target TP/EP KV layout; the remaining work is wiring the production
  typed arena into the runtime and proving layer/reference semantics.
- Sprint 329 corrected the planner/arena budget to match the physical
  row-sharded KV layout already used by `ds4_v100_tp_runtime`. The actual F8
  KV allocation at `32` slots / `256K` is `3707940864` bytes/GPU
  (`3.453 GiB`), not the ideal aggregate-sharded lower bound of
  `3646642176` bytes/GPU. The difference is only `58.46 MiB/GPU`, and the
  corrected target still passes allocation: `25.058 GiB/GPU` no-reserve,
  `27.058 GiB/GPU` with reserve, `6.366 GiB/GPU` free after allocation on the
  pod. Admission with physical row-sharded KV is now `62` slots at `256K`,
  `31` slots at `512K`, and `15` slots at `1M`.
- Sprint 330 added the first execution primitive for that production KV arena.
  `ds4_v100_tp_runtime` now exposes row views for attention and ratio-4
  indexer rows and can write/gather/decode F8 E4M3 block-128 rows from the
  physical TP shards. At `32` slots / `256K`, layer `2`, slot `31`, position
  `262140`, both attention and indexer row roundtrips pass with
  `bad decoded values=0` and `max_abs=0.000000000`. This is not yet wired into
  the full-layer attention path, but it gives that path the production typed
  row storage primitive needed to replace f32 diagnostic KV buffers.
- Sprint 331 promoted that primitive to device-to-device store/load APIs.
  `ds4_v100_tp_runtime_kv_row_store_f32_device` accepts one f32 device row per
  GPU and writes the physical F8 shard; `ds4_v100_tp_runtime_kv_row_load_f32_device`
  decodes the distributed row back to device f32 buffers using peer reads. At
  `32` slots / `256K`, layer `2`, slot `31`, position `262140`, both attention
  and indexer device roundtrips pass with `bad values=0` and
  `max_abs=0.000000000`.
- Sprint 332 wired the device KV APIs into the full-layer TP/EP attention
  state path for raw-SWA and proved full-layer typed store/load plumbing, but
  Sprint 333 found and corrected an addressing bug in that first integration:
  the generic `ATTN` row kind addresses the compressed long-attention row on
  ratio layers, not the raw-SWA ring. Sprint 333 added
  `DS4_V100_TP_KV_ROW_ATTN_RAW` and switched
  `--true-ds4-attention-typed-kv-raw-gate` to it. At `32` slots / `256K`,
  layer `2`, slot `31`, position `262140`, `attn_raw` maps to physical row
  `124` while `attn` maps to compressed row `65663`; both device roundtrips
  pass. The corrected all-layer shared-state gate emits typed raw-SWA PASS
  lines with `physical_row=124` for all `43` layers and ends with
  `pass_layers=43`, projected `72.313683` slot-step tok/s.
- Sprint 334 extended the typed KV integration to emitted compressed attention
  rows. With both typed raw-SWA and typed compressed-attention gates enabled
  at `32` slots / `256K`, position `262143`, the full-layer gate emits `43`
  raw-SWA typed rows and `41` compressed-attention typed rows, then ends with
  `pass_layers=43`, projected `51.386758` slot-step tok/s. Representative
  compressed physical rows are layer `2` ratio-4 row `65663` and layer `3`
  ratio-128 row `2175`.
- Sprint 335 completed the emitted-row typed KV set by adding ratio-4 indexer
  rows. With typed raw-SWA, typed compressed-attention, and typed indexer gates
  enabled at `32` slots / `256K`, position `262143`, the full-layer gate emits
  `43` raw-SWA rows, `41` compressed-attention rows, and `21` indexer rows,
  then ends with `pass_layers=43`, projected `53.556562` slot-step tok/s.
  The compact reference/indexer diagnostic also passes with `21` typed indexer
  rows and `21` compact-reference summaries.
- Sprint 336 added typed compressed-history reload. The full-layer path now
  records emitted compressed/indexer source positions and reloads visible
  compressed attention and ratio-4 indexer rows from the production typed TP
  KV arena before raw+compressed attention reads. A `32` slot / `256K`
  token-major run from position `262136` for `8` decode steps passes all `344`
  layer-step invocations, emits `328` typed-history lines, and reaches
  `visible_attn_rows=2`, `loaded_attn_rows=2`, `loaded_indexer_rows=2` on all
  `21` ratio-4 layers.
- Sprint 337 promoted the typed KV gates into tokenizer-enabled TP/EP HTTP
  serving. `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_HISTORY=1` now
  cascades the typed raw-SWA, compressed-attention, ratio-4 indexer, and
  history gates through the appliance launcher, and the server reports those
  gates in `/status`, `/metrics`, and response metadata. A `32` slot / `256K`
  HTTP smoke returned `200` for two `/v1/chat/completions` requests using the
  same `session_id`; the second request reused resident slot `0`
  (`cache_hit=1`) and advanced from `100014` to `100016`. Server logs show
  typed KV PASS-line counts of `685` raw, `83` compressed-attention, `83`
  indexer, and `653` history rows, including `84` lines with
  `loaded_attn_rows=2` and `loaded_indexer_rows=2`. The next integration step
  is a longer typed-KV HTTP run and an A/B against the no-typed-KV serving
  baseline so the cost of the production typed KV path is quantified.
- Sprint 338 quantified that cost. In a same-shape `32` concurrent
  `/v1/chat/completions` A/B at `32` slots / `256K` / `8` generated tokens per
  request, the no-typed-KV control returned `32/32` HTTP 200 with
  `260.529425` server wall tok/s and `698.278847` decode tok/s. The
  typed-history candidate also returned `32/32` HTTP 200 and emitted typed KV
  history evidence (`942` raw, `105` compressed, `105` indexer, `898` history
  lines, with `84` `loaded_attn_rows=2` and `84` `loaded_indexer_rows=2`
  lines), but throughput fell to `56.495098` wall tok/s and `63.381174`
  decode tok/s. Typed KV is therefore operational as the correctness-facing
  path, but remains opt-in until the staging overhead is reduced through direct
  typed-row attention reads or a narrower reload cache.
- Sprint 339 added that narrower reload cache for bounded compressed/indexer
  history rows. The cache worked: in the same `32` concurrent / `32` slot /
  `256K` / `8` token HTTP A/B, all `899` typed-history lines reported
  `reloaded_attn_rows=0` and `reloaded_indexer_rows=0` while preserving visible
  loaded-row evidence. Typed-history server wall throughput improved from
  Sprint 338's `56.495098` tok/s to `68.358523` tok/s, but remained far below
  the same-run no-typed-KV control at `311.293794` tok/s. The remaining
  bottleneck is therefore likely the same-step typed KV roundtrip for current
  raw/compressed/indexer rows. The next step is to store production typed rows
  while reusing the already-computed f32 staging row for immediate attention,
  instead of loading the same row back from typed KV in the hot layer step.
- Sprint 340 implemented that current-row skip as an explicit performance
  gate:
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_SKIP_CURRENT_LOAD=1`.
  The typed candidate still stores production typed raw-SWA,
  compressed-attention, and ratio-4 indexer rows, but avoids immediately
  loading those current rows back through typed KV in the same layer step. In
  the same `32` concurrent / `32` slot / `256K` / `8` token HTTP A/B, control
  measured `316.297621` server wall tok/s and `735.600737` decode tok/s; the
  typed skip-current-load candidate measured `74.383163` wall tok/s and
  `86.322558` decode tok/s, with `typed_current_load_0=1152` and
  `typed_current_load_1=0`. This improves typed-history serving but remains
  far below control, so the next bottleneck is likely typed KV store overhead
  itself. The next step is to measure store-only cost by row family and then
  batch row stores or fuse stores into producer kernels.
- Sprint 341 measured typed KV store-family cost. Diagnostic store suppression
  variants showed that stores are not the primary remaining regression:
  control measured `308.223158` wall tok/s / `722.800920` decode tok/s,
  typed-history baseline measured `75.577828` / `87.938497`, and the
  diagnostic no-store candidate measured only `79.039985` / `93.242875`.
  Individual family suppression was also mostly flat: no raw store
  `77.304656`, no compressed store `74.992652`, and no indexer store
  `73.921469` wall tok/s. The next bottleneck is therefore likely diagnostic
  overhead in the typed path itself: per-layer/device synchronizations, verbose
  typed PASS logging, and typed-history/indexer bookkeeping in the hot serving
  loop. The next step is to gate or remove that overhead while preserving typed
  production KV semantics.
- Sprint 342 isolated typed KV PASS logging overhead. A production-compatible
  quiet gate,
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_QUIET=1`, suppresses the
  per-layer typed raw/compressed/indexer/history PASS logs while preserving
  typed KV semantics. In the same `32` concurrent / `32` slot / `256K` /
  `8` token HTTP A/B, control measured `309.202473` wall tok/s /
  `730.769885` decode tok/s, verbose typed-history measured `73.427107` /
  `85.479279`, and typed-quiet measured `75.284862` / `87.627420`.
  Removing `2058` typed PASS lines produced only a `~2.5%` gain, so stdout
  formatting is not the main bottleneck. The next target is the typed row API
  shape itself: per-slot row calls, per-rank row kernels, broad device
  synchronizations, and hot-loop typed-history bookkeeping.
- Sprint 343 batched the typed row API across slots. The runtime now exposes
  `ds4_v100_tp_runtime_kv_rows_store_f32_device` and
  `ds4_v100_tp_runtime_kv_rows_load_f32_device`, and the serving path can use
  them with `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_BATCH_ROWS=1` for
  raw-SWA current rows, emitted compressed-attention rows, emitted ratio-4
  indexer rows, and typed-history reloads. In the same `32` concurrent /
  `32` slot / `256K` / `8` token HTTP A/B, control measured `303.282600`
  wall tok/s / `735.908031` decode tok/s, typed-quiet measured `73.452667` /
  `86.332914`, and typed-batch-rows-quiet measured `79.984163` /
  `95.624885`. Batching gives a real `+8.9%` wall and `+10.8%` decode
  improvement versus typed-quiet, but the remaining gap to control is still
  large. The next target is broad device synchronization/order around typed row
  work, ideally replacing device-wide barriers with stream-ordered row stores
  and loads.
- Sprint 344 tested that synchronization hypothesis. The new
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_TYPED_KV_STREAM_SYNC=1` gate narrows
  typed KV row barriers from device-wide synchronization to stream
  synchronization. In the same `32` concurrent / `32` slot / `256K` /
  `8` token HTTP A/B, control measured `309.709482` wall tok/s /
  `730.989696` decode tok/s, typed-batch-rows-quiet measured `79.794096` /
  `94.238623`, and typed-batch-rows-stream-sync-quiet measured `81.006809` /
  `95.558274`. The `~1.5%` gain is too small to explain the low utilization
  or the gap to control. The next sprint must collect Nsight evidence for the
  typed serving window: top kernels, tensor-core/HMMA activity, row pack/unpack
  costs, peer-read costs, and synchronization gaps.
- The system is not production-ready yet because the bridge HC sequence has
  not been proven equivalent to the DeepSeek V4 reference layer semantics, and
  production serving still needs readiness/overload/cancellation/streaming
  behavior plus a persistent deployment gate.
- Sprint 295 added stricter cached-state guardrails for downstream-serving
  work: `DS4_V100_TP_EP_KV_ALL_SLOTS=1` updates and verifies sharded KV rows
  for every active slot instead of only the old diagnostic `kv_slot=7`, and
  `DS4_V100_TP_EP_HC_PERSIST_STATE=1` prevents HC state reset between serving
  calls. The 32-slot `/v1/completions` run passes with
  `kv_runtime_resident=1`, `kv_all_slots_gate=1`,
  `hc_persist_state_gate=1`, `58.791255` wall tok/s, and `206.196887` decode
  tok/s. This is intentionally a correctness mode: all-slot KV readback is
  expensive and should be removed only after real session ownership and prefill
  are implemented.
- Sprint 296 added the first TP/EP HTTP session-slot layer, based on the
  serving semantics in `ds4.c` and llama.cpp rather than the old PP appliance.
  Requests now have cache keys, stable resident slot assignment, LRU eviction,
  cache-position bucketing, duplicate-session protection within one decode
  batch, `/v100/slots`, and hit/miss/eviction counters in status/metrics and
  responses. A V100 smoke shows a repeated `session_id` reusing slot `0` and
  advancing from `100000 -> 100001 -> 100002` with one miss and one hit. The
  endpoint is still diagnostic until tokenizer prefill, true prompt token
  accounting, selected-token feedback, and active-slot-only decode are wired
  behind this session table.
- Sprint 297 added a prompt-fingerprint guard to that session layer. Reusing a
  `session_id` with the same prompt now hits resident state; reusing it with a
  different prompt resets the slot and records a miss. This is a temporary
  string-level guardrail until tokenizer-level prefix matching and suffix
  prefill are implemented.
- Sprint 298 ran the first longer `/v1/completions` diagnostic benchmark after
  those API guardrails. At `32` concurrent requests, `32` slots, `256K`
  context, diagnostic output head, HC-current input, HC final expand, and
  persistent HC state, the `16/32/64` token cases each formed one coalesced
  batch and returned `32/32` HTTP 200 responses. Wall generated throughput
  plateaued near `195-200` tok/s and decode generated throughput near
  `329-340` tok/s, with low average GPU utilization. This is the current
  diagnostic API throughput baseline, not the final optimized serving target.
- Sprint 299 added tokenized prompt acceptance and per-session generated-token
  timelines to the TP/EP completion endpoint. Numeric `prompt_tokens` now feed
  token-sequence prompt fingerprints, resident slots expose prompt-token and
  generated-token counts, and a V100 smoke shows a repeated `session_id`
  reusing the slot while generated-token history advances from `1` to `2`.
  The next hard serving gap is real tokenizer/prompt prefill plus selected
  token feedback into the next CUDA decode input.
- Sprint 300 added the first request-boundary selected-token feedback bridge.
  The TP/EP HTTP path now loads source BF16 `token_embd.weight` once, seeds
  layer-0 HC shards from the prompt tail on a miss, and seeds from the previous
  selected token on a cache hit. This matches the core serving loop direction
  in `ds4.c` and llama.cpp, but only across one-token HTTP requests. A true
  completion endpoint still needs prompt prefill and an internal
  output-head/sample/feed loop for multi-token generation.
- Sprint 301 added that internal per-step feedback loop for diagnostic
  `max_tokens > 1` requests. The endpoint now decodes one token, runs the
  vocab-sharded output head, feeds the selected token back through the resident
  BF16 embedding seed, and repeats. This gives the TP/EP path the correct
  autoregressive shape before optimization. Text tokenizer I/O, prompt
  prefill, active-slot-only decode, and MTP remain open.
- Sprint 302 added diagnostic prompt prefill on cache misses. Tokens before
  the prompt tail are evaluated through the TP/EP loop without output-head
  selection, then generation starts from the final prompt token. This gives the
  endpoint the minimal prompt/prefix semantics needed before text I/O and
  performance optimization. Fast batched prefill is still a later optimization.
- Sprint 303 exposed generated token IDs as an explicit response array. The
  diagnostic `/v1/completions` endpoint now returns
  `ds4_v100.generated_token_sequence` plus `slot_position`, so downstream
  clients can consume token IDs and verify resident cursor advancement before
  tokenizer text rendering is wired. A 32-slot / 256K V100 smoke with
  `prompt_tokens=[31,32,33]` and `max_tokens=3` returned
  `[127885,57114,78026]`, advanced the slot to `100005`, and reported
  `214.100724` wall tok/s / `353.667490` decode tok/s for the generated
  section.
- Sprint 304 added a diagnostic `/v1/chat/completions` envelope over the same
  TP/EP resident path. Token-ID clients can now use either text-completion or
  chat-completion routes. The chat smoke returned
  `object=chat.completion`, `message.role=assistant`, matching
  `choices[0].token_ids` and `ds4_v100.generated_token_sequence`, and
  `210.355981` wall tok/s / `350.653125` decode tok/s for the generated
  section. Message text remains empty until tokenizer rendering is wired.
- Sprint 305 wired the existing DS4 tokenizer into the TP/EP binary in
  inspect-only mode. The launcher now passes
  `DS4_V100_TP_EP_TOKENIZER_MODEL`, text prompts are tokenized before prefill,
  and generated token IDs are decoded into `choices[0].text`,
  `choices[0].message.content`, and `ds4_v100.generated_text`. A text chat
  smoke with message content `"Hello"` produced `5` prompt tokens, `4` prefill
  steps, generated token IDs `[95933,89868]`, decoded text `ICCungtod`, and
  `213.595353` wall tok/s / `350.755948` decode tok/s for the generated
  section.
- Sprint 306 ran the first 32-concurrent tokenizer-enabled text chat
  benchmark. All requests coalesced into one 32-slot batch at `256K`; each
  request had `7` prompt tokens, `6` diagnostic prefill steps, and `8`
  generated tokens. The server reported `214.155740` wall tok/s /
  `355.130754` decode tok/s for `256` generated tokens.
- Sprint 307 added the first end-to-end reference-vector parity harness for
  the TP/EP HTTP path. The initial V100 gate intentionally used the official
  `short_reasoning_plain` vector and failed: expected selected text `16`
  (`3136` hex), while TP/EP returned `ICC` (`494343` hex), token ID `95933`.
  This confirms the system is askable but not yet trustworthy as DS4 output.
- Sprint 308 is closing semantic parity. The audit found that the TP/EP layer
  path still had diagnostic-only semantics: synthetic EP routing, a six-local
  expert residency cap, and a simplified attention/FFN bridge. The current
  code removes the expert cap, adds model-router route selection from
  `ffn_gate_inp.weight` plus hash-router metadata, carries per-route weights,
  and separates active-slot masking from token IDs. Full expert residency fits
  at about `27.3 GiB` observed memory per GPU with `147.17 GB` aggregate
  expert bindings. The active-mask V100 run proves nonzero model-router routes
  for real HTTP slots and reports `164.721272` wall tok/s /
  `237.349475` decode tok/s on the `short_reasoning_plain` reference, but it
  still returns `ICC` instead of `16`. The remaining blocker is true layer
  semantics: normalized routed-expert input, full shared FFN, and full DS4
  attention/compressed-KV/indexer math. The normalized routed-input diagnostic
  is now separately gated and fails at layer `0` with
  `decode_finite_bad=16384`. Follow-up tensor stats show the normalized route
  input is finite (`max_abs=38.53125`), but rank `7` produces non-finite
  TurboMind gate/down output while ranks `1` and `6` produce zero expert
  output. The failing selected experts are layer-0 rank-7 locals `30` and
  `21`; rank-6 locals `30` and `8` include the largest route weight but return
  zero output. That makes rank-local expert binding or MXFP4 scale/table
  handling the next narrow correctness target before promoting true FFN input
  semantics. The binding trace shows non-null weight/scale pointers and
  expected strides, so the likely root is now the bridge activation
  distribution rather than a missing pointer-table entry.
- Sprint 309 localized the reference-HC instability and kept the unstable
  reference path diagnostic-only. A guarded run completes the HTTP parity
  request without HTTP 500, but still returns the wrong token, so the blocker
  remains graph semantics rather than API reachability.
- Sprint 310 started replacing the simplified TP/EP attention bridge by
  binding the full DS4 attention projection tensor set for all 43 layers:
  `attn_q_a`, `attn_q_b`, `attn_kv_latent`, `attn_output_a`, and
  `attn_output_b`.
- Sprint 311 made the first true-attention projection prefix executable under
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_PROJECTION=1`. The V100 gate passes all
  43 layers at `32` slots / `256K`, executing `attn_norm -> attn_q_a ->
  attn_q_a_norm -> attn_q_b` and `attn_kv_latent -> attn_kv_a_norm`. This is
  still diagnostic; it does not yet feed q-head RoPE, raw/compressed KV,
  indexer selection, attention softmax/value read, or real attention output
  into the next hidden state.
- Sprint 312 added `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_STATE=1`, which runs
  local q-head RMSNorm over the TP8 `attn_q_b` shards and writes a diagnostic
  raw SWA KV row for all 43 layers at `32` slots / `256K`. The V100 gate has
  43 state-update passes and zero failures. The key caveat is numeric:
  q-head shards are finite, but raw SWA KV reaches FP16 saturation
  (`max_abs=65504`) in early layers, so the next work must isolate whether
  the saturation is caused by the still-simplified upstream HC/current-hidden
  bridge, missing RoPE/reference scaling, or the KV quantize/round contract.
- Sprint 313 added `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_READ=1`, which
  loads `blk.N.attn_sinks`, copies rank-local sink values, and executes a
  sink-aware one-row raw-SWA attention read for all local heads on all TP ranks.
  The V100 gate passes all 43 layers at `32` slots / `256K` with 43 raw-read
  passes and zero failures. This is still diagnostic: it proves attention-read
  plumbing, but early-layer read outputs inherit the `65504` saturation from
  raw KV state.
- Sprint 314 added `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_RAW_WINDOW=1`, which
  keeps the one-row gate intact and adds a sliding raw-window read over rows
  populated by a resident token-major run. A `32` slot / `256K` / `4` step
  V100 gate passes 172 projection/state/raw-window invocations with
  `valid_rows=1..4` and zero failures. This moves the raw-SWA read closer to
  DS4 semantics, but still does not include RoPE, compressed KV, ratio-4
  indexer selection, or attention output projection.
- Sprint 315 added `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_ROPE=1`, which applies
  DS4-style tail RoPE to q-head shards and latent KV rows before the raw-SWA
  diagnostic store/read. A `32` slot / `256K` / `4` step V100 gate passes 172
  RoPE invocations, 172 token-major layer invocations, and zero failures. One
  raw-window diagnostic line was stdout-interleaved, but the final scaffold
  reports 172 pass invocations. The remaining blocker is early-layer
  `65504` raw-KV saturation, not RoPE plumbing.
- Sprint 316 added
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_SATURATION_AUDIT=1`, which measures the
  true-attention projection/state intermediates at `32` slots / `256K`. The
  audit shows saturation first appears at `kv_normed` in layer `1`
  (`436616.219`) before KV RoPE and before raw-SWA storage. Layer `0` is not
  saturated (`kv_normed_max=6510.59814`, `raw_swa_row_max=6656`). The next
  model-correctness target is therefore the `attn_kv_latent ->
  attn_kv_a_norm` normalization/scaling contract or the upstream HC-current
  bridge, not q-head RoPE.
- Sprint 317 added
  `DS4_V100_TP_EP_TRUE_DS4_ATTENTION_KV_NORM_REFERENCE=1` and found the
  concrete implementation bug behind the KV norm drift: `block_sum_256_f32`
  and `block_max_256_f32` return the block-wide reduction only to the first
  warp, leaving threads `32..255` with the wrong reduced value. The same-input
  KV norm reference comparison shows huge per-element drift even when stable
  and reference maxima match. The next sprint must fix reduction broadcast
  before any compressed-KV/indexer work.
- Sprint 318 fixed the TP/EP block-reduction broadcast bug. The combined
  `32` slot / `256K` / `4` step V100 gate now has 172 KV-norm reference rows,
  172 saturation rows, 172 raw-window rows, and zero failures. KV norm
  reference drift dropped from `847034.125` max-abs to `9.53674316e-07`, and
  raw-SWA row max dropped from `65504` to `6.28515625`. The artificial
  attention-prefix saturation blocker is removed.
- Sprint 319 reran the official TP/EP HTTP reference parity gate after the
  reduction fix. The `short_reasoning_plain` vector still fails: expected
  `16`, received `)Skip`, token `83480`, with `193.154852` wall tok/s and
  `303.200535` decode tok/s for the one-token generated section. This is
  improved evidence, not correctness: Sprint 307 returned `ICC` / token
  `95933`, so the reduction fix does affect live output, but TP/EP still needs
  true compressed-KV/indexer attention and attention-output hidden-state
  promotion before production readiness.
- Sprint 320 added the TP/EP true-attention output projection gate. The gate
  proves the real DS4 `attn_output_a -> attn_output_b` projection sequence
  runs at `32` slots / `256K` / `4` steps with final scaffold
  `pass_invocations=172`, zero failure rows, and finite output shards. The
  pack also corrected the topology assumption: `attn_output_a` consumes
  rank-local `[slots][4096]` heads, then the runtime gathers the
  `[slots][8192]` intermediate for `attn_output_b`. The output is still
  diagnostic; the next semantic step is hidden-state promotion.
- Sprint 321 reran the official TP/EP HTTP parity vector with
  `--true-ds4-attention-output-gate` enabled. The vector still fails:
  expected `16`, received `urf`, token `64906`, at `23.926690` wall tok/s and
  `25.093416` decode tok/s for the one generated token. The output changed
  from Sprint 319's `)Skip` / token `83480`, proving the true-attention output
  projection reaches live serving. The next blocker is likely ordering:
  FFN norm/router/shared/routed FFN still need to consume the post-attention
  residual/current hidden rather than the pre-attention bridge.
- Sprint 322 added
  `--true-ds4-post-attention-ffn-input-gate`, which materializes
  `post_attn = current + attn_output_b`, recomputes FFN norm/router routes,
  repacks routed expert inputs, and fills shared-FFN gate/up inputs from that
  post-attention tensor. The `32` slot / `256K` V100 gate passed all 43 layers
  with 43 post-attention rows and zero finite failures. The HTTP parity vector
  still fails: expected `16`, received `mere`, token `88445`, at `21.484145`
  wall tok/s and `22.443315` decode tok/s. The changed token proves the
  post-attention FFN input path reaches serving; the next semantic blocker is
  true compressed-KV/indexer attention rather than FFN input ordering.
- Sprint 226 converted the TP planner into a TP8/EP8-only contract. It no
  longer exposes PP/layer-split topology modes. Against the real production
  pack bytes, the target `32` slots / `256K` / F8-KV shape fits at about
  `27.00 GiB` per GPU including a `2.00 GiB` reserve, with `5.00 GiB`
  headroom.
- Sprint 227 built the TP8 collective workbench. The doubling all-reduce
  boundary is correct and density-sensitive: `1189` overhead-only tok/s at
  32 tokens, `2119` at 64, and `3332` at 128 for the 43-layer,
  two-collective proxy. Root/direct RS+AG is correct but slower and is not the
  first runtime boundary candidate.
- Sprint 228 emitted the TP/EP pack contract from the real production pack.
  The contract has dense TP rows, replicated control/router rows, EP expert
  ownership, and KV/state descriptors, with a balanced `27.024 GiB` per-GPU
  estimate at `32` slots / `256K` / F8 KV.
- Sprint 229 added the first separate TP runtime skeleton. It opens all eight
  GPUs, enables peer access, allocates target hidden/KV/compression/scratch
  arenas for `32` slots / `256K`, runs a fixture pass, and tears down cleanly.
- Sprint 230 added explicit per-layer sharded KV row ownership to the separate
  TP runtime. Ratio-4/indexer and ratio-128 dense/KV slices pass on the V100
  pod at `32` slots / `256K` / F8 KV with `max_abs=0`.
- Sprint 231 added the bounded EP routed-expert slice. A new TP/EP-only smoke
  runs the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on
  all eight V100s at the `32` slot / `top_k=6` target, with finite exact repeat
  output and explicit route/latency reporting.
- Sprint 232 added the first one-layer TP/EP fixture gate. The same process
  opens the target TP runtime, verifies a ratio-4 sharded KV row, and runs
  real TurboMind MXFP4 EP experts on all eight GPUs at `32` slots / `256K` /
  `top_k=6`.
- Sprint 233 validated real TP/EP contract ownership for layer `2`: dense TP,
  replicated control/router, EP experts, sharded KV, and compression state are
  present and balanced across all eight GPUs with zero ownership mismatches.
- Sprints 239-242 now run a representative layer-2 TP/EP resident loop from
  production packed bytes at `32` slots / `256K`, MTP off. Sprint 242 fused
  the FP32 EP remote-sum into next-hidden compose, improving the 50-step
  layer-loop metric from `1.784008 ms/step` to `1.641832 ms/step` and from
  `17937.138290` to `19490.418145` slot-step tok/s while preserving checksum.
- Sprint 243 tested a first HMMA dense replacement in the same TP/EP path. It
  is correct/finite but slower (`3.533215 ms/step`) than the scalar dense
  control (`1.620386 ms/step`), so naive per-tile F8 decode into WMMA
  fragments is rejected.
- Sprint 244 measured the tensor-core dense ceiling for the same path:
  resident FP16/cuBLAS dense reduces dense time from `0.755645 ms/step` to
  `0.175605 ms/step` and improves the representative layer-loop metric to
  `1.050770 ms/step` / `30453.870979` slot-step tok/s. This validates dense
  as the next kernel target, while keeping expanded FP16 as diagnostic only.
- Sprint 245 added real memory admission for turning that diagnostic into a
  runtime option. At `32` slots / `256K` / F8 KV, the TP/EP contract reports
  `27.024 GiB` base per GPU including reserve and `27.701 GiB` per GPU if
  cacheable dense source tensors are replaced by FP16 runtime weights, leaving
  `4.299 GiB` physical headroom. Dense FP16 cache is therefore admissible as a
  runtime fallback/ceiling path, not a source-format change.
- Sprint 246 turned that admission into a real V100 allocation/conversion
  smoke. The separate TP/EP dense-cache tool materializes all `4096` dense TP
  rows into FP16 arenas: `13.459473 GiB` aggregate cache, `1.682434 GiB` per
  GPU, zero nonfinite values, PASS. This is now an executable runtime cache
  path, though not yet wired into the all-layer decode loop.
- Sprint 247 wired dense cache lookup into the representative layer-2 TP/EP
  resident decode loop. Cache-backed FP16/cuBLAS dense passes at `1.015128`
  ms/step and `31523.122614` slot-step tok/s, preserving the private-FP16
  checksum while using cache pointers. The remaining gap is lifting this from
  two composition tensors to a descriptor-selected dense table for every
  layer.
- Sprint 248 added that descriptor-selected dense execution table. The
  all-layer dense-table gate runs `510` transformer-layer groups and `4080`
  cache-backed FP16/cuBLAS GEMMs per 32-slot iteration, passing at
  `51.003671 ms/iteration`, `7.738345` dense-table TFLOP/s, and zero
  nonfinite outputs. The remaining gap is composing dense, EP, KV, and
  hidden-state flow into a resident all-layer TP/EP loop.
- Sprint 249 made the representative TP/EP full-layer smoke layer-parametric.
  Layers `0`, `1`, `2`, `3`, and `42` pass at `32` slots / `256K` with
  cache-backed FP16 dense compose, real TurboMind MXFP4 EP experts, sharded KV,
  and fused next-hidden composition. The representative decode-loop proxy now
  spans SWA-only, ratio-4, ratio-128, and late-layer cases with `0.999333` to
  `1.181511 ms/step`. The remaining gap is a resident all-layer TP/EP loop
  that preserves hidden shards across all 43 layers in one process.
- Sprint 250 added a one-process all-layer scaffold gate. The TP/EP full-layer
  smoke now supports `--all-layers` and passes all `43` transformer layers at
  `32` slots / `256K`. The 10-step gate reports `45.356852 ms/token` summed
  decode proxy and `705.516343` projected slot-step tok/s, with stage sums
  `12.009343 ms` EP, `8.064360 ms` dense, and `25.277469 ms` compose. This is
  still a scaffold because per-layer runtime/cache state is rebuilt; the next
  gap is making the all-layer loop truly resident.
- Sprint 251 hoisted dense FP16 cache materialization out of the per-layer
  runner in `--all-layers` mode. The shared all-layer cache has `4096` dense
  rows and `14451998720` cache bytes, builds once in `7772.591153 ms`, and the
  10-step all-layer gate still passes `43/43` layers. Wall time improves from
  `91879.358460 ms` to `74382.064295 ms`, and projected slot-step tok/s moves
  from `705.516343` to `731.369579`. The next residency targets are
  TurboMind/API handles, route buffers, expert bindings, and TP runtime state.
- Sprint 252 added an opt-in descriptor-check bypass for serving-shaped TP/EP
  scaffold runs. With shared dense cache and `--skip-descriptor-checks`, the
  10-step all-layer gate passes `43/43` layers with `descriptor_checks=0`,
  wall time drops to `46990.435640 ms`, and the projected decode proxy remains
  in the same range at `720.987187` slot-step tok/s. Strict descriptor checks
  remain the default validation gate.
- Sprint 253 repaired the decode-only all-layer harness path. With shared
  dense cache, descriptor checks off, and no one-shot compose validation, the
  10-step all-layer gate passes `43/43` layers at `44.035733 ms/token`
  summed decode proxy and `726.682578` projected slot-step tok/s. Wall time
  drops to `39951.007721 ms`. This is now the lightweight TP/EP scaffold
  benchmark to use after strict validation.
- Sprint 254 added `--skip-predecode-probes` for benchmark-only runs after
  strict validation. The all-layer decode-only gate passes `43/43` layers with
  `descriptor_checks=0` and `predecode_probes=0`, reducing wall time to
  `37819.503379 ms`. The summed decode proxy remains in the scaffold band at
  `44.848746 ms/token` / `713.509362` projected slot-step tok/s.
- Sprint 255 hoisted TurboMind API lifecycle across the all-layer TP/EP loop.
  The gate now records `shared_api=1`, passes `43/43` layers at `32` slots /
  `256K`, and reduces wall time to `35565.756621 ms`. The summed decode proxy
  is `43.957040 ms/token` / `727.983506` projected slot-step tok/s.
- Sprint 256 hoisted fixed rank buffers, route maps, streams/events, and lazy
  compose buffers across the all-layer TP/EP loop. The gate now records
  `shared_rank_buffers=1`, passes `43/43` layers, and reduces wall time to
  `33978.379725 ms`. The summed decode proxy is `43.895297 ms/token` /
  `729.007483` projected slot-step tok/s.
- Sprint 257 hoisted the TP runtime/KV allocator across the all-layer TP/EP
  loop. The gate now records `shared_tp_runtime=1`, passes `43/43` layers, and
  reduces wall time to `28437.257957 ms`. The summed decode proxy regressed to
  `46.024692 ms/token` / `695.278962` projected slot-step tok/s, so this is
  correct residency progress but needs repeat timing before performance
  promotion.
- Sprint 258 repeated the shared TP runtime path with a 50-step all-layer
  gate. The regression persisted at `45.672166 ms/token` /
  `700.645557` projected slot-step tok/s, while checksum stayed fixed. Shared
  runtime is correct residency progress, but Sprint 256 remains the current
  decode-speed base.
- Sprint 259 added a same-binary TP runtime A/B. Local per-layer TP runtime is
  the current decode-speed base at `42.723359 ms/token` /
  `749.004771` projected slot-step tok/s. Shared TP runtime remains opt-in
  because it regresses decode to `681.247356` projected slot-step tok/s.
- Sprint 260 added resident all-layer TurboMind expert bindings. Active MXFP4
  expert bytes now stay in VRAM across the 43-layer scaffold
  (`3449290752` bytes/GPU). The 50-step gate passes `43/43` layers with
  checksum `204721433`, reduces wall time to `14338.419135 ms`, and reports
  `44.131138 ms/token` / `725.111599` projected slot-step tok/s.
- Sprint 261 added EP+dense overlap with a separate dense stream per rank.
  The same-binary 50-step gate passes `43/43` layers and checksum
  `204721433`; projected scaffold throughput improves from `631.273270` to
  `846.062424` slot-step tok/s. Compose/all-to-all is now the dominant
  remaining stage.
- Sprint 262 rechecked FP16 EP return under the resident overlapped schedule.
  It is still rejected: projected throughput regresses from `831.795688` to
  `729.339500` slot-step tok/s because compose time increases.
- Sprint 263 tested direct peer-memory compose. It is rejected: direct remote
  reads regress projected throughput from `840.751688` to `634.454351`
  slot-step tok/s because compose time increases. Keep staged peer copies.
- Sprint 264 changed staged peer-copy scheduling from destination streams to
  source copy streams. It is promoted: projected throughput improves from
  `840.494594` to `999.490407` slot-step tok/s with checksum preserved.
- Sprint 265 added the first token-major serving-order scaffold. It passes
  `172/172` layer invocations for `4` token steps at `32` slots / `256K`,
  reporting `48.840011 ms/token` proxy and `655.200508` projected slot-step
  tok/s. This is closer to serving order, but still not generated-token
  serving throughput.
- Sprint 266 tested all-layer shared dense op residency in token-major mode.
  It remains correct but is not promoted: the shared-op cache regressed the
  token-major proxy from `51.991980` to `56.085843 ms/token`. Keep it as an
  opt-in diagnostic and keep the default dense op lifecycle local per layer.
- Sprint 267 rechecked shared TP runtime in token-major order and promoted it
  for token-major all-layer runs. The 4-step scaffold improves from
  `51.289549` to `47.902324 ms/token` proxy and cuts wall time from
  `34880.753622` to `11661.323548 ms`, with checksum preserved.
- Sprint 268 made token-major runs advance logical position per token step.
  The 4-step scaffold over positions `1024-1027` passes `172/172` invocations
  at `45.770462 ms/token` proxy and `699.140856` projected slot-step tok/s.
- Sprint 269 ran longer continuous token-major gates. The 32-step run passes
  `1376/1376` layer invocations at `39.290219 ms/token` proxy and
  `814.452062` projected slot-step tok/s. Compose/all-to-all is now the
  dominant measured stage: `742.079181 ms` compose versus `514.766496 ms` EP.
- Sprint 270 skipped same-GPU compose copies on the FP32 EP-return path. The
  16-step A/B improves from `40.271428` to `38.503412 ms/token` proxy, and the
  new 32-step topline is `37.912062 ms/token` / `844.058544` projected
  slot-step tok/s.
- Sprint 271 split compose timing into reduce/copy/final buckets and showed
  copy dominates. Sprint 272 tested per-destination copy streams and improved
  the 32-step scaffold topline to `36.911097 ms/token` / `866.947964`
  projected slot-step tok/s.
- Steering update: stop spending the next work cycle on compose/kernel
  micro-optimization. Focus on making TP/EP operational end-to-end with
  generated and continuation tok/s, then return to kernel selection/fusion
  with serving data.
- Sprint 273 added the first serving-shaped TP/EP metric bridge. Decode-only
  rates are now visible: `875.486234` aggregate generated tok/s and
  `931.549518` aggregate continuation tok/s at `32` slots / `256K` /
  `16` generated tokens. Wall throughput is still only `10.6 tok/s` because
  the scaffold calls the heavy per-layer runner for every token/layer.
- Sprint 274 made the TP/EP serving loop resident enough for useful
  operational metrology. With shared dense ops, `32` slots / `256K` /
  `32` generated tokens/request reports `669.222644` wall generated tok/s and
  `690.469286` wall continuation tok/s.
- Sprint 275 wrapped that resident TP/EP backend in a repeatable sustained
  serving artifact harness. The current tool-level V100 result at `32` slots /
  `256K` / `32` generated tokens/request is `749.304439` wall generated tok/s,
  `774.209856` wall continuation tok/s, `963.264018` decode-only generated
  tok/s, and `1000.823072` decode-only continuation tok/s with `32/32` token
  match. This is not yet the HTTP appliance server.
- Sprint 276 added a TP/EP-only resident HTTP harness. It keeps the TP runtime,
  dense cache, shared dense ops, rank buffers, and expert bindings loaded
  across HTTP requests and exposes `/health`, `/v100/status`, `/metrics`, and
  `POST /v100/selected-token`. The first HTTP smoke reports `719.275018` wall
  generated tok/s and `751.645517` wall continuation tok/s at `32` slots /
  `256K` / `32` generated tokens/request. It is operational as a smoke-tested
  server path, but not yet wired into the production launcher/deployment.
- Sprint 277 wired that server into `tools/ds4-v100-run-appliance.sh` via
  `DS4_V100_SERVE_MODE=tp-ep`. The launcher smoke reports `728.744669` wall
  generated tok/s and `753.022651` wall continuation tok/s at the same
  `32` slot / `256K` / `32` token shape.
- Sprint 278 added the sustained HTTP matrix driver for the launcher path. The
  current matrix reports `737.091414` wall generated tok/s at 32 tokens/request
  and `739.774102` at 64 tokens/request, both at `32` slots / `256K` with
  `32/32` token match.
- Sprint 279 made the Kubernetes deployment example point at the TP/EP
  appliance path and added GPU-utilization capture to the sustained HTTP
  matrix. The current V100 run reports `745.699174` wall generated tok/s for
  32 tokens/request and `753.708353` for 64 tokens/request, both at
  `32` slots / `256K` with `32/32` token match. GPU utilization during the
  short POST windows remains low: `15-19%` average and `38-40%` max.
- Sprint 280 extended the TP/EP HTTP harness from one generation POST per
  server to resident multi-request metrology. The current three-request V100
  matrix reports `751.114404` wall generated tok/s for 32 tokens/request and
  `762.277426` for 64 tokens/request, both at `32` slots / `256K`, with
  aggregate `96/96` token match per case. GPU utilization still peaks only at
  `40-41%`, so the next gap is request coalescing and compose/copy reduction.
- Sprint 281 exposed stage timing through the TP/EP HTTP artifacts. The
  current three-request matrix reports `742.897231` wall generated tok/s for
  32 tokens/request and `739.612937` for 64 tokens/request. The 64-token case
  shows compose-copy at `2569.208878 ms`, or `70.8%` of compose time, making
  compose-copy the next concrete performance target.
- Sprint 282 added event-wait compose copy and promoted it as the TP/EP
  appliance default. Same-binary 64-token serving A/B improves wall generated
  throughput from `752.669235` to `771.276064` tok/s while preserving
  aggregate `96/96` token match.
- Sprint 283 rechecked FP16 EP return under event-wait compose. It remains
  rejected: same-binary 64-token serving throughput regresses from
  `766.883263` to `635.936079` wall generated tok/s, despite preserving
  aggregate `96/96` token match. The FP32 return path stays default.
- Sprint 284 added compact route-compose and promoted it as the TP/EP
  appliance default. Same-binary 64-token serving A/B improves wall generated
  tok/s from `711.177884` to `791.453850`, with aggregate `96/96` token
  match. The 32-token compact sanity run reaches `802.701663` wall generated
  tok/s and `813.475877` wall continuation tok/s.
- Sprint 285 re-established the promoted default HTTP topline. At `32` slots /
  `256K` / three resident generation requests, the normal launcher path now
  reports `771.036527` wall generated tok/s for 32 tokens/request and
  `794.694599` for 64 tokens/request, both with aggregate `96/96` token match.
- Sprint 286 replaced the synthetic repeated-request serving measurement with
  true TP/EP HTTP request coalescing. At `32` slots / `256K`, `32`
  concurrent selected-token requests form one `coalesced_batch_size=32` batch.
  The practical-serving semantic baseline is now `721.446441` wall generated
  tok/s for 32 tokens/request and `787.316214` for 64 tokens/request, both
  with aggregate `32/32` token match.
- Sprint 287 added bucketed admission on top of coalescing. Mixed concurrent
  selected-token requests with pattern `32,64` now run as two same-length
  batches instead of being rejected: `32/32` token match, `bucketed_requests=16`,
  zero rejections, and `387.877251` wall generated tok/s over admitted client
  tokens. Uniform full-batch behavior remains intact at `759.490446` wall
  generated tok/s for 32 concurrent 32-token requests.
- Prior TP evidence remains useful:
  - TP8 sharded KV at `32` slots / `256K` fits, while replicated KV does not.
  - TP8 one-layer synthetic and FP16 fixture probes proved resident TP work can
    live inside an all-GPU boundary.
  - The current TurboMind MXFP4 TP8 shard-256 path failed correctness; TP4
    controls were correct but did not justify production integration.
  - Routed-only overlays and PP scheduler TP patches are rejected.

## Non-Negotiable Constraints

- No new PP/layer-split optimization sprints.
- No generic scheduler abstraction to support both PP and TP.
- TP/EP code uses separate files and a separate runtime ownership model.
- PP code may be read for reference and used as a frozen baseline, but not
  extended as the forward path.
- Single-slot tests are correctness/latency diagnostics only.
- Throughput evidence must use multi-slot server mode, report prompt tok/s,
  generated tok/s, continuation tok/s, GPU utilization, and confirm
  `active_microbatch == slots`.
- MTP stays out of the critical path until TP/EP serving is correct and
  measured.
- Performance work now follows the isolated gate discipline from
  `TEMP_THROUGHPUT_PROMPT.md`: one default-off CLI gate per activity, one
  same-binary V100 A/B per gate, promotion only with unchanged first token,
  preserved decode checksum, and improved GPU utilization or server decode
  tok/s.
- The active performance question is now launch-count and state-fragmentation
  reduction in the typed attention/KV and MoE paths. More dtype swaps, sidecar
  probes, or isolated single-layer kernels are secondary unless they directly
  reduce one of the measured serving hot spots.
- CUDA graph replay is not an active sprint path. Sprint 376 proved the current
  P2P transport is not stream-capture-compatible on this stack. A future graph
  revisit requires a scoped transport replacement plan; it should not displace
  S-C/S-D/S-E performance gates.
- Sprint 378 promoted compact MoE for the real model-router compact-compose
  path. Sprint 379 then tested S-E fused gated-SiLU and closed it as
  diagnostic-only: the generic epilogue changes tokens, while the DS4-clamped
  ABI is fast in isolation but not resident-serving validated.

## Throughput Optimization Pivot

Sprint 371 changed the performance diagnosis. At `32` slots / `256K`, server
decode stays roughly flat around `98` aggregate tok/s from `1` to `32` active
requests, while average GPU utilization stays around `10%`. Sprint 374 then
rejected a simple copied tc-grid INT8 swap for the BF16 compressor GEMMs.
`TEMP_THROUGHPUT_PROMPT.md` reframes the next work around the fact that the
runtime has many small launches and host synchronizations inside the steady
decode step, with no CUDA graph capture today.

The current throughput thesis is therefore:

```text
primary limiter = steady-state decode launch/sync latency and fragmented
                  per-layer CUDA work
not primary     = raw HBM bandwidth or a missing single dense GEMM dtype
```

This is not a permanent conclusion. It is the next falsifiable thesis. The
first two tests have already narrowed it: async output reduced output-head
syncs but regressed serving throughput, and CUDA graph replay is blocked by
P2P copy capture semantics. The roadmap therefore pivots from broad launch
replay to measured launch-count reducers in attention/KV and MoE.

The near-term implementation policy is now:

1. Execute one isolated gate at a time with same-binary V100 A/B.
2. Promote a gate only if it preserves first token/checksum or response-token
   parity and improves server decode tok/s or average GPU utilization at
   `32` slots / `256K`.
3. Do not reopen CUDA graphs unless there is a full P2P transport replacement
   plan.
4. Keep dtype conversion work scoped to measured hot paths. Sprint 374 showed
   a simple tc-grid INT8 compressor swap is slower than the FP16 tensor-op
   baseline at the target compressor shapes, so format changes are not the
   first response to the current low-utilization symptom.

The next performance sequence is ordered to test that thesis directly:

| Order | Gate | Purpose | Current status |
|---:|---|---|---|
| 1 | `--batched-paged-attn-gate` | Collapse per-slot/per-family typed-KV row store/load into block-table-indexed attention kernels | Closed diagnostic-only; row planner showed pending typed-history reloads already `0` |
| 2 | `--compact-moe-decode-gate` | Make real model-router top-k routes compatible with compact EP compose | Promoted for model-router compact compose |
| 3 | `--fused-gated-silu-gate` | Remove standalone clamp/SwiGLU launch by baking the DS4 clamp into the grouped-GEMM epilogue | Complete diagnostic; not promoted |
| 4 | `--tp-experts-ab-gate` | Measure TP-sharded expert execution against current EP8 all-to-all, without committing topology | Complete; no serving integration yet |
| 5 | `--fp8-e5m2-kv-gate` | Test alternate FP8 KV storage/load semantics for long-context serving | Complete diagnostic; promising short A/B, not promoted |
| 6 | `--mtp-decode-gate` | Add MTP only after base TP/EP decode metrology and launch strategy are stable | Deferred multiplier |

S-B is complete and rejected as a default. S-A is complete and rejected as a
promotion path: real capture is blocked by stream-capture-incompatible
`cudaMemcpyPeerAsync` transport in the existing TP/EP decode step. The roadmap
therefore pivots to S-C/S-D/S-E/S-F rather than extending audit-only graph work
or starting a broad P2P transport rewrite inside the graph sprint.

Current next step: move to S-F TP-sharded expert A/B as a topology measurement.
S-E can be revisited only after the resident dense-KV precheck failure under
`routed-normalized + fused-gated-silu` is diagnosed, or after a deterministic
fused-gate parity harness proves the DS4-clamped ABI against the two-step
reference.

## Production Readiness Sequence

The remaining work is ordered by what blocks practical use on the V100 box,
not by benchmark curiosity.

1. **Launch-count reduction gate.** Sprint 376 rejected real CUDA graph replay
   because peer-copy transport is not stream-capture compatible on this stack.
   Move directly to the throughput-prompt gates that reduce steady decode
   kernel count and fragmented state work: batched paged attention, compact
   MoE decode, and fused gated-SiLU. These gates should use the same 32-slot /
   256K A/B discipline and should not be mixed into one sprint.
2. **P2P transport revisit.** A future graph sprint is only justified after a
   scoped P2P transport plan exists for replacing `cudaMemcpyPeerAsync` with
   graph-capturable kernel/UVA copies across the whole steady decode step.
3. **Topology measurement gate.** Run the TP-sharded expert A/B if EP compose
   or all-to-all remains a measured bottleneck after launch-count work. This is
   a topology measurement, not a commitment to rip out EP8.
4. **Reference parity gate.** Keep first-token/checksum parity attached to
   every performance gate, then broaden it into fixed prompt suites and
   long-context cache-reuse checks at `128K` and `256K`.
5. **Persistent serving gate.** Run the TP/EP server as a long-lived appliance
   process with `MAX_REQUESTS=0`, readiness that reflects tokenizer/model/GPU
   residency, stable session reset/eviction semantics, overload behavior,
   cancellation/timeout handling, and operational logs/metrics.
6. **API completeness gate.** Finish role-aware multi-message chat parsing,
   stop/EOS behavior, streaming responses, and clear error contracts for bad
   requests, context overflow, queue saturation, and session conflicts.
7. **MTP gate.** Add MTP only after base TP/EP serving is correct and
   continuously benchmarkable, ideally after decode graph capture lands.
   MTP should be measured as a decode multiplier across `1`, `8`, `16`, and
   `32` active slots, not as a sidecar smoke.

## Sprint Sequence

### Sprint 307 - TP/EP Reference Parity Harness [complete]

Goal: Build a repeatable reference-comparison harness for the tokenizer-enabled
TP/EP server path.

Rationale: The API can now return text, but production readiness depends on
proving that the generated tokens are faithful DS4 behavior.

Outcome: Complete as a harness, failing as a production gate.
`tools/ds4-v100-tp-ep-reference-parity.py` now compares the live HTTP path
against official selected-token vectors. The first V100 run for
`short_reasoning_plain` expected `16` and received `ICC`, so semantic parity
remains the active blocker.

### Sprint 308 - TP/EP HC Semantic Parity [in progress]

Goal: Close the semantic gap exposed by Sprint 307 by replacing bridge HC
shortcuts with reference-faithful DS4 attention/FFN ordering and output-head
inputs.

Rationale: Persistent deployment would only make an incorrect model easier to
call. The next production sprint must identify and fix the source of the
selected-token mismatch before serving hardening or MTP.

Current finding: the mismatch is not an API-envelope issue. The TP/EP path
still uses synthetic EP routing and a simplified attention/FFN bridge. Work
now proceeds in this order: pack all local experts, add a router-driven EP
schedule, add FFN RMSNorm/router parity checks, then replace the attention
placeholder with the full DS4 attention sequence.

Progress: all-local-expert residency now builds and runs on the V100 pod. It
fits within the 32GB cards. Route buffers allocate for worst-case
`slots * top_k` per rank so a real router can produce imbalanced per-GPU
traffic without overrunning the old synthetic-route allocation. Routed
contributions carry per-route weights; the synthetic path uses `0.125` weights
to preserve behavior, while compose no longer owns a hardcoded EP scale.

Current router status: `DS4_V100_TP_EP_MODEL_ROUTER_ROUTES=1` loads router
weights, optional router bias, and optional token-hash expert IDs. The V100
active-mask run shows nonzero routes across early layers for an actual HTTP
reference request, but the top-token parity vector still fails:
`16` expected, ` ICC` returned, token `[61317]`, `164.721272` wall tok/s /
`237.349475` decode tok/s. A direct attempt to feed routed experts from
`ffn_normed` now has a separate diagnostic gate,
`DS4_V100_TP_EP_ROUTED_FFN_NORM_INPUT=1`, and fails immediately at layer `0`
with `decode_finite_bad=16384` / `rc=5`. Tensor stats show finite route input
but non-finite rank-7 TurboMind output and zero rank-1/rank-6 output. The
stable bridge currently uses FFN-normalized router logits with raw HC-current
routed expert input. The next parity work is to inspect selected expert IDs
and rank-local TurboMind pointer/scale bindings for layer-0 rank-7 locals
`30`/`21` and rank-6 locals `30`/`8`, then implement the true shared-FFN path,
then full DS4 attention semantics.

### Sprint 309 - Persistent Appliance Deployment Gate [planned]

Goal: Convert the current benchmark-run launcher into a persistent server gate.

Rationale: The V100 pod currently proves request batches, not long-lived
service operation. Production readiness needs `MAX_REQUESTS=0`, port-forward
or service access, readiness/metrics checks, graceful shutdown, overload
behavior, and a smoke that proves the server remains askable after repeated
sessions.

### Sprint 310 - API Semantics And Streaming [planned]

Goal: Finish the minimum practical chat API behavior around the TP/EP runtime.

Rationale: The current chat route is intentionally simple. Practical use needs
role-aware multi-message parsing, stop/EOS handling, streaming chunks, clear
context/queue/session errors, and compatible usage accounting.

### Sprint 311 - Prefill And Active-Slot Performance [planned]

Goal: Optimize the serving path after parity and API behavior are locked.

Rationale: Current throughput is dominated by correctness-oriented prefill and
the bridge HC sequence. The first production performance sprint should measure
prefill and decode separately, avoid full-32-slot work for low occupancy, and
only then tune kernels/fusion against the final graph shape.

### Sprint 312 - TP/EP MTP Decode Multiplier [tentative]

Goal: Add MTP to the TP/EP appliance as a measured decode accelerator.

Rationale: MTP is likely the largest user-visible speed multiplier, but it
should not be merged before base TP/EP serving is correct and operationally
measurable.

### Sprint 375 - Async Output Sync Removal [complete]

Goal: Implement `--async-output-gate`, a default-off gate that removes
steady-state host synchronization from the sampler/output path and makes the
token-major decode step eligible for CUDA graph capture.

Rationale: The active-slot matrix shows low, flat utilization at full
32-slot target shape. `TEMP_THROUGHPUT_PROMPT.md` identifies launch/sync
latency as the next highest-value performance thesis to test. S-B is the
required enabler before S-A CUDA graph capture.

Definition: audit `tools/ds4-v100-tp-ep-full-layer-smoke.cu` for
`cudaDeviceSynchronize` and `cudaStreamSynchronize` inside the steady-state
`run_one_step` region, move selected-token D2H to stream/event sequencing
where it is safely movable, and A/B the gate on the V100 pod with active-slot
matrix and same-binary HTTP profile evidence. Promote only if first
token/checksum remain stable and GPU utilization or server decode tok/s is
flat-or-up. If CPU token consumption still forces a synchronization at the
next-step embed seed, record the remaining dependency explicitly for Sprint
376 rather than hiding it.

Outcome: Implemented and rejected as a default. Direct smoke preserved first
token `98751` and output checksum `81959669916`, reducing output-head device
syncs from `26` to `0` with `8` event waits. The real HTTP `32` active request
/ `32` slot / `256K` A/B preserved first token `89340` and checksum
`101896170076`, but server decode regressed from `99.476540` to `93.764276`
tok/s and average GPU utilization stayed flat. The gate remains opt-in for
Sprint 376 graph-capture investigation.

### Sprint 376 - Decode CUDA Graph Capture [complete]

Goal: Implement `--decode-cudagraph-gate`, capturing the shape-static
32-wide per-rank decode step into CUDA graphs and replaying it across decode
steps.

Rationale: This is the make-or-break test of the launch-bound thesis. If graph
replay does not materially improve GPU utilization, later throughput work
should be replanned around a different bottleneck.

Definition: persistent graph input/output buffers, per-rank graph capture,
captured peer-copy compose/event dependencies, checksum-identical longer
decode run, and V100 A/B reporting utilization, decode tok/s, first token,
and all-layer checksum.

Planning update: Sprint 376 starts with a graph-capture audit, not immediate
graph replay. Sprint 375 showed output-head event sequencing is correct but
not a serving-speed win, and selected-token D2H still forces a CPU consumption
point outside `run_one_step`. The first deliverable is therefore an explicit
blocker/audit line for remaining graph blockers inside the token-major decode
step; replay follows only if the audit proves the region is capturable enough.

Execution note: Sprint 376 audit plumbing builds and runs on the
V100 pod. The first direct `32` slot / `256K` audit reports
`capture_eligible=0` with `172` broad in-step `sync_all` calls, `1376`
rank-stream waits, and `1376` dense-stream waits in one 43-layer decode step.
The first stream/event substitution pass preserved token/checksum parity and
moved those top-level wait counts to zero, but it remains slower before graph
capture and `capture_eligible` is still `0` because helper-level host
synchronizations remain. The HC-current helper pass removed one blocker class
and improved the graph-gated diagnostic from `44.247981` to `49.429146`
decode tok/s while preserving parity. The final-HC pass removed another
blocker class and reduced final-HC stage time, leaving five helper blocker
classes. The attention-projection pass removed one more blocker class, leaving
four. The raw-read/window pass removed another blocker class and improved the
graph-gated diagnostic to `54.144225` decode tok/s, leaving three blockers:
attention state, typed-history, and compressed-KV. The following
attention-state, typed-history, and compressed-KV passes preserved parity and
dropped helper blockers to zero. The latest direct audit reports
`capture_eligible=1`, blocker `none`, first token `54639`, output checksum
`24071637347`, scaffold checksum `3401922407`, and `54.788890` generated
decode tok/s before graph replay. The next concrete task is an actual CUDA
graph capture attempt.

Outcome: rejected as a promotion path. Separate stream captures fail with
`operation would result in a merge of separate capture sequences`; root stream
capture fails with `dependency created on uncaptured work in another stream`;
seeded root capture reaches the first HC-current `cudaMemcpyPeerAsync` and
fails with `operation not permitted when stream is capturing`; replacing that
HC-current peer copy with graph-gated device copy kernels moves the same error
to attention projection. The graph blocker is therefore pervasive P2P transport
inside the current decode step, not one residual host wait. The next sprint is
batched paged attention.

### Sprint 377 - Batched Paged Attention Gate [complete]

Outcome: Kept diagnostic-only. The row-family planner is useful, but the
observed typed-history pending reload count is already `0`, so the narrow
load-only S-C kernel was not promoted.

### Sprint 378 - Compact MoE Decode Gate [complete]

Outcome: Promoted for the real model-router compact-compose path. HTTP serving
A/B at `32` requests / `32` slots / `256K` preserved the response token stream,
improved client throughput from `37.394075` to `39.034685` tok/s, improved
server decode from `80.812914` to `81.313535` tok/s, and reduced compose time
from `19.167728` to `14.703119` ms.

### Sprint 379 - Fused Gated-SiLU Gate [complete]

Goal: Implement `--fused-gated-silu-gate` by moving the DS4 clamp/SwiGLU
boundary into the routed grouped-GEMM epilogue.

Rationale: This is a kernel-count reducer inside the routed FFN path and a
natural follow-on to compact MoE. It should bake the DS4 `10.0` routed-SwiGLU
clamp into the grouped-GEMM epilogue and avoid extra intermediates, while
preserving the source quantized weight layout.

Outcome: Not promoted. The current production-shaped model-router compact-MoE
branch already reports `routed_gate_standalone_swiglu=0`, so the fused flag is
a no-op there. The routed-normalized branch reports
`routed_gate_standalone_swiglu=1`; the generic TurboMind fused epilogue removes
it and improves direct proxy throughput from `45.368432` to `57.367413` tok/s,
but changes first token from `41432` to `54639`. A true DS4-clamped TurboMind
ABI was implemented and is fast in a layer-0 EP-only V100 run
(`4.102144` ms two-step gate versus `0.622592` ms fused gate), but resident
serving-shaped direct A/B fails before the routed gate executes because the
dense-KV precheck returns rc `4`. Keep the flag diagnostic-only.

### Sprint 380 - TP-Sharded Expert A/B [complete]

Goal: Implement `--tp-experts-ab-gate` as a measurement of TP-sharded experts
against the current EP8 all-to-all path.

Rationale: This is the topology experiment that settles whether expert
parallel all-to-all remains a structural bottleneck after launch-count work.
It is a measurement gate, not a commitment to rewrite the serving topology
before paged attention and compact MoE have been tested.

Outcome: Do not integrate TP-sharded experts into serving yet. Added
`tools/ds4-v100-tp-experts-ab.py`, a permanent measurement driver that writes
EP8 direct serving plus TP4/TP8 TurboMind workbench summaries. The V100 control
at `32` slots / `256K` / `position=262080` recorded EP8 direct decode
`66.569095` tok/s, first token `54639`, EP `18.220610` ms, and compose
`22.522762` ms. TP8 still fails correctness at `96`, `192`, and `384` routes.
TP4 is correct at all three route tiers but total speedup is only
`1.055x`, `0.891x`, and `0.927x`, so simple output reduction/compose erases
the compute win. Revisit only with a fused TP4 reduction/compose boundary.

### Sprint 381 - FP8 E5M2 KV Gate [complete]

Goal: Implement `--fp8-e5m2-kv-gate` for compressed/raw KV storage and loads.

Rationale: The typed KV memory plan already fits `32` slots / `256K`, so this
is a format and long-context optimization, not the current operational blocker.

Outcome: Implemented as a default-off diagnostic. E5M2 keeps the same
block-128 row layout as E4M3, with one E8M0 scale byte plus 128 FP8 payload
bytes. Typed-row and device-row V100 smokes passed for `attn`, `attn_raw`, and
`indexer`; E4M3 regression also passed after shared scale-byte cleanup.
Direct 4-token A/B at `32` slots / `256K` preserved checksum `13373834059`
and improved generated decode from `70.710875` to `75.787866` tok/s. HTTP
selected-token 4-token A/B returned `32/32` HTTP 200, preserved first token
`45178`, improved client tok/s from `17.212677` to `22.389190`, and reduced
compressed-KV sum from `491.310011` to `442.415827` ms. Keep E4M3 as the
default until longer parity/soak is complete because E5M2 has lower mantissa
precision and one immediate HTTP candidate run hit CUDA OOM before readiness.

### Sprint 382 - MTP Decode Gate [tentative]

Goal: Wire MTP into the TP/EP serving loop as a measured decode multiplier.

Rationale: MTP is likely a large user-visible multiplier, but it should run
after graph capture or another stable launch-amortization strategy exists;
otherwise acceptance work may be hidden by per-step launch overhead.

### Sprint 226 - TP/EP Planner And Topology Contract [complete]

Goal: Create a TP-only planner and topology report for `PP1/TP8/EP8` at
`32` slots / `256K`.

Rationale: The PP planner carries legacy assumptions that will fight the new
topology. The TP path needs its own memory, KV, expert, collective, and slot
admission contract before runtime work starts.

Outcome: Complete. `tools/ds4-v100-plan-tp.c` is now a TP8/EP8-only planner
with sharded KV, expert ownership, route-density, admission-tier, and
collective/EP traffic reporting. The real-pack V100 run reports `145.42 GiB`
total resident weight bytes, `27.00 GiB` per-GPU total at `32` slots / `256K`
/ F8 KV, and admission of `63` slots at `256K` under current assumptions.

### Sprint 227 - TP8 Collective Workbench [complete]

Goal: Build TP-only collective smokes for hidden all-reduce, reduce-scatter,
all-gather, and expert-output reduction across all eight V100s.

Rationale: The suspected TP risk is not raw NVLink bandwidth alone; it is
latency, synchronization, and whether collectives can stay resident and
overlapped inside the layer boundary.

Outcome: Complete. `tools/ds4-v100-tp8-collective-workbench` now measures
`allreduce`, `reduce-scatter`, `allgather`, `rs-ag`, and `ep-reduce` modes.
At 32 tokens, the hidden all-reduce proxy is `26.904544 ms` and the EP reduce
proxy is `27.436756 ms`; both pass correctness. At 128 tokens they improve to
`3332.257` and `3253.920` overhead-only tok/s respectively.

### Sprint 228 - TP/EP Pack Contract [complete]

Goal: Emit a TP/EP pack layout with dense TP shards, EP expert ownership, KV
shard descriptors, and per-GPU memory accounting.

Rationale: Runtime work should not reinterpret PP pack metadata. The pack
format must encode the TP/EP ownership model directly.

Outcome: Complete. `tools/ds4-v100-tp-ep-pack-contract` emits
`tp-ep-pack-contract.tsv`, `tp-ep-memory-summary.tsv`, and
`tp-ep-pack-contract.md`. The real-pack contract has `4096` dense TP rows,
`5496` replicated control/router rows, `688` EP expert rows, and `840`
KV/state rows. Per-GPU total is `27.024 GiB` at the target shape.

### Sprint 229 - TP Runtime Skeleton [complete]

Goal: Add a new TP-only runtime skeleton that opens all eight GPUs, allocates
resident hidden/KV/scratch arenas, and executes no-op or fixture layer passes.

Rationale: The runtime must prove ownership, lifecycle, and memory residency
without touching `ds4_v100_scheduler.*` as a shared abstraction.

Outcome: Complete. `ds4_v100_tp_runtime.{h,cu}` and
`tools/ds4-v100-tp-runtime-smoke.cu` now provide a separate TP runtime
skeleton. The V100 smoke allocates `7061329920` runtime bytes per GPU before
weights at the target shape and verifies fixture output with
`fixture_max_abs=0`.

### Sprint 230 - TP Dense And KV Slice [complete]

Goal: Implement a bounded dense-attention/KV slice in the TP runtime, including
sharded DS4 compressed KV at the `32` slot / `256K` target.

Rationale: TP must keep hidden state and KV in native sharded layout across
layers. This sprint answers whether dense paths and KV are viable before MoE
complexity is added.

Outcome: Complete. `ds4_v100_tp_runtime_dense_kv_slice` now computes
per-layer, per-slot sharded KV offsets and writes/reads deterministic resident
KV rows on all eight GPUs. At the target `32` slots / `256K` / F8 KV shape,
the runtime allocates `7122628608` bytes per GPU before weights. Layer 2
ratio-4 with indexer KV passes at `attn_row=384`, `indexer_row=256`,
`attn_row_bytes=65`, `indexer_row_bytes=17`, and `max_abs=0`. Layer 3
ratio-128 without indexer KV passes at `attn_row=192`, `attn_row_bytes=65`,
and `max_abs=0`. This keeps the TP runtime path viable and moves the next
implementation gate to EP routed experts.

### Sprint 231 - EP Routed Expert Slice [complete]

Goal: Implement a bounded EP routed-expert slice using real low-bit expert
kernels and measure expert dispatch, route imbalance, and grouped GEMM density
at `32` active slots.

Rationale: Expert execution dominates the useful work. EP is only valuable if
active slots create dense enough expert batches and dispatch/reduction does not
erase the kernel gains.

Outcome: Complete. `tools/ds4-v100-tp-ep-expert-smoke.cu` models EP8
ownership as `256` global experts and `32` local experts per GPU, then runs
the real TurboMind MXFP4 grouped gated-SiLU and grouped down kernels on all
eight V100s. At `32` slots / `top_k=6`, it reports `192` aggregate routes,
`1.5 MiB` dispatch, `1.5 MiB` return, balanced route imbalance `1.0`,
`repeat_max_abs=0`, `repeat_bad=0`, `repeat_nan=0`, and `PASS`. Rank `7` is
the slow rank at `0.249378 ms` versus roughly `0.059 ms` on ranks `0-6`, so
per-rank timing must remain visible in Sprint 232.

### Sprint 232 - One-Layer TP/EP Correctness Gate [complete]

Goal: Execute one TP/EP fixture layer that combines the separate TP runtime,
sharded KV, and real low-bit EP expert kernels.

Rationale: This is the first point where the separate TP runtime lifecycle,
sharded KV, and EP experts meet in one process before descriptor-backed real
layer data is introduced.

Outcome: Complete as a fixture gate. `tools/ds4-v100-tp-ep-layer-smoke.cu`
links the separate TP runtime with the TurboMind MXFP4 ABI in one process. At
`32` slots / `256K` / `top_k=6`, it opens the target runtime arenas, verifies
layer-2 ratio-4 KV with `max_abs=0`, executes `192` aggregate EP routes,
reports `1.5 MiB` dispatch and `1.5 MiB` return, and passes finite deterministic
repeat output. The fixture one-layer envelope is `1.321812 ms`, with
`1.078032 ms` in the dense/KV fixture and `0.243780 ms` worst-rank EP time.
Next: replace fixture weights/routes with descriptor-driven one-real-layer
TP/EP correctness while preserving the separate codepath.

### Sprint 233 - Descriptor Driven TP/EP Layer Gate [complete]

Goal: Validate real production-pack TP/EP contract descriptors for one
representative layer.

Rationale: Sprint 232 proved fixture execution. Before running real layer data,
the TP/EP path must prove that the production pack contract contains the dense,
control/router, EP expert, KV, and compression rows needed by the separate
runtime.

Outcome: Complete as a descriptor ownership gate. Layer `2` resolves to
`288` rows: `112` dense TP, `136` replicated control/router, `16` EP expert,
`16` KV shard, and `8` compression-state rows. Each GPU owns `36` rows and
`711945176` estimated bytes, with expert spans `0..31` through `224..255` and
zero ownership mismatches. This does not yet bind real bytes into execution;
that is the next sprint.

### Sprint 234 - Descriptor-Backed One-Layer Execution [complete]

Goal: Bind the layer-2 TP/EP descriptor rows to actual production-pack byte
spans and feed descriptor-derived expert pointers into the one-layer TP/EP
smoke.

Rationale: Descriptor ownership is now proven, but the runtime still executes
synthetic MXFP4 fixtures. The next gate must load real descriptor-backed
weights for at least the routed expert path before scaling layers.

Outcome: Complete for routed experts. `tools/ds4-v100-tp-ep-layer-smoke.cu`
now has a descriptor-backed expert mode that parses the production
`turbomind-pack-index.tsv`, loads layer-2 real packed expert weight/scale bytes,
and feeds descriptor-derived pointer tables into the TurboMind MXFP4 EP
kernels on all eight V100s. At `32` slots / `256K` / `top_k=6`, the run passes
with `192` aggregate routes, `641728512` descriptor bytes read,
`worst_ep_ms=0.246647`, `dense_kv_ms=1.121624`, `one_layer_ms=1.368271`,
KV `max_abs=0`, and deterministic finite repeat output. This is still not
serving and not logits-equivalent; dense/control/router/attention descriptor
execution is the next gate.

### Sprint 235 - Descriptor-Backed Full-Layer TP/EP Scaffold [complete]

Goal: Expand from descriptor-backed routed experts to a full layer-2 TP/EP
scaffold that parses, loads, and device-checks dense/control descriptors,
preserves sharded KV correctness, and runs descriptor-backed EP experts with
MTP off.

Rationale: TP is not operational until every layer family has a concrete
descriptor-backed runtime binding. Sprint 234 proved expert bytes; Sprint 235
must prove that the full-layer ownership model can bind real dense/control,
KV/state, and expert rows in the separate TP/EP codepath before replacing
checksum stages with true DS4 math and scaling to all 43 layers.

Outcome: Complete as a scaffold gate. `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
now parses the real TP/EP contract, binds all layer-2 descriptor families,
device-checks real dense/control bytes on the owning V100s, preserves sharded
KV correctness, and runs descriptor-backed TurboMind EP experts. At `32`
slots / `256K` / `top_k=6`, the run passes with `288` total layer rows,
`163102720` dense bytes checked, `84041408` control bytes checked,
`641728512` EP bytes loaded, KV `max_abs=0`, `worst_ep_ms=0.249378`, and
finite deterministic repeat output. This remains a scaffold, not a
logits-equivalent layer; the descriptor load/check time is startup evidence,
not serving throughput.

### Sprint 236 - Descriptor-Backed TP Dense Compute Gate [complete]

Goal: Replace one Sprint 235 dense checksum stage with real low-bit dense
computation for `blk.2.attn_q_a.weight`, using packed F8 source bytes from the
production pack and executing a TP8 row-sharded dense kernel on all V100s.

Rationale: The full-layer scaffold is not a logits-equivalent layer. The next
gate must prove that descriptor-backed packed dense bytes can feed GPU compute
inside the TP/EP path before expanding that pattern to the rest of attention
and shared dense math.

Outcome: Complete for one representative dense tensor. The TP/EP full-layer
smoke now resolves `blk.2.attn_q_a.weight`, loads real packed F8 E4M3 block-128
TP shards from the production pack, expands F8 values inside a CUDA kernel, and
computes `32` slots x `128` local rows x `4096` columns on all eight V100s.
The V100 run passes with `dense_compute_ms=0.081783`, exact repeat,
`dense_compute_oracle_max_abs=0.000000007`, KV `max_abs=0`, and the existing
descriptor-backed EP path still passing. This is not yet optimized HMMA/CUTLASS
dense math and not full-layer logits equivalence, but it proves the packed
dense compute path inside TP/EP.

### Sprint 237 - Layer-2 Dense Coverage Gate [complete]

Goal: Extend the Sprint 236 packed-F8 dense compute gate from one tensor to
all compatible layer-2 F8 dense TP tensor groups, with per-tensor timing,
repeat, and CPU oracle checks.

Rationale: Serving should not start from a path where only one dense tensor can
compute. The TP/EP layer needs broader dense-family coverage before full-layer
decode and serving gates are meaningful.

Outcome: Complete for layer-2 F8 dense tensors. The TP/EP full-layer smoke now
supports `--dense-compute-all-f8`, discovers all compatible layer-2 F8 dense TP
tensor groups, and executes all nine groups from packed production bytes. The
V100 run passes with `141606912` packed bytes loaded, worst dense compute time
`0.654029 ms`, exact repeat, worst CPU oracle error `0.000000015`, KV
`max_abs=0`, EP `worst_ep_ms=0.241766`, and final `PASS`. BF16 dense/control
math and real layer dataflow remain open.

### Sprint 238 - Layer-2 BF16 Dense Coverage Gate [complete]

Goal: Extend dense coverage to layer-2 BF16 compressor/indexer TP tensors,
expanding BF16 inside CUDA kernels and validating repeat plus CPU oracle checks
on all V100s.

Rationale: Sprint 237 covered F8 dense families. BF16 compressor/indexer
tensors are the remaining dense coverage gap before representative full-layer
dataflow can be composed.

Outcome: Complete for layer-2 BF16 dense tensors. The TP/EP full-layer smoke
now supports `--dense-compute-all-bf16` and combined `--dense-compute-all`.
It discovers all compatible layer-2 BF16 `dense_tp` groups, loads production
pack bytes, expands BF16 inside CUDA code, and validates repeat plus bounded
CPU oracle checks on the V100 pod. The BF16-only run covers five tensors with
`21495808` bytes loaded, worst BF16 compute time `0.047206 ms`, exact repeat,
and worst CPU oracle error `0.000000119`. The combined run preserves all nine
F8 dense checks with `dense_compute_pass=1`, reports `bf16_compute_pass=1`,
keeps KV `max_abs=0`, measures `worst_ep_ms=0.250368`, and ends in final
`PASS`. The next gap is no longer dense coverage; it is composing the real
layer dataflow into a next hidden state.

### Sprint 239 - Full-Layer TP/EP Decode [complete]

Goal: Combine descriptor-backed dense coverage, control/router handling,
sharded KV, and EP experts into a representative full layer that produces a
real next hidden state with MTP off.

Rationale: The current path proves bytes, KV, experts, and one dense compute
gate independently. Full-layer decode must connect those pieces into the layer
dataflow before serving.

Outcome: Complete for representative layer-2 next-hidden composition. The
TP/EP full-layer smoke now supports `--compose-next-hidden`, builds route-slot
mapping for the EP schedule, reduces TurboMind routed expert down outputs into
512-wide TP destination hidden shards, peer-copies those contributions across
all eight V100s, and composes resident next-hidden shards from
`blk.2.attn_output_b.weight`, `blk.2.ffn_down_shexp.weight`, returned EP
contributions, and deterministic residual input. The 32-slot/256K V100 run
passes with `ep_contribution_bytes=4194304`, `ep_return_bytes=4194304`,
`attn_dense_ms=0.555213`, `shared_dense_ms=0.153702`, `compose_ms=3.707477`,
checksum `4112649481`, `finite_bad=0`, exact repeat, and `compose_pass=1`.
The same run preserves combined F8/BF16 dense coverage, KV `max_abs=0`,
`worst_ep_ms=0.255590`, and final `PASS`. This is still not production
serving or logits equivalence, but it is the first resident TP/EP layer
composition gate.

### Sprint 240 - TP/EP Resident Decode Loop Gate [complete]

Goal: Convert the Sprint 239 one-shot TP/EP composition path into a resident
repeated decode-loop benchmark at `32` slots / `256K`, MTP off.

Rationale: Before server integration, the TP/EP path needs a benchmarkable
resident loop that avoids pack-byte reloads and per-step allocation.

Outcome: Complete for a representative layer-2 resident loop. The TP/EP
full-layer smoke now supports `--decode-steps N`, keeps the two F8 dense
composition tensors resident, keeps TurboMind EP weights and composition
buffers resident, and repeats EP+dense+peer-return+compose without rereading
pack bytes. The V100 pod run at `32` slots / `256K`, MTP off, `50` steps
passes with `ms_per_step=1.845548`, `slot_step_tok_s=17339.021356`,
`ep_ms_per_step=0.319095`, `dense_ms_per_step=0.756244`,
`compose_ms_per_step=0.770121`, checksum `2382924023`, `finite_bad=0`, and
`decode_pass=1`. Existing F8/BF16 dense coverage, KV check, and Sprint 239
composition still pass. This is not generated tok/s; it is the first resident
TP/EP layer-loop metric.

### Sprint 241 - TP/EP FP16 EP Return A/B [complete]

Goal: Add an opt-in FP16 EP return path and measure whether halving peer
payload improves the Sprint 240 resident loop.

Rationale: Sprint 240 showed compose/peer synchronization is a major stage
cost. FP16 return is the smallest isolated communication optimization.

Outcome: Complete and rejected as a default. `--ep-return-fp16` halves the
reported EP return payload from `4194304` bytes to `2097152` bytes and passes
finite/checksum validation, but it slows the 50-step resident loop from
`1.788149 ms/step` to `1.937399 ms/step`. Compose time rises from
`0.713836 ms/step` to `0.859697 ms/step`, so the added cast and expand kernels
cost more than the reduced peer payload saves. Keep FP32 return as default;
keep FP16 return as an opt-in diagnostic and revisit only if fused into the
EP reduction or next-hidden compose.

### Sprint 242 - TP/EP Fused Remote-Sum Compose [complete]

Goal: Fuse the FP32 EP remote contribution sum into next-hidden compose for
the separate TP/EP full-layer smoke.

Rationale: Sprint 241 showed standalone FP16 EP return is correct but slower.
The bottleneck is extra kernel/synchronization boundaries, not raw peer-copy
payload bytes.

Outcome: Complete. `--fuse-compose-sum` removes the destination `ep_sum` zero
kernel and eight add kernels per destination rank. Same-binary A/B at `32`
slots / `256K`, MTP off, and `50` resident steps: baseline FP32 return passes
at `1.784008 ms/step`, `17937.138290` slot-step tok/s, and
`0.713663 ms/step` compose; fused compose/sum passes with the same checksum at
`1.641832 ms/step`, `19490.418145` slot-step tok/s, and `0.568906 ms/step`
compose. Keep FP32 return and continue fusing TP/EP synchronization boundaries
before server integration.

### Sprint 243 - TP/EP Dense HMMA Compose Gate [complete]

Goal: Test a bounded HMMA dense replacement for the two F8 composition tensors
used by the representative TP/EP resident loop.

Rationale: After Sprint 242, scalar F8 dense compute is the largest measured
stage. V100 should compute low-bit dense paths by expanding/dequantizing on GPU
into FP16 HMMA fragments, not by scalar FP32 dot products.

Outcome: Complete and rejected as a default. `--dense-hmma-compose` adds a
32-slot-capable WMMA/HMMA kernel that keeps F8 bytes resident and decodes each
tile into FP16 fragments before FP32 accumulation. It passes finite/repeat
checks, but it slows the fused-compose resident loop from `1.620386 ms/step`
and `19748.386791` slot-step tok/s to `3.533215 ms/step` and
`9056.907248` slot-step tok/s. Dense time rises from `0.753941 ms/step` to
`2.667910 ms/step`. Keep this as a diagnostic only; the next dense path should
reuse/adapt the older shape-specific F8 HMMA kernels or use a prepacked,
software-pipelined low-bit dense design.

### Sprint 244 - TP/EP Resident Dense Tensor-Core Ceiling [complete]

Goal: Measure the best-case dense-stage improvement when the two F8
composition tensors are expanded once into resident FP16 buffers and executed
with cuBLAS FP16 Tensor Core GEMM.

Rationale: Sprint 243 rejected the naive HMMA implementation, but did not
answer whether dense tensor-core execution is worth pursuing. A resident FP16
ceiling separates the value of the compute shape from the cost of low-bit
decode/layout feeding.

Outcome: Complete as a diagnostic ceiling. `--dense-f16-cublas-compose`
expands packed F8 to resident FP16 during setup for the two layer-2
composition tensors, converts resident activations to FP16, and uses
`cublasGemmEx` to produce FP32 output shards. Same-binary A/B at `32` slots /
`256K`, MTP off, fused compose enabled, and `50` resident steps: scalar dense
passes at `1.685018 ms/step`, `18990.892348` slot-step tok/s, and
`0.755645 ms/step` dense; resident FP16/cuBLAS passes at
`1.050770 ms/step`, `30453.870979` slot-step tok/s, and `0.175605 ms/step`
dense. This is a `1.60x` layer-loop improvement and a `4.30x` dense-stage
improvement. Keep the path diagnostic; build a packed low-bit dense production
kernel next.

### Sprint 245 - TP/EP Dense FP16 Cache Admission Gate [complete]

Goal: Decide whether the Sprint 244 resident FP16 dense ceiling can fit inside
the target `32` slot / `256K` TP/EP appliance memory budget.

Rationale: V100 cannot execute BF16/FP8/FP4 natively. The source model should
remain quantized, but a practical runtime can materialize selected dense
execution weights into FP16 if that materially improves tensor-core utilization
and still fits in VRAM.

Outcome: Complete. `tools/ds4-v100-tp-ep-pack-contract` now reports dense
FP16 runtime cache admission from real pack metadata. Against the production
pack at `32` slots / `256K` / F8 KV, base memory is `27.024 GiB` per GPU
including the `2.0 GiB` reserve. F8 dense packed bytes eligible for FP16 cache
are `0.687 GiB` per GPU, the FP16 cache is `1.364 GiB`, BF16 dense shadow is
`0.319 GiB`, and the practical replace-source total is `27.701 GiB` per GPU.
That leaves `4.299 GiB` physical headroom. Dense FP16 cache is memory
admissible as a runtime option; next implement the dense-cache loader/runtime
path for all dense tensors, then benchmark the resident all-layer path.

### Sprint 246 - TP/EP Dense FP16 Cache Runtime Smoke [complete]

Goal: Materialize the dense FP16 runtime cache on the V100 pod from the real
TP/EP contract.

Rationale: Sprint 245 proved the memory budget on paper. The next risk was
whether the runtime can allocate the arenas, stage packed source shards,
convert all dense F8/BF16 tensors on GPU, and keep the cache resident without
bad values.

Outcome: Complete. `tools/ds4-v100-tp-ep-dense-cache-smoke` is a new
TP/EP-only CUDA tool. It allocates one dense FP16 cache arena per GPU and
converts `f8_e4m3_b128` and `bf16` dense shards from the production pack into
that arena. Layer-2 passes with `112` dense rows and `0.281738 GiB` aggregate
cache. The full contract passes with `4096` dense rows, `8.047012 GiB`
aggregate source bytes, and `13.459473 GiB` aggregate FP16 cache. Per GPU:
`512` rows, `1.005877 GiB` source, `1.682434 GiB` FP16 cache, `126.250 MiB`
max temp staging, and zero nonfinite values. Next wire this arena into the
resident TP/EP layer execution path and benchmark all-layer decode.

### Sprint 247 - TP/EP Dense Cache Compose Integration [complete]

Goal: Wire the dense FP16 cache arena into the representative TP/EP resident
decode loop.

Rationale: Sprint 246 proved all dense rows can be cached, but execution still
used private FP16 copies for the two composition tensors. The runtime must
look up cache-resident weights by tensor and GPU if this is going to become a
serving path.

Outcome: Complete. `--dense-f16-cache-compose` builds a layer-local dense
cache from contract rows and makes the resident FP16/cuBLAS dense path use
cache pointers. Same-binary A/B/C at `32` slots / `256K`, MTP off, fused
compose, and `50` resident steps: scalar dense passes at `1.642514 ms/step`
and `19482.326340` slot-step tok/s; private FP16/cuBLAS passes at
`1.056807 ms/step` and `30279.894858`; cache-backed FP16/cuBLAS passes at
`1.015128 ms/step` and `31523.122614`. The cache-backed path emits
`dense_f16_cache=1`, preserves checksum `2515001`, and materializes `112`
layer-2 dense rows into `302514176` cache bytes. Next lift this into a
descriptor-selected dense execution table for every layer.

### Sprint 248 - TP/EP All-Layer Dense Execution Table [complete]

Goal: Build and validate a descriptor-selected dense execution table across
the transformer layers.

Rationale: The layer-2 cache-backed decode path still selected two dense
tensors by name. TP/EP serving needs the runtime to enumerate dense work from
the contract across all layers.

Outcome: Complete. `tools/ds4-v100-tp-ep-dense-cache-smoke` now supports
`--execute-table`, which groups complete `dense_tp` rows by `(layer,
tensor_id)` and runs cache-backed FP16/cuBLAS GEMMs for each group on all TP
ranks. The layer-2 gate passes with `14` groups, `112` GEMMs per iteration,
and `1.384323 ms/iteration`. The all-layer gate passes with `510`
transformer-layer groups, `4080` GEMMs per iteration, `394684006400` FLOPs
per iteration, `51.003671 ms/iteration`, `7.738345` dense-table TFLOP/s,
checksum `15841839914005485`, and zero nonfinite outputs. Next compose this
dense table with EP routed experts, KV/update, and hidden-state flow in a
resident all-layer TP/EP loop.

### Sprint 249 - TP/EP Layer-Parametric Resident Loop [complete]

Goal: Remove layer-2 hardcoding from the representative TP/EP full-layer smoke
and validate the DS4 layer families needed for an all-layer loop.

Rationale: Sprint 248 proved all-layer dense table enumeration, but the
resident decode loop still selected layer-2 composition tensors and ratio-4 KV
behavior. The next all-layer loop needs layer-local tensor names and the DS4
SWA/ratio-4/ratio-128 compression schedule to be correct before iterating all
43 layers.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now derives
composition tensors from `--layer N` and selects indexer KV only for ratio-4
layers. The V100 representative gate at `32` slots / `256K`, MTP off,
cache-backed FP16 dense compose, real TurboMind MXFP4 EP experts, and fused
compose passes layers `0`, `1`, `2`, `3`, and `42`. Decode-loop proxy timing
ranges from `0.999333` to `1.181511 ms/step`, or `27083.969701` to
`32021.345429` slot-step tok/s. The final scaffold accepts `comp_rows=0` only
for SWA-only layers and still requires compression rows for ratio-4/ratio-128
layers. Next build the resident all-layer TP/EP loop with hidden shards carried
through all layers in one process.

### Sprint 250 - TP/EP All-Layer Scaffold Gate [complete]

Goal: Add a single-process all-layer scaffold gate for the separate TP/EP path.

Rationale: Sprint 249 proved representative layer families, but the workflow
still required shell orchestration. Before server integration, the TP/EP path
needs one command that exercises all 43 transformer layers and reports an
aggregate decode proxy.

Outcome: Complete as a scaffold. `tools/ds4-v100-tp-ep-full-layer-smoke` now
supports `--all-layers`, emitting one `tp_ep_all_layer_item` row per layer and
a final `tp_ep_all_layer_scaffold` aggregate. On the V100 pod at `32` slots /
`256K`, MTP off, cache-backed FP16 dense compose, real TurboMind MXFP4 EP
experts, and fused compose, both all-layer gates pass `43/43` layers. The
10-step gate reports `45.356852 ms/token` summed decode proxy,
`705.516343` projected slot-step tok/s, `12.009343 ms` summed EP,
`8.064360 ms` summed dense, `25.277469 ms` summed compose, and checksum
`6174401222`. This remains scaffold evidence because runtime/cache/TurboMind
state is still recreated per layer inside the process. Next make the 43-layer
loop truly resident.

### Sprint 251 - TP/EP Shared Dense Cache Residency [complete]

Goal: Hoist dense FP16 cache materialization out of the per-layer all-layer
runner.

Rationale: Sprint 250's all-layer gate was one process, but not resident: each
layer rebuilt dense cache state. Dense cache is both large enough to matter and
already memory-admitted for `32` slots / `256K`, so it is the right first
state-hoist.

Outcome: Complete. In `--all-layers` mode, the full dense contract is parsed
once and materialized into a shared FP16 cache with `4096` rows and
`14451998720` cache bytes. The cache builds in `7772.591153 ms` and is reused
across all 43 layer scaffolds. The 10-step V100 gate passes `43/43` layers,
improves wall time from `91879.358460 ms` to `74382.064295 ms`, and improves
the summed decode proxy from `45.356852 ms/token` to `43.753529 ms/token`
(`731.369579` projected slot-step tok/s). Next hoist TurboMind/API handles,
route buffers, expert bindings, and TP runtime state.

### Sprint 252 - TP/EP Descriptor Check Bypass [complete]

Goal: Add an opt-in way to skip dense/control descriptor byte checks for
serving-shaped all-layer scaffold measurements.

Rationale: Descriptor byte checks are validation work, not serving work. After
the pack has passed strict descriptor validation, the all-layer loop should not
reread and checksum dense/control rows every layer.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-descriptor-checks`. The default remains strict. With shared dense
cache, `--compose-next-hidden`, and descriptor checks disabled, the 10-step
V100 gate passes `43/43` layers at `32` slots / `256K`, reports
`descriptor_checks=0`, cuts wall time from `74382.064295 ms` to
`46990.435640 ms`, and reports `44.383590 ms/token` summed decode proxy
(`720.987187` projected slot-step tok/s). A decode-only run exposed a smoke
harness `invalid resource handle` path; keep compose validation enabled until
that is fixed.

### Sprint 253 - TP/EP Decode-Only Harness Repair [complete]

Goal: Restore the decode-only all-layer scaffold benchmark.

Rationale: Sprint 252's descriptor-bypass path still needed
`--compose-next-hidden` enabled to avoid a harness failure. That extra one-shot
compose validation is not serving-shaped and should not be required for the
standard scaffold benchmark.

Outcome: Complete. `prepare_resident_f8_dense()` now drains stale per-device
CUDA error state before launching local dense setup conversion kernels. The
decode-only all-layer V100 gate passes `43/43` layers at `32` slots / `256K`,
shared dense cache, descriptor checks off, and MTP off. It reports
`44.035733 ms/token` summed decode proxy, `726.682578` projected slot-step
tok/s, `11.804094 ms` summed EP, `7.744769 ms` summed dense,
`24.482197 ms` summed compose, and `39951.007721 ms` wall time. Next hoist
TurboMind/API handles, route buffers, expert bindings, and stream/event
lifecycle across the 43-layer loop.

### Sprint 254 - TP/EP Pre-Decode Probe Bypass [complete]

Goal: Add an opt-in benchmark mode that skips pre-decode validation probes.

Rationale: After strict gates pass, the serving-shaped scaffold should not run
extra isolated TurboMind warmup/timing/repeat probes before each layer's decode
loop.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-predecode-probes`. The default strict behavior remains unchanged. With
shared dense cache, descriptor checks disabled, predecode probes disabled, and
decode-only all-layer mode, the V100 gate passes `43/43` layers at `32` slots /
`256K`. It reports `predecode_probes=0`, `44.848746 ms/token` summed decode
proxy, `713.509362` projected slot-step tok/s, and `37819.503379 ms` wall
time. Use this only as a lightweight benchmark mode after strict validation.

### Sprint 255 - TP/EP Shared TurboMind API [complete]

Goal: Hoist TurboMind dynamic library and API lifecycle across the all-layer
TP/EP scaffold.

Rationale: Sprint 254 removed benchmark-only probes, but each layer still
performed TurboMind `dlopen`, eight-device init, shutdown, and `dlclose`.
Serving should initialize that state once and reuse it across the decode loop.

Outcome: Complete. `--all-layers` now opens TurboMind once, initializes all
eight devices once, runs all 43 layers through the shared API handle, and
shuts down once. The single-layer path preserves local lifecycle for focused
diagnostics. With shared dense cache, descriptor checks disabled, predecode
probes disabled, and decode-only all-layer mode, the V100 gate passes `43/43`
layers at `32` slots / `256K`. It reports `shared_api=1`,
`43.957040 ms/token` summed decode proxy, `727.983506` projected slot-step
tok/s, and `35565.756621 ms` wall time. Next hoist route buffers,
streams/events, expert bindings, and TP runtime/KV state.

### Sprint 256 - TP/EP Shared Rank Buffers [complete]

Goal: Hoist fixed rank buffers and stream/event lifecycle across the all-layer
TP/EP scaffold.

Rationale: Route offsets, route-to-slot maps, input/gated/down buffers,
streams, events, and compose buffers are invariant for a fixed `slots/top_k`
run. Serving should not allocate and destroy them once per layer.

Outcome: Complete. `--all-layers` now initializes shared rank buffers once and
reuses them across all 43 layers. Per-layer packed expert bindings remain
layer-specific and are still loaded/freed per layer. With shared dense cache,
shared TurboMind API, descriptor checks disabled, predecode probes disabled,
and decode-only all-layer mode, the V100 gate passes `43/43` layers at `32`
slots / `256K`. It reports `shared_rank_buffers=1`, `43.895297 ms/token`
summed decode proxy, `729.007483` projected slot-step tok/s, and
`33978.379725 ms` wall time. Next hoist TP runtime/KV state or expert
descriptor bindings.

### Sprint 257 - TP/EP Shared TP Runtime [complete]

Goal: Hoist the TP runtime/KV allocator across the all-layer TP/EP scaffold.

Rationale: The 256K KV/compression/scratch arenas are serving state. Reopening
them once per layer is setup churn and obscures the cost of the resident
decode loop.

Outcome: Complete. `--all-layers` now opens the TP runtime once, allocates
sharded KV/compression/scratch arenas once, runs `dense_kv_slice()` per layer,
and closes the runtime once. With shared dense cache, shared TurboMind API,
shared rank buffers, descriptor checks disabled, predecode probes disabled,
and decode-only all-layer mode, the V100 gate passes `43/43` layers at `32`
slots / `256K`. It reports `shared_tp_runtime=1`, `46.024692 ms/token` summed
decode proxy, `695.278962` projected slot-step tok/s, and `28437.257957 ms`
wall time. The checksum matches prior gates, but decode timing regressed versus
Sprint 256; repeat before treating this as a performance promotion.

### Sprint 258 - TP/EP Shared Runtime Repeat Gate [complete]

Goal: Repeat the shared TP runtime path with a longer decode loop.

Rationale: Sprint 257 reduced wall time but regressed the decode proxy. A
longer gate is needed before deciding whether that regression is just short-run
noise.

Outcome: Complete. The 50-step all-layer gate passes `43/43` layers at `32`
slots / `256K` with `shared_tp_runtime=1` and checksum `204721433`. It reports
`45.672166 ms/token` summed decode proxy and `700.645557` projected slot-step
tok/s. This confirms the shared-runtime decode regression is persistent enough
to respect. Keep the shared runtime as correct residency work, but use Sprint
256 as the current decode-speed base unless the EP timing interaction is fixed.

### Sprint 259 - TP Runtime A/B Gate [complete]

Goal: Add a same-binary TP runtime sharing toggle and choose the current
decode-speed base.

Rationale: Shared TP runtime reduces setup wall time but appears to disturb
the decode proxy. A same-binary A/B avoids comparing across commits or cluster
conditions.

Outcome: Complete. The tool now supports `--share-tp-runtime` and
`--local-tp-runtime`, with local TP runtime as the default. The V100 50-step
A/B passes `43/43` layers and checksum `204721433` in both modes. Local
per-layer TP runtime reports `42.723359 ms/token` summed decode and
`749.004771` projected slot-step tok/s. Shared TP runtime reports
`46.972659 ms/token` and `681.247356` projected slot-step tok/s. Keep shared
runtime as an opt-in diagnostic; do not use it as the performance base until
the EP/dense timing interaction is fixed.

### Sprint 260 - TP/EP Resident Expert Bindings [complete]

Goal: Hoist active TurboMind expert bindings into an all-layer resident cache.

Rationale: A production appliance cannot reload expert weights for every layer.
Expert weights must be device resident, with only layer selection and execution
changing during decode.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--shared-expert-bindings` and `--local-expert-bindings`; shared is the
default. The resident cache loads active gated and down MXFP4 expert bindings
for all 43 layers and all 8 GPUs, reporting `27594326016` aggregate bytes and
`3449290752` bytes/GPU. The V100 50-step A/B at `32` slots / `256K` passes
`43/43` layers and checksum `204721433`. Shared bindings reduce wall time from
`35770.339339 ms` to `14338.419135 ms`; decode proxy is `44.131138 ms/token`
and `725.111599` projected slot-step tok/s.

### Sprint 261 - TP/EP EP-Dense Overlap [complete]

Goal: Overlap routed EP work with dense tensor-core GEMMs inside the TP/EP
decode loop.

Rationale: EP and dense projections are independent until next-hidden compose.
Running them serially leaves available GPU work overlap on the table.

Outcome: Complete. Each rank now has a separate dense stream. Dense cuBLAS
GEMMs run on that stream, while routed EP stays on the existing rank stream.
The tool supports `--overlap-ep-dense` and `--serial-ep-dense`; overlap is now
the default. The V100 50-step A/B at `32` slots / `256K`, resident expert
bindings, and local TP runtime passes `43/43` layers with checksum
`204721433`. Projected scaffold throughput improves from `631.273270` to
`846.062424` slot-step tok/s. The next target is compose/all-to-all.

### Sprint 262 - TP/EP FP16 EP Return Recheck [complete]

Goal: Recheck FP16 EP return in the new resident, overlapped execution regime.

Rationale: Compose/all-to-all is now dominant, so reducing EP return payload
could have become valuable even though it was previously rejected.

Outcome: Complete. The V100 50-step A/B at `32` slots / `256K`, resident
expert bindings, local TP runtime, and EP+dense overlap passes `43/43` layers
with checksum `204721433` in both modes. FP32 return reports
`831.795688` projected slot-step tok/s; FP16 return reports `729.339500`.
FP16 return remains rejected because the cast/expand path increases compose
time from `25.608539 ms` to `31.200853 ms`.

### Sprint 263 - TP/EP Direct Remote Compose Probe [complete]

Goal: Test whether compose can skip staged peer copies and read EP
contributions directly from source GPUs over peer memory.

Rationale: The staged compose path performs explicit peer copies into
destination-local buffers, then launches the compose kernel. Direct remote
reads could remove that staging boundary if NVLink remote reads are fast enough.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--direct-remote-compose` as an opt-in diagnostic. The V100 50-step A/B at
`32` slots / `256K`, resident expert bindings, local TP runtime, EP+dense
overlap, and FP32 EP return passes `43/43` layers with checksum `204721433` in
both modes. Staged compose reports `840.751688` projected slot-step tok/s;
direct remote compose reports `634.454351`. Direct remote compose is rejected
because remote reads increase compose time from `25.368965 ms` to
`37.776787 ms`.

### Sprint 264 - TP/EP Source-Scheduled Staged Copies [complete]

Goal: Improve the staged compose/all-to-all schedule without changing math.

Rationale: Direct remote reads lost to staged peer copies, but the staged path
still has scheduling freedom. Destination-scheduled copies may underuse source
copy engines.

Outcome: Complete. Each rank now owns a `copy_stream`. The tool supports
`--source-copy-schedule` and `--dest-copy-schedule`; source scheduling is now
the default. The V100 50-step A/B at `32` slots / `256K`, resident expert
bindings, local TP runtime, EP+dense overlap, FP32 EP return, and staged
compose passes `43/43` layers with checksum `204721433`. Projected scaffold
throughput improves from `840.494594` to `999.490407` slot-step tok/s, and
compose time drops from `25.452322 ms` to `19.513090 ms`.

### Sprint 265 - TP/EP Token-Major Scaffold [complete]

Goal: Add a serving-order TP/EP scaffold that executes layers in token-major
order.

Rationale: Layer-major repeated loops are useful for kernel timing, but serving
decodes as `for token -> for layer`. We need a gate that exposes that schedule
before claiming practical serving.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--token-major-all-layers`. The V100 gate runs `4` token steps x `43` layers
at `32` slots / `256K`, using resident expert bindings, EP+dense overlap, and
source-scheduled staged copies. It passes `172/172` layer invocations and
reports `48.840011 ms/token` proxy / `655.200508` projected slot-step tok/s.
This is a serving-order scaffold, not generated-token serving throughput.

### Sprint 266 - TP/EP Shared Dense Ops Probe [complete]

Goal: Test whether token-major setup cost can be reduced by hoisting dense
operation objects across all layers.

Rationale: The token-major scaffold still constructs dense cuBLAS handles,
input buffers, and output buffers per layer invocation. If that setup is a
material part of the token-major gap, a shared dense-op cache should improve
the serving-order scaffold.

Outcome: Complete and rejected as a default. `tools/ds4-v100-tp-ep-full-layer-smoke`
now supports `--shared-dense-ops` as an opt-in diagnostic. On the V100 pod at
`32` slots / `256K`, `4` token steps, resident expert bindings, EP+dense
overlap, and source-scheduled staged copies, both local and shared dense-op
modes pass `172/172` layer invocations with checksum `296236348`. Local dense
ops report `51.991980 ms/token` proxy and `615.479538` projected slot-step
tok/s. Shared dense ops report `56.085843 ms/token` proxy and `570.553966`
projected slot-step tok/s. Shared dense ops slightly reduce wall time but
regress decode timing by `7.3%`, so the default remains local dense ops.

### Sprint 267 - TP/EP Token-Major Shared TP Runtime [complete]

Goal: Recheck shared TP runtime in token-major serving order and promote it
only if the serving-order proxy improves.

Rationale: Shared TP runtime was previously rejected in layer-major mode, but
token-major execution reuses KV/runtime state across token steps. That changes
the cost model enough to warrant a same-binary A/B before moving to generated
serving integration.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now defaults
token-major all-layer runs to shared TP runtime unless `--local-tp-runtime` is
explicitly requested. Layer-major defaults are unchanged. On the V100 pod at
`32` slots / `256K`, `4` token steps, resident expert bindings, EP+dense
overlap, source-scheduled staged copies, and local dense ops, shared TP runtime
improves the token-major proxy from `51.289549` to `47.902324 ms/token` and
projected slot-step throughput from `623.908781` to `668.026047 tok/s`.
Wall time drops from `34880.753622` to `11661.323548 ms`, with checksum
`296236348` preserved. A default one-step check confirms token-major runs now
select `shared_tp_runtime=1`.

### Sprint 268 - TP/EP Token-Major Position Advance [complete]

Goal: Make the token-major scaffold advance context position across token
steps.

Rationale: The first token-major scaffold validated execution order, but every
token step reused the same logical position. Serving decode advances position
each token while keeping the sequence slot fixed, so the scaffold should do
the same before longer continuous gates or generated-token integration.

Outcome: Complete. In `--token-major-all-layers` mode, each layer invocation
now uses `position = start_position + token_step`, and token-major item logs
include the effective position. On the V100 pod at `32` slots / `256K`, `4`
token steps, positions `1024-1027`, shared TP runtime, resident expert
bindings, EP+dense overlap, and source-scheduled staged copies, the scaffold
passes `172/172` layer invocations. It reports `45.770462 ms/token` proxy,
`699.140856` projected slot-step tok/s, `93.872406 ms` summed EP,
`89.157724 ms` summed compose, `11799.119372 ms` wall, and checksum
`296236348`.

### Sprint 269 - TP/EP Continuous Token-Major Gate [complete]

Goal: Run longer token-major gates to reduce early-token noise and expose the
steady scaffold bottleneck.

Rationale: Four token steps are useful for iteration but still include startup
effects. Before bridging to generated serving, the scaffold needs a longer
continuous run at the target `32` slots / `256K` shape.

Outcome: Complete. On the V100 pod, the 16-step and 32-step token-major gates
both pass. The 32-step run covers `1376` layer invocations with shared TP
runtime, resident expert bindings, EP+dense overlap, source-scheduled staged
copies, local dense ops, and advancing positions from `4096`. It reports
`39.290219 ms/token` proxy, `814.452062` projected slot-step tok/s,
`514.766496 ms` summed EP, `742.079181 ms` summed compose, `91515.672970 ms`
wall, and checksum `8297177632`. The bottleneck is now clearly the
compose/all-to-all boundary plus remaining orchestration, not the routed EP
kernel in isolation.

### Sprint 270 - TP/EP Skip Self Compose Copy [complete]

Goal: Remove same-GPU staged compose copies from the FP32 EP-return path.

Rationale: Sprint 269 showed compose/all-to-all dominates the continuous
token-major scaffold. The staged path still copied `src == dst` shards even
though each destination GPU can read its local EP contribution directly.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--skip-self-compose-copy` and `--copy-self-compose`; skip-self is the default.
On the FP32 return path, same-GPU copy traffic is skipped and compose reads the
local `d_ep_contrib_all` slice for that source. The V100 16-step A/B at `32`
slots / `256K` passes with checksum `8244145680` in both modes and improves
from `40.271428` to `38.503412 ms/token` proxy. Compose time drops from
`371.558564` to `342.417467 ms`. The 32-step skip-self run passes
`1376/1376` invocations at `37.912062 ms/token` proxy, `844.058544` projected
slot-step tok/s, `522.914003 ms` EP, `689.877521 ms` compose, and checksum
`8297177632`.

### Sprint 271 - TP/EP Compose Stage Breakdown [complete]

Goal: Split token-major compose timing into actionable buckets.

Outcome: Complete. The tool now reports compose reduce, copy, and final
compose timing. At `32` slots / `256K`, `16` token steps, the passing run
reports `327.657087 ms` compose total: `49.805028 ms` reduce,
`242.803068 ms` copy, and `35.048991 ms` final compose. Copy/all-to-all is
the dominant part of compose.

### Sprint 272 - TP/EP Multi Copy Streams Probe [complete]

Goal: Test whether source-scheduled peer copies benefit from multiple copy
streams per source rank.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--multi-copy-streams`. The 16-step A/B at `32` slots / `256K` improves from
`39.288036` to `37.395624 ms/token` proxy and reduces copy time from
`248.331836` to `219.221398 ms`. The 32-step opt-in run passes `1376/1376`
invocations at `36.911097 ms/token` proxy and `866.947964` projected
slot-step tok/s. Per steering, the next sprint pivots to end-to-end TP/EP
serving rather than continuing compose micro-optimization.

### Sprint 273 - TP/EP Serving Metric Bridge [complete]

Goal: Expose generated-token and continuation-token metrics from the resident
token-major TP/EP path.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke` now supports
`--serving-bench`, emitting generated/continuation token counts and tok/s
rates. At `32` slots / `256K`, `16` generated tokens/request, shared TP
runtime, resident expert bindings, source-scheduled multi-copy compose, and
MTP off, the V100 run passes with checksum `8244145680`. Decode-only metrics
are `875.486234` aggregate generated tok/s and `931.549518` aggregate
continuation tok/s. Wall metrics are only `10.612319` generated tok/s and
`10.616412` continuation tok/s because the token-major scaffold still invokes
the heavy per-layer `run_layer()` path for every token/layer. Next build a
resident serving loop that calls the decode body directly without per-layer
scaffold setup.

### Sprint 274 - TP/EP Resident Serving Loop [complete]

Goal: Remove the per-token/per-layer `run_layer()` scaffold from TP/EP
serving-bench mode.

Outcome: Complete. `--serving-bench` now uses a direct resident decode loop
when shared TP runtime, resident expert bindings, shared rank buffers, and the
shared dense cache are available. It parses layer contracts once, binds
resident expert/dense state, skips serving-mode checksum readback, and calls
the decode body directly. At `32` slots / `256K`, shared dense ops are required
for wall throughput. The best V100 run so far uses `32` generated
tokens/request and reports `669.222644` wall generated tok/s,
`690.469286` wall continuation tok/s, `876.524260` decode generated tok/s,
and `910.270244` decode continuation tok/s. Next wrap this backend in the
HTTP sustained-decode harness.

### Sprint 275 - TP/EP Sustained Serving Artifact Wrapper [complete]

Goal: Produce repeatable sustained-serving artifacts from the resident TP/EP
backend before wiring the backend into the HTTP appliance server.

Outcome: Complete. `tools/ds4-v100-tp-ep-sustained-bench.sh` runs the
resident TP/EP serving bench with the promoted `32` slot / `256K` settings,
records stdout/stderr, and writes `sustained_decode.tsv`,
`sustained_decode.json`, and per-case `result.json` artifacts. The V100 pod
run at `32` slots / `256K` / `32` generated tokens per request passes with
`32/32` token match. The current artifact topline is `749.304439` wall
generated tok/s, `774.209856` wall continuation tok/s, `963.264018`
decode-only generated tok/s, and `1000.823072` decode-only continuation tok/s.
This confirms the resident backend can be measured repeatably, but it still
needs the operational HTTP harness.

### Sprint 276 - TP/EP Resident HTTP Harness [complete]

Goal: Expose the resident TP/EP backend through an in-process HTTP harness.

Outcome: Complete as a smoke-tested server path. The TP/EP full-layer tool now
has `--serve-http`, keeps the resident backend loaded across requests, and
serves `GET /health`, `GET /v100/status`, `GET /metrics`, and
`POST /v100/selected-token`. The V100 HTTP smoke used four requests against
one resident server and the generation POST returned `32/32` token match,
`719.275018` wall generated tok/s, `751.645517` wall continuation tok/s,
`926.497242` decode-only generated tok/s, and `974.020201` decode-only
continuation tok/s. Requests are currently serialized and the harness is not
yet wired into the deployment launcher.

### Sprint 277 - TP/EP Appliance Launcher Path [complete]

Goal: Start the TP/EP resident HTTP server through the appliance launcher.

Outcome: Complete. `tools/ds4-v100-run-appliance.sh` now supports
`DS4_V100_SERVE_MODE=tp-ep`, resolves the promoted TP/EP server command, and
fails closed outside the current target shape. The V100 launcher smoke used
the launcher to start the resident TP/EP server, then exercised `/health`,
`/v100/status`, `POST /v100/selected-token`, and `/metrics`. The POST returned
`32/32` token match, `728.744669` wall generated tok/s, `753.022651` wall
continuation tok/s, `939.787471` decode-only generated tok/s, and
`976.290858` decode-only continuation tok/s.

### Sprint 278 - TP/EP Sustained HTTP Matrix [complete]

Goal: Add repeatable sustained HTTP metrology for the TP/EP launcher path.

Outcome: Complete. `tools/ds4-v100-tp-ep-http-bench.sh` starts
`DS4_V100_SERVE_MODE=tp-ep`, drives the HTTP surface using Python stdlib, and
writes matrix artifacts. The V100 run at `32` slots / `256K` reports
`737.091414` wall generated tok/s and `766.964251` wall continuation tok/s for
`32` tokens/request, and `739.774102` wall generated tok/s and `755.504630`
wall continuation tok/s for `64` tokens/request. Both cases return `32/32`
token match.

### Sprint 279 - TP/EP Deployment Defaults And GPU Utilization [complete]

Goal: Point the Kubernetes appliance example at the TP/EP serving path and
capture GPU utilization during the sustained HTTP matrix.

Outcome: Complete. The deployment example now uses `DS4_V100_SERVE_MODE=tp-ep`,
the current TP/EP production pack and contract, `32` slots / `256K` context,
the localpool workspace, and the `llm-models-local` PVC. The launcher keeps
loopback as the default bind and requires `DS4_V100_ALLOW_NONLOCAL_HOST=1` for
Kubernetes service binds. The sustained HTTP bench now samples `nvidia-smi`
during the generation POST and writes per-case GPU-util artifacts. The V100
run reports `745.699174` wall generated tok/s and `771.902910` wall
continuation tok/s for `32` tokens/request, and `753.708353` wall generated
tok/s and `766.803086` wall continuation tok/s for `64` tokens/request, with
`32/32` token match. GPU utilization peaks at `38-40%` and averages
`15-19%` across the sampled POST windows.

### Sprint 280 - TP/EP Multi-Request HTTP Metrology [complete]

Goal: Measure resident sustained serving across multiple generation requests
without restarting the TP/EP server.

Outcome: Complete. The TP/EP HTTP server now exposes cumulative prompt,
generated, continuation, timing, throughput, and logical-position counters via
`/v100/status` and `/metrics`. The sustained HTTP bench now supports
`--requests N`, writes per-request responses, and aggregates throughput across
the resident request sequence. The V100 run at `32` slots / `256K` with three
generation requests per case reports `751.114404` wall generated tok/s and
`760.078310` wall continuation tok/s for `32` tokens/request, and
`762.277426` wall generated tok/s and `766.925593` wall continuation tok/s for
`64` tokens/request. Both cases return aggregate `96/96` token match.

### Sprint 281 - TP/EP HTTP Stage Metrics [complete]

Goal: Expose EP/dense/compose stage timing in the operational HTTP artifacts.

Outcome: Complete. `/v100/selected-token` responses now include EP, dense,
compose, compose-reduce, compose-copy, and compose-final timings under
`timing_ms`. `/v100/status` and `/metrics` expose last and cumulative stage
counters. The sustained HTTP bench schema is now
`ds4_v100_tp_ep_sustained_http.v3` and aggregates stage timings across
resident generation requests. The V100 run at `32` slots / `256K` with three
generation requests per case reports `742.897231` wall generated tok/s for
`32` tokens/request and `739.612937` for `64` tokens/request. In the 64-token
case, compose-copy accounts for `2569.208878 ms` of `3626.650073 ms` compose
time.

### Sprint 282 - TP/EP Event-Wait Compose Copy [complete]

Goal: Reduce TP/EP compose-copy host synchronization by making destination
compose streams wait on peer-copy events.

Outcome: Complete. `--copy-event-compose` records per-source/per-destination
copy completion events and has destination streams wait on those events before
final compose, avoiding a global host-side copy-stream synchronization barrier.
The appliance launcher and Kubernetes defaults now enable
`DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1`. Same-binary 64-token HTTP A/B at
`32` slots / `256K` / three generation requests improves wall generated tok/s
from `752.669235` to `771.276064` and wall continuation tok/s from
`757.403683` to `775.670776`, with aggregate `96/96` token match.

### Sprint 283 - TP/EP FP16 Return Recheck [complete]

Goal: Recheck whether FP16 EP return becomes useful after event-wait compose.

Outcome: Complete. The launcher and HTTP bench now expose
`DS4_V100_TP_EP_RETURN_FP16` / `--ep-return-fp16` as a diagnostic toggle, with
the appliance default still off. Same-binary 64-token HTTP A/B at `32` slots /
`256K` / three generation requests shows FP16 return regresses wall generated
tok/s from `766.883263` to `635.936079` and decode generated tok/s from
`997.165341` to `793.283316`, while preserving aggregate `96/96` token match.
The extra cast/add/final-compose work dominates the reduced copy payload on
V100, so FP16 return remains rejected.

### Sprint 284 - TP/EP Compact Route Compose [complete]

Goal: Reduce staged FP32 contribution traffic without changing return dtype.

Outcome: Complete. `--compact-route-compose` packs EP contributions in
route-major form, copies only `routes * hidden_shard` elements per
source/destination, and composes back to slot-major hidden rows on the
destination GPU. The launcher, bench, and Kubernetes defaults now enable
`DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1`. Same-binary 64-token HTTP A/B at
`32` slots / `256K` / three generation requests improves wall generated tok/s
from `711.177884` to `791.453850` and wall continuation tok/s from
`719.489689` to `796.894336`, with aggregate `96/96` token match.

### Sprint 285 - TP/EP Promoted Serving Topline [complete]

Goal: Re-establish the normal promoted TP/EP HTTP serving topline after
Sprint 282 and Sprint 284 defaults.

Outcome: Complete. The normal launcher-backed HTTP bench now runs with
`DS4_V100_TP_EP_COPY_EVENT_COMPOSE=1`,
`DS4_V100_TP_EP_COMPACT_ROUTE_COMPOSE=1`, and
`DS4_V100_TP_EP_RETURN_FP16=0`. At `32` slots / `256K` / three resident
generation requests, the V100 pod reports `771.036527` wall generated tok/s
and `781.922821` wall continuation tok/s for `32` tokens/request, and
`794.694599` wall generated tok/s and `799.391755` wall continuation tok/s for
`64` tokens/request. Both cases return aggregate `96/96` token match.

### Sprint 286 - TP/EP HTTP Request Coalescing [complete]

Goal: Make the TP/EP HTTP serving path admit concurrent selected-token
requests into one resident decode batch instead of treating every HTTP request
as an independent synthetic 32-slot run.

Outcome: Complete. The TP/EP HTTP server now accepts pending generation
requests during a bounded `--microbatch-wait-us` window, runs one resident
decode with `slots = coalesced_batch_size`, and returns per-client responses
with `coalesced_batch_id`, `coalesced_batch_size`, per-client token counts, and
batch token counts. `/v100/status` and `/metrics` expose generation batch and
coalesced request counters. The launcher passes the resolved
`DS4_V100_MICROBATCH_WAIT_US` value into the TP/EP server.

The V100 pod matrix at `32` slots / `256K` / `32` concurrent HTTP requests
formed one `coalesced_batch_size=32` batch in both token cases:
`721.446441` wall generated tok/s and `950.363316` decode generated tok/s for
32 tokens/request, and `787.316214` wall generated tok/s and `1030.972573`
decode generated tok/s for 64 tokens/request. Both cases return aggregate
`32/32` token match.

This is now the practical-serving semantic baseline for the selected-token
harness. The next gap is a real prompt/token API and bucketed admission queues
on top of this coalescing path.

### Sprint 287 - TP/EP Bucketed Admission [complete]

Goal: Make the TP/EP HTTP serving path handle mixed concurrent generation
lengths by queueing requests into token-count buckets instead of rejecting
mismatches during coalescing.

Outcome: Complete. The TP/EP HTTP server now keeps a pending generation queue,
drains same-length queued requests before accepting new sockets for a batch,
and continues serving while pending generation requests exist. Mixed
`max_tokens` requests are no longer rejected with `409`; they are held for a
later same-length decode batch. `/v100/status` and `/metrics` expose
`bucketed_requests` and `pending_generation_requests`.

The V100 pod mixed run at `32` slots / `256K` with 32 concurrent requests using
pattern `32,64` forms two batches of 16 clients each, reports
`bucketed_requests=16`, returns aggregate `32/32` token match, and has zero
rejected requests. Admitted-client throughput is `387.877251` wall generated
tok/s and `510.747848` decode generated tok/s over 1536 generated client
tokens. A uniform 32-request sanity run still forms one full batch and reports
`759.490446` wall generated tok/s / `991.405750` decode generated tok/s.

Partial buckets intentionally run the configured 32-slot decode shape and count
only admitted client tokens in serving metrics. This keeps compact
route-compose on the validated kernel shape until a future sprint adds true
dynamic-slot compact compose or per-slot refill.

### Sprint 288 - TP/EP Diagnostic Completions Endpoint [complete]

Goal: Add a serving-shaped, OpenAI-compatible diagnostic completions endpoint
to the TP/EP HTTP harness while preserving the coalesced and bucketed admission
policy from Sprints 286-287.

Outcome: Complete. The TP/EP server now accepts `POST /v1/completions` and
`POST /v100/diagnostic-completions` in the same generation path as
`POST /v100/selected-token`. Completion responses use an OpenAI-style
`text_completion` envelope with `choices` and `usage`, while TP/EP admission,
timing, checksum, and token-match metadata are nested under `ds4_v100`.

This endpoint is deliberately diagnostic. It marks `ds4_v100.diagnostic=true`
and records that prompt prefill and output-head text/token selection are not
yet wired in this TP/EP surface.

The V100 pod mixed completion run at `32` slots / `256K` with 32 concurrent
requests using pattern `32,64` forms two 16-client buckets, returns aggregate
`32/32` token match, and reports `384.581100` wall generated tok/s /
`505.797315` decode generated tok/s over 1536 admitted client tokens. The
selected-token regression sanity still forms one full 32-client batch and
reports `726.823991` wall generated tok/s / `944.195924` decode generated
tok/s.

Next work should move from diagnostic completions to real model output in the
TP/EP path: output-head/top-token selection, tokenizer text emission, prompt
prefill, and then stop/finish handling.

### Sprint 289 - TP/EP Vocab-Sharded Output Head Gate [complete]

Goal: Add a TP/EP-only output-head primitive that exercises the real DS4
output-head tensor layout across all 8 V100s.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke.cu` now has
`--output-head-gate`. The gate loads real replicated `hc_head_fn`,
`hc_head_base`, `hc_head_scale`, and `output_norm.weight` controls, plus real
BF16 `output.weight` vocab shards. It runs synthetic HC through the DS4
output-head collapse, projects across vocab shards on all 8 GPUs, and reduces
the shard-local logits to a global top-1 token.

At `32` slots / vocab `129280`, the scalar BF16 projection passes with token
`26803`, cold projection time `2192.810195 ms`, worst per-GPU projection-kernel
time `7.593408 ms`, host top-1 reduction `6.070330 ms`, and finite logits.
The BF16-to-FP16 cuBLAS diagnostic path also passes and selects the same token,
but is slower in this cold gate: `2217.599099 ms` projection time and
`22.116352 ms` worst per-GPU kernel time. That cuBLAS result includes cold
upload, BF16-to-FP16 expansion, handle creation, and serial per-GPU
orchestration; it is not yet a serving-path rejection.

The remaining serving gap is now sharper: the TP/EP token-major loop must
carry or reconstruct final HC `[slots,4,4096]` at the end of layer 42 and feed
that into this output-head primitive. Only after that should `/v1/completions`
emit real selected tokens/text.

### Sprint 290 - TP/EP Resident Output Head Gate [complete]

Goal: Convert the cold TP/EP output-head diagnostic into a resident repeated
gate and remove full-logit host readback from the reduction path.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke.cu` now has
`--output-head-resident-gate`. It preloads the real BF16 `output.weight` vocab
shards across all 8 V100s, keeps output-head scratch resident, repeats the
synthetic-HC output path, and reports separate timing for HC prep, embedding
broadcast, vocab projection, worst per-GPU projection kernel, and token
selection.

The sprint also added GPU-side per-shard top-1 reduction. That changes the
host transfer from full logits to only `8 * slots` token/logit candidates. At
32 slots, full-logit readback measured `15.980438 ms` total and
`2002.448256` output-head tok/s. With device-side shard top-1, the same gate
measures `8.528343 ms` total, `7.474198 ms` projection wall time,
`7.427597 ms` worst per-GPU projection-kernel time, `0.211761 ms`
top-1/readback time, and `3752.194257` output-head tok/s. The 16-slot and
64-slot gates also pass at `3563.755123` and `3877.433386` output-head tok/s.

Decision: reject full-logit host readback for serving. Promote resident
vocab-sharded output projection plus GPU-side shard top-1 as the first TP/EP
output-head serving shape. The projection kernel is still scalar BF16 and
should be optimized later, after real final HC is wired into the serving loop.

Remaining gap: the TP/EP token-major loop still carries per-rank hidden shards,
not final DS4 HC `[slots,4,4096]`. The next sprint should add the HC carry
contract and call the resident output-head primitive from `/v1/completions`.

### Sprint 291 - TP/EP Final-HC Carry Scaffold [complete]

Goal: Add a TP/EP-only final-HC carry scaffold so the token-major loop has an
explicit output-head input shape.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke.cu` now has
`--final-hc-carry-gate`. When enabled, each GPU owns a resident
`[slots][4][512]` F32 shard, which collectively represents the logical
`[slots][4][4096]` HC tensor consumed by DS4 output selection. The current
kernel expands the per-rank hidden shard into a proxy HC shard; this proves
layout, finite dataflow, and timing, but it is not yet true DS4 HC row
semantics.

The 1-token all-layer V100 gate passes with `43/43` invocations,
`75.554825 ms` summed decode, `2.100054 ms` summed final-HC carry cost, and
`423.533507` decode tok/s. The matching control run without the carry gate
passes with `70.923652 ms` summed decode and `451.189400` decode tok/s. A
4-token continuation run with the carry gate passes `172/172` invocations,
reports `8.113938 ms` summed final-HC carry cost, `712.985252` aggregate
decode tok/s, and `960.823272` continuation decode tok/s.

Decision: keep the sharded HC carry shape. The overhead is small enough for the
first output-head integration path. The next work must replace the proxy HC
expansion with true DS4 HC row semantics or wire the proxy into the output head
only under an explicitly diagnostic endpoint.

### Sprint 292 - TP/EP Diagnostic Output-Head Serving Bridge [complete]

Goal: Wire the TP/EP sharded HC carry into the resident vocab-sharded output
head and surface diagnostic selected token IDs through the HTTP completions
path.

Outcome: Complete. `tools/ds4-v100-tp-ep-full-layer-smoke.cu` now has
`--diagnostic-output-head`, which implies HC carry. The new resident
`SharedOutputHead` loads real output controls and BF16 `output.weight` vocab
shards once, gathers per-rank `[slots][4][512]` HC shards into a logical
`[slots][4][4096]` tensor on GPU0, runs the DS4 output-head collapse and
vocab-sharded BF16 projection, performs GPU-side shard top-1, and returns
diagnostic token IDs/logits through the serving result.

The launcher supports `DS4_V100_TP_EP_DIAGNOSTIC_OUTPUT_HEAD=1`, and the HTTP
bench supports `--diagnostic-output-head`. `/v1/completions` responses now
include `diagnostic_output_head`, `diagnostic_output_head_proxy_hc`,
`selected_token`, `selected_logit`, and output-head timing fields under
`ds4_v100` when the flag is enabled.

Direct 32-slot V100 validation reports output-head `total_ms=8.903469`,
`projection_ms=7.690283`, `top1_ms=0.497101`, first token `122445`, finite
logits, and PASS. A launcher-level 32-concurrent completions run forms one
coalesced 32-request batch, returns `32/32` HTTP 200 responses with selected
token metadata, and reports output-head `total_ms=8.586224`,
`projection_ms=7.592902`, `top1_ms=0.341194`, `158.576748` wall generated
tok/s, and `294.331849` decode generated tok/s for the 1-token diagnostic
case.

Decision: this is the correct operational bridge, but it remains diagnostic.
The selected token IDs come from proxy HC rows, so they prove wiring and timing
rather than model-correct text generation.

## Experiment Backlog

These experiments should be run inside the TP/EP sprints, not as PP variants:

- TP8 collective roofline at `M=32/64/128`, hidden `4096`.
- TP8 dense GEMM fixture using FP16/FP8-style low-bit expansion on GPU.
- TP sharded KV allocation/update/read at `32` slots / `256K`, then `512K`
  if memory allows.
- EP routed expert smoke with real TurboMind/CUTLASS low-bit kernels at
  `32` active slots.
- Expert load-balance measurement: active experts, routes per expert, and
  worst-GPU imbalance.
- One-layer TP/EP correctness against frozen PP baseline.
- Full 43-layer TP/EP decode correctness.
- TP/EP serving throughput with generated and continuation tok/s separated.

## Parking Lot

- PP/layer-split scheduling optimizations: archived. Use only as baseline.
- Routed-only TP overlays inside the PP scheduler: rejected.
- Generic PP/TP scheduler abstraction: rejected.
- Single-slot throughput reports: rejected as practical-serving evidence.
- MTP serving: deferred until TP/EP serving is operational.
- PP-oriented MTP block-2 promotion: paused; useful correctness evidence only.
- HTTP-wrapper profiler windows: useful wiring, but not sufficient for NCU
  metrics in the current container/toolchain. Use a direct non-server TP/EP
  profile target for scoped kernel evidence.

## Pivot Log

| Date | Change | Rationale | Next |
|---|---|---|---|
| 2026-05-23 | Archived the prior PP-era vision to `docs/sprints/archive/VISION-2026-05-23-pre-tp-hard-cut.md`. | The accumulated roadmap still documents history, but it no longer reflects the strategy. | Use this file as the active alignment document. |
| 2026-05-23 | Sprint 230 proved TP sharded KV row ownership at `32` slots / `256K`. | TP/EP needs resident hidden/KV state before EP expert work is meaningful. | Build the bounded EP routed-expert slice in separate TP/EP files. |
| 2026-05-23 | Sprint 231 proved bounded EP8 routed expert execution with real TurboMind MXFP4 kernels. | The EP low-bit kernel path is live outside the PP scheduler, but rank skew is visible. | Build the one-layer TP/EP correctness gate and preserve per-rank timing. |
| 2026-05-23 | Sprint 232 proved the combined TP runtime plus EP expert fixture in one process. | The TP/EP lifecycle works at the target shape, but it is still fixture data. | Move to descriptor-driven one-real-layer TP/EP correctness. |
| 2026-05-23 | Sprint 233 proved descriptor ownership for layer `2` from the real production-pack contract. | The contract has the rows and TP/EP ownership needed, but execution still uses fixture weights. | Bind descriptor rows to actual pack bytes and feed real expert pointers into the one-layer smoke. |
| 2026-05-23 | Sprint 234 proved descriptor-backed routed expert byte binding for layer `2`. | Real packed expert bytes now flow into the separate TP/EP path; the remaining gap is full-layer math and all-layer decode. | Build descriptor-backed full-layer TP/EP decode with MTP off. |
| 2026-05-23 | Sprint 235 proved a descriptor-backed full-layer scaffold for layer `2`. | All descriptor families now have a concrete TP/EP binding outside the PP path, but dense/control rows are checksum scaffolds, not math. | Replace dense/control checksum stages with real low-bit dense execution for representative full-layer decode. |
| 2026-05-23 | Sprint 236 proved real packed-F8 dense compute for `blk.2.attn_q_a.weight` in the TP/EP path. | The runtime can now compute from packed dense bytes, but only for one representative tensor and with a straightforward FP32 dot kernel. | Extend dense compute coverage or replace this gate with fused HMMA/CUTLASS dense blocks. |
| 2026-05-23 | Sprint 237 proved packed-F8 dense compute coverage for all compatible layer-2 F8 dense tensors. | F8 dense families execute from production bytes; BF16 compressor/indexer math and real layer dataflow remain. | Add BF16 compute coverage or compose dense outputs into representative full-layer decode. |
| 2026-05-23 | Sprint 238 proved BF16 compressor/indexer dense coverage and combined F8+BF16 coverage for layer `2`. | Layer-2 dense families now execute from production bytes in the separate TP/EP path. | Compose dense, KV, control/router, and EP expert outputs into representative full-layer decode. |
| 2026-05-23 | Sprint 239 proved representative TP/EP next-hidden shard composition for layer `2`. | Dense outputs, EP returned contributions, KV update/check, and residual composition now run in one separate TP/EP execution. | Move from smoke composition to a TP/EP serving gate at `32` slots / `256K`, MTP off. |
| 2026-05-23 | Sprint 240 proved a resident repeated TP/EP layer-loop benchmark at `32` slots / `256K`. | The path now reports stage costs without per-step pack reloads: dense and compose/sync dominate over EP. | Decide whether Sprint 241 optimizes dense/compose kernels first or starts server-loop integration with known bottlenecks. |
| 2026-05-23 | Sprint 241 proved FP16 EP return is correct but slower as a standalone pass. | Payload bytes are not the limiter; extra cast/expand kernels increase compose time. | Keep FP32 return default and target fused dense/compose kernel boundaries next. |
| 2026-05-23 | Sprint 242 proved fused FP32 remote-sum compose improves the resident layer loop. | Removing zero/add kernels is more valuable than standalone EP return quantization at this shape. | Continue collapsing TP/EP dense, EP return, and compose boundaries, then move to all-layer/server integration. |
| 2026-05-23 | Sprint 243 rejected the first naive TP/EP dense HMMA candidate. | HMMA is not enough by itself; per-tile F8 decode/staging made dense time worse than scalar. | Adapt the older shape-specific HMMA kernels or design a prepacked/software-pipelined dense path. |
| 2026-05-23 | Sprint 244 proved a resident FP16 tensor-core dense ceiling is materially faster. | Dense is removable if low-bit feeding is efficient, but expanded FP16 is not the final memory format. | Implement a packed low-bit dense production kernel that approaches the FP16/cuBLAS ceiling. |
| 2026-05-23 | Sprint 245 proved dense FP16 runtime cache fits the `32` slot / `256K` TP/EP budget when replacing dense source tensors in VRAM. | This gives us a working tensor-core dense fallback while preserving the quantized source pack offline. | Build the TP/EP dense-cache loader/runtime path for all dense tensors and benchmark resident all-layer decode. |
| 2026-05-23 | Sprint 246 materialized all dense TP rows into FP16 cache arenas on the V100 pod. | The dense-cache path is now an executable runtime primitive, not just an estimate. | Wire dense cache lookup into resident layer execution and benchmark all-layer decode. |
| 2026-05-23 | Sprint 247 wired dense cache lookup into the representative TP/EP decode loop. | Execution can now consume cache-resident FP16 dense weights instead of private per-op copies. | Build a descriptor-selected dense execution table across all layers. |
| 2026-05-23 | Sprint 248 built the descriptor-selected all-layer dense execution table. | Dense no longer depends on hardcoded layer-2 tensor selection. | Compose dense, EP, KV, and hidden-state flow in a resident all-layer TP/EP loop. |
| 2026-05-23 | Sprint 249 made the representative TP/EP full-layer smoke layer-parametric across SWA-only, ratio-4, ratio-128, and late layers. | The all-layer loop no longer has layer-2 tensor-name and ratio-4 KV assumptions as blockers. | Build a resident all-layer TP/EP loop that carries hidden shards through all 43 layers in one process. |
| 2026-05-23 | Sprint 250 added a single-process all-layer TP/EP scaffold gate. | The TP/EP path now has a 43-layer correctness/timing gate, but it still recreates per-layer state. | Move runtime/cache/TurboMind state outside the per-layer runner for a truly resident all-layer loop. |
| 2026-05-23 | Sprint 251 hoisted the dense FP16 cache across all layers. | Reusing dense cache cuts all-layer scaffold wall time by about 19% and removes one class of per-layer state churn. | Hoist TurboMind/API, route buffers, expert bindings, and TP runtime state. |
| 2026-05-23 | Sprint 252 added opt-in descriptor-check bypass for serving-shaped scaffold runs. | Descriptor checks are validation work; skipping them cuts all-layer wall time by about 37% after validation has passed. | Fix decode-only harness and hoist TurboMind/API plus rank buffers. |
| 2026-05-23 | Sprint 253 repaired the decode-only all-layer scaffold harness. | The standard TP/EP scaffold benchmark no longer requires an extra one-shot compose validation path. | Hoist TurboMind/API handles, route buffers, expert bindings, and stream/event lifecycle. |
| 2026-05-23 | Sprint 254 added opt-in pre-decode probe bypass for benchmark runs. | Extra isolated TurboMind probes are validation work, not serving work. | Hoist TurboMind/API handles, route buffers, expert bindings, and stream/event lifecycle. |
| 2026-05-23 | Sprint 255 hoisted TurboMind API lifecycle across the all-layer TP/EP loop. | Removing per-layer library/API setup cuts scaffold wall time while preserving decode checksums. | Hoist route buffers, streams/events, expert bindings, and TP runtime/KV state. |
| 2026-05-23 | Sprint 256 hoisted fixed rank buffers and stream/event lifecycle across the all-layer TP/EP loop. | Removing per-layer route/core buffer allocation cuts wall time and keeps checksum stable. | Hoist TP runtime/KV state or expert descriptor bindings. |
| 2026-05-23 | Sprint 257 hoisted TP runtime/KV allocation across the all-layer TP/EP loop. | Correctness holds and wall time drops, but decode proxy regresses and needs repeat timing. | Repeat/longer gate, then decide whether to keep shared TP runtime as the performance base before expert binding hoist. |
| 2026-05-23 | Sprint 258 repeated the shared TP runtime path with a 50-step all-layer gate. | The decode regression persisted while checksum stayed stable. | Investigate EP timing under shared runtime, or keep Sprint 256 as decode-speed base while hoisting expert bindings. |
| 2026-05-23 | Sprint 259 added a same-binary TP runtime A/B and made local TP runtime the default. | Shared TP runtime is correct but slower for decode in the same executable. | Hoist expert descriptor bindings or collapse EP/dense/compose boundaries while preserving the local-runtime performance base. |
| 2026-05-23 | Sprint 260 hoisted active TurboMind expert bindings into a resident all-layer cache. | This matches the production appliance requirement and removes per-layer expert reload churn. | Move toward a real serving loop or reduce the EP/dense/compose boundary now that major setup state is resident. |
| 2026-05-23 | Sprint 261 overlapped routed EP with dense cuBLAS work on separate streams. | EP and dense are independent until compose, and overlap produced a 34% scaffold throughput gain. | Optimize compose/all-to-all or convert the scaffold into a serving loop. |
| 2026-05-23 | Sprint 262 rechecked FP16 EP return under the resident overlapped schedule. | FP16 return still regresses total decode because compose gets slower. | Keep FP32 return and target fused/direct compose-all-to-all instead of standalone cast staging. |
| 2026-05-23 | Sprint 263 tested direct peer-memory compose. | Direct remote reads preserve correctness but regress compose time and total throughput. | Keep staged peer copies; optimize staged-copy scheduling or destination-side reduction. |
| 2026-05-23 | Sprint 264 changed staged peer-copy scheduling to source copy streams. | Source-scheduled copies materially reduce compose time and raise projected scaffold throughput. | Convert scaffold into serving loop or continue destination-side compose kernel optimization. |
| 2026-05-23 | Sprint 265 added a token-major serving-order scaffold. | It exposes the real decode order and shows the next gap is resident token-loop state, not only layer-major kernel speed. | Reduce token-major setup/wall cost and then integrate generated/continuation serving measurement. |
| 2026-05-23 | Sprint 266 tested shared dense-op residency in token-major order. | Correctness holds, but decode proxy regresses despite slightly lower wall time. | Keep dense ops local per layer and target TP runtime/KV orchestration or serving integration next. |
| 2026-05-23 | Sprint 267 promoted shared TP runtime for token-major all-layer runs. | In serving order, TP/KV runtime residency improves both wall/setup and summed decode proxy. | Reduce token-major compose/all-to-all and bridge the scaffold into generated/continuation serving measurement. |
| 2026-05-23 | Sprint 268 added token-major position advance. | The scaffold now progresses logical context position across token steps and remains correct. | Run a longer continuous token-major gate, then bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 269 established the longer continuous token-major scaffold baseline. | At 32 steps the path reaches `814.452062` projected slot-step tok/s and compose dominates EP. | Collapse compose/all-to-all or bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 270 removed same-GPU staged compose copies. | Self-copy traffic was a measurable part of compose cost, but compose remains dominant after removal. | Target destination-side reduction/synchronization or bridge to generated/continuation serving measurement. |
| 2026-05-23 | Sprint 271 split compose timing and Sprint 272 tested multi-copy streams. | Copy/all-to-all dominates compose, and per-destination copy streams improve the scaffold. | Pivot to TP/EP generated/continuation serving before more kernel micro-optimization. |
| 2026-05-23 | Sprint 273 added serving-shaped TP/EP metrics. | Decode-only TP/EP rates are promising, but scaffold wall overhead prevents operational serving. | Build a resident serving loop without per-token/per-layer `run_layer()` setup. |
| 2026-05-23 | Sprint 274 built the resident TP/EP serving loop. | Shared dense ops plus direct decode remove the scaffold wall bottleneck and produce useful serving-shaped wall tok/s. | Integrate the resident TP/EP backend with the HTTP sustained-decode harness. |
| 2026-05-23 | Sprint 275 added a sustained-serving artifact wrapper over the resident TP/EP backend. | We need repeatable serving-shaped metrology before and during HTTP harness integration. | Wire the resident backend into the operational HTTP sustained-decode path. |
| 2026-05-23 | Sprint 276 added a TP/EP-only resident HTTP harness. | The backend now stays loaded across HTTP health/status/metrics/generation requests. | Wire this server mode into the appliance launcher and run sustained HTTP matrices. |
| 2026-05-23 | Sprint 277 wired the TP/EP HTTP server into the appliance launcher. | Operators can now start the TP/EP path with `DS4_V100_SERVE_MODE=tp-ep`. | Build and run sustained HTTP matrix tooling against the launcher path. |
| 2026-05-23 | Sprint 278 added sustained HTTP matrix tooling for the launcher path. | The TP/EP server now has repeatable operational metrology. | Wire Kubernetes defaults and capture GPU utilization around the matrix. |
| 2026-05-23 | Sprint 279 wired Kubernetes defaults to the TP/EP path and added GPU-util sampling. | The deployment example no longer points at the frozen PP path, and the HTTP matrix now exposes utilization as well as tok/s. | Build continuous request batching/coalescing for practical serving and keep optimizing compose/copy once metrology is stable. |
| 2026-05-23 | Sprint 280 added resident multi-request HTTP metrology. | One loaded TP/EP server now serves repeated generation requests and exposes cumulative counters. | Add request coalescing/admission so independent HTTP requests can fill the 32 active slots. |
| 2026-05-23 | Sprint 281 exposed TP/EP stage timing in HTTP artifacts. | Operational metrology now shows compose-copy as the largest individual stage. | Optimize compose-copy movement/synchronization, then add true request coalescing. |
| 2026-05-23 | Sprint 282 promoted event-wait compose copy. | Moving copy dependency waits onto CUDA events improves same-binary serving throughput by about `2.5%`. | Reduce FP32 contribution traffic or fuse staged all-to-all reduction more aggressively. |
| 2026-05-23 | Sprint 283 rejected FP16 EP return under event-wait compose. | Reduced payload bytes do not pay for the extra cast/add/final-compose work on V100. | Stay on FP32 return and attack staged contribution traffic/fusion directly. |
| 2026-05-23 | Sprint 284 promoted compact route-compose. | Route-major EP contribution packing reduces staged FP32 traffic and improves same-binary serving throughput by about `11%`. | Re-establish promoted 32/64 topline and add true request coalescing/admission. |
| 2026-05-23 | Sprint 285 established the promoted HTTP serving topline. | The normal TP/EP launcher path now reports about `771-795` wall generated tok/s at `32` slots / `256K`. | Add true request coalescing/admission, then revisit MTP. |
| 2026-05-23 | Sprint 286 added TP/EP HTTP request coalescing. | `32` independent concurrent selected-token requests now form one 32-slot resident decode batch, with `721-787` wall generated tok/s depending on tokens/request. | Replace the selected-token harness with the real prompt/token API and bucketed admission queues. |
| 2026-05-23 | Sprint 287 added bucketed TP/EP admission. | Mixed `32,64` token requests are served as same-length batches instead of rejected, with `32/32` match and zero rejected requests. | Add a prompt/token-compatible diagnostic TP/EP endpoint on top of coalesced bucketed admission. |
| 2026-05-23 | Sprint 288 added diagnostic `/v1/completions` for TP/EP. | Completion-shaped requests now exercise the real coalesced/bucketed resident decode path and return OpenAI-style envelopes, but prompt prefill/output-head text are still explicit gaps. | Wire real TP/EP output-head/top-token selection, then tokenizer text and prompt prefill. |
| 2026-05-23 | Sprint 289 added the TP/EP vocab-sharded output-head gate. | Real `output.weight` shards and output controls now produce a global top-1 token across 8 GPUs; the missing piece is final HC from the serving loop. | Carry final HC through the TP/EP token-major loop and call output-head from `/v1/completions`. |
| 2026-05-23 | Sprint 290 added a resident TP/EP output-head gate and GPU-side shard top-1. | Full-logit host readback roughly doubled output-head latency; device-side top-1 raises the 32-slot resident gate to `3752.194257` output-head tok/s. | Add the TP/EP final-HC carry contract, then feed the resident output head from `/v1/completions`. |
| 2026-05-23 | Sprint 291 added a TP/EP final-HC carry scaffold. | The sharded `[slots][4][512]` per-GPU carry buffer passes 1-token and 4-token all-layer gates with about `0.047 ms/layer` overhead, but currently uses proxy HC rows. | Replace proxy HC with true DS4 HC semantics or wire it only through an explicitly diagnostic output-head path. |
| 2026-05-23 | Sprint 292 wired proxy-HC carry into resident TP/EP output-head serving. | `/v1/completions` can now return diagnostic selected token IDs/logits from the vocab-sharded output head, and a 32-concurrent launcher run passes. | Replace proxy HC rows with true DS4 HC row semantics and feed selected tokens back into decode. |
| 2026-05-23 | Sprint 293 added TP/EP HC final-expand using real layer HC FFN controls. | The output-head bridge no longer depends on arbitrary row-scaled proxy HC; 32-concurrent completions pass with `proxy_hc=0`, `160.904882` wall tok/s, and `271.342877` decode tok/s for the 1-token diagnostic case. | Implement the full DS4 HC attention/FFN pre/post sequence, then token feedback and text output. |
| 2026-05-23 | Hard cut to TP/EP-only implementation work. | Sprint 225 showed the frozen PP path is correct but bottlenecked by layer-scheduled pipeline bubbles. User directed zero further PP variant work. | Sprint 226 starts the TP-only planner and topology contract. |
| 2026-05-23 | Deferred MTP until after TP/EP serving. | MTP can be useful only after the serving runtime has the right topology and multi-slot decode behavior. | Revisit after TP/EP serving exists and has multi-slot throughput evidence. |
| 2026-05-24 | Reframed the vision from "make the API respond" to production readiness. | Sprints 303-306 made the TP/EP path askable through text/chat APIs, but the remaining risk is trustworthiness and service hardening, not another endpoint wrapper. | Sprint 307 starts reference parity before persistent deployment and performance/MTP work. |
| 2026-05-24 | Sprint 308 identified diagnostic TP/EP semantics as the parity blocker. | Synthetic EP routes, six-local-expert packing, and simplified attention cannot produce reference DS4 tokens. | Remove diagnostic caps, implement router-driven EP, then wire full DS4 attention semantics. |
| 2026-05-24 | Sprint 308 moved TP/EP from synthetic routes to active-slot model-router routes. | Full expert residency fits, model-router routes are nonzero for active HTTP slots, and per-route weights are wired, but parity still fails (`16` expected, ` ICC` returned). | Isolate the `ffn_normed` routed-input non-finite failure, implement full shared FFN, then replace the attention bridge. |
| 2026-05-24 | Sprint 308 wired true shared FFN in the TP/EP path. | `ffn_gate_shexp`, `ffn_up_shexp`, FP32 SwiGLU midpoint, and packed-FP8 `ffn_down_shexp` now execute under `DS4_V100_TP_EP_TRUE_SHARED_FFN=1`; FP16 midpoint was rejected because it overflows/saturates, and routed-normalized input still fails inside the TurboMind routed executor. | Fix normalized routed expert input with a layer-0 microbench, then continue replacing the proxy hidden/attention bridge with true DS4 HC attention/FFN semantics. |
| 2026-05-24 | Sprint 308 fixed the routed-normalized nonfinite failure. | The reference routed path clamps gate/up at `10` before SwiGLU, while the TurboMind gated-SiLU epilogue is unclamped; the TP/EP path now uses plain gate/up plus a CUDA clamp+SwiGLU when normalized routed input is enabled, and the previous layer-0 HTTP 500 is gone. | Treat parity as a true graph-semantics gap now: replace the proxy hidden/attention/HC bridge with the full DS4 sequence, then optimize the clamped routed path back into a fused executor. |
| 2026-05-24 | Sprint 308 replaced synthetic compose residuals with current hidden shards and clamped true shared SwiGLU. | The TP/EP path now composes from real `d_current_shard`, and shared FFN midpoint magnitude drops from million-scale to about `100` by matching the reference `10.0` SwiGLU clamp. The one-token parity case still returns the wrong token (`uerak` vs `16`) at about `50` decode tok/s, so the remaining blocker is graph semantics rather than numeric blow-up. | Implement the real DS4 attention/HC bridge and token-state feedback in the TP/EP serving path; defer further FFN kernel fusion until top-token parity is closer. |
| 2026-05-24 | Sprint 308 gated reference HC reduce as a diagnostic path. | Switching HC reduce to 20 Sinkhorn iterations and removing the diagnostic weighted-sum clamp causes V100 FP16/TurboMind activation overflow at the routed FFN boundary; stable RMS plus saturating f32-to-fp16 prevents route-input infinities but still overflows gate/up. The serving default remains operational and the reference path is opt-in via `DS4_V100_TP_EP_REFERENCE_HC_REDUCE=1`. | Design an explicit activation scaling/quantization contract for reference-HC outputs before promoting the reference HC bridge; continue real attention/prefill semantics separately. |
| 2026-05-24 | Sprint 309 localized the reference-HC instability. | Route-local activation scaling keeps the normalized routed FFN path finite, but unguarded reference-HC state grows to `1e15+` by layer 30 and first becomes non-finite in `final_hc_shard` at layer 32, after `compose_next_hidden` is still finite. An explicit diagnostic guard, `DS4_V100_TP_EP_REFERENCE_HC_STATE_GUARD=1`, lets the full HTTP parity request complete with a wrong token (`[$` vs `16`) instead of HTTP 500. | Replace the simplified HC/attention bridge with true DS4 HC attention/compressed-KV/indexer semantics; keep the state guard diagnostic-only and do not treat it as model correctness. |
| 2026-05-24 | Sprint 310 starts replacing the simplified TP/EP attention bridge. | The resident TP/EP runtime can now opt into binding the full DS4 attention projection set (`attn_q_a`, `attn_q_b`, `attn_kv_latent`, `attn_output_a`, and `attn_output_b`) across all 43 layers instead of only the final attention output projection. | Wire those resident tensors into the real q/kv/RoPE/raw-KV/compressed-KV/indexer/attention/output sequence, then rerun the reference parity gate. |
| 2026-05-24 | Sprint 311 executed the first true-attention projection prefix. | The TP/EP runtime now runs `attn_norm`, `attn_q_a`, `attn_q_a_norm`, `attn_q_b`, `attn_kv_latent`, and `attn_kv_a_norm` for all 43 layers at `32` slots / `256K`; the V100 gate has 43 projection-prefix passes and zero failures. | Continue the attention sequence with q-head norm/RoPE, raw and compressed KV updates, ratio-4 indexer row selection, raw+compressed attention, inverse RoPE, and `attn_output_a -> attn_output_b`. |
| 2026-05-24 | Sprint 312 added the first true-attention state-update gate. | The TP/EP runtime now normalizes local q-head shards and writes diagnostic raw SWA KV rows for all 43 layers at `32` slots / `256K`; the state gate passes, but raw KV saturates to `65504` in early layers. | Isolate raw-KV saturation, then add q-head RoPE, attn_sinks, raw-SWA attention read, and `attn_output_a -> attn_output_b` before feeding attention output into hidden state. |
| 2026-05-24 | Sprint 313 added the first true-attention raw-read gate. | The TP/EP runtime now loads `attn_sinks` and executes a sink-aware one-row raw-SWA attention read for all 43 layers at `32` slots / `256K`; the raw-read gate passes but inherits early-layer saturation. | Replace the one-row diagnostic read with full q-RoPE, raw-window, compressed-KV/indexer, and attention-output projection semantics, then rerun reference parity. |
| 2026-05-24 | Sprint 314 added a raw-window attention-read gate. | The TP/EP runtime now reads resident raw-SWA rows accumulated across token-major steps; the `32` slot / `256K` / `4` step V100 gate has 172 raw-window passes, `valid_rows=1..4`, and zero failures. | Add RoPE plus compressed-KV/indexer read semantics, then wire `attn_output_a -> attn_output_b` only after saturation is isolated. |
| 2026-05-24 | Sprint 315 added true-attention RoPE before raw-SWA storage/read. | The TP/EP runtime now applies DS4-style tail RoPE to q-head shards and latent KV rows; the `32` slot / `256K` / `4` step V100 scaffold has 172 RoPE passes, 172 token-major layer passes, and zero failures. One raw-window diagnostic line was stdout-interleaved, but the final scaffold reports 172 pass invocations. | Isolate the early-layer `65504` raw-KV saturation in the HC-current/projection/KV-store contract before compressed-KV/indexer read or attention-output promotion. |
| 2026-05-24 | Sprint 316 localized true-attention saturation to the KV normalization path. | The new saturation audit gate passed at `32` slots / `256K` / `4` steps and showed `kv_normed` first exceeds FP16 range at layer `1`, before KV RoPE and before raw-SWA storage; q-heads remain bounded after head RMSNorm/RoPE. | Compare TP/EP `attn_kv_a_norm` against the DS4 reference normalization/scaling contract and fix that before compressed-KV/indexer work. |
| 2026-05-24 | Sprint 317 identified a TP/EP block-reduction broadcast bug. | The KV norm reference gate showed huge same-input drift between stable and plain RMSNorm; code inspection found `block_sum_256_f32` and `block_max_256_f32` only return the reduced value to the first warp, so threads `32..255` normalize with the wrong scale. | Fix the reduction helpers, then rerun KV norm reference, saturation, and raw-window gates before continuing attention semantics. |
| 2026-05-24 | Sprint 318 fixed TP/EP block-reduction broadcast. | KV norm reference drift dropped to `~1e-6`, raw-SWA max dropped from `65504` to `~6.29`, and the combined `32` slot / `256K` / `4` step gate passed all 172 layer-step invocations with zero failures. | Rerun reference parity and continue compressed-KV/indexer plus attention-output semantics. |
| 2026-05-24 | Sprint 319 reran the TP/EP HTTP reference parity gate after the reduction fix. | The official `short_reasoning_plain` vector still fails, but the live output changed from `ICC` / token `95933` to `)Skip` / token `83480`, proving the reduction fix reaches the askable serving path. | Implement the remaining true DS4 attention semantics: compressed KV/indexer row selection, raw+compressed attention merge, `attn_output_a -> attn_output_b`, and hidden-state promotion. |
| 2026-05-24 | Sprint 320 added a TP/EP true-attention output projection gate. | The real `attn_output_a -> attn_output_b` sequence now runs over rank-local 4096-wide attention heads and gathers the 8192-wide intermediate before producing per-rank hidden shards; the `32` slot / `256K` / `4` step V100 gate passes structurally with 172 layer-step invocations and zero failures. | Promote `attn_output_b` shards into the attention residual/current-hidden path, then rerun the reference parity vector. |
| 2026-05-24 | Sprint 321 reran HTTP reference parity with true-attention output enabled. | The official vector still fails, but output changed from `)Skip` / token `83480` to `urf` / token `64906`, proving the new attention output path is active in serving. | Reorder the layer path so FFN norm/router/shared/routed FFN consume post-attention residual/current hidden, then rerun parity. |
| 2026-05-24 | Sprint 322 promoted post-attention hidden into FFN inputs. | The TP/EP runtime now materializes `current + attn_output_b`, recomputes FFN norm/router/shared/routed inputs from that tensor, and passes the `32` slot / `256K` all-layer gate; HTTP parity still fails but changes to `mere` / token `88445`. | Implement true compressed-KV/indexer attention and raw+compressed attention merge, then rerun reference parity. |
| 2026-05-24 | Sprint 323 added the first TP/EP compressed-KV/indexer projection gate. | The TP/EP runtime now binds BF16 compressor/indexer dense tensors through the FP16-cache/cuBLAS resident path and executes compressor plus ratio-4 indexer projections for all 43 layers at `32` slots / `256K`. The all-layer gate passes with 43 compressed-projection rows and `19.630630` projected slot-step tok/s. HTTP parity now runs without OOM after freeing unused dense float staging buffers and moving token embeddings to host-backed per-slot row uploads; parity still fails but changes to `MARK` / token `110609`. | Implement real compressed-row storage, indexer scores/top-k over stored rows, and raw+compressed attention softmax/value merge. |
| 2026-05-24 | Sprint 324 added bounded TP/EP compressed-row storage and raw+compressed attention read. | The TP/EP runtime now gathers compressor/indexer TP shards, stores compressor state with APE, emits pooled/RMSNorm/RoPE/F16-rounded compressed rows, shifts ratio-4 state, computes a bounded one-row indexer score/top-k, and merges a visible compressed row into the attention read. The `32` slot / `256K` all-layer smoke passes with `pass_invocations=43` and `19.160884` projected slot-step tok/s. HTTP parity still fails and returns `mere` / token `88445`, so this structural path is active but not yet reference-equivalent. | Compare TP/EP layer-2 ratio-4 emitted compressed rows, indexer scores, selected rows, and raw+compressed attention output against the non-TP reference path. |
| 2026-05-24 | Sprint 325 added a compact compressed-reference diff gate and fixed layer-local attention state. | The first all-layer diagnostic found layer `4` diverging at `attn_comp_row0_compact_reference` because raw-SWA, attention-compressed, and indexer-compressed buffers were reused across layers in the smoke path. The buffers are now layer-local; `slots=1` / `position=100003` and `slots=32` / `position=262143` both pass all 43 layers, and ratio-4 compact compressed-row/indexer-score diffs pass through layer `42`. The `32` slot diagnostic reports `39.258626` projected slot-step tok/s. | Replace the compact one-row diagnostic with full production compressed-row cache/history selection and raw+compressed attention output parity against the reference layer path, then rerun HTTP parity. |
| 2026-05-24 | Sprint 326 added bounded multi-row compressed attention history. | The TP/EP path now stores up to `8` bounded compressed rows per layer, tracks visible row counts, scores all bounded visible ratio-4 indexer rows, replicates selected row indices to all TP ranks, and includes multiple compressed rows in the raw+compressed attention softmax/read. The `32` slot / `256K` / `8` step attention gate passes all `344` layer-step invocations with `visible_compressed_rows=2`, `selected_compressed_rows=2`, no compact diff failures, and `20.780883` projected slot-step tok/s. | Replace bounded diagnostic rows with production compressed-KV allocation/ownership, validate ratio-128 history, and compare raw+compressed attention output against the full reference layer path before rerunning HTTP parity. |
| 2026-05-24 | Sprint 327 made the production compressed-KV memory contract executable. | `tools/ds4-v100-plan-tp.c` now reports raw/compressed/indexer rows, persistent typed KV bytes, replicated f32 warning bytes, bounded diagnostic bytes, per-layer row tables, and JSON fields. With the real pack and F8 KV, `32` slots / `256K` fits at `27.00 GiB/GPU` with `3.40 GiB/GPU` persistent typed KV and `5.00 GiB` headroom after reserve; replicated f32 would be `107.84 GiB/GPU`. `1` slot / `1M` fits at `22.56 GiB/GPU`. | Implement the runtime allocator against this typed TP-sharded contract and validate ratio-4 plus ratio-128 row reads from the production arena. |
| 2026-05-25 | Sprint 345/346 moved performance work from tok/s guessing to profiler evidence. | Broad `nvprof` shows tensor-core-capable Cutlass and TurboMind kernels are active, but the path is launch/transform fragmented: compressor, gather, dense-fill, and many small WMMA launches dominate. TP/EP CUDA profiler windows are now wired, but HTTP-wrapper `profile-from-start off` does not emit scoped metrics. | Build a direct non-server TP/EP replay/profile target, then fuse the largest non-GEMM boundary proven by that target. |
| 2026-05-25 | Sprint 347 made direct TP/EP profiling operational. | `tools/ds4-v100-tp-ep-profile.py --run-mode direct-token-major` reuses the 32-slot / 256K typed-KV serving flags without the HTTP wrapper and produces usable `nvprof` top-kernel rows. Direct no-profiler measured `83.882587` generated tok/s decode and `91.958152` continuation tok/s decode; windowed direct `nvprof` showed TurboMind FP4 HMMA, CUTLASS WMMA, dense-fill, compressor store, and BF16/F8 transform kernels active. The dominant direct-stage timer is `sum_hc_current_input_ms=622.442653` out of `762.971220` summed decode ms. | Target HC/current-input staging and transform fragmentation directly, then validate with direct profiler and HTTP serving A/B. |
| 2026-05-25 | Sprint 348 rejected naive HC current peer-gather. | The opt-in `--tp-hc-current-input-peer-gather-gate` path is correct, but slower: control measured `87.263615` generated tok/s decode and `596.248809` HC-current ms, while peer gather measured `67.495350` and `801.525057`. Spreading the full-current gather to every rank adds more overhead than it removes from the GPU0 broadcast path. | Keep the gate diagnostic-only. Next target is HC control synchronization/fusion or direct sharded fill, not naive all-rank peer gathering. |
| 2026-05-25 | Sprint 349 promoted HC-current stream-scoped barriers. | `--tp-hc-current-input-stream-sync-gate` keeps the layout unchanged but replaces selected GPU0 device-wide barriers with rank-0 stream barriers. Direct A/B improved generated decode tok/s from `74.841520` to `81.190638` and HC-current ms from `711.608991` to `647.492171`. HTTP 32-request A/B improved server generated tok/s from `82.573137` to `83.813937`, with `32/32` HTTP 200. | Promote `DS4_V100_TP_EP_HC_CURRENT_INPUT_STREAM_SYNC=1`; next fuse or bypass the HC control/fill chain itself. |
| 2026-05-25 | Sprint 350 split the HC-current timer and corrected the bottleneck interpretation. | The actual HC-current substages sum to `83.066250` ms, while the old `sum_hc_current_input_ms` field is `557.301289` ms. The broad field includes true-attention/compressed-KV prefix work before the EP timer begins. | Stop chasing HC-current gather/broadcast as the main bottleneck. Split and optimize the true-attention/compressed-KV prefix next. |
| 2026-05-25 | Sprint 351 split the true-attention pre-EP prefix. | The prior broad pre-EP timer is now explained by measured stage totals: compressed KV projection/store `228.813152` ms, attention projection `170.865666` ms, attention state `105.654904` ms, HC-current `85.249101` ms, raw/window read `34.932798` ms, and typed-history load `1.271677` ms across `86` layer-step invocations. | Optimize compressed-KV projection/store fragmentation first, then rerun direct profiler and HTTP A/B before moving to MTP. |
| 2026-05-25 | Sprint 352 split compressed-KV internals and rejected store suppression as the next lever. | At emitted row position `262143`, the one-token 32-slot run passes and shows compressed-KV is dominated by indexer dense `36.615896` ms, attention dense `24.659453` ms, attention state/emit `24.362932` ms, combined input fill `16.776362` ms, and indexer state/emit `9.007686` ms. Suppressing compressed and indexer typed stores is flat: `81.647302` to `81.733945` generated decode tok/s. | Target fused/shared compressor-indexer input fill and compressor state/emit work before revisiting typed KV stores. |
| 2026-05-25 | Sprint 353 tested fused ratio-4 compressor/indexer input fill. | The opt-in fused fill selected all `21` ratio-4 layers and preserved the same output token. Same-binary emitted-row decode improved only from `79.011931` to `80.534845` tok/s, and pre-EP compressed-KV time moved from `130.391665` to `129.781758` ms. | Keep fused fill diagnostic-only. Target compressor/indexer state+emit fusion or dense/state boundary reduction next. |
| 2026-05-25 | Sprint 354 rejected narrow compressed RoPE+round fusion as a material lever. | The opt-in fused kernel selected `41` emitted compressed layers and preserved token `54639`, but decode moved from `79.810167` to `79.344207` tok/s while compressed-KV stayed effectively flat. | Keep the gate diagnostic-only. Target larger state/emit boundaries such as pooling+normalization or store+pooling. |
| 2026-05-25 | Sprint 355 found a real but small win from fused compressed pool+norm. | The opt-in fused pool+norm kernel selected `41` emitted layers, preserved token `54639`, reduced compressed-KV sum from `130.510967` to `127.736989` ms, and improved decode from `81.189757` to `81.687107` tok/s. | Keep opt-in pending repeat/combination testing with fused input fill before promotion. |
| 2026-05-25 | Sprint 356 wired compressed fusion gates into the serving launcher and tested the combined direct path. | New default-off env vars expose fused input-fill, fused RoPE+round, and fused pool+norm to TP/EP serving. Direct emitted-row input-fill + pool-norm preserved token `54639` and improved decode from `80.511365` to `81.311102` tok/s. | Keep opt-in. Add an emitted-row HTTP/profile mode or repeat direct A/B before promoting defaults. |
| 2026-05-25 | Sprint 357 added emitted-row selected-token HTTP profiling. | `--http-endpoint selected-token` avoids prompt-prefill ambiguity, returns `32/32` HTTP 200 responses at `position=262143`, and shows fused input-fill + pool-norm reducing parsed compressed-KV sum from `127.697384` to `123.651985` ms while one-token client tok/s remains flat. | Keep fusions opt-in. Run longer amortized serving A/B or continue reducing compressed state/emit fragmentation before default promotion. |
| 2026-05-25 | Sprint 358 ran the longer selected-token HTTP A/B. | `position=262112` leaves room for a 32-token run while still reaching the emitted-row boundary. Combined input-fill + pool-norm is not promotable; pool-norm only improves client tok/s and compressed-KV sum but regresses the scaffold decode proxy. | Keep all compressed fusions default-off. Confirm pool-norm with repeated/direct multi-step A/B or move to deeper compressed state/emit fusion. |
| 2026-05-25 | Sprint 359 promoted fused compressed pool+norm. | Direct 32-step non-HTTP A/B resolves the Sprint 358 metric conflict: pool+norm improves decode tok/s by `+1.84%`, wall tok/s by `+1.77%`, and compressed-KV sum by `62.624806 ms` with the same first token. | Keep pool+norm default-on for TP/EP serving. Continue deeper compressed state/emit fusion or rerun full HTTP chat topline with the promoted default. |
| 2026-05-25 | Sprint 360 validated the pool+norm default through the launcher. | The TP/EP launcher emits the pool+norm gate by default, and a launcher-started selected-token HTTP run returns `32/32` HTTP 200 with `187` fused pool-norm rows and `73.289956` client generated tok/s. | Use the launcher default for future TP/EP serving tests; next rerun chat/topline or continue compressed state/emit fusion. |
| 2026-05-25 | Sprint 361 ran the launcher chat/completions A/B. | The promoted pool+norm default is active through chat and stable, but short chat is flat/slightly slower: `24.118711` vs `24.280060` client generated tok/s for `8` tokens/request. | Do not claim chat topline improvement. Use longer decode-heavy chat tests or continue larger compressed-KV fusion work. |
| 2026-05-25 | Sprint 362 aligned the profile harness with launcher defaults. | HTTP profile runs now inherit the production pool+norm default, and an explicit `--disable-fused-compressed-pool-norm` flag provides the control path. V100 proof returned `1/1` HTTP 200 in both modes with `40` fused pool layers by default and `0` when disabled. | Use the permanent harness for future launcher-level TP/EP A/B tests instead of ad hoc shell scripts. |
| 2026-05-25 | Sprint 363 rejected wider emitted-row scalar fusion. | The new fused pool+norm+RoPE+round kernel is correct and selected on all emitted rows, but the full 32-step direct gate regressed: `95.463298` vs `95.908399` generated decode tok/s and `3470.682826` vs `3460.932833` ms compressed-KV sum. | Keep the gate diagnostic-only. Shift optimization upstream to compressed/indexer dense projection or current/gather staging. |
| 2026-05-25 | Sprint 364 rejected remote direct compressed input fill. | Reading `hc->d_attn_normed` directly from rank-0 memory is legal and correct but much slower: one-step compressed-KV sum regressed from `126.724613` to `260.365841` ms, with attention/indexer input-fill costs increasing sharply. | Do not use peer-read half-fill. Preserve local staged current reads and target local launch reduction or dense projection kernels. |
| 2026-05-25 | Sprint 365 rejected local attention input-fill micro-fusion as a default. | The fused local attention fill gate is correct and slightly positive in direct 32-step decode (`94.237924` to `94.396298` tok/s), but selected-token HTTP regresses (`72.886325` to `70.674037` client tok/s). | Keep the gate diagnostic-only. Move up to larger compressed/indexer dense projection or attention projection/state boundaries. |
| 2026-05-25 | Sprint 366 promoted compressed dense event waits. | Replacing host synchronizes between compressed input fills and dense launches with CUDA event dependencies preserves tokens and improves selected-token HTTP from `71.833757` to `74.432464` client tok/s at `32` slots / `256K`. | Keep the gate default-on and disableable; next target the remaining compressed/indexer dense projection and state costs. |
| 2026-05-25 | Sprint 367 confirmed the event-wait default through chat. | Valid long-context chat at `position=262080`, `32` slots, `32` requests, and `32` generated tokens/request improved client tok/s from `50.648397` to `52.022782` and server decode tok/s from `96.116667` to `99.521680`. | Keep using chat-valid start positions that reserve prompt-prefill room; next optimize the remaining dense/state costs or admission/context accounting. |
| 2026-05-25 | Sprint 368 added TP/EP chat context admission. | Over-context chat now returns HTTP 400 with `context_window_exceeded` before GPU decode; valid 32-request/32-token chat at `position=262080` still passes. | Extend admission toward active-slot/variable-length serving and continue dense/state optimization. |
| 2026-05-25 | Sprint 369 added opt-in GPU utilization sampling to the TP/EP profile harness. | `--gpu-sample-interval-ms` writes `gpu_util.csv` and summary utilization fields without overhead when disabled. A 4-request / 32-slot chat smoke passed and showed `8.412879%` average GPU util with GPU0 much busier than peers. | Use sampled active-slot matrices before changing scheduling; then optimize active-slot compaction, dense projection/state fragmentation, or EP balance with utilization evidence attached. |
| 2026-05-25 | Sprint 370 added the active-slot matrix driver. | The smoke matrix for active requests `1,4` passed and wrote aggregate TSV/JSON plus per-case profile artifacts; decode stayed flat around `101` tok/s and average GPU util stayed around `8.3%`. | Run the full `1,4,8,16,32` longer-decode matrix, then choose active-slot compaction versus deeper dense/state kernel work from the evidence. |
| 2026-05-25 | Sprint 371 ran the full active-slot matrix. | At `32` slots / `256K` / `32` tokens/request, all cases `1,4,8,16,32` passed. Client aggregate tok/s scaled with active responses, but server decode stayed `97.4-100.0` tok/s and average GPU util stayed `9.8-10.3%`. | Use active-slot compaction for low-occupancy efficiency later; next optimize the full 32-slot bottleneck in compressed/indexer dense projection, attention projection/state, and GPU0-heavy staging. |
| 2026-05-25 | Reprioritized performance work around `TEMP_THROUGHPUT_PROMPT.md`. | The full active-slot matrix plus INT8-compressor rejection makes another narrow dtype swap less compelling than testing launch/sync elimination. | Sprint 375 tested async-output synchronization removal; Sprint 376 is the CUDA graph make-or-break gate before paged attention, compact MoE, TP-expert A/B, FP8 KV, or MTP. |
| 2026-05-25 | Sprint 375 rejected async output as a default. | The gate preserved tokens and reduced output-head device syncs, but the real HTTP A/B regressed server decode tok/s and did not improve utilization. | Keep `--async-output-gate` opt-in; Sprint 376 should audit and test CUDA graph capture without assuming output-head event sequencing is a serving win. |
| 2026-05-25 | Tightened the vision around `TEMP_THROUGHPUT_PROMPT.md`. | The next performance work should not blur multiple ideas together; each gate needs a same-binary V100 A/B and a promote/reject decision. | Finish Sprint 376's graph audit, then choose graph replay, paged attention, compact MoE, TP-expert A/B, FP8 KV, or MTP from measured evidence. |
| 2026-05-25 | Sprint 376 initial graph audit ran on V100. | The decode step is not yet capturable: it has `172` broad host synchronization points across the 43-layer step. | Replace those syncs with stream/event dependencies where possible, then rerun the audit before graph replay. |
| 2026-05-25 | Sprint 376 event-barrier audit ran on V100. | Top-level `sync_all` host waits can be replaced with CUDA event ordering while preserving token/checksum parity, but the pre-graph path slows down and helper synchronizations still block capture. | Convert helper-level waits next; do not promote the event-barrier path as a performance optimization by itself. |
| 2026-05-25 | Sprint 376 HC-current helper event pass ran on V100. | HC-current stream/control waits can be event-ordered under the graph gate; parity holds and helper blocker classes drop from `7` to `6`. | Continue helper-level event ordering, starting with final HC expansion and attention helpers. |
| 2026-05-25 | Sprint 376 final-HC helper event pass ran on V100. | Final-HC host waits can be event-ordered under the graph gate; parity holds and helper blocker classes drop from `6` to `5`. | Continue with attention projection/state/output and compressed-KV helper waits. |
| 2026-05-25 | Sprint 376 attention-projection helper event pass ran on V100. | Attention projection host waits can be event-ordered under the graph gate; parity holds and helper blocker classes drop from `5` to `4`. | Continue with attention state/raw-read/output and compressed-KV helper waits. |
| 2026-05-25 | Sprint 376 raw-read helper event pass ran on V100. | Raw attention read/window host waits can be skipped under the graph gate; parity holds and helper blocker classes drop from `4` to `3`. | Continue with attention state, typed-history, and compressed-KV helper waits. |
| 2026-05-25 | Sprint 376 cleared tracked graph-audit helper blockers. | Attention-state, typed-history, and compressed-KV event-ordering passes preserve first token `54639`, output checksum `24071637347`, and scaffold checksum `3401922407`; helper blocker classes drop to `0`, and the one-step non-emitted-row audit reports `capture_eligible=1`. | Attempt real CUDA graph capture/replay next. If capture rejects an operation or replay is flat, close Sprint 376 with the blocker/performance result and pivot to batched paged attention, compact MoE, or TP-sharded expert A/B. |
| 2026-05-25 | Re-centered the vision around `TEMP_THROUGHPUT_PROMPT.md` before further performance work. | The prompt's main insight is that the current TP/EP path is flat at about `97-100` server decode tok/s and about `10%` average GPU utilization from `1` to `32` active requests, so the next work should test launch/sync elimination and launch-count reduction before broad dtype rewrites. | Keep Sprint 376 focused on one real CUDA graph capture/replay result. If it is blocked or flat, move to batched paged attention, compact MoE, fused gated-SiLU, and then TP-sharded expert A/B as isolated default-off gates. |
| 2026-05-25 | Sprint 376 real capture attempts rejected CUDA graph replay. | After audit cleanup, real capture failed first on separate capture sequence merging, then uncaptured stream dependencies, then `cudaMemcpyPeerAsync` being disallowed during stream capture. Replacing HC-current peer copies with graph-gated device copy kernels moved the same peer-copy error to attention projection. | Close Sprint 376 as REJECT, leave graph work diagnostic-only, and plan Sprint 377 around `--batched-paged-attn-gate`. |
| 2026-05-25 | Rebased the vision on `TEMP_THROUGHPUT_PROMPT.md` after Sprint 377 baseline/plumbing. | S-B and S-A are no longer future bets: async output regressed and graph capture is blocked by P2P transport. Sprint 377's fresh typed long-context baseline is `88.372350` server decode tok/s, `40.157540` client tok/s, and `7.972222%` average GPU utilization at `32` slots / `256K`. | Finish S-C row-family planning and the first batched attention/KV kernel, then move to compact MoE, fused gated-SiLU, TP-expert A/B, FP8 KV, and MTP in that order unless measured evidence changes the sequence. |
| 2026-05-25 | Re-aligned the vision to the isolated throughput prompt after S-C closed. | S-C row planning is no longer the active branch: the observed typed-history pending reload count is already `0`, so a narrow load-only paged-attention kernel is unlikely to move the topline. The active performance work is S-D compact MoE, specifically real model-router compatibility with compact EP compose. | Finish `--compact-moe-decode-gate`, then run S-E fused gated-SiLU, S-F TP-expert A/B, S-G FP8 KV, and S-H MTP as separate default-off gates with same-binary V100 A/Bs. |
| 2026-05-25 | Sprint 378 promoted compact MoE for model-router compact compose. | Direct A/B preserved first token `54639` and checksum `6840320333`, improving decode from `62.617354` to `66.481242` tok/s. HTTP serving A/B preserved response token streams, improved client throughput from `37.394075` to `39.034685` tok/s, server decode from `80.812914` to `81.313535` tok/s, and compose from `19.167728` to `14.703119` ms. | Use the promoted model-router compact-compose path as the baseline for S-E `--fused-gated-silu-gate`. |
| 2026-05-25 | Sprint 379 phase 1 tested the generic fused gated-SiLU epilogue. | The current production-shaped branch already has `routed_gate_standalone_swiglu=0`; explicit fused mode preserved first token `54639` but was effectively a no-op. The routed-normalized branch has the standalone clamped launch; generic fused mode removed it and improved direct proxy from `45.368432` to `57.367413` tok/s, but changed first token from `41432` to `54639`. | Do not promote the generic epilogue. Continue S-E only through a true DS4-clamped TurboMind epilogue ABI, or close S-E with that concrete blocker. |
| 2026-05-25 | Sprint 379 implemented the true DS4-clamped TurboMind epilogue ABI. | The ABI exports and the clamped fused gate is fast in layer-0 EP-only isolation (`4.102144` ms two-step gate versus `0.622592` ms fused gate), but resident direct serving A/B with `routed-normalized + fused-gated-silu` fails at layer 0 before the routed gate executes due to the dense-KV precheck returning rc `4`. | Keep S-E default-off and diagnostic-only. Move to S-F TP-sharded expert A/B unless we first add a deterministic fused-gate parity harness or diagnose the resident dense-KV precheck interaction. |
| 2026-05-25 | Sprint 380 started TP-sharded expert A/B measurement. | Added the permanent driver and reran TP8 TurboMind MXFP4 route tiers. TP8 still fails correctness for `96/192/384` routes and total speedup is `0.523x/0.353x/0.335x`; EP8 direct control at the target shape is `66.569095` tok/s with first token `54639`. | Do not integrate TP8 experts. Continue Sprint 380 by exposing/rerunning TP4, which was the historically correct branch. |
| 2026-05-25 | Sprint 380 reran TP4 and TP8 under one driver. | TP4 is correct at `96/192/384` routes with total speedup `1.055x/0.891x/0.927x`; TP8 remains incorrect with large NaN counts. The simple TP output reduction dominates at larger route tiers. | Do not integrate TP-sharded experts into serving yet. Revisit only with a fused TP4 reduction/compose boundary, otherwise move to the next Vision gate. |
| 2026-05-25 | Sprint 381 implemented the FP8 E5M2 KV gate. | E5M2 row/device smokes passed for `attn`, `attn_raw`, and `indexer`; direct 4-token checksum matched while decode improved `70.710875 -> 75.787866` tok/s; HTTP selected-token 4-token client throughput improved `17.212677 -> 22.389190` tok/s with first-token parity. | Keep E5M2 default-off. It is promising, but needs longer parity/soak and VRAM margin work before replacing E4M3. |
| 2026-05-25 | Sprint 389 promoted compressed dense stats skip. | Against the current real-router compact-MoE TP/EP baseline at `32` slots / `256K`, direct decode improved `91.869507 -> 102.871437` tok/s with first token `98751`; HTTP chat server decode improved `89.709430 -> 103.758804` tok/s, client throughput improved `42.183007 -> 44.592824` tok/s, first token stayed `83484`, all generated token sequences matched, and checksum stayed `17913667583206000416`. | Promote `DS4_V100_TP_EP_TRUE_DS4_COMPRESSED_KV_SKIP_DENSE_STATS=1` as the launcher/profile default; explicit `=0` or `--disable-skip-compressed-dense-stats` keeps the diagnostic stats path available. |
| 2026-05-25 | Sprint 390 made HTTP response parity permanent. | Added `tools/ds4-v100-http-response-parity.py`; it passes on Sprint 389's `32` control/candidate response pairs and fails a mutated generated-token fixture. | Use this comparator for future HTTP A/B promotion evidence. |
| 2026-05-25 | Sprint 391 reran longer E5M2 KV parity. | E5M2 preserved first token and passed `32/32` HTTP response parity pairs. HTTP server decode improved `101.206458 -> 107.281060` tok/s and client throughput improved `46.115999 -> 47.895831`, but direct decode moved `103.237368 -> 102.152512`. | Keep E5M2 default-off pending broader multi-prompt parity/soak. |
| 2026-05-25 | Sprint 392 added multi-prompt soak support. | `--prompt-file` now lets the HTTP profiler cycle JSONL chat prompts. The `16` prompt E5M2 soak passed `32/32` parity pairs, but server decode was flat (`106.390802 -> 106.483285`) and the layout does not save VRAM. | Keep E5M2 default-off; use prompt files for future risky gate promotion checks. |
| 2026-05-26 | Sprint 397 tested serving-path NCCL compose. | Added a default-off `--nccl-reduce-scatter-compose-gate` and launcher/profile wiring. The compatible non-compact FP32 EP compose path preserved checksum `1908166124`, but NCCL was slower than peer-copy fused compose at layer 2 / `32` slots (`6.401091` ms vs `2.521989` ms). Compact route compose correctly leaves the backend inactive because the production path is route-indexed and not a dense reduce-scatter. | Keep NCCL compose diagnostic-only. Use NCCL for future true TP hidden/expert collectives, but do not force it into current compact EP compose. |
| 2026-05-26 | Sprint 410 promoted HC-current NCCL at the target HTTP serving shape. | Added `tools/ds4-v100-tp-ep-nccl-http-ab.py` and ran a same-binary control/candidate A/B at `32` requests / `32` slots / `256K` / `32` tokens. Both legs passed readiness, response parity matched `32/32`, and HC-current NCCL improved server generated decode from `101.897890` to `107.723452` tok/s with `2106 MiB` minimum free VRAM. | Promote `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1` in the appliance launcher/env defaults. Keep the harness for future NCCL and TP/EP default checks; next work should target broader collectives or request orchestration because client tok/s and utilization did not improve. |

## Open Questions

1. What exact reference tolerance should gate TP/EP production readiness:
   top-token match only, bounded logit drift, or prompt-level output agreement?
2. Which prompt suite should become the fixed parity set for DS4 Flash on V100:
   short chat, long-context retrieval, tool-like JSON, coding, or all of them?
3. Should persistent service exposure first be plain port-forwarded HTTP on the
   build pod, or a Kubernetes service/deployment using the same node-local
   model paths?
4. Should active-slot-only decode land before or after streaming? Active-slot
   decode helps low-occupancy use; streaming improves practical UX and timeout
   behavior.
5. Does CUDA graph replay materially raise GPU utilization at the real
   `32` slot / `256K` decode shape, or is the remaining bottleneck inside
   kernel math/state movement rather than host launch overhead?
