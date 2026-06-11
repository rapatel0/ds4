# Sprint 599 Report - Post-C1 Layer Budget: Prefix, Overlap, Barriers

Date: 2026-06-11
Status: complete - **stretch target NOT reached; measured floor analysis
delivered; no promotions; a latent ordering hazard in the promoted layer is
the load-bearing discovery**

Environment: the s598 promoted setup (pack `...-s597`, contract, control,
launcher default `DS4_V100_TP_EP_EP_RETURN_TRANSPORT=nccl`), fixed harness,
GPUs idle-verified between runs (the s598 wait-for-idle preflight; one
self-inflicted reserve-check race recorded, no foreign-tenant overlap in any
measured window).

## Phase A - Post-C1 full-layer decomposition

Profiler extended (flag-gated, default off) with prefix stages 25-32 in
`engine/runtime_profiler.cu` + marks in `engine/decode_loop.cu` around the
existing per-stage calls and `run_final_hc_carry`; nsys kernel-class
cross-check of the s598 capture. Per rank per layer-step (replay 4.25 ms
flag-off / 4.55 ms profiler-on):

| Component | ms | Note |
|---|---:|---|
| prefix_attn_output | 0.723 | largest prefix item; kernel busy time for ALL attention classes is only ~0.05 ms -> wait/launch-bound |
| shared_swiglu_down | 0.793 | = 8x7xslots (1,792/layer) per-slot UVA remote-load copies inside `materialize_shared_swiglu_down_input`, 24/56 pair-directions crossing SYS (pre-existing since before s597) |
| ep_return_nccl | 0.614 | s598 floor |
| route_plan_pack | 0.517 | control-stream serialization |
| final_hc | 0.407 | |
| prefix_attn_projection / _state / hc_current | 0.262 / 0.250 / 0.246 | |
| barriers (954/978/1144/1373) | 0.298 | |
| gate_up + down GEMM | 0.203 | |
| typed_history + raw_read + compose + contrib | 0.083 | |

Cross-check: total GPU busy is 2.94 ms/rank/layer-step vs 4.25 ms replay -
the post-C1 layer is latency/wait-bound, not math-bound.

## Phase B - Candidates (every attempt recorded)

In-band control re-measured first: **rctl = 167.19 decode-domain /
112.70 wall**, slot-indexed tolerance 1.0/1.0 vs the s597 control, node
counts consistent with the committed s598 binary (2825/layer = 3017 minus
the 192 profiler stamps). Calibration: a flags-off 32-request probe matches
the control 32/32.

### C-A: swiglu_down exchange replacement - DROPPED (five variants, with the key finding)

| Variant | Mechanism | Perf | Tolerance | Verdict |
|---|---|---:|---|---|
| C-A1 `nccl` | pack + one grouped ncclAllGather + unpack | **197.17 (+17.9%)** | 0.781/0.935 FAIL | dropped |
| C-A2 +barrier | A1 + post-unpack cross-GPU barrier | - | 29/32 probe FAIL | dropped |
| C-A3 per-src broadcasts | s598-proven broadcast primitive | - | 14/32 probe FAIL (worse with more small collectives) | dropped |
| C-A2+`NCCL_PROTO=Simple` | protocol hammer | - | 0/32 (allreduce-order regime change; incomparable to control) | dropped |
| C-A4 `memcpy2d` | 56 strided P2P 2D-memcpy nodes (pure copies) | - | 26/32 probe FAIL | dropped |
| C-A5 `batched` | 56 strided UVA remote-load KERNELS (materialize mechanics, 1/32 the launches) | **186.33 (+11.5%)** | 32/32 at 8 tok, **0.922/0.954 at 64 tok FAIL** | dropped |

**Finding (load-bearing): the divergence is not in the exchanges.** C-A4/5
move identical bytes with pure copies - they cannot change values - and
C-A5 uses the exact remote-load kernel mechanics of the promoted path. Every
variant that makes swiglu_down *faster* eventually diverges at the reference
shape, while the slow 1,792-launch promoted path is clean. The promoted
layer therefore contains a **timing-dependent latent ordering hazard
downstream of the swiglu exchange that is currently masked by the copy
storm's own slowness**. This is a correctness debt in the promoted path, not
a defect introduced by the candidates; it blocks ~0.5-0.65 ms/layer of
measured, otherwise-bit-exact gain and is the lead Sprint 600 item
(suspects, in order: shared_op->d_out / d_x consumers on the dense stream
vs rank-stream producers under the 954/978 event-slot ordering;
attn/shared dense outputs reread by compose; the graph-order event-slot pool
under altered node timing).

### C-B: early EP return + per-rank ordering (replacing the 954/978 8x8 barriers) - CLEAN, NO GAIN, NOT PROMOTED

`DS4_V100_TP_EP_EP_RETURN_EARLY=1`: pack+NCCL return enqueued right after
the routed GEMMs (no 1144 barrier needed - same-stream + collective
semantics), 954/978 replaced with per-rank rank<->dense event edges.
Tolerance **1.0/1.0 PASS**; barriers measurably collapse (954: 0.075 ->
0.005 ms; 978: 0.152 -> 0.030 ms) but decode-domain is **168.07 (+0.5%,
run-band)** - the freed barrier time is reabsorbed by the serialized NCCL
return on the same stream. Correct but unbeneficial alone; kept as an
opt-in flag, **not promoted** (no measured benefit).

### Stack (C-A5 + C-B, evidence-only): anti-synergistic

168.94 decode-domain with profiler - C-B's reorder serializes the return
ahead of the now-fast swiglu on the same stream and destroys C-A5's gain.
Stage table (the demonstrated per-stage minima): swiglu_down 0.293,
barrier_954 0.005, barrier_978 0.030, ep_return 0.727, replay 4.11 ms,
nodes 1377/layer (copy storm gone).

### C-C (route-plan under the attention shadow): NOT ATTEMPTED

Budget went to the C-A tolerance forensics (five variants). The 0.52 ms
pool and the approach (split `run_true_ds4_post_attention_ffn_input` into
plan/pack phases; plan depends only on hc_current's router logits) are
recorded for 600.

## Phase C - Verdict (DoD 4)

**The ~220 tok/s stretch target was NOT reached.**

- Best clean (promotable) decode-domain: **167.19 tok/s** (the s598
  promoted configuration; nothing from this sprint passed both gates).
- Best demonstrated decode-domain: **186.33 tok/s** (C-A5, bit-exact-by-
  construction copies, blocked by the latent hazard); **197.17** with the
  NCCL allgather variant.

**Floor analysis (measured minima after the best attempts, ms/layer):**

| Stage | clean floor | demonstrated floor | structurally bound to |
|---|---:|---:|---|
| pre-EP prefix (attn chain + hc_current) | 1.54 | 1.54 (untouched) | launch/wait latency, NOT math (0.05 ms busy); needs launch fusion / stage overlap |
| swiglu_down exchange | 0.79 | **0.29** (C-A5) | the latent ordering hazard, not transport |
| ep_return_nccl | 0.61 | 0.61 | NCCL ring LL floor for 24 MB/layer |
| route_plan_pack | 0.52 | 0.52 (unattempted) | control-stream serialization; shadowable under attention |
| final_hc | 0.41 | 0.41 | hc expand chain |
| barriers | 0.30 | **0.04** (C-B) | replaceable by per-rank edges (proven clean) |
| expert GEMMs + compose + pack | 0.25 | 0.25 | leave alone |
| **layer replay** | **4.25** | **~3.5 achievable** = 4.25 - 0.50 - 0.26 | |

**Reachable/not-reachable conclusion: ~220 tok/s (3.13 ms/layer) is
REACHABLE within B2 scope, but it is gated on fixing the latent ordering
hazard.** The measured levers sum past the gap: swiglu fix (-0.50,
demonstrated bit-exact mechanics exist), barrier edges (-0.26, demonstrated
clean), route-plan shadowing (-0.3 to -0.5 of 0.52, unattempted), prefix
launch-latency compaction (pool 1.54, wait-bound). 4.25 - 0.5 - 0.26 - 0.35
= ~3.14 ms = ~219 tok/s. None of it is promotable until the hazard is
root-caused, because every faster-swiglu variant trips it. Outside B2
scope, the same hazard fix plus MTP-style batching remain the larger
levers. **Sprint 600 should lead with the hazard root-cause (it is both a
correctness debt in the promoted path and the single blocker on ~20-30
tok/s of already-demonstrated gain).**

## Definition of Done

1. Post-C1 decomposition table - **done** (Phase A; prefix split + nsys
   cross-check archived).
2. Each candidate: flag, tolerance, A/B, keep/drop - **done** (C-A x5
   variants all dropped with numbers; C-B clean/neutral not promoted;
   C-C not attempted, reason recorded).
3. Final stacked measurement - **done** (stack measured 168.94 and
   anti-synergistic; final configuration = s598 promoted, 167.19/112.70).
4. Stretch verdict - **done, explicit**: NOT reached; floor analysis with
   per-stage minima and a reachable-but-gated conclusion above.
5. Promotions - **none** (no candidate passed tolerance AND showed gain);
   launcher defaults unchanged (`nccl` transport, `copy` swiglu exchange,
   early-return off); all new flags retained as opt-in.
6. Report written; STATUS/steering/VISION - orchestrator's; commits -
   orchestrator's.

Gates: tolerance gate enforced on every candidate (slot-indexed vs the s597
control; the index-paired caveat from s597 still applies); flag-off
byte-identity re-proven on this binary (rctl 1.0/1.0 + node accounting);
no-SYS invariant unchanged (the promoted EP return remains NCCL-ring-only
per the s598 nsys proof; note honestly: the swiglu copy storm's small SYS
remote loads are pre-existing promoted-path behavior, unchanged by this
sprint, and quantified above).

## Deviations

- Phase A used the existing s597 profiler stamp machinery (decode_loop +
  runtime_profiler marks) rather than new marks in `hc_current.cu` /
  `layer_runner.cu` - the decode_loop boundaries were sufficient; those
  files were left untouched.
- The C-A tolerance probes used an 8-token prefix comparison against the
  64-token control (calibrated: flags-off probe = 32/32); final judgments
  always used full 64-token reference runs.
- `NCCL_PROTO=Simple` was tested once as a diagnostic; it changes allreduce
  ordering globally and is unusable under a bit-anchored control.
- The stack run doubles as the profiler/stage-minima run (tolerance-failing
  by construction since it contains C-A5; used for perf evidence only).
- No nsys run this sprint: the transport did not change on any promotable
  path (final config = s598 promoted; its no-SYS nsys proof stands).

## Artifacts

- Pod `/workspace/s599-artifacts/`: COMMANDS.md, pa-prefix-decomp/ (Phase A),
  rctl/ rca/ rcb/ rca5/ rstack/ (reference runs), probe-ca2/3/2s/4/5 +
  probe-off (tolerance forensics), logs.
- Laptop `logs/from-cluster/sprint599/`: COMMANDS.md, phaseA-decomposition,
  rstack-stage-table, all sustained_http TSVs.
- Source changes (uncommitted): `engine/decode_loop.cu`,
  `engine/runtime_options.cuh`, `engine/runtime_pack.cu` (three exchange
  variants + helpers), `engine/runtime_profiler.cu` (prefix stages),
  `tools/ds4-v100-run-tp-ep-appliance.sh` (flag plumbing);
  `docs/sprints/SPRINT-599-REPORT.md`.
