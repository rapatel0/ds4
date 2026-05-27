# Spike B Plan — Assessment (2026-05-26)

Reviewer: Spike A side. Scope: the revised 10-point Spike B plan (graph/NCCL
capture-first), checked against the latest executed work (sprints 410–414).

## Verdict: the PLAN is strong, and the make-or-break test has now PASSED.

The plan correctly targets the real lever. After a drift (412–414) it turned to
the graph path, and **sprint 415 proved the thesis: the correct semantic
all-layer TP/EP decode step CAPTURES and REPLAYS on V100.** This is the most
important positive result in the whole throughput arc — the dominant unknown
("is this dynamic MoE+MLA+HC+NCCL path even CUDA-graph-launchable on SM70?") is
resolved YES. The win is now an *engineering* path (persistent graph exec +
multi-step replay), not an open question. Two things remain before it's a banked
throughput gain: the persistent-exec architecture, and a parity gate.

## UPDATE — sprint 415: capture + replay SUCCEEDED (the milestone)

`--decode-cudagraph-replay-probe-gate` captures one decode step, instantiates,
and launches it, using graph output as the step result. On gpu-01 (sm_70), full
correct semantic path (true attention output + post-attn FFN + NCCL HC-current +
NCCL attn-output + compressed KV + compact MoE):

| Slots | Capture | Replay | sum_replay_ms | Proj. slot tok/s | Graph nodes |
|---:|---:|---:|---:|---:|---:|
| 4 | 43/43 | 43/43 | 94.3 | 42.4 | 46,134 |
| 8 | 43/43 | 43/43 | 128.4 | 62.3 | 57,758 |

What this proves and what it doesn't:
- ✅ **Launchable**: the whole correct path (incl. NCCL collectives + compressed
  KV + MoE) records into a graph and replays. NCCL-in-graph worked one-shot (my
  gap #1 didn't bite the one-shot case — but see multi-step).
- ⚠️ **Not yet a clean throughput number**: it captures+instantiates *every layer
  per invocation* — no persistent graph-exec cache, no multi-token loop, no
  serving-loop integration. The 42–62 proj. slot tok/s is preliminary; the report
  rightly says launch barrier is removed and the path is now
  kernel/collective-bound (the goal), but the util-lift payoff is unmeasured.
- ⚠️ **Multi-step replay FAILED** (the expected next hazard): recapturing every
  token step blew up at step 3 / layer 2 —
  `store_f32_device_to_f8_kv_rows_kernel: operation would make the legacy stream
  depend on a capturing blocking stream` (compressed-KV path). Correct
  conclusion in the report: **cache `cudaGraphExec_t` per layer and replay across
  steps with persistent device buffers** — exactly plan #8 / my gap #4. Do not
  re-enter capture in the token loop.
- ⚠️ **Parity not gated**: probe uses `skip_decode_checksum=1` → launchability/
  perf evidence, not the parity gate (my gap #5 still open; next task).
- A stale `capture_eligible=0` heuristic was misreporting despite success; patched.

Net: the risk has shifted from "can it be graphed?" (answered yes) to "build the
persistent-exec serving loop, confirm the util/tok-s lift, and re-attach parity."
They also distilled prior lessons into `TEMP_GRAPH_PRIOR_INSIGHTS.md` (no
steady-state recapture, no pointer drift, fixed device metadata buffers) — the
right guardrails.

### sprint 416 — persistent multi-step replay WORKS (architecture milestone)

`--decode-cudagraph-persistent-replay-gate` + a per-layer `cudaGraphExec_t` cache
(owned by shared rank buffers). gpu-01 validation:
`capture 43/43, replay 172/172` (43 layers × 4 steps), `capture_eligible=1`,
`blocker=none`, all `cudaSuccess`. **This fixes the 415 multi-step hazard** by
caching the graph exec and replaying across token steps instead of re-entering
capture — exactly plan #8 / my gap #4. The capture+replay architecture risk is
now fully retired (one-shot AND persistent multi-step both proven).

Benchmark (8-slot, 4-token, 256K): `decode 80.73 tok/s`, `continuation 88.53`,
`wall 20.48`. **Still not a throughput win** (agent's own words), and that is the
honest, important point: the graph path is operational but does not yet beat the
eager path measurably. Wall is low partly because capture/instantiate is amortized
over only 4 steps; the decode/continuation figures are the steady-state proxy.

**The risk has now shifted to its final form:** with launch overhead removed, the
binding question is *what's inside the replayed graph* — kernel/collective
efficiency. Next decisive measurement: the clean eager-vs-graph A/B + profiling.

### sprint 417–418 — the throughput win is MEASURED (the payoff landed)

**417, clean A/B (8 slots, 256K, 8 steps, correct semantic path):**
- eager: **37.6** generated decode tok/s (breakdown: hc-current 1359 ms, compressed-KV
  547 ms, attn-projection 271 ms — the launch/copy-heavy prefix).
- persistent graph: **85.3** tok/s → **2.27× over eager.** The "not a win yet" from
  416 is superseded: with enough steps to amortize capture, graphs deliver a real,
  measured speedup on the *correct* path.
- + deferred-NCCL HC-current (memory fix so NCCL allocs after expert residency):
  **89.6** tok/s (+5%).
- **16 slots: 116.85 generated decode tok/s — current best**, and it *exceeds* the
  old fast-but-less-correct path (~100 @ 32 slots). The correct path with graphs is
  now faster than the old fast path was.
- **32 slots: OOM** — all-resident experts + dense-F16 cache + KV don't fit; needs a
  memory-layout change, not just smaller scratch.

**418, peer-copy-in-graph rejected (confirms the constraint):** routing graph-mode
copies through `cudaMemcpyPeerAsync` is rejected by capture ("operation not
permitted when stream is capturing"). The graph-safe remote-read copy kernels are
"one of the few graph-capturable ways to move those remote values" — so they are
now a real *in-graph* replay cost. Correct next direction (not peer-copy): **reduce
the captured copies** — keep tensors rank-local instead of gather-to-device-0-and-
redistribute, NCCL where capture-safe, fused graph-safe kernels on rank-major
layouts, drop redundant full-hidden materialization.

**This validates the whole thesis end-to-end: capturable → operational → 2.27×
faster.** The binding constraints are now (a) **VRAM/memory-layout to reach 32
slots** (16 is the current fit point), (b) **in-graph copy reduction** for more
speed, (c) **parity re-attach** (still `skip_decode_checksum`), (d) serving-loop
integration. All tractable engineering, not open questions.

### sprint 419–427 — rank-major conversion executing (the per-slot lever), +1 open thread

The team is working through the gather-to-device-0 sites exactly as scoped:
- **420–421: rank-local attention-projection input → measured +13% decode**
  (88.4 → 100.1 tok/s at 8 slots, client +9%). First concrete win from killing a
  gather-to-0. Direction validated.
- **422–426**: extended to rank-major FFN shared+routed inputs (consume the NCCL
  all-gather'd rank-major hidden directly) and **rank-major router logits**
  (compute local expert logits per-GPU, removing another gather). Split scratch
  buffers (`d_*_full_rank_major`, ~2 MiB/GPU at 32 slots).
- **425–427: a correctness divergence appeared under the persistent-graph +
  async-route-plan regime.** Good discipline: they split the FFN gate into
  shared/route diagnostics and added a **direct `__half` buffer parity audit**
  (427) — which proved the rank-major half-input *values* are **byte-identical**
  to legacy in the sync/eager regime (0 mismatches, checksum matches). So the
  divergence is isolated to the **graph/async-route-plan path, not the values**.
  Sprint 428 chases the graph/async route-metadata behavior.

Reads as: the per-slot lever is delivering (the +13% is the first installment of
the projected per-slot-term reduction), being extended methodically, with one
**open correctness thread in the graph regime** — the exact "graph capture
silently changes behavior" risk (my gap #5). Until it's resolved, the rank-major
FFN/router wins can't be promoted *under graph replay* (where the 2.27× lives).

**Still not done:** the kernel **spill / software-pipelining sanity check**
(`-Xptxas -v` → registers/smem/spill; ncu → occupancy/stall-reasons) on the new
shapes — flagged as the pre-MTP step, not yet picked up. The build still has no
`-Xptxas -v` and no `__launch_bounds__`, so kernel spill on the changed shapes
(head_dim-512 attn, rank-major consume kernels, HC) remains unverified.

### sprint 428–451 — rank-major wins are SMALL at the served shape (caution)

The graph-regime divergence was worked through (451 response parity 16/16 clean;
discipline held — 446 rejected a 1.056x candidate for parity 0/8). But two things
temper the optimism:

1. **Wins shrink at the correct served shape.** The promotable, correctness-clean
   bundle (router + FFN rank-major) at **16 slots HTTP**: server decode
   **27.2 → 28.1 tok/s (+3.4%)**, util 10.6 → 11.8%, parity clean (451). The
   bigger wins seen earlier (444: ~2x, 41 tok/s, 27%/63% util) were at *reduced/
   non-target* shapes, and attention-rank-local **cancels** the rank-major win at
   the reduced shape (449), so it's held out. Net at target: small.
2. **Served ≪ decode-benchmark, and util stays low.** Served HTTP at 16 slots is
   ~28 tok/s with ~11% util — vs the 116 tok/s decode-only benchmark at 16 slots
   (sprint 417, graph replay). That ~4x gap needs reconciling: HTTP/output-head
   orchestration overhead, very short generation (4 tokens, setup not amortized),
   and/or **the 2.27x graph-replay win is not yet integrated into the HTTP serving
   path** (the served runs don't cite graph replay; 417 did). If so, the headline
   graph win still lives only in the standalone smoke, not in serving.

So the lever is real in isolation (+13% on attn-proj, 444's 2x at reduced shape)
but is **not yet translating to served throughput** at the target shape, and util
at the served target shape is still ~11%. The decisive open items: **(a) integrate
graph replay into the HTTP serving path and measure served throughput with longer
generation**; (b) the spill/pipelining check (still not done); (c) reach 28/32
slots. The capability is proven; converting it to a served number is the gap.

### sprint 452–463 — the reckoning: the graph win does NOT transfer to serving

The init-time confound was addressed (463: **parallel expert load** — fans the
per-GPU expert-pack loads out across all 8 GPUs instead of GPU0→GPU7 round-robin;
a *startup* fix that explicitly did **not** move decode). With cleaner metrics,
the honest served picture emerged — and it's sobering:

- **Persistent graph replay is NOT promotable in the HTTP serving path** (459).
  The usable served baseline is **graph-OFF** rank-major/NCCL.
- **Graph mode in serving is broken, not just unhelpful**: graph event-order
  (no-replay) runs at **~half speed AND fails response parity** (changes the first
  token) across 460–462 (20 → ~9.4 tok/s). They fixed real event-order dependency
  holes (461) but did not repair it.
- So the headline **2.27× graph win (sprint 417) is stranded in the standalone
  smoke — it does not exist in the path that serves users.** I over-credited that
  benchmark number in prior updates; the serving reality is graph-off.

**De-confounded served numbers (graph-off, the real deliverable):**
- 8 slots: ~20 tok/s server decode, ~11% util.
- **32 slots / 32 tokens / 256K: ~35.8 tok/s aggregate decode, ~12.5% avg util
  (32% max), 32/32 responses, fits VRAM** (max used 30.7 GiB). So 32 slots now
  *fits and serves* — the VRAM constraint is largely addressed — but per-slot is
  ~1.1 tok/s and util stays ~12%.
- Bottleneck (de-confounded request window): **EP/routed FFN 52% + HC-current
  staging 43%** — attention is now minor.

**Net:** the launch-bound problem PERSISTS in serving because graph replay won't
promote there. The central lever is stranded in the benchmark, and 4 sprints of
graph-in-serving attempts failed on correctness+speed. The throughput estimate
must come down: the served path is ~12% util / ~36 tok/s at 32 slots, and the
proven way to fix it (graphs) is blocked in the loop that matters.

## What the plan gets right (keep)

- **No re-baselining** (#1, #3) — the ~97–108 tok/s / low-util numbers are trusted.
- **No micro-opt before capture** (#6) — names the exact trap that ate S-C/S-D/S-E
  (compact MoE, fused gated-SiLU: +0.6%, util flat).
- **NCCL framed as a capture-enabler, not a throughput win** (#7) — and cites the
  right evidence (HC-current NCCL was small *because still ungraphed*).
- **Dynamic-shape problem named + correct solution** (#8) — fixed-shape capture +
  persistent device buffers updated between replays. This is the hardest concept
  and the plan has it.
- **Tight probe discipline** (#4) — slots=4/8, no 1-slot except repro, no matrix
  until capture works.
- **Sound, measurable decision gate** (#10).

## Where execution had drifted (412–414 — now being corrected by the in-flight work)

Sprints 412–414, measured against the plan:
- **412**: attention-output NCCL evaluated on the heavy *semantic* path → 21 tok/s
  vs 101 fast control, 62 VRAM failures at 32 slots. This evaluates an NCCL
  conversion by its throughput on an **ungraphed** path — the #7 misframing the
  plan warns against. No capture attempt.
- **413**: slot reduction (32→30/28) to fit VRAM. Useful operationally, but it is
  a 32-slot matrix-style sweep, not a graph probe (#4 says slots=4/8 + capture).
- **414**: removed diagnostic stat host-syncs from timed sections. Good hygiene
  (those stat syncs are also capture-hostile), but still not a capture attempt.

Net: three more sprints of measure/tune on the ungraphed path. Per the plan's own
#9 ("only measure throughput after capture eligibility improves"), the next action
should be: **slots=4 + graph gate + current NCCL gates → inspect the exact capture
blocker → patch it.** Stop measuring throughput on the ungraphed semantic path.

## Five gaps to close (additions to the plan)

1. **NCCL warmup-before-capture (likely your first "mysterious" blocker).** NCCL
   does lazy connection + buffer alloc on first call; if that lands inside
   `cudaStreamBeginCapture`, capture fails. Warm up every collective once at
   startup, outside capture.
2. **Budget VRAM for the graph pool — it's already the binding constraint.**
   Sprint 413 shows the semantic path clears the 1536 MiB NCCL reserve only at
   ≤30 slots (30 by just 20 MiB). Graph capture adds a memory pool + persistent
   I/O buffers on top. So: (a) test capture at slots=4/8 where it fits (per #4),
   and (b) treat a slots=4 capture win as NOT proving the 32-slot target — 32 may
   need <32 slots or aggressive reclaim. Make this explicit in the gate.
3. **Verify route offsets/counts + indexer top-k are DEVICE-resident.** The #8
   solution hinges on the EP grouped-GEMM reading route counts/offsets from device
   buffers (sprint 308 already allocates worst-case `slots×top_k` — good) and the
   indexer top-512 gather being fixed-shape. A single host `.item()` read of a
   route count breaks capture. Confirm before assuming peer-copies are the only
   blockers.
4. **Define "small finite remaining blocker" = piecewise capture.** If one op
   can't be made capturable, graph the rest and run that op eager between two graph
   replays (vLLM's PIECEWISE mode). Names the fallback so a single stubborn site
   doesn't stall the whole spike.
5. **Keep parity in the loop.** NCCL changes reduction order (breaks bit-exact) and
   a capture bug can silently corrupt output. Gate each probe on first-token
   unchanged + next-hidden within tolerance vs the pre-NCCL path.

## The correctness/throughput/VRAM coupling (don't capture the wrong config)

The "fast control" (~101 tok/s) is the *less-correct* path (no post-attention FFN
input / true attention output). The *correct* semantic path is ~5× slower (21
tok/s) AND VRAM-tight (≤30 slots). Graph capture must target the **correct**
path, or the throughput win lands on a configuration that isn't model-faithful.
Confirm which path the capture probe runs.

## Spike B is now the active throughput path (context)

Spike A (the vLLM port) is paused (multi-week; HC residual stream + custom kernels
+ KV-paging integration). B's advantage: the model already works and parity is
close, so B only has to make the *runtime* graph-capturable — a smaller, higher-
certainty bet for lifting util/tok-s. The plan's discipline is exactly what avoids
another 40-sprint grind. Highest-value immediate move, with gaps #1 and #3 folded
in up front: **slots=4, graph gate on, inspect the first capture blocker.**
