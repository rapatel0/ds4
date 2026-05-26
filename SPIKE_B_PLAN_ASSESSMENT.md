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
efficiency. This is the secondary wall I flagged earlier: the heavy
attention-output projection (~486 ms in sprint 412), collective time, and the
head_dim-512 occupancy/spill question. **Next decisive measurement: Nsight/ncu on
replay mode** to find the in-graph bottleneck, then optimize it. Also still open:
the parity gate (probe used `skip_decode_checksum`), and VRAM headroom at useful
slot counts.

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
