# Spike B Plan — Assessment (2026-05-26)

Reviewer: Spike A side. Scope: the revised 10-point Spike B plan (graph/NCCL
capture-first), checked against the latest executed work (sprints 410–414).

## Verdict: the PLAN is strong, and EXECUTION has now turned toward it.

The plan correctly targets the real lever and encodes the right discipline.
Sprints 412–414 had drifted (per-transfer NCCL + slot/stat tuning on the
ungraphed path — the #4/#6/#9 trap). The **latest in-flight work (uncommitted)
has pivoted to the right thing**: capture-eligibility prep in the TP runtime.
Remaining step is the actual capture+replay and a slots=4 probe — not yet done.

## UPDATE — what changed since this assessment (in-flight, uncommitted)

Working-tree changes in `ds4_v100_tp_runtime.{cu,h}` and the smoke file (no new
commit yet) are building capture eligibility under `--decode-cudagraph-gate`:
- `graph_event_order` — broad host `sync_all` waits replaced by cross-stream
  CUDA **event joins** (`capture_join_events`: `cudaEventRecord` +
  `cudaStreamWaitEvent(root_stream, …)`).
- `capture_probe_active` flag + a `CaptureStream` abstraction — scaffolding for
  an actual capture probe.
- `if (!decode_cudagraph_gate && …)` guards skipping host-sync paths when the
  gate is on; compose copy handled under the gate.

**This is the correct direction — it directly addresses the drift critique.** Two
caveats:
1. It is still **prep, not capture**: I see no `cudaStreamBeginCapture` /
   `cudaGraphInstantiate` / `cudaGraphLaunch` and no slots=4 probe result yet.
   This is re-doing the sprint-376 event-ordering inside the TP runtime.
2. **Do not judge this by throughput.** Sprint 376 already showed event-ordering
   alone *lowers* throughput (enqueue overhead) with no payoff until capture+
   replay lands (the plan's #9). Expect the same here — the win only appears
   after `cudaGraphLaunch` works.

Most imminent gap to get right *before* the first `BeginCapture`: **#1 NCCL
warmup-before-capture** (below) — it will be the first failure the moment capture
is attempted with NCCL collectives live.

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
