# SPIKE B — C Capture-First: make DS4's pure-C TP/EP runtime graph-capturable

Date: 2026-05-26
This is **one of two parallel feasibility spikes**. Spike A (separate agent,
`research/1Cat-vLLM` tree) tests porting DS4's attention into the vLLM fork. You run
Spike B only. The two results feed a single architecture decision — produce evidence,
not a product.

## ROLE

You are a feasibility-spike agent. Determine whether DS4's pure-C TP/EP runtime can
be made **capture-first** — one 32-wide decode step replayed as a CUDA graph — so GPU
utilization rises off the ~4-8% latency-bound floor, AND whether that fits in 32 GB
VRAM. **Goal is a DECISION, not a finished runtime.** Stop at the first definitive
go/no-go signal. Work ONLY in the DS4 C tree
(`tools/ds4-v100-tp-ep-full-layer-smoke.cu`, `ds4_v100_tp_runtime.{cu,h}`); do not
touch `research/`.

## CONTEXT

- DS4 is a from-scratch DeepSeek-V4-Flash TP8/EP8 appliance for 8×V100-32GB. Decode is
  latency-bound: ~4-8% GPU util, flat across 1-32 active slots, ~88-108 server decode
  tok/s aggregate.
- Sprint 376 **rejected CUDA-graph replay** because the decode step uses
  stream-capture-incompatible `cudaMemcpyPeerAsync` transport.
- NCCL migration is underway: the collective workbench
  (`tools/ds4-v100-tp8-collective-workbench.cu`) already uses
  `ncclCommInitAll`/`ncclAllReduce`/`ncclAllGather`; HC-current NCCL allgather was
  promoted (sprint 410, `DS4_V100_TP_EP_HC_CURRENT_INPUT_NCCL_ALLGATHER=1`, +5.7%
  server decode) but util did NOT rise — because no graph has been captured yet.
- Sprint 411 surfaced a **VRAM wall**: NCCL + the correct (post-attention) path
  exceeds 32 GB (62 VRAM failures, readiness blocked, min free ~1.3 GB).
- The question this spike answers: can the remaining transport be made capturable,
  does graph replay actually lift util, and does the result FIT in VRAM.

## SHARED PARITY CURRENCY (identical to Spike A)

Validate against the SAME DS4 layer-2 oracle used by the existing
`tp_ep_compressed_reference_diff` gate at the **ratio-4 emit position**. Because NCCL
breaks bit-exact reduction order, the gate here is: **first token UNCHANGED** AND
next-hidden `max_abs <= 1e-4` vs the peer-copy path. (Bit-identical checksum is not
required for the NCCL/graph path.)

## INVARIANTS

- Build + validate on gpu-01 (V100 pod), never the laptop:
  `make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke`
- All new behavior behind a CLI gate, default OFF, in the
  `std::strcmp(arg,"--...-gate")` block (~line 3408). Reuse/extend the existing
  `--decode-cudagraph-gate` and NCCL gates.
- Add `-Xptxas -v` to the build so registers/smem/spill is visible (it is not today).
- Time-box. Stop at the first definitive signal. Do not build the full runtime.
- Isolation: DS4 C tree only.

## STEPS

B1. FINISH CAPTURABLE TRANSPORT. Convert the remaining hot-path `cudaMemcpyPeerAsync`
    cross-rank transfers to NCCL collectives (or capturable kernel-copy over
    P2P-mapped memory). Sites to cover (re-grep before editing — line numbers drift):
    - attention-input / HC gathers: ~`:4883, :5052, :5191, :5236, :5920, :7205`
    - output-head shard gathers: ~`:6299, :6460`
    - EP expert-output / next-hidden reduce: `compose_next_hidden_*sum*` kernels
    Map by semantics: shard->full gather = `ncclAllGather`; expert-output reduce =
    `ncclAllReduce(sum)`. **WARM UP every collective ONCE before capture** (NCCL
    lazy connection/buffer alloc breaks capture otherwise).

B2. ATTEMPT CAPTURE. `cudaStreamBeginCapture` one 32-wide decode step PER RANK ->
    `cudaGraphInstantiate` once -> `cudaGraphLaunch` each step. Inputs/outputs through
    persistent device buffers; pass no pointers at replay. Carry forward the sprint-376
    helper event-ordering passes so `helper_host_sync_blocker_classes` reaches 0.

B3. MEASURE vs the ~4-8% util baseline (use `tools/ds4-v100-tp-ep-active-slot-matrix.py`
    and `tools/ds4-v100-tp-ep-nccl-http-ab.py`):
    - `capture_eligible` flips to 1?
    - GPU util after replay (THE test — does it leave the ~4-8% floor)?
    - server/client decode tok/s.

B4. VRAM VERDICT (the retrofit's real gate). Do NCCL buffers + graph buffers
    (persistent I/O + NCCL graph pool) + the step working set fit at 32 slots / 256K?
    Report min free VRAM + VRAM failures. Sprint 411 already shows the correct path +
    NCCL barely fits; prove graph buffers don't push it over.

B5. PARITY. First token unchanged + next-hidden within 1e-4 vs peer-copy (shared
    metric above).

## DELIVERABLE -> /Users/ravi/repos/ds4/SPIKE_B_RESULT.md

Structured fields (so it can be compared against Spike A):
- `capture_eligible`: yes/no + remaining blocker if no
- `replay_gpu_util`: % after replay vs ~4-8% baseline (the headline number)
- `replay_decode_tok_s`: server + client, vs baseline
- `vram_fit_32slot_256k`: yes/no, min free MiB, VRAM failures
- `kernel_spill_smem`: regs/thread, smem/block, spill bytes for the heavy attention
  kernels (head_dim-512 MLA) from `-Xptxas -v` / ncu
- `parity`: first token match? next-hidden max_abs vs peer-copy
- `effort_to_full`: remaining transport/stages to convert, T-shirt size
- `verdict`: RETROFIT-VIABLE / RETROFIT-BLOCKED + the one-line reason

## WHAT MAKES THIS PATH WIN (context for your investigation)

This path is recommended over porting into vLLM when: graph replay lifts util
materially (>= ~2-3x), AND it fits in 32 GB VRAM, AND parity holds — because it
preserves the C investment, stays pure-C (no Python/Torch), and keeps the strict
parity discipline. If util fails to rise after capture, OR VRAM can't hold NCCL+graph
buffers at 32 slots / 256K, this path is BLOCKED and the decision tilts to the port.
Optimize your investigation toward those decision-relevant facts; don't gold-plate.
NOTE: if util rises only to ~50% and plateaus, check the head_dim-512 MLA kernels for
register spill / smem-occupancy limits (B5) — that is the likely next ceiling.
