# DS4 V100 TP/EP Throughput Optimization — Isolated Activity Prompt

Date: 2026-05-25

## Why this exists

The TP/EP decode loop is **latency/launch-bound, not bandwidth-bound**. The
sprint-371 active-slot matrix (TEMP_STATUS_REPORT_083) shows ~10% avg GPU util
(40% max), **flat across 1-32 active slots**, ~98 aggregate decode tok/s. The
binding cost is per-layer collectives + many small per-kernel launches with
host-side device syncs. `tools/ds4-v100-tp-ep-full-layer-smoke.cu` currently has
~92 `cudaDeviceSynchronize`/`cudaStreamSynchronize` calls and **zero CUDA graphs**.

A working V100/SM70 reference for every technique below lives at
`research/1Cat-vLLM` (FULL CUDA-graph capture incl. collectives + paged attention,
a from-scratch SM70 FlashAttention-2, single batched paged KV, compact-MoE grouped
GEMM via lmdeploy TurboMind SM70 WMMA, MTP, fp8 KV).

## Isolation principle

Each trick becomes **its own CLI gate flag, default off, A/B'd same-binary against
the current default on gpu-01**, promoted only if it raises GPU util / tok/s **and**
leaves first-token/parity and the decode checksum unchanged. DS4-specific
simplification: because the runtime already pays a fixed 32-wide step cost
regardless of occupancy (the flat matrix), a **single** captured graph at 32-wide
covers every occupancy — no batch-size buckets needed to start.

---

## THE PROMPT

```text
ROLE
You are executing optimization sprints on the DS4 V100 TP8/EP8 DeepSeek-V4
appliance. Work in tools/ds4-v100-tp-ep-full-layer-smoke.cu and the TP runtime
(ds4_v100_tp_runtime.{cu,h}). Follow the existing sprint discipline exactly.

NON-NEGOTIABLE INVARIANTS
- Builds and A/B serving runs execute ON THE V100 POD (gpu-01), never the laptop.
  Build: make -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
- Each activity below is ISOLATED behind its own --<name>-gate CLI flag, default
  OFF, added in the std::strcmp(arg, "--...-gate") block near line 3288-3433.
- Every change is a same-binary A/B: current default vs the new gate, at the real
  shape (32 slots / 256K / 43 layers). Instruments:
    tools/ds4-v100-tp-ep-active-slot-matrix.py   (GPU util + decode tok/s vs occupancy)
    tools/ds4-v100-tp-ep-http-ab.py              (control vs candidate, same binary)
- PROMOTE a gate ONLY IF: GPU util OR server decode tok/s improves, AND the
  generated first-token / short_reasoning_plain parity token is UNCHANGED, AND the
  all-layer decode checksum is preserved (bit-compare). Otherwise keep it opt-in
  and record the reject, with numbers.
- One SPRINT-NNN.md + one TEMP_STATUS_REPORT_NNN.md per activity, with the cluster
  log path, the A/B table, and an explicit promote/reject decision. Do not batch
  multiple activities into one sprint.
- Read-only baseline first: capture current GPU util (~10% avg / ~40% max) and
  server decode tok/s (~98 aggregate) so every A/B has a fixed reference.

EXECUTE THESE ACTIVITIES IN ORDER. STOP after each for A/B + promote/reject.

S-B (enabler, do first) --async-output-gate
  Goal: remove the per-step HOST synchronization on the sampler/output path so the
  step becomes capture-eligible.
  Do: move selected-token D2H to a dedicated CUDA stream + event; sample on-device;
  only synchronize the copy_event when the token is actually consumed by the next
  step's embed-seed. Audit the 92 cudaStreamSynchronize/cudaDeviceSynchronize calls;
  relocate every one OUT of the steady-state region of the run_one_step lambda
  (~line 10614). Target: zero host syncs inside one decode step.
  Validate: active-slot-matrix; expect util up, tok/s flat-or-up, token UNCHANGED.

S-A (highest ROI) --decode-cudagraph-gate
  Goal: capture one decode step PER RANK into a CUDA graph and replay it; eliminate
  per-kernel + per-collective launch latency (the cause of ~10% util).
  Do: (1) make run_one_step shape-static at 32-wide (it already is). (2) Wrap each
  rank's per-step kernel sequence — attention, dense, EP grouped GEMM, AND the
  peer-copy compose — in cudaStreamBeginCapture/cudaStreamEndCapture, instantiate
  with cudaGraphInstantiate once, then cudaGraphLaunch each step. (3) Inputs/outputs
  go through PERSISTENT device buffers the graph captured; copy this step's seed
  into them, launch, read out — pass no pointers at replay. (4) Cross-rank deps
  (peer copies / event waits in compose) must be captured inside; verify the
  per-rank graphs compose correctly with a checksum-identical 50-step run.
  Risk: a kernel that branches on a host-read value (.item()-equivalent) or
  allocates mid-step breaks capture — find and remove these first.
  Validate: active-slot-matrix util MUST rise materially; checksum bit-identical;
  parity token UNCHANGED. This is the make-or-break test of the 10%-util thesis.

S-C --batched-paged-attn-gate   (extends --true-ds4-attention-typed-kv-batch-rows-gate)
  Goal: replace the per-slot, per-row-family typed-KV store/load launches with ONE
  block-table-indexed attention kernel per family (raw-SWA / compressed / indexer).
  Do: lay out KV as paged blocks; pass block_table + seq_lens once; gather paged KV
  INSIDE the matmul kernel (single-page/two-page fast paths). Grid (B*H, tiles) so
  ragged lengths are per-block loop bounds, not separate launches. Keep MLA head_dim
  512 by choosing BLOCK_N that fits the 96KB SM70 smem cap; fp32 online-softmax
  max/sum + fp32 O accumulator in smem; exp clamp -80; sum floor 1e-24.
  Validate: per-step attention kernel COUNT drops sharply; util up; checksum/parity held.

S-D --compact-moe-decode-gate
  Goal: kill expert rank-imbalance + idle-expert work at low batch.
  Do: at the active batch, sort only the top-6 routed expert IDs, gather their
  StridedPtr weight rows into a top-k-sized array, run ONE grouped GEMM over k
  one-row groups (expert_offsets = [0,1,..,k]); fused weighted-reduce of outputs.
  No scatter, no tile padding, no idle experts. Reuse the SM70 grouped-GEMM
  scheduler offsets path already in the TurboMind kernels.
  Validate: compose/EP ms/step down; util up; checksum/parity held.

S-E --fused-gated-silu-gate
  Goal: remove the separate clamp+SwiGLU kernel from sprint 308.
  Do: interleave gate+up columns at weight-prep; fuse SiLU(gate)*up into the routed
  grouped-GEMM epilogue WITH the 10.0 clamp baked in (the reason the unclamped
  TurboMind epilogue was rejected). One fewer kernel + one fewer intermediate/layer.
  Validate: routed-FFN ms/step down; checksum/parity held.

S-F (experiment, not a commitment) --tp-experts-ab-gate
  Goal: test whether TP-sharded experts eliminate the per-layer EP all-to-all that
  has dominated compose since sprint 269.
  Do: add a TP-sharded expert path (intermediate-dim split, reduced via the SAME
  hidden all-reduce), behind the gate, for a 13B-active/8-GPU shape. A/B against the
  current EP8 all-to-all path. This is a measurement to inform topology, not a rip-out.
  Validate: report compose/all-to-all ms/step and total decode tok/s for both; decide.

S-G --fp8-e5m2-kv-gate
  Goal: halve KV bandwidth/footprint at 256K.
  Do: store compressed/raw KV as fp8_e5m2 (uint8); dequant by branchless bit-shift
  (__ushort_as_half(raw<<8)); fold k_scale into the softmax scale and v_scale into
  output normalization (scalar, not per-element). Reference quality bar: operator
  max_abs <= ~9.77e-4, 256K needle stable, zero NaN.
  Validate: KV bytes/step down; parity token UNCHANGED within tolerance; no NaN.

S-H (after S-A lands) --mtp-decode-gate
  Goal: 3-4x decode multiplier via multi-token prediction.
  Do: wire the existing ds4_v100_mtp.{c,h} draft head into the TP/EP serving loop;
  run num_speculative_tokens-1 extra draft passes, rejection-sample against the
  target. CAPTURE the MTP decode batch shapes in the S-A graph or acceptance
  collapses to 1.5-2. Use tools/ds4-v100-mtp-acceptance-matrix.sh to report
  acceptance length across 1/8/16/32 active slots.
  Validate: accepted-tokens/step and effective decode tok/s up; parity preserved.

DELIVERABLE PER ACTIVITY: SPRINT-NNN.md + TEMP_STATUS_REPORT_NNN.md with the A/B
table (util, server/decode tok/s, ms/step by stage, first-token, checksum) and an
explicit PROMOTE or REJECT line with numbers. Update docs/sprints/VISION.md.
```

---

## How to add each to the current implementation

| Activity | Where it plugs in | Concrete integration |
|---|---|---|
| **S-B async output** | the 92 sync sites (`:685, 3718, 3898…`) + `run_one_step` (`:10614`) | Add an `output_copy_stream` + `cudaEvent_t` per rank; sample on device; defer the token D2H. The blocker for everything else — a single `cudaDeviceSynchronize` inside the step poisons graph capture. |
| **S-A CUDA graph** | wrap the body of `run_one_step` (`:10614`); per-rank `cudaStreamBeginCapture`→`cudaGraphInstantiate`→`cudaGraphLaunch` | Greenfield (0 graph calls today). Static 32-wide step + persistent I/O buffers. The peer-copy compose (`ComposeStats`, `:356`) must be captured inside — it's capturable. Verify checksum bit-identical over 50 steps. This is the 10%-util test. |
| **S-C batched paged attn** | extend the existing `--true-ds4-attention-typed-kv-batch-rows-gate` (`:3288`) | Replace per-slot/per-family typed-row store/load with one block-table kernel/family. Reuse the `kBoundedCompRows`/loaded-position bookkeeping (`:217-230`) as the block table. Extend the attention traits to head_dim 512 (smaller BLOCK_N for 96KB smem). |
| **S-D compact MoE** | new gate; routed-expert dispatch path | Top-k-only sort + StridedPtr gather + one grouped GEMM over k 1-row groups; reuse the SM70 grouped-GEMM `offsets`/`StridedPtr` scheduler already linked. Removes the all-to-all at low batch. |
| **S-E fused gated-SiLU** | the routed grouped-GEMM call + the sprint-308 clamp kernel | Interleave gate+up at weight-prep; bake the `kRoutedSwigluClamp=10.0` (`:73`) into the GEMM epilogue. Deletes the separate clamp+SwiGLU launch. |
| **S-F TP-experts A/B** | new gate; alternative to EP8 dispatch+compose | Intermediate-dim split, reduce via the existing hidden all-reduce; A/B compose ms/step vs EP8. Pure measurement to settle the topology question. |
| **S-G fp8 KV** | typed-KV store/load + attention load | Store uint8 e5m2; bit-shift dequant; scales folded into scalar softmax/output mults. Quality bar from the fork's audit. |
| **S-H MTP** | `ds4_v100_mtp.{c,h}` ↔ serving loop; `mtp-acceptance-matrix.sh` | Existing draft infra; wire into the per-step loop; must run after S-A so MTP shapes are graph-captured. |

**Sequencing logic:** S-B unblocks S-A (sync removal is a prerequisite for capture);
S-A is the big util win and must land before S-H (MTP needs its shapes graphed);
S-C/D/E are independent kernel-count reducers that also shrink the captured graph;
S-F is a one-off topology experiment; S-G is independent. Start with **S-B → S-A and
stop** — if S-A doesn't move util materially, that single result reshapes the whole
throughput outlook and is worth knowing before building the rest.

## Caveat to carry

The `research/1Cat-vLLM` fork validated correctness but **not** greedy-token parity
at 256K (fp16 drift can flip argmax). As S-G (fp8 KV) and any fp16 paged-attention
paths land, keep them A/B'd against DS4's exact-parity gate rather than assuming the
fork's tolerance transfers. Line numbers above are point-in-time
(`tools/ds4-v100-tp-ep-full-layer-smoke.cu`); re-grep before editing.
