# SPIKE A — Port Feasibility: DS4 DeepSeek-V4 attention inside the vLLM fork

Date: 2026-05-26
This is **one of two parallel feasibility spikes**. Spike B (separate agent, DS4 C
tree) tests the C capture-first retrofit. You run Spike A only. The two results feed
a single architecture decision — produce evidence, not a product.

## ROLE

You are a feasibility-spike agent. Determine whether DS4's DeepSeek-V4 attention can
be **hosted inside the research/1Cat-vLLM fork's runtime**, inheriting its CUDA
graphs / paged KV / NCCL / scheduler / MTP, while reproducing DS4's reference output
within tolerance on V100. **Goal is a DECISION, not a finished model.** Stop at the
first definitive go/no-go signal. Work ONLY in `research/1Cat-vLLM`; do not touch the
DS4 C tree.

## CONTEXT

- DS4 (`/Users/ravi/repos/ds4`, C/CUDA) is a from-scratch DeepSeek-V4-Flash TP8/EP8
  appliance for 8×V100-32GB. Its decode loop is latency-bound (~4-8% GPU util, no
  CUDA graphs), and it is hitting a 32 GB VRAM wall. Its differentiated asset is the
  validated DeepSeek-V4 attention: **MLA (head_dim 512, 64 heads) + sparse "indexer"
  (64 heads, top-512) + compressed-KV (ratio-4 even layers / ratio-128 odd) + raw
  sliding-window (128)**, at 256K context. MoE = 256 experts top-6, MXFP4; dense/attn
  FP8.
- `research/1Cat-vLLM` is a V100/SM70 vLLM fork: FULL CUDA-graph capture (incl.
  collectives), PagedAttention, FlashAttention-2 for SM70 (`FLASH_ATTN_V100`), NCCL,
  MTP, fp8_e5m2 KV, AWQ via lmdeploy TurboMind SM70 WMMA. It serves Qwen MoE today;
  it does NOT have DS4's V4 attention.
- The question this spike answers: can DS4's attention be expressed as a vLLM SM70
  custom attention backend that (a) reproduces the DS4 oracle within tolerance and
  (b) inherits the graphed, high-util runtime — and how big is that port.

## SHARED PARITY CURRENCY (identical to Spike B)

Diff your layer-2 attention output against the SAME DS4 oracle tensors used by DS4's
existing `tp_ep_compressed_reference_diff` gate at the **ratio-4 emit position**.
Target `max_abs <= 1e-5`; accept `<= 1e-4`. This is the apples-to-apples metric
shared with Spike B.

## INVARIANTS

- Build + validate on gpu-01 (V100 pod). Fork build: CUDA 12.8, the two-wheel
  install path per the fork README (`flash_attn_v100` + `vllm`).
- Add `-Xptxas -v` to any custom-kernel build so registers/smem/spill are visible —
  this also answers the head_dim-512 occupancy question.
- Time-box. Stop at the first definitive signal. Do not build the full model.
- Isolation: all work inside `research/1Cat-vLLM`.

## STEPS

A0. RECON FIRST (cheapest, decides scope). Does this vLLM lineage already have
    DeepSeek **MLA** support? Grep `vllm/` for: `deepseek`, `mla`, `MLACommon`,
    `MLAAttention`, `kv_lora`, `q_lora`.
    - If YES: the port = extend an existing DeepSeek/MLA model with DS4-V4's sparse
      indexer(512) + compressed-KV(ratio-4/128) + raw-SWA + an SM70 backend.
    - If NO: MLA is from-scratch.
    Report which — this single fact dominates port effort.

A1. Stand up ONE layer (layer 2) of DS4 attention — MLA head_dim 512 + sparse
    indexer top-512 + compressed-KV ratio-4 + raw-SWA 128 — as a vLLM custom
    attention op/backend, modeled on
    `vllm/v1/attention/backends/flash_attn_v100.py`. Feed it fixture inputs matching
    the DS4 oracle; load layer-2 FP8 attention weights directly from the DS4 pack
    (read-only; no full AWQ-loader integration required for the spike).

A2. HEAD_DIM-512 VERDICT. The fork's FA-SM70 dispatch tops out at head_dim 256. Does
    512 fit the 96 KB SM70 smem (union-aliased tiles, smaller BLOCK_N) or need a new
    tile/kernel? Report `-Xptxas -v` reg/smem/spill + achieved occupancy from ncu.
    NOTE: if MLA decode uses absorbed projections, the effective attended width may be
    far smaller than 512 (latent space) — determine this; it may make the kernel
    trivial and the verdict trivially YES.

A3. PARITY. Diff the backend's layer-2 output vs the DS4 oracle (see shared metric).
    Report `max_abs`.

A4. RUNTIME-INHERIT CHECK. Run the layer through the fork's CUDA-graph capture path;
    confirm it captures cleanly and report launch/util behavior. This is the payoff
    question: does DS4 attention inherit graphed, high-util execution for free?

## DELIVERABLE -> /Users/ravi/repos/ds4/SPIKE_A_RESULT.md

Structured fields (so it can be compared against Spike B):
- `mla_already_present`: yes/no + where (decides effort)
- `parity_max_abs`: value vs oracle, pass/fail at 1e-5 / 1e-4
- `head_dim_512_fits`: yes (existing tiles) / yes (new tile) / no
- `kernel_spill_smem`: regs/thread, smem/block, spill bytes, achieved occupancy
- `graph_capture_inherited`: yes/no + observed util/launch behavior
- `effort_to_full_port`: remaining model pieces, pack-loader work, T-shirt size
- `deployment_notes`: Python/Torch dependency, container/k8s implications
- `verdict`: PORT-VIABLE / PORT-BLOCKED + the one-line reason

## WHAT MAKES THIS PATH WIN (context for your investigation)

This path is recommended over the C retrofit when: Spike B stalls (its transport
stays non-capturable OR it can't fit NCCL+graph buffers in 32 GB), AND this spike
shows the port reproduces the oracle, fits head_dim 512, and inherits graphed
high-util execution. If MLA already exists in the fork (A0=yes) and parity holds,
this path becomes strongly favored regardless. Optimize your investigation toward
those decision-relevant facts; don't gold-plate.
