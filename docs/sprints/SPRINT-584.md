# Sprint 584 — EP=8 MTP integration plan (correcting the LP-vs-EP conflation)

Date: 2026-05-30

## Why this sprint exists (user correction)

The DS4 V100 appliance has had **two** parallelism configurations:

1. **LP=8 (layer-parallel) — the FIRST pass.** Each layer's full 256 routed
   experts live on one GPU; the 43 layers spread across the 8 GPUs (uneven:
   ~6/6/6/6/6/5/5/3). The deployed `s181` pack
   (`/workspace/packs/ds4-appliance-full-tm-gated-s181`) is an LP pack — its
   `turbomind-pack-index.tsv` has exactly 86 rows (43 layers x {ffn_down_exps,
   ffn_gate_up_exps}), each on a single `owning_gpu`, `experts_packed=256`, and
   the shard sizes are unequal (gpu0 22.5 GB .. gpu7 11.8 GB).
2. **EP=8 (expert-parallel) — the CURRENT target.** Each of the 8 ranks holds a
   32-expert slice (experts `[rank*32 .. rank*32+32)`) of **every** layer, with
   a decode-time all-to-all dispatch (the "EP 65% of decode" the tuning sprint
   measured). The pack is described by `tp-ep-pack-contract.c` `ep_expert` rows
   (24 for a single MTP layer = 3 expert tensors x 8 ranks, `expert_first`
   0/32/../224, `expert_count=32`).

**The existing MTP code is LP-era and must be replaced for EP=8.** Concretely,
`engine/mtp_step.cu` + the MTP sidecar read experts from
`ds4_mtp_sidecar_q4_k_expert_view` in **Q4_K** with **separate** gate/up
(`ffn_gate_exps`/`ffn_up_exps`) and do self-contained, non-EP expert compute.
That is the first-pass (LP) MTP. The earlier "MTP weight-pack" sprints (582/583)
and `tools/mtp-pack-fragment.c` inherited this LP framing (separate gate/up,
single-GPU emission). **They are not the EP=8 path and should not be carried
forward as-is.**

## Code evidence (EP=8 is the production runtime)

- Launcher `tools/ds4-v100-run-tp-ep-appliance.sh` execs
  `appliance/ds4-v100-tp-ep-appliance` with BOTH `--contract`
  (default `sprint245-tp-ep-dense-f16-cache-contract/.../tp-ep-pack-contract.tsv`,
  the EP-split contract with `ep_expert` rows) AND `--tm-index`
  (the turbomind expert blobs). The EP contract's `expert_first`/`expert_count`
  slice the per-layer blobs into per-rank 32-expert sets at load.
- `engine/runtime_pack.cu` parses `record_type == "ep_expert"` with
  `expert_first` (f[14]) / `expert_count` (f[15]) (lines 157-163, 387).
- `engine/runtime_types.cuh`: `kGlobalExperts=256`, `kLocalExperts=32`.
- The per-layer routed-FFN dispatch the EP path uses (the reusable building
  block for MTP) is in `engine/layer_execute_ffn.inc:129-194`:
  when `state->has_turbomind_routed` and `has_turbomind_fused_gate_up`, it calls
  `ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32(...,
  &state->turbomind_gate_up_view, &state->turbomind_down_view, ...,
  state->routed_experts, ...)`. The experts are **mxfp4**, **fused gate_up**,
  EP-split per rank.

## The EP=8 MTP design (target)

Treat MTP as **layer 43 of the unified EP=8 model**, not a sidecar. Layer 43
has the same per-layer structure as layers 0-42 (input norm, MLA attention, HC
mixing, routed MoE FFN) plus an MTP-specific embedding-combine prologue
(`enorm` + `e_proj`: combine the previous hidden state with the embedding of the
last accepted token) and it shares the model's output head.

Therefore the EP=8 MTP reuses the existing per-layer EP execution and the
existing `ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32`
dispatch — no new expert kernel. The genuinely new work is the prologue, the
layer-43 binding through the unified path, and the speculative-decode loop.

## Plan (phased; each phase builds+validates on the pod per the steering process)

1. **EP pack emission (NOT a converter change).** Correction: the gate/up
   FUSION is a downstream `appliance-pack` option (`--fuse-gate-up-interleaved`;
   it reads SEPARATE `ffn_gate_exps`/`ffn_up_exps` at `tools/appliance-pack.cu:302-314`
   and fuses to `turbomind_mxfp4_grouped_gate_up_interleaved`). The main-model
   SOURCE manifest also has separate gate/up. So `tools/mtp-pack-fragment.c`'s
   separate gate/up is ALREADY correct and needs no change. The real Phase-1
   work is operational: drive the pipeline from the EP `ep_expert` contract
   (`mtp-contract2` already has the 24 EP rows) through turbomind-pack +
   `appliance-pack --fuse-gate-up-interleaved`, so each rank gets its 32-expert
   slice and gate/up is fused. (The earlier single-GPU emission ran against the
   wrong owning_gpu=0 index and without the fuse flag.)

   **Storage is by-layer, NOT equal-shard (firsthand-confirmed).** EP-split
   happens at LOAD, not in the pack: `engine/turbomind_bindings.cu:138`
   computes `global_expert = rank * kLocalExperts + active[i]`
   (`kLocalExperts=32`, `active=[0..32)`), so each rank reads its own
   `[rank*32, rank*32+32)` slice from the single by-layer sidecar blob via
   `weight_offset + global_expert*weight_bytes_per_expert`. So the MTP layer-43
   experts are one by-layer blob (256 experts, fused gate_up, mxfp4) on one
   owning_gpu, exactly like s181 (shards stay unequal). Validate: a fused
   `blk.43.ffn_gate_up_exps` turbomind-index row (256 experts) matching a
   main-model `blk.N` row, NOT equal shard sizes.
2. **Runtime layer-43 bind.** Two loops hardcode `layer < 43` and must become
   `< 44`: (a) `engine/runtime_pack.cu` (dense/attn/HC/norm + the MTP
   `enorm`/`e_proj`), and (b) `engine/turbomind_bindings.cu:198`
   `open_shared_expert_bindings` (the per-rank 32-expert blob load). Validate:
   appliance loads layer 43, each rank binds its 32-expert slice.
3. **MTPBlock.forward (EP).** Add the embedding-combine prologue + run layer 43
   through the shared EP per-layer execution + the output head. Validate against
   the LP sidecar's MTP logits as a reference (same draft distribution).
4. **Sidecar delete.** Remove `engine/mtp_step.cu` + `engine/mtp_sidecar.{c,h}`
   once (3) matches the reference.
5. **TP/EP speculative-decode loop (Phase B).** The draft-K / verify / accept
   loop across ranks — the actual throughput sprint; opts into perf measurement.

## Definition of Done (this planning sprint)

- The LP-vs-EP distinction and EP=8 MTP design recorded, with code evidence.
- Prior LP-framed MTP records (582/583, steering, vision) annotated as LP-era.
- Phase 1 (EP pack emission via the ep_expert contract + `--fuse-gate-up-interleaved`)
  scoped as the first build increment; converter confirmed already correct.
- Steering + vision updated; committed (excluding user-owned
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`).

## Phase 1 progress (2026-05-30) — guided by MTP_IMPLEMENTATION.md

`MTP_IMPLEMENTATION.md` is the authoritative guide (user-confirmed). It defines
the canonical **32-family** MTP tensor set, sourced from the safetensors
(shard 46) and copied verbatim, experts kept separate (appliance-pack fuses).

**Converter completed + validated (commit a24b978a).** `tools/mtp-pack-fragment.c`
was emitting only 20 of the 32 families. Added the 12 missing — the
hyper-connection sets (`hc_attn_{fn,scale,base}`, `hc_ffn_{fn,scale,base}`,
`hc_head_{fn,scale,base}`), `h_proj`, `norm`, and the router bias `exp_probs_b`.
GGUF names verified against the main model (`blk.N.hc_attn_fn` etc. are bare;
`exp_probs_b` is bare, not `.bias`); HF source names verified against
safetensors shard 46. Rebuilt + ran on the pod: **32 tensors, EMIT_OK=1, no
duplicates**, all 12 new families present with correct dtype/shape (hc F32,
`h_proj` f8_e4m3_b128 `[4096x4096]`, `norm` f32, `exp_probs_b` f32 `[256]`).
Output: `/workspace/mtp-fragment-ep.gguf` + `/workspace/mtp-manifest-ep.tsv`.
The converter is the only from-scratch code in the weight-integration steps
(per the guide); it is now complete.

**Remaining Phase 1 (mechanical, "re-run existing tools" per the guide):**
from the 32-family GGUF: `pack.c --manifest mtp-manifest-ep.tsv --write-index`
-> `tp-ep-pack-contract --pack-dir <dir>` (emits the layer-43 `ep_expert` rows,
32/rank) -> `appliance-pack --fuse-gate-up-interleaved` (by-layer expert blob +
fused gate_up). Validate: a fused `blk.43.ffn_gate_up_exps` turbomind row
(256 experts) matching a main-model `blk.N` row. (Schema-matching across pack.c
/contract/appliance-pack indices is the known fiddly part.)

## Status

Architecture established and the converter (the sole new-code deliverable)
completed + validated to all 32 families. Prior MTP weight-pack work was
LP-framed and is superseded. Next: the mechanical pack chain above, then the
runtime layer-43 bind (Phase 2).
