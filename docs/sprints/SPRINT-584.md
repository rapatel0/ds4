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

### Phase 1 COMPLETE + validated end-to-end (2026-05-30)

The full EP=8 MTP weight pack ran on the pod from the 32-family GGUF:

1. `pack.c --manifest mtp-manifest-ep.tsv --gpus 8 --write-index` ->
   `/workspace/mtp-pack-ep/pack-index.tsv` (32 tensors, 3.35 GiB).
2. `appliance-pack --index pack-index.tsv --source mtp-fragment-ep.gguf
   --layer 43 --fuse-gate-up-interleaved --lib <turbomind>` -> packed
   `blk.43.ffn_gate_up_exps experts=256/256 fused_N=4096 interleaved=1` +
   `blk.43.ffn_down_exps experts=256/256` into `/workspace/mtp-shards-ep`.
   **The fused row matches the main-model `blk.N.ffn_gate_up_exps` format
   exactly** (same `fused_N=4096 interleaved=1`). gpu0.weights=3.6 GB,
   gpu1-7=0 -- correct by-layer storage for a single layer (runtime EP-slices
   at load).
3. `tp-ep-pack-contract --pack-dir mtp-shards-ep` ->
   `/workspace/mtp-contract-ep/tp-ep-pack-contract.tsv`: **16 `ep_expert` rows**
   (2 fused expert tensors x 8 ranks; the fusion collapsed the old 24 from 3
   separate tensors), **32 experts/rank** EP-split (`efirst` 0/32/../224,
   `ecount=32`) across all 8 GPUs, plus 80 `dense_tp`, 512 `kv_shard`,
   328 `kv_comp_state`, 152 `replicated_control`. A correct TP8/EP8 plan.

This is the EP=8 MTP weight pack: fused gate_up (production format), experts
EP-sliced 32/rank, dense TP-sharded, loaded per-rank at runtime via the
contract's `efirst`/`ecount` (`turbomind_bindings.cu:138`). Artifacts:
`/workspace/mtp-fragment-ep.gguf`, `mtp-manifest-ep.tsv`, `mtp-shards-ep/`
(shards + fused turbomind-pack-index), `mtp-contract-ep/` (EP contract).

## Phase 2 design (runtime layer-43 bind) — scoped 2026-05-30

**Key finding: a naive `layer < 43` -> `< 44` loop extension is WRONG.**
`ds4_layer_ratio(43)` returns 128 (43 is odd, >=2), so the standard per-layer
binding in `runtime_pack.cu` would try to load `attn_compress_ape` / `indexer`
tensors for layer 43 -- but the MTP attention has NONE of those. The safetensors
`mtp.0.attn.*` set is only `wq_a/b`, `wkv`, `wo_a/b`, `q_norm`, `kv_norm`,
`attn_sink` -- the simple ratio=0 attention form. So the MTP layer needs
DEDICATED binding, not the ratio-driven 0-42 loop.

**The binding template already exists.** `engine/mtp_step.cu`'s `mtpf_views`
struct (lines 34-67) is exactly the 32-family MTP weight set: `enorm`, `hnorm`,
`e_proj`, `h_proj`, `hc_attn_{fn,scale,base}`, `attn_norm`, `attn_q_a`,
`attn_q_a_norm`, `attn_q_b`, `attn_kv`, `attn_kv_norm`, `attn_sinks`,
`attn_output_a/b`, `hc_ffn_{fn,scale,base}`, `ffn_norm`, `ffn_gate_inp`,
`exp_probs_b`, `ffn_{gate,up,down}_shexp`, `ffn_{gate,up,down}_exps`,
`hc_head_{fn,scale,base}`, `output_norm` (= our `blk.43.norm.weight`). Ratio=0
(no compress/indexer), prologue + head -- matches our 32-family pack exactly.

**The only EP change vs the LP sidecar bind:** experts move from
`ds4_gpu_q4_k_expert_view` (bound via `ds4_mtp_sidecar_*` from the Q4_K sidecar)
to the turbomind mxfp4 EP-split views, bound via
`turbomind_bindings.cu` `open_shared_expert_bindings` /
`pack_descriptor_set` -- the same path layers 0-42 use, which slices 32/rank
at load (`global_expert = rank*32 + active[i]`). The non-expert families
(norms/hc/proj, all F32/F8 control+dense) load from the unified pack-ep
artifacts via `runtime_pack.cu`'s control/dense loaders.

**Phase 2 implementation:** (a) extend `open_shared_expert_bindings`
`layer < 43` -> `< 44` + size `LayerExpertCache layers[]` for 44, so layer 43's
experts load from `mtp-shards-ep` + `mtp-contract-ep`; (b) add a dedicated
MTP non-expert bind (the 29 non-expert families) into MTP storage, sourced
from the unified pack (not the sidecar). Validate: appliance loads layer 43,
each rank binds its 32-expert slice + the non-expert families, no crash, byte
counts match. Requires a CUDA rebuild + load test on the pod.

## Phase 3/5 integration point found (2026-05-30)

The LP `mtp_step.cu` forward (`ds4_mtp_forward_*`) is **dead code in the TP/EP
appliance** -- it's not invoked anywhere in `engine/*.cu|*.inc` or
`appliance/*.cu`. The real EP decode driver is the **replay pipeline**
(`engine/replay.c` + `replay_step_pipeline.inc`), and the **verify half of
speculative decode already exists** there: `ds4_replay_verify_token_block`
(`replay_step_pipeline.inc:1237`) feeds a draft token block through the main
model and selects argmax per position, reporting `accepted_prefix_len` /
`speculative_saves`. The sprint181 `mtp_serving=verify` run exercised it
(mtp_attempted=16, accepted=16) with externally-supplied draft tokens.

So the EP MTP integration is in the replay pipeline, NOT `mtp_step.cu`:
- **Phase 3 (the missing piece):** an MTP draft forward (layer 43) that runs
  in the replay pipeline -- embedding-combine prologue (`enorm`/`hnorm`/`e_proj`/
  `h_proj`) + the shared EP per-layer execution (attention + HC + the
  `ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32` routed FFN)
  + the MTP head (`hc_head_*` + `output_norm`) -> draft logits -> K draft tokens.
- **Phase 5 (the specdec loop):** wire draft -> `ds4_replay_verify_token_block`
  (exists) -> accept/reject -> advance, coordinated across the 8 ranks.

**Runtime-validation prerequisite — DE-RISKED (no re-pack needed).** Because
the MTP layer uses DEDICATED storage (not the main [43] arrays), it can load
from its OWN pack dir. `pack_descriptor_set` already takes a `pack_dir` arg
(`turbomind_bindings.cu:110/113`: `sidecar_path = path_join(pack_dir, sidecar_file)`).
So the appliance loads layers 0-42 from the s181 pack and layer 43 (MTP) from
the `mtp-shards-ep` + `mtp-contract-ep` artifacts via a dedicated bind + a new
`--mtp-pack-dir`/`--mtp-contract` option. No unified 146 GB re-pack; the MTP
weights stay decoupled (like the sidecar) but in the unified EP format
(mxfp4/turbomind, fused gate_up, EP-split 32/rank). This is validatable: the
appliance loads the MTP layer from its own dir, byte counts match, no crash.

### Phase 2 expert bind IMPLEMENTED + runtime-validated (2026-05-30)

Added a dedicated MTP expert bind (commit 193b555a): `Options` MTP fields,
`SharedExpertBindings.mtp_layer`, `open_mtp_expert_bindings()` (reuses
`parse_tm_index` + `pack_descriptor_set` for layer 43 from the MTP pack dir),
wired into `appliance_runtime.cu`, with `--mtp-pack-dir`/`--mtp-tm-index`/
`--mtp-contract` args. Built clean (nvcc, make rc=0).

**Runtime load test PASSED** (no re-pack): ran the appliance against the s181
pack (layers 0-42) + the MTP dir, GPUs free. Log:
`tp_ep_shared_expert_bindings_load layers 43 ... PASS` then
`tp_ep_mtp_expert_bindings_load layer 43 bytes 3422552064 PASS` -- the MTP
layer's experts loaded EP-split 32/rank (3.42 GB, same as a main layer's expert
load), and the appliance proceeded to decode normally (`decode_pass=1`, `rc=0`)
-- the MTP bind did not disturb the serving path. The dedicated-pack-dir design
works: layers 0-42 from s181, layer 43 from `mtp-shards-ep`, no unified re-pack.

**Phase 2 non-expert bind DONE + runtime-validated (commit 08d839a3).**
`open_mtp_nonexpert_bindings` loads the 29 non-expert families (dense_tp +
replicated_control) from the MTP contract+pack into per-(name,gpu) device
buffers (`MtpNonExpertWeights`), reusing `parse_contract`/`physical_row_offset`/
`read_exact_at`. All families were confirmed already packed in
`mtp-shards-ep/gpu0.weights` (the `--layer 43` filter only gates experts).
Load test PASS: `tp_ep_mtp_nonexpert_bindings_load layer 43 tensors 99 bytes
174661100 PASS` (99 = 80 dense_tp [10 families x 8 TP ranks] + 19
replicated_control). **Phase 2 (runtime layer-43 bind) is COMPLETE**: layer
43's full weights (experts + dense + control) load from the dedicated MTP pack
dir, serving path undisturbed, no re-pack.

## Phase 3 (MTP draft forward) design — scoped 2026-05-30

The complete forward sequence already exists in `engine/mtp_step.cu:605-650`
(the LP path). It is a single-token arena sequence:
prologue (`enorm` RMS -> `e_proj` matmul -> repeat HC; `hnorm` RMS -> `h_proj`
matmul -> add) -> HC attn mix -> attention (`attn_q_a` -> `q_a_norm` -> `attn_q_b`
-> head RMS -> `attn_kv` -> `kv_norm` -> kv store -> decode heads -> output) ->
HC expand -> HC ffn mix -> `ffn_norm` -> router (`ffn_gate_inp` ->
`router_select_bias`) -> **routed MoE** -> shared expert (gate/up/swiglu/down) ->
add -> HC expand -> HC head mix -> `output_hc_weights` -> weighted sum ->
`output_norm` -> output matmul -> logits.

Reuse this skeleton; THREE substantive EP changes make it sprint-sized:
1. **Dense projections F8, not Q8_0.** The LP path uses
   `ds4_gpu_arena_q8_0_matmul_f32` (sidecar Q8_0). The EP MTP weights are
   F8_E4M3 (e_proj/h_proj/attn_q_a/q_b/kv/output_a-b/shared experts), so these
   become the F8 dense matmul path (as layers 0-42 use).
2. **Routed FFN: EP all-to-all, not single-GPU Q4_K.** The LP path uses
   `ds4_gpu_arena_q4_k_routed_moe_one_f32` (single-GPU). The EP version uses the
   shared `ds4_gpu_arena_turbomind_mxfp4_routed_gate_up_swiglu_down_sum_f32`
   dispatch across the 8 ranks (the EP-split experts from Phase 2 +
   dispatch/combine all-to-all) -- this is the throughput point.
3. **Weights from the bound structures, multi-rank.** Source the weights from
   the Phase-2 `MtpNonExpertWeights` (by name/rank) + the MTP expert cache, not
   the sidecar views, and run the sequence across the 8 ranks in the replay
   pipeline (not the single-GPU arena).

Validation gate: the EP MTP draft logits must match the LP sidecar's draft
distribution (the reference) within the determinism floor. This is the
"MTPBlock.forward" Phase A (~1 sprint per MTP_IMPLEMENTATION.md) and is the next
major engine implementation; the weight integration it builds on is now done.

### Phase 3 implementation architecture DECIDED (reuse run_layer)

Two paths considered for the multi-rank EP MTP body:
- **(A) Reuse `run_layer`** (the validated multi-rank per-layer EP execution:
  attention + HC + the turbomind mxfp4 all-to-all routed FFN) for the MTP
  transformer body, wrapped by a new prologue + head. CHOSEN.
- (B) A from-scratch dedicated multi-rank MTP forward. Rejected -- it would
  reimplement the EP all-to-all (huge, error-prone).

**Key finding that shapes the bind:** `run_layer` sources weights from
`SharedDenseOps->layers[opt.layer]` (`runtime_types.cuh:684`, a `[43]` array,
loaded by `open_shared_dense_ops`), `SharedHcControls` (loaded by
`open_shared_hc_controls`), and `shared_expert_bindings->layers[opt.layer]`.
It does NOT read the name-keyed `MtpNonExpertWeights` I built in Phase 2. So
the clean path requires the MTP non-expert weights in THOSE structures at
slot 43:
- Extend `SharedDenseOps.layers[43]->[44]` + `SharedHcControls` for layer 43
  (a CONTAINED `[43]->[44]` -- just these two structs + their loaders, not the
  full `runtime_types` `[43]` sprawl).
- Extend `open_shared_dense_ops` / `open_shared_hc_controls` to load layer 43
  from the MTP contract+pack (reusing the Phase-2 load mechanism --
  parse_contract + read -- only the target struct changes; the Phase-2 load
  validation still confirmed the MTP non-expert tensors are readable/correct).
- `run_layer` for `opt.layer==43`: source the contract from `mtp_contract_path`
  and the expert cache from `shared_expert_bindings->mtp_layer`; force ratio=0.
- New: the prologue (`enorm`/`e_proj` + `hnorm`/`h_proj`) before, and the head
  (`hc_head_*` + `output_norm` + output matmul) after.

This reuses the validated EP all-to-all (the hard, throughput-critical part) and
confines new code to the prologue/head + the contained `[44]` extension + the
`run_layer` MTP redirect. The Phase-2 `MtpNonExpertWeights` is superseded as the
consumption format (its load logic is reused); the expert bind
(`shared_expert_bindings.mtp_layer`) is directly reusable by `run_layer`.

### Phase 3 implementation progress + KV-management finding (2026-05-30)

Implemented (commits cd572ae8..this): converter body-tensor renames
(`attn_kv`->`attn_kv_latent`, `attn_sinks.weight`->`attn_sinks`) so the MTP pack
matches the main convention for run_layer reuse (re-ran the pack chain); a
`run_layer` MTP redirect (`ds4_layer_ratio(43)=0`; layer-43 sources the contract
from `mtp_contract_path`, experts from `shared_expert_bindings->mtp_layer`, graph
cache disabled); and an MTP layer-43 scaffold in the active token-major path
(`token_major_loop.cu`). Builds clean.

**Load-test finding (the incremental gate working):** the MTP scaffold reached
`run_layer` (ratio=0 applied correctly) but FAILED with
`tp_runtime_dense_kv_slice_failed: layer is outside DS4 range`
(`tp_runtime.cu:1231`, `layer >= kLayers`). This is NOT just a bound to bump:
`run_layer` does **persistent per-layer KV-cache management** over the [43]
layer arrays, but the MTP attention uses a **dedicated draft KV buffer**
(`mtp_step.cu` `raw_t` / `MTPF_RAW_CAP`), not the persistent 43-layer cache.
So the clean split is:
- **FFN (routed MoE)**: reuse the EP all-to-all dispatch via run_layer -- the
  throughput point, experts already bound in `mtp_layer`. Feasible.
- **Attention**: needs MTP-specific KV handling (dedicated draft KV), NOT
  run_layer's persistent-cache path. run_layer cannot be reused wholesale.

Resolution options for Phase 3 attention: (a) give the MTP layer a KV slot
(`kLayers`->44 + the [43] KV/state arrays -> [44]) and run it like a main layer
over the context KV; or (b) an MTP-specific attention path (dedicated draft KV,
mirroring `mtp_step.cu`), composed with the reused EP FFN. (b) is closer to the
canonical MTP draft semantics and avoids the [43] KV-array sprawl. This is the
next decision/implementation step.

### Phase 3 rc=14 root cause CONFIRMED (2026-05-30): optimized decode infra has no contract fallback

After kLayers->44 + KV arrays + the 9-file guard migration, the MTP layer
cleared all layer-range guards and reached the decode kernels, failing at rc=14.
Root cause (code-confirmed in `engine/attention_projection.cu:1-30`):
`run_true_ds4_attention_projection_prefix` returns 1 when `ops` (LayerDenseOps)
is null -- and the MTP scaffold passes `shared_dense_ops=nullptr`. It ALSO
requires `hc->d_attn_norm_weight[layer]` / `d_q_a_norm_weight[layer]` /
`d_kv_a_norm_weight[layer]` (SharedHcControls per-layer arrays). So the
appliance's OPTIMIZED decode path (`true_ds4_attention_projection_gate`,
`dense_f16_*_compose`) has NO contract fallback for dense/HC -- it requires
`SharedDenseOps->layers[layer]` + `SharedHcControls` populated per-layer.

**Scope finding:** the run_layer-reuse path for the MTP body requires extending
the WHOLE optimized-decode infrastructure to layer 43, not just kLayers/KV:
`SharedDenseOps.layers[43]->[44]` + `open_shared_dense_ops` for layer 43,
`SharedHcControls` per-layer arrays ->[44] + `open_shared_hc_controls` for layer
43, and likely the `DenseF16Cache` (the f16 cache the dense-ops loader consumes)
for layer 43. This is a large, invasive multi-structure extension -- the bulk of
the MTPBlock.forward (Phase A) sprint. The appliance decode is a deeply
43-layer-specialized optimized pipeline; adding a 44th layer touches its core
caches. Alternative (disable the optimization gates for the MTP layer to use a
baseline path) is uncertain -- the decode loop has no obvious non-optimized
attention-projection fallback. Decision/implementation of the optimized-infra
extension is the next substantial chunk.

## Status

Phase 1 (EP=8 MTP weight pack) COMPLETE + validated. Phase 2 (runtime layer-43
bind) COMPLETE + runtime-validated. Phase 3 (MTP draft forward) IN PROGRESS:
run_layer MTP redirect + kLayers->44 + KV arrays + guard migration landed (MTP
layer reaches the decode kernels; existing serving intact); rc=14 root cause
confirmed -- the optimized decode path needs SharedDenseOps + SharedHcControls
populated for layer 43 (no contract fallback), a large multi-structure [44]
extension that is the bulk of Phase A. Earlier design notes (reuse the mtp_step.cu sequence
with 3 EP changes (F8 dense matmuls, EP all-to-all routed FFN, bound-structure
weights + multi-rank); validated against the LP sidecar draft logits. This is
the sprint-sized Phase A engine work; weight integration (steps 1-4) is done.
Architecture fully mapped:
Phase 2 = dedicated layer-43 bind (ratio-0, templated on `mtpf_views`, experts
via the shared turbomind EP path); Phase 3 = MTP draft forward in the REPLAY
pipeline (verify half `ds4_replay_verify_token_block` already exists); Phase 5
= the specdec accept/reject loop. These are the core B1 engine effort (CUDA
builds + serving validation), sequenced as multiple sprints per
MTP_IMPLEMENTATION.md. Next concrete step: the Phase 2 bind code +
`turbomind_bindings.cu` `layer < 44` extension, compiled as the first checkpoint.
