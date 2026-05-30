# Sprint 585 — MTP TP/EP speculative-decode loop (Phase B / the B1 throughput sprint)

Date: 2026-05-30 (planned; execute as a fresh focused session per user decision)

## Why this is its own sprint

Sprint 584 completed + validated the entire MTP weight integration
(MTP_IMPLEMENTATION.md steps 1-4) AND the **MTP forward body executing through
the appliance's multi-rank EP decode** -- the structural breakthrough. All
weight loaders are built. The ONE remaining piece is the speculative-decode
loop: a from-scratch multi-rank serve-path decode driver whose only validation
is a live pod serving run. The user chose to run it as a fresh focused session
(full context coherence + serving bench), not as continued single-turn churn in
the shared decode path that carries the shipped 1.22x serving.

## What's already built + validated (Sprint 584, on the pod)

- **EP=8 MTP weight pack**: `tools/mtp-pack-fragment.c` (32-family converter) ->
  `mtp-fragment-ep.gguf` -> pack.c -> `appliance-pack --fuse-gate-up-interleaved`
  -> `mtp-shards-ep` -> `tp-ep-pack-contract` -> `mtp-contract-ep` (16 ep_expert
  rows, fused gate_up, EP-split 32/rank).
- **Runtime bind (engine/, all load-tested)**:
  - `open_mtp_expert_bindings` (turbomind_bindings.cu) -- EP-split experts.
  - `open_mtp_nonexpert_bindings` (turbomind_bindings.cu) -- 99 non-expert tensors.
  - `load_mtp_hc_layer43` (runtime_pack.cu) -- HC/norm/router controls + rank-dist
    + router EP/shard + norm-rank, slot 43.
  - `load_mtp_dense_layer43` (output_head.cu) -- dense F8 projections, slot 43.
  - `load_mtp_output_head` (output_head.cu) -- MTP draft head (shares LM matmul,
    overrides head HC weights).
  - Structural: `kLayers`->44, `ds4_layer_ratio(43)=0`, `layer_ratio(43)=0`,
    `SharedHcControls`/`SharedDenseOps` `[44]`, KV arrays `[44]`, 9-file guard
    migration (`>=43`->`>=44`).
- **MTP forward body**: `run_layer(43)` with the MTP redirect (contract from
  `mtp_contract_path`, experts from `shared_expert_bindings->mtp_layer`) executes
  through the full EP decode -- `decode_pass=1`, `decode_checksum=6684186189`,
  `expert_rows=16` (EP-split). (rc=1 is the benign `kv_rows>0` scaffold flag,
  inapplicable to the raw-SWA MTP.)
- Appliance options: `--mtp-pack-dir`, `--mtp-tm-index`, `--mtp-contract`.

## The remaining work (this sprint): the specdec serve-driver

**Integration point:** the per-request generation loop in
`appliance/http_server.cu` (`decode_input_token` / `generated_token_ids`).
Today it generates 1 token/step; the specdec driver generates K drafts then
verifies.

**Components (all exist):**
- prologue: embedding-combine -> MTP input. `ds4_replay_read_token_embedding_f32`
  (token embedding) + `e_proj`/`h_proj` dense-F8 matmuls (weights bound in the
  MTP dense ops / `MtpNonExpertWeights`) + `enorm`/`hnorm` rms-norm. The
  previous hidden state is already in the rank HC state after layer 42.
- body: `run_layer(43)` (working, validated).
- head: `run_shared_output_head_from_rank_hc` with the MTP head (built via
  `load_mtp_output_head`) -> draft logits -> sample.
- verify: `ds4_replay_verify_token_block` (`replay_step_pipeline.inc:1237`) --
  EXISTS but currently HAS NO CALLERS; this sprint wires it.

**Driver to build (the new code):**
1. Per request, after the main forward produces the position-p hidden state:
   run the MTP prologue+body+head K times (K = `mtp_draft_tokens`, <=16),
   autoregressively, the MTP attending its raw-SWA window over the drafts ->
   K draft tokens.
2. Feed (prev + K drafts) to `ds4_replay_verify_token_block` (main model
   verifies K+1 in parallel -- this is the EP-fill throughput win: each step
   sees (K+1)x tokens, filling the grouped-GEMM tiles).
3. Accept the longest matching prefix; advance position by 1 + accepted_k.
4. Cross-rank coordination: all 8 ranks must agree on accept/reject so KV stays
   consistent (this is what the TP/EP launcher refused MTP for).

**Validation gate (the whole point):** a live pod serving run via
`tools/ds4-v100-sustained-decode-bench.sh` (mtp_serving=verify) and
`tools/ds4-v100-mtp-acceptance-matrix.sh` -- measure acceptance rate +
aggregate decode throughput at the reference shape (32 slots / 256K), comparing
against the no-MTP baseline (26.8 tok/s agg decode, Sprint 581). Also validate
the MTP draft logits against the LP sidecar's draft distribution within the
determinism floor (correctness) before measuring throughput.

**Then:** Phase A finish = the prologue/head are validated by the above;
sidecar deletion (`engine/mtp_sidecar.{c,h}`, `mtp_step.cu`, Q4_K/Q8_0 paths)
once the unified-pack draft matches the sidecar reference.

## Definition of Done

- The specdec serve-driver wired into the generation loop (prologue + body +
  head + verify + accept/reject + cross-rank coordination).
- Draft logits match the LP sidecar within the determinism floor.
- A serving run shows MTP acceptance > 0 and a decode-throughput delta vs the
  no-MTP baseline at the reference shape (the EP-fill win, banked).
- Sidecar deleted; steering + vision updated; committed (excluding
  `docs/sprints/VALIDATION_CONTROL_POLICY.md` and `research/`).

## Status

Planned. All components + the integration point are built/identified and
de-risked in Sprint 584; this sprint is the serve-driver composition + the
serving-validation measurement -- the final B1 throughput sprint.
