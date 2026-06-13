# MTP Implementation Plan

Discovery and planning for collapsing the V100 MTP sidecar into the
appliance's main weight-pack pipeline. Prerequisite-discharged record for
SPIKE B item **B1** — research is done, the existing pack tools do most of
the work, and the remaining scope is small and mechanical.

## FRESH REASON TO RESUME (2026-06-13, Sprint 604): the s604 hazard is a prime 0/71 suspect

The 0/71 investigation (s590-595) punted with the blocker localized to
"attention-output handoff, post-attention/FFN handoff, or routed-FFN activation
order" — all checked **statically** (same-activation CPU oracles) and found
clean. Sprint 604 then found and fixed a real bug in the MAIN decode path at
**exactly the attention-output handoff**: a cross-rank dense→rank ordering
hazard (`attn_output_a.d_out` written on src's dense stream, read cross-rank on
dst's rank stream, no ordering edge — `engine/attention_output.cu:48/87-98/51`).

Three facts make this the leading 0/71 hypothesis:

1. **The MTP draft path reuses the same code.** The draft runs
   `run_layer(mtp_opt, ...)` for layer 43 (`engine/token_major_loop.cu:450`),
   which goes through the same attention-output allgather handoff that carried
   the s604 hazard. The MTP draft inherited the bug.
2. **It explains the static-vs-live paradox.** In a full-capture graph a
   missing dependency edge is NOT a random race — the captured schedule replays
   identically, so it manifests as a *deterministically wrong* ordering. The
   s590-595 oracles synchronized before reading (seeing correct math); the live
   captured draft replayed a fixed mis-ordering. That is precisely "deterministic
   0/71 with every static check passing."
3. **0/71 was last measured 2026-05-30, before DENSE_FIX existed** (s604,
   promoted default-on 2026-06-13). MTP acceptance has never been measured with
   the attention-output handoff correctly ordered.

**The test is cheap and concrete** (queued for the MTP gate, after the s605+
step-floor campaign owning the GPU): re-run MTP serving acceptance on current
HEAD with `DENSE_FIX=1` (default) vs `DENSE_FIX=0`, and under the s604
amplifier (`DENSE_HAZARD_AMP`). If acceptance moves off 0 with the fix on, or
changes with the amplifier, the ordering hazard was (part of) the blocker. If
it stays 0/71 deterministically regardless, the blocker is a genuine
draft-math/semantic error and the s585-596 conclusion stands — but this must be
ruled out first, because it is the cheapest test with the highest information
value and it directly targets the s590-595-named suspect stage.

Caveat: s604's main-path hazard was rare (~1/256 steps) in the un-amplified
measured window, which argues a pure race cannot alone produce *deterministic*
0/71 — but the full-capture-determinism point (#2) is the reconciliation: the
draft is computed through a fixed captured schedule, so a mis-order there is
fixed per replay, not rare. Resolve empirically.

## Current Status: V100 TP/EP MTP Does Not Work

As of 2026-05-30, the V100 TP/EP serving MTP path was **not working** (see the
fresh-reason-to-resume note above — this status predates the s604 fix).

What is known:

- The integrated draft path runs without corrupting normal serving output; the
  main model token stream remains byte-identical with MTP on/off.
- The layer-43 MTP weight loading, dense F8 pack/orientation, output-head HC
  slicing, HC-current reduction, raw-SWA state/frontier, and raw-SWA attention
  math have all been checked or repaired through Sprints 590-595.
- Despite that, deterministic draft acceptance remains `0/71` in the serving
  harness, so the MTP draft is numerically wrong and provides no throughput
  benefit.
- Debug MTP runs can show higher GPU activity while lowering useful tok/s,
  because they add rejected draft work and diagnostic host synchronizations.

The remaining blocker is downstream of raw-SWA attention and upstream of the
already-cleared output head, likely in attention-output handoff,
post-attention/FFN handoff, routed-FFN activation order, or another subtle
layer-43 semantic mismatch. Work is intentionally punted until there is a fresh
reason to resume it.

## TL;DR

- The sidecar (`engine/mtp_sidecar.{c,h}`) runs **complete canonical MTP**,
  not a truncated probe — its 32 tensors are the full MTPBlock in GGUF
  packing convention. Upstream `research/ds4/ds4.c:3068-3104`
  `mtp_weights_bind()` requires exactly the same 32 tensor families.
- **Upstream ds4.c has no sidecar.** It loads MTP from the *same* GGUF as
  the main model. The V100 sidecar exists because the appliance's main
  GGUF was produced via a pipeline that stripped MTP (HF transformers
  `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`), and someone
  preserved MTP in a separate Q4_K/Q8_0 GGUF + parallel runtime as a
  band-aid.
- **The existing pack pipeline already handles 99% of what we need.** The
  appliance's pack-contract / appliance-pack / TurboMind-pack /
  runtime-load chain is format-agnostic and already shards by TP8/EP8.
  Adding MTP is a one-off converter + small contract extension, not new
  infrastructure.
- **Actual scope:** ~200 LoC for a new safetensors→GGUF converter, ~50–100
  LoC to extend `tools/tp-ep-pack-contract.c` for layer 43, mechanical
  binding additions in `engine/runtime_pack.cu`, and a sidecar deletion.
  Three sprint-sized tasks. Then the actual B1 throughput work
  (MTPBlock.forward + specdec loop) is unchanged.
- **Schedule:** still behind C5 / B2 / C1 / tuning sprint. In-place
  optimization first; MTP integration last.

## Evidence (verified on the pod and against upstream, 2026-05-28)

### What's on the pod

| File | Size | Tensors | MTP content |
|---|---:|---:|---|
| `/models/DSv4-Flash-256e-fixed.gguf` (appliance main) | 146 GB | 1,328 | **0** — stripped at conversion |
| `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (sidecar source) | 3.6 GB | 32 | Complete MTPBlock at Q4_K/Q8_0/F32 |
| `/models/deepseek-v4-flash-safetensors-cache/` (canonical HF release) | 46 shards | — | Complete MTPBlock at FP8/MXFP4 in shard 46 |

Config from the canonical safetensors:

```
num_nextn_predict_layers: 1
num_hidden_layers:        43
architectures:            DeepseekV4ForCausalLM
```

### What upstream ds4.c expects (the canonical tensor set)

`research/ds4/ds4.c:3068-3104` — `mtp_weights_bind()` binds **exactly 32
tensor families** as the full MTP draft model. Naming/packing convention:

```
Head + prologue (8):
  mtp.0.hc_head_base / fn / scale
  mtp.0.e_proj / h_proj
  mtp.0.enorm / hnorm / norm

Attention sublayer (11):
  mtp.0.hc_attn_fn / scale / base
  mtp.0.attn_norm
  mtp.0.attn_q_a / q_a_norm / q_b
  mtp.0.attn_kv / attn_kv_a_norm
  mtp.0.attn_sinks
  mtp.0.attn_output_a / attn_output_b

FFN/MoE sublayer (13):
  mtp.0.hc_ffn_fn / scale / base
  mtp.0.ffn_norm
  mtp.0.ffn_gate_inp                              (router weight)
  mtp.0.exp_probs_b.bias                          (router bias)
  mtp.0.ffn_gate_exps / up_exps / down_exps       (256 routed experts, stacked)
  mtp.0.ffn_gate_shexp / up_shexp / down_shexp    (shared expert)
```

The "32 vs 1,575" gap I worried about earlier was just packing convention.
GGUF stacks the 256 routed experts into 3 tensors (`ffn_*_exps`); HF
safetensors unpacks each expert as its own tensor. Same logical weights.

### Upstream speculative-decode interface

`research/ds4/ds4.h`:

```c
int  ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                         int max_tokens, int eos_token,
                                         int *accepted, int accepted_cap, ...);
bool ds4_engine_has_mtp(ds4_engine *e);
int  ds4_engine_mtp_draft_tokens(ds4_engine *e);
```

Plus engine options `mtp_draft_tokens` (cap 16, default 1) and `mtp_margin`
(default 3.0). The upstream Mac path runs real MTP speculative decoding
through this interface.

## Why the V100 has a sidecar (the actual history)

1. Upstream ds4.c expects MTP in the **same GGUF** as the main model.
   `mtp_weights_bind(w, m)` calls `required_tensor(m, "mtp.0.*")` against
   the same model handle as the main layers.
2. The V100 appliance's main GGUF (`DSv4-Flash-256e-fixed.gguf`) was
   produced by a conversion pipeline that used the HF transformers loader,
   which silently strips MTP via
   `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]`.
3. Someone produced a separate MTP-only GGUF
   (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`) at Q4_K/Q8_0/F32 quantization
   to preserve MTP.
4. A sidecar loader (`engine/mtp_sidecar.{c,h}` + Q4_K/Q8_0 paths in
   `ds4_pack.{c,h}` / `ds4_source_formats.{c,h}`) was added to bridge it.

**The sidecar is a packaging band-aid, not an architectural choice.** It
runs the complete canonical MTPBlock — just from a separate file at lower
precision via parallel infrastructure.

## The existing pack pipeline

The V100 appliance already has a complete offline-pack + runtime-load
pipeline. Tools and what they do:

```
input:  GGUF + index TSV
            ↓
    tools/tp-ep-pack-contract.c (717 lines)
        - knows DS4_N_LAYER=43, DS4_N_EXPERT=256, EP8/TP8 sharding rules
        - emits pack-contract.tsv: one row per (tensor, gpu) assignment
        - columns: record_type, tensor_id, source_name, layer_id, family,
                   source_dtype, owning_gpu, source_offset, byte_length, ...
            ↓
    tools/appliance-pack.cu (1,298 lines)
        - args: --source GGUF --index TSV --out-dir DIR
        - uses ds4_pack.h + ds4_source_formats.h + TurboMind API
        - emits: gpu{0..7}.weights + pack-index.tsv + turbomind-pack-index.tsv
            ↓
    tools/turbomind-pack.cu
        - per-layer TurboMind-specific routed-expert layout (--layer N)
            ↓
output: per-rank shard files at /workspace/logs/sprint-NNN-.../contract/
        consumed at runtime via DS4_V100_TP_EP_CONTRACT → ds4_pack_open()
        → engine/runtime_pack.cu
```

**Format support is already complete.** `ds4_source_formats.h` handles
`f8_e4m3_b128` (FP8 with per-128-block scales), `mxfp4` (MXFP4 with
per-32-block scales), `bf16`, and `f32` — exactly the formats the
canonical HF safetensors ships in.

**Sharding is already complete.** The pack contract assigns each tensor a
`(layer_id, family, owning_gpu)` triple using the TP8/EP8 rules. Adding
layer 43 (MTP) means emitting more rows that reuse these existing rules.

## Implementation scope

Five concrete pieces; only one is new code from scratch.

### 1. A one-off safetensors → GGUF converter (~200 LoC, NEW)

Lives in `tools/`. Reads canonical HF safetensors shard 46, emits a
GGUF fragment containing the `mtp.0.*` tensors in the naming + packing
convention upstream `mtp_weights_bind()` expects.

Responsibilities:

- Parse safetensors header (`u64 header_length || JSON || tensor blobs`),
  resolve offsets and dtypes for `mtp.0.*` entries.
- **Stack the 256 routed experts.** HF safetensors stores each expert as
  its own tensor (`mtp.0.ffn.experts.0.gate_up.weight` through
  `mtp.0.ffn.experts.255.gate_up.weight`). The GGUF/upstream convention
  wants three stacked tensors: `mtp.0.ffn_gate_exps`,
  `mtp.0.ffn_up_exps`, `mtp.0.ffn_down_exps`. Read 256 reads, write one
  stacked blob per family.
- **Apply the naming remap.** HF names use `attn.wq_a.weight`,
  `attn.kv_norm.weight`, `ffn.gate.weight`; upstream uses `attn_q_a.weight`,
  `attn_kv_a_norm.weight`, `ffn_gate_inp.weight`. A small static table.
- **Preserve dtypes verbatim.** No re-quantization. FP8 stays FP8, MXFP4
  stays MXFP4, BF16 stays BF16. Just copy bytes.
- Emit a GGUF with the MTP tensors in the expected naming.

Output options:

- **(a) Standalone MTP-GGUF fragment** the pack contract can reference as a
  separate `shard_file`. Smallest run.
- **(b) Unified main + MTP GGUF** by appending the MTP fragment to a copy
  of `DSv4-Flash-256e-fixed.gguf` (preferred — single source file matches
  upstream's expectation; the cost is one 146 GB file copy).

Recommend (b) for the cleanest match to upstream and the simplest runtime
story. Run once per model artifact.

### 2. Extend `tools/tp-ep-pack-contract.c` for layer 43 (~50–100 LoC)

The file already knows about 43 layers, 256 experts, and the EP8/TP8
sharding rules. Add emission for layer 43 (MTP). Most tensor families
reuse existing logic:

- Attention LoRA (`attn_q_a/b`, `attn_kv`, `attn_output_a/b`,
  `attn_q_a_norm`, `attn_kv_a_norm`, `attn_sinks`, `attn_norm`) — same
  TP8 column/row sharding as layers 0–42.
- Router (`ffn_gate_inp`, `exp_probs_b.bias`) — replicated, same as
  existing.
- Routed experts (`ffn_gate_exps / up_exps / down_exps`) — EP8 with 32
  experts per rank, same as existing.
- Shared expert (`ffn_gate_shexp / up_shexp / down_shexp`) — same
  treatment as existing.
- HC sublayer params (`hc_attn_*`, `hc_ffn_*`) — same as existing layer's
  HC mixing.

Genuinely new tensor families (9):

- `e_proj`, `h_proj` — small dense `[dim, dim]`. Same handling as
  `attn_q_a` (TP-shardable but maybe replicated for simplicity).
- `enorm`, `hnorm`, `norm` — small `[dim]` norms, replicated.
- `hc_head_fn`, `hc_head_base`, `hc_head_scale` — same as existing HC head
  parameters at the model boundary.

### 3. Re-run the pack pipeline against the unified GGUF (mechanical)

Once the pack contract emits MTP rows and the unified GGUF exists:

```bash
tools/tp-ep-pack-contract \
    --source DSv4-Flash-256e-fixed-with-mtp.gguf \
    --out contract/tp-ep-pack-contract.tsv

tools/ds4-v100-appliance-pack \
    --index contract/tp-ep-pack-contract.tsv \
    --source DSv4-Flash-256e-fixed-with-mtp.gguf \
    --out-dir contract/

tools/ds4-v100-turbomind-pack \
    --index contract/tp-ep-pack-contract.tsv \
    --source DSv4-Flash-256e-fixed-with-mtp.gguf \
    --out-dir contract/ \
    --layer 43
```

No code changes — re-run the existing tools against the new input.

### 4. Engine binding rules in `engine/runtime_pack.cu` (small)

Tell the engine layer 43 exists. The `runtime_pack.cu` file already binds
layers 0–42 via the pack contract. Most of the change is "extend the
per-layer loop to 44 layers" plus a few bindings for the 9 new tensor
families (allocate small device buffers, copy in from the pack-contract
arena). Comparable in scope to any "add a layer family" diff.

### 5. Delete the sidecar (mechanical)

- `engine/mtp_sidecar.c`
- `engine/mtp_sidecar.h`
- Q4_K and Q8_0 source paths in `ds4_pack.c`, `ds4_pack.h`,
  `ds4_source_formats.c`, `ds4_source_formats.h` — verify no other
  consumer outside the sidecar; many tests may consume these.
- The standalone MTP smokes in `smokes/mtp-*.{c,cu}` that target the
  sidecar interface. These were already flagged as "smokes that grew up"
  debt in the repo review.
- Launcher / profile references to `DS4_V100_MTP_*` env vars that were
  sidecar-specific.

Operator-side: the separate
`/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` file becomes unreferenced
and can be removed from `/models/` on the pod.

## What stays the same — the actual B1 work

Steps 1–5 above land MTP weights in the engine. They do not enable TP/EP
MTP serving. The actual throughput work is unchanged:

### Phase A — MTPBlock.forward in `engine/`

Add `engine/mtp_step.cu` (extends the existing file from sprint 525) with
`run_mtp_step(...)` that calls existing sublayer primitives:

```c
e = engine_embed(input_ids);          // shared with main
e = rms_norm(e, enorm);
x = rms_norm(x, hnorm);
x = e_proj(e) + h_proj(x);

run_hc_current(...);                  // engine/hc_current.cu
run_attention(...);                   // engine/attention_*.cu
run_ep_compose(...);                  // engine/ep_compose.cu
run_post_attention_ffn(...);          // engine/post_attention_ffn.cu

logits = run_mtp_head(x, hc_head_fn, hc_head_scale, hc_head_base, norm);
```

All sublayer calls reuse the existing kernels (`kernels/v100/norm.cuh`,
`attention.cuh`, `ep_compose.cuh`, `hc_mix.cuh`). **No new kernel work.**

### Phase B — TP/EP speculative-decode loop (the real B1)

Wire the speculative pattern across 8 ranks:

- MTP forward proposes K draft tokens (K = `mtp_draft_tokens`, ≤ 16).
- Main model forward processes (prev + K drafts) in parallel.
- Accept/reject by comparing main verify logits to draft probabilities,
  using `mtp_margin` as the threshold — same semantics as upstream
  `ds4_session_eval_speculative_argmax`.
- Advance position by 1 + accepted_k.
- **Cross-rank coordination: every rank must agree on accept/reject so KV
  state stays consistent.** This is the work the TP/EP launcher refuses
  MTP for today.

This is the largest piece of B1 — raising effective `M` from <1 to
`(K+1)`/expert and lifting the EP-bound 53% bucket.

## Sequencing relative to the rest of SPIKE B

In-place model tuning still goes first.

```
1. C5 sync-point reduction pass 2          ← in flight (sprint 529)
2. B2 compact EP variable-size NCCL compose
3. C1 piecewise graph capture
4. Tuning sprint (reference-shape perf, shape envelope, NCCL pinning, C4 spill)
5. MTP weight integration (steps 1–5 above — 3 small sprints)
6. MTPBlock.forward (Phase A — 1 sprint)
7. TP/EP speculative-decode loop (Phase B / actual B1 — N sprints)
```

Steps 5–6 are correctness-only per the per-sprint validation policy.
Step 7 is the throughput sprint and opts into perf measurement at the
reference shape per `docs/sprints/VALIDATION_CONTROL_POLICY.md`.

## Files affected (forward-looking inventory)

**Add:**

- `tools/ds4-v100-mtp-from-safetensors.{c,cu}` (or similar name) — the
  ~200-line one-off converter.
- `engine/mtp_step.cu` extensions for `run_mtp_step` (Phase A).

**Extend:**

- `tools/tp-ep-pack-contract.c` — emit layer-43 rows (~50–100 LoC).
- `engine/runtime_pack.cu` — bind MTP tensor families.
- `engine/api.h` — expose `run_mtp_step` if Phase A lands.
- `appliance/main.cu` — call into MTP step from the decode loop (Phase B).

**Delete (when steps 1–5 land):**

- `engine/mtp_sidecar.c`
- `engine/mtp_sidecar.h`
- Q4_K / Q8_0 source paths in `ds4_pack.{c,h}` and
  `ds4_source_formats.{c,h}` (after verifying no other consumer).
- `smokes/mtp-*.{c,cu}` smokes that target the sidecar.
- `DS4_V100_MTP_*` env vars in launcher/profile that were sidecar-only.

**No engine kernel changes anywhere.**

## One-line summary

The existing pack pipeline (`tp-ep-pack-contract.c` → `appliance-pack.cu`
→ runtime mmap) already handles formats, sharding, and per-rank staging.
Adding MTP is a one-off ~200-LoC safetensors→GGUF converter plus a small
contract extension and engine-side binding rules — three sprint-sized
tasks that delete the sidecar runtime. The actual B1 throughput work
(MTPBlock.forward + TP/EP-coordinated specdec loop) is unchanged and
still sits behind C5 / B2 / C1 / tuning.
