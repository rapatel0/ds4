# MTP Implementation Plan

This document captures the discovery and planning work for replacing the
current Multi-Token Prediction (MTP) sidecar with a fully-integrated MTP path
on the V100 TP/EP appliance. It is the prerequisite-discharged record for
SPIKE B item **B1** — research is done, the weights are local, the
implementation is now a sequenced engineering task.

## TL;DR

- The current `engine/mtp_sidecar.{c,h}` runtime is **not running real MTP**.
  It loads only 32 tensors of a 1,575-tensor MTPBlock. The canonical
  attention + MoE expert weights for layer 43 (the MTP block) are absent
  from the sidecar's source file.
- The canonical weights **are already on the V100 pod** at
  `/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`,
  at the same FP8/MXFP4 quantization as the main model (with scales).
- The main appliance file `/models/DSv4-Flash-256e-fixed.gguf` **does not
  contain MTP** — it was produced via the stock HuggingFace transformers
  loader, which silently strips MTP keys.
- The implementation path is therefore a packaging + binding + forward-pass
  exercise on top of the existing pack tools and engine, not a research
  expedition. It scopes into roughly **3 sprints** plus the
  speculative-decode-loop work that B1 was always about.

## Evidence (verified on the pod, 2026-05-28)

| File | Size | Tensors | MTP tensors |
|---|---:|---:|---:|
| `/models/DSv4-Flash-256e-fixed.gguf` (the appliance file) | 146 GB | 1,328 | **0** |
| `/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (sidecar source) | 3.6 GB | **32** | 32 |
| `/models/deepseek-v4-flash-safetensors-cache/` (canonical HF release) | 46 shards | — | **1,575** in shard 46 |

Canonical config from the safetensors cache:

```
num_nextn_predict_layers: 1
num_hidden_layers: 43
architectures: DeepseekV4ForCausalLM
```

Tensor-name sample from the canonical release (all under `mtp.0.*`):

```
mtp.0.attn.attn_sink
mtp.0.attn.kv_norm.weight
mtp.0.attn.q_norm.weight
mtp.0.attn.wkv.scale
mtp.0.attn.wo_a.scale
mtp.0.attn.wo_b.scale
mtp.0.attn.wq_a.scale
mtp.0.attn.wq_b.scale
mtp.0.attn_norm.weight
mtp.0.enorm.weight
mtp.0.e_proj.scale
mtp.0.ffn_norm.weight
mtp.0.hnorm.weight
mtp.0.norm.weight
mtp.0.hc_head_fn / base / scale
mtp.0.hc_attn_fn / base / scale
mtp.0.hc_ffn_fn / base / scale
mtp.0.ffn.gate.weight / bias
mtp.0.ffn.<routed-expert weights>.weight / scale
... (1,575 total, including 256 routed experts + the shared expert)
```

## MTP architecture (from the authoritative reference)

`reference/model.py:738-766` — `MTPBlock`:

```python
class MTPBlock(Block):
    def __init__(self, layer_id, args):
        super().__init__(layer_id, args)            # FULL Block: attention + MoE FFN
        self.e_proj = Linear(dim, dim)              # embedding projection
        self.h_proj = Linear(dim, dim)              # hidden projection
        self.enorm  = RMSNorm(dim)
        self.hnorm  = RMSNorm(dim)
        self.norm   = RMSNorm(dim)
        self.hc_head_fn / base / scale = Parameter(...)   # MTP-specific HC head
        self.embed: ParallelEmbedding = None        # SHARED with main Transformer
        self.head:  ParallelHead       = None       # SHARED with main Transformer

    def forward(self, x, start_pos, input_ids):
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)   # combine embed + prev hidden
        x = super().forward(x, start_pos, input_ids)        # full Block forward
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale,
                              self.hc_head_base, self.norm)
        return logits
```

Two crucial facts:

1. **MTPBlock IS a Block.** It inherits the full attention + 256-expert MoE
   FFN + HC infrastructure. There are no kernels to write that the engine
   doesn't already have.
2. **It shares the main model's `embed` and `head`.** No separate
   lm_head, no separate embedding table — the MTP block reuses what the main
   Transformer already has. Only the 6 extras (`e_proj`, `h_proj`, `enorm`,
   `hnorm`, `norm`) and the 3 HC-head tensors (`hc_head_fn/base/scale`) are
   new tensor families.

## Why the current sidecar is what it is

The story explains itself once you know two facts:

1. **The HF transformers DSv4 modeling class strips MTP at load time:**

   ```python
   _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
   ```

   Any conversion pipeline that uses `transformers.from_pretrained()`
   silently drops every MTP tensor. That's why `DSv4-Flash-256e-fixed.gguf`
   has zero MTP tensors despite the canonical release shipping 1,575 of them.

2. **The sidecar was built when the appliance pack contract didn't have a
   place for MTP tensors.** Rather than extend the contract, the original
   ds4.c team produced a separate small GGUF and a parallel loader. That
   separate file (`DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`) was
   hand-assembled from a partial source and contains only **32 tensors** —
   the HC-head matrices, the 6 extras, and a handful of attention
   projections. **It does not contain the routed experts or most of the
   attention LoRA factors.** Whatever serves today is running a
   heavily-truncated approximation of MTP, not the canonical block.

The sidecar therefore is two debts at once: a parallel quantization runtime
(Q4_K/Q8_0 vs the main path's FP8/MXFP4) **and** a degraded model artifact.
Removing it isn't just cleanup; it's a quality upgrade.

## Implementation plan

The work splits into three concrete sprints plus the speculative-decode-loop
work B1 was always about. Phases 1–3 below collapse the sidecar; the
speculative-decode loop is the actual B1 throughput lever.

### Phase 1 — Pack canonical MTP weights into the appliance contract

Convert the canonical `mtp.0.*` tensors from the safetensors cache into the
appliance's pack format and add them to the pack-index TSV.

- **Source:** `/models/deepseek-v4-flash-safetensors-cache/model-00046-of-00046.safetensors`
  for MTP-specific tensors, plus the relevant earlier shards for the shared
  embed and head (no change to those — just confirm they're already in the
  main GGUF, which they are).
- **Target format:** same FP8/MXFP4 + scales the main path already uses.
  The canonical safetensors are already at this quantization (e.g.
  `mtp.0.attn.wq_a.scale` exists), so this is a **format-shuffle, not a
  re-quantization**.
- **Pack tool:** extend `tools/ds4-v100-turbomind-pack.cu` and/or
  `tools/ds4-v100-appliance-pack.cu` to handle the `mtp.0.*` tensor
  namespace. The shard rules for `mtp.0.attn.*` and `mtp.0.ffn.experts.*`
  are identical to the rules for `layers.42.attn.*` and
  `layers.42.ffn.experts.*` (TP8 + EP8 with 32 routed experts per rank).
  The 9 new tensor families (`e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`,
  `hc_head_fn`, `hc_head_base`, `hc_head_scale`) need binding rules added —
  small dense tensors, replicated or sharded by feature dim depending on
  whether the engine's MTP forward consumes them rank-local.
- **Output:** a new pack contract file
  (`/workspace/logs/sprint-XXX-tp-ep-mtp-contract/contract/tp-ep-pack-contract.tsv`)
  that includes `mtp.0.*` entries.

This sprint produces a pack contract and verifies via `runtime_pack.cu`
that the binding succeeds (no kernel changes, no decode-path changes).
Tolerance gate against the prior promoted control.

### Phase 2 — Bind MTP tensors through the main weight-load path

Make `engine/runtime_pack.cu` and `engine/runtime_resources.cu` aware of the
layer-43 MTP block: allocate the per-rank attention + MoE expert buffers for
it, allocate the 9 extras, hold pointers consistent with the main layers.

- The 256 routed experts at layer 43 use the existing EP8 layout (32 per
  rank). The shared expert at layer 43 uses the existing replicated layout.
- The attention LoRA factors at layer 43 use the existing TP8 column/row
  shard rules.
- The 9 extras are small and probably replicated; finalize that during the
  sprint.
- The shared `embed` and `head` are already loaded by the main path —
  expose them to the MTP step via the engine API.

No decode-path changes yet. Phase 2 ends with "MTP weights are resident, the
sidecar is no longer referenced by the load path." The sidecar code (`engine/
mtp_sidecar.{c,h}`, the Q4_K/Q8_0 helpers in `ds4_pack.{c,h}` and
`ds4_source_formats.{c,h}`) gets deleted in this sprint's promote commit.

Tolerance gate: serving must still pass at promoted-control parity with MTP
loaded but not exercised.

### Phase 3 — Implement `MTPBlock.forward` in `engine/`

Add the actual MTP forward function. It runs as one additional sublayer the
decode loop can call when MTP is enabled.

The forward, in engine terms:

```c
// engine/mtp_step.cu (extends the existing file)
int run_mtp_step(...) {
    // prologue
    e = engine_embed(input_ids);          // shared with main
    e = rms_norm(e, enorm);
    x = rms_norm(x, hnorm);
    x = e_proj(e) + h_proj(x);            // small dense ops

    // body — reuses the existing Block forward primitives
    run_hc_current(...);                  // engine/hc_current.cu
    run_attention(...);                   // engine/attention_*.cu
    run_ep_compose(...);                  // engine/ep_compose.cu
    run_post_attention_ffn(...);          // engine/post_attention_ffn.cu

    // epilogue — MTP-specific HC head
    logits = run_mtp_head(x, hc_head_fn, hc_head_scale, hc_head_base, norm);
    return logits;
}
```

All sublayer calls reuse the existing kernels — `kernels/v100/norm.cuh`,
`kernels/v100/attention.cuh`, `kernels/v100/ep_compose.cuh`,
`kernels/v100/hc_mix.cuh`. No new kernel work.

Phase 3 ends with: MTP forward exists and produces correct logits at
single-step, gated behind an explicit `--mtp-serving-gate` (or env var).
The decode loop does not yet use it for speculative decoding.

Tolerance gate: with MTP enabled and run once per decode step,
selected-token agreement vs control ≥ 0.99.

### Phase 4 — The speculative-decode loop (the actual B1 throughput lever)

This is what the SPIKE_B_STEERING B1 entry was really about: use MTP to
verify K draft tokens per main-model step, raising effective batch size and
fixing the M < 1 token/expert problem.

- Run MTP forward to propose K draft tokens.
- Run main model forward on (prev + K drafts) in parallel.
- Accept-or-reject draft tokens by comparing main-model verify logits to
  draft probabilities.
- Advance position by 1 + accepted_k.
- Coordinate accepts/rejects across all 8 ranks so KV state stays consistent.

This is the cross-rank coordination work the TP/EP launcher currently
refuses MTP for. It is the largest piece of work in the B1 program and is
where the "MTP unsupported on TP/EP yet" stops being honest. **It depends
on phases 1–3 being done first** but is *not* unblocked by them — it's
a separate body of work on top.

## Sequencing relative to the rest of SPIKE B

In-place model tuning still goes first. The full sequence:

```
1. C5 sync-point reduction pass 2          ← in flight (sprint 529)
2. B2 compact EP variable-size NCCL compose
3. C1 piecewise graph capture
4. Tuning sprint (reference-shape perf, shape envelope, NCCL pinning, C4 spill)
5. MTP Phase 1 — pack canonical weights         ← this document
6. MTP Phase 2 — bind through main path; delete sidecar
7. MTP Phase 3 — MTPBlock.forward
8. MTP Phase 4 — speculative-decode loop (B1)
```

Phases 5–7 are sprint-sized cleanups, each with a tolerance gate against the
prior promoted control. Phase 4 is the actual throughput sprint, and it's
where the per-sprint perf measurement opt-in (per the validation policy)
applies — the gate is whether speculative-decode acceptance × verify-cost
beats baseline tok/s.

## Quality implication of fixing this

The current sidecar runs a stripped-down MTP — 32 of 1,575 canonical tensors
— so draft predictions are made from a partial network missing the routed
experts and most of the attention. Whatever quality the current `DS4_V100_MTP_SERVING=on`
path gets is from running on the wrong model. After Phase 3, MTP serves from
the full canonical MTPBlock at the same quantization as the main path. Draft
acceptance rate should improve materially — which is exactly the variable
that determines whether speculative decoding pays off in Phase 4.

## Files affected (forward-looking inventory)

**Add / extend:**

- `tools/ds4-v100-turbomind-pack.cu` — extend for `mtp.0.*` namespace
- `tools/ds4-v100-appliance-pack.cu` — same
- A new pack contract TSV
- `engine/runtime_pack.cu` — bind the new tensor families
- `engine/runtime_resources.cu` — allocate per-rank buffers for layer 43
- `engine/runtime_options.cuh` — promote MTP defaults
- `engine/mtp_step.cu` — implement `MTPBlock.forward`
- `engine/api.h` — expose the MTP step to the appliance
- `appliance/main.cu` — call into the MTP step from the decode loop (Phase 4)

**Delete (in Phase 2's promote commit):**

- `engine/mtp_sidecar.c`
- `engine/mtp_sidecar.h`
- The Q4_K and Q8_0 paths in `ds4_pack.c`, `ds4_pack.h`,
  `ds4_source_formats.c`, `ds4_source_formats.h` (verify no other consumer)
- The standalone MTP smokes in `smokes/mtp-*.{c,cu}` that target the sidecar
  (they were already flagged as "smokes that grew up" debt; this is when
  they retire)
- The sidecar source file `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` becomes
  unreferenced (operator-side cleanup, not a code change)

**Out of scope here:**

- TP/EP MTP launcher refusal removal (lifted in Phase 3 or 4 by the same
  sprint that lands the working integration).
- Any speculative-decode tuning (K choice, acceptance threshold) — that's
  Phase 4 sprint-internal work.

## One-line summary

The canonical MTP weights are 1,575 tensors at `mtp.0.*` already on the pod
at FP8/MXFP4 with scales; the local appliance GGUF stripped them via the HF
loader gotcha, and the sidecar runs a 32-tensor truncated approximation.
Real MTP integration is three sprint-sized binding/forward-pass steps
(Phases 1–3) plus the speculative-decode loop (Phase 4), all sequenced
behind C5/B2/C1/Tuning.
