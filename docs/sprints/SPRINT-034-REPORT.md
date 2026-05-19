# SPRINT-034 Report

## Verdict

`SHIP`

Sprint 034 shipped the resident MTP prefix composition probe. The MTP sidecar
gpu7 arena can now supply F32 norm weights and Q8_0 projection weights for the
native prefix sequence:

`enorm -> e_proj -> HC repeat -> hnorm -> h_proj -> add -> mtp_input_hc`.

The full V100 appliance gate still reports `ready=false` with only
`missing=mtp_forward`, which is expected because this sprint does not implement
the dense MTP block, MTP output logits, or draft/verify/rollback.

## Implementation

- Added `ds4_v100_mtp_sidecar_f32_vector_view` for validated 1D F32 sidecar
  tensors.
- Added `ds4_gpu_arena_f32_rms_norm_f32` with CUDA and fail-closed stub
  implementations.
- Extended `tools/ds4-v100-mtp-prefix-smoke` to validate:
  - exact CUDA-vs-CUDA Q8_0 parity for `mtp.0.e_proj.weight`;
  - exact CUDA-vs-CUDA Q8_0 parity for `mtp.0.h_proj.weight`;
  - resident F32 `enorm` and `hnorm` against a CPU RMSNorm reference;
  - full prefix-chain composition against an independent CPU F32/Q8_0
    reference.

## Focused V100 Evidence

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 ./tools/ds4-v100-mtp-prefix-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-034-MTP-PREFIX-CHAIN/mtp_prefix.report
```

Result: `mtp_prefix_smoke PASS`.

Key evidence from
`docs/sprints/drafts/SPRINT-034-MTP-PREFIX-CHAIN/mtp_prefix.report`:

- MTP sidecar resident arena: `3807601408` bytes on gpu7.
- Uploaded tensors: `32`; uploaded bytes: `3807600108`.
- Free after upload: `29937369088` bytes, above the 4096 MiB reserve.
- `mtp.0.e_proj.weight`: `max_abs=0`, `max_rel=0` against existing CUDA Q8_0.
- `mtp.0.h_proj.weight`: `max_abs=0`, `max_rel=0` against existing CUDA Q8_0.
- `enorm`: `max_abs=2.23517418e-08`.
- `hnorm_hc`: `max_abs=4.47034836e-08`.
- `mtp_input_hc`: `max_abs=0.00392527878` against CPU Q8_0 reference under
  the explicit `cpu_q8_abs_tol=0.01`.

The CPU Q8_0 reference is not expected to be bit-exact with the CUDA Q8_0 path
because accumulation order differs. The exact Q8_0 resident proof remains the
CUDA-vs-CUDA projection parity lines.

## Full Gate Evidence

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-034-GATE-CLUSTER-8GPU
```

Result:

- `gate mtp_prefix PASS`
- `gate v100_replay_tool PASS`
- `gate v100_appliance_http PASS`
- `gate v100_appliance_http_long PASS`
- `gate readiness NOT_READY missing=mtp_forward`
- `gate summary PASS failures=0 ready=false`

Replay timing from `v100_replay_tool.log`:

- `open_total_ms=230837.504`
- `prompt_replay_ms=3486.792`
- `continuation_decode_ms=142.928`
- generated tokens: `3136`, then EOS
- generated tokens/sec: `0.550017`

Long HTTP smoke:

- request 1: `generated_tokens=2`, first token hex `3136`,
  `continuation_ms=145.205`
- request 2: `generated_tokens=2`, first token hex `3136`,
  `continuation_ms=144.025`

## Notes

- Sprint 034 does not change readiness beyond the prefix sub-surface. Level 3
  remains incomplete until the MTP block produces logits/top-k and draft state
  can be verified/rolled back safely.
- The generic CUDA model-map range cache showed fragile behavior with repeated
  tiny malloc-backed sidecar copies during smoke development. The shipped
  prefix-chain reference avoids relying on that cache for F32 norms.
