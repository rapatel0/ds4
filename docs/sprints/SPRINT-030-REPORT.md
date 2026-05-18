# SPRINT-030 Report

## Verdict

`SHIP`

## Summary

Sprint 030 shipped the MTP sidecar readiness gate:

- `ds4_mtp_sidecar_report` validates the actual DS4 MTP companion GGUF through
  the existing `ds4.c` model parser and strict `mtp.0.*` tensor binder.
- `tools/ds4-v100-mtp-sidecar-gate` emits a durable dtype/shape/bytes/kernel
  contract for the sidecar.
- `tools/ds4-v100-gate.sh --mtp-model` now distinguishes a valid sidecar from
  the still-missing MTP runtime implementation.

Speculative decode remains disabled in the V100 appliance path. The remaining
work is now specifically `mtp_runtime`: binding/uploading the sidecar tensors
into V100 resident arenas, implementing the MTP forward pass, and validating
draft/verify/rollback correctness.

## V100 Evidence

Full gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-gate.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-030-GATE-CLUSTER-8GPU-FULL
```

Readiness:

```text
gate	mtp_sidecar	PASS
gate	scheduler_output_head	PASS
gate	v100_replay_tool	PASS
gate	v100_appliance_http	PASS
gate	readiness	NOT_READY	missing=mtp_runtime
gate	summary	PASS	failures=0 ready=false
```

Sidecar inventory:

```text
architecture: deepseek4_mtp_support
file_bytes: 3807602400
described_tensor_bytes: 3807600108
f32: 19 tensors, 7691756 bytes
q8_0: 10 tensors, 176029696 bytes
q4_k: 3 tensors, 3623878656 bytes
```

Key tensors:

```text
mtp.0.e_proj.weight        q8_0 [4096,4096]       17,825,792 bytes
mtp.0.h_proj.weight        q8_0 [4096,4096]       17,825,792 bytes
mtp.0.ffn_gate_exps.weight q4_k [4096,2048,256] 1,207,959,552 bytes
mtp.0.ffn_up_exps.weight   q4_k [4096,2048,256] 1,207,959,552 bytes
mtp.0.ffn_down_exps.weight q4_k [2048,4096,256] 1,207,959,552 bytes
```

Replay timing from the same full gate:

```text
prompt_tokens=18
generated_tokens=2
first token id=926 hex=3136 text=16 logit=35.250885
second token id=1 text=<|end of sentence|> logit=39.3052406
open_total_ms=243489.833
prompt_replay_ms=3455.847
continuation_decode_ms=144.044
output_head_ms=6.805
prompt_tokens_per_second=5.208564
continuation_tokens_per_second=6.942314
uploaded_bytes=156142862684
```

HTTP smoke from the same full gate:

```text
requests=2
prompt_tokens=18
generated_tokens=1
first_token=926
first_hex=3136
```

Artifacts:

- `docs/sprints/drafts/SPRINT-030-GATE-CLUSTER-8GPU-FULL/`

## Interpretation

This sprint did not make MTP faster or correct yet. It made the MTP problem
concrete inside the appliance:

- The sidecar format is valid and exactly known.
- The memory footprint is dominated by three Q4_K routed expert tensors.
- Dense MTP projections and shared experts are Q8_0.
- Control, norms, HC functions, and routing bias are F32.
- The full appliance baseline remains green with the sidecar gate enabled.

The next sprint should implement a V100 MTP runtime milestone around K=1 draft
parity, not jump directly to speculative serving.
