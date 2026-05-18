# SPRINT-031 Report

## Verdict

`SHIP`

## Summary

Sprint 031 shipped the V100 MTP resident sidecar bridge:

- `ds4_mtp_sidecar_inspect` now exposes typed MTP tensor descriptors with
  dtype, shape, GGUF offset, byte length, kernel family, and compact resident
  offset.
- `ds4_v100_mtp_sidecar_open` owns the MTP fd/mmap separately from the base
  model, allocates a gpu7 device arena, uploads all sidecar tensors, and
  spot-checks resident bytes.
- `tools/ds4-v100-gate.sh --mtp-model` now runs both sidecar validation and
  sidecar residency before reporting `missing=mtp_forward`.

Speculative decode remains disabled. The remaining MTP work is now the forward
path itself: Q8_0 e/h projections, MTP attention and KV state, Q4_K routed
experts, output-head logits, and draft/verify/rollback correctness.

## V100 Evidence

Standalone residency smoke:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ./tools/ds4-v100-mtp-residency-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096
```

Residency result:

```text
mtp_runtime	gpu	7
mtp_runtime	arena_kind	device
mtp_runtime	arena_bytes	3807601408
mtp_runtime	uploaded_tensors	32
mtp_runtime	uploaded_bytes	3807600108
mtp_runtime	spot_checks	60
mtp_runtime	free_after_upload_bytes	29937369088
mtp_runtime	PASS	resident_sidecar=1
reserve_bytes	4294967296
mtp_residency_smoke	PASS
```

Full gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-031-GATE-CLUSTER-8GPU-FULL
```

Readiness:

```text
gate	mtp_sidecar	PASS
gate	mtp_residency	PASS
gate	scheduler_output_head	PASS
gate	v100_replay_tool	PASS
gate	v100_appliance_http	PASS
gate	readiness	NOT_READY	missing=mtp_forward
gate	summary	PASS	failures=0 ready=false
```

Replay timing from the same full gate:

```text
prompt_tokens=18
generated_tokens=2
first token id=926 hex=3136 text=16 logit=35.250885
second token id=1 text=<|end of sentence|> logit=39.3052406
open_total_ms=279591.369
prompt_replay_ms=3527.238
continuation_decode_ms=148.968
output_head_ms=6.735
prompt_tokens_per_second=5.103144
continuation_tokens_per_second=6.712850
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

- `docs/sprints/drafts/SPRINT-031-MTP-RESIDENCY/`
- `docs/sprints/drafts/SPRINT-031-GATE-CLUSTER-8GPU-FULL/`

## Interpretation

The MTP sidecar is no longer just a validated file. It is a resident V100
runtime asset with a stable tensor directory and arena offsets. The bridge is
intentionally separate from the main V100 pack/layer-state path because the MTP
sidecar uses Q4_K routed experts and Q8_0 dense tensors, while the main model
path is built around source-layout MXFP4/F8/BF16 descriptors.

The next sprint should implement a K=1 MTP forward probe against this resident
sidecar object and compare draft logits/tokens before enabling speculative
serving.
