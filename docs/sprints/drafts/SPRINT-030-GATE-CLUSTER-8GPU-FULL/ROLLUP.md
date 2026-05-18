# Sprint 030 Full Gate Rollup

Command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-gate.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-030-GATE-CLUSTER-8GPU-FULL
```

Result:

```text
gate	mtp_sidecar	PASS
gate	scheduler_output_head	PASS
gate	v100_replay_tool	PASS
gate	v100_appliance_http	PASS
gate	readiness	NOT_READY	missing=mtp_runtime
gate	summary	PASS	failures=0 ready=false
```

Key sidecar facts:

```text
architecture=deepseek4_mtp_support
tensors=32
described_tensor_bytes=3807600108
f32=19 tensors / 7691756 bytes
q8_0=10 tensors / 176029696 bytes
q4_k=3 tensors / 3623878656 bytes
```

Interpretation:

The sidecar is valid and the appliance baseline remains green. Readiness is
still false because the V100 runtime does not yet execute the MTP draft/verify
path.
