# Sprint 035 Report: Resident MTP Q4_K Routed Experts

## Verdict

`SHIP`.

Sprint 035 adds the resident V100 Q4_K routed expert primitive needed by the
MTP FFN path. The full gate now proves:

- MTP sidecar residency on gpu7.
- Resident F32/Q8_0 prefix composition from Sprint 034.
- Resident Q4_K gate/up/down routed expert execution for one token and six
  experts.
- Base selected-token replay and loopback HTTP serving remain green.

The gate remains conservative and still reports `ready=false
missing=mtp_forward` because full MTP block execution, logits/top-k, and
draft/verify/rollback are not implemented yet.

## Implementation

- Added `ds4_gpu_q4_k_expert_view` and
  `ds4_v100_mtp_sidecar_q4_k_expert_view()` for typed resident 3D Q4_K expert
  tensors.
- Added `ds4_gpu_arena_q4_k_routed_moe_one_f32()` in CUDA. It bypasses
  `cuda_model_range_ptr()` and resolves weights directly from
  `arena->ptr + resident_offset`.
- Reused the existing V100 Q4_K decode kernels:
  - F32 input -> Q8_K activation quantization.
  - Q4_K gate/up dot products.
  - clamp + SwiGLU + router weight.
  - mid -> Q8_K quantization.
  - direct six-expert Q4_K down-sum.
- Added `tools/ds4-v100-mtp-q4k-smoke.c`, Makefile target, and gate step
  `mtp_q4k`.
- Updated MTP sidecar reporting so Q4_K routed expert tensors are labeled
  `v100_q4_k_routed_moe` instead of pending.

## Evidence

Focused V100 smoke:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 900 ./tools/ds4-v100-mtp-q4k-smoke \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --gpu 7 --require-gpus 8 --reserve-mib 4096 \
  --report docs/sprints/drafts/SPRINT-035-MTP-Q4K/mtp_q4k.report
```

Result:

```text
mtp_q4k_smoke PASS
mtp_q4k_routed arena_ms=3.476 reference_ms=38.194 max_abs=1.43051147e-06 tol=0.05 PASS
```

Full V100 gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2700 ./tools/ds4-v100-gate.sh --build \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 --slots 1 \
  --log-dir docs/sprints/drafts/SPRINT-035-GATE-CLUSTER-8GPU
```

Result:

```text
gate mtp_q4k PASS
gate v100_replay_tool PASS
gate v100_appliance_http PASS
gate v100_appliance_http_long PASS
gate readiness NOT_READY missing=mtp_forward
gate summary PASS failures=0 ready=false
```

Replay timing from the full gate:

```text
prompt_replay=3432.057 ms
continuation_decode=144.740 ms
generated_tokens_per_second=0.557913
first token hex=3136
```

## Notes

- The new API is intentionally decode-only for now: one token, six selected
  experts, 256 resident experts.
- The smoke CPU reference is local to the tool because the CPU runtime did not
  already have a Q4_K routed expert path.
- The Q4_K primitive is not the full MTP FFN block. Router selection, shared
  Q8_0 expert, HC post, MTP attention, logits/top-k, and draft rollback remain
  separate integration work.
