# SPRINT-026 Report

## Verdict

`SHIP`

## Summary

Sprint 026 added output-head diagnostics for the selected-token mismatch from
Sprint 025. The new deterministic HC parity smoke proves that the V100
output-head adapter matches the CPU reference for the final collapse and BF16
vocab projection path.

The official prompt replay still fails the expected-token check. Since the
isolated output-head path now matches, the remaining divergence should be
localized inside the 43-layer body: attention/cache updates, compressor/indexer
row selection, routing, MoE execution, residual/norm ordering, or cross-stage
HC state.

## Implementation Notes

- `ds4_v100_stage_scheduler_write_hc` lets tests inject a known HC tensor into
  a stage scheduler.
- `ds4_v100_stage_scheduler_select_topk` returns top-k token ids and logits.
- `ds4_v100_stage_scheduler_select_token` remains the top-1 compatibility
  wrapper.
- `cuda_v100_output_head_parity_smoke` compares CPU and V100 output-head top-5
  for a deterministic HC vector.
- `cuda_v100_selected_token_smoke --top-k N` prints selected token ids, logits,
  and token bytes while preserving oracle failure behavior.

## V100 Evidence

Full appliance gate:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 3600 ./tools/ds4-v100-gate.sh --build --cuda-arch sm_70 --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --log-dir /tmp/ds4-sprint026-gate-full
gate	full_scheduler	PASS	command=./tests/cuda_v100_full_scheduler_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --token 16 --position 16
gate	output_head_parity	PASS	command=./tests/cuda_v100_output_head_parity_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf
gate	scheduler_output_head	PASS	command=./tests/cuda_v100_selected_token_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt
gate	readiness	NOT_READY	missing=real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Output-head parity:

```text
ds4: CUDA loading model tensors into device cache
cuda_v100_output_head_parity_smoke: cpu_top1=83253 gpu_top1=83253 cpu_logit=21.160488 gpu_logit=21.160488 ok
```

Prompt replay with explicit oracle and top-k:

```text
cuda_v100_selected_token_smoke: selected token mismatch expected=3136 got=0a0a token=271 logit=20.014683
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=271 logit=20.014683 expected=3136 topk=271:20.014683:0a0a,6328:16.67861:0a0a0a,1613:16.59844:2e2e2e,15:16.51009:2d,223:15.869394:20,16:15.587646:2e,426:15.115372:2e2e,743:15.038074:e68890 FAIL
```

## Readiness

The appliance is still not ready for use. `real_model_selected_token` remains a
blocking correctness gap. Serving, MTP, and throughput should stay behind that
gate.

## Interpretation

The output-head adapter is not the current blocker. The top-k list is dominated
by punctuation/newline-like tokens, and the expected token bytes `3136` are not
near the top of the observed logits. The next useful implementation is an
automated stage/layer checkpoint harness that compares V100 HC against a CPU
source-layout replay for the same prompt.
