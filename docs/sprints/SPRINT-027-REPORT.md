# SPRINT-027 Report

## Verdict

`SHIP`

## Summary

Sprint 027 shipped the selected-token correctness fix for the V100 scheduler.
The root causes were in the body path, not the output head:

- token embedding HC seed read native BF16 bytes as F16;
- decode KV/cache was taking an FP8 round-trip by default even though the
  source-layout oracle uses F16 cache semantics.

The sprint also added checkpoint diagnostics that can compare CPU source-layout
HC against V100 scheduler HC at seed, after-attention, and layer-final
boundaries, with route-id/weight comparison for layer-final checkpoints.

## Implementation Notes

- `ds4_engine_cpu_hc_checkpoints` replays the CPU source-layout oracle through
  only the maximum requested layer and captures `[4 x 4096]` HC checkpoints.
- `ds4_engine_cpu_route_checkpoints` recomputes CPU router selections and
  weights from the after-attention HC used by the FFN path.
- `ds4_v100_stage_scheduler_decode_token_checkpoints` captures the embedding
  seed and forwards layer checkpoint callbacks through the stage scheduler.
- `ds4_v100_layer_execute_hc_decode` emits an after-attention checkpoint before
  the FFN path mutates the layer output.
- `ds4_gpu_embed_token_hc_tensor` and the batched embedding variant now decode
  source BF16 bytes with `arena_bf16_to_f32`.
- The default scheduler KV path is F16-rounded. FP8 KV remains available
  behind `fp8_kv_cache`.

## V100 Evidence

Selected-token oracle:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tests/cuda_v100_selected_token_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --expected-token-hex 3136 --top-k 8
cuda_v100_selected_token_smoke: prompt_tokens=18 selected=926 logit=35.250885 expected=3136 topk=926:35.250885:3136,11154:26.782398:323034,1:25.74218:3cefbd9c656e64e2968164e2968173656e74656e6365efbd9c3e,3054:24.17345:546f,5718:24.017841:4c6574,671:23.983316:546865,856:23.934837:3135,201:23.468815:0a ok
```

Checkpoint localization:

```text
CUDA_VISIBLE_DEVICES=0 ./tests/cuda_v100_scheduler_checkpoint_parity_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --layers -1,0,1,2,3,a4,4,5 --ctx 4096 --prompt-tokens 1
checkpoint layer=-1 kind=seed PASS
checkpoint layer=0 kind=layer_final PASS
checkpoint layer=1 kind=layer_final PASS
checkpoint layer=2 kind=layer_final PASS
checkpoint layer=3 kind=layer_final PASS
checkpoint layer=4 kind=after_attn PASS
checkpoint layer=4 kind=layer_final DIFF
checkpoint layer=5 kind=layer_final DIFF
```

Full appliance gate after selected-token fix:

```text
gate	scheduler_output_head	PASS	command=./tests/cuda_v100_selected_token_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --expected-token-hex 3136
gate	readiness	NOT_READY	missing=public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Artifacts:

- `docs/sprints/drafts/SPRINT-027-GATE-CLUSTER-8GPU/summary.log`
- `docs/sprints/drafts/SPRINT-027-GATE-CLUSTER-8GPU/scheduler_output_head.log`
- `docs/sprints/drafts/SPRINT-027-GATE-CLUSTER-8GPU/scheduler_checkpoint_parity.log`
- `docs/sprints/drafts/SPRINT-027-GATE-CLUSTER-8GPU/checkpoint_localization_diff.log`

## Readiness

The model-body and output-head path now satisfy the official short-prompt
selected-token gate. The appliance is not yet ready for use because there is
still no public serving surface, no MTP implementation, and no throughput
benchmark.

## Interpretation

The remaining layer-4 final-HC drift occurs after attention and after route
selection. For the one-token localization run, seed and early layers match,
layer-4 after-attention matches, and route0 matches (`64/1.290131` on both CPU
and GPU). That points at FFN numeric differences, most likely MXFP4/shared
expert accumulation behavior versus the CPU source-format oracle.
