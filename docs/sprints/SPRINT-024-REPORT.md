# SPRINT-024 Report

## Verdict

`SHIP`

## Summary

Sprint 024 shipped the full 8-stage resident scheduler smoke. The runtime now
opens all stage arenas, executes layers 0-42 across gpu0-gpu7, peer-copies HC
between every stage, and produces a finite nonzero final HC state on gpu7.

## Implementation Notes

- Added `tests/cuda_v100_full_scheduler_smoke.c`.
- Added Makefile build/stub/clean coverage for the new CUDA smoke.
- Added `full_scheduler` to `tools/ds4-v100-gate.sh`.
- Made readiness dynamic: `full_43_layer_scheduler` is removed only when the
  full-chain gate passes.

## V100 Evidence

Standalone full-chain run:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 layers=43 last=40-42 gpu=7 uploaded_tensors=1328 uploaded_bytes=156142862684 expert_last=75 ok
```

Full appliance gate:

```text
gate	full_scheduler	PASS	command=./tests/cuda_v100_full_scheduler_smoke --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --token 16 --position 16
gate	readiness	NOT_READY	missing=real_model_selected_token,public_serving,mtp,throughput_benchmark
gate	summary	PASS	failures=0 ready=false
```

Observed during standalone and gate runs, resident memory stayed below the
32 GB V100 limit. The largest observed stage was gpu0 at about 21.8 GiB; middle
stages were about 20.8 GiB, and later stages were lower.

## Remaining Gaps

- Final HC is not yet collapsed through `hc_head_*`, `output_norm.weight`, and
  BF16 `output.weight`.
- No selected-token comparison against the source oracle yet.
- No server, MTP, throughput benchmark, or multi-slot scheduler.
