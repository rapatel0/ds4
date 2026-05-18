# SPRINT-028 Report

## Verdict

`SHIP`

## Summary

Sprint 028 moved the working selected-token scheduler orchestration out of test
code and into a reusable runtime layer:

- `ds4_v100_replay_open` owns tokenizer metadata, model mapping, model fd
  registration, and the eight resident stage schedulers.
- `ds4_v100_replay_generate` replays prompt tokens and greedy continuation
  tokens through the 8x V100 body.
- `tools/ds4-v100-replay` exposes that path as a CUDA-only command with JSON
  token, timing, and memory output.

This gives us the first appliance-shaped entrypoint and a throughput/timing
baseline. It is still not a network server.

## V100 Evidence

Replay command:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-replay --index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv --model /models/DSv4-Flash-256e-fixed.gguf --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt --tokens 2 --expected-token-hex 3136 --json
```

Observed baseline from the full Sprint 028 gate:

```text
prompt_tokens=18
generated_tokens=2
first token id=926 hex=3136 text=16 logit=35.250885
second token id=1 text=<|end of sentence|> logit=39.3052406
open_total_ms=280822.215
prompt_replay_ms=3536.169
continuation_decode_ms=143.322
output_head_ms=6.733
total_ms=3686.246
prompt_tokens_per_second=5.090255
continuation_tokens_per_second=6.977295
uploaded_bytes=156142862684
```

Full gate evidence is stored under:

- `docs/sprints/drafts/SPRINT-028-GATE-CLUSTER-8GPU/`

## Readiness

After Sprint 028, `throughput_benchmark` should no longer be a missing
readiness item when the replay tool passes. `public_serving` remains missing
because the shipped artifact is a command-line appliance surface, not an HTTP
or long-running process endpoint. `mtp` remains deferred.

## Interpretation

The biggest immediate performance issue is not the decode step; it is opening
the eight resident stages and uploading roughly 156 GB of packed weights. The
first baseline saw about 281 seconds of open/upload time versus about 149 ms
for one continuation decode step after the prompt. The next serving sprint
should keep the scheduler resident across requests instead of reopening it.
