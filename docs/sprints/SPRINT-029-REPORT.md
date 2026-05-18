# SPRINT-029 Report

## Verdict

`SHIP`

## Summary

Sprint 029 shipped the first resident HTTP surface for the V100 appliance:

- `ds4_v100_stage_scheduler_reset` resets scheduler-owned mutable KV, compressed
  state, indexer state, and HC buffers.
- `ds4_v100_replay_reset` resets all eight resident stages and clears the
  replay runtime's one-shot guard.
- `tools/ds4-v100-replay --serve` exposes a narrow loopback endpoint for
  deterministic one-slot selected-token/greedy replay.
- `tools/ds4-v100-appliance-smoke.sh` validates the endpoint without requiring
  `curl` or `python3` in the CUDA pod.
- `tools/ds4-v100-gate.sh` now treats `public_serving` as a measured readiness
  item.

This is intentionally minimal serving. It is not a production API, it is not
concurrent, and it is not streaming. It is the first deployed-process shape that
keeps the model resident and accepts an HTTP request.

## V100 Evidence

Full gate HTTP smoke:

```text
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/ds4-v100-appliance-smoke.sh \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
  --tokens 1 \
  --requests 2 \
  --expected-token-hex 3136 \
  --host 127.0.0.1 \
  --port 18080 \
  --log-dir docs/sprints/drafts/SPRINT-029-GATE-CLUSTER-8GPU/v100_appliance_http
```

Observed request 1:

```text
prompt_tokens=18
generated_tokens=1
first token id=926 hex=3136 text=16 logit=35.250885
open_total_ms=311299.886
prompt_replay_ms=3601.463
output_head_ms=4.677
total_ms=3606.161
prompt_tokens_per_second=4.997969
uploaded_bytes=156142862684
```

Observed request 2 after reset in the same resident process:

```text
prompt_tokens=18
generated_tokens=1
first token id=926 hex=3136 text=16 logit=35.250885
open_total_ms=311299.886
prompt_replay_ms=2619.038
output_head_ms=3.038
total_ms=2622.088
prompt_tokens_per_second=6.872753
uploaded_bytes=156142862684
```

Full gate replay timing:

```text
prompt_tokens=18
generated_tokens=2
first token id=926 hex=3136 text=16 logit=35.250885
second token id=1 text=<|end of sentence|> logit=39.3052406
open_total_ms=233228.202
prompt_replay_ms=3487.222
continuation_decode_ms=149.479
output_head_ms=8.114
total_ms=3644.844
prompt_tokens_per_second=5.161702
continuation_tokens_per_second=6.689903
uploaded_bytes=156142862684
```

Full gate readiness:

```text
gate	v100_appliance_http	PASS
gate	readiness	NOT_READY	missing=mtp
gate	summary	PASS	failures=0 ready=false
```

Artifacts:

- `docs/sprints/drafts/SPRINT-029-GATE-CLUSTER-8GPU/`

## Readiness

After Sprint 029, `public_serving` is no longer a readiness blocker for the
minimal appliance definition. `mtp` remains the only missing readiness item.

## Interpretation

The serving process itself is now proven, but startup remains expensive:
opening/uploading all resident stages still takes roughly 5 minutes. The next
performance work should parallelize stage upload and then run longer decode
baselines from the already-resident process.
