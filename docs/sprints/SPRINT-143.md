# Sprint 143 - Prefill/Decode Metric Split

Date: 2026-05-21

## Objective

Make future throughput experiments report prefill and decode separately. Recent
kernel probes can be hidden by aggregate generated tok/s, so each benchmark
artifact should expose prompt replay/prefill rate, continuation decode rate,
and aggregate generated rate side by side.

## Implementation

- Extended `tools/ds4-v100-appliance-soak.sh` summary output with:
  - `aggregate_prompt_tokens_per_second`
  - `prompt_tokens`
  - `prefill_prompt_replay_ms_avg`
  - `prompt_response_tokens_per_second_avg`
  - `continuation_decode_ms_avg`
  - `continuation_response_tokens_per_second_avg`
  - `generated_response_tokens_per_second_avg`
- Extended each appliance soak response row with prompt and continuation timing
  fields.
- Made appliance soak tolerate serial one-slot latency configs where
  `async_pipeline_decode=false`; timed responses only require `async_pipeline`
  timing when the server reports the async path active.
- Extended `tools/ds4-v100-sustained-decode-bench.sh` with aggregate prompt
  tok/s in JSON and TSV reports.
- Extended `tools/ds4-v100-aggregate-throughput.sh` with aggregate prompt and
  continuation tok/s in JSON and TSV reports.

## V100 Validation

Shell checks:

```text
bash -n tools/ds4-v100-appliance-soak.sh \
  tools/ds4-v100-sustained-decode-bench.sh \
  tools/ds4-v100-aggregate-throughput.sh

git diff --check
```

Real V100 one-request metrics smoke:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_GATE_UP_PROBE=auto
DS4_V100_TURBOMIND_DOWN_PROBE=off
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
tools/ds4-v100-appliance-soak.sh \
  --appliance-dir /workspace/ds4-appliance-full-tm-gated-s127 \
  --ctx 32768 --slots 1 --active-microbatch 1 \
  --tokens 2 --requests 1 --warmup-requests 0 \
  --async-pipeline-mode off
```

Result:

```text
aggregate_prompt_tokens_per_second=6.841274
aggregate_generated_tokens_per_second=0.760142
aggregate_continuation_tokens_per_second=0.380071
prefill_prompt_replay_ms_avg=2476.522
continuation_decode_ms_avg=71.761
prompt_response_tokens_per_second_avg=7.268257
continuation_response_tokens_per_second_avg=13.935101
token_match=1/1
```

Artifact:

- `logs/from-cluster/sprint143-prefill-decode-metrics-smoke/`

## Related Profile

The Sprint 142 follow-up 128-slot routed-FFN profile still points at packed
MXFP4 GEMM dataflow rather than route plumbing:

| Bucket | Share |
|---|---:|
| gated gate/up GEMM | about 55-60% |
| down GEMM | about 29-30% |
| route/gather/scatter combined | small single-digit to low-teens share |

## Decision

Ship the metric split. Future A/B tables should include at least:

- aggregate prompt/prefill tok/s
- aggregate continuation decode tok/s
- aggregate generated tok/s
- response-local prompt and continuation rates when useful

This does not change the runtime path. It improves visibility for the next
kernel sprint.
