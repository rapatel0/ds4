# Sprint 052 Report: Sustained Decode And Utilization Baseline

## Result

`SHIP`.

## Changes Implemented

1. Added `tools/ds4-v100-sustained-decode-bench.sh`.
   - Runs multi-token request cases against the resident replay service.
   - Supports context, slot, and queue-policy matrices.
   - Separates aggregate generated tok/s from aggregate continuation tok/s.
   - Captures replay timing averages, per-stage decode timings, and handoff
     timings from response JSON.
   - Captures `nvidia-smi` samples when available.
2. Added optional sustained decode gate profile support to
   `tools/ds4-v100-gate.sh`.
   - `--sustained-profile off|smoke|full`
   - Explicit overrides for sustained context tiers, slot tiers, policies,
     requests, tokens, warmup requests, host, port base, and sample interval.
   - Default readiness behavior remains unchanged because the sustained profile
     defaults to `off`.
3. Updated `docs/operations/DS4-V100-APPLIANCE.md` with sustained decode
   commands, profile defaults, and artifact descriptions.
4. Added cluster evidence under
   `logs/from-cluster/sprint052-sustained-baseline`.

## Validation

Local:

```bash
bash -n tools/ds4-v100-sustained-decode-bench.sh
bash -n tools/ds4-v100-gate.sh
tools/ds4-v100-sustained-decode-bench.sh --help
tools/ds4-v100-gate.sh --help
```

Cluster build:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint052 &&
  CUDA_ARCH=sm_70 make tools/ds4-v100-replay
'
```

## Cluster Execution

Executed on `llamacpp-build-8gpu` (`gpu-01`) with all eight V100s visible and
idle before the run:

```bash
kubectl -n llm exec llamacpp-build-8gpu -- bash -lc '
  cd /workspace/ds4-sprint052 &&
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  timeout 900 bash ./tools/ds4-v100-sustained-decode-bench.sh \
    --model /models/DSv4-Flash-256e-fixed.gguf \
    --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
    --prompt-file tests/test-vectors/prompts/short_reasoning_plain.txt \
    --ctx-tiers 1048576 \
    --slot-tiers 1 \
    --queue-policies sequential \
    --tokens 16 \
    --requests 4 \
    --warmup-requests 1 \
    --expected-token-hex 3136 \
    --sample-ms 500 \
    --log-dir logs/sprint052-sustained-baseline
'
```

Result:

- Status: `PASS`
- Context: `1048576`
- Slots: `1`
- Timed requests: `4`
- Generated tokens per request: `16`
- Token correctness: `4/4`, first token bytes `3136`
- Aggregate generated tok/s: `3.304551`
- Aggregate continuation tok/s: `3.098017`
- Average per-response continuation tok/s: `6.869750`
- Average request latency: `4841.506 ms`
- p95 request latency: `4846.450 ms`
- Average GPU utilization: `10.804%`
- Max GPU utilization: `22.000%`
- Max memory used: `23594 MiB`

Artifacts:

- `logs/from-cluster/sprint052-sustained-baseline/sustained_decode.tsv`
- `logs/from-cluster/sprint052-sustained-baseline/sustained_decode.json`
- `logs/from-cluster/sprint052-sustained-baseline/cases/case_1_ctx1048576_s1_sequential_tok16/result.json`
- `logs/from-cluster/sprint052-sustained-baseline/cases/case_1_ctx1048576_s1_sequential_tok16/gpu_util.csv`

## Remaining Gap

The baseline confirms the practical-use problem: sustained decode is now being
measured honestly, but the service still does not keep the V100s busy. The next
sprint should implement continuous token-step batching across active slots so
multi-token requests can remain resident and advance together instead of
falling back to mostly serial per-request generation.
