# Sprint 051 Report: Gate Aggregate Matrix Profiles

## Result

`SHIP`.

## Changes Implemented

1. Added aggregate profile selection to the full gate:
   - `--aggregate-profile fast|full`
2. Added matrix/request override knobs:
   - `--aggregate-ctx-tiers`
   - `--aggregate-slot-tiers`
   - `--aggregate-queue-policies`
   - `--aggregate-requests`
   - `--aggregate-tokens`
   - `--aggregate-host`
   - `--aggregate-port-base`
3. Added gate-side resolved profile log line:
   - `gate aggregate_profile ...`
4. Updated runbook with:
   - fast/full default matrix definitions;
   - full-profile cluster invocation example.

## Fast vs Full Defaults

- `fast` (default):
  - `ctx`: `262144,1048576`
  - `slots`: `2`
  - `policies`: `sequential`
  - `requests`: `8`
  - `tokens`: `1`
- `full`:
  - `ctx`: `131072,262144,524288,1048576`
  - `slots`: `1,2,4,8`
  - `policies`: `sequential,reject-busy`
  - `requests`: `4`
  - `tokens`: `1`

## Validation

```bash
bash -n tools/ds4-v100-gate.sh
tools/ds4-v100-gate.sh --help
```

## Cluster Execution

Executed on `llamacpp-build-8gpu` (`gpu-01`) with:

```bash
bash ./tools/ds4-v100-gate.sh \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --mtp-model /models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --pack-index docs/sprints/drafts/SPRINT-003-PACK-INDEX.tsv \
  --ctx 1048576 \
  --slots 2 \
  --aggregate-profile full \
  --log-dir logs/sprint051-full-gate
```

Gate result:

- `gate	readiness	READY	missing=`
- `gate	summary	PASS	failures=0 ready=true`

Artifacts copied to:

- `logs/from-cluster/sprint051-full-profile`
- `aggregate_slot_context_throughput/aggregate_throughput.tsv` includes all
  32 full-profile cases:
  - contexts: `131072,262144,524288,1048576`
  - slots: `1,2,4,8`
  - policies: `sequential,reject-busy`

Quick aggregate summary (one-token request shape, `requests=4` per case):

- Case pass count: `32/32` (`status_200=4`, `errors=0`, `token_mismatch=0` in every case)
- Aggregate tok/s envelope across full matrix:
  - minimum observed: `0.320543` tok/s
  - maximum observed: `0.382304` tok/s

## Notes

This sprint now includes both control-plane implementation and cluster
execution evidence.
