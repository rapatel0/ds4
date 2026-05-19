# Sprint 046 Report: Aggregate Slot/Context Envelope

## Result

`SHIP (runtime contract only)`.

Sprint 046 replaced the generic `aggregate_slot_context_envelope` blocker with a
runtime-enforced admission contract for configured slots, context tiers, queueing,
and service-level status/metrics. The full 8-GPU gate now checks `slot_context_admission`.

The remaining blocker is that we have not executed the new gate end-to-end on the
cluster from this workspace after the changes.

## Implementation Summary

- Added slot and queue admission to `tools/ds4-v100-replay`:
  - `--slots`
  - `--active-microbatch`
  - `--queue-policy {reject-busy,sequential}`
  - `ds4_v100_replay` now allocates KV state for configured slots.
  - Rejection counters now differentiate busy/context/bad request rejects.
  - `ds4_v100_scheduler` and `ds4_v100_replay` carry and use
    `kv_active_slots`.
- Updated appliance status and metrics surfaces:
  - `slots`, `configured_slots`, `active_slots`, `active_microbatch`,
    `queue_policy`, and `scheduler_slots_ready` in `/v100/status`.
  - `ds4_v100_configured_slots`, `ds4_v100_active_microbatch`,
    `ds4_v100_active_slots`, `ds4_v100_rejected_*` metrics in `/metrics`.
- Added a machine-readable planner output and matrix in `tools/ds4-v100-plan`:
  - `--json` emits envelope, per-gpu totals, `admission_tiers`, and
    `target_matrix` (1/2/4/8 slots).
- Added `tools/ds4-v100-slot-context-envelope.sh`:
  - emits planner JSON + TSV rows,
  - runs a conservative slot/concurrency smoke,
  - validates an intentional over-context request returns HTTP `413 context_exceeded`.
- Wired slot/context admission into the full gate:
  - `slot_context_admission` added to `tools/ds4-v100-gate.sh` and `readiness`.
- Added operational config surface:
  - launcher env additions:
    `DS4_V100_ACTIVE_MICROBATCH`, `DS4_V100_QUEUE_POLICY`.
  - updated `docs/operations/DS4-V100-APPLIANCE.md` and appliance smoke checks.

## Planner Contract

- Default device capacity used for planning is now explicit via
  `--device-total-bytes` (defaults to 32 GiB).
- Admission cap is currently bounded to 8 slots for practical device-wide runtime
  concurrency assumptions.

## Local Validation

```bash
bash -n tools/ds4-v100-gate.sh \
  tools/ds4-v100-production-deployment-gate.sh \
  tools/ds4-v100-appliance-smoke.sh \
  tools/ds4-v100-slot-context-envelope.sh \
  tools/ds4-v100-run-appliance.sh \
  tools/ds4-v100-replay.c
```

```bash
cc -fsyntax-only -I. tools/ds4-v100-replay.c
cc -fsyntax-only tools/ds4-v100-plan.c
```

```bash
./tools/ds4-v100-plan --json --ctx 262144 --slots 8 --gpus 8 \
  --device-total-bytes 34359738368 > /tmp/ds4_v100_plan_046.json
```

```bash
python3 - <<'PY'
import json
with open('/tmp/ds4_v100_plan_046.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print('configured', data['configured'])
print('target slots at 1M', [row for row in data['target_matrix'] if row['ctx_tokens'] == 1048576])
PY
```

## Remaining Blocker

`slot_context_admission` is now the explicit next gate rung, but this repo does
not include a completed 8x V100 run of `tools/ds4-v100-slot-context-envelope.sh`
since `tools/ds4-v100-replay` requires a CUDA build and runtime.

## Next Sprint Candidate

- `active_microbatch_scheduler` should replace reject-only mode as a true
  tensor-resident batched scheduler path.
- Keep queue/busy rejection semantics and expand throughput evidence for
  aggregate tok/s under each context tier.
