# Sprint 046 Follow-Ups

## P0: Active-Microbatch Scheduler

Delivered in Sprint 047 at scheduler-core level:

- stage scheduler now owns per-slot KV and HC state;
- batch scheduler APIs exist for token-seed decode, HC decode, and stage handoff;
- full scheduler smoke and gate have multi-slot (`--slots`) coverage.

Remaining P0 work is request-loop integration across concurrent prompts.

Sprint 46 implemented admission/queue policy and metrics, but requests are not yet executed
with true tensor-resident batched slots.

- Implement the runtime path that executes `active_microbatch > 1` decode batches in one
  device pass.
- Extend layer/context state so each active slot has independent KV and HC cursors.
- Keep admission intact; reject only on explicit memory/context overrun, not by
  conservative single-slot scheduling behavior.

## P1: Throughput Evidence

Admission now provides a safety envelope; true throughput proof is still pending.

- Run 1/2/4/8 configured slot benches at 128K/256K/512K/1M context with
  `ds4-v100-slot-context-envelope.sh`.
- Capture aggregate token throughput, p99 latency, and per-stage timing for each active
  slot configuration.
- Compare `reject-busy` and `sequential` queue policies under load with identical workloads.

## P2: Scheduler vs Queue Policy

- Expose `scheduler_slots_ready` as true only when slot batching is implemented.
- Add a gate rung that explicitly requires `active_microbatch_scheduler`.
- Keep queue policy behavior (`reject-busy` default; `sequential` optional) stable for
  mixed workloads until true batching is fully validated.
