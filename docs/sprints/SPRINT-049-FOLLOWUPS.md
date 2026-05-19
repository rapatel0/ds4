# Sprint 049 Follow-Ups

## P0: Extend Tier Coverage

- Add 128K and 512K tier runs with the same slot/policy matrix already captured
  for 256K and 1M.

## P1: Gate Integration Depth

- Decide whether full-gate default should keep a fast subset (current) and add
  a separate full-envelope profile for 1/2/4/8 slots.
- Add a machine-readable pass/fail threshold policy per case family.
- Execute and archive a full gate run that reaches `ready=true`.

## P2: Scheduler Path Expansion

- Extend request-loop batching from first-token only to multi-token token-step
  batching so aggregate tok/s claims include continuation decode.
