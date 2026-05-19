# Sprint 051 Follow-Ups

## P0: Throughput Policy Thresholds

- Add machine-readable min-success and max-error thresholds for aggregate cases
  so `aggregate_slot_context_throughput` can fail hard on regressions.

## P1: Continuation Decode Coverage

- Add multi-token aggregate runs (`tokens > 1`) once batch token-step scheduling
  is expanded beyond first-token batching for non-MTP requests.
