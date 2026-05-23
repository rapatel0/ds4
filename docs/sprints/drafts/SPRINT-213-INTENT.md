# Sprint 213 Intent - Routed FFN Materialized-Reduce Gate

Validate or reject the current dirty `fused6_split_reduce` routed-FFN artifacts
on the V100 cluster. Sprint 212 rejected TP4/PP1 as the next runtime branch, so
this sprint returns to the production six-route routed-FFN hot path. The goal is
not another abstract plan: build, run, compare, and decide whether the
materialized half down-route reducer is a production candidate or diagnostic.

Constraints: no TP runtime integration, no PP scheduler changes, no default
promotion without served 16-slot/256K A/B evidence, and no generic scheduler
abstraction.
