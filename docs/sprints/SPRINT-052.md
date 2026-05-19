# Sprint 052: Sustained Decode And Utilization Baseline

## Status

Complete.

## Overview

Sprint 052 replaces the one-token aggregate gate as the practical performance
reference with sustained multi-token decode measurements. The goal is not to
claim optimization yet; it is to produce a repeatable benchmark that separates
prompt replay, continuation decode, stage timing, handoff timing, request
latency, and observed GPU utilization.

## Goals

1. Add a sustained decode benchmark that runs multi-token requests against the
   resident V100 replay service.
2. Capture generated tok/s and continuation tok/s separately.
3. Capture per-response timing fields already emitted by `ds4-v100-replay`.
4. Capture `nvidia-smi` utilization samples when available.
5. Add an optional gate profile so sustained decode can be run from the normal
   8-GPU gate without changing readiness defaults.
6. Document the operator command for practical-use performance baselining.

## Scope

- `tools/ds4-v100-sustained-decode-bench.sh`:
  - starts one resident `ds4-v100-replay --serve` instance per case;
  - supports context/slot/policy matrices;
  - defaults to multi-token requests;
  - records request latency, generated tok/s, continuation tok/s, timing
    averages, per-stage decode averages, and handoff averages;
  - records GPU utilization samples when `nvidia-smi` exists.
- `tools/ds4-v100-gate.sh`:
  - add optional sustained profile flags;
  - keep existing readiness behavior unchanged unless the sustained profile is
    explicitly enabled.
- `docs/operations/DS4-V100-APPLIANCE.md`:
  - document the sustained decode benchmark and expected artifacts.

## Out of Scope

- Continuous token-step batching across multi-token requests.
- New low-bit expert kernels.
- True MTP draft commit.
- A performance win claim. This sprint establishes the measurement baseline.

## Definition of Done

- `tools/ds4-v100-sustained-decode-bench.sh --help` documents all supported
  benchmark knobs.
- `bash -n tools/ds4-v100-sustained-decode-bench.sh` passes.
- `tools/ds4-v100-gate.sh --help` documents the optional sustained profile.
- `bash -n tools/ds4-v100-gate.sh` passes.
- The benchmark JSON includes:
  - `aggregate_generated_tokens_per_second`;
  - `aggregate_continuation_tokens_per_second`;
  - latency percentiles;
  - timing averages;
  - stage decode averages;
  - handoff averages;
  - GPU utilization summary or an explicit skipped reason.
- The runbook includes a cluster invocation that writes durable artifacts.
- Cluster execution captures at least one sustained decode run on `gpu-01`, or
  the report records the exact blocker preventing cluster execution.
