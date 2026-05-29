# Sprint 544 - C1 Full-Capture Recheck

Date: 2026-05-29

## Goal

Recheck full graph capture on the current audit-clean TP/EP surface, without
the promoted suffix stage, and determine the next C1 blocker.

## Setup

Remote workspace:

- `/workspace/s544-full-capture`

Build command:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Result:

- PASS

Probe command shape:

- `DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY=0`
- `--decode-cudagraph-gate`
- `--decode-cudagraph-replay-probe-gate`
- `--decode-cudagraph-persistent-replay-gate`
- no `--decode-cudagraph-suffix-stage`
- selected-token endpoint
- `8` requests / `8` slots / `256K` context / position `262080`
- `4` generated tokens

Candidate artifact:

- `/workspace/s544-full-capture-artifacts/none-none-s544-fullcap8x4-p262080-serverargs-h396a9fa7`

Control artifact:

- `/workspace/s538-c2-parity/none-s538-eager8x4`

## Result

Correctness:

- `http_200=8`
- response sequence multiset matched eager
- decode-step checksum multiset matched eager
- peer-copy/SYS `0`
- NCCL graph SYS edges `0`
- `graph_audit_blocker=none`
- `graph_audit_capture_eligible=1`

Graph behavior:

- `graph_audit_capture_attempted=43`
- `graph_audit_capture_succeeded=43`
- `graph_audit_replay_attempted=43`
- `graph_audit_replay_succeeded=43`
- `graph_audit_persistent_cache_hits=0`
- `graph_audit_persistent_cache_misses=43`
- `graph_audit_persistent_invalidate_position=43`

Performance:

- client generated tok/s `4.394849929`
- not comparable to the promoted suffix path for promotion, because full
  capture recaptured by position and paid instantiate overhead every step.

## Decision

Do not promote full capture yet.

The current full-capture path is no longer blocked by helper host
synchronization or serving parity, but it is still position-keyed. This is the
same structural limitation that previously stranded full graph serving: the
graph is correct only when recaptured per decode position.

## Follow-Up

The next C1 sprint should make full-capture position dynamic instead of part of
the persistent cache key. The likely implementation target is a device-resident
decode-position scalar/state that graph replay updates or reads without
changing graph topology. Only after position invalidations drop to `0` should a
warmed full-capture performance gate run.
