# Sprint 460: TP/EP Graph Event-Order No-Replay Isolation

## Objective

Determine whether the CUDA graph event-order path is correctness-safe when
graph replay is disabled.

## Rationale

Sprint 459 showed persistent replay is not promotable and is not actually
reusing stable graphs across HTTP token positions: every layer invalidates on
position and recaptures. The candidate still changes output tokens. Before
moving dynamic state into device metadata, we need to know whether the
non-replay graph-safe synchronization mode itself preserves serving semantics.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- No new scheduler abstraction.
- Run the current rank-major/NCCL baseline as control.
- Run candidate with `--decode-cudagraph` only, without
  `--persistent-decode-cudagraph`.
- Use the focused reduced shape:

```text
8 requests / 8 slots / 256K context / 3 generated tokens/request
```

## Definition of Done

- Run starts only when the node is clear.
- HTTP A/B completes or records a concrete environmental blocker.
- Result records readiness, response parity, token output, server decode,
  request-window GPU utilization, graph capture counters, and decision.
- Sprint docs/status are updated.

## Decision Gate

- If no-replay graph event-order preserves parity, graph replay is the unsafe
  boundary and the next work is capture/no-replay guard plus device dynamic
  metadata.
- If no-replay graph event-order fails parity, fix event-order synchronization
  before touching persistent replay again.

## Outcome

First attempt:

```text
/localpool/ds4/workspace/logs/s460-graph-gate-s8-t3
```

The control server exited with `rc=-15` before readiness and before emitting
startup output. The run was discarded as environmental.

Clean retry artifact:

```text
/localpool/ds4/workspace/logs/s460-graph-gate-s8-t3-r2
```

Shape:

```text
8 requests / 8 slots / 256K context / 3 generated tokens/request
```

| Metric | Control | Graph event-order, no replay |
|---|---:|---:|
| readiness | pass | fail |
| response parity | - | 0/8 |
| HTTP 200 | 8 | 8 |
| server generated decode tok/s | 20.009325 | 9.388085 |
| server continuation decode tok/s | 19.967099 | 9.350279 |
| client generated tok/s | 2.057285 | 0.839831 |
| output-head first token | 52762 | 57097 |
| graph capture attempted/succeeded | 0/0 | 43/43 |
| graph replay attempted/succeeded | 0/0 | 0/0 |
| graph blocker | n/a | none |
| HC-current gather ms | 4.487008 | 157.184537 |
| HC-current input ms | 228.028590 | 331.324808 |
| min free VRAM MiB | 5092 | 5086 |

## Decision

Do not promote graph event-order mode.

This rules out "persistent replay only" as the sole correctness problem. Even
without replay, enabling `--decode-cudagraph` changes the output token and fails
response parity. The next graph work must fix the graph-safe event-order path
itself before persistent graph reuse or device dynamic metadata can be
promotion candidates.

The clearest timing clue is HC-current gather:

```text
4.487008 ms -> 157.184537 ms
```

That suggests the event-order path is either missing a required dependency and
forcing later synchronization fallout, or it is serializing/ordering the
cross-rank HC-current gather badly compared with the eager stream-sync baseline.
