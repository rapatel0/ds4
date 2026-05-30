# Sprint 588 - MTP layer-43 eager correctness isolation

Date: 2026-05-30

## Why This Sprint Exists

Sprint 587 rejected the two token/position semantic candidates. During the
same work, temporary layer-43 tensor diagnostics crashed because the MTP body
entered CUDA graph capture:

```text
cuda error ./engine/diagnostics_support.cu:38: operation not permitted when stream is capturing
tp_ep_decode_cudagraph_replay_probe_start layer 43
```

MTP is still a correctness scaffold, not a promoted graph path. Running layer
43 through graph capture adds an unvalidated execution mode and blocks the
same-logical-point diagnostics needed to localize the remaining numerical bug.

## Scope

1. Force only the MTP layer-43 run to eager mode by disabling decode graph
   capture in `mtp_opt`.
2. Keep the promoted main 0-42 serving graph path unchanged.
3. Rebuild on the pod.
4. Re-run the deterministic MTP acceptance harness once, because this is a real
   candidate path change.

## Non-Goals

- Do not change MTP token conditioning or position semantics in this sprint.
- Do not add permanent debug flags or smokes.
- Do not promote MTP unless acceptance improves and main-serving parity holds.

## Definition of Done

- Pod build passes.
- The acceptance harness completes without graph-capture diagnostic crashes.
- The result is recorded as either a promoted eager-MTP correctness fix or a
  rejected isolation candidate.

## Execution Result

Implemented the isolation by forcing only `mtp_opt` to disable graph capture,
persistent replay, and replay-probe before `run_layer(43)`. The promoted main
0-42 serving graph path is unchanged.

Pod build:

```text
make appliance/ds4-v100-tp-ep-appliance
BUILD_EXIT=0
```

Validation:

```text
/workspace/s585_accept2.sh
ACCEPT_EXIT=0
pairs 71 same_index_match 0 (0.00) next_index_match 0
main[:12]  [53022, 94385, 64581, 109502, 109502, 27525, 32461, 119065, 55222, 46965, 63082, 70194]
draft[:12] [112865, 5743, 8373, 13151, 82318, 84941, 5626, 124211, 49721, 21859, 27674, 67132]
```

Graph isolation check:

```text
layer43_graph_lines=0
```

The MTP layer no longer enters CUDA graph capture, and the scaffold layer-43
decode time in this harness drops to about `9-10 ms` from the prior graph
capture range around `42 ms`. Acceptance remains `0/71`, so this is not a B1
correctness promotion. It is kept as a prerequisite cleanup for future MTP
debugging: layer 43 is unpromoted and should stay eager until MTP parity exists.

Next direction: same-point layer-43 raw-ring/body diagnostics can now run
without graph-capture host-copy failures. Start with raw-ring isolation, then
attention output and routed-FFN output.
