# Sprint 538 - C2 Graph Serving Parity and Replay Repair

Date: 2026-05-29

## Goal

Close the graph-in-serving parity gap found in Sprint 537 so persistent suffix
replay can be evaluated for real serving performance without changing output.

## Starting Evidence

Sprint 537 proved the direct graph suffix is usable:

- Direct `8` slot / `256K` / `4` token suffix replay matched eager first token
  `123327`.
- Direct graph replay produced `43` misses, `129` cache hits, `172/172`
  successful replays, zero invalidations, and zero NCCL SYS edges.
- Reduced HTTP selected-token executed graph replay (`43/43` cache hits) with
  zero peer-copy/SYS and zero NCCL SYS edges, but failed serving parity:
  eager first token `29361`, graph first token `61012`.

Performance is not the decision criterion for this sprint. Long warmed
generation is useful only after parity passes. If the graph path is
correctness-clean and does not regress steady serving performance, promotion is
allowed as a structural default even without a large immediate throughput win.

## Scope

1. Run a reduced selected-token eager/graph checksum pair with
   `--decode-stage-checksum-gate` enabled to localize the first serving-mode
   divergence.
2. Compare stage/tensor/rank checksums by `(step, layer, stage, tensor, rank)`.
3. Fix the first confirmed ordering/state bug with precise CUDA events or
   device-buffer state repair. Do not add broad device synchronizes.
4. Re-run the reduced selected-token parity probe after each repair.
5. If reduced selected-token parity passes, run a larger selected-token parity
   gate. Only then run a warmed long-generation/lorem-ipsum performance probe.
6. For prompt-level checks, use deterministic request settings where the
   endpoint supports them. The selected-token endpoint is the hard parity gate
   because it has no sampling surface; fixed-prompt checks are supplemental.

## Non-goals

- No graph default promotion before selected-token or fixed-prompt generated
  sequence parity passes.
- No broad performance claims from short probes.
- No MTP work.
- No new permanent smoke harness or flag matrix.
- No promotion based only on a stochastic text sample.

## Validation

Minimum correctness gate:

- Eager and graph selected-token first token match at the reduced shape.
- Generated sequence agreement passes for the same fixed selected-token shape
  if more than one token is requested.
- Fixed-prompt output checks use constrained prompts such as
  `The capital of France is` with deterministic sampling controls where
  available. These checks are supplemental to selected-token parity.
- `graph_audit_replay_succeeded == graph_audit_replay_attempted`.
- `peer_copy_ops=0`, `peer_copy_sys_bytes=0`.
- `nccl_graph_sys_edge_count=0`.
- No VRAM admission failures.

Output tolerance:

- Selected-token parity remains strict: token ids must match.
- Natural-language prompt checks may tolerate benign numeric/tokenization drift
  if the output-level answer is semantically correct, but only as supplemental
  evidence after graph-state parity is established.

Performance gate, only after correctness:

- Use startup warmup where serving supports it.
- Use enough warmed requests/tokens to make startup and graph instantiation
  negligible.
- Compare request-window or steady-state fields, not full-run elapsed time or
  full-run GPU averages.
- A long fixed-prompt/lorem-ipsum generation is acceptable for this warmed
  performance probe.

## Execution

Workspace:

- Local repo: `/Users/ravi/repos/ds4`
- Remote build workspace: `/workspace/s538-c2-repair`
- Main artifact root: `/workspace/s538-c2-repair-artifacts`

Build:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`
- Result: PASS

Reduced checksum pair:

- Eager: `/workspace/s538-c2-checksum/none-s538-eager-checksum8x1-serverargs-h91108f54`
- Graph: `/workspace/s538-c2-checksum/none-s538-graph-checksum8x1-compose-serverargs-h1f9e1eca`
- Result: selected-token output matched for all `8` slots at `1` token.

Bug reproduced at the prior failing shape:

- Eager control: `/workspace/s538-c2-parity/none-s538-eager8x4`
- Graph candidate before repair:
  `/workspace/s538-c2-parity/none-s538-graph8x4-compose-serverargs-h2180dc1d`
- Result: reproduced Sprint 537 parity failure. Eager first output-head token
  was `29361`; graph first output-head token was `61012`.
- Response sequences diverged after the first generated token. Example slot 0:
  eager `[128819, 68338, 29361, 57097]`; graph
  `[128819, 101261, 61012, 80437]`.

Root cause:

- The Sprint 537 suffix-only persistent graph cache was made reusable across
  decode positions.
- That is unsafe for the current routed suffix because routed FFN/compose still
  capture host-side launch geometry such as per-rank route counts.
- The first decode position captures with one route shape; later positions
  update dynamic prefix buffers but replay the suffix graph with stale captured
  launch parameters.

Repair:

- `engine/decode_loop.cu` now keeps persistent suffix graphs position-keyed.
- This restores the safe replay-after-capture behavior and prevents stale
  route-shape reuse across decode positions.
- No broad device synchronizes, no permanent smoke harness, and no new feature
  flag were added.

Validation after repair:

- `8` requests / `8` slots / `256K` / `4` tokens:
  - Graph artifact:
    `/workspace/s538-c2-repair-artifacts/none-s538-repair-graph8x4-compose-serverargs-h2180dc1d`
  - Eager reference: `/workspace/s538-c2-parity/none-s538-eager8x4`
  - Result: all `8` response token sequences matched exactly.
  - `graph_audit_replay_succeeded=43`, `graph_audit_replay_attempted=43`
  - `graph_audit_persistent_cache_hits=0`
  - `graph_audit_persistent_cache_misses=43`
  - `graph_audit_persistent_invalidate_position=43`
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

- Larger `8` requests / `8` slots / `256K` / `8` tokens:
  - Eager artifact:
    `/workspace/s538-c2-repair-artifacts/none-s538-repair-eager8x8`
  - Graph artifact:
    `/workspace/s538-c2-repair-artifacts/none-s538-repair-graph8x8-compose-serverargs-h2180dc1d`
  - Result: all `8` response token sequences matched exactly.
  - Output-head first token matched: `42395`
  - `graph_audit_replay_succeeded=43`, `graph_audit_replay_attempted=43`
  - `graph_audit_persistent_cache_hits=0`
  - `graph_audit_persistent_cache_misses=43`
  - `graph_audit_persistent_invalidate_position=43`
  - `peer_copy_ops=0`, `peer_copy_sys_bytes=0`
  - `nccl_graph_sys_edge_count=0`

Decision:

- C2 serving parity is repaired for the selected-token graph suffix path.
- Do not promote graph serving defaults as a performance feature yet. The safe
  repair deliberately invalidates by position, so short-run timing is
  correctness evidence, not steady-state throughput evidence.
- The next graph-performance step is to make routed suffix launch geometry
  graph-stable, likely through fixed/full-shape device-side route masking or a
  route-signature-aware cache that can be validated without stale host launch
  parameters.
