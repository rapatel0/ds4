# TEMP Status Report 444

## Current Focus

Sprint 444 focused on making the HTTP profile/A-B harness safe enough to run
persistent CUDA graph serving experiments without stale servers, port collisions,
or invalid request budgets.

## Implemented

- Added `--kill-stale-server` to `tools/ds4-v100-tp-ep-profile.py`.
- Added DS4 managed-server discovery by HTTP port.
- Added port preflight failure when cleanup is not requested.
- Wired automatic stale-server cleanup into `tools/ds4-v100-tp-ep-nccl-http-ab.py`.
- Fixed profile request budgeting so `/health`, `/status`, and `/metrics` are
  included in the server-side `max_requests` allowance.
- Added graph-stream typed KV store/load plumbing while investigating graph
  serving blockers.
- Added position to the persistent graph cache key.
- Added expert contiguous-allocation fallback.

## Topline Results

Valid single persistent-graph HTTP smoke:

- Artifact: `/localpool/ds4/workspace/logs/s444-graph-single-fixed`
- Shape: `8` slots, `256K`, `2` requests, `2` tokens/request.
- HTTP 200: `2/2`
- Server generated decode: `41.291192 tok/s`
- Server continuation decode: `41.323516 tok/s`
- Average GPU util: `27.16%`
- Max GPU util: `63%`
- Min free VRAM: `4410 MiB`

Invalid A/B speed signal:

- Artifact: `/localpool/ds4/workspace/logs/s444-graph-ab-fixed`
- Control server generated decode: `19.817413 tok/s`
- Candidate server generated decode: `41.035001 tok/s`
- Speedup: `2.07x`
- Rejected: response parity failed `0/8`.
- Root cause: graph cache reused position-dependent captures across decode
  positions.

Position-keyed graph follow-up:

- Artifact: `/localpool/ds4/workspace/logs/s444-graph-ab-position-key`
- Correctly invalidates stale-position graphs, but candidate failed to serve
  valid responses.
- Subsequent candidate smokes moved through several capture blockers:
  compressed/indexer typed KV stores, typed KV loads, and indexer top-k copy.
- Latest minimal graph candidate still fails before HTTP 200; this path is not
  promotable.

## Assessment

Persistent graph serving has a real launch-overhead speed signal, but the
current implementation cannot safely reuse graphs across positions. Keying by
position restores correctness intent but turns the path into repeated
capture/instantiate work and exposes capture/VROOM fragility.

The next graph design should not cache one graph per position. It should move
position and similar dynamic values into device-side state buffers that are
updated before replay, so graph launch shape stays static while semantics change
per step.

## Cluster State

After cleanup, `nvidia-smi --query-compute-apps` and DS4 harness process checks
reported no active DS4 GPU jobs.
