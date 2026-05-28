# Sprint 466: TP/EP Graph First-Divergence Checksums

## Objective

Find the first decode-stage state divergence between eager TP/EP serving and
graph-event-ordered TP/EP serving.

## Rationale

Sprint 465 ruled out output-head readiness as the graph no-replay root cause:
even a full device sync before output-head gather still emitted token `42549`
instead of the eager baseline token `52762`. The wrong state is produced before
the output head. Continuing broad serving graph A/Bs without intermediate state
visibility is now wasteful.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add default-off per-stage checksum diagnostics.
- Wire the diagnostics through the appliance launcher and A/B harness.
- Run a short V100 A/B to identify the first divergent stage.

## Definition of Done

- Diagnostic gate is default-off and has launcher/profile/A-B plumbing.
- CUDA binary rebuilds on the V100 node.
- Focused A/B completes on a clean node.
- The artifact contains eager and graph-event-order stage checksum lines.
- The sprint records the first observed divergence or the next missing
  instrumentation point.

## Implementation

Added default-off stage checksum instrumentation:

```text
--decode-stage-checksum-gate
DS4_V100_TP_EP_DECODE_STAGE_CHECKSUM=1
```

The gate is wired through:

- `tools/ds4-v100-tp-ep-full-layer-smoke.cu`
- `tools/ds4-v100-run-appliance.sh`
- `tools/ds4-v100-tp-ep-profile.py`
- `tools/ds4-v100-tp-ep-nccl-http-ab.py`
- `deploy/v100/ds4-v100-appliance.env.example`

The first version logged many tensors and was too heavy. It still produced a
useful partial artifact:

```text
/localpool/ds4/workspace/logs/s466-stage-checksum-s8-t1
```

The first overlapping mismatch was:

```text
step=0 layer=0 stage=hc_current tensor=current_shard rank=0
control:   bytes=16384 checksum=260522477
candidate: bytes=16384 checksum=264538364
```

The gate was then trimmed to stage-specific primary tensors so future runs are
usable.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
```

Remote:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Completed focused A/B:

```text
/localpool/ds4/workspace/logs/s466-stage-checksum-lite-s8-t1
shape: 8 requests / 8 slots / 256K context / 1 generated token
```

| Metric | Control | Candidate |
|---|---:|---:|
| response parity | 8/8 | 8/8 |
| stage checksum keys | 6880 | 6880 |
| checksum mismatches | 0 | 0 |
| server generated decode tok/s | 5.889001 | 4.122498 |
| HC-current input ms | 827.941246 | 1050.573522 |
| graph captures | 0 | 43/43 |

The completed run is not a performance result. The checksum gate intentionally
synchronizes after stage probes and materially slows both legs.

## Decision

Keep the diagnostic gate. It proves that adding synchronization during decode
restores graph-event-order response parity and matching intermediate state.
That makes the graph failure an ordering/race issue, not a tensor math or output
head issue.

The partial heavy run points the first observed divergence at HC-current
`current_shard` on layer 0. The completed lite run shows that stage-level
synchronization repairs it.

## Next

Find the minimal non-diagnostic ordering fix around HC-current. Start by adding
a diagnostic host/device sync immediately after HC-current only. If that restores
parity, replace it with a precise event or NCCL dependency. If it does not,
move the sync earlier inside HC-current: seed, control split, split copy,
weighted sum, and current-full allgather.
