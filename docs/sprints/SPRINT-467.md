# Sprint 467: TP/EP HC-Current Graph Ordering Probe

## Objective

Test whether a narrow HC-current-only synchronization point restores TP/EP
graph-event-order HTTP parity at the 8-slot 256K serving shape.

## Rationale

Sprint 466 showed that broad stage checksum synchronization makes eager and
graph-event-order state match, and the first heavy diagnostic mismatch appeared
at layer 0 HC-current `current_shard`. The next step is to remove the broad
diagnostic probes and test only the suspected ordering boundary.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Add a default-off diagnostic HC-current sync gate.
- Wire the gate through the appliance launcher, profile wrapper, and A/B
  harness.
- Run an 8-slot 256K graph-event-order A/B without stage checksums.

## Definition of Done

- `--decode-cudagraph-hc-current-sync-gate` exists and is default-off.
- The launcher exposes `DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC`.
- The profile and A/B harness can enable the gate independently for control and
  candidate.
- The V100 node build succeeds.
- The focused A/B records HTTP parity and decode throughput.

## Implementation

Added two default-off diagnostic controls:

```text
--decode-cudagraph-hc-current-sync-gate
DS4_V100_TP_EP_DECODE_CUDAGRAPH_HC_CURRENT_SYNC=1

--decode-cudagraph-stage-sync-gate STAGES
DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC=typed_history
```

The stage-sync gate accepts comma-separated decode stage names or `all`. It is
wired through the appliance launcher, profile wrapper, and HTTP A/B harness.

Also added `third_party/nccl_compat/nccl.h` so direct-node builds can continue
when the V100 host has `libnccl.so` but not `nccl.h` after a restart.

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

All remote builds completed on `gpu-01` with only the known unused-kernel
warnings.

## Experiments

All A/Bs used `8` requests / `8` slots / `256K` context / position `262000` /
`1` token unless noted.

| Artifact | Candidate sync | Parity | Candidate token | Candidate decode tok/s | Read |
|---|---|---:|---:|---:|---|
| `s467-hc-current-sync-s8-t3` | `hc_current` post-sync, `3` tokens | `0/8` | `42549` | `9.360714` | HC-current-only sync is not enough. |
| `s467-stage-sync-all-s8-t1` | `all` | `8/8` | `32974` | `8.265516` | Stage sync reproduces Sprint 466 checksum correctness without checksum logging. |
| `s467-stage-sync-pre-ep-s8-t1` | pre-EP stages | `8/8` | `32974` | `8.781688` | The required ordering is before routed/shared FFN. |
| `s467-stage-sync-pre-a-s8-t1` | `hc_current,attention_projection,compressed_kv,attention_state` | `0/8` | `91699` | `9.075232` | First half of pre-EP is insufficient. |
| `s467-stage-sync-pre-b-s8-t1` | `typed_history,raw_read,attention_output,post_attention_ffn_input` | `8/8` | `32974` | `8.985362` | Required ordering is in the latter pre-EP group. |
| `s467-stage-sync-attn-out-post-s8-t1` | `attention_output,post_attention_ffn_input` | `0/8` | `75105` | `9.265810` | Attention-output handoff alone is insufficient. |
| `s467-stage-sync-history-raw-s8-t1` | `typed_history,raw_read` | `8/8` | `32974` | `9.186428` | The issue is before or at raw history visibility. |
| `s467-stage-sync-raw-read-r2-s8-t1` | `raw_read` | `0/8` | `32974` | `9.328220` | Raw-read-only sync is insufficient. |
| `s467-stage-sync-typed-history-s8-t1` | `typed_history` | `8/8` | `32974` | `9.358436` | Minimal stage-level correctness barrier found. |
| `s467-typed-kv-boundary-fix-s8-t1` | graph event barrier in `sync_typed_kv_boundary()` | `0/8` | `32974` | `8.686312` | Event ordering alone does not replace the host sync. |
| `s467-typed-kv-fence-fix-s8-t1` | event barrier plus store-side `__threadfence_system()` | `0/8` | `75105` | `8.474409` | Store fence alone still does not replace the host sync. |

## Decision

HC-current was a useful starting hypothesis, but the minimal correctness
barrier is now localized to the typed KV history boundary. A host stream sync
after `typed_history` restores response parity; graph event barriers and
store-side system fences do not.

Do not promote graph serving yet. Keep `DS4_V100_TP_EP_DECODE_CUDAGRAPH_STAGE_SYNC`
as a diagnostic control, not a production setting.

## Next

Replace the typed-history host sync with a real graph-safe data-movement fix.
Most likely directions:

- avoid peer-read typed KV loads by loading each GPU's local shard then
  explicitly allgathering/broadcasting rows with NCCL or peer copies;
- split typed-history load into local-shard load plus graph-ordered row
  assembly;
- keep the host sync only as a correctness fallback while measuring the cost at
  `32` slots / `256K`.
