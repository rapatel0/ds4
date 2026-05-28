# TEMP Status Report 473: Compose-Suffix Graph Serving Boundary

## Topline

Persistent compose-suffix replay is now correct in the direct all-layer harness
and partially correct in HTTP serving.

The useful speed signal is real:

- Direct `8` slot / `256K` / `1` token: `4.785078 -> 11.558355` tok/s.
- Direct `8` slot / `256K` / `2` tokens: `5.870644 -> 14.196876` tok/s.
- Current-source HTTP `32` requests / `32` slots / `256K` / `3` tokens
  strict-passes with GPU route planning, compose-suffix persistent graph replay,
  and scratch `1024 MiB`: `10.067904 -> 55.253943` server decode tok/s.
- The longer `32` request / `32` slot / `256K` / `32` token soak fails visible
  response parity: `9.454185 -> 57.197275` server decode tok/s, but `0/32`
  response pairs match.

The graph path is memory-admitted and fast, but not production-safe for longer
generations yet. The launcher defaults have been updated to the short-run
passing configuration so the next bisection can run without experiment-only
overrides.

## What Changed

- Added `compose_eager_final_hc` graph suffix mode.
- Plumbed graph suffix selection through the appliance launcher, env example,
  profile wrapper, and HTTP A/B wrapper.
- Added startup warmup control to the profile/A-B harness.
- Added persistent graph cache invalidation when decode position changes.
- Added HTTP A/B `--control-gpu-route-plan` and
  `--candidate-gpu-route-plan` flags.
- Added `--experimental-ctx-slot-cap` to the profile and HTTP A/B wrappers so
  target-shape graph diagnostics can override conservative admission.

## Evidence

Direct all-layer checks at `8` slots / `256K`:

| Tokens | Eager checksum | Persistent checksum | Graph captures | Graph replays | Verdict |
|---:|---:|---:|---:|---:|---|
| 1 | `1126925252` | `1126925252` | 43 | 43 | pass |
| 2 | `8349369606` | `8349369606` | 86 | 86 | pass |

HTTP bisection summary:

| Shape | Candidate | Result |
|---|---|---|
| `8x1` selected-token, no GPU route plan | failed `0/8`; `128818 -> 0` |
| `8x1` selected-token, GPU route plan | passed `8/8` |
| `8x3` chat, GPU route plan + masked copy | passed `8/8`; `6.533721 -> 20.691050` tok/s |
| `16x3` chat, GPU route plan + masked copy | failed strict checksum `0/16`; token sequences matched |
| `16x3` chat, GPU route plan, no masked copy | passed `16/16`; `8.839339 -> 37.293446` tok/s |
| `32x3` chat, GPU route plan, no masked copy | failed strict checksum `0/32`; token sequences matched; `10.020431 -> 53.064769` tok/s |

Current-source slot/headroom matrix:

| Shape | Scratch | Endpoint | Result |
|---|---:|---|---|
| `32x2` | `1536 MiB` | selected-token | passed `32/32`; no prompt prefill; step checksums match |
| `16x3` | `1536 MiB` | chat | passed `16/16`; `8.827543 -> 32.789088` tok/s |
| `24x3` | `1536 MiB` | chat | passed `24/24`; `9.109601 -> 48.878137` tok/s |
| `28x3` | `1536 MiB` | chat | passed `28/28`; `9.786744 -> 51.415096` tok/s |
| `30x3` | `1536 MiB` | chat | failed checksum `0/30`; token/logit match; candidate min free `1430 MiB` |
| `30x3` | `1280 MiB` | chat | passed `30/30`; `9.855309 -> 53.397590` tok/s; candidate min free `1686 MiB` |
| `32x3` | `1024 MiB` | chat | passed `32/32`; `10.067904 -> 55.253943` tok/s; candidate min free `1714 MiB` |
| `32x3` | `1024 MiB` | chat, fixed eager control | failed `0/32`; `35.699743 -> 58.932234` tok/s; 5/32 visible token mismatches |
| `32x4` | `1024 MiB` | chat, fixed eager control | failed `0/32`; `35.206155 -> 56.932814` tok/s; 21/32 visible token mismatches |
| `32x4` | `1024 MiB` | chat, non-persistent graph | failed `0/32`; `35.036379 -> 27.574525` tok/s; token sequences matched but checksum drifted |
| `32x4` | `1024 MiB` | chat + stage checksum | failed `0/32`; step checksums differ from step 0 for all slots |
| `32x4` | `1024 MiB` | chat + output sync | failed `0/32`; output-head device sync does not fix it |
| `32x32` | `1024 MiB` | chat | failed `0/32`; `9.454185 -> 57.197275` tok/s; real token divergence |

Key artifacts:

```text
/localpool/ds4/workspace/s473-http-selected-gpu-route-plan-masked-t1
/localpool/ds4/workspace/s473-http-chat-gpu-route-plan-masked-t3
/localpool/ds4/workspace/s473-http-chat-gpu-route-plan-nomask-s16-t3
/localpool/ds4/workspace/s473-http-chat-gpu-route-plan-nomask-s32-t3
/localpool/ds4/workspace/s473-current-chat-gpu-route-plan-nomask-s30-t3-scratch1280
/localpool/ds4/workspace/s473-current-chat-gpu-route-plan-nomask-s32-t3-scratch1024
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-bisect
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-stage
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-outputsync
/localpool/ds4/workspace/s474-graph-nonpersistent-s32-t4-compose
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-bisect-fixedctl
/localpool/ds4/workspace/s474-promoted-graph-s32-t3-fixedctl
/localpool/ds4/workspace/s475-rankmajor-graph-s32-t1-replaystage
/localpool/ds4/workspace/s474-promoted-graph-s32-t32-soak
```

## Current Hypothesis

GPU route planning fixed the real selected-token graph bug. The hidden checksum
drift was slot/headroom sensitive for short runs: scratch `1536 MiB` drifted at
`30` and `32` slots, while scratch `1280 MiB` restored `30` slots and scratch
`1024 MiB` restored the full `32` slot / `3` token target.

The fixed-control reruns found a harness/launcher interaction: the launcher
graph suffix default was overriding the wrapper's explicit empty suffix for
control legs. That made recent controls graph-audit controls instead of true
eager controls. The launcher now preserves explicit empty suffix values and the
appliance graph defaults have been reverted to off.

With true eager controls, persistent graph is not correct even at `32x3`: 5 of
32 response pairs have visible token mismatches and 27 have checksum-only
drift. At `32x4`, 21 of 32 pairs have visible token mismatches. Non-persistent
compose graph is a useful isolate because all `32x4` visible token sequences
match, but checksums still drift and the graph candidate is slower than eager.
Adding a device sync before output-head did not restore parity, which points to
replayed decode state rather than output gather/top1 ordering.

In the `32` token soak, most slots share the first three tokens with control
and then diverge; the observed longest-common-prefix counts were:

```text
{0:1, 1:2, 3:8, 5:2, 6:2, 7:2, 8:1, 9:3, 10:3, 11:1, 12:1,
 13:1, 14:3, 23:1, 30:1}
```

The masked compact-copy route is rejected because it causes the same hidden
checksum drift already at `16` slots. Rank-major graph is also rejected for
this path: `32x1` rank-major replay-stage failed `0/32` parity despite a fast
`8.771138 -> 64.908951` server decode result.

The correct production default is therefore the eager TP/EP path with GPU route
planning and fixed-capacity route planning. Graph suffix replay remains an
opt-in diagnostic until it matches true eager.

## Next

Next: use the non-persistent compose graph isolate to find the first stream or
state dependency that drifts checksum while preserving visible `32x4` tokens,
then re-enable persistent replay only after non-persistent graph matches true
eager.
