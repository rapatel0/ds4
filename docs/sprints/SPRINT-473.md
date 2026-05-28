# Sprint 473: Compose-Suffix Graph Serving Bisection

## Objective

Move final-HC carry/expand out of the captured persistent CUDA graph suffix,
wire the new serving controls through the appliance harness, and determine
whether the compose-only suffix can be promoted to HTTP serving.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Default-off graph diagnostics only.
- Keep the target context at `256K`.
- Validate direct all-layer correctness before serving promotion.

## Implementation

- Added `compose_eager_final_hc` as a suffix stage: the graph captures through
  compose, then final-HC carry/expand runs eagerly after replay.
- Wired `DS4_V100_TP_EP_DECODE_CUDAGRAPH_SUFFIX_STAGE` through:
  - `tools/ds4-v100-run-appliance.sh`
  - `deploy/v100/ds4-v100-appliance.env.example`
  - `tools/ds4-v100-tp-ep-profile.py`
  - `tools/ds4-v100-tp-ep-nccl-http-ab.py`
- Added harness support for `--startup-warmup`.
- Added persistent graph cache invalidation on decode position.
- Added per-leg HTTP A/B controls:
  - `--control-gpu-route-plan`
  - `--candidate-gpu-route-plan`
- Added an experimental context/slot admission override for target-shape
  graph diagnostics:
  - `--experimental-ctx-slot-cap`

## Validation

Local harness checks:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py tools/ds4-v100-tp-ep-nccl-http-ab.py
bash -n tools/ds4-v100-run-appliance.sh
git diff --check
PASS
```

V100 build:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Direct all-layer validation at `8` slots / `256K`:

| Run | Tokens | Eager checksum | Persistent checksum | Eager tok/s | Persistent tok/s | Verdict |
|---|---:|---:|---:|---:|---:|---|
| direct masked | 1 | `1126925252` | `1126925252` | `4.785078` | `11.558355` | pass |
| direct masked | 2 | `8349369606` | `8349369606` | `5.870644` | `14.196876` | pass |

HTTP A/B bisection:

| Shape | Endpoint | Candidate | Control server tok/s | Candidate server tok/s | Parity | Notes |
|---|---|---|---:|---:|---|---|
| `8x1` | chat | `compose_eager_final_hc` | `6.536650` | `20.553654` | `0/8` | Fast but wrong |
| `8x1` | chat | graph + masked compact copy | `6.493966` | `20.685753` | `0/8` | Route mask alone not sufficient |
| `8x1` | chat | masked + position invalidation | `6.630770` | `19.717539` | `0/8` | Cache hits `0`; stale position cache was not the only bug |
| `8x1` | selected-token | masked + position invalidation | n/a | n/a | `0/8` | Removes chat prompt prefill and still failed |
| `8x1` | selected-token | graph + GPU route plan | n/a | n/a | `8/8` | Fixed the selected token `0` bug |
| `8x1` | chat | graph + GPU route plan + masked copy | `6.574589` | `20.111171` | `8/8` | First serving-correct graph route-plan result |
| `8x3` | chat | graph + GPU route plan + masked copy | `6.533721` | `20.691050` | `8/8` | Continuation `6.528784 -> 20.574184` |
| `16x3` | chat | graph + GPU route plan + masked copy | `8.738857` | `36.872701` | `0/16` | Token sequences matched; hidden checksum drifted |
| `16x3` | chat | graph + GPU route plan, no masked copy | `8.839339` | `37.293446` | `16/16` | Strict pass; masked copy is diagnostic-only |
| `32x3` | chat | graph + GPU route plan, no masked copy | `10.020431` | `53.064769` | `0/32` | Token sequences and sampled logits matched; hidden checksum drifted |

After syncing the current local full-layer source and adding per-step checksum
telemetry, the slot/headroom boundary is:

| Shape | Scratch | Endpoint | Control server tok/s | Candidate server tok/s | Parity | Candidate min free | Notes |
|---|---:|---|---:|---:|---|---:|---|
| `32x2` | `1536 MiB` | selected-token | n/a | n/a | `32/32` | `1204 MiB` | No prompt prefill; step checksums match |
| `16x3` | `1536 MiB` | chat | `8.827543` | `32.789088` | `16/16` | `3484 MiB` | Current-source strict pass |
| `24x3` | `1536 MiB` | chat | `9.109601` | `48.878137` | `24/24` | `2304 MiB` | Current-source strict pass |
| `28x3` | `1536 MiB` | chat | `9.786744` | `51.415096` | `28/28` | `1664 MiB` | Current-source strict pass |
| `30x3` | `1536 MiB` | chat | `9.980393` | `54.742297` | `0/30` | `1430 MiB` | Token/logit match; checksum drift below reserve |
| `30x3` | `1280 MiB` | chat | `9.855309` | `53.397590` | `30/30` | `1686 MiB` | Headroom fix restores strict parity |
| `32x3` | `1024 MiB` | chat | `10.067904` | `55.253943` | `32/32` | `1714 MiB` | Full target strict pass |

The `32x3` / scratch `1024 MiB` run queued behind the global benchmark lock
while a steady-profile job completed:

```text
/localpool/ds4/workspace/logs/s474-steady-profile-s32-r256-t64-c32d-lock
```

Final full-target artifact:

```text
/localpool/ds4/workspace/s473-current-chat-gpu-route-plan-nomask-s32-t3-scratch1024
```

Longer target soak:

| Shape | Scratch | Endpoint | Control server tok/s | Candidate server tok/s | Parity | Candidate min free | Notes |
|---|---:|---|---:|---:|---|---:|---|
| `32x3` | `1024 MiB` | chat, fixed eager control | `35.699743` | `58.932234` | `0/32` | `1714 MiB` | Previous `32x3` promotion was invalid; true eager control exposes 5/32 visible token mismatches |
| `32x4` | `1024 MiB` | chat, fixed eager control | `35.206155` | `56.932814` | `0/32` | `1714 MiB` | 21/32 visible token mismatches |
| `32x4` | `1024 MiB` | chat, non-persistent graph | `35.036379` | `27.574525` | `0/32` | `1838 MiB` | All token sequences matched, but checksum drifted and candidate was slower |
| `32x4` | `1024 MiB` | chat, graph-audit control | `27.054120` | `57.909852` | `0/32` | `1714 MiB` | Superseded by fixed-control run; control accidentally carried graph-audit suffix |
| `32x4` | `1024 MiB` | chat + stage checksum | `9.394376` | `57.929667` | `0/32` | `1714 MiB` | Step checksums differed from step 0 for all slots; only 2/32 visible token mismatches in this instrumented run |
| `32x4` | `1024 MiB` | chat + output sync | `29.785221` | `53.396183` | `0/32` | `1714 MiB` | Device sync before output head did not repair parity |
| `32x32` | `1024 MiB` | chat | `9.454185` | `57.197275` | `0/32` | `1714 MiB` | Real token divergence after short prefix |

Artifact:

```text
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-bisect
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-stage
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-outputsync
/localpool/ds4/workspace/s474-graph-nonpersistent-s32-t4-compose
/localpool/ds4/workspace/s474-promoted-graph-s32-t4-bisect-fixedctl
/localpool/ds4/workspace/s474-promoted-graph-s32-t3-fixedctl
/localpool/ds4/workspace/s474-promoted-graph-s32-t32-soak
```

The long soak confirms that the graph path is not yet production-safe for
longer generations. It is fast and memory-admitted, but visible generated
tokens diverge. The first-divergence distribution across response pairs was:

```text
LCP counts: {0:1, 1:2, 3:8, 5:2, 6:2, 7:2, 8:1, 9:3, 10:3, 11:1, 12:1,
             13:1, 14:3, 23:1, 30:1}
```

The fixed-control rerun invalidates the previous short-run graph promotion.
The wrapper was correctly passing an empty graph suffix for control legs, but
the launcher default used shell `:=` assignment and replaced the empty value
with `compose_eager_final_hc`. That made recent control legs graph-audit
controls instead of pure eager controls. The launcher now preserves explicit
empty suffix values, and graph defaults have been reverted to off.

Against a true eager control, persistent graph fails by `32x3`: 5 of 32
response pairs have visible token differences and the other 27 drift in
checksum. At `32x4`, 21 of 32 pairs have visible token differences. A full
device sync before output-head did not repair the mismatch, so the remaining
bug is in replayed decode state, not merely output gather/top1 ordering.
Non-persistent compose graph preserves visible `32x4` token sequences but still
drifts checksum and is slower than eager, which points at capture/replay
ordering before persistent-cache optimization.

Rank-major graph diagnostics remain rejected. The `32x1` rank-major graph
replay-stage run failed `0/32` response parity despite improving server decode
`8.771138 -> 64.908951` tok/s, so rank-major stays default-off for the graph
serving path.

Selected-token artifact:

```text
/localpool/ds4/workspace/s473-http-selected-position-invalidated-masked-t1
```

The selected-token response pair is sharp:

```text
control selected_token   = 128818
candidate selected_token = 0
```

## Outcome

Do not promote persistent graph suffix serving as the `32` slot default yet.

The direct all-layer path proves that compose-suffix replay plus eager final-HC
can be correct and faster. HTTP serving requires GPU route-plan semantics for
graph correctness: without it, selected-token graph serving returned token `0`.
With GPU route planning, the graph path is strict-correct through `16` slots at
`256K`, and the `32` slot run preserves visible token output while drifting in
the hidden checksum.

The masked compact-copy route is rejected. It is correct at `8` slots but
introduces hidden checksum drift at `16` slots. The unmasked GPU route-plan
path is the current best graph candidate.

The correct serving tier is now the target `32` slots / `256K` eager TP/EP path
with `DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB=1024`, GPU route planning,
fixed-capacity route planning, masked compact copy disabled, and graph replay
disabled. The launcher still admits `32` slots at `256K` without requiring
`DS4_V100_EXPERIMENTAL_CTX_SLOT_CAP`, but graph capture/replay is opt-in only.

Do not treat graph serving as production-ready: it fails true eager parity by
`32x3`, and the `32` token soak also fails visible token parity. Graph flags
are useful diagnostics, but production promotion requires fixing replay-state
parity first.

## Next

1. Localize graph capture/replay ordering drift using non-persistent compose
   graph as the safer isolate: visible tokens match at `32x4`, but checksum
   drift remains.
2. Keep appliance defaults on the true eager TP/EP path until graph replay
   matches true eager.
3. Keep masked compact copy and rank-major graph paths default-off and
   diagnostic-only.
