# TEMP_STATUS_REPORT_039

Date: 2026-05-24

## Topline

Sprint 327 converted the compressed-KV memory discussion into an executable
TP/EP planner contract.

`tools/ds4-v100-plan-tp.c` now prints and emits JSON for:

- raw SWA rows
- ratio-4 compressed attention rows
- ratio-128 compressed attention rows
- ratio-4 indexer rows
- persistent KV bytes by dtype
- per-TP-rank persistent KV bytes
- replicated f32 bytes per GPU as a warning case
- current bounded diagnostic f32 bytes per GPU
- per-layer row/byte table

## V100 Validation

Build:

- Command: `make -j80 tools/ds4-v100-plan-tp`
- Result: PASS

Real pack:

- `/workspace/packs/ds4-appliance-full-tm-gated-s181`
- total weight bytes from pack: `156142896212` bytes, about `145.42 GiB`

Key outputs:

| Case | Verdict | Per-GPU total | Headroom after reserve | Persistent KV / GPU | Replicated f32 / GPU |
|---|---:|---:|---:|---:|---:|
| `slots=32`, `ctx=262144`, `kv=f8` | fits | `27.00 GiB` | `5.00 GiB` | `3.40 GiB` | `107.84 GiB` |
| `slots=1`, `ctx=1048576`, `kv=f8` | fits | `22.56 GiB` | `9.44 GiB` | `0.42 GiB` | `13.45 GiB` |

Admission tiers for F8 with the real pack:

| Context | Max slots | Per-GPU total at max |
|---:|---:|---:|
| `131072` | `126` | `31.94 GiB` |
| `262144` | `63` | `31.92 GiB` |
| `524288` | `31` | `31.75 GiB` |
| `1048576` | `15` | `31.43 GiB` |

## Interpretation

The target `32` slot / `256K` configuration is feasible only if KV is typed and
TP-sharded. Replicated f32 KV is numerically impossible at this target.

This explains the Sprint 326 behavior: bounded diagnostic f32 rows can pass,
but production serving needs a real typed persistent compressed-KV arena before
we expand history or rely on long-context serving.

## Current Gap

The planner contract is now explicit, but the runtime still uses diagnostic
f32 row buffers. The next sprint should implement the production allocator
against this contract and validate row reads from that arena.

## Artifacts

- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots32-ctx262144-f8.md`
- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots1-ctx1048576-f8.md`
- `logs/from-cluster/sprint327-tp-kv-contract/cluster/plan-slots32-ctx262144-f8.json`
