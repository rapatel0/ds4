# SYS Transport Sweep — replace every hot-path peer_copy with NCCL

Single self-contained sprint spec. Replace every `ds4_peer_copy_async` and
`enqueue_graph_f32_copy_*` call in the hot path with a non-reducing NCCL
collective. Same arithmetic, same bytes, NCCL-routed → topology-aware,
NVLink-only, **0 SYS**, graph-capturable. Bit-exact by construction.

Do all sites in one sprint — order does not matter; gate as a single A/B at the
end. Builds and tests are cheaper bundled.

## Why now

`s478-peer-site-self` peer-accounting found Direct SYS ≈ **256 MB / 35 K ops** at
the small shape and **184 GB / 20.5 M ops** at the full shape. On
V100-SXM2, NVLink is not full all-to-all, so `ds4_peer_copy_async` between
non-NVLink-adjacent ranks falls back to PCIe/SYS. NCCL ring/tree respects the
topology → 0 SYS, every reported NCCL graph already shows this. The single fix
class is "move every cross-rank byte through NCCL."

The HC top-site (~67 MB of 256 MB at the small shape) is only ~26% of SYS
bytes. The rest is **EP all-to-all** (pairwise dispatch and combine over the full
shard, O(N²) per layer × 43 layers), **router-plan broadcasts**, **attention /
indexer / sinks staging**, and the **graph-capturable peer-copy wrappers** that
share the same underlying transport. All of them go in this sweep.

## Bit-exactness guardrails (read this before you touch anything)

This sweep is bit-exact **because every replacement is a non-reducing
collective**. Three concrete traps that would silently break that property —
do not fall into them:

1. **Do not promote `peer_copy + local kernel-sum` into `ncclReduceScatter`**
   (or `ncclAllReduce`) **anywhere.** The EP compose region (around
   `tools/ds4-v100-tp-ep-full-layer-smoke.cu` ~12886–12914) is tempting
   because it currently does pairwise peer_copy then a local kernel sums
   `d_ep_remote[0..7]` in fixed `for src=0..7` order — fusing into
   `ncclReduceScatter` would replace that fixed-order sum with NCCL's
   tree-reduction order and change fp32 results at ulp scale.
   **That is arithmetic-changing** and belongs under the tolerance gate in a
   separate sprint, not here. In this sweep: replace **only** the peer_copy
   with `ncclAlltoAll` / grouped send/recv, and **keep** the local kernel-sum
   that follows.
2. **The sweep covers `enqueue_graph_f32_copy_*` wrappers too, not just
   `ds4_peer_copy_async`.** `enqueue_graph_f32_copy_from_device0` and
   `enqueue_graph_f32_copy_between_devices` are graph-capturable peer-copy
   wrappers — same SYS exposure, same swap. Don't miss them by grepping only
   the macro name.
3. **Compact-route copy is sparse — use grouped Send/Recv (AlltoAllv idiom),
   not `ncclAlltoAll`.** `copy_compact_active_route_shard_kernel` elides
   zero-routed rows, so per-pair byte counts are variable. `ncclAlltoAll`
   would move zero rows wastefully (or wrong); grouped `ncclSend`/`ncclRecv`
   with the existing per-pair active-route counts is the right fit.

A fourth implicit guardrail: only the four **non-reducing** NCCL collectives are
allowed in this sweep — `ncclBroadcast`, `ncclAlltoAll`, `ncclSend`,
`ncclRecv`. Anything that reduces (`ncclAllReduce`, `ncclReduceScatter`,
`ncclReduce`) is out of scope here because it changes summation order.

## Site map

All targets are in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`. Line numbers
are from the current tree; verify with a fresh grep before editing — recent
edits have shifted line numbers.

| # | Domain | Symbols / sites | Pattern | Replacement |
|---|---|---|---|---|
| 1 | HC current GPU0→all (d_hc_split, d_current_full, d_attn_normed, d_ffn_normed, …) | `ds4_peer_copy_async` GPU0→ranks at multiple HC sites | small scalar broadcast and slot-major full-hidden broadcast (~512 KB) | `ncclBroadcast(root=0, ..., r.compose_nccl)` |
| 2 | Router-plan GPU0→all (`d_router_selected_plan`, `d_router_weights_plan`) | 6 peer_copy sites | small metadata, many ops | `ncclBroadcast(root=0, ...)` |
| 3 | EP dense pairwise dispatch (`d_ep_remote[src]` ← `d_ep_contrib_all + dst*shard`) | rank↔rank pairwise peer_copy, uniform per pair | shard-sized × O(N²) per layer | **`ncclAlltoAll`** (per-rank, wrapped in `ncclGroupStart/End`) |
| 4 | EP routed pairwise dispatch — sparse (compact_route) | per-pair peer_copy with variable counts (`routed_compose_rows`-driven) | variable per pair | **grouped `ncclSend`/`ncclRecv` with per-pair active-route counts** (AlltoAllv idiom) |
| 5 | Attention compressed KV / score / kv_full | GPU0→all peer_copy of normed/compressed buffers | medium-byte broadcasts | `ncclBroadcast(root=0, ...)` |
| 6 | Sparse-indexer state (`d_indexer_topk`, comp_kv / comp_score) | GPU0→all + some rank↔rank | mixed | `ncclBroadcast` for GPU0-sourced; grouped `ncclSend`/`ncclRecv` for pairwise |
| 7 | Attention sinks (`d_attn_sinks`) | GPU0→all per-layer slice | small broadcasts | `ncclBroadcast(root=0, ...)` |
| 8 | Input embedding distribution | GPU0→all peer_copy of embedding-normed | one-shot per request batch | `ncclBroadcast(root=0, ...)` |
| 9 | **Graph-capturable peer-copy wrappers** | `enqueue_graph_f32_copy_from_device0` and `enqueue_graph_f32_copy_between_devices` (callers throughout the hot path) | same SYS exposure as raw peer_copy | same replacements as the matching pattern in 1–8 |

Note for the EP compose region: the peer_copy in mode (a) is in scope; the
local kernel-sum that follows is **not** to be modified (guardrail #1). Mode
(b)'s existing `ncclReduceScatter` path is already NCCL — leave it as-is, do
not enable it as part of this sweep.

## Replacement patterns

### GPU0→all (categories 1, 2, 5, 7, 8, and the matching 9 wrapper sites)

```cpp
// before
ds4_peer_copy_async(r.d_dst, r.device,
                    hc->d_src, opt.devices[0],
                    bytes, r.stream);

// after — issue from every rank inside a group, root=0
ncclGroupStart();
for (int rk = 0; rk < kGpus; ++rk) {
  cudaSetDevice(ranks[rk].device);
  ncclBroadcast(/*sendbuff*/ rk == 0 ? hc->d_src : nullptr,
                /*recvbuff*/ ranks[rk].d_dst,
                count, ncclDataType, /*root*/ 0,
                ranks[rk].compose_nccl, ranks[rk].stream);
}
ncclGroupEnd();
```

### EP dense alltoall (category 3) — uniform per-pair

```cpp
// before — O(N²) pairwise peer_copy
for (src) for (dst != src)
  ds4_peer_copy_async(remote[dst][src], src.contrib + dst*shard, ...);

// after — single NCCL Alltoall per rank, group-wrapped
ncclGroupStart();
for (int rk = 0; rk < kGpus; ++rk) {
  cudaSetDevice(ranks[rk].device);
  ncclAllToAll(ranks[rk].d_ep_contrib_all,    // [N, shard_elems]
               ranks[rk].d_ep_remote_flat,    // [N, shard_elems]
               shard_elems, ncclDataType,
               ranks[rk].compose_nccl, ranks[rk].stream);
}
ncclGroupEnd();
// downstream local kernel-sum of d_ep_remote[0..7] is UNCHANGED
```

### EP routed sparse alltoall (category 4) — variable per-pair (AlltoAllv)

```cpp
// before — per-pair peer_copy with copy_elems driven by routed_compose_rows
//          (compact_route path; the kernel copy_compact_active_route_shard_kernel
//           variant uses the same per-pair element counts)

// after — grouped send/recv per pair, reusing the existing counts table
ncclGroupStart();
for (int src = 0; src < kGpus; ++src) {
  cudaSetDevice(ranks[src].device);
  for (int dst = 0; dst < kGpus; ++dst) {
    if (dst == src) continue;             // honor existing skip_self_copy
    const size_t send_elems = counts[src][dst];  // existing per-pair elems
    const size_t recv_elems = counts[dst][src];
    if (send_elems) ncclSend(send_buf(src, dst), send_elems, dt, dst,
                             ranks[src].compose_nccl, ranks[src].stream);
    if (recv_elems) ncclRecv(recv_buf(src, dst), recv_elems, dt, dst,
                             ranks[src].compose_nccl, ranks[src].stream);
  }
}
ncclGroupEnd();
```

If `compact_route` is OFF, category 4 collapses to category 3 (`ncclAlltoAll`).

### Rank↔rank pairwise (category 6 indexer pairs, any other non-EP pairwise)

Same as category 4: grouped `ncclSend`/`ncclRecv` with whatever per-pair byte
counts the existing code computed for `ds4_peer_copy_async`'s `bytes` argument.

## Gating policy (inline, self-contained)

This sweep does not depend on any external policy doc; the gates below are the
gates.

- **Parity — strict bit-exact selected-token, 256/256 vs control** on the
  reference shape (32 slots / 256K context / 256 requests / 64 generated tokens).
  A mismatch on a transport-only swap is a real bug — find it, do not relax
  the gate. The most common bug class will be that something assumed to be
  transport-only actually changed arithmetic (e.g., a kernel-sum got
  accidentally promoted into a reducing collective — see guardrail #1).
- **SYS — per-site Direct SYS bytes/ops → 0** with peer accounting enabled.
  Verify per site, not just aggregate; the only acceptable end state is that
  every replaced site individually reports 0 SYS bytes in the post-swap run.
  Aggregate Direct SYS bytes for the run should be **≈ 0**.
- **Perf — server decode tok/s and request-window GPU util must not regress**
  vs control. A net gain is expected (SYS bytes carving out of EP's 473 ms
  bucket and HC-current's 357 ms bucket), but it is not required to promote:
  transport-only is justified by SYS alone, the perf upside is a bonus.
- **NCCL hygiene:**
  - Every new collective uses `r.compose_nccl` (already `ncclCommInitAll`'d in
    `open_compose_nccl`); wrap multi-rank issue in `ncclGroupStart/End`.
  - **Warm up each new collective before any graph capture.** NCCL collectives
    are graph-capturable; peer_copy is not. This sweep therefore also unblocks
    `C1` piecewise graph capture later, but capture itself is out of scope here.
  - Small payloads (HC split, router plan): NCCL will auto-pick Tree+LL128
    based on size. Verify once with `NCCL_DEBUG=INFO`; pin
    `NCCL_ALGO=Tree NCCL_PROTO=LL128` if needed.
- **Tolerance vs other sweeps:** this sweep's bit-exact gate is **non-negotiable**
  — do not relax it to accommodate any arithmetic-changing change that
  arrives in parallel (A2 mix all-reduce, A3 router all-reduce, A4b
  row-parallel consumers, A6 rank-local norm). Those live in other sprints and
  have their own (tolerance) gate.

## Telemetry to report (control vs candidate)

- Selected-token parity (must be **256/256**).
- Aggregate Direct SYS bytes / ops (target **≈ 0**).
- **Per-site** Direct SYS bytes — every site in categories 1–9 must
  individually report **0**.
- Server decode tok/s, projected slot-step tok/s.
- Request-window GPU util (avg + max).
- rxpci / txpci trace during the request window (visual evidence the PCIe
  spikes are gone).
- Updated decode-domain table (EP / HC-current / final HC / compose / dense /
  other) — EP's share is expected to drop visibly if transport was a real
  component, which steers the next priority.

## Stop condition

Stop when **all** of the following hold on the reference shape:

1. Selected-token parity 256/256.
2. Aggregate Direct SYS bytes ≈ 0; per-site SYS bytes 0 at every replaced site.
3. rxpci/txpci traces show no per-request-window PCIe traffic during decode.
4. Decode tok/s and GPU util are at-or-above control.

Then report the updated domain table. If EP shrank meaningfully → next priority
is the structural EP work (MTP / TP-experts / fused dispatch+combine). If
HC-current is still dominant → resume the arithmetic-changing rank-local work
(A2 mix all-reduce / A3 router all-reduce / A6 rank-local norms) under the
tolerance gate.

## Out of scope

- **Arithmetic-changing changes:** A2 mix/RMS all-reduce, A3 router all-reduce,
  A4b row-parallel consumers, A6 rank-local norms. These all reorder a
  reduction and require the tolerance gate, not the bit-exact gate this sweep
  uses. Do not bundle them; do not let them sneak into this sprint as
  "while-we're-in-there" optimizations.
- **Structural EP redesign:** MTP, TP-sharded experts, dispatch /
  grouped-GEMM / combine fusion. The transport swap may carve a real chunk
  off the 473 ms EP bucket on its own; structural work happens separately.
- **Graph capture:** keep new collectives capturable (and warm them up before
  any future capture), but capture itself is its own sprint (C1).
- **EP compose mode (b)** — `ncclReduceScatter` is already NCCL; leave it as
  it is, do not toggle it on as part of this sweep.

## One-line summary

Every `ds4_peer_copy_async` and `enqueue_graph_f32_copy_*` in the hot path
becomes `ncclBroadcast` / `ncclAlltoAll` / grouped `ncclSend`+`ncclRecv` on
`r.compose_nccl`; the local kernel-sum after EP dispatch stays; the gate is
selected-token 256/256 plus per-site Direct SYS bytes = 0.
