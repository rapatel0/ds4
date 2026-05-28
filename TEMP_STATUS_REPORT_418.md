# TEMP Status Report 418

Date: 2026-05-26

## Topline

Current best validated full all-layer TP/EP serving path remains:

- 8 slots
- 256K context
- persistent CUDA graph replay
- NCCL current-HC allgather
- aggregate decode: about 90.8 generated tok/s, 95.7 continuation tok/s

The new peer-copy experiment did not become a production candidate. CUDA graph capture rejects `cudaMemcpyPeerAsync` in the captured replay path with:

```text
cuda error tools/ds4-v100-tp-ep-full-layer-smoke.cu:4478:
operation not permitted when stream is capturing
```

## What Changed

Added an opt-in gate in `tools/ds4-v100-tp-ep-full-layer-smoke.cu`:

```text
--decode-cudagraph-peer-copy-gate
```

The gate routes graph-mode `copy_f32_kernel` transfers through `cudaMemcpyAsync` / `cudaMemcpyPeerAsync` instead of the remote-read copy kernel. It covers the main graph-mode copy sites:

- HC split replication
- current hidden replication
- attention projection input replication
- compressed KV and indexer-current row replication
- raw KV replication
- attention sink replication
- post-attention FFN input replication
- EP compose all-to-all copy path

## Validation

Build passed on the V100 node:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

### Resident Layer 2 Baseline

Artifact:

```text
/localpool/ds4/workspace/logs/sprint418-peer-copy/resident-layer2-baseline/
```

Result:

```text
layer=2
slots=8
ctx=262144
decode_steps=4
decode_ms_per_step=2.399744
decode_slot_step_tok_s=3333.688880
capture_succeeded=1
replay_succeeded=1
PASS
```

### Resident Layer 2 Peer-Copy Gate

Artifact:

```text
/localpool/ds4/workspace/logs/sprint418-peer-copy/resident-layer2-peer/
```

Result:

```text
FAIL
cudaMemcpyPeerAsync during graph capture is not permitted in this path.
```

Conclusion: replacing graph-mode remote copy kernels with peer memcpys is not viable without restructuring capture boundaries or staging the copies outside the graph.

## Full All-Layer A/B Attempts

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint418-peer-copy/slot8-tokens8-clean/
/localpool/ds4/workspace/logs/sprint418-peer-copy/baseline-after-rebuild-slot8-tokens8/
/localpool/ds4/workspace/logs/sprint418-peer-copy/reduced-baseline-slot8-tokens8/
/localpool/ds4/workspace/logs/sprint418-peer-copy/reduced-baseline-exclusive-slot8-tokens8/
/localpool/ds4/workspace/logs/sprint418-peer-copy/reduced-defer-baseline-slot8-tokens4/
/localpool/ds4/workspace/logs/sprint418-peer-copy/reduced-defer-peer-slot8-tokens4/
```

Findings:

- Full all-layer runs became sensitive to resident-load headroom after rebuild.
- Baseline without deferred NCCL OOMed during expert scale allocation.
- Reduced memory with `--tp-runtime-skip-unused-comp-state-gate --tp-runtime-scratch-mib 512` still needed deferred NCCL to pass reliably.
- Competing DS4 smoke processes repeatedly appeared during full A/B attempts and caused additional OOM / `cublasCreate` failures.

The clean deferred-NCCL baseline did pass:

```text
artifact: /localpool/ds4/workspace/logs/sprint418-peer-copy/reduced-defer-baseline-slot8-tokens4/
slots=8
ctx=262144
decode_steps=4
aggregate_generated_tok_s_decode=85.703793
aggregate_continuation_tok_s_decode=94.353496
capture_succeeded=43/43
replay_succeeded=172/172
PASS
```

The peer-copy full run did not reach decode because a separate slot-1 smoke process started during the run and consumed about 15 GiB/GPU. The resident-layer result is still sufficient to reject the peer-copy approach under current graph capture semantics.

## Bottleneck Interpretation

The late-window nvprof trace still points at graph-safe copy kernels as a real replay cost, but the direct peer-copy replacement is blocked by CUDA graph capture rules. The current copy path is therefore not just a poor choice; it is one of the few graph-capturable ways we currently move those remote values.

The likely next useful direction is not `cudaMemcpyPeerAsync` inside the graph. It is to reduce or avoid the graph-captured copies:

- keep more intermediate tensors rank-local instead of gathering to device 0 and redistributing
- use NCCL collectives where they are capture-compatible and memory-safe
- add fused graph-safe kernels that consume remote or rank-major layouts directly
- remove redundant full-hidden materialization before dense/route input packing

## Next Step

The next TP/EP-only sprint should focus on rank-local layout propagation:

1. Audit which `copy_f32_kernel` sites are caused by device-0 canonical tensors.
2. Pick one high-count family, likely attention projection input or compressed KV current-row replication.
3. Change the downstream consumer to accept rank-local/rank-major layout directly.
4. Re-profile resident layer 2 and then full all-layer deferred-NCCL 8-slot decode.

This keeps the graph replay model intact while attacking the real data movement.
