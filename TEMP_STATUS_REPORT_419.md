# TEMP Status Report 419

Date: 2026-05-26

## Sprint

Sprint 416: Graph-Safe Attention Projection Direct Fill

## Objective

Test whether the true DS4 attention projection prefix can avoid one
graph-captured device-0-to-rank hidden-state copy by filling both Q_A and
KV-latent half inputs directly from `hc->d_attn_normed`.

## Implementation

Added an opt-in gate:

```text
--true-ds4-attention-projection-direct-input-fill-gate
```

Added:

```text
fill_two_hidden_inputs_half_from_current_kernel
```

When enabled, `run_true_ds4_attention_projection_prefix` skips:

```text
r.d_current_full <- hc->d_attn_normed
fill q_a input from r.d_current_full
fill kv_latent input from r.d_current_full
```

and replaces that with one fused two-output half-fill directly from
`hc->d_attn_normed`.

## Build

V100 build passed:

```text
make -B -j80 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_70 tools/ds4-v100-tp-ep-full-layer-smoke
```

## Resident Layer 2 A/B

Artifacts:

```text
/localpool/ds4/workspace/logs/sprint416-attn-proj-direct-fill/resident-layer2-baseline/
/localpool/ds4/workspace/logs/sprint416-attn-proj-direct-fill/resident-layer2-directfill/
```

Results:

| Mode | Checksum | Capture | Replay ms | Decode ms/step | Slot-step tok/s | Nodes |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 8290057485 | pass | 10.302464 | 2.575616 | 3106.053217 | 789 |
| direct fill | 8290057485 | pass | 9.886720 | 2.471680 | 3236.665037 | 773 |

Resident layer result was positive: same checksum, fewer nodes, and about
4.0% faster replay for layer 2.

## All-Layer Reduced/Deferred Check

Artifact:

```text
/localpool/ds4/workspace/logs/sprint416-attn-proj-direct-fill/all-layer-directfill-slot8-tokens4/
```

Shape:

```text
slots=8
ctx=262144
decode_steps=4
tp-runtime-scratch=512 MiB
defer-nccl-init=on
hc-current-nccl=on
persistent graph replay=on
```

Result:

```text
aggregate_generated_tok_s_decode=84.710170
aggregate_continuation_tok_s_decode=94.058547
checksum=4335215310
capture_succeeded=43/43
replay_succeeded=172/172
nodes=111492
PASS
```

Comparison baseline from `TEMP_STATUS_REPORT_418.md`:

```text
aggregate_generated_tok_s_decode=85.703793
aggregate_continuation_tok_s_decode=94.353496
nodes=114244
PASS
```

## Decision

Do not promote.

The gate is correct and graph-capturable, and it reduces graph nodes, but the
full all-layer path regresses slightly. This matches the earlier compressed
direct-input-fill evidence: replacing local staged reads with remote-source
fills can reduce launches/nodes while losing enough memory locality to regress
the aggregate path.

Keep the gate diagnostic-only.

## Next Bottleneck Direction

Continue rank-local layout work, but avoid direct remote-source fill as the
main pattern. Better candidates:

- preserve local staging while fusing downstream consumers
- make rank-major/current-full layouts consumable directly by dense kernels
- remove full-hidden materialization where the following op only needs a shard
- target compressed/indexer dense input staging or post-attention FFN input
  with local layout, not GPU0 remote read
