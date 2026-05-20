# Sprint 090: Full Appliance Pack And Scheduler Run

## Goal

Move from bounded stage-0 validation to a full 8-GPU appliance directory that
can run the scheduler without the source GGUF model map.

## Implementation Plan

- Generate a full appliance pack with `tools/ds4-v100-appliance-pack`:
  `gpu0.weights` through `gpu7.weights`, `pack-index.tsv`, and
  `turbomind-pack-index.tsv`.
- Verify that each shard fits the 32 GB V100 budget with enough headroom for
  KV, scratch, relay buffers, CUDA overhead, and MTP-disabled baseline decode.
- Run `cuda_v100_full_scheduler_smoke --appliance-dir ... --stages 8` on the
  cluster.
- If the full scheduler smoke passes, run a bounded one-slot replay timing from
  the appliance directory and capture generated tok/s.

## Definition Of Done

- [x] Full appliance pack generation completes on V100.
- [x] Per-GPU shard sizes are recorded and fit the baseline budget.
- [x] Full 8-stage scheduler smoke executes all 43 layers from the appliance
  directory.
- [x] If full smoke passes, one appliance replay timing run is captured.
- [x] Cluster log is committed.

## Result

Sprint 090 promotes the TurboMind appliance format from bounded stage-0 smoke
to a full 8-GPU appliance artifact on the V100 cluster.

The original build pod wrote to `/dev/md0`; that is the mirrored host/root
disk and is not appropriate for 142 GiB appliance artifacts. The sprint
recreated `llamacpp-build-8gpu` with `/workspace` mounted from
`/var/lib/rancher/k3s/storage/ds4-sprint090-workspace`, which resolves to
`localpool/k8s-local` on `gpu-01`.

Full pack generation:

```text
ds4-v100-appliance-pack: wrote /workspace/ds4-appliance-full-tm-s090 source_rows=1199 tm_rows=129 skipped_rows=0 source_bytes=8973123932 tm_weight_bytes=138512695296 tm_scale_bytes=8657043456
ds4-v100-appliance-pack: gpu0.weights bytes=22524134668
ds4-v100-appliance-pack: gpu1.weights bytes=21494393612
ds4-v100-appliance-pack: gpu2.weights bytes=21494393612
ds4-v100-appliance-pack: gpu3.weights bytes=21494393612
ds4-v100-appliance-pack: gpu4.weights bytes=21494393612
ds4-v100-appliance-pack: gpu5.weights bytes=17922654732
ds4-v100-appliance-pack: gpu6.weights bytes=17901334540
ds4-v100-appliance-pack: gpu7.weights bytes=11817197824
```

Full scheduler smoke:

```text
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=1 layers=43 tm_layers=43 last=40-42 gpu=7 uploaded_tensors=8 uploaded_bytes=156142896212 expert_last=26 ok
```

Replay timing from the same appliance directory:

```text
tokens: 926 / "16" / 3136, then EOS
open_total_ms: 73897.062
prompt_replay_ms: 3109.929
continuation_decode_ms: 105.353
continuation_tokens_per_second: 9.491896
generated_tokens_per_second: 0.620997
uploaded_tensors: 8
uploaded_bytes: 156142896212
```

This is a correctness and residency milestone, not the final practical serving
target. It proves the single appliance directory works and eliminates the
source GGUF scheduler residency path for baseline decode. The next throughput
work should wire `--appliance-dir` into the launcher/service path and then run
multi-slot async appliance benchmarks against this format.

## Stop Conditions

- Stop and document if any shard exceeds the VRAM budget.
- Stop and document if the full scheduler cannot execute from shard-backed
  model offsets without a bounded code fix.
- Stop before broad optimization if the runtime still falls back to source GGUF
  residency or transient TurboMind repacking.
