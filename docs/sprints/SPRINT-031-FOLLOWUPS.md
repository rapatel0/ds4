# SPRINT-031 Follow-Ups

## Next Runtime Slice

- Implement a K=1 MTP forward probe using the resident sidecar object.
- Keep the probe gpu7-local at first and consume the final base-model HC state
  after normal selected-token decode.
- Start with the MTP prefix path: token embedding, `enorm`, Q8_0 `e_proj`,
  HC repeat, `hnorm`, Q8_0 `h_proj`, and HC add.
- Then add the MTP block path: HC attention split, MTP attention/KV update,
  router, Q4_K routed experts, Q8_0 shared expert, and HC expansion.
- Finish with MTP output logits/top-k using the base model output head.

## Kernel Work

- Add a test that directly exercises `ds4_gpu_routed_moe_one_tensor` on Q4_K
  MTP expert tensors.
- Keep Q4_K MTP expert execution separate from the main MXFP4 expert arena
  path.
- Decide whether MTP Q8_0/Q4_K kernels should consume the compact resident
  arena directly or keep using GGUF offsets with a dedicated MTP map/cache.
- Add K=1 CPU/GPU parity for Q8_0 `e_proj` and `h_proj` before comparing full
  draft tokens.

## Scheduler Integration

- Add an optional MTP path to `ds4_v100_replay_options`.
- Expose a gpu7 scheduler hook that can read the current final HC state without
  copying it through host memory.
- Keep draft generation probe-only until rollback and verifier state are
  explicit.
- Keep readiness at `missing=mtp_forward` until the MTP draft token matches a
  trusted oracle on the official prompt.

## Performance Follow-Ups

- Parallelize resident uploads across stage schedulers after K=1 MTP forward
  correctness is underway.
- Add longer resident decode baselines after MTP forward probe correctness.
- Revisit gpu7 memory budgeting once MTP scratch and raw-cache tensors are
  allocated alongside the resident sidecar arena.
