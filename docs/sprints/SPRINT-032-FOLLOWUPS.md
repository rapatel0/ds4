# SPRINT-032 Follow-Ups

## Next Runtime Slice

- Implement the K=1 MTP forward probe using the resident gpu7 sidecar arena.
- Start with Q8_0 MTP prefix projections: token embedding, `enorm`, `e_proj`,
  HC repeat, `hnorm`, `h_proj`, and HC add.
- Add MTP block attention/KV, router, Q4_K routed experts, Q8_0 shared expert,
  output-head logits, and top-k comparison after prefix parity is proven.
- Keep speculative serving disabled until draft/verify/rollback state is
  explicit and the draft token matches a trusted oracle.

## Base Appliance Hardening

- Rename or split the status counter if needed: the current `served_requests`
  counts accepted HTTP requests, including health and status probes.
- Add structured request IDs and per-request failure JSON once the endpoint
  grows beyond the loopback smoke surface.
- Add a status field for configured limits: one slot, sequential serving, max
  generated tokens, MTP disabled, and streaming disabled.
- Keep `/v100/selected-token` as the internal correctness endpoint until a
  production API shape is chosen.

## Performance Follow-Ups

- Parallelize stage open/upload. Sprint 032 evidence shows fresh-process upload
  dominates at roughly 244-293 seconds.
- Add resident decode baselines for longer continuations after MTP forward
  correctness starts.
- Measure one-slot decode with context tiers beyond the short official fixture:
  4K, 32K, 128K, 256K, 512K, and 1M.
- Defer 2/4/8-slot aggregate throughput until the one-slot MTP path and request
  state ownership are correct.

## Deployment Follow-Ups

- Decide the first production packaging target: supervised internal endpoint,
  Kubernetes Deployment/Service, or OpenAI-compatible facade.
- Add process supervision and restart behavior only after the runtime can avoid
  repeated multi-minute upload costs on common operator workflows.
- Keep external exposure disabled until auth, request limits, and failure
  isolation are implemented.
