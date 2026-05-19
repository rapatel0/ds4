# Sprint 045 Follow-Ups

## P0: Aggregate Slot/Context Envelope

The full gate now passes with `missing=aggregate_slot_context_envelope`.
Next sprint should define and validate the practical operating envelope:

- explicit admission for 1/2/4/8 configured slots;
- context tiers such as 128K, 256K, 512K, and 1M;
- per-tier memory reports for weights, KV, scratch, relay, MTP, and reserve;
- active microbatch size separate from configured slots;
- queueing or explicit rejection semantics for concurrent requests;
- aggregate tok/s and per-request latency reports.

## P1: True MTP Commit Path

Sprint 045 reports exact MTP draft acceptance but still returns the base token
after computing the target token. Future speculative serving work should add:

- target-state mutation for accepted drafts without recomputing target logits;
- rollback path for rejected served drafts;
- MTP-on versus MTP-off latency and tok/s comparisons;
- accepted-token accounting that distinguishes diagnostic acceptance from
  committed speculative speedup.

## P1: MTP Service Hardening

The resident MTP service is deliberately narrow. Production hardening should
cover:

- repeated MTP-enabled request loops beyond the single fixture;
- one-token requests where MTP diagnostics are skipped;
- MTP sidecar or output-head upload failure messages in `/metrics` or logs;
- configurable top-k and gpu/reserve values through the deployment templates;
- automated rollback-mode smoke after an MTP-enabled smoke.

## P2: Startup And Upload Refinement

Full-gate parallel open remains around one minute and varies by run. Future
work can investigate:

- persistent resident process strategy and operator restart policy;
- avoiding duplicate context parsing for the output-head binding;
- retaining sidecar/output-head uploads across repeated MTP service opens;
- per-stage startup regression thresholds.
