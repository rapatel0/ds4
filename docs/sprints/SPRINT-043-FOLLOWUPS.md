# Sprint 043 Follow-Ups

## P0: Throughput Optimization And Operating Envelope

The full gate now passes with `missing=throughput_optimization`. Next sprint
should stop treating timing as incidental diagnostics and produce an explicit
operating envelope:

- fresh-process startup/upload timing;
- resident prompt replay timing;
- resident continuation decode timing;
- context-tier measurements;
- 1/2/4/8 slot admission analysis;
- first targeted optimization with before/after evidence.

The most obvious optimization target is parallel stage open/upload. Sprint043
evidence shows fresh-process open/upload around `289-345 s`, while the
continuation decode path is around `143-153 ms` per token in the short fixture.

## P1: Production MTP Serving Object

The production service remains `base_one_slot` with `mtp_enabled=false`.
Native MTP correctness is proven by gates, but speculative serving still needs:

- a resident draft-session object;
- device-local embedding and HC handoff where practical;
- accept/reject transaction boundaries;
- integration with the HTTP replay loop;
- rollback-safe metrics and diagnostics.

## P1: Request Surface Hardening

Before any external exposure, add:

- authentication or a protected proxy;
- request IDs;
- bounded request body size;
- structured failure JSON;
- concurrency rejection or queueing semantics;
- optional OpenAI-compatible facade.

## P2: Deployment Manifest Hardening

The systemd and Kubernetes files are templates. Before using them as a
long-lived production controller, confirm:

- the final workspace path on `gpu-01`;
- persistent artifact/log path;
- image and dependency installation strategy;
- readiness/liveness timeout values after startup upload is optimized;
- whether the service should remain loopback only or sit behind an internal
  authenticated proxy.
