# Sprint 600 Follow-Ups

## 1. Promoted path corrupts tokens ~1/256 steps (CRITICAL disclosure)

- **What**: The NCCL-collective race fires on the PROMOTED serving path
  (~1/256 steps, caught live in rctl600; whole-batch events, up to 12/32
  token flips). Every tolerance result in the repo's history sampled around
  a real, rare nondeterminism. Until a fix lands, deterministic-parity
  claims should be read as "modulo a ~0.4%-of-steps NCCL race", and any
  future bit-exact gate should run the s600 per-step checksum tool to
  detect events rather than rely on one lucky run.
- **Severity**: Critical (correctness disclosure; mitigated by rarity).
- **Suggested sprint**: 601 (the /dev/shm-unblocked dedicated-comm retest is
  the first fix candidate).
- **Files**: `tools/` s600 first-divergence checksums;
  `logs/from-cluster/sprint600/`.

## 2. /dev/shm cap is pod config, not platform (orchestrator finding)

- **What**: The dedicated-head-comm fix candidate was "blocked" by NCCL
  needing 33.5 MiB /dev/shm per communicator vs the pod's 64 MiB — but that
  is the bare Kubernetes emptyDir default; the manifest mounts no shm
  volume. One-line fix: `emptyDir: {medium: Memory, sizeLimit: 16Gi}` at
  /dev/shm in `deploy/v100/ds4-v100-build-localpool.pod.yaml` (pod recreate
  is safe — /workspace is hostPath). This reopens the comm-isolation fix
  family (dedicated comms per collective class, possibly per layer-group).
- **Severity**: Critical (gates the 220 verdict revision and item #1's fix).
- **Suggested sprint**: 601, first task.
- **Files**: `deploy/v100/ds4-v100-build-localpool.pod.yaml`.

## 3. NVIDIA escalation package

- **What**: The sprint assembled a reproducer/evidence package for the
  captured-NCCL race (single 8-rank-per-process comm, V100, graph capture).
  File it (NCCL GitHub issue) — version-independence (2.19.3, 2.27.7) and
  the pacing trigger law make it a strong report.
- **Severity**: Important (parallel-path insurance).
- **Suggested sprint**: any; orchestrator/user action (outward-facing).
- **Files**: `/workspace/s600-artifacts/` reproducer legs.

## 4. HEAD_COMM=host changes reduction associativity

- **What**: The host-side head-reduction mode produces 0.969 selected-token
  agreement vs control (sum-order change, expected). If it ever becomes a
  candidate again, it needs its own re-anchored control, not the s597 one.
- **Severity**: Nice-to-have (mode is opt-in/diagnostic).
- **Files**: `engine/output_head.cu`.

## 5. C-B restack / C-C route-plan shadow still unattempted

- **What**: Carried from s599; moot until an exchange/fix promotes. Revisit
  in the post-601 stack.
- **Severity**: Important (part of the ≥50/slot budget).

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| Promoted-path ~1/256 corruption disclosure | Critical | 601 fix | s600 checksum tools |
| /dev/shm cap is pod config — lift + retest dedicated comm | Critical | 601 first task | deploy/v100 pod manifest |
| NVIDIA escalation filing | Important | user/orchestrator | s600 artifacts |
| host head-reduction needs own control | Nice-to-have | if revisited | engine/output_head.cu |
| C-B/C-C restack | Important | post-601 stack | engine/ |
