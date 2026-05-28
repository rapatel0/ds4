# Sprint 439: Masked-Copy Output-Head Validation Fix

## Objective

Fix the masked compact-copy validation gap from Sprint 438 and rerun token-level
output-head checks.

No PP/layer-split variants are in scope.

## Implementation

Fixed the parser for:

```text
--post-attention-masked-compact-copy-gate
```

The branch incorrectly advanced `i`, which skipped the next CLI flag. In the
Sprint 438 commands the skipped flag was `--diagnostic-output-head-lazy-gate`,
which is why proxy runs emitted serving-bench lines without output-head parity.

## V100 Evidence

Build:

```text
gpu-01:/localpool/ds4/workspace/ds4-sprint181
CUDA_ARCH=sm_70 make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Logs:

```text
/localpool/ds4/workspace/logs/sprint439-masked-copy-parity/maskedcopy-lazy-slots8-graph-head.stdout
/localpool/ds4/workspace/logs/sprint439-masked-copy-parity/fullcap-lazy-slots8-graph-head.stdout
```

Same rebuilt harness, `8` slots / `256K` / all layers / persistent graph:

```text
masked copy:
  projected_slot_step_tok_s=54.281002
  first_token=50845
  first_logit=20.302221298
  checksum=4331108964

full cap repeat:
  projected_slot_step_tok_s=38.706401
  first_token=164
  first_logit=19.380706787
  checksum=6566249846
```

## Decision

Do not promote masked compact copy yet.

The parser fix is valid and the masked-copy proxy remains faster, but token
parity is not proven. More importantly, the full-cap output-head diagnostic is
not stable across recent repeats: earlier full-cap runs selected `50845`, while
the same rebuilt full-cap repeat selected `164`. That makes a single all-layer
smoke output-head comparison too weak for promotion.

## Next

Move validation to an operational parity harness:

- run same-binary HTTP A/B at a small admitted shape with and without
  `--post-attention-masked-compact-copy-gate`;
- compare returned selected tokens / generated token sequences across all
  requests;
- only if HTTP parity passes, repeat at larger slot counts and profile route
  copy traffic.

Until then, keep masked compact copy diagnostic-only.
