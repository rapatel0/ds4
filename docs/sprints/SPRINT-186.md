# Sprint 186 - Synthetic 4096-Token Context Tier

Date: 2026-05-22

## Objective

Extend the synthetic filled-context measurement track from 1024 tokens to 4096
tokens on the persistent production appliance pack.

## Scope

- Run direct replay with:
  - `--synthetic-prompt-token 926`
  - `--synthetic-prompt-len 4096`
  - `--ctx 262144`
  - `--tokens 2`
- Use the persistent Sprint 181 appliance pack:
  `/workspace/packs/ds4-appliance-full-tm-gated-s181`.
- Keep Sprint 183 online attention default-off.
- Record prompt replay, prompt tok/s, continuation tok/s, output IDs, and any
  failure mode.
- Copy cluster evidence and update the vision.

## Non-Goals

- No full 256K prefill in this sprint.
- No new serving API.
- No promotion of online-single attention.
- No kernel rewrite unless the 4096 tier exposes a blocking correctness or
  capacity bug.

## Definition of Done

- [x] V100 4096-token synthetic prompt run completes or a concrete blocking
      failure is recorded.
- [x] Evidence is archived under `logs/from-cluster/`.
- [x] Sprint result records timing or failure details.
- [x] Vision is updated with the 4096-tier result.
- [x] Changes are committed.

## Outcome

The 4096-token synthetic filled-context tier completed on the V100 pod using
the persistent Sprint 181 production appliance pack.

Command shape:

```text
tools/ds4-v100-replay \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --appliance-dir /workspace/packs/ds4-appliance-full-tm-gated-s181 \
  --synthetic-prompt-token 926 \
  --synthetic-prompt-len 4096 \
  --ctx 262144 \
  --tokens 2 \
  --json
```

## Evidence

| Synthetic token | Prompt len | Context | Generated tokens | Prompt replay ms | Prompt tok/s | Continuation tok/s | Output ids |
|---:|---:|---:|---:|---:|---:|---:|---|
| 926 | 4096 | 262144 | 2 | `288102.638` | `14.217155` | `13.354373` | `271, 5` |

Stage decode timing:

```text
[38176.502, 38836.970, 38837.897, 38926.719, 38555.177, 42646.587, 31632.196, 19903.468]
```

Cluster evidence:

```text
logs/from-cluster/sprint186-synthetic-4096-tier/len4096/synthetic-len4096.json
```

## Decision

This confirms the synthetic long-context harness can advance beyond the old
short-fixture regime. The 4096-token tier is still far below the target 256K+
context, but it is enough to show the practical measurement shape:

- prompt replay dominates wall time
- continuation decode remains in the `13-15 tok/s` direct one-slot range for
  these synthetic filled-context tiers
- full 256K prefill will be expensive enough that we should schedule it
  deliberately, not run it casually inside every optimization sprint

Next useful step is a decision sprint: either optimize prompt/prefill replay
before larger tiers, or schedule a long overnight-style synthetic prefill tier
such as 16384/65536 to get the next scaling point.
