# Sprint 185 - Synthetic Context Tier Timing

Date: 2026-05-22

## Objective

Use the Sprint 184 synthetic prompt path to collect the first larger
filled-context timing tier beyond 256 synthetic prompt tokens.

## Scope

- Run direct replay with `--synthetic-prompt-token 926`.
- Use the persistent Sprint 181 production appliance pack.
- Record prompt replay and continuation decode timing for a `1024`-token
  synthetic prompt at `ctx=262144`.
- Keep online attention default-off for this control measurement.
- Fix synthetic prompt cache sizing if the first real long-context tier exposes
  an immediate capacity bug.
- Copy cluster evidence and update the vision.

## Non-Goals

- No runtime code change unless the measurement exposes a blocking bug.
- No full 256K prefill in this sprint.
- No online-attention promotion.

## Definition of Done

- [x] V100 `1024` synthetic prompt run completes.
- [x] Generated output and timing JSON are archived under `logs/from-cluster/`.
- [x] Sprint result records prompt replay ms, prompt tok/s, continuation tok/s,
      and output token IDs.
- [x] Vision is updated.
- [x] Changes are committed.

## Outcome

The first 1024-token run failed at layer 2 with:

```text
decode cache attention compressed capacity exceeded
```

That was a real filled-context blocker hidden by the previous short-prompt
benchmarks. Direct replay defaulted to `attn_comp_cap=64` and
`index_comp_cap=64`, which is enough for the old short fixture but not enough
for ratio-4 compressed rows beyond roughly 256 positions.

Updated synthetic prompt mode so direct replay sizes compressed decode cache
capacity from the synthetic prompt length:

```text
ceil((synthetic_prompt_len + generated_tokens) / 4) + 4
```

with the existing default as a floor. This keeps normal text-prompt serving
unchanged and avoids allocating full `ctx/4` compressed rows when a synthetic
measurement only needs a bounded prefix.

## Evidence

Build:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed on `llm/llamacpp-build-8gpu`.

1024-token synthetic filled-context run:

| Synthetic token | Prompt len | Context | Generated tokens | Prompt replay ms | Prompt tok/s | Continuation tok/s | Output ids |
|---:|---:|---:|---:|---:|---:|---:|---|
| 926 | 1024 | 262144 | 2 | `66918.694` | `15.302152` | `15.198459` | `926, 926` |

Stage decode timing from the JSON:

```text
[9876.568, 9247.724, 9248.722, 9217.706, 9172.163, 7721.094, 7622.962, 4708.878]
```

Cluster evidence:

```text
logs/from-cluster/sprint185-synthetic-context-tier/len1024/synthetic-len1024.json
```

## Decision

Keep the synthetic cache-sizing fix. It is scoped to synthetic prompt mode and
it allowed the first 1024-token filled-context replay to complete.

This result confirms that the previous 256K serving numbers were not measuring
filled 256K decode. They are still valid capacity/serving-throughput numbers,
but long-context performance needs its own tiered measurement track.

Next tiers should be `4096` and then a deliberately scheduled full-context
prefill if the operator wants the wall-clock cost. The 1024-token tier suggests
prompt replay is already the dominant cost for filled-context tests, while the
single continuation step remains around `15 tok/s` in direct one-slot mode.
