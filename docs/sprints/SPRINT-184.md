# Sprint 184 - Synthetic Long-Context Replay

Date: 2026-05-22

## Objective

Add a replay path that can prefill a synthetic token sequence without a huge
text prompt file, so we can measure actual filled-context decode rather than
only 256K capacity/admission.

## Rationale

The current sustained benchmarks use a short prompt while configuring 256K or
1M context capacity. That validates memory residency, admission, and scheduler
behavior, but it does not exercise a filled long-context KV state. Sprint 182's
profile still found attention-stage cost, and Sprint 183 showed online
attention can move throughput, but the next decision needs a benchmark that
places decode near a real long-context position.

## Scope

- Add CLI options to `tools/ds4-v100-replay`:
  - `--synthetic-prompt-token ID`
  - `--synthetic-prompt-len N`
- Make synthetic prompt mode mutually exclusive with `--prompt` and
  `--prompt-file`.
- Build `ds4_tokens` directly from the repeated token ID.
- Validate a small synthetic prompt locally/on V100.
- Run one bounded long-context V100 measurement that reaches a materially
  larger position than the normal 18-token fixture.

## Non-Goals

- No HTTP serving request-schema change.
- No promotion of Sprint 183 online attention.
- No requirement to prefill a full 256K sequence in this sprint if runtime is
  too long; the important change is creating the harness path.

## Definition of Done

- [x] `tools/ds4-v100-replay --help` documents synthetic prompt options.
- [x] Invalid combinations fail clearly.
- [x] V100 build passes.
- [x] A small synthetic prompt selected-token smoke passes or records the
      expected token if no fixed expected token is known.
- [x] At least one longer synthetic prompt run records prompt/decode timing.
- [x] Sprint 184 evidence is copied into `logs/from-cluster/`.
- [x] Vision is updated.
- [x] Changes are committed.

## Outcome

Added direct replay synthetic prompt support:

```text
--synthetic-prompt-token ID
--synthetic-prompt-len N
```

This creates a `ds4_tokens` prompt by repeating a token ID, bypassing tokenizer
and huge prompt text files. Synthetic mode is mutually exclusive with
`--prompt`, `--prompt-file`, and non-empty `--system`.

Parser validation on the V100 pod:

- `tools/ds4-v100-replay --help` includes the synthetic options.
- Combining `--prompt` with synthetic mode exits with rc `2` and a clear
  message.

Build validation:

```text
make -j80 CUDA_ARCH=sm_70 tools/ds4-v100-replay
```

passed on `llm/llamacpp-build-8gpu`.

## Evidence

Small synthetic prompt smoke:

| Synthetic token | Prompt len | Context | Generated tokens | Prompt tok/s | Continuation tok/s | Output ids |
|---:|---:|---:|---:|---:|---:|---|
| 926 | 8 | 4096 | 2 | `4.184295` | `14.055557` | `201, 926` |

Longer bounded synthetic prompt:

| Mode | Synthetic token | Prompt len | Context | Generated tokens | Prompt replay ms | Prompt tok/s | Continuation tok/s | Output ids |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control | 926 | 256 | 262144 | 2 | `20108.067` | `12.731209` | `14.071008` | `3955, 361` |
| online-single | 926 | 256 | 262144 | 2 | `20170.419` | `12.691853` | `12.038632` | `3955, 361` |

Cluster evidence:

```text
logs/from-cluster/sprint184-synthetic-long-context/
```

## Decision

Keep the synthetic prompt mode. It closes a measurement gap: we can now create
filled-context replay measurements without huge prompt files.

The first bounded len-256 result also tempers the Sprint 183 online-attention
signal. Online-single improved the short-prompt 16-slot serving benchmark, but
it was slower on the direct len-256 synthetic continuation check. That means
online attention is still a useful experimental lever, but not a default.

The next practical serving sprint should use this synthetic prompt mode to
measure actual longer contexts before promoting any attention/KV kernel change.
The obvious next measurement tiers are `1024`, `4096`, and then a timed decision
about whether a full `256K` prefill is operationally worth running.
