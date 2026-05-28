# TEMP Status Report 446

## Current Focus

TP/EP serving only. Sprint 446 isolated the rank-major correctness failure from
Sprint 445.

## Harness Fixes

- `tools/ds4-v100-tp-ep-profile.py` now sets `DS4_LOCK_FILE` to the per-case
  artifact directory. This avoids failures from stale root-owned `/tmp/ds4.lock`
  files left by prior sudo/root runs.
- `tools/ds4-v100-tp-ep-nccl-http-ab.py` now aborts after a nonzero control or
  candidate profile return code, instead of launching the next leg and leaving
  extra GPU work running.

## Sprint 445 Topline

Combined rank-major candidate at `8` requests / `8` slots / `256K` / `2`
tokens:

| Leg | HTTP 200 | First token | Server decode tok/s | Avg GPU util | Min free VRAM |
|---|---:|---:|---:|---:|---:|
| Control | 8/8 | 72960 | 19.279431 | 9.840278% | 4674 MiB |
| Candidate | 8/8 | 81401 | 20.362245 | 10.173611% | 4836 MiB |

Decision: do not promote. Response parity failed `0/8` despite a `1.056x`
server decode speed signal.

## Sprint 446 Isolation Results

Reduced isolation shape: `8` slots, `4` requests, `256K`, `2` tokens,
`512 MiB` scratch, deferred NCCL.

| Candidate gate | Parity | First-token result | Server decode speedup |
|---|---:|---|---:|
| Attention input only | 0/4 | `72960 -> 81401` | 1.018x |
| FFN input only | 4/4 | `72960 -> 72960` | 1.011x |
| Router logits only | 4/4 | `72960 -> 72960` | 1.016x |

Artifacts:

- `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-attn-s512`
- `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-ffn-s512`
- `/localpool/ds4/workspace/logs/s446-rankmajor-isolate-router-s512`

## Assessment

The attention projection rank-local/rank-major input path is the current
correctness blocker. It changes tokens by itself. FFN rank-major input and
rank-major router logits are parity-clean in isolation, but neither is a
standalone promotion at this reduced shape.

Next work should compare the attention projection input buffers directly:
legacy slot-major projection input versus rank-local/rank-major projection
input, per layer/rank/slot before the Q/KV projection consumers. Avoid broad
combined A/Bs until this path is parity-clean.

## Cluster State

After the Sprint 446 runs, the V100 node reported no active DS4 GPU jobs.
