# Sprint 392: Multi-Prompt HTTP Soak Harness

## Overview

Add prompt-set support to the TP/EP HTTP profile harness and use it to rerun
E5M2 KV over varied prompts.

Sprint 391 kept E5M2 KV default-off because one repeated chat prompt template
is not enough parity evidence for a lower-mantissa KV format. This sprint
turns the requested broader multi-prompt parity/soak into a permanent harness
feature.

## Scope

- Add `--prompt-file` to `tools/ds4-v100-tp-ep-profile.py`.
- Support JSONL prompt records for `/v1/chat/completions`.
- Cycle prompts across requests while keeping deterministic session IDs.
- Add a small DS4 V100 soak prompt set under `tests/`.
- Run HTTP control/candidate A/B with:
  `32` slots, `32` requests, `256K` context, `position=262080`,
  model-router routes, compact MoE, VRAM admission, and E5M2 as the only
  candidate gate.
- Compare responses with `tools/ds4-v100-http-response-parity.py`.

## Out Of Scope

- No PP/layer-split work.
- No MTP work.
- No new KV dtype implementation.
- No default promotion unless the multi-prompt evidence is strong and
  unambiguous.

## Definition Of Done

- Prompt-file support is implemented and syntax checked.
- The prompt file is committed.
- V100 HTTP control and E5M2 candidate complete using the prompt file.
- Permanent parity comparator summary is produced.
- Decision is documented and committed.

## Risks

- Different prompt token lengths can shift cache positions and expose context
  admission bugs.
- Varied prompts may make generated token sequences harder to compare if the
  control and candidate diverge after several steps. That is exactly what this
  sprint is meant to catch.

## Execution Plan

1. Implement JSONL prompt-file parsing in the profile harness.
2. Add a compact prompt set covering chat, code, JSON, long-context reference,
   arithmetic, instruction-following, and multilingual text.
3. Validate local syntax and prompt parsing.
4. Sync harness changes to gpu-01.
5. Run control and E5M2 HTTP A/B.
6. Run the permanent parity comparator.
7. Document promote/reject/defer decision.

## Outcome

Complete. The multi-prompt soak harness is implemented and E5M2 remains
default-off.

Changes:

- Added `--prompt-file` to `tools/ds4-v100-tp-ep-profile.py`.
- Added JSONL prompt parsing for chat/completions runs.
- Added prompt metadata to profile summaries:
  `prompt_file`, `prompt_count`, and `prompt_digest`.
- Added `tests/v100_tp_ep_soak_prompts.jsonl` with `16` prompt records.

## Validation

Local:

```text
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
prompt parser loaded 16 records
prompt digest 03f814a38f6f5f89...
```

V100 multi-prompt HTTP A/B:

| Metric | Control | E5M2 KV |
|---|---:|---:|
| HTTP 200 | `32/32` | `32/32` |
| Prompt count | `16` | `16` |
| Prompt digest | `03f814a38f6f5f89...` | `03f814a38f6f5f89...` |
| First token | `83484` | `83484` |
| Parity matched pairs | `32/32` | `32/32` |
| Client generated tok/s | `38.912861` | `39.774181` |
| Server generated tok/s | `88.358577` | `88.351220` |
| Server decode tok/s | `106.390802` | `106.483285` |
| Compressed-KV sum | `3343.550356 ms` | `3301.691102 ms` |
| Avg GPU util | `9.772727%` | `9.691860%` |
| Max GPU util | `52%` | `50%` |
| VRAM failures | `0` | `0` |
| Min free VRAM | `1746 MiB` | `1746 MiB` |

Permanent parity comparator:

```text
match=true
paired_count=32
matched_pairs=32
failed_pairs=0
```

## Decision

Keep E5M2 KV default-off. It is parity-clean on this broader prompt soak, but
the performance result is effectively flat and the current block-128 E5M2
layout is not a VRAM-capacity win over E4M3. The precision tradeoff is not
worth making the production default without a larger quality/long-context
reason.

## Artifacts

- Cluster:
  - `/workspace/logs/sprint392-multiprompt-e5m2/http-control`
  - `/workspace/logs/sprint392-multiprompt-e5m2/http-candidate-e5m2`
- Local:
  - `logs/from-cluster/sprint392-multiprompt-e5m2`
