# Sprint 299 - TP/EP Token Timeline Smoke

Date: 2026-05-23

## Goal

Move the TP/EP diagnostic completion endpoint one step closer to real serving
semantics by accepting tokenized prompts and tracking per-session generated
token timelines.

This follows the shape used by `ds4.c` sessions and llama.cpp slots: a request
has prompt tokens, the slot owns resident KV/HC state, each sampled token is
recorded, and the next decode step advances from the slot cursor instead of
discarding the conversation.

## Implementation

- `POST /v1/completions` and `POST /v100/diagnostic-completions` now accept a
  numeric `prompt_tokens` array.
- A numeric `prompt` array is also accepted for compatibility with clients that
  submit token IDs through the OpenAI-style prompt field.
- Prompt fingerprints use the token-ID sequence when token IDs are supplied.
- Resident session slots now record:
  - prompt token ID count
  - generated token ID count
  - last selected token
  - cache hit/miss state
  - slot cursor position
- `/v100/slots`, `/v100/status`, and response metadata expose the tokenized
  prompt/session counters.

The CUDA path is still diagnostic: selected-token output is recorded, but the
generated token is not yet fed back as the next real model token through a full
tokenizer/prefill path.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/completions`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off
- Request sequence:
  - `session_id=tokalpha`, `prompt_tokens=[1,2,3,4]`, `max_tokens=1`
  - repeat the same request
  - query `/v100/slots`
  - query `/v100/status`

Result:

```text
first request cache_hit                  0
second request cache_hit                 1
request prompt token IDs                 4
slot prompt token IDs                    4
slot generated token IDs after request 1 1
slot generated token IDs after request 2 2
slot cursor                              100000 -> 100001 -> 100002
status cache_hits/cache_misses           1 / 1
status total_prompt_tokens               8 request-observed tokens
slot prompt_tokens                       4 resident-committed tokens
status total_generated_tokens            2
```

The first request committed the prompt to the slot. The second request reused
the same resident prompt/KV/HC state and only extended the generated-token
timeline.

## Evidence

```text
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/summary.json
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/resp1.json
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/resp2.json
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/slots.json
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/status.json
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/server.log
logs/from-cluster/sprint299-tp-ep-token-timeline/cluster/server.err
```

## Remaining Gap

The next completion step is to wire real tokenizer/prompt prefill and selected
token feedback into the TP/EP endpoint. At that point the server can stop being
a diagnostic token harness and start behaving like a minimal DeepSeek text
serving appliance.
