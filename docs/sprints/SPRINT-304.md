# Sprint 304 - TP/EP Chat Completions Envelope

Date: 2026-05-23

## Goal

Make the diagnostic TP/EP serving path accept the common
`/v1/chat/completions` route instead of only `/v1/completions`.

## Implementation

- Added `/v1/chat/completions` to the TP/EP HTTP generation route set.
- Added a diagnostic OpenAI-style chat completion response envelope:
  `object=chat.completion`, `choices[0].message.role=assistant`, and
  `choices[0].message.content=""`.
- Mirrored generated token IDs into `choices[0].token_ids` while preserving
  `ds4_v100.generated_token_sequence`.
- Kept text content empty until tokenizer rendering is wired.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/chat/completions`
- Request: `session_id=chatseq`, `prompt_tokens=[41,42,43]`, `max_tokens=3`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
object                         chat.completion
message role                   assistant
choice token_ids               [0, 57085, 104170]
generated token sequence       [0, 57085, 104170]
generated token IDs            3
slot generated token IDs       3
prompt prefill tokens          2
slot cursor / cache_pos_out    100005
wall generated tok/s           210.355981
decode generated tok/s         350.653125
```

## Evidence

```text
logs/from-cluster/sprint304-tp-ep-chat-completions/cluster/summary.json
logs/from-cluster/sprint304-tp-ep-chat-completions/cluster/response.json
logs/from-cluster/sprint304-tp-ep-chat-completions/cluster/response.status
logs/from-cluster/sprint304-tp-ep-chat-completions/cluster/server.out
logs/from-cluster/sprint304-tp-ep-chat-completions/cluster/server.err
```

## Remaining Gap

The chat route is still diagnostic because tokenizer text I/O is not wired.
It is useful now for token-ID clients and for benchmarking the same resident
TP/EP decode path through a route that practical clients expect.
