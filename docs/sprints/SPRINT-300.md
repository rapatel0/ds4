# Sprint 300 - TP/EP Request-Boundary Token Feedback

Date: 2026-05-23

## Goal

Replace the remaining synthetic request-boundary input in the TP/EP diagnostic
HTTP path with a real token-embedding seed. This is the first serving step that
feeds a generated token back into the next decode request, matching the
`ds4.c` / llama.cpp pattern where the sampled token becomes the next model
input.

## Implementation

- Added a resident `token_embd.weight` loader for the TP/EP HTTP path.
- The loader keeps the BF16 token embedding table on GPU0:
  - shape `[4096 x 129280]`
  - source dtype `bf16`
  - loaded bytes `1059061760`
- Added a GPU seed kernel that expands a token embedding row into each rank's
  local HC shard `[slot][4][512]`.
- For a cache miss, the request uses the last supplied prompt token as the
  first decode input.
- For a cache hit, the request uses the slot's previous selected token as the
  next decode input.
- Response metadata now includes:
  - `token_input_seed`
  - `decode_input_token`

This is deliberately request-boundary feedback. Multi-token `max_tokens > 1`
still needs a per-step output-head/sample/feed loop before it is real text
generation inside one HTTP request.

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
token embedding load        PASS, 1059061760 bytes on GPU0
first request cache_hit     0
first request input token   4
first selected token        77960
second request cache_hit    1
second request input token  77960
second selected token       77960
slot generated token IDs    2
slot cursor                 100000 -> 100001 -> 100002
status cache_hits/misses    1 / 1
```

The first request seeded layer 0 from the prompt tail token. The second request
seeded layer 0 from the previous selected token, proving the basic selected
token feedback path across HTTP requests.

## Evidence

```text
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/summary.json
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/resp1.json
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/resp2.json
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/slots.json
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/status.json
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/server.log
logs/from-cluster/sprint300-tp-ep-token-feedback/cluster/server.err
```

## Remaining Gap

The endpoint is still diagnostic until these are implemented:

- tokenizer text input/output
- prompt prefill into resident TP/EP KV/HC state
- per-step output-head/sample/feed loop for `max_tokens > 1`
- active-slot-only decode instead of always computing all configured slots
- MTP after the target stream is correct
