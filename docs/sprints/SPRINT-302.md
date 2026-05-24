# Sprint 302 - TP/EP Prompt Prefill Bridge

Date: 2026-05-23

## Goal

Add a diagnostic prompt-prefill path for TP/EP HTTP cache misses so a new
session no longer starts generation from only the final prompt token.

## Implementation

- On a cache miss, prompt tokens before the tail are evaluated as prefill
  tokens through the TP/EP one-token loop.
- Those prefill tokens update resident KV/HC state but are not returned as
  generated tokens.
- Generation then starts from the final prompt token and uses the Sprint 301
  per-step output-head/sample/feed loop.
- Session commit advances the resident cursor by
  `prompt_prefill_tokens + generated_tokens`.
- Response metadata now reports `prompt_prefill_tokens`.

This is a correctness bridge, not a fast batched prefill implementation.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/completions`
- Request: `session_id=prefill`, `prompt_tokens=[21,22,23]`, `max_tokens=2`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
decode input token       23
prompt prefill tokens    2
generated token IDs      2
slot generated tokens    2
slot cursor              100000 -> 100004
status next_position     100004
wall generated tok/s     212.692685
decode generated tok/s   351.116767
```

The server log shows two prefill passes without output-head selection,
followed by two generation passes with output-head selection.

## Evidence

```text
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/summary.json
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/resp.json
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/slots.json
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/status.json
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/server.log
logs/from-cluster/sprint302-tp-ep-prompt-prefill/cluster/server.err
```

## Remaining Gap

The endpoint now has tokenized prompt prefill and per-step generated-token
feedback. The remaining serving gaps are tokenizer text I/O, active-slot-only
decode, optimized/batched prefill, and MTP.
