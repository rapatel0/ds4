# Sprint 301 - TP/EP Multi-Token Feedback Loop

Date: 2026-05-23

## Goal

Make `max_tokens > 1` in the TP/EP diagnostic HTTP endpoint advance
autoregressively instead of running several decode steps from one static input
seed.

## Implementation

- The HTTP generation path now runs a one-token TP/EP decode loop repeatedly
  for the requested token count.
- After each step, the diagnostic output head selects a token for each active
  slot.
- The selected token is fed back through the resident BF16 token embedding
  seed as the next step's layer-0 HC input.
- Per-request generated token history now records every selected token, not
  only the final selected token.
- Session commit now appends the full generated-token sequence and advances
  the slot cursor by the requested token count.

This is the correctness-oriented loop. It runs the output head once per token
and is not yet the optimized fused serving loop.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/completions`
- Request: `session_id=multi`, `prompt_tokens=[11,12,13]`, `max_tokens=3`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
decode input token          13
generated token IDs         3
slot generated token IDs    3
slot cursor                 100000 -> 100003
response generated_tokens   3
response continuation_tokens 2
wall generated tok/s        153.126777
decode generated tok/s      252.798645
```

The server log shows three one-token target passes and three output-head
passes. The final selected token was `51394`.

## Evidence

```text
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/summary.json
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/resp.json
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/slots.json
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/status.json
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/server.log
logs/from-cluster/sprint301-tp-ep-multitoken-feedback/cluster/server.err
```

## Remaining Gap

The endpoint now has tokenized per-step feedback, but it is still diagnostic.
The next serving gaps are tokenizer text I/O, prompt prefill into resident
TP/EP KV/HC state, active-slot-only decode, and then MTP.
