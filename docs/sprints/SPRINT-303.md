# Sprint 303 - TP/EP Token Sequence Response

Date: 2026-05-23

## Goal

Expose the generated token sequence from the diagnostic TP/EP completion
endpoint so downstream clients can consume real token IDs before tokenizer text
rendering is wired.

## Implementation

- Added JSON serialization for `std::vector<uint32_t>` token ID arrays.
- Added `ds4_v100.generated_token_sequence` to `/v1/completions` response
  metadata.
- Added `ds4_v100.slot_position` as an explicit alias for the committed
  resident cache cursor.
- Kept tokenizer text output explicitly diagnostic; the returned completion
  text is still empty until tokenizer rendering is connected.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/completions`
- Request: `session_id=seq`, `prompt_tokens=[31,32,33]`, `max_tokens=3`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
generated token sequence       [127885, 57114, 78026]
generated token IDs            3
slot generated token IDs       3
prompt prefill tokens          2
slot cursor / cache_pos_out    100005
wall generated tok/s           214.100724
decode generated tok/s         353.667490
```

The generated token sequence is now available in the response body, and
`slot_position == cache_pos_out`, which confirms the session cursor advances by
`prompt_prefill_tokens + generated_tokens`.

## Evidence

```text
logs/from-cluster/sprint303-tp-ep-token-sequence-response/cluster/summary.json
logs/from-cluster/sprint303-tp-ep-token-sequence-response/cluster/response.json
logs/from-cluster/sprint303-tp-ep-token-sequence-response/cluster/response.status
logs/from-cluster/sprint303-tp-ep-token-sequence-response/cluster/server.out
logs/from-cluster/sprint303-tp-ep-token-sequence-response/cluster/server.err
```

## Remaining Gap

The endpoint can now accept tokenized prompts, prefill prompt tokens, run
multi-token autoregressive feedback, persist resident slot state, and return
generated token IDs. Remaining serving gaps are tokenizer text I/O,
active-slot-only decode, optimized/batched prefill, production DS4 HC sequence
parity, and MTP.
