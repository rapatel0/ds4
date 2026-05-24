# Sprint 305 - TP/EP Tokenizer Text I/O

Date: 2026-05-23

## Goal

Wire the existing DS4 tokenizer into the TP/EP diagnostic serving binary so
the API can accept text prompts and return decoded text, not only token IDs.

## Implementation

- Linked the TP/EP full-layer binary with the existing `ds4.c` CPU tokenizer
  path in inspect-only mode.
- Added `--tokenizer-model PATH` to the TP/EP binary.
- Added `DS4_V100_TP_EP_TOKENIZER_MODEL`, defaulting to `DS4_V100_MODEL`, to
  the launcher.
- Tokenizes top-level text `prompt` for `/v1/completions`.
- Tokenizes simple chat `messages[].content` via the DS4 chat prompt encoder
  for `/v1/chat/completions`.
- Decodes generated token IDs with `ds4_token_text` and returns the text in
  `choices[0].text`, `choices[0].message.content`, and
  `ds4_v100.generated_text`.
- Fixed the request fingerprint path so tokenizer-derived prompt tokens are
  preserved instead of being cleared by the old text-only fallback.

This is a simple tokenizer bridge. It does not yet implement full multi-message
role-aware chat parsing or streaming UTF-8 boundary buffering.

## V100 Smoke

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/chat/completions`
- Request: `session_id=textchat`, message content `"Hello"`, `max_tokens=2`
- Tokenizer model: `/models/DSv4-Flash-256e-fixed.gguf`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
tokenizer ready                 1
request prompt token IDs        5
prompt prefill tokens           4
generated token sequence        [95933, 89868]
generated text                  ICCungtod
message content                 ICCungtod
generated token IDs             2
slot cursor / cache_pos_out     100006
wall generated tok/s            213.595353
decode generated tok/s          350.755948
```

## Evidence

```text
logs/from-cluster/sprint305-tp-ep-tokenizer-text-io/cluster/summary.json
logs/from-cluster/sprint305-tp-ep-tokenizer-text-io/cluster/response.json
logs/from-cluster/sprint305-tp-ep-tokenizer-text-io/cluster/response.status
logs/from-cluster/sprint305-tp-ep-tokenizer-text-io/cluster/server.out
logs/from-cluster/sprint305-tp-ep-tokenizer-text-io/cluster/server.err
```

## Remaining Gap

The TP/EP API can now accept text and return decoded text, but it is still a
diagnostic model path. Remaining serving gaps are full role-aware chat parsing,
streaming output, active-slot-only decode, optimized/batched prefill, exact DS4
HC sequence parity, and MTP.
