# Sprint 296 - TP/EP HTTP Session Slots

Date: 2026-05-23

## Goal

Move the TP/EP diagnostic HTTP endpoint from a global request cursor toward
production serving semantics: requests need stable session-to-slot ownership,
visible cache hit/miss state, and explicit resident position accounting before
real tokenizer/prefill work can safely land.

## Reference Review

Reviewed the existing `ds4.c` session model and llama.cpp server slot model
before implementation.

- `ds4_session` treats a session as one mutable inference timeline: token
  checkpoint, live KV/frontier state, logits, invalidation, rewind, and
  prefix-extension versus rebuild decisions.
- llama.cpp server slots keep per-slot prompt tokens, `seq_id`, LRU selection,
  prompt cache save/load, and per-sequence KV removal/shift operations.
- `llama-memory-deepseek4.cpp` tracks per-sequence min/max positions and uses
  a bounded rollback journal for DS4 tail rewinds.

The TP/EP path now copies the useful serving semantics, not those codebases'
runtime implementations.

## Implementation

- Added a TP/EP-only HTTP session table inside
  `tools/ds4-v100-tp-ep-full-layer-smoke.cu`.
- Requests now derive a cache key from `session_id`, `cache_key`,
  `conversation_id`, or a prompt hash fallback.
- Added stable slot assignment with LRU eviction.
- Added position preview before batching, so requests only coalesce when they
  have the same `max_tokens` and same resident cache position.
- Added duplicate-session protection inside a single decode batch.
- Added `/v100/slots`.
- Extended `/v100/status`, `/metrics`, and per-response `ds4_v100` metadata
  with:
  - `cache_slots_total`
  - `cache_slots_used`
  - `cache_hits`
  - `cache_misses`
  - `cache_evictions`
  - `cache_key`
  - `cache_hit`
  - `cache_slot`
  - `cache_pos_in`
  - `cache_pos_out`
  - `cache_evicted`

## Validation

Build on the V100 pod:

```text
cd /workspace/ds4-sprint181
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

HTTP cache smoke, same `session_id=alpha`, with HC-current input, HC final
expand, HC persist, KV all-slots, and diagnostic output head enabled:

```text
resp1.json slot 0 hit 0 pos 100000 -> 100001 token 117160
resp2.json slot 0 hit 1 pos 100001 -> 100002 token 117160
slots.json {'slots_used': 1, 'cache_hits': 1, 'cache_misses': 1, 'cache_evictions': 0}
status.json {'cache_slots_used': 1, 'cache_hits': 1, 'cache_misses': 1, 'cache_evictions': 0, 'next_position': 100002}
```

Evidence:

```text
logs/from-cluster/sprint296-tp-ep-http-session-cache/cluster/
```

## Decision

This is the correct minimal serving primitive for the TP/EP path. It does not
make the endpoint real text serving yet, but it prevents the next prefill work
from being bolted onto a global scratch-style cursor.

## Remaining Gap

- Connect tokenizer/prompt prefill to the assigned slot.
- Replace the diagnostic prompt-token accounting with real token counts.
- Feed selected tokens back into the next decode step.
- Stop touching all 32 slots when only a subset is active.
- Add real prefix/rebuild semantics using the same session table.
- Add MTP only after the TP/EP serving loop is real and measurable.
