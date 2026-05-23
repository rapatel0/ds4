# Sprint 297 - TP/EP Prompt Fingerprint Cache Guard

Date: 2026-05-23

## Goal

Prevent unsafe resident KV/HC reuse when a client reuses a session key with a
different prompt. Sprint 296 added stable session slots, but cache hits were
based only on the cache key. Until real tokenizer prefix matching exists, an
explicit prompt fingerprint is the simplest correctness guardrail.

## Implementation

- Added prompt fingerprint extraction for JSON `prompt` strings.
- Extended TP/EP HTTP session slots with:
  - `prompt_fingerprint_known`
  - `prompt_fingerprint`
- Cache preview now returns a resident position only when the session has
  valid KV/HC and the prompt fingerprint matches.
- Assignment of an existing session now resets resident slot state when the
  prompt fingerprint mismatches.
- Added response metadata:
  - `cache_prompt_match`
  - `cache_prompt_fingerprint`
- Added prompt fingerprint visibility to `/v100/slots`.

This remains a string-level fingerprint, not tokenizer-prefix reuse. It is an
intermediate safety mechanism for the diagnostic endpoint.

## Validation

Build on the V100 pod:

```text
cd /workspace/ds4-sprint181
make -j80 tools/ds4-v100-tp-ep-full-layer-smoke
```

Result: PASS.

HTTP smoke with the same `session_id=alpha`:

```text
resp1.json slot 0 hit 0 prompt_match 1 pos 100000 -> 100001 fp 25347132070217633
resp2.json slot 0 hit 1 prompt_match 1 pos 100001 -> 100002 fp 25347132070217633
resp3_mismatch.json slot 0 hit 0 prompt_match 0 pos 100000 -> 100001 fp 959878203752375754
slots.json {'slots_used': 1, 'cache_hits': 1, 'cache_misses': 2, 'cache_evictions': 0}
status.json {'cache_slots_used': 1, 'cache_hits': 1, 'cache_misses': 2, 'cache_evictions': 0, 'next_position': 100002}
```

Evidence:

```text
logs/from-cluster/sprint297-tp-ep-http-prompt-fingerprint/cluster/
```

## Decision

Keep this guardrail until tokenizer-level prefix matching replaces it. It
protects downstream users from accidentally continuing from stale resident
state when they reuse a session id with a different prompt.

## Remaining Gap

- Tokenize prompts and compare token prefixes, not string hashes.
- Prefill only the changed suffix for matching prefixes.
- Expose explicit reset/rewind operations.
- Feed selected tokens back into slot prompt state.
