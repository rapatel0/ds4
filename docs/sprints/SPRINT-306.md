# Sprint 306 - TP/EP 32-Concurrent Text Chat Benchmark

Date: 2026-05-23

## Goal

Benchmark the tokenizer-enabled TP/EP diagnostic chat path with a full 32-slot
coalesced request batch.

## V100 Run

Configuration:

- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Endpoint: `/v1/chat/completions`
- Concurrent requests: `32`
- Request shape: one text chat message, `max_tokens=8`
- Tokenizer model: `/models/DSv4-Flash-256e-fixed.gguf`
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

Result:

```text
HTTP responses                  32/32
coalesced batches               1
coalesced batch size            32
prompt tokens/request           7
prefill tokens/request          6
generated tokens total          256
client elapsed                  2.326499939 s
client effective tok/s          110.036538
server wall generated tok/s     214.155740
server decode generated tok/s   355.130754
```

The client-side number includes server startup readiness timing and HTTP
request/response overhead. The server-side metric is the current comparable
TP/EP generated-section rate for the resident text chat path.

## Evidence

```text
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/summary.json
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/responses-summary.json
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/response-00.json
...
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/response-31.json
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/server.out
logs/from-cluster/sprint306-tp-ep-text-chat-32concurrent/cluster/server.err
```

## Interpretation

This confirms the text chat API can fill all 32 slots and exercise tokenizer
input, prompt prefill, multi-token decode, detokenized output, and resident
session cursor updates in one batch. Throughput is still diagnostic because the
current path performs correctness-oriented prompt prefill and still runs the
bridge HC sequence rather than final DS4 parity/optimized kernels.
