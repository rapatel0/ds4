# Sprint 298 - Longer TP/EP HTTP Completion Benchmark

Date: 2026-05-23

## Goal

Run a longer serving-shaped benchmark against the current TP/EP diagnostic
completion API after the session-slot and prompt-fingerprint guardrails landed.

## Configuration

- Endpoint: `/v1/completions`
- Topology: TP8 / EP8 / PP1
- Context: `256K`
- Slots: `32`
- Concurrent requests: `32`
- Token cases: `16`, `32`, `64`
- MTP: off
- Diagnostic output head: on
- HC current input: on
- HC final expand: on
- HC persistent state: on
- KV all-slot readback verifier: off

The KV all-slot verifier was intentionally disabled for this throughput run
because it is a correctness/readback guardrail, not the intended serving mode.

## Result

```text
tokens  wall generated tok/s  wall continuation tok/s  decode generated tok/s  decode continuation tok/s  gpu util avg/max
16      194.530928            199.977851               329.048680             339.279225                7.40% / 36%
32      199.286944            203.199004               340.196025             346.097972                8.02% / 37%
64      200.272837            203.295374               338.142261             343.440198                8.39% / 37%
```

Each case formed one coalesced `32` request batch, returned `32/32` HTTP 200
responses, and had zero token mismatches in the diagnostic output-head path.

## Interpretation

The API path is operational for diagnostic sustained HTTP benchmarking, but it
is not yet real text serving. The throughput plateau near `200` wall tok/s and
`340` decode tok/s is with the current HC bridge path, not the older synthetic
selected-token benchmark. GPU utilization remains low, so the next performance
work should still focus on making the serving loop real first, then reducing
HC bridge / compose movement and enabling active-slot-only decode.

## Evidence

```text
logs/from-cluster/sprint298-tp-ep-long-http-completions/cluster/sustained_http.tsv
logs/from-cluster/sprint298-tp-ep-long-http-completions/cluster/sustained_http.json
```
