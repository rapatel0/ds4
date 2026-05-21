# Sprint 158 - 256K Practical Serving Routed-FFN Executor Boundary

Date: 2026-05-21

## Objective

Create a DS4-specific routed-FFN executor boundary for the practical serving
target, not just the short-context benchmark target.

The primary benchmark target is the current long-context throughput mode:

```text
ctx = 262144
slots = 16
routes = 16 * 6 = 96
```

The secondary long-context sanity target is:

```text
ctx = 1048576
slots = 4
routes = 4 * 6 = 24
```

The 128-slot / 32K / 768-route shape remains useful as a diagnostic stress
shape, but it is not the product goal and must not be the only success gate.

The existing fixed 96-route TurboMind probe is narrower than the product
shape: it requires `total_routes=96` **and** exactly six compacted active
expert groups. The real 16-slot / 256K served case may scatter those 96 routed
rows across more than six experts. Sprint 158 therefore treats fixed96 as a
guarded executor probe and active-expert visibility point, not as proof that
the final fused kernel can be six-group-only.

## Rationale

Sprints 154-157 ruled out several smaller levers:

- down-reduce epilogue alone was flat;
- host stream-per-expert pipelining was not production-safe and safe auto-group
  readback regressed;
- CUDA Graph replay around the current default-stream path did not capture.

The next useful abstraction is:

```text
route build / active-expert compaction
  -> routed-FFN executor family
       gate_up
       activation
       down
       weighted reduce
  -> routed output
```

This executor boundary should support multiple served route shapes. That
matters for future tensor/expert parallel work: the same runtime call can later
select a different compiled kernel family, TP degree, or dense projection
sharding strategy without rewiring the layer scheduler again.

## Corrections From Planning Discussion

- Do not assume 2-way TP is the final or only useful topology. Existing 2-way
  evidence is just the current measured proxy, not a design constraint.
- Do not optimize only for 32K. 32K is a stress benchmark. The appliance goal is
  useful throughput at `>=256K` context.
- If the single-GPU executor boundary does not create a credible path to a
  fused/persistent win at 256K, the next sprint should pivot to topology work
  that considers 2/4/8-way TP or EP/dense sharding under the 256K memory budget.
- The main topology hypothesis to test after this sprint is not that any
  specific TP degree is inherently faster. It is that TP/EP may reshape the
  routed and dense projections into denser fused HMMA-heavy kernels while
  keeping activations fp16 inside the fused executor, with quantize/dequantize
  only at kernel-family boundaries.

## Scope

- Add `DS4_V100_TURBOMIND_ROUTED_EXECUTOR`.
- Support:
  - `off`
  - `auto`
  - `fixed96` for 16-slot / 256K
  - `fixed768` for 128-slot / 32K diagnostic benchmarking
  - `chain96`, `ffn96`, `chain768`, and `ffn768` as aliases
- Keep default behavior unchanged unless explicitly enabled.
- Start with `fixed96` as the primary implementation target.
- Reuse existing TurboMind fixed-shape gate_up support where possible, but only
  when active-expert compaction makes the shape legal.
- Fall back to the current generic path when guards miss.

## Shape Guards

Primary `fixed96` guard:

- `n_tokens = 16`
- `routes_per_token = 6`
- `total_routes = 96`
- compacted active expert groups = 6 for the fixed TurboMind gate_up probe
- `hidden = 4096`
- `mid = 2048`
- fused interleaved gate_up with gated-SiLU
- compact schedule enabled
- graph and group-pipeline diagnostics disabled

Diagnostic `fixed768` guard:

- `n_tokens = 128`
- `routes_per_token = 6`
- `total_routes = 768`
- same `hidden`, `mid`, gate_up, and compact schedule requirements

## Non-Goals

- No broad TP scheduler rewrite in this sprint.
- No assumption that 2-way is the final topology.
- No CUDA Graph replay or explicit-stream rewrite.
- No default promotion without served V100 A/B at `ctx >= 256K`.

## Implementation Plan

1. Add the routed-executor config flag to the launcher and env example.
2. Add a guarded routed-executor dispatch layer inside the TurboMind routed-FFN
   CUDA path after route build and active-expert inspection.
3. Implement `fixed96` first, using the existing fixed 96-route gated-SiLU
   TurboMind gate_up support only when compaction yields six active expert
   groups, and the current down/reduce path as needed.
4. Add `fixed768` only as a diagnostic executor shape if it is low-cost after
   the dispatch boundary exists.
5. Keep the generic path as fallback for all guard misses.
6. Add profiling/logging so a served run proves both the compacted active
   expert count and whether the executor was actually selected.

## Definition Of Done

- Build passes locally.
- V100 build passes for:
  - `ds4_cuda.o`
  - `tools/ds4-v100-replay`
  - `tests/cuda_v100_full_scheduler_smoke`
- Launcher `--check` accepts and prints the routed executor flag.
- Full 43-layer scheduler smoke passes with executor off.
- Full 43-layer scheduler smoke passes with
  `DS4_V100_TURBOMIND_ROUTED_EXECUTOR=fixed96`.
- Served 16-slot/256K A/B is run if smoke passes.
- 4-slot/1M sanity is run if the 256K run passes and time permits.
- 128-slot/32K is optional diagnostic evidence only.
- Results are recorded in:
  - `TEMP_CURRENT_REPORT.md`
  - `TEMP_STATUS_REPORT.md`
  - `docs/sprints/STATUS.md`
  - `docs/sprints/EXPERIMENT-STATUS.md`

## Decision Gate

Do not promote by default unless `ctx=262144`, `slots=16` continuation/decode
throughput improves outside run noise and correctness remains exact for the
selected-token fixture.

If `fixed96` is flat but correct and visibly selected, keep the executor
boundary only if profiling shows it creates a credible next fused/persistent
kernel target.

If `fixed96` does not expose that path, pivot the next development activity to
parallel topology work for `>=256K` context:

- evaluate TP/EP degrees as variables, not as a fixed 2-way assumption;
- include dense-layer TP/sharding probes, not only routed experts;
- require 256K memory admission and served A/B evidence before accepting a
  topology change.

## Results

Build:

```text
make ds4_cuda.o tools/ds4-v100-replay tests/cuda_v100_full_scheduler_smoke CUDA_ARCH=sm_70 -j80
```

passed on the V100 pod.

Full 43-layer scheduler smoke at the product context tier passed with executor
off:

```text
DS4_V100_TURBOMIND_GATED_SILU=1
DS4_V100_TURBOMIND_COMPACT_SCHEDULE=1
DS4_V100_TURBOMIND_ROUTED_EXECUTOR=off
tests/cuda_v100_full_scheduler_smoke --appliance-dir /workspace/ds4-appliance-full-tm-gated-s127 --slots 16 --ctx 262144 --expect-tm-layers 43
```

Result:

```text
layers=43 tm_layers=43 ... ok
```

Full 43-layer scheduler smoke also passed with `fixed96`, and proved the fixed
gate_up kernel is selected when the scheduler presents the 96-route shape:

```text
ds4: TurboMind routed executor fixed96 shape total_routes=96 active_experts=6 max_routes_per_expert=16
ds4: TurboMind routed executor selected fixed gate_up total_routes=96
cuda_v100_full_scheduler_smoke: stages=8 token=16 pos=16 slots=16 layers=43 tm_layers=43 ... ok
```

Served 16-slot/256K A/B:

| Mode | Generated tok/s | Continuation tok/s | Correctness | Notes |
|---|---:|---:|---:|---|
| Control | `46.113721` | `43.231614` | `16/16` | Gated appliance, compact schedule |
| Unguarded fixed96 | `45.010403` | `42.197253` | `16/16` | Regressed from host active-group readback on a six-route served shape |
| Guarded fixed96 | `46.167311` | `43.281854` | `16/16` | Neutral; fixed96 was not selected in HTTP serving |

The key diagnostic is that full-scheduler smoke and HTTP serving expose
different routed-FFN shapes:

```text
full scheduler: total_routes=96 active_experts=6 max_routes_per_expert=16
HTTP served:    total_routes=6  active_experts=6 max_routes_per_expert=1
```

## Decision

Keep `DS4_V100_TURBOMIND_ROUTED_EXECUTOR` default-off and explicit opt-in.

The fixed96 executor is correct and useful as a kernel boundary probe, but it
does not currently move production serving because the served HTTP path is not
coalescing concurrent requests into the 96-route FFN shape before calling the
layer executor. The next material sprint should target served batch formation
for `>=256K` context, or pivot to TP/EP topology work that makes per-layer
execution denser without relying on the current HTTP coalescing behavior.
