# TEMP GRAPH PRIOR INSIGHTS

Date: 2026-05-26

Source reviewed: `/Users/ravi/repos/deepseek`, especially:

- `cuda-patches/0003-cuda-graphs.patch`
- `cuda-patches/0004-cuda-graph-skip-reason-counters.patch`
- `cuda-patches/0005-cuda-graph-drift-granularity.patch`
- `cuda-patches/0006-cuda-graph-allow-ptr-drift.patch`
- `cuda-patches/0007-cuda-buffer-async-set-cpy-clear.patch`
- `docs/sprints/SPRINT-025-PATCH.md`
- `docs/sprints/SPRINT-025-PATCH-REPORT.md`

## What The Old Work Teaches Us

### 1. Count graph behavior explicitly

The old fork added counters for:

- graph launches
- eager fallbacks
- captures
- recaptures
- skip reasons
- drift reasons

This was valuable. Without these counters, it is too easy to think graphs are
active while the runtime silently falls back or repeatedly recaptures.

For this DS4 TP appliance, we should keep permanent counters:

- capture attempted/succeeded/failed
- instantiate attempted/succeeded/failed
- replay attempted/succeeded/failed
- eager fallback count
- graph invalidation reason
- replay steps per graph exec

### 2. Recapture is the trap

The old ggml path had to classify graph drift:

- node-count drift
- op/shape/stride drift
- data-pointer-only drift
- other property drift

Pointer-only drift was common enough to justify special handling. The patch tried
to preserve graph usage by treating pointer drift separately from structural
drift, but still needed recapture/update to refresh kernel node params.

For this TP appliance, the better strategy is stricter:

- no pointer drift inside captured graphs
- no topology drift
- no host-side graph property comparison
- no recapture in steady-state decode

The graph should consume stable device pointers. Dynamic inputs should be values
inside persistent device buffers, not changed host pointers.

### 3. Dynamic values must be device-resident metadata

The old path had trouble because changing graph properties made warmup reset or
required recapture/update. In our TP path, the dynamic fields are known:

- decode position
- KV row index
- raw-window row
- compressed row counters
- route counts and offsets
- selected expert ids / route indices
- active slot mask

These should live in fixed device buffers, updated before graph replay by tiny
H2D copies or update kernels. Captured kernels should read metadata from those
buffers. The graph should not see a new pointer or a different launch topology.

### 4. Host sync removal helped but was not enough

The old `0007` patch removed device-direction `cudaStreamSynchronize` calls from
buffer set/copy/clear operations after profiling found around 1k syncs/token and
large CPU wait time. Host-direction reads kept synchronization because sampling
and readback require valid host data.

For this DS4 TP path:

- no host synchronization in decode replay
- no device-to-host route reads in replay
- output-token readback stays outside the replay graph
- route generation must either be captured or run as a separate fixed metadata
  update stage before replay

### 5. CUDA graph capture was not the TurboMind correctness bug

The old Sprint 025 patch report tested `GGML_CUDA_DISABLE_GRAPHS=1`; gibberish
persisted. The actual fixed issue was MXFP4 nibble-lane mapping in the
GGML-to-TurboMind pack path.

For this repo, graph work should not be used to explain correctness drift unless
the non-graph TP path is already parity-clean. Graph replay needs its own parity
gate, but kernel/packing correctness remains separate.

## Impact On Current DS4 TP Work

The current Spike B result proves:

- all 43 layers can capture
- all 43 layers can instantiate
- all 43 layers can graph launch once

But the failed multi-step run proves:

- recapturing each token is not viable
- repeated capture hits stream-capture dependency hazards in compressed KV
- persistent graph execs are required

Therefore the next implementation should be:

1. Capture each layer once.
2. Instantiate and store one `cudaGraphExec_t` per layer.
3. Reuse it for steady-state decode.
4. Update only stable device metadata between launches.
5. Treat any need to recapture as a bug or explicit fallback, not normal control
   flow.

## Concrete Design Constraints For The Next Patch

- Add a `TpGraphLayerExec` structure for each layer.
- Keep root stream, graph, graph exec, node count, capture status, and replay stats.
- Add stable metadata buffers per GPU or per runtime:
  - `d_decode_position`
  - `d_kv_slot`
  - `d_raw_row`
  - route metadata buffers
  - compact KV row/counter metadata
- The captured graph reads metadata from device buffers.
- The host updates metadata outside capture, then launches the graph.
- No `cudaStreamBeginCapture` in the steady-state decode loop.
- No `cudaMemcpyPeerAsync` inside capture; use graph-safe kernels or NCCL paths.
- No legacy/per-thread stream operations inside capture unless explicitly joined
  into the captured stream DAG.
- Metrics must make silent fallback impossible.

## Practical Next Step

Implement persistent graph exec storage for the all-layer TP/EP serving harness,
initially for a fixed-position synthetic replay:

- capture 43 layer graphs once
- instantiate 43 graph execs once
- launch them for N decode steps without recapture
- report setup wall time separately from replay-only continuation time

Then add dynamic metadata updates one by one:

1. position/KV row update
2. compressed KV row counters
3. route metadata update
4. output-head token selection
