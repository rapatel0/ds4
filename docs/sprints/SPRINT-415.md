# Sprint 415: Operationalize Persistent Graph + Deferred NCCL

Date: 2026-05-26

## Objective

Promote the latest TP/EP direct-decode controls from one-off benchmark flags
into the permanent appliance launcher and profiling harness:

- persistent per-layer CUDA graph replay
- configurable TP runtime scratch size
- deferred NCCL initialization after model residency allocation

This sprint remains TP/EP-only. No PP/layer-split variants.

## Evidence Baseline

Source: `TEMP_STATUS_REPORT_417.md`.

At `8` slots / `256K` / `8` decode steps:

```text
eager generated decode tok/s:      37.617796
persistent graph generated tok/s:  85.272661
replay_succeeded:                  344/344
```

With current-HC NCCL, deferred NCCL init, and scratch512:

```text
16-slot generated decode tok/s:      116.852459
16-slot continuation decode tok/s:   121.222428
```

The direct `32` slot / `256K` case still OOMs during expert allocation, so the
next performance target is memory residency/layout before assuming slot32 is
available in this all-resident direct path.

## Implementation

- `tools/ds4-v100-run-appliance.sh`
  - Added `DS4_V100_TP_EP_DECODE_CUDAGRAPH_PERSISTENT`.
  - Added `DS4_V100_TP_EP_TP_RUNTIME_SCRATCH_MIB`.
  - Added `DS4_V100_TP_EP_DEFER_NCCL_INIT`.
  - Renders `--decode-cudagraph-persistent-replay-gate`,
    `--tp-runtime-scratch-mib`, and `--defer-nccl-init-gate` into the TP/EP
    command.

- `tools/ds4-v100-tp-ep-profile.py`
  - Added matching HTTP/direct harness flags and environment propagation.
  - Direct token-major profile commands now use the same scratch/deferred-NCCL
    knobs as the validated direct benchmarks.

- `deploy/v100/ds4-v100-appliance.env.example`
  - Documented the new controls and the Sprint 417 results.

- `docs/sprints/VISION.md`
  - Updated the graph/NCCL direction: broad graph capture remains rejected, but
    persistent per-layer TP/EP graph replay is now an active measured path.

## Validation

Local validation:

```text
bash -n tools/ds4-v100-run-appliance.sh
python3 -m py_compile tools/ds4-v100-tp-ep-profile.py
```

Launcher command rendering:

```text
--tp-runtime-scratch-mib 512
--defer-nccl-init-gate
--decode-cudagraph-persistent-replay-gate
```

Cluster validation:

```text
pod: llm/ds4-tp-bench
workspace: /workspace -> /localpool/ds4/workspace
models: /models -> /srv/models
build: PASS
```

The first HTTP promotion attempt did not reach request execution. It failed
during all-resident expert allocation. That exposed a real memory-layout issue:
the loader was doing thousands of per-expert `cudaMalloc`s. This sprint changed
the loader to allocate one contiguous weight buffer and one contiguous scale
buffer per descriptor, then point the existing expert table ABI into those
buffers.

After the contiguous allocation change, the build still passed, but full
all-layer expert residency remains too tight in the CUDA pod. The current
process reports several GiB used at the first CUDA memory checkpoint, then adds
the dense F16 cache, TP runtime/KV/scratch, and roughly `17-18 GiB/GPU` of
expert residency. This is now the next blocker before HTTP graph serving can
be promoted.

## Next

1. Add an explicit expert-residency planner/report before allocation.
2. Reduce duplicate residency, starting with dense F16 cache and expert loading
   strategy, then retry HTTP persistent graph serving.
3. Keep Nsight measurements attached to the serving harness so bottleneck
   movement is visible after graph replay reduces launch overhead.
