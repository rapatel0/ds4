# Sprint 546 - C1 Device Decode Position Stage 1

Date: 2026-05-29

## Goal

Start the staged full-capture position work from Sprint 545 by moving pure
kernel position consumers from host scalar launch arguments to replay-updated
device memory, without changing full-capture cache-key safety.

## Changes

Added one per-rank device scalar:

- `RankState::d_decode_position`

The scalar is allocated with the rank buffers, initialized to zero, and updated
from `opt.position` before decode work is enqueued.

Converted pure kernel consumers to read the device scalar:

- RoPE row kernels
- compressed-row RoPE emit kernels
- compressed-state store kernels
- raw SWA store/read/window kernels

The remaining host-side position consumers are intentionally unchanged:

- compressed-KV emitted-row branch
- typed-KV runtime row selection
- compressed-row position bookkeeping
- full-capture persistent cache key

## Validation

Remote workspace:

- `/workspace/s546-device-position`

Build command:

- `CUDA_ARCH=sm_70 make -B -j80 appliance/ds4-v100-tp-ep-appliance`

Result:

- PASS

Local checks:

- `git diff --check` PASS

No serving or performance gate was run in this sprint. This is a prerequisite
mechanical change, and full-capture persistent reuse remains disabled by the
position cache key.

## Decision

Keep the change. It removes one class of position-baked graph launch arguments
while preserving current correctness safety. Do not remove position from the
full-capture cache key yet.

## Follow-Up

The next full-capture position stage is compressed-KV topology: the emitted-row
decision still comes from a host branch over `opt.position`, so capture topology
can still differ by decode position.
