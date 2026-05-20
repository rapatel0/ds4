# tc-grid V100 Kernel Source Copy

This directory contains the tc-grid V100 kernel files copied from the local
DeepSeek/llama.cpp working tree:

```text
/Users/ravi/repos/deepseek/tools/tc-grid
```

Copied during Sprint 080 from source repo commit
`5903432d826b7b10cdc6d02d8d5da1bbe65371b8`.

The first `ds4` proof uses:

- `kernels/v13_kernels.cuh`
- `kernels/mma_sm70.cuh`
- `include/dispatch.h`
- `include/tc_grid.h`

`v12_kernels.cuh` is copied alongside the v13 path because the tc-grid
dispatcher documents it as the small-M fallback. The current `ds4` smoke only
launches `v13_rf_v6` directly.

This is copied source, not a build-time dependency on `~/repos/deepseek`.
Runtime and test targets in this repository must include and compile these
files from this directory.
