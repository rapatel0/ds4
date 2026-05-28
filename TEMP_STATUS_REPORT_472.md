# TEMP Status Report 472: Final-HC Is the Unsafe Graph Suffix

## Topline

The persistent graph bisection reached a concrete answer:

- `routed_ffn` suffix replay matches eager.
- `dense` suffix replay matches eager.
- `compose` suffix replay matches eager.
- `final_hc` suffix replay does **not** match eager.

So the current graph correctness blocker is final-HC carry/expand under replay,
not routed FFN, dense matmuls, or EP compose.

## Results

Layer 0, `8` slots, `256K` context, `3` decode steps:

| Stage | Control checksum | Persistent checksum | Verdict |
|---|---:|---:|---|
| routed_ffn | `1510241683` | `1510241683` | match |
| dense | `5035503764` | `5035503764` | match |
| compose | `5035503764` | `5035503764` | match |
| final_hc | `5306391750` | `2880063635` | mismatch |

Speed signals for matched slices:

| Stage | Control ms/step | Persistent ms/step | Approx speedup |
|---|---:|---:|---:|
| routed_ffn | `35.897593` | `25.696161` | `1.40x` |
| dense | `43.008403` | `30.500961` | `1.41x` |
| compose | `35.169200` | `27.413582` | `1.28x` |

## Interpretation

Graph replay is viable through compose at this layer-0 diagnostic shape. The
unsafe operation is the HC state update after compose, probably the final-HC
carry/expand path or one of its stream/device ordering assumptions.

## Next Action

Do not run another broad HTTP graph A/B yet.

Implement a compose-suffix persistent mode where:

1. dynamic prefix remains eager;
2. graph replay runs through compose;
3. final-HC carry/expand runs eagerly after replay;
4. direct layer-0 checksum must match eager;
5. only then run small HTTP response parity.
