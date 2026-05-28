# Sprint 472: Suffix Replay Boundary Bisection

## Objective

Finish the layer-0 suffix bisection after Sprint 470 by testing `dense`,
`compose`, and full `final_hc` suffix boundaries.

## Scope

- TP/EP path only.
- No PP/layer-split work.
- Default-off diagnostic gates only.
- Extend `--decode-cudagraph-suffix-stage-gate` to:
  - `dense`
  - `compose`
  - `final_hc`
- Use layer-0 direct diagnostics before any HTTP graph A/B.

## Implementation

- `dense` suffix stage now returns after routed FFN plus dense/shared work and
  hashes dense outputs.
- `compose` suffix stage now returns after EP pack/copy/compose and hashes
  `next_hidden`.
- `final_hc` runs the full suffix and hashes the normal next-hidden plus
  final-HC carry state.
- The profile wrapper accepts the suffix stage value and uses stable hash
  suffix shortening when artifact names are too long.

## Validation

V100 build:

```text
make -B -j80 tools/ds4-v100-tp-ep-full-layer-smoke
PASS
```

Layer-0 direct diagnostic at `8` slots / `256K` / `3` decode steps:

| Stage | Mode | rc | Checksum | Capture | Replay | Nodes | Decode ms/step | Slot-step tok/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dense | eager/control | 0 | `5035503764` | 0 | 0 | 0 | `43.008403` | `186.010162` |
| dense | persistent | 0 | `5035503764` | 1 | 1 | 569 | `30.500961` | `262.286820` |
| compose | eager/control | 0 | `5035503764` | 0 | 0 | 0 | `35.169200` | `227.471767` |
| compose | persistent | 0 | `5035503764` | 1 | 1 | 601 | `27.413582` | `291.826150` |
| final_hc | eager/control | 0 | `5306391750` | 0 | 0 | 0 | `45.910823` | `174.250851` |
| final_hc | persistent | 0 | `2880063635` | 1 | 1 | 628 | `39.367301` | `203.214336` |

Artifacts:

```text
/localpool/ds4/workspace/logs/s472-dense-control3
/localpool/ds4/workspace/logs/s472-dense-persistent3
/localpool/ds4/workspace/logs/s472-compose-control3
/localpool/ds4/workspace/logs/s472-compose-persistent3
/localpool/ds4/workspace/logs/s472-final_hc-control3
/localpool/ds4/workspace/logs/s472-final_hc-persistent3
```

## Outcome

The first unsafe suffix stage is `final_hc`.

`routed_ffn`, `dense`, and `compose` all match eager checksums under persistent
replay. Full suffix replay changes the final-HC checksum, so persistent graph
serving should not capture final-HC carry/expand yet.

## Next

Move final-HC carry/expand out of the captured suffix and run it eagerly after
compose replay. Then rerun the same layer-0 checksum pair, followed by the
small HTTP parity A/B only if the direct checksum matches.
