# DS4 V100 MTP Speculative Gate

Decision: `fail_serial_target_replay`

| Mode | Generated tok/s | Continuation tok/s | Match | Target forwards | Effective output tokens | Spec saves |
|---|---:|---:|---:|---:|---:|---:|
| baseline off | 4.954613 | 4.644949 | 1/1 | n/a | n/a | n/a |
| mtp commit | 4.561292 | 4.276211 | 1/1 | 16 | 16 | 0 |

Current commit mode is a real speedup candidate only when `speculative_saves > 0`; accepted or committed draft counts alone do not qualify.
