# MTP Block-2 Commit Summary

- cases: 5
- ok_cases: 4
- failed_cases: 1
- average_block2_tps_ok_cases: 3.663043
- average_baseline_tps_ok_cases: 2.032918
- ok_case_speedup_ratio: 1.801865

| Case | Status | Match | Blocks | Full | Partial | Reject | Drafts Accepted | Target Forwards | Spec Saves | Block2 tok/s | Baseline tok/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| long_code_audit | ok | True | 3 | 3 | 0 | 0 | 6 | 7 | 1 | 0.033014 | 0.032855 |
| long_memory_archive | fail |  |  |  |  |  |  |  |  |  |  |
| short_code_completion | ok | True | 4 | 3 | 0 | 1 | 6 | 7 | 1 | 4.105815 | 2.430585 |
| short_italian_fact | ok | True | 3 | 3 | 0 | 0 | 6 | 7 | 1 | 4.969561 | 2.624934 |
| short_reasoning_plain | ok | True | 4 | 0 | 2 | 2 | 2 | 7 | 1 | 5.543783 | 3.043298 |

Failure note: long_memory_archive failed token parity at token 1 (`baseline=16`, `got=8773`). A follow-up target-block smoke on the same prompt also failed reset determinism (`prompt[0] token mismatch got=32085 want=10220`), so this is not isolated to the block-2 commit path.
