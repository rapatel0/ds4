# MTP Acceptance Matrix Summary

- cases: 15
- ok_cases: 15
- failed_cases: 0
- average_accepted_prefix: 1.533
- max_accepted_prefix: 2
- cases_with_accepted_prefix_ge_2: 10
- total_speculative_saves: 4
- decision: continue-mtp-evaluation

| Block | Cases | Accepted Prefix >= 2 |
|---:|---:|---:|
| 2 | 5 | 4 |
| 4 | 5 | 3 |
| 8 | 5 | 3 |

## Cases

| Case | Prompt | Block | Accepted | Effective | Target Forwards | Spec Saves |
|---|---|---:|---:|---:|---:|---:|
| short_reasoning_plain_block2 | tests/test-vectors/prompts/short_reasoning_plain.txt | 2 | 1 | 2 | 2 | 0 |
| short_reasoning_plain_block4 | tests/test-vectors/prompts/short_reasoning_plain.txt | 4 | 1 | 2 | 4 | 0 |
| short_reasoning_plain_block8 | tests/test-vectors/prompts/short_reasoning_plain.txt | 8 | 1 | 2 | 8 | 0 |
| short_code_completion_block2 | tests/test-vectors/prompts/short_code_completion.txt | 2 | 2 | 3 | 2 | 1 |
| short_code_completion_block4 | tests/test-vectors/prompts/short_code_completion.txt | 4 | 2 | 3 | 4 | 0 |
| short_code_completion_block8 | tests/test-vectors/prompts/short_code_completion.txt | 8 | 2 | 3 | 8 | 0 |
| short_italian_fact_block2 | tests/test-vectors/prompts/short_italian_fact.txt | 2 | 2 | 3 | 2 | 1 |
| short_italian_fact_block4 | tests/test-vectors/prompts/short_italian_fact.txt | 4 | 2 | 3 | 4 | 0 |
| short_italian_fact_block8 | tests/test-vectors/prompts/short_italian_fact.txt | 8 | 2 | 3 | 8 | 0 |
| long_code_audit_block2 | tests/test-vectors/prompts/long_code_audit.txt | 2 | 2 | 3 | 2 | 1 |
| long_code_audit_block4 | tests/test-vectors/prompts/long_code_audit.txt | 4 | 2 | 3 | 4 | 0 |
| long_code_audit_block8 | tests/test-vectors/prompts/long_code_audit.txt | 8 | 2 | 3 | 8 | 0 |
| long_memory_archive_block2 | tests/test-vectors/prompts/long_memory_archive.txt | 2 | 2 | 3 | 2 | 1 |
| long_memory_archive_block4 | tests/test-vectors/prompts/long_memory_archive.txt | 4 | 0 | 1 | 4 | 0 |
| long_memory_archive_block8 | tests/test-vectors/prompts/long_memory_archive.txt | 8 | 0 | 1 | 8 | 0 |
