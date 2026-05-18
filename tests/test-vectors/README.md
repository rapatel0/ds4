# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

The source-layout oracle runner verifies the guarded CPU oracle against the
native DS4-Flash source model without manual JSON inspection:

```sh
make tools/ds4-source-oracle-vector
tools/ds4-source-oracle-vector \
  --model /models/DSv4-Flash-256e-fixed.gguf \
  --only short_reasoning_plain
```

Use `--dry-parse` for a model-less fixture sanity check. Use `--guards-only`
on the cluster to prove normal source-layout serving, non-CPU oracle use, MTP
sidecars, and sessions without the diagnostic unlock still fail closed.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```
