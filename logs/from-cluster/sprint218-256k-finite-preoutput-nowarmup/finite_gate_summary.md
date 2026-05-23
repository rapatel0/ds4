# DS4 V100 256K Finite Gate

Decision: `hc_nonfinite_localized`

| Ctx | Slots | Requests | Status 200 | Status other | Max GPU util | Max memory MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 262144 | 18 | 18 | 0 | 18 | 67.000% | 24076.0 |

First HC non-finite:

```text
ds4-v100-scheduler: HC non-finite: phase=pre-output-head stage=7 gpu=7 layer=-1 slot=0 token=4294967295 position=4294967295 index=0 value=nan
```

First non-200 body:

```json
{"error":"HC non-finite: phase=pre-output-head stage=7 gpu=7 layer=-1 slot=0 token=4294967295 position=4294967295 index=0 value=nan"}

```
