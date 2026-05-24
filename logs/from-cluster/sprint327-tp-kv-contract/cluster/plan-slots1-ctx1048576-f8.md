DS4 V100 TP/EP planner contract
topology: PP=1(no pipeline) TP=8 EP=8 KV=sharded
configured: slots=1 ctx=1048576 kv=f8_e4m3_b128 reserve=2.00 GiB scratch=1.50 GiB
weights: total 145.42 GiB, per TP rank 18.18 GiB (from --pack-dir)
verdict: fits; per-GPU total 22.56 / 32.00 GiB; headroom after reserve 9.44 GiB

## KV aggregate before TP sharding
| Component | Aggregate | Per GPU |
|---|---:|---:|
| attn_kv | 2.73 GiB | 0.34 GiB |
| indexer_kv | 0.66 GiB | 0.08 GiB |
| comp_state envelope | 1.68 GiB | 0.21 GiB |

## Production compressed-KV contract
| Item | Value |
|---|---:|
| raw SWA rows, all layers | 5504 |
| ratio-4 attention compressed rows | 5505024 |
| ratio-128 attention compressed rows | 163840 |
| ratio-4 indexer compressed rows | 5505024 |
| persistent KV values, all slots | 3609919488 |
| persistent KV bytes, aggregate f8_e4m3_b128 | 3.39 GiB |
| persistent KV bytes, per TP rank | 0.42 GiB |
| if replicated f32 per GPU | 13.45 GiB |
| current bounded diagnostic f32 per GPU | 0.05 GiB |

## Per-layer compressed-KV row contract
| Layer | Ratio | Raw SWA rows | Attn comp rows | Indexer rows | Persistent per GPU | Replicated f32 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 128 | 0 | 0 | 0.008 MiB | 0.250 MiB |
| 1 | 0 | 128 | 0 | 0 | 0.008 MiB | 0.250 MiB |
| 2 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 3 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 4 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 5 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 6 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 7 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 8 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 9 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 10 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 11 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 12 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 13 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 14 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 15 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 16 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 17 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 18 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 19 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 20 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 21 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 22 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 23 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 24 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 25 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 26 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 27 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 28 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 29 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 30 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 31 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 32 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 33 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 34 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 35 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 36 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 37 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 38 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 39 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 40 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |
| 41 | 128 | 128 | 8192 | 0 | 0.512 MiB | 16.250 MiB |
| 42 | 4 | 128 | 262144 | 262144 | 20.164 MiB | 640.250 MiB |

## Per-GPU resident budget
| GPU | Weights | KV | Comp | Scratch | Collectives | Globals | Reserve | Total | Headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpu0 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu1 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu2 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu3 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu4 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu5 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu6 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |
| gpu7 | 18.18 | 0.42 | 0.21 | 1.50 | 0.000 | 0.25 | 2.00 | 22.56 | 9.44 |

## Admission tiers
| Context | Max slots | Per-GPU total at max |
|---:|---:|---:|
| 131072 | 126 | 31.94 GiB |
| 262144 | 63 | 31.92 GiB |
| 524288 | 31 | 31.75 GiB |
| 1048576 | 15 | 31.43 GiB |

## Decode-shape traffic estimates
| Path | Per decode step |
|---|---:|
| hidden payload per rank | 0.008 MiB |
| one ring all-reduce per rank | 0.014 MiB |
| hidden collectives, 2/layer x 43 | 1.176 MiB |
| EP dispatch + return aggregate | 0.094 MiB |

## Expert ownership and density
| Metric | Value |
|---|---:|
| experts per GPU | 32 |
| active routes per decode step | 6 |
| average routes per GPU | 0.75 |
| average routes per expert | 0.023 |

notes:
- Planner intentionally exposes no PP/layer-split topology modes.
- KV is always TP-sharded here; replicated KV is not a production target for 32-slot/256K.
- Weight bytes are a residency estimate until the TP/EP pack contract lands.
- No-reserve per-GPU total is 20.56 GiB.
