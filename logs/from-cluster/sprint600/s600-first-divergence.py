#!/usr/bin/env python3
"""S600: slot-indexed tolerance + first-divergence localization.
Pairs candidate responses with control by (coalesced_batch_id, coalesced_slot_index);
reports selected-token/sequence agreement and first differing checksum/token step."""
import json, glob, sys, pathlib, collections

def load_dir(d):
    out = {}
    for p in glob.glob(d + "/response*"):
        text = pathlib.Path(p).read_text(encoding="utf-8", errors="replace")
        if "\nHTTP_STATUS:" in text:
            text = text.rsplit("\nHTTP_STATUS:", 1)[0]
        try:
            b = json.loads(text)
        except Exception as e:
            print("parse fail", p, e); continue
        if not isinstance(b, dict): continue
        key = (b.get("coalesced_batch_id"), b.get("coalesced_slot_index"))
        out[key] = (p, b)
    return out

ctl = load_dir(sys.argv[1])
cand = load_dir(sys.argv[2])
common = sorted(set(ctl) & set(cand))
print(f"pairs={len(common)} ctl_only={len(set(ctl)-set(cand))} cand_only={len(set(cand)-set(ctl))}")
ck_hist = collections.Counter()
tok_hist = collections.Counter()
sel_match = sel_tot = seq_match = seq_tot = 0
diverged = 0
for key in common:
    _, cb = ctl[key]
    _, xb = cand[key]
    if cb.get("selected_token") is not None and xb.get("selected_token") is not None:
        sel_tot += 1
        sel_match += int(cb["selected_token"] == xb["selected_token"])
    cseq = cb.get("generated_token_sequence") or []
    xseq = xb.get("generated_token_sequence") or []
    for a, b in zip(cseq, xseq):
        seq_tot += 1
        seq_match += int(a == b)
    cks_c = cb.get("decode_step_checksums") or []
    cks_x = xb.get("decode_step_checksums") or []
    first_ck = next((i for i, (a, b) in enumerate(zip(cks_c, cks_x)) if a != b), None)
    first_tok = next((i for i, (a, b) in enumerate(zip(cseq, xseq)) if a != b), None)
    ck_hist[first_ck] += 1
    tok_hist[first_tok] += 1
    if first_ck is not None or first_tok is not None:
        diverged += 1
        if diverged <= 8:
            print(f"batch={key[0]} slot={key[1]} first_ck_step={first_ck} first_tok_step={first_tok}")
print("selected_token_agreement:", round(sel_match / sel_tot, 6) if sel_tot else None, f"({sel_match}/{sel_tot})")
print("sequence_agreement:", round(seq_match / seq_tot, 6) if seq_tot else None, f"({seq_match}/{seq_tot})")
print("first_ck_step histogram:", dict(sorted(ck_hist.items(), key=lambda kv: (kv[0] is None, kv[0]))))
print("first_tok_step histogram:", dict(sorted(tok_hist.items(), key=lambda kv: (kv[0] is None, kv[0]))))
