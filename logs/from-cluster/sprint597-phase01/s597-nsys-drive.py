#!/usr/bin/env python3
"""Post N concurrent selected-token requests to the resident TP/EP server."""
import json
import sys
import threading
import urllib.request

port, n, tokens = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out_prefix = sys.argv[4] if len(sys.argv) > 4 else None
base = f"http://127.0.0.1:{port}"
results = [None] * n

def post(i):
    req = urllib.request.Request(
        base + "/v100/selected-token",
        data=json.dumps({"max_tokens": tokens, "request_index": i}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        body = r.read()
    results[i] = body
    if out_prefix:
        with open(f"{out_prefix}_{i:03d}.json", "wb") as f:
            f.write(body)

threads = [threading.Thread(target=post, args=(i,)) for i in range(n)]
for t in threads:
    t.start()
for t in threads:
    t.join()
ok = sum(1 for r in results if r)
print(f"drive_done ok={ok}/{n}")
