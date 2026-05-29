import concurrent.futures, json, os, pathlib, signal, subprocess, time, urllib.request
REPO=pathlib.Path("/workspace/s573-continuation-instrument")
ART=pathlib.Path("/workspace/s573-determinism-artifacts"); ART.mkdir(parents=True,exist_ok=True)
SLOTS=32
BASE_ENV=os.environ.copy()
BASE_ENV.update({
 "DS4_V100_SERVE_MODE":"tp-ep","DS4_V100_CTX":"262144","DS4_V100_SLOTS":"32","DS4_V100_ACTIVE_MICROBATCH":"32",
 "DS4_V100_CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7","CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7",
 "DS4_V100_APPLIANCE_DIR":"/workspace/packs/ds4-appliance-full-tm-gated-s181",
 "DS4_V100_TP_EP_CONTRACT":"/workspace/logs/sprint245-tp-ep-dense-f16-cache-contract/contract/tp-ep-pack-contract.tsv",
 "DS4_V100_TURBOMIND_LIB":"/workspace/ds4-sprint181/build/turbomind-v100/libggml-turbomind.so",
 "DS4_V100_TP_EP_TOKENIZER_MODEL":"/models/DSv4-Flash-256e-fixed.gguf",
 "DS4_V100_STARTUP_WARMUP":"auto","DS4_V100_TP_EP_VRAM_MIN_FREE_MIB":"64","DS4_V100_TP_EP_NCCL_MIN_FREE_MIB":"1536",
 "DS4_V100_RESERVE_MIB":"0","DS4_V100_NCCL_TOPOLOGY_POLICY":"no-sys","DS4_V100_NCCL_ALLOW_VISIBLE_REMAP":"1",
 "NCCL_P2P_LEVEL":"NVL","NCCL_RINGS":"0 3 2 1 5 7 6 4",
})
PROMPT=("The capital of France is Paris. Continue with a precise, factual paragraph. "+"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "*80+"Return deterministic prose without lists.")
TOKENS=32; POSITION=250000
def req_json(method,port,path,payload=None,timeout=1800):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(f"http://127.0.0.1:{port}{path}",data=data,method=method,headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as resp: return resp.status,resp.read().decode(errors="replace")
def post_one(port,phase,i):
    payload={"model":"ds4","messages":[{"role":"user","content":PROMPT}],"max_tokens":TOKENS,"temperature":0,"top_p":1,"session_id":"s573det-%s-%03d"%(phase,i)}
    s,b=req_json("POST",port,"/v1/chat/completions",payload,1800); d=json.loads(b)
    return {"i":i,"status":s,"seq":d.get("ds4_v100",{}).get("generated_token_sequence")}
def run_batch(port,phase):
    with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as ex:
        return list(ex.map(lambda i: post_one(port,phase,i), range(SLOTS)))
LEGS={
 "eager":      {"sr":"0","extra":""},
 "control-A":  {"sr":"1","extra":""},
 "control-B":  {"sr":"1","extra":""},
 "full":       {"sr":"0","extra":"\n".join(["--decode-cudagraph-gate","--decode-cudagraph-replay-probe-gate","--decode-cudagraph-persistent-replay-gate"])},
}
def run_leg(leg,port):
    cfg=LEGS[leg]; case=ART/leg; case.mkdir(parents=True,exist_ok=True)
    env=BASE_ENV.copy(); env["DS4_V100_PORT"]=str(port); env["DS4_V100_LOG_DIR"]=str(case/"launcher"); env["DS4_LOCK_FILE"]=str(case/"ds4.lock")
    env["DS4_V100_TOKENS"]=str(TOKENS); env["DS4_V100_TP_EP_POSITION"]=str(POSITION); env["DS4_V100_MAX_REQUESTS"]=str(2*SLOTS+32)
    env["DS4_V100_TP_EP_GRAPH_SUFFIX_REPLAY"]=cfg["sr"]; env["DS4_V100_TP_EP_EXTRA_ARGS"]=cfg["extra"]
    with open(case/"server.out","wb") as out, open(case/"server.err","wb") as err:
        proc=subprocess.Popen(["./tools/ds4-v100-run-tp-ep-appliance.sh"],cwd=REPO,env=env,stdout=out,stderr=err,preexec_fn=os.setsid)
    try:
        for _ in range(900):
            if proc.poll() is not None: raise RuntimeError(f"{leg} exited {proc.returncode}")
            try: req_json("GET",port,"/health",None,2); break
            except Exception: time.sleep(1)
        run_batch(port,"warmup")            # one warmup batch (excluded)
        m=run_batch(port,"measured")
        seqs=[tuple(r["seq"] or []) for r in sorted(m,key=lambda r:r["i"])]
        json.dump([list(s) for s in seqs],open(case/"seqs.json","w"))
        return seqs
    finally:
        try: os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=30)
        except Exception:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except Exception: pass
res={}; port=18810
for li,leg in enumerate(LEGS): res[leg]=run_leg(leg,port+li)
def cmp(a,b):
    mis=[i for i in range(SLOTS) if res[a][i]!=res[b][i]]
    offs=[next((j for j,(x,y) in enumerate(zip(res[a][i],res[b][i])) if x!=y),None) for i in mis]
    return {"mismatch":len(mis),"first_offsets":offs[:8]}
summary={
 "control-A_vs_control-B(determinism)":cmp("control-A","control-B"),
 "eager_vs_control-A":cmp("eager","control-A"),
 "eager_vs_full":cmp("eager","full"),
 "control-A_vs_full":cmp("control-A","full"),
 "distinct_per_leg":{k:len({tuple(s) for s in res[k]}) for k in LEGS},
}
json.dump(summary,open(ART/"summary.json","w"),indent=2)
print(json.dumps(summary)); print("DONE")
