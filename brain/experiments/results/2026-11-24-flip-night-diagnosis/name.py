# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,collections,re
R="_scratch/stage2/read/";A="_scratch/brainv2/aws/"
LIVES=[(a,s,f"{R}life{a}_s{s}.json") for a in("Bg","KP","PRa","PRl","PRx","S1F") for s in(319,320,321)]
LIVES+=[("M1",s,f"{A}{d}/lifeM1_s{s}.json.json") for s,d in((319,"20261123-1552"),(320,"20261123-1632"),(321,"20261123-1713"))]
REF=re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai",re.I)
def own(t): return bool(re.search(r"\b(my|mine)\b",t,re.I) or re.match(r"I(?:'m| am| | \w+)",t.strip())) and not REF.search(t)
T=collections.defaultdict(lambda:[0,0]); V=collections.defaultdict(list); tr=collections.defaultdict(lambda:[0,0])
for arm,seed,path in LIVES:
    r=json.load(open(path)); L=r["ledger"]; g="keep" if arm in("KP","M1") else "nokeep"
    for p in sorted(r["probes"],key=lambda p:p["night"]):
        if p.get("line","trunk")!="trunk" or p.get("tag") or p["arm"]!="hippocampus-off" or p["night"]<2: continue
        conv=[x for x in p["rows"] if not x["kind"].startswith("open")]
        t=[x for x in p["rows"] if L.get(x.get("item") or "",{}).get("telling")]
        v=sum(own(x["brain"]) for x in t)>=4
        yn=conv[2]["brain"].lower(); nm="Nova" if "nova" in yn else "Dan" if "dan" in yn else "other"
        T[(g,nm)][0]+=v; T[(g,nm)][1]+=1
        V[(arm,seed)].append((p["night"],int(v)))
for k,v in sorted(T.items()): print(k,v)
for arm in("KP","M1","Bg"):
    a=b=c=d=0
    for (ar,s),seq in V.items():
        if ar!=arm: continue
        for (n1,v1),(n2,v2) in zip(seq,seq[1:]):
            if v1: a+=v2; b+=1
            else: c+=v2; d+=1
    print(arm,"voice nights",{s:[n for n,v in V[(arm,s)] if v] for s in(319,320,321)},"P(v|prev v)",a,b,"P(v|prev not)",c,d)
