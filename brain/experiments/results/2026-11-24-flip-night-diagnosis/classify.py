# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,collections,csv,re
from brain.experiments.strict_scorer import is_first_person as fp, is_echo
R="_scratch/stage2/read/";A="_scratch/brainv2/aws/"
LIVES=[(a,s,f"{R}life{a}_s{s}.json") for a in("Bg","KP","PRa","PRl","PRx","S1F") for s in(319,320,321)]
LIVES+=[("M1",s,f"{A}{d}/lifeM1_s{s}.json.json") for s,d in((319,"20261123-1552"),(320,"20261123-1632"),(321,"20261123-1713"))]
REF=re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai",re.I)
def own(t): return bool(re.search(r"\b(my|mine)\b",t,re.I) or re.match(r"I(?:'m| am| | \w+)",t.strip())) and not REF.search(t)
S=collections.defaultdict(collections.Counter); det=[]
for arm,seed,path in LIVES:
    r=json.load(open(path)); L=r["ledger"]
    for p in r["probes"]:
        if p.get("line","trunk")!="trunk" or p.get("tag") or p["arm"]!="hippocampus-off" or p["night"]<2: continue
        t=[(x,L[x["item"]]) for x in p["rows"] if L.get(x.get("item") or "",{}).get("telling")]
        nfp=sum(fp(x["brain"]) for x,e in t)
        if nfp/len(t)<0.5: 
            S[arm]["nonflip"]+=1; continue
        no=sum(own(x["brain"]) for x,e in t); nr=sum(bool(REF.search(x["brain"])) for x,e in t)
        top=collections.Counter(re.sub(r"your .*","your X",x["brain"]) for x,e in t).most_common(1)[0]
        ne=sum(any(is_echo(x["brain"],k,e["telling"]) for k in e["keys"]) for x,e in t)
        refecho=sum(any(is_echo(x["brain"],k,e["telling"]) for k in e["keys"]) and bool(REF.search(x["brain"])) for x,e in t)
        kind="voice" if no>=4 else "refusal/idk" if nr>=4 else "other-loop" if top[1]>=4 else "mixed"
        S[arm][kind]+=1
        if arm in("KP","M1"): det.append((arm,seed,p["night"],f"fp{nfp} own{no} refuse{nr} echo{ne} refuse-echo{refecho}",kind,top[0][:45],top[1]))
for a,c in S.items(): print(a,dict(c))
for d in det: print(d)
