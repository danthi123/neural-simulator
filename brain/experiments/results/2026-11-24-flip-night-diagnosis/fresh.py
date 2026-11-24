# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,collections,csv
from brain.experiments.strict_scorer import is_first_person as fp
R="_scratch/stage2/read/";A="_scratch/brainv2/aws/"
LIVES=[(a,s,f"{R}life{a}_s{s}.json") for a in("Bg","KP","PRa","PRl","PRx","S1F") for s in(319,320,321)]
LIVES+=[("M1",s,f"{A}{d}/lifeM1_s{s}.json.json") for s,d in((319,"20261123-1552"),(320,"20261123-1632"),(321,"20261123-1713"))]
flip={(r["arm"],int(r["seed"]),int(r["night"])):(int(r["flip"]),int(r["off_fp"])) for r in csv.DictReader(open("_scratch/flip/nights.tsv"),delimiter="\t")}
T=collections.defaultdict(lambda:[0,0]); per=[]
for arm,seed,path in LIVES:
    r=json.load(open(path)); grp="keep" if arm in("KP","M1") else "nokeep"
    for src,key in (("cue","cue_probes"),("intv","interview_probes")):
        for p in r.get(key) or []:
            if p["arm"]!="hippocampus-off": continue
            rows=[x for x in p["rows"] if (src=="cue" or x.get("kind")=="forced") and not x["brain"].startswith("<think>")]
            if src=="cue": rows=[x for x in rows if x.get("rung") in("C0","C1")]
            f,ofp=flip[(arm,seed,p["night"])]
            k=sum(fp(x["brain"]) for x in rows)
            T[(grp,src,f)][0]+=k; T[(grp,src,f)][1]+=len(rows)
            b="lo" if ofp<=1 else "hi" if ofp>=6 else "mid"
            T[(grp,src,"off_fp "+b)][0]+=k; T[(grp,src,"off_fp "+b)][1]+=len(rows)
            if arm=="KP": per.append((arm,seed,p["night"],src,f,ofp,f"{k}/{len(rows)}",[x["brain"][:50] for x in rows if fp(x["brain"])][:2]))
for k in sorted(T,key=str): print(k,T[k],round(T[k][0]/max(T[k][1],1),3))
for x in per: print(x)
