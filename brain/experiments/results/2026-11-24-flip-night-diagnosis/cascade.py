# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,re,collections,sys
sys.path.insert(0,'_scratch/flip')
from brain.experiments.strict_scorer import is_first_person as fp
R="_scratch/stage2/read/";A="_scratch/brainv2/aws/"
LIVES=[(a,s,f"{R}life{a}_s{s}.json") for a in("Bg","KP","PRa","PRl","PRx","S1F") for s in(319,320,321)]
LIVES+=[("M1",s,f"{A}{d}/lifeM1_s{s}.json.json") for s,d in((319,"20261123-1552"),(320,"20261123-1632"),(321,"20261123-1713"))]
def trunk(x): return x.get("line","trunk")=="trunk" and not x.get("tag")
tr=collections.Counter(); onset=collections.Counter(); pat=collections.Counter()
name=collections.defaultdict(lambda:[0,0]); first=collections.defaultdict(lambda:[0,0,0])
armpat=collections.defaultdict(collections.Counter)
mixed=[]
for arm,seed,path in LIVES:
    r=json.load(open(path))
    for p in r["probes"]:
        if not trunk(p) or p["arm"]!="hippocampus-off" or p["night"]<2: continue
        conv=[x for x in p["rows"] if not x["kind"].startswith("open")]
        # conv: greet, name, yourname, cat, work, sister, car, rot, rot, nd, nd, nev*3
        q=conv[3:]  # the 11 'my' questions after the two name rows
        s=[int(fp(x["brain"])) for x in q]
        yn=conv[2]["brain"]; nm="Nova" if "nova" in yn.lower() else ("Dan" if "dan" in yn.lower() else "other")
        k=sum(s)
        for a,b in zip(s,s[1:]): tr[(a,b)]+=1
        # is the fp set a suffix run?
        if 0<k<len(s):
            o=s.index(1); suffix=all(s[o:]); mixed.append((arm,seed,p["night"],"".join(map(str,s))))
            pat["suffix" if suffix else "broken"]+=1
        else: pat["all" if k else "none"]+=1
        armpat[arm]["all" if k==len(s) else "none" if k==0 else "mixed"]+=1
        name[(arm in("KP","M1"),nm)][0]+=k>=6; name[(arm in("KP","M1"),nm)][1]+=1
        f=first[(arm in("KP","M1"),s[0])]; f[0]+=sum(s[1:]); f[1]+=len(s)-1; f[2]+=1
print("transitions (prev fp, this fp):",dict(tr))
print("P(fp|prev fp)=%.3f  P(fp|prev not)=%.3f"%(tr[(1,1)]/(tr[(1,1)]+tr[(1,0)]),tr[(0,1)]/(tr[(0,1)]+tr[(0,0)])))
print("night patterns over 11 conversation 'my' questions:",dict(pat))
print("by arm:",{a:dict(c) for a,c in armpat.items()})
print("first told row (cat) fp -> share of later 10 rows fp: (keepbank?,catfp):[fp rows,rows,nights]",dict(first))
print("your-name reply vs >=6/11 fp nights:",dict(name))
for m in mixed: print(m)
