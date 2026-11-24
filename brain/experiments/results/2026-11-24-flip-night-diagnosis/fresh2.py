# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
import json,collections,csv,re
R="_scratch/stage2/read/";A="_scratch/brainv2/aws/"
LIVES=[(a,s,f"{R}life{a}_s{s}.json") for a in("Bg","KP","PRa","PRl","PRx","S1F") for s in(319,320,321)]
LIVES+=[("M1",s,f"{A}{d}/lifeM1_s{s}.json.json") for s,d in((319,"20261123-1552"),(320,"20261123-1632"),(321,"20261123-1713"))]
flip={(r["arm"],int(r["seed"]),int(r["night"])):(int(r["flip"]),int(r["off_fp"])) for r in csv.DictReader(open("_scratch/flip/nights.tsv"),delimiter="\t")}
REF=re.compile(r"sorry|don't|do not|can't|cannot|not sure|as an ai|i think|i can |i'd|i'm glad|i hope|i remember|i'll",re.I)
def own(t):  # user-voice answer: 'my/mine' or a bare 'I <verb>' statement, not a refusal/hedge
    t=t.strip()
    return bool(re.search(r"\b(my|mine)\b",t,re.I) or re.match(r"I(?:'m| am| | \w+)",t)) and not REF.search(t) and "my friend" not in t.lower()
T=collections.defaultdict(lambda:[0,0]); chk=collections.Counter()
for arm,seed,path in LIVES:
    r=json.load(open(path)); grp=arm if arm in("KP","M1","Bg") else "PR/S1F"
    # sanity: strict marker on the battery told rows
    for p in r["probes"]:
        if p.get("line","trunk")!="trunk" or p.get("tag") or p["arm"]!="hippocampus-off": continue
        f,ofp=flip[(arm,seed,p["night"])]
        rows=[x for x in p["rows"] if x["kind"] in("H","rotating","neardup","never")][2:]
        T[(grp,"battery-conv",f)][0]+=sum(own(x["brain"]) for x in rows); T[(grp,"battery-conv",f)][1]+=len(rows)
    for src,key in (("fresh",'cue_probes'),("fresh","interview_probes")):
        for p in r.get(key) or []:
            if p["arm"]!="hippocampus-off": continue
            rows=[x for x in p["rows"] if (key=="cue_probes" or x.get("kind")=="forced") and not x["brain"].startswith("<think>") and re.search(r"\b(my|I|me)\b",x["you"])]
            f,ofp=flip[(arm,seed,p["night"])]
            T[(grp,src,f)][0]+=sum(own(x["brain"]) for x in rows); T[(grp,src,f)][1]+=len(rows)
            if ofp>=7 and arm in("KP","M1"): T[(grp,"fresh on 7-8/8 nights","")][0]+=sum(own(x["brain"]) for x in rows); T[(grp,"fresh on 7-8/8 nights","")][1]+=len(rows)
for k in sorted(T,key=str): print(k,T[k],round(T[k][0]/max(T[k][1],1),3))
