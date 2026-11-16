# Capability: Self and identity
# Record: brain/research/2026-11-15-self-recall-after-night-1.5b.md (P0-P3)
"""Score the self-recall probe (brain/research/2026-11-15-self-recall-after-night-1.5b.md, P0-P3). Usage: self_recall_score.py <dir with srp_*.json>"""
import json,glob,sys
P=sys.argv[1]
Q="What's your name?"
def is_self(t): return t is not None and "Assistant:" in t
homes=[]
for f in sorted(glob.glob(P+"/srp_*.json")):
    d=json.load(open(f))
    for h in d["homes"]:
        h["_host"]=d["host"]; homes.append(h)
def seed(h): return h["home"].split("_s")[-1].split("/")[0]
rows={"off":[],"on":[]}
for h in homes:
    arm="on" if "workSC_on" in h["home"] else "off"
    r={"seed":seed(h),"host":h["_host"],"bar":h.get("S0_key_bar"),"sall_bar":h.get("S_all_key_bar")}
    for k in ["S0","S_all"]:
        for rd in ["live","raw"]:
            e=h[f"{k}|{rd}|{Q}"]
            r[f"{k}_{rd}"]=(e["winner"][:40], e["passes"], round(e["best3"][0][1],4), e["bar"])
            r[f"{k}_{rd}_scores"]=[s for _,s in e["best3"]]
            r[f"{k}_{rd}_self"]= is_self(e["winner"]) and e["passes"]
            r[f"{k}_{rd}_selfwin"]= is_self(e["winner"])
            for c,tag in [("What is my sister's name?","sis"),("What is my cat called?","cat")]:
                ec=h[f"{k}|{rd}|{c}"]
                r[f"{k}_{rd}_{tag}"]=(("sister" in ec["winner"]) if tag=="sis" else ("cat" in ec["winner"])) and ec["passes"]
    rows[arm].append(r)
for arm in ["off","on"]:
    R=sorted(rows[arm],key=lambda r:r["seed"])
    print("==",arm,"homes",len(R))
    for r in R:
        print(r["seed"],r["host"],"S0bar",r["bar"],"| S0 live",r["S0_live"],"raw",r["S0_raw"],"| S_all live",r["S_all_live"],"raw",r["S_all_raw"],"| ctrl S_all live sis/cat",r["S_all_live_sis"],r["S_all_live_cat"])
    n=len(R)
    p0a=sum(r["S_all_live_scores"]!=r["S_all_raw_scores"] for r in R)
    p0b=sum(r["S0_live_scores"]!=r["S0_raw_scores"] for r in R)
    p1=sum(r["S0_raw_self"] and not r["S0_live_self"] for r in R)
    p2=sum((not r["S0_live_self"]) and (not r["S0_raw_self"]) for r in R)
    p3=sum(r["S_all_live_self"] and r["S_all_live_sis"] and r["S_all_live_cat"] for r in R)
    p3w=sum(r["S_all_live_selfwin"] for r in R)
    print(f"P0 S_all live!=raw {p0a}/{n} (S0 live!=raw {p0b}/{n}); P1 {p1}/{n}; P2 {p2}/{n}; P3 {p3}/{n} (S_all live self WINS ignoring bar {p3w}/{n})")
