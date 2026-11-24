# Capability: Self and identity
# Record: brain/experiments/results/2026-11-24-flip-night-diagnosis.md (host-only read of banked lives)
# Expected-fail: no (a read-out; no model run)
"""First-person flip nights across banked slept DEV lives. Read-only on life JSONs; no model.
Usage: PYTHONPATH=. python _scratch/flip/flip.py  -> prints tables, writes _scratch/flip/nights.tsv"""
import json, glob, re, os, statistics as st, collections
from brain.experiments.strict_scorer import is_echo, is_first_person, content_words, _overlap, _words
R = "_scratch/stage2/read/"; A = "_scratch/brainv2/aws/"
LIVES = []
for arm in ("Bg", "KP", "PRa", "PRl", "PRx", "S1F"):
    for s in (319, 320, 321): LIVES.append((arm, s, f"{R}life{arm}_s{s}.json"))
for s, d in ((319, "20261123-1552"), (320, "20261123-1632"), (321, "20261123-1713")):
    LIVES.append(("M1", s, f"{A}{d}/lifeM1_s{s}.json.json"))
LIVES.append(("AL", 322, R + "lifeAL_s322.json"))
LIVES.append(("S1FULL", 318, A + "20261120-1917/life_s318_S1FULL.json.json"))
for t in ("B", "S", "SAB"): LIVES.append((t + "7d", 318, A + f"20261120-0412/life_s318_{t}.json.json"))
LIVES.append(("SABE7d", 318, A + "20261120-1226/life_s318_SABE.json.json"))
SECOND = {"you", "your", "yours", "you're", "youre", "you've"}
def fp(t): return is_first_person(t)
def trunk(x): return x.get("line", "trunk") == "trunk" and not x.get("tag")
def slog(n, i, key):
    try: return float(re.search(key + r" ([\d.]+)", n["sleep_log"][i]).group(1))
    except Exception: return None
rows_out = []; BAT = {}; SCHED = {}
for arm, seed, path in LIVES:
    r = json.load(open(path)); assert r["split"] == "dev", path
    L = r["ledger"]
    P = {(p["night"], p["arm"]): p for p in r["probes"] if trunk(p)}
    N = {n["night"]: n for n in r["nights"] if trunk(n)}
    LV = {l["day"]: l for l in r["lived"] if trunk(l)}
    for d in sorted(N):
        n = N[d]; o = dict(arm=arm, seed=seed, night=d)
        def told(pa, H=None):
            p = P.get((d, pa)); out = []
            for row in (p["rows"] if p else []):
                e = L.get(row.get("item") or "")
                if e and e.get("telling") and (H is None or (row["kind"] == "H") == H):
                    out.append((row, e))
            return out
        for pa, tg in (("hippocampus-off", "off"), ("intact", "int"), ("intact-frozen", "frz")):
            t = told(pa)
            o[tg + "_n"] = len(t)
            o[tg + "_fp"] = sum(fp(x["brain"]) for x, e in t)
            o[tg + "_echo"] = sum(any(is_echo(x["brain"], k, e["telling"]) for k in e["keys"]) for x, e in t)
            o[tg + "_ok"] = sum(bool(x.get("ok")) for x, e in t)
        tH = told("hippocampus-off", True); o["offH_fp"] = sum(fp(x["brain"]) for x, e in tH); o["offH_n"] = len(tH)
        p = P.get((d, "hippocampus-off"))
        if p:   # 'my' questions with no told answer (never / neardup): first-person share
            nv = [x for x in p["rows"] if x["kind"] in ("never", "neardup")]
            o["offNV_fp"] = sum(fp(x["brain"]) for x in nv); o["offNV_n"] = len(nv)
            op = [x for x in p["rows"] if x["kind"].startswith("open")]
            o["collapse"] = p["summary"].get("collapse"); o["factor_norms"] = p.get("factor_norms")
            o["dT"] = (p.get("transfer") or {}).get("dT")
            BAT[(arm, seed, d)] = tuple(x["you"] for x in p["rows"])
        o["flip"] = int(o["off_n"] > 0 and o["off_fp"] / o["off_n"] >= 0.5)
        o.update(eps=n.get("episodes_before"), pend=n.get("pending"), store=n.get("store"), anorm=n.get("adapter_norm"),
                 kl=(n.get("lang_guard") or {}).get("kl"), nll_on=(n.get("lang_guard") or {}).get("nll_on"),
                 rep0=slog(n, 0, "replay"), repN=slog(n, -1, "replay"), keep0=slog(n, 0, "keep"), keepN=slog(n, -1, "keep"),
                 keepkl=(n.get("keep_bank") or {}).get("mean_keep_kl"), sec=n.get("seconds"))
        rs = n.get("replay_share") or {}; tot = sum(rs.values()) or 1
        for c in ("AB", "D", "E", "F", "H", "T", "other"): o["rs_" + c] = round(rs.get(c, 0) / tot, 3)
        lv = LV.get(d, {"rows": []})["rows"]
        o["lv_n"] = len(lv); o["lv_kinds"] = ",".join(f"{k}{v}" for k, v in sorted(collections.Counter(x["kind"] for x in lv).items()))
        o["lv_fp"] = sum(fp(x["brain"]) for x in lv)
        o["lv_too"] = sum(bool(re.search(r"\btoo\b", x["brain"].lower())) for x in lv)
        # brain mirrors the user's statement in first person (own 'me too' talk): fp reply holding >half the user's content words
        o["lv_mirror"] = sum(fp(x["brain"]) and not set(_words(x["brain"])) & SECOND and len(content_words(x["you"])) > 0
                             and _overlap(x["brain"], content_words(x["you"])) > 0.5 for x in lv)
        o["lv_my"] = sum(bool(re.search(r"\b(my|mine)\b", x["brain"].lower())) for x in lv)
        q = [x for x in lv if x["you"].strip().endswith("?") and x["kind"] != "greet"]
        o["lv_qfp"] = sum(bool(re.search(r"\b(my|mine)\b", x["brain"].lower())) for x in q); o["lv_q"] = len(q)
        o["lv_own"] = int(o["lv_my"] + o["lv_too"] + o["lv_mirror"] > 0)
        o["lv_tell"] = sum(x["kind"] == "tell" for x in lv); o["lv_fpuser"] = sum(fp(x["you"]) for x in lv)
        o["lv_recalled"] = sum(bool(x.get("recalled")) for x in lv)
        sd = [x for x in lv if x["kind"] == "sameday"]; o["sd_fp"] = sum(fp(x["brain"]) for x in sd); o["sd_n"] = len(sd)
        sc = [s for s in r["schedule"] if s["day"] == d]
        SCHED[(arm, seed, d)] = tuple((t["kind"], t["item"]) for t in sc[0]["turns"]) if sc else ()
        o["told_cls"] = ",".join(f"{k}{v}" for k, v in sorted(collections.Counter(L[t["item"]]["cls"] for t in (sc[0]["turns"] if sc else []) if t["item"] in L and t["kind"] == "tell").items()))
        rows_out.append(o)
cols = [c for c in rows_out[0] if c != "factor_norms"] + ["factor_norms"]
with open("_scratch/flip/nights.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for o in rows_out: f.write("\t".join(str(o.get(c)) for c in cols) + "\n")
G = collections.defaultdict(dict)
for o in rows_out: G[(o["arm"], o["seed"])][o["night"]] = o
print("== A. hippocampus-off told rows: first-person/n per night (* = flip >=0.5); second line echo off|intact")
for k, v in G.items():
    print(f"{k[0]:7}{k[1]} fp  ", " ".join(f"{v[d]['off_fp']}/{v[d]['off_n']}{'*' if v[d]['flip'] else ' '}" for d in sorted(v)))
    print(f"{'':10} ech ", " ".join(f"{v[d]['off_echo']}|{v[d]['int_echo']} " for d in sorted(v)))
print("\n== B. flip nights, with intact/frozen fp, H-only fp, never/neardup fp, next-day sameday fp")
for k, v in G.items():
    for d in sorted(v):
        o = v[d]
        if o["flip"]:
            nx = v.get(d + 1)
            print(k, "n", d, f"off {o['off_fp']}/{o['off_n']} echo {o['off_echo']} ok {o['off_ok']} | intact fp {o['int_fp']}/{o['int_n']} echo {o['int_echo']} ok {o['int_ok']} | frozen fp {o['frz_fp']}/{o['frz_n']} | H-only {o['offH_fp']}/{o['offH_n']} | never+neardup fp {o['offNV_fp']}/{o['offNV_n']} | next off fp {nx['off_fp'] if nx else '-'}/{nx['off_n'] if nx else '-'} | next-day lived fp {nx['lv_fp'] if nx else '-'}/{nx['lv_n'] if nx else '-'}")
print("\n== C. flip count by night across 15-day lives (n lives flipping), and by seed")
byn = collections.Counter(); bys = collections.Counter(); nl = collections.Counter()
for o in rows_out:
    if o["arm"].endswith("7d"): continue
    nl[o["seed"]] += 1; byn[o["night"]] += o["flip"]; bys[o["seed"]] += o["flip"]
print("by night", dict(sorted(byn.items()))); print("by seed", {s: f"{bys[s]}/{nl[s]}" for s in sorted(nl)})
for s in (319, 320, 321):
    print(s, {a: [d for d in sorted(v) if v[d]["flip"]] for (a, ss), v in G.items() if ss == s})
print("\n== D. flip vs adjacent non-flip nights (same life, night-1 and night+1), numeric fields: mean flip | mean prev | mean next | n pairs | share of pairs flip>prev")
F = ["eps", "pend", "store", "anorm", "kl", "nll_on", "rep0", "repN", "keep0", "keepN", "keepkl", "sec", "rs_AB", "rs_D", "rs_E", "rs_F", "rs_H", "rs_T", "rs_other",
     "lv_n", "lv_fp", "lv_too", "lv_mirror", "lv_my", "lv_qfp", "lv_own", "lv_tell", "lv_fpuser", "lv_recalled", "collapse", "dT", "d_anorm", "d_kl"]
for k, v in G.items():
    for d in v:
        for a, b in (("d_anorm", "anorm"), ("d_kl", "kl")):
            v[d][a] = (v[d][b] - v[d - 1][b]) if d - 1 in v and v[d][b] is not None and v[d - 1][b] is not None else None
for f in F:
    fl, pv, nx, up, npair = [], [], [], 0, 0
    for k, v in G.items():
        for d in v:
            if not v[d]["flip"] or v[d].get(f) is None: continue
            fl.append(v[d][f])
            if d - 1 in v and not v[d - 1]["flip"] and v[d - 1].get(f) is not None:
                pv.append(v[d - 1][f]); npair += 1; up += v[d][f] > v[d - 1][f]
            if d + 1 in v and not v[d + 1]["flip"] and v[d + 1].get(f) is not None: nx.append(v[d + 1][f])
    allnf = [o[f] for o in rows_out if not o["flip"] and o.get(f) is not None and o["night"] >= 5]
    m = lambda x: round(st.mean(x), 4) if x else None
    print(f"{f:11} flip {m(fl)} (n{len(fl)}) | prev {m(pv)} | next {m(nx)} | all non-flip n>=5 {m(allnf)} | flip>prev {up}/{npair}")
print("\n== G. per arm (nights 2-15): flips, off_fp histogram (0-1 / 2-5 / 6-8), P(flip|prev flip), P(flip|prev not), mean KL & adapter norm at n15; P(flip | lived own-talk that day) with prev night non-flip")
ARMS = collections.OrderedDict()
for (a, s_), v in G.items(): ARMS.setdefault(a, []).append(v)
for a, vs in ARMS.items():
    fl = tot = pf = pfn = nf = nfn = 0; h = [0, 0, 0]; own = [0, 0]; noown = [0, 0]; my = [0, 0]; nomy = [0, 0]
    for v in vs:
        for d in sorted(v):
            if d < 2: continue
            o = v[d]; tot += 1; fl += o["flip"]; h[0 if o["off_fp"] <= 1 else 1 if o["off_fp"] <= 5 else 2] += 1
            if d - 1 in v:
                if v[d - 1]["flip"]: pfn += 1; pf += o["flip"]
                else:
                    nfn += 1; nf += o["flip"]
                    t = own if o["lv_own"] else noown; t[0] += o["flip"]; t[1] += 1
                    t = my if o["lv_my"] else nomy; t[0] += o["flip"]; t[1] += 1
    last = [v[max(v)] for v in vs]
    kl = [o["kl"] for o in last if o["kl"] is not None]
    print(f"{a:7} flips {fl}/{tot} hist {h} P(f|f) {pf}/{pfn} P(f|nf) {nf}/{nfn} | KL_last {round(st.mean(kl),3) if kl else None} anorm_last {round(st.mean(o['anorm'] for o in last),1)} | own-talk day: {own[0]}/{own[1]} vs none {noown[0]}/{noown[1]} | lived 'my' reply: {my[0]}/{my[1]} vs none {nomy[0]}/{nomy[1]}")
print("\n== H. keep-bank arms (KP, M1) only: flip vs non-flip nights (n>=2), mean of each field")
for a in ("KP", "M1", "Bg"):
    for f in F:
        x = [v[d][f] for v in ARMS[a] for d in v if d >= 2 and v[d]["flip"] and v[d].get(f) is not None]
        y = [v[d][f] for v in ARMS[a] for d in v if d >= 2 and not v[d]["flip"] and v[d].get(f) is not None]
        if x and y: print(f"{a:3} {f:11} flip {st.mean(x):.4f} (n{len(x)}, min {min(x):.3f} max {max(x):.3f}) | non-flip {st.mean(y):.4f} (n{len(y)}, sd {st.pstdev(y):.4f})")
print("\n== I. lived own-talk rows on KP/M1 flip days (brain reply holding my/mine, or 'too', or mirror)")
for a in ("KP", "M1"):
    for (aa, s_, path) in LIVES:
        if aa != a: continue
        r = json.load(open(path)); v = G[(a, s_)]
        for l in r["lived"]:
            if l.get("line", "trunk") == "trunk" and v[l["day"]]["flip"]:
                hits = [x for x in l["rows"] if re.search(r"\b(my|mine|too)\b", x["brain"].lower())]
                print(a, s_, "day", l["day"], "prev flip", v.get(l["day"] - 1, {}).get("flip"), "|", " || ".join(f"[{x['kind']}] {x['you'][:50]} -> {x['brain'][:80]}" for x in hits[:3]) or "none")
print("\n== E. per-night table (15-day lives, arms KP/Bg/M1), fields")
for k, v in G.items():
    if k[0] not in ("KP", "Bg", "M1"): continue
    print(k, "night offFP echo intEcho | eps pend anorm kl rep0>repN keep0>keepN keepkl | rs AB/D/E/F/H/T/other | lived n fp too mirror | told cls")
    for d in sorted(v):
        o = v[d]
        print(f" {d:2} {o['off_fp']}/{o['off_n']}{'*' if o['flip'] else ' '} {o['off_echo']} {o['int_echo']} | {o['eps']} {o['pend']} {o['anorm']} {o['kl']} {o['rep0']}>{o['repN']} {o['keep0']}>{o['keepN']} {round(o['keepkl'],4) if o['keepkl'] else '-'} | "
              + "/".join(str(o['rs_' + c]) for c in ("AB", "D", "E", "F", "H", "T", "other")) + f" | {o['lv_n']} {o['lv_fp']} {o['lv_too']} {o['lv_mirror']} | {o['told_cls']}")
print("\n== F. schedule / battery identity")
for s in (319, 320, 321):
    arms = [a for (a, ss) in G if ss == s]
    same_s = all(SCHED[(a, s, d)] == SCHED[(arms[0], s, d)] for a in arms for d in range(1, 16))
    same_b = all(BAT.get((a, s, d)) == BAT.get((arms[0], s, d)) for a in arms for d in range(1, 16))
    print(s, "schedule identical across arms:", same_s, "| off-battery questions identical across arms:", same_b)
print("battery identical across seeds per night:", {d: len({BAT[("KP", s, d)] for s in (319, 320, 321)}) == 1 for d in range(1, 16)})
print("schedule (kind,item) identical across seeds per day:", {d: len({SCHED[("KP", s, d)] for s in (319, 320, 321)}) == 1 for d in range(1, 16)})
print("battery size per night KP s320:", [len(BAT[("KP", 320, d)]) for d in range(1, 16)])
fixed = set.intersection(*[set(BAT[("KP", 320, d)]) for d in range(1, 16)])
print("questions asked every night (KP s320):", len(fixed))
