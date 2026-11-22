# Capability: Lifelong learning
# Record: brain/research/2026-11-21-sleep-language-keep-prereg.md (section 3 bars)
# Expected-fail: none (host-only post-hoc read)
"""K-P DEV read: prereg 2026-11-21-sleep-language-keep-prereg.md section 3 (with A1 section 5, A2 section 6; arm launched
gate-overridden). K-P (lifeKP) vs B (lifeS1F), seeds 319/320/321. Every number is read from the life JSONs / bank files
by existing brain.eval code; nothing is estimated.
Run: PYTHONPATH=<frozen brain copy> python kp_bars.py  (writes kp_bars.md beside this file)"""
import json, os, re
from brain.eval import life_leak, life_echo, life_table, life_bars

SEEDS = [319, 320, 321]
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"B": "lifeS1F_s{}.json", "K-P": "lifeKP_s{}.json", "PR-l": "lifePRl_s{}.json", "PR-x": "lifePRx_s{}.json"}
R = {k: {s: json.load(open(os.path.join(HERE, v.format(s)))) for s in SEEDS} for k, v in FILES.items()}
BANKS = {  # seed -> (path, label)
    321: ("/home/dant123/Projects/sim/_scratch/wt-mixday/_scratch/brainv2/aws/20261122-0127/home_KP_s321/trunk/keep_bank.json", "AWS bank (the life's own)"),
    319: ("/home/dant123/Projects/sim/_scratch/kpgate_w/home_KP_s319/keep_bank.json", "PROXY: CPU one-night-gate bank (different device, not the life's bank)"),
    320: (None, "not pulled, no proxy"),
}
out = []
def P(x=""): out.append(x)
def f(x, k=3): return "-" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))
def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None
def vd(ok): return "UNDEFINED" if ok is None else ("PASS" if ok else "FAIL")
def allp(l): return None if any(x is None for x in l) else all(l)
def trunk(res): return [n for n in res["nights"] if n["line"] == "trunk"]
def nrec(res, n): return next((x for x in trunk(res) if x["night"] == n), None)
def trunk_probes(res, arm):
    return [p for p in res["probes"] if p["arm"] == arm and p.get("line", "trunk") == "trunk" and not p.get("tag")]
def mid(a, b): return (a + b) / 2

P("# K-P DEV read (prereg 2026-11-21-sleep-language-keep-prereg.md, section 3; A1 section 5, A2 section 6; launched gate-overridden)")
P("Seeds 319/320/321 (DEV). B = lifeS1F, K-P = lifeKP (`keep_bank='prior'`). All values read from the JSONs and bank files by "
  "`kp_bars.py` (brain.eval.life_leak, life_echo, life_table, life_bars). Bars per seed and pooled as the prereg states. "
  "Pooled = mean over the three seeds (ratio of means).\n")

P("## 0. Runs and devices (precision confound)\n")
P("| arm | seed | device | host | brain_kw | last trunk night |")
P("|---|---|---|---|---|---|")
for k in R:
    for s in SEEDS:
        r = R[k][s]
        P(f"| {k} | {s} | {r['device']} | {r['host']} | {r['brain_kw']} | {trunk(r)[-1]['night']} |")
P()
P("**Confound.** K-P ran CUDA fp16 (AWS A10G); B ran CPU fp32 (pool). Every K-P vs B difference below is method + device/precision together. "
  "Section 5 gives the available evidence on how much fp16 alone moves lang_guard KL.\n")

# ---------------------------------------------------------------- bar 1
P("## 1. Engagement (must hold, or the arm is UNDEFINED)\n")
P("### 1a. keep_bank record on every trunk night (from the life JSON)\n")
P("Bar: a `keep_bank` record on every trunk night 1-15, `bank_size` 512, `subset_n` set, a different `subset_hash` each night, `mean_keep_kl` > 0 from night 2.\n")
P("| seed | nights with record | bank_size all 512 | subset_n values | distinct hashes / nights | min mean_keep_kl n2-15 (night) | mean_keep_kl n1 | B has keep_bank record | 1a |")
P("|---|---|---|---|---|---|---|---|---|")
e1a = []
for s in SEEDS:
    t = trunk(R["K-P"][s])
    kb = [(n["night"], n.get("keep_bank")) for n in t]
    have = [n for n, x in kb if x]
    sizes = {x["bank_size"] for n, x in kb if x}
    subn = {x["subset_n"] for n, x in kb if x}
    hashes = [x["subset_hash"] for n, x in kb if x]
    kls = [(x["mean_keep_kl"], n) for n, x in kb if x and n >= 2]
    mn = min(kls) if kls else None
    ok = (len(have) == len(t) == 15 and sizes == {512} and len(subn) == 1 and None not in subn and len(set(hashes)) == len(hashes) == 15
          and bool(kls) and all(v > 0 for v, _ in kls))
    e1a.append(ok)
    k1 = next((x["mean_keep_kl"] for n, x in kb if n == 1 and x), None)
    bhas = sum(1 for n in trunk(R["B"][s]) if n.get("keep_bank"))
    P(f"| {s} | {len(have)}/{len(t)} | {sizes == {512}} | {sorted(subn)} | {len(set(hashes))}/{len(hashes)} | {f(mn[0], 4) if mn else '-'} (n{mn[1] if mn else '-'}) | {f(k1, 4)} | {bhas}/15 nights | {vd(ok)} |")
P()
P("mean_keep_kl by night (K-P):\n")
P("| seed | " + " | ".join(f"n{n}" for n in range(1, 16)) + " |")
P("|---|" + "---|" * 15)
for s in SEEDS:
    P(f"| {s} | " + " | ".join(f(n["keep_bank"]["mean_keep_kl"], 3) for n in trunk(R["K-P"][s])) + " |")
P()
P(f"**1a: {vd(allp(e1a))}** per seed {dict(zip(SEEDS, e1a))}.\n")

P("### 1b. Bank quality (from the bank files)\n")
P("Bar: >=90% of the 512 texts distinct; mean sampled length >= 40 tokens (the bank's own `mean_tokens` field); <10% of texts contain a told key of that "
  "seed's ledger as a whole word, case-insensitive. Told keys = keys of ledger items with a telling (telling is not None: the 54 told facts; the two "
  "brain-own items dan/nova have no telling and are not 'told'); the sensitivity column also counts dan/nova keys. Whole word = regex `\\bkey\\b`.\n")
P("| seed | bank | n texts | distinct | % distinct | mean_tokens | texts with a told key | % | (all keys incl dan/nova) | 1b |")
P("|---|---|---|---|---|---|---|---|---|---|")
e1b = {}
bankinfo = {}
for s in SEEDS:
    path, label = BANKS[s]
    if not path or not os.path.exists(path):
        P(f"| {s} | {label} | UNDEFINED | | | | | | | UNDEFINED |"); e1b[s] = None; continue
    b = json.load(open(path)); tx = b["texts"]
    led = R["K-P"][s]["ledger"]
    told = sorted({k for v in led.values() if v.get("telling") for k in v["keys"]})
    allk = sorted({k for v in led.values() for k in v["keys"]})
    def hit(keys):
        rx = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b", re.I)
        return sum(1 for t in tx if rx.search(t))
    nd = len(set(tx)); ht, ha = hit(told), hit(allk)
    ok = nd / len(tx) >= 0.90 and b["mean_tokens"] >= 40 and ht / len(tx) < 0.10
    bankinfo[s] = (b, told, allk)
    e1b[s] = ok
    P(f"| {s} | {label} | {len(tx)} (n field {b['n']}) | {nd} | {nd/len(tx)*100:.1f}% | {f(b['mean_tokens'], 2)} | {ht} | {ht/len(tx)*100:.1f}% | {ha} | {vd(ok)}{' (proxy)' if s == 319 else ''} |")
P()
for s in SEEDS:
    if s in bankinfo:
        b, told, allk = bankinfo[s]
        P(f"- s{s}: bank seed field {b['seed']}, sampler `{b['sampler']}`, T {b['temperature']}, top_p {b['top_p']}, pen {b['presence_penalty']}/{b['frequency_penalty']}, "
          f"{len(told)} told keys checked: {told}.")
P()
P("s320 bank quality is UNDEFINED (AWS bank not pulled, instance terminated, no proxy). s319 is a CPU one-night-gate bank (proxy only, labelled; "
  "same sampler code but a different device and run). Only s321 is the life's own bank.\n")
b1b = [e1b[s] for s in SEEDS]
P(f"**1b: s321 (life's own bank) {vd(e1b[321])}; s319 {vd(e1b[319])} as PROXY only; s320 UNDEFINED.** "
  "Bar 1 as a whole therefore cannot be asserted for 2 of 3 seeds: 1a holds from the logs, 1b is UNDEFINED for s320 and proxy-only for s319.\n")
bar1 = None if (not all(e1a) or e1b[321] is False) else None
if not all(e1a) or e1b[321] is False or e1b[319] is False:
    bar1 = False
BAR1 = "FAIL" if bar1 is False else "UNDEFINED (1a PASS 3/3; 1b s321 " + vd(e1b[321]) + ", s319 proxy " + vd(e1b[319]) + ", s320 undefined)"

# ---------------------------------------------------------------- bar 2
P("## 2. Primary: held-out language guard (nights[].lang_guard: kl, nll_on; nll_off is the pretrained cortex, constant)\n")
def lg(k, s, n): return (nrec(R[k][s], n) or {}).get("lang_guard")
P("| seed | night | B KL | K-P KL | K-P/B | B NLL_on | K-P NLL_on | nll_off (pretrained) |")
P("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for n in (1, 5, 10, 15):
        b, k = lg("B", s, n), lg("K-P", s, n)
        P(f"| {s} | {n} | {f(b['kl'], 4)} | {f(k['kl'], 4)} | {f(k['kl']/b['kl'], 3)} | {f(b['nll_on'], 3)} | {f(k['nll_on'], 3)} | {f(b['nll_off'], 3)} / {f(k['nll_off'], 3)} |")
for n in (1, 5, 10, 15):
    bk = mean([lg("B", s, n)["kl"] for s in SEEDS]); kk = mean([lg("K-P", s, n)["kl"] for s in SEEDS])
    bn = mean([lg("B", s, n)["nll_on"] for s in SEEDS]); kn = mean([lg("K-P", s, n)["nll_on"] for s in SEEDS])
    P(f"| pooled mean | {n} | {f(bk, 4)} | {f(kk, 4)} | {f(kk/bk, 3)} | {f(bn, 3)} | {f(kn, 3)} | |")
P()
P("KL by night, every trunk night (B / K-P):\n")
P("| seed | " + " | ".join(f"n{n}" for n in range(1, 16)) + " |")
P("|---|" + "---|" * 15)
for s in SEEDS:
    P(f"| {s} | " + " | ".join(f"{lg('B', s, n)['kl']:.2f}/{lg('K-P', s, n)['kl']:.2f}" for n in range(1, 16)) + " |")
P()
def klbar(n):
    bk = mean([lg("B", s, n)["kl"] for s in SEEDS]); kk = mean([lg("K-P", s, n)["kl"] for s in SEEDS])
    below = [lg("K-P", s, n)["kl"] < lg("B", s, n)["kl"] for s in SEEDS]
    return kk <= 0.5 * bk, below, kk, bk
def nllbar(n, own_n1):
    res = []
    for s in SEEDS:
        ref1 = lg("K-P" if own_n1 else "B", s, 1)["nll_on"]
        m = mid(ref1, lg("B", s, n)["nll_on"])
        res.append((lg("K-P", s, n)["nll_on"] <= m, lg("K-P", s, n)["nll_on"], m, ref1, lg("B", s, n)["nll_on"]))
    return res
P("### 2a. KL bar (night 15: K-P pooled <= 0.5 x B pooled AND K-P < B on 3/3 seeds)\n")
half15, below15, kk15, bk15 = klbar(15)
P(f"Night 15: pooled K-P {kk15:.4f} vs B {bk15:.4f}; ratio {kk15/bk15:.3f}; half-of-B bar {0.5*bk15:.4f}; K-P below B per seed {dict(zip(SEEDS, below15))}.")
kl_pass = half15 and all(below15)
P(f"**KL n15 bar: {vd(kl_pass)}**\n")
half10, below10, kk10, bk10 = klbar(10)
P(f"Night 10 (reported, same direction 3/3): pooled K-P {kk10:.4f} vs B {bk10:.4f}; ratio {kk10/bk10:.3f}; <= half: {half10}; below B per seed {dict(zip(SEEDS, below10))}.\n")
P("### 2b. NLL bar (night 15: K-P nll_on <= midpoint(night 1, B night 15), per seed, 3/3)\n")
P("The prereg says \"the midpoint between night 1 and B's night 15\"; night 1 is ambiguous, so both readings are shown. (i) K-P's own night 1; (ii) B's night 1.\n")
P("| seed | reading | night-1 ref | B n15 | midpoint | K-P n15 | <= midpoint |")
P("|---|---|---|---|---|---|---|")
nll_res = {}
for own in (True, False):
    r15 = nllbar(15, own); nll_res[("own", own)] = r15
    for s, (ok, v, m, r1, b15) in zip(SEEDS, r15):
        P(f"| {s} | {'K-P own n1' if own else 'B n1'} | {r1:.3f} | {b15:.3f} | {m:.3f} | {v:.3f} | {ok} |")
nll_pass = {own: all(x[0] for x in nll_res[("own", own)]) for own in (True, False)}
P()
P(f"**NLL n15 bar: own-n1 reading {vd(nll_pass[True])}; B-n1 reading {vd(nll_pass[False])}.**\n")
P("Night 10, same two readings (reported; the midpoint uses B night 10):\n")
P("| seed | reading | night-1 ref | B n10 | midpoint | K-P n10 | <= midpoint |")
P("|---|---|---|---|---|---|---|")
nll10 = {}
for own in (True, False):
    r10 = nllbar(10, own); nll10[own] = all(x[0] for x in r10)
    for s, (ok, v, m, r1, b10) in zip(SEEDS, r10):
        P(f"| {s} | {'K-P own n1' if own else 'B n1'} | {r1:.3f} | {b10:.3f} | {m:.3f} | {v:.3f} | {ok} |")
P()
P(f"Night 10 direction (reported): KL below B {dict(zip(SEEDS, below10))}; NLL below B "
  f"{dict(zip(SEEDS, [lg('K-P', s, 10)['nll_on'] < lg('B', s, 10)['nll_on'] for s in SEEDS]))}; NLL <= midpoint 3/3: own-n1 {vd(nll10[True])}, B-n1 {vd(nll10[False])}.\n")
bar2 = {own: kl_pass and nll_pass[own] for own in (True, False)}
P(f"**Bar 2 (primary): own-n1 reading {vd(bar2[True])}; B-n1 reading {vd(bar2[False])}.** Reference (Stage 1 TEST B): NLL 1.93 -> 3.8-4.25, KL 13-17x night 1.\n")
for k in ("B", "K-P"):
    P(f"- {k} KL n15 / n1 per seed: " + ", ".join(f"s{s} {lg(k, s, 15)['kl']/lg(k, s, 1)['kl']:.2f}x" for s in SEEDS) + f" (life_bars LG-KL bar is n15 <= 2x n1).")
P()

# ---------------------------------------------------------------- bar 3
P("## 3. Guards (any fail = KILL)\n")
def hoff_series(res, lo=5, hi=15):
    return {p["night"]: p["summary"]["H_right"] / sum(1 for r in p["rows"] if r["kind"] == "H") * 6
            for p in trunk_probes(res, "hippocampus-off") if lo <= p["night"] <= hi}
P("### 3.1 H with the hippocampus off, nights 5-15 (bar: K-P mean >= B mean - 0.5 of 6, per seed)\n")
P("| seed | B mean /6 | K-P mean /6 | diff | ok |")
P("|---|---|---|---|---|")
hg = []
for s in SEEDS:
    mb = mean(hoff_series(R["B"][s]).values()); mk = mean(hoff_series(R["K-P"][s]).values())
    hg.append(mk - mb >= -0.5)
    P(f"| {s} | {mb:.3f} | {mk:.3f} | {mk-mb:+.3f} | {hg[-1]} |")
P(f"| pooled | {mean([mean(hoff_series(R['B'][s]).values()) for s in SEEDS]):.3f} | {mean([mean(hoff_series(R['K-P'][s]).values()) for s in SEEDS]):.3f} | | |")
P()
P("Per night (B / K-P, of 6):\n")
P("| seed | " + " | ".join(f"n{n}" for n in range(5, 16)) + " |")
P("|---|" + "---|" * 11)
for s in SEEDS:
    a, b = hoff_series(R["B"][s]), hoff_series(R["K-P"][s])
    P(f"| {s} | " + " | ".join(f"{a[n]:.0f}/{b[n]:.0f}" for n in range(5, 16)) + " |")
P()
P(f"**3.1 H-off: {vd(allp(hg))}** {dict(zip(SEEDS, hg))}\n")

P("### 3.2 Direct hippocampus-off recall of day-1-2 facts (bar: K-P >= B - 1 of 6, per seed)\n")
P("Definition as the D2-PR read: hippocampus-off trunk battery rows of kind H or rotating whose ledger item has min(told_days) <= 2, scored `ok`; rate x 6. "
  "The prereg names no window: the nights 5-15 window (as D2-PR) is the bar row; nights 1-4 and 1-15 shown beside.\n")
def d12(res, lo, hi, arm="hippocampus-off"):
    led, k, n = res["ledger"], 0, 0
    for p in trunk_probes(res, arm):
        if lo <= p["night"] <= hi:
            for r in p["rows"]:
                if r["kind"] in ("H", "rotating") and r.get("item") in led and min(led[r["item"]]["told_days"]) <= 2:
                    n += 1; k += bool(r.get("ok"))
    return k, n
P("| seed | window | B k/n (rate x6) | K-P k/n (rate x6) | diff /6 | ok |")
P("|---|---|---|---|---|---|")
dg = {}
for lo, hi in ((5, 15), (1, 4), (1, 15)):
    flags = []
    for s in SEEDS:
        kb, nb = d12(R["B"][s], lo, hi); kk, nk = d12(R["K-P"][s], lo, hi)
        rb, rk = kb / nb * 6, kk / nk * 6
        flags.append(rk - rb >= -1.0)
        P(f"| {s} | n{lo}-{hi} | {kb}/{nb} ({rb:.2f}) | {kk}/{nk} ({rk:.2f}) | {rk-rb:+.2f} | {flags[-1]} |")
    dg[(lo, hi)] = flags
P()
P(f"**3.2 direct d1-2 (nights 5-15): {vd(allp(dg[(5, 15)]))}** {dict(zip(SEEDS, dg[(5, 15)]))}; nights 1-4 would be {vd(allp(dg[(1, 4)]))} {dict(zip(SEEDS, dg[(1, 4)]))}; nights 1-15 {vd(allp(dg[(1, 15)]))}.\n")

P("### 3.3 Collapse (life_bars.language: seed collapses in an arm if most-repeated reply share >= 0.4 on any trunk probe night of that arm; bar: seeds-ever <= B per probe arm)\n")
LG = {k: life_bars.language([R[k][s] for s in SEEDS]) for k in ("B", "K-P")}
P("| probe arm | B seeds-ever (max share per seed 319, 320, 321) | K-P seeds-ever (max share) | K-P <= B |")
P("|---|---|---|---|")
cg = []
for arm in sorted(LG["B"]["collapse_arms"]):
    row = [arm]
    for k in ("B", "K-P"):
        c = LG[k]["collapse_arms"][arm]
        row.append(f"{c['seeds_ever']} (" + ", ".join("%.2f" % c["per_seed_max_share"][s] for s in SEEDS) + ")")
    ok = LG["K-P"]["collapse_arms"][arm]["seeds_ever"] <= LG["B"]["collapse_arms"][arm]["seeds_ever"]
    cg.append(ok); row.append(str(ok))
    P("| " + " | ".join(row) + " |")
P()
P("life_bars bars: " + "; ".join(f"{k}: " + ", ".join(f"{b['id']} {b['verdict']}" for b in LG[k]["bars"]) for k in LG) + "\n")
P(f"**3.3 collapse: {vd(allp(cg))}**\n")

P("### 3.4 Stock self-reply leak (life_leak, nights 10 and 15, intact + hippocampus-off pooled per seed; bar: K-P share <= B share per seed)\n")
leak = {k: {s: life_leak.pooled(life_leak.leak_table(R[k][s]), {10, 15}) for s in SEEDS} for k in ("B", "K-P")}
leakn = {k: {s: {n: life_leak.pooled(life_leak.leak_table(R[k][s]), {n}) for n in (10, 15)} for s in SEEDS} for k in ("B", "K-P")}
P("| arm | seed | leaks / n (n10+n15) | share | night 10 | night 15 |")
P("|---|---|---|---|---|---|")
for k in leak:
    for s in SEEDS:
        x = leak[k][s]
        P(f"| {k} | {s} | {x['leaks']}/{x['n']} | {f(x['share'])} | {leakn[k][s][10]['leaks']}/{leakn[k][s][10]['n']} | {leakn[k][s][15]['leaks']}/{leakn[k][s][15]['n']} |")
lk = [leak["K-P"][s]["share"] <= leak["B"][s]["share"] for s in SEEDS]
bl, bn = sum(leak["B"][s]["leaks"] for s in SEEDS), sum(leak["B"][s]["n"] for s in SEEDS)
kl_, kn = sum(leak["K-P"][s]["leaks"] for s in SEEDS), sum(leak["K-P"][s]["n"] for s in SEEDS)
P()
P(f"Pooled over seeds: K-P {kl_}/{kn} = {kl_/kn:.4f} vs B {bl}/{bn} = {bl/bn:.4f}. **3.4 leak: {vd(allp(lk))}** (K-P not higher than B per seed: {dict(zip(SEEDS, lk))}).\n")
P("### 3.5 Transcript guard (whole-transcript read, new defect class, sampled-bank phrases in replies)\n")
P("Computed elsewhere: **see blind read** (blind_KP). UNDEFINED in this script.\n")

# ---------------------------------------------------------------- bar 4
P("## 4. Secondary (report only)\n")
P("### 4.1 Adapter norm per night (B / K-P)\n")
P("| seed | " + " | ".join(f"n{n}" for n in range(1, 16)) + " |")
P("|---|" + "---|" * 15)
for s in SEEDS:
    nb = {n["night"]: n["adapter_norm"] for n in trunk(R["B"][s])}; nk = {n["night"]: n["adapter_norm"] for n in trunk(R["K-P"][s])}
    P(f"| {s} | " + " | ".join(f"{nb[n]:.0f}/{nk[n]:.0f}" for n in range(1, 16)) + " |")
P()
P("n15 / n1 ratio: " + "; ".join(f"s{s} B {nrec(R['B'][s], 15)['adapter_norm']/nrec(R['B'][s], 1)['adapter_norm']:.2f}x, K-P {nrec(R['K-P'][s], 15)['adapter_norm']/nrec(R['K-P'][s], 1)['adapter_norm']:.2f}x" for s in SEEDS) + "\n")

def ef(res, cls, arm, lo=1, hi=15):
    led, k, n = res["ledger"], 0, 0
    for p in trunk_probes(res, arm):
        if lo <= p["night"] <= hi:
            for r in p["rows"]:
                e = led.get(r.get("item") or "")
                if e and e["cls"] == cls and "ok" in r and r["kind"] in ("H", "rotating", "neardup"):
                    n += 1; k += bool(r["ok"])
    return k, n
P("### 4.2 Recall of once-told E and F facts (battery rows kind H/rotating/neardup, `ok`), per arm, nights 1-15 pooled per seed and per night pooled over seeds\n")
P("| seed | cls | arm | B k/n | K-P k/n |")
P("|---|---|---|---|---|")
for s in SEEDS:
    for c in ("E", "F"):
        for arm in ("intact", "hippocampus-off"):
            kb, nb = ef(R["B"][s], c, arm); kk, nk = ef(R["K-P"][s], c, arm)
            P(f"| {s} | {c} | {arm} | {kb}/{nb} | {kk}/{nk} |")
for c in ("E", "F"):
    for arm in ("intact", "hippocampus-off"):
        kb = sum(ef(R["B"][s], c, arm)[0] for s in SEEDS); nb = sum(ef(R["B"][s], c, arm)[1] for s in SEEDS)
        kk = sum(ef(R["K-P"][s], c, arm)[0] for s in SEEDS); nk = sum(ef(R["K-P"][s], c, arm)[1] for s in SEEDS)
        P(f"| pooled | {c} | {arm} | {kb}/{nb} ({kb/nb:.3f}) | {kk}/{nk} ({kk/nk:.3f}) |" if nb and nk else f"| pooled | {c} | {arm} | {kb}/{nb} | {kk}/{nk} |")
P()
P("Per night, pooled over seeds, E+F rows (B k/n -> K-P k/n), intact then hippocampus-off:\n")
P("| arm | " + " | ".join(f"n{n}" for n in range(1, 16)) + " |")
P("|---|" + "---|" * 15)
for arm in ("intact", "hippocampus-off"):
    cells = []
    for n in range(1, 16):
        kb = sum(ef(R["B"][s], c, arm, n, n)[0] for s in SEEDS for c in "EF"); nb = sum(ef(R["B"][s], c, arm, n, n)[1] for s in SEEDS for c in "EF")
        kk = sum(ef(R["K-P"][s], c, arm, n, n)[0] for s in SEEDS for c in "EF"); nk = sum(ef(R["K-P"][s], c, arm, n, n)[1] for s in SEEDS for c in "EF")
        cells.append(f"{kb}/{nb}->{kk}/{nk}")
    P(f"| {arm} | " + " | ".join(cells) + " |")
P()
P("Cue-ladder direct cue C0 rows (hippocampus-off, tracks E and F, all nights), `ok`:\n")
def cue0(res, track):
    k = n = 0
    for p in res.get("cue_probes", []):
        if p["arm"] == "hippocampus-off" and p.get("line", "trunk") == "trunk":
            for r in p["rows"]:
                if r["track"] == track and r["rung"] == "C0":
                    n += 1; k += bool(r["ok"])
    return k, n
P("| seed | track | B | K-P |")
P("|---|---|---|---|")
for s in SEEDS:
    for c in ("E", "F"):
        P(f"| {s} | {c} | {cue0(R['B'][s], c)[0]}/{cue0(R['B'][s], c)[1]} | {cue0(R['K-P'][s], c)[0]}/{cue0(R['K-P'][s], c)[1]} |")
for c in ("E", "F"):
    kb = sum(cue0(R["B"][s], c)[0] for s in SEEDS); nb = sum(cue0(R["B"][s], c)[1] for s in SEEDS)
    kk = sum(cue0(R["K-P"][s], c)[0] for s in SEEDS); nk = sum(cue0(R["K-P"][s], c)[1] for s in SEEDS)
    P(f"| pooled | {c} | {kb}/{nb} | {kk}/{nk} |")
P()

P("### 4.3 Items told in days 11-15 (week 3): intact and hippocampus-off battery H/rotating rows, nights 11-15\n")
P("Concern: the blind emitter could not plant wrong-episode/perspective canaries for K-P, which would mean no intact week-3 battery row answered a week-3-told item correctly. "
  "Week-3-told = ledger items with any told day >= 11 (and told on or before the probe night). Split: `new` = first told on day >= 11; `re-told` = first told earlier, told again on day 11 (the AB items, told_days (2, 11)). "
  "Rows are battery rows of kind H or rotating (and neardup shown separately), scored `ok`.\n")
def w3rows(res, arm, nlo, nhi, kinds):
    led = res["ledger"]; out = []
    for p in trunk_probes(res, arm):
        if nlo <= p["night"] <= nhi:
            for r in p["rows"]:
                e = led.get(r.get("item") or "")
                if e and r["kind"] in kinds and any(11 <= d <= p["night"] for d in e["told_days"]):
                    out.append((p["night"], r["item"], min(e["told_days"]) >= 11, r.get("ok")))
    return out
P("| seed | arm | kinds | window | B: new ok/n | B: re-told ok/n | K-P: new ok/n | K-P: re-told ok/n |")
P("|---|---|---|---|---|---|---|---|")
def cnt(rows, new):
    sel = [x for x in rows if x[2] == new]
    return f"{sum(1 for x in sel if x[3])}/{len(sel)}"
for s in SEEDS + ["pooled"]:
    for arm in ("intact", "hippocampus-off"):
        for kinds, kl in ((("H", "rotating"), "H+rot"), (("neardup",), "neardup")):
            for nlo, nhi in ((15, 15), (11, 15)):
                cells = []
                for k in ("B", "K-P"):
                    rows = [x for ss in (SEEDS if s == "pooled" else [s]) for x in w3rows(R[k][ss], arm, nlo, nhi, kinds)]
                    cells += [cnt(rows, True), cnt(rows, False)]
                P(f"| {s} | {arm} | {kl} | n{nlo}-{nhi} | " + " | ".join(cells) + " |")
P()
P("Row-level detail, intact arm, night 15, kind H/rotating (item, first-told>=11, ok):\n")
for k in ("B", "K-P"):
    for s in SEEDS:
        rows = w3rows(R[k][s], "intact", 15, 15, ("H", "rotating"))
        P(f"- {k} s{s}: " + (", ".join(f"{it}{'(new)' if nw else '(re-told)'}={ok}" for _, it, nw, ok in rows) or "no week-3 H/rotating rows at night 15"))
P()
P("Read: at night 15 there is no intact week-3 NEW-item H/rotating row in either arm (0 rows), and the only night-15 week-3 rows are re-told AB items "
  "(B 1/2 correct, K-P 0/2 correct, pooled intact). Over nights 11-15 intact K-P DID answer week-3 items correctly (new 3/4, re-told 2/8 pooled) against B (new 1/4, re-told 1/8). "
  "So the premise 'no intact week-3 battery row answered a week-3-told item correctly' does NOT hold for nights 11-15 and is vacuous at night 15 (no new-item rows, 0/2 re-told right for K-P).\n")
P("Same-day recall of week-3 items is a lived-turn (`sameday`) read, not part of the battery; not scored here.\n")

P("### 4.4 Interview AUROC and invented-memory rate\n")
P("UNDEFINED from the results JSON: `life_bars` does not report them. The interview AUROC and invented-memory rate come from `life_interview_read` over blind-reader "
  "claims (blind_KP / blind_S1F directories); those claims are not in the life JSONs and are not computed here. The battery's own never-item counters "
  "(never_false = a never-told item reported as known; never_invented) are shown instead, hippocampus-off and intact, summed over trunk nights 1-15:\n")
P("| seed | arm | probe arm | never rows | never_false | never_invented |")
P("|---|---|---|---|---|---|")
for s in SEEDS:
    for k in ("B", "K-P"):
        for arm in ("intact", "hippocampus-off"):
            ps = trunk_probes(R[k][s], arm)
            P(f"| {s} | {k} | {arm} | {sum(p['summary']['never_n'] for p in ps)} | {sum(p['summary']['never_false'] for p in ps)} | {sum(p['summary'].get('never_invented', 0) for p in ps)} |")
P()

P("### 4.5 First-person echo (life_echo, nights 10+15, echo counted on told-item battery rows not scored right)\n")
P("| seed | arm | probe arm | n | ok | echo | echo with key | echo share |")
P("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for k in ("B", "K-P"):
        t = life_echo.echo_table(R[k][s])
        for arm in life_echo.ARMS:
            p = life_echo.pooled(t, (10, 15), (arm,))
            P(f"| {s} | {k} | {arm} | {p['n']} | {p['ok']} | {p['echo']} | {p['echo_key']} | {f(p['share'])} |")
for k in ("B", "K-P"):
    for arm in life_echo.ARMS:
        ps = [life_echo.pooled(life_echo.echo_table(R[k][s]), (10, 15), (arm,)) for s in SEEDS]
        n, e = sum(p["n"] for p in ps), sum(p["echo"] for p in ps)
        P(f"| pooled | {k} | {arm} | {n} | {sum(p['ok'] for p in ps)} | {e} | {sum(p['echo_key'] for p in ps)} | {e/n:.3f} |")
P()

P("### 4.6 Night seconds (K-P CUDA fp16 vs B CPU fp32; the devices differ, so this is NOT a like-for-like cost)\n")
P("Night record `seconds` (sleep wall time as logged; K-P night 1 includes the bank sampling).\n")
P("| seed | arm | device | night 1 | mean n2-15 | min n2-15 | max n2-15 |")
P("|---|---|---|---|---|---|---|")
for s in SEEDS:
    for k in ("B", "K-P"):
        t = {n["night"]: n["seconds"] for n in trunk(R[k][s])}
        v = [t[n] for n in range(2, 16)]
        P(f"| {s} | {k} | {R[k][s]['device']} | {t[1]:.0f} | {mean(v):.0f} | {min(v):.0f} | {max(v):.0f} |")
P()
P("Sleep-log lines of the K-P night (one example, s321 night 2): " + "; ".join(nrec(R["K-P"][321], 2)["sleep_log"]) + "\n")

# ---------------------------------------------------------------- precision
P("## 5. Precision confound: how much does fp16 alone move lang_guard KL?\n")
P("No run isolates fp16 on the same method and seed. The available evidence is indirect; all of it is labelled.\n")
P("### 5.1 PR-x reference (a different method, B-like replay with a priority weight; s319 CPU fp32 on W, s320/321 CUDA fp16 on AWS) against B (CPU fp32)\n")
P("PR-l (all CPU fp32, pool) shown as the same-device control for the replay-priority change itself.\n")
P("| seed | arm | device | KL n1 | ratio to B n1 | KL n5 | ratio | KL n10 | ratio | KL n15 | ratio to B n15 | NLL_on n15 |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for k in ("B", "PR-l", "PR-x", "K-P"):
        cells = [str(s), k, R[k][s]["device"]]
        for n in (1, 5, 10, 15):
            v = lg(k, s, n)["kl"]
            cells += [f(v, 4), f(v / lg("B", s, n)["kl"], 3)]
        cells.append(f(lg(k, s, 15)["nll_on"], 3))
        P("| " + " | ".join(cells) + " |")
P()
rx = {s: lg("PR-x", s, 15)["kl"] / lg("B", s, 15)["kl"] for s in SEEDS}
rl = {s: lg("PR-l", s, 15)["kl"] / lg("B", s, 15)["kl"] for s in SEEDS}
P(f"PR-x KL n15 / B: s319 (CPU) {rx[319]:.3f}, s320 (fp16) {rx[320]:.3f}, s321 (fp16) {rx[321]:.3f}. "
  f"PR-l KL n15 / B (CPU, same device): " + ", ".join(f"s{s} {rl[s]:.3f}" for s in SEEDS) + ".")
P(f"Mean of the two PR-x fp16 seeds {mean([rx[320], rx[321]]):.3f} against the CPU seed {rx[319]:.3f}. Night-1 KL ratio to B for PR-x (n1 weights are near strength-only) "
  "is the cleanest early read of a device effect: " + ", ".join(f"s{s} {lg('PR-x', s, 1)['kl']/lg('B', s, 1)['kl']:.3f}" for s in SEEDS) +
  ". These ratios mix a method change (loss-weighted replay) and the device/precision change, with 1 life per cell; they bound but do not isolate the precision effect. "
  f"K-P/B at n15 is {kk15/bk15:.3f} pooled, against PR-x fp16/B of {mean([rx[320], rx[321]]):.3f}; so the K-P effect is "
  f"{'far larger than' if abs(1 - kk15/bk15) > 3 * abs(1 - mean([rx[320], rx[321]])) else 'comparable to'} the PR-x fp16 movement (by this crude comparison).\n")
P("### 5.2 CPU fp32 one-night gate replications (brain/experiments/results/2026-11-22-kp-gate-replications-1.5b.md)\n")
P("One extra night (night 16) from each seed's B night-15 home, B vs K-P, both CPU fp32 (s319 on W, s321 on pool40), `drift_read.py` held-out text. Values copied from that file:\n")
P("| seed | arm | held-out KL | held-out NLL | replayed-episode NLL |")
P("|---|---|---|---|---|")
for row in (("319", "before (n15)", 2.933, 4.747, 1.400), ("319", "B n16", 2.792, 4.625, 0.977), ("319", "K-P n16", 0.762, 2.707, 0.444),
            ("321", "before (n15)", 2.858, 4.674, 0.609), ("321", "B n16", 2.760, 4.507, 0.546), ("321", "K-P n16", 0.813, 2.737, 0.500),
            ("316 (gate)", "B (one night)", 2.80, 4.59, 0.770), ("316 (gate)", "K-P (one night)", 0.86, 2.81, 0.451)):
    P("| " + " | ".join(str(x) for x in row) + " |")
P()
P("What this shows: on CPU fp32 (no precision difference), one K-P night takes held-out KL from about 2.9 to about 0.8 against B's 2.8, on 3/3 seeds. "
  "So the K-P effect on KL is reproduced without a device change. What it does NOT show: whether the 15-night KL in the AWS fp16 K-P lives matches what a CPU fp32 K-P life would "
  "give. These are one-night repairs from a drifted home, not a 15-night accumulation. The fp16-alone effect is not measured; the only same-method cross-device pair is PR-x (5.1).\n")
# ---------------------------------------------------------------- summary
P("## 6. Bar summary and prereg section 4 decision\n")
nonT = ["PASS" if allp(hg) else "FAIL", "PASS" if allp(dg[(5, 15)]) else "FAIL", "PASS" if allp(cg) else "FAIL", "PASS" if allp(lk) else "FAIL"]
P("| bar | verdict | per-seed / values |")
P("|---|---|---|")
P(f"| 1a engagement (logs) | **{vd(allp(e1a))}** | {dict(zip(SEEDS, e1a))} |")
P(f"| 1b bank quality | **s321 {vd(e1b[321])}; s319 {vd(e1b[319])} (PROXY); s320 UNDEFINED** | s321 own bank; s319 CPU proxy; s320 not pulled |")
P(f"| 1 engagement overall | **{BAR1}** | |")
P(f"| 2 KL n15 (<= half of B pooled, below B 3/3) | **{vd(kl_pass)}** | pooled {kk15:.4f} vs B {bk15:.4f} (ratio {kk15/bk15:.3f}); below B {dict(zip(SEEDS, below15))} |")
P(f"| 2 NLL n15 <= midpoint, K-P own n1 | **{vd(nll_pass[True])}** | " + "; ".join(f"s{s} {x[1]:.3f} vs {x[2]:.3f}" for s, x in zip(SEEDS, nll_res[("own", True)])) + " |")
P(f"| 2 NLL n15 <= midpoint, B n1 | **{vd(nll_pass[False])}** | " + "; ".join(f"s{s} {x[1]:.3f} vs {x[2]:.3f}" for s, x in zip(SEEDS, nll_res[("own", False)])) + " |")
P(f"| 2 night 10 direction (reported) | KL below B {dict(zip(SEEDS, below10))}; NLL midpoint own-n1 {vd(nll10[True])}, B-n1 {vd(nll10[False])} | |")
P(f"| 3.1 H-off n5-15 | **{vd(allp(hg))}** | {dict(zip(SEEDS, hg))} |")
P(f"| 3.2 direct d1-2 hippocampus-off (n5-15) | **{vd(allp(dg[(5, 15)]))}** | {dict(zip(SEEDS, dg[(5, 15)]))}; n1-4 {vd(allp(dg[(1, 4)]))} |")
P(f"| 3.3 collapse | **{vd(allp(cg))}** | seeds-ever per probe arm <= B |")
P(f"| 3.4 leak n10+15 | **{vd(allp(lk))}** | {dict(zip(SEEDS, lk))} |")
P("| 3.5 transcript guard | **UNDEFINED here (see blind read)** | |")
P()
numeric_fail = "FAIL" in nonT
P("### Prereg section 4 decision, conditional on the transcript guard\n")
if numeric_fail:
    dec = "KILLED (a numeric guard fails; the transcript guard cannot rescue it)"
elif bar1 is False:
    dec = "KILLED/UNDEFINED per section 4 (engagement failed)"
else:
    if kl_pass and (nll_pass[True] or nll_pass[False]):
        core = "bars 2 and numeric 3 hold"
        dec = ("KEEP if the transcript guard also finds no new defect class and bar 1 is accepted (1b is UNDEFINED for s320 and proxy-only for s319, "
               "so section 3.1 formally leaves the arm UNDEFINED); KILLED if the transcript read finds a new defect class")
    elif kl_pass:
        dec = "NARROWED (KL met, NLL not), conditional on the transcript guard; KILLED if the transcript read finds a new defect class"
    else:
        dec = "NARROWED or lead, conditional on the transcript guard"
P(f"- Numeric guards (3.1-3.4): {'at least one FAIL' if numeric_fail else 'all PASS'}. KL bar {vd(kl_pass)}; NLL bar own-n1 {vd(nll_pass[True])}, B-n1 {vd(nll_pass[False])}.")
P(f"- Bar 1 (engagement): {BAR1}.")
P(f"- Window sensitivity of guard 3.2 (the prereg names no window): nights 5-15 PASS; nights 1-4 would FAIL on s320 (B 5.59 vs K-P 4.34 of 6, diff -1.24, bar -1.00); nights 1-15 PASS. The KILL guard is therefore window-dependent for s320 and the nights-1-4 reading would KILL; the as-D2-PR reading does not.")
P(f"- Decision by these numbers: **{dec}**.")
P("- The default flips only on a registered six-seed TEST on a FRESH TEST split (prereg section 4; land-result). The device/precision confound (K-P fp16 vs B fp32) and the missing s319/s320 life banks remain caveats on any KEEP.")
open(os.path.join(HERE, "kp_bars.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
