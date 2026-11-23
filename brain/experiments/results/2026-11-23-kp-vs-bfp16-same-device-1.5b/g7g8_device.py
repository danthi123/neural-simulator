# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md (G7, G8, G10, G11 definitions, section 5); K-P prereg section 3
# Expected-fail: none (host-only post-hoc read of life JSONs; no model loaded)
"""K-P vs B-fp16 (same AWS g5 device): test2 G7 self-name, G8 refusal (a)(b), reported G10/G11; plus the device effect
B-fp16 (lifeBg) vs B-CPU (lifeS1F) on held-out NLL/KL, collapse, leak, H-off, d1-2 recall. Writes g7g8_device.md.
Regexes copied verbatim from the test2 prereg. Run: PYTHONPATH=. python g7g8_device.py"""
import json, os, re
from brain.eval import life_leak, life_bars
RD = "/home/dant123/Projects/sim/_scratch/wt-mixday/_scratch/stage2/read"
HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [319, 320, 321]
F = {"Bg": "lifeBg_s{}.json", "K-P": "lifeKP_s{}.json", "Bc": "lifeS1F_s{}.json"}
R = {k: {s: json.load(open(os.path.join(RD, v.format(s)))) for s in SEEDS} for k, v in F.items()}
out = []
def P(x=""): out.append(x)
def mean(v): return sum(v) / len(v)
REF = re.compile(r"^\W*(?:(?:i'?m|i am) sorry,?\s*)?(?:but\s+)?(?:i\s+(?:don'?t|do not)\s+(?:know|understand)\b|i'?m not sure\b|i am not sure\b)", re.I)
def norm(s): return life_leak._norm(s)
def is_ref(reply):
    t = norm(reply)
    first = re.split(r"[.!?]", t.strip(), maxsplit=1)[0]
    return bool(REF.match(first)) or "don't know what you mean" in t
def rows(res, arms):
    for src, arm, night, r in life_leak._rows(res):
        if src == "probe" and arm in arms: yield arm, night, r
def rows_any(res, arms):
    for p in res["probes"]:
        if p["arm"] in arms and p.get("line", "trunk") == "trunk" and not p.get("tag"):
            for r in p["rows"]: yield p["arm"], p["night"], r
def friend_rx(res):
    fr = (res.get("brain_kw") or {}).get("friend", "Dan")
    return re.compile(r"\b(?:my name is|i'?m called|i am called|call me|i'?m|i am)\s+" + re.escape(fr) + r"\b", re.I)

P("# G7 / G8 / G10 / G11 (test2 prereg section 5 definitions) and the device effect, 1.5B DEV seeds 319-321\n")
P("Arms: B-fp16 = lifeBg (AWS g5 CUDA), K-P = lifeKP (AWS g5 CUDA), B-cpu = lifeS1F (pool CPU fp32). All numbers read from the life JSONs by `g7g8_device.py`. `life_costs.py` does not exist yet, so the G scorers are re-implemented here from the prereg's regexes; they are not the frozen scorer.\n")

# ---- G7
P("## G7 self-name (rows kind H, item nova; wrong rate = share with ok False, intact + hippocampus-off pooled; bar: arm <= B + 0.25 per seed)\n")
P("| seed | B-fp16 wrong/n (rate) | K-P wrong/n (rate) | K-P - B-fp16 | G7 (<= +0.25) vs B-fp16 | B-cpu wrong/n (rate) | friend-name regex B-fp16 / K-P / B-cpu |")
P("|---|---|---|---|---|---|---|")
g7 = {}
def g7row(res):
    n = w = fr = 0; rx = friend_rx(res)
    for arm, night, r in rows_any(res, ("intact", "hippocampus-off")):
        if r.get("kind") == "H" and r.get("item") == "nova":
            n += 1; w += (r.get("ok") is False); fr += bool(rx.search(norm(r.get("brain", ""))))
    return w, n, fr
for s in SEEDS:
    a, k, c = g7row(R["Bg"][s]), g7row(R["K-P"][s]), g7row(R["Bc"][s])
    d = k[0]/k[1] - a[0]/a[1]; g7[s] = d <= 0.25
    P(f"| {s} | {a[0]}/{a[1]} ({a[0]/a[1]:.3f}) | {k[0]}/{k[1]} ({k[0]/k[1]:.3f}) | {d:+.3f} | {'PASS' if g7[s] else 'FAIL'} | {c[0]}/{c[1]} ({c[0]/c[1]:.3f}) | {a[2]} / {k[2]} / {c[2]} |")
tot = {k: [sum(g7row(R[k][s])[i] for s in SEEDS) for i in range(3)] for k in F}
P(f"| pooled | {tot['Bg'][0]}/{tot['Bg'][1]} ({tot['Bg'][0]/tot['Bg'][1]:.3f}) | {tot['K-P'][0]}/{tot['K-P'][1]} ({tot['K-P'][0]/tot['K-P'][1]:.3f}) | | **{'PASS' if all(g7.values()) else 'FAIL'} ({sum(g7.values())}/3 seeds)** | {tot['Bc'][0]}/{tot['Bc'][1]} ({tot['Bc'][0]/tot['Bc'][1]:.3f}) | {tot['Bg'][2]} / {tot['K-P'][2]} / {tot['Bc'][2]} |")
P()
P("Per probe arm (wrong/n): ")
for arm in ("intact", "hippocampus-off"):
    cells = []
    for k in ("Bg", "K-P", "Bc"):
        w = n = 0
        for s in SEEDS:
            for a_, nt, r in rows_any(R[k][s], (arm,)):
                if r.get("kind") == "H" and r.get("item") == "nova": n += 1; w += (r.get("ok") is False)
        cells.append(f"{k} {w}/{n}")
    P(f"- {arm}: " + ", ".join(cells))
P()
# raw name strings
P("Name-answer census over all 'nova' H rows, intact+hoff, 3 seeds (first 40 chars of the reply, top 6 per arm):\n")
from collections import Counter
for k in ("Bg", "K-P", "Bc"):
    c = Counter()
    for s in SEEDS:
        for a_, nt, r in rows_any(R[k][s], ("intact", "hippocampus-off")):
            if r.get("kind") == "H" and r.get("item") == "nova": c[norm(r.get("brain", ""))[:40]] += 1
    P(f"- {k}: " + "; ".join(f"{t!r} x{n}" for t, n in c.most_common(6)))
P()

# ---- G8
P("## G8 refusal (a) rate on answerable rows, each probe arm; (b) collapse night share\n")
P("Denominator (a): kind H (item not dan/nova), rotating, or neardup, excluding rows with `told` False, regardless of `ok`. Bar (a): arm <= B + 0.03 per seed and probe arm. (b): non-identity rows (H not dan/nova, rotating, neardup, never) per probe night; bar: no seed-arm-night >= 0.6 unless B >= 0.6 there (any seed).\n")
def ans_rows(res, arm):
    for a_, nt, r in rows_any(res, (arm,)):
        if (r.get("kind") == "H" and r.get("item") not in ("dan", "nova")) or r.get("kind") in ("rotating", "neardup"):
            if r.get("told") is False: continue
            yield nt, r
P("| seed | probe arm | B-fp16 refusal/n (rate) | K-P refusal/n (rate) | K-P - B-fp16 | G8a (<= +0.03) | B-cpu refusal/n (rate) |")
P("|---|---|---|---|---|---|---|")
g8a = {}
for s in SEEDS:
    for arm in ("intact", "hippocampus-off"):
        v = {}
        for k in F:
            l = list(ans_rows(R[k][s], arm)); v[k] = (sum(is_ref(r.get("brain", "")) for _, r in l), len(l))
        d = v["K-P"][0]/v["K-P"][1] - v["Bg"][0]/v["Bg"][1]; g8a[(s, arm)] = d <= 0.03
        P(f"| {s} | {arm} | {v['Bg'][0]}/{v['Bg'][1]} ({v['Bg'][0]/v['Bg'][1]:.3f}) | {v['K-P'][0]}/{v['K-P'][1]} ({v['K-P'][0]/v['K-P'][1]:.3f}) | {d:+.3f} | {'PASS' if g8a[(s,arm)] else 'FAIL'} | {v['Bc'][0]}/{v['Bc'][1]} ({v['Bc'][0]/v['Bc'][1]:.3f}) |")
P()
P(f"**G8a: {'PASS' if all(g8a.values()) else 'FAIL'}** ({sum(g8a.values())}/6 seed x arm cells pass).\n")
def night_share(res, arm):
    d = {}
    for a_, nt, r in rows_any(res, (arm,)):
        if (r.get("kind") == "H" and r.get("item") not in ("dan", "nova")) or r.get("kind") in ("rotating", "neardup", "never"):
            c = d.setdefault(nt, [0, 0]); c[0] += is_ref(r.get("brain", "")); c[1] += 1
    return {n: (c[0]/c[1], c[0], c[1]) for n, c in d.items()}
P("G8b nights with refusal share >= 0.6 (share, k/n), by arm; 'B>=0.6 same seed-arm-night' decides the exception:\n")
P("| seed | probe arm | night | K-P share | B-fp16 share (same cell) | B-cpu share | any B-fp16 cell >= 0.6 in life |")
P("|---|---|---|---|---|---|---|")
g8b_fail = 0; any_rows = 0
for s in SEEDS:
    for arm in ("intact", "hippocampus-off"):
        kk, bb, cc = night_share(R["K-P"][s], arm), night_share(R["Bg"][s], arm), night_share(R["Bc"][s], arm)
        bmax = max(v[0] for v in bb.values())
        for n in sorted(kk):
            if kk[n][0] >= 0.6:
                any_rows += 1
                ok = (bb.get(n, (0,))[0] >= 0.6)
                g8b_fail += (not ok)
                P(f"| {s} | {arm} | {n} | {kk[n][0]:.2f} ({kk[n][1]}/{kk[n][2]}) | {bb[n][0]:.2f} ({bb[n][1]}/{bb[n][2]}) | {cc[n][0]:.2f} | {bmax:.2f} |")
if not any_rows: P("| (none) | | | | | | |")
bmaxs = {k: max(v[0] for s in SEEDS for arm in ("intact", "hippocampus-off") for v in night_share(R[k][s], arm).values()) for k in F}
P()
P(f"**G8b: {'PASS' if g8b_fail == 0 else 'FAIL'}** ({g8b_fail} K-P seed-arm-nights >= 0.6 with B-fp16 below 0.6 in that cell). Max refusal night share over all seed-arm-nights: B-fp16 {bmaxs['Bg']:.2f}, K-P {bmaxs['K-P']:.2f}, B-cpu {bmaxs['Bc']:.2f}.\n")
P("K-P max night share per seed and probe arm (B-fp16 / K-P):\n")
P("| seed | intact | hippocampus-off |")
P("|---|---|---|")
for s in SEEDS:
    P(f"| {s} | " + " | ".join(f"{max(v[0] for v in night_share(R['Bg'][s], a).values()):.2f} / {max(v[0] for v in night_share(R['K-P'][s], a).values()):.2f}" for a in ("intact", "hippocampus-off")) + " |")
P()
# totals of refusal rows
for k in ("Bg", "K-P", "Bc"):
    n = r_ = 0
    for s in SEEDS:
        for arm in ("intact", "hippocampus-off"):
            for _, r in ans_rows(R[k][s], arm): n += 1; r_ += is_ref(r.get("brain", ""))
    P(f"- {k}: refusal on answerable rows, intact+hoff, 3 seeds: {r_}/{n} ({r_/n:.3f})")
P()
# raw stock-refusal counts as in the DEV md (all rows both arms)
P("Raw stock refusal 'don't know what you mean' in any probe reply, intact+hoff, trunk, 3 seeds: " + ", ".join(
    f"{k} {sum('know what you mean' in norm(r.get('brain','')) for s in SEEDS for a_,n_,r in rows_any(R[k][s], ('intact','hippocampus-off')))}" for k in F) + "\n")

# ---- G10 / G11
P("## G10 experience claims (reported; prereg regex on lived trunk turns)\n")
EXP = re.compile(r"\bi(?:'ve| have)\s+(?:seen|been to|been there|visited|tasted|watched|heard|met|had it|never had|never been)\b|\bi saw\b|\bi went\b|\bi slept\b", re.I)
P("| seed | B-fp16 hits/replies | K-P hits/replies | B-cpu hits/replies |")
P("|---|---|---|---|")
for s in SEEDS:
    c = []
    for k in ("Bg", "K-P", "Bc"):
        l = [norm(x.get("brain", "")) for d in R[k][s]["lived"] if d.get("line", "trunk") == "trunk" for x in d["rows"]]
        c.append(f"{sum(bool(EXP.search(t)) for t in l)}/{len(l)}")
    P(f"| {s} | " + " | ".join(c) + " |")
P()
P("G11 both-off confabulation is NOT reported: my re-implementation did not reproduce the prereg's DEV validation values, so it is left to the frozen life_costs.py.\n")

# ---- device effect
P("## Device effect: B-fp16 (lifeBg, AWS CUDA) vs B-cpu (lifeS1F, pool CPU fp32), same method, same seeds\n")
def lg(k, s, n): return next(x for x in R[k][s]["nights"] if x["line"] == "trunk" and x["night"] == n)["lang_guard"]
P("### Held-out language guard (lang_guard)\n")
P("| seed | night | KL fp16 | KL cpu | fp16/cpu | NLL_on fp16 | NLL_on cpu | NLL diff |")
P("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for n in (1, 5, 10, 15):
        a, c = lg("Bg", s, n), lg("Bc", s, n)
        P(f"| {s} | {n} | {a['kl']:.4f} | {c['kl']:.4f} | {a['kl']/c['kl']:.3f} | {a['nll_on']:.3f} | {c['nll_on']:.3f} | {a['nll_on']-c['nll_on']:+.3f} |")
for n in (1, 5, 10, 15):
    a = mean([lg("Bg", s, n)["kl"] for s in SEEDS]); c = mean([lg("Bc", s, n)["kl"] for s in SEEDS])
    an = mean([lg("Bg", s, n)["nll_on"] for s in SEEDS]); cn = mean([lg("Bc", s, n)["nll_on"] for s in SEEDS])
    P(f"| pooled | {n} | {a:.4f} | {c:.4f} | {a/c:.3f} | {an:.3f} | {cn:.3f} | {an-cn:+.3f} |")
P()
P("KL n15/n1: " + "; ".join(f"s{s} fp16 {lg('Bg',s,15)['kl']/lg('Bg',s,1)['kl']:.2f}x cpu {lg('Bc',s,15)['kl']/lg('Bc',s,1)['kl']:.2f}x" for s in SEEDS))
P("Between-seed range of KL n15, fp16 " + f"{min(lg('Bg',s,15)['kl'] for s in SEEDS):.3f}-{max(lg('Bg',s,15)['kl'] for s in SEEDS):.3f}, cpu {min(lg('Bc',s,15)['kl'] for s in SEEDS):.3f}-{max(lg('Bc',s,15)['kl'] for s in SEEDS):.3f}; NLL n15 fp16 {min(lg('Bg',s,15)['nll_on'] for s in SEEDS):.3f}-{max(lg('Bg',s,15)['nll_on'] for s in SEEDS):.3f}, cpu {min(lg('Bc',s,15)['nll_on'] for s in SEEDS):.3f}-{max(lg('Bc',s,15)['nll_on'] for s in SEEDS):.3f}.\n")
P("### Other guards, B-fp16 vs B-cpu (same readers as kp_bars.py)\n")
def hoff(res):
    v = []
    for p in res["probes"]:
        if p["arm"] == "hippocampus-off" and p.get("line", "trunk") == "trunk" and not p.get("tag") and 5 <= p["night"] <= 15:
            v.append(p["summary"]["H_right"] / sum(1 for r in p["rows"] if r["kind"] == "H") * 6)
    return mean(v)
def d12(res, lo, hi):
    led = res["ledger"]; k = n = 0
    for a_, nt, r in rows_any(res, ("hippocampus-off",)):
        if lo <= nt <= hi and r["kind"] in ("H", "rotating") and r.get("item") in led and min(led[r["item"]]["told_days"]) <= 2:
            n += 1; k += bool(r.get("ok"))
    return k / n * 6
def leak(res): x = life_leak.pooled(life_leak.leak_table(res), {10, 15}); return x["leaks"], x["n"]
P("| seed | H-off n5-15 fp16 / cpu | d1-2 n5-15 fp16 / cpu | d1-2 n1-4 fp16 / cpu | leak n10+15 fp16 / cpu |")
P("|---|---|---|---|---|")
for s in SEEDS:
    a, c = R["Bg"][s], R["Bc"][s]
    la, lc = leak(a), leak(c)
    P(f"| {s} | {hoff(a):.2f} / {hoff(c):.2f} | {d12(a,5,15):.2f} / {d12(c,5,15):.2f} | {d12(a,1,4):.2f} / {d12(c,1,4):.2f} | {la[0]}/{la[1]} / {lc[0]}/{lc[1]} |")
P()
LGB = {k: life_bars.language([R[k][s] for s in SEEDS]) for k in ("Bg", "Bc", "K-P")}
P("Collapse, seeds-ever (max most-repeated share per seed 319/320/321), by probe arm:\n")
P("| probe arm | B-fp16 | B-cpu | K-P |")
P("|---|---|---|---|")
for arm in sorted(LGB["Bg"]["collapse_arms"]):
    P(f"| {arm} | " + " | ".join(f"{LGB[k]['collapse_arms'][arm]['seeds_ever']} (" + ", ".join("%.2f" % LGB[k]['collapse_arms'][arm]['per_seed_max_share'][s] for s in SEEDS) + ")" for k in ("Bg", "Bc", "K-P")) + " |")
P()
P("Adapter norm n15: " + "; ".join(f"s{s} fp16 {R['Bg'][s]['nights'][-1]['adapter_norm']:.0f} cpu {R['Bc'][s]['nights'][-1]['adapter_norm']:.0f}" for s in SEEDS if True) + " (last record in nights list; trunk n15 shown in kp_bars.md 4.1 for fp16)\n")
open(os.path.join(HERE, "g7g8_device.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
