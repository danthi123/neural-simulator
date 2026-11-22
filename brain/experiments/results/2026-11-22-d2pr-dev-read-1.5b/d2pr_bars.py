# Capability: Lifelong learning
# Record: brain/research/2026-11-21-replay-priority-prereg.md (section 3 bars; read script written by a sonnet agent 11-22)
# Expected-fail: none (host-only post-hoc read of life JSONs)
"""D2-PR DEV read: prereg 2026-11-21-replay-priority-prereg.md section 3, arms PR-l and PR-x against B (lifeS1F).
Every number is read from the life JSONs by existing brain.eval code; nothing estimated.
Run: PYTHONPATH=<worktree> python d2pr_bars.py  (cwd = this folder)"""
import json, os
from brain.eval import life_leak, life_echo, life_table, life_bars

SEEDS = [319, 320, 321]
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {"B": "lifeS1F_s{}.json", "PR-l": "lifePRl_s{}.json", "PR-x": "lifePRx_s{}.json"}
ARMS = ["PR-l", "PR-x"]
R = {k: {s: json.load(open(os.path.join(HERE, v.format(s)))) for s in SEEDS} for k, v in FILES.items()}
out = []
def P(x=""): out.append(x)
def f(x, k=3): return "-" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))
def mean(v): v = [x for x in v if x is not None]; return sum(v) / len(v) if v else None
def vd(ok): return "UNDEFINED" if ok is None else ("PASS" if ok else "FAIL")
def allp(l): return None if any(x is None for x in l) else all(l)
def nrec(res, n): return next((x for x in res["nights"] if x["night"] == n and x["line"] == "trunk"), None)
def trunk_probes(res, arm):
    return [p for p in res["probes"] if p["arm"] == arm and p.get("line", "trunk") == "trunk" and not p.get("tag")]

P("# D2-PR DEV read (prereg 2026-11-21-replay-priority-prereg.md, section 3)")
P("Seeds 319/320/321 (DEV). B = lifeS1F, arms PR-l (`loss`) and PR-x (`strength_x_loss`). All values read from the JSONs by "
  "`d2pr_bars.py` (brain.eval.life_leak, life_echo, life_table, life_bars). Bars are evaluated per seed and pooled as the prereg states.\n")
summary = {a: {} for a in ARMS}

# ---------------------------------------------------------------- 3.1
P("## 3.1 Engagement (nights 5 and 10; permanent share and top-6 share lower than B on 3/3 seeds)\n")
P("**Comparison used.** B's night records have no `replay_priority` block (no per-episode weights), and its `replay_share` lumps "
  "the permanent exchanges into class `other`, so B's permanent share and top-6 share cannot be read from the B JSON. Each PR night "
  "record carries `strength_permanent_share` / `strength_top6_share`: the strength-only weighting (exactly B's rule) computed on the SAME "
  "night store, inside the same night. That within-night reference is what B would have replayed from that state, so the paired comparison is "
  "actual (PR weighting) vs that reference. It is exact only to the degree the PR life's store at that night equals B's: "
  "check below that the B `replay_share` class values equal the PR record's (identical = same store and same strengths).\n")
P("| arm | seed | night | permanent share | strength-only ref | top-6 share | strength-only ref | max share | ref | lower perm? | lower top6? |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
eng = {a: [] for a in ARMS}
for a in ARMS:
    for s in SEEDS:
        for n in (5, 10):
            rp = nrec(R[a][s], n).get("replay_priority")
            if not rp:
                P(f"| {a} | {s} | {n} | UNDEFINED (no replay_priority) | | | | | | | |"); eng[a].append(None); continue
            lp, lt = rp["permanent_share"] < rp["strength_permanent_share"], rp["top6_share"] < rp["strength_top6_share"]
            eng[a].append(lp and lt)
            P(f"| {a} | {s} | {n} | {f(rp['permanent_share'])} | {f(rp['strength_permanent_share'])} | {f(rp['top6_share'])} | "
              f"{f(rp['strength_top6_share'])} | {f(rp['max_share'])} | {f(rp['strength_max_share'])} | {lp} | {lt} |")
P()
P("Reference sanity (does the strength-only reference match across the two arms, and B's class replay_share match the PR record's?):\n")
P("| seed | night | ref perm PR-l / PR-x | ref top6 PR-l / PR-x | B replay_share == PR-l record | == PR-x record |")
P("|---|---|---|---|---|---|")
for s in SEEDS:
    for n in (5, 10):
        l, x = nrec(R["PR-l"][s], n), nrec(R["PR-x"][s], n)
        b = nrec(R["B"][s], n)
        P(f"| {s} | {n} | {f(l['replay_priority']['strength_permanent_share'],4)} / {f(x['replay_priority']['strength_permanent_share'],4)} | "
          f"{f(l['replay_priority']['strength_top6_share'],4)} / {f(x['replay_priority']['strength_top6_share'],4)} | "
          f"{b['replay_share'] == l['replay_share']} | {b['replay_share'] == x['replay_share']} |")
P()
P("Class-level cross-check against B (B `replay_share` normalised by life_table.replay_class_shares vs the PR arm's actual `shares_by_class`; "
  "B has no permanent class, it sits inside `other`):\n")
P("| arm | seed | night | class: B share -> arm share |")
P("|---|---|---|---|")
for a in ARMS:
    for s in SEEDS:
        bs = life_table.replay_class_shares(R["B"][s])
        for n in (5, 10):
            sh = nrec(R[a][s], n)["replay_priority"]["shares_by_class"]
            parts = [f"{c} {bs[n]['shares'].get(c, 0):.3f}->{sh.get(c, 0):.3f}" for c in sorted(bs[n]["shares"])]
            P(f"| {a} | {s} | {n} | " + "; ".join(parts) + f"; (arm permanent {sh.get('permanent', 0):.3f}) |")
P()
for a in ARMS:
    v = allp(eng[a])
    summary[a]["3.1 engagement"] = ("PASS" if v else "FAIL, arm UNDEFINED per prereg 3.1", "perm and top-6 below strength-only ref on " + f"{sum(1 for x in eng[a] if x)}/{len(eng[a])} seed-nights")
    P(f"**3.1 {a}: {'PASS' if v else 'FAIL, so the arm is UNDEFINED per prereg 3.1'}** ({sum(1 for x in eng[a] if x)}/{len(eng[a])} seed-nights both shares lower; the bar needs 3/3 seeds on each of nights 5 and 10).")
P()

# ---------------------------------------------------------------- 3.2
P("## 3.2 Primary: stock self-reply leak (life_leak, nights 10 and 15, intact + hippocampus-off pooled per seed)\n")
leak = {k: {s: life_leak.pooled(life_leak.leak_table(R[k][s]), {10, 15}) for s in SEEDS} for k in R}
leakn = {k: {s: {n: life_leak.pooled(life_leak.leak_table(R[k][s]), {n}) for n in (10, 15)} for s in SEEDS} for k in R}
P("| arm | seed | leaks / n (nights 10+15) | share | night 10 | night 15 |")
P("|---|---|---|---|---|---|")
for k in R:
    for s in SEEDS:
        x = leak[k][s]
        P(f"| {k} | {s} | {x['leaks']}/{x['n']} | {f(x['share'])} | {leakn[k][s][10]['leaks']}/{leakn[k][s][10]['n']} | {leakn[k][s][15]['leaks']}/{leakn[k][s][15]['n']} |")
P()
bl, bn = sum(leak["B"][s]["leaks"] for s in SEEDS), sum(leak["B"][s]["n"] for s in SEEDS)
for a in ARMS:
    al, an = sum(leak[a][s]["leaks"] for s in SEEDS), sum(leak[a][s]["n"] for s in SEEDS)
    below = [leak[a][s]["share"] < leak["B"][s]["share"] for s in SEEDS]
    red = 1 - (al / an) / (bl / bn)
    ok = all(below) and red >= 0.30
    summary[a]["3.2 leak"] = (vd(ok), f"below B on {sum(below)}/3 seeds; pooled {al}/{an}={al/an:.3f} vs B {bl}/{bn}={bl/bn:.3f}, reduction {red*100:.1f}% (bar >=30%)")
    P(f"**3.2 {a}: {vd(ok)}**. Below B per seed: {dict(zip(SEEDS, below))}. Pooled {al}/{an} = {al/an:.4f} vs B {bl}/{bn} = {bl/bn:.4f}; "
      f"relative reduction {red*100:.1f}% (needs >= 30%).")
P()

# ---------------------------------------------------------------- 3.3
P("## 3.3 Guards (any fail = KILL)\n")
def hoff_series(res, arm="hippocampus-off", lo=5, hi=15):
    return {p["night"]: p["summary"]["H_right"] / sum(1 for r in p["rows"] if r["kind"] == "H") * 6
            for p in trunk_probes(res, arm) if lo <= p["night"] <= hi}
P("### Guard H: hippocampus-off H recall, nights 5-15\n")
P("Scoring: per hippocampus-off trunk probe night, `summary.H_right` (count of kind-H rows with `ok` true) out of the night's H rows (6), "
  "so the value is already 'of 6'; the per-seed value is the mean over probe nights 5-15 (nights 5-15 that have a hippocampus-off probe, all 11). "
  "Bar: arm mean >= B mean - 0.5 on every seed.\n")
P("| seed | B mean /6 | PR-l mean /6 | diff | PR-x mean /6 | diff |")
P("|---|---|---|---|---|---|")
hg = {a: [] for a in ARMS}
for s in SEEDS:
    sb = hoff_series(R["B"][s]); mb = mean(sb.values())
    row = [str(s), f(mb)]
    for a in ARMS:
        sa = hoff_series(R[a][s]); ma = mean(sa.values())
        hg[a].append(ma - mb >= -0.5)
        row += [f(ma), f"{ma-mb:+.3f}"]
    P("| " + " | ".join(row) + " |")
P()
P("Per night (B / PR-l / PR-x, of 6):\n")
P("| seed | " + " | ".join(f"n{n}" for n in range(5, 16)) + " |")
P("|---|" + "---|" * 11)
for s in SEEDS:
    ser = {k: hoff_series(R[k][s]) for k in R}
    P(f"| {s} | " + " | ".join(f"{ser['B'][n]:.0f}/{ser['PR-l'][n]:.0f}/{ser['PR-x'][n]:.0f}" for n in range(5, 16)) + " |")
P()
P("### Guard direct: hippocampus-off recall of day-1-2 facts\n")
P("Definition (as life_table.extra_rows): hippocampus-off trunk battery rows of kind H or rotating whose ledger item has min(told_days) <= 2 "
  "(day-1-2 facts; all six H items are told on day 1, the dan/nova facts have no telling but are H rows), scored with `ok`. "
  "Rate pooled over probe nights 5-15 per seed, converted to /6 as rate x 6. Bar: arm >= B - 1 (of 6) on every seed. Nights 1-15 shown beside.\n")
def d12(res, lo, hi):
    led, k, n = res["ledger"], 0, 0
    for p in trunk_probes(res, "hippocampus-off"):
        if lo <= p["night"] <= hi:
            for r in p["rows"]:
                if r["kind"] in ("H", "rotating") and r.get("item") in led and min(led[r["item"]]["told_days"]) <= 2:
                    n += 1; k += bool(r.get("ok"))
    return k, n
P("| seed | B k/n (rate x6) | PR-l | diff /6 | PR-x | diff /6 | nights 1-15: B / PR-l / PR-x (rate x6) |")
P("|---|---|---|---|---|---|---|")
dg = {a: [] for a in ARMS}
for s in SEEDS:
    kb, nb = d12(R["B"][s], 5, 15); rb = kb / nb * 6
    row = [str(s), f"{kb}/{nb} ({rb:.2f})"]
    for a in ARMS:
        k, n = d12(R[a][s], 5, 15); r = k / n * 6
        dg[a].append(r - rb >= -1.0)
        row += [f"{k}/{n} ({r:.2f})", f"{r-rb:+.2f}"]
    allr = [d12(R[k][s], 1, 15) for k in ("B", "PR-l", "PR-x")]
    row.append(" / ".join(f"{k}/{n} ({k/n*6:.2f})" for k, n in allr))
    P("| " + " | ".join(row) + " |")
P()
P("### Guard collapse (life_bars.language: collapse = most-repeated reply share >= 0.4 on any trunk probe night of the arm)\n")
LG = {k: life_bars.language([R[k][s] for s in SEEDS]) for k in R}
P("| probe arm | B seeds-ever (max share per seed) | PR-l | PR-x |")
P("|---|---|---|---|")
cg = {a: [] for a in ARMS}
for arm in sorted(LG["B"]["collapse_arms"]):
    row = [arm]
    for k in R:
        c = LG[k]["collapse_arms"][arm]
        mx = ", ".join("%.2f" % c["per_seed_max_share"][s] for s in SEEDS)
        row.append(f"{c['seeds_ever']} ({mx})")
    P("| " + " | ".join(row) + " |")
    for a in ARMS:
        cg[a].append(LG[a]["collapse_arms"][arm]["seeds_ever"] <= LG["B"]["collapse_arms"][arm]["seeds_ever"])
P()
P("Collapse 'not worse' = for every probe arm the number of collapsed seeds is <= B's (strict reading; PR-x hippocampus-off 3 seeds vs B 2 is the failing cell, though LG-COL's own <=3/6 rule still passes). Also LG-COL / KILL-10 as life_bars reports: "
  + "; ".join(f"{k}: " + ", ".join(f"{b['id']} {b['verdict']}" for b in LG[k]['bars'] if b['id'] in ('LG-COL', 'KILL-10')) for k in R) + "\n")
P("### Guard language KL, night 15 (bar: <= 1.25 x B per seed)\n")
P("| seed | B KL n15 | PR-l KL | ratio | PR-x KL | ratio | NLL on/off n15 B | PR-l | PR-x |")
P("|---|---|---|---|---|---|---|---|---|")
kg = {a: [] for a in ARMS}
for s in SEEDS:
    g = {k: nrec(R[k][s], 15).get("lang_guard") for k in R}
    row = [str(s), f(g["B"]["kl"], 4)]
    for a in ARMS:
        r = g[a]["kl"] / g["B"]["kl"]; kg[a].append(r <= 1.25)
        row += [f(g[a]["kl"], 4), f(r)]
    row += [f"{g[k]['nll_on']:.3f}/{g[k]['nll_off']:.3f}" for k in R]
    P("| " + " | ".join(row) + " |")
P()
P("Whole-transcript read (3 sonnet readers plus refuters): NOT computed here; it needs the blind readers (not in the JSONs). UNDEFINED in this script; "
  "the guards below are the four numeric ones.\n")
for a in ARMS:
    P(f"- {a}: H-off {vd(allp(hg[a]))} {dict(zip(SEEDS, hg[a]))}; direct d1-2 {vd(allp(dg[a]))} {dict(zip(SEEDS, dg[a]))}; "
      f"collapse {vd(allp(cg[a]))}; KL n15 {vd(allp(kg[a]))} {dict(zip(SEEDS, kg[a]))}")
    summary[a]["3.3 H-off"] = (vd(allp(hg[a])), str(dict(zip(SEEDS, hg[a]))))
    summary[a]["3.3 direct d1-2"] = (vd(allp(dg[a])), str(dict(zip(SEEDS, dg[a]))))
    summary[a]["3.3 collapse"] = (vd(allp(cg[a])), "seeds-ever per probe arm <= B")
    summary[a]["3.3 KL n15"] = (vd(allp(kg[a])), str(dict(zip(SEEDS, kg[a]))))
    summary[a]["3.3 transcript read"] = ("UNDEFINED", "needs blind readers, not in JSON")
P()

# ---------------------------------------------------------------- 3.4
P("## 3.4 Secondary (report only)\n")
P("### Hippocampus-off recall of once-told E and F facts, nights 5-15\n")
P("(a) battery rows (kinds H / rotating / neardup, `ok`) whose ledger class is E or F, hippocampus-off trunk; (b) cue-ladder direct cue C0 rows (hippocampus-off, tracks E and F, all nights), `ok`.\n")
def ef(res, cls):
    led, k, n = res["ledger"], 0, 0
    for p in trunk_probes(res, "hippocampus-off"):
        if 5 <= p["night"] <= 15:
            for r in p["rows"]:
                e = led.get(r.get("item") or "")
                if e and e["cls"] == cls and "ok" in r:
                    n += 1; k += bool(r["ok"])
    return k, n
def cue0(res, track):
    k = n = 0
    for p in res.get("cue_probes", []):
        if p["arm"] == "hippocampus-off" and p.get("line", "trunk") == "trunk":
            for r in p["rows"]:
                if r["track"] == track and r["rung"] == "C0":
                    n += 1; k += bool(r["ok"])
    return k, n
P("| seed | cls | B battery k/n | PR-l | PR-x | B cue C0 | PR-l | PR-x |")
P("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for c in ("E", "F"):
        row = [str(s), c] + [f"{ef(R[k][s], c)[0]}/{ef(R[k][s], c)[1]}" for k in R] + [f"{cue0(R[k][s], c)[0]}/{cue0(R[k][s], c)[1]}" for k in R]
        P("| " + " | ".join(row) + " |")
for c in ("E", "F"):
    row = ["pooled", c]
    for fn in (ef, cue0):
        for k in R:
            kk = sum(fn(R[k][s], c)[0] for s in SEEDS); nn = sum(fn(R[k][s], c)[1] for s in SEEDS)
            row.append(f"{kk}/{nn}" + (f" ({kk/nn:.3f})" if nn else ""))
    P("| " + " | ".join(row) + " |")
P()
P("### Wrong fetches and hub top-1 (hippocampal read-outs; PR should not change them)\n")
P("wrong fetch = life_bars._h_wrong (kind-H rows with wrong_fetch, all arms and probes), plus the intact-arm wrong/src sum over nights 5-15 (life_table.seed_rows); "
  "hub top-1 = life_table.extra_rows live_top1 (share of the most fetched episode among live fetches) mean and max over nights 5-15.\n")
P("| seed | arm | _h_wrong wf / H rows | intact wf/src (n5-15) | hub top-1 mean (max) n5-15 |")
P("|---|---|---|---|---|")
for s in SEEDS:
    for k in R:
        w = life_bars._h_wrong(R[k][s])
        sr = [r for r in life_table.seed_rows(R[k][s]) if 5 <= r["night"] <= 15]
        ex = [r["live_top1"] for r in life_table.extra_rows(R[k][s]) if 5 <= r["night"] <= 15 and r["live_top1"] is not None]
        P(f"| {s} | {k} | {w['wrong_fetch']}/{w['n_H_rows']} | {sum(r['wf'] for r in sr)}/{sum(r['src'] for r in sr)} | {f(mean(ex))} ({f(max(ex) if ex else None)}) |")
P()
P("### Interview AUROC and invented-memory rate\n")
P("UNDEFINED from the results JSON: life_bars does not report them; they come from `life_interview_read` over blind-reader claims on the interview units "
  "(blind_PRl / blind_PRx / blind_S1F directories, reader output not part of the life JSONs). Not computed here.\n")
P("### Adapter norm (night 15, and x night 1)\n")
P("| seed | B n15 (x n1) | PR-l | PR-x | n5 B / PR-l / PR-x | n10 B / PR-l / PR-x |")
P("|---|---|---|---|---|---|")
for s in SEEDS:
    nn = {k: {n["night"]: n["adapter_norm"] for n in R[k][s]["nights"] if n["line"] == "trunk"} for k in R}
    P(f"| {s} | " + " | ".join(f"{nn[k][15]:.1f} ({nn[k][15]/nn[k][1]:.2f}x)" for k in R) + " | "
      + " / ".join(f"{nn[k][5]:.0f}" for k in R) + " | " + " / ".join(f"{nn[k][10]:.0f}" for k in R) + " |")
P()
P("### First-person echo (life_echo, report only, not a bar; nights 10+15, echo counted on told-item battery rows not scored right)\n")
P("| seed | arm | probe arm | n | ok | echo | echo with key | echo share |")
P("|---|---|---|---|---|---|---|---|")
for s in SEEDS:
    for k in R:
        t = life_echo.echo_table(R[k][s])
        for arm in life_echo.ARMS:
            p = life_echo.pooled(t, (10, 15), (arm,))
            P(f"| {s} | {k} | {arm} | {p['n']} | {p['ok']} | {p['echo']} | {p['echo_key']} | {f(p['share'])} |")
for k in R:
    for arm in life_echo.ARMS:
        ps = [life_echo.pooled(life_echo.echo_table(R[k][s]), (10, 15), (arm,)) for s in SEEDS]
        n, e = sum(p["n"] for p in ps), sum(p["echo"] for p in ps)
        P(f"| pooled | {k} | {arm} | {n} | {sum(p['ok'] for p in ps)} | {e} | {sum(p['echo_key'] for p in ps)} | {e/n:.3f} |")
P()

# ---------------------------------------------------------------- summary
P("## Bar summary\n")
P("| bar | PR-l | PR-x |")
P("|---|---|---|")
for key in summary["PR-l"]:
    P(f"| {key} | **{summary['PR-l'][key][0]}** ({summary['PR-l'][key][1]}) | **{summary['PR-x'][key][0]}** ({summary['PR-x'][key][1]}) |")
P()
for a in ARMS:
    e = summary[a]["3.1 engagement"][0]; l = summary[a]["3.2 leak"][0]
    g = [summary[a][k][0] for k in summary[a] if k.startswith("3.3") and "transcript" not in k]
    gv = "FAIL" if "FAIL" in g else "PASS (numeric guards; transcript read UNDEFINED)"
    dec = "KILLED (a numeric guard fails)" if "FAIL" in g else ("engagement bar failed: arm UNDEFINED per prereg 3.1 (leak/guards shown for information)" if e != "PASS" else ("passes 1+2+3 numeric, to registered test pending transcript read" if l == "PASS" else "NARROWED (2 fails, 3 holds)"))
    P(f"- {a}: engagement {e}; leak {l}; guards {gv}. Prereg section 4 outcome by these numbers: {dec}.")
open(os.path.join(HERE, "d2pr_bars.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
