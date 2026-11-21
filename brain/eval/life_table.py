# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: schema transfer and cued recall by context at 1.5B (design section 2)
"""Per-night tracking of the two known defects from life_sS.json files (design section 6). Report-only.

python -m brain.eval.life_table <life_s310.json> [...]   (or a directory: every life_s*.json in it)   [--out table.md]

D1 wrong-episode reinstatement: wrong-fetch rate (fetched episode is not the ledger source) and the paired cost
(hippocampus-off right minus intact right on the same direct questions).
D2 slept-fact intrusion: open-turn intrusions per arm and the excess over the both-off arm (full-probe nights only).
Also H right, collapse (most-repeated reply share), night seconds, store size and adapter norm. A row per seed and night
for the trunk life, and the sums over seeds. The nosleep line (hippocampus only, zero nights) is listed last.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _by(res: dict, line: str) -> dict:
    """{night: {arm: summary}} for one life line. Tagged extra probes (--pruned-a's trunk AB recall) are not arms."""
    out: dict = {}
    for p in res["probes"]:
        if p["line"] == line and not p.get("tag"):
            out.setdefault(p["night"], {})[p["arm"]] = p
    return out


def seed_rows(res: dict, line: str = "trunk") -> list:
    rows = []
    nights = {n["night"]: n for n in res["nights"] if n["line"] == line}
    for night, arms in sorted(_by(res, line).items()):
        i, o, b = (arms.get(k) for k in ("intact", "hippocampus-off", "both-off"))
        si, so = i["summary"], o["summary"]
        r = {"seed": res["seed"], "line": line, "night": night, "seconds": nights.get(night, {}).get("seconds"),
             "store": i["store"], "norm": i["adapter_norm"],
             "H_int": si["H_right"], "H_off": so["H_right"], "H_n": si["H_n"],
             "wf": si["wrong_fetch"], "src": si["src_n"], "fetched": si["fetched"],
             "right_int": si["direct_right"], "right_off": so["direct_right"], "direct_n": si["direct_n"],
             "intr_int": si["open_intrusions"], "intr_off": so["open_intrusions"], "open_n": si["open_n"],
             "intr_both": b["summary"]["open_intrusions"] if b else None,
             "col_int": si["collapse"], "col_off": so["collapse"],
             "never_false_int": si["never_false"], "never_false_off": so["never_false"], "never_n": si["never_n"]}
        lg = nights.get(night, {}).get("lang_guard")     # --lang-guard lives only
        if lg:
            r["kl"], r["nll_on"], r["nll_off"] = lg["kl"], lg["nll_on"], lg["nll_off"]
        r["d1_cost"] = r["right_off"] - r["right_int"]
        r["d2_excess"] = None if b is None else r["intr_int"] - r["intr_both"]
        rows.append(r)
    kl1 = next((r["kl"] for r in rows if r.get("kl") is not None), None)   # the first measured night = night 1
    for r in rows:
        if r.get("kl") is not None:
            r["kl_ratio"] = round(r["kl"] / kl1, 3) if kl1 else None
    return rows


def _f(x):
    return "-" if x is None else str(x)


def _top1(texts: list):
    """(share of the most fetched episode, distinct episodes) over a list of fetched episode texts; (None, 0) if none."""
    if not texts:
        return None, 0
    from collections import Counter
    c = Counter(texts)
    return round(c.most_common(1)[0][1] / len(texts), 2), len(c)


def extra_rows(res: dict, line: str = "trunk") -> list:
    """M-use / M-scale read-outs per night (prereg 2026-11-19-life-muse-mscale-prereg.md section 3). Gate counts and
    adapter fields exist only for runs with the new flags (else None); hub shares read the fetched episode text, so
    they exist for every run. Direct questions (H + rotating) are split by the first day their fact was told."""
    rows = []
    nights = {n["night"]: n for n in res["nights"] if n["line"] == line}
    ledger = res.get("ledger", {})
    for night, arms in sorted(_by(res, line).items()):
        i, o, fz = arms.get("intact"), arms.get("hippocampus-off"), arms.get("intact-frozen")
        lived = [r for x in res["lived"] if x["line"] == line and x["day"] == night for r in x["rows"]]
        live_f = [r["recalled"] for r in lived if r.get("recalled")]
        gate = [r["fetch"] for r in lived if r.get("fetch") and r["fetch"].get("idx") is not None]
        top, distinct = _top1(live_f)
        ftop, _ = _top1([r["recalled"] for r in fz["rows"] if r.get("recalled")] if fz else [])
        n = nights.get(night, {})
        irows = i["rows"]
        r = {"seed": res["seed"], "night": night, "fetched": i["summary"]["fetched"],
             "right": sum(1 for x in irows if x.get("right_episode")), "wrong": i["summary"]["wrong_fetch"],
             "granted": sum(1 for g in gate if g["confirmed"]) if gate or any(x.get("fetch") for x in lived) else None,
             "denied": sum(1 for g in gate if not g["confirmed"]) if gate or any(x.get("fetch") for x in lived) else None,
             "max_strength": i.get("max_strength_open"), "live_top1": top, "frozen_top1": ftop, "live_distinct": distinct,
             "norm_pre": n.get("norm_pre_night"), "g": n.get("downscale_g"), "norm_post": i["adapter_norm"],
             "A": (i.get("factor_norms") or [None, None])[0], "B": (i.get("factor_norms") or [None, None])[1]}

        def split(p):
            out = {}
            for k, lo in (("d12", True), ("d3p", False)):
                sel = [x for x in p["rows"] if x["kind"] in ("H", "rotating") and x.get("item") in ledger
                       and (min(ledger[x["item"]]["told_days"]) <= 2) == lo]
                out[k] = (sum(1 for x in sel if x.get("ok")), len(sel))
            return out
        si, so = split(i), split(o) if o else {"d12": (0, 0), "d3p": (0, 0)}
        r["d12"], r["d3p"] = (si["d12"], so["d12"]), (si["d3p"], so["d3p"])
        rows.append(r)
    return rows


def render(results: list) -> str:
    L = ["# Life table (per night: D1 wrong fetch, D2 intrusion, guard)\n",
         "D1 wf = wrong fetches / questions with a ledger source (fetched = an episode came back). D1 cost = off right - intact right "
         "(direct questions). D2 = open-turn intrusions (intact / off / both-off); excess = intact - both-off (full nights only). "
         "col = most-repeated reply share (0.4+ collapse).\n"]
    for line in ("trunk", "nosleep"):
        rows = [r for res in results for r in seed_rows(res, line)]
        if not rows:
            continue
        L.append(f"\n## {line}\n")
        lgc = any(r.get("kl") is not None for r in rows)     # --lang-guard columns only when the field exists
        L.append("| seed | night | s | store | norm | H int/off | wf | fetched | D1 cost | D2 int/off/both | D2 excess | never false int/off | col int/off |"
                 + (" KL | KL / night-1 |" if lgc else ""))
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if lgc else ""))
        for r in rows:
            L.append(f"| {r['seed']} | {r['night']} | {_f(r['seconds'])} | {r['store']} | {r['norm']} | {r['H_int']}/{r['H_off']} of {r['H_n']} "
                     f"| {r['wf']}/{r['src']} | {r['fetched']} | {r['d1_cost']:+d} of {r['direct_n']} | {r['intr_int']}/{r['intr_off']}/{_f(r['intr_both'])} of {r['open_n']} "
                     f"| {_f(r['d2_excess'])} | {r['never_false_int']}/{r['never_false_off']} of {r['never_n']} | {r['col_int']}/{r['col_off']} |"
                     + (f" {_f(r.get('kl'))} | {_f(r.get('kl_ratio'))} |" if lgc else ""))
        L.append("\n### M-use / M-scale read-outs (new columns)\n")
        L.append("fetched/right/wrong = intact-probe fetches of the direct questions with a source (counts). boosts = live-day "
                 "fetches whose use update was granted/denied (M-use runs only). max strength = largest non-permanent episode "
                 "strength when the probe copy opened. hub top-1 = share of the most fetched episode among live trunk fetches "
                 "(live) and among intact-frozen probe fetches (frozen); distinct = distinct episodes fetched live. "
                 "norm pre/g/post = adapter norm before the night's cut, the cut factor, the norm in the intact probe; "
                 "A, B = factor norms (M-scale runs). direct d1-2 / d3+ = right/n int and off on H + rotating questions whose fact was "
                 "first told on day 1-2 / day 3+ (every H fact is told on day 1).\n")
        L.append("| seed | night | fetched | right | wrong | boosts granted/denied | max strength | hub top-1 live | frozen | distinct | norm pre | g | norm post | A | B | direct d1-2 int/off | direct d3+ int/off |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for res in results:
            for r in extra_rows(res, line):
                (a1, n1), (a2, _) = r["d12"]
                (b1, n2), (b2, _) = r["d3p"]
                L.append(f"| {r['seed']} | {r['night']} | {r['fetched']} | {r['right']} | {r['wrong']} | {_f(r['granted'])}/{_f(r['denied'])} "
                         f"| {_f(r['max_strength'])} | {_f(r['live_top1'])} | {_f(r['frozen_top1'])} | {r['live_distinct']} "
                         f"| {_f(r['norm_pre'] if r['norm_pre'] is None else round(r['norm_pre'], 4))} | {_f(r['g'] if r['g'] is None else round(r['g'], 4))} "
                         f"| {r['norm_post']} | {_f(r['A'])} | {_f(r['B'])} "
                         f"| {a1}/{a2} of {n1} | {b1}/{b2} of {n2} |")
        L.append("\n### sums over seeds, per night\n")
        L.append("| night | seeds | H int | H off | wrong fetch | D1 cost | intrusions int | off | D2 excess (both-off nights) | max col | mean s |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for n in sorted({r["night"] for r in rows}):
            g = [r for r in rows if r["night"] == n]
            ex = [r["d2_excess"] for r in g if r["d2_excess"] is not None]
            secs = [r["seconds"] for r in g if r["seconds"] is not None]
            L.append(f"| {n} | {len(g)} | {sum(r['H_int'] for r in g)}/{sum(r['H_n'] for r in g)} | {sum(r['H_off'] for r in g)} "
                     f"| {sum(r['wf'] for r in g)}/{sum(r['src'] for r in g)} | {sum(r['d1_cost'] for r in g):+d} "
                     f"| {sum(r['intr_int'] for r in g)}/{sum(r['open_n'] for r in g)} | {sum(r['intr_off'] for r in g)} "
                     f"| {sum(ex) if ex else '-'} ({len(ex)} seeds) | {max(max(r['col_int'], r['col_off']) for r in g)} "
                     f"| {round(sum(secs) / len(secs), 1) if secs else '-'} |")
    if any("cue_ladder" in res for res in results):     # --cue-ladder lives only
        from brain.eval import life_ladder
        L += life_ladder.pooled_md(results)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "life_s*.json"))) if os.path.isdir(p) else [p]
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    text = render(results)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    print(text)
    return text


def replay_class_shares(res: dict, line: str = "trunk") -> dict:
    """Post-hoc, from the banked night records of ANY life (arm B included; nothing is written into the JSON): the
    night's replay share by class, normalised from `replay_share` (expected steps per class under the effective-strength
    weights). What the JSON cannot give: the permanent episodes and the top-6 / max share, because replay_share lumps
    the permanent self exchanges (their text is not a ledger telling) into class "other" and keeps no per-episode
    weights. Those come from a trunk home (_scratch/stage2/replay_shares_home.py). For a life run with
    --replay-priority the night record carries the exact values in n["replay_priority"], returned alongside.
    {night: {"shares": {cls: share}, "permanent_share": None | float, "top6_share": None | float, ...}}"""
    out = {}
    for n in res["nights"]:
        if n.get("line") != line or not n.get("replay_share"):
            continue
        tot = sum(n["replay_share"].values()) or 1.0
        row = {"shares": {c: round(v / tot, 4) for c, v in n["replay_share"].items()},
               "permanent_share": None, "top6_share": None, "max_share": None}
        rp = n.get("replay_priority")
        if rp:
            row.update({k: rp[k] for k in ("permanent_share", "top6_share", "max_share")})
        out[n["night"]] = row
    return out


if __name__ == "__main__":
    main()

