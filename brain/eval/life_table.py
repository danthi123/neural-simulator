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
    """{night: {arm: summary}} for one life line."""
    out: dict = {}
    for p in res["probes"]:
        if p["line"] == line:
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
        r["d1_cost"] = r["right_off"] - r["right_int"]
        r["d2_excess"] = None if b is None else r["intr_int"] - r["intr_both"]
        rows.append(r)
    return rows


def _f(x):
    return "-" if x is None else str(x)


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
        L.append("| seed | night | s | store | norm | H int/off | wf | fetched | D1 cost | D2 int/off/both | D2 excess | never false int/off | col int/off |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| {r['seed']} | {r['night']} | {_f(r['seconds'])} | {r['store']} | {r['norm']} | {r['H_int']}/{r['H_off']} of {r['H_n']} "
                     f"| {r['wf']}/{r['src']} | {r['fetched']} | {r['d1_cost']:+d} of {r['direct_n']} | {r['intr_int']}/{r['intr_off']}/{_f(r['intr_both'])} of {r['open_n']} "
                     f"| {_f(r['d2_excess'])} | {r['never_false_int']}/{r['never_false_off']} of {r['never_n']} | {r['col_int']}/{r['col_off']} |")
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


if __name__ == "__main__":
    main()
