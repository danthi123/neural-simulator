# Capability: Affect shaping speech
# Record: brain/research/2026-11-03-mood-state-design.md (section 12: iteration 3, pre-registered, frozen before the run)
"""Iteration 3: dev selection (register, 0.4B GAIN) and the six-seed table with the pre-registered criteria.

  python -m brain.experiments.mood_v3 register <dir>     # off-arm dev JSONs regdev_<size>_<register>_s<seed>.json
  python -m brain.experiments.mood_v3 gain <dir>         # gaindev_0.4b_g<gain>_s42.json
  python -m brain.experiments.mood_v3 table <dir> <size> # mood3_<size>_s<seed>.json
"""
import glob
import json
import re
import sys

import numpy as np

from brain.experiments import sentiment_instrument as si

CANDIDATES = ("plain", "mixed")       # "friendly" (iteration 2's register) is run as a reference only
SIZES = ("0.1b", "0.4b")
COVERAGE_MIN = 0.25
GAINS = (8.0, 16.0, 32.0)


def _rows(j, arm):
    return j["rows"][arm]


def coverage(rows):
    return float(np.mean([any(w in si.POS or w in si.NEG for w in si.words(r["reply"])) for r in rows]))


def select_register(d):
    stats = {}
    for reg in CANDIDATES + ("friendly",):
        for size in SIZES:
            rows = [r for f in glob.glob(f"{d}/regdev_{size}_{reg}_s*.json") for r in _rows(json.load(open(f)), "off")]
            if rows:
                stats[(reg, size)] = {"n": len(rows), "off_lex": float(np.mean([r["lex"] for r in rows])),
                                      "coverage": coverage(rows)}
    for (reg, size), v in sorted(stats.items()):
        print(f"{reg:9s} {size}: n={v['n']} off lex {v['off_lex']:+.2f}  coverage {v['coverage']:.2f}")
    ok = [r for r in CANDIDATES if all(stats[(r, s)]["coverage"] >= COVERAGE_MIN for s in SIZES)]
    if ok:
        win = min(ok, key=lambda r: max(abs(stats[(r, s)]["off_lex"]) for s in SIZES))
    else:
        win = max(CANDIDATES, key=lambda r: min(stats[(r, s)]["coverage"] for s in SIZES))
    print(f"SELECTED register: {win} (rule: coverage >= {COVERAGE_MIN} at both sizes, then min over registers of "
          f"max over sizes |off lex|)")
    return win


def select_gain(d):
    best, res = None, {}
    for g in GAINS:
        fs = glob.glob(f"{d}/gaindev_0.4b_g{int(g)}_s42.json")
        if not fs:
            continue
        s = json.load(open(fs[0]))["summary"]
        effect = s["pos"]["lex"] - s["neg"]["lex"]
        echo = (s["pos"]["echo_rate"] + s["neg"]["echo_rate"]) / 2 - s["off"]["echo_rate"]
        nll = (s["pos"]["nll"] + s["neg"]["nll"]) / 2 / s["off"]["nll"]
        ok = echo <= 0.10 and nll <= 1.25
        res[g] = (effect, echo, nll, ok)
        print(f"GAIN {g:4.0f}: lex pos-neg {effect:+.2f}  echo mood-off {echo:+.3f}  NLL mood/off {nll:.3f}  "
              f"{'ok' if ok else 'excluded'}")
        if ok and (best is None or effect > res[best][0]):
            best = g
    best = best if best is not None else min(GAINS)
    print(f"SELECTED 0.4B GAIN: {best:.0f} (rule: max lex pos-neg on dev s.t. echo <= 0.10 and NLL ratio <= 1.25; "
          f"tie -> smaller)")
    return best


def criteria(j):
    acc, ind, c, s = j["appraisal_sign_acc"], j["induction"], j["contrasts"], j["summary"]
    echo_d = (s["pos"]["echo_rate"] + s["neg"]["echo_rate"]) / 2 - s["off"]["echo_rate"]
    return {
        "C1_heldout_appraisal": acc["heldout_pos"] >= 0.75 and acc["heldout_neg"] >= 0.75,
        "C2_mood_signs": ind["m_pos"] > 0 > ind["m_neg"],
        "C3_pos_vs_null_ci_above_0": c["pos_null"]["lex_ci95"][0] > 0,
        "C4_neg_vs_null_ci_below_0": c["neg_null"]["lex_ci95"][1] < 0,
        "C5_embedding_vs_null": c["pos_null"]["emb_diff"] > 0 > c["neg_null"]["emb_diff"],
        "C6_echo_le_0.10": echo_d <= 0.10,
    }, echo_d


def table(d, size):
    files = sorted(glob.glob(f"{d}/mood3_{size}_s*.json"), key=lambda x: int(re.findall(r"_s(\d+)\.json", x)[0]))
    print("| seed | held-out acc +/- | m_pos | m_neg | lex pos-null [CI] | lex neg-null [CI] | emb pos-null / neg-null "
          "| null-off pos / neg (lex) | lex pos-off [CI] | lex neg-off [CI] | off lex | echo mood-off | "
          "NLL pos/neg/off | criteria met |")
    print("|" + "---|" * 14)
    tally, rep = {}, {"pos_off_ci_above_0": 0, "neg_off_ci_below_0": 0}
    means = {k: [] for k in ("pn", "nn", "po", "no", "nullp", "nulln", "off")}
    for f in files:
        j = json.load(open(f))
        acc, ind, c, s = j["appraisal_sign_acc"], j["induction"], j["contrasts"], j["summary"]
        cr, echo_d = criteria(j)
        for k, v in cr.items():
            tally[k] = tally.get(k, 0) + int(v)
        pn, nn, po, no = c["pos_null"], c["neg_null"], c["pos_off"], c["neg_off"]
        rep["pos_off_ci_above_0"] += po["lex_ci95"][0] > 0
        rep["neg_off_ci_below_0"] += no["lex_ci95"][1] < 0
        for k, v in (("pn", pn["lex_diff"]), ("nn", nn["lex_diff"]), ("po", po["lex_diff"]), ("no", no["lex_diff"]),
                     ("nullp", pn["lex_null_minus_off"]), ("nulln", nn["lex_null_minus_off"]),
                     ("off", s["off"]["lex"])):
            means[k].append(v)
        ci = lambda x: f"{x['lex_diff']:+.2f} [{x['lex_ci95'][0]:+.2f},{x['lex_ci95'][1]:+.2f}]"  # noqa: E731
        print(f"| {j['seed']} | {acc['heldout_pos']:.2f}/{acc['heldout_neg']:.2f} | {ind['m_pos']:+.3f} | "
              f"{ind['m_neg']:+.3f} | {ci(pn)} | {ci(nn)} | {pn['emb_diff']:+.3f} / {nn['emb_diff']:+.3f} | "
              f"{pn['lex_null_minus_off']:+.2f} / {nn['lex_null_minus_off']:+.2f} | {ci(po)} | {ci(no)} | "
              f"{s['off']['lex']:+.2f} | {echo_d:+.3f} | {s['pos']['nll']:.2f}/{s['neg']['nll']:.2f}/"
              f"{s['off']['nll']:.2f} | {sum(cr.values())}/6 (fail: "
              f"{','.join(k[:2] for k, v in cr.items() if not v) or '-'}) |")
    n = len(files)
    need = {"C2_mood_signs": 6, "C6_echo_le_0.10": 6}
    print(f"\nCriteria over {n} seeds at {size} (bar 5/6; C2 and C6 6/6):")
    allpass = True
    for k, v in tally.items():
        ok = v >= need.get(k, 5)
        allpass &= ok
        print(f"- {k}: {v}/{n} {'PASS' if ok else 'FAIL'}")
    print(f"Reported (not gated): pos vs off CI above 0 {rep['pos_off_ci_above_0']}/{n}; "
          f"neg vs off CI below 0 {rep['neg_off_ci_below_0']}/{n}")
    print("Means over seeds: " + ", ".join(f"{k} {np.mean(v):+.2f}" for k, v in means.items()))
    print(f"\nOVERALL {size}: {'PASS' if allpass and n == 6 else 'NOT PASS'}")


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    {"register": lambda: select_register(d), "gain": lambda: select_gain(d),
     "table": lambda: table(d, sys.argv[3])}[cmd]()
