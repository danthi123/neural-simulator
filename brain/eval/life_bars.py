# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md (sections 5, 10, 13)
# Expected-fail: none (a read-out)
"""Six-seed aggregator for the pre-registered bars of the multi-week life (design section 5 table, as amended by
section 13, plus the section-10 per-seed kill). HOST-ONLY read-out: it reads life_s*.json files and computes; it never
runs a brain, never changes a record, and every number comes from the JSONs.

python -m brain.eval.life_bars <life jsons or directories...> --out PREFIX      (writes PREFIX.json and PREFIX.md)

Bars (ids): P0 (forgetting precondition), SAV-* (savings, L-prunedA), CUE-* (cued recall), TRF-* (transfer),
LG-* (language guard, collapse, the section-10 kill), plus the report-only adapter-norm table.

Verdicts: PASS / FAIL / UNDEFINED, thresholds exactly as registered. With fewer than 6 seeds the verdict label is
"DEV (n seeds)" (counts only; the registered 5-of-6 rule is applied proportionally as ceil(5n/6) for the `verdict_core`
field, which exists only so the logic can be checked on DEV). A clause whose data is missing is UNDEFINED, never a crash
and never a silent pass; a per-seed clause with a missing seed value counts that seed as NOT passing.

Fixed permutation settings: N_PERM = 10000 permutations, numpy default_rng(RNG_SEED), one-sided p = (1 + #{perm >= obs}) / (1 + N).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from brain.eval import life as L
from brain.eval import life_ladder
from brain.eval import life_table

N_PERM = 10000
RNG_SEED = 20261121
EPS = 1e-9
LOW_CLS = ("F", "E", "D", "AB")       # LOW = every item told once (F, E, the near-duplicate frames D, A and B sets)
P_LEVEL = 0.05
AGE_OFFSET = 1                        # age in nights slept = harness age (night - last told day) + 1; see judgment calls
COLLAPSE = 0.4


# ---------------------------------------------------------------- small helpers
def load(paths) -> list:
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "life_s*.json"))) if os.path.isdir(p) else [p]
    out = []
    for f in files:
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def _need(n: int) -> int:
    """5 of 6 for six seeds; ceil(5n/6) otherwise (counts only on DEV)."""
    return math.ceil(5 * n / 6 - 1e-9)


def _label(core: str, n: int) -> str:
    if n >= 6:
        return core
    return f"DEV ({n} seeds)" + (" UNDEFINED" if core == "UNDEFINED" else "")


def _mean(v):
    v = [x for x in v if x is not None]
    return None if not v else sum(v) / len(v)


def _r(x, k=4):
    return None if x is None else round(float(x), k)


def count_rule(flags: dict, need: int | None = None) -> dict:
    """flags {seed: True/False/None}; None (undefined) counts as not passing."""
    n = len(flags)
    need = _need(n) if need is None else need
    c = sum(1 for v in flags.values() if v)
    nd = sum(1 for v in flags.values() if v is not None)
    return {"count": c, "n": n, "defined": nd, "need": need, "ok": c >= need}


def combine(clauses: list) -> str:
    """FAIL if any clause FAIL; else UNDEFINED if any UNDEFINED; else PASS."""
    if any(c == "FAIL" for c in clauses):
        return "FAIL"
    if any(c == "UNDEFINED" for c in clauses):
        return "UNDEFINED"
    return "PASS"


def bar(bid, group, name, definition, registered, n, core, **kw) -> dict:
    d = {"id": bid, "group": group, "name": name, "definition": definition, "registered_bar": registered,
         "verdict_core": core, "verdict": _label(core, n), "n_seeds": n}
    d.update(kw)
    return d


def perm_label_p(groups: list, rng) -> tuple:
    """Item-level permutation, items nested in seeds. groups = [(values, nA)] per seed: the first nA values carry label A,
    the rest label B. stat = mean over seeds of (mean A - mean B). Labels are permuted WITHIN each seed. One-sided p."""
    groups = [(np.asarray(v, float), na) for v, na in groups if na and len(v) > na]
    if not groups:
        return None, None
    obs = float(np.mean([v[:na].mean() - v[na:].mean() for v, na in groups]))
    tot = np.zeros(N_PERM)
    for v, na in groups:
        idx = rng.random((N_PERM, len(v))).argsort(axis=1)
        pv = v[idx]
        tot += pv[:, :na].mean(axis=1) - pv[:, na:].mean(axis=1)
    tot /= len(groups)
    return obs, float((1 + np.sum(tot >= obs - 1e-12)) / (1 + N_PERM))


# ---------------------------------------------------------------- P0: forgetting precondition
def _b_ids(ledger: dict) -> set:
    """The fresh B set: AB items first told on the day the A items are re-told (design 3: LOW is F, A and E; B is not LOW)."""
    rd = {d for e in ledger.values() if e["cls"] == "AB" for d in e["told_days"][1:]}
    return {i for i, e in ledger.items() if e["cls"] == "AB" and e["told_days"] and min(e["told_days"]) in rd}


def _told_once(ledger: dict, item: str, night: int):
    e = ledger.get(item)
    if not e or e["cls"] not in LOW_CLS or item in _b_ids(ledger):
        return None
    told = [d for d in e["told_days"] if d <= night]
    return told[0] if len(told) == 1 else None


def _arm_probes(res: dict, arm: str, line: str = "trunk"):
    return [p for p in res["probes"] if p["line"] == line and p["arm"] == arm and not p.get("tag")]


def low_recall_obs(res: dict, arm: str = "hippocampus-off") -> list:
    """Recall observations of LOW items (told exactly once by that morning) in one arm: (a) battery rows kind 'rotating' and
    told 'neardup' (row['ok'], strict scorer) and (b) the cue ladder's C0 direct rung (cue_probes rows, row['ok']).
    age = night - telling day (harness age; nights slept = age + AGE_OFFSET)."""
    led, out = res["ledger"], []
    for p in _arm_probes(res, arm):
        for x in p["rows"]:
            if x["kind"] == "rotating" or (x["kind"] == "neardup" and x.get("told")):
                t = _told_once(led, x.get("item"), p["night"])
                if t is not None and "ok" in x:
                    out.append({"item": x["item"], "age": p["night"] - t, "ok": bool(x["ok"]), "src": "battery"})
    for pr in res.get("cue_probes", []):
        if pr["arm"] != arm:
            continue
        for x in pr["rows"]:
            if x.get("rung") == "C0":
                t = _told_once(led, x["item"], pr["night"])
                if t is not None:
                    out.append({"item": x["item"], "age": pr["night"] - t, "ok": bool(x["ok"]), "src": "cue-C0"})
    return out


def low_span_rows(res: dict, arm: str = "hippocampus-off") -> list:
    out, bset = [], _b_ids(res["ledger"])
    for p in _arm_probes(res, arm):
        for s in p.get("span") or []:
            if s["cls"] in LOW_CLS and s["item"] not in bset and s["n_told"] == 1 and s.get("logodds") is not None:
                out.append(s)
    return out


def _rate(obs, lo, hi=None):
    sel = [o for o in obs if o["age"] + AGE_OFFSET >= lo and (hi is None or o["age"] + AGE_OFFSET <= hi)]
    return (sum(o["ok"] for o in sel) / len(sel), len(sel)) if sel else (None, 0)


def _lo(rows, lo, hi=None, off=None):
    off = AGE_OFFSET if off is None else off
    sel = [r["logodds"] for r in rows if r["age"] + off >= lo and (hi is None or r["age"] + off <= hi)]
    return (sum(sel) / len(sel), len(sel)) if sel else (None, 0)


def never_false_rate(res: dict, arm: str = "hippocampus-off"):
    ps = _arm_probes(res, arm)
    k = sum(p["summary"]["never_false"] for p in ps)
    n = sum(p["summary"]["never_n"] for p in ps)
    return (k / n if n else None), n


def h_last_mention(res: dict) -> dict:
    """{H id: sorted days it was told or re-mentioned}: told_days plus the schedule's 'remention' turns."""
    days = {i: set(e["told_days"]) for i, e in res["ledger"].items() if e["cls"] == "H"}
    for d in res.get("schedule", []):
        for t in d.get("turns", []):
            if t.get("kind") == "remention" and t.get("item") in days:
                days[t["item"]].add(d["day"])
    return {i: sorted(v) for i, v in days.items()}


def high_recall_obs(res: dict, arm: str) -> list:
    """H items asked in the battery (kind 'H'), age = night - last told-or-re-mentioned day."""
    md, out = h_last_mention(res), []
    for p in _arm_probes(res, arm):
        for x in p["rows"]:
            if x["kind"] == "H" and x.get("item") in md:
                last = [d for d in md[x["item"]] if d <= p["night"]]
                if last:
                    out.append({"item": x["item"], "age": p["night"] - max(last), "ok": bool(x["ok"]), "src": "battery-H"})
    return out


def p0_seed(res: dict, offset: int | None = None) -> dict:
    global AGE_OFFSET
    keep = AGE_OFFSET
    if offset is not None:
        AGE_OFFSET = offset
    try:
        obs, sp = low_recall_obs(res), low_span_rows(res)
        r1, n1 = _rate(obs, 1, 1)
        rL, nL = _rate(obs, 10)
        l1, m1 = _lo(sp, 1, 1)
        lL, mL = _lo(sp, 10)
        nf, nfn = never_false_rate(res)
        hs = [p["summary"] for p in _arm_probes(res, "hippocampus-off")]
        hr = sum(s["H_right"] for s in hs) / max(sum(s["H_n"] for s in hs), 1) if hs else None
        c_drop = None if r1 is None or rL is None else r1 - rL
        c_floor = None if rL is None or nf is None else rL - nf
        c_lo = None if l1 is None or lL is None else lL < l1
        flags = (None if None in (c_drop, c_floor, c_lo) else
                 (c_drop >= 0.15 - EPS, c_lo, c_floor >= 0.10 - EPS))
        return {"seed": res["seed"], "recall_age1": _r(r1), "n_age1": n1, "recall_age10p": _r(rL), "n_age10p": nL,
                "logodds_age1": _r(l1), "n_lo_age1": m1, "logodds_age10p": _r(lL), "n_lo_age10p": mL,
                "drop": _r(c_drop), "floor_margin": _r(c_floor), "never_false": _r(nf), "never_n": nfn, "H_off": _r(hr),
                "clauses": None if flags is None else {"drop>=0.15": flags[0], "logodds_lower": flags[1],
                                                       "above_false+0.10": flags[2]},
                "pass": None if flags is None else all(flags)}
    finally:
        AGE_OFFSET = keep


# ---------------------------------------------------------------- P0-v2 (age ladder, design 2026-11-22 section 3)
LADDER_FLOOR_NIGHTS = (11, 15)    # the floor and FM-v2 window, inclusive
FM_BAR_NIGHT = 15                 # FM-v2: night 15 is the bar for intact and both-off
FM_MARGIN = 0.10


def ladder_rows(res: dict, arm: str = "hippocampus-off") -> list:
    """The age-ladder rows of one arm (trunk, untagged probes), each with its probe night."""
    return [dict(x, night=p["night"]) for p in _arm_probes(res, arm) for x in p.get("age_ladder") or []]


def _item_weighted(rows: list):
    """Each item's mean over its rows, then the mean over items. -> (rate, n_items, {item: mean})."""
    per: dict = {}
    for r in rows:
        per.setdefault(r["item"], []).append(bool(r["ok"]))
    m = {i: sum(v) / len(v) for i, v in per.items()}
    return (sum(m.values()) / len(m) if m else None), len(m), m


def _per_question(rows: list, field: str, lo: int, hi: int):
    """Mean over never questions of the per-question rate of `field` on nights lo..hi. -> (rate, n_questions, n_rows)."""
    per: dict = {}
    for r in rows:
        if r["kind"] == "never" and lo <= r["night"] <= hi:
            per.setdefault(r["item"], []).append(bool(r[field]))
    if not per:
        return None, 0, 0
    return sum(sum(v) / len(v) for v in per.values()) / len(per), len(per), sum(len(v) for v in per.values())


def low_recall_obs_dedup(res: dict, arm: str = "hippocampus-off") -> list:
    """The pooled v1 observation set of a ladder life: battery + cue-C0 + ladder rows (LOW items), deduplicated by
    (item, night) keeping the ladder row. age = harness age (age_nights - AGE_OFFSET for ladder rows)."""
    lad = [{"item": r["item"], "age": r["age_nights"] - AGE_OFFSET, "ok": bool(r["ok"]), "src": "ladder",
            "night": r["night"]} for r in ladder_rows(res, arm) if r["kind"] in ("ladder", "control")]
    have = {(o["item"], o["night"]) for o in lad}
    led, out = res["ledger"], []
    for o in low_recall_obs(res, arm):
        t = _told_once(led, o["item"], 99)
        night = None if t is None else o["age"] + t
        if (o["item"], night) not in have:
            out.append(o)
    return out + [{k: v for k, v in o.items() if k != "night"} for o in lad]


def p0_seed_ladder(res: dict) -> dict:
    rows = ladder_rows(res)
    it = [r for r in rows if r["kind"] in ("ladder", "control")]
    r1, n1, m1 = _item_weighted([r for r in it if r["age_nights"] == 1])
    rL, nL, mL = _item_weighted([r for r in it if r["age_nights"] >= 10])
    rc, nc, _ = _item_weighted([r for r in it if r["kind"] == "control"])
    sp = low_span_rows(res)
    l1, k1 = _lo(sp, 1, 1)
    lL, kL = _lo(sp, 10)
    fl, fq, fr = _per_question(rows, "slot_false", *LADDER_FLOOR_NIGHTS)
    hs = [p["summary"] for p in _arm_probes(res, "hippocampus-off")]
    hr = sum(s["H_right"] for s in hs) / max(sum(s["H_n"] for s in hs), 1) if hs else None
    c_drop = None if r1 is None or rL is None else r1 - rL
    c_floor = None if rL is None or fl is None else rL - fl
    c_lo = None if l1 is None or lL is None else lL < l1
    flags = None if None in (c_drop, c_floor, c_lo) else (c_drop >= 0.15 - EPS, c_lo, c_floor >= 0.10 - EPS)
    return {"seed": res["seed"], "instrument": "age-ladder", "recall_age1": _r(r1), "n_items_age1": n1,
            "recall_age10p": _r(rL), "n_items_age10p": nL, "recall_controls": _r(rc), "n_controls": nc,
            "matched_night_drop": _r(None if rc is None or rL is None else rc - rL),
            "logodds_age1": _r(l1), "n_lo_age1": k1, "logodds_age10p": _r(lL), "n_lo_age10p": kL,
            "drop": _r(c_drop), "floor_margin": _r(c_floor), "slot_floor": _r(fl), "floor_questions": fq,
            "floor_rows": fr, "H_off": _r(hr),
            "item_means_age1": {i: _r(v) for i, v in sorted(m1.items())},
            "item_means_age10p": {i: _r(v) for i, v in sorted(mL.items())},
            "clauses": None if flags is None else {"drop>=0.15": flags[0], "logodds_lower": flags[1],
                                                   "above_floor+0.10": flags[2]},
            "pass": None if flags is None else all(flags)}


def p0_ladder(results: list) -> dict:
    n = len(results)
    seeds = [p0_seed_ladder(r) for r in results]
    cr = count_rule({s["seed"]: s["pass"] for s in seeds})
    core = "PASS" if cr["ok"] else ("UNDEFINED" if cr["defined"] == 0 else "FAIL")
    groups = []
    for s in seeds:     # pooled item-level permutation: items nested in seeds, age-1 vs age>=10 item means within seed
        a, b = list(s["item_means_age1"].values()), list(s["item_means_age10p"].values())
        if a and b:
            groups.append((a + b, len(a)))
    obs, pv = perm_label_p(groups, np.random.default_rng(RNG_SEED + 1))
    allrows = [r for res in results for r in ladder_rows(res) if r["kind"] in ("ladder", "control")]
    bins = []
    for a in sorted({r["age_nights"] for r in allrows}):
        sel = [r["ok"] for r in allrows if r["age_nights"] == a]
        bins.append((a, sum(sel) / len(sel), len(sel)))
    fit = _fit([b[0] for b in bins], [b[1] for b in bins], [b[2] for b in bins])
    mnd = [s["matched_night_drop"] for s in seeds if s["matched_night_drop"] is not None]
    pooled_drop = _mean([s["drop"] for s in seeds])
    v1 = [o for r in results for o in low_recall_obs_dedup(r)]
    v1_bins = []
    for a in sorted({o["age"] + AGE_OFFSET for o in v1}):
        sel = [o["ok"] for o in v1 if o["age"] + AGE_OFFSET == a]
        v1_bins.append({"age_nights": a, "recall": _r(sum(sel) / len(sel)), "n": len(sel)})
    label = ("night-confounded" if mnd and _mean(mnd) < 0.15 - EPS and pooled_drop is not None and pooled_drop >= 0.15 - EPS
             else "")
    return {"bar": bar("P0", "P0", "Forgetting precondition P0-v2 (age ladder, hippocampus-off, LOW items)",
                       "per seed: item-weighted ladder recall(age 1: 12 ladder + 8 matched-night controls) - recall(age>=10) "
                       ">= 0.15 AND mean log-odds(age>=10) < log-odds(age 1) (span rows) AND recall(age>=10) - slot-matched "
                       "never-told floor (nights 11-15, per question) >= 0.10; VALID if >=5/6 seeds",
                       "VALID if >=5/6 seeds pass; else savings UNDEFINED (design 2026-11-22 section 3)", n, core,
                       per_seed=seeds,
                       pooled={"seeds_passing": cr["count"], "defined_seeds": cr["defined"], "need": cr["need"],
                               "sensitivity_age_offset0_passing": "n/a (ladder)", "perm_stat": _r(obs),
                               "mean_matched_night_drop": _r(_mean(mnd))},
                       p=pv, null=None,
                       note="a seed with an empty age bin is UNDEFINED and counts as not passing; the pooled permutation "
                            "is reported beside the 5/6 count" + (f"; drop labelled {label}" if label else "")),
            "valid": core == "PASS", "age_bins_pooled": [{"age_nights": a, "recall": _r(r), "n": m} for a, r, m in bins],
            "fit": fit, "valid_label": _label(core, n), "instrument": "age-ladder", "v1_style_bins_dedup": v1_bins,
            "drop_label": label or None}


def fm_v2_seed(res: dict) -> dict:
    """FM-v2 rates (per question, then averaged) per arm: hippocampus-off on nights 11-15, intact and both-off on night 15
    (nights 5 and 10 reported). v1-style (STOCK included) from the same rows."""
    out = {"seed": res["seed"]}
    for arm in ("hippocampus-off", "intact", "both-off"):
        rows = ladder_rows(res, arm)
        lo, hi = LADDER_FLOOR_NIGHTS if arm == "hippocampus-off" else (FM_BAR_NIGHT, FM_BAR_NIGHT)
        fm, q, nr = _per_question(rows, "fm_v2", lo, hi)
        v1, _, _ = _per_question(rows, "v1_false", lo, hi)
        out[arm] = {"fm_v2": _r(fm), "v1_style": _r(v1), "questions": q, "rows": nr,
                    "by_night": {nt: _r(_per_question(rows, "fm_v2", nt, nt)[0]) for nt in sorted({r["night"] for r in rows
                                                                                                  if r["kind"] == "never"})}}
    b, i, h = (out[a]["fm_v2"] for a in ("both-off", "intact", "hippocampus-off"))
    out["pass"] = None if None in (b, i, h) else (i <= b + FM_MARGIN + EPS and h <= b + FM_MARGIN + EPS)
    return out


def fm_v2(results: list) -> dict:
    n = len(results)
    seeds = [fm_v2_seed(r) for r in results]
    cr = count_rule({s["seed"]: s["pass"] for s in seeds})
    core = "PASS" if cr["ok"] else ("UNDEFINED" if cr["defined"] == 0 else "FAIL")
    m = {a: _r(_mean([s[a]["fm_v2"] for s in seeds])) for a in ("hippocampus-off", "intact", "both-off")}
    v1 = {a: _r(_mean([s[a]["v1_style"] for s in seeds])) for a in ("hippocampus-off", "intact", "both-off")}
    return bar("FM-v2", "P0", "False memory FM-v2 (never-told ladder questions, any told key, STOCK out)",
               "per question then averaged: hippocampus-off nights 11-15, intact and both-off night 15; per seed intact <= "
               "both-off + 0.10 AND hippocampus-off <= both-off + 0.10; >=5/6 seeds", "PASS if >=5/6 seeds; its own "
               "verdict, never gates savings (design 2026-11-22 section 3b)", n, core, per_seed=seeds,
               pooled={"seeds_passing": cr["count"], "defined": cr["defined"], "need": cr["need"], "mean_fm_v2": m,
                       "mean_v1_style": v1}, p=None, null={"both-off": m["both-off"]},
               note="v1_style = the Stage-1 rate (any told key, STOCK included) on the same rows")


def _fit(ages: list, rates: list, ns: list):
    """Weighted least squares in log space over bins with recall > 0 and age >= 1: power law c*a^-b and exponential c*exp(-l*a).
    R2 is on the raw (rate) scale, weighted by n. Report only."""
    pts = [(a, r, n) for a, r, n in zip(ages, rates, ns) if r and r > 0 and a >= 1]
    if len(pts) < 3:
        return None
    a = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    w = np.array([p[2] for p in pts], float)
    out = {"bins_used": len(pts)}
    for name, x in (("power", np.log(a)), ("exp", a)):
        b, c0 = np.polyfit(x, np.log(y), 1, w=np.sqrt(w))
        pred = np.exp(c0 + b * x)
        mu = np.average(y, weights=w)
        ss_res, ss_tot = float(np.sum(w * (y - pred) ** 2)), float(np.sum(w * (y - mu) ** 2))
        out[name] = {"c": _r(math.exp(c0)), "rate": _r(-b), "R2": None if ss_tot <= 0 else _r(1 - ss_res / ss_tot)}
    return out


def p0(results: list) -> dict:
    n = len(results)
    seeds = [p0_seed(r) for r in results]
    flags = {s["seed"]: s["pass"] for s in seeds}
    cr = count_rule(flags)
    core = "PASS" if cr["ok"] else ("UNDEFINED" if cr["defined"] == 0 else "FAIL")
    alt = count_rule({r["seed"]: p0_seed(r, 0)["pass"] for r in results})
    allobs = [o for r in results for o in low_recall_obs(r)]
    ages = sorted({o["age"] + AGE_OFFSET for o in allobs})
    bins = []
    for a in ages:
        sel = [o for o in allobs if o["age"] + AGE_OFFSET == a]
        bins.append((a, sum(o["ok"] for o in sel) / len(sel), len(sel)))
    fit = _fit([b[0] for b in bins], [b[1] for b in bins], [b[2] for b in bins])
    return {"bar": bar("P0", "P0", "Forgetting precondition (hippocampus-off, LOW items)",
                       "per seed: LOW recall(age>=10 nights) <= recall(age 1) - 0.15 AND mean log-odds(age>=10) < log-odds(age 1) AND "
                       "recall(age>=10) >= never-told false rate + 0.10; VALID if >=5/6 seeds",
                       "VALID if >=5/6 seeds pass; else savings UNDEFINED", n, core, per_seed=seeds,
                       pooled={"seeds_passing": cr["count"], "defined_seeds": cr["defined"], "need": cr["need"],
                               "sensitivity_age_offset0_passing": alt["count"]},
                       p=None, null=None,
                       note="a seed with an empty age bin is UNDEFINED and counts as not passing"),
            "valid": core == "PASS", "age_bins_pooled": [{"age_nights": a, "recall": _r(r), "n": m} for a, r, m in bins],
            "fit": fit, "valid_label": _label(core, n)}


# ---------------------------------------------------------------- savings (L-prunedA)
def _sav(res: dict):
    pa = res.get("pruned_a") or {}
    if "report" in pa:       # the harness's own function over the same records (a life written before the 13 amendment lacks
        return pa, L.savings_table(res)     # the null-subtracted fields in its stored table)
    return pa, pa.get("savings") or None


def _items_null_sub(off: dict, null: dict):
    lo, ln = off.get("logodds") or {}, null.get("logodds") or {}
    side = lambda k: [lo[k][i] - ln[k][i] for i in lo.get(k, {}) if i in ln.get(k, {})]      # noqa: E731
    return side("A"), side("B")


def savings(results: list, p0_valid: bool, p0_label: str, rng) -> dict:
    n = len(results)
    per_night: dict = {}
    seeds_missing = []
    perm_groups = {}
    for res in results:
        pa, tab = _sav(res)
        if not tab:
            seeds_missing.append(res["seed"])
            continue
        for nk, row in tab.items():
            off = row.get("prunedA", {}).get("hippocampus-off")
            nul = row.get("prunedA", {}).get("both-off")
            tr = row.get("trunk", {}).get("hippocampus-off")
            bn = int(nk) - pa["day"] + 1      # branch night (1 = night pa["day"], the next morning), per seed
            rec = {"seed": res["seed"], "branch_night": bn}
            if off:
                rec.update(G_null_sub=off.get("G_null_sub"), G_null_sub_excl=off.get("G_null_sub_excl"), G_raw=off.get("G"),
                           G_excl=off.get("G_excl"), dR=off.get("dR"), dR_excl=off.get("dR_excl"),
                           dR_null_sub=off.get("dR_null_sub"), recall_A=off.get("recall_A"), recall_B=off.get("recall_B"))
                if nul:
                    rec.update(G_both_off=nul.get("G"), dR_both_off=nul.get("dR"))
                    a, b = _items_null_sub(off, nul)
                    if a and b:
                        rec["G_null_sub_items"] = _r(sum(a) / len(a) - sum(b) / len(b))
                        perm_groups.setdefault(bn, []).append((rec["seed"], a + b, len(a)))
            if tr:
                rec.update(trunk_G_null_sub=tr.get("G_null_sub"), trunk_G_raw=tr.get("G"), trunk_dR=tr.get("dR"),
                           trunk_dR_null_sub=tr.get("dR_null_sub"))
            per_night.setdefault(bn, []).append(rec)
    bars, detail = [], {}
    for nk in sorted(per_night):
        recs = per_night[nk]
        gs = {r["seed"]: r.get("G_null_sub") for r in recs}
        obs, pv = perm_label_p([(v, na) for _, v, na in perm_groups.get(nk, [])], rng)
        detail[nk] = {"per_seed": recs, "mean_G_null_sub": _r(_mean(list(gs.values()))), "pooled_perm_stat": _r(obs), "p": pv,
                      "mean_dR_null_sub": _r(_mean([r.get("dR_null_sub") for r in recs])),
                      "mean_G_raw": _r(_mean([r.get("G_raw") for r in recs])),
                      "mean_G_both_off": _r(_mean([r.get("G_both_off") for r in recs])),
                      "mean_G_excl": _r(_mean([r.get("G_null_sub_excl") for r in recs])),
                      "mean_trunk_G_null_sub": _r(_mean([r.get("trunk_G_null_sub") for r in recs]))}
    if 1 not in per_night:
        for bid, nm in (("SAV-G", "Savings G_null_sub night 1"), ("SAV-dR", "Savings dR_null_sub night 1")):
            bars.append(bar(bid, "Savings", nm, "needs --pruned-a records", "see design 5/13", n, "UNDEFINED",
                            per_seed={}, pooled=None, p=None, null=None, note="no pruned_a savings section in any life"))
        return {"bars": bars, "detail": detail, "perm_groups_missing": seeds_missing}
    n1 = 1                # branch night 1 (night bd = the L(bd+1) morning), keyed per seed from its own pruned_a day
    d = detail[n1]
    have = {r["seed"]: r for r in per_night[n1]}
    gflags = {x["seed"]: (None if have.get(x["seed"], {}).get("G_null_sub") is None else have[x["seed"]]["G_null_sub"] > 0) for x in results}
    cr = count_rule(gflags)
    g_core = "UNDEFINED" if d["p"] is None else ("PASS" if (cr["ok"] and d["p"] < P_LEVEL) else "FAIL")
    rflags = {x["seed"]: (None if have.get(x["seed"], {}).get("dR_null_sub") is None else have[x["seed"]]["dR_null_sub"] >= 1.5 - EPS) for x in results}
    mdr = _mean([r.get("dR_null_sub") for r in per_night[n1]])
    dr_core = "UNDEFINED" if mdr is None else ("PASS" if mdr >= 1.5 - EPS else "FAIL")
    if not p0_valid:      # savings is VALID only if P0 is VALID: the numbers stay beside an UNDEFINED verdict
        g_core = dr_core = "UNDEFINED"
    note = "" if p0_valid else f"UNDEFINED because P0 is not VALID (P0: {p0_label}); numbers shown are what the threshold would give: "
    thr_g = "PASS" if (cr["ok"] and d["p"] is not None and d["p"] < P_LEVEL) else "FAIL"
    thr_dr = None if mdr is None else ("PASS" if mdr >= 1.5 - EPS else "FAIL")
    bars.append(bar("SAV-G", "Savings", "G_null_sub, branch night 1 (all 8 A vs 8 B, hippocampus-off minus both-off)",
                    "G_null_sub = mean_A(lo_off - lo_bothoff) - mean_B(same); G>0 on >=5/6 seeds AND pooled item-level permutation "
                    f"(A/B labels permuted within seed, {N_PERM} perms, one-sided) p<0.05",
                    "G>0 on >=5/6 seeds and pooled p<0.05 (design 5 as amended by 13); valid only if P0 VALID",
                    n, g_core, per_seed={x["seed"]: next((r.get("G_null_sub") for r in per_night[n1] if r["seed"] == x["seed"]), None) for x in results},
                    pooled={"mean_G_null_sub": d["mean_G_null_sub"], "perm_stat": d["pooled_perm_stat"],
                            "seeds_positive": cr["count"], "need": cr["need"]},
                    p=d["p"], null={"G_both_off_raw_mean": d["mean_G_both_off"], "note": "both-off raw G is not a bar after 13 "
                                    "(item difficulty); reported beside"},
                    threshold_verdict=thr_g, note=note + f"raw G mean {d['mean_G_raw']}, excl mean {d['mean_G_excl']}"))
    bars.append(bar("SAV-dR", "Savings", "dR_null_sub at branch night 1 (recall A minus B, minus both-off dR), of 8",
                    "mean over seeds of per-seed dR_null_sub >= +1.5 (of 8)", "recall difference at night 1 >= +1.5 of 8 pooled",
                    n, dr_core, per_seed={x["seed"]: next((r.get("dR_null_sub") for r in per_night[n1] if r["seed"] == x["seed"]), None) for x in results},
                    pooled={"mean_dR_null_sub": d["mean_dR_null_sub"],
                            "seeds_ge_1.5": count_rule(rflags)["count"]},
                    p=None, null=None, threshold_verdict=thr_dr, note=note.strip()))
    return {"bars": bars, "detail": detail, "perm_groups_missing": seeds_missing}


# ---------------------------------------------------------------- cued recall
def rcue_combined(results: list, arm: str, delay: int, rng) -> dict:
    """R_cue = P(C2) - max(P(C3), P(C4)), tracks E and F pooled (the registered unit is 'E and LOW items'); items nested in
    seeds; permutation shuffles each item's (C2, C3, C4) outcomes within the item (the life_ladder statistic, vectorised)."""
    g = life_ladder._groups(results)
    trip, seeds = [], {}
    for (a, track, dl), items in g.items():
        if a != arm or dl != delay:
            continue
        for (s, _), v in items.items():
            if all(k in v for k in ("C2", "C3", "C4")):
                t = tuple(int(v[k]["ok"]) for k in ("C2", "C3", "C4"))
                trip.append(t)
                seeds.setdefault(s, []).append(t)

    def stat(t):
        t = np.asarray(t, float)
        return float(t[:, 0].mean() - max(t[:, 1].mean(), t[:, 2].mean()))
    if not trip:
        return {"n_items": 0, "R_cue": None, "p": None, "per_seed": {}, "seeds_positive": 0}
    arr = np.asarray(trip, float)
    obs = stat(arr)
    idx = rng.random((N_PERM, len(arr), 3)).argsort(axis=2)
    pv = np.take_along_axis(np.broadcast_to(arr, (N_PERM,) + arr.shape), idx, axis=2)
    sims = pv[:, :, 0].mean(1) - np.maximum(pv[:, :, 1].mean(1), pv[:, :, 2].mean(1))
    ps = {s: _r(stat(t)) for s, t in seeds.items()}
    return {"n_items": len(trip), "R_cue": _r(obs), "p": float((1 + np.sum(sims >= obs - 1e-12)) / (1 + N_PERM)),
            "per_seed": ps, "seeds_positive": sum(1 for v in ps.values() if v is not None and v > 0)}


def _h_wrong(res: dict) -> dict:
    """H answers wrong as the harness marks them: row kind 'H' with wrong_fetch True (WRONG-FETCH tag: an episode other than the
    ledger source came back), over every probe of every arm and line (non-tagged; tagged probes carry only AB rows).
    Distinct from 'not right' (ok False). Also counted for information: not-right, and a not-right reply that names another
    told ledger key (the harness's `false` test used for never-told questions)."""
    wf = nr = tot = kfalse = 0
    by_arm: dict = {}
    nr_arm: dict = {}
    for p in res["probes"]:
        if p.get("tag"):
            continue
        keys = None
        for x in p["rows"]:
            if x["kind"] != "H":
                continue
            tot += 1
            w = bool(x.get("wrong_fetch"))
            wf += w
            by_arm[p["arm"]] = by_arm.get(p["arm"], 0) + w
            nr_arm[p["arm"]] = nr_arm.get(p["arm"], 0) + (not x.get("ok"))
            if not x.get("ok"):
                nr += 1
                if keys is None:
                    keys = L.told_keys(res["ledger"], p["night"])
                if L._has(x["brain"], [k for k in keys if not L._has(x["you"], [k])]):
                    kfalse += 1
    return {"wrong_fetch": wf, "n_H_rows": tot, "not_right": nr, "not_right_names_other_key": kfalse, "wrong_fetch_by_arm": by_arm,
            "not_right_by_arm": nr_arm, "H_n_by_arm": {a: sum(1 for p in res["probes"] if not p.get("tag") and p["arm"] == a
                                                          for x in p["rows"] if x["kind"] == "H") for a in nr_arm}}


def cue(results: list, p0_valid: bool, rng) -> dict:
    n = len(results)
    bars, detail = [], {}
    has = any(res.get("cue_probes") for res in results)
    # R_cue at +1 (intact headline) and +3 (hippocampus-off headline)
    cl, cells = [], {}
    for arm, dl in (("intact", 1), ("hippocampus-off", 3)):
        c = rcue_combined(results, arm, dl, rng) if has else {"n_items": 0, "R_cue": None, "p": None, "per_seed": {}, "seeds_positive": 0}
        cells[f"{arm} +{dl}"] = c
        if c["R_cue"] is None:
            cl.append("UNDEFINED")
        else:
            cl.append("PASS" if (c["R_cue"] > 0 and c["p"] < P_LEVEL) else "FAIL")
    # both-off reference on the same cells where it was probed
    null_cells = {f"both-off +{dl}": (rcue_combined(results, "both-off", dl, rng) if has else None) for dl in (1, 3)}
    bars.append(bar("CUE-R", "Cued recall", "R_cue at +1 (intact) and +3 (hippocampus-off), E+F pooled",
                    "R_cue = P(C2) - max(P(C3), P(C4)); pooled item-level permutation p<0.05 and R_cue>0 at each of +1 and +3 "
                    "(per-seed direction count beside it; pooled permutation is primary per 13)",
                    "R_cue>0 at +1 and +3 (pooled p<0.05 primary, 13); per-seed 5/6 reported",
                    n, combine(cl), per_seed={k: v["per_seed"] for k, v in cells.items()},
                    pooled={k: {"R_cue": v["R_cue"], "n_items": v["n_items"], "seeds_positive": v["seeds_positive"]} for k, v in cells.items()},
                    p={k: v["p"] for k, v in cells.items()},
                    null={k: ({"R_cue": v["R_cue"], "n_items": v["n_items"]} if v else None) for k, v in null_cells.items()},
                    note="cue ladder absent" if not has else ""))
    # direct beats uncued by >= 0.4 pooled at age <= 3 (headline arms)
    g = life_ladder._groups(results)
    c0 = c4 = []
    c0, c4 = [], []
    per_seed_d: dict = {}
    for (arm, track, dl), items in g.items():
        if dl <= 3 and life_ladder.headline(arm, dl):
            for (s, _), v in items.items():
                if "C0" in v and "C4" in v:
                    c0.append(int(v["C0"]["ok"]))
                    c4.append(int(v["C4"]["ok"]))
                    per_seed_d.setdefault(s, []).append((int(v["C0"]["ok"]), int(v["C4"]["ok"])))
    if c0:
        diff = sum(c0) / len(c0) - sum(c4) / len(c4)
        core = "PASS" if diff >= 0.4 - EPS else "FAIL"
    else:
        diff, core = None, "UNDEFINED"
    bars.append(bar("CUE-D", "Cued recall", "direct (C0) minus uncued (C4), pooled, age <= 3 nights, headline arms",
                    "mean C0 right - mean C4 right over items in (intact +1) and (hippocampus-off +3) >= 0.4", "pooled >= 0.4", n, core,
                    per_seed={s: _r(sum(a for a, _ in v) / len(v) - sum(b for _, b in v) / len(v)) for s, v in per_seed_d.items()},
                    pooled={"diff": _r(diff), "n_items": len(c0), "C0": _r(sum(c0) / len(c0)) if c0 else None,
                            "C4": _r(sum(c4) / len(c4)) if c4 else None}, p=None, null=None))
    # H wrong answers
    hw = {res["seed"]: _h_wrong(res) for res in results}
    # wrong = not RIGHT (strict scorer ok False; the blind reader overrides when run, design 5 common rules), on the intact arm
    # (the lived brain; both-off is the null arm). wrong_fetch is a RETRIEVAL mark (set on right replies too) and design 5
    # logs retrieval apart from the reply, so it is reported, never the verdict.
    flags = {s: (None if not v["H_n_by_arm"].get("intact") else v["not_right_by_arm"].get("intact", 0) == 0) for s, v in hw.items()}
    cr = count_rule(flags, need=n)
    bars.append(bar("CUE-H", "Cued recall", "H has 0 wrong answers on every probe, every seed",
                    "wrong = battery row kind 'H' not RIGHT (strict scorer ok False; reader wins when run), intact arm, every "
                    "non-tagged probe of every line; per-arm not-right and wrong_fetch (retrieval, not an answer) reported", "0 wrong on every probe on 6/6 seeds (any wrong H answer = fail)", n,
                    "UNDEFINED" if cr["defined"] < n else ("PASS" if cr["ok"] else "FAIL"), per_seed=hw,
                    pooled={"seeds_clean": cr["count"], "intact_not_right": sum(v["not_right_by_arm"].get("intact", 0) for v in hw.values()), "total_wrong_fetch": sum(v["wrong_fetch"] for v in hw.values()),
                            "total_H_rows": sum(v["n_H_rows"] for v in hw.values())}, p=None, null=None))
    # HIGH >= LOW at matched age
    out_ = {}
    for arm in ("hippocampus-off", "intact"):
        per, fl = {}, {}
        for res in results:
            hi, lo = high_recall_obs(res, arm), low_recall_obs(res, arm)
            diffs, ws = [], []
            for a in sorted({o["age"] for o in hi} & {o["age"] for o in lo}):
                h = [o["ok"] for o in hi if o["age"] == a]
                l_ = [o["ok"] for o in lo if o["age"] == a]
                diffs.append(sum(h) / len(h) - sum(l_) / len(l_))
                ws.append(len(l_))
            v = None if not ws else sum(d_ * w for d_, w in zip(diffs, ws)) / sum(ws)
            per[res["seed"]] = {"diff": _r(v), "matched_ages": len(ws), "n_LOW": sum(ws)}
            fl[res["seed"]] = None if v is None else v >= -EPS
        out_[arm] = (per, count_rule(fl))
    per, cr = out_["hippocampus-off"]
    nodata = cr["defined"] == 0
    bars.append(bar("CUE-HL", "Cued recall", "HIGH-use >= LOW-use recall at matched age (hippocampus-off)",
                    "per seed: LOW-weighted mean over exactly matched ages of recall(H) - recall(LOW) >= 0; H age counted from its last "
                    "told-or-re-mentioned day (schedule 'remention' turns), LOW = battery rotating/told-neardup + cue C0 rows",
                    "HIGH>=LOW at matched age on >=5/6 seeds", n,
                    "UNDEFINED" if nodata else ("PASS" if cr["ok"] else "FAIL"), per_seed=per,
                    pooled={"seeds_ok": cr["count"], "defined": cr["defined"], "intact_arm": {"seeds_ok": out_["intact"][1]["count"],
                            "defined": out_["intact"][1]["defined"]}}, p=None, null=None,
                    note="no matched ages on any seed" if nodata else "an undefined seed counts as not passing"))
    ov = combine([b["verdict_core"] for b in bars])
    if not p0_valid:
        pass    # the cued-recall bars do not depend on P0
    return {"bars": bars, "overall_core": ov, "overall": _label(ov, n), "cells": cells}


# ---------------------------------------------------------------- transfer
def _tr_by_arm(res: dict) -> dict:
    out: dict = {}
    for p in res["probes"]:
        if p["line"] == "trunk" and not p.get("tag") and p.get("transfer"):
            out.setdefault(p["arm"], {})[p["night"]] = p["transfer"]
    return out


def _k_nights(tr_intact: dict, k: int) -> list:
    """Nights at which both domains have exactly k incidents told (read from the record's own k)."""
    out = []
    for night in sorted(tr_intact):
        kk = tr_intact[night].get("k") or {}
        if kk and all(v == k for v in kk.values()) and len(kk) == 2:
            out.append(night)
    return out


def transfer(results: list, rng) -> dict:
    n = len(results)
    rows = {}
    for res in results:
        arms = _tr_by_arm(res)
        ti = arms.get("intact", {})
        r = {"nights": {}}
        for k in (1, 2, 4):
            cand = _k_nights(ti, k)
            r["nights"][k] = cand
            for arm in ("intact", "hippocampus-off", "consolidated-off", "both-off", "nosleep"):
                # the arm's first probed night among the nights with exactly k incidents in both domains (the full-probe-night
                # arms are not read every night, so their k=4 read is night 15 when night 13 was not probed)
                nt = next((x for x in cand if x in arms.get(arm, {})), None)
                tr = arms.get(arm, {}).get(nt) if nt is not None else None
                r[(arm, k)] = None if tr is None else {"T": tr["T"], "T0": tr["T0"], "dT": tr["dT"], "per": tr["per"],
                                                       "variant": tr["variant"], "night": nt}
        rows[res["seed"]] = r
    seeds = list(rows)

    def val(arm, k, f="T"):
        return {s: (None if rows[s][(arm, k)] is None else rows[s][(arm, k)][f]) for s in seeds}
    bars = []
    # T>0 at k=4 on >=5/6 AND pooled permutation p<0.05 (intact)
    t4 = val("intact", 4)
    cr = count_rule({s: (None if v is None else v > 0) for s, v in t4.items()})
    groups = []
    for s in seeds:
        c = rows[s][("intact", 4)]
        if c and c["per"].get("sysadmin") and c["per"].get("home repair"):
            sg = 1 if c["variant"].get("sysadmin") == "X" else -1
            a, b = c["per"]["sysadmin"], c["per"]["home repair"]
            groups.append((sg * np.asarray(a + b, float), len(a)))
    obs, pv = perm_label_p(groups, rng)
    core = "UNDEFINED" if cr["defined"] == 0 or pv is None else ("PASS" if cr["ok"] and pv < P_LEVEL else "FAIL")
    bars.append(bar("TRF-T4", "Transfer", "T_life > 0 at k=4 (intact copy)",
                    "T = s*(mean_sys - mean_home of lp(X)-lp(Y) over the never-told incident questions) read the morning both domains "
                    "have 4 incidents told; >0 on >=5/6 seeds AND pooled permutation (domain labels permuted within seed, "
                    f"{N_PERM} perms) p<0.05. RAW T (the registered quantity); dT = T - T0 is shown in the table",
                    "T>0 at k=4 on >=5/6 seeds (+ common-rule pooled p<0.05, 13)", n, core, per_seed=t4,
                    pooled={"mean_T": _r(_mean(list(t4.values()))), "perm_stat": _r(obs), "seeds_positive": cr["count"],
                            "need": cr["need"], "mean_dT": _r(_mean(list(val("intact", 4, "dT").values())))},
                    p=pv, null={"both_off_T_k4": val("both-off", 4)}))
    # increasing with k
    t1 = val("intact", 1)
    pair = [(t4[s], t1[s]) for s in seeds if t4[s] is not None and t1[s] is not None]
    if pair:
        m4, m1 = sum(a for a, _ in pair) / len(pair), sum(b for _, b in pair) / len(pair)
        core = "PASS" if m4 > m1 else "FAIL"
    else:
        m4 = m1 = None
        core = "UNDEFINED"
    bars.append(bar("TRF-K", "Transfer", "T increases with k (k=4 above k=1, pooled, intact)",
                    "mean over seeds of T(k=4) > mean of T(k=1)", "k=4 above k=1 pooled", n, core,
                    per_seed={"k1": t1, "k2": val("intact", 2), "k4": t4},
                    pooled={"mean_T_k1": _r(m1), "mean_T_k2": _r(_mean(list(val("intact", 2).values()))), "mean_T_k4": _r(m4)},
                    p=None, null=None))

    def about_zero(arm, k):
        v = [x for x in val(arm, k).values() if x is not None]
        if len(v) < 3:
            return "UNDEFINED", None, None
        m, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        se = sd / math.sqrt(len(v))
        return ("PASS" if abs(m) <= 2 * se else "FAIL"), m, se
    c, m, se = about_zero("both-off", 4)
    bars.append(bar("TRF-B0", "Transfer", "both-off T about 0 at k=4",
                    "|mean T| <= 2 SE across seeds (n>=3); no numeric threshold is registered for 'about 0', this rule is mine",
                    "both-off T about 0", n, c, per_seed=val("both-off", 4), pooled={"mean_T": _r(m), "se": _r(se)}, p=None, null=None))
    nsl = {s: rows[s][("nosleep", 4)] for s in seeds}
    if all(v is None for v in nsl.values()):
        bars.append(bar("TRF-N0", "Transfer", "L-nosleep T about 0 at k=4", "needs a nosleep line with transfer records",
                        "L-nosleep T about 0", n, "UNDEFINED", per_seed={}, pooled=None, p=None, null=None,
                        note="no L-nosleep transfer records in these lives"))
    else:
        c, m, se = about_zero("nosleep", 4)
        bars.append(bar("TRF-N0", "Transfer", "L-nosleep T about 0 at k=4", "same rule as both-off", "L-nosleep T about 0", n, c,
                        per_seed=val("nosleep", 4), pooled={"mean_T": _r(m), "se": _r(se)}, p=None, null=None))
    # consolidated-off reduces T
    tc = val("consolidated-off", 4)
    # intact read on the SAME night as the consolidated-off cell (consolidated-off is probed on full nights only)
    ti_c = {}
    for res in results:
        c = rows[res["seed"]][("consolidated-off", 4)]
        tr_i = _tr_by_arm(res).get("intact", {}).get(c["night"]) if c else None
        ti_c[res["seed"]] = None if tr_i is None else tr_i["T"]
    t4c = ti_c
    red = {s: (None if tc[s] is None or t4c[s] is None else t4c[s] - tc[s] > 0) for s in seeds}
    cr = count_rule(red)
    dm = _mean([t4c[s] - tc[s] for s in seeds if tc[s] is not None and t4c[s] is not None])
    core = "UNDEFINED" if cr["defined"] == 0 else ("PASS" if cr["ok"] and dm > 0 else "FAIL")
    bars.append(bar("TRF-C", "Transfer", "consolidated-off reduces T at k=4",
                    "T_intact - T_consolidated-off at the consolidated-off k=4 night (same night) > 0 on >=5/6 seeds and mean > 0 (no permutation test is registered for this clause)",
                    "consolidated-off must reduce T", n, core, per_seed={s: _r(t4c[s] - tc[s]) if red[s] is not None else None for s in seeds},
                    pooled={"mean_reduction": _r(dm), "seeds_reduced": cr["count"], "defined": cr["defined"],
                            "mean_T_consolidated_off": _r(_mean(list(tc.values())))}, p=None, null=None))
    return {"bars": bars, "overall_core": combine([b["verdict_core"] for b in bars]),
            "overall": _label(combine([b["verdict_core"] for b in bars]), n), "table": {
                str(s): {f"{arm}@k{k}": rows[s][(arm, k)] and {x: rows[s][(arm, k)][x] for x in ("T", "T0", "dT", "night")}
                         for arm in ("intact", "hippocampus-off", "consolidated-off", "both-off") for k in (1, 2, 4)}
                for s in seeds}}


# ---------------------------------------------------------------- language guard, collapse, per-seed kill, adapter norm
def _collapse_by_arm(res: dict) -> dict:
    """{arm: {night: collapse share}} over the trunk's probes (life_table._by)."""
    out: dict = {}
    for night, arms in life_table._by(res, "trunk").items():
        for arm, p in arms.items():
            out.setdefault(arm, {})[night] = p["summary"]["collapse"]
    return out


def kill_check(res: dict) -> dict:
    """Section 10 as amended by 13: collapse (>=0.4) on 2 consecutive probe nights of the life (intact arm) ends the seed.
    Consecutive = adjacent in the sorted list of that arm's probe nights. Reports the night it would have ended (the second)."""
    c = _collapse_by_arm(res).get("intact", {})
    nights = sorted(c)
    for a, b in zip(nights, nights[1:]):
        if c[a] >= COLLAPSE - EPS and c[b] >= COLLAPSE - EPS:
            return {"seed": res["seed"], "triggered": True, "night": b, "first_night": a}
    return {"seed": res["seed"], "triggered": False, "night": None, "first_night": None}


def language(results: list) -> dict:
    n = len(results)
    bars, detail = [], {}
    # night-15 KL <= 2x night 1
    per, flags = {}, {}
    for res in results:
        rows = {n_["night"]: {"kl": (n_.get("lang_guard") or {}).get("kl"), "nll_on": (n_.get("lang_guard") or {}).get("nll_on"),
                              "nll_off": (n_.get("lang_guard") or {}).get("nll_off")}
                for n_ in res["nights"] if n_["line"] == "trunk"}
        r1, r15 = rows.get(1), rows.get(15)
        ok = r1 is not None and r15 is not None and r1.get("kl") is not None and r15.get("kl") is not None
        per[res["seed"]] = {"kl_n1": r1 and r1.get("kl"), "kl_n15": r15 and r15.get("kl"),
                            "ratio": _r(r15["kl"] / r1["kl"]) if ok and r1["kl"] else None,
                            "nll_on_n1": r1 and r1.get("nll_on"), "nll_off_n1": r1 and r1.get("nll_off"),
                            "nll_on_n15": r15 and r15.get("nll_on"), "nll_off_n15": r15 and r15.get("nll_off"),
                            "kl_by_night": {k: v.get("kl") for k, v in rows.items() if v.get("kl") is not None}}
        flags[res["seed"]] = None if not ok or not r1["kl"] else r15["kl"] <= 2 * r1["kl"] + EPS
    cr = count_rule(flags, need=n)   # registered: night-15 KL <= 2x night 1 (per seed; reported as a count, never a kill)
    core = "UNDEFINED" if cr["defined"] < n else ("PASS" if cr["count"] == n else "FAIL")
    bars.append(bar("LG-KL", "Language guard", "night-15 KL <= 2x night-1 KL (per seed)",
                    "KL of adapters-on vs pretrained on the frozen held-out set (nights[].lang_guard.kl); every seed must satisfy it; "
                    "UNDEFINED if any seed lacks the lang_guard record at night 1 or 15", "night-15 KL <= 2x night 1 (never a kill)",
                    n, core, per_seed=per, pooled={"seeds_ok": cr["count"], "defined": cr["defined"]}, p=None, null=None,
                    note="reading a seed without lang_guard records as UNDEFINED, not a pass" if cr["defined"] < n else ""))
    # collapse per arm
    col = {res["seed"]: _collapse_by_arm(res) for res in results}
    arms = sorted({a for v in col.values() for a in v})
    per_arm = {}
    for a in arms:
        ever = {s: any(x >= COLLAPSE - EPS for x in col[s].get(a, {}).values()) if a in col[s] else None for s in col}
        last = {s: (col[s][a][max(col[s][a])] >= COLLAPSE - EPS) if col[s].get(a) else None for s in col}
        per_arm[a] = {"seeds_ever": sum(1 for v in ever.values() if v), "seeds_at_last_probe": sum(1 for v in last.values() if v),
                      "per_seed_ever": ever, "per_seed_max_share": {s: max(col[s][a].values()) if col[s].get(a) else None for s in col}}
    ok_all = all(v["seeds_ever"] <= 3 for v in per_arm.values())
    core = "UNDEFINED" if not per_arm else ("PASS" if ok_all else "FAIL")
    bars.append(bar("LG-COL", "Language guard", "collapse on no more than 3/6 seeds, per arm",
                    "a seed collapses in an arm if the most-repeated reply share is >=0.4 on ANY probe night of that arm (trunk probes); "
                    "every arm must have <=3 collapsed seeds (counts only when n<6)", "collapse on <=3/6 seeds per arm", n, core,
                    per_seed={a: v["per_seed_ever"] for a, v in per_arm.items()},
                    pooled={a: {"seeds_ever": v["seeds_ever"], "seeds_at_last_probe": v["seeds_at_last_probe"]} for a, v in per_arm.items()},
                    p=None, null=None))
    kills = [kill_check(res) for res in results]
    # kill is report-only (a rule, not a bar); verdict field states which seeds would have been ended
    bars.append(bar("KILL-10", "Language guard", "section-10 per-seed kill (collapse on 2 consecutive probe nights, as amended by 13)",
                    "intact-arm collapse share >=0.4 on two adjacent probe nights ends that seed's life; norm and KL clauses are dropped by 13",
                    "no seed ended", n, "FAIL" if any(k["triggered"] for k in kills) else "PASS",
                    per_seed={k["seed"]: k for k in kills},
                    pooled={"triggered": [k["seed"] for k in kills if k["triggered"]]}, p=None, null=None,
                    note="would have ended: " + (", ".join(f"s{k['seed']} at night {k['night']}" for k in kills if k["triggered"]) or "none")))
    norms = {}
    for res in results:
        nn = {n_["night"]: n_.get("adapter_norm") for n_ in res["nights"] if n_["line"] == "trunk"}
        norms[res["seed"]] = {k: {"norm": v, "x_night1": _r(v / nn[min(nn)]) if v and nn.get(min(nn)) else None}
                              for k, v in sorted(nn.items())}
    return {"bars": bars, "collapse_arms": per_arm, "norms": norms}


# ---------------------------------------------------------------- assemble and render
def analyse(results: list) -> dict:
    results = sorted(results, key=lambda r: r["seed"])
    n = len(results)
    rng = np.random.default_rng(RNG_SEED)
    lad = [bool(r.get("age_ladder")) for r in results]
    if any(lad) and not all(lad):
        raise ValueError("life_bars: some lives have the age ladder and some do not; read them apart")
    p = p0_ladder(results) if any(lad) else p0(results)
    sv = savings(results, p["valid"], p["valid_label"], rng)
    cu = cue(results, p["valid"], rng)
    tr = transfer(results, rng)
    lg = language(results)
    bars = [p["bar"]] + ([fm_v2(results)] if any(lad) else []) + sv["bars"] + cu["bars"] + tr["bars"] + lg["bars"]
    return {"n_seeds": n, "seeds": [r["seed"] for r in results], "n_perm": N_PERM, "rng_seed": RNG_SEED, "bars": bars,
            "p0": {k: v for k, v in p.items() if k != "bar"}, "savings": sv["detail"], "cue": {"cells": cu["cells"], "overall": cu["overall"]},
            "transfer": {"overall": tr["overall"], "table": tr["table"]}, "language": {"collapse_arms": lg["collapse_arms"]},
            "adapter_norm": lg["norms"],
            "sections": {"Cued recall": cu["overall"], "Transfer": tr["overall"]}}


def _f(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.3f}"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}: {_f(v)}" for k, v in x.items()) + "}"
    return str(x)


def render(A: dict) -> str:
    n = A["n_seeds"]
    L_ = [f"# Life bars ({n} seeds: {A['seeds']}){' - DEV, thresholds as written, counts only' if n < 6 else ''}\n",
          f"_Permutation tests: {A['n_perm']} perms, numpy default_rng({A['rng_seed']}), one-sided. Verdict label is `DEV (n seeds)` "
          "when n<6; `core` is the threshold result (5/6 applied as ceil(5n/6)) shown for logic checking only._\n",
          "## Verdicts\n", "| id | bar | verdict | core | pooled | p |", "|---|---|---|---|---|---|"]
    for b in A["bars"]:
        L_.append(f"| {b['id']} | {b['name']} | {b['verdict']} | {b['verdict_core']} | {_f(b['pooled'])} | {_f(b['p'])} |")
    for b in A["bars"]:
        L_ += [f"\n### {b['id']} {b['name']}\n", f"- definition: {b['definition']}", f"- registered bar: {b['registered_bar']}",
               f"- verdict: **{b['verdict']}** (core {b['verdict_core']})", f"- per seed: {_f(b['per_seed'])}",
               f"- pooled: {_f(b['pooled'])}; p: {_f(b['p'])}"]
        if b.get("null") is not None:
            L_.append(f"- both-off / null arm: {_f(b['null'])}")
        if b.get("threshold_verdict"):
            L_.append(f"- threshold result ignoring the P0 gate: {b['threshold_verdict']}")
        if b.get("note"):
            L_.append(f"- note: {b['note']}")
    p0_ = A["p0"]
    lad = p0_.get("instrument") == "age-ladder"
    L_ += ["\n## P0 detail: LOW recall by age (hippocampus-off, pooled over seeds; age = nights slept)"
           + (" - age-ladder rows (P0-v2)" if lad else "") + "\n", "| age | recall | n |", "|---|---|---|"]
    L_ += [f"| {b['age_nights']} | {_f(b['recall'])} | {b['n']} |" for b in p0_["age_bins_pooled"]]
    L_.append(f"\nFits (report only): {_f(p0_['fit'])}")
    if lad:
        L_ += ["\nOffset-0 sensitivity: n/a (ladder). Drop label: " + (p0_.get("drop_label") or "none") + ".",
               "\n### v1 observation set (battery + cue-C0 + ladder, deduplicated by (item, night), ladder row kept)\n",
               "| age | recall | n |", "|---|---|---|"]
        L_ += [f"| {b['age_nights']} | {_f(b['recall'])} | {b['n']} |" for b in p0_["v1_style_bins_dedup"]]
    L_ += ["\n## Savings detail per branch night\n", "| branch night | seed | G_null_sub | excl | G raw | G both-off | dR | dR_null_sub | recall A/B | trunk G_null_sub |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for nk, d in sorted(A["savings"].items()):
        for r in d["per_seed"]:
            L_.append(f"| {r['branch_night']} | {r['seed']} | {_f(r.get('G_null_sub'))} | {_f(r.get('G_null_sub_excl'))} | {_f(r.get('G_raw'))} "
                      f"| {_f(r.get('G_both_off'))} | {_f(r.get('dR'))} | {_f(r.get('dR_null_sub'))} | {_f(r.get('recall_A'))}/{_f(r.get('recall_B'))} "
                      f"| {_f(r.get('trunk_G_null_sub'))} |")
        L_.append(f"| {d['per_seed'][0]['branch_night']} | mean | {_f(d['mean_G_null_sub'])} | {_f(d['mean_G_excl'])} | {_f(d['mean_G_raw'])} "
                  f"| {_f(d['mean_G_both_off'])} | - | {_f(d['mean_dR_null_sub'])} | - | {_f(d['mean_trunk_G_null_sub'])} | (p {_f(d['p'])}) |")
    L_ += ["\n## Transfer table (T raw, T0, dT = T - T0, night)\n", "| seed | cell | T | T0 | dT | night |", "|---|---|---|---|---|---|"]
    for s, cells in A["transfer"]["table"].items():
        for k, v in cells.items():
            if v:
                L_.append(f"| {s} | {k} | {_f(v['T'])} | {_f(v['T0'])} | {_f(v['dT'])} | {v['night']} |")
    L_ += ["\n## Collapse by arm\n", "| arm | seeds ever >=0.4 | seeds at last probe | max share per seed |", "|---|---|---|---|"]
    for a, v in A["language"]["collapse_arms"].items():
        L_.append(f"| {a} | {v['seeds_ever']} | {v['seeds_at_last_probe']} | {_f(v['per_seed_max_share'])} |")
    L_ += ["\n## Adapter norm per night (report only; x night 1)\n", "| seed | " + " | ".join(
        str(k) for k in sorted({k for v in A["adapter_norm"].values() for k in v})) + " |"]
    nights = sorted({k for v in A["adapter_norm"].values() for k in v})
    L_.append("|---|" + "---|" * len(nights))
    for s, v in A["adapter_norm"].items():
        L_.append(f"| {s} | " + " | ".join("-" if k not in v else f"{_f(v[k]['norm'])} ({_f(v[k]['x_night1'])}x)" for k in nights) + " |")
    return "\n".join(L_) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = analyse(load(a.paths))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(A, fh, indent=1, default=str)
    md = render(A)
    with open(a.out + ".md", "w") as fh:
        fh.write(md)
    print(md)
    return A


if __name__ == "__main__":
    main()
