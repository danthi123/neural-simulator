# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md (sections 5, 13); brain/eval/life_bars.py
# Expected-fail: none (a host-only read-out: a label-shuffle null for the life_bars composite verdicts)
"""Permutation (label-shuffle) null for the composite verdicts of brain/eval/life_bars.py (imported, never edited).

python -m brain.eval.life_bars_null LIFE.json [LIFE.json ...] --out PREFIX [--k 200] [--inner-perm 300]
                                    [--sections p0,savings,transfer,cue] [--synthetic-seeds 6]      (PREFIX.json, PREFIX.md)

PRE-REGISTRATION (fixed before any real run; change only with a new dated note)
  K = 200 null replicates (default), inner permutation count N_PERM monkeypatched to 300 inside each null replicate
  (life_bars' own 10000 is used for the OBSERVED re-run beside it), master seed 20261127. The scoring code is life_bars',
  unmodified; only the INPUT labels are shuffled, always within seed, always from the items, and every derived field is
  recomputed by life_bars from the shuffled items (no precomputed summary is shuffled):
    savings (SAV-G, SAV-dR)  A/B membership of the pruned_a items (pa["A"], pa["B"]); life.savings_table recomputes G and dR.
                             The P0 gate is OPEN in the null (ungated threshold verdicts), P0 has its own null.
    transfer (TRF-T4/K/C)    domain label (sysadmin / home repair) of each per-question value, per transfer cell, after the
                             variant sign is applied; T and d are recomputed from the shuffled lists.
    cue (CUE-R, CUE-D)       the rung label (C0..C4, C0g left alone) of each item's outcomes, within (probe, item).
    p0 (P0, v1 and ladder)   age: v1 shuffles the telling days of the told-once LOW items among themselves (age is derived
                             from the ledger) and the age of the qualifying span rows; the ladder shuffles age_nights among the
                             ladder/control rows of the hippocampus-off arm.
  Unshuffled bars (CUE-H, CUE-HL, TRF-B0/N0, LG-*, KILL-10, FM-v2) do not depend on these labels and are not in the null.
  Composite false-pass = ALL label-dependent bars of a section have verdict_core PASS in a null replicate ("composite" per
  section; "all" = every selected section at once, sections shuffled independently). Per bar: false-pass rate, null mean,
  95th and 99th percentile (numpy linear) of the bar's statistic, the null distribution of its seed count, and the observed
  statistic's rank: frac_ge = (1 + #null >= obs) / (1 + K), rank 1 = highest.
  PERCENTILE RULE: an observed bar is "beyond null" iff its statistic >= null 95th percentile (frac_ge <= 0.05), "strongly" iff
  >= 99th (frac_ge <= 0.01); the false-pass rule is applied PER BAR for the p-value bars (SAV-G, TRF-T4, CUE-R): trusted only if the bar's
  null false-pass rate is <= 0.05; reported 'inconclusive' when within 2 binomial SE at K (nominal 0.05) of 0.05. Section
  composites are an upper bound only. The observed run shares one rng in life_bars.analyse order (savings, cue, transfer). Statistics: P0 mean drop, SAV-G pooled perm stat, SAV-dR mean
  dR_null_sub, TRF-T4 mean T, TRF-K mean T(k4)-T(k1), TRF-C mean reduction, CUE-R min R_cue over the two headline cells,
  CUE-D pooled direct-minus-uncued.

--synthetic-seeds N: builds N pseudo-seeds by item-level bootstrap of the given lives' item pools (items resampled with
replacement within class, duplicates renamed, transfer per-question lists resampled within domain; donor = lives cycled).
Everything it prints is labelled SIMULATED: it shows what the bars do at six seeds on these pools, it is not data.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

from brain.eval import life_bars as LB

K_DEFAULT = 200
INNER_PERM = 300
MASTER_SEED = 20261127
SECTIONS = ("p0", "savings", "transfer", "cue")
RUNGS = ("C0", "C1", "C2", "C3", "C4")


# ---------------------------------------------------------------- shuffles (each returns a new life, input untouched)
def _perm_values(objs: list, key: str, rng):
    """Permute the `key` values among the dicts `objs` (in place on those dicts)."""
    vals = [o[key] for o in objs]
    for o, v in zip(objs, [vals[i] for i in rng.permutation(len(vals))]):
        o[key] = v


def shuffle_savings(res: dict, rng) -> dict:
    pa = res.get("pruned_a")
    if not pa or not pa.get("A") or not pa.get("B"):
        return res
    ids = list(pa["A"]) + list(pa["B"])
    p = [ids[i] for i in rng.permutation(len(ids))]
    return {**res, "pruned_a": {**pa, "A": p[:len(pa["A"])], "B": p[len(pa["A"]):]}}


def _retransfer(tr: dict, per: dict) -> dict:
    """Recompute d and T from per-question lists (T = sign * (mean sysadmin - mean home repair), as life_bars reads it)."""
    sg = 1 if tr["variant"].get("sysadmin") == "X" else -1
    d = {k: float(np.mean(v)) if v else None for k, v in per.items()}
    T = None if d.get("sysadmin") is None or d.get("home repair") is None else round(sg * (d["sysadmin"] - d["home repair"]), 4)
    return {**tr, "per": per, "d": d, "T": T, "dT": None if T is None or tr.get("T0") is None else round(T - tr["T0"], 4)}


def shuffle_transfer(res: dict, rng) -> dict:
    out = []
    for p in res["probes"]:
        tr = p.get("transfer")
        a, b = (tr or {}).get("per", {}).get("sysadmin"), (tr or {}).get("per", {}).get("home repair")
        if not a or not b:
            out.append(p)
            continue
        sg = 1 if tr["variant"].get("sysadmin") == "X" else -1
        v = sg * np.asarray(list(a) + list(b), float)
        v = v[rng.permutation(len(v))]
        # signed values go back with the sign folded in, so the cell is read with variant X for sysadmin
        tr2 = _retransfer({**tr, "variant": {**tr["variant"], "sysadmin": "X"}},
                          {"sysadmin": [float(x) for x in v[:len(a)]], "home repair": [float(x) for x in v[len(a):]]})
        out.append({**p, "transfer": tr2})
    return {**res, "probes": out}


def shuffle_cue(res: dict, rng) -> dict:
    cps = []
    for pr in res.get("cue_probes", []):
        rows = [dict(r) for r in pr["rows"]]
        by: dict = {}
        for r in rows:
            if r["rung"] in RUNGS:
                by.setdefault(r["item"], []).append(r)
        for rs in by.values():
            _perm_values(rs, "ok", rng)
        cps.append({**pr, "rows": rows})
    return {**res, "cue_probes": cps}


def shuffle_p0(res: dict, rng) -> dict:
    probes = [dict(p) for p in res["probes"]]
    off = [p for p in probes if p["line"] == "trunk" and p["arm"] == "hippocampus-off" and not p.get("tag")]
    out = {**res, "probes": probes}
    if res.get("age_ladder"):
        rows = []
        for p in off:
            p["age_ladder"] = [dict(x) for x in p.get("age_ladder") or []]
            rows += [x for x in p["age_ladder"] if x["kind"] in ("ladder", "control")]
        if rows:
            _perm_values(rows, "age_nights", rng)
    else:
        led = {i: dict(e) for i, e in res["ledger"].items()}
        bset = LB._b_ids(led)
        once = [i for i, e in led.items() if e["cls"] in LB.LOW_CLS and i not in bset and len(e["told_days"]) == 1]
        days = [led[i]["told_days"][0] for i in once]
        for i, d in zip(once, [days[j] for j in rng.permutation(len(days))]):
            led[i]["told_days"] = [d]
        out["ledger"] = led
    spans = []
    bset = LB._b_ids(res["ledger"])
    for p in off:
        p["span"] = [dict(s) for s in p.get("span") or []]
        spans += [s for s in p["span"] if s["cls"] in LB.LOW_CLS and s["item"] not in bset and s["n_told"] == 1
                  and s.get("logodds") is not None]
    if spans:
        _perm_values(spans, "age", rng)
    return out


SHUFFLE = {"p0": shuffle_p0, "savings": shuffle_savings, "transfer": shuffle_transfer, "cue": shuffle_cue}


# ---------------------------------------------------------------- scoring: life_bars unmodified, stats read from its bars
def _g(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _min(*v):
    v = [x for x in v if x is not None]
    return min(v) if v else None


def _bar_out(b: dict, stat, count) -> dict:
    return {"verdict": b["verdict_core"], "stat": stat, "count": count}


def score_section(name: str, results: list, rng) -> dict:
    """{bar id: {verdict, stat, count}} for the label-dependent bars of one section, by life_bars' own functions."""
    if name == "p0":
        p = LB.p0_ladder(results) if any(r.get("age_ladder") for r in results) else LB.p0(results)
        b = p["bar"]
        return {"P0": _bar_out(b, LB._mean([s["drop"] for s in b["per_seed"]]), _g(b, "pooled", "seeds_passing"))}
    if name == "savings":
        bs = {b["id"]: b for b in LB.savings(results, True, "", rng)["bars"]}
        g, r = bs["SAV-G"], bs["SAV-dR"]
        return {"SAV-G": _bar_out(g, _g(g, "pooled", "perm_stat"), _g(g, "pooled", "seeds_positive")),
                "SAV-dR": _bar_out(r, _g(r, "pooled", "mean_dR_null_sub"), _g(r, "pooled", "seeds_ge_1.5"))}
    if name == "transfer":
        bs = {b["id"]: b for b in LB.transfer(results, rng)["bars"]}
        t, k, c = bs["TRF-T4"], bs["TRF-K"], bs["TRF-C"]
        k4, k1 = _g(k, "pooled", "mean_T_k4"), _g(k, "pooled", "mean_T_k1")
        return {"TRF-T4": _bar_out(t, _g(t, "pooled", "mean_T"), _g(t, "pooled", "seeds_positive")),
                "TRF-K": _bar_out(k, None if k4 is None or k1 is None else k4 - k1, None),
                "TRF-C": _bar_out(c, _g(c, "pooled", "mean_reduction"), _g(c, "pooled", "seeds_reduced"))}
    if name == "cue":
        bs = {b["id"]: b for b in LB.cue(results, True, rng)["bars"]}
        r, d = bs["CUE-R"], bs["CUE-D"]
        cells = list(r["pooled"].values())
        rc = [c["R_cue"] for c in cells]
        stat = None if any(x is None for x in rc) else _min(*rc)      # a missing headline cell -> no statistic
        return {"CUE-R": _bar_out(r, stat, _min(*[c["seeds_positive"] for c in cells])),
                "CUE-D": _bar_out(d, _g(d, "pooled", "diff"), None)}
    raise ValueError(name)


ANALYSE_ORDER = ("p0", "savings", "cue", "transfer")   # life_bars.analyse draws savings, cue, transfer from one rng in this order
P_BARS = ("SAV-G", "TRF-T4", "CUE-R")                    # the permutation-p bars: the percentile / false-pass rule applies per bar


def _score(sections, results, seed) -> dict:
    """Observed run: ONE shared rng, sections scored in life_bars.analyse order, so verdicts reproduce life_bars exactly."""
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(sections, key=ANALYSE_ORDER.index):
        out[s] = score_section(s, results, rng)
    return {s: out[s] for s in sections}


def _with_inner(n: int, f):
    keep = LB.N_PERM
    LB.N_PERM = n
    try:
        return f()
    finally:
        LB.N_PERM = keep


# ---------------------------------------------------------------- the null run
def _pct(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0, "mean": None, "p95": None, "p99": None}
    a = np.asarray(v, float)
    return {"n": len(v), "mean": round(float(a.mean()), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "p99": round(float(np.percentile(a, 99)), 4)}


def _trust(rate: float, se0: float) -> str:
    """'yes' if the false-pass rate is clearly <= 0.05, 'no' if clearly above, 'inconclusive' within 2 SE (binomial at K) of 0.05."""
    if abs(rate - 0.05) <= 2 * se0:
        return "inconclusive"
    return "yes" if rate < 0.05 else "no"


def run_null(results: list, sections=SECTIONS, k: int = K_DEFAULT, inner: int = INNER_PERM, seed: int = MASTER_SEED,
             progress=None) -> dict:
    results = sorted(results, key=lambda r: r["seed"])
    sections = [s for s in sections if s in SECTIONS]
    obs_reg = _score(sections, results, LB.RNG_SEED)                         # registered N_PERM, as life_bars reports it
    obs = _with_inner(inner, lambda: _score(sections, results, LB.RNG_SEED))  # same inner N as the null (rank basis)
    master = np.random.default_rng(seed)
    reps = []
    for i in range(k):
        shuf = np.random.default_rng(master.integers(2**32))
        rep = {}
        for s in sections:
            lives = [SHUFFLE[s](res, shuf) for res in results]
            rep[s] = _with_inner(inner, lambda: score_section(s, lives, np.random.default_rng(int(master.integers(2**32)))))
        reps.append(rep)
        if progress:
            progress(i + 1, k)
    bars, comp = {}, {}
    for s in sections:
        for bid in obs[s]:
            nv = [r[s][bid]["stat"] for r in reps]
            o = obs[s][bid]["stat"]
            nn = [x for x in nv if x is not None]
            cnt = {}
            for r in reps:
                c = r[s][bid]["count"]
                cnt[str(c)] = cnt.get(str(c), 0) + 1
            fp = sum(r[s][bid]["verdict"] == "PASS" for r in reps) / k
            ge = None if o is None or not nn else (1 + sum(x >= o - 1e-12 for x in nn)) / (1 + len(nn))
            st = _pct(nv)
            bars[bid] = {"section": s, "observed_verdict": obs_reg[s][bid]["verdict"],
                         "observed_verdict_at_null_inner_perm": obs[s][bid]["verdict"], "observed_stat": o,
                         "observed_seed_count": obs[s][bid]["count"], "null_stat": st, "false_pass_rate": fp,
                         "frac_ge": None if ge is None else round(ge, 4),
                         "rank": None if o is None or not nn else 1 + sum(x > o + 1e-12 for x in nn),
                         "beyond_p95": None if o is None or st["p95"] is None else o >= st["p95"],
                         "beyond_p99": None if o is None or st["p99"] is None else o >= st["p99"],
                         "null_seed_count": dict(sorted(cnt.items()))}
        fps = [all(v["verdict"] == "PASS" for v in r[s].values()) for r in reps]
        comp[s] = {"bars": list(obs[s]), "false_pass_rate": sum(fps) / k,
                   "observed_composite": ("PASS" if all(v["verdict"] == "PASS" for v in obs_reg[s].values()) else "not PASS")}
    allp = [all(all(v["verdict"] == "PASS" for v in r[s].values()) for s in sections) for r in reps]
    comp["all"] = {"bars": [b for s in sections for b in obs[s]], "false_pass_rate": sum(allp) / k,
                   "observed_composite": ("PASS" if all(c["observed_composite"] == "PASS" for c in comp.values()) else "not PASS")}
    se0 = float(np.sqrt(0.05 * 0.95 / k))      # Monte Carlo SE of a rate measured at K replicates when the true rate is 0.05
    for c in comp.values():
        c["false_pass_se"] = round(float(np.sqrt(c["false_pass_rate"] * (1 - c["false_pass_rate"]) / k)), 4)
        c["mc_margin_2se"] = round(2 * se0, 4)
        c["trusted_rule"] = _trust(c["false_pass_rate"], se0)
        c["note"] = "upper bound: the all-bars-PASS composite is not the registered per-bar rule"
    for bid, b in bars.items():
        if bid in P_BARS:
            b["trusted_rule"] = _trust(b["false_pass_rate"], se0)
            b["false_pass_se"] = round(float(np.sqrt(b["false_pass_rate"] * (1 - b["false_pass_rate"]) / k)), 4)
    return {"K": k, "inner_perm": inner, "registered_n_perm": LB.N_PERM, "master_seed": seed, "sections": sections,
            "seeds": [r["seed"] for r in results], "n_seeds": len(results), "bars": bars, "composite": comp}


# ---------------------------------------------------------------- synthetic pseudo-seeds (SIMULATED)
def bootstrap_life(res: dict, rng, seed_id: int) -> dict:
    """Item-level bootstrap of one life's pools into a pseudo-seed: within each ledger class (AB split into A/B by pruned_a),
    items are drawn with replacement, copies are renamed id~m, and every row keyed by a ledger item is expanded to its copies
    (rows of unsampled items drop; rows of non-ledger items, e.g. never-told questions, stay). Transfer per-question lists
    are resampled within domain."""
    led = res["ledger"]
    pa = res.get("pruned_a") or {}
    cls: dict = {}
    for i, e in led.items():
        c = e["cls"] if e["cls"] != "AB" else ("AB-A" if i in pa.get("A", ()) else "AB-B" if i in pa.get("B", ()) else "AB")
        cls.setdefault(c, []).append(i)
    mp: dict = {i: [] for i in led}
    for items in cls.values():
        for j in rng.integers(0, len(items), len(items)):
            i = items[int(j)]
            mp[i].append(i if not mp[i] else f"{i}~{len(mp[i])}")

    def ex(rows):
        o = []
        for r in rows:
            it = r.get("item")
            if it in led:
                o += [{**r, "item": n} for n in mp[it]]
            else:
                o.append(r)
        return o

    def ids(v):
        return [n for i in v for n in mp.get(i, [i])]
    new = dict(res)
    new["seed"] = seed_id
    new["ledger"] = {n: copy.deepcopy(led[i]) for i, ns in mp.items() for n in ns}
    if pa:
        new["pruned_a"] = {**pa, "A": ids(pa.get("A", [])), "B": ids(pa.get("B", [])), "sameday_items": ids(pa.get("sameday_items", []))}
    probes = []
    for p in res["probes"]:
        q = dict(p)
        for f in ("rows", "span", "age_ladder"):
            if f in q and q[f]:
                q[f] = ex(q[f])
        tr = q.get("transfer")
        if tr and tr.get("per", {}).get("sysadmin") and tr["per"].get("home repair"):
            per = {d: [float(v[int(j)]) for j in rng.integers(0, len(v), len(v))] for d, v in tr["per"].items()}
            q["transfer"] = _retransfer(tr, per)
        probes.append(q)
    new["probes"] = probes
    new["cue_probes"] = [{**pr, "rows": ex(pr["rows"])} for pr in res.get("cue_probes", [])]
    new["schedule"] = [{**d, "turns": ex(d.get("turns", []))} for d in res.get("schedule", [])]
    return new


def synthetic_lives(lives: list, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    lives = sorted(lives, key=lambda r: r["seed"])
    return [bootstrap_life(lives[j % len(lives)], rng, 9000 + j) for j in range(n)]


# ---------------------------------------------------------------- output
def _f(x):
    return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render(R: dict) -> str:
    sim = R.get("synthetic")
    L_ = [f"# Life bars permutation null ({R['n_seeds']} seeds: {R['seeds']})"
          + (" - SIMULATED, bootstrap of the given lives' item pools, not data" if sim else ""),
          f"\nK={R['K']} null replicates, inner permutations {R['inner_perm']} (registered observed run: {R['registered_n_perm']}), "
          f"master seed {R['master_seed']}. Labels shuffled within seed; life_bars scoring unmodified.\n",
          "## Per bar\n", "| bar | observed verdict | obs stat | null mean | null p95 | null p99 | frac_ge | rank | beyond p95 | beyond p99 | false-pass | rule (<=0.05, +-2 SE MC) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for bid, b in R["bars"].items():
        n = b["null_stat"]
        L_.append(f"| {bid} | {b['observed_verdict']} | {_f(b['observed_stat'])} | {_f(n['mean'])} | {_f(n['p95'])} | {_f(n['p99'])} "
                  f"| {_f(b['frac_ge'])} | {_f(b['rank'])} | {b['beyond_p95']} | {b['beyond_p99']} | {_f(b['false_pass_rate'])} | {b.get('trusted_rule', 'n/a (not a p-value bar)')} |")
    L_ += ["\n## Composite verdicts (all label-dependent bars of the section PASS) - UPPER BOUND ONLY; the rule is applied per bar above\n",
           "| section | observed | null false-pass rate | SE | rule (<=0.05, +-2 SE MC) |", "|---|---|---|---|---|"]
    for s, c in R["composite"].items():
        L_.append(f"| {s} | {c['observed_composite']} | {_f(c['false_pass_rate'])} | {_f(c['false_pass_se'])} | {c['trusted_rule']} |")
    L_ += ["\n## Null distribution of the seed count (count: replicates)\n"]
    for bid, b in R["bars"].items():
        L_.append(f"- {bid} (observed {_f(b['observed_seed_count'])}): {b['null_seed_count']}")
    return "\n".join(L_) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="life json files (explicit paths)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--inner-perm", type=int, default=INNER_PERM)
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--synthetic-seeds", type=int, default=0, help="N>0: bootstrap N pseudo-seeds from the lives (SIMULATED)")
    a = ap.parse_args(argv)
    lives = LB.load(a.paths)
    if a.synthetic_seeds:
        lives = synthetic_lives(lives, a.synthetic_seeds, a.seed)
    t0 = time.time()
    R = run_null(lives, a.sections.split(","), a.k, a.inner_perm, a.seed,
                 progress=lambda i, k: print(f"null {i}/{k} ({time.time() - t0:.0f}s)", file=sys.stderr) if i % 20 == 0 else None)
    if a.synthetic_seeds:
        R["synthetic"] = f"SIMULATED: {a.synthetic_seeds} pseudo-seeds bootstrapped from {a.paths}"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    md = render(R)
    with open(a.out + ".md", "w") as fh:
        fh.write(md)
    print(md)
    return R


if __name__ == "__main__":
    main()
