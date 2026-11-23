# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md (sections 4, 5 and 7: LG-P1/P2, D2-PR2 primary, G1-G4, G6)
# Expected-fail: none (host-only post-hoc read of life JSONs); K-P is expected to FAIL the "language-drift fixed" bar (n15 KL <= 2x n1)
"""Arm vs B comparison for the Stage-2 TEST v2 (prereg 2026-11-23-stage2-test2-prereg.md, sections 4-5). HOST-ONLY: reads
life results JSONs, pairs arm and B lives by seed, and computes every vs-B bar this prereg gives to `life_compare.py`:

  LG-P1  night-15 lang_guard.kl: arm mean over seeds <= 0.5 x B mean, AND arm < B on >= 5/6 seeds.
  LG-P2  night-15 lang_guard.nll_on, per seed: NLL_arm(n15) - NLL_B(n1) <= 0.5 x (NLL_B(n15) - NLL_B(n1)) on >= 5/6.
         The arm's own night 1 and night 10 (same formula, B n10) are reported, not bars.
  LEAK   D2-PR2 primary (--d2pr2): stock leak (life_leak, nights 10+15, intact + hippocampus-off) below B on >= 5/6 AND
         >= 30% lower pooled, on all rows AND on non-refusal rows (G8-pattern replies dropped from numerator and denominator).
  G1     H with the hippocampus off, nights 5-15, mean of 6: arm >= B - 1.0, >= 5/6.
  G2a/b  hippocampus-off day-1-2 recall (d12), nights 1-4 (G2a) and 5-15 (G2b), rate x 6: arm >= B - 1.0, >= 5/6 each.
         Nights 1-15 reported.
  G3     collapse (life_bars.language, share >= 0.4): seeds-ever per probe arm <= B's and <= 3 (of 6).
  G4     stock leak (n10+15), all rows AND non-refusal rows: arm share not higher than B, pooled and on >= 5/6.
  G6     (--d2pr2 only) language-guard KL n15 <= 1.25 x B, per seed >= 5/6 (pooled ratio reported).
  LG-FIX the "language-drift defect fixed" bar: the arm's own n15 KL <= 2x n1 on >= 5/6 (reported; expected FAIL for K-P).
Under --d2pr2, LG-P1/LG-P2 are reported for that arm, not bars (prereg section 4); every reported-only verdict is printed
"(reported, not a bar)".

The hoff_series, d12, KL/NLL and leak computations are copied from
brain/experiments/results/2026-11-22-kp-dev-read-1.5b/kp_bars.py (and _scratch/stage2/read/d2pr2_bars.py, a relabelled
copy of the same code); the refusal rule is imported from brain/eval/life_costs.py (one frozen copy of the prereg
section 5 G8 regex). Pooled = mean over seeds for KL/NLL/H (ratio of means, as kp_bars) and sum(leaks)/sum(n) for the
leak (as kp_bars and the D2-PR2 read).

Seeds (Amendment T1; the rule is life_costs.seed_status, shared verbatim with life_costs). By default the six registered
seeds (42/43/44/100/101/102) are EXPECTED and need is a fixed 5 of 6. Lives of other seeds are ignored and listed. `--dev`
(expect_seeds=None) scores the seeds given (union of arm and B), need ceil(5n/6), every verdict labelled "DEV (n seeds)".
- UNDEFINED seed (neither pass nor fail, listed with the reason; dropped from every bar, pooled values included): B life
  missing, B life ended before night 15, or a setup mismatch: split, device / cuda_precision / torch / torch_cuda /
  gpu_name differing where recorded on both lives (absent on either = "provenance unrecorded": UNDEFINED in registered
  mode, reported only under --dev), or both-off
  rows differing. A bar with fewer than need scorable seeds is UNDEFINED.
- ARM life ended early (completed nights k < 15, a section 10 kill): scored on nights 1..k with B restricted to the same
  nights; a bar that needs a later night (night-15 KL/NLL, the n10+15 leak, an empty window) counts that seed as not
  passing. A missing arm life is not passing (flag None).

Verdict of a bar (count clause, optional pooled clause): a seed whose flag is None (unscored life, or a missing
lang_guard record) counts as not passing. FAIL if the count misses need or the pooled clause is False; PASS if the count
reaches need and the pooled clause is True; else (count holds, pooled clause undefined) UNDEFINED. A pooled clause is None (undefined) whenever any seed is unscored, because the
registered pooled value is over all six seeds.

Completed nights (life_costs.completed_nights): the longest run 1..k with the trunk night record, intact and
hippocampus-off trunk probes, and on nights 10/15 their interview probes. lang_guard records are checked per bar (a
missing one makes that bar's seed None). brain_kw / device / split / model / provenance of every life are in the header.

Leak edge rules: a seed with zero rows (after the refusal drop) on the arm side does not pass the D2-PR2 primary
("below B" is not shown) and does pass G4 (no leak rows, not higher than B); zero rows on the B side reads B's share as
0. B pooled 0 makes the primary's pooled clause False (no arm can be 30% lower than 0).

Sign-flip test (section 4 common rule): exact paired sign-flip over the per-seed differences d = arm - B (2^n flips, 64
for six seeds), one-sided in the arm-better direction: p = share of flips whose mean is <= the observed mean (for
"lower is better") or >= it (for "higher is better"), ties counted. It is REPORTED beside a bar with the number of
seeds it used (seeds with an undefined difference are dropped); it gates no verdict: the section 4/5 bars computed here
state no p condition. The pooled item-level permutation of section 4 is not used by any of these bars (section 5 never
asks for it); this reading goes into the freeze note.

python -m brain.eval.life_compare --arm-name KP --arm A_s1.json ... --b B_s1.json ... [--d2pr2] [--dev] [--out PREFIX]
(writes PREFIX.json and PREFIX.md; prints the markdown)"""
from __future__ import annotations

import argparse
import itertools
import json

from brain.eval import life_bars, life_leak
from brain.eval import life_costs
from brain.eval.life_costs import is_refusal  # noqa: F401  (the one frozen copy of the prereg G8 refusal rule)
from brain.eval.life_costs import LAST_NIGHT, UNDEF, completed_nights, provenance, restrict, seed_status

REGISTERED_SEEDS = (42, 43, 44, 100, 101, 102)
NIGHTS = tuple(range(1, 16))
LEAK_NIGHTS = (10, 15)
PROBE_ARMS = ("intact", "hippocampus-off")
P_LEVEL = 0.05
EPS = 1e-9


# ---------------------------------------------------------------- helpers copied from kp_bars.py
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def trunk(res):
    return [n for n in res["nights"] if n["line"] == "trunk"]


def nrec(res, n):
    return next((x for x in trunk(res) if x["night"] == n), None)


def trunk_probes(res, arm):
    return [p for p in res["probes"] if p["arm"] == arm and p.get("line", "trunk") == "trunk" and not p.get("tag")]


def lg(res, n):
    if res is None:
        return {}
    return (nrec(res, n) or {}).get("lang_guard") or {}


def hoff_series(res, lo=5, hi=15):
    if res is None:
        return {}
    return {p["night"]: p["summary"]["H_right"] / sum(1 for r in p["rows"] if r["kind"] == "H") * 6
            for p in trunk_probes(res, "hippocampus-off") if lo <= p["night"] <= hi}


def d12(res, lo, hi, arm="hippocampus-off"):
    if res is None:
        return 0, 0
    led, k, n = res["ledger"], 0, 0
    for p in trunk_probes(res, arm):
        if lo <= p["night"] <= hi:
            for r in p["rows"]:
                if r["kind"] in ("H", "rotating") and r.get("item") in led and min(led[r["item"]]["told_days"]) <= 2:
                    n += 1
                    k += bool(r.get("ok"))
    return k, n


# ---------------------------------------------------------------- completeness
def completeness(res) -> list:
    """Problems that make a life unscorable for these bars (empty list = complete)."""
    probs = []
    tn = {n["night"] for n in res.get("nights", []) if n.get("line") == "trunk"}
    miss = [n for n in NIGHTS if n not in tn]
    if miss:
        probs.append(f"trunk nights missing {miss}")
    for arm in PROBE_ARMS:
        pn = {p["night"] for p in res.get("probes", []) if p["arm"] == arm and p.get("line", "trunk") == "trunk" and not p.get("tag")}
        miss = [n for n in NIGHTS if n not in pn]
        if miss:
            probs.append(f"{arm} trunk probes missing nights {miss}")
        iv = {p["night"] for p in res.get("interview_probes", []) if p["arm"] == arm and (p.get("line") or "trunk") == "trunk"
              and not p.get("tag")}
        miss = [n for n in LEAK_NIGHTS if n not in iv]
        if miss:
            probs.append(f"{arm} interview probes missing nights {miss}")
    return probs


def life_info(res) -> dict:
    return dict({k: res.get(k) for k in ("brain_kw", "device", "split", "model", "days")}, provenance=provenance(res),
                completed_nights=completed_nights(res))


# ---------------------------------------------------------------- leak (refusal rule from life_costs)
def leak(res, nights=LEAK_NIGHTS, drop_refusals=False) -> dict:
    """life_leak.leak_table + pooled over `nights`, both arms; with drop_refusals the G8-pattern replies leave numerator
    and denominator (same rows, needles and identity exclusion as life_leak.leak_table)."""
    if not drop_refusals:
        return life_leak.pooled(life_leak.leak_table(res), set(nights))
    kw = res.get("brain_kw") or {}
    needles = life_leak.permanent_replies(kw.get("name") or life_leak.NAME, kw.get("friend") or life_leak.FRIEND)
    ident = life_leak.identity_items(res)
    n = k = 0
    for _src, _arm, night, r in life_leak._rows(res):
        if night in nights and r.get("item") not in ident and not is_refusal(r.get("brain", "")):
            n += 1
            k += life_leak.is_leak(r.get("brain", ""), needles)
    return {"n": n, "leaks": k, "share": k / n if n else None}


# ---------------------------------------------------------------- statistics
def sign_flip_p(diffs, lower_better=True) -> float | None:
    """Exact paired sign-flip test over the per-seed differences (2^n assignments), one-sided, ties counted.
    None differences are dropped (sign_flip_n reports how many were used)."""
    d = [x for x in diffs if x is not None]
    if not d:
        return None
    obs = sum(d) / len(d)
    hit = 0
    tot = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        m = sum(s * x for s, x in zip(signs, d)) / len(d)
        hit += (m <= obs + EPS) if lower_better else (m >= obs - EPS)
        tot += 1
    return hit / tot


def _sf(diffs, lower_better=True) -> dict:
    return {"sign_flip_p": sign_flip_p(diffs, lower_better), "sign_flip_n": sum(1 for x in diffs if x is not None)}


def count(flags: dict, need: int | None = None) -> dict:
    """>= 5/6 rule (life_bars.count_rule); None counts as not passing. flags carries the scorable seeds (UNDEFINED seeds
    are dropped); need is the fixed need of the expected seeds (5 of 6), or ceil(5n/6) of the flags when not given."""
    return life_bars.count_rule(flags, need)


def decide(c: dict, pooled_ok=True) -> str:
    """Undefined seeds count as not passing (count_rule). FAIL if the count misses need or the pooled clause is False;
    PASS if the count reaches need and the pooled clause is True; else (count holds, pooled undefined) UNDEFINED."""
    if not c["ok"] or pooled_ok is False:
        return "FAIL"
    if c["ok"] and pooled_ok is True:
        return "PASS"
    return "UNDEFINED"


def combine(verdicts) -> str:
    v = list(verdicts)
    return "FAIL" if "FAIL" in v else ("UNDEFINED" if "UNDEFINED" in v or not v else "PASS")


def pair(arm_lives: list, b_lives: list) -> tuple:
    a = {r["seed"]: r for r in arm_lives}
    b = {r["seed"]: r for r in b_lives}
    if len(a) != len(arm_lives) or len(b) != len(b_lives):
        raise ValueError("duplicate seed among the arm or B lives")
    seeds = sorted(set(a) & set(b))
    return seeds, a, b, sorted(set(a) ^ set(b))


def _perseed(seeds, flags, raw):
    return {s: dict(raw[s], ok=flags[s]) for s in seeds}


def _has(s, A, B):
    return s in A and s in B


# ---------------------------------------------------------------- bars
def lg_p1(seeds, A, B, night=15, need=None) -> dict:
    raw = {s: {"arm": lg(A.get(s), night).get("kl"), "b": lg(B.get(s), night).get("kl")} for s in seeds}
    flags = {s: None if None in (v["arm"], v["b"]) else v["arm"] < v["b"] for s, v in raw.items()}
    ma, mb = mean([v["arm"] for v in raw.values()]), mean([v["b"] for v in raw.values()])
    c = count(flags, need)
    pooled_ok = None if None in (ma, mb) or any(f is None for f in flags.values()) else ma <= 0.5 * mb + EPS
    diffs = [None if None in (v["arm"], v["b"]) else v["arm"] - v["b"] for v in raw.values()]
    return {"id": "LG-P1" if night == 15 else f"LG-P1 (night {night}, reported)", "night": night,
            "per_seed": _perseed(seeds, flags, raw), "count": c,
            "pooled": {"arm": ma, "b": mb, "ratio": ma / mb if ma is not None and mb else None, "half_b": 0.5 * mb if mb is not None else None,
                       "ok": pooled_ok},
            **_sf(diffs), "verdict": decide(c, pooled_ok)}


def lg_p2(seeds, A, B, night=15, need=None) -> dict:
    raw, flags, diffs = {}, {}, []
    for s in seeds:
        a, b1, bn, a1 = (lg(A.get(s), night).get("nll_on"), lg(B.get(s), 1).get("nll_on"), lg(B.get(s), night).get("nll_on"),
                         lg(A.get(s), 1).get("nll_on"))
        if None in (a, b1, bn):
            raw[s] = {"arm": a, "b_n1": b1, "b": bn, "arm_n1": a1, "rise_arm": None, "half_rise_b": None}
            flags[s] = None
            diffs.append(None)
            continue
        rise, half = a - b1, 0.5 * (bn - b1)
        raw[s] = {"arm": a, "b_n1": b1, "b": bn, "arm_n1": a1, "rise_arm": rise, "half_rise_b": half}
        flags[s] = rise <= half + EPS
        diffs.append(a - bn)
    c = count(flags, need)
    return {"id": "LG-P2" if night == 15 else f"LG-P2 (night {night}, reported)", "night": night,
            "per_seed": _perseed(seeds, flags, raw), "count": c,
            "pooled": {"arm": mean([v["arm"] for v in raw.values()]), "b": mean([v["b"] for v in raw.values()]),
                       "b_n1": mean([v["b_n1"] for v in raw.values()]), "arm_n1": mean([v["arm_n1"] for v in raw.values()])},
            **_sf(diffs), "verdict": decide(c)}


def lg_fix(seeds, A, need=None) -> dict:
    raw = {s: {"n1": lg(A.get(s), 1).get("kl"), "n15": lg(A.get(s), 15).get("kl")} for s in seeds}
    for v in raw.values():
        v["ratio"] = v["n15"] / v["n1"] if v["n1"] and v["n15"] is not None else None
    flags = {s: None if v["ratio"] is None else v["n15"] <= 2 * v["n1"] + EPS for s, v in raw.items()}
    c = count(flags, need)
    return {"id": "LG-FIX", "per_seed": _perseed(seeds, flags, raw), "count": c, "verdict": decide(c)}


def _leak_rows(seeds, A, B, drop):
    raw = {}
    for s in seeds:
        if not _has(s, A, B) or A[s].get("_completed_nights", LAST_NIGHT) < max(LEAK_NIGHTS):
            raw[s] = {"arm": None, "b": None, "arm_k": None, "arm_n": None, "b_k": None, "b_n": None}
            continue   # missing arm life, or it ended before night 15: not passing
        a, b = leak(A[s], drop_refusals=drop), leak(B[s], drop_refusals=drop)
        raw[s] = {"arm": a["share"], "b": b["share"], "arm_k": a["leaks"], "arm_n": a["n"], "b_k": b["leaks"], "b_n": b["n"]}
    sc = [v for v in raw.values() if v["arm_n"] is not None]
    ak, an = sum(v["arm_k"] for v in sc), sum(v["arm_n"] for v in sc)
    bk, bn = sum(v["b_k"] for v in sc), sum(v["b_n"] for v in sc)
    pa, pb = (ak / an if an else None), (bk / bn if bn else None)
    complete = len(sc) == len(raw)
    pooled = {"arm_k": ak, "arm_n": an, "b_k": bk, "b_n": bn, "arm": pa, "b": pb,
              "reduction": (1 - pa / pb) if pa is not None and pb else None, "complete": complete}
    diffs = [None if None in (v["arm"], v["b"]) else v["arm"] - v["b"] for v in raw.values()]
    return raw, pooled, diffs


def _b_share(v):
    return 0.0 if v["b_n"] == 0 else v["b"]


def leak_primary(seeds, A, B, need=None) -> dict:
    out = {"id": "D2-PR2 leak primary", "rows": {}}
    for label, drop in (("all", False), ("non_refusal", True)):
        raw, pooled, diffs = _leak_rows(seeds, A, B, drop)
        # missing seed: None; arm with zero rows: False ("below B" not shown); B with zero rows: B share 0
        flags = {s: None if v["arm_n"] is None else (False if v["arm_n"] == 0 else v["arm"] < _b_share(v)) for s, v in raw.items()}
        c = count(flags, need)
        if not pooled["complete"] or pooled["arm"] is None:
            pooled_ok = None if not pooled["complete"] else False
        elif not pooled["b"]:
            pooled_ok = False                                   # B pooled 0 (or no B rows): no arm is 30% lower than 0
        else:
            pooled_ok = pooled["reduction"] >= 0.30 - EPS
        out["rows"][label] = {"id": f"{out['id']} ({label})", "per_seed": _perseed(seeds, flags, raw), "count": c,
                              "pooled": dict(pooled, ok=pooled_ok), **_sf(diffs), "verdict": decide(c, pooled_ok)}
    out["verdict"] = combine(r["verdict"] for r in out["rows"].values())
    return out


def g4(seeds, A, B, need=None) -> dict:
    out = {"id": "G4", "rows": {}}
    for label, drop in (("all", False), ("non_refusal", True)):
        raw, pooled, diffs = _leak_rows(seeds, A, B, drop)
        # missing seed: None; arm with zero rows: True (no leak rows); B with zero rows: B share 0
        flags = {s: None if v["arm_n"] is None else (True if v["arm_n"] == 0 else v["arm"] <= _b_share(v) + EPS) for s, v in raw.items()}
        c = count(flags, need)
        if not pooled["complete"]:
            pooled_ok = None
        else:
            pa = pooled["arm"] or 0.0
            pb = pooled["b"] or 0.0
            pooled_ok = pa <= pb + EPS
        out["rows"][label] = {"id": f"{out['id']} ({label})", "per_seed": _perseed(seeds, flags, raw), "count": c,
                              "pooled": dict(pooled, ok=pooled_ok), **_sf(diffs), "verdict": decide(c, pooled_ok)}
    out["verdict"] = combine(r["verdict"] for r in out["rows"].values())
    return out


def g1(seeds, A, B, margin=1.0, need=None) -> dict:
    raw = {s: {"arm": mean(hoff_series(A.get(s)).values()) if _has(s, A, B) else None,
               "b": mean(hoff_series(B.get(s)).values()) if _has(s, A, B) else None} for s in seeds}
    flags = {s: None if None in (v["arm"], v["b"]) else v["arm"] - v["b"] >= -margin - EPS for s, v in raw.items()}
    for v in raw.values():
        v["diff"] = None if None in (v["arm"], v["b"]) else v["arm"] - v["b"]
    c = count(flags, need)
    return {"id": "G1", "margin": margin, "per_seed": _perseed(seeds, flags, raw), "count": c,
            "pooled": {"arm": mean([v["arm"] for v in raw.values()]), "b": mean([v["b"] for v in raw.values()])},
            **_sf([v["diff"] for v in raw.values()], lower_better=False), "verdict": decide(c)}


def g2(seeds, A, B, lo, hi, bid, margin=1.0, need=None) -> dict:
    raw = {}
    for s in seeds:
        ka, na = d12(A.get(s), lo, hi) if _has(s, A, B) else (0, 0)
        kb, nb = d12(B.get(s), lo, hi) if _has(s, A, B) else (0, 0)
        ra, rb = (ka / na * 6 if na else None), (kb / nb * 6 if nb else None)
        raw[s] = {"arm": ra, "b": rb, "arm_k": ka, "arm_n": na, "b_k": kb, "b_n": nb,
                  "diff": None if None in (ra, rb) else ra - rb}
    flags = {s: None if v["diff"] is None else v["diff"] >= -margin - EPS for s, v in raw.items()}
    c = count(flags, need)
    return {"id": bid, "nights": [lo, hi], "margin": margin, "per_seed": _perseed(seeds, flags, raw), "count": c,
            "pooled": {"arm": mean([v["arm"] for v in raw.values()]), "b": mean([v["b"] for v in raw.values()])},
            **_sf([v["diff"] for v in raw.values()], lower_better=False), "verdict": decide(c)}


def g3(seeds, A, B, cap=3) -> dict:
    """seeds-ever per probe arm <= B's and <= cap. With an unscored seed the comparison is incomplete: an arm already over
    the cap FAILS, otherwise that probe arm is undefined."""
    scored = [s for s in seeds if _has(s, A, B)]
    complete = len(scored) == len(seeds)
    LA = life_bars.language([A[s] for s in scored])["collapse_arms"] if scored else {}
    LB = life_bars.language([B[s] for s in scored])["collapse_arms"] if scored else {}
    arms = {}
    for arm in sorted(set(LA) | set(LB)):
        a, b = LA.get(arm), LB.get(arm)
        ea, eb = (a or {}).get("seeds_ever"), (b or {}).get("seeds_ever")
        if ea is not None and ea > cap:
            ok = False
        elif None in (ea, eb) or not complete:
            ok = None
        else:
            ok = ea <= eb
        arms[arm] = {"arm_seeds_ever": ea, "b_seeds_ever": eb,
                     "arm_max_share": (a or {}).get("per_seed_max_share"), "b_max_share": (b or {}).get("per_seed_max_share"), "ok": ok}
    oks = [v["ok"] for v in arms.values()]
    core = "FAIL" if False in oks else ("UNDEFINED" if not oks or None in oks else "PASS")
    return {"id": "G3", "cap": cap, "per_arm": arms, "complete": complete, "verdict": core}


def g6(seeds, A, B, factor=1.25, need=None) -> dict:
    raw = {s: {"arm": lg(A.get(s), 15).get("kl"), "b": lg(B.get(s), 15).get("kl")} for s in seeds}
    for v in raw.values():
        v["ratio"] = v["arm"] / v["b"] if v["arm"] is not None and v["b"] else None
    flags = {s: None if v["ratio"] is None else v["arm"] <= factor * v["b"] + EPS for s, v in raw.items()}
    ma, mb = mean([v["arm"] for v in raw.values()]), mean([v["b"] for v in raw.values()])
    c = count(flags, need)
    return {"id": "G6", "factor": factor, "per_seed": _perseed(seeds, flags, raw), "count": c,
            "pooled": {"arm": ma, "b": mb, "ratio": ma / mb if ma is not None and mb else None}, "verdict": decide(c)}


def _label_all(bar: dict, dev_n: int | None):
    """verdict_core = PASS/FAIL/UNDEFINED; verdict = the same, or 'DEV (n seeds) <core>' off the registered six."""
    for x in [bar] + list(bar.get("rows", {}).values()):
        x["verdict_core"] = x["verdict"]
        if dev_n is not None:
            x["verdict"] = f"DEV ({dev_n} seeds) {x['verdict_core']}"


def _mark_undefined(bar: dict, undefined: dict, n_scorable: int, need: int):
    """Add the UNDEFINED seeds to a bar's per-seed table (and its row clauses), and make the bar UNDEFINED when fewer
    than need seeds are scorable."""
    xs = [bar] + list(bar.get("rows", {}).values())
    for x in xs:
        if "per_seed" in x:
            for s, why in undefined.items():
                x["per_seed"][s] = {"ok": UNDEF, "undefined": why}
            x["per_seed"] = dict(sorted(x["per_seed"].items()))
        if "count" in x:
            x["count"] = dict(x["count"], undefined=len(undefined), scorable=n_scorable)
        if n_scorable < need:
            x["verdict"] = UNDEF
    bar["undefined_seeds"] = sorted(undefined)


def compare(arm_lives: list, b_lives: list, arm_name: str = "arm", d2pr2: bool = False,
            expect_seeds=REGISTERED_SEEDS, setup_check: bool = True) -> dict:
    """expect_seeds: the seeds every bar is scored over (default the six registered; need fixed at 5).
    None = DEV: score the seeds given, need ceil(5n/6), verdicts labelled DEV. setup_check False is a diagnostic (setup
    mismatches are reported in the notes, not applied); a TEST read never sets it."""
    _, A_all, B_all, unpaired = pair(arm_lives, b_lives)
    if expect_seeds is None:
        seeds = sorted(set(A_all) | set(B_all))
        ignored = []
    else:
        seeds = sorted(expect_seeds)
        ignored = sorted((set(A_all) | set(B_all)) - set(seeds))
    st = seed_status(seeds, A_all, B_all, setup_check, registered=expect_seeds is not None)
    undefined = {s: st[s]["undefined"] for s in seeds if st[s]["undefined"]}
    scored = [s for s in seeds if s not in undefined]
    A, B = {}, {}
    for s in scored:
        k = st[s]["arm_k"] if s in A_all else LAST_NIGHT
        if s in A_all:
            A[s] = restrict(A_all[s], k)
        B[s] = restrict(B_all[s], k)
    unscored = {s: [n for n in st[s]["notes"] if n.startswith("arm life")] for s in scored
                if st[s]["arm_missing"] or st[s]["arm_k"] < LAST_NIGHT}
    need = life_bars._need(len(seeds))
    lives = {s: {"arm": life_info(A_all[s]) if s in A_all else None, "b": life_info(B_all[s]) if s in B_all else None}
             for s in sorted(set(A_all) | set(B_all))}
    same_kw = [s for s in seeds if s in A_all and s in B_all and A_all[s].get("brain_kw") == B_all[s].get("brain_kw")]
    kw = {"need": need}
    bars = {"LG-P1": lg_p1(scored, A, B, **kw), "LG-P1 n10 (reported)": lg_p1(scored, A, B, 10, **kw),
            "LG-P2": lg_p2(scored, A, B, **kw), "LG-P2 n10 (reported)": lg_p2(scored, A, B, 10, **kw),
            "LG-FIX": lg_fix(scored, A, **kw),
            "G1": g1(scored, A, B, **kw), "G2a": g2(scored, A, B, 1, 4, "G2a", **kw),
            "G2b": g2(scored, A, B, 5, 15, "G2b", **kw),
            "G2 n1-15 (reported)": g2(scored, A, B, 1, 15, "G2 n1-15 (reported)", **kw),
            "G3": g3(scored, A, B), "G4": g4(scored, A, B, **kw)}
    if d2pr2:
        bars["LEAK"] = leak_primary(scored, A, B, **kw)
        bars["G6"] = g6(scored, A, B, **kw)
    guards = ["G1", "G2a", "G2b", "G3", "G4"] + (["G6"] if d2pr2 else [])
    primary = ["LEAK"] if d2pr2 else ["LG-P1", "LG-P2"]
    dev_n = len(seeds) if expect_seeds is None or len(seeds) != 6 else None
    for k, x in bars.items():
        x["role"] = "primary" if k in primary else ("guard" if k in guards else "reported")
        _mark_undefined(x, undefined, len(scored), need)
        _label_all(x, dev_n)
    return {"arm_name": arm_name, "d2pr2": d2pr2, "mode": "DEV" if dev_n is not None else "registered",
            "seeds": seeds, "unpaired_seeds": unpaired, "ignored_seeds": ignored, "unscored": unscored,
            "undefined": undefined, "scorable_seeds": scored, "setup_check": setup_check,
            "notes": {s: st[s]["notes"] for s in seeds if st[s]["notes"]},
            "need": need, "lives": lives, "same_brain_kw_seeds": same_kw,
            "bars": bars, "primary": {k: bars[k]["verdict"] for k in primary}, "guards": {k: bars[k]["verdict"] for k in guards}}


# ---------------------------------------------------------------- render
def _f(x, k=3):
    return "-" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))


def _vline(x, role, unscored) -> str:
    us = f"; unscored seeds (not passing): {sorted(unscored)}" if unscored else ""
    if role == "reported":
        return f"{x['id']}: {x['verdict']} (reported, not a bar){us}"
    if role == "clause":
        return f"{x['id']}: {x['verdict']} (clause of the bar below){us}"
    return f"**{x['id']}: {x['verdict']}**{us}"


def _seed_table(bar, cols, role, unscored, k=3):
    L = ["| seed | " + " | ".join(c[0] for c in cols) + " | pass |", "|---|" + "---|" * (len(cols) + 1)]
    for s, v in bar["per_seed"].items():
        if v["ok"] == UNDEF:
            L.append(f"| {s} | " + " | ".join("-" for _ in cols) + f" | UNDEFINED ({'; '.join(v['undefined'])}) |")
            continue
        L.append(f"| {s} | " + " | ".join(_f(v.get(c[1]), k) if not callable(c[1]) else c[1](v) for c in cols) + f" | {v['ok']} |")
    c = bar["count"]
    sf = f"; sign-flip p (one-sided, arm better, reported) {_f(bar['sign_flip_p'], 4)} over {bar['sign_flip_n']} seeds" if "sign_flip_p" in bar else ""
    ud = f" ({c['undefined']} UNDEFINED seeds dropped)" if c.get("undefined") else ""
    L.append(f"\nCount {c['count']}/{c['n']}{ud} (need {c['need']}){sf}; {_vline(bar, role, unscored)}\n")
    return L


def render(r: dict) -> str:
    a = r["arm_name"]
    b = r["bars"]
    us = r["unscored"]
    L = [f"# {a} vs B (test2 prereg sections 4-5, `life_compare.py`)\n",
         f"Mode: {r['mode']}. Seeds scored over: {r['seeds']}; >= rule need {r['need']}/{len(r['seeds'])}. "
         f"UNDEFINED seeds (neither pass nor fail): {r['undefined'] or 'none'}. "
         f"Arm missing or ended early (scored on completed nights; not passing where a later night is needed): {us or 'none'}. "
         f"Unpaired lives: {r['unpaired_seeds'] or 'none'}. "
         f"Ignored lives (seed not expected): {r['ignored_seeds'] or 'none'}.\n",
         f"Primary: {r['primary']}. Guards: {r['guards']}.\n",
         ("" if r["setup_check"] else "**Setup check OFF (diagnostic): verdicts are not valid reads.**\n"),
         "| seed | side | split | device | provenance | completed nights | model | days | brain_kw |", "|---|---|---|---|---|---|---|---|---|"]
    for s, d in r["lives"].items():
        for side in ("arm", "b"):
            i = d[side]
            if i is None:
                L.append(f"| {s} | {side} | - | - | - | - | - | - | (no life) |")
            else:
                pv = ", ".join(f"{k}={v}" for k, v in i["provenance"].items() if k != "device") or "unrecorded"
                L.append(f"| {s} | {side} | {i['split']} | {i['device']} | {pv} | {i['completed_nights']} | {i['model']} | {i['days']} | "
                         f"{json.dumps(i['brain_kw'], sort_keys=True)} |")
    for sd, ns in r["notes"].items():
        L.append(f"\nNotes s{sd}: " + "; ".join(ns))
    if r["same_brain_kw_seeds"]:
        L.append(f"\nNote: arm and B lives have identical brain_kw on seeds {r['same_brain_kw_seeds']} (expected only for B-vs-B).")
    L.append("")
    for key in ("LG-P1", "LG-P1 n10 (reported)"):
        x = b[key]
        L.append(f"## {x['id']}: lang_guard.kl night {x['night']} (arm < B per seed; pooled arm <= 0.5 x B)\n")
        L += _seed_table(x, [(f"{a} KL", "arm"), ("B KL", "b")], x["role"], us, k=4)
        p = x["pooled"]
        L.append(f"Pooled (mean over seeds) {a} {_f(p['arm'], 4)} vs B {_f(p['b'], 4)}, ratio {_f(p['ratio'], 3)}, half of B {_f(p['half_b'], 4)}, pooled ok {p['ok']}.\n")
    for key in ("LG-P2", "LG-P2 n10 (reported)"):
        x = b[key]
        L.append(f"## {x['id']}: NLL_arm(n{x['night']}) - NLL_B(n1) <= 0.5 x (NLL_B(n{x['night']}) - NLL_B(n1))\n")
        L += _seed_table(x, [(f"{a} NLL n{x['night']}", "arm"), (f"{a} own n1", "arm_n1"), ("B n1", "b_n1"), (f"B n{x['night']}", "b"),
                             (f"{a} rise", "rise_arm"), ("0.5 x B rise", "half_rise_b")], x["role"], us)
    x = b["LG-FIX"]
    L.append("## LG-FIX: the arm's own n15 KL <= 2x n1 ('language-drift defect fixed')\n")
    L += _seed_table(x, [("KL n1", "n1"), ("KL n15", "n15"), ("ratio", "ratio")], x["role"], us, k=4)
    x = b["G1"]
    L.append("## G1: H hippocampus-off nights 5-15, mean of 6 (arm >= B - 1.0)\n")
    L += _seed_table(x, [(a, "arm"), ("B", "b"), ("diff", "diff")], x["role"], us)
    L.append(f"Pooled {a} {_f(x['pooled']['arm'])} vs B {_f(x['pooled']['b'])}.\n")
    for key in ("G2a", "G2b", "G2 n1-15 (reported)"):
        x = b[key]
        L.append(f"## {key}: hippocampus-off day-1-2 recall, nights {x['nights'][0]}-{x['nights'][1]}, rate x 6 (arm >= B - 1.0)\n")
        L += _seed_table(x, [(f"{a} k/n", lambda v: f"{v['arm_k']}/{v['arm_n']}"), (f"{a} x6", "arm"),
                             ("B k/n", lambda v: f"{v['b_k']}/{v['b_n']}"), ("B x6", "b"), ("diff", "diff")], x["role"], us, k=2)
    x = b["G3"]
    L.append(f"## G3: collapse seeds-ever per probe arm (arm <= B and <= {x['cap']})\n")
    L.append(f"| probe arm | {a} seeds-ever | B seeds-ever | {a} max share per seed | B max share per seed | ok |")
    L.append("|---|---|---|---|---|---|")
    for arm, v in x["per_arm"].items():
        ms = lambda d: ", ".join(f"{s}:{_f(m, 2)}" for s, m in (d or {}).items())
        L.append(f"| {arm} | {v['arm_seeds_ever']} | {v['b_seeds_ever']} | {ms(v['arm_max_share'])} | {ms(v['b_max_share'])} | {v['ok']} |")
    L.append(f"\n{_vline(x, x['role'], us)}\n")
    blocks = [("G4", "G4: stock leak n10+15, arm not higher than B (per seed and pooled)")]
    if r["d2pr2"]:
        blocks.insert(0, ("LEAK", "D2-PR2 primary: stock leak n10+15 below B per seed and >= 30% lower pooled"))
    for key, title in blocks:
        x = b[key]
        L.append(f"## {title}\n")
        for label, row in x["rows"].items():
            L.append(f"### {label} rows\n")
            L += _seed_table(row, [(f"{a} k/n", lambda v: f"{v['arm_k']}/{v['arm_n']}"), (f"{a} share", "arm"),
                                   ("B k/n", lambda v: f"{v['b_k']}/{v['b_n']}"), ("B share", "b")], "clause", us)
            p = row["pooled"]
            L.append(f"Pooled {a} {p['arm_k']}/{p['arm_n']} = {_f(p['arm'], 4)} vs B {p['b_k']}/{p['b_n']} = {_f(p['b'], 4)}; "
                     f"reduction {_f(p['reduction'], 3)}; pooled ok {p['ok']}.\n")
        L.append(f"{_vline(dict(x, id=key), x['role'], us)} (all rows AND non-refusal rows)\n")
    if r["d2pr2"]:
        x = b["G6"]
        L.append("## G6: KL n15 <= 1.25 x B\n")
        L += _seed_table(x, [(a, "arm"), ("B", "b"), ("ratio", "ratio")], x["role"], us, k=4)
        L.append(f"Pooled ratio {_f(x['pooled']['ratio'], 3)}.\n")
    return "\n".join(L) + "\n"


def _load(paths):
    out = []
    for p in paths:
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm-name", default="arm")
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--d2pr2", action="store_true", help="D2-PR2 arm: leak primary and G6; LG-P1/P2 reported only")
    ap.add_argument("--dev", action="store_true", help="score the seeds that pair (need ceil(5n/6)), verdicts labelled DEV")
    ap.add_argument("--no-setup-check", action="store_true", help="diagnostic: report setup mismatches, do not apply them")
    ap.add_argument("--out", help="PREFIX: writes PREFIX.json and PREFIX.md")
    a = ap.parse_args(argv)
    r = compare(_load(a.arm), _load(a.b), a.arm_name, a.d2pr2, expect_seeds=None if a.dev else REGISTERED_SEEDS,
                setup_check=not a.no_setup_check)
    md = render(r)
    if a.out:
        with open(a.out + ".json", "w") as fh:
            json.dump(r, fh, indent=1, default=str)
        with open(a.out + ".md", "w") as fh:
            fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
