# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md
# Expected-fail: cued recall by context under S0 (a matched-overlap decoy may cue as well as the context cue)
"""The cue ladder (design section 5, cued recall) for brain/eval/life.py --cue-ladder. Report-only until the blind reader.

Plan (ladder_plan, pure and deterministic per seed; the lived life is NOT changed: build_life and the battery are the
flag-off ones). Eligible = every E and F item told once on a day t >= 2 and never asked in the live conversation after
its telling day (the same-day question on day t is allowed). Delay k (nights slept since the telling) in 1, 3, 7, 14:
the cell is probed the morning after night n = t + k - 1 (age in life.span_read terms = n - t = k - 1). A delay is
feasible when n <= days and the item is not the battery's rotating question on that morning. Each item goes to exactly
one delay cell: items in a seeded shuffle, the most constrained first, each to its least-filled feasible delay in its
track (ties by the seeded rng). Items with no feasible delay are listed under "unplaced" with the reason.

Per cell item the copy hears every rung, each from a FRESH conversation (state = base, no recent replies, the world clock
reset to that morning's 09:00, a per (seed, night, item, rung) reseed shared by all arms):
  C0 direct, C0g gist question (E only), C1 reworded, C2 context cue, C3 decoy cue (brain/eval/life_cues.py, hashed),
  C4 an uncued neutral prompt (life_cues.C4_PROMPTS, one per seed and item), all scored for THAT item's keys
  (gist keys for C0g) with the strict scorer, the cue text standing in for the question.
Independence: every episode's strength / uses / last_used (and the store list) is snapshotted before each turn and
restored after it (a host operation on the disk copy), so no rung or cell boosts another; use_restored logs whether
the turn had changed anything. The block runs on its OWN disk copy per arm per morning, opened after the battery copy
was released and deleted after use; the trunk is never opened. Arms: intact and hippocampus-off every morning with a
cell, both-off too on --full-probe-nights.

R_cue = P(C2 right) - max(P(C3 right), P(C4 right)) per arm, track, delay (auto scorer, PROVISIONAL until the blind
reader; rows keep the full replies). Headline: hippocampus-off at delay >= 3, intact at delay <= 1. Per-seed R_cue on
5/6 seeds has low power (a few items per cell); the pooled item-level permutation (pooled(), life_table) is the
registered primary.
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import socket
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.eval import life as L
from brain.eval import life_cues as LC
from brain.eval import life_cues_split as CS     # test2 cue texts, life_cues.py kept frozen
from brain.experiments.strict_scorer import correct_answer

CUES_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_cues.sha256")
CUES_TEST2_HASH_FILE = os.path.join(os.path.dirname(__file__), "life_cues_test2.sha256")   # TEST v2 cue texts
DELAYS = (1, 3, 7, 14)
RUNGS = {"E": ("C0", "C0g", "C1", "C2", "C3", "C4"), "F": ("C0", "C1", "C2", "C3", "C4")}
FULL_EXTRA = ("both-off",)
NOTE = ("auto scorer (strict_scorer.correct_answer, cue text as the question), PROVISIONAL until the blind reader; "
        "per-seed R_cue > 0 on 5/6 seeds has low power (a few items per cell), the pooled item-level permutation is the "
        "registered primary")


def _cue_file(split: str | None) -> str:
    if split == "test2":     # TEST v2 cue texts live in their own data file (design 2026-11-22 section 4)
        import importlib
        return importlib.import_module("brain.eval.life_cues_test2").__file__
    return LC.__file__


def cues_sha(split: str | None = None) -> str:
    """sha256 of the file holding the split's cue texts (life_cues.py for dev/test, life_cues_test2.py for test2)."""
    return hashlib.sha256(open(_cue_file(split), "rb").read()).hexdigest()


def load_cues(split: str) -> dict:
    """{item: (C2, C3)} of the split. TEST: life_cues.py must hash to life_cues.sha256; TEST v2: life_cues_test2.py
    must hash to life_cues_test2.sha256 (fail closed, also when the hash file is missing)."""
    cs = CS.cues(split)
    if cs is None:
        raise ValueError(f"no cue texts for split {split!r}")
    if split in ("test", "test2"):
        hfile = CUES_HASH_FILE if split == "test" else CUES_TEST2_HASH_FILE
        if not os.path.exists(hfile):
            raise RuntimeError(f"cue texts of split {split!r} are not frozen: the hash file {hfile} is missing")
        want = open(hfile).read().split()[0]
        got = cues_sha(split)
        if got != want:
            raise RuntimeError(f"cue texts changed: sha256 {got[:12]} != recorded {want[:12]} ({hfile})")
    return cs


def live_asked_late(sched: list, ledger: dict) -> set:
    """Items asked in the live conversation on a day after their first telling (the same-day question is allowed)."""
    out = set()
    for day in sched:
        for t in day["turns"]:
            i = t.get("item")
            if t["kind"] in ("sameday", "remention") and i in ledger and ledger[i]["told_days"] \
                    and day["day"] > min(ledger[i]["told_days"]):
                out.add(i)
    return out


def ladder_plan(seed: int, days: int, sched: list, ledger: dict, split: str = "dev", weekend: bool = False,
                exclude=None) -> dict:
    # The bank of THIS split, set on the module life_ladder imports: under `python -m brain.eval.life` the running life
    # is __main__, a different module object whose use_split() does not reach L.LB (L.LB would stay DEV on a TEST life).
    bank = L.use_split(split)
    late = live_asked_late(sched, ledger)
    bkw = {} if exclude is None else {"exclude": exclude}    # --age-ladder: the battery the life actually asks
    rot = {n: {q["item"] for q in L.battery(seed, n, ledger, **bkw) if q["kind"] == "rotating"} for n in range(1, days + 1)}
    ov = CS.overlaps(split, bank)
    elig, unplaced = [], []
    for iid in sorted(ledger):
        e = ledger[iid]
        if e["cls"] not in RUNGS or not e["told_days"]:
            continue
        t = min(e["told_days"])
        why = ("told more than once" if len(e["told_days"]) > 1 else "told on day 1" if t < 2
               else "asked live after its telling day" if iid in late else None)
        feas = [k for k in DELAYS if t + k - 1 <= days and iid not in rot[t + k - 1]] if why is None else []
        if why is None and not feas:
            why = ("no delay morning within the horizon" if all(t + k - 1 > days for k in DELAYS)
                   else "the rotating question on every feasible morning")
        if why:
            unplaced.append({"item": iid, "track": e["cls"], "told_day": t, "reason": why})
        else:
            elig.append((iid, e["cls"], t, feas))
    rng = random.Random(f"cue-ladder:{seed}")
    rng.shuffle(elig)
    elig.sort(key=lambda x: len(x[3]))       # stable: the most constrained items first
    count: dict = {}
    cells = []
    for iid, cls, t, feas in elig:
        lo = min(count.get((cls, k), 0) for k in feas)
        k = rng.choice([k for k in feas if count.get((cls, k), 0) == lo])
        count[(cls, k)] = count.get((cls, k), 0) + 1
        n = t + k - 1
        c2, c3, c4 = ov.get(iid, (None, None, None))
        cells.append({"item": iid, "track": cls, "told_day": t, "delay": k, "night": n, "age": n - t,
                      "calendar_days": round((L.day_start(n + 1, weekend) - L.day_start(t, weekend)) / 86400, 2),
                      "rungs": list(RUNGS[cls]), "ov_C2": c2, "ov_C3": c3, "ov_C4": c4})
    cells.sort(key=lambda c: (c["night"], c["item"]))
    return {"days": days, "cells": cells, "unplaced": unplaced,
            "counts": {f"{cls}+{k}": count.get((cls, k), 0) for cls in RUNGS for k in DELAYS}}


def cue_text(item, rung: str, cues: dict, seed: int) -> str:
    if rung == "C0":
        return item.direct
    if rung == "C0g":
        return item.gist_q
    if rung == "C1":
        return item.reworded
    if rung in ("C2", "C3"):
        return cues[item.id][0 if rung == "C2" else 1]
    if rung == "C4":
        return LC.C4_PROMPTS[random.Random(f"{seed}:{item.id}:C4").randrange(len(LC.C4_PROMPTS))]
    raise ValueError(rung)


def _turn_seed(seed: int, night: int, iid: str, rung: str) -> int:
    return int(hashlib.sha256(f"cue:{seed}:{night}:{iid}:{rung}".encode()).hexdigest()[:8], 16)


def _rank(b: Brain, item, text: str) -> dict:
    """Read-only: where the item's own episode ranks in the strength-weighted competition for this cue (0 = winner)."""
    hp = b.hippocampus
    idx = [i for i, e in enumerate(hp.episodes) if L.episode_is_source(item, e.text)]
    if not idx:
        return {"rank_target": None, "margin_target": None}
    sc = hp.scores(b.cortex, b.base, text)
    best = max(idx, key=lambda i: float(sc[i]))
    order = [int(i) for i in torch.argsort(sc, descending=True)]
    others = [float(sc[j]) for j in range(len(sc)) if j not in idx]
    return {"rank_target": order.index(best),
            "margin_target": round(float(sc[best]) - max(others), 4) if others else None}


def cue_turn(b: Brain, item, rung: str, text: str, cell: dict, sim, t_morning: float, seed: int) -> dict:
    hp = b.hippocampus
    eps = list(hp.episodes)
    vals = [(e.strength, e.uses, e.last_used) for e in eps]
    recent, mon, turns = list(b._recent_replies), len(getattr(b, "monitor_log", [])), b.meta.get("turns")
    b.state = [x.clone() for x in b.base]
    b._recent_replies = []
    sim.to(t_morning)
    s = _turn_seed(seed, cell["night"], item.id, rung)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    rank = _rank(b, item, text) if hp.enabled and eps else {}
    try:
        reply, info = b.hear(text)
    finally:
        changed = len(hp.episodes) != len(eps) or any((e.strength, e.uses, e.last_used) != v for e, v in zip(eps, vals))
        hp.episodes = eps
        for e, (st, u, lu) in zip(eps, vals):
            e.strength, e.uses, e.last_used = st, u, lu
        b._recent_replies = recent
        if hasattr(b, "monitor_log"):
            del b.monitor_log[mon:]
        if turns is not None:
            b.meta["turns"] = turns
        b.state = [x.clone() for x in b.base]
    keys = item.gist_keys if rung == "C0g" else item.keys
    rec = info.get("recalled")
    right = L.episode_is_source(item, rec)
    ok = any(correct_answer(reply, k, text, item.telling) for k in keys)
    row = {"item": item.id, "track": cell["track"], "delay": cell["delay"], "rung": rung,
           "slot": "gist" if rung == "C0g" else "detail", "you": text, "brain": reply, "recalled": rec,
           "fetched": rec is not None, "right_episode": right, "wrong_fetch": rec is not None and not right,
           "ok": ok, "key_hit": L._has(reply, [k for k in keys if not L._has(text, [k])]),
           "reply_uses_retrieved": ok and right, "use_restored": changed, **rank}
    if cell["track"] == "E" and rung != "C0g":
        row["gist_hit"] = L._has(reply, [g for g in item.gist_keys if not L._has(text, [g])])
    return row


def cue_probe(home: str, cortex, night: int, arms, cells: list, ledger: dict, sim, seed: int, brain_kw: dict,
              work: str, cues: dict, weekend: bool = False, check_pending: bool = True, bank=None) -> list:
    """The cue block of one morning: one fresh disk copy of the trunk per arm (never the battery's copy). bank = the
    running life's bank module (life.LB of the caller; L.LB may be another module object's under -m)."""
    bank = bank or L.LB
    out = []
    for arm in arms:
        h_off, c_off = L.ARMS[arm]
        cp = os.path.join(work, f"cue_L{night}_{arm}")
        shutil.rmtree(cp, ignore_errors=True)
        shutil.copytree(home, cp)
        t0 = time.time()
        t_m = L.day_start(night + 1, weekend)
        sim.to(t_m)
        b = Brain(cp, cortex=cortex, lock=True, **{**brain_kw, "sleep_on_restart": False})
        try:
            if check_pending:
                assert not b.pending(), "the trunk has an unslept day: a cue copy must start from a slept home"
            b.set_lesions(hippocampus_off=h_off, consolidated_off=c_off)
            rows = []
            for c in cells:
                it = bank.ALL[c["item"]]
                for rung in c["rungs"]:
                    rows.append(cue_turn(b, it, rung, cue_text(it, rung, cues, seed), c, sim, t_m, seed))
            rec = {"night": night, "arm": arm, "rows": rows, "seconds": round(time.time() - t0, 1),
                   "store": len(b.hippocampus.episodes), "host": socket.gethostname()}
        finally:
            b.release()
            shutil.rmtree(cp, ignore_errors=True)
        out.append(rec)
    return out


# ---------------------------------------------------------------- read-outs
def _p(rows):
    return round(sum(1 for r in rows if r["ok"]) / len(rows), 4) if rows else None


def r_cue(p: dict):
    if None in (p.get("C2"), p.get("C3"), p.get("C4")):
        return None
    return round(p["C2"] - max(p["C3"], p["C4"]), 4)


def _groups(results: list) -> dict:
    """{(arm, track, delay): {(seed, item): {rung: row}}}"""
    g: dict = {}
    for res in results:
        for pr in res.get("cue_probes", []):
            for r in pr["rows"]:
                g.setdefault((pr["arm"], r["track"], r["delay"]), {}).setdefault((res["seed"], r["item"]), {})[r["rung"]] = r
    return g


def headline(arm: str, delay: int) -> bool:
    return (arm == "intact" and delay <= 1) or (arm == "hippocampus-off" and delay >= 3)


def cue_summary(res: dict) -> dict:
    """Per arm, track and delay for one life (one seed): n items, P(right) per rung, R_cue, retrieval logged apart."""
    out = []
    for (arm, track, delay), items in sorted(_groups([res]).items()):
        rungs = sorted({k for v in items.values() for k in v})
        by = {k: [v[k] for v in items.values() if k in v] for k in rungs}
        p = {k: _p(v) for k, v in by.items()}
        c2 = by.get("C2", [])
        ret = [r for r in c2 if r["right_episode"]]
        out.append({"arm": arm, "track": track, "delay": delay, "n_items": len(items), "p": p, "R_cue": r_cue(p),
                    "headline": headline(arm, delay),
                    "direct_minus_uncued": None if None in (p.get("C0"), p.get("C4")) else round(p["C0"] - p["C4"], 4),
                    "retrieved_target": {k: round(sum(r["right_episode"] for r in v) / len(v), 4) for k, v in by.items()},
                    "wrong_fetch": {k: round(sum(r["wrong_fetch"] for r in v) / len(v), 4) for k, v in by.items()},
                    "C2_reply_given_retrieval": _p(ret), "C2_retrieved_n": len(ret),
                    "gist_hit_C2": (round(sum(r.get("gist_hit", False) for r in c2) / len(c2), 4)
                                    if c2 and track == "E" else None),
                    "use_restored_turns": sum(r["use_restored"] for v in by.values() for r in v)})
    return {"note": NOTE, "cells": out}


def pooled(results: list, n_perm: int = 2000, seed: int = 0) -> list:
    """Over seeds, per arm/track/delay: pooled R_cue, per-seed R_cue, sign count, and a one-sided permutation p for
    R_cue > 0 that shuffles each item's (C2, C3, C4) outcomes within the item (items nested in seeds)."""
    rng = random.Random(seed)
    out = []
    for (arm, track, delay), items in sorted(_groups(results).items()):
        trip = [tuple(int(v[k]["ok"]) for k in ("C2", "C3", "C4")) for v in items.values()
                if all(k in v for k in ("C2", "C3", "C4"))]
        if not trip:
            continue

        def stat(t):
            n = len(t)
            return sum(x[0] for x in t) / n - max(sum(x[1] for x in t) / n, sum(x[2] for x in t) / n)
        obs = stat(trip)
        ge = 0
        for _ in range(n_perm):
            perm = []
            for x in trip:
                y = list(x)
                rng.shuffle(y)
                perm.append(y)
            ge += stat(perm) >= obs - 1e-12
        per_seed = {}
        for s in sorted({s for s, _ in items}):
            t = [tuple(int(v[k]["ok"]) for k in ("C2", "C3", "C4")) for (s2, _), v in items.items()
                 if s2 == s and all(k in v for k in ("C2", "C3", "C4"))]
            per_seed[s] = round(stat(t), 4) if t else None
        out.append({"arm": arm, "track": track, "delay": delay, "n_items": len(trip), "R_cue": round(obs, 4),
                    "p_perm": round((1 + ge) / (1 + n_perm), 4), "per_seed": per_seed,
                    "seeds_positive": sum(1 for v in per_seed.values() if v is not None and v > 0),
                    "n_seeds": len(per_seed), "headline": headline(arm, delay)})
    return out


def _f(x):
    return "-" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def cue_md(res: dict) -> list:
    cl = res["cue_ladder"]
    plan = cl["plan"]
    L_ = ["\n## Cue ladder (--cue-ladder, design 5 cued recall)\n",
          f"_Cells: {plan['counts']}; unplaced {len(plan['unplaced'])} "
          f"({', '.join(sorted({u['reason'] for u in plan['unplaced']})) or 'none'}). Delay = nights slept since the "
          "telling. Each cell item hears C0 direct, C0g gist (E), C1 reworded, C2 context cue, C3 decoy cue, C4 uncued "
          "prompt, each from a fresh conversation on its own disk copy per arm per morning, with every episode's "
          "strength/uses/last_used restored after each turn (host op on the copy), so turns are independent. "
          f"Cue texts sha256 {cl['config']['cues_sha256'][:12]}. Scores: {NOTE}._\n",
          "| arm | track | delay | items | C0 | C0g | C1 | C2 | C3 | C4 | R_cue | C0-C4 | C2 retrieved target | headline |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in res.get("cue_summary", {}).get("cells", []):
        p = c["p"]
        L_.append(f"| {c['arm']} | {c['track']} | +{c['delay']} | {c['n_items']} | "
                  + " | ".join(_f(p.get(k)) for k in ("C0", "C0g", "C1", "C2", "C3", "C4"))
                  + f" | {_f(c['R_cue'])} | {_f(c['direct_minus_uncued'])} | {_f(c['retrieved_target'].get('C2'))} | "
                  f"{'yes' if c['headline'] else ''} |")
    L_.append("\n_Plan (item: track +delay on morning after night n; overlap with the telling C2/C3/C4):_ "
              + "; ".join(f"{c['item']} {c['track']}+{c['delay']} n{c['night']} ({c['ov_C2']}/{c['ov_C3']}/{c['ov_C4']})"
                          for c in plan["cells"]))
    for pr in res.get("cue_probes", []):
        L_.append(f"\n#### cue block, morning after night {pr['night']}, arm: {pr['arm']} ({pr['seconds']} s)\n")
        for r in pr["rows"]:
            tag = (" ✅" if r["ok"] else " ❌") + (" WRONG-FETCH" if r["wrong_fetch"] else "")
            recd = f" _(recalled: {r['recalled']})_" if r.get("recalled") else ""
            L_.append(f"- [{r['rung']} {r['item']} +{r['delay']}] **you:** {r['you']}\n  **brain:** {r['brain']}{tag}{recd}")
    return L_


def pooled_md(results: list) -> list:
    rows = pooled(results)
    if not rows:
        return []
    L_ = ["\n## Cue ladder, pooled over seeds\n",
          f"_R_cue = P(C2) - max(P(C3), P(C4)); {NOTE}. p = one-sided permutation (rungs shuffled within item)._\n",
          "| arm | track | delay | items | R_cue pooled | p perm | seeds R>0 | per seed | headline |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L_.append(f"| {r['arm']} | {r['track']} | +{r['delay']} | {r['n_items']} | {r['R_cue']:+.3f} | {r['p_perm']} | "
                  f"{r['seeds_positive']}/{r['n_seeds']} | {r['per_seed']} | {'yes' if r['headline'] else ''} |")
    return L_
