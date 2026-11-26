# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-26-routing-screen3-prereg.md (part X, as amended before data);
#   results/2026-11-25-sall-live-dev-read-1.5b.md; brain/experiments/routing_screen2.py
# Expected-fail: partly. Near-duplicate separation is MISSING (PREREQUISITES.md); freed self rows may go to no-fetch
#   (the told episodes' weight stays low, X5 / X7) rather than right. Fetch level only, offline.
"""Routing screen 3, part X: on arm S's own night snapshots (live S_all + slept bar), arm K = the S life's competition
(must reproduce the banked fetch, with the amended fp16/fp32 tolerance) vs arm KX = the same similarities and weights
with every permanent episode excluded (-1e9, as recall(exclude=) does; what Brain(recall_skip_permanent=True) passes).
Host-side harness; FREEZE on; nothing is fitted.

    python -m brain.experiments.routing_screen3x run --life L.json --snaps DIR --trunk T --out O.json
    python -m brain.experiments.routing_screen3x score O_s319.json O_s320.json O_s321.json [--out V.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import torch

from brain import clock
from brain.experiments import routing_replay as RR
from brain.experiments.routing_screen2 import SALL_SLEPT, keyed_from

ARMS = ("K", "KX")
TOL = 0.01                    # amended: a K/banked mismatch within TOL of a bar drops the row, not the night
NO_FETCH_EXCESS_MAX_SEED = 5  # X7
X5_RIGHT_FRAC, X6_WRONG_FRAC = 1 / 3, 1 / 4
ENGAGE_MIN = 10


def decide(k, cortex, base, cue: str, now: float, exclude=()) -> dict:
    cue = cue.strip()
    question = cue.endswith("?")
    sims = k.similarities(cortex, base, cue)
    w = torch.tensor([math.tanh(k.effective_strength(e, now)) for e in k.episodes])
    score = sims * w
    if exclude:
        sims, score = sims.clone(), score.clone()
        for i in exclude:
            sims[i] = score[i] = -1e9
    saved = k.last_fetch
    try:
        _, s = k._compete(score, question, now, sims=sims)
        idx = k.last_fetch["idx"]
    finally:
        k.last_fetch = saved
    srt = torch.sort(score, descending=True).values
    return {"idx": idx, "score_best": float(srt[0]), "weighted_winner": int(torch.argmax(score)),
            "margin_w": float(srt[0] - srt[1]) if len(srt) > 1 else None, "question": question}


def near_bar(fx: dict, dk: dict) -> bool:
    bar, margin = SALL_SLEPT
    mw = fx.get("margin_w")
    return (mw is not None and abs(mw - margin) < TOL) or abs(dk["score_best"] - bar) < TOL or \
        (dk["margin_w"] is not None and abs(dk["margin_w"] - margin) < TOL)


def screen_night(res, night, snap, trunk, cortex, work, weekend) -> dict:
    from brain.eval import life
    from brain.brain import _is_telling
    LB = life.use_split(res.get("split", "dev"))
    ledger, seed = res["ledger"], res["seed"]
    rows_b = RR.banked_rows(res, night)
    for r in rows_b:
        if _is_telling(r["you"]):
            raise ValueError(f"night {night}: battery cue {r['you']!r} is a telling")
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{seed}_n{night}")
    try:
        sim.to(life.day_start(night + 1, weekend))
        life._reseed(seed, 500 + night)
        b = RR.open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=True)
        try:
            hc = b.hippocampus
            before = [(e.strength, e.uses, e.last_used) for e in hc.episodes]
            k = keyed_from(hc, b, "1.5b")
            texts = [e.text for e in hc.episodes]
            perm = tuple(i for i, e in enumerate(hc.episodes) if e.permanent)
            out_rows, void, dropped = [], [], []
            for i, r in enumerate(rows_b):
                now = clock.now()
                dk = decide(k, b.cortex, b.base, r["you"], now)
                dx = decide(k, b.cortex, b.base, r["you"], now, exclude=perm)
                fx = r.get("fetch") or {}
                cls = RR.row_class(r, ledger)
                it = LB.ALL[r["item"]] if cls in ("told", "identity") else None
                row = {"night": night, "i": i, "item": r.get("item"), "kind": r["kind"], "cue": r["you"], "cls": cls,
                       "banked_idx": fx.get("idx"), "K_weighted_winner_perm": dk["weighted_winner"] in perm,
                       "K_diag": {x: dk[x] for x in ("score_best", "margin_w", "weighted_winner")}}
                for a, d in (("K", dk), ("KX", dx)):
                    kidx = d["idx"]
                    row[a] = {"idx": kidx, "text": texts[kidx] if kidx is not None else None,
                              "perm": kidx in perm if kidx is not None else False,
                              "right_episode": bool(life.episode_is_source(it, texts[kidx])) if it is not None and
                              kidx is not None else (False if it is not None else None)}
                mismatch = row["K"]["idx"] != row["banked_idx"] or row["K"]["text"] != r.get("recalled")
                if it is not None and row["K"]["right_episode"] != bool(r.get("right_episode")):
                    mismatch = True
                if mismatch:
                    (dropped if near_bar(fx, dk) else void).append(i)
                out_rows.append(row)
            after = [(e.strength, e.uses, e.last_used) for e in hc.episodes]
            if after != before:
                raise RuntimeError(f"night {night}: an episode changed during the screen (FREEZE broken)")
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
    finally:
        clock.set(None)
    kept = [r for r in out_rows if r["i"] not in set(dropped)]
    return {"night": night, "store": len(texts), "n_perm": len(perm), "rows": kept, "dropped_rows": dropped,
            "void_rows": sorted(set(void)), "void": bool(void)}


def run(res, snaps, trunk, cortex, work, weekend=True, nights=RR.NIGHTS, log=print) -> dict:
    if res["brain_kw"].get("key") != "S_all" or res["brain_kw"].get("key_bar") != "slept":
        raise SystemExit(f"part X is registered on arm S lives (key S_all, slept bar), not {res['brain_kw']}")
    out = {"seed": res["seed"], "sall_slept_bar": list(SALL_SLEPT), "tol": TOL, "arms": list(ARMS), "nights": []}
    os.makedirs(work, exist_ok=True)
    for n in nights:
        rec = screen_night(res, n, os.path.join(snaps, f"n{n}"), trunk, cortex, work, weekend)
        log(f"s{res['seed']} n{n}: store {rec['store']} perm {rec['n_perm']} rows {len(rec['rows'])} "
            f"dropped {len(rec['dropped_rows'])} void {rec['void']}")
        out["nights"].append(rec)
    return out


def _count(rows, arm):
    told = [r for r in rows if r["cls"] == "told"]
    fire = [r for r in rows if r["cls"] == "fire"]
    return {"told_n": len(told), "right": sum(r[arm]["right_episode"] for r in told),
            "wrong": sum(r[arm]["idx"] is not None and not r[arm]["right_episode"] for r in told),
            "wrong_nonperm": sum(r[arm]["idx"] is not None and not r[arm]["right_episode"] and not r[arm]["perm"]
                                 for r in told),
            "fetch_perm": sum(r[arm]["perm"] for r in told),
            "no_fetch": sum(r[arm]["idx"] is None for r in told), "fires": sum(r[arm]["idx"] is not None for r in fire),
            "identity_fires": sum(r[arm]["idx"] is not None for r in rows if r["cls"] == "identity"),
            "identity_right": sum(bool(r[arm]["right_episode"]) for r in rows if r["cls"] == "identity")}


def _self_rows(rows):
    s = [r for r in rows if r["cls"] == "told" and r["K_weighted_winner_perm"]]
    return {"n": len(s), "to_right": sum(r["KX"]["right_episode"] for r in s),
            "to_wrong": sum(r["KX"]["idx"] is not None and not r["KX"]["right_episode"] for r in s),
            "to_none": sum(r["KX"]["idx"] is None for r in s)}


def score(outs: list) -> dict:
    seeds = {}
    for o in outs:
        nights = [n for n in o["nights"] if not n.get("void")]
        nvoid = len(o["nights"]) - len(nights)
        rows = [r for n in nights for r in n["rows"]]
        seeds[str(o["seed"])] = {"void": nvoid > RR.VOID_NIGHTS_MAX, "void_nights": nvoid,
                                 "dropped_rows": sum(len(n["dropped_rows"]) for n in o["nights"]),
                                 **{a: _count(rows, a) for a in ARMS}, "self": _self_rows(rows)}
    live = {s: v for s, v in seeds.items() if not v["void"]}
    verdict = {"seeds": seeds}
    if len(live) < 3:
        verdict["status"] = f"INCONCLUSIVE: {3 - len(live)} seed(s) void; no bar is read"
        return verdict
    K = sum(v["K"]["right"] for v in live.values())
    X = sum(v["KX"]["right"] for v in live.values())
    selfn = sum(v["self"]["n"] for v in live.values())
    eng = {"KX_fetch_perm": sum(v["KX"]["fetch_perm"] for v in live.values()), "self_rows": selfn}
    if eng["KX_fetch_perm"] != 0 or selfn < ENGAGE_MIN:
        verdict.update(status="UNDEFINED", engagement=eng)
        return verdict
    need = max(math.ceil((1 + RR.GAIN_FRAC) * K), K + RR.GAIN_MIN)
    up = sum(v["KX"]["right"] > v["K"]["right"] for v in live.values())
    down = [s for s, v in live.items() if v["KX"]["right"] < v["K"]["right"] - 1]
    kills = []
    if X < need:
        kills.append(f"X1 right {X} < {need} (K {K})")
    if up < RR.DIRECTION_SEEDS or down:
        kills.append(f"X2 right rises on {up}/3 seeds, falls by >1 on {down}")
    w3 = {s: v["KX"]["wrong_nonperm"] - v["K"]["wrong_nonperm"] for s, v in live.items()}
    if any(x > 5 for x in w3.values()):
        kills.append(f"X3 wrong non-permanent excess {w3} (max 5/seed)")
    f4 = {s: v["KX"]["fires"] - v["K"]["fires"] for s, v in live.items()}
    if any(x > RR.FIRE_EXCESS_MAX_SEED for x in f4.values()):
        kills.append(f"X4 fire excess {f4} (max {RR.FIRE_EXCESS_MAX_SEED}/seed)")
    to_r = sum(v["self"]["to_right"] for v in live.values())
    to_w = sum(v["self"]["to_wrong"] for v in live.values())
    if to_r < X5_RIGHT_FRAC * selfn:
        kills.append(f"X5 self -> right {to_r}/{selfn} < 1/3")
    if to_w > X6_WRONG_FRAC * selfn:
        kills.append(f"X6 self -> wrong {to_w}/{selfn} > 1/4")
    nf = {s: v["KX"]["no_fetch"] - v["K"]["no_fetch"] for s, v in live.items()}
    if any(x > NO_FETCH_EXCESS_MAX_SEED for x in nf.values()):
        kills.append(f"X7 told no-fetch excess {nf} (max {NO_FETCH_EXCESS_MAX_SEED}/seed)")
    verdict.update(engagement=eng, X={"K_right": K, "KX_right": X, "need": need, "self_to_right": to_r,
                                      "self_to_wrong": to_w, "self_rows": selfn, "kills": kills},
                   status="KILLED" if kills else "PASS (fetch level: earns live arm SX)")
    return verdict


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True)
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--work", default=None)
    r.add_argument("--no-weekend", action="store_true")
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        with open(a.life) as fh:
            res = json.load(fh)
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
        out = run(res, a.snaps, a.trunk, cortex, a.work or a.out + ".work", not a.no_weekend)
        with open(a.out, "w") as fh:
            json.dump(out, fh)
        return out
    outs = []
    for f in a.outs:
        with open(f) as fh:
            outs.append(json.load(fh))
    v = score(outs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(v, fh, indent=1)
    print(json.dumps({k: v[k] for k in v if k != "seeds"}, indent=1))
    for s_, x in v["seeds"].items():
        print(s_, {a_: {k: x[a_][k] for k in ("right", "wrong", "no_fetch", "fetch_perm", "fires")} for a_ in ARMS},
              x["self"], "void_nights", x["void_nights"], "dropped", x["dropped_rows"])
    return v


if __name__ == "__main__":
    main()
