# Capability: Episodic recall
# Record: brain/research/2026-11-25-routing-screen2-prereg.md (prereg); results/2026-11-25-routing-ceiling-read.md
# Expected-fail: partly. Near-duplicate separation and a paraphrase-robust cue are MISSING (PREREQUISITES.md); the
#   S_all key may fire on untold near-duplicate cues (K4) and no key ranks "admin" first. Fetch level only.
"""Routing screen 2: arm A (K-P as is) vs arm K, the existing S_all key with its slept bar
(KeyedHippocampus(key="S_all", familiarity=0.165, margin=0.02) = Brain(key="S_all", key_bar="slept")), offline on the
KR night snapshots. Host-side harness; no brain default changes, nothing is fitted.

Per non-void-able night: the snapshot store is opened with the final trunk's deltas (routing_replay.open_snapshot,
keys checked against keys.pt); arm A = routing_replay.decide's A (must reproduce the banked fetch, else the night is
void, as in the 11-24 screen); arm K = the live KeyedHippocampus over the same episodes, FREEZE on, its own
similarities() weighted by tanh(effective_strength) (= scores()), then _compete.

    python -m brain.experiments.routing_screen2 run --life L.json --snaps DIR --trunk T --out O.json
    python -m brain.experiments.routing_screen2 score O_s319.json O_s320.json O_s321.json --validity V1 V2 V3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from brain import clock
from brain.experiments import routing_replay as RR
from brain.hippocampus_keys import KeyedHippocampus

SALL_SLEPT = (0.165, 0.02)       # brain.Brain.KEY_BARS_SLEPT_BY_SIZE[("S_all", "1.5b")]; asserted below
ARMS = ("A", "K")
NO_FETCH_EXCESS_MAX_SEED = 5     # K5


def keyed_from(hc, brain=None, size: str | None = "1.5b") -> KeyedHippocampus:
    """The S_all arm over hc's episodes (shared objects; FREEZE on so no fetch changes any of them). With a Brain, the
    arm is built by the brain's own code (key_bar="slept", Brain._make_hippocampus("S_all")) and must be 1.5B; without
    one, or with size=None (stub tests: a stub has no model size), the same KeyedHippocampus built directly."""
    from brain.brain import Brain, model_size
    assert Brain.KEY_BARS_SLEPT_BY_SIZE[("S_all", "1.5b")] == SALL_SLEPT
    if brain is not None and size is not None:
        if model_size(brain.cortex_model) != size:
            raise ValueError(f"routing screen 2 is registered for 1.5B, not {brain.cortex_model!r}")
        saved = brain.key_bar
        try:
            brain.key_bar = "slept"
            k = brain._make_hippocampus("S_all")
        finally:
            brain.key_bar = saved
    else:
        k = KeyedHippocampus(key="S_all", familiarity=SALL_SLEPT[0], margin=SALL_SLEPT[1])
        k.GAIN = hc.GAIN
    assert isinstance(k, KeyedHippocampus) and (k.FAMILIARITY, k.MARGIN) == SALL_SLEPT
    k.episodes = hc.episodes
    k.FREEZE_USE = True
    return k


def decide_K(k: KeyedHippocampus, cortex, base, cue: str, now: float) -> dict:
    cue = cue.strip()
    question = cue.endswith("?")
    sims = k.similarities(cortex, base, cue)
    w = torch.tensor([math.tanh(k.effective_strength(e, now)) for e in k.episodes])
    score = sims * w
    saved = k.last_fetch
    try:
        _, s = k._compete(score, question, now, sims=sims)
        idx = k.last_fetch["idx"]
    finally:
        k.last_fetch = saved
    raw = int(torch.argmax(sims))
    wb = int(torch.argmax(score))
    srt = torch.sort(score, descending=True).values
    # diagnostics (11-25, after the registered read; no bar uses them): what blocks a raw-first episode
    return {"idx": idx, "score": s, "raw_winner": raw, "sim_raw_max": float(sims.max()), "weighted_winner": wb,
            "score_best": float(srt[0]), "margin_w": float(srt[0] - srt[1]) if len(srt) > 1 else None,
            "w_raw": float(w[raw]), "score_raw": float(score[raw]), "question": question}


def screen_night(res, night, snap, trunk, cortex, work, weekend, size="1.5b") -> dict:
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
            k = keyed_from(hc, b, size)
            texts = [e.text for e in hc.episodes]
            out_rows, void = [], []
            for i, r in enumerate(rows_b):
                now = clock.now()
                d = RR.decide(hc, b.cortex, b.base, r["you"], now=now)
                dk = decide_K(k, b.cortex, b.base, r["you"], now)
                fx = r.get("fetch") or {}
                row = {"night": night, "i": i, "item": r.get("item"), "kind": r["kind"], "cue": r["you"],
                       "cls": RR.row_class(r, ledger), "banked_idx": fx.get("idx"),
                       "K_raw_winner": dk["raw_winner"], "K_sim_raw_max": dk["sim_raw_max"],
                       "K_diag": {k_: dk[k_] for k_ in ("weighted_winner", "score_best", "margin_w", "w_raw",
                                                       "score_raw", "question")}}
                it = LB.ALL[r["item"]] if row["cls"] in ("told", "identity") else None
                for a, kidx in (("A", d["A"]["idx"]), ("K", dk["idx"])):
                    row[a] = {"idx": kidx, "text": texts[kidx] if kidx is not None else None,
                              "right_episode": bool(life.episode_is_source(it, texts[kidx])) if it is not None and
                              kidx is not None else (False if it is not None else None)}
                if it is not None:
                    row["K_raw_right"] = bool(life.episode_is_source(it, texts[dk["raw_winner"]]))
                if row["A"]["idx"] != row["banked_idx"] or row["A"]["text"] != r.get("recalled"):
                    void.append(i)
                if it is not None and row["A"]["right_episode"] != bool(r.get("right_episode")):
                    void.append(i)
                out_rows.append(row)
            after = [(e.strength, e.uses, e.last_used) for e in hc.episodes]
            if after != before:
                raise RuntimeError(f"night {night}: an episode changed during the screen (FREEZE broken)")
        finally:
            b.release()
            import shutil
            shutil.rmtree(home, ignore_errors=True)
    finally:
        clock.set(None)
    return {"night": night, "store": len(texts), "rows": out_rows, "void_rows": sorted(set(void)), "void": bool(void)}


def run(res, snaps, trunk, cortex, work, weekend=True, nights=RR.NIGHTS, log=print, size="1.5b") -> dict:
    out = {"seed": res["seed"], "split": res.get("split"), "brain_kw": res["brain_kw"], "weekend": weekend,
           "sall_slept_bar": list(SALL_SLEPT), "arms": list(ARMS), "nights": []}
    os.makedirs(work, exist_ok=True)
    for n in nights:
        snap = os.path.join(snaps, f"n{n}")
        if not os.path.isdir(snap):
            out["nights"].append({"night": n, "void": True, "missing_snapshot": True, "rows": []})
            continue
        rec = screen_night(res, n, snap, trunk, cortex, work, weekend, size)
        log(f"s{res['seed']} n{n}: store {rec['store']}, void rows {rec['void_rows']}")
        out["nights"].append(rec)
    return out


# ------------------------------------------------------------------ scorer (prereg section 3)
def _count(rows, arm):
    c = RR._count(rows, arm)
    told = [r for r in rows if r["cls"] == "told"]
    c["no_fetch"] = sum(r[arm]["idx"] is None for r in told)
    c["identity_fires"] = sum(r[arm]["idx"] is not None for r in rows if r["cls"] == "identity")
    return c


def score(outs: list, validity: dict) -> dict:
    seeds = {}
    for o in outs:
        s = str(o["seed"])
        bad_life = bool(validity.get(s))
        nights = [n for n in o["nights"] if not n.get("void")]
        nvoid = len(o["nights"]) - len(nights)
        rows = [r for n in nights for r in n["rows"]]
        nonH = [r for r in rows if r["kind"] != "H"]
        seeds[s] = {"void": bad_life or nvoid > RR.VOID_NIGHTS_MAX, "void_life": bad_life, "void_nights": nvoid,
                    **{a: _count(rows, a) for a in ARMS}, "nonH": {a: _count(nonH, a) for a in ARMS},
                    "K_raw_right": sum(r.get("K_raw_right", False) for r in rows if r["cls"] == "told")}
    per_night, items = [], {}
    for o in outs:
        for n in o["nights"]:
            per_night.append({"seed": o["seed"], "night": n["night"], "void": n.get("void", True),
                              **({a: _count(n["rows"], a) for a in ARMS} if n.get("rows") and not n.get("void") else {})})
            if n.get("void"):
                continue
            for r in n["rows"]:
                if r["kind"] == "H" and r["cls"] == "told":
                    x = items.setdefault(r["item"], {"n": 0, **{a: {"right": 0, "wrong": 0} for a in ARMS}})
                    x["n"] += 1
                    for a in ARMS:
                        x[a]["right"] += r[a]["right_episode"]
                        x[a]["wrong"] += r[a]["idx"] is not None and not r[a]["right_episode"]
    live = {s: v for s, v in seeds.items() if not v["void"]}
    verdict = {"seeds": seeds, "per_night": per_night, "H_items": items}
    if len(live) < 3:
        verdict["status"] = f"INCONCLUSIVE: {3 - len(live)} seed(s) void; no bar is read"
        return verdict
    A_right = sum(v["A"]["right"] for v in live.values())
    right = sum(v["K"]["right"] for v in live.values())
    need = max(math.ceil((1 + RR.GAIN_FRAC) * A_right), A_right + RR.GAIN_MIN)
    up = sum(v["K"]["right"] > v["A"]["right"] for v in live.values())
    down = [s for s, v in live.items() if v["K"]["right"] < v["A"]["right"] - 1]
    kills = []
    if right < need:
        kills.append(f"K1 right {right} < {need} (A {A_right})")
    if up < RR.DIRECTION_SEEDS or down:
        kills.append(f"K2 right rises on {up}/3 seeds, falls by >1 on {down}")
    nw = {s: v["K"]["new_wrong_on_A_right"] for s, v in live.items()}
    if any(x > RR.NEW_WRONG_MAX_SEED for x in nw.values()):
        kills.append(f"K3 new wrong on A-right rows {nw} (max {RR.NEW_WRONG_MAX_SEED}/seed)")
    fx = {s: v["K"]["fires"] - v["A"]["fires"] for s, v in live.items()}
    if any(x > RR.FIRE_EXCESS_MAX_SEED for x in fx.values()):
        kills.append(f"K4 fire excess over A {fx} (max {RR.FIRE_EXCESS_MAX_SEED}/seed)")
    nf = {s: v["K"]["no_fetch"] - v["A"]["no_fetch"] for s, v in live.items()}
    if any(x > NO_FETCH_EXCESS_MAX_SEED for x in nf.values()):
        kills.append(f"K5 told no-fetch excess over A {nf} (max {NO_FETCH_EXCESS_MAX_SEED}/seed)")
    verdict["K"] = {"right": right, "need": need, "A_right": A_right, "kills": kills,
                    "status": "KILLED" if kills else "PASS (fetch level only: earns a live DEV life)"}
    verdict["status"] = "READ"
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
    r.add_argument("--out", required=True)
    r.add_argument("--work", default=None)
    r.add_argument("--nights", default="5-15")
    s = sub.add_parser("score")
    s.add_argument("outs", nargs="+")
    s.add_argument("--validity", nargs="+", required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        with open(a.life) as fh:
            res = json.load(fh)
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
        lo, _, hi = a.nights.partition("-")
        out = run(res, a.snaps, a.trunk, cortex, a.work or a.out + ".work", True, range(int(lo), int(hi or lo) + 1))
        out["life"], out["device"] = a.life, a.device
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        return out
    outs = [json.load(open(f)) for f in a.outs]
    if len(a.validity) != len(outs):
        raise SystemExit("score: one --validity file per output, same order")
    validity = {}
    for o, f in zip(outs, a.validity):
        x = json.load(open(f))
        validity[str(o["seed"])] = x if isinstance(x, list) else (x.get("first", []) if x.get("mismatches") else [])
    v = score(outs, validity)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(v, fh, indent=1)
    print(json.dumps({k: v[k] for k in ("status", "K") if k in v}, indent=1))
    for s_, x in v["seeds"].items():
        print(s_, "void" if x["void"] else "", {a: {k: x[a][k] for k in ("right", "wrong", "no_fetch", "fires",
              "new_wrong_on_A_right", "identity_fires")} for a in ARMS}, "nonH right", {a: x["nonH"][a]["right"]
              for a in ARMS}, "K raw-first right", x["K_raw_right"])
    return v


if __name__ == "__main__":
    main()
