# Capability: Memory selection (pattern completion and separation)
# Record: brain/research/2026-11-27-dg-separation-port-prereg-DRAFT.md (prereg); brain/dg_separation.py;
#   results/2026-11-25-routing-ceiling-read.md (DG(S0) dead; S_all_live ranks the right episode first 168/217);
#   results/2026-11-25-routing-screen2-read.md (S_all through the live competition KILLED at the bar, not the rank)
# Expected-fail: yes for the rank (a fixed expansion keeps rank order); the open question is the SHAPE of the score
#   (chance floor, top-1 margin) for the bar and margin. Near-duplicate separation is MISSING in PREREQUISITES.md and
#   the strength-weighted competition (hubs) is outside this module. Fetch level only, never a reply claim.
"""Offline DG screen on the KR night snapshots (s319/s320/s321, nights 5-15), the routing screens' own population.

For every night the snapshot store is opened with the final trunk's deltas (routing_replay.open_snapshot), arm A
(K-P as is) must reproduce the banked fetch (else the night is void, as in screens 1 and 2), and every battery cue is
read ONCE from base. From that one read each arm's similarities are made and recorded as (weighted winner, its weighted
score, weighted margin, raw winner, raw rank of the right episode, right flags). Nothing about a bar is decided at run
time: `score` applies a bar to the records, so the bar rule reads no told row (prereg section 5).

Arms (names as in the records): A, K (S_all, registered slept bar), then DG arms made by brain.dg_separation:
  DG:<f>[:<mapseed>]  k-WTA code, active fraction f, map seed (42 unless given)
  DGcc:<f>            the same with CA3 completion (softmax beta CA3_BETA, 2 steps)
  DGshuf:<f>          stored codes' units permuted (the shuffled-code control)
  DGlesion            no k-WTA, the dense centred projection (the sparsity lesion)

    python -m brain.experiments.dg_screen run --life L.json --snaps DIR --trunk T --out O.json [--model M --device cpu]
    python -m brain.experiments.dg_screen score O_s319.json O_s320.json O_s321.json --validity V1 V2 V3 [--out v.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil

import torch

from brain import clock
from brain.hippocampus import Hippocampus
from brain.dg_separation import DGHippocampus, DGSeparator
from brain.experiments import routing_replay as RR
from brain.experiments import routing_screen2 as S2
from brain.experiments.routing_ceiling import rank_of

E, FAN_IN = 8192, 64
FRACS = (0.02, 0.05)
PRIMARY_FRAC = 0.05            # the registered cell; the other fraction is reported, never picked
MAP_SEEDS = (42, 43, 44, 100, 101, 102)
CA3_BETA = 50.0                # overlap / k spans ~0.0-0.5; beta 50 makes a 0.05 overlap lead a 2.5 nat lead
MARGIN = 0.02                  # the class margin every routing screen used
STMT_RATIO = 0.45 / 0.12       # Hippocampus.FAMILIARITY_SPONTANEOUS / FAMILIARITY: the statement bar keeps this ratio
BAR_GRID = [round(0.005 * i, 3) for i in range(0, 201)]
RIGHT_GAIN_MIN = 5             # D6/D9: rows over the lesion / the refit dense key
SHUF_MAX_FRAC = 0.25           # D7: the shuffled-code control keeps at most this fraction of the DG arm's right fetches
MAP_SEEDS_MIN = 4              # D8: map seeds (of 6) that must clear K1


def arm_specs() -> list:
    """[(name, kwargs for DGHippocampus)] in record order."""
    out = []
    for f in FRACS:
        out += [(f"DG:{f}", dict(frac=f)), (f"DGcc:{f}", dict(frac=f, ca3_beta=CA3_BETA)),
                (f"DGshuf:{f}", dict(frac=f, shuffle=True))]
    out.append(("DGlesion", dict(frac=PRIMARY_FRAC, lesion=True)))
    out += [(f"DG:{PRIMARY_FRAC}:{s}", dict(frac=PRIMARY_FRAC, seed=s)) for s in MAP_SEEDS if s != 42]
    return out


def build_arms(episodes: list, E_: int | None = None, fan_in: int | None = None, specs=None) -> dict:
    """{name: DGHippocampus} over the same episode list (FREEZE on). Arms with one map seed share one separator, so each
    episode is projected once per map. Bars are placeholders: the scorer applies the bar."""
    E_, fan_in = E if E_ is None else E_, FAN_IN if fan_in is None else fan_in
    seps, arms = {}, {}
    for name, kw in (arm_specs() if specs is None else specs):
        kw = dict(kw)
        frac, seed = kw.pop("frac"), kw.pop("seed", 42)
        sep = DGSeparator(E_, frac, fan_in, seed)
        sep = seps.setdefault((frac, seed), sep)
        h = DGHippocampus(familiarity=0.0, margin=MARGIN, sep=sep, **kw)
        h.episodes, h.FREEZE_USE = episodes, True
        arms[name] = h
    return arms


def record(h, sims: torch.Tensor, now: float, cue: str, right: list | None, texts_right) -> dict:
    """Everything a bar needs from one arm's similarities for one cue (no bar applied)."""
    w = torch.tensor([math.tanh(h.effective_strength(e, now)) for e in h.episodes])
    score = sims * w
    srt = torch.sort(score, descending=True)
    raw = int(torch.argmax(sims))
    wb = int(srt.indices[0])
    rec = {"wbest": wb, "score_best": float(srt.values[0]),
           "margin_w": float(srt.values[0] - srt.values[1]) if len(srt.values) > 1 else None,
           "raw_winner": raw, "sim_raw_max": float(sims.max()), "question": cue.strip().endswith("?")}
    if texts_right is not None:
        rec["right_w"], rec["right_raw"] = bool(texts_right(wb)), bool(texts_right(raw))
    if right:
        rec["rank_right"] = rank_of(sims, right)["rank"]
    return rec


def screen_night(res, night, snap, trunk, cortex, work, weekend, size="1.5b", specs=None) -> dict:
    from brain.brain import _is_telling
    from brain.eval import life
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
            texts = [e.text for e in hc.episodes]
            k = S2.keyed_from(hc, b, size)
            arms = build_arms(hc.episodes, specs=specs)
            out_rows, void = [], []
            for i, r in enumerate(rows_b):
                now = clock.now()
                d = RR.decide(hc, b.cortex, b.base, r["you"], now=now)
                dk = S2.decide_K(k, b.cortex, b.base, r["you"], now)
                cls = RR.row_class(r, ledger)
                it = LB.ALL[r["item"]] if cls in ("told", "identity") else None
                is_right = (lambda j, it=it: life.episode_is_source(it, texts[j])) if it is not None else None
                right = [j for j, t in enumerate(texts) if life.episode_is_source(it, t)] if cls == "told" else None
                _, st = b.cortex.read(f"User: {r['you'].strip()}\n\n", b.base)    # the cue, read once from base
                dl = b.cortex.matrix_delta(st, b.base)
                row = {"night": night, "i": i, "item": r.get("item"), "kind": r["kind"], "cue": r["you"], "cls": cls,
                       "banked_idx": (r.get("fetch") or {}).get("idx"), "right": right}
                for a, idx in (("A", d["A"]["idx"]), ("K", dk["idx"])):
                    row[a] = {"idx": idx, "right_episode": bool(is_right(idx)) if is_right and idx is not None else False}
                row["K_rec"] = record(k, k.similarities(b.cortex, b.base, r["you"]), now, r["you"], right, is_right)
                row["arms"] = {name: record(h, h.similarities_from_delta(dl), now, r["you"], right, is_right)
                               for name, h in arms.items()}
                if row["A"]["idx"] != row["banked_idx"] or (row["A"]["idx"] is not None
                                                            and texts[row["A"]["idx"]] != r.get("recalled")):
                    void.append(i)
                if it is not None and row["A"]["right_episode"] != bool(r.get("right_episode")):
                    void.append(i)
                out_rows.append(row)
            after = [(e.strength, e.uses, e.last_used) for e in hc.episodes]
            if after != before:
                raise RuntimeError(f"night {night}: an episode changed during the screen (FREEZE broken)")
        finally:
            b.release()
            shutil.rmtree(home, ignore_errors=True)
    finally:
        clock.set(None)
    return {"night": night, "store": len(texts), "rows": out_rows, "void_rows": sorted(set(void)), "void": bool(void)}


def run(res, snaps, trunk, cortex, work, weekend=True, nights=RR.NIGHTS, log=print, size="1.5b", specs=None) -> dict:
    out = {"seed": res["seed"], "split": res.get("split"), "E": E, "fan_in": FAN_IN, "fracs": list(FRACS),
           "ca3_beta": CA3_BETA, "map_seeds": list(MAP_SEEDS), "nights": []}
    os.makedirs(work, exist_ok=True)
    for n in nights:
        snap = os.path.join(snaps, f"n{n}")
        if not os.path.isdir(snap):
            out["nights"].append({"night": n, "void": True, "missing_snapshot": True, "rows": []})
            continue
        rec = screen_night(res, n, snap, trunk, cortex, work, weekend, size, specs)
        log(f"s{res['seed']} n{n}: store {rec['store']}, void rows {rec['void_rows']}")
        out["nights"].append(rec)
    return out


# ------------------------------------------------------------------ scorer (prereg section 5; no run-time decisions)
def apply_bar(rec: dict, bar: float, margin: float = MARGIN, stmt: float | None = None) -> int | None:
    """The arm's fetch under a question bar `bar` (statements keep the class ratio): Hippocampus._compete with
    RANK_ON_RAW off, on the recorded weighted score and margin."""
    b = bar if rec["question"] else (stmt if stmt is not None else bar * STMT_RATIO)
    m = rec["margin_w"] if rec["margin_w"] is not None else 1.0
    return None if (rec["score_best"] < b or m < margin) else rec["wbest"]


def materialise(rows: list, arm: str, bar: float | None, stmt: float | None = None) -> list:
    """Rows in routing_replay._count's shape for `arm` (key arm) under `bar`: copies with row[arm_key] = {idx, right_episode}."""
    out = []
    for r in rows:
        rec = r["K_rec"] if arm == "K" else r["arms"][arm]
        idx = apply_bar(rec, bar, stmt=stmt) if bar is not None else None
        out.append({**{k: r[k] for k in ("cls", "kind", "item", "night")}, "A": r["A"],
                    "X": {"idx": idx, "right_episode": bool(idx is not None and rec.get("right_w") and idx == rec["wbest"])}})
    return out


def fit_bar(fire_rows: list, arm: str, target_fires: int) -> float:
    """The registered bar rule: the smallest question bar on the grid at which the arm fires on the pooled fire rows no
    more often than arm A does. Reads fire rows only (open, greet, never-told, untold near-duplicate), never a told row."""
    for b in BAR_GRID:
        if sum(apply_bar(r["K_rec"] if arm == "K" else r["arms"][arm], b) is not None for r in fire_rows) <= target_fires:
            return b
    return BAR_GRID[-1]


def auroc(pos: list, neg: list) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _count_arm(rows: list) -> dict:
    c = RR._count(rows, "X")
    told = [r for r in rows if r["cls"] == "told"]
    c["no_fetch"] = sum(r["X"]["idx"] is None for r in told)
    return c


def score(outs: list, validity: dict) -> dict:
    seeds, live_rows = {}, {}
    for o in outs:
        s = str(o["seed"])
        bad = bool(validity.get(s))
        nights = [n for n in o["nights"] if not n.get("void") and not n.get("void_rows")]
        nvoid = len(o["nights"]) - len(nights)
        seeds[s] = {"void": bad or nvoid > RR.VOID_NIGHTS_MAX, "void_life": bad, "void_nights": nvoid}
        live_rows[s] = [r for n in nights for r in n["rows"]]
    live = [s for s, v in seeds.items() if not v["void"]]
    verdict = {"seeds": seeds}
    if len(live) < 3:
        verdict["status"] = f"INCONCLUSIVE: {3 - len(live)} seed(s) void; no bar is read"
        return verdict
    allrows = [r for s in live for r in live_rows[s]]
    fire_rows = [r for r in allrows if r["cls"] == "fire"]
    A_fires = sum(r["A"]["idx"] is not None for r in fire_rows)
    arm_names = list(allrows[0]["arms"]) + ["K"]
    bars = {a: fit_bar(fire_rows, a, A_fires) for a in arm_names}
    bars["K_registered"] = S2.SALL_SLEPT[0]
    per_arm = {}
    for a in arm_names + ["K_registered"]:
        src, bar = ("K", bars[a]) if a == "K_registered" else (a, bars[a])
        stmt = Hippocampus.FAMILIARITY_SPONTANEOUS if a == "K_registered" else None
        per_seed = {s: _count_arm(materialise(live_rows[s], src, bar, stmt)) for s in live}
        told = [r for r in allrows if r["cls"] == "told"]
        recs = [(r["K_rec"] if src == "K" else r["arms"][src]) for r in told]
        qfire = [(r["K_rec"] if src == "K" else r["arms"][src]) for r in fire_rows if r["K_rec"]["question"]]
        per_arm[a] = {"bar": bar, "seeds": per_seed,
                      "right": sum(v["right"] for v in per_seed.values()),
                      "raw_right1": sum(1 for x in recs if x.get("rank_right") == 1),
                      "auroc_told_vs_fire": auroc([x["score_best"] for x in recs], [x["score_best"] for x in qfire])}
    A_seed = {s: _count_arm([{**r, "X": r["A"]} for r in live_rows[s]]) for s in live}
    A_right = sum(v["right"] for v in A_seed.values())
    need = max(math.ceil((1 + RR.GAIN_FRAC) * A_right), A_right + RR.GAIN_MIN)
    cells = {}
    for name in [a for a in arm_names if a.startswith("DG:")]:
        pa = per_arm[name]
        kills = []
        if pa["right"] < need:
            kills.append(f"D1 right {pa['right']} < {need} (A {A_right})")
        up = sum(pa["seeds"][s]["right"] > A_seed[s]["right"] for s in live)
        down = [s for s in live if pa["seeds"][s]["right"] < A_seed[s]["right"] - 1]
        if up < RR.DIRECTION_SEEDS or down:
            kills.append(f"D2 right rises on {up}/3 seeds, falls by >1 on {down}")
        nw = {s: pa["seeds"][s]["new_wrong_on_A_right"] for s in live}
        if any(x > RR.NEW_WRONG_MAX_SEED for x in nw.values()):
            kills.append(f"D3 new wrong on A-right rows {nw}")
        fx = {s: pa["seeds"][s]["fires"] - A_seed[s]["fires"] for s in live}
        if any(x > RR.FIRE_EXCESS_MAX_SEED for x in fx.values()):
            kills.append(f"D4 fire excess over A {fx}")
        nf = {s: pa["seeds"][s]["no_fetch"] - A_seed[s]["no_fetch"] for s in live}
        if any(x > S2.NO_FETCH_EXCESS_MAX_SEED for x in nf.values()):
            kills.append(f"D5 told no-fetch excess over A {nf}")
        cells[name] = {"right": pa["right"], "kills": kills}
    primary = f"DG:{PRIMARY_FRAC}"
    v = {"A_right": A_right, "need": need, "A_fires_pooled": A_fires, "bars": bars, "arms": per_arm, "cells": cells}
    if primary in cells:
        pr = per_arm[primary]
        extra = []
        les, shuf, kfit = per_arm.get("DGlesion"), per_arm.get(f"DGshuf:{PRIMARY_FRAC}"), per_arm["K"]
        if les is not None and pr["right"] < les["right"] + RIGHT_GAIN_MIN:
            extra.append(f"D6 sparsity is not load-bearing: DG {pr['right']} < lesion {les['right']} + {RIGHT_GAIN_MIN}")
        if shuf is not None and shuf["right"] > SHUF_MAX_FRAC * pr["right"]:
            extra.append(f"D7 shuffled-code control keeps {shuf['right']} of DG's {pr['right']} right fetches")
        ok_seeds = [n for n in cells if n.startswith(f"DG:{PRIMARY_FRAC}") and cells[n]["right"] >= need]
        if len(ok_seeds) < MAP_SEEDS_MIN:
            extra.append(f"D8 map seeds clearing D1: {len(ok_seeds)} < {MAP_SEEDS_MIN}")
        if pr["right"] < kfit["right"] + RIGHT_GAIN_MIN:
            extra.append(f"D9 DG adds nothing over its own input key refit by the same rule: DG {pr['right']} < "
                         f"S_all refit {kfit['right']} + {RIGHT_GAIN_MIN}")
        kills = cells[primary]["kills"] + extra
        v["primary"] = {"arm": primary, "kills": kills,
                        "status": "KILLED" if kills else "PICK (fetch level only: earns a live DEV life)"}
        cc = per_arm.get(f"DGcc:{PRIMARY_FRAC}")
        if cc is not None:
            v["ca3_completion"] = {"right": cc["right"], "adds_over_DG": cc["right"] - pr["right"],
                                   "carried": cc["right"] - pr["right"] >= RIGHT_GAIN_MIN}
    verdict["verdict"] = v
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
            json.dump(out, fh)
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
    print(json.dumps(v.get("verdict", {}).get("primary", v), indent=1))
    return v


if __name__ == "__main__":
    main()
