# Capability: Episodic recall
# Record: brain/research/2026-11-25-routing-ceiling-prereg.md (prereg); results/2026-11-25-routing-screen-read.md;
#   brain/hippocampus_keys.py (S<l>, S_all); research/findings/2026-06-11-cortex-fixed-expansion-decorrelation-NEGATIVE.md
# Expected-fail: partly. A measurement, not a method. DG(S0) is predicted not to reorder (a random sparse map keeps
#   rank order to first order); near-duplicate separation and a paraphrase-robust cue are MISSING (PREREQUISITES.md),
#   so every key may rank the right episode deep on hub-capture rows.
"""Offline routing ceiling on the K-P(-rerun) night snapshots: where does the right episode rank under each key?
Host-side harness only; nothing here is a brain default and hippocampus.py is not edited.

For each non-void night of the 11-25 routing replay, the snapshot's store is opened WITH the final trunk's episode
deltas (routing_replay.open_snapshot, which checks every key against keys.pt), and every told row's cue is read once
from base (cortex.read + matrix_delta, as Hippocampus.similarities does). Keys (prereg section 3):
  S0        layer-0 matrix delta, centred cosine (must reproduce the 11-25 replay's raw_winner: K-sane)
  DG:E:f:s  centred S0 through a fixed sparse random map (E units, each the signed sum of F=128 input dims, map seed s),
            top fraction f kept as a binary code; similarity = overlap / k; same map for cue and store
  S0q       the cue's S0 key minus the mean S0 key of the night's untold question cues (never / untold near-duplicate,
            ending in "?"), then centred cosine; no told cue enters the mean
  S<l>      one layer's matrix delta (l in 6, 12, 18, 23), centred cosine
  S_all     every layer, blocks centred + normalised, averaged (hippocampus_keys.centred_cos)
  S0drift   every stored episode re-keyed by reading its own text through the night's cortex (S0)

    python -m brain.experiments.routing_ceiling run --replay R.json --life L.json --snaps DIR --trunk T --out O.json
    python -m brain.experiments.routing_ceiling table O_s319.json O_s320.json O_s321.json [--out table.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn.functional as F

from brain import clock

LAYERS = (6, 12, 18, 23)
DG_E = (4096, 16384)
DG_F = 128
DG_FRAC = (0.02, 0.05)
DG_SEEDS = (42, 43, 44, 100, 101, 102)


# ------------------------------------------------------------------ similarity primitives
def centred_cos_one(K: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Hippocampus.similarities' math: centre on the stored keys' mean, L2-normalise, cosine."""
    if K.shape[0] > 1:
        mu = K.mean(0)
        K, q = K - mu, q - mu
    return F.normalize(K, dim=1) @ F.normalize(q, dim=0)


def dg_map(dim: int, E: int, seed: int, fan_in: int = DG_F):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, dim, (E, fan_in), generator=g)
    sign = torch.randint(0, 2, (E, fan_in), generator=g).float() * 2 - 1
    return idx, sign


def dg_store(K: torch.Tensor, idx, sign) -> tuple:
    """The store side of DG(S0), once per night: centre on the stored keys' mean, then the signed fan-in sums."""
    mu = K.mean(0) if K.shape[0] > 1 else torch.zeros_like(K[0])
    return mu, _fanin(K - mu, idx, sign)


def _fanin(X, idx, sign, chunk: int = 1024):
    E = idx.shape[0]
    h = torch.empty(X.shape[0], E)
    for a in range(0, E, chunk):
        h[:, a:a + chunk] = (X[:, idx[a:a + chunk]] * sign[a:a + chunk]).sum(-1)
    return h


def _binary(h: torch.Tensor, frac: float) -> torch.Tensor:
    k = max(1, round(frac * h.shape[1]))
    code = torch.zeros_like(h)
    code.scatter_(1, h.topk(k, dim=1).indices, 1.0)
    return code


def dg_sims_cached(store_h: torch.Tensor, mu: torch.Tensor, q: torch.Tensor, idx, sign, frac: float) -> torch.Tensor:
    """DG(S0) similarity of a cue against a cached store: overlap / k of the binary top-k codes."""
    cs = _binary(store_h, frac)
    cq = _binary(_fanin((q - mu)[None], idx, sign), frac)[0]
    return cs @ cq / cq.sum()


def dg_sims(K: torch.Tensor, q: torch.Tensor, idx, sign, frac: float) -> torch.Tensor:
    mu, h = dg_store(K, idx, sign)
    return dg_sims_cached(h, mu, q, idx, sign, frac)


def live_store(blocks: dict) -> dict:
    """{state index: (n, d) stored blocks} -> the cache sall_live reads (KeyedHippocampus.similarities' math)."""
    out = {}
    for l, Kb in blocks.items():
        Kn = F.normalize(Kb.float(), dim=1)
        mun = Kn.mean(0) if Kn.shape[0] > 1 else torch.zeros_like(Kn[0])
        out[l] = (Kn.half(), mun, Kn @ mun, float(mun @ mun))
    return out


def sall_live(live_K: dict, dl: list, n: int) -> torch.Tensor:
    """Live S_all: each block unit-normalised, the concatenation centred on the stored mean, cosine."""
    B = len(live_K)
    qk, muk, qmu, mumu = torch.zeros(n), torch.zeros(n), 0.0, 0.0
    for l, (Kn, mun, kmu, m2) in live_K.items():
        q = F.normalize(dl[l].flatten().float().cpu(), dim=0)
        qk += Kn.float() @ q
        muk += kmu
        qmu += float(q @ mun)
        mumu += m2
    num = qk - qmu - muk + mumu
    return num / max(B - 2 * qmu + mumu, 1e-12) ** 0.5 / (B - 2 * muk + mumu).clamp_min(1e-12).sqrt()


def rank_of(sims: torch.Tensor, right: list) -> dict:
    """Rank (1 = first) of the best-ranked right episode, and its margin to the top-1."""
    best = max(float(sims[i]) for i in right)
    rs = set(right)
    ties = sum(1 for j in range(len(sims)) if float(sims[j]) == best and j not in rs)   # a tie counts against it
    return {"rank": 1 + int((sims > best).sum()) + ties, "margin": float(sims.max()) - best,
            "top": int(torch.argmax(sims))}


# ------------------------------------------------------------------ one night
def ceiling_night(res: dict, rep_night: dict, snap: str, trunk: str, cortex, work: str, weekend: bool,
                  dg: bool = True, drift: bool = True) -> dict:
    from brain.eval import life
    from brain.experiments.routing_replay import open_snapshot
    LB = life.use_split(res.get("split", "dev"))
    night = rep_night["night"]
    sim = life.SimClock()
    clock.set(sim.now)
    home = os.path.join(work, f"s{res['seed']}_n{night}")
    sim.to(life.day_start(night + 1, weekend))
    b = open_snapshot(snap, home, cortex, res["brain_kw"], trunk=trunk, with_deltas=True)
    try:
        hc, cx, base = b.hippocampus, b.cortex, b.base
        eps = hc.episodes
        texts = [e.text for e in eps]
        K0 = torch.stack([e.key for e in eps]).float()

        def cue_delta(text):
            _, st = cx.read(f"User: {text.strip()}\n\n", base)   # decide() strips the cue
            return cx.matrix_delta(st, base)

        # Routing reads the plain change where one is kept (completion / contrast episodes), as KeyedHippocampus does
        rd = [getattr(e, "route_delta", None) or e.delta for e in eps]
        layers = [l for l in LAYERS if 3 * l + 1 < len(rd[0]) and rd[0][3 * l + 1] is not None]
        Kl = {l: torch.stack([d[3 * l + 1].flatten().float() for d in rd]) for l in layers}
        qcues = [r["cue"] for r in rep_night["rows"]
                 if r["cls"] == "fire" and r["kind"] in ("never", "neardup") and r["cue"].strip().endswith("?")]
        qmean = torch.stack([cx.key_of(cue_delta(c)).cpu() for c in qcues]).mean(0) if qcues else None
        Kd = None
        if drift:
            Kd = torch.stack([cx.key_of(cx.matrix_delta(cx.read(t + "\n\n", base)[1], base)).cpu() for t in texts])
        maps = {}
        if dg:
            for E in DG_E:
                for s in DG_SEEDS:
                    idx, sign = dg_map(K0.shape[1], E, s)
                    maps[(E, s)] = (idx, sign) + dg_store(K0, idx, sign)
        n_route = sum(1 for e in eps if getattr(e, "route_delta", None))
        # S_all store side once per night. S_all (prereg 3.5, hippocampus_keys.centred_cos): each block centred on its
        # stored mean, normalised, block cosines averaged. S_all_live (KeyedHippocampus.similarities, the key the 11-03
        # / 11-14 / 11-18 results used): each block normalised first, then the concatenation centred (review 11-25).
        sall_K, live_K = {}, {}
        for l, d in enumerate(rd[0]):
            if d is not None:
                Kb = torch.stack([x[l].flatten().float() for x in rd])
                mu = Kb.mean(0) if Kb.shape[0] > 1 else torch.zeros_like(Kb[0])
                sall_K[l] = (F.normalize(Kb - mu, dim=1).half(), mu)
                live_K.update(live_store({l: Kb}))
                del Kb
        for e in eps:          # nothing below reads the stored deltas (memory, review 11-25)
            e.delta = None
            if getattr(e, "route_delta", None) is not None:
                e.route_delta = None
        del rd
        top_strength = sorted(range(len(eps)), key=lambda i: -hc.effective_strength(eps[i], clock.now()))[:3]
        rows = []
        for r in rep_night["rows"]:
            if r["cls"] != "told":
                continue
            it = LB.ALL[r["item"]]
            right = [i for i, t in enumerate(texts) if life.episode_is_source(it, t)]
            row = {"night": night, "i": r["i"], "item": r["item"], "cue": r["cue"], "A_idx": r["A"]["idx"],
                   "A_right": r["A"]["right_episode"], "replay_raw_winner": r["raw_winner"], "right": right,
                   "A_text": texts[r["A"]["idx"]] if r["A"]["idx"] is not None else None,
                   "A_is_hub": r["A"]["idx"] in top_strength, "keys": {}}
            dl = cue_delta(r["cue"])
            q0 = cx.key_of(dl).cpu()
            s0 = centred_cos_one(K0, q0)
            row["S0_top"] = int(torch.argmax(s0))
            if not right:
                rows.append(row)
                continue
            row["keys"]["S0"] = rank_of(s0, right)
            if qmean is not None:
                row["keys"]["S0q"] = rank_of(centred_cos_one(K0, q0 - qmean), right)
            for l in layers:
                row["keys"][f"S{l}"] = rank_of(centred_cos_one(Kl[l], dl[3 * l + 1].flatten().float().cpu()), right)
            sall = 0.0
            for l, (Kn, mu) in sall_K.items():
                sall = sall + Kn.float() @ F.normalize(dl[l].flatten().float().cpu() - mu, dim=0)
            row["keys"]["S_all"] = rank_of(sall / len(sall_K), right)
            row["keys"]["S_all_live"] = rank_of(sall_live(live_K, dl, len(eps)), right)
            if Kd is not None:
                row["keys"]["S0drift"] = rank_of(centred_cos_one(Kd, q0), right)
            for (E, s), (idx, sign, mu, h) in maps.items():
                for f in DG_FRAC:
                    row["keys"][f"DG:{E}:{f}:{s}"] = rank_of(dg_sims_cached(h, mu, q0, idx, sign, f), right)
            rows.append(row)
    finally:
        b.release()
        clock.set(None)
    sane = all(x["S0_top"] == x["replay_raw_winner"] for x in rows)
    return {"night": night, "store": len(texts), "n_qcues": len(qcues), "n_route_delta": n_route, "rows": rows,
            "sane": sane}


def run(res: dict, rep: dict, snaps: str, trunk: str, cortex, work: str, dg=True, drift=True, log=print) -> dict:
    out = {"seed": res["seed"], "layers": list(LAYERS), "dg": {"E": list(DG_E), "F": DG_F, "frac": list(DG_FRAC),
           "seeds": list(DG_SEEDS)}, "nights": []}
    os.makedirs(work, exist_ok=True)
    for n in rep["nights"]:
        if n.get("void"):
            out["nights"].append({"night": n["night"], "void": True, "rows": []})
            continue
        rec = ceiling_night(res, n, os.path.join(snaps, f"n{n['night']}"), trunk, cortex, work, rep["weekend"], dg, drift)
        log(f"s{res['seed']} n{rec['night']}: store {rec['store']}, told {len(rec['rows'])}, sane {rec['sane']}")
        out["nights"].append(rec)
    return out


# ------------------------------------------------------------------ table + kill rules (prereg section 5)
def _measures(rows: list, key: str) -> dict:
    rs = [r for r in rows if key in r["keys"]]
    ranks = [r["keys"][key]["rank"] for r in rs]
    aw = [r for r in rs if r["A_idx"] is not None and not r["A_right"]]
    return {"n": len(rs), "r1": sum(x == 1 for x in ranks), "r3": sum(x <= 3 for x in ranks),
            "r5": sum(x <= 5 for x in ranks), "median_rank": statistics.median(ranks) if ranks else None,
            "A_wrong_n": len(aw), "A_wrong_r3": sum(r["keys"][key]["rank"] <= 3 for r in aw),
            "A_wrong_gt5": sum(r["keys"][key]["rank"] > 5 for r in aw),
            "median_margin": statistics.median([r["keys"][key]["margin"] for r in rs]) if rs else None}


def table(outs: list) -> dict:
    per_seed, pooled_rows = {}, []
    void = []
    insane = {}
    for o in outs:   # prereg 2: an insane night (S0 top-1 != replay raw winner) is void and dropped; a seed with more
        #              than 2 dropped nights (replay + ceiling) is void, as in the 11-25 screen
        ok = [n for n in o["nights"] if not n.get("void") and n.get("sane", True)]
        insane[str(o["seed"])] = [n["night"] for n in o["nights"] if not n.get("void") and not n.get("sane", True)]
        if sum(1 for n in o["nights"] if n.get("void") or not n.get("sane", True)) > 2:
            void.append(o["seed"])
        rows = [r for n in ok for r in n["rows"]]
        pooled_rows += rows
        keys = sorted({k for r in rows for k in r["keys"]})
        per_seed[str(o["seed"])] = {k: _measures(rows, k) for k in keys}
        per_seed[str(o["seed"])]["unscorable"] = sum(1 for r in rows if not r["right"])
    keys = sorted({k for r in pooled_rows for k in r["keys"]})
    pooled = {k: _measures(pooled_rows, k) for k in keys}
    seeds = [str(o["seed"]) for o in outs]
    if len(seeds) != 3 or any("S0" not in per_seed[s] for s in seeds):
        raise SystemExit(f"table: need 3 seeds with scored rows, got {seeds}")
    s0 = pooled["S0"]["r1"]

    def gain(k, rows=pooled_rows):     # k's right@1 minus S0's, on the rows that have k (review 11-25)
        rs = [r for r in rows if k in r["keys"]]
        return sum(r["keys"][k]["rank"] == 1 for r in rs) - sum(r["keys"]["S0"]["rank"] == 1 for r in rs)
    n_scored = pooled["S0"]["n"]
    v = {"K_sane": "VOID" if void else "ok", "void_seeds": void, "insane_nights": insane}
    # K-sane, second half: S0 top-1 right count equals the replay raw winner's (same rows, same store)
    v["S0_raw_winner_right"] = sum(1 for r in pooled_rows if r["right"] and r["replay_raw_winner"] in r["right"])
    if v["S0_raw_winner_right"] != s0:
        v["K_sane"] = "VOID"
    dg_cells = {}
    for E in DG_E:
        for f in DG_FRAC:
            ks = [f"DG:{E}:{f}:{s}" for s in DG_SEEDS if f"DG:{E}:{f}:{s}" in pooled]
            if not ks:
                continue
            gains = [gain(k) for k in ks]
            dg_cells[f"{E}:{f}"] = {"gains": gains, "alive": min(gains) >= 5 and len({g > 0 for g in gains}) == 1}
    v["DG"] = {"cells": dg_cells, "status": ("not run" if not dg_cells else
               "alive" if any(c["alive"] for c in dg_cells.values()) else "dead"),
               "hub_rule": "judged by hand from rows' A_is_hub (prereg 5, third DG rule)"}
    if "S0q" in pooled:
        g = gain("S0q")
        v["S0q"] = {"gain": g, "status": "alive" if g >= 10 else "dead"}
    deep = {}
    by_seed = {str(o["seed"]): [r for n in o["nights"] if not n.get("void") and n.get("sane", True)
                                for r in n["rows"]] for o in outs}
    for k in [k for k in [f"S{l}" for l in LAYERS] + ["S_all", "S_all_live"] if k in pooled]:
        wins = sum(1 for s in seeds if gain(k, by_seed[s]) >= 0.10 * per_seed[s]["S0"]["n"])
        deep[k] = {"gain": gain(k), "seeds_10pct": wins, "status": "alive" if wins >= 2 else "dead"}
    v["deep"] = deep
    if "S0drift" in pooled:
        g = gain("S0drift")
        v["drift"] = {"gain": g, "status": "not the cause" if abs(g) <= 5 else "possible cause"}
    buried = all(pooled[k]["A_wrong_gt5"] > pooled[k]["A_wrong_n"] / 2 for k in keys)
    v["separation_dead_on_store"] = buried
    return {"verdict": v, "pooled": pooled, "seeds": per_seed, "n_scored": n_scored}


def main(argv=None, cortex=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--replay", required=True, help="the 11-25 routing_replay output for this seed")
    r.add_argument("--life", required=True)
    r.add_argument("--snaps", required=True)
    r.add_argument("--trunk", required=True, help="the life's final trunk home (episode deltas)")
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--work", default=None)
    r.add_argument("--no-dg", action="store_true")
    r.add_argument("--no-drift", action="store_true")
    t = sub.add_parser("table")
    t.add_argument("outs", nargs="+")
    t.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        with open(a.life) as fh:
            res = json.load(fh)
        with open(a.replay) as fh:
            rep = json.load(fh)
        if rep["seed"] != res["seed"]:
            raise SystemExit(f"replay seed {rep['seed']} != life seed {res['seed']}")
        if cortex is None:
            from brain.cortex import Cortex, resolve_model
            cortex = Cortex(model_path=resolve_model(a.model), device=a.device)
        out = run(res, rep, a.snaps, a.trunk, cortex, a.work or a.out + ".work", not a.no_dg, not a.no_drift)
        with open(a.out, "w") as fh:
            json.dump(out, fh)
        return out
    outs = []
    for f in a.outs:
        with open(f) as fh:
            outs.append(json.load(fh))
    tab = table(outs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(tab, fh, indent=1)
    print(json.dumps(tab["verdict"], indent=1))
    for k, m in tab["pooled"].items():
        if not k.startswith("DG:"):
            print(f"{k:8s} n {m['n']:3d}  r1 {m['r1']:3d}  r3 {m['r3']:3d}  r5 {m['r5']:3d}  med {m['median_rank']}  "
                  f"A-wrong r<=3 {m['A_wrong_r3']}/{m['A_wrong_n']}  >5 {m['A_wrong_gt5']}")
    return tab


if __name__ == "__main__":
    main()
