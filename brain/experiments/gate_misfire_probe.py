# Capability: Metacognition
# Record: brain/experiments/results/2026-11-03-told-gate-1.5b.md (the lesson overrides recall when the gate fires)
"""Gate misfire: how often the "not known" familiarity gate (Brain.knows: known iff hippocampal familiarity >= bar)
fires on told questions, and how often it stays quiet on never-told ones. Pre-registration + results:
brain/experiments/results/2026-11-03-gate-misfire-1.5b.md. Measurement only: no speech for questions, nothing stored,
no Brain default changed.

  run     one seed: day 1 = the 12 FACTS told through hear; familiarity of every question under S0 and S_all (same
          episodes), unslept, then after one Brain.sleep() (--no-sleep skips it)
  report  pre-registered bars, the DEV sweep, AUROC, per-seed table over gm_s*.json

  python -m brain.experiments.gate_misfire_probe run --model 1.5b --device cuda --seed 42 --out <dir>/gm_s42.json
  python -m brain.experiments.gate_misfire_probe report --dir <dir>
"""
import argparse
import glob
import json
import os
import shutil
import time

import numpy as np
import torch

from brain.brain import Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments.abstain_in_brain import FRESH_TOLD
from brain.experiments.confab_m2 import DEV_NEVER, FRESH_NEVER3
from brain.experiments.facts import FACTS, PARA
from brain.hippocampus_keys import KeyedHippocampus

SETS = {"told": [q for _, q, _ in FACTS], "para": list(PARA), "dev_told": [q for q, _, _ in FRESH_TOLD],
        "dev_never": [q for q, _, _ in DEV_NEVER], "v3_never": [q for q, _, _ in FRESH_NEVER3]}
TOLD_EP = {"told": [s for s, _, _ in FACTS], "para": [s for s, _, _ in FACTS], "dev_told": [s for s, _, _ in FACTS]}
KEYS = ("S0", "S_all")
CONDS = ("unslept", "slept")
HERE = os.path.dirname(os.path.abspath(__file__))
F_BARS = os.path.join(HERE, "results", "2026-11-03-abstain-v2", "bars.json")
# Pre-registered (key, bar name) pairs; bar value per seed from bar_value().
PAIRS = (("S0", "0.22"), ("S_all", "F0.104"), ("S_all", "0.22"))
MISFIRE_MAX, CATCH_MIN, SEEDS_MIN = 1 / 12, 0.80, 5


def bar_value(name, seed):
    if name == "F0.104":
        return json.load(open(F_BARS))["bars"][str(seed)]["F"]["fam"]
    return float(name)


def signals(b, hips, max_q):
    out = {}
    for key, h in hips.items():
        out[key] = {}
        for st, qs in SETS.items():
            rows = []
            for i, q in enumerate(qs[:max_q or None]):
                ep, fam = h.best(b.cortex, b.base, q)
                r = {"q": q, "fam": round(fam, 5), "best": ep.text if ep is not None else None}
                if st in TOLD_EP:
                    r["right"] = r["best"] == f"User: {TOLD_EP[st][i]}"
                rows.append(r)
            out[key][st] = rows
    return out


def run(a):
    c = Cortex(model_path=resolve_model(a.model), device=a.device)
    home = os.path.join(a.home, f"s{a.seed}")
    shutil.rmtree(home, ignore_errors=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    b = Brain(home, cortex=c, sleep_seed=a.seed, sleep_on_restart=False)
    for s, _, _ in FACTS:
        b.hear(s)
    kh = KeyedHippocampus(key="S_all")   # familiarity is bar-free; same episodes (abstain_v2.keyed)
    kh.episodes, kh.enabled = b.hippocampus.episodes, b.hippocampus.enabled
    hips = {"S0": b.hippocampus, "S_all": kh}
    res = {"seed": a.seed, "model": c.model_name, "episodes": len(b.hippocampus.episodes), "conds": {}}
    res["conds"]["unslept"] = signals(b, hips, a.max_q)
    if not a.no_sleep:
        t = time.time()
        b.sleep()
        res["sleep_s"] = round(time.time() - t, 1)
        res["conds"]["slept"] = signals(b, hips, a.max_q)
    if a.device == "cuda":
        res["cuda_peak_bytes"] = torch.cuda.max_memory_allocated()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, res.get("sleep_s"), flush=True)


def fires(vals, bar):
    return sum(v < bar for v in vals)


def sweep(told, never, catch_min=CATCH_MIN):
    """Lowest bar (gate fires iff value < bar) with catch of `never` >= catch_min; returns (bar, told misfire rate,
    catch rate) or (None, None, None) if no bar reaches it."""
    for t in sorted(set(never) | set(told)) + [float("inf")]:
        bar = t if t != float("inf") else max(never + told) + 1e-6
        if fires(never, bar) / len(never) >= catch_min:
            return bar, fires(told, bar) / len(told), fires(never, bar) / len(never)
    return None, None, None


def auroc(told, never):
    """P(told familiarity > never familiarity), ties 1/2."""
    return float(np.mean([(t > n) + 0.5 * (t == n) for t in told for n in never]))


def report(d):
    runs = {}
    for f in glob.glob(os.path.join(d, "gm_s*.json")):
        r = json.load(open(f))
        runs[r["seed"]] = r
    seeds = sorted(runs)
    conds = [c for c in CONDS if all(c in runs[s]["conds"] for s in seeds)]
    V = lambda s, c, k, st: [x["fam"] for x in runs[s]["conds"][c][k][st]]  # noqa: E731
    out = {"seeds": seeds, "pairs": {}, "sweep": {}, "auroc": {}, "right_episode": {}}
    for c in conds:
        for k, name in PAIRS:
            per = {}
            for s in seeds:
                bar = bar_value(name, s)
                nv = V(s, c, k, "dev_never") + V(s, c, k, "v3_never")
                per[str(s)] = {"bar": round(bar, 5), "told_misfire": fires(V(s, c, k, "told"), bar),
                               "para_misfire": fires(V(s, c, k, "para"), bar),
                               "dev_told_misfire": fires(V(s, c, k, "dev_told"), bar),
                               "never_catch": fires(nv, bar), "n_never": len(nv),
                               "dev_never_catch": fires(V(s, c, k, "dev_never"), bar),
                               "v3_never_catch": fires(V(s, c, k, "v3_never"), bar)}
            n_t = len(V(seeds[0], c, k, "told"))
            mis = float(np.mean([p["told_misfire"] for p in per.values()])) / n_t
            para = float(np.mean([p["para_misfire"] for p in per.values()])) / n_t
            ok_seeds = sum(p["never_catch"] / p["n_never"] >= CATCH_MIN for p in per.values())
            out["pairs"][f"{c}/{k}@{name}"] = {
                "per_seed": per, "told_misfire_mean_rate": round(mis, 4), "para_misfire_mean_rate": round(para, 4),
                "never_catch_mean_rate": round(float(np.mean([p["never_catch"] / p["n_never"] for p in per.values()])), 4),
                "seeds_catch_ok": ok_seeds,
                "misfire_ok": mis <= MISFIRE_MAX + 1e-9, "catch_ok": ok_seeds >= SEEDS_MIN,
                "usable": mis <= MISFIRE_MAX + 1e-9 and ok_seeds >= SEEDS_MIN}
        for k in KEYS:
            pool = lambda st: [v for s in seeds for v in V(s, c, k, st)]  # noqa: E731
            bar, dmis, dcat = sweep(pool("dev_told"), pool("dev_never"))
            sw = {"dev_bar": bar, "dev_told_misfire_rate": dmis, "dev_never_catch_rate": dcat}
            if bar is not None:
                for st in ("told", "para", "v3_never"):
                    sw[f"test_{st}_fire_rate"] = round(fires(pool(st), bar) / len(pool(st)), 4)
                sw["per_seed_test_told_misfire"] = {str(s): fires(V(s, c, k, "told"), bar) for s in seeds}
            out["sweep"][f"{c}/{k}"] = sw
            nev = pool("dev_never") + pool("v3_never")
            out["auroc"][f"{c}/{k}"] = {"told_vs_never": round(auroc(pool("told"), nev), 4),
                                        "para_vs_never": round(auroc(pool("para"), nev), 4),
                                        "dev_told_vs_dev_never": round(auroc(pool("dev_told"), pool("dev_never")), 4)}
            out["right_episode"][f"{c}/{k}"] = {st: sum(x["right"] for s in seeds for x in runs[s]["conds"][c][k][st])
                                                for st in TOLD_EP}
    json.dump(out, open(os.path.join(d, "report.json"), "w"), indent=1)
    for name, p in out["pairs"].items():
        print(name, "misfire", p["told_misfire_mean_rate"], "para", p["para_misfire_mean_rate"], "catch",
              p["never_catch_mean_rate"], "seeds_ok", p["seeds_catch_ok"], "USABLE" if p["usable"] else "not usable")
    for name, sw in out["sweep"].items():
        print("sweep", name, sw)
    print("auroc", out["auroc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    ap.add_argument("--dir")
    ap.add_argument("--home", default="/home/dant123/Projects/sim/_scratch/brainv2/gate-misfire/homes")
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--no-sleep", action="store_true")
    a = ap.parse_args()
    run(a) if a.phase == "run" else report(a.dir)


if __name__ == "__main__":
    main()
