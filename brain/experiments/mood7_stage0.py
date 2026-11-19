# Capability: Affect
# Record: brain/research/2026-11-19-mood7-design.md
# Expected-fail: the pair-axis appraisal is a text-side scaffold whose neutral |v| read 0.24-0.30 in iteration 5
"""Mood iteration 7, Stage 0: does the cortex's pair-axis appraisal gate bad news apart from neutral and surprising
events? Reads only: no Brain change, no generation (design doc section 20, prereg).

Per seed: grow Affect(readout="pairs") from draw_pairs(seed); for every DEV-FIT / DEV-VAL text record v (pair axis),
v under the sign-shuffled axis, and surprise (brain.novelty: max word surprisal from the birth state). Fit on DEV-FIT:
dz = max(p95 |v| NEU, p90 |v| SNEU+HARD); TAU 4 fixed; B = 0.15 dz. Simulate on DEV-VAL the conversation
T1 event / T2, T3 fillers / T4 probe with m += (v_gated - m) / 4 and the low pole firing while m <= -B.
S0e: on DEV-CONV a 1.5B Brain stores the T1 telling and reports whether ordinary recall (or the inquiry gate) would
take T2-T4 before a mood branch could.

  python -m brain.experiments.mood7_stage0 --seed 42 --out <json> [--model 1.5b] [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import tempfile
import time

import numpy as np
import torch

from brain.affect import Affect, draw_pairs
from brain.brain import BIRTH, Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments import mood7_banks as mb
from brain.novelty import novelty

TAU, B_FRAC = 4.0, 0.15
CLASSES = ("NEG", "NEU", "SNEU", "HARD", "POS")


def auroc(lo, hi):
    """P(a draw from `hi` exceeds a draw from `lo`), ties half."""
    s = 0.0
    for a in hi:
        for b in lo:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / max(len(hi) * len(lo), 1)


def fires(vs, dz, b):
    """Low-pole fire per turn for a conversation's appraisals vs (dead zone dz, bar b, TAU 4)."""
    m, out = 0.0, []
    for v in vs:
        vg = v if abs(v) > dz else 0.0
        m += (vg - m) / TAU
        out.append(m <= -b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    cortex = Cortex(resolve_model(a.model))
    _, base = cortex.read(BIRTH, None)
    aff = Affect(enabled=False, readout="pairs")
    aff.grow(cortex, base, pairs=draw_pairs(random.Random(a.seed)))
    _, shuf_axis = aff.shuffled(random.Random(f"{a.seed}|shuf"))

    def read(text):
        return {"v": round(aff.project_pure(cortex, base, text)[1], 4),
                "v_shuf": round(aff.project_pure(cortex, base, text, axis=shuf_axis)[1], 4),
                "surprise": round(novelty(cortex, base, text), 3)}

    rows = {}
    for bank in ("DEV_FIT", "DEV_VAL"):
        rows[bank] = {c: [dict(text=t, **read(t)) for t in mb.BANKS[bank][c]] for c in CLASSES}
    filler_v = [read(t)["v"] for t in mb.FILLERS]
    probe_v = read(mb.PROBE)["v"]

    fit = rows["DEV_FIT"]
    absv = lambda c: [abs(r["v"]) for r in fit[c]]  # noqa: E731
    dz = max(float(np.percentile(absv("NEU"), 95)), float(np.percentile(absv("SNEU") + absv("HARD"), 90)))
    b = B_FRAC * dz

    val = rows["DEV_VAL"]
    conv = {c: [fires([r["v"]] + filler_v + [probe_v], dz, b) for r in val[c]] for c in CLASSES}
    t1 = {c: float(np.mean([f[0] for f in conv[c]])) for c in CLASSES}
    all_turns = {c: float(np.mean([x for f in conv[c] for x in f])) for c in CLASSES}
    neg_v = [r["v"] for r in val["NEG"]]
    sneu_v = [r["v"] for r in val["SNEU"]]
    auc_v = auroc([-x for x in sneu_v], [-x for x in neg_v])          # NEG more negative than SNEU
    auc_s = auroc([r["surprise"] for r in val["SNEU"]], [r["surprise"] for r in val["NEG"]])
    auc_s_sym = max(auc_s, 1 - auc_s)
    auc_shuf = auroc([-r["v_shuf"] for r in val["SNEU"]], [-r["v_shuf"] for r in val["NEG"]])

    # S0e: would ordinary recall (or the inquiry gate) take T2-T4 after the T1 telling is stored?
    blocked, n_turns = 0, 0
    detail = []
    with tempfile.TemporaryDirectory() as home:
        br = Brain(home, cortex=cortex)
        n0 = len(br.hippocampus.episodes)
        for c in ("NEG", "NEU", "SNEU"):
            for t in mb.BANKS["DEV_CONV"][c]:
                del br.hippocampus.episodes[n0:]
                br.hippocampus.store(br.cortex, br.base, t)
                for turn in mb.FILLERS + [mb.PROBE]:
                    ep, _ = br.hippocampus.recall(br.cortex, br.base, turn, question=turn.endswith("?"))
                    inq = br.inquiry is not None and br.inquiry_episode() is not None and \
                        not turn.endswith("?") and br._novelty(turn) > br.inquiry
                    took = (ep is not None and ep.text.endswith(t)) or inq
                    blocked += bool(ep is not None or inq)
                    n_turns += 1
                    detail.append({"cls": c, "t1": t, "turn": turn, "recalled": None if ep is None else ep.text,
                                   "inquiry": bool(inq), "recalled_t1": bool(took and ep is not None)})
    s0e = blocked / max(n_turns, 1)

    crit = {"S0a NEG T1 fire >= 0.70": (round(t1["NEG"], 3), t1["NEG"] >= 0.70),
            "S0b NEU+SNEU fire <= 5% of turns": (round((all_turns["NEU"] + all_turns["SNEU"]) / 2, 3),
                                                 (all_turns["NEU"] + all_turns["SNEU"]) / 2 <= 0.05),
            "S0c HARD fire <= 15% of turns": (round(all_turns["HARD"], 3), all_turns["HARD"] <= 0.15),
            "S0d AUROC v NEG vs SNEU >= 0.80, surprise <= 0.65": (round(auc_v, 3), round(auc_s_sym, 3),
                                                                   auc_v >= 0.80 and auc_s_sym <= 0.65),
            "S0e blocked by recall/inquiry <= 20% of T2-T4": (round(s0e, 3), s0e <= 0.20)}
    out = {"seed": a.seed, "model": a.model, "dz": round(dz, 4), "B": round(b, 4), "tau": TAU,
           "filler_v": filler_v, "probe_v": probe_v, "t1_fire": t1, "turn_fire": all_turns,
           "auroc_v_neg_sneu": auc_v, "auroc_surprise_neg_sneu": auc_s, "auroc_shuf_neg_sneu": auc_shuf,
           "abs_v_mean": {bk: {c: round(float(np.mean([abs(r["v"]) for r in rows[bk][c]])), 3) for c in CLASSES}
                          for bk in rows},
           "sign_neg": {bk: round(float(np.mean([r["v"] < 0 for r in rows[bk]["NEG"]])), 3) for bk in rows},
           "criteria": crit, "PASS": all(v[-1] for v in crit.values()), "rows": rows, "s0e_detail": detail,
           "host": socket.gethostname(), "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("seed", "dz", "B", "t1_fire", "turn_fire", "auroc_v_neg_sneu",
                                           "auroc_surprise_neg_sneu", "auroc_shuf_neg_sneu", "criteria", "PASS")},
                     indent=1))


if __name__ == "__main__":
    main()
