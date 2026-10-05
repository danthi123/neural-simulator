"""Organ-level DEV de-risk of the BG neural seam (`bg_action_selection_production_organ`, region `neural-seam`,
flag BRAIN_BG_NEURAL_SEAM). Not a verdict: dev seeds only. The live-turn measurement is
`research/runners/host_share_bg_seam.py`.

Per seed and arm (intact, lesion value / latch / arousal / direct_path) and per message class (content-empty,
one word, content), `--trials` races on ONE warm organ: P(commit), P(STAY-SILENT | commit), the membrane range over
every step, and the READOUT INTERVENTION: the same trial re-read by alternative host rules (latch threshold x0.5 / x2,
the production first-to-12 motor crossing, first-to-6, first-to-24, motor majority). The fraction of trials whose
committed action changes under a rule is that rule's share of the commitment (neural state held fixed).

  SIM_BACKEND=numpy python -m research.runners._bg_neural_seam_derisk --seeds 1 2 3 --trials 20 \
      --out research/findings/raw/_bg_neural_seam/dev_organ_s1-3.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from research.runners import bg_action_selection_production_organ as BG
from tools.lab import attributable_to

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
MESSAGES = {"empty": ["...", "?", "..", "!"], "one_word": ["hi", "ok", "yes", "hmm"],
            "content": ["what does the dog chase", "tell me about cats", "the wolf bites the apple",
                        "what does the cat eat"]}
ARMS = (None, "value", "latch", "arousal", "direct_path")


def run_arm(seed, lesion, trials):
    org = BG.NeuralSeamSelector(seed=seed, lesion=lesion)
    out = {}
    for cls, msgs in MESSAGES.items():
        rows = [org.select_once(msgs[i % len(msgs)]) for i in range(int(trials))]
        c = [r for r in rows if r["committed"]]
        dis = {k: round(sum(1 for r in rows if r["readouts"][k] != r["winner"]) / len(rows), 4)
               for k in rows[0]["readouts"]}
        out[cls] = {"n": len(rows), "p_commit": round(len(c) / len(rows), 4),
                    "p_silent_given_commit": (round(sum(1 for r in c if r["winner"] == BG.STAY_SILENT) / len(c), 4)
                                              if c else None),
                    "winners": "".join("-" if r["winner"] is None else str(r["winner"]) for r in rows),
                    "readout_change_fraction": dis,
                    "v_range": [round(min(r["v_min"] for r in rows), 1), round(max(r["v_max"] for r in rows), 1)]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args(argv)
    if set(a.seeds) & set(GATE_SEEDS):
        raise SystemExit("dev de-risk only: gate seeds are refused")
    res = {"probe": "bg_neural_seam_organ_derisk", "label": "DEV, not a verdict", "seeds": a.seeds,
           "constants": {k: getattr(BG, k) for k in ("VALUE_WEIGHT", "HOLD_TONIC_PA", "HEARD_PA", "LATCH_INTERNAL_W",
                                                     "LATCH_CROSS_W", "MOTOR_TO_LATCH_W", "T_DRIVE", "T_HOLD",
                                                     "READ_WINDOW", "LATCH_READ_RATE")},
           "per_seed": {}}
    for s in a.seeds:
        res["per_seed"][s] = {str(arm or "intact"): run_arm(s, arm, a.trials) for arm in ARMS}
        print(json.dumps({"seed": s, "summary": {arm: {c: [v["p_commit"], v["p_silent_given_commit"]]
                                                        for c, v in d.items()}
                                                  for arm, d in res["per_seed"][s].items()}}), flush=True)
        # ATTRIBUTION (gates/attribution_required): ask WHOSE the p_commit change is, per lesion arm, rather
        # than just banking intact vs lesioned numbers side by side (the gap#5 shape -- two correct numbers one
        # key apart, never subtracted). Treatment = the intact (unlesioned) organ's content-class p_commit;
        # control = the same statistic with that pathway lesioned. Diagnostic only; does not touch ARMS,
        # trial construction or any registered gate bar.
        intact_content_p_commit = res["per_seed"][s]["intact"]["content"]["p_commit"]
        for arm in ARMS:
            if arm is None:
                continue
            lesioned_p_commit = res["per_seed"][s][str(arm)]["content"]["p_commit"]
            attributable_to("seed %d lesion=%s content p_commit (intact vs lesioned)" % (s, arm),
                             intact_content_p_commit, lesioned_p_commit)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
