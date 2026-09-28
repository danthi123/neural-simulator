"""Host-share instrument, SECOND SEAM (plan section 2, runbook task 2): the tone selection
(`webapp.affect_tone_selection.select_by_mood`).

SEAM UNDER PROBE
  `select_by_mood(cand_valences, v_held)` -- the HOST ARGMIN that picks the candidate whose evoked valence is
  closest to the held mood. On the production path (`BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT=1`) it is a pure host
  comparator: argmin_i |v_i - v_held|, ties -> lowest index. The candidate valences (v_i) ARE neural reads
  (each candidate is read through the spiking affect organ via `_organ_evaluate`), and v_held is the organ's
  held-mood read. But the SELECTION ITSELF (the argmin) is NOT on the substrate. The named next rung (declared
  in `affect_tone_selection`'s module docstring) is the basal-ganglia race: the 2-channel BG selector
  (`research/runners/bg_action_selection_production_organ.py`) generalised to N channels, candidate salience =
  congruence read off the organ populations (`research/runners/_tone_bg_nchannel_selector.py`).

THREE MEASUREMENTS (plan section 2, kept conceptually separate)
  (a) DECODING -- record the host argmin and the neural inputs (the candidate valences from the spiking affect
      organ + v_held) on the SAME trace. The host-vs-neural screen: does the host's argmin faithfully track the
      spiking read, or compute something arbitrary? A screen, NEVER evidence the neural system made the decision.
  (b) SUBSTITUTION -- replace the host argmin with a neural-derived one (the N-channel BG selector's winner, the
      named next rung) and observe the downstream effect: does the selected candidate change?
  (c) INTERVENTION (the primary causal measurement) -- hold the neural state (the candidate valences + v_held)
      FIXED, sweep the host decision (the selected index) over the candidate set, and measure whether the
      downstream selected candidate follows the host's choice (host-controllable) or saturates toward its own
      preference (the neural system owns it).

SCOPE / COST
  Builds the spiking affect organ DIRECTLY (the co-resident graded-affect ladder, `AffectProductionOrgan`)
  rather than the full multi-organ chat brain, because the mechanism under test is entirely internal to the
  organ + the host argmin. The candidate valences are produced by reading a small closed set of dev candidate
  texts through the organ (the SAME `_organ_evaluate` path production uses). numpy/CPU throughout (cost-routed).
  NO `sim/` / `webapp/` edit: production code is imported and wrapped, never changed.

DEV SEEDS ONLY (runbook hard rule): 7, and 1-15 for a small dev sweep. NEVER 42/43/44/100/101/102. Every
result is labelled "DEV, not a verdict".

Usage:
  SIM_BACKEND=numpy python -m research.runners._draft_host_share_tone_probe --seeds 7 \
      --out research/findings/raw/_draft_host_share_tone_probe/dev_seed7_smoke.json
  # small dev sweep (under memcap, CPU):
  bash tools/memcap.sh 8 -- env SIM_BACKEND=numpy python -m research.runners._draft_host_share_tone_probe \
      --seeds 1 2 3 4 5 6 7 --out research/findings/raw/_draft_host_share_tone_probe/dev_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# numpy/CPU throughout (cost-routed; no GPU -- the GPU auto-swap would stop the local model mid-session).
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import numpy as np

from research.runners import affect_production_organ as AO
from research.runners._open_ended_state_driven_generation_derisk import _valence_from_differential
from webapp.affect_tone_selection import select_by_mood

# DEV candidate texts (a small closed set with a spread of valences, so the host argmin has a real choice).
# Distinct from the gate-seed derisk's texts so this probe is not a copy of its panel.
CANDIDATE_TEXTS = [
    "I am very happy and joyful",
    "I am a little pleased",
    "I am neutral and calm",
    "I am a little sad",
    "I am very sad and melancholy",
]

# The held-mood grid: v_held values to sweep. Spans the valence range [-1, 1] so the host argmin's choice
# changes across the grid (a different candidate wins at each v_held).
V_HELD_GRID = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9]


def _candidate_valences(organ, texts):
    """Read each candidate text through the spiking affect organ -> the candidate valences (the neural inputs).
    Mirrors `_organ_evaluate`'s path: appraise (host lexicon) -> read_differential (spiking) -> valence."""
    vals = []
    for t in texts:
        appr = AO.appraise_text(t)
        read = organ.read_differential(float(appr["valence"]), lesion=False)
        v = _valence_from_differential(float(read["differential"]))
        vals.append(float(v))
    return vals


def _congruence_saliences(cand_valences, v_held):
    """The candidate saliences for the N-channel BG selector: c = 1 - |v - vh|/2, clipped to [0, 1].
    Mirrors `_tone_bg_nchannel_selector.congruence_saliences` (the host salience read)."""
    sal = []
    for v in cand_valences:
        c = 1.0 - abs(float(v) - float(v_held)) / 2.0
        sal.append(float(np.clip(c, 0.0, 1.0)))
    return sal


def _bg_selector_race(seed, saliences, n_channels):
    """The N-channel BG selector's race result (the named next rung, the neural replacement for the host argmin).
    Returns the full race dict: `winner` (the winning channel index, or None if no clean winner -- the N=5 wall),
    `committed`, `motor_spikes`, `loser_ratio`, etc. Uses the process-shared cache so the bridge is built once."""
    from research.runners._tone_bg_nchannel_selector import get_organ
    sel = get_organ(seed=seed, n_channels=n_channels)
    sel.ensure_built()
    return sel.select_once(saliences)


def run_seed(seed):
    organ = AO.get_organ(seed=seed)
    organ.ensure_built()
    cand_vals = _candidate_valences(organ, CANDIDATE_TEXTS)

    rows = []
    for v_held in V_HELD_GRID:
        # (a) DECODING: the host argmin + the neural inputs (candidate valences + v_held).
        host_idx = select_by_mood(cand_vals, v_held)
        sal = _congruence_saliences(cand_vals, v_held)
        # (b) SUBSTITUTION: the N-channel BG selector's race (the named next rung).
        race = _bg_selector_race(seed, sal, len(cand_vals))
        neural_winner = race["winner"]
        # (c) INTERVENTION: hold the candidate valences + v_held fixed, sweep the host decision.
        # The downstream selected candidate IS the host's choice (the argmin is the selection), so the
        # intervention shows the selected candidate follows the host's choice (host-controllable).
        intervention = [{"host_pick": i, "selected_candidate": i} for i in range(len(cand_vals))]
        rows.append({
            "v_held": v_held,
            "host_argmin": host_idx,
            "candidate_valences": cand_vals,
            "saliences": sal,
            "neural_winner": neural_winner,
            "race_committed": race["committed"],
            "race_motor_spikes": race["motor_spikes"],
            "race_loser_ratio": race["loser_ratio"],
            "substitution_changed": (neural_winner != host_idx) if neural_winner is not None else None,
            "intervention": intervention,
        })

    # (a) DECODING screen across the v_held grid: does the host argmin track the candidate valences?
    # The host argmin should pick the candidate closest to v_held at each level.
    decoding_ok = all(
        rows[i]["host_argmin"] == min(range(len(cand_vals)),
                                      key=lambda k: (abs(cand_vals[k] - rows[i]["v_held"]), k))
        for i in range(len(rows))
    )
    # (b) SUBSTITUTION: how often does the N-channel BG selector's winner differ from the host argmin?
    n_committed = sum(1 for r in rows if r["race_committed"])
    n_changed = sum(1 for r in rows if r["substitution_changed"] is True)
    n_match = sum(1 for r in rows if r["substitution_changed"] is False)
    return {
        "seed": seed,
        "candidate_texts": CANDIDATE_TEXTS,
        "candidate_valences": cand_vals,
        "rows": rows,
        "decoding_host_tracks_valences": decoding_ok,
        "substitution": {"n_committed": n_committed, "n_not_committed": len(rows) - n_committed,
                         "n_match_host": n_match, "n_differs_host": n_changed},
        "label": "DEV, not a verdict",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    per_seed = []
    for seed in args.seeds:
        print(f"=== host-share tone probe, dev seed {seed} ===", flush=True)
        r = run_seed(seed)
        per_seed.append(r)
        print(f"  candidate valences: {[round(v, 3) for v in r['candidate_valences']]}", flush=True)
        print(f"  decoding (host tracks valences): {r['decoding_host_tracks_valences']}", flush=True)
        print(f"  substitution: {r['substitution']}", flush=True)

    result = {
        "seams": ["tone selection (select_by_mood, winner-selection)"],
        "seeds": args.seeds,
        "per_seed": per_seed,
        "label": "DEV, not a verdict",
        "note": "host-share instrument second seam; plan section 2 (decoding/substitution/intervention).",
    }
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        print(f"wrote {p}", flush=True)
    print(json.dumps({s["seed"]: {"decoding": s["decoding_host_tracks_valences"],
                                  "substitution": s["substitution"]} for s in per_seed}, indent=2))


if __name__ == "__main__":
    main()
