"""ORGAN-LEVEL DIAGNOSTIC for the prospective-memory driving-probe fix (finding
2026-09-20-prospective-memory-drive-v2-intervening-turns-flip).

WHY THIS EXISTS. The FIRST integrated driving-probe (a 2-turn formation->cue group) read treat=0 on the brain build.
This probe pins the reason at the ORGAN level, cheaply (no full brain build): the SFA/NMDA held x cue coincidence only
reaches its operating point (rel_A >= FIRE_THR) AFTER the intention is held across intervening turns. It sweeps the
number of intervening (distractor) turns between FORMATION and CUE, for the INTACT and the BRAIN_PMEM_LESION arms, using
the EXACT production organ (`ProspectiveMemoryOrgan`) and the EXACT probe messages the integrated driving group uses,
and records intact/lesion rel_A + the fired decision at each n. Expectation (measured): intact does NOT fire at n=0
(rel_A<FIRE_THR) but fires for n>=1; the lesion arm stays at the floor at every n -> the flip appears only once a delay
is held, which is why the zero-delay 2-turn probe was hollow.

Brain-based-only note: this drives the REAL spiking substrate (the reads are `cp_firing_states`); host supplies only the
turn text + the intention/cue text->slot mapping (the declared language boundary). Not a verdict runner (no GO/NO-GO) —
a diagnostic artifact the finding cites for its ramp table.

Run (numpy-CPU, fast; organ-only, no full brain build):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' .venv/bin/python -m research.runners._pmem_intervening_ramp_probe \
      --out research/findings/raw/_load_bearing/pmem_intervening_ramp.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import research.runners.prospective_memory_production_organ as PM  # noqa: E402

# the EXACT messages the integrated driving group uses (onebrain_regression_battery._EXTRA_TURNS: pmem_form2/d0/d1/d2/cue)
_FORM = "remind me to feed the dog when the bird sings"
_CUE = "the bird sings"
_DISTRACTORS = ["what does the cat eat", "how is the weather today", "tell me about the sky",
                "what colour is the grass", "who is at the door"]


def _arm(lesion: bool, n_intervening: int) -> dict:
    """Form the intention, hold it across n_intervening distractor turns, then read the cue turn. Returns the cue read."""
    org = PM.ProspectiveMemoryOrgan(seed=42)
    f = PM.parse_intention(_FORM)
    assert f is not None, "formation utterance did not parse"
    finfo = org.form_intention(f["action"], f["cue_clause"], f["cue_keywords"], lesion=lesion, hebbian_lesion=False)
    for i in range(n_intervening):
        d = _DISTRACTORS[i % len(_DISTRACTORS)]
        assert not PM.cue_present(d, org.cue_keywords), "a distractor is a cue: " + d
        org.read_turn(d)
    cue = org.read_turn(_CUE)
    return {"n_intervening": n_intervening, "lesion": lesion, "is_cue": cue["is_cue"],
            "fired": bool(cue["fired"]), "rel": round(float(cue["rel"]), 4),
            "held": round(float(cue["held"]), 4), "threshold": float(cue["threshold"]),
            "held_after_lesion": finfo.get("held_after_lesion")}


def main():
    ap = argparse.ArgumentParser(description="Organ-level prospective-memory intervening-turn ramp diagnostic.")
    ap.add_argument("--out", default="research/findings/raw/_load_bearing/pmem_intervening_ramp.json")
    ap.add_argument("--max-intervening", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for n in range(0, args.max_intervening + 1):
        intact = _arm(False, n)
        lesion = _arm(True, n)
        rows.append({"n_intervening": n, "intact_rel": intact["rel"], "intact_fired": intact["fired"],
                     "lesion_rel": lesion["rel"], "lesion_fired": lesion["fired"],
                     "flip": intact["fired"] != lesion["fired"], "threshold": intact["threshold"]})
        print("n=%d intact(rel=%.4f fired=%s) lesion(rel=%.4f fired=%s) FLIP=%s"
              % (n, intact["rel"], intact["fired"], lesion["rel"], lesion["fired"],
                 intact["fired"] != lesion["fired"]), flush=True)

    # ATTRIBUTION (tools.lab): at the operating point (the deepest held delay measured), how much of the cue-turn
    # release is owed to the HELD LATCH vs the lesioned floor? treatment = intact rel_A; control = BRAIN_PMEM_LESION
    # rel_A at the SAME n (the cue drive is identical in both arms; only the latch is zeroed). A large attributable
    # fraction = the fire is the held x cue coincidence, not the cue alone (the load-bearing spiking property).
    from tools.lab import attributable_to
    deepest = rows[-1] if rows else {"intact_rel": 0.0, "lesion_rel": 0.0, "n_intervening": None}
    fire_attribution = attributable_to(
        "prospective cue-fire rel owed to the held latch (intact vs BRAIN_PMEM_LESION at n=%s)" % deepest["n_intervening"],
        float(deepest["intact_rel"]), float(deepest["lesion_rel"]))

    out = {"runner": "research.runners._pmem_intervening_ramp_probe", "kind": "diagnostic",
           "form": _FORM, "cue": _CUE, "cue_keywords": PM._cue_keywords(_CUE.replace("the ", "")),
           "fire_thr": float(rows[0]["threshold"]) if rows else None,
           "first_firing_n": next((r["n_intervening"] for r in rows if r["intact_fired"]), None),
           "fire_attribution": fire_attribution, "ramp": rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=str)
    print("[saved]", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
