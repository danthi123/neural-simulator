"""ORGAN-LEVEL VERIFY for the W5 affective-ToM production organ (empathy), the direct load-bearing proof that does
NOT need the slow full ChatBrain build (the handler no-regression is `_affective_tom_flip_soak.py`).

Per seed, reads the OTHER-tagged region on a BAD and a GOOD other-situation, INTACT and LESIONED, and asserts:
  * VARY  (the faculty changes the output): the intact bad-other lead ("That sounds really hard ...") differs from the
    intact good-other lead ("That's wonderful ..."), and the neural tone_sign flips (-1 vs +1). Load-bearing (B).
  * LESION (the finding's other-output oracle): cutting the OTHER region's affect_out collapses the recall
    differential (|diff| -> ~0) so tone_sign -> 0 and the empathic lead VANISHES, while the appraised valence is
    unchanged. The empathic tone rides the OTHER-region SPIKING read, not a host `if valence<0`.
  * FLAG-OFF inertness: an ordinary message detects no other-agent -> observe_turn acted=False, no lead, no bridge
    built (byte-identical + no RNG perturbation).

Writes a durable artifact for the finding to cite.

  Run: SIM_BACKEND=numpy python -m research.runners._affective_tom_production_organ_verify --seeds 42 43 44
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners.affective_tom_production_organ as O

OUT = _REPO / "research" / "findings" / "raw" / "_affective_tom_prodflip" / "organ_verify.json"


def _read(seed, valence_sign, lesion):
    O._ORGAN = O.AffectiveToMOrgan(seed=int(seed))
    r = O.get_organ(seed=int(seed)).read_other_tone(int(valence_sign), lesion=bool(lesion))
    lead = O.empathic_lead(int(r["tone_sign"]), "Sam")
    return {"valence_sign": int(valence_sign), "lesion": bool(lesion), "tone_sign": int(r["tone_sign"]),
            "differential": round(float(r["differential"]), 6), "lead": lead}


def evaluate_seed(seed, tol_neutral=O._NEUTRAL_TOL):
    bad_intact = _read(seed, -1, False)
    good_intact = _read(seed, +1, False)
    bad_lesion = _read(seed, -1, True)
    good_lesion = _read(seed, +1, True)
    # VARY: intact bad lead != intact good lead; signs flip -1 vs +1.
    vary_ok = bool(bad_intact["lead"] != good_intact["lead"] and bad_intact["lead"] and good_intact["lead"])
    sign_ok = bool(bad_intact["tone_sign"] == -1 and good_intact["tone_sign"] == 1)
    # LESION: both collapse to neutral (tone 0, empty lead) with |diff| under tol.
    lesion_collapse = bool(bad_lesion["tone_sign"] == 0 and good_lesion["tone_sign"] == 0
                           and bad_lesion["lead"] == "" and good_lesion["lead"] == ""
                           and abs(bad_lesion["differential"]) < tol_neutral
                           and abs(good_lesion["differential"]) < tol_neutral)
    go = bool(vary_ok and sign_ok and lesion_collapse)
    return {"seed": int(seed), "go": go, "vary_ok": vary_ok, "sign_ok": sign_ok,
            "lesion_collapse": lesion_collapse, "bad_intact": bad_intact, "good_intact": good_intact,
            "bad_lesion": bad_lesion, "good_lesion": good_lesion}


def _flag_off_inertness():
    """Ordinary message -> no other-agent -> acted=False, no lead, bridge NOT built."""
    O._ORGAN = None
    class C: pass
    c = C()
    infos = [O.observe_turn(c, m) for m in ("what does the cat eat", "I am sad", "you are great")]
    return {"all_inert": bool(all((not i["acted"]) and i["lead"] == "" for i in infos)),
            "organ_built": bool(O._ORGAN is not None), "reasons": [i["reason"] for i in infos]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    per_seed = [evaluate_seed(s) for s in a.seeds]
    n_go = sum(1 for r in per_seed if r["go"])
    inert = _flag_off_inertness()
    all_vary = all(r["vary_ok"] for r in per_seed)
    all_sign = all(r["sign_ok"] for r in per_seed)
    all_lesion = all(r["lesion_collapse"] for r in per_seed)
    GO = bool(n_go == len(a.seeds) and inert["all_inert"] and not inert["organ_built"])

    # EARN the verdict (tools.verdict): the load-bearing preconditions must each be measured and hold.
    from tools.verdict import Verdict  # noqa: E402
    _v = Verdict("W5 affective-ToM production organ (empathy lead)", chance=0.0)
    _v.require("6 seeds (project bar)", len(a.seeds) >= 6, expect=True)
    _v.require("VARY on EVERY seed (bad-other lead != good-other lead)", all_vary, expect=True,
               note="the empathic lead flips with the OTHER agent's situation valence")
    _v.require("sign correct on EVERY seed (bad -> tone -1, good -> tone +1)", all_sign, expect=True)
    _v.require("LESION collapses on EVERY seed (affect_out=0 -> |diff|<tol -> tone 0 -> lead '')", all_lesion,
               expect=True, note="the empathic tone rides the OTHER-region spiking read, not a host if valence<0")
    _v.require("FLAG-OFF inert (ordinary turn: acted=False, no lead, bridge NOT built)",
               bool(inert["all_inert"] and not inert["organ_built"]), expect=True)
    _vb = _v.decide(go=GO)

    out = {"runner": "_affective_tom_production_organ_verify",
           "faculty": "W5 affective theory of mind — production organ (empathy lead), load-bearing verify",
           "seeds": a.seeds, "n_seeds_go": n_go, "GO": GO, "verdict": _vb["status"], "neutral_tol": O._NEUTRAL_TOL,
           "flag_off_inertness": inert, "per_seed": per_seed,
           **{k: _vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
           "elapsed_seconds": round(time.time() - t0, 2)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    for r in per_seed:
        print(f"[organ-verify] seed {r['seed']}: GO={r['go']} vary={r['vary_ok']} sign={r['sign_ok']} "
              f"lesion_collapse={r['lesion_collapse']} | bad_intact diff={r['bad_intact']['differential']:+.4f} "
              f"lead={r['bad_intact']['lead']!r} | bad_lesion diff={r['bad_lesion']['differential']:+.4f} "
              f"lead={r['bad_lesion']['lead']!r}", flush=True)
    print(f"[organ-verify] flag-off inertness: all_inert={inert['all_inert']} organ_built={inert['organ_built']}",
          flush=True)
    print(f"[organ-verify] VERDICT: {'GO' if GO else 'NO-GO'} ({n_go}/{len(a.seeds)} seeds) -> {a.out} "
          f"({out['elapsed_seconds']}s)", flush=True)
    return 0 if GO else 1


if __name__ == "__main__":
    sys.exit(main())
