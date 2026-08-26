"""SOAK / no-regression gate for the SELECTIVE-ATTENTION (biased-competition) DEFAULT-ON flip.

The PARENT runs this on the pool BEFORE flipping BRAIN_BIASED_COMPETITION default-on. It runs a conversation twice
per seed — flag OFF vs flag ON — through the PRODUCTION wiring (the same `enable_biased_competition=_bc_enabled()`
the live build sites use, driven by the env flag), and asserts:

  NO-REGRESSION (the gate): every ORDINARY turn is BYTE-IDENTICAL OFF vs ON. An ordinary turn holds < 2 discourse
  referents, so it never enters the biased-competition path: a UNIQUE-referent pronoun ("dog chase cat" -> "what does
  it eat?"), a plain non-pronoun fact query ("what does cat eat?"), a moat abstain (unknown subject; content-silent
  verb), and a truth query all resolve through the plain single-attractor anaphora path unchanged. Flag ON only
  builds the biased-competition buffer, it does not alter these replies.

  FACULTY LIVE (the flip is not a no-op): a TRIGGERED turn — a bare pronoun over >=2 held referents of opposing
  animacy ({cat, ball}) with a content-selecting verb — RESOLVES DIFFERENTLY OFF vs ON. OFF the plain path reads the
  seed-dependent intrinsic attractor; ON the WTA biased competition binds the content-favored referent (eat->cat,
  roll->ball). The OFF-vs-ON difference on the 'roll' (inanimate-selecting) direction is the load-bearing signal.

GO(seed) = no_regression AND faculty_live. The 6-seed bar is 6/6 no_regression (the flip is safe) with faculty_live
on the decisive seeds (the de-risk's GO-arm is 5/6; a seed whose intrinsic asymmetry is extreme ABSTAINS OFF and ON
alike on 'roll' — moat-preserving, NOT a regression — so faculty_live is reported per-seed, and the GATE is
no_regression).

De-risk GO: research/findings/2026-06-19-multireferent-biased-competition-derisk.md (spiking substrate, GO-arm 5/6).
Organ: research/runners/biased_competition_buffer.py + multi_turn_agent.py. Flag organ: biased_competition_prod.py.

  Run: SIM_BACKEND=numpy python -m research.runners._biased_competition_flip_soak --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners.multi_turn_agent import MultiTurnAgent          # noqa: E402
from research.runners.biased_competition_prod import (                # noqa: E402
    BRAIN_BIASED_COMPETITION_ENV, biased_competition_enabled)

NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat", "roll", "hill", "mat", "dragon"]
OUT = _REPO / "research" / "findings" / "raw" / "_biased_competition_prodflip" / "soak_seed42.json"


def _mk_agent(seed, bias_pA=2500.0):
    """Construct EXACTLY as a live build site does: enable_biased_competition=biased_competition_enabled() (the
    env-gated organ call). The env flag (set by the caller) therefore drives the config the same way production does.
    `bias_pA` overrides the content-bias magnitude for the LESION arm (0.0 = the de-risk's own bias-lesion: the WTA
    reverts to the seed-dependent intrinsic attractor -> the content-steer vanishes)."""
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed,
                       enable_biased_competition=biased_competition_enabled(), biased_competition_bias_pA=bias_pA)
    # shared fact store (identical OFF vs ON): both cat & ball have eat + roll facts -> a wrong bind returns a
    # DIFFERENT non-None answer (fact-availability is controlled; only WHICH referent binds differs).
    for f in [("cat", "eat", "fish"), ("ball", "eat", "worm"),
              ("ball", "roll", "hill"), ("cat", "roll", "mat"), ("dog", "chase", "cat")]:
        a.agent.composer.store(*f)
    return a


def _ordinary_turns(seed):
    """A fresh agent that never holds >=2 referents -> every turn takes the plain anaphora path (must be OFF==ON)."""
    a = _mk_agent(seed)
    a.hear("dog chase cat")                                  # ONE referent held ('cat')
    return {
        "unique_pronoun": a.what_does("it", "eat"),          # it -> cat (single attractor) -> fish
        "plain_fact": a.what_does("cat", "eat"),             # no pronoun -> fish
        "moat_unknown_subject": a.what_does("dragon", "eat"),  # unknown -> None (moat)
        "truth_query": a.is_it_true("cat", "eat", "fish"),   # -> yes/true
        "moat_pronoun_silent": a.what_does("it", "chase"),   # 'chase' has a fact for cat? -> resolve it->cat
    }


def _triggered_turn(seed, bias_pA=2500.0):
    """A fresh agent holding TWO opposing-animacy referents {cat, ball}. OFF reads the intrinsic attractor; ON runs
    the biased competition. Follows the validated CI read sequence (resolve eat, what_does eat, resolve roll).
    `bias_pA=0.0` (the LESION) zeroes the content bias -> the WTA reverts to the intrinsic attractor."""
    a = _mk_agent(seed, bias_pA=bias_pA)
    a._write_referent("cat")
    a._write_referent("ball")
    held_ok = (a._held_set() == ["ball", "cat"])
    eat = a._resolve("it", query_verb="eat")                 # ON: biased(eat)->cat ; OFF: held_referent (intrinsic)
    wd_eat = a.what_does("it", "eat")                        # advances state as the validated sequence
    roll = a._resolve("it", query_verb="roll")               # ON: biased(roll)->ball ; OFF/LESION: intrinsic
    wd_roll = a.what_does("it", "roll")
    return {"held_ok": held_ok, "resolve_eat": eat, "wd_eat": wd_eat, "resolve_roll": roll, "wd_roll": wd_roll,
            "enable_biased_competition": a.enable_biased_competition, "bcw_built": a.bcw is not None}


def _run_arm(seed, flag_on):
    os.environ[BRAIN_BIASED_COMPETITION_ENV] = "1" if flag_on else "0"
    assert biased_competition_enabled() is flag_on
    return {"ordinary": _ordinary_turns(seed), "triggered": _triggered_turn(seed)}


def run_one(seed, backend):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[bc-soak] seed={seed} backend={backend} — conversation OFF vs ON: ORDINARY turns byte-identical; the "
          f"multi-referent TRIGGERED turn resolves the content-favored referent when ON.", flush=True)
    result = {"seed": seed, "backend": backend}
    try:
        off = _run_arm(seed, flag_on=False)
        on = _run_arm(seed, flag_on=True)
        # LESION arm (the de-risk's OWN bias-lesion): flag ON but content bias zeroed -> the WTA reverts to the
        # seed-dependent intrinsic attractor, so the verb no longer steers the winner. Run under the ON flag.
        os.environ[BRAIN_BIASED_COMPETITION_ENV] = "1"
        lesion = _triggered_turn(seed, bias_pA=0.0)
        os.environ[BRAIN_BIASED_COMPETITION_ENV] = "0"

        # ── NO-REGRESSION GATE: every ordinary turn byte-identical OFF vs ON ──
        no_regression = (off["ordinary"] == on["ordinary"])
        # config sanity: OFF never builds the buffer; ON does (on the triggered scenario, >=2 held)
        off_buffer_absent = (off["triggered"]["enable_biased_competition"] is False
                             and off["triggered"]["bcw_built"] is False)
        on_buffer_built = (on["triggered"]["enable_biased_competition"] is True
                           and on["triggered"]["bcw_built"] is True)
        # ── FACULTY LIVE: the triggered turn resolves DIFFERENTLY OFF vs ON (load-bearing) ──
        on_content_flip = (on["triggered"]["resolve_eat"] == "cat"
                           and on["triggered"]["resolve_roll"] == "ball")
        # the 'roll' direction differs OFF vs ON (OFF intrinsic != ON's ball); the reply differs too
        roll_differs = (off["triggered"]["resolve_roll"] != on["triggered"]["resolve_roll"]
                        and off["triggered"]["wd_roll"] != on["triggered"]["wd_roll"])
        # ── LESION LOAD-BEARING: zeroing the bias DESTROYS the feature-flip (roll no longer binds ball) and the
        #    winner reverts to the intrinsic attractor (roll's answer == eat's answer -> content ignored). The ON
        #    content-flip must be present for this to be meaningful (only-if the faculty was live in the first place).
        lesion_reverts = (on_content_flip and lesion["resolve_roll"] != "ball"
                          and lesion["resolve_roll"] == lesion["resolve_eat"])
        faculty_live = bool(on_content_flip and roll_differs and lesion_reverts)

        GO = bool(no_regression and off_buffer_absent and on_buffer_built and faculty_live)
        result.update(dict(GO=GO, no_regression=no_regression, faculty_live=faculty_live,
                           off_buffer_absent=off_buffer_absent, on_buffer_built=on_buffer_built,
                           on_content_flip=on_content_flip, roll_differs=roll_differs,
                           lesion_reverts=lesion_reverts, off=off, on=on, lesion=lesion))
        print(f"[bc-soak] ordinary OFF: {off['ordinary']}", flush=True)
        print(f"[bc-soak] ordinary ON : {on['ordinary']}", flush=True)
        print(f"[bc-soak] NO_REGRESSION(ordinary OFF==ON) = {no_regression}", flush=True)
        print(f"[bc-soak] triggered OFF: eat={off['triggered']['resolve_eat']!r} roll={off['triggered']['resolve_roll']!r}"
              f" | ON: eat={on['triggered']['resolve_eat']!r} roll={on['triggered']['resolve_roll']!r}"
              f" | LESION: eat={lesion['resolve_eat']!r} roll={lesion['resolve_roll']!r}", flush=True)
        print(f"[bc-soak] faculty_live(ON content-flip + roll differs OFF + lesion reverts) = {faculty_live}", flush=True)
        print(f"[bc-soak] seed={seed} NO_REGRESSION={no_regression} FACULTY_LIVE={faculty_live} "
              f"=> {'GO' if GO else 'NO-GO'}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    finally:
        os.environ[BRAIN_BIASED_COMPETITION_ENV] = "0"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    backend = os.environ.get("SIM_BACKEND", "numpy")
    results = {}; go = []; noreg = []
    for seed in seeds:
        r = run_one(seed, backend)
        results[seed] = r; go.append(bool(r.get("GO"))); noreg.append(bool(r.get("no_regression")))
    out_path = Path(a.out)
    if len(seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(seeds)}seed.json"
        print("\n" + "#" * 118)
        print(f"[bc-soak] {len(seeds)}-SEED SOAK: NO_REGRESSION {int(sum(noreg))}/{len(seeds)} (the flip-safety GATE) | "
              f"GO(no-reg AND faculty-live) {int(sum(go))}/{len(seeds)} seeds={seeds}")
        print("#" * 118)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": seeds, "n_go": int(sum(go)), "n_no_regression": int(sum(noreg)),
                                    "go": go, "no_regression": noreg, "backend": backend,
                                    "results": {str(s): results[s] for s in seeds}}, indent=2, default=str))
    print(f"[bc-soak] wrote {out_path}")
    # the GATE the parent flips on is NO-REGRESSION across all seeds (faculty_live is per-seed diagnostic).
    return 0 if (noreg and all(noreg)) else 1


if __name__ == "__main__":
    sys.exit(main())
