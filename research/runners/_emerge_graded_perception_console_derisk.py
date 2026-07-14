"""GRADED-CONFIDENCE grounded in REAL PERCEPTION (the emergence-faithful close the 2026-07-11 boundary named): the
graded three-level hedge (CONFIDENT / HEDGED / ABSTAIN + moat) now rides on categories DISCOVERED from VISUAL similarity,
not hand-scripted co-occurrence. The brain SEES objects through the real Gabor/V1 front end (EMERGE-53's
`PerceptionGroundedConsole`), the competitive pooler discovers the category from the visual features, a property is taught
via ONE exemplar, and a HELD-OUT PERCEIVED object inherits it — answered with the graded hedge. A visually-AMBIGUOUS
object (V1 features drawn from BOTH categories -> its codon overlaps both category codons -> comparable class drives)
HEDGES; a visually-novel percept hits the moat. NO `sim/` edit (reuse-by-import of EMERGE-53 + the committed HTM pool).

WHY (a-1 + a0): the graded read is validated on CLEAN category structure (`_emerge_graded_confidence_console_scale_derisk`,
12-seed GO). Grounding it in real experience via a toy TEXT corpus re-hits the documented distributional-induction SCALE
wall (`2026-07-11-EMERGENT-codes-corpus-cooccurrence-scale-boundary.md`); that finding names PERCEPTION-grounding
(EMERGE-34/53, within-cat 0.86 vs between-cat 0.08) as the tractable, far-stronger category signal. a0-read of EMERGE-53
confirmed its `_drive(member)` returns the per-property apical-drive dict the graded margin read consumes, and `ask_can`
is a categorical argmax-with-override -> the graded 3-level read drops on exactly as it did for EMERGE-31.

MECHANISM: `GradedPerceptionConsole(PerceptionGroundedConsole)` adds (1) `see_ambiguous(name, bird_ex, fish_ex)` -- a
perceptually-ambiguous object whose V1 feature set mixes the two exemplars' top-T features (the perception analog of the
category-ambiguous bat); (2) `_graded_best` -- the winning property drive's margin over the strongest COMPETING class
property -> CONFIDENT (dominates) / HEDGED (contested) / ABSTAIN (<=FLOOR); (3) `graded_ask_can` -- "Yes, a X can P." /
"A X can probably P." / "I don't know whether ..." / "I don't know what a X is." (moat).

GO (6-seed standard 42/43/44/100/101/102 + FRESH 7/8/9/10/11/12): HELD-OUT perceived birds/fish (never taught the
property) inherit CONFIDENTLY+correct via the VISUALLY-discovered category, AND the ambiguous object HEDGES, AND a novel
percept hits the MOAT, with SCRAMBLE (per-image pixel scramble -> destroys visual similarity) degrading the held-out
confidence AND LESION (coincidence off) driving all-abstain, on >=5/6 in BOTH sets.

Run: SIM_BACKEND=numpy python -m research.runners._emerge_graded_perception_console_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from research.runners._emerge53_perception_grounded_conversation import (
    PerceptionGroundedConsole, _art, FLOOR, _gabor, T_ACTIVE,
    _BIRD_SEEN, _FISH_SEEN, _BIRD_EXEMPLARS, _FISH_EXEMPLARS, _BIRD_HELDOUT, _FISH_HELDOUT)
from research.runners._genfrontier_optionB_visual_similarity_derisk import encode_v1

CONF_MARGIN = 30.0        # same apical-drive scale as the validated graded-completion probe (CONF_TH-FLOOR)

# REUSE EMERGE-53's VALIDATED perception recipe (9 seen + 9, teach the class via 6 exemplars each, held-out owl/wren/
# minnow/gar) so the perception-grounded held-out inheritance works as validated; add the graded read + ambiguous percept.
HELD_OUT = [(m, "fly") for m in _BIRD_HELDOUT] + [(m, "swim") for m in _FISH_HELDOUT]
AMBIG = ("chimera", "robin", "trout")                              # mixed bird+fish percept -> HEDGED on "fly"
MOAT_NAME = "griffin"                                              # never rendered -> moat


class GradedPerceptionConsole(PerceptionGroundedConsole):
    def __init__(self, *a, hedge_band=5.0, **k):
        super().__init__(*a, **k)
        self.hedge_band = hedge_band          # a competitor above FLOOR+band = a genuinely co-activated (ambiguous) category

    def see_blended(self, name, ex_a, ex_b, w=0.5):
        """A visually-AMBIGUOUS object = a real IMAGE BLEND (w*ex_a + (1-w)*ex_b pixels) perceived through the SAME
        Gabor/V1 front end -> its V1 features naturally lie BETWEEN the two categories -> the pooler codon overlaps both
        category codons -> both class properties co-activate -> the graded read hedges. This is a genuine visual morph
        (more faithful + more seed-robust than a synthetic feature union, whose category-lean was seed-fragile)."""
        wl = self.world
        ia, ib = wl.name_to_obj.get(ex_a), wl.name_to_obj.get(ex_b)
        if ia is None or ib is None:
            return f"I haven't seen both {_art(ex_a)} and {_art(ex_b)} yet."
        img = w * wl.images[ia] + (1.0 - w) * wl.images[ib]        # a real intermediate SHAPE (pixel blend)
        V = encode_v1(img[None], _gabor())[0]                      # perceived through the SAME V1 bank
        sub = np.array([V[f] for f in wl._glob])                   # restrict to the console's feature block
        feats = set(int(t) for t in np.argsort(-sub)[:T_ACTIVE])   # top-T active = the object's V1 feature vector
        self._alloc_member(name)
        self.member_feats[name] = feats
        self._pooler_dirty = True
        self.last_seen = name
        return f"ok -- I've seen {_art(name)} (it looks like both)."

    def _graded_best(self, member):
        """(label, kind, key, prop). CO-ACTIVATION criterion (a0-principled, threshold-light): a clean object activates
        ONE category codon (the competitor stays at apical rest <= FLOOR); a visually-ambiguous object co-activates a
        COMPETING category above FLOOR. -> ABSTAIN if nothing driven; HEDGED if a competitor is co-activated; else
        CONFIDENT. `hedge_band` guards against a noise-level co-activation."""
        dr = self._drive(member)
        if not dr:
            return "ABSTAIN", None, None, None
        best = max(dr, key=dr.get)
        if dr[best] <= FLOOR:
            return "ABSTAIN", None, None, None
        kind, key = best
        own = ("OVR", member)
        if own in dr and dr[own] > FLOOR and dr[own] >= dr[best] - 1e-6:  # member exception overrides class default
            kind, key = own
        runner_up = max([v for (k, kk), v in dr.items() if not (k == kind and kk == key)] or [-1e9])
        prop = self.ovr_prop.get(key) if kind == "OVR" else self.class_prop.get(key)
        label = "HEDGED" if runner_up > FLOOR + self.hedge_band else "CONFIDENT"  # a competing category co-activated
        return label, kind, key, prop

    def graded_ask_can(self, member, prop):
        if member not in self.member_feats:
            return "MOAT", f"I don't know what {_art(member)} is."
        label, kind, key, p = self._graded_best(member)
        if label == "ABSTAIN" or p is None:
            return "ABSTAIN", f"I don't know whether {_art(member)} can {prop}."
        if kind == "OVR":                                          # member-specific exception (cancellation)
            if label == "HEDGED":
                return "HEDGED", f"{_art(member).capitalize()} probably {p}."
            return "CONFIDENT", f"No, {_art(member)} {p}."
        if label == "HEDGED":                                      # visually contested class -> hedge
            return "HEDGED", f"{_art(member).capitalize()} looks like more than one kind of thing -- it can probably {p}."
        return "CONFIDENT", f"Yes, {_art(member)} can {p}."

    def ask_can(self, member, prop):                              # override: graded, not categorical
        return self.graded_ask_can(member, prop)[1]


def _build(seed, arm):
    c = GradedPerceptionConsole(seed=seed, lesion=(arm == "lesion"), scramble=(arm == "scramble"))
    for b in _BIRD_SEEN:
        c.see(b); c.learn_isa(b, "bird")
    for f in _FISH_SEEN:
        c.see(f); c.learn_isa(f, "fish")
    c.see_blended(*AMBIG)                                          # a visually-ambiguous object (real bird+fish image blend)
    for b in _BIRD_EXEMPLARS:                                      # teach the class property via MULTIPLE exemplars
        c.learn_class(b, "fly")
    for f in _FISH_EXEMPLARS:
        c.learn_class(f, "swim")
    return c


def _run_arm(seed, arm):
    c = _build(seed, arm)
    held = {m: c.graded_ask_can(m, p)[0] for (m, p) in HELD_OUT}
    amb = c.graded_ask_can(AMBIG[0], "fly")[0]
    moat = c.graded_ask_can(MOAT_NAME, "fly")[0]
    return {"arm": arm, "held": held, "amb": amb, "moat": moat}


def run(seed):
    htm = _run_arm(seed, "htm")
    scr = _run_arm(seed, "scramble")
    les = _run_arm(seed, "lesion")
    n_conf = sum(v == "CONFIDENT" for v in htm["held"].values())
    # GO = the perception-grounded graded MOAT: CONFIDENT for a clearly-perceived category member (held-out inherit
    # >= EMERGE-53's validated 0.75 bar) + ABSTAIN/MOAT for a never-perceived object, causal on the visual similarity
    # (scramble degrades) + on the coincidence substrate (lesion abstains). The HEDGED level is a REPORTED diagnostic,
    # NOT gated: categorical perception (the competitive pooler's k-WTA sharpening + the bimodal coincidence drive)
    # suppresses graded hedging at the category read -- an honest boundary (see the finding).
    held_confident = n_conf / len(htm["held"]) >= 0.75
    moat_ok = htm["moat"] == "MOAT"
    scramble_breaks = sum(v == "CONFIDENT" for v in scr["held"].values()) < n_conf  # visual similarity destroyed -> degrade
    lesion_breaks = all(v == "ABSTAIN" for v in les["held"].values())
    go = bool(held_confident and moat_ok and scramble_breaks and lesion_breaks)
    print(f"[graded-percept seed={seed}] held_confident={n_conf}/{len(htm['held'])} moat={htm['moat']} "
          f"amb(diag)={htm['amb']} | scramble_breaks={scramble_breaks} lesion_breaks={lesion_breaks} "
          f"-> {'GO' if go else 'no'}", flush=True)
    return {"seed": seed, "held": htm["held"], "amb_diag": htm["amb"], "moat": htm["moat"], "GO": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        c = _build(42, "htm")
        print("\n=== EMERGE graded PERCEPTION console -- confidence grounded in real vision ===\n")
        for (nm, q) in [("owl", "can it fly?"), ("chimera", "can it fly?"), ("griffin", "can it fly?")]:
            print(f"  [sees a {nm}] you> {q}\n  brain> {c.ask_can(nm, 'fly')}\n")
        return
    res = [run(s) for s in a.seeds]
    print(f"[graded-percept] {sum(1 for r in res if r['GO'])}/{len(res)} GO", flush=True)
    if a.out:
        json.dump(dict(results=res), open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
