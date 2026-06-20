"""Multi-referent disambiguation — CONTENT-GRADED bias polish: close the seed-100 extreme-asymmetry boundary.

THE GAP THIS CLOSES. The validated biased-competition de-risk (2026-06-19-multireferent-biased-competition-derisk.md,
GO 5/6) had ONE pre-registered miss: **seed 100**. There the content-favored referent's intrinsic accumulator
feed-forward is ESSENTIALLY ZERO relative to a strong rival (`roll` favors ball, but unbiased sel ball=0.000 vs
cat=0.292 — the most extreme intrinsic asymmetry across the 6 seeds), so a FIXED-magnitude content bias (~1x,
2500 pA) cannot lift ball past cat → the WTA either keeps the wrong (cat) winner or neither reaches the 1.3x margin
→ ABSTAIN (None). The abstention is moat-preserving (NOT a confabulation), but the named fix is a **content-GRADED
bias**: scale the bias by how badly the content-favored referent is intrinsically DOMINATED by its rival, so an
extreme-asymmetry referent gets a proportionally stronger steer — WITHOUT over-steering the easy cases (where the
favored is already competitive, so the magnitude stays at base and the no-confab moat is unperturbed).

THE GRADED RULE (principled, tied to CONTENT — NOT a relabelled global gain). On a pronoun query:
  1. the host content helper (`content_bias_target`, unchanged) selects WHICH held referent the pronoun+verb favors
     (animacy/number agreement + selectional restriction). If content is silent/ambiguous -> abstain (moat).
  2. a CHEAP UNBIASED PROBE read of the per-referent accumulator (sel) competition measures the favored referent's
     intrinsic sel `fav_sel` and its strongest rival's sel `rival_sel`. The probe is non-destructive (the held
     attractors persist across reads — verified: cat 0.2925 -> 0.31 on re-read).
  3. the CONTENT-GRADED bias magnitude is set by the favored referent's competitive DEFICIT:
        deficit   = max(0, rival_sel - fav_sel)            # how much the content-favored referent is dominated
        bias_pA   = min(cap, base * (1 + gain * deficit/ref))
     and the (graded) bias current is injected into ONLY the content-favored sel pool, exactly as the fixed bias was.
  4. resolve the WTA winner, gated by the no-confab moat (the winner must be a referent actually held in WM; ties /
     content-silent / empty WM -> abstain).

WHY THIS IS NOT A GLOBAL BOOST (the decisive distinction). The magnitude is (a) applied ONLY to the content-favored
referent (a global gain would lift every sel pool, including the rival), and (b) scaled by THAT referent's content-
vs-rival competitive deficit (deficit=0 for an already-competitive favored -> bias stays at base -> easy cases
UNCHANGED, no over-steer). And critically, the **bias-LESION removes the bias entirely** (bias_pA=0, no probe), so
graded(lesioned)=0 -> the WTA reverts to the intrinsic winner -> the lesion STILL BREAKS resolution. A graded bias
that survives the lesion = a global gain = NOT a GO (the runner checks this).

GO BAR (the point is to close seed-100 WITHOUT breaking anything that already worked):
  1. GO-arm now 6/6 (or at least seed-100 RESOLVES correctly where it previously abstained), both write-orders +
     feature-flip-flips-winner;
  2. the bias-LESION still BREAKS it 6/6 (graded bias still load-bearing -> THE decisive control);
  3. recency-baseline + salience-baseline still FAIL 6/6 on the identical setup;
  4. the no-confab MOAT still holds (0 breaches): empty WM / content-silent / genuinely-tied -> abstain. A graded
     bias that makes a TIED case resolve = a moat regression = NOT a GO.
  5. 3-referent scale check still holds.

Reuse-by-import (the production buffer + helpers verbatim — single source of truth; NO sim/ edit). The graded layer
is computed HERE (the probe-read + the deficit scaling); it is the named follow-on to the production
`BiasedCompetitionContextBuffer`. If GO -> recommend updating `MultiTurnAgent`'s biased-competition to the graded bias.

Run: SIM_BACKEND=numpy python -m research.runners._phaseB_biased_competition_graded_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.content_selection_spiking import SpikingLoopContextBuffer
from research.runners.biased_competition_buffer import (
    BiasedCompetitionContextBuffer,
    content_bias_target,
    resolve_referent,
)
# Re-use the validated baseline battery (recency + salience-4x on the identical {cat, ball} setup) VERBATIM, so the
# control arms are byte-faithful to the GO de-risk and any difference is attributable solely to the graded bias.
from research.runners._phaseB_biased_competition_derisk import run_baselines_on_pair, PAIR, DISTRACTORS


# ---------------------------------------------------------------------------
# The CONTENT-GRADED bias.
# ---------------------------------------------------------------------------
def graded_bias_pA(fav_sel, rival_sel, base_pA, gain, ref, cap_pA):
    """The content-graded bias magnitude: scale `base_pA` UP by how much the content-favored referent is
    intrinsically DOMINATED by its strongest rival (its competitive deficit). deficit=0 (already competitive)
    -> base (no over-steer); a large deficit (extreme intrinsic asymmetry, e.g. seed 100) -> a proportionally
    stronger steer, capped at `cap_pA`. Tied to CONTENT (the favored referent's deficit), not a global gain."""
    deficit = max(0.0, float(rival_sel) - float(fav_sel))
    return float(min(cap_pA, base_pA * (1.0 + gain * deficit / ref)))


def resolve_pronoun_graded(w, verb, candidates, base_pA, spec_threshold, window,
                           gain, ref, cap_pA, lesion=False):
    """Full pronoun resolution with a CONTENT-GRADED bias.

    (1) the content helper selects the favored referent; content-silent -> abstain (moat).
    (2) a cheap UNBIASED PROBE read measures the favored referent's intrinsic sel vs its strongest rival.
    (3) the bias magnitude is graded by the favored referent's competitive deficit (graded_bias_pA), injected
        into ONLY the favored sel pool.
    (4) resolve the WTA, gated by the no-confab moat.

    lesion=True keeps the competition but DROPS the bias entirely (no probe, bias_pA=0) -> graded(lesioned)=0 ->
    the WTA reverts to the intrinsic winner -> the content control is broken (the decisive load-bearing check)."""
    fav = content_bias_target(candidates, verb)
    if fav is None:
        return None, None, {}, None   # content silent -> abstain (moat)
    if lesion:
        rates = w.read(window=window, bias_concept=None, bias_pA=0.0)
        return resolve_referent(rates, spec_threshold), fav, rates, 0.0
    # (2) cheap unbiased probe of the intrinsic accumulator competition (non-destructive — held attractors persist).
    probe = w.read(window=window, bias_concept=None, bias_pA=0.0)
    fav_sel = probe["sel"].get(fav, 0.0)
    rival_sel = max((v for c, v in probe["sel"].items() if c != fav), default=0.0)
    # (3) content-graded magnitude, into the favored sel only.
    pA = graded_bias_pA(fav_sel, rival_sel, base_pA, gain, ref, cap_pA)
    rates = w.read(window=window, bias_concept=fav, bias_pA=pA)
    return resolve_referent(rates, spec_threshold), fav, rates, pA


def _disp(read):
    if isinstance(read, dict) and "sel" in read:
        return {c: [round(read["sel"][c], 4), round(read["held"][c], 4)] for c in read["sel"]}
    return {c: round(v, 4) for c, v in read.items()}


def run_seed(seed, base_pA, spec_threshold, window, gain, ref, cap_pA, verbose=False):
    cat, ball = PAIR

    def buf(concepts):
        return BiasedCompetitionContextBuffer(concepts, n=600, pattern_size=40, seed=seed,
                                              enable_ou=False, competition=True, verbose=verbose)

    out = {"seed": seed}

    def trial(order, verb, lesion=False):
        cands = [cat, ball]
        w = buf(cands)
        w.update([order[0]]); w.update([order[1]])
        resolved, fav, rates, pA = resolve_pronoun_graded(
            w, verb, cands, base_pA, spec_threshold, window, gain, ref, cap_pA, lesion=lesion)
        return {"order": list(order), "verb": verb, "favored": fav, "rates": _disp(rates),
                "bias_pA": pA, "resolved": resolved, "correct": bool(resolved == fav and fav is not None)}

    # --- GO arm: graded biased competition, both write-orders + feature-flip ---
    out["bc_cat_first_eat"] = trial((cat, ball), "eat")
    out["bc_ball_first_eat"] = trial((ball, cat), "eat")
    out["bc_cat_first_roll"] = trial((cat, ball), "roll")
    out["bc_ball_first_roll"] = trial((ball, cat), "roll")
    out["go_arm"] = bool(out["bc_cat_first_eat"]["correct"] and out["bc_ball_first_eat"]["correct"]
                         and out["bc_cat_first_roll"]["correct"] and out["bc_ball_first_roll"]["correct"])
    # seed-100 close: did the two previously-failing roll cases now resolve to ball?
    out["seed100_roll_closes"] = bool(out["bc_cat_first_roll"]["correct"]
                                      and out["bc_ball_first_roll"]["correct"])

    # --- LESION: competition present, bias REMOVED (graded(0)=0). Must NOT match both opposite favoreds. ---
    les_a = trial((cat, ball), "eat", lesion=True)
    les_b = trial((cat, ball), "roll", lesion=True)
    out["lesion_eat"] = les_a
    out["lesion_roll"] = les_b
    out["lesion_breaks"] = bool(not (les_a["correct"] and les_b["correct"]))

    # --- MOAT: empty WM -> abstain; content-silent query -> abstain. ---
    w_empty = buf([cat, ball])  # nothing written
    er, ef, erates, _ = resolve_pronoun_graded(w_empty, "eat", [cat, ball], base_pA, spec_threshold,
                                               window, gain, ref, cap_pA)
    out["moat_empty"] = {"rates": _disp(erates), "resolved": er}
    w_sil = buf([cat, ball]); w_sil.update([cat]); w_sil.update([ball])
    sr, sf, srates, _ = resolve_pronoun_graded(w_sil, "see", [cat, ball], base_pA, spec_threshold,
                                               window, gain, ref, cap_pA)  # 'see' not in VERB_SELECTS
    out["moat_silent"] = {"rates": _disp(srates), "resolved": sr, "favored": sf}
    out["moat_intact"] = bool(er is None and sr is None)

    # --- 3-referent scale check: {cat(animate), ball, river(inanimate)}, eat -> cat ---
    three = [cat, ball, "river"]
    w3 = buf(three)
    w3.update([ball]); w3.update(["river"]); w3.update([cat])
    res3, fav3, rates3, _ = resolve_pronoun_graded(w3, "eat", three, base_pA, spec_threshold,
                                                   window, gain, ref, cap_pA)
    out["three_ref"] = {"favored": fav3, "rates": _disp(rates3),
                        "resolved": res3, "correct": bool(res3 == fav3 and fav3 is not None)}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--base-pA", type=float, default=2500.0,
                    help="base content-bias current (= 1x the per-assembly drive; the floor magnitude for an "
                         "already-competitive favored referent).")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="content-grading gain: bias_pA = base*(1 + gain*deficit/ref), deficit = the favored "
                         "referent's intrinsic sel deficit vs its rival.")
    ap.add_argument("--ref", type=float, default=0.20,
                    help="reference sel scale that normalizes the deficit (a full-deficit ~0.2 -> ~base*(1+gain)).")
    ap.add_argument("--cap-pA", type=float, default=8000.0,
                    help="cap on the graded bias (stay within a safe envelope; never an unbounded turn-up).")
    ap.add_argument("--spec-threshold", type=float, default=1.3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="research/findings/raw/_phaseB_biased_competition_graded.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("[content-graded bias polish] does a CONTENT-GRADED bias (scaled by the favored referent's intrinsic "
          "competitive deficit)\n  close the seed-100 extreme-asymmetry miss WITHOUT over-steering the easy cases "
          "or weakening the moat?\n"
          f"  base_pA={a.base_pA} gain={a.gain} ref={a.ref} cap_pA={a.cap_pA} spec_threshold={a.spec_threshold}\n",
          flush=True)

    results = []
    for seed in a.seeds:
        r = run_seed(seed, a.base_pA, a.spec_threshold, a.window, a.gain, a.ref, a.cap_pA, verbose=a.verbose)
        r["baselines"] = run_baselines_on_pair(seed, a.window, a.spec_threshold)
        results.append(r)
        ea = r["bc_cat_first_eat"]; eb = r["bc_ball_first_eat"]
        ra = r["bc_cat_first_roll"]; rb = r["bc_ball_first_roll"]
        bl = r["baselines"]
        print(f"  [seed {seed}] GO-arm: eat(cat-1st)->{ea['resolved']}({'OK' if ea['correct'] else 'X'},{ea['bias_pA']:.0f}pA) "
              f"eat(ball-1st)->{eb['resolved']}({'OK' if eb['correct'] else 'X'},{eb['bias_pA']:.0f}pA) "
              f"roll(cat-1st)->{ra['resolved']}({'OK' if ra['correct'] else 'X'},{ra['bias_pA']:.0f}pA) "
              f"roll(ball-1st)->{rb['resolved']}({'OK' if rb['correct'] else 'X'},{rb['bias_pA']:.0f}pA) || "
              f"go_arm={r['go_arm']}", flush=True)
        print(f"            lesion: eat->{r['lesion_eat']['resolved']} roll->{r['lesion_roll']['resolved']} "
              f"(breaks={r['lesion_breaks']}) | moat empty->{r['moat_empty']['resolved']} "
              f"silent->{r['moat_silent']['resolved']} (intact={r['moat_intact']}) | "
              f"3ref->{r['three_ref']['resolved']}({'OK' if r['three_ref']['correct'] else 'X'})", flush=True)
        print(f"            baselines on {PAIR}: recency_resolves={bl['recency']['resolves']} "
              f"salience4x_resolves={bl['salience_4x']['resolves']}", flush=True)

    n = len(results)
    go_seeds = sum(r["go_arm"] for r in results)
    s100_closes = sum(r["seed100_roll_closes"] for r in results)
    lesion_seeds = sum(r["lesion_breaks"] for r in results)
    moat_seeds = sum(r["moat_intact"] for r in results)
    three_seeds = sum(r["three_ref"]["correct"] for r in results)
    recency_fail = sum(not r["baselines"]["recency"]["resolves"] for r in results)
    salience_fail = sum(not r["baselines"]["salience_4x"]["resolves"] for r in results)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "summary": {
            "n": n, "go_arm_seeds": go_seeds, "seed100_roll_closes_seeds": s100_closes,
            "lesion_breaks_seeds": lesion_seeds, "moat_intact_seeds": moat_seeds,
            "three_ref_seeds": three_seeds, "recency_fail_seeds": recency_fail,
            "salience_fail_seeds": salience_fail, "base_pA": a.base_pA, "gain": a.gain,
            "ref": a.ref, "cap_pA": a.cap_pA, "spec_threshold": a.spec_threshold}}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  GO-arm (favored wins both orders + feature-flip): {go_seeds}/{n}", flush=True)
    print(f"  every roll case (incl. seed-100 close) resolves:  {s100_closes}/{n}", flush=True)
    print(f"  bias-LESION breaks resolution (load-bearing):     {lesion_seeds}/{n}", flush=True)
    print(f"  no-confab MOAT intact (empty/tie abstain):        {moat_seeds}/{n}", flush=True)
    print(f"  recency baseline FAILS (identical setup):         {recency_fail}/{n}", flush=True)
    print(f"  salience-4x baseline FAILS (identical setup):     {salience_fail}/{n}", flush=True)
    print(f"  3-referent scale (in-probe):                      {three_seeds}/{n}", flush=True)
    bar = 6 if n >= 6 else n
    GO = (go_seeds >= bar and lesion_seeds == n and moat_seeds == n
          and recency_fail == n and salience_fail == n and three_seeds == n)
    if GO:
        print(f"\n  ==> GO: the CONTENT-GRADED bias closes the seed-100 extreme-asymmetry miss -> GO-arm {go_seeds}/{n}. "
              "The bias\n  stays load-bearing (lesion breaks 6/6 -> NOT a relabelled global gain), the moat holds "
              f"({moat_seeds}/{n}, 0 breaches),\n  the recency + salience baselines still fail. ==> recommend updating "
              "the production MultiTurnAgent biased-\n  competition to the content-graded bias (a small follow-on).",
              flush=True)
    else:
        print(f"\n  ==> NEGATIVE: the content-graded bias did NOT cleanly close seed-100 within the GO bar "
              "(GO-arm {go}/{n},\n  lesion {les}/{n}, moat {moat}/{n}). If the lesion stopped breaking -> it became a "
              "global gain; if the moat\n  regressed -> a tied case resolved. seed-100 stays the documented "
              "extreme-asymmetry boundary; do NOT escalate\n  into a config search.".format(
                  go=go_seeds, n=n, les=lesion_seeds, moat=moat_seeds), flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
