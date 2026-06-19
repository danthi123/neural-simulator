"""Multi-referent disambiguation via WTA BIASED COMPETITION — the mechanism the two converging NEGATIVEs
(2026-06-17-multireferent-disambiguation-NEGATIVE.md: recency 0/3 + salience-boost even-4x-fails) named as the
fix, and scoped in 2026-06-19-multireferent-wta-biased-competition-scoping.md.

THE QUESTION. When the spiking working memory holds several discourse referents, which one does a bare pronoun
("it") bind to? The plain SpikingLoopContextBuffer holds each referent in an INDEPENDENT attractor with NO
cross-referent coupling (loop_weight=0, internal_density=0 by construction), so neither recency (no position
signal in the rate read) nor a salience boost (only ADDS activity to an independent attractor; cannot SUPPRESS
the competitor) can pick the right one. THE FIX (Desimone-Duncan 1995 biased competition; Wong-Wang 2006 attractor
WTA): MUTUAL INHIBITION between the held referent assemblies (each referent's assembly drives a dedicated FS
inhibitory pool that suppresses the OTHER referents' assemblies — the Rutishauser selective-inhibition motif the
navigation sel_X/sel_FS_X read-out already uses) + a small CONTENT-based top-down BIAS (a feed-forward current
into the referent whose features — animacy/number agreement with the pronoun + selectional compatibility with the
query verb — match). The crux: the bias is a CONTENT signal (NOT position=recency, NOT magnitude=boost — the two
already-disproven signals); the recurrence amplifies the small content asymmetry into a SUPPRESSIVE winner.

THE NEW WIRING (additive; NO sim/ edit). BiasedCompetitionContextBuffer wraps the validated loop bridge: it builds
the cortex_ctx<->dlpfc_wm loop PLUS one all-inhibitory FS region per referent (exc_fraction=0.0 -> the FS neurons
get inhibitory traits, so their out-synapses route to g_i), then installs, via set_pathway_weights(add_missing):
  * cortex_assembly[X] -> ref_FS_X[all]        (EXCITATORY: a referent recruits its own interneuron)
  * ref_FS_X[all]      -> cortex_assembly[Y!=X] (INHIBITORY: that interneuron suppresses the OTHER referents)
plus a bias(concept, pA) injector that adds a SMALL feed-forward current into a concept's cortex assembly during
the competitive read window. The held attractors + read() are reused verbatim from SpikingLoopContextBuffer.

THE CONTENT-BIAS HELPER is host-side (a teaching scaffold, FLAGGED for conversion to a learned synaptic
feature-compatibility map per BRAIN-BASED-ONLY): given the pronoun's features + the query verb's selectional
restriction, it returns WHICH held referent receives the bias current. The win is brain-based (spiking competition
+ suppression); the content SCORING is host in this probe (the follow-on neuralizes it).

GO BAR (pre-registered, FROZEN; >=5/6 seeds):
  1. the content-bias-favored referent WINS the WTA (the pronoun resolves to it), in BOTH write-orders, AND the
     feature-flip FLIPS the winner (proves it's content, not position/magnitude).
  2. the recency baseline FAILS on the identical {cat, ball} setup (re-run in-probe).
  3. the salience-boost baseline FAILS on the identical setup (re-run in-probe, even 4x).
  4. bias-LESION (remove the content bias, keep the competition) -> the WTA picks at chance / wrong (THE decisive
     control proving genuine content-steered competition, not a relabelled boost).
  5. the no-confab MOAT intact: empty WM or a TIE -> abstain (None), 0 breaches.
A bias-lesion that does NOT break resolution = the bias wasn't load-bearing = NOT a GO.

Run: SIM_BACKEND=numpy python -m research.runners._phaseB_biased_competition_derisk --seeds 42 43 44 100 101 102
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
# The mechanism (the buffer + the content-bias helper + the feature lexicons + resolve_referent) lives in the
# production module now; this de-risk imports it verbatim (single source of truth — the runner stays unchanged).
from research.runners.biased_competition_buffer import (  # noqa: F401  (ANIMACY/VERB_SELECTS re-exported for parity)
    ANIMACY,
    VERB_SELECTS,
    BiasedCompetitionContextBuffer,
    content_bias_target,
    resolve_referent,
)


# ---------------------------------------------------------------------------
# The de-risk
# ---------------------------------------------------------------------------
# Two referents of OPPOSING content features (so the bias has a content handle and the test is NOT
# recency-solvable): cat (animate) vs ball (inanimate). Pronoun "it" + a query verb whose selectional
# restriction picks one of them.
PAIR = ("cat", "ball")            # (animate, inanimate)
DISTRACTORS = ["fish", "worm", "fox", "dog"]   # extra referents for the >=2 setup / 3-referent scale check


def _favored(candidates, verb):
    return content_bias_target(candidates, verb)


def _disp(read):
    """Compact display of a structured read: {concept: (sel_rate, held_rate)}."""
    if isinstance(read, dict) and "sel" in read:
        return {c: [round(read["sel"][c], 4), round(read["held"][c], 4)] for c in read["sel"]}
    return {c: round(v, 4) for c, v in read.items()}


def resolve_pronoun(w, verb, candidates, bias_pA, spec_threshold, window, lesion=False):
    """The full pronoun-resolution decision. (1) The CONTENT bias selects which held referent matches the
    pronoun+verb; if content is SILENT (no/ambiguous match) -> abstain (None) — the no-confab moat refuses
    to pick by intrinsic strength. (2) Else run the biased competition (re-present + bias the favored sel)
    and resolve the WTA winner (gated on the moat held-floor). lesion=True keeps the competition but DROPS
    the bias (bias_pA=0) -> the WTA reverts to the intrinsic winner -> the content control is broken."""
    fav = content_bias_target(candidates, verb)
    if fav is None:
        return None, None, {}   # content silent -> abstain (moat)
    rates = w.read(window=window, bias_concept=(None if lesion else fav),
                   bias_pA=(0.0 if lesion else bias_pA))
    resolved = resolve_referent(rates, spec_threshold)
    return resolved, fav, rates


def run_seed(seed, bias_pA, spec_threshold, window, verbose=False):
    cat, ball = PAIR

    def buf(concepts, competition=True):
        return BiasedCompetitionContextBuffer(concepts, n=600, pattern_size=40, seed=seed,
                                              enable_ou=False, competition=competition, verbose=verbose)

    out = {"seed": seed}

    # --- GO arm: biased competition, BOTH write-orders, on the {cat, ball} pair ---
    # query "eat" selects animate -> favored = cat ; query "roll" selects inanimate -> favored = ball.
    def trial(order, verb, lesion=False):
        cands = [cat, ball]
        w = buf(cands, competition=True)
        w.update([order[0]]); w.update([order[1]])
        resolved, fav, rates = resolve_pronoun(w, verb, cands, bias_pA, spec_threshold, window, lesion=lesion)
        return {"order": list(order), "verb": verb, "favored": fav, "rates": _disp(rates),
                "resolved": resolved, "correct": bool(resolved == fav and fav is not None)}

    out["bc_cat_first_eat"] = trial((cat, ball), "eat")    # favored cat, cat written first
    out["bc_ball_first_eat"] = trial((ball, cat), "eat")   # favored cat, ball written first -> if recency, ball would win
    out["bc_cat_first_roll"] = trial((cat, ball), "roll")  # FEATURE-FLIP: favored ball, cat written first
    out["bc_ball_first_roll"] = trial((ball, cat), "roll") # FEATURE-FLIP: favored ball, ball written first
    out["go_arm"] = bool(out["bc_cat_first_eat"]["correct"] and out["bc_ball_first_eat"]["correct"]
                         and out["bc_cat_first_roll"]["correct"] and out["bc_ball_first_roll"]["correct"])

    # --- LESION: competition present, bias REMOVED. For the SAME held WM {cat,ball}, the unbiased WTA picks
    # the SAME intrinsic winner regardless of verb -> it cannot match BOTH opposite favoreds (eat->cat,
    # roll->ball) -> >=1 is wrong -> the bias is load-bearing. ---
    les_a = trial((cat, ball), "eat", lesion=True)
    les_b = trial((cat, ball), "roll", lesion=True)
    out["lesion_eat"] = les_a
    out["lesion_roll"] = les_b
    out["lesion_breaks"] = bool(not (les_a["correct"] and les_b["correct"]))

    # --- MOAT: (a) empty WM -> abstain ; (b) content-silent query (verb with no selectional restriction, OR
    # two same-feature candidates) -> abstain (the agent refuses to pick by intrinsic strength). ---
    w_empty = buf([cat, ball], competition=True)  # nothing written (empty registry)
    er, ef, erates = resolve_pronoun(w_empty, "eat", [cat, ball], bias_pA, spec_threshold, window)
    out["moat_empty"] = {"rates": _disp(erates), "resolved": er}
    # content-silent: a verb with no selectional restriction -> favored None -> abstain
    w_sil = buf([cat, ball], competition=True); w_sil.update([cat]); w_sil.update([ball])
    sr, sf, srates = resolve_pronoun(w_sil, "see", [cat, ball], bias_pA, spec_threshold, window)  # 'see' not in VERB_SELECTS
    out["moat_silent"] = {"rates": _disp(srates), "resolved": sr, "favored": sf}
    out["moat_intact"] = bool(er is None and sr is None)

    # --- 3-referent scale check (one compatible + two incompatible): {cat(animate), ball, river(inanimate)} ---
    three = [cat, ball, "river"]
    w3 = buf(three, competition=True)
    w3.update([ball]); w3.update(["river"]); w3.update([cat])   # cat written last but bias is content, not recency
    res3, fav3, rates3 = resolve_pronoun(w3, "eat", three, bias_pA, spec_threshold, window)  # animate -> cat
    out["three_ref"] = {"favored": fav3, "rates": _disp(rates3),
                        "resolved": res3, "correct": bool(res3 == fav3 and fav3 is not None)}

    return out


def run_baselines_on_pair(seed, window, spec_threshold):
    """Re-run the recency + salience baselines on the IDENTICAL {cat, ball} setup (no competition substrate),
    proving the setup is genuinely ambiguous without the new mechanism."""
    cat, ball = PAIR

    def plain(concepts):
        return SpikingLoopContextBuffer(concepts, n=600, pattern_size=40, seed=seed, enable_ou=False)

    # recency: write cat then ball (ball recent). Does the read carry a usable order gradient either way?
    w = plain([cat, ball]); w.update([cat]); w.update([ball])
    r_nat = w.read(window=window)
    nat_res = resolve_referent(r_nat, spec_threshold)
    w2 = plain([cat, ball]); w2.update([ball]); w2.update([cat])
    r_ord = w2.read(window=window)
    ord_res = resolve_referent(r_ord, spec_threshold)
    # recency PASSES only if the read flips with order (recent always wins). It FAILS (the documented NEGATIVE)
    # if it does not produce a recency-aligned, order-flipping winner.
    recency_resolves = bool(nat_res == ball and ord_res == cat)

    # salience: write cat normal, ball boosted 4x. Does the boost win? order-control: ball normal, cat boosted.
    def boosted(w, c, f=4.0):
        w.update([c], drive_pA=2500.0 * f, stim=int(40 * f), settle=15)
    ws = plain([cat, ball]); ws.update([cat]); boosted(ws, ball)
    s_nat = ws.read(window=window)
    s_nat_res = resolve_referent(s_nat, spec_threshold)
    ws2 = plain([cat, ball]); ws2.update([ball]); boosted(ws2, cat)
    s_ord = ws2.read(window=window)
    s_ord_res = resolve_referent(s_ord, spec_threshold)
    salience_resolves = bool(s_nat_res == ball and s_ord_res == cat)

    return {
        "recency": {"natural_resolved": nat_res, "order_resolved": ord_res,
                    "nat_rates": {k: round(v, 4) for k, v in r_nat.items()},
                    "ord_rates": {k: round(v, 4) for k, v in r_ord.items()},
                    "resolves": recency_resolves},
        "salience_4x": {"natural_resolved": s_nat_res, "order_resolved": s_ord_res,
                        "nat_rates": {k: round(v, 4) for k, v in s_nat.items()},
                        "ord_rates": {k: round(v, 4) for k, v in s_ord.items()},
                        "resolves": salience_resolves},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--bias-pA", type=float, default=2500.0,
                    help="content-bias feed-forward current (~1x the per-assembly drive scale 2500 — SMALL on "
                         "purpose, the magnitude a uniform boost already FAILED at, so any win is from the "
                         "competition amplifying a small content asymmetry).")
    ap.add_argument("--spec-threshold", type=float, default=1.3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="research/findings/raw/_phaseB_biased_competition.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("[biased-competition de-risk] does WTA biased competition (mutual inhibition + small CONTENT bias) "
          "bind a bare pronoun\n  to the correct one of >=2 held referents, where recency + salience cannot?\n"
          f"  bias_pA={a.bias_pA} (~1x drive; small on purpose), spec_threshold={a.spec_threshold}\n", flush=True)

    results = []
    for seed in a.seeds:
        r = run_seed(seed, a.bias_pA, a.spec_threshold, a.window, verbose=a.verbose)
        r["baselines"] = run_baselines_on_pair(seed, a.window, a.spec_threshold)
        results.append(r)
        ea = r["bc_cat_first_eat"]; eb = r["bc_ball_first_eat"]
        ra = r["bc_cat_first_roll"]; rb = r["bc_ball_first_roll"]
        bl = r["baselines"]
        print(f"  [seed {seed}] GO-arm: eat(cat-1st) ->{ea['resolved']}({'OK' if ea['correct'] else 'X'}) "
              f"eat(ball-1st)->{eb['resolved']}({'OK' if eb['correct'] else 'X'}) "
              f"roll(cat-1st)->{ra['resolved']}({'OK' if ra['correct'] else 'X'}) "
              f"roll(ball-1st)->{rb['resolved']}({'OK' if rb['correct'] else 'X'}) || go_arm={r['go_arm']}",
              flush=True)
        print(f"            lesion: eat->{r['lesion_eat']['resolved']} roll->{r['lesion_roll']['resolved']} "
              f"(breaks={r['lesion_breaks']}) | moat empty->{r['moat_empty']['resolved']} "
              f"silent->{r['moat_silent']['resolved']} (intact={r['moat_intact']}) | "
              f"3ref->{r['three_ref']['resolved']}({'OK' if r['three_ref']['correct'] else 'X'})", flush=True)
        print(f"            baselines on {PAIR}: recency_resolves={bl['recency']['resolves']} "
              f"salience4x_resolves={bl['salience_4x']['resolves']}", flush=True)

    n = len(results)
    go_seeds = sum(r["go_arm"] for r in results)
    lesion_seeds = sum(r["lesion_breaks"] for r in results)
    moat_seeds = sum(r["moat_intact"] for r in results)
    three_seeds = sum(r["three_ref"]["correct"] for r in results)
    recency_fail = sum(not r["baselines"]["recency"]["resolves"] for r in results)
    salience_fail = sum(not r["baselines"]["salience_4x"]["resolves"] for r in results)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "summary": {
            "n": n, "go_arm_seeds": go_seeds, "lesion_breaks_seeds": lesion_seeds,
            "moat_intact_seeds": moat_seeds, "three_ref_seeds": three_seeds,
            "recency_fail_seeds": recency_fail, "salience_fail_seeds": salience_fail,
            "bias_pA": a.bias_pA, "spec_threshold": a.spec_threshold}}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  GO-arm (favored wins both orders + feature-flip): {go_seeds}/{n}", flush=True)
    print(f"  bias-LESION breaks resolution (load-bearing):     {lesion_seeds}/{n}", flush=True)
    print(f"  no-confab MOAT intact (empty/tie abstain):        {moat_seeds}/{n}", flush=True)
    print(f"  recency baseline FAILS (identical setup):         {recency_fail}/{n}", flush=True)
    print(f"  salience-4x baseline FAILS (identical setup):     {salience_fail}/{n}", flush=True)
    print(f"  3-referent scale (in-probe):                      {three_seeds}/{n}", flush=True)
    bar = 5 if n >= 6 else n
    GO = (go_seeds >= bar and lesion_seeds >= bar and moat_seeds == n
          and recency_fail == n and salience_fail == n)
    if GO:
        print(f"\n  ==> GO: WTA biased competition resolves multi-referent pronouns where recency + salience "
              "CANNOT. The favored\n  referent wins (both write-orders + feature-flip), the bias is load-bearing "
              "(lesion breaks it), the moat holds.\n  ==> recommend wiring into MultiTurnAgent behind a "
              "default-OFF enable_biased_competition flag (follow-on).", flush=True)
    elif go_seeds >= bar and lesion_seeds >= bar and moat_seeds == n and three_seeds < bar:
        print(f"\n  ==> BOUNDARY: the 2-referent case resolves (lesion+moat+baselines hold) but the 3-referent "
              "case degrades\n  -> localizes competition-strength-vs-N as the next tuning sub-problem (within the "
              "alpha<1 envelope).", flush=True)
    else:
        print(f"\n  ==> NEGATIVE: even with mutual inhibition + a small content bias the intrinsic-attractor "
              "asymmetry dominates\n  (or the lesion did not break resolution = the bias was not load-bearing). "
              "Honest rate-attractor substrate boundary\n  -> re-scope to gamma-cycle (N.19) phase segregation; "
              "do NOT escalate into a config search.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
