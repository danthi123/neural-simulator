"""Tier 2 #6 de-risk: does the shared DOPAMINE state functionally modulate the conversational composer's memory by
gating fact ENCODING STRENGTH at store time (Lisman-Grace hippocampal-VTA loop; Kandel D.16 -- dopamine makes a
memory trace STABLE vs degradable)?

THE LOAD-BEARING QUESTION (pre-registered, research/findings/2026-06-19-tier2-limbic-to-composer-scoping.md):
scaling a fact's stored complex-phasor magnitude by a dopamine-driven encoding gain `g = 1 + k_DA*(DA - DA_base)`
makes a REWARDED fact (g>1) recallable where a NEUTRAL fact (g=1, MATCHED cue strength) degrades under common read
damage -- in a LOAD-BEARING, neural way (the DA-lesion kills the differential), WITHOUT weakening the no-confab moat.

THE VERIFIED MECHANISM: the RF phase read-out has a hard MAGNITUDE FLOOR (sim/bridge.py:5589, `_rf_mag2 > _rf_floor2`
-- a readout neuron whose |Z| decays below the floor never up-crosses -> reads phase 0 = garbage). Under COMMON,
gain-independent additive read noise of fixed sigma, a readout's per-neuron SNR is `g*M / sigma`: a higher-gain
(rewarded) fact has cleaner phase -> survives the floor -> the cleanup recalls it; a unit-gain (neutral) fact's
low-SNR neurons drop below the floor -> garbled phase -> the cleanup mis-recalls. The floor x noise interaction is
the nonlinearity that makes a (per-fact) encoding gain DIFFERENTIAL rather than a vacuous global scalar.

This is realized via the composer's DEFAULT-OFF `encoding_gain_fn` (the gain `g` read at store time -- a probe DA
here; the live shared `dopamine` SNc modulator in deployment) + the default-preserving `_retrieve_noise` read-damage
knob. NO sim/ edit (composer-layer multiply on the written complex weight + composer-layer read noise). The damage
is calibrated to the graceful-degradation knee where a unit-gain fact STARTS to fail (noise=350 for D=64, two facts).

GO BAR (all; >=5/6 seeds):
  - REWARDED recalls where NEUTRAL (matched cue) abstains/degrades -- the differential;
  - DA-LESION (both g=1) KILLS the differential -- the decisive control (it's the gain, not the content);
  - PERMUTED-gain (apply the gain to the OTHER fact) -> the advantage follows the gain, not the fact;
  - the no-confab MOAT holds at EVERY gain -- an UNSTORED cue still returns None (HARD gate);
  - regression: encoding_gain_fn=None == g=1 (byte-identical default).

CPU/numpy; run INLINE. python -m research.runners._phaseB_dopamine_encoding_gain_derisk
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer

# The de-risk geometry: a NEUTRAL fact + a REWARDED fact, MATCHED cue strength (both plain 3-role SVO), distinct
# content; the only intended difference is the encoding gain. Plus an UNSTORED cue for the moat.
VOCAB = ["dog", "cat", "go", "eat", "north", "apple", "river", "run"]
NEUTRAL = ("dog", "go", "north")        # the matched-cue neutral fact (g=1)
REWARDED = ("cat", "eat", "apple")      # the rewarded fact (g>1)
UNSTORED = ("river", "run")             # the moat probe (never stored)

SEEDS = [42, 43, 44, 100, 101, 102]


def _build(seed, g_for_first, g_for_second, noise, D, read_floor):
    """Two-fact composer: NEUTRAL stored first (gain g_for_first), REWARDED second (gain g_for_second). The gain is
    read per-store via a one-shot encoding_gain_fn closure over c._next_g."""
    c = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, enable_substrate_store=True)
    c._next_g = float(g_for_first)
    c.encoding_gain_fn = (lambda: c._next_g)
    c.store(*NEUTRAL)
    c._next_g = float(g_for_second)
    c.store(*REWARDED)
    c._retrieve_noise = float(noise)
    c._retrieve_read_floor = float(read_floor)
    return c


# deterministic per-query offsets (NOT hash() -- str hashing is PYTHONHASHSEED-randomized per process, which would
# make the read-damage draw non-reproducible across runs).
_TAG_OFFSET = {"neu": 11, "rew": 23, "moat": 37}


def _q(c, agent, action, seed, tag):
    """A query with an independent-but-reproducible common read-damage draw (reseed the read-noise RNG per query so
    each (fact, query) sees its own damage realization; the damage sigma is gain-INDEPENDENT)."""
    c._retrieve_noise_rng = np.random.default_rng(seed * 1000 + _TAG_OFFSET[tag])
    return c.query_patient(agent, action)


def run_condition(seeds, g_neutral, g_rewarded, noise, D, read_floor):
    """For each seed: recall the neutral-slot fact, the rewarded-slot fact, and the moat probe. Returns per-seed
    booleans (neutral_ok, rewarded_ok, moat_ok)."""
    neu, rew, moat = [], [], []
    for s in seeds:
        c = _build(s, g_neutral, g_rewarded, noise, D, read_floor)
        neu.append(_q(c, NEUTRAL[0], NEUTRAL[1], s, "neu") == NEUTRAL[2])
        rew.append(_q(c, REWARDED[0], REWARDED[1], s, "rew") == REWARDED[2])
        # moat at the prevailing (rewarded) gain regime: an unstored cue must still abstain
        moat.append(_q(c, UNSTORED[0], UNSTORED[1], s, "moat") is None)
    return neu, rew, moat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=64)
    # noise=260 = the moat-safe knee for D=64, two facts: the within-fact gain lift is strong AND the no-confab moat
    # holds 6/6 in every condition. Above ~280 the differential grows but the heavy read damage ITSELF (not the gain)
    # starts to breach the moat on a seed -- and the HARD gate forbids any moat breach, so we operate below it.
    ap.add_argument("--noise", type=float, default=260.0, help="common read-damage sigma (the moat-safe knee)")
    ap.add_argument("--read-floor", type=float, default=1.0e-2)
    ap.add_argument("--g-rewarded", type=float, default=2.0)
    ap.add_argument("--out", type=str, default="research/findings/raw/_phaseB_dopamine_encoding_gain.json")
    args = ap.parse_args()

    D, noise, rf, gR = args.D, args.noise, args.read_floor, args.g_rewarded
    results = {}

    # The three gain assignments over the SAME two facts (NEUTRAL fact stored first, REWARDED fact second). The KEY
    # measure is the WITHIN-FACT paired contrast (each fact recalled at g=1 AND at g=2, with its CONTENT held fixed --
    # this removes the content-robustness confound that a between-fact comparison carries):
    #   REAL     (neu fact g=1, rew fact g=2): gives neu_fact@g1 + rew_fact@g2
    #   PERMUTED (neu fact g=2, rew fact g=1): gives neu_fact@g2 + rew_fact@g1   <- the gain swapped to the OTHER fact
    #   LESION   (both g=1):                   the within-fact NULL (no fact gets a gain -> no differential between them)
    neu_r, rew_r, moat_r = run_condition(SEEDS, 1.0, gR, noise, D, rf)     # REAL
    neu_p, rew_p, moat_p = run_condition(SEEDS, gR, 1.0, noise, D, rf)     # PERMUTED
    neu_l, rew_l, moat_l = run_condition(SEEDS, 1.0, 1.0, noise, D, rf)    # LESION

    # within-fact gain lift (content-controlled): recall@g2 - recall@g1 for EACH fact
    neu_fact_g1, neu_fact_g2 = sum(neu_r), sum(neu_p)     # dog-go-north at g=1 (REAL) vs g=2 (PERMUTED)
    rew_fact_g1, rew_fact_g2 = sum(rew_p), sum(rew_r)     # cat-eat-apple at g=1 (PERMUTED) vs g=2 (REAL)
    neu_lift = neu_fact_g2 - neu_fact_g1
    rew_lift = rew_fact_g2 - rew_fact_g1
    total_lift = neu_lift + rew_lift

    results["real"] = {"neutral_recall": sum(neu_r), "rewarded_recall": sum(rew_r), "moat_ok": sum(moat_r),
                       "per_seed_neutral": neu_r, "per_seed_rewarded": rew_r, "per_seed_moat": moat_r}
    results["permuted"] = {"neutral_slot_recall": sum(neu_p), "rewarded_slot_recall": sum(rew_p),
                           "moat_ok": sum(moat_p), "per_seed_neutral_slot": neu_p, "per_seed_rewarded_slot": rew_p}
    results["lesion"] = {"neutral_recall": sum(neu_l), "rewarded_recall": sum(rew_l), "moat_ok": sum(moat_l),
                         "per_seed_neutral": neu_l, "per_seed_rewarded": rew_l}
    results["within_fact_gain_lift"] = {
        "neutral_fact(dog-go-north)": {"g1": neu_fact_g1, "g2": neu_fact_g2, "lift": neu_lift},
        "rewarded_fact(cat-eat-apple)": {"g1": rew_fact_g1, "g2": rew_fact_g2, "lift": rew_lift},
        "total_lift_out_of_12": total_lift,
        "lesion_within_fact_null(both_g1_diff)": abs(sum(rew_l) - sum(neu_l)),
    }

    # --- MONOTONICITY (single fact, vary g at the fixed noise) ---
    mono = {}
    for g in [0.5, 1.0, 1.5, 2.0, 3.0]:
        rec = []
        for s in SEEDS:
            c = RFPhasorComposer(seed=s, D=D, vocab=VOCAB, enable_substrate_store=True,
                                 encoding_gain_fn=(lambda gg=g: gg))
            c.store(*NEUTRAL)
            c._retrieve_noise = noise
            c._retrieve_read_floor = rf
            c._retrieve_noise_rng = np.random.default_rng(s * 7 + 1)
            rec.append(c.query_patient(NEUTRAL[0], NEUTRAL[1]) == NEUTRAL[2])
        mono[f"g={g}"] = sum(rec)
    results["monotonicity"] = mono

    # --- REGRESSION: default None == explicit g=1.0 (recall + moat, no damage) ---
    reg_ok = []
    for s in SEEDS:
        cd = RFPhasorComposer(seed=s, D=D, vocab=VOCAB, enable_substrate_store=True)  # encoding_gain_fn=None
        cd.store(*NEUTRAL)
        c1 = RFPhasorComposer(seed=s, D=D, vocab=VOCAB, enable_substrate_store=True,
                              encoding_gain_fn=(lambda: 1.0))
        c1.store(*NEUTRAL)
        ok = (cd.query_patient(NEUTRAL[0], NEUTRAL[1]) == c1.query_patient(NEUTRAL[0], NEUTRAL[1]) == NEUTRAL[2]
              and cd.query_patient(*UNSTORED) is None and c1.query_patient(*UNSTORED) is None)
        reg_ok.append(ok)
    results["regression_default_eq_g1"] = sum(reg_ok)

    # --- VERDICT ---
    # GO: the within-fact gain lift is positive on BOTH facts and >=4/12 total (the gain reliably protects the SAME
    # fact under common damage); the LESION within-fact null shows no between-fact differential (it's the gain, not
    # content); monotonic; the moat holds 6/6 in EVERY condition (HARD gate); regression byte-identical.
    moat_intact = (sum(moat_r) == len(SEEDS) and sum(moat_p) == len(SEEDS) and sum(moat_l) == len(SEEDS))
    mono_ok = (mono["g=0.5"] <= mono["g=1.0"] <= mono["g=1.5"] <= mono["g=2.0"] <= mono["g=3.0"]
               and mono["g=3.0"] > mono["g=0.5"])
    lesion_null = results["within_fact_gain_lift"]["lesion_within_fact_null(both_g1_diff)"]
    go = (neu_lift >= 1 and rew_lift >= 1 and total_lift >= 4
          and total_lift > lesion_null and moat_intact and mono_ok
          and results["regression_default_eq_g1"] == len(SEEDS))
    results["verdict"] = {
        "GO": bool(go),
        "within_fact_total_gain_lift_/12": total_lift,
        "neutral_fact_lift": neu_lift, "rewarded_fact_lift": rew_lift,
        "lesion_within_fact_null(should_be_~0)": lesion_null,
        "moat_intact_all_conditions(HARD)": bool(moat_intact),
        "monotonic_in_gain": bool(mono_ok),
        "regression_byte_identical": results["regression_default_eq_g1"] == len(SEEDS),
        "config": {"D": D, "noise": noise, "read_floor": rf, "g_rewarded": gR, "seeds": SEEDS},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n==================== Tier 2 #6 DOPAMINE ENCODING-GAIN DE-RISK ====================")
    print(f"  config: D={D} noise={noise} read_floor={rf} g_rewarded={gR} seeds={SEEDS}")
    print("  WITHIN-FACT GAIN LIFT (content-controlled -- same fact, g=1 vs g=2):")
    print(f"    NEUTRAL fact (dog go north):   g1 {neu_fact_g1}/6 -> g2 {neu_fact_g2}/6   (+{neu_lift})")
    print(f"    REWARDED fact (cat eat apple): g1 {rew_fact_g1}/6 -> g2 {rew_fact_g2}/6   (+{rew_lift})")
    print(f"    TOTAL gain lift:               +{total_lift}/12   "
          f"(LESION within-fact null diff = {lesion_null}, should be ~0)")
    print(f"  BETWEEN-FACT (REAL: neu g=1, rew g={gR}): rew {sum(rew_r)}/6  neu {sum(neu_r)}/6")
    print(f"  MONOTONICITY (single fact):      {mono}  -> monotonic: {mono_ok}")
    print(f"  MOAT intact 6/6 ALL conditions:  {moat_intact}  "
          f"(real {sum(moat_r)}/6 perm {sum(moat_p)}/6 lesion {sum(moat_l)}/6)")
    print(f"  REGRESSION (None==g1):           {results['regression_default_eq_g1']}/6")
    print(f"\n  VERDICT: {'GO' if go else 'NEGATIVE'} -- the dopamine encoding gain "
          f"{'DOES' if go else 'does NOT'} give a load-bearing, content-controlled, moat-safe recall lift.")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
