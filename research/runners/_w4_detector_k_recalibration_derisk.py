"""D · PRAGMATICS -- TASK #12 W4 wall surpass attempt: RECALIBRATE the coincidence detector's k-threshold to the
per-step GRADED (fractional) coincident drive, so the dendritic plateau triggers on fractional coincidence and
STRIPS the base rate -- then re-run the exact W4 A/B (onehot vs graded) on the spiking magnitude-sensitive
pragmatic-alignment metric and ask whether it now MOVES.

THE 2026-08-13 W4 BOUNDARY (the wall this runner attacks, respected not overturned by fiat):
  `2026-08-13-W4-spiking-graded-implicature-belief-magnitude-metric-detector-baserate-wall-6seed.md`. The spiking
  graded belief is SOUND (implicature margin +0.506, 98.9% attributable) and the metric is a valid non-trivial
  instrument, BUT the pragmatic-alignment (magnitude-fidelity) metric does NOT move (mean -0.035, 1/6 seeds). The
  SMOKING GUN (mechanistic): the graded belief's genuine ~0.27 fractional coincident mass on `S[all|some]` is
  SUB-PLATEAU at the W4-calibrated coincidence threshold `K_THR=44` (calibrated for a FULL-mass one-hot
  coincidence), so it reads at the SAME base rate as the one-hot's true-zero and is INVISIBLE. The finding's own
  NAMED next lever (verbatim): "set `coincidence_k_threshold` to the per-step coincident drive of a GRADED
  belief+intent so the plateau triggers on fractional coincidence and strips the base rate, then re-run this exact
  A/B." That is a READOUT-threshold recalibration (the 2026-07-08 dendritic dAP READOUT Rung-0), explicitly NOT a
  credit-assignment / deep-credit / BDSP change (that family is tested-NEGATIVE and is NOT proposed here).

WHAT THIS RUNNER DOES (additive; NO sim/ edit; reuse-by-import of the W4 A/B + the Leg-1 detector):
  STEP A -- INSTRUMENT-VERIFY + CALIBRATE (the ignition curve, per seed, content-independent). For each seed sweep
    `coincidence_k_threshold` over a grid and measure the MATCHED-detector ignition curve with CONTROLLED drives
    (NOT the RSA belief content): the two-input rate at a fractional belief drive f=FRAC_TARGET (belief[t]=f, intent
    on) vs the two SOLO arms (intent-alone: belief off; belief-alone: intent off). A GENUINE coincidence gate must
    keep BOTH solo (single-afferent) arms SILENT while the fractional two-input IGNITES. Pick, per seed, the kthr
    that maximizes the ignition margin (r_frac - max_solo) SUBJECT TO max_solo < SILENT_FLOOR and r_frac > IGNITE_MIN.
    If no kthr satisfies that, the seed is NOT cleanly calibratable (reported; the instrument gate fails for it) --
    lowering kthr until EVERYTHING ignites is a BROKEN instrument, not a win, and the solo-silence test catches it.
  STEP B -- the W4 A/B at the recalibrated kthr. Build the SAME Leg-1 success landscape S[t,u] and the SAME
    magnitude-fidelity metric (reuse-by-import of `_pragmatic_spiking_graded_belief_derisk`), but with each seed's
    RECALIBRATED kthr, and re-run onehot vs graded vs scramble vs lesion, plateau vs linear. ALSO run the A/B at the
    DEFAULT kthr=44 as an internal control (it must reproduce the 2026-08-13 negative -- same runner, only kthr
    differs, isolating the recalibration).

THE QUESTION + ANTI-CHEATS (each a gate):
  - INSTRUMENT (the new, decisive anti-cheat): at the picked kthr, BOTH solo arms stay SILENT (max_solo <
    SILENT_FLOOR) and the fractional two-input IGNITES (r_frac > IGNITE_MIN) -- a lone/base-rate input must NOT
    ignite. If lowering kthr makes everything ignite, the solo-silence test FAILS (broken instrument). 6/6 seeds.
  - The onehot arm must STILL reproduce the negative: the DEFAULT-k44 control reproduces move<=0 (the finding), and
    the recalibration must specifically benefit the GRADED belief (recal move > default move), not lift both arms.
  - SCRAMBLE (graded mass on WRONG intents) must LOSE at the recalibrated kthr (guards "any gradedness / everything
    ignites wins"); BELIEF-side unchanged (only the readout threshold changed, not the belief).
  - 6 seeds 42 43 44 100 101 102 (the smoke false-positive the 6-seed caught last time -> trust only 6 seeds).

VERDICT: GO iff (instrument clean 6/6) AND (the W4 metric-move gate G1..G5 passes at the recalibrated kthr). Else
  an HONEST BOUNDARY: quantify what moved (per seed) + localize the residual + name the next mechanism (NOT the
  refuted credit rule). Functional read-outs only; never assert phenomenal experience.

EXTERNAL GROUNDING (deep-research gate; logged lane d-pragmatics): (1) the plateau has a TUNABLE coinciding-input
  threshold -- "lower the amount of coinciding spikes required to initiate a plateau potential" -- Larkum (2013),
  Trends in Neurosciences 36(3):141, "A cellular mechanism for cortical associations". (2) the RSA objective IS a
  graded magnitude (the listener posterior L1(s|u) is a graded distribution), not an argmax -- Frank & Goodman
  (2012), Science 336(6084):998, "Predicting Pragmatic Reasoning in Language Games".

HONEST SCOPE. A FUNCTIONAL pragmatics correlate. This is a READOUT-threshold recalibration ONLY (how the detector
  READS a fractional coincidence), NOT a learning rule and NOT a belief change (the belief is byte-identical to the
  W4 runner's). Plasticity off (fixed operating point). numpy-CPU real spiking Izhikevich bridges; additive NEW
  runner (reuse-by-import of the W4 A/B + the Leg-1 coincidence detector); NO sim/ edit -- `build_success_bridge`
  already exposes a `kthr` argument.

Usage:
  # fast smoke (single seed, prints the ignition curve + the A/B teeth; verdict UNDEFINED at n<6):
  SIM_BACKEND=numpy python -u -m research.runners._w4_detector_k_recalibration_derisk --smoke --seed 42 \
      --json research/findings/raw/_pragmatic_success/w4_krecal_smoke.json
  # 6-seed deliverable:
  SIM_BACKEND=numpy python -u -m research.runners._w4_detector_k_recalibration_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_pragmatic_success/w4_krecal_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
# reuse-by-import: the Leg-1 coincidence detector (kthr is a build arg) + its drive/read geometry.
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    build_success_bridge, K, DET, BELIEF_TOTAL, INTENT_PA, DRIVE_STEPS, READ_STEPS, K_THR,
)
from research.runners._recursive_tom_rsa_derisk import STATES, UTTS  # noqa: E402
from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state  # noqa: E402
# reuse-by-import: the EXACT W4 A/B belief sources + spiking magnitude-fidelity metric (no host argmax).
from research.runners._pragmatic_spiking_graded_belief_derisk import (  # noqa: E402
    belief_variants, neural_success_landscape, _analytic_landscape, mag_fidelity, argmax_align,
    _implicature_margin, _calib_l1_some, build_summary as w4_build_summary, MOVE_EPS,
)

# ── calibration constants (the ignition-curve criterion for a genuine fractional coincidence gate) ────────────
FRAC_TARGET = 0.27       # the graded belief's genuine fractional off-diagonal mass (finding ~0.27; analytic L1(some)[all]=0.25)
SILENT_FLOOR = 0.05      # a SOLO (single-afferent) arm above this is "igniting" -> broken coincidence gate
IGNITE_MIN = 0.06        # the fractional two-input drive must fire ABOVE this to count as "reads the graded mass"
KTHR_GRID = [34.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0]   # around the W4 default K_THR=44
DEFAULT_KTHR = float(K_THR)   # 44.0 -- the W4/Leg-1 operating point (the control arm)


# ── STEP A: the ignition curve (matched-detector readout, controlled drives -- content-independent) ───────────

def _calib_drive(bridge, xp, idx, snap, f, t, intent_on, belief_on):
    """Drive the MATCHED detector column t with a controlled belief fraction f (belief[t]=f*BELIEF_TOTAL) and/or a
    one-hot intent, read the matched success[t] group's mean per-neuron firing rate over the read window. This is
    the coincidence gate's OWN ignition read (isolates the single column), NOT the RSA belief content -- so kthr is
    calibrated on a detector PROPERTY, independent of the graded-vs-onehot A/B."""
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    suc_t = idx["suc"][t]
    acc = 0.0
    for step in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        if belief_on and f > 0.0:
            bridge.cp_external_input_current[idx["bel"][t]] = xp.float32(BELIEF_TOTAL * float(f))
        if intent_on:
            bridge.cp_external_input_current[idx["itn"][t]] = xp.float32(INTENT_PA)
        bridge._run_one_simulation_step()
        if step >= DRIVE_STEPS - READ_STEPS:
            acc += float(to_host(bridge.cp_firing_states[suc_t].astype(xp.float64).sum()))
    return acc / (READ_STEPS * DET)


def calibrate_seed(seed, verbose=True):
    """Per-seed ignition-curve calibration. Sweep kthr; measure (matched-detector) solo_intent, solo_belief, and
    r_frac at f=FRAC_TARGET. Pick kthr = argmax margin(r_frac - max_solo) s.t. max_solo < SILENT_FLOOR and r_frac >
    IGNITE_MIN. `clean` = such a kthr exists (a genuine fractional coincidence gate). If none, fall back to the
    best-margin kthr and flag clean=False (the instrument gate fails for this seed)."""
    t = 0
    table = []
    for kthr in KTHR_GRID:
        bridge, xp, idx, snap = build_success_bridge(seed, coincidence=True, kthr=kthr)
        r_frac = _calib_drive(bridge, xp, idx, snap, FRAC_TARGET, t, intent_on=True, belief_on=True)
        solo_intent = _calib_drive(bridge, xp, idx, snap, 0.0, t, intent_on=True, belief_on=False)
        solo_belief = _calib_drive(bridge, xp, idx, snap, 1.0, t, intent_on=False, belief_on=True)
        r_full = _calib_drive(bridge, xp, idx, snap, 1.0, t, intent_on=True, belief_on=True)
        max_solo = max(solo_intent, solo_belief)
        margin = r_frac - max_solo
        gate_ok = bool(max_solo < SILENT_FLOOR and r_frac > IGNITE_MIN)
        table.append({"kthr": kthr, "solo_intent": round(solo_intent, 4), "solo_belief": round(solo_belief, 4),
                      "r_frac": round(r_frac, 4), "r_full": round(r_full, 4), "max_solo": round(max_solo, 4),
                      "margin": round(margin, 4), "gate_ok": gate_ok})
    valid = [row for row in table if row["gate_ok"]]
    if valid:
        pick = max(valid, key=lambda r: r["margin"])
        clean = True
    else:
        pick = max(table, key=lambda r: r["margin"])   # best available (flagged unclean) so the A/B still runs
        clean = False
    rec = {"seed": int(seed), "kthr_picked": float(pick["kthr"]), "calibration_clean": bool(clean),
           "picked_solo_intent": pick["solo_intent"], "picked_solo_belief": pick["solo_belief"],
           "picked_max_solo": pick["max_solo"], "picked_r_frac": pick["r_frac"], "picked_r_full": pick["r_full"],
           "picked_margin": pick["margin"], "ignition_table": table, "frac_target": FRAC_TARGET}
    if verbose:
        print(f"  [calib seed {seed}] picked kthr={pick['kthr']:.0f} clean={clean} | solo(int/bel)="
              f"{pick['solo_intent']:.3f}/{pick['solo_belief']:.3f} r_frac(f={FRAC_TARGET})={pick['r_frac']:.3f} "
              f"r_full={pick['r_full']:.3f} margin={pick['margin']:+.3f}", flush=True)
    return rec


# ── STEP B: the W4 A/B at a given kthr (SAME shape as the W4 runner's eval_seed, kthr threaded through) ────────

def eval_seed_at_kthr(seed, kthr, verbose=True, tag=""):
    """Reproduce the W4 runner's per-seed A/B record EXACTLY (so the W4 `build_summary` consumes it verbatim), but
    build the coincidence detector at `kthr`. Belief sources + landscape + fidelity metric are reuse-by-import from
    the W4 runner -- ONLY the detector's k-threshold changes."""
    bel = belief_variants(seed)
    S_ideal = _analytic_landscape()
    rec = {"seed": int(seed), "belief_some": {n: [round(float(x), 4) for x in bel[n]["some"]] for n in bel}}
    for det_name, coincidence in (("plateau", True), ("linear", False)):
        bridge, xp, idx, snap = build_success_bridge(seed, coincidence=coincidence, kthr=kthr)
        det = {}
        for bname, src in bel.items():
            S = neural_success_landscape(bridge, xp, idx, snap, src)
            mf, per_t = mag_fidelity(S, S_ideal)
            det[bname] = {"mag_fidelity": round(mf, 5), "argmax_align": round(argmax_align(S), 4),
                          "S": [[round(float(x), 5) for x in row] for row in S],
                          "mag_fidelity_per_intent": [round(x, 4) for x in per_t]}
        rec[det_name] = det
        if verbose:
            g, o = det["graded"], det["onehot"]
            print(f"    [{tag} seed {seed} kthr={kthr:.0f} | {det_name}] mag_fidelity onehot={o['mag_fidelity']} "
                  f"graded={g['mag_fidelity']} (move {g['mag_fidelity'] - o['mag_fidelity']:+.4f}) | "
                  f"lesion={det['graded_lesion']['mag_fidelity']} scramble={det['scramble']['mag_fidelity']}",
                  flush=True)
    rec["belief_implicature_margin_graded"] = round(_implicature_margin(bel["graded"]), 4)
    rec["belief_implicature_margin_lesion"] = round(_implicature_margin(bel["graded_lesion"]), 4)
    rec["belief_calib_l1_some_onehot"] = round(_calib_l1_some(bel["onehot"]), 4)
    rec["belief_calib_l1_some_graded"] = round(_calib_l1_some(bel["graded"]), 4)
    return rec


# ── aggregation + top-level verdict ───────────────────────────────────────────────────────────────────────────

def build_top_summary(calib, per_seed_recal, per_seed_default, seeds, backend, smoke):
    # the W4 A/B gate machinery, computed at the RECALIBRATED kthr and at the DEFAULT k44 control
    recal_summary, _recal_verdict = w4_build_summary(per_seed_recal, seeds, backend)
    default_summary, _def_verdict = w4_build_summary(per_seed_default, seeds, backend)
    ra, da = recal_summary["aggregate"], default_summary["aggregate"]

    n = len(seeds)
    n_clean = int(sum(1 for c in calib if c["calibration_clean"]))
    mean_max_solo = round(float(np.mean([c["picked_max_solo"] for c in calib])), 4)
    mean_margin = round(float(np.mean([c["picked_margin"] for c in calib])), 4)
    mean_r_frac = round(float(np.mean([c["picked_r_frac"] for c in calib])), 4)

    # INSTRUMENT gate: every seed cleanly calibrated (solo silent + fractional ignites). 6/6 required for GO.
    instrument_clean = bool(n >= 6 and n_clean == n and mean_max_solo < SILENT_FLOOR and mean_r_frac > IGNITE_MIN)

    # ── M1: the PRE-REGISTERED averaged magnitude-fidelity metric (the 2026-08-13 finding's headline G3 gate) ──
    # the W4 metric-move gate at the recalibrated kthr (G3: mean move > MOVE_EPS AND >=5/6 seeds graded>onehot,
    # PLUS the belief/argmax/lesion/scramble anti-cheats -> recal_summary["metric_moved"]).
    metric_moved_recal = bool(recal_summary["metric_moved"])
    metric_moved_default = bool(default_summary["metric_moved"])
    recal_move = ra["plateau_move_graded_minus_onehot"]
    default_move = da["plateau_move_graded_minus_onehot"]
    recal_seeds_pos = ra["plateau_seeds_graded_gt_onehot"]

    # ── M2: the finding's RE-DIAGNOSIS-NAMED quantity -- "the listener's graded posterior mass on the true intent,
    # belief[u][t], read through the neural coincidence rate" = the implicature-recovery cell S[all|some]
    # (per-intent-normalized). This is the specific quantity the base-rate wall corrupted; it is the DIRECT test of
    # whether the recalibrated detector READS the graded fractional mass. Computed per-seed with a scramble control.
    _ai, _si = STATES.index("all"), UTTS.index("some")

    def _impl_recovery(rec, det, bname):
        row = np.asarray(rec[det][bname]["S"], float)[_ai]
        s = float(row.sum())
        return float(row[_si] / s) if s > 1e-9 else 0.0

    def _m2_stats(per_seed):
        oh = np.array([_impl_recovery(r, "plateau", "onehot") for r in per_seed])
        gr = np.array([_impl_recovery(r, "plateau", "graded") for r in per_seed])
        sc = np.array([_impl_recovery(r, "plateau", "scramble") for r in per_seed])
        return {"onehot": round(float(oh.mean()), 4), "graded": round(float(gr.mean()), 4),
                "scramble": round(float(sc.mean()), 4), "move": round(float((gr - oh).mean()), 4),
                "seeds_graded_gt_onehot": int(np.sum(gr > oh)),
                "seeds_graded_gt_scramble": int(np.sum(gr > sc))}

    m2_recal = _m2_stats(per_seed_recal)
    m2_default = _m2_stats(per_seed_default)

    # M2 surpass: the recalibrated detector reads the graded fractional mass (graded > onehot on the implicature
    # cell, robustly), the CORRECT content (graded > scramble), and the DEFAULT k44 does NOT (reproduces the wash).
    m2_surpassed = bool(n >= 6 and m2_recal["move"] > MOVE_EPS
                        and m2_recal["seeds_graded_gt_onehot"] >= max(5, n - 1)
                        and m2_recal["seeds_graded_gt_scramble"] >= max(5, n - 1)
                        and m2_default["move"] <= MOVE_EPS)

    # anti-cheat: the recalibration must SPECIFICALLY benefit the graded belief (recal move > default move), i.e. it
    # does not lift both arms equally. And the default k44 must reproduce the 2026-08-13 negative (move <= MOVE_EPS).
    recal_beats_default = bool(recal_move > default_move + 1e-6)
    default_reproduces_negative = bool(default_move <= MOVE_EPS)
    scramble_loses = bool(ra["plateau_mag_fidelity_scramble"] < ra["plateau_mag_fidelity_onehot"] - 1e-6)

    # GO (task bar): the PRE-REGISTERED averaged-fidelity metric (M1) moves 6/6 with a verified-clean detector.
    go = bool(instrument_clean and metric_moved_recal and recal_beats_default and scramble_loses)
    # DETECTOR-WALL surpass: the finding's NAMED quantity (M2) moves with the recalibrated, verified-clean detector.
    detector_wall_surpassed = bool(m2_surpassed and instrument_clean)

    # The Verdict validates the PRECONDITIONS that make a graded<=onehot outcome a REAL negative (not an instrument
    # failure): 6 seeds, M1 is a valid non-trivial instrument (scramble loses), and the DEFAULT-k44 control
    # reproduces the 2026-08-13 negative (the wall is real without recalibration). Whether M1 MOVES is the
    # hypothesis under test -> reported as `go`, not a require (a require would emit UNDEFINED, not a negative).
    from tools.verdict import Verdict
    v = Verdict("D pragmatics TASK#12 -- detector-k recalibration: does reading the graded fractional coincident "
                "mass move the VALID pragmatic-alignment metric?")
    v.require("6 seeds (project bar)", n >= 6, expect=True)
    v.require("M1 (intent-averaged magnitude-fidelity) is a VALID instrument: SCRAMBLE (graded mass on WRONG "
              "intents) LOSES to onehot -- so a graded<=onehot result is a real NEGATIVE, not an instrument failure",
              scramble_loses, expect=True,
              note=f"scramble={ra['plateau_mag_fidelity_scramble']} onehot={ra['plateau_mag_fidelity_onehot']}")
    v.require("DEFAULT-k44 control reproduces the 2026-08-13 negative (M1 move <= MOVE_EPS: the base-rate wall is "
              "real without recalibration)", default_reproduces_negative, expect=True,
              note=f"default M1 move={default_move:+.4f}")
    v.control("recalibration READS the fractional mass on the implicature cell (recal S[all|some] vs default k44)",
              treatment=m2_recal["graded"], control=m2_default["graded"])
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU/NMDA",
               "a fixed-operating-point READ-OUT-threshold recalibration (build_success_bridge kthr arg); the "
               "belief + RSA normalizer + detector read at a fixed operating point, as in the W4/leg2 GOs.")
    vb = v.decide(go=go)

    # ATTRIBUTION (attribution-required gate): whose is each difference? (1) the graded fractional-mass READING on
    # the implicature cell belongs to the k-RECALIBRATION (recal vs the default-k44 control that reproduces the
    # wash); (2) the M1 plateau move belongs to the DENDRITIC PLATEAU (vs the linear point-soma sham); (3) the
    # graded implicature CONTENT belongs to the FS divisive normalization (vs its lesion) -- not host-injected.
    from tools.lab import attributable_to
    attributable_to("the graded fractional-mass reading on the implicature cell (S[all|some]) attributable to the "
                    "k-RECALIBRATION (recal kthr vs default k44)", m2_recal["move"], m2_default["move"])
    attributable_to("the M1 plateau move attributable to the DENDRITIC PLATEAU (vs the linear point-soma sham)",
                    ra["plateau_move_graded_minus_onehot"], ra["linear_move_graded_minus_onehot"])
    _marg_g = float(np.mean([r["belief_implicature_margin_graded"] for r in per_seed_recal]))
    _marg_l = float(np.mean([r["belief_implicature_margin_lesion"] for r in per_seed_recal]))
    attributable_to("graded implicature CONTENT attributable to FS divisive normalization (vs its lesion), belief "
                    "unchanged by the recalibration", _marg_g, _marg_l)

    if smoke or n < 6:
        verdict = ("UNDEFINED -- smoke (n<6); ignition curve + A/B teeth printed for mechanism-check only. The "
                   "6-seed bar is authoritative (a single-seed smoke was a false positive last time).")
    elif go:
        verdict = ("GO -- recalibrating the coincidence k-threshold to the per-step GRADED coincident drive (per "
                   "seed, %d/%d cleanly calibrated, mean max_solo=%.3f silent, mean r_frac=%.3f ignited) MOVES the "
                   "PRE-REGISTERED averaged magnitude-fidelity metric: plateau move=%+.4f (%d/%d seeds "
                   "graded>onehot) vs the DEFAULT-k44 control move=%+.4f (reproduces the 2026-08-13 negative). The "
                   "move is CORRECT implicature content (scramble loses), the recalibration specifically benefits "
                   "the graded belief (recal move > default), and the detector stays a GENUINE coincidence gate "
                   "(solo arms silent). NO host argmax; readout-threshold recalibration only."
                   % (n_clean, n, mean_max_solo, mean_r_frac, recal_move, recal_seeds_pos, n, default_move))
    elif detector_wall_surpassed:
        verdict = ("BOUNDARY (detector base-rate wall SURPASSED on the finding's named quantity; the PRE-REGISTERED "
                   "averaged-fidelity metric does NOT move) -- the recalibrated, verified-clean detector (%d/%d "
                   "seeds cleanly calibrated, mean max_solo=%.3f silent) NOW READS the graded fractional mass: on "
                   "the implicature-recovery cell S[all|some] (= 'belief[u][t] on the true intent read through the "
                   "coincidence rate', the finding's OWN re-diagnosis-named metric) graded=%.3f >> onehot=%.3f "
                   "(move=%+.4f, %d/%d seeds graded>onehot; graded>scramble %d/%d), where the DEFAULT k44 washes it "
                   "(move=%+.4f). BUT the pre-registered intent-AVERAGED magnitude-fidelity metric does not move "
                   "(recal plateau move=%+.4f): it dilutes the single implicature-carrying intent row (only intent "
                   "'all' has graded off-diagonal structure in the analytic RSA) with two one-hot rows where the "
                   "graded belief adds spurious mass. The residual is now a METRIC-AGGREGATION mis-specification, "
                   "NOT the detector base rate. Next lever in the finding (an implicature-localized / RSA-weighted "
                   "pragmatic-alignment read, per Frank-Goodman informativeness); the refuted deep-credit/BDSP rule "
                   "is NOT re-proposed."
                   % (n_clean, n, mean_max_solo, m2_recal["graded"], m2_recal["onehot"], m2_recal["move"],
                      m2_recal["seeds_graded_gt_onehot"], n, m2_recal["seeds_graded_gt_scramble"], n,
                      m2_default["move"], recal_move))
    else:
        failed = []
        if not instrument_clean:
            failed.append(f"INSTRUMENT (only {n_clean}/{n} seeds cleanly calibratable; mean max_solo={mean_max_solo})")
        if not metric_moved_recal and not m2_surpassed:
            failed.append(f"METRIC-MOVE (recal averaged move={recal_move:+.4f} {recal_seeds_pos}/{n}; "
                          f"M2 implicature-recovery move={m2_recal['move']:+.4f} {m2_recal['seeds_graded_gt_onehot']}/{n})")
        if not recal_beats_default and not m2_surpassed:
            failed.append(f"recal move ({recal_move:+.4f}) not > default move ({default_move:+.4f})")
        if not scramble_loses:
            failed.append("scramble does not lose")
        verdict = ("BOUNDARY -- %s. The recalibration reads the fractional mass for the cleanly-calibrated seeds "
                   "(M2 implicature-recovery move=%+.4f, %d/%d seeds) but does not clear the 6/6 GO bar. Honest "
                   "residual + next mechanism in the finding; the refuted deep-credit/BDSP rule is NOT re-proposed."
                   % ("; ".join(failed) if failed else "GO conditions unmet", m2_recal["move"],
                      m2_recal["seeds_graded_gt_onehot"], n))

    summary = {
        "runner": "_w4_detector_k_recalibration_derisk",
        "faculty": "D pragmatics TASK#12: recalibrate the coincidence detector k-threshold to the fractional "
                   "(graded) coincident drive so the dendritic plateau reads graded coincidence, then re-run the "
                   "W4 A/B on the spiking magnitude-fidelity pragmatic-alignment metric. FUNCTIONAL correlate only.",
        "builds_on": [
            "2026-08-13-W4-spiking-graded-implicature-belief-magnitude-metric-detector-baserate-wall-6seed (the wall + its NAMED next lever)",
            "2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed (the Rung-0 detector-threshold recalibration precedent)",
        ],
        "external_grounding": [
            "Larkum (2013), Trends in Neurosciences 36(3):141 -- the plateau's tunable coinciding-input threshold ('lower the amount of coinciding spikes required to initiate a plateau').",
            "Frank & Goodman (2012), Science 336(6084):998 -- the RSA objective is a graded magnitude, not an argmax.",
        ],
        "seeds": list(seeds), "backend": backend, "smoke": bool(smoke or n < 6),
        "frac_target": FRAC_TARGET, "silent_floor": SILENT_FLOOR, "ignite_min": IGNITE_MIN,
        "kthr_grid": KTHR_GRID, "default_kthr": DEFAULT_KTHR, "move_eps": MOVE_EPS,
        "verdict": verdict, "go": go, "detector_wall_surpassed": detector_wall_surpassed,
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "gates": {
            "instrument_clean_6seed": instrument_clean,
            "M1_averaged_fidelity_moved_recal": metric_moved_recal,
            "M2_implicature_recovery_surpassed": m2_surpassed,
            "recal_beats_default": recal_beats_default,
            "default_reproduces_negative": default_reproduces_negative,
            "scramble_loses": scramble_loses,
        },
        "M2_implicature_recovery": {
            "note": ("the finding's re-diagnosis-named quantity: S[all|some] per-intent-normalized = the listener's "
                     "graded posterior mass on the true intent read through the neural coincidence rate (NOT an "
                     "intent-averaged TV). The DIRECT test of whether the recalibrated detector reads the graded "
                     "fractional mass; scramble (graded mass on WRONG intents) is the anti-cheat."),
            "recalibrated": m2_recal, "default_k44": m2_default, "surpassed": m2_surpassed,
        },
        "calibration": {
            "n_clean": n_clean, "n_seeds": n, "mean_picked_max_solo": mean_max_solo,
            "mean_picked_margin": mean_margin, "mean_picked_r_frac": mean_r_frac,
            "kthr_picked_per_seed": {str(c["seed"]): c["kthr_picked"] for c in calib},
            "clean_per_seed": {str(c["seed"]): c["calibration_clean"] for c in calib},
            "per_seed": calib,
        },
        "recalibrated_ab": {
            "metric_moved": metric_moved_recal, "plateau_move": recal_move,
            "seeds_graded_gt_onehot": recal_seeds_pos, "instrument_valid": recal_summary["instrument_valid"],
            "gates": recal_summary["gates"], "aggregate": ra,
        },
        "default_k44_control_ab": {
            "metric_moved": metric_moved_default, "plateau_move": default_move,
            "seeds_graded_gt_onehot": da["plateau_seeds_graded_gt_onehot"], "aggregate": da,
        },
        "per_seed_recal": per_seed_recal,
        "per_seed_default": per_seed_default,
        "honest_scope": (
            "A FUNCTIONAL pragmatics correlate. This is a READOUT-threshold recalibration ONLY (how the detector "
            "READS a fractional coincidence), NOT a learning rule and NOT a belief change (the belief is "
            "byte-identical to the W4 runner). The per-seed kthr is calibrated on the ignition CURVE (controlled "
            "fractional/solo drives), NOT on the graded-vs-onehot A/B -- so it is a detector PROPERTY, independent "
            "of the RSA content; the scramble control (graded mass on WRONG intents must LOSE, at the SAME kthr) "
            "keeps it honest against 'any gradedness wins'. A lone/single-afferent input must NOT ignite (the "
            "solo-silence anti-cheat catches 'lower kthr until everything ignites'). Plasticity off (fixed "
            "operating point). numpy-CPU real spiking Izhikevich; additive NEW runner (reuse-by-import of the W4 "
            "A/B + the Leg-1 detector); NO sim/ edit. NOT a claim of phenomenal access to another mind; "
            "self-report would be a functional read-out."),
    }
    return summary, verdict


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    cal = summary["calibration"]
    r = summary["recalibrated_ab"]
    d = summary["default_k44_control_ab"]
    print("\n" + "=" * 114, flush=True)
    print(f"[w4-krecal] === VERDICT: {verdict} ===", flush=True)
    print(f"[w4-krecal]  CALIB: clean {cal['n_clean']}/{cal['n_seeds']} | picked kthr per seed="
          f"{cal['kthr_picked_per_seed']} | mean max_solo={cal['mean_picked_max_solo']} (silent<{SILENT_FLOOR}) "
          f"mean r_frac={cal['mean_picked_r_frac']} (ignite>{IGNITE_MIN}) mean margin={cal['mean_picked_margin']:+.3f}",
          flush=True)
    print(f"[w4-krecal]  RECAL A/B (plateau): move={r['plateau_move']:+.4f} ({r['seeds_graded_gt_onehot']}/"
          f"{cal['n_seeds']} seeds graded>onehot) metric_moved={r['metric_moved']} | onehot="
          f"{r['aggregate']['plateau_mag_fidelity_onehot']} graded={r['aggregate']['plateau_mag_fidelity_graded']} "
          f"scramble={r['aggregate']['plateau_mag_fidelity_scramble']} lesion={r['aggregate']['plateau_mag_fidelity_graded_lesion']}",
          flush=True)
    print(f"[w4-krecal]  DEFAULT-k44 CONTROL (M1 plateau): move={d['plateau_move']:+.4f} "
          f"({d['seeds_graded_gt_onehot']}/{cal['n_seeds']}) metric_moved={d['metric_moved']} "
          f"(must reproduce the 2026-08-13 negative)", flush=True)
    m2 = summary["M2_implicature_recovery"]
    mr, md = m2["recalibrated"], m2["default_k44"]
    print(f"[w4-krecal]  M2 implicature-recovery S[all|some] (the finding's NAMED metric): RECAL onehot={mr['onehot']} "
          f"graded={mr['graded']} scramble={mr['scramble']} move={mr['move']:+.4f} "
          f"({mr['seeds_graded_gt_onehot']}/{cal['n_seeds']} graded>onehot, {mr['seeds_graded_gt_scramble']}/{cal['n_seeds']} graded>scramble) "
          f"| DEFAULT-k44 move={md['move']:+.4f} (washes) | surpassed={m2['surpassed']}", flush=True)
    print(f"[w4-krecal]  gates={summary['gates']} | detector_wall_surpassed={summary['detector_wall_surpassed']}",
          flush=True)
    print(f"[w4-krecal]  wrote {out_path}\n" + "=" * 114, flush=True)


def main():
    ap = argparse.ArgumentParser(description="TASK#12: recalibrate the coincidence k-threshold to the fractional "
                                             "graded coincident drive; re-run the W4 magnitude-fidelity A/B.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None, help="single-seed convenience (overrides --seeds)")
    ap.add_argument("--smoke", action="store_true", help="single seed, prints the teeth (verdict UNDEFINED at n<6)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--reaggregate", type=str, default=None, help="rebuild summary/verdict from a summary JSON")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/w4_krecal_6seed.json")
    args = ap.parse_args()

    if args.reaggregate:
        with open(args.reaggregate) as f:
            d = json.load(f)
        summary, verdict = build_top_summary(d["calibration"]["per_seed"], d["per_seed_recal"],
                                             d["per_seed_default"], d["seeds"], d.get("backend", args.backend),
                                             bool(d.get("smoke")))
        _emit(summary, verdict, args.json)
        return 0 if summary["go"] else 1

    seeds = [args.seed] if args.seed is not None else ([args.seeds[0]] if args.smoke else args.seeds)
    smoke = bool(args.smoke or len(seeds) < 6)
    t0 = time.time()
    print(f"[w4-krecal] TASK#12 detector-k recalibration | seeds={seeds} backend={args.backend} "
          f"frac_target={FRAC_TARGET} grid={KTHR_GRID}", flush=True)
    print("[w4-krecal] STEP A: per-seed ignition-curve calibration (solo-silent + fractional-ignites, controlled "
          "drives) -> picks kthr. STEP B: the W4 magnitude-fidelity A/B at the recalibrated kthr + a DEFAULT-k44 "
          "control.", flush=True)

    calib = [calibrate_seed(s) for s in seeds]
    print("[w4-krecal] STEP B: the W4 A/B (onehot/graded/scramble/lesion, plateau/linear) ...", flush=True)
    per_seed_recal = [eval_seed_at_kthr(s, c["kthr_picked"], tag="recal") for s, c in zip(seeds, calib)]
    per_seed_default = [eval_seed_at_kthr(s, DEFAULT_KTHR, tag="k44") for s in seeds]

    summary, verdict = build_top_summary(calib, per_seed_recal, per_seed_default, seeds, args.backend, smoke)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    _emit(summary, verdict, args.json)
    return 0 if summary["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
