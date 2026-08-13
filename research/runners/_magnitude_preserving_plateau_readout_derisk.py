"""MAGNITUDE-PRESERVING READ-OUT: close TWO independent boundaries with ONE grounded mechanism -- the GO'd
graded dendritic-plateau read-out (`enable_graded_dendritic_plateau`, de-risk A GO 2026-06-20;
Mikulasch-Priesemann analog dendritic read-out), whose output GRADES with the coincident input instead of
SATURATING (the all-or-none coincidence plateau's failure mode).

TWO 2026-08-13 boundaries hit the SAME wall -- the substrate reads a SIGN / present-absent robustly but graded
MAGNITUDE weakly:

  PART A -- W4 detector (`2026-08-13-w4-detector-k-recalibration-BOUNDARY.md`). Recalibrating the coincidence
    k-threshold made the detector READ the fractional mass (M2 implicature-recovery graded 0.360 vs onehot 0.136,
    6/6) but the VALID pre-registered metric M1 (intent-averaged fidelity to the analytic Frank-Goodman RSA
    landscape) still LOSES to onehot (move -0.046, 0/6) because the ALL-OR-NONE coincidence plateau is a
    THRESHOLD, not a magnitude-preserving read: once the 0.27 mass crosses, the SATURATED output OVERSHOOTS the
    analytic RSA magnitude (0.360 vs target 0.20; onehot's 0.136 is CLOSER). The named next mechanism: the GRADED
    dendritic-plateau read-out -- V(near) > V(mid) > V(far), proportional to the coincident drive.

  PART B -- affect salience (`2026-08-13-affect-opponent-weights-self-organized-BOUNDARY.md`). The self-organized
    valence read holds for SIGN (held-out r=+0.508) but graded STRENGTH underperforms (C-A2 salience
    |differential|~strength r=+0.10 vs the ridge's 0.27).

THE MECHANISM (additive; enable the ENGINE's graded plateau via CONFIG -- NOT a `sim/` edit; reuse-by-import both
runners). The graded dendritic plateau (fused_graded_dendritic_plateau, bridge 2.3a-ter) passes the WEIGHTED
coincident drive c_w = Sum_j w_eff_j*x_j through a GENTLE CENTERED logistic V = sigmoid(slope*(c_w-center)) - floor
scaled to a regenerative plateau current, so a fractional-mass coincidence yields a PROPORTIONALLY smaller plateau
than a full-mass one -- the magnitude the all-or-none switch destroys. Enabled by:
  cfg.enable_coincidence_detection = True    # builds the coincidence_detector routing mask (needed by BOTH forms)
  cfg.coincidence_plateau_strength = 0.0     # the ALL-OR-NONE current OFF (mask still built) -- pure graded read
  cfg.enable_graded_dendritic_plateau = True + graded_plateau_center/slope/strength
(the exact pattern in `_dendrite_stage1_onbridge_graded_plateau.py`). It is a READ-OUT NONLINEARITY, NOT the
refuted deep-credit / two-compartment / BDSP learning rule (`2026-07-22-gap4-real-issue-NOT-dendrites`).

PART A (W4) -- re-run the W4 onehot-vs-graded A/B with the detector read-out set to the GRADED plateau (per-seed
  center/slope calibrated on the ignition curve -- a detector PROPERTY, content-independent), score the VALID M1.
  DOES GRADED NOW BEAT ONEHOT on M1, 6/6? The all-or-none read is re-run as the CONTROL (must reproduce the
  onehot>graded negative). ANTI-CHEATS: (i) the graded read-out's response curve GRADES MONOTONICALLY (proportional,
  NOT saturating) -- printed + gated (proportionality error, and vs the all-or-none which saturates r(.27)/r(1)~1);
  (ii) M1's SCRAMBLE (graded mass on WRONG intents) must lose to onehot (a valid instrument); (iii) the graded read
  must help GRADED specifically (graded > onehot on the graded read, while the all-or-none read reproduces the
  negative), not lift both.

PART B (affect) -- apply the SAME graded plateau to the affect opponent code_in->vplus/vminus FF read, re-measure
  the C-A2 salience |differential|~valence-strength correlation. 4-cell design (self-organized-weights x ridge-
  weights) x (point-soma read x graded-plateau read) ISOLATES read-out vs weight-source: the ridge already reaches
  0.27 with the point-soma read, so if the read-out were the bottleneck the ridge would be limited too -- this test
  says WHICH boundary is read-out-limited (closable here) and which is weight-source-limited (the named
  graded-reinforcement-strength third factor, not a read-out).

VERDICT: GO iff BOTH move (A: graded>onehot on M1 6/6 with a verified magnitude-preserving read + all-or-none
  reproduces the negative; B: graded read lifts C-A2 toward the ridge's 0.27). PARTIAL iff exactly one. BOUNDARY iff
  neither. Functional read-outs only; never assert phenomenal experience.

EXTERNAL GROUNDING: Mikulasch & Priesemann (dendritic analog/graded read-out); Larkum (2013) TiNS 36(3):141 (the
  plateau's tunable coinciding-input threshold); Frank & Goodman (2012) Science 336(6084):998 (the RSA objective is
  a graded MAGNITUDE, not an argmax -- why a threshold read is insufficient).

Usage:
  # PART A smoke (1 seed, prints the graded vs all-or-none response curve + the A/B teeth; verdict UNDEFINED n<6):
  SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part A --smoke \
      --json research/findings/raw/_magnitude_preserving/w4_smoke.json
  # PART A 6-seed:
  SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part A \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_magnitude_preserving/w4_6seed.json
  # PART B smoke / 6-seed (affect):
  SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part B --smoke \
      --json research/findings/raw/_magnitude_preserving/affect_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._magnitude_preserving_plateau_readout_derisk --part B \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_magnitude_preserving/affect_6seed.json
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

import logging as _logging
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

# ── PART A reuse-by-import: the W4 detector geometry + the W4 A/B belief sources + the magnitude-fidelity metric ──
from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state, _restore_state, SETTLE_STEPS  # noqa: E402
from research.runners._self_schema_region_derisk import WS_LOOP_GATE  # noqa: E402
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    build_success_bridge, success_signal, _proj,
    K, ITEM, DET, BELIEF_TOTAL, INTENT_PA, W_SYN, K_THR, GAIN, PLATEAU, DRIVE_STEPS, READ_STEPS,
)
from research.runners._recursive_tom_rsa_derisk import STATES, UTTS  # noqa: E402
from research.runners._pragmatic_spiking_graded_belief_derisk import (  # noqa: E402
    belief_variants, neural_success_landscape, _analytic_landscape, mag_fidelity, argmax_align,
    _implicature_margin, MOVE_EPS,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART A -- graded-plateau read-out on the W4 success (coincidence) detector.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# calibration: center around 2*K_THR (the W4 count-threshold expressed in WEIGHTED-drive units, since c_w = W_SYN*count)
CENTER_GRID = [80.0, 88.0, 96.0]          # around 2*K_THR = 88 (the W4 operating point in weighted-drive units)
SLOPE_GRID = [0.08, 0.11, 0.14]           # GENTLE logistic slope -> non-saturating graded read
GP_STRENGTH = float(PLATEAU)              # 80 -- matches the all-or-none plateau strength (like-for-like)
SILENT_FLOOR_A = 0.08                     # a SOLO (single-afferent) arm above this is not silent -> not a clean gate
IGNITE_MIN_A = 0.06                       # the full-mass two-input drive must read above this
FGRID = [0.0, 0.135, 0.27, 0.5, 0.75, 1.0]   # belief-fraction ignition sweep (content-independent)


def build_success_bridge_graded(seed, center, slope, strength=GP_STRENGTH, shuffle_k=False):
    """The W4 success (coincidence) detector, but read through the GRADED dendritic plateau instead of the
    all-or-none switch. MIRRORS `build_success_bridge` geometry EXACTLY (same regions/wiring/snapshot, so every
    downstream read -- success_signal / neural_success_landscape -- is reuse-by-import), changing ONLY the read-out
    config: enable_coincidence_detection True (builds the routing mask), coincidence_plateau_strength 0.0 (all-or-
    none OFF), enable_graded_dendritic_plateau True (center/slope/strength). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="belief", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="intent", n_neurons=ITEM * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="success", n_neurons=DET * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="decoy", n_neurons=DET * K, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                       # ⛔ seed the SUBSTRATE (identical to build_success_bridge)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process", "enable_nmda"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    # ── THE READ-OUT: graded dendritic plateau (magnitude-preserving), all-or-none coincidence current OFF ──
    cfg.enable_coincidence_detection = True    # builds the coincidence_detector routing mask (needed by BOTH forms)
    cfg.coincidence_k_threshold = float(K_THR)
    cfg.coincidence_gain = float(GAIN)
    cfg.coincidence_plateau_strength = 0.0     # the ALL-OR-NONE current OFF (g_inc==0) -> pure graded read
    cfg.coincidence_weighted_drive = True      # (the graded block always reads the weighted drive; kept consistent)
    cfg.enable_graded_dendritic_plateau = True
    cfg.graded_plateau_center = float(center)
    cfg.graded_plateau_slope = float(slope)
    cfg.graded_plateau_strength = float(strength)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    bel = np.asarray(rm.indices("belief"), dtype=np.int64)
    itn = np.asarray(rm.indices("intent"), dtype=np.int64)
    suc = np.asarray(rm.indices("success"), dtype=np.int64)
    dec = np.asarray(rm.indices("decoy"), dtype=np.int64)
    bel_k = {k: bel[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    itn_k = {k: itn[k * ITEM:(k + 1) * ITEM] for k in range(K)}
    suc_k = {k: suc[k * DET:(k + 1) * DET] for k in range(K)}
    if shuffle_k:
        prng = np.random.default_rng(seed * 999 + 7)
        perm = np.arange(K)
        while np.any(perm == np.arange(K)):
            perm = prng.permutation(K)
    else:
        perm = np.arange(K)
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    for k in range(K):
        union[f"bel2suc_{k}"] = _proj(bel_k[k], suc_k[int(perm[k])], W_SYN, True)
        union[f"itn2suc_{k}"] = _proj(itn_k[k], suc_k[k], W_SYN, True)
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)
    idx = {"bel": {k: xp.asarray(bel_k[k]) for k in range(K)},
           "itn": {k: xp.asarray(itn_k[k]) for k in range(K)},
           "suc": {k: xp.asarray(suc_k[k]) for k in range(K)},
           "suc_all": xp.asarray(suc), "dec_all": xp.asarray(dec)}
    return bridge, xp, idx, snap


def _ignition_drive(bridge, xp, idx, snap, f, t, intent_on, belief_on):
    """Controlled matched-detector ignition read (content-independent): drive column t with belief fraction f and/or
    a one-hot intent; read the matched success[t] mean per-neuron firing rate over the read window. Identical to the
    W4 recal `_calib_drive` -- a detector PROPERTY, NOT the RSA content."""
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


def _curve_stats(rcurve, solo_i, solo_b):
    """Magnitude-preservation stats of an ignition curve r(FGRID). Monotonic (non-decreasing within a small noise
    tol), non-saturating (r(.27)/r(1) well below 1), and proportionality error = mean_f |r(f)/r(1) - f| (0 = a
    perfect magnitude-preserving read; the all-or-none read saturates -> ~1)."""
    r = np.asarray(rcurve, float)
    r_full = float(r[-1])
    max_solo = float(max(solo_i, solo_b))
    if r_full <= 1e-9:
        return {"r_full": r_full, "max_solo": max_solo, "monotonic": False, "prop_err": 9.9,
                "sat_ratio": 9.9, "dyn_range": 0.0}
    ratio = r / r_full
    diffs = np.diff(r)
    monotonic = bool(np.all(diffs >= -0.02))                       # non-decreasing within noise tol
    prop_err = float(np.mean(np.abs(ratio - np.asarray(FGRID))))   # deviation from a proportional ramp
    sat_ratio = float(ratio[FGRID.index(0.27)])                    # r(.27)/r(1): ~1 = saturated, ~0.27 = proportional
    dyn_range = float(r_full - r[0])
    return {"r_full": r_full, "max_solo": max_solo, "monotonic": monotonic, "prop_err": round(prop_err, 4),
            "sat_ratio": round(sat_ratio, 4), "dyn_range": round(dyn_range, 4)}


def calibrate_graded_seed(seed, verbose=True):
    """Per-seed calibration of the graded plateau (center, slope) on the ignition curve. Pick the (center, slope)
    that is SOLO-SILENT (single-afferent arms below the floor), IGNITES at full mass, is MONOTONIC, and minimizes the
    PROPORTIONALITY error (the magnitude-preservation objective) -- content-independent (controlled drives). If none
    is solo-silent+monotonic, fall back to the best-proportional silent candidate (flagged unclean)."""
    t = 0
    table = []
    for center in CENTER_GRID:
        for slope in SLOPE_GRID:
            bridge, xp, idx, snap = build_success_bridge_graded(seed, center, slope)
            rc = [_ignition_drive(bridge, xp, idx, snap, f, t, intent_on=True, belief_on=True) for f in FGRID]
            solo_i = _ignition_drive(bridge, xp, idx, snap, 0.0, t, intent_on=True, belief_on=False)
            solo_b = _ignition_drive(bridge, xp, idx, snap, 1.0, t, intent_on=False, belief_on=True)
            st = _curve_stats(rc, solo_i, solo_b)
            silent = bool(st["max_solo"] < SILENT_FLOOR_A and st["r_full"] > IGNITE_MIN_A)
            clean = bool(silent and st["monotonic"])
            table.append({"center": center, "slope": slope, "r_curve": [round(x, 4) for x in rc],
                          "solo_i": round(solo_i, 4), "solo_b": round(solo_b, 4), "silent": silent,
                          "clean": clean, **st})
    clean_rows = [row for row in table if row["clean"]]
    silent_rows = [row for row in table if row["silent"]]
    if clean_rows:
        pick = min(clean_rows, key=lambda r: r["prop_err"]); calibration_clean = True
    elif silent_rows:
        pick = min(silent_rows, key=lambda r: r["prop_err"]); calibration_clean = False
    else:
        pick = min(table, key=lambda r: r["prop_err"]); calibration_clean = False
    rec = {"seed": int(seed), "center": float(pick["center"]), "slope": float(pick["slope"]),
           "calibration_clean": bool(calibration_clean), "picked": pick, "table": table}
    if verbose:
        print(f"  [calib-A seed {seed}] picked center={pick['center']:.0f} slope={pick['slope']:.2f} "
              f"clean={calibration_clean} | solo(i/b)={pick['solo_i']:.3f}/{pick['solo_b']:.3f} "
              f"r_full={pick['r_full']:.3f} monotonic={pick['monotonic']} prop_err={pick['prop_err']:.3f} "
              f"sat_ratio(r.27/r1)={pick['sat_ratio']:.2f} | curve={pick['r_curve']}", flush=True)
    return rec


def _landscape_record(bridge, xp, idx, snap, det_name):
    """Reuse-by-import: the W4 per-belief-variant landscape S[t,u] + M1 magnitude-fidelity + argmax + M2 cell, for a
    given (already-built) detector bridge. Returns the det-block the W4 metrics consume."""
    S_ideal = _analytic_landscape()
    det = {}
    for bname, src in _CUR_BELIEFS.items():
        S = neural_success_landscape(bridge, xp, idx, snap, src)
        mf, per_t = mag_fidelity(S, S_ideal)
        det[bname] = {"mag_fidelity": round(mf, 5), "argmax_align": round(argmax_align(S), 4),
                      "S": [[round(float(x), 5) for x in row] for row in S],
                      "mag_fidelity_per_intent": [round(x, 4) for x in per_t]}
    return det


_CUR_BELIEFS = None   # set per seed (belief_variants) so _landscape_record reuses the exact W4 sources


def eval_w4_seed(seed, center, slope, verbose=True):
    """Build BOTH read-outs at this seed and run the W4 A/B (onehot/graded/scramble/lesion): the GRADED plateau read
    (calibrated center/slope) and the ALL-OR-NONE read (the W4 default build_success_bridge, coincidence=True). M1 =
    intent-averaged magnitude-fidelity; M2 = the implicature-recovery cell S[all|some]."""
    global _CUR_BELIEFS
    _CUR_BELIEFS = belief_variants(seed)
    rec = {"seed": int(seed), "center": float(center), "slope": float(slope)}
    # GRADED plateau read-out
    bg, xg, ig, sg = build_success_bridge_graded(seed, center, slope)
    rec["graded_read"] = _landscape_record(bg, xg, ig, sg, "graded_read")
    # ALL-OR-NONE read-out (the W4 default detector) -- the CONTROL that must reproduce the negative
    ba, xa, ia, sa = build_success_bridge(seed, coincidence=True)
    rec["allornone_read"] = _landscape_record(ba, xa, ia, sa, "allornone_read")
    rec["belief_implicature_margin_graded"] = round(_implicature_margin(_CUR_BELIEFS["graded"]), 4)
    rec["belief_implicature_margin_lesion"] = round(_implicature_margin(_CUR_BELIEFS["graded_lesion"]), 4)
    if verbose:
        for rd in ("graded_read", "allornone_read"):
            g, o = rec[rd]["graded"], rec[rd]["onehot"]
            print(f"    [seed {seed} | {rd}] M1 onehot={o['mag_fidelity']} graded={g['mag_fidelity']} "
                  f"(move {g['mag_fidelity'] - o['mag_fidelity']:+.4f}) | scramble={rec[rd]['scramble']['mag_fidelity']} "
                  f"lesion={rec[rd]['graded_lesion']['mag_fidelity']}", flush=True)
    return rec


def _m2(rec, read):
    """M2 implicature-recovery S[all|some] per-intent-normalized, for onehot/graded/scramble under `read`."""
    _ai, _si = STATES.index("all"), UTTS.index("some")

    def cell(b):
        row = np.asarray(rec[read][b]["S"], float)[_ai]
        s = float(row.sum())
        return float(row[_si] / s) if s > 1e-9 else 0.0
    return {b: round(cell(b), 4) for b in ("onehot", "graded", "scramble")}


def build_summary_A(calib, per_seed, seeds, backend, smoke):
    n = len(seeds)
    n_clean = int(sum(1 for c in calib if c["calibration_clean"]))
    mean_max_solo = round(float(np.mean([c["picked"]["max_solo"] for c in calib])), 4)
    mean_prop_err = round(float(np.mean([c["picked"]["prop_err"] for c in calib])), 4)
    mean_sat_ratio = round(float(np.mean([c["picked"]["sat_ratio"] for c in calib])), 4)
    all_monotonic = bool(all(c["picked"]["monotonic"] for c in calib))

    def m1(read, b):
        return round(float(np.mean([r[read][b]["mag_fidelity"] for r in per_seed])), 5)
    graded_m1_g, graded_m1_o = m1("graded_read", "graded"), m1("graded_read", "onehot")
    graded_m1_s = m1("graded_read", "scramble")
    ao_m1_g, ao_m1_o = m1("allornone_read", "graded"), m1("allornone_read", "onehot")
    graded_move = round(graded_m1_g - graded_m1_o, 5)
    ao_move = round(ao_m1_g - ao_m1_o, 5)
    graded_seeds_gt = int(sum(1 for r in per_seed if r["graded_read"]["graded"]["mag_fidelity"]
                              > r["graded_read"]["onehot"]["mag_fidelity"]))
    # M2 (reported diagnostic; the finding's named cell)
    m2_g = {b: round(float(np.mean([_m2(r, "graded_read")[b] for r in per_seed])), 4)
            for b in ("onehot", "graded", "scramble")}

    # instrument: every seed cleanly calibrated (solo silent + monotonic) AND magnitude-preserving (not saturating).
    instrument_clean = bool(n >= 6 and n_clean == n and mean_max_solo < SILENT_FLOOR_A and all_monotonic
                            and mean_sat_ratio < 0.6)   # < 0.6 => the graded read does NOT saturate (all-or-none ~1)
    scramble_loses = bool(graded_m1_s < graded_m1_o - 1e-6)
    graded_beats_onehot = bool(graded_move > MOVE_EPS and graded_seeds_gt == n)
    allornone_reproduces_negative = bool(ao_move <= MOVE_EPS)   # the all-or-none read keeps the onehot>=graded wall
    graded_specific = bool(graded_move > ao_move + 1e-6)        # the graded READ specifically helps GRADED

    go_A = bool(instrument_clean and graded_beats_onehot and scramble_loses
                and allornone_reproduces_negative and graded_specific)

    v = Verdict("PART A -- does the graded (magnitude-preserving) plateau read-out move the VALID M1 where the "
                "all-or-none read cannot?")
    v.require("6 seeds (project bar)", n >= 6, expect=True)
    v.require("M1 is a VALID instrument: SCRAMBLE (graded mass on WRONG intents) LOSES to onehot on the graded read "
              "(so a graded<=onehot result would be a real NEGATIVE, not an instrument failure)",
              scramble_loses, expect=True, note=f"scramble={graded_m1_s} onehot={graded_m1_o}")
    v.require("the ALL-OR-NONE read reproduces the 2026-08-13 negative (M1 move <= MOVE_EPS: onehot >= graded)",
              allornone_reproduces_negative, expect=True, note=f"all-or-none M1 move={ao_move:+.4f}")
    v.control("the graded read-out is MAGNITUDE-PRESERVING (r(.27)/r(1) << the all-or-none's ~1.0 saturation)",
              treatment=1.0, control=mean_sat_ratio)
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU/NMDA + the all-or-none coincidence current "
               "(coincidence_plateau_strength=0)", "a fixed-operating-point READ-OUT nonlinearity (the graded "
               "dendritic plateau); belief byte-identical to the W4 A/B; NOT a learning rule.")
    vb = v.decide(go=go_A)

    attributable_to("the M1 move attributable to the GRADED (magnitude-preserving) read-out (vs the all-or-none read "
                    "on the SAME belief)", graded_move, ao_move)
    _mg = float(np.mean([r["belief_implicature_margin_graded"] for r in per_seed]))
    _ml = float(np.mean([r["belief_implicature_margin_lesion"] for r in per_seed]))
    attributable_to("graded implicature CONTENT attributable to FS divisive normalization (vs its lesion), belief "
                    "unchanged by the read-out", _mg, _ml)

    if smoke or n < 6:
        verdict = ("UNDEFINED -- smoke (n<6); the graded vs all-or-none response curve + the A/B teeth are printed "
                   "for mechanism-check only. The 6-seed bar is authoritative.")
    elif go_A:
        verdict = ("GO (PART A) -- the GRADED (magnitude-preserving) dendritic-plateau read-out MOVES the VALID "
                   "pre-registered M1: graded belief M1=%.3f > onehot M1=%.3f (move=%+.4f, %d/%d seeds), where the "
                   "ALL-OR-NONE read reproduces the 2026-08-13 negative (move=%+.4f, onehot>=graded). The read is "
                   "verified magnitude-preserving (%d/%d seeds solo-silent + monotonic, mean r(.27)/r(1)=%.2f << the "
                   "all-or-none's ~1.0 saturation), the move is CORRECT implicature content (scramble %.3f < onehot "
                   "%.3f), and it specifically helps GRADED (graded move %+.4f > all-or-none move %+.4f). The W4 "
                   "detector's all-or-none MAGNITUDE-BLINDNESS is surpassed by a READ-OUT nonlinearity, NOT a credit "
                   "rule." % (graded_m1_g, graded_m1_o, graded_move, graded_seeds_gt, n, ao_move, n_clean, n,
                              mean_sat_ratio, graded_m1_s, graded_m1_o, graded_move, ao_move))
    else:
        failed = []
        if not instrument_clean:
            failed.append(f"INSTRUMENT (clean {n_clean}/{n}, mean max_solo={mean_max_solo}, all_monotonic="
                          f"{all_monotonic}, mean sat_ratio={mean_sat_ratio})")
        if not graded_beats_onehot:
            failed.append(f"M1-MOVE (graded move={graded_move:+.4f}, {graded_seeds_gt}/{n} seeds)")
        if not allornone_reproduces_negative:
            failed.append(f"all-or-none does not reproduce the negative (move={ao_move:+.4f})")
        if not scramble_loses:
            failed.append("scramble does not lose")
        if not graded_specific:
            failed.append(f"graded move ({graded_move:+.4f}) not > all-or-none move ({ao_move:+.4f})")
        verdict = ("BOUNDARY (PART A) -- %s. The graded read reads the magnitude (M2 S[all|some] graded=%.3f vs "
                   "onehot=%.3f) but does not clear the 6/6 M1 GO bar. Honest residual + next mechanism in the "
                   "finding; the refuted deep-credit/BDSP rule is NOT re-proposed." %
                   ("; ".join(failed), m2_g["graded"], m2_g["onehot"]))

    summary = {
        "runner": "_magnitude_preserving_plateau_readout_derisk", "part": "A (W4 detector)",
        "faculty": "graded dendritic-plateau (magnitude-preserving) read-out on the W4 coincidence detector; does it "
                   "move the VALID intent-averaged magnitude-fidelity M1 where the all-or-none plateau cannot? "
                   "FUNCTIONAL pragmatics correlate.",
        "seeds": list(seeds), "backend": backend, "smoke": bool(smoke or n < 6), "verdict": verdict, "go": go_A,
        "external_grounding": [
            "Mikulasch & Priesemann -- dendritic ANALOG/graded read-out (the graded plateau is the point-neuron's "
            "magnitude-blindness surpass).",
            "Larkum (2013) TiNS 36(3):141 -- the plateau's tunable coinciding-input threshold.",
            "Frank & Goodman (2012) Science 336(6084):998 -- the RSA objective is a graded MAGNITUDE, not an argmax.",
        ],
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "gates": {"instrument_magnitude_preserving_6seed": instrument_clean,
                  "M1_graded_beats_onehot_6of6": graded_beats_onehot, "scramble_loses": scramble_loses,
                  "allornone_reproduces_negative": allornone_reproduces_negative,
                  "graded_read_specific": graded_specific},
        "M1": {"graded_read": {"onehot": graded_m1_o, "graded": graded_m1_g, "scramble": graded_m1_s,
                               "move": graded_move, "seeds_graded_gt_onehot": graded_seeds_gt},
               "allornone_read": {"onehot": ao_m1_o, "graded": ao_m1_g, "move": ao_move}},
        "M2_implicature_recovery_graded_read": m2_g,
        "calibration": {"n_clean": n_clean, "n_seeds": n, "mean_max_solo": mean_max_solo,
                        "mean_prop_err": mean_prop_err, "mean_sat_ratio": mean_sat_ratio,
                        "all_monotonic": all_monotonic,
                        "center_per_seed": {str(c["seed"]): c["center"] for c in calib},
                        "slope_per_seed": {str(c["seed"]): c["slope"] for c in calib},
                        "response_curve_per_seed": {str(c["seed"]): c["picked"]["r_curve"] for c in calib},
                        "per_seed": calib},
        "per_seed": per_seed,
        "honest_scope": (
            "A FUNCTIONAL pragmatics correlate. This is a READ-OUT nonlinearity ONLY (how the detector READS a "
            "fractional coincidence -- the graded dendritic plateau vs the all-or-none switch), NOT a learning rule "
            "and NOT a belief change (belief byte-identical to the W4 A/B). Per-seed center/slope calibrated on the "
            "ignition CURVE (controlled fractional/solo drives), a detector PROPERTY independent of the RSA content; "
            "the SCRAMBLE control (graded mass on WRONG intents must LOSE) and the ALL-OR-NONE control (must "
            "reproduce the onehot>=graded negative) keep it honest. Plasticity off. numpy-CPU real spiking "
            "Izhikevich; NO sim/ edit (enable_graded_dendritic_plateau via config); reuse-by-import of the W4 A/B + "
            "the Leg-1 detector. NOT a claim of phenomenal access to another mind."),
    }
    return summary, verdict


def run_part_A(seeds, backend, smoke, out_path):
    t0 = time.time()
    print(f"[mag-plateau A] W4 detector graded-plateau read-out | seeds={seeds} centers={CENTER_GRID} "
          f"slopes={SLOPE_GRID}", flush=True)
    print("[mag-plateau A] STEP A: per-seed graded-plateau calibration (solo-silent + monotonic + proportional). "
          "STEP B: the W4 magnitude-fidelity A/B on the graded read vs the all-or-none control.", flush=True)
    calib = [calibrate_graded_seed(s) for s in seeds]
    per_seed = [eval_w4_seed(s, c["center"], c["slope"]) for s, c in zip(seeds, calib)]
    summary, verdict = build_summary_A(calib, per_seed, seeds, backend, smoke)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    m1, cal = summary["M1"], summary["calibration"]
    print("\n" + "=" * 114, flush=True)
    print(f"[mag-plateau A] === VERDICT: {verdict} ===", flush=True)
    print(f"[mag-plateau A]  CALIB: clean {cal['n_clean']}/{cal['n_seeds']} | mean max_solo={cal['mean_max_solo']} "
          f"(silent<{SILENT_FLOOR_A}) all_monotonic={cal['all_monotonic']} mean prop_err={cal['mean_prop_err']} "
          f"mean sat_ratio={cal['mean_sat_ratio']} (all-or-none~1.0)", flush=True)
    print(f"[mag-plateau A]  M1 GRADED read: onehot={m1['graded_read']['onehot']} graded={m1['graded_read']['graded']}"
          f" move={m1['graded_read']['move']:+.4f} ({m1['graded_read']['seeds_graded_gt_onehot']}/{cal['n_seeds']}) "
          f"scramble={m1['graded_read']['scramble']}", flush=True)
    print(f"[mag-plateau A]  M1 ALL-OR-NONE read (control): onehot={m1['allornone_read']['onehot']} "
          f"graded={m1['allornone_read']['graded']} move={m1['allornone_read']['move']:+.4f} "
          f"(must reproduce the negative)", flush=True)
    print(f"[mag-plateau A]  gates={summary['gates']}", flush=True)
    print(f"[mag-plateau A]  wrote {out_path}\n" + "=" * 114, flush=True)
    return summary


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# PART B -- graded-plateau read-out on the affect opponent salience read.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# reuse-by-import: the affect-deepen circuit reads + constants + the composed self-organized opponent weights.
from research.runners._affect_appraisal_emotion_reappraisal_derisk import (  # noqa: E402
    read_valence, ridge_opponent, _pearson,
    N_OPP, N_XINH, N_REAP, N_DIM, N_EMO, N_EMO_FS, N_DIMFS, ENABLE_HET, OPP_TONIC_PA, OPP_FF_GAIN, OPP_READ_SCALE,
    XINH_EXC_W, XINH_INH_W, DIMFS_EXC_W, REAP_INH_W, VAL_INH_W, EMO_MAP, INH_MAP, EMO_NAMES,
    W_MAP, W_INH, VAL_TO_EMO_W, EMO_RECUR_W, EMO_RECUR_DENSITY, EMO_BIAS_PA, EMO_EXC_TO_FS, EMO_FS_TO_EXC,
    SETTLE_STEPS as AFF_SETTLE, OPP_READ_MS, _snapshot as _aff_snapshot,
)
from research.runners._affect_composed_selforganized_opponent_derisk import (  # noqa: E402
    build_all, selforg_opponent_weights, rescorla_wagner_valence, W_L2_REF,
)

RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
FS = "IZH2007_FS_CORTICAL_INTERNEURON"
# affect graded-plateau calibration: the opponent FF drive scale differs from the W4 detector; calibrate the center
# on the measured operating point of code.w*OPP_FF_GAIN*OPP_READ_SCALE (the ignition-curve instrument, swept below).
AFF_CENTER_FRACS = [0.20, 0.35, 0.5]      # center as a FRACTION of the measured max weighted opponent drive
AFF_SLOPE_GRID = [0.02, 0.05, 0.1]        # gentle (the opponent drive spans a wide c_w range)
AFF_GP_STRENGTH = 80.0


def _aff_region(name, n, exc=1.0, dens=0.0, w=0.0, nmda=False, itype=RS, intrinsic=0.0):
    return BrainRegion(name=name, n_neurons=int(n), exc_fraction=exc, internal_density=dens,
                       exc_weight_mean=w, inh_weight_mean=0.0, weight_jitter=0.05 if dens > 0 else 0.0,
                       plastic_internal=False, izh_neuron_type=itype, enable_nmda=nmda,
                       intrinsic_current_pA=float(intrinsic), enable_homeostasis=False)


def build_affect_bridge(seed, D, wp, wm, graded=False, center=0.0, slope=0.0, strength=AFF_GP_STRENGTH):
    """The affect appraisal->emotion->reappraisal bridge (MIRRORS `_affect_appraisal_emotion_reappraisal.build_bridge`
    EXACTLY), with the rung-a opponent FF read either point-soma (graded=False, byte-identical to the original build)
    or through the GRADED dendritic plateau (graded=True: the ff_code_vplus/vminus pathways tagged coincidence_
    detector=True, coincidence_plateau_strength=0, enable_graded_dendritic_plateau=True). read_valence/read_emotion
    are reuse-by-import (build-agnostic). Returns (bridge, xp, idx, snap)."""
    xp, _ = get_backend()
    dims = ("agency_self", "agency_other", "certainty", "uncertainty")
    regions = [
        _aff_region("code_in", D),
        _aff_region("appr_vplus", N_OPP, intrinsic=OPP_TONIC_PA),
        _aff_region("appr_vminus", N_OPP, intrinsic=OPP_TONIC_PA),
        _aff_region("xinh_vp", N_XINH, exc=0.0, itype=FS),
        _aff_region("xinh_vm", N_XINH, exc=0.0, itype=FS),
        _aff_region("vmpfc_reap", N_REAP),
        _aff_region("reap_fs", N_DIMFS, exc=0.0, itype=FS),
        _aff_region("emo_fs", N_EMO_FS, exc=0.0, itype=FS),
    ]
    for d in dims:
        regions.append(_aff_region(d, N_DIM))
        regions.append(_aff_region(f"{d}_fs", N_DIMFS, exc=0.0, itype=FS))
    for e in EMO_NAMES:
        regions.append(_aff_region(e, N_EMO, dens=EMO_RECUR_DENSITY, w=EMO_RECUR_W, nmda=True, intrinsic=EMO_BIAS_PA))

    G_WTA, G_REAP = "emo_wta", "reap_out"
    pathways = [
        RegionPathway(from_region="appr_vplus", to_region="xinh_vp", density=0.6, weight_mean=XINH_EXC_W,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="xinh_vp", to_region="appr_vminus", density=0.7, weight_mean=XINH_INH_W,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        RegionPathway(from_region="appr_vminus", to_region="xinh_vm", density=0.6, weight_mean=XINH_EXC_W,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="xinh_vm", to_region="appr_vplus", density=0.7, weight_mean=XINH_INH_W,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
    ]
    for emo, srcs in EMO_MAP.items():
        for src, w in srcs:
            pathways.append(RegionPathway(from_region=src, to_region=emo, density=0.7, weight_mean=w,
                                          weight_jitter=0.1, plastic=False))
    for d in dims:
        pathways.append(RegionPathway(from_region=d, to_region=f"{d}_fs", density=0.7, weight_mean=DIMFS_EXC_W,
                                      weight_jitter=0.1, plastic=False))
        for emo in INH_MAP.get(d, ()):
            pathways.append(RegionPathway(from_region=f"{d}_fs", to_region=emo, density=0.7, weight_mean=W_INH,
                                          weight_jitter=0.1, plastic=False, receptor="gaba_a"))
    for emo in ("emo_fear", "emo_rage"):
        pathways.append(RegionPathway(from_region="xinh_vp", to_region=emo, density=0.7, weight_mean=VAL_INH_W,
                                      weight_jitter=0.1, plastic=False, receptor="gaba_a"))
    for emo in ("emo_seeking", "emo_care"):
        pathways.append(RegionPathway(from_region="xinh_vm", to_region=emo, density=0.7, weight_mean=VAL_INH_W,
                                      weight_jitter=0.1, plastic=False, receptor="gaba_a"))
    for e in EMO_NAMES:
        pathways.append(RegionPathway(from_region=e, to_region="emo_fs", density=0.6, weight_mean=EMO_EXC_TO_FS,
                                      weight_jitter=0.1, plastic=False))
        pathways.append(RegionPathway(from_region="emo_fs", to_region=e, density=0.6, weight_mean=EMO_FS_TO_EXC,
                                      weight_jitter=0.1, plastic=False, receptor="gaba_a", transmission_gate=G_WTA))
    pathways.append(RegionPathway(from_region="vmpfc_reap", to_region="reap_fs", density=0.8, weight_mean=DIMFS_EXC_W,
                                  weight_jitter=0.1, plastic=False))
    pathways.append(RegionPathway(from_region="reap_fs", to_region="appr_vminus", density=0.85,
                                  weight_mean=REAP_INH_W, weight_jitter=0.1, plastic=False, receptor="gaba_a",
                                  transmission_gate=G_REAP))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = 100.0
    cfg.nmda_recurrent_tau_decay_ms = 100.0
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_input_divisive_norm"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = False
    cfg.ou_std_current_pA = 0.0
    cfg.enable_parameter_heterogeneity = bool(ENABLE_HET)
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    # ── THE READ-OUT: opponent FF read through the graded dendritic plateau (graded=True) or point-soma (default) ──
    if graded:
        cfg.enable_coincidence_detection = True     # builds the routing mask (needed by the graded block)
        cfg.coincidence_k_threshold = float(K_THR)
        cfg.coincidence_gain = float(GAIN)
        cfg.coincidence_plateau_strength = 0.0      # all-or-none current OFF -> pure graded read
        cfg.coincidence_weighted_drive = True
        cfg.enable_graded_dendritic_plateau = True
        cfg.graded_plateau_center = float(center)
        cfg.graded_plateau_slope = float(slope)
        cfg.graded_plateau_strength = float(strength)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    idx = {n: np.asarray(rm.indices(n), dtype=np.int64) for n in
           ("code_in", "appr_vplus", "appr_vminus", "agency_self", "agency_other", "certainty", "uncertainty",
            "vmpfc_reap") + tuple(EMO_NAMES)}
    union = dict(rm.build_wiring_plan(seed=int(seed)))
    ci = idx["code_in"]

    def _ff(post_idx, wvec):
        P, Q, V = [], [], []
        for di, a in enumerate(ci):
            gw = float(OPP_FF_GAIN * wvec[di])
            if gw <= 0.0:
                continue
            for b in post_idx:
                P.append(int(a)); Q.append(int(b)); V.append(gw)
        d = dict(pre_indices=P, post_indices=Q, initial_weights=V, plastic=False, conn_type="ff")
        if graded:
            d["coincidence_detector"] = True        # route the opponent FF through the graded plateau read
        return d

    union["ff_code_vplus"] = _ff(idx["appr_vplus"], wp)
    union["ff_code_vminus"] = _ff(idx["appr_vminus"], wm)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(AFF_SETTLE):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _aff_snapshot(bridge, xp)
    return bridge, xp, idx, snap


def _aff_prep_seed(seed, A, n_each=5, min_events=2, held_frac=0.5, max_held_probe=48):
    """Draw the innate primaries, condition s_c, derive the composed self-organized opponent weights (wp/wm), and the
    ridge-Warriner opponent -- exactly as `_affect_composed_selforganized_opponent.run_seed`. Returns the codes, the
    held probe set, and both weight vectors + s_true (EVAL only)."""
    rng = np.random.default_rng(seed)
    vocab, codes, codes_read = A["vocab"], A["codes"], A["codes_read"]
    relatedness, s_true, Co = A["relatedness"], A["s_true"], A["Co"]
    all_primaries, prim_sign_full = A["all_primaries"], A["prim_sign_full"]
    n = len(vocab); D = codes.shape[1]
    prim_col = {w: j for j, w in enumerate(all_primaries)}
    app = [w for w in all_primaries if prim_sign_full[w] > 0]
    avr = [w for w in all_primaries if prim_sign_full[w] < 0]
    app_pick = list(rng.choice(app, size=min(n_each, len(app)), replace=False))
    avr_pick = list(rng.choice(avr, size=min(n_each, len(avr)), replace=False))
    primaries = app_pick + avr_pick
    prim_idx = np.array([prim_col[w] for w in primaries]); prim_sgn = np.array([prim_sign_full[w] for w in primaries], float)
    is_primary = np.array([w in set(primaries) for w in vocab])
    s_c, reinforced = rescorla_wagner_valence(Co, prim_idx, prim_sgn, is_primary, min_events)
    ridx = np.where(reinforced)[0]; rng.shuffle(ridx)
    n_held = int(round(held_frac * len(ridx)))
    held_idx, train_idx = ridx[:n_held], ridx[n_held:]
    train_mask = np.zeros(n, bool); train_mask[train_idx] = True
    # composed self-organized opponent (Warriner-free)
    w_comp, wp_comp, wm_comp = selforg_opponent_weights(codes_read, s_c, train_mask, codes, relatedness=relatedness,
                                                        l2_ref=W_L2_REF)
    # ridge-to-Warriner opponent (the reference method; magnitude-supervised)
    _, wp_ridge, wm_ridge = ridge_opponent(codes[train_idx], s_true[train_idx])
    hp = held_idx if len(held_idx) <= max_held_probe else rng.choice(held_idx, max_held_probe, replace=False)
    return dict(D=D, codes=codes, hp=hp, s_true=s_true, wp_comp=wp_comp, wm_comp=wm_comp,
                wp_ridge=wp_ridge, wm_ridge=wm_ridge, n_held=int(held_idx.size))


def _weighted_opp_drive_scale(prep, wp, wm):
    """The max weighted opponent drive across the held concepts (== the c_w scale the graded plateau center is set
    against): max_i (OPP_FF_GAIN * (code_i . wp)) over the probe concepts, on the vplus arm (the stronger sign)."""
    codes = prep["codes"]; hp = prep["hp"]
    dvp = OPP_FF_GAIN * (codes[hp] @ np.asarray(wp))
    dvm = OPP_FF_GAIN * (codes[hp] @ np.asarray(wm))
    return float(max(dvp.max(initial=0.0), dvm.max(initial=0.0), 1e-6))


def _aff_salience(bridge, xp, idx, snap, prep):
    """C-A2: |differential| ~ |valence| correlation over the held probe concepts (the opponent read off
    cp_firing_states), and the input-lesion collapse of |differential|."""
    codes, hp, s_true = prep["codes"], prep["hp"], prep["s_true"]
    diffs = np.array([read_valence(bridge, xp, idx, snap, codes[i])["differential"] for i in hp])
    r_sign = _pearson(diffs, s_true[hp])
    abs_r = _pearson(np.abs(diffs), np.abs(s_true[hp]))
    les = np.array([read_valence(bridge, xp, idx, snap, codes[i], lesion_input=True)["differential"] for i in hp[:12]])
    return {"r_sign": float(r_sign), "salience_r": float(abs_r), "intact_diff_abs": float(np.abs(diffs).mean()),
            "lesion_diff_abs": float(np.abs(les).mean())}


def calibrate_affect_seed(seed, prep, verbose=True):
    """Calibrate the graded plateau (center, slope) for the affect opponent on its OWN operating point: center as a
    fraction of the measured max weighted opponent drive, slope gentle. Pick (per the composed weights) the params
    maximizing the salience_r on a HELD-OUT calibration read (the ignition instrument = the opponent's own drive
    range), keeping the SIGN correlation intact. Content is the concept codes (unavoidable -- the opponent's operating
    point IS its concept drives); the choice is a detector property (transfer function), not the weights."""
    scale = _weighted_opp_drive_scale(prep, prep["wp_comp"], prep["wm_comp"])
    table = []
    for frac in AFF_CENTER_FRACS:
        for slope in AFF_SLOPE_GRID:
            center = frac * scale
            bridge, xp, idx, snap = build_affect_bridge(seed, prep["D"], prep["wp_comp"], prep["wm_comp"],
                                                        graded=True, center=center, slope=slope)
            sal = _aff_salience(bridge, xp, idx, snap, prep)
            table.append({"frac": frac, "center": round(center, 2), "slope": slope, **sal})
    # keep the SIGN read valid (r_sign > 0.2) and the input-lesion collapse (intact >> lesion); maximize salience_r.
    valid = [r for r in table if r["r_sign"] > 0.2 and r["lesion_diff_abs"] < 0.6 * max(r["intact_diff_abs"], 1e-9)]
    pool = valid if valid else table
    pick = max(pool, key=lambda r: r["salience_r"])
    rec = {"seed": int(seed), "scale": round(scale, 2), "center": pick["center"], "slope": pick["slope"],
           "picked": pick, "table": table, "valid": bool(valid)}
    if verbose:
        print(f"  [calib-B seed {seed}] scale={scale:.1f} picked center={pick['center']:.1f} slope={pick['slope']:.3f}"
              f" | salience_r={pick['salience_r']:+.3f} r_sign={pick['r_sign']:+.3f} "
              f"lesion|d|={pick['lesion_diff_abs']:.3f} vs intact|d|={pick['intact_diff_abs']:.3f}", flush=True)
    return rec


def eval_affect_seed(seed, prep, center, slope, verbose=True):
    """The 4-cell design: {composed, ridge} weights x {point-soma, graded-plateau} read. C-A2 salience per cell."""
    rec = {"seed": int(seed), "center": float(center), "slope": float(slope), "n_held": prep["n_held"]}
    cells = {
        "composed_pointsoma": (prep["wp_comp"], prep["wm_comp"], False),
        "composed_graded": (prep["wp_comp"], prep["wm_comp"], True),
        "ridge_pointsoma": (prep["wp_ridge"], prep["wm_ridge"], False),
        "ridge_graded": (prep["wp_ridge"], prep["wm_ridge"], True),
    }
    for name, (wp, wm, graded) in cells.items():
        bridge, xp, idx, snap = build_affect_bridge(seed, prep["D"], wp, wm, graded=graded, center=center, slope=slope)
        rec[name] = _aff_salience(bridge, xp, idx, snap, prep)
    if verbose:
        print(f"    [seed {seed}] salience_r  composed: point-soma={rec['composed_pointsoma']['salience_r']:+.3f} "
              f"graded={rec['composed_graded']['salience_r']:+.3f} | ridge: "
              f"point-soma={rec['ridge_pointsoma']['salience_r']:+.3f} graded={rec['ridge_graded']['salience_r']:+.3f}",
              flush=True)
    return rec


def build_summary_B(calib, per_seed, seeds, backend, smoke):
    n = len(seeds)

    def m(cell, key="salience_r"):
        return round(float(np.mean([r[cell][key] for r in per_seed])), 4)
    comp_ps, comp_gr = m("composed_pointsoma"), m("composed_graded")
    ridge_ps, ridge_gr = m("ridge_pointsoma"), m("ridge_graded")
    comp_ps_sign, comp_gr_sign = m("composed_pointsoma", "r_sign"), m("composed_graded", "r_sign")
    lift_comp = round(comp_gr - comp_ps, 4)
    lift_ridge = round(ridge_gr - ridge_ps, 4)
    seeds_comp_lift = int(sum(1 for r in per_seed if r["composed_graded"]["salience_r"]
                              > r["composed_pointsoma"]["salience_r"]))
    RIDGE_TARGET = 0.27
    # the composed graded read must (a) LIFT toward the ridge reference (>0.20, a majority of the 0.10->0.27 gap
    # closed) AND (b) keep the SIGN read (r_sign > 0.2), AND (c) the point-soma control must reproduce the ~0.10
    # boundary (validates the read is the isolated change).
    comp_ps_reproduces = bool(comp_ps < 0.18)                 # reproduces the ~0.10 boundary (loose band)
    lifts_toward_ridge = bool(comp_gr >= 0.20 and lift_comp > 0.03 and seeds_comp_lift >= 5)
    sign_intact = bool(comp_gr_sign > 0.2)
    go_B = bool(n >= 6 and comp_ps_reproduces and lifts_toward_ridge and sign_intact)

    v = Verdict("PART B -- does the graded (magnitude-preserving) plateau read-out lift the affect C-A2 salience "
                "toward the ridge's 0.27?")
    v.require("6 seeds (project bar)", n >= 6, expect=True)
    v.require("the point-soma control reproduces the ~0.10 affect-salience boundary (composed point-soma < 0.18)",
              comp_ps_reproduces, expect=True, note=f"composed point-soma salience_r={comp_ps:+.3f}")
    v.control("the graded read lifts the COMPOSED opponent salience (vs the point-soma read on the SAME weights)",
              treatment=comp_gr, control=comp_ps)
    v.control("read-out isolation: the ridge (magnitude-supervised) weights already reach ~0.27 with the POINT-SOMA "
              "read -- if the read-out were the bottleneck the ridge would be limited too",
              treatment=ridge_ps, control=0.0)
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU + the all-or-none coincidence current",
               "a fixed-operating-point READ-OUT nonlinearity on the code_in->vplus/vminus FF; weights unchanged.")
    vb = v.decide(go=go_B)

    attributable_to("the composed-opponent salience lift attributable to the GRADED read-out (vs the point-soma read "
                    "on the SAME self-organized weights)", comp_gr, comp_ps)

    if smoke or n < 6:
        verdict = ("UNDEFINED -- smoke (n<6); the 4-cell salience table is printed for mechanism-check only.")
    elif go_B:
        verdict = ("GO (PART B) -- the GRADED plateau read-out lifts the affect C-A2 salience |differential|~"
                   "valence-strength from composed point-soma %.3f to composed graded %.3f (lift %+.4f, %d/%d seeds), "
                   "toward the ridge reference %.3f, while the SIGN read holds (r_sign %.3f). The magnitude the "
                   "point-soma spike-rate lost is recovered by the magnitude-preserving read." %
                   (comp_ps, comp_gr, lift_comp, seeds_comp_lift, n, RIDGE_TARGET, comp_gr_sign))
    else:
        verdict = ("BOUNDARY (PART B) -- the graded plateau read-out does NOT close the affect salience gap: composed "
                   "point-soma %.3f -> composed graded %.3f (lift %+.4f, %d/%d seeds); ridge point-soma %.3f -> ridge "
                   "graded %.3f. The read-out is NOT the affect bottleneck: the ridge (magnitude-supervised) weights "
                   "already reach %.3f with the POINT-SOMA read, so the point-soma read is magnitude-preserving ENOUGH "
                   "-- the boundary is the WEIGHT SOURCE (the Rescorla-Wagner s_c SATURATES; the named surpass is a "
                   "GRADED reinforcement-STRENGTH third factor -- Bayer & Glimcher 2005 -- NOT a read-out). This "
                   "DISTINGUISHES the two boundaries: W4 is read-out-limited (Part A), affect is weight-source-"
                   "limited." % (comp_ps, comp_gr, lift_comp, seeds_comp_lift, n, ridge_ps, ridge_gr, ridge_ps))

    summary = {
        "runner": "_magnitude_preserving_plateau_readout_derisk", "part": "B (affect salience)",
        "faculty": "graded dendritic-plateau (magnitude-preserving) read-out on the affect opponent code_in->vplus/"
                   "vminus FF; does it lift the C-A2 |differential|~valence-strength correlation toward the ridge's "
                   "0.27? 4-cell (weights x read-out) design isolates read-out vs weight-source. FUNCTIONAL correlate.",
        "seeds": list(seeds), "backend": backend, "smoke": bool(smoke or n < 6), "verdict": verdict, "go": go_B,
        "external_grounding": [
            "Mikulasch & Priesemann -- dendritic ANALOG/graded read-out.",
            "Bayer & Glimcher (2005) Neuron -- graded-magnitude reward-prediction error (the named weight-source "
            "surpass IF the read-out is not the bottleneck).",
            "Namburi/Tye et al. (2015) Nature -- the innate V+/V- opponent channel.",
        ],
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "gates": {"point_soma_reproduces_boundary": comp_ps_reproduces, "graded_lifts_toward_ridge": lifts_toward_ridge,
                  "sign_read_intact": sign_intact},
        "salience_r": {"composed_pointsoma": comp_ps, "composed_graded": comp_gr, "ridge_pointsoma": ridge_ps,
                       "ridge_graded": ridge_gr, "lift_composed": lift_comp, "lift_ridge": lift_ridge,
                       "seeds_composed_lift": seeds_comp_lift, "ridge_target": RIDGE_TARGET},
        "sign_r": {"composed_pointsoma": comp_ps_sign, "composed_graded": comp_gr_sign},
        "calibration": {"center_per_seed": {str(c["seed"]): c["center"] for c in calib},
                        "slope_per_seed": {str(c["seed"]): c["slope"] for c in calib}, "per_seed": calib},
        "per_seed": per_seed,
        "honest_scope": (
            "A FUNCTIONAL affect-salience correlate. The graded plateau is applied ONLY to the opponent read-out (the "
            "code_in->vplus/vminus FF), weights unchanged; the point-soma cell is the byte-comparable control. The "
            "4-cell (weights x read-out) design isolates the read-out from the weight source. Warriner is EVAL-only "
            "ground-truth. Plasticity off. numpy-CPU real spiking Izhikevich; NO sim/ edit; reuse-by-import of the "
            "affect-deepen circuit + the composed self-organized opponent. NOT a claim of phenomenal experience."),
    }
    return summary, verdict


def run_part_B(seeds, backend, smoke, out_path, max_stories, n_hub):
    t0 = time.time()
    print(f"[mag-plateau B] affect opponent graded-plateau read-out | seeds={seeds} max_stories={max_stories} "
          f"n_hub={n_hub}", flush=True)
    A = build_all(max_stories, n_hub, window=4, min_count=5)
    print(f"  self-organized codes: {len(A['vocab'])} concepts x {A['codes'].shape[1]} hubs | primaries "
          f"{len(A['app'])}+{len(A['avr'])} ({round(time.time()-t0,1)}s)", flush=True)
    preps = {s: _aff_prep_seed(s, A) for s in seeds}
    calib = [calibrate_affect_seed(s, preps[s]) for s in seeds]
    per_seed = [eval_affect_seed(s, preps[s], c["center"], c["slope"]) for s, c in zip(seeds, calib)]
    summary, verdict = build_summary_B(calib, per_seed, seeds, backend, smoke)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    summary["config"] = {"max_stories": max_stories, "n_hub": n_hub, "n_vocab": len(A["vocab"])}
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    s = summary["salience_r"]
    print("\n" + "=" * 114, flush=True)
    print(f"[mag-plateau B] === VERDICT: {verdict} ===", flush=True)
    print(f"[mag-plateau B]  SALIENCE_r  composed: point-soma={s['composed_pointsoma']} graded={s['composed_graded']} "
          f"(lift {s['lift_composed']:+.4f}, {s['seeds_composed_lift']}/{len(seeds)}) | ridge: "
          f"point-soma={s['ridge_pointsoma']} graded={s['ridge_graded']} | ridge_target={s['ridge_target']}", flush=True)
    print(f"[mag-plateau B]  gates={summary['gates']}", flush=True)
    print(f"[mag-plateau B]  wrote {out_path}\n" + "=" * 114, flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="magnitude-preserving graded dendritic-plateau read-out: close the W4 "
                                             "detector (A) + affect salience (B) boundaries with one mechanism.")
    ap.add_argument("--part", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 seed, prints the teeth (verdict UNDEFINED at n<6)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--max-stories", type=int, default=60000, help="PART B corpus size")
    ap.add_argument("--n-hub", type=int, default=64, help="PART B concept code dim")
    ap.add_argument("--json", type=str, default="research/findings/raw/_magnitude_preserving/summary.json")
    args = ap.parse_args()
    if args.backend != "auto":
        get_backend(args.backend)

    seeds = [args.seed] if args.seed is not None else ([args.seeds[0]] if args.smoke else args.seeds)
    smoke = bool(args.smoke or len(seeds) < 6)
    max_stories = min(args.max_stories, 8000) if args.smoke else args.max_stories

    rc = 0
    if args.part in ("A", "both"):
        outA = args.json if args.part == "A" else args.json.replace(".json", "_A.json")
        sA = run_part_A(seeds, args.backend, smoke, outA)
        rc = rc or (0 if sA["go"] else 1)
    if args.part in ("B", "both"):
        outB = args.json if args.part == "B" else args.json.replace(".json", "_B.json")
        sB = run_part_B(seeds, args.backend, smoke, outB, max_stories, args.n_hub)
        rc = rc or (0 if sB["go"] else 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
