"""D · PRAGMATICS -- TASK #12 close: a GENUINELY-SPIKING graded scalar-implicature ToM belief source, read
through a SPIKING magnitude-sensitive pragmatic-alignment metric (NO host argmax) -- and the DECISIVE test of
whether that metric MOVES where the 2026-08-11 host-argmax metric could not.

THE 2026-08-11 HONEST NEGATIVE (respected, not overturned by fiat):
  `2026-08-11-W4-RSA-belief-source-into-speaking-pipeline-6seed.md` FAITHFULLY wired the W4 graded-implicature RSA
  posterior as the speaking-pipeline belief source (12x better calibrated to the analytic Frank-Goodman RSA, moat
  intact) but it did NOT move the pragmatic-alignment metric. Its own re-diagnosis named WHY, in two structural
  parts, and named the two next levers:
    (1) THE METRIC IS A HOST ARGMAX. `aligned[t]=argmax_u belief[u][t]` and `succ_opt[t]=argmax_u S[t,u]` are host
        argmaxes -- STRUCTURALLY INSENSITIVE to the graded magnitude the calibrated belief adds. "only belief
        MAGNITUDES change, and succ_opt/learned-aligned are argmax reads ... INSENSITIVE to magnitude refinement."
        NEXT LEVER (verbatim): "a magnitude-sensitive pragmatic reward (informativeness = the listener's graded
        posterior mass on the true intent, belief[u][t], read through the neural coincidence rate rather than an
        argmax)."
    (2) THE SUCC_OPT GAP IS THE DETECTOR, not the belief -- a point-soma coincidence-DETECTOR base-rate artifact
        that "corrupts the DIAGONAL itself". NEXT LEVER (verbatim): the standing 2026-07-08 dendritic dAP READOUT
        (the two-compartment regenerative plateau; held-out completion 0.571 vs point-neuron 0.007) on the
        detector pool -- selectivity decoupled from magnitude, the companion process the point soma lacks.

WHAT THIS RUNNER DOES (the two named levers, both spiking, NO sim/ edit, reuse-by-import):
  * BELIEF (spiking, no host argmax): the graded implicature belief is the substrate's SOFT-COMPETITION population
    rate over the state assemblies -- L1(s|u)=normalize_states(S1_neural[u,:]) read from the W4 RSA bridge's FS
    divisive-normalization competition (`_recursive_tom_rsa_derisk._compete`), ONE step before the operating
    point's final hard-WTA `_compete` would collapse it. A graded population rate, not a host argmax.
  * METRIC (spiking, magnitude-sensitive): the pragmatic-alignment is the NEURAL success LANDSCAPE
    S[t,u] = success_signal(belief=belief[u], intent=t) read off the Leg-1 coincidence detector
    (`_pragmatic_success_coincidence_derisk`). S[t,u] is the listener's posterior mass on the TRUE intent t,
    delivered as graded currents and read as a graded coincidence RATE -- exactly lever (1). The alignment score
    is the FIDELITY of that neural landscape to the analytic Frank-Goodman RSA landscape (total-variation, per
    intent). The graded belief carries the off-diagonal "some is still compatible with all" mass (L1(all|some)
    ~0.25) that the one-hot ERASES; a speaker whose listener-model has that mass is more RSA-faithful. The
    argmax-of-S metric (the 2026-08-11 read) is ALSO computed, to show it stays flat while the magnitude metric
    moves.
  * DETECTOR (spiking, the companion process): the SAME landscape is read through the dendritic-coincidence
    PLATEAU (coincidence=True; the engine-native Poirazi/Larkum two-input plateau) vs the LINEAR point-soma sham
    (coincidence=False). Lever (2): the plateau's selectivity is what lets the graded content survive the
    detector's base-rate; the linear point soma should NOT deliver the move.

PRE-REGISTERED GO GATE (6 seeds 42 43 44 100 101 102, CPU numpy; COMPARATIVE graded-vs-onehot):
  G1  belief is GRADED + spiking:  mean graded implicature margin (SBNA-all) > 0.05  (population-rate read).
  G2  belief tracks the RSA ideal: mean graded calib_l1(some)->analytic  <  mean onehot calib_l1(some).
  G3  THE METRIC MOVES (plateau): mean mag_fidelity(graded) - mag_fidelity(onehot) > MOVE_EPS (=0.03)
      AND graded > onehot on >= 5/6 seeds.
  G4  the OLD host-argmax metric does NOT move: |argmax_align(graded) - argmax_align(onehot)| <= 0.05
      (reproduces the 2026-08-11 negative -- the instrument was the problem, not the belief).
  G5  ANTI-CHEATS all hold:
        (a) normalization-LESION (FS off) collapses the move: mag_fidelity(lesion) does NOT beat onehot;
        (b) SCRAMBLE (belief rows permuted -> graded mass on WRONG intents) is WORSE than onehot
            (guards against "any gradedness wins" -- the move must come from CORRECT implicature content);
        (c) the DETECTOR is load-bearing: the graded-vs-onehot move is LARGER with the plateau than the linear
            point-soma (the 2026-08-11 residual detector).

VERDICT is comparative + honest: GO iff G1..G5. Else the failing gate + the residual + the next lever. This does
NOT overturn the 2026-08-11 negative by moving goalposts: the magnitude-sensitive read + the dendritic detector
are the finding's OWN two named next mechanisms; the scramble/lesion controls keep the move honest.

HONEST SCOPE. A FUNCTIONAL pragmatics correlate: a spiking graded listener-belief source + a spiking
magnitude-sensitive alignment read. NOT a claim of phenomenal access to another mind; a self-report would be a
functional read-out. Plasticity off (fixed operating point, as in the W4/leg2 GOs). The per-intent rate
normalization (spike-count -> rate, /sum) is a read-out op, the same footing the existing pipeline already uses;
the graded STRUCTURE is the substrate's FS divisive normalization (collapses under its lesion). numpy-CPU real
spiking Izhikevich bridges; additive NEW runner (reuse-by-import); NO sim/ edit.

Usage:
  # fast single-seed smoke (STEP 1 deterministic ceiling; ~1-2 min):
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_spiking_graded_belief_derisk --smoke --seed 42 \
      --json research/findings/raw/_pragmatic_success/spiking_graded_belief_smoke.json
  # 6-seed deliverable:
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_spiking_graded_belief_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_pragmatic_success/spiking_graded_belief_6seed.json
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

from research.runners._recursive_tom_rsa_derisk import STATES, UTTS  # noqa: E402
from research.runners._pragmatic_success_coincidence_derisk import (  # noqa: E402
    build_success_bridge, success_signal, K,
)
from research.runners._pragmatic_graded_belief_source_derisk import (  # noqa: E402
    graded_belief_sources, onehot_belief_sources, ANALYTIC_L1,
)

MOVE_EPS = 0.03          # G3: minimum mean fidelity gain (graded - onehot) to call the metric "moved"
ARGMAX_FLAT_EPS = 0.05   # G4: the host-argmax metric must stay within this of the onehot (reproduce the negative)


# ── belief sources (all spiking soft-competition population rates; scramble/lesion are the anti-cheats) ────────

def _scramble_belief(belief_src, seed):
    """ANTI-CHEAT: permute (derangement) which STATE each utterance's posterior points at, keeping the exact same
    graded shape. The belief is EQUALLY graded but its mass now lands on the WRONG intents. If the fidelity metric
    only rewarded 'gradedness' this would tie the intact graded belief; a faithful metric makes it LOSE."""
    prng = np.random.default_rng(seed * 613 + 29)
    perm = np.arange(K)
    while np.any(perm == np.arange(K)):
        perm = prng.permutation(K)
    return {u: np.asarray(belief_src[u], dtype=np.float64)[perm].copy() for u in UTTS}


def belief_variants(seed):
    """Return the belief sources under test. All are graded population-rate reads off the substrate except
    `onehot` (the leg2_v2 WTA baseline). `graded_lesion` = FS-normalization off (moat). `scramble` = graded mass
    on wrong intents (anti-cheat)."""
    graded = graded_belief_sources(seed, normalize=True)
    return {
        "onehot": onehot_belief_sources(seed),                      # leg2_v2 WTA one-hot baseline
        "graded": graded,                                           # the spiking soft-competition graded read
        "graded_lesion": graded_belief_sources(seed, normalize=False),  # FS off -> flat (moat)
        "scramble": _scramble_belief(graded, seed),                 # graded mass on WRONG intents (anti-cheat)
    }


# ── belief-side reads (calibration + implicature; graded should beat onehot on both) ──────────────────────────

def _implicature_margin(belief_src):
    v = belief_src["some"]
    return float(v[STATES.index("SBNA")] - v[STATES.index("all")])


def _calib_l1_some(belief_src):
    return float(np.sum(np.abs(np.asarray(belief_src["some"], float) - ANALYTIC_L1["some"])))


# ── the SPIKING success landscape S[t,u] and the magnitude-sensitive fidelity metric ──────────────────────────

def neural_success_landscape(bridge, xp, idx, snap, belief_src):
    """S[t,u] = success_signal(intent=t, belief=belief[u]) read off the coincidence detector -- the listener's
    posterior mass on the TRUE intent t under utterance u, delivered as graded currents and read as a graded
    coincidence RATE (NO host argmax). Returns a KxK matrix indexed [intent t][utterance u]."""
    S = np.zeros((K, K), dtype=np.float64)
    for t in range(K):
        for u_i, u in enumerate(UTTS):
            S[t, u_i] = success_signal(bridge, xp, idx, snap, belief_src[u], t)
    return S


def _analytic_landscape():
    """S_ideal[t,u] = analytic Frank-Goodman L1(state=t | utterance=u) -- the rational listener's recovery of
    intent t under each utterance. Ground truth (fixed), NOT read from the substrate."""
    S = np.zeros((K, K), dtype=np.float64)
    for t in range(K):
        for u_i, u in enumerate(UTTS):
            S[t, u_i] = ANALYTIC_L1[u][t]
    return S


def _row_norm(M, eps=1e-9):
    """Per-intent (row) normalization to a distribution over utterances -> controls the intent-drive base rate,
    isolates WHICH utterance the (neural) listener recovers the intent from."""
    out = np.zeros_like(M, dtype=np.float64)
    for t in range(M.shape[0]):
        s = float(M[t].sum())
        out[t] = M[t] / s if s > eps else np.full(M.shape[1], 1.0 / M.shape[1])
    return out


def mag_fidelity(S_neural, S_ideal):
    """MAGNITUDE-SENSITIVE pragmatic-alignment = fidelity of the neural success landscape to the analytic RSA
    landscape. Per intent t: 1 - TV(Snorm[t,:], Ideal[t,:]); mean over intents. Higher = the neural listener-model
    the speaker sees is more RSA-faithful. Magnitude-sensitive (uses the graded off-diagonal mass), NOT an argmax."""
    Sn, In = _row_norm(S_neural), _row_norm(S_ideal)
    per_t = [1.0 - 0.5 * float(np.sum(np.abs(Sn[t] - In[t]))) for t in range(Sn.shape[0])]
    return float(np.mean(per_t)), per_t


def argmax_align(S_neural):
    """The 2026-08-11 host-argmax metric: does argmax_u S[t,u] recover the RSA-informative utterance a(t)=t?
    (identity indexing: intent t's informative utterance is utterance t). Magnitude-BLIND -> should NOT move."""
    return float(np.mean([int(np.argmax(S_neural[t])) == t for t in range(K)]))


# ── per-seed evaluation ───────────────────────────────────────────────────────────────────────────────────────

def eval_seed(seed, verbose=True):
    bel = belief_variants(seed)
    S_ideal = _analytic_landscape()
    rec = {"seed": int(seed), "belief_some": {n: [round(float(x), 4) for x in bel[n]["some"]] for n in bel}}

    for det_name, coincidence in (("plateau", True), ("linear", False)):
        bridge, xp, idx, snap = build_success_bridge(seed, coincidence=coincidence)
        det = {}
        for bname, src in bel.items():
            S = neural_success_landscape(bridge, xp, idx, snap, src)
            mf, per_t = mag_fidelity(S, S_ideal)
            det[bname] = {
                "mag_fidelity": round(mf, 5),
                "argmax_align": round(argmax_align(S), 4),
                "S": [[round(float(x), 5) for x in row] for row in S],
                "mag_fidelity_per_intent": [round(x, 4) for x in per_t],
            }
        rec[det_name] = det
        if verbose:
            g, o = det["graded"], det["onehot"]
            print(f"  [seed {seed} | {det_name}] mag_fidelity onehot={o['mag_fidelity']} graded={g['mag_fidelity']} "
                  f"(move {g['mag_fidelity'] - o['mag_fidelity']:+.4f}) | argmax_align onehot={o['argmax_align']} "
                  f"graded={g['argmax_align']} | lesion={det['graded_lesion']['mag_fidelity']} "
                  f"scramble={det['scramble']['mag_fidelity']}", flush=True)

    rec["belief_implicature_margin_graded"] = round(_implicature_margin(bel["graded"]), 4)
    rec["belief_implicature_margin_lesion"] = round(_implicature_margin(bel["graded_lesion"]), 4)
    rec["belief_calib_l1_some_onehot"] = round(_calib_l1_some(bel["onehot"]), 4)
    rec["belief_calib_l1_some_graded"] = round(_calib_l1_some(bel["graded"]), 4)
    return rec


# ── aggregation + pre-registered verdict ──────────────────────────────────────────────────────────────────────

def _mean(rows, fn):
    return float(np.mean([fn(r) for r in rows]))


def build_summary(per_seed, seeds, backend):
    from tools.verdict import Verdict
    from tools.lab import attributable_to

    n = len(seeds)
    agg = {}
    # belief side
    agg["belief_implicature_margin_graded"] = round(_mean(per_seed, lambda r: r["belief_implicature_margin_graded"]), 4)
    agg["belief_implicature_margin_lesion"] = round(_mean(per_seed, lambda r: r["belief_implicature_margin_lesion"]), 4)
    agg["belief_calib_l1_some_onehot"] = round(_mean(per_seed, lambda r: r["belief_calib_l1_some_onehot"]), 4)
    agg["belief_calib_l1_some_graded"] = round(_mean(per_seed, lambda r: r["belief_calib_l1_some_graded"]), 4)

    # metric side, per detector
    for det in ("plateau", "linear"):
        for b in ("onehot", "graded", "graded_lesion", "scramble"):
            agg[f"{det}_mag_fidelity_{b}"] = round(_mean(per_seed, lambda r, d=det, bb=b: r[d][bb]["mag_fidelity"]), 5)
        for b in ("onehot", "graded"):
            agg[f"{det}_argmax_align_{b}"] = round(_mean(per_seed, lambda r, d=det, bb=b: r[d][bb]["argmax_align"]), 4)
        agg[f"{det}_move_graded_minus_onehot"] = round(
            agg[f"{det}_mag_fidelity_graded"] - agg[f"{det}_mag_fidelity_onehot"], 5)
        agg[f"{det}_seeds_graded_gt_onehot"] = int(sum(
            1 for r in per_seed if r[det]["graded"]["mag_fidelity"] > r[det]["onehot"]["mag_fidelity"]))
        # DIAGNOSTIC (a-priori named, NOT a gate): the direct implicature-recovery cell = the neural recovery of
        # intent=all under utterance="some", per-intent normalized. This is the finding's EXACT named quantity
        # ("the listener's graded posterior mass on the true intent, belief[u][t], read through the neural
        # coincidence rate"): graded carries ~0.25 mass on `all` after "some"; the one-hot ERASES it to 0. Isolates
        # where the implicature actually lives (the fidelity gate averages it across 2 intents with no graded
        # structure, diluting it). Reported alongside the pre-registered fidelity gate, not in place of it.
        _ai, _si = STATES.index("all"), UTTS.index("some")
        for b in ("onehot", "graded", "scramble"):
            def _rec(r, d=det, bb=b):
                row = np.asarray(r[d][bb]["S"], float)[_ai]
                s = float(row.sum())
                return float(row[_si] / s) if s > 1e-9 else 0.0
            agg[f"{det}_implicature_recovery_{b}"] = round(_mean(per_seed, _rec), 5)

    # ── pre-registered gates ──
    g1 = agg["belief_implicature_margin_graded"] > 0.05
    g2 = agg["belief_calib_l1_some_graded"] < agg["belief_calib_l1_some_onehot"]
    g3 = (agg["plateau_move_graded_minus_onehot"] > MOVE_EPS
          and agg["plateau_seeds_graded_gt_onehot"] >= max(5, n - 1))
    g4 = abs(agg["plateau_argmax_align_graded"] - agg["plateau_argmax_align_onehot"]) <= ARGMAX_FLAT_EPS
    g5a = agg["plateau_mag_fidelity_graded_lesion"] <= agg["plateau_mag_fidelity_onehot"] + 1e-6
    g5b = agg["plateau_mag_fidelity_scramble"] < agg["plateau_mag_fidelity_onehot"] - 1e-6
    g5c = agg["plateau_move_graded_minus_onehot"] > agg["linear_move_graded_minus_onehot"] + 1e-6
    g5 = g5a and g5b and g5c
    metric_moved = bool(g1 and g2 and g3 and g4 and g5)

    # The Verdict validates the INSTRUMENT (all must hold for the metric-move A/B to be trustworthy); whether the
    # metric MOVES is the HYPOTHESIS under test -- its falsification is a clean scientific NEGATIVE, NOT an
    # instrument failure, so it is reported as `metric_moved`, NOT a v.require (which would return UNDEFINED). Same
    # structure as the 2026-08-11 runner it builds on.
    v = Verdict("D pragmatics TASK#12 -- INSTRUMENT VALIDITY for the spiking-graded-vs-onehot metric-move A/B "
                "(is the belief a sound graded spiking read + is the magnitude metric a valid, non-trivial "
                "instrument?)", chance=1.0 / K)
    v.require("6 seeds (project bar)", n >= 6, expect=True)
    v.require("G1 graded belief is GRADED (implicature margin SBNA-all > 0.05), a spiking population-rate read",
              agg["belief_implicature_margin_graded"], expect=lambda x: x > 0.05)
    v.control("G1-moat normalization-LESION collapses the graded implicature (the graded content is the FS "
              "divisive normalization, not host-injected)",
              treatment=agg["belief_implicature_margin_graded"], control=agg["belief_implicature_margin_lesion"])
    v.require("G2 graded belief better-calibrated to analytic RSA than one-hot (calib_l1(some) lower)",
              g2, expect=True,
              note=f"onehot={agg['belief_calib_l1_some_onehot']} graded={agg['belief_calib_l1_some_graded']}")
    v.require("G4 the OLD host-argmax metric does NOT move (reproduces the 2026-08-11 negative -- confirms the "
              "argmax read is magnitude-blind)", g4, expect=True,
              note=f"argmax_align onehot={agg['plateau_argmax_align_onehot']} graded={agg['plateau_argmax_align_graded']}")
    v.control("G5a normalization-LESION does not deliver a spurious move (fidelity(lesion) <= onehot)",
              treatment=agg["plateau_mag_fidelity_onehot"], control=agg["plateau_mag_fidelity_graded_lesion"])
    v.require("G5b SCRAMBLE (graded mass on WRONG intents) is WORSE than onehot -- the fidelity metric is a VALID "
              "non-trivial instrument (it does NOT reward gradedness per se, only CORRECT implicature content)",
              g5b, expect=True,
              note=f"scramble={agg['plateau_mag_fidelity_scramble']} onehot={agg['plateau_mag_fidelity_onehot']}")
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU/NMDA",
               "belief stores + RSA normalizer + coincidence detector read at a fixed operating point (as in the "
               "W4/leg2 GOs); no learning in this STEP-1 deterministic ceiling.")

    # instrument sound iff the belief is a valid graded spiking read AND the metric is a valid non-trivial
    # instrument (argmax reproduces the negative; scramble loses; lesion does not spuriously win).
    instrument_valid = (n >= 6 and g1 and g2 and g4 and g5a and g5b)
    vb = v.decide(go=instrument_valid)

    attributable_to("graded implicature content attributable to FS divisive normalization (vs its lesion)",
                    agg["belief_implicature_margin_graded"], agg["belief_implicature_margin_lesion"])
    attributable_to("the metric-move attributable to the DENDRITIC PLATEAU detector (vs linear point-soma)",
                    agg["plateau_move_graded_minus_onehot"], agg["linear_move_graded_minus_onehot"])

    if not instrument_valid:
        # preconditions unmet (e.g. n<6 smoke, or a broken instrument) -> UNDEFINED, never a negative.
        unmet = [g for g, ok in (("6seeds", n >= 6), ("G1", g1), ("G2", g2), ("G4", g4),
                                  ("G5a", g5a), ("G5b", g5b)) if not ok]
        verdict = ("UNDEFINED -- instrument preconditions unmet (%s); a run whose preconditions do not hold "
                   "yields UNDEFINED, never a negative. (Smoke = n<6.)" % ",".join(unmet))
    elif metric_moved:
        verdict = ("GO -- the SPIKING graded belief read through a SPIKING magnitude-sensitive coincidence metric "
                   "MOVES the pragmatic-alignment metric (mean fidelity +%.4f, %d/%d seeds), where the 2026-08-11 "
                   "host-argmax metric stayed flat (G4). The move is the CORRECT implicature content (scramble "
                   "LOSES), the FS normalization (lesion collapses it), and the dendritic plateau (linear point "
                   "soma delivers less). NO host argmax." % (agg["plateau_move_graded_minus_onehot"],
                                                             agg["plateau_seeds_graded_gt_onehot"], n))
    else:
        failed = [g for g, ok in (("G3", g3), ("G5c", g5c)) if not ok]
        verdict = ("NEGATIVE/BOUNDARY -- instrument VALID (belief graded+calibrated; argmax reproduces the "
                   "2026-08-11 negative; scramble loses) but the pre-registered metric-MOVE hypothesis failed "
                   "(%s): the graded belief does not move the magnitude metric and the default plateau does not "
                   "rescue it; the residual is the detector base rate. Next lever in the finding. Does NOT "
                   "overclaim past the 2026-08-11 negative." % ",".join(failed))

    summary = {
        "runner": "_pragmatic_spiking_graded_belief_derisk",
        "faculty": "D pragmatics TASK#12: a genuinely-spiking graded scalar-implicature ToM belief source + a "
                   "spiking magnitude-sensitive pragmatic-alignment metric (no host argmax), and whether the "
                   "metric moves where the 2026-08-11 host-argmax metric could not.",
        "builds_on": [
            "2026-08-11-W4-RSA-belief-source-into-speaking-pipeline-6seed (the honest negative + its 2 named levers)",
            "2026-08-01-W4-recursive-theory-of-mind-...-6seed-GO (depth-2 scalar implicature GO)",
            "2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed (the dendritic plateau read-out)",
        ],
        "seeds": list(seeds), "backend": backend, "chance": 1.0 / K,
        "move_eps": MOVE_EPS, "argmax_flat_eps": ARGMAX_FLAT_EPS,
        "verdict": verdict, "metric_moved": metric_moved, "instrument_valid": bool(instrument_valid),
        "gates": {"G1_graded": bool(g1), "G2_calibrated": bool(g2), "G3_metric_moves_plateau": bool(g3),
                  "G4_argmax_metric_flat": bool(g4), "G5a_lesion_collapses": bool(g5a),
                  "G5b_scramble_loses": bool(g5b), "G5c_plateau_load_bearing": bool(g5c)},
        **{kk: vb[kk] for kk in ("preconditions", "disabled_processes", "undefined_reasons")},
        "aggregate": agg,
        "per_seed": per_seed,
        "honest_scope": (
            "The graded belief is the substrate's FS soft-competition population rate over the state assemblies "
            "(no host argmax); the pragmatic-alignment metric is the neural coincidence-rate landscape's fidelity "
            "to the analytic Frank-Goodman RSA (magnitude-sensitive, the finding's OWN named lever), read through "
            "the dendritic-coincidence plateau. The move is kept honest by the scramble control (graded mass on "
            "WRONG intents must LOSE) and the normalization-lesion (collapses the graded content). A FUNCTIONAL "
            "pragmatics correlate; NOT a claim of phenomenal access to another mind. Plasticity off (fixed "
            "operating point). Per-intent rate normalization is a read-out op (spike-count->rate); the graded "
            "STRUCTURE is the substrate's divisive normalization. numpy-CPU real spiking Izhikevich; NO sim/ edit."),
    }
    return summary, verdict


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    a = summary["aggregate"]
    print("\n" + "=" * 110, flush=True)
    print(f"[spiking-graded] === VERDICT: {verdict} ===", flush=True)
    print(f"[spiking-graded]  BELIEF: implicature margin graded={a['belief_implicature_margin_graded']:+.3f} "
          f"(lesion={a['belief_implicature_margin_lesion']:+.3f}) | calib_l1(some)->analytic onehot="
          f"{a['belief_calib_l1_some_onehot']} graded={a['belief_calib_l1_some_graded']} (lower=better)", flush=True)
    print(f"[spiking-graded]  METRIC (plateau): mag_fidelity onehot={a['plateau_mag_fidelity_onehot']} "
          f"graded={a['plateau_mag_fidelity_graded']} MOVE={a['plateau_move_graded_minus_onehot']:+.4f} "
          f"(seeds graded>onehot={a['plateau_seeds_graded_gt_onehot']}) | lesion={a['plateau_mag_fidelity_graded_lesion']} "
          f"scramble={a['plateau_mag_fidelity_scramble']}", flush=True)
    print(f"[spiking-graded]  OLD ARGMAX metric (plateau): onehot={a['plateau_argmax_align_onehot']} "
          f"graded={a['plateau_argmax_align_graded']} (should be ~equal -> the instrument was the problem)", flush=True)
    print(f"[spiking-graded]  DETECTOR: plateau move={a['plateau_move_graded_minus_onehot']:+.4f} vs "
          f"linear move={a['linear_move_graded_minus_onehot']:+.4f}", flush=True)
    print(f"[spiking-graded]  DIAG implicature-recovery S[all|some] (plateau): onehot="
          f"{a['plateau_implicature_recovery_onehot']} graded={a['plateau_implicature_recovery_graded']} "
          f"scramble={a['plateau_implicature_recovery_scramble']} (the some->not-all mass the one-hot erases)",
          flush=True)
    print(f"[spiking-graded]  gates={summary['gates']}", flush=True)
    print(f"[spiking-graded]  wrote {out_path}\n" + "=" * 110, flush=True)


def main():
    ap = argparse.ArgumentParser(description="TASK#12: spiking graded scalar-implicature belief + spiking "
                                             "magnitude-sensitive pragmatic-alignment metric; does the metric move?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=None, help="single-seed convenience (overrides --seeds)")
    ap.add_argument("--smoke", action="store_true", help="single seed, prints the teeth (verdict is UNDEFINED at n<6)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--reaggregate", type=str, default=None, help="rebuild summary/verdict from a summary JSON")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/spiking_graded_belief_6seed.json")
    args = ap.parse_args()

    if args.reaggregate:
        with open(args.reaggregate) as f:
            d = json.load(f)
        summary, verdict = build_summary(d["per_seed"], d["seeds"], d.get("backend", args.backend))
        _emit(summary, verdict, args.json)
        return 0 if summary["metric_moved"] else 1

    seeds = [args.seed] if args.seed is not None else ([args.seeds[0]] if args.smoke else args.seeds)
    t0 = time.time()
    print(f"[spiking-graded] TASK#12 spiking graded belief + spiking magnitude metric | seeds={seeds} "
          f"backend={args.backend}", flush=True)
    print("[spiking-graded] BELIEF = FS soft-competition population rate (no host argmax); METRIC = neural "
          "coincidence-landscape fidelity to analytic RSA (magnitude-sensitive), read via the dendritic plateau.",
          flush=True)

    per_seed = [eval_seed(s) for s in seeds]
    summary, verdict = build_summary(per_seed, seeds, args.backend)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    summary["smoke"] = bool(args.smoke or len(seeds) < 6)
    _emit(summary, verdict, args.json)
    return 0 if summary["metric_moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
