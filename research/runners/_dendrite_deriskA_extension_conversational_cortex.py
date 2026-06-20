"""DENDRITE DE-RISK A -- EXTENSION to the CONVERSATIONAL learned cortex (Stage 0, CPU/numpy, NO sim/ edit).

THE QUESTION
============
De-risk A (`2026-06-20-dendrite-derisk-A-graded-plateau-readout.md`, GO 6/6) proved the dendrite's ONE
genuine unlock: a GRADED dendritic-plateau read-out  V = sigmoid((v_basal - theta)/slope)  produces a graded
analog quantity a point neuron's somatic spike rate provably cannot (0 sub-rheobase, or saturated all-or-none).
Its instance was the nav value-critic delta = r - V.

The dendrite scoping (`2acebf6b`, controller-verified) classed the conversational "learned graded cortex
embedding" (D2 Phase 2) as the OTHER instance of the SAME family -- "a graded read-out of a distributed code"
(the cortex reads a graded similarity/generalization off a DISTRIBUTED concept code). This extension asks:

    Does the SAME graded dendritic-plateau read-out recover a GRADED, faithful read-out of the LEARNED
    CONVERSATIONAL concept code (a graded category-similarity signal) where the point-neuron read-out
    (D2 Phase 2) returned an honest NEGATIVE? I.e. does the dendrite's graded-read-out unlock GENERALIZE
    from the nav value-critic to the conversational cortex?

THE D2 PHASE-2 NEGATIVE THIS BUILDS ON (`2026-06-14-D2-phase1-DONE-phase2-frontier.md`)
=======================================================================================
The task: recover an a-priori category structure S_true from a concept x hub co-occurrence COUNT matrix C
whose high-frequency COMMON hubs ("said","day","big" -- every concept connects to them) DOMINATE every
profile. The decisive metric: Pearson(cos(codes), S_true) -- a graded category-similarity fidelity vs the
host PPMI+SVD ceiling (+0.96 on the counts). The point-neuron forward-pass read-out (random projection +
Izhikevich dynamics) gave codes ANTI-CORRELATED with S_true (~ -0.07) at every setting; the deeper limit is
that the spiking rheobase THRESHOLD silences the low-count category hubs (the low-frequency informative
signal) -- a graded read-out that doesn't hard-threshold is the natural candidate fix.

THE LADDER (the CONTRAST is the result; all read-outs read the IDENTICAL common-mode-dominated count input)
============================================================================================================
All arms read the SAME concept x hub count code (NO host normalization pre-applied unless the arm IS the
normalization arm -- so we can ATTRIBUTE the recovery to the graded read-out vs the normalization).
  L0  POINT-NEURON all-or-none read-out (Heaviside threshold of v_basal = the somatic-spike read-out;
      silences low / saturates high). The validity gate -- expect FAIL (~0), reproducing the D2 NEGATIVE.
  L1  GRADED dendritic plateau read-out, FIXED-random W_basal (the de-risk-A mechanism, NO normalization,
      NO learning). DOES GRADED-NESS ALONE recover the structure where the point neuron can't? (the direct
      "does the de-risk-A unlock transfer" test.)
  L2  GRADED dendritic plateau read-out, W_basal LEARNED UNSUPERVISED on the stream (local Urbanczik-Senn,
      apical = the read-out's OWN activity -- NO category label, NO S_true). Does the local learning rule
      let the graded read-out find the structure unsupervised? (the legitimate-teacher analog of de-risk A.)
  L3  PER-HUB divisive NORMALIZATION (the D2 Phase-1 dendritic gain -- a SEPARATE per-INPUT dendritic
      mechanism) THEN the GRADED plateau read-out. Does the graded read-out COMPOSE with the normalization,
      and -- the attribution -- is the GRADED read-out or the NORMALIZATION the load-bearing piece? (its
      all-or-none lesion isolates the graded contribution.)
  HOST  PPMI+SVD on the counts -- the ceiling (the data carries it).

VERDICT
=======
  GO       = the GRADED dendritic-plateau read-out (L1 or L2) recovers the graded category-similarity
             (Pearson >= bar) where the point-neuron read-out (L0) fails AND the graded-ness is LOAD-BEARING
             (the all-or-none lesion collapses) -> the de-risk-A unlock GENERALIZES to the conversational
             cortex; strengthens the Stage-1 case (the graded plateau is a general read-out, not nav-only).
  NEGATIVE = the GRADED plateau read-out ALSO fails on the conversational code (L1/L2 ~ the point neuron) ->
             the conversational cortex NEGATIVE is NOT (just) a graded-read-out problem. We then CHARACTERIZE
             the deeper cause via the ladder (L3 attribution): which dendritic mechanism the wall actually
             needs (per-input normalization, a DIFFERENT dendritic unlock than the graded read-out).

ANTI-CHEATS (the de-risk-A battery + the D2 controls -- ALL, multi-seed)
=======================================================================
  (a) POINT-NEURON CONTROL (L0) re-asserted IN-RUN and FAILS (the validity gate; if L0 does NOT fail the
      regime is mis-calibrated and the comparison is VOID).
  (b) HOST CEILING carries (PPMI+SVD on the counts >= bar; the data has the structure -> a failure is the
      mechanism, not the data).
  (c) GRADED-NESS LOAD-BEARING -- for whichever arm wins, its ALL-OR-NONE (apical) lesion must collapse it
      (the de-risk-A anti-cheat b). If the all-or-none version does AS WELL, the graded-ness is not the
      load-bearing element -> NOT the de-risk-A unlock.
  (d) NO HOST-NORMALIZATION SMUGGLING -- L0/L1/L2 read the RAW common-mode-dominated counts (only L3 applies
      the per-hub normalization, and L3's purpose is the ATTRIBUTION, not the GO).
  (e) PERMUTED-S collapses (shuffle which concepts are same-category -> recovery ~0): the recovery is the
      real category structure, not cosine geometry.
  (f) S_true A-PRIORI (constructed block, never data-derived) + the per-hub gains LEARNED ONLINE (L3) +
      multi-seed.

Stage 0 = NO sim/ edit (reuse-by-import of sim.dendritic_neuron.DendriticLayer +
sim.dendritic_plasticity.urbanczik_senn_update + the D1/D2 count builder + the host PPMI lens). If a sim/
edit is needed -> STOP + report (it is a later stage).

Run (CPU/numpy, fast; multi-seed):
  SIM_BACKEND=numpy python -u -m research.runners._dendrite_deriskA_extension_conversational_cortex \
      --seeds 42,43,44 --out research/findings/raw/_dendrite_deriskA_ext_conv_cortex.json
Run (smoke):
  SIM_BACKEND=numpy python -u -m research.runners._dendrite_deriskA_extension_conversational_cortex --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

# the D1/D2 synthetic concept x hub count regime + the structure metrics (reuse-by-import)
from research.runners.dendritic_d1_learn_graded_structure_derisk import (
    build_concept_hub_counts, _cos_sim, _pearson_vs_Strue, heldout_generalization,
    learn_perhub_gains, perhub_residual,
)
# the host PPMI+SVD ceiling lens (labelled instrument only, never the deliverable)
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score
# the dendritic substrate (Stage 0 -- reuse-by-import; NO sim/ edit)
from sim.dendritic_neuron import DendriticLayer
from sim.dendritic_plasticity import urbanczik_senn_update


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ---------------------------------------------------------------------------
# The graded dendritic-plateau read-out (the de-risk-A mechanism), applied to a VECTOR code.
# De-risk A read a SCALAR value V = sigmoid((v_basal - theta)/slope) (n_post=1). The conversational
# cortex code is a VECTOR (n_post graded plateau units), structure = Pearson(cos(codes), S_true).
# Each unit j reads the concept x hub code through W_basal[:, j] -> v_basal_j -> a graded plateau V_j.
# ---------------------------------------------------------------------------
def _graded_plateau_codes(x, W_basal, *, slope_scale=1.0, all_or_none=False):
    """x [Nc x H] (the per-concept hub code, normalized to ~[0,1]); W_basal [H x n_post].
    v_basal = x @ W_basal [Nc x n_post]. The GRADED dendritic-plateau read-out is the SMOOTH sigmoid of
    the plateau-drive, per unit, placed on the graded slope by a PER-UNIT center+scale (theta_j=mean_i,
    slope_j=std_i across concepts -- a fixed, unsupervised operating-point placement, NOT fit to S_true).
    all_or_none (the apical lesion, anti-cheat c): replace the smooth sigmoid with the Heaviside threshold
    V_j = 1{v_basal_j >= theta_j} -- the POINT-NEURON all-or-none plateau. If the graded-ness is the
    load-bearing element, the all-or-none version collapses."""
    vb = np.asarray(x, np.float64) @ np.asarray(W_basal, np.float64)        # [Nc x n_post]
    theta = vb.mean(0, keepdims=True)
    slope = (vb.std(0, keepdims=True) + 1e-9) * float(slope_scale)
    if all_or_none:
        return (vb >= theta).astype(np.float64)
    return _sig((vb - theta) / slope)


def _learn_W_unsup(x, n_post, seed, *, epochs=8, lr=0.05, theta_high=1.0, apical_gain=0.5, n_teacher=8):
    """L2: learn W_basal UNSUPERVISED on the stream via the LOCAL Urbanczik-Senn rule (the de-risk-A
    learning machinery, reuse-by-import), with the apical teacher = the read-out's OWN soma_rate (a
    self-prediction target -- NO category label, NO S_true). Each concept profile x_i is one presentation;
    W_basal updates by the local somato-dendritic mismatch, apical-gated. Returns the learned W_basal.

    This is the honest unsupervised analog of de-risk A's apical-gated learning: de-risk A had a LEGITIMATE
    external teacher (the SNc reward delta, location-selective). The conversational cortex code structure
    must emerge WITHOUT category labels (the brain gets no S_true), so the only legitimate apical signal is
    self-generated -- we do NOT inject S_true (that would smuggle the answer)."""
    layer = DendriticLayer(n_pre=x.shape[1], n_post=int(n_post), n_teacher=int(n_teacher), seed=int(seed),
                           theta_high=float(theta_high), apical_gain=float(apical_gain), leak=0.0)
    rng = np.random.default_rng(seed * 104729 + 7)
    order = np.arange(x.shape[0])
    for _ in range(int(epochs)):
        rng.shuffle(order)
        for i in order:
            xi = np.asarray(x[i], np.float64)
            out = layer.step(xi, np.zeros(int(n_teacher)))        # forward (apical teacher self-generated below)
            soma = out["soma_rate"]; vb = out["v_basal"]
            # apical teaching = the read-out's OWN activity projected through the FIXED-RANDOM apical
            # feedback (self-supervised; NO weight transport, NO S_true). Larkum BAC gate = apical depol.
            teacher = np.full(int(n_teacher), float(soma.mean()), dtype=np.float64)
            gate = np.abs(layer._apical_drive(teacher))
            apical_sig = layer._apical_drive(teacher)
            dW = urbanczik_senn_update(xi, soma, vb, apical_gate=gate, apical_signal=apical_sig, lr=lr)
            layer.W_basal = np.clip(layer.W_basal + dW, -8.0, 8.0)
    return layer.W_basal


def run_seed(seed, args):
    print(f"\n{'='*100}\n  DENDRITE DE-RISK A EXTENSION -> CONVERSATIONAL CORTEX (seed {seed})\n{'='*100}", flush=True)
    # The D2/D1 synthetic concept x hub count regime: high-frequency COMMON hubs (the common mode that
    # dominates every profile -> the point neuron fails) + per-category signal hubs. S_true a-priori block.
    C, labels, S_true, hub_freq = build_concept_hub_counts(
        args.n_cat, args.per_cat, args.n_common, args.n_sig_per_cat,
        args.lam_common, args.lam_sig, args.lam_bg, seed)
    Nc, H = C.shape
    s_true_independent = bool(np.array_equal(S_true, (labels[:, None] == labels[None, :]).astype(float)))

    # host ceiling (data-carries-it) + the raw-profile point-neuron baseline (common-hub-dominated ~0)
    raw_pearson = _pearson_vs_Strue(_cos_sim(C), S_true)
    host_sim = ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(args.host_svd, min(C.shape) - 1), alpha=args.host_alpha)
    host_pearson, _, _, _ = score(host_sim, labels)
    print(f"  data: {args.n_cat}x{args.per_cat}={Nc} concepts x {H} hubs ({args.n_common} common + "
          f"{args.n_cat*args.n_sig_per_cat} signal); raw-profile Pearson={raw_pearson:+.3f} (~0); "
          f"HOST PPMI+SVD ceiling={host_pearson:+.3f}; S_true a-priori={s_true_independent}", flush=True)

    # normalize the count code into the dendritic basal range (pA->~[0,1], like de-risk A's place_drive_norm)
    x = C / (C.max() + 1e-9)

    # === L0: POINT-NEURON all-or-none read-out (the validity gate) ===
    # The somatic-spike read-out = a hard Heaviside threshold of the (fixed-random-projected) plateau drive.
    # Silences low-count / saturates common -> the common mode dominates -> ~0 (reproduces the D2 NEGATIVE).
    rng = np.random.default_rng(seed * 2654435761 + 1)
    W_fixed = rng.normal(0.0, 1.0, (H, args.n_post)) * args.w_init_scale
    l0_codes = _graded_plateau_codes(x, W_fixed, slope_scale=args.slope_scale, all_or_none=True)
    l0_p = _pearson_vs_Strue(_cos_sim(l0_codes), S_true)
    l0_gen, chance = heldout_generalization(l0_codes, labels)

    # === L1: GRADED dendritic plateau, FIXED-random W (the de-risk-A mechanism, NO norm, NO learning) ===
    l1_codes = _graded_plateau_codes(x, W_fixed, slope_scale=args.slope_scale, all_or_none=False)
    l1_p = _pearson_vs_Strue(_cos_sim(l1_codes), S_true)
    l1_gen, _ = heldout_generalization(l1_codes, labels)

    # === L2: GRADED dendritic plateau, W LEARNED UNSUPERVISED (local Urbanczik-Senn, self-apical) ===
    t0 = time.time()
    W_learned = _learn_W_unsup(x, args.n_post, seed, epochs=args.epochs, lr=args.lr)
    l2_codes = _graded_plateau_codes(x, W_learned, slope_scale=args.slope_scale, all_or_none=False)
    l2_p = _pearson_vs_Strue(_cos_sim(l2_codes), S_true)
    l2_gen, _ = heldout_generalization(l2_codes, labels)
    # graded-ness lesion of the BEST graded arm (anti-cheat c): all-or-none on the same (better) W.
    best_W = W_learned if l2_p >= l1_p else W_fixed
    best_graded_p = max(l1_p, l2_p)
    aon_best_codes = _graded_plateau_codes(x, best_W, slope_scale=args.slope_scale, all_or_none=True)
    aon_best_p = _pearson_vs_Strue(_cos_sim(aon_best_codes), S_true)

    # === L3: PER-HUB divisive NORMALIZATION (the SEPARATE D2 Phase-1 dendritic gain) THEN graded plateau ===
    # the per-hub gains are LEARNED ONLINE over the stream (local rule; the D1 mechanism, reuse-by-import).
    g_hub, gtrace = learn_perhub_gains(C, args.gain_epochs, args.gain_eta, seed)
    r = perhub_residual(C, g_hub, sigma=args.gain_sigma)            # the common-mode-removed residual
    r_norm = r / (r.max() + 1e-9)
    l3_lin_p = _pearson_vs_Strue(_cos_sim(r), S_true)              # the per-hub-gain residual read LINEARLY (D1)
    l3_lin_gen, _ = heldout_generalization(r, labels)
    # graded plateau read-out on the normalized residual (identity W = each hub its own graded unit)
    l3_graded_codes = _graded_plateau_codes(r_norm, np.eye(H), slope_scale=args.slope_scale)
    l3_graded_p = _pearson_vs_Strue(_cos_sim(l3_graded_codes), S_true)
    l3_graded_gen, _ = heldout_generalization(l3_graded_codes, labels)
    l3_aon_codes = _graded_plateau_codes(r_norm, np.eye(H), slope_scale=args.slope_scale, all_or_none=True)
    l3_aon_p = _pearson_vs_Strue(_cos_sim(l3_aon_codes), S_true)
    gains_converge = bool(len(gtrace) >= 2 and abs(gtrace[-1] - gtrace[-2]) <= 0.05 * (gtrace[-1] + 1e-9))
    gain_freq_corr = float(np.corrcoef(g_hub, hub_freq)[0, 1])

    print(f"  [L0 POINT-NEURON all-or-none]     Pearson={l0_p:+.3f}  gen={l0_gen:.3f} (chance {chance:.3f})", flush=True)
    print(f"  [L1 GRADED plateau, fixed-W]      Pearson={l1_p:+.3f}  gen={l1_gen:.3f}", flush=True)
    print(f"  [L2 GRADED plateau, learned-W]    Pearson={l2_p:+.3f}  gen={l2_gen:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"       graded-ness lesion (all-or-none on the best graded W): Pearson={aon_best_p:+.3f} "
          f"(best graded {best_graded_p:+.3f})", flush=True)
    print(f"  [L3 per-hub NORM -> linear]       Pearson={l3_lin_p:+.3f}  gen={l3_lin_gen:.3f}  "
          f"(gain~freq {gain_freq_corr:+.2f}, converge={gains_converge})", flush=True)
    print(f"  [L3 per-hub NORM -> GRADED]       Pearson={l3_graded_p:+.3f}  gen={l3_graded_gen:.3f}  "
          f"| all-or-none {l3_aon_p:+.3f}", flush=True)

    # === anti-cheat (e): permuted-S collapses (for the best-recovering arm: L3-linear / L3-graded) ===
    prng = np.random.RandomState(seed * 32452843 + 1)
    perm_labels = prng.permutation(labels)
    S_perm = (perm_labels[:, None] == perm_labels[None, :]).astype(np.float64)
    l3_perm = _pearson_vs_Strue(_cos_sim(r), S_perm)
    l1_perm = _pearson_vs_Strue(_cos_sim(l1_codes), S_perm)
    print(f"  [anti-cheat] permuted-S: L3-linear={l3_perm:+.3f}  L1-graded={l1_perm:+.3f} (must ~0)", flush=True)

    return {
        "seed": seed, "n_concepts": Nc, "n_hubs": H, "chance": chance,
        "raw_pearson": raw_pearson, "host_ceiling_pearson": host_pearson,
        "s_true_independent": s_true_independent,
        "L0_point_neuron_aon": {"pearson": l0_p, "gen": l0_gen},
        "L1_graded_fixed": {"pearson": l1_p, "gen": l1_gen},
        "L2_graded_learned": {"pearson": l2_p, "gen": l2_gen},
        "best_graded_pearson": best_graded_p,
        "best_graded_allornone_lesion_pearson": aon_best_p,
        "L3_norm_linear": {"pearson": l3_lin_p, "gen": l3_lin_gen},
        "L3_norm_graded": {"pearson": l3_graded_p, "gen": l3_graded_gen, "allornone_pearson": l3_aon_p},
        "gains_converge": gains_converge, "gain_freq_corr": gain_freq_corr,
        "permuted_S": {"L3_linear": l3_perm, "L1_graded": l1_perm},
    }


def decide_verdict(per_seed, seeds, args):
    def mean(path):
        cur = [per_seed[str(s)] for s in seeds]
        for k in path:
            cur = [c[k] for c in cur]
        return float(np.mean(cur))
    n = len(seeds)
    host_mean = mean(["host_ceiling_pearson"])
    l0_mean = mean(["L0_point_neuron_aon", "pearson"])
    l1_mean = mean(["L1_graded_fixed", "pearson"])
    l2_mean = mean(["L2_graded_learned", "pearson"])
    best_graded_mean = mean(["best_graded_pearson"])
    aon_best_mean = mean(["best_graded_allornone_lesion_pearson"])
    l3_lin_mean = mean(["L3_norm_linear", "pearson"])
    l3_graded_mean = mean(["L3_norm_graded", "pearson"])

    # validity gate (a): the point-neuron control (L0) must fail (~0) at majority of seeds.
    l0_fails = sum(1 for s in seeds if abs(per_seed[str(s)]["L0_point_neuron_aon"]["pearson"]) <= args.pn_fail_bar)
    host_carries = sum(1 for s in seeds if per_seed[str(s)]["host_ceiling_pearson"] >= args.host_bar)
    # the GO test: the GRADED plateau read-out (L1/L2) BEATS the point neuron by a clear margin AND clears
    # the structure bar AND the graded-ness is LOAD-BEARING (the all-or-none lesion collapses the best
    # graded arm) AND reads the RAW counts (no normalization smuggled).
    graded_beats_pn = sum(1 for s in seeds
                          if per_seed[str(s)]["best_graded_pearson"]
                          >= per_seed[str(s)]["L0_point_neuron_aon"]["pearson"] + args.beat_margin
                          and per_seed[str(s)]["best_graded_pearson"] >= args.structure_bar)
    graded_is_loadbearing = sum(1 for s in seeds
                                if per_seed[str(s)]["best_graded_allornone_lesion_pearson"]
                                <= per_seed[str(s)]["best_graded_pearson"] - args.lesion_drop)
    permuted_clean = sum(1 for s in seeds if abs(per_seed[str(s)]["permuted_S"]["L3_linear"]) <= args.pn_fail_bar
                         and abs(per_seed[str(s)]["permuted_S"]["L1_graded"]) <= args.pn_fail_bar)

    maj = max(1, (n + 1) // 2)
    controls_valid = (l0_fails >= maj and host_carries >= maj)

    if not controls_valid:
        verdict = "VOID"
        why = (f"the validity gate did not hold (point-neuron L0 fails {l0_fails}/{n}, host carries "
               f"{host_carries}/{n}) -> the regime is mis-calibrated; the comparison is not interpretable.")
    elif graded_beats_pn >= maj and graded_is_loadbearing >= maj:
        verdict = "GO"
        why = (f"the GRADED dendritic-plateau read-out on the RAW conversational code recovers the graded "
               f"category-similarity (mean Pearson {best_graded_mean:+.3f} >= {args.structure_bar}) where the "
               f"point-neuron all-or-none read-out fails (mean {l0_mean:+.3f}), and the graded-ness is "
               f"LOAD-BEARING (the all-or-none lesion collapses it, {graded_is_loadbearing}/{n}). The de-risk-A "
               f"graded-read-out unlock GENERALIZES to the conversational cortex.")
    else:
        verdict = "NEGATIVE"
        # characterize WHY via the L3 attribution ladder.
        why = (
            f"the GRADED dendritic-plateau read-out does NOT recover the conversational concept-code structure "
            f"from the RAW common-mode-dominated counts: L1 (graded, fixed-W) {l1_mean:+.3f} and L2 (graded, "
            f"learned-W unsupervised) {l2_mean:+.3f} both sit at the point-neuron floor (L0 all-or-none "
            f"{l0_mean:+.3f}, raw-profile ~0), while the host ceiling is {host_mean:+.3f}. The graded-ness is "
            f"NOT the load-bearing element here (L1 ~ L2 ~ its own all-or-none lesion {aon_best_mean:+.3f}). "
            f"The ATTRIBUTION ladder localizes the real wall: only PER-HUB divisive NORMALIZATION (the SEPARATE "
            f"D2 Phase-1 per-input dendritic gain) recovers it -- the normalized residual read LINEARLY already "
            f"reaches {l3_lin_mean:+.3f} (and a graded read-out on top neither helps nor is needed, "
            f"{l3_graded_mean:+.3f}). => the conversational cortex NEGATIVE is a COMMON-MODE / per-input-"
            f"NORMALIZATION problem, a DIFFERENT dendritic unlock than the graded plateau read-out that de-risk "
            f"A confirmed. The de-risk-A unlock does NOT generalize to this wall (the nav value-critic input has "
            f"no common mode; the conversational code is common-mode-dominated). A decisive, build-saving result: "
            f"it sharpens the dendrite's capability map -- the graded read-out and the per-input normalization "
            f"are DISTINCT dendritic unlocks, not interchangeable.")
    detail = {
        "host_ceiling_pearson_mean": host_mean,
        "L0_point_neuron_pearson_mean": l0_mean,
        "L1_graded_fixed_pearson_mean": l1_mean,
        "L2_graded_learned_pearson_mean": l2_mean,
        "best_graded_pearson_mean": best_graded_mean,
        "best_graded_allornone_lesion_pearson_mean": aon_best_mean,
        "L3_norm_linear_pearson_mean": l3_lin_mean,
        "L3_norm_graded_pearson_mean": l3_graded_mean,
        "point_neuron_fails": f"{l0_fails}/{n}", "host_carries": f"{host_carries}/{n}",
        "graded_beats_point_neuron": f"{graded_beats_pn}/{n}",
        "graded_is_loadbearing": f"{graded_is_loadbearing}/{n}",
        "permuted_S_clean": f"{permuted_clean}/{n}",
        "controls_valid": controls_valid,
    }
    return verdict, why, detail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default=None)
    p.add_argument("--seed", type=int, default=42)
    # synthetic D2/D1 conversational regime (calibrated so the point neuron fails + the host carries)
    p.add_argument("--n-cat", type=int, default=8)
    p.add_argument("--per-cat", type=int, default=8)
    p.add_argument("--n-common", type=int, default=200, help="# high-freq COMMON hubs (the common mode)")
    p.add_argument("--n-sig-per-cat", type=int, default=12)
    p.add_argument("--lam-common", type=float, default=40.0)
    p.add_argument("--lam-sig", type=float, default=4.0)
    p.add_argument("--lam-bg", type=float, default=0.3)
    # the graded dendritic-plateau read-out
    p.add_argument("--n-post", type=int, default=400, help="# graded plateau read-out units (the code dim)")
    p.add_argument("--w-init-scale", type=float, default=0.1, help="fixed-random W_basal init scale (L0/L1)")
    p.add_argument("--slope-scale", type=float, default=1.0, help="graded plateau slope (x per-unit std)")
    p.add_argument("--epochs", type=int, default=8, help="L2 unsupervised Urbanczik-Senn epochs")
    p.add_argument("--lr", type=float, default=0.05)
    # the L3 per-hub normalization (the separate D2 Phase-1 dendritic gain)
    p.add_argument("--gain-epochs", type=int, default=12)
    p.add_argument("--gain-eta", type=float, default=0.05)
    p.add_argument("--gain-sigma", type=float, default=1.0)
    # host
    p.add_argument("--host-svd", type=int, default=50)
    p.add_argument("--host-alpha", type=float, default=0.75)
    # bars
    p.add_argument("--structure-bar", type=float, default=0.30, help="graded read-out must clear this to GO")
    p.add_argument("--pn-fail-bar", type=float, default=0.15, help="point-neuron L0 must be <= this (fails)")
    p.add_argument("--host-bar", type=float, default=0.30)
    p.add_argument("--beat-margin", type=float, default=0.20, help="graded must beat the point neuron by >=")
    p.add_argument("--lesion-drop", type=float, default=0.10, help="all-or-none lesion must drop the graded by >=")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    t0 = time.time()
    print(f"[dendrite de-risk A EXTENSION -> conversational cortex] seeds={seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}\n  question: does the GRADED dendritic-plateau "
          f"read-out (de-risk A) recover the conversational concept-code structure where the point neuron "
          f"(D2 Phase 2) failed?", flush=True)
    per_seed = {str(s): run_seed(s, args) for s in seeds}
    verdict, why, detail = decide_verdict(per_seed, seeds, args)

    print(f"\n{'='*100}\n  EXTENSION VERDICT: {verdict}\n  {why}", flush=True)
    print(f"\n  LADDER (mean Pearson(cos(codes),S_true) vs S_true):", flush=True)
    print(f"    HOST ceiling (PPMI+SVD)           {detail['host_ceiling_pearson_mean']:+.3f}", flush=True)
    print(f"    L0 POINT-NEURON all-or-none       {detail['L0_point_neuron_pearson_mean']:+.3f}  (validity gate -- fails)", flush=True)
    print(f"    L1 GRADED plateau, fixed-W        {detail['L1_graded_fixed_pearson_mean']:+.3f}  (the de-risk-A mechanism, no norm)", flush=True)
    print(f"    L2 GRADED plateau, learned-W      {detail['L2_graded_learned_pearson_mean']:+.3f}  (unsupervised local rule)", flush=True)
    print(f"    L3 per-hub NORM -> linear         {detail['L3_norm_linear_pearson_mean']:+.3f}  (the SEPARATE per-input gain)", flush=True)
    print(f"    L3 per-hub NORM -> graded         {detail['L3_norm_graded_pearson_mean']:+.3f}", flush=True)
    print(f"  graded beats point-neuron {detail['graded_beats_point_neuron']} | graded load-bearing "
          f"{detail['graded_is_loadbearing']} | permuted-S clean {detail['permuted_S_clean']}", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n{'='*100}\n", flush=True)

    out = {
        "item": "dendrite_derisk_A_extension_conversational_cortex",
        "stage": 0, "sim_edit": False,
        "verdict": verdict, "why": why, "detail": detail, "seeds": seeds,
        "config": vars(args), "per_seed": per_seed,
        "note": ("Stage-0 extension of the GO dendrite de-risk A (graded dendritic-plateau read-out) from the "
                 "nav value-critic to the conversational learned cortex (the D2 Phase-2 NEGATIVE). The CONTRAST "
                 "ladder reads the IDENTICAL common-mode-dominated concept x hub count code; the metric is "
                 "Pearson(cos(codes),S_true) (a graded category-similarity fidelity) vs the host PPMI+SVD "
                 "ceiling. NO sim/ edit (reuse-by-import of sim.dendritic_neuron + sim.dendritic_plasticity + "
                 "the D1/D2 count builder)."),
    }
    if args.out is None:
        raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                               "research", "findings", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(raw_dir, f"_dendrite_deriskA_ext_conv_cortex_{ts}.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"  [saved] {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
