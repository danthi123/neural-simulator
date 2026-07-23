"""nav close-out R5 RANK-1 CLOSURE — the brain's OWN LEARNED SPIKING value drives a value-driven choice
(retire R5-R1a's HOST value stand-in `build_concept_value` / `make_option_values`).

THE GAP THIS CLOSES (scoping `2026-07-23-value-critic-closure-scoping.md`, RANK 1):
R5-R1a proved a spiking value-driven WTA is load-bearing BY ITS FUNCTION (6/6 GO) — BUT the VALUE scalar fed into
the WTA was `make_option_values`, "a CPU stand-in for the merged-bridge spiking SNc/striosome_value critic, seeded
from a reward-tagging RNG." So the DECISION organ was spiking, but the VALUE it read was HOST-injected. This runner
CLOSES that residual: it trains the REAL spiking `striosome_value` critic (DA-gated STDP on the merged one-brain
bridge — neurons + synapses), READS its firing rate as V_i (a real `cp_firing_states` read), and drives the SAME
spiking value-WTA's drift from that LEARNED spiking V. ⇒ *spiking decision + SPIKING value*, no host value tag in
the decision path.

THE BUILD (reuse-by-import; NO `sim/` edit):
  - the LEARNED spiking value: `_merged_navcritic_valuetrain.{build_merged, run_value_train, _critic_rate_via_afferent,
    lesion_gabab, check_moat}` — the ported g11_bg_runner._run_place_value_training DA-gated STDP value-train grows
    the plastic vs_place_context->striosome_value weight until the critic fires MORE at the rewarded goal place; the
    critic's firing rate when a place-cue is presented IS V(cue).
  - the spiking DECISION: `_navcloseout_R5_value_driven_choice.{SpikingValueChoice, _drives, make_salience_bias}` —
    the Wang-2002 biased-competition WTA (a neural pool's FIRING = the choice, NOT a host argmax). drift(option_i) =
    base + value_gain * V_i + salience_gain * SALIENCE_i (catalog O.19/C.34: value modulates the accumulator DRIFT).
    Here V_i is the LEARNED spiking critic rate (raw Hz), not a host tag.

THE CUES: K place-cues along the diagonal from the trained GOAL (near, high learned V) out to its far reflection
(low learned V). After value-train, the critic's learned place field gives a graded V that DECREASES with distance
to the goal -> the choice must pick the cue CLOSEST to the goal (highest LEARNED value). K=2 is the decisive RANK-1
(near vs far); K>2 is the RANK-3 graded generalization.

THE 5 GATES (validate-by-function; GO = all >= 5/6 seeds):
  (G_HEADLINE) value-ON picks the higher-LEARNED-V cue >> chance.
  (G_LESION, drive-level matched) replace each cue's LEARNED V with the MEAN learned V (remove the gradient, hold
      the operating point) -> the choice collapses to chance. [the R4 fix; here the lesion HAS something to collapse]
  (G_UNTRAINED, the NEW anti-cheat R5-R1a LACKED — the LOAD-BEARING one) read V from the UNTRAINED critic (init
      weight, no value-train) -> no learned gradient (the position-blind A1 floor) -> the trained-cue advantage
      VANISHES (choice at/ below chance). THIS is what makes the substrate's LEARNING load-bearing (the value is the
      brain's LEARNED spiking V, not a host tag or a wired-in prior).
  (G_DISCRIM, validate-by-function) EQUAL-value cues (same V on all options) -> the value carries no gradient -> the
      lesion is NEUTRAL (intact-vs-lesion trial-by-trial choice agreement ~1.0). Value-SPECIFIC, not a lesion artifact.
  (G_PERMUTE) permute which option gets which LEARNED V (deterministic permutation-AVERAGE) -> advantage -> chance.
  (+ NON-CIRCULAR) corr(learned V, salience) ~ 0 -> the choice is driven by the learned value, not a relabeled salience.
  (+ MOAT) the WTA decision organ has NO RF/conversational slices -> array-disjoint -> the no-confab moat is preserved
      by construction; check_moat re-asserted on the merged critic agent (the DA scope=all broadcast must not perturb
      the frozen conversational slice).

DISCIPLINE: CPU/numpy (a multi-day GPU training run is live — do NOT contend). The merged bridge value-train is the
expensive part (~4 min/seed CPU); the WTA arms run on a separate tiny bridge (fast). Read the runner's OWN verdict;
an honest NEGATIVE is a first-class result. NO git.

Run:
  # CPU smoke (well-formedness, no long train):
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R5b_learned_value_choice --smoke --seed 42
  # 1-seed calibration (build+train+arms, inspect regime):
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R5b_learned_value_choice --seeds 42 \
      --value-train-trials 40 --out research/findings/raw/navcloseout_R5/R5b_seed42.json
  # the 6-seed gate:
  SIM_BACKEND=numpy python -m research.runners._navcloseout_R5b_learned_value_choice \
      --seeds 42,43,44,100,101,102 --value-train-trials 40 \
      --out research/findings/raw/navcloseout_R5/R5b_learned_value_6seed.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # the merged bridge + WTA are the numpy-CPU brain slices

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the LEARNED spiking value critic (the merged one-brain bridge + DA-gated STDP value-train)
from research.runners import _merged_navcritic_valuetrain as VT   # noqa: E402
# reuse-by-import: the spiking value-WTA DECISION organ (a neural pool's firing) + the drive builder + salience
from research.runners._navcloseout_R5_value_driven_choice import (  # noqa: E402
    SpikingValueChoice, _drives, make_salience_bias,
)


# ── Normalization of the LEARNED spiking V (raw Hz) into the WTA drift ─────────────────────────────
# We feed the RAW learned critic rate (Hz) into R5's `_drives(values, ..., value_gain_pA=VALUE_HZ_GAIN)`.
# VALUE_HZ_GAIN is a FIXED pA-per-Hz constant applied IDENTICALLY to the untrained AND trained reads (and every
# arm) — it is NOT re-fit per seed, so it CANNOT smuggle the answer: an untrained critic with flat/anti-goal V
# produces flat/anti-goal drives regardless of the gain (the gate that the LEARNING is load-bearing holds for any
# gain). The gain only sets how strongly a REAL learned gradient drives the WTA vs the salience noise.
VALUE_HZ_GAIN_PA = 5.0     # pA per Hz of learned critic firing (trained sep ~15-30 Hz -> ~75-150 pA, R5's regime)
SPEAK_BASE_PA = 70.0       # R5 option-pool base drive
SALIENCE_GAIN_PA = 40.0    # R5 salience -> drift gain (the value-INDEPENDENT 'default pull'; the lesion baseline)


def _cue_places(k):
    """K place-cues along the diagonal from the trained GOAL (near, high learned V) to its far reflection (low V).
    cue 0 = the GOAL (highest LEARNED value after training); cue K-1 = the far reflection (lowest). For K=2 this is
    exactly (near, far). The 'true value ranking' = by distance to the goal (closer == higher learned V)."""
    near = np.asarray(VT.GOAL, dtype=float)
    far = np.asarray(VT._far_of(VT.GOAL), dtype=float)
    ts = np.linspace(0.0, 1.0, int(k))
    return [tuple((near + t * (far - near)).tolist()) for t in ts]


def _read_V(bridge, idx, prefs, cues, xp, cfg):
    """Read the critic firing rate (Hz) at each cue place -> V_i. NO direct critic drive, NO teacher: only the
    perceived place code into the place afferents (the faithful deployment read). Frozen (value_input closed,
    reward_learning_rate 0) so the read does NOT grow the weight."""
    return [float(VT._critic_rate_via_afferent(bridge, idx, prefs, gx, gy, xp, cfg)) for (gx, gy) in cues]


def _score_arm(accumulator, V_vec, seed, n_trials, *, lesion, equal, permute, salience_gain=SALIENCE_GAIN_PA,
               true_best_from=None):
    """Run one arm over n_trials of the K-option value-WTA choice. Each trial randomly ASSIGNS the learned-V vector
    to the option pools (so the higher-value option is not fixed to one pool) + draws an INDEPENDENT salience per
    option. drift = _drives(values=assigned V (Hz), salience, value_gain=VALUE_HZ_GAIN, lesion_value=lesion). The
    DECISION = the winning pool's spiking (SpikingValueChoice.decide). CORRECT = pick the option assigned the TRUE
    highest-value cue.

    V_vec         : the per-cue learned V (Hz), indexed by cue (cue 0 = goal = highest trained V).
    true_best_from: if given (the UNTRAINED arm), score correctness against the cue that is TRUE-highest in THIS
                    reference (the trained V) — i.e. 'does the untrained critic pick the to-be-rewarded cue?'. If
                    None, the true-best is argmax(V_vec) itself (the intact/trained/lesion arms).
    equal=True    : all options get the SAME value (V_vec.mean()) -> no gradient (the G_DISCRIM control).
    Returns (accuracy, choices[np], true_bests[np], mean_margin)."""
    n_opt = accumulator.n_options
    Vv = np.asarray(V_vec, dtype=float)
    ref = np.asarray(true_best_from, dtype=float) if true_best_from is not None else Vv
    rng = np.random.default_rng(int(seed) * 7919 + 3)

    n_correct = 0.0
    choices, true_bests, margins = [], [], []
    for t in range(int(n_trials)):
        assign = rng.permutation(n_opt)         # which cue -> which option pool (random per trial)
        # the values presented to the pools (assigned), + which pool is the TRUE best (highest REF value cue)
        vals = Vv[assign].copy()
        if equal:
            vals = np.full(n_opt, float(Vv.mean()), dtype=float)
        true_best_cue = int(np.argmax(ref))                       # the cue with the highest reference (learned) value
        true_best_pool = int(np.where(assign == true_best_cue)[0][0])
        salience = make_salience_bias(n_opt, seed=int(seed) * 1000 + t)
        if permute:
            # G_PERMUTE (deterministic permutation-AVERAGE): average correctness over ALL value<->pool permutations
            # of this trial. A value-driven choice follows the permuted drives, so correctness averages to chance
            # EXACTLY by construction; only a fixed-pool bias would keep it high.
            perms = list(itertools.permutations(range(n_opt)))
            tc = 0.0; last_ch = 0; last_m = 0.0
            for perm in perms:
                pv = vals[list(perm)]
                drives = _drives(pv, salience, speak_base_pA=SPEAK_BASE_PA, value_gain_pA=VALUE_HZ_GAIN_PA,
                                 salience_gain_pA=salience_gain, lesion_value=lesion)
                ch, _, m = accumulator.decide(drives)
                if ch == true_best_pool:
                    tc += 1.0
                last_ch, last_m = int(ch), m
            n_correct += tc / max(1, len(perms))
            choices.append(last_ch); true_bests.append(true_best_pool); margins.append(last_m)
        else:
            drives = _drives(vals, salience, speak_base_pA=SPEAK_BASE_PA, value_gain_pA=VALUE_HZ_GAIN_PA,
                             salience_gain_pA=salience_gain, lesion_value=lesion)
            ch, _, m = accumulator.decide(drives)
            choices.append(int(ch)); true_bests.append(true_best_pool); margins.append(m)
            if ch == true_best_pool:
                n_correct += 1.0
    return (n_correct / max(1, int(n_trials)), np.array(choices), np.array(true_bests),
            float(np.mean(margins)) if margins else 0.0)


def _value_salience_corr(V_vec, seed, n_trials, n_opt):
    """Non-circularity: correlate the (assigned) LEARNED value with the (independent) salience over all
    (trial, option) pairs. ~0 -> the choice's value-dependence is not a relabeled salience."""
    rng = np.random.default_rng(int(seed) * 7919 + 3)
    Vv = np.asarray(V_vec, dtype=float)
    all_v, all_s = [], []
    for t in range(int(n_trials)):
        assign = rng.permutation(n_opt)
        all_v.extend(Vv[assign].tolist())
        all_s.extend(make_salience_bias(n_opt, seed=int(seed) * 1000 + t).tolist())
    av, as_ = np.array(all_v), np.array(all_s)
    return float(np.corrcoef(av, as_)[0, 1]) if av.std() > 0 and as_.std() > 0 else 0.0


def run_seed(seed, a, accumulator):
    """One seed: build the merged bridge, read UNTRAINED V, value-train the critic, read TRAINED V, then run the
    5-gate value-WTA choice on the LEARNED spiking V. Returns the per-seed row."""
    from sim.backend import get_backend
    xp, backend = get_backend()
    t0 = time.time()
    print(f"\n{'='*90}\n[R5b seed={seed}] backend={backend}  building the merged one-brain bridge...", flush=True)

    b, _ = VT.build_merged(seed, convergent_upstate=True)
    cfg = b.core_config
    cfg.gabab_conductance_max = float(VT.GIRK_CAP)
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = 100.0
    cfg.homeostasis_threshold_adapt_rate = 0.0
    # FREEZE the value arm for every read (the merged default reward_learning_rate=0.01 + value_input open would
    # grow the weight during a read -> the 'untrained' read would not be untrained). The value-train re-opens it.
    _vt_lr = float(cfg.reward_learning_rate) if (cfg.reward_learning_rate and cfg.reward_learning_rate > 0) else 0.01
    cfg.reward_learning_rate = 0.0
    cfg.current_reward_signal = 0.0
    b.set_plasticity_gate("value_input", 0.0)

    idx = {nm: VT._idx(b, nm, xp) for nm in (VT.SNC, VT.CRITIC, VT.REWARD_US, VT.CRITIC_AFFERENT, VT.UPSTATE_AFFERENT)}
    prefs = VT._vs_place_prefs(int(len(VT._host(idx[VT.CRITIC_AFFERENT]))))
    cues = _cue_places(int(a.n_options))
    near = VT.GOAL; far = VT._far_of(VT.GOAL)
    w_init = float(VT._mean_afferent_weight(b, idx))
    print(f"[R5b seed={seed}] build {time.time()-t0:.1f}s  cues={[tuple(round(c,1) for c in cc) for cc in cues]} "
          f"w_init={w_init:.3f}", flush=True)

    # ── ANTI-CHEAT (G_UNTRAINED): read V from the UNTRAINED critic (no learned gradient) ──
    V_untr = _read_V(b, idx, prefs, cues, xp, cfg)
    print(f"[R5b seed={seed}] UNTRAINED V(Hz) per cue = {[round(v,1) for v in V_untr]} "
          f"(goal={V_untr[0]:.1f} far={V_untr[-1]:.1f})", flush=True)

    # ── VALUE-TRAIN: grow vs_place_context->striosome_value via pair-then-reward DA-gated STDP (learn V) ──
    cfg.reward_learning_rate = _vt_lr
    tvt = time.time()
    vt = VT.run_value_train(b, idx, prefs, xp, cfg, near=near, far=far, trials=int(a.value_train_trials),
                            verbose=a.verbose)
    cfg.reward_learning_rate = 0.0
    cfg.current_reward_signal = 0.0
    b.set_plasticity_gate("value_input", 0.0)
    print(f"[R5b seed={seed}] VALUE-TRAIN {time.time()-tvt:.1f}s  w {vt['w_near_pre']:.3f}->{vt['w_near_post']:.3f} "
          f"({vt['w_grew']:.2f}x)  tonic_frac={vt['tonic_frac']:.4f}", flush=True)

    # ── read the TRAINED V (the learned spiking value the WTA will read) ──
    V_train = _read_V(b, idx, prefs, cues, xp, cfg)
    print(f"[R5b seed={seed}] TRAINED   V(Hz) per cue = {[round(v,1) for v in V_train]} "
          f"(goal={V_train[0]:.1f} far={V_train[-1]:.1f}  grade goal/far={V_train[0]/max(V_train[-1],1e-3):.2f})",
          flush=True)

    # ── the WTA choice arms (on the SEPARATE tiny spiking decision bridge; fast) ──
    n_opt = int(a.n_options)
    chance = 1.0 / n_opt
    # INTACT (trained learned V):
    acc_intact, _, _, m_intact = _score_arm(accumulator, V_train, seed, a.n_trials, lesion=False, equal=False,
                                            permute=False)
    # LESION (drive-level matched: pin learned V to its mean -> remove the gradient, hold the op-point):
    acc_lesion, _, _, m_lesion = _score_arm(accumulator, V_train, seed, a.n_trials, lesion=True, equal=False,
                                            permute=False)
    # UNTRAINED (the SAME pipeline with the untrained critic's flat/anti-goal V; scored vs the trained best cue):
    acc_untr, _, _, _ = _score_arm(accumulator, V_untr, seed, a.n_trials, lesion=False, equal=False, permute=False,
                                   true_best_from=V_train)
    # EQUAL-value (G_DISCRIM): all options share the mean learned V -> lesion NEUTRAL (choice agreement):
    _, ch_eq_intact, _, _ = _score_arm(accumulator, V_train, seed, a.n_trials, lesion=False, equal=True,
                                       permute=False)
    _, ch_eq_lesion, _, _ = _score_arm(accumulator, V_train, seed, a.n_trials, lesion=True, equal=True,
                                       permute=False)
    equal_agreement = float(np.mean(ch_eq_intact == ch_eq_lesion)) if a.n_trials > 0 else 1.0
    # PERMUTE (deterministic permutation-average):
    acc_permuted, _, _, _ = _score_arm(accumulator, V_train, seed, a.n_trials, lesion=False, equal=False,
                                       permute=True)
    # non-circularity:
    vs_corr = _value_salience_corr(V_train, seed, a.n_trials, n_opt)

    # ── the 5 gates ──
    headline_ok = (acc_intact - chance) >= a.above_chance_bar
    lesion_ok = ((acc_intact - acc_lesion) >= a.lesion_drop_bar and (acc_lesion - chance) <= a.near_chance_tol)
    untrained_ok = ((acc_intact - acc_untr) >= a.learning_drop_bar and (acc_untr - chance) <= a.near_chance_tol)
    discrim_ok = (equal_agreement >= a.discrim_agreement_bar)
    permute_ok = (abs(acc_permuted - chance) <= a.near_chance_tol)
    noncircular = abs(vs_corr) <= a.max_value_salience_corr
    moat_construction = (not accumulator.has_rf_slices)

    print(f"[R5b seed={seed}] chance={chance:.3f} | INTACT {acc_intact:.3f} (m{m_intact:.1f}) | "
          f"LESION {acc_lesion:.3f} (m{m_lesion:.1f}) | UNTRAINED {acc_untr:.3f} | "
          f"EQUAL-agree {equal_agreement:.3f} | PERMUTE {acc_permuted:.3f} | corr(V,sal) {vs_corr:+.3f}", flush=True)
    print(f"[R5b seed={seed}] gates: HEADLINE={headline_ok} LESION={lesion_ok} UNTRAINED={untrained_ok} "
          f"DISCRIM={discrim_ok} PERMUTE={permute_ok} noncirc={noncircular} moat_by_constr={moat_construction}",
          flush=True)

    # ── MOAT (agent-level; optional per seed) ──
    moat = None
    if a.check_moat:
        try:
            moat = VT.check_moat(seed)
            print(f"[R5b seed={seed}] MOAT what_does(dog,go)={moat['positive']!r} "
                  f"what_does(river,look)={moat['negative']!r} -> holds={moat['moat_holds']}", flush=True)
        except Exception as e:
            moat = {"error": str(e)}
            print(f"[R5b seed={seed}] MOAT ERROR: {e}", flush=True)

    return {
        "seed": int(seed), "backend": backend, "n_options": n_opt, "n_trials": int(a.n_trials), "chance": chance,
        "cues": [list(c) for c in cues],
        "w_afferent_init": w_init, "value_train": vt,
        "V_untrained_hz": [round(v, 2) for v in V_untr],
        "V_trained_hz": [round(v, 2) for v in V_train],
        "trained_grade_goal_far": float(V_train[0] / max(V_train[-1], 1e-3)),
        "untrained_grade_goal_far": float(V_untr[0] / max(V_untr[-1], 1e-3)),
        "acc_intact": acc_intact, "acc_lesion": acc_lesion, "acc_untrained": acc_untr,
        "equal_value_choice_agreement": equal_agreement, "acc_permuted": acc_permuted,
        "margin_intact": m_intact, "margin_lesion": m_lesion, "value_salience_corr": vs_corr,
        "headline_above_chance": bool(headline_ok), "lesion_collapses": bool(lesion_ok),
        "untrained_no_advantage": bool(untrained_ok), "equal_value_neutral": bool(discrim_ok),
        "permute_collapses": bool(permute_ok), "noncircular": bool(noncircular),
        "moat_preserved_by_construction": bool(moat_construction), "moat_agent": moat,
        "elapsed_s": time.time() - t0,
    }


def decide_verdict(rows, a):
    def col(k):
        return [r[k] for r in rows]

    def n_pass(k):
        return int(sum(1 for r in rows if r[k]))
    nseed = len(rows)
    headline_n = n_pass("headline_above_chance")
    lesion_n = n_pass("lesion_collapses")
    untrained_n = n_pass("untrained_no_advantage")
    discrim_n = n_pass("equal_value_neutral")
    permute_n = n_pass("permute_collapses")
    noncirc_all = all(col("noncircular"))
    moat_all = all(col("moat_preserved_by_construction"))
    bar = a.min_seed_pass    # >= 5/6

    detail = {
        "n_seeds": nseed, "min_seed_pass": bar,
        "acc_intact_mean": float(np.mean(col("acc_intact"))), "acc_intact_min": float(np.min(col("acc_intact"))),
        "acc_lesion_mean": float(np.mean(col("acc_lesion"))),
        "acc_untrained_mean": float(np.mean(col("acc_untrained"))),
        "equal_agreement_mean": float(np.mean(col("equal_value_choice_agreement"))),
        "acc_permuted_mean": float(np.mean(col("acc_permuted"))),
        "trained_grade_mean": float(np.mean(col("trained_grade_goal_far"))),
        "untrained_grade_mean": float(np.mean(col("untrained_grade_goal_far"))),
        "value_salience_corr_absmax": float(np.max(np.abs(col("value_salience_corr")))),
        "headline_pass": headline_n, "lesion_pass": lesion_n, "untrained_pass": untrained_n,
        "discrim_pass": discrim_n, "permute_pass": permute_n,
        "noncircular_all": noncirc_all, "moat_by_construction_all": moat_all,
    }
    if not noncirc_all:
        verdict = "INVALID_value_is_relabeled_salience"
    elif headline_n < bar:
        verdict = "HONEST_NEGATIVE_learned_value_does_not_drive_choice"
    elif lesion_n < bar:
        verdict = "HONEST_NEGATIVE_lesion_does_not_collapse"
    elif untrained_n < bar:
        verdict = "HONEST_NEGATIVE_learning_not_load_bearing"     # the untrained critic already picks the best cue
    elif discrim_n < bar:
        verdict = "HONEST_NEGATIVE_lesion_artifact_not_value_specific"
    elif permute_n < bar:
        verdict = "HONEST_NEGATIVE_advantage_is_fixed_pool_bias"
    else:
        verdict = "GO"
    return verdict, detail


def run(a):
    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[R5b] LEARNED spiking value -> value-driven choice. seeds={seeds} K={a.n_options} "
          f"value_train_trials={a.value_train_trials}", flush=True)
    print(f"[R5b] building the tiny spiking value-WTA decision organ (Wang-2002 biased competition)...", flush=True)
    accumulator = SpikingValueChoice(seed=12345, n_options=int(a.n_options), n_steps=int(a.acc_steps))
    rows = [run_seed(s, a, accumulator) for s in seeds]
    verdict, detail = decide_verdict(rows, a)

    print(f"\n{'#'*100}", flush=True)
    print(f"  R5b VERDICT: {verdict}   (>= {detail['min_seed_pass']}/{detail['n_seeds']} seeds per gate)", flush=True)
    print(f"  ACC: INTACT {detail['acc_intact_mean']:.3f} (min {detail['acc_intact_min']:.3f}) | "
          f"LESION {detail['acc_lesion_mean']:.3f} | UNTRAINED {detail['acc_untrained_mean']:.3f} | "
          f"chance {1.0/a.n_options:.3f}", flush=True)
    print(f"  learned grade goal/far: trained {detail['trained_grade_mean']:.2f} vs untrained "
          f"{detail['untrained_grade_mean']:.2f}", flush=True)
    print(f"  EQUAL-agree {detail['equal_agreement_mean']:.3f} | PERMUTE {detail['acc_permuted_mean']:.3f} | "
          f"|corr(V,sal)|max {detail['value_salience_corr_absmax']:.3f}", flush=True)
    print(f"  gate pass-counts: HEADLINE {detail['headline_pass']} LESION {detail['lesion_pass']} "
          f"UNTRAINED {detail['untrained_pass']} DISCRIM {detail['discrim_pass']} PERMUTE {detail['permute_pass']} "
          f"(of {detail['n_seeds']})", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'#'*100}\n", flush=True)

    out = {
        "probe": "navcloseout_R5b_learned_value_driven_choice", "verdict": verdict, "seeds": seeds,
        "config": {"n_options": a.n_options, "n_trials": a.n_trials, "acc_steps": a.acc_steps,
                   "value_train_trials": a.value_train_trials, "value_hz_gain_pA": VALUE_HZ_GAIN_PA,
                   "speak_base_pA": SPEAK_BASE_PA, "salience_gain_pA": SALIENCE_GAIN_PA,
                   "above_chance_bar": a.above_chance_bar, "lesion_drop_bar": a.lesion_drop_bar,
                   "learning_drop_bar": a.learning_drop_bar, "near_chance_tol": a.near_chance_tol,
                   "discrim_agreement_bar": a.discrim_agreement_bar, "min_seed_pass": a.min_seed_pass,
                   "max_value_salience_corr": a.max_value_salience_corr},
        "mechanism": (
            "the LEARNED spiking striosome_value critic (DA-gated STDP on the merged one-brain bridge) supplies the "
            "per-option VALUE (its firing rate V(cue), a real cp_firing_states read) that sets the drift of the "
            "spiking value-WTA (a neural pool's FIRING = the choice). Retires R5-R1a's host make_option_values "
            "stand-in: spiking decision + SPIKING value. CORRECT = pick the higher-LEARNED-V cue."),
        "anti_cheats": {
            "G_LESION": "pin each cue's learned V to the MEAN (drive-level matched: remove the gradient, hold the "
                        "op-point) -> choice collapses to chance.",
            "G_UNTRAINED": "read V from the UNTRAINED critic (init weight, no value-train) -> flat/anti-goal V (A1 "
                           "floor) -> the trained-cue advantage vanishes. Proves the substrate's LEARNING is "
                           "load-bearing (the NEW anti-cheat R5-R1a lacked; its value was a host tag).",
            "G_DISCRIM": "EQUAL learned V on all options -> lesion NEUTRAL (choice agreement ~1.0). Value-SPECIFIC.",
            "G_PERMUTE": "permute which option gets which learned V (deterministic permutation-average) -> chance.",
            "non_circular_value": "corr(learned V, salience) ~ 0 -> the choice is driven by the learned value, not a "
                                  "relabeled salience.",
            "moat_by_construction": "the WTA decision organ has NO RF/conversational slices -> array-disjoint from "
                                    "any composer; check_moat re-asserted on the merged critic agent.",
        },
        "detail": detail, "per_seed": rows, "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "navcloseout_R5", "R5b_learned_value.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


def smoke(seed):
    """CPU well-formedness (no long value-train): cues are graded; the WTA reads a synthetic learned-V vector and
    picks the highest; the untrained-flat and lesion arms collapse; the verdict aggregator is sound."""
    print("=" * 78)
    print(f"[R5b SMOKE seed={seed}] harness well-formedness (no merged-bridge build)")
    print("=" * 78)
    ok = True

    cues2 = _cue_places(2); cues4 = _cue_places(4)
    print(f"  (a) cues K=2 {[tuple(round(c,1) for c in cc) for cc in cues2]}  K=4 "
          f"{[tuple(round(c,1) for c in cc) for cc in cues4]}")
    graded_ok = (cues2[0] == tuple(np.asarray(VT.GOAL, float).tolist())
                 and cues4[0] == tuple(np.asarray(VT.GOAL, float).tolist()))
    ok = ok and graded_ok

    # (b) the drive builder: a graded learned V favors the goal cue; pin-to-mean removes the gradient.
    Vg = np.array([35.0, 12.0])          # trained: goal 35 Hz >> far 12 Hz
    sal = np.array([0.3, 0.7])
    d_int = _drives(Vg, sal, speak_base_pA=SPEAK_BASE_PA, value_gain_pA=VALUE_HZ_GAIN_PA,
                    salience_gain_pA=SALIENCE_GAIN_PA, lesion_value=False)
    d_les = _drives(Vg, sal, speak_base_pA=SPEAK_BASE_PA, value_gain_pA=VALUE_HZ_GAIN_PA,
                    salience_gain_pA=SALIENCE_GAIN_PA, lesion_value=True)
    intact_favors_goal = bool(np.argmax(d_int) == 0)
    lesion_value_flat = bool(np.isclose(d_les[0] - SALIENCE_GAIN_PA * sal[0], d_les[1] - SALIENCE_GAIN_PA * sal[1]))
    print(f"  (b) intact drive favors GOAL cue: {intact_favors_goal} (drives {np.round(d_int,1).tolist()}); "
          f"lesion value-term FLAT: {lesion_value_flat} (drives {np.round(d_les,1).tolist()})")
    ok = ok and intact_favors_goal and lesion_value_flat

    # (c) the tiny spiking WTA picks the higher learned-V pool (a few decisions), moat by construction.
    print("  (c) building the tiny spiking value-WTA (K=2)...")
    acc = SpikingValueChoice(seed=12345, n_options=2, n_steps=80)
    moat_ok = (not acc.has_rf_slices)

    class _A:
        n_options = 2; n_trials = 24; above_chance_bar = 0.20; lesion_drop_bar = 0.20; learning_drop_bar = 0.20
        near_chance_tol = 0.15; discrim_agreement_bar = 0.80; max_value_salience_corr = 0.35; min_seed_pass = 5
    a = _A()
    # synthetic learned V: goal 35 vs far 12 (trained) -> INTACT should pick the goal cue >> chance.
    V_train = np.array([35.0, 12.0]); V_untr = np.array([13.0, 20.0])   # untrained anti-goal floor (probe-like)
    acc_int, _, _, _ = _score_arm(acc, V_train, seed, a.n_trials, lesion=False, equal=False, permute=False)
    acc_les, _, _, _ = _score_arm(acc, V_train, seed, a.n_trials, lesion=True, equal=False, permute=False)
    acc_unt, _, _, _ = _score_arm(acc, V_untr, seed, a.n_trials, lesion=False, equal=False, permute=False,
                                  true_best_from=V_train)
    acc_prm, _, _, _ = _score_arm(acc, V_train, seed, a.n_trials, lesion=False, equal=False, permute=True)
    _, ce_i, _, _ = _score_arm(acc, V_train, seed, a.n_trials, lesion=False, equal=True, permute=False)
    _, ce_l, _, _ = _score_arm(acc, V_train, seed, a.n_trials, lesion=True, equal=True, permute=False)
    eq_agree = float(np.mean(ce_i == ce_l))
    print(f"      INTACT {acc_int:.3f} LESION {acc_les:.3f} UNTRAINED {acc_unt:.3f} PERMUTE {acc_prm:.3f} "
          f"EQUAL-agree {eq_agree:.3f}  moat_by_constr={moat_ok}")
    wta_sane = (acc_int > 0.70 and acc_les <= 0.65 and acc_unt <= 0.65 and abs(acc_prm - 0.5) <= 0.20
                and eq_agree >= 0.80)
    ok = ok and wta_sane and moat_ok

    # (d) the verdict aggregator: synthetic GO / untrained-not-load-bearing NEGATIVE / lesion NEGATIVE.
    def _row(**kw):
        base = dict(headline_above_chance=True, lesion_collapses=True, untrained_no_advantage=True,
                    equal_value_neutral=True, permute_collapses=True, noncircular=True,
                    moat_preserved_by_construction=True, acc_intact=0.9, acc_lesion=0.5, acc_untrained=0.5,
                    equal_value_choice_agreement=0.95, acc_permuted=0.5, trained_grade_goal_far=2.5,
                    untrained_grade_goal_far=0.65, value_salience_corr=0.05)
        base.update(kw); return base
    go_rows = [_row() for _ in range(6)]
    neg_rows = [_row(untrained_no_advantage=(i >= 4)) for i in range(6)]  # untrained passes only 2/6 -> below bar
    v_go, _ = decide_verdict(go_rows, a)
    v_neg, _ = decide_verdict(neg_rows, a)
    agg_ok = (v_go == "GO" and v_neg == "HONEST_NEGATIVE_learning_not_load_bearing")
    print(f"  (d) verdict aggregator: GO->{v_go}; untrained-fails->{v_neg}  (sound: {agg_ok})")
    ok = ok and agg_ok

    print("=" * 78)
    print(f"[R5b SMOKE] {'PASS' if ok else 'FAIL'}")
    print("=" * 78)
    return ok


def main():
    p = argparse.ArgumentParser(description="nav close-out R5 RANK-1 CLOSURE — the LEARNED spiking value drives a "
                                            "value-driven choice (retire R5-R1a's host value stand-in).")
    p.add_argument("--smoke", action="store_true", help="CPU well-formedness (no merged-bridge build)")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--seed", type=int, default=42, help="(smoke) seed")
    p.add_argument("--n-options", type=int, default=2, help="K choice options / cues (2 = near-vs-far RANK 1)")
    p.add_argument("--n-trials", type=int, default=60, help="value-choice trials per seed")
    p.add_argument("--acc-steps", type=int, default=120, help="spiking WTA integration window (steps)")
    p.add_argument("--value-train-trials", type=int, default=40, help="DA-gated STDP value-train trials")
    p.add_argument("--above-chance-bar", type=float, default=0.20)
    p.add_argument("--lesion-drop-bar", type=float, default=0.20)
    p.add_argument("--learning-drop-bar", type=float, default=0.20)
    p.add_argument("--near-chance-tol", type=float, default=0.15)
    p.add_argument("--discrim-agreement-bar", type=float, default=0.80)
    p.add_argument("--max-value-salience-corr", type=float, default=0.35)
    p.add_argument("--min-seed-pass", type=int, default=5, help="min seeds passing each gate for GO (>=5/6)")
    p.add_argument("--check-moat", action="store_true", help="also run the agent-level moat check per seed (slow)")
    p.add_argument("--verbose", action="store_true", help="verbose value-train progress")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    logging.getLogger().setLevel(logging.WARNING)
    for nm in ("SIM_BRIDGE", "sim", "sim.bridge"):
        logging.getLogger(nm).setLevel(logging.WARNING)

    if a.smoke:
        ok = smoke(a.seed)
        raise SystemExit(0 if ok else 1)
    run(a)


if __name__ == "__main__":
    main()
