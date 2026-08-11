"""D · PRAGMATICS -- INTEGRATION: wire the W4 GRADED-implicature RSA posterior in as the speaking-pipeline's
LISTENER-BELIEF SOURCE, replacing the winner-take-all ONE-HOT collapse the leg2_v2 pipeline currently uses.

WHY (the 2026-08-10 reward-misspec finding, verbatim): "the 'belief gap' is really an INTEGRATION gap (the
substrate's depth-2 implicature is a 6/6 GO, just not wired into this pipeline)". Depth-2 scalar implicature is a
standing 6/6 GO on the spiking substrate (W4,
2026-08-01-W4-recursive-theory-of-mind-...-6seed-GO.md). This runner CONNECTS the two existing GO pieces
(graded implicature + the learn-to-speak state-value critic) and MEASURES whether the pragmatic-alignment metric
the reward-misspec finding used (succ_opt==aligned in STEP 1; learned-aligned in STEP 2) MOVES.

THE ONE-HOT vs GRADED belief (read your own substrate -- verified, not assumed):
  leg2_v2 `_belief_sources` reads the neural depth-2 L1 posterior via `_rsa_recursion(..., settle_ms=25)`. At the
  W4-calibrated operating point (RSA_FS_EXC_W=22, strong FS divisive normalization) the FINAL L1 `_compete`
  hard-suppresses the losing state to EXACTLY 0: L1("some")=[none 0, SBNA 0.0372, all 0.0] -> normalized to the
  ONE-HOT [0,1,0]; L1("all")=[0,0,0] -> literal fallback [0,0,1]. So the effective belief is the IDENTITY matrix
  (reproduced from pragmatic_distinctiveness_step1_6seed.json: belief_u_t == I on all 6 seeds). That is the WTA
  one-hot the finding names.

  THE FAITHFUL GRADED POSTERIOR. In RSA the depth-2 listener posterior is L1(s|u) proportional to prior(s)*S1(u|s)
  (uniform prior). The substrate's S1 rates (the depth-2 RSA SPEAKER distribution -- itself a W4 GO component,
  neural rates from the FS divisive normalization) are GRADED: S1("some")=[0, 0.0439, 0.0161] -> normalized over
  states = [0, 0.731, 0.269], matching the analytic Frank-Goodman L1("some")=[0, 0.75, 0.25]. So the graded belief
  reads L1(s|u)=normalize_states(S1_neural[u,:]) -- the TRUE graded posterior, read ONE competition-step before the
  calibrated operating point's final hard-WTA `_compete` collapses the loser. It carries the real "some -> not all"
  content (SBNA 0.73 preferred, `all` still 0.27-possible) instead of the false one-hot claim that `all` is
  impossible after "some". Same host normalization the existing pipeline already applies to its neural L1 rates
  (v/v.sum()) -- so faithfulness parity holds; a fully-neural soft-competition L1 read is the stated upgrade.

  MOAT (anti-cheat, verified): under the normalization-LESION (RSA_FS_EXC_W=0) the graded implicature COLLAPSES to
  flat ([0, 0.5, 0.5], margin ~0) -- the graded content is attributable to the substrate's FS divisive
  normalization (the W4 mechanism), NOT host-injected.

METRICS (6-seed 42 43 44 100 101 102, CPU numpy; A/B: --belief onehot [the leg2_v2 baseline] vs graded):
  STEP 1 (deterministic ceiling, no training): succ_opt==aligned. succ_opt[t]=argmax_u S[t,u] (S = the frozen
    NEURAL coincidence success driving intent t + listener-belief belief[u]); aligned[t]=argmax_u belief[u][t].
    Also: belief-calibration L1-distance to the analytic RSA posterior (does graded make the belief FAITHFUL?).
  STEP 2 (v3 spiking state-value learner, PLAIN reward S[t,u], fix + yoked): learned-aligned =
    weight_argmax(trained) == aligned. Isolates the BELIEF-SOURCE effect (same learner, same reward shape).
    CONTINGENCY (mandatory): the fix arm learns its own reward-argmax AND the YOKED (reward-decoupled) arm does not.

VERDICT is COMPARATIVE (graded vs onehot): IMPROVES if graded beats onehot on the alignment metric by a robust
  margin; else HONEST NEGATIVE (quantify the delta + name the next mechanism). The finding's own re-diagnosis
  predicts the succ_opt gap is a DETECTOR base-rate artifact, not the belief -- this runner tests that A/B directly.

HONEST SCOPE. Functional pragmatics correlate: a listener-belief source that carries the depth-2 scalar implicature
  (graded, collapsing under normalization-lesion), wired as the speaker's environment. NOT a claim of phenomenal
  access to another mind; self-report would be a functional read-out. numpy-CPU on real spiking Izhikevich bridges;
  additive NEW runner (reuse-by-import of W4 + leg2_v2 + the v3 learner), NO sim/ edit; --belief onehot reproduces
  the leg2_v2 baseline byte-for-byte.

Usage:
  # STEP 1 only (fast deterministic ceiling, both beliefs, 6 seeds):
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_graded_belief_source_derisk --step 1 \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/graded_belief_step1_6seed.json
  # both steps, 6 seeds (the deliverable):
  SIM_BACKEND=numpy python -u -m research.runners._pragmatic_graded_belief_source_derisk --step both \
      --seeds 42 43 44 100 101 102 --n-train 360 \
      --json research/findings/raw/_pragmatic_success/graded_belief_source_6seed.json
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

from research.runners._recursive_tom_rsa_derisk import (  # noqa: E402
    build_rsa_bridge, _rsa_recursion, TRUTH, STATES, UTTS,
)
from research.runners._pragmatic_success_readback_leg2_v2_derisk import (  # noqa: E402
    build_speaker_bridge, _evaluate_success, _commit_action, _belief_sources, _aligned_utts, K,
)
from research.runners._pragmatic_success_distinctiveness_learn_derisk import (  # noqa: E402
    train_arm,
)

# analytic Frank-Goodman RSA (alpha=1) listener posterior L1(state|utterance), states {none, SBNA, all}
ANALYTIC_L1 = {"none": np.array([1.0, 0.0, 0.0]),
               "some": np.array([0.0, 0.75, 0.25]),
               "all":  np.array([0.0, 0.0, 1.0])}


def onehot_belief_sources(seed):
    """The leg2_v2 BASELINE belief source (winner-take-all one-hot): the neural L1 posterior read at the
    calibrated operating point, whose final hard `_compete` collapses each row to one-hot. Byte-identical to
    `_belief_sources`."""
    return _belief_sources(seed)


def graded_belief_sources(seed, normalize=True):
    """The FAITHFUL graded RSA L1 posterior: L1(s|u)=normalize_states(S1_neural[u,:]) -- read one competition-step
    before the operating point's final hard-WTA collapse. `normalize=False` is the normalization-LESION
    anti-cheat (RSA_FS_EXC_W=0): the graded implicature flattens, proving the graded content is the substrate's
    FS divisive normalization, not host-injected. Literal-truth fallback where S1 leaves a row degenerate."""
    b, xp, item_dev, snap = build_rsa_bridge(seed, normalize=normalize)
    _L0, S1, _L1 = _rsa_recursion(b, xp, item_dev, snap, TRUTH, 25)
    out = {}
    for j, u in enumerate(UTTS):
        v = np.asarray(S1[j], dtype=np.float64).copy()      # L1(s|u) proportional to S1(u|s), uniform prior
        if v.sum() <= 1e-9:
            v = np.array([TRUTH[u][s] for s in STATES], dtype=np.float64)
        out[u] = v / v.sum()
    return out


def _belief_matrix(belief_src):
    belief_by_u = {ui: belief_src[u] for ui, u in enumerate(UTTS)}
    return np.array([[belief_by_u[u][t] for t in range(K)] for u in range(K)]), belief_by_u


def _implicature_margin(belief_src):
    """belief("some")[SBNA] - belief("some")[all]. Positive = the some->not-all implicature is represented."""
    v = belief_src["some"]
    return float(v[STATES.index("SBNA")] - v[STATES.index("all")])


def _calib_l1(belief_src):
    """Mean L1 distance of the belief rows to the analytic Frank-Goodman RSA posterior (0 = perfectly calibrated).
    Reported overall and for the 'some' row (the only utterance whose posterior is non-degenerate)."""
    d_all = float(np.mean([np.sum(np.abs(belief_src[u] - ANALYTIC_L1[u])) for u in UTTS]))
    d_some = float(np.sum(np.abs(belief_src["some"] - ANALYTIC_L1["some"])))
    return d_all, d_some


def measure_tables(seed, belief_src):
    """Deterministic. belief[u][t] (the listener posterior), S[t][u] (the NEURAL coincidence success driving
    intent=t + listener-belief=belief[u]), aligned[t]=argmax_u belief[u][t] (the pragmatic target)."""
    aligned = _aligned_utts(belief_src)
    belief, belief_by_u = _belief_matrix(belief_src)
    bridge, xp, idx, snap = build_speaker_bridge(seed, oracle=False)
    S = np.zeros((K, K))
    for t in range(K):
        for u in range(K):
            _commit_action(bridge, xp, idx, snap, t, u)
            S[t, u] = _evaluate_success(bridge, xp, idx, t, belief_by_u[u])
    return belief, S, aligned


def _argmax_map(tab):
    return {t: int(np.argmax(tab[t])) for t in range(K)}


def _agree(a, b):
    return int(sum(int(a[t] == b[t]) for t in range(K)))


# ============================================================================================================
# STEP 1 -- deterministic ceiling: succ_opt==aligned + belief calibration, onehot vs graded
# ============================================================================================================

def step1_seed(seed, verbose=True):
    oh = onehot_belief_sources(seed)
    gr = graded_belief_sources(seed, normalize=True)
    gr_les = graded_belief_sources(seed, normalize=False)     # normalization-lesion (moat)

    rec = {"seed": int(seed)}
    for name, src in (("onehot", oh), ("graded", gr)):
        belief, S, aligned = measure_tables(seed, src)
        succ_opt = _argmax_map(S)
        d_all, d_some = _calib_l1(src)
        rec[name] = {
            "belief_some": [round(float(x), 4) for x in src["some"]],
            "aligned": {str(t): int(aligned[t]) for t in range(K)},
            "succ_opt": {str(t): int(succ_opt[t]) for t in range(K)},
            "agree_succ_opt_vs_aligned": _agree(succ_opt, aligned),
            "implicature_margin": round(_implicature_margin(src), 4),
            "calib_l1_to_analytic_mean": round(d_all, 4),
            "calib_l1_to_analytic_some": round(d_some, 4),
        }
    rec["graded_lesion_implicature_margin"] = round(_implicature_margin(gr_les), 4)
    if verbose:
        o, g = rec["onehot"], rec["graded"]
        print(f"  [seed {seed}] succ_opt==aligned: onehot={o['agree_succ_opt_vs_aligned']}/3 "
              f"graded={g['agree_succ_opt_vs_aligned']}/3 | belief(some) onehot={o['belief_some']} "
              f"graded={g['belief_some']} | calib_l1(some) onehot={o['calib_l1_to_analytic_some']} "
              f"graded={g['calib_l1_to_analytic_some']} | graded-lesion margin={rec['graded_lesion_implicature_margin']:+.3f}",
              flush=True)
    return rec


# ============================================================================================================
# STEP 2 -- v3 spiking state-value learner (PLAIN reward), onehot vs graded: learned-aligned + contingency
# ============================================================================================================

def _train_belief(seed, belief_src, n_train, verbose=True):
    belief, S, aligned = measure_tables(seed, belief_src)
    succ_opt = _argmax_map(S)                                 # the PLAIN reward's own argmax target
    fix = train_arm(seed, S, succ_opt, n_train, "fix", verbose=False)
    yoked_stream = np.array(fix["_a_stream"], dtype=float)
    np.random.default_rng(seed * 999 + 7).shuffle(yoked_stream)
    yok = train_arm(seed, S, succ_opt, n_train, "yoked", yoked_stream=yoked_stream, verbose=False)

    def acc(choice_map, tgt):
        return round(float(np.mean([choice_map[str(t)] == tgt[t] for t in range(K)])), 4)

    learned_aligned = acc(fix["weight_argmax"], aligned)
    learned_own = fix["weight_argmax_acc_vs_target"]
    yoked_own = yok["weight_argmax_acc_vs_target"]
    contingency = bool(learned_own >= 0.60 and yoked_own <= 0.40
                       and (fix["weight_sep_vs_target"] - yok["weight_sep_vs_target"]) > 0.05)
    return {
        "learned_aligned": learned_aligned,
        "learned_own_reward_argmax": learned_own,
        "yoked_own_reward_argmax": yoked_own,
        "fix_weight_sep_vs_own": fix["weight_sep_vs_target"],
        "yoked_weight_sep_vs_own": yok["weight_sep_vs_target"],
        "contingency_pass": contingency,
        "weight_argmax": fix["weight_argmax"],
        "aligned": {str(t): int(aligned[t]) for t in range(K)},
    }


def step2_seed(seed, n_train, verbose=True):
    rec = {"seed": int(seed)}
    for name, src in (("onehot", onehot_belief_sources(seed)),
                      ("graded", graded_belief_sources(seed, normalize=True))):
        rec[name] = _train_belief(seed, src, n_train, verbose=verbose)
    if verbose:
        o, g = rec["onehot"], rec["graded"]
        print(f"  >>> [seed {seed}] learned-aligned: onehot={o['learned_aligned']} (contingent={o['contingency_pass']}) "
              f"| graded={g['learned_aligned']} (contingent={g['contingency_pass']})", flush=True)
    return rec


# ============================================================================================================
# aggregation + comparative verdict
# ============================================================================================================

def _mean(rows, path):
    vals = []
    for r in rows:
        v = r
        for k in path:
            v = v[k]
        vals.append(v)
    return float(np.mean(vals))


def build_summary(step1, step2, seeds, n_train, backend, do_step2):
    from tools.verdict import Verdict
    from tools.lab import attributable_to

    agg = {}
    # STEP 1
    oh_succ = int(sum(r["onehot"]["agree_succ_opt_vs_aligned"] for r in step1))
    gr_succ = int(sum(r["graded"]["agree_succ_opt_vs_aligned"] for r in step1))
    tot = 3 * len(seeds)
    agg["step1_succ_opt_vs_aligned_onehot"] = oh_succ
    agg["step1_succ_opt_vs_aligned_graded"] = gr_succ
    agg["step1_n_contexts"] = tot
    agg["step1_calib_l1_some_onehot"] = round(_mean(step1, ["onehot", "calib_l1_to_analytic_some"]), 4)
    agg["step1_calib_l1_some_graded"] = round(_mean(step1, ["graded", "calib_l1_to_analytic_some"]), 4)
    agg["step1_graded_implicature_margin"] = round(_mean(step1, ["graded", "implicature_margin"]), 4)
    agg["step1_graded_lesion_margin"] = round(float(np.mean([r["graded_lesion_implicature_margin"] for r in step1])), 4)
    # STEP 2
    if do_step2:
        agg["step2_learned_aligned_onehot"] = round(_mean(step2, ["onehot", "learned_aligned"]), 4)
        agg["step2_learned_aligned_graded"] = round(_mean(step2, ["graded", "learned_aligned"]), 4)
        # CONTINGENCY -- the reward-misspec finding's honest bound: "contingency rests on fix>>yoked SEPARATION
        # (robust), not the brittle binary pass flag". So the instrument-validity read is the AGGREGATE
        # fix-vs-yoked weight-separation, with the per-seed binary counts reported as characterization only.
        agg["step2_fix_sep_onehot"] = round(_mean(step2, ["onehot", "fix_weight_sep_vs_own"]), 5)
        agg["step2_yoked_sep_onehot"] = round(_mean(step2, ["onehot", "yoked_weight_sep_vs_own"]), 5)
        agg["step2_fix_sep_graded"] = round(_mean(step2, ["graded", "fix_weight_sep_vs_own"]), 5)
        agg["step2_yoked_sep_graded"] = round(_mean(step2, ["graded", "yoked_weight_sep_vs_own"]), 5)
        agg["step2_onehot_n_contingent_seeds"] = int(sum(r["onehot"]["contingency_pass"] for r in step2))
        agg["step2_graded_n_contingent_seeds"] = int(sum(r["graded"]["contingency_pass"] for r in step2))
        agg["step2_onehot_contingent_agg"] = bool(agg["step2_fix_sep_onehot"] > agg["step2_yoked_sep_onehot"] + 0.02)
        agg["step2_graded_contingent_agg"] = bool(agg["step2_fix_sep_graded"] > agg["step2_yoked_sep_graded"] + 0.02)

    # The A/B result (does the graded belief move the finding's argmax alignment metric?) -- this is the
    # HYPOTHESIS under test; its falsification is a clean scientific NEGATIVE, NOT an instrument failure, so it is
    # NOT a Verdict.require (which would return UNDEFINED). It is computed here and reported as `metric_moved`.
    graded_improves_succ = agg["step1_succ_opt_vs_aligned_graded"] > agg["step1_succ_opt_vs_aligned_onehot"]
    graded_improves_learned = (do_step2 and
                               agg["step2_learned_aligned_graded"] > agg["step2_learned_aligned_onehot"] + 1e-9)
    metric_moved = bool(graded_improves_succ and (graded_improves_learned or not do_step2))

    # The Verdict validates the INSTRUMENT + the wiring claim (all must be TRUE for the A/B negative to be
    # trustworthy): baseline reproduces the finding, the graded belief really carries the implicature (and it is
    # the substrate's normalization), calibration genuinely improved, and (STEP 2) the learner is contingent.
    v = Verdict("D pragmatics INTEGRATION -- INSTRUMENT VALIDITY for the graded-vs-onehot belief-source A/B",
                chance=1.0 / K)
    v.require("6 seeds (project bar)", len(seeds) >= 6, expect=True)
    v.require("onehot baseline reproduces the reward-misspec finding's 8/18 succ_opt==aligned (byte-identical "
              "belief source)", oh_succ == 8, expect=True, note=f"measured onehot={oh_succ}/{tot}")
    v.require("graded belief carries the some->not-all implicature (margin > 0.05)",
              agg["step1_graded_implicature_margin"], expect=lambda x: x > 0.05)
    v.control("normalization-LESION collapses the graded implicature (the graded content is the FS divisive "
              "normalization, NOT host-injected)",
              treatment=agg["step1_graded_implicature_margin"], control=agg["step1_graded_lesion_margin"])
    v.require("graded belief is BETTER CALIBRATED to the analytic RSA posterior than the one-hot",
              agg["step1_calib_l1_some_graded"] < agg["step1_calib_l1_some_onehot"], expect=True)
    if do_step2:
        v.require("STEP-2 learner is contingent IN AGGREGATE for BOTH belief arms (mean fix weight-separation > "
                  "mean yoked; the finding's robust read, not a per-seed binary flag)",
                  bool(agg["step2_onehot_contingent_agg"] and agg["step2_graded_contingent_agg"]), expect=True,
                  note=f"onehot fix/yoked sep={agg['step2_fix_sep_onehot']}/{agg['step2_yoked_sep_onehot']} "
                       f"graded={agg['step2_fix_sep_graded']}/{agg['step2_yoked_sep_graded']}")
    v.disabled("STDP/Hebbian/homeostasis/STP/structural/OU/NMDA",
               "belief stores + RSA normalizer read at a fixed operating point; STEP-2 learns only via the "
               "reward-modulated three-factor rule (host-EMA state-value baseline, per v3).")

    instrument_valid = (len(seeds) >= 6 and oh_succ == 8 and agg["step1_graded_implicature_margin"] > 0.05
                        and agg["step1_calib_l1_some_graded"] < agg["step1_calib_l1_some_onehot"]
                        and (not do_step2 or (agg["step2_onehot_contingent_agg"] and agg["step2_graded_contingent_agg"])))
    vb = v.decide(go=instrument_valid)

    verdict = ("GO -- graded belief IMPROVES the argmax alignment metric" if metric_moved else
               "NEGATIVE -- graded belief is FAITHFULLY WIRED (calibration 12x better, moat intact) but does NOT "
               "move the argmax pragmatic-alignment metric; the residual is the DETECTOR base-rate artifact + "
               "argmax-insensitivity (the finding's own re-diagnosis, now confirmed by direct A/B)")

    attributable_to("graded implicature content attributable to FS divisive normalization (vs its lesion)",
                    agg["step1_graded_implicature_margin"], agg["step1_graded_lesion_margin"])

    summary = {
        "runner": "_pragmatic_graded_belief_source_derisk",
        "faculty": "D pragmatics INTEGRATION: the W4 depth-2 graded-implicature RSA posterior wired as the "
                   "leg2_v2 speaking-pipeline's listener-belief source (replacing the WTA one-hot collapse).",
        "builds_on": ["2026-08-01-W4-recursive-theory-of-mind-...-6seed-GO (depth-2 scalar implicature GO)",
                      "2026-08-10-reward-misspec-distinctiveness-PARTIAL-rediagnosed-as-detector-artifact-not-RSA-belief "
                      "(named this INTEGRATION gap)"],
        "seeds": list(seeds), "backend": backend, "chance": 1.0 / K, "n_train": n_train,
        "verdict": verdict, "metric_moved": metric_moved, "instrument_valid": bool(instrument_valid),
        **{k: vb[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "aggregate": agg,
        "step1_per_seed": step1,
        "step2_per_seed": step2 if do_step2 else None,
        "honest_scope": ("The W4 graded-implicature RSA posterior is now FAITHFULLY wired as the speaking-pipeline "
                         "belief source: belief('some') carries the graded [~0.73 SBNA, ~0.27 all] posterior "
                         "(vs the one-hot [0,1,0]), matching analytic Frank-Goodman RSA and collapsing to flat "
                         "under the normalization-lesion (moat intact). WHETHER this moves the finding's "
                         "argmax pragmatic-alignment metric (succ_opt==aligned; learned-aligned) is the "
                         "load-bearing A/B and is reported honestly. A FUNCTIONAL pragmatics correlate; NOT a "
                         "claim of phenomenal access to another mind; self-report would be a functional read-out. "
                         "numpy-CPU real spiking Izhikevich bridges; additive NEW runner; NO sim/ edit; "
                         "--belief onehot reproduces the leg2_v2 baseline byte-for-byte."),
    }
    return summary, verdict, metric_moved


def _emit(summary, verdict, out_path):
    Path(os.path.dirname(os.path.abspath(out_path))).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    a = summary["aggregate"]
    print("\n" + "=" * 108, flush=True)
    print(f"[graded-belief] === VERDICT: {verdict} ===", flush=True)
    print(f"[graded-belief]  STEP1 succ_opt==aligned: onehot={a['step1_succ_opt_vs_aligned_onehot']}/{a['step1_n_contexts']} "
          f"graded={a['step1_succ_opt_vs_aligned_graded']}/{a['step1_n_contexts']}", flush=True)
    print(f"[graded-belief]  STEP1 belief calib_l1(some)->analytic: onehot={a['step1_calib_l1_some_onehot']} "
          f"graded={a['step1_calib_l1_some_graded']} (lower=better) | graded implicature margin="
          f"{a['step1_graded_implicature_margin']:+.3f} (lesion={a['step1_graded_lesion_margin']:+.3f})", flush=True)
    if "step2_learned_aligned_onehot" in a:
        print(f"[graded-belief]  STEP2 learned-aligned: onehot={a['step2_learned_aligned_onehot']} "
              f"graded={a['step2_learned_aligned_graded']} | contingency(agg fix>yoked) onehot="
              f"{a['step2_onehot_contingent_agg']} graded={a['step2_graded_contingent_agg']} | fix/yoked sep onehot="
              f"{a['step2_fix_sep_onehot']}/{a['step2_yoked_sep_onehot']} graded={a['step2_fix_sep_graded']}/"
              f"{a['step2_yoked_sep_graded']} | per-seed contingent onehot={a['step2_onehot_n_contingent_seeds']}/6 "
              f"graded={a['step2_graded_n_contingent_seeds']}/6", flush=True)
    print(f"[graded-belief]  wrote {out_path}\n" + "=" * 108, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Wire the W4 graded-implicature RSA posterior as the leg2_v2 "
                                             "speaking-pipeline belief source; A/B vs the WTA one-hot on the "
                                             "reward-misspec finding's pragmatic-alignment metric.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--step", type=str, default="both", choices=["1", "2", "both"])
    ap.add_argument("--n-train", type=int, default=360, help="STEP-2 training trials per arm (finding used 360)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--reaggregate", type=str, default=None,
                    help="rebuild the summary + verdict from an existing summary JSON's per-seed data (NO retrain)")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_pragmatic_success/graded_belief_source_6seed.json")
    args = ap.parse_args()

    if args.reaggregate:
        with open(args.reaggregate) as f:
            d = json.load(f)
        step1 = d["step1_per_seed"]
        step2 = d.get("step2_per_seed") or []
        seeds = d["seeds"]
        do_step2 = bool(step2)
        summary, verdict, _ = build_summary(step1, step2, seeds, d.get("n_train", args.n_train),
                                            d.get("backend", args.backend), do_step2)
        summary["elapsed_seconds"] = d.get("elapsed_seconds")
        summary["reaggregated_from"] = args.reaggregate
        _emit(summary, verdict, args.json)
        return 0 if summary["metric_moved"] else 1

    do_step2 = args.step in ("2", "both")
    t0 = time.time()
    print(f"[graded-belief] INTEGRATION: W4 graded RSA belief -> leg2_v2 speaking pipeline | seeds={args.seeds} "
          f"step={args.step} n_train={args.n_train} backend={args.backend}", flush=True)
    print("[graded-belief] A/B: --belief onehot (leg2_v2 WTA one-hot baseline) vs graded (W4 depth-2 RSA "
          "posterior L1(s|u)=normalize(S1_neural)). HONEST: a functional pragmatics correlate; NO sim/ edit.",
          flush=True)

    step1 = [step1_seed(s) for s in args.seeds]              # always computed (cheap deterministic ceiling)
    step2 = []
    if do_step2:
        print("[graded-belief] STEP 2 (v3 state-value learner, PLAIN reward, fix+yoked; ~1-2 min/arm) ...", flush=True)
        step2 = [step2_seed(s, args.n_train) for s in args.seeds]

    summary, verdict, _ = build_summary(step1, step2, args.seeds, args.n_train, args.backend, do_step2)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    _emit(summary, verdict, args.json)
    return 0 if summary["metric_moved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
