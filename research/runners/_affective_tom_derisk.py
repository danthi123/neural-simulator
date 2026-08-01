"""W5 (Stage-4 social build): AFFECTIVE THEORY OF MIND -- infer ANOTHER agent's EMOTION from THEIR (witnessed)
situation, held in an OTHER-tagged affect model that is DISSOCIABLE from the system's OWN affect. The affective
companion to W3 (the agent-keyed false-BELIEF register): W3 turns the self-schema outward to model another agent's
BELIEF; W5 turns the P0.3 affect region outward to model another agent's FEELING. Shamay-Tsoory affective
perspective-taking / cognitive empathy (mPFC/TPJ + the appraisal system applied to an OTHER-tagged input).

WHY THIS IS ToM AND NOT EGOCENTRIC PROJECTION (the load-bearing dissociation). The trivial way to "report another's
emotion" is to project your OWN affect (you feel good -> you say they feel good). That FAILS whenever the other is in
a DIFFERENT situation from you. Genuine affective ToM maintains a SEPARATE, OTHER-tagged affect state driven by the
OTHER's situation, so on INCONGRUENT scenarios (self got good news, other got bad news) the attribution tracks the
OTHER (bad), not the self (good). The de-risk's decisive arm is exactly these incongruent trials.

MECHANISM (reuse-by-import, NO `sim/` edit; two P0.3 AffectStateBrain instances on separate numpy bridges = a clean
self/other separation, the same "separate slot per agent" motif W3 uses for belief):
  - SELF affect model  = AffectStateBrain (verbatim import), appraised on the SYSTEM's OWN situation valence.
  - OTHER affect model = AffectStateBrain (verbatim import), OTHER-tagged, appraised on the OTHER agent's WITNESSED
    situation valence (F3 appraisal on the other-schema). Each region is the P0.3 opponent slow-NMDA attractor
    (affect_vplus/vminus + Namburi-Tye cross-inhibition); appraisal enters via the diffuse neuromodulator bus.
  - THE EMOTIONAL ATTRIBUTION / SPEECH TONE = a SYNAPTIC read of the OTHER model's gated output: the affect state
    biases recall_pos (V+) vs recall_neg (V-) through the ONE `affect_out` transmission gate (Bower mood-congruent
    pathway, reused as the tone read-out). tone_sign = sign(rate(recall_pos) - rate(recall_neg)). tone_sign=+1 =>
    "share-joy / positive" tone; -1 => "comfort / negative-acknowledging" tone. This number is NEVER host-set: it is
    a difference of two SPIKE RATES from the OTHER-tagged region; you cannot get it without running that region.
  - Scoped to VALENCE (good/bad), matching the P0.3 substrate's CHARACTERIZED bistable good/bad latch (P0.3 is a
    QUALIFIED-GO/BOUNDARY: it holds valence sign robustly, not a graded discrete-emotion circumplex). Fine discrete
    emotions need the graded-circumplex surpass P0.3 already named (a line/bump attractor with SFA eviction / the
    dendritic substrate) -- the SAME wall, not a new one. HONEST: a FUNCTIONAL affective-mentalizing correlate
    (a separate, other-driven, dissociable affect attribution), NOT a claim of access to another mind's feelings.

TASK. Each scenario draws a SELF situation valence and an OTHER situation valence, independently, in {+ (good news),
- (bad news)} -- a balanced 2x2 factorial so half the trials are INCONGRUENT (self != other). The OTHER's situation
is what the other WITNESSED (their belief about their own situation), which on a false-belief subset differs from
reality (the other feels according to what THEY perceived).

GO GATE (6-seed 42/43/44/100/101/102, CPU numpy):
  (a) other_attribution_acc      >= 0.85  -- the OTHER-model tone tracks the OTHER's situation valence (chance 0.5).
  (b) SELF/OTHER DISSOCIATION (the keystone, on INCONGRUENT trials):
        other_acc_incongruent    >= 0.85  -- attribution follows the OTHER even when self differs, AND
        egocentric_acc_incong    <= 0.35  -- reading the SELF affect as the attribution is WRONG on incongruent
                                             trials (self-tone tracks self, != other). Proves a separate other-model
                                             is load-bearing; projection cannot solve it.
  (c) OTHER-LESION collapses: lesion the OTHER model's OUTPUT (affect_out gate=0) -> no affective attribution ->
        other_attribution_acc_lesion <= 0.65 (-> chance 0.5). The other pools keep appraising; only the gated
        read-out is severed => "the model runs; its OUTPUT is what is load-bearing for the attribution."
  (d) SCRAMBLE the other's witnessed valence across trials -> the attribution tracks the WRONG situation ->
        scored vs the TRUE other valence, other_attribution_acc_scramble <= 0.65 (-> chance). Proves it rides the
        ACTUAL other-situation, not a fixed response.
  CHARACTERIZATION (reported; the W3 x P0.3 integration): on the false-belief-of-affect subset the OTHER's witnessed
  situation != reality -> the inferred emotion tracks the other's BELIEF (witnessed), and a reality-appraised
  baseline would be WRONG (affect_tracks_belief high, reality_baseline low).

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit. cfg.seed set per-seed inside
AffectStateBrain (NOT actual_seed_used -- the CLAUDE.md substrate-seeding gotcha; verified in the P0.3 class).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affective_tom_derisk --smoke --seed 42 \
                  --json research/findings/raw/_affective_tom_smoke.json
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affective_tom_derisk \
                  --seeds 42 43 44 100 101 102 --n-trials 24 \
                  --json research/findings/raw/_affective_tom/summary_6seed.json
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

# reuse-by-import: the P0.3 affect region (verbatim) + its read constants.
from research.runners._affect_state_region_derisk import (  # noqa: E402
    AffectStateBrain, N_RECALL, RECALL_CUE_PA,
)

# ---- read protocol constants (mirror P0.3 _establish_mood_and_probe) -----------------------------------------
SETTLE_BASE = 40        # base settle before appraisal (a per-trial jitter is added for genuine trial variance)
SETTLE_JITTER = 40      # per-trial random extra settle -> the OU-noise phase at read time varies across trials
ESTABLISH_MS = 120      # establish the appraised affect state (slow-NMDA integrates)
PROBE_MS = 80           # read window: equal recall cue to both pools, mood biases which fires more


def read_tone(brain: AffectStateBrain, valence_sign: int, lesion: bool, settle_ms: int):
    """Appraise `brain` on a situation valence (+1 good / -1 bad), then read the SYNAPTIC tone = which recall pool
    (pos vs neg) the affect state drives harder through the affect_out gate. Returns (tone_sign, pos_rate, neg_rate,
    mood). NB set the affect lesion AFTER reset() -- reset() restores the transmission gate to 1.0."""
    brain.reset()
    brain.set_affect_lesion(lesion)
    brain.step(int(settle_ms))                                   # settle (variable length -> noise-phase variance)
    vp = 1.0 if valence_sign > 0 else 0.0
    vm = 1.0 if valence_sign < 0 else 0.0
    brain.step(ESTABLISH_MS, vp=vp, vm=vm, ar=0.5)               # establish the appraised affect state
    c = brain.step(PROBE_MS, vp=vp, vm=vm, ar=0.4,
                   cue_pos=RECALL_CUE_PA, cue_neg=RECALL_CUE_PA,  # EQUAL cue: the mood breaks the symmetry
                   record=("recall_pos", "recall_neg", "affect_vplus", "affect_vminus"))
    pos = c["recall_pos"] / (N_RECALL * PROBE_MS)
    neg = c["recall_neg"] / (N_RECALL * PROBE_MS)
    tone_sign = 1 if pos > neg else -1
    mood = brain.mood_rate(c, PROBE_MS)
    return tone_sign, float(pos), float(neg), float(mood)


def make_trials(seed: int, n_trials: int):
    """Balanced 2x2 factorial over (self_valence, other_valence) in {+1, -1}. Half the trials are INCONGRUENT
    (self != other) -- the decisive self/other-projection dissociation set. Also tags a false-belief-of-affect
    subset (the other's WITNESSED valence != reality) for the W3 x P0.3 characterization arm."""
    rng = np.random.default_rng(seed * 131 + 7)
    per_cell = max(1, n_trials // 4)
    combos = [(+1, +1), (+1, -1), (-1, +1), (-1, -1)]
    trials = []
    for (sv, ov) in combos:
        for _ in range(per_cell):
            # false-belief-of-affect on ~1/3 of trials: the other WITNESSED `ov` but reality is the opposite.
            false_belief = bool(rng.random() < 1.0 / 3.0)
            reality_ov = (-ov) if false_belief else ov
            trials.append({"self_v": int(sv), "other_v": int(ov),   # other_v = the other's WITNESSED (believed) valence
                           "reality_other_v": int(reality_ov), "false_belief": false_belief,
                           "settle": int(SETTLE_BASE + rng.integers(0, SETTLE_JITTER + 1))})
    rng.shuffle(trials)
    return trials


def evaluate_seed(seed: int, n_trials: int, thresholds: dict, verbose: bool = False):
    trials = make_trials(seed, n_trials)
    self_v = np.array([t["self_v"] for t in trials], dtype=int)
    other_v = np.array([t["other_v"] for t in trials], dtype=int)          # witnessed/believed
    reality_v = np.array([t["reality_other_v"] for t in trials], dtype=int)
    incong = self_v != other_v
    fb_mask = np.array([t["false_belief"] for t in trials], dtype=bool)

    # build the two affect models ONCE per seed (reset per trial); same seed -> both deterministic per appraisal,
    # the per-trial settle jitter supplies genuine noise-phase variance so the accuracies are real averages.
    self_brain = AffectStateBrain(seed, nmda_on=True)
    other_brain = AffectStateBrain(seed, nmda_on=True)

    self_tone = np.zeros(len(trials), dtype=int)          # tone from the SELF model on self_v (the egocentric read)
    other_tone = np.zeros(len(trials), dtype=int)         # tone from the OTHER model on the other's witnessed valence
    other_tone_lesion = np.zeros(len(trials), dtype=int)  # OTHER model output-lesioned
    other_mood = np.zeros(len(trials), dtype=float)
    self_mood = np.zeros(len(trials), dtype=float)

    # SCRAMBLE: permute which witnessed valence the OTHER model is appraised on (belief attached to the wrong trial)
    rng = np.random.default_rng(seed * 977 + 23)
    scr_perm = rng.permutation(len(trials))
    scr_other_v = other_v[scr_perm]
    other_tone_scramble = np.zeros(len(trials), dtype=int)

    for i, t in enumerate(trials):
        s_sign, _sp, _sn, s_mood = read_tone(self_brain, int(self_v[i]), lesion=False, settle_ms=t["settle"])
        o_sign, _op, _on, o_mood = read_tone(other_brain, int(other_v[i]), lesion=False, settle_ms=t["settle"])
        ol_sign, _lp, _ln, _lm = read_tone(other_brain, int(other_v[i]), lesion=True, settle_ms=t["settle"])
        os_sign, _xp, _xn, _xm = read_tone(other_brain, int(scr_other_v[i]), lesion=False, settle_ms=t["settle"])
        self_tone[i] = s_sign; other_tone[i] = o_sign
        other_tone_lesion[i] = ol_sign; other_tone_scramble[i] = os_sign
        self_mood[i] = s_mood; other_mood[i] = o_mood

    # ---- accuracies (chance 0.5; tone tracks a binary valence sign) ----
    other_attribution_acc = float(np.mean(other_tone == other_v))
    self_attribution_acc = float(np.mean(self_tone == self_v))               # the self model tracks the self situation
    other_acc_incong = float(np.mean(other_tone[incong] == other_v[incong])) if incong.any() else 0.0
    # EGOCENTRIC read on incongruent trials: use the SELF affect as the attribution -> scored vs the OTHER's truth.
    egocentric_acc_incong = float(np.mean(self_tone[incong] == other_v[incong])) if incong.any() else 0.0
    # LESION the OTHER model output -> attribution non-specific (scored vs the OTHER's truth).
    other_attribution_acc_lesion = float(np.mean(other_tone_lesion == other_v))
    # SCRAMBLE: the OTHER model was appraised on scr_other_v[i] (the wrong witnessed valence); score vs TRUE other_v.
    other_attribution_acc_scramble = float(np.mean(other_tone_scramble == other_v))

    # CHARACTERIZATION (W3 x P0.3): on false-belief-of-affect trials, does the inferred emotion track the other's
    # BELIEF (witnessed) rather than reality? reality-baseline = scoring the OTHER tone against reality.
    if fb_mask.any():
        affect_tracks_belief = float(np.mean(other_tone[fb_mask] == other_v[fb_mask]))
        reality_baseline_acc = float(np.mean(other_tone[fb_mask] == reality_v[fb_mask]))
    else:
        affect_tracks_belief = reality_baseline_acc = 0.0

    ch = thresholds["chance_margin"]
    go_attr = bool(other_attribution_acc >= thresholds["attribution_acc"])
    go_dissoc = bool(other_acc_incong >= thresholds["attribution_acc"]
                     and egocentric_acc_incong <= thresholds["egocentric_max"])
    go_lesion = bool(other_attribution_acc_lesion <= ch)
    go_scramble = bool(other_attribution_acc_scramble <= ch)
    go = bool(go_attr and go_dissoc and go_lesion and go_scramble)

    r = {
        "seed": int(seed), "n_trials": len(trials),
        "n_incongruent": int(incong.sum()), "n_false_belief": int(fb_mask.sum()), "chance": 0.5,
        "intact": {
            "other_attribution_acc": other_attribution_acc,
            "self_attribution_acc": self_attribution_acc,
            "other_acc_incongruent": other_acc_incong,
            "mean_other_mood": float(np.mean(other_mood)), "mean_self_mood": float(np.mean(self_mood)),
        },
        "self_other_dissociation": {
            "other_acc_incongruent": other_acc_incong,
            "egocentric_acc_incongruent": egocentric_acc_incong,
            "dissociation_ok": go_dissoc,
        },
        "other_lesion": {"attribution_acc": other_attribution_acc_lesion, "collapsed": go_lesion},
        "scramble": {"attribution_acc_vs_true": other_attribution_acc_scramble, "collapsed": go_scramble},
        "belief_characterization": {
            "affect_tracks_belief_on_false_belief": affect_tracks_belief,
            "reality_baseline_on_false_belief": reality_baseline_acc,
        },
        "go_components": {"attribution": go_attr, "self_other_dissociation": go_dissoc,
                          "other_lesion_collapses": go_lesion, "scramble_collapses": go_scramble},
        "go": go,
    }
    if verbose:
        _print_seed(r)
    return r


def _print_seed(r):
    it = r["intact"]; ds = r["self_other_dissociation"]; le = r["other_lesion"]; sc = r["scramble"]
    bc = r["belief_characterization"]
    print(f"  [seed {r['seed']}]  {r['n_trials']} trials ({r['n_incongruent']} incongruent, "
          f"{r['n_false_belief']} false-belief; chance 0.5)", flush=True)
    print(f"    INTACT   other_attribution={it['other_attribution_acc']:.3f}  self_attribution="
          f"{it['self_attribution_acc']:.3f}  other|incong={it['other_acc_incongruent']:.3f}", flush=True)
    print(f"    DISSOC   other|incong={ds['other_acc_incongruent']:.3f}  EGOCENTRIC|incong="
          f"{ds['egocentric_acc_incongruent']:.3f} (must be LOW)  ok={ds['dissociation_ok']}", flush=True)
    print(f"    LESION   other_attribution={le['attribution_acc']:.3f}  collapsed={le['collapsed']}", flush=True)
    print(f"    SCRAMBLE attribution_vs_true={sc['attribution_acc_vs_true']:.3f}  collapsed={sc['collapsed']}",
          flush=True)
    print(f"    BELIEF   affect_tracks_belief={bc['affect_tracks_belief_on_false_belief']:.3f}  "
          f"reality_baseline={bc['reality_baseline_on_false_belief']:.3f} (belief should WIN)", flush=True)
    print(f"    >>> seed GO = {r['go']}  {r['go_components']}", flush=True)


DEFAULT_THRESHOLDS = {
    "attribution_acc": 0.85,   # intact: the OTHER-model tone tracks the other's situation valence (chance 0.5)
    "egocentric_max": 0.35,    # reading SELF affect as the attribution FAILS on incongruent trials (-> ~0)
    "chance_margin": 0.65,     # lesion/scramble collapse toward the 0.5 chance floor; 0.65 separates from ~1.0 intact
}


def main():
    ap = argparse.ArgumentParser(description="W5 AFFECTIVE THEORY OF MIND de-risk (infer another agent's emotion "
                                             "from their witnessed situation; dissociable from the self's affect).")
    ap.add_argument("--seed", type=int, default=42, help="single seed (used by --smoke)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed list (overrides --seed)")
    ap.add_argument("--n-trials", type=int, default=60, help="scenarios per seed (balanced 2x2 factorial; >=48 "
                    "keeps the lesion/scramble chance-floor estimate tight)")
    ap.add_argument("--smoke", action="store_true", help="tiny 1-seed smoke (fewer trials)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--json", type=str, default="research/findings/raw/_affective_tom/summary_6seed.json")
    args = ap.parse_args()

    if args.smoke:
        seeds = [args.seed]
        n_trials = min(args.n_trials, 8)
    else:
        seeds = args.seeds if args.seeds is not None else [args.seed]
        n_trials = args.n_trials

    print(f"[affective-ToM] W5 -- infer another agent's EMOTION (valence) from their WITNESSED situation, in an "
          f"OTHER-tagged affect model dissociable from the self | seeds={seeds} n_trials={n_trials} "
          f"backend={args.backend}", flush=True)
    print("[affective-ToM] two P0.3 AffectStateBrain models (self + other, separate bridges); tone = SYNAPTIC "
          "recall_pos-vs-recall_neg read of the OTHER model's affect_out-gated output. Scoped to VALENCE "
          "(good/bad) = the P0.3 bistable-latch substrate; discrete emotions need the graded-circumplex surpass.",
          flush=True)
    print("[affective-ToM] HONEST: a FUNCTIONAL affective-mentalizing correlate (a separate, other-driven, "
          "dissociable affect attribution) -- NOT a claim of access to another mind's feelings.", flush=True)

    t0 = time.time()
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, n_trials, DEFAULT_THRESHOLDS, verbose=True))

    n_go = sum(1 for r in per_seed if r["go"])
    all_go = bool(n_go == len(per_seed))
    verdict = "GO" if all_go else ("PARTIAL" if n_go > 0 else "NEGATIVE")

    def _mean(path):
        vals = []
        for r in per_seed:
            v = r
            for k in path:
                v = v[k]
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    agg = {
        "mean_other_attribution_acc": _mean(["intact", "other_attribution_acc"]),
        "mean_self_attribution_acc": _mean(["intact", "self_attribution_acc"]),
        "mean_other_acc_incongruent": _mean(["self_other_dissociation", "other_acc_incongruent"]),
        "mean_egocentric_acc_incongruent": _mean(["self_other_dissociation", "egocentric_acc_incongruent"]),
        "mean_lesion_attribution_acc": _mean(["other_lesion", "attribution_acc"]),
        "mean_scramble_attribution_acc": _mean(["scramble", "attribution_acc_vs_true"]),
        "mean_affect_tracks_belief": _mean(["belief_characterization", "affect_tracks_belief_on_false_belief"]),
        "mean_reality_baseline": _mean(["belief_characterization", "reality_baseline_on_false_belief"]),
        "all_dissociation_ok": all(r["self_other_dissociation"]["dissociation_ok"] for r in per_seed),
        "all_lesion_collapse": all(r["other_lesion"]["collapsed"] for r in per_seed),
        "all_scramble_collapse": all(r["scramble"]["collapsed"] for r in per_seed),
    }

    # ---- ATTRIBUTE the attribution to its manipulations (tools.lab): what fraction of the ABOVE-CHANCE
    # attribution effect is removed by the lesion / by scrambling the witnessed valence (chance=0.5, so
    # attribute over the above-chance effect, not the raw accuracy whose 0.5 floor would masquerade as
    # "present in the control").
    from tools.lab import attributable_to                                      # noqa: E402
    _les_attr = attributable_to("other-model OUTPUT (affect_out lesion), above chance",
                                treatment_value=agg["mean_other_attribution_acc"] - 0.5,
                                control_value=agg["mean_lesion_attribution_acc"] - 0.5)
    _scr_attr = attributable_to("ACTUAL other-situation (witnessed-valence scramble), above chance",
                                treatment_value=agg["mean_other_attribution_acc"] - 0.5,
                                control_value=agg["mean_scramble_attribution_acc"] - 0.5)
    agg["frac_attribution_to_lesion"] = _les_attr
    agg["frac_attribution_to_scramble"] = _scr_attr

    # ---- EARN the verdict (tools.verdict): a GO cannot be asserted while any precondition fails or is unmeasured.
    from tools.verdict import Verdict                                          # noqa: E402
    _v = Verdict("affective ToM (W5)", chance=0.5)
    _v.require("6 seeds (project bar)", len(seeds) >= 6, expect=True)
    _v.floor("other-attribution acc vs chance", agg["mean_other_attribution_acc"], 0.5)
    _v.require("self/other dissociation on EVERY seed (other tracks other, egocentric fails on incongruent)",
               agg["all_dissociation_ok"], expect=True,
               note="the egocentric self-affect read must FAIL on incongruent trials, else projection could pass")
    _v.control("other-lesion collapses the attribution", treatment=agg["mean_other_attribution_acc"],
               control=agg["mean_lesion_attribution_acc"])
    _v.require("other-lesion collapsed on EVERY seed", agg["all_lesion_collapse"], expect=True)
    _v.control("scrambled other-situation collapses the attribution", treatment=agg["mean_other_attribution_acc"],
               control=agg["mean_scramble_attribution_acc"])
    _v.require("scramble collapsed on EVERY seed", agg["all_scramble_collapse"], expect=True)
    _verdict_block = _v.decide(go=all_go)
    if _verdict_block["status"] != "GO" and verdict == "GO":
        verdict = _verdict_block["status"]

    out = {
        "runner": "_affective_tom_derisk",
        "faculty": "W5 affective theory of mind (infer another agent's emotion from their witnessed situation; "
                   "Stage-4 social build; ToM ladder affective rung)",
        "theory": "Shamay-Tsoory affective perspective-taking / cognitive empathy (mPFC/TPJ + the appraisal system "
                  "applied to an OTHER-tagged input); the P0.3 affect region turned OUTWARD, the affective companion "
                  "to W3's outward-turned self-schema (FUNCTIONAL correlate only, NOT access to another mind)",
        "mechanism": "two P0.3 AffectStateBrain instances (self + other, separate numpy bridges); OTHER-tagged model "
                     "appraised on the other's WITNESSED situation valence; emotional attribution / speech tone = "
                     "SYNAPTIC sign(rate(recall_pos)-rate(recall_neg)) of the OTHER model's affect_out-gated output",
        "scope": "VALENCE (good/bad) only -- matches the P0.3 bistable good/bad latch (QUALIFIED-GO/BOUNDARY). "
                 "Fine discrete emotions need the SAME graded-circumplex surpass P0.3 named (line/bump attractor "
                 "with SFA eviction / dendritic substrate), NOT a new wall.",
        "seeds": seeds, "n_trials": n_trials, "backend": args.backend, "chance": 0.5,
        "verdict": verdict, "GO": all_go, "n_seeds_go": n_go,
        "thresholds": DEFAULT_THRESHOLDS,
        "aggregate": agg,
        **{k: _verdict_block[k] for k in ("preconditions", "disabled_processes", "undefined_reasons")},
        "per_seed": per_seed,
        "HONEST_NOTE": "numpy-CPU read (real spiking Izhikevich bridges; 'numpy' is the backend, not a host "
                       "shortcut). The situation->valence appraisal is the legitimate world/perceptual input (P0.3's "
                       "interface, DR-2 learned-tag precedent); the ToM-specific neural work is (a) a SEPARATE "
                       "OTHER-tagged affect state and (b) the synaptic tone read-out, verified load-bearing by the "
                       "self/other dissociation (egocentric self-read fails on incongruent trials), the output-lesion "
                       "collapse, and the witnessed-situation scramble collapse. NO sim/ edit (reuse-by-import).",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[affective-ToM] VERDICT: {verdict}  ({n_go}/{len(seeds)} seeds GO)", flush=True)
    print(f"  other-attribution {agg['mean_other_attribution_acc']:.3f} | other|incong "
          f"{agg['mean_other_acc_incongruent']:.3f} vs EGOCENTRIC|incong {agg['mean_egocentric_acc_incongruent']:.3f} "
          f"| lesion {agg['mean_lesion_attribution_acc']:.3f} | scramble {agg['mean_scramble_attribution_acc']:.3f} "
          f"| belief-track {agg['mean_affect_tracks_belief']:.3f} vs reality {agg['mean_reality_baseline']:.3f}",
          flush=True)
    print(f"[affective-ToM] wrote {args.json}  ({out['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
