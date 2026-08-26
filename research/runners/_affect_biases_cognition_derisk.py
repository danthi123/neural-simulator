"""Lane A · Affect (Phase-0 emotion) — AFFECT-BIASES-COGNITION: a persistent core-affect state that CAUSALLY
biases HOW the brain talks along FOUR axes at once. This is the faculty that COLORS conversation — the same words
come out warmer/faster/more-hedged depending on the standing mood x arousal.

REUSE (NO new organ): this de-risk SUBCLASSES the P0.3 GO affect-state region
(`research/runners/_affect_state_region_derisk.py::AffectStateBrain`) — the SAME three opponent slow-NMDA pools
(affect_vplus / affect_vminus / affect_arousal) with Namburi-Tye cross-inhibition on ONE numpy Izhikevich
SimulationBridge, the SAME diffuse-neuromodulator appraisal injection, the SAME `mood = rate(V+)-rate(V-)` state
read-out, and the SAME single `affect_out` transmission gate as the clean lesion. Two of the four bias axes
(mood-congruent recall, arousal speak-rate) are the organ's own, reused by importing its measurement functions;
the runner ADDS two more axes through the organ's pre-existing `extra_regions`/`extra_pathways` seam (no `sim/`
edit, no organ edit).

THE FOUR BIAS AXES (each an independent way core-affect colors cognition; all four must be present AND
lesion-load-bearing for a GO):
  (1) MOOD-CONGRUENT RECALL (Bower 1981) — positive mood facilitates valence-matched (positive) recall, negative
      mood the negative pool: affect_vplus -> recall_pos, affect_vminus -> recall_neg (organ-native).
  (2) AROUSAL-MODULATED ENCODING (Mather-Sutherland 2011 GANE; McGaugh; LC-NE gain) — high arousal AMPLIFIES the
      encoding of an attended item: affect_arousal -> encode_pool raises the firing (encoding gain) of a co-active
      stimulus pool (ADDED axis).
  (3) SPEAK-RATE (Damasio somatic vigor; Niv 2007 tonic-arousal-gates-vigor) — high arousal raises the spiking
      speak/silence accumulator's speak rate: affect_arousal -> speak_acc (organ-native).
  (4) HEDGE PROBABILITY (affective pragmatics; negative/anxious affect -> more tentative, hedged speech) — negative
      mood raises a HEDGE accumulator over an ASSERT accumulator in a biased WTA: affect_vminus -> hedge_acc,
      affect_vplus -> assert_acc; hedge-rate = hedge/(hedge+assert) (ADDED axis).

EVERY step from appraised input to biased output is neurons/synapses: pool firing (slow-NMDA attractor) + diffuse
volume transmission (appraisal) + fixed synaptic projections (the four biases) + biased WTA competition
(speak/silence, hedge/assert). NO host formula sits between sensation and the biased read-out. => brain-based-only.

CONVERSATION-DRIVING (the DEPTH property): a downstream turn's CONTENT (which valence gets recalled), ENCODING
STRENGTH (what is laid down), RATE (how fast it speaks) and STANCE (how hedged) all CHANGE with the affect state,
and every one of those changes VANISHES when the single `affect_out` gate is lesioned. That lesion-load-bearing
collapse of all four axes IS the GO.

MANDATORY ANTI-CHEATS (wired into the printed verdict; a GO whose anti-cheats fail is a NO-GO):
  * AFFECT-LESION -> ALL FOUR biases vanish: each axis's intact margin collapses to <=0.2 of intact under the
    affect_out lesion (the affect pools keep firing; only their synaptic bias onto cognition is cut).
  * YOKED-RANDOM affect -> biases point the WRONG direction: when the established affect state is a RANDOM sign
    uncorrelated with the trial's target, each axis's margin drops below 0.5x intact (mis-directed / collapsed),
    proving the bias tracks the CONGRUENT affect state, not a generic drive.
  * NO-CONFAB MOAT intact: |corr(concept signed-valence, PPMI relatedness)| < 0.15 — valence is its own circumplex
    dimension, not relabeled factual likelihood (reused from the organ; separate-RNG independent tags).

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit. cfg.seed per-seed (NOT actual_seed_used).

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_biases_cognition_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_biases_cognition_derisk --seeds 42 43 44 100 101 102
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
from tools.lab import attributable_to  # noqa: E402  (lesion-collapse attribution — which arm owns the bias)

# --- reuse-by-import: the P0.3 GO affect-state organ + its concept/appraisal machinery ------------------------
from research.runners._affect_state_region_derisk import (  # noqa: E402
    AffectStateBrain, build_concepts, _pearson,
    measure_congruent_recall, measure_speak_rate,
    DEFAULT_RECUR_WEIGHT, N_AFF, N_RECALL, N_ACC, N_WTA, RECALL_CUE_PA,
    SPEAK_BASE_PA, SILENCE_BASE_PA, BIAS_WEIGHT,
)

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_biases_cognition_6seed.json"

# ---- ADDED-axis pool sizes / drives --------------------------------------------------------------------------
N_ENC = 40                 # encoding pool (arousal-modulated encoding gain)
ENCODE_STIM_PA = 60.0      # tonic "attended-stimulus" drive to encode_pool (present in BOTH arousal conditions,
                           # so the arousal margin is the arousal-ADDED encoding, not the baseline stimulus)
HEDGE_BASE_PA = 90.0       # tonic drive to the hedge accumulator (utterance-is-uncertain default pressure)
ASSERT_BASE_PA = 90.0      # tonic drive to the assert accumulator (equal to hedge -> the affect bias breaks the tie)


# =============================================================================================================
# AffectBiasesBrain: the P0.3 organ + two ADDED bias axes appended through the organ's extra_regions/pathways
# seam. The affect pools + opponent inhibition + appraisal bus + affect_out gate are UNCHANGED (byte-identical
# region indices for the native pools); we only append encode/hedge/assert pools and their gated afferents.
# =============================================================================================================
class AffectBiasesBrain(AffectStateBrain):
    def __init__(self, seed, recur_weight=DEFAULT_RECUR_WEIGHT, ou_pA=8.0):
        from sim.regions import BrainRegion, RegionPathway
        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
        FS = "IZH2007_FS_CORTICAL_INTERNEURON"
        G = "affect_out"   # SAME single lesion gate as the organ -> lesioning it cuts ALL four axes at once

        def exc_pool(name, n, dens=0.0, w=0.0):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=dens,
                               exc_weight_mean=w, inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        def fs_pool(name, n):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False, izh_neuron_type=FS)

        extra_regions = [
            exc_pool("encode_pool", N_ENC),                              # (2) arousal-modulated encoding
            exc_pool("hedge_acc", N_ACC, dens=0.4, w=0.3),               # (4) hedge accumulator
            exc_pool("assert_acc", N_ACC, dens=0.4, w=0.3),             # (4) assert accumulator
            fs_pool("hedge_wta_fs", N_WTA),                             # (4) hedge/assert biased competition
        ]
        extra_pathways = [
            # (2) arousal -> encode_pool: high arousal amplifies the encoding of a co-active stimulus (GANE / LC-NE
            #     gain). Gated by affect_out -> the lesion removes the arousal boost, leaving only the stimulus.
            RegionPathway(from_region="affect_arousal", to_region="encode_pool", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            # (4) negative mood -> hedge, positive mood -> assert (affective pragmatics), gated by affect_out.
            RegionPathway(from_region="affect_vminus", to_region="hedge_acc", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            RegionPathway(from_region="affect_vplus", to_region="assert_acc", density=0.6, weight_mean=BIAS_WEIGHT,
                          weight_jitter=0.1, plastic=False, transmission_gate=G),
            # hedge vs assert biased WTA (shared hedge_wta_fs) — the affect bias breaks the symmetric base drive.
            RegionPathway(from_region="hedge_acc", to_region="hedge_wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="assert_acc", to_region="hedge_wta_fs", density=0.5, weight_mean=8.0,
                          weight_jitter=0.1, plastic=False),
            RegionPathway(from_region="hedge_wta_fs", to_region="hedge_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
            RegionPathway(from_region="hedge_wta_fs", to_region="assert_acc", density=0.6, weight_mean=6.0,
                          weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        ]
        super().__init__(seed, nmda_on=True, recur_weight=recur_weight, ou_pA=ou_pA, opponent_style="cross",
                         extra_regions=extra_regions, extra_pathways=extra_pathways)

    def step_cues(self, n_steps, vp=0.0, vm=0.0, ar=0.0, cues=None, record=("affect_vplus", "affect_vminus")):
        """Generalized step: holds the appraisal broadcast (vp/vm/ar) and injects a dict of {region: pA} afferent
        currents (cues). Returns {region: summed_spike_count} for the recorded regions. Mirrors the organ's step()
        but with an arbitrary cue set so the added encode/hedge/assert pools can be driven."""
        b = self._bridge
        cues = cues or {}
        counts = {r: 0.0 for r in record}
        for _ in range(int(n_steps)):
            if vp or vm or ar:
                self._set_appraisal(vp, vm, ar)
            b.cp_external_input_current[:] = 0.0
            for reg, pA in cues.items():
                if pA:
                    b.cp_external_input_current[self._idx[reg]] = np.float32(pA)
            b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            for r in record:
                counts[r] += float(fs[self._idx[r]].sum())
        return counts


# =============================================================================================================
# Axis (2): AROUSAL-MODULATED ENCODING
# =============================================================================================================
def measure_encoding(seed, recur_weight, arousal_level, lesion=False, probe_ms=120, ou_pA=8.0):
    """Encoding gain = firing rate of encode_pool under a sustained arousal state while an attended stimulus is
    co-active. High arousal -> affect_arousal fires -> synaptic drive to encode_pool -> stronger encoding.
    lesion=True clamps affect_out -> only the stimulus drive remains."""
    brain = AffectBiasesBrain(seed, recur_weight=recur_weight, ou_pA=ou_pA)
    brain.set_affect_lesion(lesion)
    brain.step_cues(40)
    brain.step_cues(100, ar=float(arousal_level))                                   # establish arousal state
    c = brain.step_cues(probe_ms, ar=float(arousal_level), cues={"encode_pool": ENCODE_STIM_PA},
                        record=("encode_pool",))
    return c["encode_pool"] / (N_ENC * probe_ms)


# =============================================================================================================
# Axis (4): HEDGE PROBABILITY
# =============================================================================================================
def measure_hedge_rate(brain, mood_sign, lesion=False, probe_ms=100, establish_steps=120):
    """Establish a mood then probe hedge/assert with EQUAL base drive to both accumulators. hedge-rate =
    hedge/(hedge+assert). Negative mood biases hedge; positive mood biases assert. NB affect_out gate is set
    AFTER reset() (reset restores the gate array to 1.0)."""
    brain.reset()
    brain.set_affect_lesion(lesion)
    brain.step_cues(40)
    vp = 1.0 if mood_sign > 0 else 0.0
    vm = 1.0 if mood_sign < 0 else 0.0
    brain.step_cues(establish_steps, vp=vp, vm=vm, ar=0.4)                          # establish mood
    c = brain.step_cues(probe_ms, vp=vp, vm=vm, ar=0.3,
                        cues={"hedge_acc": HEDGE_BASE_PA, "assert_acc": ASSERT_BASE_PA},
                        record=("hedge_acc", "assert_acc"))
    h = c["hedge_acc"]; a = c["assert_acc"]
    return h / (h + a) if (h + a) > 1e-9 else 0.5


def measure_hedge_bias(seed, recur_weight, lesion=False, yoked=False, n_trials=6, ou_pA=8.0, rng=None):
    """Hedge bias = mean(hedge_rate under NEGATIVE-target mood) - mean(hedge_rate under POSITIVE-target mood).
    Positive => negative mood hedges more (the affective-pragmatics direction). lesion collapses it; yoked
    (random established sign) mis-directs it."""
    brain = AffectBiasesBrain(seed, recur_weight=recur_weight, ou_pA=ou_pA)
    if rng is None:
        rng = np.random.default_rng(seed * 7 + 5)
    neg_rates, pos_rates = [], []
    for t in range(int(n_trials)):
        target_sign = -1 if (t % 2 == 0) else 1                 # half neg-target, half pos-target
        drive_sign = int(rng.choice([-1, 1])) if yoked else target_sign
        hr = measure_hedge_rate(brain, drive_sign, lesion=lesion)
        (neg_rates if target_sign < 0 else pos_rates).append(hr)
    return float(np.mean(neg_rates) - np.mean(pos_rates)), float(np.mean(neg_rates)), float(np.mean(pos_rates))


# =============================================================================================================
# One seed = all four axes x {intact, lesion, yoked}
# =============================================================================================================
def run_seed(seed, concepts, recur_weight, ou_pA=8.0):
    t0 = time.time()

    # (1) MOOD-CONGRUENT RECALL (organ-native) — intact / lesion / yoked
    rec_intact, _ = measure_congruent_recall(seed, recur_weight, lesion=False, yoked=False, ou_pA=ou_pA)
    rec_lesion, _ = measure_congruent_recall(seed, recur_weight, lesion=True, yoked=False, ou_pA=ou_pA)
    rec_yoked, _ = measure_congruent_recall(seed, recur_weight, lesion=False, yoked=True, ou_pA=ou_pA)

    # (2) AROUSAL-MODULATED ENCODING — margin = enc(high arousal) - enc(low arousal)
    enc_hi = measure_encoding(seed, recur_weight, arousal_level=1.0, lesion=False, ou_pA=ou_pA)
    enc_lo = measure_encoding(seed, recur_weight, arousal_level=0.0, lesion=False, ou_pA=ou_pA)
    enc_les_hi = measure_encoding(seed, recur_weight, arousal_level=1.0, lesion=True, ou_pA=ou_pA)
    enc_les_lo = measure_encoding(seed, recur_weight, arousal_level=0.0, lesion=True, ou_pA=ou_pA)
    enc_intact = enc_hi - enc_lo
    enc_lesion = enc_les_hi - enc_les_lo
    # yoked encoding: arousal level randomized so the "high" condition is not reliably high
    ry = np.random.default_rng(seed * 11 + 2)
    enc_y_hi = measure_encoding(seed, recur_weight, arousal_level=float(ry.uniform(0, 1)), lesion=False, ou_pA=ou_pA)
    enc_y_lo = measure_encoding(seed, recur_weight, arousal_level=float(ry.uniform(0, 1)), lesion=False, ou_pA=ou_pA)
    enc_yoked = enc_y_hi - enc_y_lo

    # (3) SPEAK-RATE (organ-native) — margin = speak(high arousal) - speak(low arousal); lesion / yoked
    sr_hi, _ = measure_speak_rate(seed, recur_weight, arousal_level=1.0, lesion=False, ou_pA=ou_pA)
    sr_lo, _ = measure_speak_rate(seed, recur_weight, arousal_level=0.0, lesion=False, ou_pA=ou_pA)
    sr_les_hi, _ = measure_speak_rate(seed, recur_weight, arousal_level=1.0, lesion=True, ou_pA=ou_pA)
    sr_les_lo, _ = measure_speak_rate(seed, recur_weight, arousal_level=0.0, lesion=True, ou_pA=ou_pA)
    spk_intact = sr_hi - sr_lo
    spk_lesion = sr_les_hi - sr_les_lo
    sr_y_hi, _ = measure_speak_rate(seed, recur_weight, arousal_level=float(ry.uniform(0, 1)), lesion=False, ou_pA=ou_pA)
    sr_y_lo, _ = measure_speak_rate(seed, recur_weight, arousal_level=float(ry.uniform(0, 1)), lesion=False, ou_pA=ou_pA)
    spk_yoked = sr_y_hi - sr_y_lo

    # (4) HEDGE PROBABILITY (added) — bias = hedge_rate(neg) - hedge_rate(pos); lesion / yoked
    hed_intact, hr_neg, hr_pos = measure_hedge_bias(seed, recur_weight, lesion=False, yoked=False, ou_pA=ou_pA)
    hed_lesion, _, _ = measure_hedge_bias(seed, recur_weight, lesion=True, yoked=False, ou_pA=ou_pA)
    hed_yoked, _, _ = measure_hedge_bias(seed, recur_weight, lesion=False, yoked=True, ou_pA=ou_pA)

    # no-confab moat (value-perp-plausibility)
    vpp = _pearson(concepts["s_signed"], concepts["relatedness"])

    axes = {
        "mood_congruent_recall": {"intact": rec_intact, "lesion": rec_lesion, "yoked": rec_yoked},
        "arousal_encoding":      {"intact": enc_intact, "lesion": enc_lesion, "yoked": enc_yoked},
        "speak_rate":            {"intact": spk_intact, "lesion": spk_lesion, "yoked": spk_yoked},
        "hedge_probability":     {"intact": hed_intact, "lesion": hed_lesion, "yoked": hed_yoked},
    }

    def ratio(x, base):
        return (x / base) if abs(base) > 1e-9 else 1.0

    # per-axis checks: present (intact>0) AND lesion-collapses (<=0.2x) AND yoked-misdirects (<0.5x).
    # attributable_to(intact, lesion): what FRACTION of each axis's bias is REMOVED by the affect lesion (i.e.
    # is genuinely carried by the affect_out projection) vs still present in the lesioned control. A load-bearing
    # bias attributes ~100% to the affect state; if most of the "bias" survived the lesion it would be a confound.
    axis_ok = {}
    axis_attrib = {}
    for name, d in axes.items():
        present = d["intact"] > 1e-4
        lesion_collapse = ratio(d["lesion"], d["intact"]) <= 0.2
        yoked_misdirect = ratio(d["yoked"], d["intact"]) < 0.5
        frac = attributable_to(f"{name} (seed {seed}): affect-lesion", d["intact"], d["lesion"], warn_below=0.8)
        axis_attrib[name] = frac
        axis_ok[name] = {"present": bool(present), "lesion_collapse<=0.2": bool(lesion_collapse),
                         "yoked_misdirect<0.5": bool(yoked_misdirect),
                         "affect_attributable_frac": frac,
                         "ok": bool(present and lesion_collapse and yoked_misdirect)}

    moat_ok = abs(vpp) < 0.15
    all_axes_present = all(axis_ok[n]["present"] for n in axes)
    all_lesion = all(axis_ok[n]["lesion_collapse<=0.2"] for n in axes)
    all_yoked = all(axis_ok[n]["yoked_misdirect<0.5"] for n in axes)
    go = all(axis_ok[n]["ok"] for n in axes) and moat_ok

    row = {
        "seed": int(seed), "recur_weight": float(recur_weight), "GO": bool(go),
        "axes": axes, "axis_checks": axis_ok, "axis_affect_attributable_frac": axis_attrib,
        "all_four_present": bool(all_axes_present), "all_four_lesion_collapse": bool(all_lesion),
        "all_four_yoked_misdirect": bool(all_yoked),
        "hedge_rate_neg": hr_neg, "hedge_rate_pos": hr_pos,
        "value_plausibility_corr": vpp, "moat_ok": bool(moat_ok),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"  [seed {seed}] recall {rec_intact:+.3f}(les {rec_lesion:+.3f}/yok {rec_yoked:+.3f}) | "
          f"enc {enc_intact:+.3f}(les {enc_lesion:+.3f}/yok {enc_yoked:+.3f}) | "
          f"speak {spk_intact:+.3f}(les {spk_lesion:+.3f}/yok {spk_yoked:+.3f}) | "
          f"hedge {hed_intact:+.3f}(les {hed_lesion:+.3f}/yok {hed_yoked:+.3f}) | "
          f"moat r {vpp:+.3f} | GO={go} ({row['elapsed_seconds']}s)", flush=True)
    return row


# =============================================================================================================
# SMOKE — one seed, all four axes intact-vs-lesion (the load-bearing collapse) + moat
# =============================================================================================================
def run_smoke(seed, concepts, recur_weight, ou_pA=8.0):
    print(f"[affect-biases SMOKE] seed={seed} recur_weight={recur_weight} — four bias axes intact vs lesion",
          flush=True)
    row = run_seed(seed, concepts, recur_weight, ou_pA=ou_pA)
    ax = row["axis_checks"]
    for name in ("mood_congruent_recall", "arousal_encoding", "speak_rate", "hedge_probability"):
        d = row["axes"][name]; ok = ax[name]
        print(f"  [{name:22s}] intact {d['intact']:+.4f} | lesion {d['lesion']:+.4f} | yoked {d['yoked']:+.4f} "
              f"-> present={ok['present']} lesion_collapse={ok['lesion_collapse<=0.2']} "
              f"yoked_misdirect={ok['yoked_misdirect<0.5']}", flush=True)
    all_four = row["all_four_present"]
    all_les = row["all_four_lesion_collapse"]
    smoke_go = bool(all_four and all_les and row["moat_ok"])
    verdict = (f"SMOKE {'GO' if smoke_go else 'NO-GO'} — four_axes_present={all_four} "
               f"all_four_lesion_collapse={all_les} moat_intact(|r|={abs(row['value_plausibility_corr']):.3f}<0.15)"
               f"={row['moat_ok']} -> {'PROCEED to 6-seed battery' if smoke_go else 'INVESTIGATE before battery'}")
    print(f"\n[affect-biases SMOKE] VERDICT: {verdict}", flush=True)
    return {"smoke_go": smoke_go, "verdict": verdict, "seed_row": row}


# =============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1-seed: four axes intact-vs-lesion + moat")
    ap.add_argument("--recur-weight", type=float, default=DEFAULT_RECUR_WEIGHT)
    ap.add_argument("--ou-pA", type=float, default=8.0)
    ap.add_argument("--max-stories", type=int, default=20000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    print(f"[affect-biases] building DR-2 concept valence tags (max_stories={a.max_stories}) ...", flush=True)
    concepts = build_concepts(a.max_stories if not a.smoke else min(a.max_stories, 8000),
                              independent_valence=True)
    print(f"  {concepts['n']} concepts | no-confab moat corr(valence, PPMI relatedness) independent "
          f"{concepts['independent_value_plausibility_corr']:+.3f} (want |r|<0.15)", flush=True)

    if a.smoke:
        smoke = run_smoke(a.seeds[0], concepts, a.recur_weight, ou_pA=a.ou_pA)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(a.out).replace(".json", "_smoke.json")).write_text(json.dumps(smoke, indent=2, default=str))
        print(f"[affect-biases SMOKE] wrote {str(a.out).replace('.json', '_smoke.json')} "
              f"({round(time.time() - t0, 1)}s)", flush=True)
        return 0

    print(f"[affect-biases] 6-seed anti-cheat battery @ recur_weight={a.recur_weight}", flush=True)
    rows = [run_seed(s, concepts, a.recur_weight, ou_pA=a.ou_pA) for s in a.seeds]

    n_go = sum(1 for r in rows if r["GO"])
    axis_names = ["mood_congruent_recall", "arousal_encoding", "speak_rate", "hedge_probability"]

    def axis_mean(name, kind):
        return float(np.mean([r["axes"][name][kind] for r in rows]))

    agg_checks = {}
    for name in axis_names:
        agg_checks[f"{name}_present_all_seeds"] = all(r["axis_checks"][name]["present"] for r in rows)
        agg_checks[f"{name}_lesion_collapse_all_seeds"] = all(
            r["axis_checks"][name]["lesion_collapse<=0.2"] for r in rows)
        agg_checks[f"{name}_yoked_misdirect_all_seeds"] = all(
            r["axis_checks"][name]["yoked_misdirect<0.5"] for r in rows)
    agg_checks["no_confab_moat_intact(|r|<0.15)"] = abs(axis_mean("mood_congruent_recall", "intact")) >= 0 and \
        all(r["moat_ok"] for r in rows)

    all_four_present = all(agg_checks[f"{n}_present_all_seeds"] for n in axis_names)
    all_four_lesion = all(agg_checks[f"{n}_lesion_collapse_all_seeds"] for n in axis_names)
    all_four_yoked = all(agg_checks[f"{n}_yoked_misdirect_all_seeds"] for n in axis_names)
    moat_intact = all(r["moat_ok"] for r in rows)
    go = all(agg_checks.values())

    means = {name: {kind: axis_mean(name, kind) for kind in ("intact", "lesion", "yoked")} for name in axis_names}

    if go:
        verdict = (f"GO ({len(a.seeds)}-seed) — persistent core-affect CAUSALLY biases cognition along ALL FOUR "
                   f"axes, every one lesion-load-bearing on ONE spiking bridge: (1) mood-congruent recall "
                   f"{means['mood_congruent_recall']['intact']:+.3f}; (2) arousal-modulated encoding "
                   f"{means['arousal_encoding']['intact']:+.3f}; (3) speak-rate {means['speak_rate']['intact']:+.3f}; "
                   f"(4) hedge-probability {means['hedge_probability']['intact']:+.3f}. ALL FOUR collapse to <=0.2x "
                   f"under the affect_out lesion and mis-direct (<0.5x) under yoked-random affect; no-confab moat "
                   f"intact (|r|<0.15). This is the faculty that COLORS conversation (content/encoding/rate/stance "
                   f"all move with mood x arousal). Reuse of the P0.3 affect-state organ; numpy-CPU; NO sim/ edit.")
    else:
        miss = [k for k, v in agg_checks.items() if not v]
        verdict = (f"PARTIAL/BOUNDARY ({len(a.seeds)}-seed, {n_go}/{len(a.seeds)} seeds GO) — FAILED {miss}. "
                   f"present4={all_four_present} lesion4={all_four_lesion} yoked4={all_four_yoked} "
                   f"moat={moat_intact}. means "
                   + " | ".join(f"{n}:{means[n]['intact']:+.3f}(les {means[n]['lesion']:+.3f})" for n in axis_names))

    summary = {
        "probe": "affect_biases_cognition (Lane A · four-axis affective coloring)", "verdict": verdict, "GO": bool(go),
        "n_seeds_go": n_go, "aggregate_checks": agg_checks, "means": means,
        "all_four_present": bool(all_four_present), "all_four_lesion_collapse": bool(all_four_lesion),
        "all_four_yoked_misdirect": bool(all_four_yoked), "no_confab_moat_intact": bool(moat_intact),
        "per_seed": rows,
        "config": {"seeds": a.seeds, "recur_weight": a.recur_weight, "ou_pA": a.ou_pA,
                   "max_stories": a.max_stories, "n_concepts": concepts["n"],
                   "N_ENC": N_ENC, "N_ACC": N_ACC, "BIAS_WEIGHT": BIAS_WEIGHT,
                   "ENCODE_STIM_PA": ENCODE_STIM_PA, "HEDGE_BASE_PA": HEDGE_BASE_PA},
        "mechanism": "Subclass of the P0.3 GO affect-state organ (3 opponent slow-NMDA pools + Namburi-Tye "
                     "cross-inhibition + diffuse-neuromodulator appraisal + affect_out gate). Four synaptic bias "
                     "axes, all through the single affect_out gate: affect_vplus/vminus->recall (mood-congruent), "
                     "affect_arousal->encode_pool (GANE encoding gain), affect_arousal->speak_acc (vigor), "
                     "affect_vminus/vplus->hedge/assert biased WTA (hedge probability).",
        "HONEST_NOTE": "numpy-CPU read (real spiking Izhikevich bridge — 'numpy' is the backend, not a host "
                       "shortcut). Reuse-by-import of the affect-state organ (no organ edit, no sim/ edit — "
                       "encode/hedge/assert pools appended via the organ's pre-existing extra_regions/pathways "
                       "seam). Each axis is measured on a fresh brain (separate seeds share cfg.seed); the four "
                       "axes share the SAME affect_out lesion, so the single-gate collapse is a genuine "
                       "one-mechanism dependency. DR-2 Warriner-approximate core lexicon (independent-RNG tags for "
                       "the moat). Recall/encode/hedge read-outs are firing-rate reads of the relevant pools.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[affect-biases] VERDICT: {verdict}", flush=True)
    print(f"[affect-biases] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
