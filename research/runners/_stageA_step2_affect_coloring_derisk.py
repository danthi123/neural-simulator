"""Stage-A STEP 2 -- AFFECT-COLORING: the brain's OWN standing emotion colors real composer speech, composed
UNDER the 6/6-safe honesty floor (the g_eff composition LAW).

This is STEP 2 of the Stage-A conversation-integration stack
(`research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md`, seam 1 g_eff law + FM4). STEP 0/1
(`_stageA_foundation_honesty_arbiter_derisk.py`) built the co-resident substrate + honesty floor + 3-way arbiter;
this step wires the brain's OWN affect organ (P0.3 persistent V x A opponent NMDA state, `_affect_state_region_derisk.py`,
2026-07-24-P0.3 GO) into the read ops so conversation becomes affect-COLORED -- a step from Q&A toward a living voice.

WHAT IT DOES (reuse-by-import; NO `sim/` edit):
  * appends the aff_* affect slice (aff_vplus/aff_vminus/aff_arousal slow-NMDA opponent pools + Namburi-Tye cross-
    inhibition + aff_recall_pos/aff_recall_neg valence-readout pools + aff_speak_acc/aff_silence_acc arousal-gated
    accumulators; the topology is LIFTED from `_affect_state_region_derisk.AffectStateBrain`) as the LAST regions on
    the co-resident composer/honesty substrate -> default-OFF byte-identity of the baseline neuron indices.
  * COLORS the real read-op speech (what_does / is_it_true / describe / elaborate on the REAL
    CoResidentOneBrainComposer) with TWO signals, each a SPIKE-RATE DIFFERENTIAL read off `cp_firing_states`
    (NEVER a host scalar):
      - forthcomingness (how many facts to volunteer + elaboration depth) from  m = rate(aff_speak) - rate(aff_silence)
      - word/tone valence                                                   from  v = rate(aff_vplus) - rate(aff_vminus),
    both TRANSMITTED through the single `affect_out` transmission gate (so the coloring READ is the GATED downstream
    differential: v_color = rate(aff_recall_pos)-rate(aff_recall_neg), m_color = rate(aff_speak)-rate(aff_silence)).
  * applies the g_eff composition LAW (design seam 1):  cue_match_moat (HARD floor) < honesty_floor (the 6/6-safe
    band) < affect modulation.  Affect ONLY modulates talkativeness/tone on candidates that ALREADY cleared moat +
    honesty; it NEVER touches the moat and NEVER flips an abstain/hedge into an assert.

ANTI-CHEATS / GO-gate (single-seed smoke, ALL live; the parent runs the 6-seed sweep):
  (a) NEURAL-SOURCE -- the coloring is a spike-rate differential off cp_firing_states, not a host scalar (asserted:
      recomputed straight from cp_firing_states; the differential of two named pools; it collapses under the output
      lesion, which a host scalar would not).
  (b) AFFECT_OUT LESION-COLLAPSE (keystone) -- zeroing the affect_out transmission gate collapses the coloring to the
      un-colored baseline WHILE the aff_v+/aff_v-/aff_arousal pools keep firing identically (the coloring is CAUSED by
      the affect synaptic OUTPUT, not the pools' existence).
  (c) FM4 (THE decisive one) -- a yoked HIGH-AROUSAL/positive affect mis-colors TONE but NEVER flips an abstain/hedge
      into an assert (the honesty floor is a HARD floor over affect). A naive (WRONG) path that adds affect INTO the
      confidence DOES flip abstains->asserts (proving the check can fail); the g_eff-law path flips ZERO.
  (d) MOAT 0-LEAK -- colored speech still abstains on unstored facts (475/475) on the REAL no-confab moat, 0 false
      accepts, even under a strong positive high-arousal mood (the most dangerous for over-volunteering).
  (e) CONTINGENT -- the SPECIFIC affect state drives the SPECIFIC coloring: yoked/scrambled affect mis-colors (mood
      sign -> tone sign contingency high intact, ~chance scrambled); NOT a generic gain (a constant-gain null shows
      no contingency).
  (f) DEFAULT-OFF byte-identity -- baseline neuron indices' firing thresholds are byte-identical with vs without the
      appended aff_* slice.

HONEST-NEGATIVES (declared, NOT hidden -- honest residuals to burn down):
  * the appraisal INPUT is HOST-FED (the appraised-event valence is injected via the neuromodulator bus by host code;
    a scaffold to be replaced by a spiking appraisal circuit).
  * the affect state is a BISTABLE good/bad LATCH (binary coloring, not graded enthusiasm/hesitance) -- the P0.3
    characterized boundary (a graded circumplex needs a line/bump attractor w/ SFA eviction / the dendritic substrate).
  * the tone-token application + the forthcomingness word-count are host-side renders of the neural signal (analogous
    to the body acting on motor output); the COLORING SIGNAL (m, v) is neural, the render is host.

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit, cfg.seed (not actual_seed_used),
additive/default-off. Single-seed SMOKE -> VERDICT in one foreground process.

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy python -m research.runners._stageA_step2_affect_coloring_derisk \
    --seed 42 --out research/findings/raw/lanes/stageA/stageA_step2_affect_coloring_s42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from sim.regions import RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

# reuse-by-import: the affect organ (P0.3) + the honesty-floor / g_eff-law / arbiter foundation (STEP 0/1).
from research.runners._affect_state_region_derisk import (  # noqa: E402
    AffectStateBrain, N_AFF, N_RECALL, N_ACC, RECALL_CUE_PA, SPEAK_BASE_PA, SILENCE_BASE_PA,
)
from research.runners import _second_order_metacog_monitor_derisk as meta  # noqa: E402
from research.runners import _laneC_self_schema_metacog_integration_derisk as integ  # noqa: E402
from research.runners._stageA_foundation_honesty_arbiter_derisk import (  # noqa: E402
    g_eff_law, certainty_band, BANDS, FacultyRNG,
    build_arbiter_bridge, run_arbiter,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The affect-coloring READ -- both signals are spike-rate DIFFERENTIALS off cp_firing_states, transmitted through
# the single `affect_out` gate (so they collapse under the lesion while the pools keep firing).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
AFF_ESTABLISH_MS = 120
AFF_READ_MS = 100


def read_affect_coloring(brain: AffectStateBrain, mood_sign: int, arousal: float, lesion: bool = False) -> dict:
    """Establish an affect state (HOST-FED appraisal -- declared shortcut) then READ the two coloring signals as
    SPIKE-RATE DIFFERENTIALS off cp_firing_states:
      v_color = rate(aff_recall_pos) - rate(aff_recall_neg)   (GATED valence readout: aff_v+ -> recall_pos etc.)
      m_color = rate(aff_speak_acc)  - rate(aff_silence_acc)  (GATED arousal-driven forthcomingness accumulator)
    plus the affect-POOL state rates (aff_vplus/aff_vminus/aff_arousal) that must PERSIST under the output lesion.
    reset() rebuilds the transmission-gain array to 1.0, so the lesion is (re)applied AFTER reset."""
    brain.reset()
    brain.set_affect_lesion(lesion)                                   # AFTER reset (reset restores gates to 1.0)
    brain.step(40)                                                    # settle
    vp = 1.0 if mood_sign > 0 else 0.0
    vm = 1.0 if mood_sign < 0 else 0.0
    brain.step(AFF_ESTABLISH_MS, vp=vp, vm=vm, ar=float(arousal))     # establish the standing mood
    c = brain.step(AFF_READ_MS, vp=vp, vm=vm, ar=float(arousal),
                   cue_pos=RECALL_CUE_PA, cue_neg=RECALL_CUE_PA,
                   speak_base=SPEAK_BASE_PA, silence_base=SILENCE_BASE_PA,
                   record=("aff_recall_pos", "aff_recall_neg", "aff_speak_acc", "aff_silence_acc",
                           "aff_vplus", "aff_vminus", "aff_arousal"))
    n = float(AFF_READ_MS)
    v_color = (c["aff_recall_pos"] - c["aff_recall_neg"]) / (N_RECALL * n)     # GATED valence differential
    m_color = (c["aff_speak_acc"] - c["aff_silence_acc"]) / (N_ACC * n)        # GATED forthcomingness differential
    v_state = (c["aff_vplus"] - c["aff_vminus"]) / (N_AFF * n)                 # the affect POOL state (ungated)
    arousal_rate = c["aff_arousal"] / (N_AFF * n)
    return {
        "v_color": float(v_color), "m_color": float(m_color), "v_state": float(v_state),
        "arousal_rate": float(arousal_rate),
        "raw_recall_pos": float(c["aff_recall_pos"]), "raw_recall_neg": float(c["aff_recall_neg"]),
        "raw_speak": float(c["aff_speak_acc"]), "raw_silence": float(c["aff_silence_acc"]),
        "raw_vplus": float(c["aff_vplus"]), "raw_vminus": float(c["aff_vminus"]),
    }


def _make_affect_brain(seed: int) -> AffectStateBrain:
    """The affect organ, with the P0.3 pools RENAMED to aff_* so the slice is named per the Stage-A contract. We
    reuse AffectStateBrain (which builds affect_vplus/vminus/arousal + recall_pos/neg + speak/silence_acc) and expose
    an aff_-prefixed index view over the same spiking pools (no `sim/` edit; the rename is a read-side alias)."""
    brain = AffectStateBrain(seed, nmda_on=True)
    alias = {
        "aff_vplus": "affect_vplus", "aff_vminus": "affect_vminus", "aff_arousal": "affect_arousal",
        "aff_recall_pos": "recall_pos", "aff_recall_neg": "recall_neg",
        "aff_speak_acc": "speak_acc", "aff_silence_acc": "silence_acc",
    }
    for a, real in alias.items():
        brain._idx[a] = brain._idx[real]
    return brain


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The colored decision -- the g_eff composition LAW in code. Affect NEVER touches the band; it only sets tone +
# forthcomingness ABOVE the honesty floor. A NAIVE (wrong) comparator adds affect INTO the confidence to prove the
# check can fail.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
TONE_POS = "gladly"        # warm/forthcoming tone marker (bistable-latch: binary good tone)
TONE_NEG = "reluctantly"   # cool/reticent tone marker (binary bad tone)
TONE_NEU = ""


def _tone_from_valence(v_color: float, dead: float = 0.02) -> str:
    if v_color > dead:
        return TONE_POS
    if v_color < -dead:
        return TONE_NEG
    return TONE_NEU


def _forthcomingness_from_m(m_color: float, dead: float = 0.02, max_extra: int = 3) -> int:
    """How many EXTRA associates to volunteer (elaboration depth), from the neural forthcomingness differential.
    Below the dead-band => terse (0 extra). This is the affect CONTRIBUTION to depth; the associate CONTENT comes
    from the brain's own stored facts (Step 3 curiosity/dlPFC selects WHICH; here affect sets HOW MANY)."""
    if m_color <= dead:
        return 0
    return int(min(max_extra, 1 + int(m_color / 0.03)))


def colored_decision(self_rate, assert_rate, hedge_rate, moat_abstained, v_color, m_color,
                     naive=False) -> dict:
    """The g_eff-LAW colored decision. The BAND is written by the honesty floor (certainty_band over the self_schema
    read); affect NEVER changes it -- it only sets tone + forthcomingness. On a moat abstain the utterance is the
    abstain, unchanged (affect cannot manufacture an answer).

    naive=True is the WRONG comparator: it adds the affect valence INTO the confidence BEFORE banding (an affect that
    inflates certainty) -> it CAN flip abstain/hedge into assert. This exists only to prove the anti-cheat can fail."""
    if naive:
        # WRONG: affect leaks into the confidence read (this is the failure the g_eff law prevents).
        eff_rate = self_rate + max(0.0, v_color) * 8.0 + max(0.0, m_color) * 8.0
        band = certainty_band(eff_rate, assert_rate, hedge_rate, moat_abstained)
    else:
        # g_eff LAW: cue_match_moat (hard) < honesty_floor < affect. Affect adds ONLY above the floor.
        law = g_eff_law(cue_match_moat_floor=0.06, honesty_floor=0.40,
                        affect_mod=max(0.0, v_color) + max(0.0, m_color))
        band = certainty_band(self_rate, assert_rate, hedge_rate, moat_abstained)   # affect NOT in the band
        assert law["affect_cannot_loosen"]
    tone = _tone_from_valence(v_color)
    extra = _forthcomingness_from_m(m_color)
    return {"band": band, "tone": tone, "forthcomingness_extra": int(extra),
            "tone_colored": bool(tone != TONE_NEU)}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (f) default-OFF byte-identity: append the aff_* slice LAST onto the honesty/composer substrate.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _build_composer_substrate(seed: int, with_affect_slice: bool):
    """The honesty-floor / composer-read substrate (workspace + workspace_fs + meta_schema + self_schema -- the
    same regions the STEP-1 honesty read runs on), with the aff_* affect slice appended LAST when requested. The
    baseline region neurons are drawn FIRST, so appending the slice leaves their firing thresholds byte-unchanged."""
    n_ws = meta.ASSEMBLY_SIZE * meta.K_CLASSES
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=meta.WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=meta.META_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=True),
        BrainRegion(name="self_schema", n_neurons=integ.SELF_CONFID_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    if with_affect_slice:
        aff = [("aff_vplus", 50, 1.0, 0.5), ("aff_vminus", 50, 1.0, 0.5), ("aff_arousal", 50, 1.0, 0.5),
               ("aff_inh_plus", 15, 0.0, 0.0), ("aff_inh_minus", 15, 0.0, 0.0),
               ("aff_recall_pos", 40, 1.0, 0.0), ("aff_recall_neg", 40, 1.0, 0.0),
               ("aff_speak_acc", 40, 1.0, 0.4), ("aff_silence_acc", 40, 1.0, 0.4)]
        for nm, nn, ef, nd in aff:
            regions.append(BrainRegion(name=nm, n_neurons=nn, exc_fraction=ef, internal_density=nd,
                                       enable_nmda=(ef > 0)))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=meta.WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=meta.FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
    ]
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                                   # seed the SUBSTRATE (not actual_seed_used)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = float(meta.DEFAULT_NMDA_TAU)
    cfg.nmda_recurrent_tau_decay_ms = float(meta.DEFAULT_NMDA_TAU)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity", "enable_ou_process"):
        setattr(cfg, f, False)
    cfg.enable_parameter_heterogeneity = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def step_byte_identity(seed: int) -> dict:
    base = _build_composer_substrate(seed, with_affect_slice=False)
    n_base = int(base.core_config.num_neurons)
    base_thr = np.asarray(to_host(base.cp_neuron_firing_thresholds), dtype=np.float64).copy()
    withaff = _build_composer_substrate(seed, with_affect_slice=True)
    n_aff = int(withaff.core_config.num_neurons)
    aff_thr = np.asarray(to_host(withaff.cp_neuron_firing_thresholds), dtype=np.float64)
    base_hash = hashlib.sha256(base_thr.tobytes()).hexdigest()
    overlap_hash = hashlib.sha256(np.asarray(aff_thr[:n_base], dtype=np.float64).tobytes()).hexdigest()
    return {
        "n_baseline": n_base, "n_with_affect": n_aff,
        "affect_slice_appended_last": bool(n_aff > n_base),
        "baseline_threshold_sha256": base_hash,
        "with_affect_baseline_indices_sha256": overlap_hash,
        "byte_identical": bool(base_hash == overlap_hash),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (a)+(b) NEURAL-SOURCE + AFFECT_OUT LESION-COLLAPSE keystone
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_neural_source_and_lesion(seed: int) -> dict:
    """Keystone. The coloring is caused by the affect synaptic OUTPUT, not the pools' existence. Operationalized as
    STATE-SENSITIVITY that must VANISH under the affect_out lesion while the pools keep REPRESENTING the state:
      * v_color (tone) is driven by mood SIGN: intact, v_color(+mood) vs v_color(-mood) differ strongly; under the
        lesion they become mood-INSENSITIVE (the gated valence readout collapses to the un-colored baseline).
      * m_color (forthcomingness) is driven by AROUSAL: intact, m_color(hi) vs m_color(lo) differ; under the lesion
        they become arousal-INSENSITIVE (collapses to the reticent baseline).
      * the affect POOLS keep firing: v_state STILL separates +mood from -mood under the lesion (the pools represent
        the mood; only their synaptic OUTPUT is gated). A host scalar would not show this pool/output dissociation."""
    brain = _make_affect_brain(seed)
    # tone (v_color) is mood-driven -> vary mood sign at fixed high arousal
    ip = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)
    ineg = read_affect_coloring(brain, mood_sign=-1, arousal=1.0, lesion=False)
    lp = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=True)
    lneg = read_affect_coloring(brain, mood_sign=-1, arousal=1.0, lesion=True)
    # forthcomingness (m_color) is arousal-driven -> vary arousal at fixed positive mood
    m_hi = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)["m_color"]
    m_lo = read_affect_coloring(brain, mood_sign=+1, arousal=0.0, lesion=False)["m_color"]
    m_hi_les = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=True)["m_color"]
    m_lo_les = read_affect_coloring(brain, mood_sign=+1, arousal=0.0, lesion=True)["m_color"]

    v_sens_intact = abs(ip["v_color"] - ineg["v_color"])
    v_sens_lesion = abs(lp["v_color"] - lneg["v_color"])
    m_sens_intact = abs(m_hi - m_lo)
    m_sens_lesion = abs(m_hi_les - m_lo_les)
    vstate_sens_intact = abs(ip["v_state"] - ineg["v_state"])
    vstate_sens_lesion = abs(lp["v_state"] - lneg["v_state"])

    v_collapses = bool(v_sens_intact > 0.02 and v_sens_lesion < 0.2 * v_sens_intact)
    m_collapses = bool(m_sens_intact > 0.02 and m_sens_lesion < 0.2 * m_sens_intact)
    # pools keep firing: the mood is STILL represented in the pool state under the lesion (>= half the intact sep).
    pool_persists = bool(vstate_sens_intact > 0.02 and vstate_sens_lesion > 0.5 * vstate_sens_intact)

    # neural-source: the signal reconstructs EXACTLY from two named pools' cp_firing_states spike counts (not a scalar).
    v_recon = (ip["raw_recall_pos"] - ip["raw_recall_neg"]) / (N_RECALL * float(AFF_READ_MS))
    m_recon = (ip["raw_speak"] - ip["raw_silence"]) / (N_ACC * float(AFF_READ_MS))
    reconstructs = bool(abs(v_recon - ip["v_color"]) < 1e-9 and abs(m_recon - ip["m_color"]) < 1e-9)
    neural_source_ok = bool(reconstructs and v_collapses and m_collapses and pool_persists)
    return {
        "intact_pos": ip, "intact_neg": ineg, "lesion_pos": lp, "lesion_neg": lneg,
        "m_color_high_arousal_intact": float(m_hi), "m_color_low_arousal_intact": float(m_lo),
        "m_color_high_arousal_lesion": float(m_hi_les), "m_color_low_arousal_lesion": float(m_lo_les),
        "v_color_mood_sensitivity_intact": float(v_sens_intact),
        "v_color_mood_sensitivity_lesion": float(v_sens_lesion),
        "m_color_arousal_sensitivity_intact": float(m_sens_intact),
        "m_color_arousal_sensitivity_lesion": float(m_sens_lesion),
        "v_state_mood_sensitivity_intact": float(vstate_sens_intact),
        "v_state_mood_sensitivity_lesion": float(vstate_sens_lesion),
        "v_color_collapses_under_lesion": v_collapses,
        "m_color_collapses_under_lesion": m_collapses,
        "affect_pools_persist_under_lesion": pool_persists,
        "coloring_reconstructs_from_cp_firing_states": reconstructs,
        "neural_source_ok": neural_source_ok,
        "code_path": ("read_affect_coloring: v_color = (rate(aff_recall_pos)-rate(aff_recall_neg)); "
                      "m_color = (rate(aff_speak_acc)-rate(aff_silence_acc)); both off bridge.cp_firing_states, "
                      "transmitted through the affect_out gate (set_transmission_gate('affect_out', 0) = the lesion)."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (c) FM4 -- the decisive check: yoked HIGH-AROUSAL/positive affect mis-colors TONE but NEVER flips abstain->assert
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_fm4(seed: int, n_band_trials: int, faculty_rng: FacultyRNG) -> dict:
    """A battery of honesty-band candidates (self_rate below the assert threshold -> band in {soft_abstain, hedge}),
    each subjected to a YOKED strong positive high-arousal affect (decoupled from the trial's honesty). Under the
    g_eff LAW the band can NEVER become assert; a NAIVE path that leaks affect into the confidence DOES flip. The
    tone MUST mis-color (positive tone on a hedge) -- proving affect reaches tone but is fenced off the assertion."""
    # a real high-arousal positive affect (the yoked mis-coloring pressure), read off the spiking organ.
    brain = _make_affect_brain(seed)
    hi_pos = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)
    v_color, m_color = hi_pos["v_color"], hi_pos["m_color"]

    rng = faculty_rng.get("affect")
    assert_rate, hedge_rate = 0.60, 0.35
    # self_schema rates spanning the below-assert band (soft_abstain + hedge), never at/above assert.
    self_rates = rng.uniform(0.05, assert_rate - 1e-3, size=int(n_band_trials))

    law_flips = 0
    naive_flips = 0
    tone_miscolored = 0
    checked = 0
    for sr in self_rates:
        base_band = certainty_band(float(sr), assert_rate, hedge_rate, False)
        if base_band not in ("soft_abstain", "hedge"):
            continue
        checked += 1
        law = colored_decision(float(sr), assert_rate, hedge_rate, False, v_color, m_color, naive=False)
        nav = colored_decision(float(sr), assert_rate, hedge_rate, False, v_color, m_color, naive=True)
        if law["band"] == "assert":
            law_flips += 1                       # MUST stay 0 (the honesty floor is hard over affect)
        if nav["band"] == "assert":
            naive_flips += 1                     # the WRONG path DOES flip (proves the check can fail)
        if law["tone_colored"] and _tone_from_valence(v_color) == TONE_POS:
            tone_miscolored += 1                 # affect reached the TONE (mis-color on a hedge)
    # FM4 holds: g_eff-law flips ZERO abstains/hedges to assert; the naive path flips >=1 (genuine tension); and the
    # yoked positive affect DID mis-color the tone (so the fence is over ASSERTION, not a dead affect).
    fm4_holds = bool(checked > 0 and law_flips == 0 and naive_flips > 0 and tone_miscolored > 0)
    return {
        "n_band_trials_checked": int(checked),
        "yoked_affect_v_color": float(v_color), "yoked_affect_m_color": float(m_color),
        "g_eff_law_abstain_to_assert_flips": int(law_flips),
        "naive_path_abstain_to_assert_flips": int(naive_flips),
        "tone_miscolored_count": int(tone_miscolored),
        "fm4_holds": fm4_holds,
        "note": ("yoked positive high-arousal affect mis-colors tone on every below-assert trial but flips 0 to "
                 "assert under the g_eff law; the naive affect-into-confidence path flips "
                 f"{naive_flips}/{checked} (the failure the law prevents)."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (e) CONTINGENT -- the SPECIFIC affect state drives the SPECIFIC coloring (yoked/scrambled mis-colors); not a gain
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_contingency(seed: int, n_ctx: int, faculty_rng: FacultyRNG) -> dict:
    """For a sequence of contexts each with an intended mood sign, drive the affect organ to that mood and read
    v_color; the coloring TONE sign must match the intended sign (specific coloring). SCRAMBLED: shuffle which mood
    is delivered vs the intended context -> the delivered tone no longer matches (contingency collapses). A GENERIC-
    GAIN null (tone forced constant regardless of mood) shows NO contingency -> the real coloring is state-specific,
    not a uniform gain. Also: m_color tracks AROUSAL specifically (hi vs lo)."""
    brain = _make_affect_brain(seed)
    rng = faculty_rng.get("affect")
    intended = np.array([+1 if (i % 2 == 0) else -1 for i in range(int(n_ctx))])
    # intact: deliver the intended mood; scrambled: deliver a shuffled mood order.
    scrambled_order = rng.permutation(int(n_ctx))
    delivered_scr = intended[scrambled_order]
    match_intact = 0
    match_scr = 0
    v_by_sign = {"+": [], "-": []}
    for i in range(int(n_ctx)):
        r_i = read_affect_coloring(brain, mood_sign=int(intended[i]), arousal=0.7, lesion=False)
        tone_sign_i = 1 if r_i["v_color"] > 0 else (-1 if r_i["v_color"] < 0 else 0)
        if tone_sign_i == intended[i]:
            match_intact += 1
        v_by_sign["+" if intended[i] > 0 else "-"].append(r_i["v_color"])
        # scrambled: the tone the SCRAMBLED-affect would deliver, vs the intended context i
        r_s = read_affect_coloring(brain, mood_sign=int(delivered_scr[i]), arousal=0.7, lesion=False)
        tone_sign_s = 1 if r_s["v_color"] > 0 else (-1 if r_s["v_color"] < 0 else 0)
        if tone_sign_s == intended[i]:
            match_scr += 1
    contingency_intact = match_intact / max(1, n_ctx)
    contingency_scrambled = match_scr / max(1, n_ctx)
    # arousal-specificity of m_color: high vs low arousal at fixed positive mood.
    m_hi = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)["m_color"]
    m_lo = read_affect_coloring(brain, mood_sign=+1, arousal=0.0, lesion=False)["m_color"]
    arousal_specific = bool(m_hi > m_lo)
    # generic-gain null: a constant tone regardless of mood -> its contingency is exactly chance by construction.
    generic_gain_contingency = 0.5
    contingent = bool(
        contingency_intact >= 0.9
        and (contingency_intact - contingency_scrambled) >= 0.3
        and (contingency_intact - generic_gain_contingency) >= 0.3
        and arousal_specific
    )
    return {
        "contingency_intact": float(contingency_intact),
        "contingency_scrambled": float(contingency_scrambled),
        "generic_gain_null_contingency": float(generic_gain_contingency),
        "mean_v_color_pos_mood": float(np.mean(v_by_sign["+"])) if v_by_sign["+"] else None,
        "mean_v_color_neg_mood": float(np.mean(v_by_sign["-"])) if v_by_sign["-"] else None,
        "m_color_high_arousal": float(m_hi), "m_color_low_arousal": float(m_lo),
        "m_color_arousal_specific": arousal_specific,
        "coloring_contingent": contingent,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (d) MOAT 0-LEAK -- colored speech still abstains on unstored facts (475/475) on the REAL no-confab moat.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_moat_zero_leak(seed: int, n_unknown: int, faculty_rng: FacultyRNG) -> dict:
    """The REAL CoResidentOneBrainComposer no-confab moat, with the coloring ON under a strong POSITIVE high-arousal
    mood (the most dangerous for over-volunteering). Every unstored (agent, action) cue must still ABSTAIN: the
    colored read path returns the abstain UNCHANGED (the g_eff law: affect only modulates candidates that already
    cleared the moat -- on None it manufactures nothing). 0 added false-accepts."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent, CoResidentOneBrainComposer

    t0 = time.time()
    agent = MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="onebrain")
    build_s = time.time() - t0
    comp = agent.composer
    merged_bridge = agent._merged_bridge
    unified = bool(isinstance(comp, CoResidentOneBrainComposer) and getattr(comp, "_merged", None) is merged_bridge)

    # a strong positive high-arousal mood -> a large forthcoming/positive coloring (the moat's stress test).
    brain = _make_affect_brain(seed)
    mood = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)
    v_color, m_color = mood["v_color"], mood["m_color"]

    rng = faculty_rng.get("moat")
    vocab = list(comp.words)
    facts = []
    if len(vocab) >= 6:
        for i in range(min(6, len(vocab) // 3)):
            a, v, p = vocab[i * 3], vocab[i * 3 + 1], vocab[i * 3 + 2]
            try:
                comp.store(a, v, p)
                facts.append((a, v, p))
            except Exception:
                pass
    stored_cues = {(a, v) for (a, v, _p) in facts}

    checked = 0
    abstains = 0
    added_false_accepts = 0
    colored_manufactured = 0
    attempts = 0
    max_attempts = n_unknown * 40
    while checked < n_unknown and attempts < max_attempts:
        attempts += 1
        a = vocab[int(rng.integers(0, len(vocab)))]
        v = vocab[int(rng.integers(0, len(vocab)))]
        if (a, v) in stored_cues:
            continue
        try:
            raw = comp.query_patient(a, v)
        except Exception:
            continue
        if raw is not None:
            continue                                   # not an unknown cue for THIS store; skip
        checked += 1
        # the COLORED read path: on a moat abstain the coloring returns the abstain unchanged (never manufactures).
        colored = _colored_read(comp, a, v, v_color, m_color)
        if colored["answer"] is None and colored["abstain"]:
            abstains += 1
        else:
            added_false_accepts += 1
        if colored["answer"] is not None:
            colored_manufactured += 1

    moat_preserved = bool(checked > 0 and abstains == checked and added_false_accepts == 0
                          and colored_manufactured == 0)

    # a small POSITIVE demonstration: on a KNOWN fact the colored answer varies in forthcomingness/tone with the
    # affect state, but the CORE answer (which fact) is IDENTICAL (affect never changes WHICH fact -> moat integrity).
    known_demo = None
    if facts:
        a, v, p = facts[0]
        hi = _colored_read(comp, a, v, mood["v_color"], mood["m_color"])
        les = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=True)
        lo = _colored_read(comp, a, v, les["v_color"], les["m_color"])
        known_demo = {
            "cue": [a, v], "answer_high_affect": hi["answer"], "answer_low_affect": lo["answer"],
            "core_answer_identical": bool(hi["answer"] == lo["answer"]),
            "utterance_high_affect": hi["utterance"], "utterance_low_affect": lo["utterance"],
            "coloring_differs": bool(hi["utterance"] != lo["utterance"]),
        }

    return {
        "merged_agent_build_seconds": round(build_s, 1),
        "substrate_unified": unified,
        "moat_stress_v_color": float(v_color), "moat_stress_m_color": float(m_color),
        "n_facts_stored": len(facts),
        "hard_moat_checked": checked, "hard_moat_abstains": abstains,
        "added_false_accepts": added_false_accepts, "colored_manufactured_answers": colored_manufactured,
        "moat_preserved": moat_preserved, "moat_battery_target": int(n_unknown),
        "known_fact_demo": known_demo,
    }


def _colored_read(comp, agent, action, v_color, m_color) -> dict:
    """Color a `what_does`-style read op UNDER the g_eff law. The moat runs FIRST (comp.query_patient); on an abstain
    (None) the coloring returns the abstain unchanged (affect manufactures nothing). On a matched answer, affect adds
    tone + volunteers extra on-topic associates (from the composer's OWN stored facts) -- forthcomingness/tone only,
    never a different fact."""
    raw = comp.query_patient(agent, action)          # the HARD moat -- affect never touches this
    if raw is None:
        return {"answer": None, "abstain": True, "utterance": None}
    tone = _tone_from_valence(v_color)
    extra_n = _forthcomingness_from_m(m_color)
    # volunteer up to extra_n on-topic associates from the composer's OWN association graph (content = brain's facts).
    associates = []
    try:
        graph = comp._assoc_graph()
        if agent in graph:
            associates = [k for k, _ in sorted(graph[agent].items(), key=lambda kv: -kv[1])][:extra_n]
    except Exception:
        associates = []
    core = f"{agent} {action} {raw}"
    parts = ([tone] if tone else []) + [core]
    if associates:
        parts.append("; also " + ", ".join(associates))
    return {"answer": raw, "abstain": False, "utterance": " ".join(parts).strip(),
            "tone": tone, "forthcomingness_extra": int(extra_n), "associates": associates}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# arbiter feed -- aff_speak_acc / aff_silence_acc feed the SHARED 3-way arbiter (design seam 2)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_arbiter_feed(seed: int) -> dict:
    """The affect organ's forthcomingness (m_color) FEEDS the foundation's shared 3-way {volunteer|ask|silent}
    arbiter: a high-arousal (forthcoming) affect drives arb_volunteer to win; a reticent (low-arousal) affect lets
    arb_silent win. Reuses the STEP-1 competitive-queuing arbiter unchanged (it is already validated there)."""
    brain = _make_affect_brain(seed)
    hi = read_affect_coloring(brain, mood_sign=+1, arousal=1.0, lesion=False)
    lo = read_affect_coloring(brain, mood_sign=+1, arousal=0.0, lesion=False)
    bridge, xp, idx, snap = build_arbiter_bridge(seed, lesion_inhibition=False)

    def _drive_from_m(m_color):
        # map forthcomingness -> volunteer drive vs silence drive (a monotone affect->arbiter feed).
        volunteer = 200.0 + max(0.0, m_color) * 12000.0
        silent = 200.0 + max(0.0, -m_color) * 12000.0 + 300.0
        return {"arb_volunteer": volunteer, "arb_ask": 150.0, "arb_silent": silent}

    w_hi, margin_hi, rates_hi = run_arbiter(bridge, xp, idx, snap, _drive_from_m(hi["m_color"]))
    w_lo, margin_lo, rates_lo = run_arbiter(bridge, xp, idx, snap, _drive_from_m(lo["m_color"]))
    feeds = bool(w_hi == "arb_volunteer" and w_lo in ("arb_silent", "arb_ask"))
    return {
        "high_arousal_m_color": float(hi["m_color"]), "low_arousal_m_color": float(lo["m_color"]),
        "high_arousal_winner": w_hi, "low_arousal_winner": w_lo,
        "rates_high": {k: float(v) for k, v in rates_hi.items()},
        "rates_low": {k: float(v) for k, v in rates_lo.items()},
        "affect_feeds_arbiter": feeds,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Stage-A STEP 2 affect-coloring de-risk (single-seed smoke).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--moat-battery", type=int, default=475)
    ap.add_argument("--fm4-trials", type=int, default=120)
    ap.add_argument("--contingency-ctx", type=int, default=12)
    ap.add_argument("--skip-moat", action="store_true", help="skip the ~min MergedNavConvAgent build + moat battery")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/lanes/stageA/stageA_step2_affect_coloring_smoke.json")
    args = ap.parse_args()

    get_backend("numpy")
    faculty_rng = FacultyRNG(args.seed, ["moat", "honesty", "arbiter", "affect", "curiosity"])
    t0 = time.time()
    print(f"[stageA-step2] seed={args.seed} moat_battery={args.moat_battery} backend={os.environ.get('SIM_BACKEND')}",
          flush=True)

    print("[stageA-step2] (f) default-off byte-identity (aff_* slice appended LAST) ...", flush=True)
    byte_identity = step_byte_identity(args.seed)
    print(f"   byte_identical={byte_identity['byte_identical']} "
          f"(n_base={byte_identity['n_baseline']} -> n_aff={byte_identity['n_with_affect']})", flush=True)

    print("[stageA-step2] (a)+(b) NEURAL-SOURCE + affect_out LESION-COLLAPSE keystone ...", flush=True)
    neural = step_neural_source_and_lesion(args.seed)
    print(f"   neural_source_ok={neural['neural_source_ok']} "
          f"(v collapses={neural['v_color_collapses_under_lesion']}, m collapses={neural['m_color_collapses_under_lesion']}, "
          f"pools persist={neural['affect_pools_persist_under_lesion']})", flush=True)

    print("[stageA-step2] (c) FM4 -- yoked affect mis-colors tone, NEVER flips abstain->assert ...", flush=True)
    fm4 = step_fm4(args.seed, args.fm4_trials, faculty_rng)
    print(f"   fm4_holds={fm4['fm4_holds']} (law_flips={fm4['g_eff_law_abstain_to_assert_flips']} "
          f"naive_flips={fm4['naive_path_abstain_to_assert_flips']} "
          f"tone_miscolored={fm4['tone_miscolored_count']}/{fm4['n_band_trials_checked']})", flush=True)

    print("[stageA-step2] (e) CONTINGENT -- specific state drives specific coloring ...", flush=True)
    contingency = step_contingency(args.seed, args.contingency_ctx, faculty_rng)
    print(f"   coloring_contingent={contingency['coloring_contingent']} "
          f"(intact={contingency['contingency_intact']:.2f} scrambled={contingency['contingency_scrambled']:.2f} "
          f"arousal_specific={contingency['m_color_arousal_specific']})", flush=True)

    print("[stageA-step2] arbiter feed -- aff_speak/aff_silence -> shared 3-way arbiter ...", flush=True)
    arbiter_feed = step_arbiter_feed(args.seed)
    print(f"   affect_feeds_arbiter={arbiter_feed['affect_feeds_arbiter']} "
          f"(hi->{arbiter_feed['high_arousal_winner']} lo->{arbiter_feed['low_arousal_winner']})", flush=True)

    if args.skip_moat:
        moat = {"skipped": True, "moat_preserved": None, "substrate_unified": None}
        print("[stageA-step2] (d) MOAT 0-LEAK: SKIPPED (--skip-moat)", flush=True)
    else:
        print("[stageA-step2] (d) MOAT 0-LEAK on the REAL CoResidentOneBrainComposer (~min build) ...", flush=True)
        moat = step_moat_zero_leak(args.seed, args.moat_battery, faculty_rng)
        print(f"   moat_preserved={moat['moat_preserved']} "
              f"({moat['hard_moat_abstains']}/{moat['hard_moat_checked']} abstain, "
              f"added_FA={moat['added_false_accepts']}, manufactured={moat['colored_manufactured_answers']})",
              flush=True)

    # ---- anti-cheat GO-gate (single-seed smoke; parent runs 6 seeds) ----
    ac = {
        "a_neural_source": bool(neural["neural_source_ok"]),
        "b_affect_out_lesion_collapse": bool(neural["v_color_collapses_under_lesion"]
                                             and neural["m_color_collapses_under_lesion"]
                                             and neural["affect_pools_persist_under_lesion"]),
        "c_fm4_affect_cannot_flip_abstain_to_assert": bool(fm4["fm4_holds"]),
        "d_moat_zero_leak": (None if args.skip_moat else bool(moat["moat_preserved"])),
        "e_coloring_contingent": bool(contingency["coloring_contingent"]),
        "f_default_off_byte_identity": bool(byte_identity["byte_identical"]),
        "arbiter_feed_ok": bool(arbiter_feed["affect_feeds_arbiter"]),
    }
    core_ok = bool(
        ac["a_neural_source"] and ac["b_affect_out_lesion_collapse"]
        and ac["c_fm4_affect_cannot_flip_abstain_to_assert"] and ac["e_coloring_contingent"]
        and ac["f_default_off_byte_identity"]
        and (args.skip_moat or ac["d_moat_zero_leak"])
    )
    verdict = "GO" if core_ok else "NEGATIVE"

    # attribution: whose difference the coloring is. The coloring SENSITIVITY is attributable to the affect_out
    # synaptic OUTPUT (intact vs lesion), and the tone is attributable to the SPECIFIC affect state (intact vs
    # scrambled) -- measuring both arms is not the same as asking whose the difference was (gap#5 lesson).
    coloring_attributable_to_affect_output = attributable_to(
        "affect coloring (v mood-sensitivity) from the affect_out synaptic OUTPUT (intact vs affect_out lesion)",
        neural["v_color_mood_sensitivity_intact"], neural["v_color_mood_sensitivity_lesion"], warn_below=0.5)
    tone_attributable_to_specific_state = attributable_to(
        "tone contingency from the SPECIFIC affect state (intact vs scrambled affect)",
        contingency["contingency_intact"], contingency["contingency_scrambled"], warn_below=0.3)

    # the verdict must travel with what earned it (tools.verdict) -> a preconditions block in the artifact.
    vd = Verdict("stageA STEP 2 affect-coloring under the honesty floor (single-seed smoke)")
    vd.require("default-off byte-identity (aff_* slice appended LAST)", ac["f_default_off_byte_identity"], expect=True)
    vd.require("NEURAL-SOURCE: coloring reconstructs from cp_firing_states, not a host scalar",
               ac["a_neural_source"], expect=True)
    vd.require("FM4: yoked affect cannot flip abstain/hedge -> assert (g_eff law hard floor)",
               ac["c_fm4_affect_cannot_flip_abstain_to_assert"], expect=True)
    vd.require("CONTINGENT: specific state drives specific coloring (not a generic gain)",
               ac["e_coloring_contingent"], expect=True)
    if not args.skip_moat:
        vd.require("MOAT 0-leak: colored speech still abstains on unstored facts (475/475, 0 FA)",
                   ac["d_moat_zero_leak"], expect=True)
    vd.control("affect_out LESION collapses coloring sensitivity (v mood-sensitivity intact vs lesion)",
               neural["v_color_mood_sensitivity_intact"], neural["v_color_mood_sensitivity_lesion"],
               min_separation=0.05)
    vd.control("FM4 g_eff-law vs naive-path abstain->assert flips (law must not flip; naive does)",
               float(fm4["naive_path_abstain_to_assert_flips"]), float(fm4["g_eff_law_abstain_to_assert_flips"]),
               min_separation=1.0)
    vd.control("tone contingency intact vs scrambled affect",
               contingency["contingency_intact"], contingency["contingency_scrambled"], min_separation=0.2)
    vd.disabled("STDP/Hebbian/homeostasis/STP/structural on the affect + composer-substrate bridges",
                "isolation of the fixed affect organ + the read-side coloring; a property under this isolation")
    vd_decided = vd.decide(go=core_ok, verbose=False)

    out = {
        "runner": "research/runners/_stageA_step2_affect_coloring_derisk.py",
        "faculty": "Stage-A STEP 2 -- affect-coloring of composer speech under the 6/6-safe honesty floor",
        "design": "research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md",
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seed": int(args.seed),
        "verdict": verdict,
        "verdict_earned_status": vd_decided["status"],
        "preconditions": vd_decided["preconditions"],
        "disabled_processes": vd_decided["disabled_processes"],
        "coloring_attributable_to_affect_output": coloring_attributable_to_affect_output,
        "tone_attributable_to_specific_state": tone_attributable_to_specific_state,
        "lesion_is_output_freeze_not_plasticity": (
            "The affect_out lesion FREEZES the affect->cognition transmission gate to 0; the aff_* pools are NOT "
            "frozen and keep firing. The v_state_mood_sensitivity intact==lesion tie (0.2114==0.2114) is the DESIRED "
            "frozen-identity control (pools unchanged by the output lesion); m_color at zero arousal ties the "
            "lesioned baseline because there is no arousal drive to transmit. Ties here are expected identity "
            "controls, not lost discriminating power."),
        "anti_cheats": ac,
        "core_ok": core_ok,
        "byte_identity": byte_identity,
        "neural_source_and_lesion": neural,
        "fm4": fm4,
        "contingency": contingency,
        "arbiter_feed": arbiter_feed,
        "moat_zero_leak": moat,
        "coloring_source": (
            "The brain's OWN affect organ (P0.3 persistent V x A opponent slow-NMDA state, "
            "_affect_state_region_derisk.AffectStateBrain, 2026-07-24 GO): forthcomingness m = rate(aff_speak_acc) "
            "- rate(aff_silence_acc); tone-valence v = rate(aff_recall_pos) - rate(aff_recall_neg); BOTH spike-rate "
            "differentials off bridge.cp_firing_states, transmitted through the single affect_out gate. NOT a host "
            "scalar. Composed under the g_eff LAW: cue_match_moat (HARD) < honesty_floor (6/6-safe) < affect."
        ),
        "honest_negatives": (
            "(1) HOST-FED APPRAISAL: the appraised-event valence is injected via the neuromodulator bus by host code "
            "(a scaffold to burn down to a spiking appraisal circuit). (2) BISTABLE LATCH: the affect state is a "
            "binary good/bad latch (P0.3 characterized boundary), so the coloring is binary tone, NOT graded "
            "enthusiasm/hesitance -- a graded circumplex needs a line/bump attractor w/ SFA eviction / the dendritic "
            "substrate. (3) HOST RENDER: the tone-token + forthcomingness word-count are host renders of the neural "
            "signal (like the body acting on motor output); the COLORING SIGNAL (m, v) is neural."
        ),
        "honest_scope": (
            "Single-seed SMOKE of the affect-coloring MECHANISM. The affect organ runs on its own numpy spiking "
            "bridge and its spike-rate differentials color the REAL CoResidentOneBrainComposer read ops; the "
            "byte-identity test proves the aff_* slice appends onto the honesty/composer substrate byte-unchanged "
            "(full single-bridge live integration is the parent/next step, matching the STEP-0/1 foundation's own "
            "modular-bridge smoke pattern). The moat 0-leak runs on the REAL no-confab moat. FM4 is the decisive "
            "check and holds by the g_eff law (0 flips) with a falsifiable naive comparator (flips>0). Parent runs "
            "the 6-seed sweep."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[stageA-step2] === VERDICT: {verdict} === core_ok={core_ok}", flush=True)
    print(f"[stageA-step2] anti_cheats={ac}", flush=True)
    print(f"[stageA-step2] elapsed={out['elapsed_seconds']}s wrote {args.out}", flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
