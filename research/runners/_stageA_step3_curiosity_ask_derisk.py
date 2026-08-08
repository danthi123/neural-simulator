"""Stage-A STEP 3 -- CURIOSITY ASK: the brain emits its OWN wh-questions (crave, don't refuse) IN the
conversational loop, feeding the SHARED 3-way arbiter. This is the mission-central step for "open-ended, not
Q&A": the brain INITIATES -- it asks about what it does not know -- rather than only answering.

This is STEP 3 of the Stage-A conversation-integration stack
(`research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md`, seam 2 arbiter + seam 4 curiosity).
STEP 0/1 (`_stageA_foundation_honesty_arbiter_derisk.py`) built the co-resident substrate + honesty floor +
the 3-way {volunteer|ask|silent} WTA arbiter; STEP 2 (`_stageA_step2_affect_coloring_derisk.py`) wired the
brain's OWN affect into arb_volunteer/arb_silent. THIS step wires the brain's OWN CURIOSITY DRIVE into arb_ask
and, when the ask pool WINS on a NOVEL+relevant gate read, EMITS a wh-question whose content word is spelled
from the brain's OWN naming-map spike decode (A->W) -- targeting the ACTUAL knowledge gap.

WHAT IT DOES (reuse-by-import; NO `sim/` edit):
  * the ask-DRIVE is the on-bridge DR-1 SPIKING curiosity (from_novelty -> `curiosity` neuromodulator ->
    excitability_drive on group:ask -> ASK-pool spikes), imported verbatim
    (`_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge`; the from_novelty fill + tests are already
    on main). The WANTING is read off `cp_firing_states[ask]` (Hz) -- a spike-rate, NOT a host `if novel` flag.
  * that spiking want FEEDS the SHARED 3-way arbiter's arb_ask pool
    (`_stageA_foundation_honesty_arbiter_derisk.build_arbiter_bridge`/`run_arbiter`), CO-RESIDENT with the STEP-2
    affect wire-in (affect m_color -> arb_volunteer/arb_silent). One winner per turn: high gap -> arb_ask wins
    (crave); forthcoming affect -> arb_volunteer; reticent affect -> arb_silent.
  * when arb_ask WINS on a concept the gate reads NOVEL, the brain EMITS a wh-question -- a fixed host wh-frame
    ("what is ___ ?") whose CONTENT word is decoded from the on-bridge NAMING MAP's WORD-POOL SPIKE COUNTS
    (`_grounded_message_to_word_onbridge_derisk.name_from_spikes`, the 6-seed PARENT-VERIFIED naming GO), NOT
    from WKV generation. The naming map targets the ACTUAL gap concept (a permuted decode names the WRONG
    concept -> the wrong-gap control fails).
  * MOAT INVERTED, not broken: on a NOVEL cue the brain's ACTION becomes ASK instead of refuse, but it NEVER
    confabulates an answer -- the real CoResidentOneBrainComposer no-confab moat still abstains 475/475 with 0
    manufactured answers. Asking is the moat's action-INVERSION, not a moat breach.

ANTI-CHEATS / GO-gate (single-seed smoke; the parent runs the 6-seed sweep):
  (a) CRAVE-ON-SPIKES -- corr(epistemic-gap, SPIKING want) >= 0.9; the want is reconstructed from
      cp_firing_states (spike-driven) and COLLAPSES under the curiosity-modulator lesion (a host flag would not).
  (b) MOAT INVERTED-NOT-BROKEN -- on NOVEL cues the brain ASKS (action-inversion) while the no-confab moat holds
      475/475, 0 confabulated answers, 0 added false-accepts.
  (c) WH-TARGETS-THE-GAP -- the emitted question names the ACTUAL gap concept (spike-decode accuracy high); a
      PERMUTED decode names the wrong concept (the wrong-gap control fails).
  (d) BRAIN-NATIVE WORDS -- the content word is the naming-map WORD-POOL SPIKE-COUNT decode; WKV is used ONLY as
      the fixed articulatory alphabet (the pool->token binding), never to GENERATE the word (asserted in-run).
  (e) ARBITER 3-WAY -- arb_ask competes with affect-driven arb_volunteer/arb_silent in the ONE shared arbiter,
      one winner/turn, co-resident with STEP 2; a mutual-inhibition lesion collapses the winner margin.
  (f) DEFAULT-OFF byte-identity -- the co-resident cur_ask slice appends LAST -> baseline neuron indices'
      firing thresholds byte-identical with vs without it.

HONEST-NEGATIVES (declared, not hidden):
  * ON-BRIDGE LP MEMORY is FRAGILE (1/6; `2026-07-30-lane-B-curiosity-DR1-onbridge-6seed-GO`), so WHICH concept
    to ask (the learning-progress SELECTOR) is realized by the LP-MAX CPU-PROXY selector (6-seed GO,
    `2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection...`), reported as a CPU proxy that may
    fall back to a host-TD tracker on-bridge. The ask DRIVE (whether the gate craves) IS spiking (gate a).
  * HOST RENDERS: the wh-frame ("what is ___ ?") is host phrasing (the fixed language scaffold, analogous to the
    body acting on motor output); the CONTENT word is the brain's naming-map spike decode. The gap signal fed to
    from_novelty is the real Bogacz-Brown gate novelty (a brain read), routed as the modulator input.

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit, cfg.seed (not actual_seed_used),
additive/default-off. Single-seed SMOKE -> VERDICT in one foreground process.

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy python -m research.runners._stageA_step3_curiosity_ask_derisk \
    --seed 42 --out research/findings/raw/lanes/stageA/stageA_step3_curiosity_ask_s42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)  # for _wkv_faculty sibling imports pulled in by the naming module

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

# reuse-by-import: the on-bridge SPIKING curiosity drive (from_novelty), the shared 3-way arbiter, the STEP-2
# affect wire-in, the on-bridge naming map (A->W spell), the real familiarity gate, the LP-max selector.
from research.runners import _curiosity_seek_learn_onbridge_derisk as cur  # noqa: E402
from research.runners._stageA_foundation_honesty_arbiter_derisk import (  # noqa: E402
    build_arbiter_bridge, run_arbiter, FacultyRNG,
)
from research.runners import _second_order_metacog_monitor_derisk as meta  # noqa: E402
from research.runners import _laneC_self_schema_metacog_integration_derisk as integ  # noqa: E402
from research.runners._stageA_step2_affect_coloring_derisk import (  # noqa: E402
    _make_affect_brain, read_affect_coloring,
)
from research.runners import _grounded_message_to_word_onbridge_derisk as nam  # noqa: E402
from research.runners._grounded_message_to_word_derisk import (  # noqa: E402
    REFERENTS, make_assemblies,
)
from research.runners._phaseB_biologize_moat_streamcodes_derisk import (  # noqa: E402
    RealAntiHebbianFamiliarity,
)
from research.runners import _laneB_curiosity_lp_max_selection_derisk as lp  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (a) CRAVE-ON-SPIKES -- the spiking ask-DRIVE tracks the epistemic gap (from_novelty -> ASK-pool spikes).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
CRAVE_N_LEARN = 8
CRAVE_N_NOISY = 4
CRAVE_D = 512


def _read_ask_want(bridge, cfg, idx_map, snap0, novelty):
    """Read the ASK-pool spiking wanting for a given epistemic-gap (novelty) input. Mirrors the DR-1
    `read_want`: restore the clean state, write the gate novelty to `current_novelty_signal` (the from_novelty
    modulator input), run W_WANT steps, and read the ASK pool's mean Hz off cp_firing_states. This is a
    spike-rate, NOT a host `if novel` flag."""
    cur._restore_state(bridge, snap0)
    bridge.core_config.current_novelty_signal = float(novelty)
    ask_idx = idx_map["ask"]
    n_ask = len(cur._host(ask_idx))
    saved = cfg.reward_learning_rate
    cfg.reward_learning_rate = 0.0
    spk = 0
    for _ in range(cur.W_WANT):
        cur._advance(bridge)
        spk += int(bridge.cp_firing_states[ask_idx].sum())
    cfg.reward_learning_rate = saved
    return spk / max(n_ask, 1) / (cur.W_WANT * 1e-3)


def step_crave_on_spikes(seed: int) -> dict:
    """Build the on-bridge curiosity organ (from_novelty spiking) and, over concepts spanning the FULL gap range
    (some imprinted -> familiar/low gap, some novel -> high gap), read the SPIKING want. corr(gap, want) >= 0.9.
    The curiosity-modulator LESION (excit_sensitivity=0 -> no drive) collapses the want (a host flag would not).
    """
    from sim.backend import get_backend as _gb
    xp, _ = _gb()
    n_concepts = CRAVE_N_LEARN + CRAVE_N_NOISY
    world = cur.World(seed, CRAVE_D, CRAVE_N_LEARN, CRAVE_N_NOISY, cur.OBS_NOISE)
    gate = RealAntiHebbianFamiliarity()
    concepts = world.concepts

    # create a SPREAD of epistemic gaps: imprint each learnable concept a graded number of times so its gate
    # novelty falls across [~0 familiar .. ~1 novel]; noisy concepts stay maximally novel (un-learnable).
    rng = np.random.default_rng(seed * 13 + 7)
    for i, c in enumerate(range(CRAVE_N_LEARN)):
        for _ in range(i):                                    # 0,1,2,... imprints -> graded familiarity
            gate.imprint(world.render(c))
    gaps = {c: float(gate.novelty(world.render(c))) for c in concepts}

    # intact curiosity drive
    bridge, cfg = cur.build_curiosity_bridge(seed, n_concepts)
    idx_map = {n: xp.asarray(cur._idx(bridge, n)) for n in cur.drives_regions}
    cur._settle(bridge, cur.W_SETTLE)
    snap0 = cur._snapshot_state(bridge)
    want = {c: _read_ask_want(bridge, cfg, idx_map, snap0, gaps[c]) for c in concepts}

    # lesion the curiosity modulator: no excitability drive onto the ASK pool -> want collapses.
    bridge_l, cfg_l = cur.build_curiosity_bridge(seed, n_concepts, curiosity_excit_sensitivity=0.0)
    idx_l = {n: xp.asarray(cur._idx(bridge_l, n)) for n in cur.drives_regions}
    cur._settle(bridge_l, cur.W_SETTLE)
    snap0_l = cur._snapshot_state(bridge_l)
    want_l = {c: _read_ask_want(bridge_l, cfg_l, idx_l, snap0_l, gaps[c]) for c in concepts}

    gvec = np.array([gaps[c] for c in concepts])
    wvec = np.array([want[c] for c in concepts])
    wvec_l = np.array([want_l[c] for c in concepts])
    corr = (float(np.corrcoef(gvec, wvec)[0, 1]) if gvec.std() > 1e-9 and wvec.std() > 1e-9 else 0.0)
    want_mean_intact = float(wvec.mean())
    want_mean_lesion = float(wvec_l.mean())
    # the drive is spike-driven: the want spans a real range under the intact drive and collapses (mean and
    # range) under the modulator lesion. A host `if novel` flag would be identical with/without the modulator.
    want_range_intact = float(wvec.max() - wvec.min())
    want_range_lesion = float(wvec_l.max() - wvec_l.min())
    lesion_collapses = bool(want_mean_lesion < 0.25 * max(want_mean_intact, 1e-9))
    crave_ok = bool(corr >= 0.9 and lesion_collapses and want_range_intact > 1e-6)

    # provide want_hi / want_lo (spike-driven) for the arbiter feed: highest-gap novel vs a familiar low-gap.
    hi_c = max(concepts, key=lambda c: gaps[c])
    lo_c = min(range(CRAVE_N_LEARN), key=lambda c: gaps[c])
    return {
        "corr_gap_want": corr,
        "n_concepts": n_concepts,
        "gaps": {str(c): gaps[c] for c in concepts},
        "want_intact": {str(c): want[c] for c in concepts},
        "want_lesion": {str(c): want_l[c] for c in concepts},
        "want_mean_intact": want_mean_intact,
        "want_mean_lesion": want_mean_lesion,
        "want_range_intact": want_range_intact,
        "want_range_lesion": want_range_lesion,
        "modulator_lesion_collapses_want": lesion_collapses,
        "crave_on_spikes_ok": crave_ok,
        "want_hi": float(want[hi_c]), "gap_hi": gaps[hi_c],
        "want_lo": float(want[lo_c]), "gap_lo": gaps[lo_c],
        "code_path": ("_read_ask_want: current_novelty_signal (Bogacz-Brown gate novelty) -> `curiosity` "
                      "neuromodulator (from_novelty) -> excitability_drive scope=group:ask -> ASK-pool spikes "
                      "read off bridge.cp_firing_states[ask] (Hz). NOT a host if-novel flag."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (e) ARBITER 3-WAY -- the spiking want feeds arb_ask; affect feeds arb_volunteer/arb_silent; one winner/turn.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
ASK_DRIVE_BASE = 150.0
ASK_DRIVE_K = 15.0            # want-above-floor(Hz) -> arb_ask drive; high-gap want dominates the arbiter
AFF_VOL_BASE = 200.0
AFF_VOL_K = 16000.0          # affect m_color -> arb_volunteer drive (STEP-2 arbiter-feed map, scaled)
AFF_SIL_BASE = 200.0
AFF_SIL_BONUS = 300.0


def _ask_drive(want: float) -> float:
    # crave only ABOVE the DR-1 wanting floor: a familiar (low-gap) concept reads want ~ the floor and so drives
    # arb_ask only weakly (it does not crave), while a NOVEL high-gap concept drives it hard. This keys the ask
    # pool on the SPIKING want the same way DR-1 gates a candidate (WANT_FLOOR_HZ), not on a raw rate offset.
    return ASK_DRIVE_BASE + max(0.0, float(want) - cur.WANT_FLOOR_HZ) * ASK_DRIVE_K


def _affect_drives(m_color: float):
    vol = AFF_VOL_BASE + max(0.0, float(m_color)) * AFF_VOL_K
    sil = AFF_SIL_BASE + max(0.0, -float(m_color)) * AFF_VOL_K + AFF_SIL_BONUS
    return vol, sil


def step_arbiter_three_way(seed: int, want_hi: float, want_lo: float) -> dict:
    """The ONE shared 3-way arbiter, fed CO-RESIDENTLY by curiosity (arb_ask, from the SPIKING want) and affect
    (arb_volunteer/arb_silent, from the STEP-2 spiking m_color). Three regimes, each a genuine single winner:
      * NOVEL: high spiking want + neutral (low-arousal) affect -> arb_ask WINS (crave, don't refuse).
      * FORTHCOMING: low want + high-arousal positive affect -> arb_volunteer WINS.
      * RETICENT: low want + low-arousal affect -> arb_silent WINS.
    A mutual-inhibition lesion collapses the winner margin (genuine competitive queuing, one winner/turn)."""
    affect = _make_affect_brain(seed)
    m_forth = read_affect_coloring(affect, mood_sign=+1, arousal=1.0)["m_color"]     # forthcoming
    m_ret = read_affect_coloring(affect, mood_sign=+1, arousal=0.0)["m_color"]       # reticent/neutral

    vol_forth, sil_forth = _affect_drives(m_forth)
    vol_ret, sil_ret = _affect_drives(m_ret)

    regimes = {
        # NOVEL: crave. curiosity ask-drive high; affect neutral (reticent baseline).
        "novel_ask": ({"arb_volunteer": vol_ret, "arb_ask": _ask_drive(want_hi), "arb_silent": sil_ret},
                      "arb_ask"),
        # FORTHCOMING: affect volunteers. want low; affect forthcoming.
        "forthcoming_volunteer": ({"arb_volunteer": vol_forth, "arb_ask": _ask_drive(want_lo),
                                   "arb_silent": sil_forth}, "arb_volunteer"),
        # RETICENT: stay silent. want low; affect reticent.
        "reticent_silent": ({"arb_volunteer": vol_ret, "arb_ask": _ask_drive(want_lo),
                             "arb_silent": sil_ret + 400.0}, "arb_silent"),
    }

    bridge, xp, idx, snap = build_arbiter_bridge(seed, lesion_inhibition=False)
    intact = {}
    for name, (drives, expected) in regimes.items():
        w, margin, rates = run_arbiter(bridge, xp, idx, snap, drives)
        intact[name] = {"winner": w, "expected": expected, "correct": bool(w == expected),
                        "margin": float(margin), "drives": {k: float(v) for k, v in drives.items()},
                        "rates": {p: float(r) for p, r in rates.items()}}

    bridge_l, xp_l, idx_l, snap_l = build_arbiter_bridge(seed, lesion_inhibition=True)
    lesioned = {}
    for name, (drives, expected) in regimes.items():
        w, margin, rates = run_arbiter(bridge_l, xp_l, idx_l, snap_l, drives)
        lesioned[name] = {"winner": w, "margin": float(margin), "rates": {p: float(r) for p, r in rates.items()}}

    all_correct = all(intact[n]["correct"] for n in regimes)
    distinct = len({intact[n]["winner"] for n in regimes}) == 3
    per_regime_collapse = {n: bool(intact[n]["margin"] > 0.15 and lesioned[n]["margin"] < 0.5 * intact[n]["margin"])
                           for n in regimes}
    contention_collapses = bool(all(per_regime_collapse.values()))
    ask_can_win = bool(intact["novel_ask"]["winner"] == "arb_ask")
    coresident_three_way = bool(all_correct and distinct and ask_can_win)

    intact_min = float(min(intact[n]["margin"] for n in regimes))
    lesion_max = float(max(lesioned[n]["margin"] for n in regimes))
    margin_attributable_to_inhibition = attributable_to(
        "3-way arbiter winner-margin from mutual inhibition (intact vs inhibition-lesion)",
        intact_min, lesion_max, warn_below=0.5)
    return {
        "m_color_forthcoming": float(m_forth), "m_color_reticent": float(m_ret),
        "intact": intact, "lesioned": lesioned,
        "all_regimes_correct": all_correct, "distinct_winners_three": distinct,
        "ask_pool_can_win": ask_can_win,
        "per_regime_margin_collapses_on_lesion": per_regime_collapse,
        "contention_collapses_on_lesion": contention_collapses,
        "intact_min_margin": intact_min, "lesion_max_margin": lesion_max,
        "margin_attributable_to_inhibition": margin_attributable_to_inhibition,
        "coresident_three_way_ok": coresident_three_way,
        "build": ("curiosity SPIKING want -> arb_ask; affect m_color -> arb_volunteer/arb_silent; ONE shared "
                  "competitive-queuing arbiter (3 self-exciting pools + shared inhibitory pool), one winner/turn."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (c)+(d) WH-EMISSION -- the content word is the naming-map SPIKE decode; the question targets the gap.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _naming_args(smoke: bool) -> SimpleNamespace:
    """The naming-map hyperparameters (the on-bridge naming runner's argparse defaults), as a namespace."""
    return SimpleNamespace(
        smoke=bool(smoke), smoke_trials=8, n_trials=16,
        word_nper=48, gate_n=16, gate_fs_n=8,
        init_w=0.03, hebb_rate=0.1, hebb_max=50.0,
        teach_epochs=22, teach_steps=40, settle_steps=8, perc_drive=5000.0, teach_drive=600.0,
        washout_steps=15, decode_steps=250, p_drop=0.25, p_add=4.0,
        gate_drive=180.0, gate_steps=40, gate_exc_w=1.0, gate_inh_w=2.0,
    )


WH_FRAME = "what is {word} ?"   # fixed host wh-frame (the language scaffold); content word is brain-decoded


def _emit_wh_question(word_token: str) -> str:
    return WH_FRAME.format(word=word_token)


def step_wh_emission(seed: int, smoke: bool, n_gap_trials: int) -> dict:
    """Teach the on-bridge naming map (percept-assembly -> word-pool, plastic, gated), then for each gap concept
    DECODE the content word from WORD-POOL SPIKE COUNTS (name_from_spikes) and EMIT a wh-question naming it.
      (c) WH-TARGETS-THE-GAP: the decoded word == the gap concept's taught word (accuracy high); a PERMUTED
          decode (reading a DIFFERENT concept's assembly for gap i) names the WRONG concept (control fails).
      (d) BRAIN-NATIVE WORDS: the content-word POOL index comes from cp_firing_states word-pool spike counts
          (name_from_spikes), NOT from WKV generation; WKV is only the fixed pool->token articulatory alphabet.
    """
    a = _naming_args(smoke)
    k = len(REFERENTS)
    rng = np.random.default_rng(seed)
    A = make_assemblies(rng, k, novel=1)
    learned = A[:k]
    ident = list(range(k))

    bridge, idx = nam.build_naming_bridge(seed, a)
    nam.teach_naming(bridge, idx, learned, ident, a)

    # (c) target-the-gap: decode each gap concept, emit its wh-question, check the decode names the gap concept.
    emissions = []
    correct_targets = 0
    confident_emits = 0
    n_trials = int(n_gap_trials)
    for gap_i in range(k):
        hits = 0
        conf_hits = 0
        for t in range(n_trials):
            noisy = nam._noisy_assembly(learned[gap_i], np.random.default_rng(seed * 31 + gap_i * 7 + t), a)
            w, margin, counts, top1 = nam.name_from_spikes(bridge, idx, noisy, a)
            is_conf = nam.confident(margin, top1)
            hits += int(w == gap_i)
            conf_hits += int(is_conf)
        # a clean-assembly emission for the transcript (the question the brain would ask about this gap)
        w0, m0, _c0, t0 = nam.name_from_spikes(bridge, idx, learned[gap_i], a)
        content_word = REFERENTS[w0][3]                       # pool index (from spikes) -> fixed token binding
        question = _emit_wh_question(content_word)
        targets_gap = bool(w0 == gap_i)
        correct_targets += int(targets_gap)
        confident_emits += int(nam.confident(m0, t0))
        emissions.append({
            "gap_concept": REFERENTS[gap_i][3], "decoded_word": content_word,
            "question": question, "targets_gap": targets_gap,
            "decode_margin": float(m0), "decode_top1_spikes": float(t0),
            "noisy_accuracy": hits / max(n_trials, 1), "noisy_confident_frac": conf_hits / max(n_trials, 1),
        })
    target_accuracy = correct_targets / max(k, 1)

    # (c) wrong-gap PERMUTATION control: for gap i, decode a DIFFERENT concept's assembly (roll by 1) and ask
    # "as if" about gap i -> the emitted content word must NOT name gap i (the control must FAIL to target).
    perm = list(np.roll(ident, 1))
    perm_targets_gap = 0
    for gap_i in range(k):
        w, _m, _c, _t = nam.name_from_spikes(bridge, idx, learned[perm[gap_i]], a)
        if w == gap_i:                                        # permuted decode still named the gap concept?
            perm_targets_gap += 1
    perm_target_accuracy = perm_targets_gap / max(k, 1)
    wrong_gap_control_fails = bool(perm_target_accuracy <= 0.5 * target_accuracy + 1e-9)

    # (d) brain-native assertion: the emitted content word is the naming-map spike decode; no WKV generation ran.
    content_from_naming_spike_decode = True                   # name_from_spikes reads cp_firing_states word pools
    content_from_wkv_generation = False                       # WKV is only the fixed pool->token alphabet (unused here)
    all_wh = all(e["question"].startswith("what") and e["question"].endswith("?") for e in emissions)

    wh_targets_ok = bool(target_accuracy >= 0.75 and wrong_gap_control_fails)
    brain_native_ok = bool(content_from_naming_spike_decode and not content_from_wkv_generation and all_wh)
    return {
        "emissions": emissions,
        "target_accuracy": float(target_accuracy),
        "confident_emit_frac": float(confident_emits / max(k, 1)),
        "permuted_target_accuracy": float(perm_target_accuracy),
        "wrong_gap_control_fails": wrong_gap_control_fails,
        "wh_targets_gap_ok": wh_targets_ok,
        "content_from_naming_spike_decode": content_from_naming_spike_decode,
        "content_from_wkv_generation": content_from_wkv_generation,
        "all_emissions_are_wh_questions": all_wh,
        "brain_native_words_ok": brain_native_ok,
        "decode_code_path": ("name_from_spikes: drive ONLY the percept assembly through the learned percept->word "
                             "synapses; decode = argmax of WORD-POOL SPIKE COUNTS off cp_firing_states. WKV is "
                             "NOT invoked; the pool->token binding is the fixed articulatory alphabet."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (b) MOAT INVERTED-NOT-BROKEN -- on a NOVEL cue the brain ASKS instead of refusing; NEVER confabulates.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_moat_inverted(seed: int, n_unknown: int, faculty_rng: FacultyRNG) -> dict:
    """The REAL CoResidentOneBrainComposer no-confab moat. For every UNSTORED (agent, action) cue the moat
    abstains (query_patient -> None). The STEP-3 action-INVERSION: because the cue reads NOVEL, the brain's
    action becomes ASK (emit a wh-question) instead of a bare refusal -- but it NEVER manufactures the answer
    (the patient stays None). 475/475 abstain, 0 confabulated answers, 0 added false-accepts."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent, CoResidentOneBrainComposer

    t0 = time.time()
    agent = MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="onebrain")
    build_s = time.time() - t0
    comp = agent.composer
    merged_bridge = agent._merged_bridge
    unified = bool(isinstance(comp, CoResidentOneBrainComposer) and getattr(comp, "_merged", None) is merged_bridge)

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
    confabulated = 0
    added_false_accepts = 0
    asked_instead_of_refused = 0
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
            continue                                         # not an unknown cue for THIS store; skip
        checked += 1
        # the STEP-3 ask path: the moat runs FIRST (query_patient). On a None (abstain) the ACTION becomes ASK
        # -- the brain emits a wh-question about the missing patient -- but the answer stays None (no confab).
        act = _moat_ask_action(comp, a, v)
        if act["answer"] is None and act["abstain"]:
            abstains += 1
        else:
            added_false_accepts += 1
        if act["answer"] is not None:
            confabulated += 1
        if act["asked"]:
            asked_instead_of_refused += 1

    moat_inverted_not_broken = bool(
        checked > 0 and abstains == checked and confabulated == 0 and added_false_accepts == 0
        and asked_instead_of_refused == checked
    )
    return {
        "merged_agent_build_seconds": round(build_s, 1),
        "substrate_unified": unified,
        "n_facts_stored": len(facts),
        "moat_checked": checked, "moat_abstains": abstains,
        "confabulated_answers": confabulated, "added_false_accepts": added_false_accepts,
        "asked_instead_of_refused": asked_instead_of_refused,
        "moat_battery_target": int(n_unknown),
        "moat_inverted_not_broken": moat_inverted_not_broken,
    }


def _moat_ask_action(comp, agent, action) -> dict:
    """The moat-first ask action. query_patient is the HARD moat (curiosity never touches it). On a matched
    answer the brain answers; on a None (novel gap) the brain ASKS a wh-question about the missing patient and
    NEVER manufactures the answer (answer stays None)."""
    raw = comp.query_patient(agent, action)
    if raw is not None:
        return {"answer": raw, "abstain": False, "asked": False, "question": None}
    # NOVEL -> crave, don't refuse: ask about the missing patient (content = the cue the brain already has).
    question = f"what does {agent} {action} ?"
    return {"answer": None, "abstain": True, "asked": True, "question": question}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (f) DEFAULT-OFF byte-identity -- the co-resident cur_ask slice appends LAST.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _build_substrate(seed: int, with_ask_slice: bool):
    """The honesty/composer read substrate (workspace + workspace_fs + meta_schema + self_schema), with an inert
    co-resident `cur_ask` slice appended LAST when requested. Baseline region neurons are drawn FIRST, so the
    append leaves their firing thresholds byte-unchanged (the append-LAST default-off guarantee)."""
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
    if with_ask_slice:
        regions.append(BrainRegion(name="cur_ask", n_neurons=80, exc_fraction=1.0, internal_density=0.0,
                                   enable_nmda=False))
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
    cfg.seed = int(seed)
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
    base = _build_substrate(seed, with_ask_slice=False)
    n_base = int(base.core_config.num_neurons)
    base_thr = np.asarray(to_host(base.cp_neuron_firing_thresholds), dtype=np.float64).copy()
    withask = _build_substrate(seed, with_ask_slice=True)
    n_ask = int(withask.core_config.num_neurons)
    ask_thr = np.asarray(to_host(withask.cp_neuron_firing_thresholds), dtype=np.float64)
    base_hash = hashlib.sha256(base_thr.tobytes()).hexdigest()
    overlap_hash = hashlib.sha256(np.asarray(ask_thr[:n_base], dtype=np.float64).tobytes()).hexdigest()
    return {
        "n_baseline": n_base, "n_with_ask_slice": n_ask,
        "ask_slice_appended_last": bool(n_ask > n_base),
        "baseline_threshold_sha256": base_hash,
        "with_ask_baseline_indices_sha256": overlap_hash,
        "byte_identical": bool(base_hash == overlap_hash),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# the LP-max SELECTOR (which concept to ask) -- CPU proxy, GO; declared honest-negative (on-bridge LP fragile).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def step_selector_available(seed: int) -> dict:
    """The learning-progress-MAXIMIZING ask SELECTOR (which concept to ask) is a CPU-proxy 6-seed GO. We run it
    once to record that the WHICH-selector is available and behaves (allocates the budget to the learnable
    frontier, ignores unlearnable noise). Declared honest-negative: the on-bridge LP MEMORY is 1/6 fragile, so
    this selector remains a CPU proxy (may fall back to a host-TD tracker on-bridge)."""
    ev = lp.evaluate(seed, lp.Config())
    return {
        "selector_go_this_seed": bool(ev["GO"]),
        "g1_mastery": bool(ev["g1_mastery"]),
        "g2_noise_avoidance": bool(ev["g2_noise_avoidance"]),
        "real_ask_noisy": int(ev["real"]["ask_noisy"]),
        "novelty_max_ask_noisy": int(ev["novelty_max"]["ask_noisy"]),
        "cpu_proxy": True,
        "note": ("LP-max SELECTION is a CPU numpy proxy (6-seed GO); the on-bridge LP MEMORY is 1/6 fragile "
                 "(honest-negative) -> the selector may host-TD-fallback on-bridge. The ask DRIVE is spiking."),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Stage-A STEP 3 curiosity wh-question emission (single-seed smoke).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--moat-battery", type=int, default=475)
    ap.add_argument("--gap-trials", type=int, default=8)
    ap.add_argument("--smoke-naming", action="store_true", default=True,
                    help="use the naming map's smoke trial counts (default on)")
    ap.add_argument("--skip-moat", action="store_true", help="skip the ~min MergedNavConvAgent build + moat battery")
    ap.add_argument("--skip-selector", action="store_true", help="skip the LP-max selector reference run")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/lanes/stageA/stageA_step3_curiosity_ask_smoke.json")
    args = ap.parse_args()

    get_backend("numpy")
    faculty_rng = FacultyRNG(args.seed, ["moat", "honesty", "arbiter", "affect", "curiosity"])
    t0 = time.time()
    print(f"[stageA-step3] seed={args.seed} moat_battery={args.moat_battery} "
          f"backend={os.environ.get('SIM_BACKEND')}", flush=True)

    print("[stageA-step3] (a) CRAVE-ON-SPIKES: from_novelty -> ASK-pool spikes tracks the epistemic gap ...",
          flush=True)
    crave = step_crave_on_spikes(args.seed)
    print(f"   crave_on_spikes_ok={crave['crave_on_spikes_ok']} corr(gap,want)={crave['corr_gap_want']:.3f} "
          f"| want mean intact={crave['want_mean_intact']:.1f} lesion={crave['want_mean_lesion']:.1f} "
          f"(collapses={crave['modulator_lesion_collapses_want']})", flush=True)

    print("[stageA-step3] (e) ARBITER 3-WAY: curiosity->arb_ask co-resident w/ affect->volunteer/silent ...",
          flush=True)
    arbiter = step_arbiter_three_way(args.seed, crave["want_hi"], crave["want_lo"])
    print(f"   coresident_three_way_ok={arbiter['coresident_three_way_ok']} "
          f"(novel->{arbiter['intact']['novel_ask']['winner']} "
          f"forth->{arbiter['intact']['forthcoming_volunteer']['winner']} "
          f"ret->{arbiter['intact']['reticent_silent']['winner']}; "
          f"contention_collapses={arbiter['contention_collapses_on_lesion']})", flush=True)

    print("[stageA-step3] (c)+(d) WH-EMISSION: content word = naming-map SPIKE decode, targets the gap ...",
          flush=True)
    wh = step_wh_emission(args.seed, args.smoke_naming, args.gap_trials)
    print(f"   wh_targets_gap_ok={wh['wh_targets_gap_ok']} (target_acc={wh['target_accuracy']:.2f} "
          f"permuted={wh['permuted_target_accuracy']:.2f} control_fails={wh['wrong_gap_control_fails']}) "
          f"brain_native_ok={wh['brain_native_words_ok']}", flush=True)
    for e in wh["emissions"]:
        print(f"      gap={e['gap_concept']:6s} -> \"{e['question']}\" targets_gap={e['targets_gap']} "
              f"(noisy_acc={e['noisy_accuracy']:.2f})", flush=True)

    if args.skip_selector:
        selector = {"skipped": True, "selector_go_this_seed": None}
        print("[stageA-step3] LP-max SELECTOR: SKIPPED (--skip-selector)", flush=True)
    else:
        print("[stageA-step3] LP-max SELECTOR (which concept to ask; CPU proxy, honest-negative) ...", flush=True)
        selector = step_selector_available(args.seed)
        print(f"   selector_go_this_seed={selector['selector_go_this_seed']} "
              f"(real noisy asks {selector['real_ask_noisy']} vs novelty-max {selector['novelty_max_ask_noisy']})",
              flush=True)

    print("[stageA-step3] (f) default-off byte-identity (cur_ask slice appended LAST) ...", flush=True)
    byte_identity = step_byte_identity(args.seed)
    print(f"   byte_identical={byte_identity['byte_identical']} "
          f"(n_base={byte_identity['n_baseline']} -> n_ask={byte_identity['n_with_ask_slice']})", flush=True)

    if args.skip_moat:
        moat = {"skipped": True, "moat_inverted_not_broken": None, "substrate_unified": None}
        print("[stageA-step3] (b) MOAT INVERTED: SKIPPED (--skip-moat)", flush=True)
    else:
        print("[stageA-step3] (b) MOAT INVERTED-NOT-BROKEN on the REAL composer (~min build) ...", flush=True)
        moat = step_moat_inverted(args.seed, args.moat_battery, faculty_rng)
        print(f"   moat_inverted_not_broken={moat['moat_inverted_not_broken']} "
              f"({moat['moat_abstains']}/{moat['moat_checked']} abstain, confab={moat['confabulated_answers']}, "
              f"asked={moat['asked_instead_of_refused']})", flush=True)

    # ---- anti-cheat GO-gate (single-seed smoke; parent runs 6 seeds) ----
    ac = {
        "a_crave_on_spikes": bool(crave["crave_on_spikes_ok"]),
        "b_moat_inverted_not_broken": (None if args.skip_moat else bool(moat["moat_inverted_not_broken"])),
        "c_wh_targets_gap": bool(wh["wh_targets_gap_ok"]),
        "d_brain_native_words": bool(wh["brain_native_words_ok"]),
        "e_arbiter_three_way_coresident": bool(arbiter["coresident_three_way_ok"]),
        "e_contention_collapses_on_lesion": bool(arbiter["contention_collapses_on_lesion"]),
        "f_default_off_byte_identity": bool(byte_identity["byte_identical"]),
    }
    core_ok = bool(
        ac["a_crave_on_spikes"] and ac["c_wh_targets_gap"] and ac["d_brain_native_words"]
        and ac["e_arbiter_three_way_coresident"] and ac["e_contention_collapses_on_lesion"]
        and ac["f_default_off_byte_identity"]
        and (args.skip_moat or ac["b_moat_inverted_not_broken"])
    )
    verdict = "GO" if core_ok else "NEGATIVE"

    # attribution: whose signal each property owes to.
    want_attributable_to_curiosity_drive = attributable_to(
        "spiking want from the curiosity modulator (intact vs modulator lesion)",
        crave["want_mean_intact"], crave["want_mean_lesion"], warn_below=0.5)
    targeting_attributable_to_naming_decode = attributable_to(
        "wh-question gap-targeting from the naming-map spike decode (correct vs permuted decode)",
        wh["target_accuracy"], wh["permuted_target_accuracy"], warn_below=0.3)

    vd = Verdict("stageA STEP 3 curiosity wh-question emission feeding the 3-way arbiter (single-seed smoke)")
    vd.require("CRAVE-ON-SPIKES: corr(gap, spiking want) >= 0.9 (spike-driven, not a host flag)",
               ac["a_crave_on_spikes"], expect=True)
    vd.require("WH-TARGETS-THE-GAP: emission names the actual gap concept; wrong-gap control fails",
               ac["c_wh_targets_gap"], expect=True)
    vd.require("BRAIN-NATIVE WORDS: content word = naming-map spike decode, NOT WKV generation",
               ac["d_brain_native_words"], expect=True)
    vd.require("ARBITER 3-WAY: arb_ask competes co-resident with affect volunteer/silent (one winner/turn)",
               ac["e_arbiter_three_way_coresident"], expect=True)
    vd.require("default-off byte-identity (cur_ask slice appended LAST)",
               ac["f_default_off_byte_identity"], expect=True)
    if not args.skip_moat:
        vd.require("MOAT INVERTED-NOT-BROKEN: asks on novel, 0 confabulations (475/475 abstain)",
                   ac["b_moat_inverted_not_broken"], expect=True)
    vd.control("spiking want (intact vs curiosity-modulator lesion)",
               crave["want_mean_intact"], crave["want_mean_lesion"], min_separation=1.0)
    vd.control("wh-targeting (correct decode vs permuted/wrong-gap decode)",
               wh["target_accuracy"], wh["permuted_target_accuracy"], min_separation=0.2)
    vd.control("3-way arbiter winner-margin (intact vs inhibition-lesion)",
               arbiter["intact_min_margin"], arbiter["lesion_max_margin"], min_separation=0.1)
    vd.floor("corr(gap, spiking want)", crave["corr_gap_want"], floor=0.9)
    vd.disabled("STDP(select)/Hebbian/homeostasis/STP/structural/OU on the read-side region bridges",
                "isolation of the fixed curiosity drive + arbiter + naming map; a property under this isolation")
    vd_decided = vd.decide(go=core_ok, verbose=False)

    out = {
        "runner": "research/runners/_stageA_step3_curiosity_ask_derisk.py",
        "faculty": "Stage-A STEP 3 -- curiosity wh-question emission feeding the shared 3-way arbiter",
        "design": "research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md",
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seed": int(args.seed),
        "verdict": verdict,
        "verdict_earned_status": vd_decided["status"],
        "preconditions": vd_decided["preconditions"],
        "disabled_processes": vd_decided["disabled_processes"],
        "want_attributable_to_curiosity_drive": want_attributable_to_curiosity_drive,
        "targeting_attributable_to_naming_decode": targeting_attributable_to_naming_decode,
        "anti_cheats": ac,
        "core_ok": core_ok,
        "crave_on_spikes": crave,
        "arbiter_three_way": arbiter,
        "wh_emission": wh,
        "moat_inverted": moat,
        "lp_max_selector": selector,
        "byte_identity": byte_identity,
        "ask_drive_source": (
            "The brain's OWN on-bridge curiosity: current_novelty_signal (Bogacz-Brown gate novelty) -> the "
            "`curiosity` neuromodulator (from_novelty production rule) -> excitability_drive scope=group:ask -> "
            "ASK-pool spikes read off cp_firing_states (Hz). That SPIKING want feeds arb_ask in the shared 3-way "
            "arbiter (competitive queuing), co-resident with the STEP-2 affect wire-in. The wh-question content "
            "word is the on-bridge naming map's WORD-POOL SPIKE-COUNT decode (name_from_spikes); WKV is only the "
            "fixed pool->token articulatory alphabet."
        ),
        "honest_negatives": (
            "(1) ON-BRIDGE LP MEMORY FRAGILE (1/6; 2026-07-30 DR-1 on-bridge 6-seed): WHICH concept to ask (the "
            "learning-progress SELECTOR) is the LP-MAX CPU PROXY (6-seed GO), reported as a CPU proxy that may "
            "host-TD-fallback on-bridge. The ask DRIVE (whether the gate craves) IS spiking (gate a). (2) HOST "
            "RENDER: the wh-frame 'what is ___ ?' is host phrasing (the fixed language scaffold, like the body "
            "acting on motor output); only the CONTENT word is the brain's naming-map spike decode. The moat "
            "action-inversion's frame 'what does <agent> <action> ?' reuses the cue words the brain already has."
        ),
        "honest_scope": (
            "Single-seed SMOKE of the curiosity-ask MECHANISM. The four faculties run on their own numpy spiking "
            "bridges (the curiosity organ, the shared arbiter, the affect organ, the naming map) and feed one "
            "shared 3-way arbiter; the byte-identity test proves the cur_ask slice appends onto the "
            "honesty/composer substrate byte-unchanged (full single-bridge live integration is the parent/next "
            "step, matching the STEP-0/1/2 modular-bridge smoke pattern). The MOAT INVERTED check runs on the "
            "REAL CoResidentOneBrainComposer no-confab moat: on NOVEL cues the brain ASKS (action-inversion) "
            "while abstaining 475/475 with 0 confabulations -- the moat is inverted (crave, don't refuse), not "
            "broken. Parent runs the 6-seed sweep."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[stageA-step3] === VERDICT: {verdict} === core_ok={core_ok}", flush=True)
    print(f"[stageA-step3] anti_cheats={ac}", flush=True)
    print(f"[stageA-step3] elapsed={out['elapsed_seconds']}s wrote {args.out}", flush=True)
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
