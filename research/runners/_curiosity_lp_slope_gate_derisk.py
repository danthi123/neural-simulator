"""CURIOSITY LEARNING-PROGRESS -- per-topic LP SLOPE GATE (the brake becomes a learning-progress slope), Stage-2 C3, method 3.

Approved by RULING c3-methods-1018b (method 3 approved as a parallel dev-seed de-risk: new module, never imported by
webapp/ or the production organ, no sim/ edit) and RULING c3-methods-1018c (own branch with tests; ASK Claude for
review before its dev results are read as a verdict; dev seeds only; a gate-seed prereg comes after a dev PASS).
Design proposal: research/coordination/c3-method3-proposal-1018.md (a MiMo Pro delegate over the record, persisted by
the driver); packet bars: docs/drafts/curiosity-lp/PLAN.md section 6 (B1-B9), section 7 commands. Priority: RULING
priorities-1016 (3) / priorities-1018 (4). Written BEFORE any run (pre-registered constants below).

WHAT THIS MODULE IS, RELATIVE TO METHODS 1 AND 2
  Method 1 (research/runners/_curiosity_lp_trace_derisk.py): the brake on ASKING is the per-topic exposure-count
  depression trace (Tsodyks-Markram cp_stp_x on the topic slice's -> ASK synapses, tau_d = TAU_HAB_MS = 12 s).
  Dev verdict NO-GO on four bars (B1 1/8, B1b 0/8, B6 0/8, B7 1/8): recorded cause = the background OU process
  fires the slice during the silent inter-turn gaps, so the synapses are already depressed before the first probe
  (PLAN.md's pre-battery note and post-verdict: "the brake runs on a clock, not on unanswered asks").
  Method 2 (--no-ou flag on method 1, RULING c3-methods-1018): background noise off, depression brake kept; dev read
  all bars 8/8 (PROVISIONAL, research/coordination/c3-method2-devread-1018.md).
  Method 3 (THIS module): the brake BECOMES the per-topic learning-progress (LP) slope gate. The depression trace is
  made inert (stp_tau_d = M3_TAU_MS = 200 ms, fully recovered within the 400 ms inter-turn gap, so x ~ 1 every turn)
  and the background OU process is off in this module's build (method 2's settled diagnosis; --ou-process restores it
  for a controlled comparison, default off), so the ONLY thing that stops an ask is the per-topic omission veto. That
  is the ruling's B5 discriminator: with the gate removed (arm lesion_slope_gate) the veto never accumulates and the
  brain keeps asking on the 4th unanswered toad probe; with the gate intact it stays silent there.

THE MECHANISM (one bridge, identical wiring to methods 1/2; the ask decision per topic = want_hz >= threshold + veto)
  * Presentation (abstain turn): the topic's slice is driven at GRADED novelty * WORD_DRIVE (unchanged host boundary),
    the ASK pool spikes, want_hz = the spiking read. The LP state for that topic updates from the want read:
    slope = fast EMA - slow EMA of the topic's LP samples. A sample whose |slope| <= SLOPE_EPS_HZ is "slope stays ~0"
    (probed, never answered): the omission veto grows by VETO_STEP_HZ (capped at VETO_MAX_HZ), which raises that
    topic's slice -> ASK threshold. slope > SLOPE_EPS_HZ (progress) clears the veto (veto-protected: slow-but-imving
    keeps asking); slope < -SLOPE_EPS_HZ doubles the step. The FIRST probe of any topic sees veto = 0 (both EMAs
    initialize on its first sample), so every novel topic is asked on its first probe (B1/B1b).
  * Teach turn (the teacher's store write on a sliced topic): method 1's answer() read is reused unchanged -- the
    DR-1 read, PLAN.md:46-47: reward_us -> snc spiking DA, phasic-above-tonic in Hz (snc_hz - snc_neutral_hz, the
    same clip as the DA read) plus the topic's last want read -- and that jump initializes/raises slope, clearing the
    veto. An answered topic reads familiar afterwards (word-level novelty, the organ's gate) and its probes recall,
    so it is never presented again (B3).
  * DECLARED HOST SCAFFOLD (as required by RULING c3-methods-1018b): the fast/slow LP state, the SLOPE_EPS test and
    the omission-veto accumulator are host-side state held by this module (numpy scalars per topic, declared in the
    module header, recorded per turn in each artifact's responses and gate_state). The engine has no learning-progress
    target on any synapse; the realization rung would be an additive, default-off sim/ target (the same class as
    method 1's DA-gated STP restore, PLAN.md:79-81 -- which stays declared and inert here: stp_tau_d = 200 ms leaves
    x ~ 1, so the restore is a no-op, arm lesion_reset stays informational). Also declared, unchanged from the organ:
    the topic word (extract_topic), the graded novelty scalar (TopicNoveltyGate), the word -> slice table (first-sight
    allocation), the question frame, and the battery's store/recall world scaffold (not under test).
  * Arms (PLAN section 5, B5 retargeted per RULING c3-methods-1018b): intact, intact_b (the same-seed null),
    lesion_slope_gate (THE slope/veto gate removed -- the counted B5 lesion), lesion_drive (B9), perm_order (B6),
    perm_exposure (B7); informational: shared_slice, lesion_reset. The scorer is method 1's own B1-B9 code with only
    B5 retargeted (score_seed maps lesion_slope_gate onto method 1's lesion slot; aggregate retargets the
    attribution label). Bars frozen in PLAN.md section 6; W_TOPIC_ASK = 80 frozen (design_s9); dev seeds 1-8 only.

BIOLOGY (cited in PLAN.md section 3): learning progress as the intrinsic reward that should steer asking
(Oudeyer, Kaplan & Hafner 2007; Gottlieb, Oudeyer, Lopes & Baranes 2013; Kidd & Hayden 2015); infants withdrawing
attention from sequences they cannot learn (Kidd, Piantadosi & Aslin 2012); midbrain DA signalling information arrival
(Bromberg-Martin & Hikosaka 2009). Why not method 2: an exposure-count depression is blind to progress -- a topic
being learned is silenced exactly like one yielding nothing (the identifiability limit named in the
2026-08-02-laneB-curiosity-learning-progress-slope CPU-proxy finding, whose fast-minus-slow slope is reused here).

PRE-REGISTERED CONSTANTS (written before any run; do not change after data):
  ALPHA_FAST = 0.5, ALPHA_SLOW = 0.125 (EMA rates of the per-topic LP samples),
  SLOPE_EPS_HZ = 5.0 (|slope| below = "stays ~0"; above = progress),
  VETO_STEP_HZ = 10.0 per zero-slope probe (threshold + veto must cross a ~35 Hz fresh want by the 4th probe:
    19 + 3*10 = 49 > 38.9 worst-case pooled fresh want), VETO_STEP_DECLINE_HZ = 20.0 on negative slope,
  VETO_MAX_HZ = 120.0, M3_TAU_MS = 200.0 (the engine's default E->E STP recovery; depression inert by design),
  ou_process default False (method 2's diagnosis; --ou-process restores the noise for a controlled comparison).
  Dev seeds 1-8 only; dev seeds need no prereg; no gate-seed claim of any kind (RULING c3-methods-1018c).

STATUS: PROVISIONAL de-risk. Per RULING c3-methods-1018c raw dev reads may land first, but the dev results are read
as a verdict only after an ASK to Claude; gate-seed prereg comes after a dev PASS. FUNCTIONAL CORRELATE, NOT
phenomenal: any self-report of this gate is a read-out ("I asked about toads three times and got nothing, so my
drive to ask about toads has been vetoed"), never felt boredom.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import _curiosity_lp_trace_derisk as M1  # noqa: E402  (the same B1-B9 battery code, REUSED)
from research.runners._curiosity_lp_trace_derisk import (  # noqa: E402  (the same B1-B9 battery code, REUSED)
    ABSTAIN_TEXT,
    BATTERY_SEEDS,
    FIRST_GAP_PROBES,
    INTER_TURN_MS,
    N_ASK,
    N_REWARD_US,
    N_SLOTS,
    N_SNC,
    REWARD_US_TO_SNC_W,
    TOAD,
    TOPIC_ASK_DENSITY,
    TOPIC_ASK_JITTER,
    W_TOPIC_ASK,
    BatteryWorld,
    extract_topic,
    followup_question,
    parse_probe,
    parse_teach,
)
from research.runners.spiking_novelty_habituation_organ import (  # noqa: E402  (the de-risked STP regime)
    STP_TAU_F,
    STP_U,
    INPOP,
)
from research.runners._curiosity_seek_learn_onbridge_derisk import (  # noqa: E402  (bridge helpers)
    _idx,
    _settle,
    _snapshot_state,
)

RUNNER = "research/runners/_curiosity_lp_slope_gate_derisk.py"
PLAN = "docs/drafts/curiosity-lp/PLAN.md"
RULINGS = ("c3-methods-1018", "c3-methods-1018b", "c3-methods-1018c")

# ── pre-registered constants (module header; written BEFORE any run) ────────────────────────────────────────────────
ALPHA_FAST = 0.5            # EMA rate of the per-topic LP samples (fast)
ALPHA_SLOW = 0.125          # EMA rate (slow); slope = fast - slow (the 2026-08-02 CPU-proxy finding's estimator)
SLOPE_EPS_HZ = 5.0          # |slope| <= this = "slope stays ~0" (probed, never answered)
VETO_STEP_HZ = 10.0         # omission-veto growth per zero-slope probe (raises that topic's slice->ASK threshold)
VETO_STEP_DECLINE_HZ = 20.0  # a declining slope (slope < -SLOPE_EPS_HZ) vetoes at double step
VETO_MAX_HZ = 120.0         # cap (threshold stays finite)
M3_TAU_MS = 200.0           # stp_tau_d: the engine's default E->E recovery; the depression trace is INERT by design
                            # (x ~ 1 every turn over the 400 ms gap) so the slope gate alone is the brake
OUT_PARENT = "research/findings/raw/_curiosity_lp_slope/m3_dev"

BARS = dict(M1.BARS)
BARS["B5_lesion_continues"] = (
    "lesion_slope_gate: asked on u_toad_4, 8/8 seeds, and intact != lesion on u_toad_4 on every seed "
    "(attributable fraction vs the intact_b null >= 0.5) -- B5's counted lesion arm retargeted by "
    "RULING c3-methods-1018b (the slope/veto gate removed instead of lesion_habituation)"
)

# ── the arms (PLAN section 5; B5's counted arm = lesion_slope_gate) ────────────────────────────────────────────────
ARMS = {
    "intact": {},
    "intact_b": {},                                          # same-seed rebuild: the null
    "lesion_slope_gate": {"slope_gate_off": True},           # B5: the slope/veto gate removed (the counted lesion)
    "lesion_drive": {"drive_lesion": True},                  # B9: topic -> ASK transmission zeroed
    "perm_order": {"turns": "perm_order"},                   # B6: a seeded shuffle of the probe order
    "perm_exposure": {"turns": "perm_exposure"},             # B7: Stage-2's permuted exposure
    "shared_slice": {"shared_slice": True},                  # informational: topic specificity removed
    "lesion_reset": {"reset_on": False},                     # informational: the DA-gated restore removed (inert here)
}
COUNTED_ARMS = ("intact", "intact_b", "lesion_slope_gate", "lesion_drive", "perm_order", "perm_exposure")

PREDICTIONS_INFORMATIONAL = dict(M1.PREDICTIONS_INFORMATIONAL)


def turns_for(arm, seed):
    kind = ARMS[arm].get("turns")
    if kind == "perm_order":
        return M1.perm_order_turns(seed)
    if kind == "perm_exposure":
        return list(M1.ASK_PERMUTED_TURNS)
    return list(M1.ASK_TURNS)


# ── the bridge (method 1's wiring, verbatim, with two declared build-state changes) ─────────────────────────────────
def build_slope_gate_bridge(seed, *, w_topic_ask, tau_hab_ms=M3_TAU_MS, ou_process=False):
    """Method 1's build_lp_bridge, unchanged in wiring, with the module's declared build state: stp_tau_d =
    M3_TAU_MS (depression inert -- the slope gate, not the trace, is the brake) and enable_ou_process =
    ou_process (default False, method 2's settled diagnosis; pass True for the controlled noise comparison)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_neuron_ou_seed = True
    cfg.enable_ou_process = bool(ou_process)          # METHOD 3's build state (default off; declared in the header)
    cfg.enable_homeostasis = False
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    for _flag in ("enable_stdp", "enable_hebbian_learning", "enable_reward_modulation", "enable_structural_plasticity",
                  "enable_neuromodulator_subsystem"):
        setattr(cfg, _flag, False)
    cfg.enable_parameter_heterogeneity = False
    cfg.per_region_threshold_heterogeneity = True
    cfg.per_region_parameter_heterogeneity = True
    cfg.current_novelty_signal = 0.0
    cfg.current_reward_signal = 0.0
    cfg.enable_short_term_plasticity = True
    cfg.enable_per_type_stp = False
    cfg.stp_U = float(STP_U)
    cfg.stp_tau_f = float(STP_TAU_F)
    cfg.stp_tau_d = float(tau_hab_ms)                 # M3_TAU_MS (200 ms): inert depression by design

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="topic_aff", n_neurons=N_SLOTS * INPOP, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        BrainRegion(name="reward_us", n_neurons=N_REWARD_US, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        BrainRegion(name="snc", n_neurons=N_SNC, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        BrainRegion(name="ask", n_neurons=N_ASK, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
    ]
    for _rg in cfg.brain_regions:
        _rg.enable_heterogeneity = True
    cfg.region_pathways = [
        RegionPathway(from_region="topic_aff", to_region="ask", density=TOPIC_ASK_DENSITY,
                      weight_mean=float(w_topic_ask), weight_jitter=TOPIC_ASK_JITTER, plastic=False),
        RegionPathway(from_region="reward_us", to_region="snc", density=0.6, weight_mean=REWARD_US_TO_SNC_W,
                      weight_jitter=0.2, plastic=False, stp_disabled=True),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = int(seed)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge, cfg


class CuriositySlopeGateOrgan(M1.CuriosityLPOrgan):
    """Method 1's organ with the per-topic slope/veto gate. Inherits build, calibration, reads, present()'s
    presentation mechanics and answer()'s DA read; overrides _build_one (module bridge), present() (the ask
    decision reads threshold + the topic's omission veto and updates the LP slope from the want read) and answer()
    (the teach jump: DA phasic-above-tonic Hz + the last want, slope > eps clears the veto). The gate's host
    state is `gate_state[topic] = {fast, slow, veto, last_want, n}` (DECLARED HOST SCAFFOLD)."""

    def __init__(self, seed, *, w_topic_ask, tau_hab_ms=M3_TAU_MS, reset_on=True, drive_lesion=False,
                 shared_slice=False, slope_gate_off=False, ou_process=False):
        super().__init__(seed=seed, w_topic_ask=w_topic_ask, tau_hab_ms=tau_hab_ms, reset_on=reset_on,
                         drive_lesion=drive_lesion, shared_slice=shared_slice)
        self.slope_gate_off = bool(slope_gate_off)
        self.ou_process = bool(ou_process)
        self.gate_state = {}

    def _build_one(self, lesion=False):
        from sim.backend import get_backend, from_host
        xp, _ = get_backend()
        bridge, cfg = build_slope_gate_bridge(self.seed, w_topic_ask=self.w_topic_ask,
                                               tau_hab_ms=self.tau_hab_ms, ou_process=self.ou_process)
        topic_idx = _idx(bridge, "topic_aff")
        self._slot_neurons = [topic_idx[k * INPOP:(k + 1) * INPOP] for k in range(N_SLOTS)]
        self._idx_topic_all = xp.asarray(topic_idx)
        self._idx_snc = xp.asarray(_idx(bridge, "snc"))
        self._idx_us = xp.asarray(_idx(bridge, "reward_us"))
        self._n_snc = int(len(_idx(bridge, "snc")))
        ask_idx = _idx(bridge, "ask")
        self._n_ask = int(len(ask_idx))
        self._num = int(bridge.cp_membrane_potential_v.shape[0])
        coo = bridge.cp_connections.tocoo()
        rows, cols = M1._h(coo.row), M1._h(coo.col)
        ask_set = np.zeros(self._num, bool); ask_set[ask_idx] = True
        self._slot_syn = []
        for k in range(N_SLOTS):
            pre = np.zeros(self._num, bool); pre[self._slot_neurons[k]] = True
            self._slot_syn.append(np.nonzero(pre[rows] & ask_set[cols])[0])
        self._n_syn = int(bridge.cp_connections.nnz)
        if self.drive_lesion:
            data = M1._h(bridge.cp_connections.data).copy()
            for k in range(N_SLOTS):
                data[self._slot_syn[k]] = 0.0
            bridge.cp_connections.data[:] = from_host(data.astype(np.float32))
        _settle(bridge, M1.W_SETTLE)
        snap0 = _snapshot_state(bridge)
        self._snap_neuron = {k: v for k, v in snap0.items() if k not in M1._STP_KEYS}
        return bridge, cfg, xp, xp.asarray(ask_idx), snap0

    # ── the slope/veto gate (DECLARED HOST STATE) ─────────────────────────────────────────────────────────────────
    def _gate(self, topic):
        st = self.gate_state.get(topic)
        if st is None:
            st = {"fast": None, "slow": None, "veto": 0.0, "last_want": 0.0, "n": 0}
            self.gate_state[topic] = st
        return st

    def gate_update(self, topic, sample):
        """One LP observation: slope = fast EMA - slow EMA of the topic's samples (both initialize on the first,
        so slope starts 0), then the omission veto: |slope| <= eps grows VETO_STEP (stays ~0: probed, never
        answered), slope > eps clears it (progress: veto-protected), slope < -eps grows double. Returns the slope
        (or None under a gate lesion, where the veto never moves)."""
        if self.slope_gate_off:
            return None
        st = self._gate(topic)
        sample = float(sample)
        if st["fast"] is None:
            st["fast"] = st["slow"] = sample
            slope = 0.0
        else:
            fast = ALPHA_FAST * sample + (1.0 - ALPHA_FAST) * st["fast"]
            slow = ALPHA_SLOW * sample + (1.0 - ALPHA_SLOW) * st["slow"]
            slope = fast - slow
            st["fast"], st["slow"] = fast, slow
        st["n"] += 1
        if slope > SLOPE_EPS_HZ:
            st["veto"] = 0.0
        elif slope < -SLOPE_EPS_HZ:
            st["veto"] = min(VETO_MAX_HZ, st["veto"] + VETO_STEP_DECLINE_HZ)
        else:
            st["veto"] = min(VETO_MAX_HZ, st["veto"] + VETO_STEP_HZ)
        return float(slope)

    def present(self, topic, novelty):
        """One presentation with the gate: the ask decision reads want >= threshold + veto (pre-update: the first
        probe of any topic sees veto 0), then the want read updates the topic's LP slope."""
        self.ensure_built()
        slot = self.slot_for(topic)
        x_pre = self._x_mean(slot)
        self._wash_neurons()
        ask_spk, slot_spk = self._drive_slot(slot, novelty)
        want = self._want_hz(ask_spk)
        x_post = self._x_mean(slot)
        self._wash_neurons()
        st = self._gate(topic)
        st["last_want"] = float(want)
        veto_pre = float(st["veto"])
        threshold_eff = float(self.threshold) + veto_pre
        slope = self.gate_update(topic, want)
        return {"topic": topic, "slot": int(slot), "novelty": float(novelty), "want_hz": float(want),
                "threshold": float(self.threshold), "threshold_eff": threshold_eff,
                "veto_pre": veto_pre, "veto_post": float(st["veto"]), "slope": slope,
                "curious": bool(want >= threshold_eff),
                "x_pre": x_pre, "x_post": x_post, "afferent_spikes": int(slot_spk)}

    def answer(self, word, lp):
        """The teach jump: method 1's DA read (phasic-above-tonic), then the topic's LP sample = the DA read in Hz
        plus the topic's last want read (the proposal's progress read); a positive slope clears the veto."""
        rec = super().answer(word, lp)
        da_hz = max(0.0, float(rec.get("snc_hz") or 0.0) - float(rec.get("snc_neutral_hz") or 0.0))
        st = self._gate(word)
        sample = da_hz + float(st.get("last_want") or 0.0)
        veto_before = float(st["veto"])
        slope = self.gate_update(word, sample)
        rec.update({"lp_sample": float(sample), "slope": slope, "veto_before": veto_before,
                    "veto_after": float(st["veto"])})
        return rec

    def gate_snapshot(self):
        return {t: dict(st) for t, st in sorted(self.gate_state.items())}


def run_arm(seed, arm, *, w_topic_ask):
    """Method 1's battery loop, verbatim, on the slope-gate organ; records the gate state at the end."""
    cfg = dict(ARMS[arm])
    cfg.pop("turns", None)
    t0 = time.time()
    organ = CuriositySlopeGateOrgan(seed, w_topic_ask=w_topic_ask, **cfg)
    organ.ensure_built()
    world = BatteryWorld()
    turns = turns_for(arm, seed)
    recs = {}
    for label, msg, kind in turns:
        organ.idle(INTER_TURN_MS)
        if kind == "teach":
            fact = parse_teach(msg)
            vocab0 = world.vocab()
            before = {w: organ.novelty(w, vocab0) for w in (fact[0], fact[2])}
            world.store(fact)
            vocab1 = world.vocab()
            ans = []
            for w in (fact[0], fact[2]):
                lp = max(0.0, before[w] - organ.novelty(w, vocab1))
                if w in organ._slot_of:
                    ans.append(organ.answer(w, lp))
            recs[label] = {"kind": kind, "message": msg, "fact": list(fact), "answer": "",
                           "abstained": False, "asked": None, "answers": ans}
            continue
        sv = parse_probe(msg)
        obj = world.recall(*sv)
        if obj is not None:
            recs[label] = {"kind": kind, "message": msg, "abstained": False, "recalled": obj, "asked": None,
                           "answer": "the %s %ss the %s" % (sv[0], sv[1], obj)}
            continue
        topic = extract_topic(msg)
        nov = organ.novelty(topic, world.vocab())
        r = organ.present(topic, nov)
        text = ABSTAIN_TEXT + (followup_question(topic) if r["curious"] else "")
        recs[label] = dict(r, kind=kind, message=msg, abstained=True, recalled=None, asked=bool(r["curious"]),
                           answer=text)
    data = M1._h(organ.bridge.cp_connections.data)
    holds = {"stp_tau_d": float(organ.bridge.core_config.stp_tau_d),
             "enable_ou_process": bool(organ.bridge.core_config.enable_ou_process),
             "slope_gate_off": bool(organ.slope_gate_off),
             "topic_ask_weight_sum": float(sum(float(np.abs(data[organ._slot_syn[k]]).sum())
                                               for k in range(N_SLOTS)))}
    return {"seed": int(seed), "arm": arm, "turn_order": [t[0] for t in turns], "responses": recs,
            "manipulation_at_end": holds,
            "calib": organ.calib, "threshold": float(organ.threshold), "snc_neutral_hz": organ.snc_neutral_hz,
            "idle_topic_spikes": int(organ.idle_topic_spikes), "w_topic_ask": float(w_topic_ask),
            "tau_hab_ms": organ.tau_hab_ms, "gate_state": organ.gate_snapshot(),
            "wall_s": round(time.time() - t0, 1)}


def score_seed(shards):
    """The same B1-B9 code as method 1 with only B5 retargeted (RULING c3-methods-1018b): map this module's
    counted lesion arm onto method 1's lesion slot before scoring, then rename the recorded B5 keys."""
    m1_shards = dict(shards)
    if "lesion_slope_gate" in m1_shards:
        m1_shards["lesion_habituation"] = m1_shards["lesion_slope_gate"]
    out = M1.score_seed(m1_shards)
    ta = out.get("toad_asks")
    if isinstance(ta, dict) and "lesion_habituation" in ta:
        ta["lesion_slope_gate"] = ta.pop("lesion_habituation")
    return out


BAR_KEYS = M1.BAR_KEYS     # the same B1-B9 key set (B5's keys are arm-agnostic)


def aggregate(per_seed):
    """Method 1's aggregate, verbatim, with B5 retargeted: the treatment is the lesion_slope_gate arm's excess
    toad asks over intact, against the intact_b null (tools.lab.attributable_to)."""
    from tools.lab import attributable_to
    seeds = sorted(per_seed)
    bars = {}
    for k in BAR_KEYS:
        vals = [per_seed[s].get(k) for s in seeds]
        bars[k] = {"n_true": sum(1 for v in vals if v is True), "n": len(seeds),
                   "holds": len(seeds) == len(BATTERY_SEEDS) and all(v is True for v in vals)}
    t = sum((per_seed[s].get("toad_asks") or {}).get("lesion_slope_gate", 0) -
            (per_seed[s].get("toad_asks") or {}).get("intact", 0) for s in seeds)
    c = sum((per_seed[s].get("toad_asks") or {}).get("intact_b", 0) -
            (per_seed[s].get("toad_asks") or {}).get("intact", 0) for s in seeds)
    att = attributable_to("lesion_slope_gate toad asks", t, c)
    bars["B5_attribution"] = {"treatment": t, "control": c, "fraction": att,
                              "holds": att is not None and att >= 0.5}
    verdict = "GO" if all(v["holds"] for v in bars.values()) else "NO-GO"
    info = {
        "shared_slice_later_first_gap_unasked": sum(1 for s in seeds
                                                    if per_seed[s].get("shared_slice_later_first_gap_unasked")),
        "lesion_reset_reply_identical": sum(1 for s in seeds if per_seed[s].get("lesion_reset_reply_identical")),
        "asks_before_stop": {s: per_seed[s].get("asks_before_stop") for s in seeds},
        "stage2_c3_all": sum(1 for s in seeds if all((per_seed[s].get("stage2_c3") or {}).get(k)
                                                     for k in ("discriminates", "lp_learnable", "lp_unlearnable"))),
    }
    return {"verdict": verdict, "bars": bars, "informational": info}


def run_battery(seeds, *, w_topic_ask, arms=tuple(ARMS)):
    shards, per_seed = {}, {}
    for s in seeds:
        shards[s] = {}
        for a in arms:
            sh = run_arm(s, a, w_topic_ask=w_topic_ask)
            shards[s][a] = sh
            print("  seed %d %-18s asked=%s  thr=%.1f  wall=%.1fs" % (
                s, a, "".join("1" if (sh["responses"][l].get("asked")) else "." for l in sh["turn_order"]),
                sh["threshold"], sh["wall_s"]), flush=True)
        per_seed[s] = score_seed(shards[s])
    return shards, per_seed


def selftest():
    """Pure-logic checks (no bridge): the world/turns (method 1's, reasserted), the gate's EMA/veto math, the
    rekeyed scorer equivalence against method 1's code, and the aggregate's all-8-seeds rule."""
    checks = {}
    checks["parse_teach"] = parse_teach("the ibex chases the marmot") == ("ibex", "chase", "marmot")
    checks["parse_probe"] = parse_probe("what does the toad chase") == ("toad", "chase")
    checks["extract_topic"] = extract_topic("what does the toad chase") == "toad"
    w = BatteryWorld()
    checks["builtin_recall"] = w.recall("dog", "chase") == "cat"
    w.store(parse_teach("the ibex chases the marmot"))
    checks["taught_recall"] = w.recall("ibex", "chase") == "marmot"

    # turn orders are method 1's (imported), reasserted for the bars they carry
    for s in BATTERY_SEEDS:
        lab = [x[0] for x in M1.perm_order_turns(s)]
        checks["perm_order_seed%d" % s] = (sorted(lab) == sorted(x[0] for x in M1.ASK_TURNS)
                                           and lab.index("lp_ibex_1") < lab.index("t_ibex")
                                           < lab.index("lp_ibex_2")
                                           and [l for l in lab if l in TOAD] == list(TOAD))

    # the gate: EMAs initialize on the first sample; flat slope accumulates VETO_STEP; progress clears it;
    # a falling series crosses the negative eps (decline doubles); the cap holds
    org = CuriositySlopeGateOrgan(1, w_topic_ask=W_TOPIC_ASK)
    checks["gate_first_sample_vetoes_step"] = (org.gate_update("t_flat", 35.0) == 0.0
                                               and org._gate("t_flat")["veto"] == VETO_STEP_HZ)
    for _ in range(3):
        org.gate_update("t_flat", 35.0)
    checks["gate_flat_accumulates"] = org._gate("t_flat")["veto"] == 4 * VETO_STEP_HZ
    slope = org.gate_update("t_flat", 35.0 + 500.0)
    checks["gate_progress_clears"] = slope > SLOPE_EPS_HZ and org._gate("t_flat")["veto"] == 0.0
    org.gate_update("t_dec", 100.0)
    org.gate_update("t_dec", 100.0)
    slope_down = org.gate_update("t_dec", 35.0)
    checks["gate_decline_doubles"] = (slope_down < -SLOPE_EPS_HZ
                                      and org._gate("t_dec")["veto"] == 2 * VETO_STEP_HZ + VETO_STEP_DECLINE_HZ)
    for _ in range(14):
        org.gate_update("t_cap", 35.0)
    checks["gate_caps_at_max"] = org._gate("t_cap")["veto"] == VETO_MAX_HZ
    org_l = CuriositySlopeGateOrgan(1, w_topic_ask=W_TOPIC_ASK, slope_gate_off=True)
    checks["gate_lesion_never_moves"] = org_l.gate_update("t_x", 35.0) is None and org_l._gate("t_x")["veto"] == 0.0

    def fake(asked_labels, turns=M1.ASK_TURNS):
        r = {}
        for l, m, k in turns:
            if k == "teach":
                r[l] = {"kind": k, "asked": None}
            elif l in ("lp_ibex_2",):
                r[l] = {"kind": k, "asked": None, "recalled": "marmot"}
            elif l.startswith("c_dog"):
                r[l] = {"kind": k, "asked": None, "recalled": "cat"}
            else:
                r[l] = {"kind": k, "asked": l in asked_labels}
        return {"responses": r, "turn_order": [t[0] for t in turns]}

    good = fake(set(FIRST_GAP_PROBES))
    no_stop = fake(set(FIRST_GAP_PROBES) | set(TOAD))
    perm_good = fake({"m_yak", "m_moose", "m_goat", "lp_ibex_1", "u_toad_1"}, M1.ASK_PERMUTED_TURNS)
    none = fake(set())
    shards = {"intact": good, "intact_b": good, "lesion_slope_gate": no_stop, "lesion_drive": none,
              "perm_order": good, "perm_exposure": perm_good}
    sc = score_seed(shards)
    checks["scorer_intact_passes"] = all(sc[k] for k in BAR_KEYS)
    # B5 is retargeted: the scorer reads the lesion_slope_gate arm (rekey proof against method 1's code)
    m1_shards = dict(shards); m1_shards["lesion_habituation"] = m1_shards["lesion_slope_gate"]
    checks["scorer_is_method1_rekeyed"] = M1.score_seed(m1_shards) == M1.score_seed({
        **{k: v for k, v in shards.items() if k != "lesion_slope_gate"},
        "lesion_habituation": shards["lesion_slope_gate"]})
    checks["b5_key_is_slope_gate"] = "lesion_slope_gate" in sc.get("toad_asks", {})
    sc2 = score_seed({"intact": no_stop, "intact_b": no_stop, "lesion_slope_gate": no_stop,
                      "lesion_drive": none, "perm_order": no_stop, "perm_exposure": perm_good})
    checks["scorer_no_stop_fails_B2_B5"] = (not sc2["B2_stop_by_4th"]) and (not sc2["B5_differs"])
    agg = aggregate({s: sc for s in BATTERY_SEEDS})
    checks["aggregate_go_on_synthetic"] = agg["verdict"] == "GO"
    agg2 = aggregate({s: sc for s in BATTERY_SEEDS[:7]})
    checks["aggregate_needs_all_8_seeds"] = agg2["verdict"] == "NO-GO"
    # the module never leaks into production
    checks["b5_bar_mentions_slope_gate"] = "lesion_slope_gate" in BARS["B5_lesion_continues"]
    checks["counted_arms_b5_retargeted"] = ("lesion_slope_gate" in COUNTED_ARMS
                                            and "lesion_habituation" not in ARMS)
    for k, v in checks.items():
        print("  %s %s" % ("PASS" if v else "FAIL", k))
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in BATTERY_SEEDS))
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--ou-process", action="store_true",
                    help="restore the background OU process (default OFF -- method 3's declared build state)")
    ap.add_argument("--w-topic-ask", type=float, default=None,
                    help="override the frozen topic->ASK weight (default: W_TOPIC_ASK); recorded in the artifact")
    args = ap.parse_args()
    if args.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if not args.battery:
        ap.error("one of --selftest / --battery")
    from sim.backend import get_backend
    _, be = get_backend()
    w_topic_ask = W_TOPIC_ASK if args.w_topic_ask is None else float(args.w_topic_ask)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if any(s not in range(1, 16) for s in seeds):
        sys.exit("dev seeds 1-15 only")
    arms = [a for a in args.arms.split(",") if a.strip()]
    shards, per_seed = run_battery(seeds, w_topic_ask=w_topic_ask, arms=arms)
    res = {"runner": RUNNER, "plan": PLAN, "rulings": list(RULINGS), "backend": str(be),
           "w_topic_ask": w_topic_ask, "ou_process": bool(args.ou_process), "tau_hab_ms": M3_TAU_MS,
           "preregistered": {"w_topic_ask": W_TOPIC_ASK, "tau_hab_ms": M3_TAU_MS, "ou_process": False,
                             "alpha_fast": ALPHA_FAST, "alpha_slow": ALPHA_SLOW, "slope_eps_hz": SLOPE_EPS_HZ,
                             "veto_step_hz": VETO_STEP_HZ, "veto_max_hz": VETO_MAX_HZ},
           "bars": BARS, "predictions_informational": PREDICTIONS_INFORMATIONAL,
           "status": "PROVISIONAL de-risk; dev seeds 1-8 only; verdict read only after an ASK to Claude "
                     "(RULING c3-methods-1018c)",
           "per_seed": per_seed, "shards": shards}
    if (sorted(seeds) == sorted(BATTERY_SEEDS) and set(arms) >= set(COUNTED_ARMS)
            and w_topic_ask == W_TOPIC_ASK and not args.ou_process):
        res["aggregate"] = aggregate(per_seed)
        print("VERDICT:", res["aggregate"]["verdict"])
        for k, v in res["aggregate"]["bars"].items():
            print("  %-28s %s" % (k, v))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("wrote", args.out)


if __name__ == "__main__":
    main()