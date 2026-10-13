"""CURIOSITY LEARNING-PROGRESS TRACE -- a per-topic, synaptic brake on asking (Stage-2 C3 gap), section-5 de-risk.

Packet: docs/drafts/curiosity-lp/PLAN.md (bars, arms and the design-calibration rule were written there and here
BEFORE any run). Dev seeds only (1-8 battery, 9 design smoke). Default-off by construction: this module is never
imported by webapp/ or by the production organ, and the production organ (`curiosity_production_organ.py`) is not
edited. No sim/ edit.

THE GAP (Stage-2 C3, dev seed 7, `research/findings/raw/_stage2_exit/dev/s7/ask__intact_a.json`). The production ask
decision is `want_hz * da_crave_gain >= threshold` (`webapp/da_curiosity_drives_chat.py`). The only brake on asking is
the GLOBAL self-produced DA level: on s7 it fell 0.55 -> 0.26 -> 0.06 over the first two unanswered asks, the crave gain
fell to 0.32, and every later probe (g_newt, lp_ibex_1, u_toad_1..4) went unasked although each novel topic still drove
the ASK pool to 38.9 Hz against a 19.1 Hz threshold. Nothing in the organ knows WHICH topic went unanswered, so the
brake cannot be per-topic: it stops a fresh topic as hard as the one that has been asked three times.

THE MECHANISM (one bridge; the ASK pool is the production organ's own ASK-pool build; the per-topic brake lives in the
synapses' Tsodyks-Markram resource variable `cp_stp_x`, deepened by real presynaptic spikes):
  * Each topic word drives its own slice of a `topic_aff` population (25 RS neurons per slice, the habituation organ's
    INPOP) with a short burst whose amplitude is the topic's GRADED novelty (the production `TopicNoveltyGate` read,
    the same host boundary the organ already declares). Slices are assigned on first sight of a word (a fixed
    word -> population code, like the organ's per-word phase code; the slice table is the input encoding, not a count).
  * Every slice projects onto the ONE ASK pool (convergent, density 1.0) through depression-dominant STP synapses
    (U = 0.35, tau_f = 10 ms: the de-risked regime of `spiking_novelty_habituation_organ.py`) whose recovery constant is
    the habituation scale TAU_HAB_MS (12 s of substrate time against a 0.4 s silent gap between turns: per-turn
    recovery 3.3 % of the deficit). A presentation releases transmitter and depletes ONLY that slice's synapses, so
    each unanswered ask on a topic weakens THAT topic's novelty -> ASK drive and no other's. The novelty read itself is
    untouched: the gap stays a gap (the brain still abstains, the familiarity gate still reads novel); what habituates
    is the drive to ASK about it.
  * The answer arriving = the teacher's store write. Its learning progress (the topic's novelty before minus after the
    write, read off the same familiarity gate) drives a `reward_us` population onto an `snc` population (the
    curiosity bridge's reward_us -> snc wiring and constants); the SNc burst above its tonic rate is the DA read, and
    that read scales a dishabituating restore of the topic's synapses: x <- x + DA * (1 - x). DECLARED HOST STEP: the
    engine has no neuromodulator target on the STP resource variable, so the restore is applied by this module, gated
    by the spiking SNc read (the same routing class as the critic's current_reward_signal). The realization rung is an
    additive sim/ target (a DA -> STP-recovery term), not done here.
  * The ask decision is the organ's own: `want_hz >= threshold`, with the threshold calibrated by the organ's own
    `ensure_built` rule (midpoint of the NOVEL vs FAMILIAR want, fresh synapses, a dedicated reference slice 0). This
    class SUBCLASSES `CuriosityProductionOrgan` and overrides only the build and the calibration read.

BIOLOGY (cited in PLAN.md section 3): habituation as homosynaptic depression that lasts minutes and is stimulus
specific (Kandel PNS 6e ch. 53, Fig. 53-2; Castellucci & Kandel 1974; Thompson & Spencer 1966; Rankin et al. 2009);
dishabituation by a modulatory transmitter acting on the depressed terminal (Hochner, Klein, Schacher & Kandel 1986);
depressing neocortical synapses (Tsodyks & Markram 1997); learning progress as the intrinsic reward that should steer
asking (Oudeyer, Kaplan & Hafner 2007; Gottlieb, Oudeyer, Lopes & Baranes 2013; Kidd & Hayden 2015); infants withdraw
attention from sequences they cannot learn (Kidd, Piantadosi & Aslin 2012); midbrain DA neurons signal the arrival of
information (Bromberg-Martin & Hikosaka 2009); curiosity-gated DA-hippocampal encoding (Gruber, Gelman & Ranganath 2014).

THE ORGAN BATTERY replays the Stage-2 `ask` group's turn order (`s/research/stage2-exit-gate:
research/runners/stage2_exit_gate.py`, GROUPS["ask"]) at organ level: a teach turn is a store write; a probe recalls
iff the world store holds (subject, verb); an abstain runs the curiosity read on the topic. The store/recall is the
battery's world scaffold, NOT under test (the chat's parse/store can fail -- s7 t_goat did -- and that is out of scope).
No global DA crave gain is applied (tonic, gain 1.0): the question is whether the per-topic trace alone gives the C3
shape. Commands: see PLAN.md section 7.

FUNCTIONAL CORRELATE, NOT phenomenal: a habituating ask drive is reported as a read-out ("I asked about toads three
times and got nothing, so my drive to ask about toads has habituated"), never as felt boredom.
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

from research.runners.curiosity_production_organ import (  # noqa: E402  (reuse-by-import, organ NOT edited)
    CuriosityProductionOrgan,
    TopicNoveltyGate,
    extract_topic,
    followup_question,
    NOVEL_SIGNAL,
    FAMILIAR_SIGNAL,
    N_READ_REPS,
    _reset_ou_read_state,
)
from research.runners.spiking_novelty_habituation_organ import (  # noqa: E402  (the de-risked STP regime)
    STP_U,
    STP_TAU_F,
    INPOP,
    DRIVE_STEPS,
    READ_STEPS,
    WORD_DRIVE,
)
from research.runners._curiosity_seek_learn_onbridge_derisk import (  # noqa: E402  (bridge helpers + DA constants)
    _advance,
    _idx,
    _settle,
    _snapshot_state,
    _restore_state,
    W_SETTLE,
    SNC_TONIC_PA,
    US_GAIN_PA,
    SNC_REWARD_SCALE,
    W_WARMUP,
    W_MEASURE,
)

# ── pre-registered constants (PLAN.md section 4; frozen before the battery) ─────────────────────────────────────────
BATTERY_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)       # dev seeds (owner rule: dev seeds 1-15 only)
DESIGN_SEED = 9                                # the design-calibration smoke seed (NOT a battery seed)
TOPIC_GATE_SEED = 42                           # production `topic_novelty(...)` reads the process gate at its default
                                               # seed 42 whatever BRAIN_CHAT_SEED is; mirrored here
N_ASK = 80                                     # the production organ's ASK pool size (build_curiosity_bridge n_ask)
N_SLOTS = 12                                   # slice 0 = calibration reference; 1..11 = topic slices
N_REWARD_US = 40                               # build_curiosity_bridge n_reward_us
N_SNC = 30                                     # build_curiosity_bridge n_snc
REWARD_US_TO_SNC_W = 10.0                      # build_curiosity_bridge reward_us_to_snc_weight
TAU_HAB_MS = 12000.0                           # habituation recovery constant (substrate ms)
TAU_LESION_MS = 200.0                          # the engine's default E->E STP recovery (stp_tau_d_per_type[0]): the
                                               # habituation-memory lesion (depression recovers between turns)
INTER_TURN_MS = 400                            # silent substrate time between turns (ratio to TAU_HAB_MS = 1/30)
TOPIC_ASK_DENSITY = 1.0
TOPIC_ASK_JITTER = 0.2
W_GRID = (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 80.0)   # the design grid for the topic -> ASK weight
TARGET_WANT_NOVEL_HZ = 40.0                    # the organ's FAITHFUL/CALMER on-spec want_novel (~42 Hz; s7 pooled 38.2)
# Frozen by the design rule (PLAN section 4.2): the W_GRID value whose fresh want_novel on DESIGN_SEED is closest to
# TARGET_WANT_NOVEL_HZ. None until the design smoke has been run and its artifact committed.
W_TOPIC_ASK = None

BUILTIN_FACTS = (("brain", "use", "spikes"), ("brain", "learn", "words"), ("brain", "store", "memory"),
                 ("dog", "chase", "cat"), ("cat", "eat", "fish"))   # tiny-demo built-ins (s7 shard, before teaches)
VERB_LEMMA = {"chases": "chase", "eats": "eat", "chase": "chase", "eat": "eat"}

# ── the Stage-2 `ask` group, verbatim (s/research/stage2-exit-gate @ 4b19aafb8, stage2_exit_gate.py GROUPS["ask"]) ───
ASK_PROBES = [
    ("g_eel", "what does the eel chase", "probe"),
    ("m_yak", "what does the yak chase", "probe"),
    ("g_crab", "what does the crab chase", "probe"),
    ("m_moose", "what does the moose chase", "probe"),
    ("g_newt", "what does the newt chase", "probe"),
    ("m_goat", "what does the goat chase", "probe"),
    ("lp_ibex_1", "what does the ibex chase", "probe"),
    ("t_ibex", "the ibex chases the marmot", "teach"),
    ("lp_ibex_2", "what does the ibex chase", "probe"),
    ("u_toad_1", "what does the toad chase", "probe"),
    ("c_dog_1", "what does the dog chase", "control"),
    ("u_toad_2", "what does the toad chase", "probe"),
    ("c_dog_2", "what does the dog chase", "control"),
    ("u_toad_3", "what does the toad chase", "probe"),
    ("c_dog_3", "what does the dog chase", "control"),
    ("u_toad_4", "what does the toad chase", "probe"),
]
ASK_TURNS = [("t_yak", "the yak eats the grass", "teach"), ("t_moose", "the moose eats the leaf", "teach"),
             ("t_goat", "the goat eats the hay", "teach")] + ASK_PROBES
ASK_PERMUTED_TURNS = [("t_yak", "the eel eats the grass", "teach"), ("t_moose", "the crab eats the leaf", "teach"),
                      ("t_goat", "the newt eats the hay", "teach")] + ASK_PROBES

GAPS = ("g_eel", "g_crab", "g_newt")
MOATS = ("m_yak", "m_moose", "m_goat")
FIRST_GAP_PROBES = ("g_eel", "g_crab", "g_newt", "lp_ibex_1", "u_toad_1")
TOAD = ("u_toad_1", "u_toad_2", "u_toad_3", "u_toad_4")
ASK_RATIO_BAR = 2.0          # Stage-2's own C3 bars (stage2_exit_gate.py)
ASK_GAP_FLOOR = 2.0 / 3.0

# ── the arms (PLAN section 5) ───────────────────────────────────────────────────────────────────────────────────────
ARMS = {
    "intact": {},
    "intact_b": {},                                         # same-seed rebuild: the null
    "lesion_habituation": {"tau_hab_ms": TAU_LESION_MS},    # B5: the habituation memory removed
    "lesion_drive": {"drive_lesion": True},                 # B9: topic -> ASK transmission zeroed
    "perm_order": {"turns": "perm_order"},                  # B6: a seeded shuffle of the probe order
    "perm_exposure": {"turns": "perm_exposure"},            # B7: Stage-2's permuted exposure
    "shared_slice": {"shared_slice": True},                 # informational: topic specificity removed
    "lesion_reset": {"reset_on": False},                    # informational: the DA-gated restore removed
}
COUNTED_ARMS = ("intact", "intact_b", "lesion_habituation", "lesion_drive", "perm_order", "perm_exposure")

BARS = {
    "B1_first_novel_asked": "intact: asked on g_eel (the first novel probe), 8/8 seeds",
    "B1b_every_first_gap_asked": "intact: asked on the first probe of every novel topic "
                                 "(g_eel, g_crab, g_newt, lp_ibex_1, u_toad_1), 8/8 seeds",
    "B2_stop_by_4th": "intact: NOT asked on u_toad_4 (the 4th unanswered probe of one topic), 8/8 seeds",
    "B3_answered_not_reasked": "intact: NOT asked on lp_ibex_2 (the answered gap), 8/8 seeds (0/8 re-asked)",
    "B4_moats_silent": "intact: 0 asks on the taught-topic moats m_yak, m_moose, m_goat (0 of 24)",
    "B5_lesion_continues": "lesion_habituation: asked on u_toad_4, 8/8 seeds, and intact != lesion on u_toad_4 on "
                           "every seed (attributable fraction vs the intact_b null >= 0.5)",
    "B6_perm_order": "perm_order: B1b, B2, B3 and B4 hold, 8/8 seeds",
    "B7_perm_exposure": "perm_exposure: asked on >= 2 of m_moose, m_goat, m_yak (now untaught; yak is untaught "
                        "here too) and on none of g_eel, g_crab, g_newt (now taught); B2 and B3 hold; 8/8 seeds",
    "B8_null": "intact_b: the per-label asked vector equals intact's, 8/8 seeds",
    "B9_drive_lesion": "lesion_drive: 0 asks on every turn, 8/8 seeds",
}
PREDICTIONS_INFORMATIONAL = {
    "shared_slice": "every topic on ONE slice -> the trace becomes a global brake: at least one later first-gap "
                    "probe (g_crab, g_newt, lp_ibex_1, u_toad_1) unasked on >= 6/8 seeds",
    "lesion_reset": "reply-identical to intact (the answered topic reads familiar after its store write, so it is "
                    "never presented again); the ibex slice's x after t_ibex stays below intact's",
    "asks_before_stop": "number of toad probes asked before the first unasked one: reported, not barred",
}


# ── the battery world (the store/recall scaffold; NOT under test) ────────────────────────────────────────────────────
_TEACH_RE = re.compile(r"^\s*the (\w+) (\w+) the (\w+)\s*$", re.IGNORECASE)
_PROBE_RE = re.compile(r"^\s*what does the (\w+) (\w+)\s*$", re.IGNORECASE)


def parse_teach(msg):
    m = _TEACH_RE.match(msg or "")
    if not m:
        return None
    s, v, o = (g.lower() for g in m.groups())
    lemma = VERB_LEMMA.get(v)
    return (s, lemma, o) if lemma else None


def parse_probe(msg):
    m = _PROBE_RE.match(msg or "")
    if not m:
        return None
    s, v = (g.lower() for g in m.groups())
    return (s, VERB_LEMMA.get(v, v))


class BatteryWorld:
    """The organ battery's fact store: built-ins + taught facts. `vocab()` mirrors `webapp/server.py::_brain_vocab`
    (agents U actions U patients of the stored facts), the set production imprints into the topic gate."""

    def __init__(self):
        self.facts = list(BUILTIN_FACTS)

    def store(self, fact):
        if fact not in self.facts:
            self.facts.append(fact)

    def recall(self, subj, verb):
        for s, v, o in self.facts:
            if s == subj and v == verb:
                return o
        return None

    def vocab(self):
        out = set()
        for f in self.facts:
            out |= {w.lower() for w in f}
        return out


def perm_order_turns(seed):
    """B6: a deterministic per-seed shuffle of the 16 probe/teach turns after the three teaches. Constraints (so the
    labels keep their meaning): lp_ibex_1 < t_ibex < lp_ibex_2, and u_toad_1..4 and c_dog_1..3 keep their own order.
    The shuffle fills the shuffled positions of each constrained family in its canonical order."""
    head = ASK_TURNS[:3]
    body = list(ASK_PROBES)
    # single-draw-null: ok the draw is a per-seed stimulus ORDER (8 distinct orders over the 8 seeds) under which the
    # same per-label clauses must hold; it is not a null distribution for any statistic.
    rng = np.random.default_rng(10_000 + int(seed))
    order = list(rng.permutation(len(body)))
    shuffled = [body[i] for i in order]
    families = (("lp_ibex_1", "t_ibex", "lp_ibex_2"), TOAD, ("c_dog_1", "c_dog_2", "c_dog_3"))
    for fam in families:
        pos = [i for i, t in enumerate(shuffled) if t[0] in fam]
        canon = [t for t in body if t[0] in fam]
        for p, t in zip(pos, canon):
            shuffled[p] = t
    return head + shuffled


def turns_for(arm, seed):
    kind = ARMS[arm].get("turns")
    if kind == "perm_order":
        return perm_order_turns(seed)
    if kind == "perm_exposure":
        return list(ASK_PERMUTED_TURNS)
    return list(ASK_TURNS)


# ── the bridge ──────────────────────────────────────────────────────────────────────────────────────────────────────
def build_lp_bridge(seed, *, w_topic_ask, tau_hab_ms=TAU_HAB_MS):
    """ONE SimulationBridge: `topic_aff` (N_SLOTS x INPOP RS) --[depression-dominant STP]--> `ask` (N_ASK RS, the
    production ASK-pool settings: homeostasis-free, per-neuron OU seed, name-keyed per-region heterogeneity), plus
    `reward_us` --> `snc` (the curiosity bridge's DA read; STP disabled on it so the DA read never habituates)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.per_neuron_ou_seed = True
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
    # THE mechanism: depression-dominant Tsodyks-Markram STP (the habituation organ's de-risked U and tau_f), with the
    # habituation recovery constant. Global (not per-type) parameters so only these values apply.
    cfg.enable_short_term_plasticity = True
    cfg.enable_per_type_stp = False
    cfg.stp_U = float(STP_U)
    cfg.stp_tau_f = float(STP_TAU_F)
    cfg.stp_tau_d = float(tau_hab_ms)

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


def _h(a):
    from sim.backend import to_host
    try:
        return np.asarray(to_host(a))
    except Exception:
        return np.asarray(a)


_STP_KEYS = ("cp_stp_x", "cp_stp_u")


class CuriosityLPOrgan(CuriosityProductionOrgan):
    """The production curiosity organ with a per-topic habituating novelty -> ASK synapse. Reuses the organ's
    `ensure_built` (calibration rule), `extract_topic`, `followup_question` and `TopicNoveltyGate`; overrides only the
    bridge build (`_build_one`) and the calibration read (`_read_want_raw`, on the reference slice with fresh synapses).
    Live reads go through `present(topic)`, which does NOT restore the synapses' STP state: that state is the trace."""

    def __init__(self, seed, *, w_topic_ask, tau_hab_ms=TAU_HAB_MS, reset_on=True, drive_lesion=False,
                 shared_slice=False):
        super().__init__(seed=seed, shared=None)
        self.w_topic_ask = float(w_topic_ask)
        self.tau_hab_ms = float(tau_hab_ms)
        self.reset_on = bool(reset_on)
        self.drive_lesion = bool(drive_lesion)
        self.shared_slice = bool(shared_slice)
        self.gate = TopicNoveltyGate(seed=TOPIC_GATE_SEED)
        self._slot_of = {}
        self.snc_neutral_hz = None
        self.idle_topic_spikes = 0

    # ── build ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _build_one(self, lesion=False):
        from sim.backend import get_backend, from_host
        xp, _ = get_backend()
        bridge, cfg = build_lp_bridge(self.seed, w_topic_ask=self.w_topic_ask, tau_hab_ms=self.tau_hab_ms)
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
        rows, cols = _h(coo.row), _h(coo.col)
        ask_set = np.zeros(self._num, bool); ask_set[ask_idx] = True
        self._slot_syn = []
        for k in range(N_SLOTS):
            pre = np.zeros(self._num, bool); pre[self._slot_neurons[k]] = True
            self._slot_syn.append(np.nonzero(pre[rows] & ask_set[cols])[0])
        self._n_syn = int(bridge.cp_connections.nnz)
        if self.drive_lesion:
            # B9: the topic -> ASK transmission removed (the analog of BRAIN_CURIOSITY_LESION's drive-pathway cut)
            data = _h(bridge.cp_connections.data).copy()
            for k in range(N_SLOTS):
                data[self._slot_syn[k]] = 0.0
            bridge.cp_connections.data[:] = from_host(data.astype(np.float32))
        _settle(bridge, W_SETTLE)
        snap0 = _snapshot_state(bridge)
        self._snap_neuron = {k: v for k, v in snap0.items() if k not in _STP_KEYS}
        return bridge, cfg, xp, xp.asarray(ask_idx), snap0

    def ensure_built(self):
        if self._built:
            return
        super().ensure_built()
        self.snc_neutral_hz = self._snc_rate(0.0)

    # ── reads ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _wash_neurons(self):
        """The organ's drift-free wash-out (restore the clean post-settle neuron state + a deterministic OU start),
        WITHOUT touching the synapses' STP resource state."""
        _restore_state(self.bridge, self._snap_neuron)
        _reset_ou_read_state(self.bridge)

    def _drive_slot(self, slot, novelty):
        """One presentation: a DRIVE_STEPS burst on the slot's neurons at novelty * WORD_DRIVE, then READ_STEPS silent.
        Returns the ASK-pool spike count over the whole window and the slot's spike count (the afferent volley)."""
        from sim.backend import from_host
        b = self.bridge
        cur = np.zeros(self._num, np.float32)
        cur[self._slot_neurons[slot]] = float(novelty) * float(WORD_DRIVE)
        sl = self.xp.asarray(self._slot_neurons[slot])
        ask_spk = slot_spk = 0
        b.cp_external_input_current[:] = from_host(cur)
        for _ in range(DRIVE_STEPS):
            _advance(b)
            ask_spk += int(b.cp_firing_states[self.idx_ask].sum())
            slot_spk += int(b.cp_firing_states[sl].sum())
        b.cp_external_input_current[:] = 0.0
        for _ in range(READ_STEPS):
            _advance(b)
            ask_spk += int(b.cp_firing_states[self.idx_ask].sum())
            slot_spk += int(b.cp_firing_states[sl].sum())
        return ask_spk, slot_spk

    def _want_hz(self, ask_spk):
        return ask_spk / max(self._n_ask, 1) / ((DRIVE_STEPS + READ_STEPS) * 1e-3)

    def _read_want_raw(self, novelty, bridge, xp, idx_ask, snap0):
        """Calibration read (called by the organ's `ensure_built`): the reference slice 0 with FRESH synapses each rep
        (full restore incl. STP), N_READ_REPS reps like the organ; the full clean state is restored afterwards."""
        vals = []
        for _ in range(N_READ_REPS):
            _restore_state(bridge, snap0)
            _reset_ou_read_state(bridge)
            ask_spk, _ = self._drive_slot(0, novelty)
            vals.append(self._want_hz(ask_spk))
        _restore_state(bridge, snap0)
        return float(np.mean(vals))

    def _x_mean(self, slot):
        x = _h(self.bridge.cp_stp_x)
        s = self._slot_syn[slot]
        return float(x[s].mean()) if len(s) else float("nan")

    def slot_for(self, topic):
        if self.shared_slice:
            self._slot_of[topic] = 1          # every topic on ONE slice (the topic-specificity anti-cheat)
            return 1
        if topic not in self._slot_of:
            if len(self._slot_of) >= N_SLOTS - 1:
                raise RuntimeError("topic slices exhausted (N_SLOTS=%d)" % N_SLOTS)
            self._slot_of[topic] = len(self._slot_of) + 1
        return self._slot_of[topic]

    def novelty(self, word, vocab):
        self.gate.imprint_vocab(vocab)
        return float(self.gate.novelty(word))

    def present(self, topic, novelty):
        """The live curiosity read on an abstain: one presentation of the topic's slice. The synapses' STP state
        carries over (the trace); the neuron state is washed before and after (drift-free, like the organ)."""
        self.ensure_built()
        slot = self.slot_for(topic)
        x_pre = self._x_mean(slot)
        self._wash_neurons()
        ask_spk, slot_spk = self._drive_slot(slot, novelty)
        want = self._want_hz(ask_spk)
        x_post = self._x_mean(slot)
        self._wash_neurons()
        return {"topic": topic, "slot": int(slot), "novelty": float(novelty), "want_hz": float(want),
                "threshold": float(self.threshold), "curious": bool(want >= self.threshold),
                "x_pre": x_pre, "x_post": x_post, "afferent_spikes": int(slot_spk)}

    def _snc_rate(self, lp):
        """The SNc firing over W_MEASURE after W_WARMUP at tonic drive + an LP-scaled reward_us drive."""
        from sim.backend import from_host
        b = self.bridge
        self._wash_neurons()
        cur = np.zeros(self._num, np.float32)
        cur[_h(self._idx_snc)] = SNC_TONIC_PA
        cur[_h(self._idx_us)] = float(max(0.0, lp)) * US_GAIN_PA
        b.cp_external_input_current[:] = from_host(cur)
        spk = 0
        for i in range(W_WARMUP + W_MEASURE):
            _advance(b)
            if i >= W_WARMUP:
                spk += int(b.cp_firing_states[self._idx_snc].sum())
        b.cp_external_input_current[:] = 0.0
        self._wash_neurons()
        return spk / max(self._n_snc, 1) / (W_MEASURE * 1e-3)

    def answer(self, word, lp):
        """The answer's store write evoked learning progress `lp` on `word`: read the SNc burst, and if the word has a
        topic slice, restore its synapses by the DA read (dishabituation). Returns the trace."""
        self.ensure_built()
        rate = self._snc_rate(lp)
        da = float(np.clip((rate - self.snc_neutral_hz) / SNC_REWARD_SCALE, 0.0, 1.0))
        rec = {"word": word, "lp": float(lp), "snc_hz": float(rate), "snc_neutral_hz": float(self.snc_neutral_hz),
               "da": da, "slot": None, "x_before": None, "x_after": None, "reset_applied": False}
        slot = self._slot_of.get(word)
        if slot is not None:
            rec["slot"] = int(slot)
            rec["x_before"] = self._x_mean(slot)
            if self.reset_on and da > 0.0:
                from sim.backend import from_host
                x = _h(self.bridge.cp_stp_x).copy()
                s = self._slot_syn[slot]
                x[s] = x[s] + da * (1.0 - x[s])
                self.bridge.cp_stp_x[:] = from_host(x.astype(np.float32))
                rec["reset_applied"] = True
            rec["x_after"] = self._x_mean(slot)
        return rec

    def idle(self, ms=INTER_TURN_MS):
        """Silent substrate time between turns (STP recovers by its own tau_d). Counts any topic-slice spikes (a
        spontaneous volley would deplete synapses outside a presentation; expected 0)."""
        self.ensure_built()
        b = self.bridge
        b.cp_external_input_current[:] = 0.0
        n = 0
        for _ in range(int(ms)):
            _advance(b)
            n += int(b.cp_firing_states[self._idx_topic_all].sum())
        self.idle_topic_spikes += n
        self._wash_neurons()
        return n


# ── the battery ─────────────────────────────────────────────────────────────────────────────────────────────────────
ABSTAIN_TEXT = "I don't know about that."


def run_arm(seed, arm, *, w_topic_ask, tau_hab_ms=TAU_HAB_MS):
    cfg = dict(ARMS[arm])
    cfg.pop("turns", None)
    cfg.setdefault("tau_hab_ms", float(tau_hab_ms))     # the lesion arm carries its own tau_d
    t0 = time.time()
    organ = CuriosityLPOrgan(seed, w_topic_ask=w_topic_ask, **cfg)
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
    # TERMS "lesion": the manipulation still holds at the end of the run (no plasticity is live, so it should)
    data = _h(organ.bridge.cp_connections.data)
    holds = {"stp_tau_d": float(organ.bridge.core_config.stp_tau_d),
             "topic_ask_weight_sum": float(sum(float(np.abs(data[organ._slot_syn[k]]).sum()) for k in range(N_SLOTS)))}
    return {"seed": int(seed), "arm": arm, "turn_order": [t[0] for t in turns], "responses": recs,
            "manipulation_at_end": holds,
            "calib": organ.calib, "threshold": float(organ.threshold), "snc_neutral_hz": organ.snc_neutral_hz,
            "idle_topic_spikes": int(organ.idle_topic_spikes), "w_topic_ask": float(w_topic_ask),
            "tau_hab_ms": organ.tau_hab_ms, "wall_s": round(time.time() - t0, 1)}


def asked(shard, label):
    r = (shard.get("responses") or {}).get(label)
    if r is None:
        return None
    return bool(r.get("asked"))


def score_seed(shards):
    """Per-seed clause table over the arms present (a dict arm -> shard)."""
    it = shards.get("intact") or {}
    out = {}

    def first_gaps_asked(sh):
        return all(asked(sh, l) is True for l in FIRST_GAP_PROBES)

    def stop4(sh):
        return asked(sh, "u_toad_4") is False

    def ibex2(sh):
        r = (sh.get("responses") or {}).get("lp_ibex_2") or {}
        return r.get("recalled") == "marmot" and not r.get("asked")

    def moats_asks(sh):
        return sum(1 for l in MOATS if asked(sh, l))

    out["B1_first_novel_asked"] = asked(it, "g_eel") is True
    out["B1b_every_first_gap_asked"] = first_gaps_asked(it)
    out["B2_stop_by_4th"] = stop4(it)
    out["B3_answered_not_reasked"] = ibex2(it)
    out["moat_asks"] = moats_asks(it)
    out["B4_moats_silent"] = out["moat_asks"] == 0
    le = shards.get("lesion_habituation")
    if le is not None:
        out["B5_lesion_asks_4th"] = asked(le, "u_toad_4") is True
        out["B5_differs"] = asked(le, "u_toad_4") != asked(it, "u_toad_4")
        out["toad_asks"] = {"intact": sum(1 for l in TOAD if asked(it, l)),
                            "lesion_habituation": sum(1 for l in TOAD if asked(le, l))}
        if shards.get("intact_b") is not None:
            out["toad_asks"]["intact_b"] = sum(1 for l in TOAD if asked(shards["intact_b"], l))
    po = shards.get("perm_order")
    if po is not None:
        out["B6_perm_order"] = first_gaps_asked(po) and stop4(po) and ibex2(po) and moats_asks(po) == 0
    pe = shards.get("perm_exposure")
    if pe is not None:
        n_new_gaps = sum(1 for l in MOATS if asked(pe, l))
        n_new_moats = sum(1 for l in GAPS if asked(pe, l))
        out["B7_perm_exposure"] = n_new_gaps >= 2 and n_new_moats == 0 and stop4(pe) and ibex2(pe)
        out["perm_exposure_asks"] = {"untaught_m": n_new_gaps, "taught_g": n_new_moats}
    ib = shards.get("intact_b")
    if ib is not None:
        labels = sorted(set((it.get("responses") or {})) | set((ib.get("responses") or {})))
        out["B8_null"] = all(asked(it, l) == asked(ib, l) for l in labels)
    ld = shards.get("lesion_drive")
    if ld is not None:
        out["B9_drive_lesion"] = sum(1 for l, r in (ld.get("responses") or {}).items() if r.get("asked")) == 0
    ss = shards.get("shared_slice")
    if ss is not None:
        out["shared_slice_later_first_gap_unasked"] = any(asked(ss, l) is False for l in FIRST_GAP_PROBES[1:])
    lr = shards.get("lesion_reset")
    if lr is not None:
        labels = sorted(set((it.get("responses") or {})) | set((lr.get("responses") or {})))
        out["lesion_reset_reply_identical"] = all(asked(it, l) == asked(lr, l) for l in labels)

        def ibex_x(sh):
            for a in ((sh.get("responses") or {}).get("t_ibex") or {}).get("answers", []):
                if a.get("word") == "ibex":
                    return a.get("x_after")
            return None
        out["ibex_x_after_answer"] = {"intact": ibex_x(it), "lesion_reset": ibex_x(lr)}
    # asks before the first unasked toad probe (reported)
    n = 0
    for l in TOAD:
        if asked(it, l):
            n += 1
        else:
            break
    out["asks_before_stop"] = n
    # Stage-2 C3 clauses at organ level (score_ask's own logic, organ-level recall)
    a_gap = [bool(asked(it, l)) for l in GAPS]
    a_moat = [bool(asked(it, l)) for l in MOATS]
    rg, rm = sum(a_gap) / 3.0, sum(a_moat) / 3.0
    out["stage2_c3"] = {"ask_rate_gap": rg, "ask_rate_moat": rm,
                        "discriminates": rg >= ASK_GAP_FLOOR and rg >= ASK_RATIO_BAR * rm,
                        "lp_learnable": asked(it, "lp_ibex_1") is True and ibex2(it),
                        "lp_unlearnable": asked(it, "u_toad_1") is True and stop4(it)}
    return out


BAR_KEYS = ("B1_first_novel_asked", "B1b_every_first_gap_asked", "B2_stop_by_4th", "B3_answered_not_reasked",
            "B4_moats_silent", "B5_lesion_asks_4th", "B5_differs", "B6_perm_order", "B7_perm_exposure", "B8_null",
            "B9_drive_lesion")


def aggregate(per_seed):
    """Every bar must hold on every battery seed. B5's attribution: the lesion's toad-ask excess over intact vs the
    intact_b null's excess (tools.lab.attributable_to)."""
    from tools.lab import attributable_to
    seeds = sorted(per_seed)
    bars = {}
    for k in BAR_KEYS:
        vals = [per_seed[s].get(k) for s in seeds]
        bars[k] = {"n_true": sum(1 for v in vals if v is True), "n": len(seeds),
                   "holds": len(seeds) == len(BATTERY_SEEDS) and all(v is True for v in vals)}
    t = sum((per_seed[s].get("toad_asks") or {}).get("lesion_habituation", 0) -
            (per_seed[s].get("toad_asks") or {}).get("intact", 0) for s in seeds)
    c = sum((per_seed[s].get("toad_asks") or {}).get("intact_b", 0) -
            (per_seed[s].get("toad_asks") or {}).get("intact", 0) for s in seeds)
    att = attributable_to("lesion_habituation toad asks", t, c)
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


def run_battery(seeds, *, w_topic_ask, tau_hab_ms=TAU_HAB_MS, arms=tuple(ARMS)):
    shards, per_seed = {}, {}
    for s in seeds:
        shards[s] = {}
        for a in arms:
            sh = run_arm(s, a, w_topic_ask=w_topic_ask, tau_hab_ms=tau_hab_ms)
            shards[s][a] = sh
            print("  seed %d %-18s asked=%s  thr=%.1f  wall=%.1fs" % (
                s, a, "".join("1" if (sh["responses"][l].get("asked")) else "." for l in sh["turn_order"]),
                sh["threshold"], sh["wall_s"]), flush=True)
        per_seed[s] = score_seed(shards[s])
    return shards, per_seed


# ── design smoke (PLAN section 4.2) ─────────────────────────────────────────────────────────────────────────────────
def design_smoke(seed=DESIGN_SEED, tau_hab_ms=TAU_HAB_MS):
    """Pick W_TOPIC_ASK by the pre-registered rule and record the design reads: fresh want_novel / want_familiar per
    W_GRID value, the afferent volley and the slice's x after one presentation, the SNc neutral vs LP=1 rate, and any
    idle topic spikes. Does NOT run the battery turns."""
    rows = []
    for w in W_GRID:
        org = CuriosityLPOrgan(seed, w_topic_ask=w, tau_hab_ms=tau_hab_ms)
        org.ensure_built()
        r1 = org.present("__design_a__", NOVEL_SIGNAL)
        org.idle(INTER_TURN_MS)
        r2 = org.present("__design_a__", NOVEL_SIGNAL)
        da1 = org._snc_rate(1.0)
        rows.append({"w": w, "want_novel_hz": org.calib["want_novel_hz"],
                     "want_familiar_hz": org.calib["want_familiar_hz"], "threshold_hz": org.threshold,
                     "present1": r1, "present2_after_idle": r2, "snc_neutral_hz": org.snc_neutral_hz,
                     "snc_lp1_hz": da1, "idle_topic_spikes": org.idle_topic_spikes})
        print("  w=%5.1f want_novel=%.1f fam=%.1f thr=%.1f | p1 want=%.1f x %.3f->%.3f volley=%d | p2 want=%.1f "
              "x->%.3f | snc %.1f -> %.1f" % (w, org.calib["want_novel_hz"], org.calib["want_familiar_hz"],
                                               org.threshold, r1["want_hz"], r1["x_pre"], r1["x_post"],
                                               r1["afferent_spikes"], r2["want_hz"], r2["x_post"],
                                               org.snc_neutral_hz, da1), flush=True)
    best = min(rows, key=lambda r: (abs(r["want_novel_hz"] - TARGET_WANT_NOVEL_HZ), r["w"]))
    return {"seed": int(seed), "rule": "W_GRID value whose fresh want_novel is closest to %.1f Hz (ties -> smaller w)"
            % TARGET_WANT_NOVEL_HZ, "rows": rows, "chosen_w": best["w"]}


# ── selftest (no bridge) ───────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    checks = {}
    checks["parse_teach"] = parse_teach("the ibex chases the marmot") == ("ibex", "chase", "marmot")
    checks["parse_probe"] = parse_probe("what does the toad chase") == ("toad", "chase")
    checks["extract_topic"] = extract_topic("what does the toad chase") == "toad"
    w = BatteryWorld()
    checks["builtin_recall"] = w.recall("dog", "chase") == "cat" and w.recall("yak", "chase") is None
    w.store(parse_teach("the ibex chases the marmot"))
    checks["taught_recall"] = w.recall("ibex", "chase") == "marmot"
    for s in BATTERY_SEEDS:
        t = perm_order_turns(s)
        lab = [x[0] for x in t]
        ok = (sorted(lab) == sorted(x[0] for x in ASK_TURNS) and lab.index("lp_ibex_1") < lab.index("t_ibex")
              < lab.index("lp_ibex_2") and [l for l in lab if l in TOAD] == list(TOAD))
        checks["perm_order_seed%d" % s] = ok
    checks["perm_order_differs"] = len({tuple(x[0] for x in perm_order_turns(s)) for s in BATTERY_SEEDS}) > 1

    def fake(asked_labels, turns=ASK_TURNS):
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
    perm_good = fake({"m_yak", "m_moose", "m_goat", "lp_ibex_1", "u_toad_1"}, ASK_PERMUTED_TURNS)
    none = fake(set())
    sc = score_seed({"intact": good, "intact_b": good, "lesion_habituation": no_stop, "lesion_drive": none,
                     "perm_order": good, "perm_exposure": perm_good})
    checks["scorer_intact_passes"] = all(sc[k] for k in BAR_KEYS)
    sc2 = score_seed({"intact": no_stop, "intact_b": no_stop, "lesion_habituation": no_stop, "lesion_drive": none,
                      "perm_order": no_stop, "perm_exposure": perm_good})
    checks["scorer_no_stop_fails_B2_B5"] = (not sc2["B2_stop_by_4th"]) and (not sc2["B5_differs"])
    agg = aggregate({s: sc for s in BATTERY_SEEDS})
    checks["aggregate_go_on_synthetic"] = agg["verdict"] == "GO"
    agg2 = aggregate({s: sc for s in BATTERY_SEEDS[:7]})
    checks["aggregate_needs_all_8_seeds"] = agg2["verdict"] == "NO-GO"
    for k, v in checks.items():
        print("  %s %s" % ("PASS" if v else "FAIL", k))
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--design", action="store_true", help="the design smoke on DESIGN_SEED (picks W_TOPIC_ASK)")
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in BATTERY_SEEDS))
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tau-hab-ms", type=float, default=TAU_HAB_MS,
                    help="habituation recovery tau_d (substrate ms); the pre-registered value is TAU_HAB_MS")
    ap.add_argument("--w-topic-ask", type=float, default=None,
                    help="override the frozen topic->ASK weight (default: W_TOPIC_ASK); recorded in the artifact")
    args = ap.parse_args()
    if args.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    from sim.backend import get_backend
    _, be = get_backend()
    if args.design:
        res = design_smoke(DESIGN_SEED, tau_hab_ms=args.tau_hab_ms)
        res["tau_hab_ms"] = float(args.tau_hab_ms)
        res["backend"] = str(be)
        print("chosen W_TOPIC_ASK =", res["chosen_w"])
    elif args.battery:
        w_topic_ask = W_TOPIC_ASK if args.w_topic_ask is None else float(args.w_topic_ask)
        if w_topic_ask is None:
            sys.exit("W_TOPIC_ASK is not frozen: run --design and commit its artifact first (PLAN section 4.2)")
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if any(s not in range(1, 16) for s in seeds):
            sys.exit("dev seeds 1-15 only")
        arms = [a for a in args.arms.split(",") if a.strip()]
        shards, per_seed = run_battery(seeds, w_topic_ask=w_topic_ask, tau_hab_ms=args.tau_hab_ms, arms=arms)
        res = {"runner": "research/runners/_curiosity_lp_trace_derisk.py", "plan": "docs/drafts/curiosity-lp/PLAN.md",
               "backend": str(be), "w_topic_ask": w_topic_ask, "tau_hab_ms": float(args.tau_hab_ms),
               "preregistered": {"w_topic_ask": W_TOPIC_ASK, "tau_hab_ms": TAU_HAB_MS}, "bars": BARS,
               "predictions_informational": PREDICTIONS_INFORMATIONAL, "per_seed": per_seed,
               "shards": shards}
        if (sorted(seeds) == sorted(BATTERY_SEEDS) and set(arms) >= set(COUNTED_ARMS)
                and w_topic_ask == W_TOPIC_ASK and float(args.tau_hab_ms) == TAU_HAB_MS):
            res["aggregate"] = aggregate(per_seed)
            print("VERDICT:", res["aggregate"]["verdict"])
            for k, v in res["aggregate"]["bars"].items():
                print("  %-28s %s" % (k, v))
    else:
        ap.error("one of --selftest / --design / --battery")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
