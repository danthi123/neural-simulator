"""Emotion-category channels of the affect ladder, held on spiking neurons and named by a spiking commit (Stage-2 C4).

Packet `s2-c4-build` (Claude, 2026-10-06; docs/drafts/stage2-c4-build/PLAN.md). Target: the Stage-2 exit criterion C4
`emo` as amended by AMENDMENT-A3 (docs/drafts/stage2-exit/AMENDMENT-A3.md): on a `how do you feel` turn the reply NAMES
the discrete emotion category the brain's affect ladder holds, and the ladder lesion (`BRAIN_AFFECT_LESION=1`, the
`affect_out` gate) changes the named category on every feel turn. Today the production feel read-out
(`affect_production_organ.feel_readout`) prints a sign, a number and an arousal band, and the production ladder is a
signed scalar: fear and anger read identically (prereg 4.4). DEFAULT ON since 2026-10-15 behind `BRAIN_EMOTION_CATEGORY` (=0 is the escape; the switch: `webapp/server.py`
`_emotion_category_on`).

WHAT THIS ORGAN IS (one `SimulationBridge`, Izhikevich neurons, explicit wiring, no region framework; the same
construction as `word_role_parser_organ.py`, whose commit stage it reuses in form and parameters):

  * `word` inputs: one assembly (`NW` neurons) per word the brain heard during development (the labeling episodes
    below). A word it never heard has no assembly and drives nothing.
  * four APPRAISAL ensembles A_k, k in (joy, fear, anger, sadness) (`R` neurons each), with lateral inhibition: A_k
    drives its own inhibitory pool, which inhibits the other three ensembles (one channel wins a mixed message).
    Panksepp's primary-process systems (SEEKING/PLAY, FEAR, RAGE, PANIC/GRIEF) as separate populations: separate
    valence-specific and behaviour-specific populations in amygdala and hypothalamus (Namburi et al. 2015 Nature
    520:675; Lin et al. 2011 Nature 470:221 VMHvl aggression; Silva et al. 2013 Cell 155:1596 VMHdm predator fear).
  * INNATE trigger synapses (fixed, wired): a few unconditioned-stimulus words per channel (predator words -> FEAR,
    goal-blocking/violation words -> RAGE, loss/separation words -> PANIC/GRIEF, reward/play words -> SEEKING) drive
    their channel from birth. Declared: the innate US map is hand-wired, the analogue of the Warriner seed norms.
  * LEARNED word -> channel synapses (plastic, the engine's Hebbian co-firing rule): during development a teacher
    narrates labeling episodes ("the monster came, you are scared"): the trigger word fires its channel through the
    innate synapse while the emotion word is heard, so the emotion word's synapses onto that channel potentiate
    (evaluative conditioning, Staats & Staats 1957; emotion words learned by caregiver labeling of the child's own
    state, Widen & Russell 2008; Hoemann, Xu & Barrett 2019 Dev Psychol 55:1830). No teacher current enters the
    brain: the channel fires from the innate trigger. Frozen after development.
  * the LADDER: per channel, `NL` rungs (`NR` neurons, dense AMPA self-recurrence) driven by A_k at staggered weights
    (rung j at W_AL x (1 - STAG x j)): a stronger appraisal latches more rungs (Koulakov et al. 2002 robust
    integrator; Goldman et al. 2003 staircase; the production valence ladder's own motif). The rungs SELF-SUSTAIN
    after the message ends: the held affect state between turns. A new appraisal QUENCHES the other channels' rungs
    (A_k -> inhibitory pool QI_k -> rungs of every other channel): the active-clear motif of
    research/findings/2026-08-01-affect-BRAIN-BASED-active-clear-spiking-quench-fs-gate-6seed-GO-operating-region-mapped.md.
  * READOUT O_k (`NO` neurons) from every rung of channel k, through the transmission gate `affect_out` (the same
    name and semantics as the production ladder's readout gate; `BRAIN_AFFECT_LESION=1` clamps it to 0).
  * the SPIKING COMMIT (docs/drafts/spiking-commit/PLAN.md, parameters of COMMIT_GRID row 0 in
    word_role_parser_organ.py): O_k -> commit gate G_k (self-recurrent, each gate's inhibitory pool GI_k inhibits the
    other gates: at most one bursts) -> latch S_k (bistable). The category is WHICH latch is on after a zero-input hold,
    read by a threshold-invariant transducer (re-read at x0.5 and x2 on the same spikes). Exactly one latch -> that
    category; none -> no category; two or more -> no category (a co-commit, counted). No count is compared with
    another channel's count.

PER-SESSION STATE. The bridge's dynamical state after each affect-bearing turn is kept per session key (the held
ladder); `observe()` restores it, presents the message's known words, runs a zero-input hold (the appraisal ensembles
fall silent, latched rungs persist) and stores it again. `read()` restores it, opens the read window (O -> G) for
`READ_WINDOW` steps, closes it, holds, and reads the commit latches; the read never writes the session state back, so
a read leaves the held state unchanged (like the production ladder's snapshot-isolated read).

LESIONS (each a transmission gate on this bridge; dev arms):
  `affect_out`         rungs -> readout      BRAIN_AFFECT_LESION=1 (the gate's registered C4 lesion)
  `emo_word`           word -> appraisal     BRAIN_EMOTION_CATEGORY_WORD_LESION=1 (learned AND innate word synapses)
  `emo_ladder_loop`    rung self-recurrence  BRAIN_EMOTION_CATEGORY_LATCH_LESION=1 (the ladder cannot hold)
  `emo_commit`         readout -> gate       BRAIN_EMOTION_CATEGORY_COMMIT_LESION=1
  shuffle (a build)    O_k -> G_{pi(k)}      BRAIN_EMOTION_CATEGORY_SHUFFLE=1 (pi = rotate by one channel)
  teach swap (a build) fear<->anger triggers in development   BRAIN_EMOTION_CATEGORY_TEACH_SWAP=1

BOUNDED CONDUCTANCE (`BRAIN_EMOTION_CATEGORY_BOUNDED`, default OFF; research/findings/2026-10-26-emotion-organ-bounded-
conductance-derisk.md). The explicit-Euler conductance term (dv = g_i * (E_i - v) * dt / C) overshoots E_i whenever
g_i * dt / C exceeds ~2. The organ's heterogeneity draw gives about 4% of builds an appraisal neuron of low capacitance
(s101: C = 53.6 pF against a mean near 100); when the other channel wins, its inhibitory pool drives that neuron's g_i
to 100-136 nS, v oscillates with growing amplitude (-96, -24, -116, +18, -137 mV), and the spurious spikes recruit the
sadness quench pool that silences the joy rungs. The flag sets the engine's own opt-in guard
`cfg.enable_bounded_conductance_integration` (sim/config.py; exponential-Euler, one step can only move v toward its
reversal) at CoreSimConfig construction. Nothing else changes: no capacitance truncation, no host table. Every read
reports `bounded_flag` (the flag the organ resolved), `bounded_env` (the raw env string seen), `bounded_readback` (the
bridge's own core_config value) and `oor_count` / `oor_count_session` (distinct organ neurons whose v went below
-110 mV during the read / during the session's observes and the read; 0 expected under the fix). The monitor is a
read-only min over v after each step and changes no state.

NAMED RESIDUALS (host code that remains; declared):
  * the development curriculum (which trigger and which emotion word the teacher puts in one episode) is the teacher
    as social environment; the innate trigger map is hand-wired (a seed lexicon, like WARRINER);
  * the word-form front end (a dictionary word -> assembly index) and the stop-word strip are host;
  * the per-session state store is host bookkeeping of the bridge's own state (the process sleeps between turns);
  * the read window (when O -> G opens) is host timing: the feel question opens the read;
  * the latch read is a host level read of spike counts, registered threshold-invariant (x0.5 / x2);
  * the host renders the latched channel's fixed word (`CATEGORY_WORD`);
  * the organ is its own bridge (one-brain debt; the production valence ladder is another bridge).
NOT a phenomenal claim: the clause is a functional read of which channel latched (CLAUDE.md non-negotiable 5).

Backend: the process backend (numpy on the pool). No `sim/` edit. Building/training restores the global RNG state.
"""
from __future__ import annotations

import os
import re
import sys
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402

from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state, _reset_quiescent, _restore_state  # noqa: E402
from research.runners.word_role_parser_organ import _dense, _isolated  # noqa: E402

CATS = ("joy", "fear", "anger", "sadness")
# the word the host renders for the latched channel (each is a member of exactly one category of the Stage-2 frozen
# EMOTION_LEXICON; pinned by tests/test_emotion_category_organ.py)
CATEGORY_WORD = {"joy": "joy", "fear": "fear", "anger": "anger", "sadness": "sadness"}

# ── development: the innate trigger map and the teacher's labeling episodes ─────────────────────────────────────────
# INNATE unconditioned triggers per channel (wired, fixed). None of them is a Stage-2 situation frame word
# (win/gift/wolf/attacking/someone/cheats/alone) nor an affect word of a Stage-2 situation: pinned by the tests.
TRIGGERS: Dict[str, tuple] = {
    "joy": ("candy", "cake", "toy", "party", "treat", "puppy"),          # reward / play (SEEKING, PLAY)
    "fear": ("monster", "ghost", "snake", "spider", "storm", "dragon"),  # predator / threat (FEAR)
    "anger": ("took", "broke", "hit", "pushed", "grabbed", "teased"),    # violation / goal-blocking (RAGE)
    "sadness": ("lost", "gone", "goodbye", "died", "missed", "left"),     # loss / separation (PANIC/GRIEF)
}
# the TEACHER's emotion labels per channel (common child-directed emotion words; the teacher is the social
# environment). They overlap the Stage-2 lexicon and situations BY DESIGN: C4 measures naming the held channel with
# words learned this way, not generalisation to never-heard emotion words (declared in PLAN.md).
LABELS: Dict[str, tuple] = {
    "joy": ("happy", "glad", "joyful", "delighted", "pleased", "cheerful"),
    "fear": ("scared", "afraid", "frightened", "terrified", "fearful"),
    "anger": ("angry", "mad", "furious", "annoyed", "cross"),
    "sadness": ("sad", "unhappy", "lonely", "miserable", "cry"),
}
STOP = {"a", "an", "the", "and", "you", "are", "is", "was", "were", "be", "to", "of", "in", "on", "at", "it", "i",
        "me", "my", "your", "so", "very", "came", "come", "now", "how", "do", "feel", "what", "who", "that", "this"}


def curriculum(teach_swap: bool = False) -> List[tuple]:
    """The labeling episodes, in replay order: (trigger word, emotion word, channel). Each emotion word is narrated with
    two of its channel's triggers. `teach_swap`: the fear and anger TRIGGERS are exchanged (a dev control: the learned
    meaning of the fear words must then follow the anger trigger)."""
    trig = dict(TRIGGERS)
    if teach_swap:
        trig["fear"], trig["anger"] = TRIGGERS["anger"], TRIGGERS["fear"]
    eps = []
    for c in CATS:
        t = trig[c]
        for i, w in enumerate(LABELS[c]):
            for j in (i, i + 1):
                eps.append((t[j % len(t)], w, c))
    return eps


# ── operating points (frozen order; selection rule in docs/drafts/stage2-c4-build/PLAN.md) ───────────────────────────
_G = dict(NW=8, R=12, NI=8, NL=3, NR=20, NQ=8, NO=12,
          DRIVE_PA=1500.0, LR=0.2, W_MAX=20.0, W_INIT=0.05, EPOCHS=8, TRAIN_STEPS=40,
          W_US=20.0, W_RI=200.0, W_IR=80.0,
          W_AL=20.0, STAG=0.25, W_LL=60.0, W_AQ=60.0, W_QL=120.0, W_LO=10.0,
          EXPOSE_STEPS=60, SETTLE_STEPS=100, READ_WINDOW=60,
          # the commit stage (word_role_parser_organ COMMIT_GRID row 0, unchanged)
          NG=12, NGI=8, NS=20, W_RG=50.0, W_GG=20.0, W_GGI=60.0, W_GIG=60.0, W_GS=12.0, W_SS=60.0,
          HOLD_STEPS=100, READ_STEPS=30, READ_FRAC=0.03)
# Frozen order (written before the dev-seed 1-8 battery; the first smoke on dev seeds 1-4 found LR 0.1 / W_AL 14 too
# weak for the two-word fear situation, which set row 0): selection = the first row whose battery passes every case on
# all eight dev seeds (research/runners/emotion_category_dev.py --select). Gate seeds are never run by the dev battery.
GRID: List[dict] = [
    dict(_G),                                   # 0: reference
    dict(_G, W_LL=80.0),                        # 1: stronger rung latch
    dict(_G, W_QL=200.0),                       # 2: stronger quench
    dict(_G, W_AL=14.0),                        # 3: weaker appraisal -> ladder
    dict(_G, LR=0.1, EPOCHS=12),                # 4: slower learning, longer development
    dict(_G, W_RG=60.0),                        # 5: more sensitive commit gate
]

GATE_WORD = "emo_word"
GATE_LOOP = "emo_ladder_loop"
GATE_QUENCH = "emo_quench"
GATE_OUT = "affect_out"
GATE_COMMIT = "emo_commit"
GATE_EXCL = "emo_commit_excl"
GATE_SLOT = "emo_slot_loop"
GATE_WORD_PLASTIC = "emo_word_plastic"
GATE_FIXED = "emo_fixed"

BOUNDED_ENV = "BRAIN_EMOTION_CATEGORY_BOUNDED"   # default OFF: the organ builds with the engine's unbounded Euler step
OOR_MV = -110.0                                    # an organ neuron below this left the physiological range (E_i = -75)


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


DEFAULT_OP = 0     # the GRID row the dev battery selected (docs/drafts/stage2-c4-build/PLAN.md section 5)


def operating_point() -> int:
    """`BRAIN_EMOTION_CATEGORY_OP` (an int GRID row) or DEFAULT_OP. The on/off switch lives in webapp/server.py
    (`_EMOTION_CATEGORY_DEFAULT_ON`, `_emotion_category_on`), so flag off never imports this module."""
    try:
        v = os.environ.get("BRAIN_EMOTION_CATEGORY_OP")
        return int(v) if v is not None and v.strip() else DEFAULT_OP
    except ValueError:
        return DEFAULT_OP


def bounded_requested() -> bool:
    """`BRAIN_EMOTION_CATEGORY_BOUNDED` (1/true/yes/on): build the organ with the engine's bounded conductance
    integration. Unset or anything else: the unbounded step, byte-identical to the organ before this flag existed."""
    return _truthy(BOUNDED_ENV)


_WORD_RE = re.compile(r"[a-z']+")


def content_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in STOP]


class EmotionCategoryOrgan:
    """Development (`train`), per-session held state (`observe`), and the committed category (`read`)."""

    def __init__(self, seed: int = 1, params: Optional[dict] = None, shuffle: bool = False, teach_swap: bool = False,
                 bounded: Optional[bool] = None):
        self.p = dict(params if params is not None else GRID[0])
        self.seed = int(seed)
        # None = follow BRAIN_EMOTION_CATEGORY_BOUNDED (read once, here, so a cached organ keeps the value it was built with)
        self.bounded_env = os.environ.get(BOUNDED_ENV)
        self.bounded = bounded_requested() if bounded is None else bool(bounded)
        self.shuffle = bool(shuffle)
        self.teach_swap = bool(teach_swap)
        self.episodes = curriculum(teach_swap)
        vocab: List[str] = []
        for t, w, _c in self.episodes:
            for x in (t, w):
                if x not in vocab:
                    vocab.append(x)
        for c in CATS:                       # every innate trigger has an assembly even if no episode used it
            for t in TRIGGERS[c]:
                if t not in vocab:
                    vocab.append(t)
        self.vocab = vocab
        self._bridge = self._xp = self._snap = None
        self._lock = threading.RLock()
        self._sessions: "OrderedDict[object, dict]" = OrderedDict()
        self._oor_sessions: "OrderedDict[object, set]" = OrderedDict()   # distinct out-of-range neurons per session
        self.trained = False

    # ── build ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _build(self):
        p = self.p
        xp, _ = get_backend()
        NW, R, NI, NL, NR, NQ, NO = (int(p[k]) for k in ("NW", "R", "NI", "NL", "NR", "NQ", "NO"))
        NG, NGI, NS = int(p["NG"]), int(p["NGI"]), int(p["NS"])
        K = len(CATS)
        o = 0

        def block(n):
            nonlocal o
            a = np.arange(o, o + n, dtype=np.int64)
            o += n
            return a

        self.word_idx = {w: block(NW) for w in self.vocab}
        self.app_idx = {c: block(R) for c in CATS}
        self.app_inh = {c: block(NI) for c in CATS}
        self.rung_idx = {c: [block(NR) for _ in range(NL)] for c in CATS}
        self.quench_idx = {c: block(NQ) for c in CATS}
        self.out_idx = {c: block(NO) for c in CATS}
        self.gate_idx = {c: block(NG) for c in CATS}
        self.gate_inh = {c: block(NGI) for c in CATS}
        self.slot_idx = {c: block(NS) for c in CATS}
        n = o
        self.all_rungs = np.concatenate([np.concatenate(self.rung_idx[c]) for c in CATS])
        self.rest_at_read = np.concatenate([np.concatenate([self.out_idx[c], self.gate_idx[c], self.gate_inh[c],
                                                            self.slot_idx[c]]) for c in CATS])

        cfg = CoreSimConfig()
        cfg.num_neurons = int(n)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_reward_modulation", "enable_homeostasis", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_watts_strogatz", "enable_ou_process"):
            setattr(cfg, f, False)
        cfg.enable_hebbian_learning = True
        cfg.hebbian_learning_rate = float(p["LR"])
        cfg.hebbian_max_weight = float(p["W_MAX"])
        cfg.enable_parameter_heterogeneity = True
        if self.bounded:        # the engine's own opt-in guard (sim/config.py); default OFF leaves cfg untouched
            cfg.enable_bounded_conductance_integration = True
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                             gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b._initialize_simulation_data(called_from_playback_init=False)

        trig_of = {}
        for c in CATS:
            for t in TRIGGERS[c]:
                trig_of[t] = c
        if self.teach_swap:   # the innate map itself is unchanged; only the curriculum pairs swap (see curriculum())
            pass
        plan = {}
        all_app = np.concatenate([self.app_idx[c] for c in CATS])
        learn_words = [w for w in self.vocab if w not in trig_of]
        if learn_words:
            lw = np.concatenate([self.word_idx[w] for w in learn_words])
            plan["word_app"] = _dense(lw, all_app, p["W_INIT"], plastic=True, conn_type="E_TO_E",
                                      plasticity_gate=GATE_WORD_PLASTIC, transmission_gate=GATE_WORD)
        for t, c in trig_of.items():
            plan["us_" + t] = _dense(self.word_idx[t], self.app_idx[c], p["W_US"], plastic=False, conn_type="E_TO_E",
                                     plasticity_gate=GATE_FIXED, transmission_gate=GATE_WORD)
        inh = []
        for k, c in enumerate(CATS):
            others_app = np.concatenate([self.app_idx[x] for x in CATS if x != c])
            plan["ai_" + c] = _dense(self.app_idx[c], self.app_inh[c], p["W_RI"], plastic=False, conn_type="E_TO_I",
                                     plasticity_gate=GATE_FIXED)
            plan["ia_" + c] = _dense(self.app_inh[c], others_app, p["W_IR"], plastic=False, conn_type="I_TO_E",
                                     plasticity_gate=GATE_FIXED)
            inh.append(self.app_inh[c])
            for j, rung in enumerate(self.rung_idx[c]):
                w_al = float(p["W_AL"]) * max(0.05, 1.0 - float(p["STAG"]) * j)
                plan["al_%s_%d" % (c, j)] = _dense(self.app_idx[c], rung, w_al, plastic=False, conn_type="E_TO_E",
                                                    plasticity_gate=GATE_FIXED)
                plan["ll_%s_%d" % (c, j)] = _dense(rung, rung, p["W_LL"], plastic=False, conn_type="E_TO_E",
                                                    plasticity_gate=GATE_FIXED, transmission_gate=GATE_LOOP)
                plan["lo_%s_%d" % (c, j)] = _dense(rung, self.out_idx[c], p["W_LO"], plastic=False,
                                                    conn_type="E_TO_E", plasticity_gate=GATE_FIXED,
                                                    transmission_gate=GATE_OUT)
            other_rungs = np.concatenate([np.concatenate(self.rung_idx[x]) for x in CATS if x != c])
            plan["aq_" + c] = _dense(self.app_idx[c], self.quench_idx[c], p["W_AQ"], plastic=False,
                                     conn_type="E_TO_I", plasticity_gate=GATE_FIXED)
            plan["ql_" + c] = _dense(self.quench_idx[c], other_rungs, p["W_QL"], plastic=False, conn_type="I_TO_E",
                                     plasticity_gate=GATE_FIXED, transmission_gate=GATE_QUENCH)
            inh.append(self.quench_idx[c])
            # the commit stage: O_k -> G_{pi(k)} (pi = identity, or rotate-by-one under the shuffle build)
            tgt = CATS[(k + 1) % len(CATS)] if self.shuffle else c
            g, gi, s = self.gate_idx[c], self.gate_inh[c], self.slot_idx[c]
            other_g = np.concatenate([self.gate_idx[x] for x in CATS if x != c])
            plan["og_" + c] = _dense(self.out_idx[c], self.gate_idx[tgt], p["W_RG"], plastic=False,
                                     conn_type="E_TO_E", plasticity_gate=GATE_FIXED, transmission_gate=GATE_COMMIT)
            plan["gg_" + c] = _dense(g, g, p["W_GG"], plastic=False, conn_type="E_TO_E", plasticity_gate=GATE_FIXED)
            plan["ggi_" + c] = _dense(g, gi, p["W_GGI"], plastic=False, conn_type="E_TO_I", plasticity_gate=GATE_FIXED)
            plan["gig_" + c] = _dense(gi, other_g, p["W_GIG"], plastic=False, conn_type="I_TO_E",
                                      plasticity_gate=GATE_FIXED, transmission_gate=GATE_EXCL)
            plan["gs_" + c] = _dense(g, s, p["W_GS"], plastic=False, conn_type="E_TO_E", plasticity_gate=GATE_FIXED)
            plan["ss_" + c] = _dense(s, s, p["W_SS"], plastic=False, conn_type="E_TO_E", plasticity_gate=GATE_FIXED,
                                     transmission_gate=GATE_SLOT)
            inh.append(gi)
        b.inject_explicit_wiring(plan, output_inhibitory_indices=np.concatenate(inh))
        b.set_plasticity_gate(GATE_FIXED, 0.0)
        b.set_plasticity_gate(GATE_WORD_PLASTIC, 1.0)
        for gname in (GATE_WORD, GATE_LOOP, GATE_QUENCH, GATE_OUT, GATE_EXCL, GATE_SLOT):
            b.set_transmission_gate(gname, 1.0)
        b.set_transmission_gate(GATE_COMMIT, 0.0)       # the read window is closed outside a read
        b.cp_external_input_current[:] = 0.0
        for _ in range(60):
            b._run_one_simulation_step()
        self._bridge, self._xp, self._n = b, xp, int(n)
        self._snap = _snapshot_state(b, xp)

    def _ensure(self):
        if self._bridge is None:
            _isolated(self._build)

    # ── development ────────────────────────────────────────────────────────────────────────────────────────────
    def train(self):
        """Replay the labeling episodes (trigger + emotion word together, no teacher current). Idempotent."""
        with self._lock:
            if self.trained:
                return self
            _isolated(self._train)
            self.trained = True
        return self

    def _train(self):
        self._ensure()
        p, b, xp = self.p, self._bridge, self._xp
        b.set_plasticity_gate(GATE_WORD_PLASTIC, 1.0)
        for _ in range(int(p["EPOCHS"])):
            for t, w, _c in self.episodes:
                _reset_quiescent(b, xp, self._snap)
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[xp.asarray(self.word_idx[t])] = xp.float32(p["DRIVE_PA"])
                cur[xp.asarray(self.word_idx[w])] = xp.float32(p["DRIVE_PA"])
                b.cp_external_input_current[:] = cur
                for _s in range(int(p["TRAIN_STEPS"])):
                    b._run_one_simulation_step()
                b.cp_external_input_current[:] = 0.0
        b.set_plasticity_gate(GATE_WORD_PLASTIC, 0.0)    # frozen: a conversation never rewrites these weights
        _reset_quiescent(b, xp, self._snap)
        self.weight_summary = self._weights()

    def _weights(self) -> dict:
        """Mean learned word -> channel weight, taught channel vs the others (a diagnostic host read)."""
        try:
            coo = self._bridge.cp_connections.tocoo()
            row, col, dat = to_host(coo.row), to_host(coo.col), to_host(coo.data)
            taught = {w: c for _t, w, c in self.episodes}
            tw, uw = [], []
            for w, c in taught.items():
                for x in CATS:
                    m = np.isin(row, self.word_idx[w]) & np.isin(col, self.app_idx[x])
                    if m.any():
                        (tw if x == c else uw).append(float(dat[m].mean()))
            return {"word_taught": round(float(np.mean(tw)), 3) if tw else 0.0,
                    "word_untaught": round(float(np.mean(uw)), 3) if uw else 0.0}
        except Exception:   # noqa: BLE001 -- diagnostic only
            return {}

    # ── sessions ───────────────────────────────────────────────────────────────────────────────────────────────
    def _state_of(self, key):
        st = self._sessions.get(key)
        if st is not None:
            self._sessions.move_to_end(key)
        return st

    def _save(self, key):
        b = self._bridge
        self._sessions[key] = {name: getattr(b, name).copy() for name in self._snap}
        self._sessions.move_to_end(key)
        while len(self._sessions) > 64:
            self._sessions.popitem(last=False)

    def forget(self, key):
        with self._lock:
            self._sessions.pop(key, None)
            self._oor_sessions.pop(key, None)

    def _restore(self, key):
        b, xp = self._bridge, self._xp
        st = self._state_of(key)
        _reset_quiescent(b, xp, self._snap)
        if st is not None:
            _restore_state(b, st)
        b.cp_external_input_current[:] = 0.0

    def _step(self, oor: set) -> None:
        """One simulation step, then a read-only census: every organ neuron now below OOR_MV joins `oor`. The census
        reads v after the step and changes no state, so it cannot move a spike; an error in it is swallowed (an
        instrument failure must not crash a turn)."""
        b = self._bridge
        b._run_one_simulation_step()
        try:
            v = to_host(b.cp_membrane_potential_v)
            if float(v.min()) < OOR_MV:
                oor.update(int(i) for i in np.flatnonzero(v < OOR_MV))
        except Exception:   # noqa: BLE001 -- diagnostic only
            pass

    def _counts(self, idx_map, xp):
        f = self._bridge.cp_firing_states
        return {c: int(to_host(f[xp.asarray(ix)].astype(xp.float64).sum())) for c, ix in idx_map.items()}

    def observe(self, key, text: str, word_lesion: bool = False, latch_lesion: bool = False,
                drive_channels: Optional[Dict[str, float]] = None, lesion: bool = False) -> dict:
        """One heard message: restore this session's held state, present the message's known words for EXPOSE_STEPS,
        then SETTLE_STEPS of zero input, and store the state. A message with no known word (and no `drive_channels`)
        leaves the state untouched. `drive_channels` (dev probe only): drive appraisal ensembles directly (pA)
        instead of words -- the ladder-only probe."""
        self.train()
        words = [w for w in content_words(text) if w in self.word_idx]
        if not words and not drive_channels:
            return {"acted": False, "words": []}
        with self._lock:
            return _isolated(lambda: self._observe(key, words, word_lesion, latch_lesion, drive_channels, lesion))

    def _observe(self, key, words, word_lesion, latch_lesion, drive_channels, lesion=False):
        b, xp, p = self._bridge, self._xp, self.p
        self._restore(key)
        b.set_transmission_gate(GATE_OUT, 0.0 if lesion else 1.0)
        b.set_transmission_gate(GATE_WORD, 0.0 if word_lesion else 1.0)
        b.set_transmission_gate(GATE_LOOP, 0.0 if latch_lesion else 1.0)
        try:
            cur = xp.zeros(self._n, dtype=xp.float32)
            for w in words:
                cur[xp.asarray(self.word_idx[w])] = xp.float32(p["DRIVE_PA"])
            for c, pa in (drive_channels or {}).items():
                cur[xp.asarray(self.app_idx[c])] = xp.float32(pa)
            b.cp_external_input_current[:] = cur
            app = {c: 0 for c in CATS}
            oor = set(self._oor_sessions.get(key, ()))
            for _ in range(int(p["EXPOSE_STEPS"])):
                self._step(oor)
                for c, v in self._counts(self.app_idx, xp).items():
                    app[c] += v
            b.cp_external_input_current[:] = 0.0
            for _ in range(int(p["SETTLE_STEPS"])):
                self._step(oor)
            rungs = {c: [int(to_host(b.cp_firing_states[xp.asarray(r)].astype(xp.float64).sum())) for r in
                         self.rung_idx[c]] for c in CATS}
            self._save(key)
            self._oor_sessions[key] = oor
            self._oor_sessions.move_to_end(key)
            while len(self._oor_sessions) > 64:
                self._oor_sessions.popitem(last=False)
        finally:
            b.cp_external_input_current[:] = 0.0
            b.set_transmission_gate(GATE_OUT, 1.0)
            b.set_transmission_gate(GATE_WORD, 1.0)
            b.set_transmission_gate(GATE_LOOP, 1.0)
        return {"acted": True, "words": list(words), "appraisal_counts": app, "rungs_active_last_step": rungs,
                "oor_count_session": len(oor)}

    def read(self, key, lesion: bool = False, commit_lesion: bool = False, latch_lesion: bool = False) -> dict:
        """The committed category of this session's held ladder. `lesion` = the `affect_out` lesion."""
        self.train()
        with self._lock:
            rec = _isolated(lambda: self._read(key, lesion, commit_lesion, latch_lesion))
        return read_category(rec, self.p)

    def _read(self, key, lesion, commit_lesion, latch_lesion):
        b, xp, p = self._bridge, self._xp, self.p
        self._restore(key)
        # the readout and the commit populations start each read from rest: the held state is the ladder's rungs only
        keep_out = xp.asarray(self.rest_at_read)
        for name, arr in self._snap.items():
            cur = getattr(b, name, None)
            if cur is not None and cur.ndim >= 1 and cur.shape[0] == self._n:
                cur[keep_out] = arr[keep_out]
        b.set_transmission_gate(GATE_OUT, 0.0 if lesion else 1.0)
        b.set_transmission_gate(GATE_LOOP, 0.0 if latch_lesion else 1.0)
        out_c = {c: 0 for c in CATS}
        gate_c = {c: 0 for c in CATS}
        slot_c = {c: 0 for c in CATS}
        oor = set()
        try:
            b.set_transmission_gate(GATE_COMMIT, 0.0 if commit_lesion else 1.0)    # the read window opens
            for _ in range(int(p["READ_WINDOW"])):
                self._step(oor)
                for c, v in self._counts(self.out_idx, xp).items():
                    out_c[c] += v
                for c, v in self._counts(self.gate_idx, xp).items():
                    gate_c[c] += v
            b.set_transmission_gate(GATE_COMMIT, 0.0)                               # ... and closes
            hold, rd = int(p["HOLD_STEPS"]), int(p["READ_STEPS"])
            for t in range(hold):
                self._step(oor)
                if t >= hold - rd:
                    for c, v in self._counts(self.slot_idx, xp).items():
                        slot_c[c] += v
        finally:
            b.set_transmission_gate(GATE_COMMIT, 0.0)
            b.set_transmission_gate(GATE_OUT, 1.0)
            b.set_transmission_gate(GATE_LOOP, 1.0)
            b.cp_external_input_current[:] = 0.0
        return {"readout_counts": out_c, "gate_counts": gate_c, "slot_counts": slot_c,
                "lesioned": bool(lesion), "commit_lesioned": bool(commit_lesion), "latch_lesioned": bool(latch_lesion),
                # ENGAGEMENT: what this read actually ran with (see BOUNDED CONDUCTANCE in the module docstring)
                "bounded_flag": bool(self.bounded), "bounded_env": self.bounded_env,
                "bounded_readback": bool(getattr(b.core_config, "enable_bounded_conductance_integration", False)),
                "oor_count": len(oor), "oor_count_session": len(oor | self._oor_sessions.get(key, set()))}


def read_category(rec: dict, p: dict) -> dict:
    """The transducer read on recorded latch counts (pure; the bar-sanity spec calls it on synthetic records). A latch
    is ON iff its hold-window count reaches READ_FRAC x scale x NS x READ_STEPS. Exactly one ON -> that category; none
    -> None; several -> None (a co-commit). Re-read at x0.5 and x2; `read_invariant` says whether the outcome held."""
    thr0 = float(p["READ_FRAC"]) * int(p["NS"]) * int(p["READ_STEPS"])

    def at(scale):
        on = [c for c in CATS if rec["slot_counts"][c] >= thr0 * scale]
        return (on[0] if len(on) == 1 else None), on

    cat, on = at(1.0)
    variants = {tag: at(sc)[0] for tag, sc in (("x0.5", 0.5), ("x2", 2.0))}
    out = dict(rec)
    out.update({"category": cat, "latched": on, "co_commit": len(on) > 1, "threshold": thr0,
                "read_variants": variants, "read_invariant": all(v == cat for v in variants.values())})
    return out


def feel_clause(category: Optional[str]) -> str:
    """The sentence the feel read-out appends. Names exactly one category word when a channel latched; otherwise says
    none did (and names no category word). A functional read, never a claim of felt experience."""
    if category in CATEGORY_WORD:
        return " The emotion channel that latched on my affect ladder is %s." % CATEGORY_WORD[category]
    return " No single emotion channel latched on my affect ladder."


# ── process-shared organs and the production hooks (only reached when BRAIN_EMOTION_CATEGORY is on) ─────────────────
_ORGANS: Dict[tuple, EmotionCategoryOrgan] = {}
_ORGANS_LOCK = threading.Lock()


def _seed() -> int:
    try:
        return int(os.environ.get("BRAIN_CHAT_SEED", "42") or 42)
    except ValueError:
        return 42


def get_organ(seed: Optional[int] = None) -> EmotionCategoryOrgan:
    seed = _seed() if seed is None else int(seed)
    op = operating_point()
    shuffle, swap = _truthy("BRAIN_EMOTION_CATEGORY_SHUFFLE"), _truthy("BRAIN_EMOTION_CATEGORY_TEACH_SWAP")
    bounded = bounded_requested()
    k = (seed, op, shuffle, swap, bounded)
    with _ORGANS_LOCK:
        org = _ORGANS.get(k)
        if org is None:
            org = EmotionCategoryOrgan(seed=seed, params=GRID[op], shuffle=shuffle, teach_swap=swap, bounded=bounded)
            _ORGANS[k] = org
    return org


def observe_turn(key, text: str, affect_lesion: bool = False) -> dict:
    """Production hook (every affect-on turn): fold the message into this session's held ladder."""
    return get_organ().observe(key, text, word_lesion=_truthy("BRAIN_EMOTION_CATEGORY_WORD_LESION"),
                               latch_lesion=_truthy("BRAIN_EMOTION_CATEGORY_LATCH_LESION"),
                               lesion=bool(affect_lesion))


def read_turn(key, affect_lesion: bool) -> dict:
    """Production hook (a feel query): the committed category, with every trace field the dev scorer needs."""
    org = get_organ()
    r = org.read(key, lesion=bool(affect_lesion), commit_lesion=_truthy("BRAIN_EMOTION_CATEGORY_COMMIT_LESION"),
                 latch_lesion=_truthy("BRAIN_EMOTION_CATEGORY_LATCH_LESION"))
    r.update({"route": "emotion_category", "seed": org.seed, "op": operating_point(), "shuffle": org.shuffle,
              "teach_swap": org.teach_swap})
    return r


def apply_to_feel(feel: dict, key, affect_lesion: bool) -> dict:
    """Production hook (the feel branch of the handler): a copy of the feel response with the category clause appended
    to the answer and the read recorded under `emotion_category`. An organ error leaves the answer unchanged and is
    recorded (never raised), so a broken organ reads as an UNDEFINED instrument, not as a missing capability."""
    out = dict(feel)
    try:
        ec = read_turn(key, affect_lesion)
    except Exception as e:   # noqa: BLE001
        out["emotion_category"] = {"route": "emotion_category", "error": "%s: %s" % (type(e).__name__, e)}
        return out
    ec = {k: v for k, v in ec.items()}
    ec["error"] = None
    out["answer"] = str(out.get("answer", "")) + feel_clause(ec.get("category"))
    out["emotion_category"] = ec
    return out


def forget_session(key) -> None:
    for org in list(_ORGANS.values()):
        org.forget(key)
