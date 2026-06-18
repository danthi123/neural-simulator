"""Learned ATTRIBUTED-ENTITY parser on the bridge -- the brain-based comprehend piece for 'S V [adj]* N' sentences,
extending the flat-SVO conjunctive parser (`conjunctive_parser` / `BridgeParser`) with a position-from-END
conjunction factor (the adjacency-to-the-head cue). Validated GO 6/6 (2026-06-18-neural-attributed-parser-GO.md):
per-position role read-out in SPIKES at 1.000; the from-END factor is load-bearing (it disambiguates the head
noun=patient from the modifiers=attribute at the same from-start position). End-to-end with the RF composer:
2026-06-18-neural-attributed-endtoend-GO.md (0.993, 6/6).

The conjunction space is (position-from-START x position-from-END x voice); a conjunction unit per (s_bucket,
e_bucket, voice) Hebbian-learns -> its role ensemble (the v16 embodied co-firing teacher, as in BridgeParser). At
parse time, driving a word's (s, e, voice) conjunction ALONE reads its role off the bridge in spikes -- no host role
rule. Roles: {agent, action, patient, attribute, attribute2}. Scope: active 'S V adj* N' (1-2 adjectives) + flat SVO
(active/passive); passive-attributed + >=3 adjectives are bounded follow-ons (more buckets + teacher frames).

This is the named home for the parser (extracted from the de-risk runner so production code -- the agent -- imports
it cleanly). GPU for real (Hebbian training on the bridge); numpy is a tiny smoke. NO sim/ edit (reuse-by-import).
"""
from __future__ import annotations

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

ROLES = ["agent", "action", "patient", "attribute", "attribute2"]
S_CAP, E_CAP = 3, 2                       # from-start bucket 0..3, from-end bucket 0..2


def role_for(s, e):
    """The ground-truth STRUCTURAL role for a word at (from-start s, from-end e) in an active 'S V adj* N' frame:
    s=0 agent; s=1 action; e=0 patient (head noun); else s=2 attribute, s>=3 attribute2."""
    sc, ec = min(s, S_CAP), min(e, E_CAP)
    if sc == 0:
        return "agent"
    if sc == 1:
        return "action"
    if ec == 0:
        return "patient"
    return "attribute" if sc == 2 else "attribute2"


def conj_index(s, e, voice, use_end=True):
    """Flat conjunction id for (s_bucket, e_bucket, voice). use_end=False drops the from-END factor (the control)."""
    sc, ec, v = min(s, S_CAP), (min(e, E_CAP) if use_end else 0), (0 if voice in (0, "active") else 1)
    ne = (E_CAP + 1) if use_end else 1
    return sc * (ne * 2) + ec * 2 + v


class AttributedBridgeParser:
    """(from-start x from-end x voice) -> 5-role Hebbian parser on a private Izhikevich bridge. Mirrors BridgeParser
    (embodied-Hebbian co-firing teacher; firing-rate role read-out), with the from-END conjunction added."""

    def __init__(self, seed=42, R=40, n_epochs=24, train_steps=120, test_steps=80, drive=2500.0, use_end=True):
        self.R = R; self.test_steps = test_steps; self.drive = drive; self.use_end = use_end
        self.n_conj = (S_CAP + 1) * ((E_CAP + 1) if use_end else 1) * 2
        self.conj = list(range(self.n_conj))
        self.role_idx = {r: [self.n_conj + i * R + j for j in range(R)] for i, r in enumerate(ROLES)}
        # the teacher: which role each conjunction (s,e,voice) drives. Only (s,e) combos that occur in real frames
        # (s+e = n-1, n in 3..5 active; n=3 passive) are trained; others stay unbound.
        self.teacher = {}
        for n in (3, 4, 5):
            for pos in range(n):
                self.teacher[conj_index(pos, n - 1 - pos, 0, use_end)] = role_for(pos, n - 1 - pos)
        for pos, role in ((0, "patient"), (1, "action"), (2, "agent")):     # flat passive (n=3) SVO flip
            self.teacher[conj_index(pos, 3 - 1 - pos, 1, use_end)] = role
        pre, post, w = [], [], []
        for k in self.conj:
            for r in ROLES:
                for j in self.role_idx[r]:
                    pre.append(k); post.append(j); w.append(0.5)
        cfg = CoreSimConfig()
        cfg.num_neurons = self.n_conj + len(ROLES) * R
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = int(seed); cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0; cfg.num_traits = 1
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = True
        cfg.hebbian_max_weight = 400.0; cfg.hebbian_learning_rate = 0.005
        for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
                  "enable_reward_modulation", "enable_watts_strogatz"):
            setattr(cfg, f, False)
        cfg.ou_std_current_pA = 20.0
        self.bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                       runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self.bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge.inject_explicit_wiring({"parse": {"pre_indices": pre, "post_indices": post,
                                                      "initial_weights": np.array(w, dtype=np.float32),
                                                      "plastic": True, "conn_type": "E_TO_E", "count": len(pre)}})
        xp, _ = get_backend()
        self.conj_arr = xp.asarray(self.conj, dtype=xp.int64)
        self.role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in self.role_idx.items()}
        self._n = self.bridge.core_config.num_neurons
        self._train(n_epochs, train_steps)

    def _step_reset(self, reset=20):
        self.bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset):
            self.bridge._run_one_simulation_step()

    def _train(self, n_epochs, train_steps):
        xp, _ = get_backend()
        ks = sorted(self.teacher)
        for _ in range(n_epochs):
            for k in ks:
                self._step_reset()
                cur = xp.zeros(self._n, dtype=xp.float32)
                cur[self.conj_arr[k]] = self.drive
                cur[self.role_arr[self.teacher[k]]] = self.drive
                self.bridge.cp_external_input_current[:] = cur
                for _ in range(train_steps):
                    self.bridge._run_one_simulation_step()
        self.bridge.cp_external_input_current[:] = 0.0

    def role_of(self, s, e, voice=0):
        xp, _ = get_backend()
        k = conj_index(s, e, voice, self.use_end)
        self._step_reset()
        cur = xp.zeros(self._n, dtype=xp.float32)
        cur[self.conj_arr[k]] = self.drive
        self.bridge.cp_external_input_current[:] = cur
        rates = {r: 0.0 for r in ROLES}
        for _ in range(self.test_steps):
            self.bridge._run_one_simulation_step()
            for r in ROLES:
                rates[r] += float(to_host(self.bridge.cp_firing_states[self.role_arr[r]].astype(xp.float64).mean()))
        self.bridge.cp_external_input_current[:] = 0.0
        return max(rates, key=rates.get)

    def parse_roles(self, n, voice=0):
        """The per-position roles the bridge reads out for an n-word sentence (active/passive)."""
        return [self.role_of(pos, n - 1 - pos, voice) for pos in range(n)]

    def parse(self, words, voice="active"):
        """Comprehend 'S V [adj]* N' (>=3 words) -> {role: word}, attributes included, from the bridge in spikes."""
        v = 0 if voice in (0, "active") else 1
        return {role: w for role, w in zip(self.parse_roles(len(words), voice=v), list(words))}
