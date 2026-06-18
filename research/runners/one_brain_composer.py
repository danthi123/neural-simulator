"""The production `OneBrainComposer` (roadmap phase 2, the real "one brain"): the whole who/what conversational
pipeline on ONE persistent co-resident `SimulationBridge`, with no host round-trips between operations. An
`RFPhasorComposer` API-sibling the conversational agent can use via `composer_kind="onebrain"`.

Assembled from the validated GO pieces of the Phase-2 arc (each de-risked multi-seed this session):
  - the PARSER front-end (GAP B, `2026-06-18-one-brain-parser-frontend-GO.md`): a `BridgeParser` on slice [0:P]
    comprehends a sentence; the role it FIRES for each word selects that word's bind (no host {role:word} dict;
    voice-invariant).
  - the persistent multi-fact STORE (GAP A, `2026-06-18-one-brain-multifact-store-GAP-A-GO.md`): each fact = a 3-role
    composite written into a (1+D) trigger->readout block in the bridge's complex weights (register-reset-safe; GO to
    K=32).
  - the CUE-matching SCAN + on-bridge cleanup + the no-confab moat (`2026-06-18-one-brain-composer-A3-GO.md`): a
    who/what question reconstructs each stored block, unbinds all three roles IN PARALLEL (one reconstruction, no phase
    drift), cleans up, and the first block whose cue roles match answers; an absent cue / unstored fact abstains.

The parser (Izhikevich, voltage in v/u) and the resonate-and-fire composer registers (a complex phasor in v/u)
co-reside as disjoint slices on ONE bridge (the merged-bridge regime), the resonate-and-fire ops masked to their slice.

HONEST SCOPE: this first cut handles AFFIRMATIVE facts (who / what / affirmative yes-no). Negation (a bound polarity
tag = a 4th role) + the richer agent capabilities (`render_fact`/`query_chain`/`elaborate`) are bounded follow-ons.

NO sim/ edit (reuse-by-import: BridgeParser + RFPhasorComposer + the masked rf_kick). GPU for real use (the parser
trains on the bridge); numpy is the test oracle.
"""
from __future__ import annotations

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import to_host
from research.runners.brain_conversational_agent import BridgeParser
from research.runners.rf_phasor_composer import RFPhasorComposer

ROLES3 = ["agent", "action", "patient"]


def build_coresident_bridge(seed, n_total):
    """An Izhikevich bridge (Hebbian ON for the parser); the RF region has no cp_connections wiring (its memory is in
    cp_rf_w_re/im), so global Hebbian has nothing to touch there."""
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
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
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


class OneBrainComposer:
    """The who/what pipeline on ONE persistent co-resident bridge. Parser [0:P]; RF region from P: fill_0..2,
    bound_0..2, acc (7 blocks), the persistent store (k_max (1+D) blocks), 3 parallel-read Q registers, 3 V-cleanup
    blocks. API mirrors `RFPhasorComposer` for the conversational agent (`store`/`hear`/`query_patient`/`query_agent`/
    `ask_yes_no`; `kb` bookkeeping)."""

    def __init__(self, seed=42, D=128, vocab=None, k_max=32, period=200):
        self.seed = int(seed); self.D = int(D); self.period = int(period)
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=period)
        self.words = list(self.comp.words)              # the cleanup codebook = the composer's ACTUAL vocab
        self.V = len(self.words)
        self.R = 40; self.P = 6 + 3 * self.R; self.k_max = int(k_max)
        self.pol_words = list(self.comp.pol_words)                   # ["AFFIRM","NEGATE"] -- cleaned up SEPARATELY from
        self.NP = len(self.pol_words)                                # the main vocab (a 2-word polarity codebook)
        # 4 fillable roles ALWAYS bound: agent, action, patient, polarity (default AFFIRM) -> yes/no/negation. The
        # 4-role coherence is GO, so the 4th bind is within the substrate's per-fact capacity.
        self.store_base = self.P + 9 * D                            # work: fill_0..3 (4) + bound_0..3 (4) + acc (1) = 9
        self.block = 1 + D
        self.q_base = self.store_base + self.k_max * self.block      # 4 Q regs: agent/action/patient/polarity
        self.c_base = self.q_base + 4 * D                            # cleanup: 3 V-blocks (main roles) + 1 NP-block (pol)
        self.n_total = self.c_base + 3 * self.V + self.NP
        self.b = build_coresident_bridge(seed, self.n_total)
        self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=0)   # wires+trains [0:P]
        self.rf_mask = np.zeros(self.n_total, dtype=bool); self.rf_mask[self.P:self.n_total] = True
        self.kb = []          # bookkeeping: list of (fact_dict, None) -- the agent's _assoc_graph reads fact dicts;
        #                       the bound VECTOR is on-substrate (the None placeholder keeps the (fact, vec) shape)
        self.store_conns = []

    # --- comprehend + store ---
    def _pol(self, polarity):
        return polarity if polarity in self.pol_words else "AFFIRM"

    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO sentence with the on-bridge parser (its role firing selects each bind) + store the fact.
        `polarity` (AFFIRM default / NEGATE) is bound as a 4th role -> `ask_yes_no` returns yes/no/unknown."""
        words = sentence.split() if isinstance(sentence, str) else list(sentence)
        roles = [self.parser.role_of(pos, voice) for pos in range(3)]
        fact = {roles[i]: words[i] for i in range(3)}
        pol = self._pol(polarity)
        self._store_composite([fact.get(r) for r in ROLES3] + [pol], ROLES3 + ["polarity"])
        fact["polarity"] = pol
        self.kb.append((fact, None))
        return fact

    def store(self, agent, action, patient, polarity=None):
        """Store a fact whose roles are already resolved (API parity with RFPhasorComposer; used when the caller's
        parser comprehends). Binds agent/action/patient + the polarity tag (AFFIRM default)."""
        pol = self._pol(polarity)
        self._store_composite([agent, action, patient, pol], ROLES3 + ["polarity"])
        self.kb.append(({"agent": agent, "action": action, "patient": patient, "polarity": pol}, None))

    def _store_composite(self, fillers, roles):
        comp, b, D, P, Pd = self.comp, self.b, self.D, self.P, self.period
        nr = len(roles)
        binds, bundle = [], []
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(nr):
            zr = comp._to_phasor(comp.roles[roles[i]]); zf = comp._to_phasor(comp.concepts[fillers[i]])
            kick[P + i * D:P + (i + 1) * D] = zf                                                  # fill_i at block i
            binds += [(P + (4 + i) * D + k, P + i * D + k, complex(zr[k])) for k in range(D)]     # bound_i at block 4+i
            bundle += [(P + 8 * D + k, P + (4 + i) * D + k, 1.0) for k in range(D)]               # acc at block 8
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        b.rf_set_complex_weights(binds); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        b.rf_set_complex_weights(bundle); b.rf_resonate_steps(Pd + 8)
        zc = comp._to_phasor(np.asarray(b.rf_read_phases())[P + 8 * D:P + 9 * D])
        i = len(self.kb)
        if i >= self.k_max:
            raise RuntimeError(f"OneBrainComposer store full: k_max={self.k_max} reached (shard or raise k_max)")
        trig = self.store_base + i * self.block
        self.store_conns += [(trig + 1 + k, trig, complex(zc[k])) for k in range(D)]

    # --- query (cue-matching scan; reconstruct ONCE per block, read all 4 roles in PARALLEL) ---
    def _read_block(self, block_idx):
        """Reconstruct block_idx + unbind all 4 roles IN PARALLEL (one settle, no phase drift). The 3 main roles clean
        up against the main vocab; the polarity role cleans up against the 2-word polarity codebook (a separate small
        block). Returns (agent, action, patient, polarity)."""
        comp, b, D, Pd, V, NP = self.comp, self.b, self.D, self.period, self.V, self.NP
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        trig = self.store_base + block_idx * self.block
        kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=self.rf_mask)
        b.rf_resonate_steps(Pd + 8)
        unbind = []
        for ri, role in enumerate(ROLES3 + ["polarity"]):
            zc = np.conj(comp._to_phasor(comp.roles[role]))
            unbind += [(self.q_base + ri * D + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        clean = []
        for ri in range(3):                                              # 3 main roles -> the main vocab codebook
            for j in range(V):
                cc = np.conj(comp._to_phasor(comp.concepts[self.words[j]]))
                clean += [(self.c_base + ri * V + j, self.q_base + ri * D + k, complex(cc[k])) for k in range(D)]
        for j in range(NP):                                              # polarity role -> the 2-word polarity codebook
            cc = np.conj(comp._to_phasor(comp.concepts[self.pol_words[j]]))
            clean += [(self.c_base + 3 * V + j, self.q_base + 3 * D + k, complex(cc[k])) for k in range(D)]
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        out = []
        for ri in range(3):
            scores = np.maximum(mem[self.c_base + ri * V:self.c_base + (ri + 1) * V], 0.0)
            out.append(self.words[int(np.argmax(scores))])
        pol_scores = np.maximum(mem[self.c_base + 3 * V:self.c_base + 3 * V + NP], 0.0)
        out.append(self.pol_words[int(np.argmax(pol_scores))])
        return tuple(out)            # (agent, action, patient, polarity)

    def _scan(self, cue, answer_idx):
        for i in range(len(self.kb)):
            wa, wv, wp, _pol = self._read_block(i)
            got = {"agent": wa, "action": wv, "patient": wp}
            if all(got[role] == want for role, want in cue.items()):
                return (wa, wv, wp)[answer_idx]
        return None

    def query_patient(self, agent, action, order_fn=None):
        return self._scan({"agent": agent, "action": action}, 2)

    def query_agent(self, action, patient):
        return self._scan({"action": action, "patient": patient}, 0)

    def ask_yes_no(self, agent, action, patient):
        """yes / no / unknown: the first fact matching the full SVO answers by its polarity tag (AFFIRM -> yes,
        NEGATE -> no); no matching fact -> 'unknown' (the no-confab moat)."""
        for i in range(len(self.kb)):
            wa, wv, wp, wpol = self._read_block(i)
            if wa == agent and wv == action and wp == patient:
                return "yes" if wpol == "AFFIRM" else "no"
        return "unknown"
