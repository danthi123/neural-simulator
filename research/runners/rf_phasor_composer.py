"""FHRR-on-bridge layer (b): a PARALLEL RF phasor composer running the conversational composition on the bridge's
resonate-and-fire neurons + complex synapses -- so the opponency (the rate-coded composer's SNR wall) is GONE (the
phasor algebra has no common mode). Same conversational API as core_sim_composition.CoreSimComposer; validated at
parity before the BrainConversationalAgent switches (layer c). Design:
docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md.

Reuse-by-import the RF + complex-synapse substrate already on the bridge (NeuronModel.RESONATE_AND_FIRE +
rf_kick / rf_read_phases / rf_set_complex_weights, layers RF-on-bridge + layer-a). NO sim/ edits here.

Representation: each concept/role is a PHASOR vector (phases in [0,1)^D, deterministic per seed). bind = role (x)
filler via a DIAGONAL complex synapse (weight = the role phasor); bundle = unit complex synapses (the sum -- NO
opponency); unbind = conj diagonal synapse; cleanup = phase-cosine argmax. Abstention (the no-confab moat): the
relational query returns None when no stored fact's cue roles match (architecture-preserved).
"""
from collections import namedtuple

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge
from sim.backend import to_host

ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2")
DEFAULT_VOCAB = ["dog", "cat", "go", "run", "come", "stop", "look", "north", "south", "east", "west", "apple",
                 "river", "big", "small", "hot", "cold"]
# A recursive SVO clause that can be a filler ('dog look (cat go north)'). Mirrors core_sim_composition.Clause.
Clause = namedtuple("Clause", ["agent", "action", "patient"])


def _is_clause(x):
    """A clause-like filler: any namedtuple with (agent, action, patient) fields. Duck-typed so it recognizes BOTH
    this module's Clause AND core_sim_composition.Clause (the BrainConversationalAgent passes the latter) -- they are
    distinct namedtuple classes, so isinstance() would miss across them. A plain tuple (e.g. an ('adj', 'noun')
    attribute) has no _fields -> correctly NOT a clause."""
    return getattr(x, "_fields", None) == ("agent", "action", "patient")


def _build_rf_bridge(n, seed=42):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return bridge


class RFPhasorComposer:
    def __init__(self, seed=42, D=64, vocab=None, period=200, enable_spiking_cleanup=False,
                 enable_substrate_store=False, grounded_codes=None):
        self.seed = int(seed)
        self.D = int(D)
        self.period = int(period)
        # (cheat-C conversion, opt-in) hold each fact's bound composite in the SUBSTRATE (per-fact trigger->readout
        # complex weights) instead of a numpy array in self.kb; retrieve via firing. Default OFF: numpy kb fast path.
        self.enable_substrate_store = bool(enable_substrate_store)
        # (cheat-B conversion, opt-in) route _cleanup through the fully-on-bridge spiking cleanup (matched filter on
        # the complex synapse + Izhikevich WTA). Default OFF: numpy argmax stays the fast path (the rate composer's
        # NEF-cleanup opt-in pattern). Validated == numpy multi-seed.
        self.enable_spiking_cleanup = bool(enable_spiking_cleanup)
        self._izh_bank_cache = {}      # Stage-2 Izhikevich WTA banks, keyed by candidate count
        self._cleanup_drive_pA = 60.0  # input-normalized drive for the winner (sane band 20-100; >=200 over-drives)
        self._cleanup_window = 120
        self.words = sorted(vocab) if vocab is not None else sorted(DEFAULT_VOCAB)
        rng = np.random.default_rng(seed)
        # phasor codes: phases in [0,1)^D per concept + per role (deterministic per seed)
        self.concepts = {w: rng.uniform(0.0, 1.0, self.D) for w in self.words}
        # (cheat-A conversion, opt-in) SENSORY-GROUNDED codes: a {word: phases[D]} dict (e.g. real V1 Gabor responses
        # projected to phases) overrides the random codes for those words. Validated == random at parity (the
        # grounding INTERFACE works on the RF substrate). HONEST boundary: producing meaningful grounded codes (real
        # object images + abstract-concept grounding) is the open problem -- the embodied-cognition limit; this is the
        # interface, not full semantic grounding.
        if grounded_codes:
            for w, ph in grounded_codes.items():
                if w in self.concepts:
                    self.concepts[w] = np.asarray(ph, dtype=float)
        # AFFIRM/NEGATE polarity fillers (phasor codes; cleaned up only against pol_words, not the main vocab)
        self.pol_words = ["AFFIRM", "NEGATE"]
        for tag in self.pol_words:
            self.concepts[tag] = rng.uniform(0.0, 1.0, self.D)
        self.roles = {r: rng.uniform(0.0, 1.0, self.D) for r in ROLES}
        self.kb = []  # (fact_dict, composite_phases)
        self._dlpfc = None       # dialogue-planning Control (lazy; rebuilt only when the association graph changes)
        self._dlpfc_key = None
        self._bridge_cache = {}  # (c-opt) reuse RF bridges by neuron count -> avoid _initialize_simulation_data per op

    # --- RF complex-synapse ops (each op a per-op RF bridge; reuse-by-import the substrate) ---
    def _resonate(self, n, conns, kick):
        # (c-opt) reuse a cached bridge per neuron count; zero its complex weights (rf_set_complex_weights appends)
        # and rf_kick resets the RF state -> each op is clean. Avoids _initialize_simulation_data per op.
        b = self._bridge_cache.get(n)
        if b is None:
            b = _build_rf_bridge(n, self.seed)
            self._bridge_cache[n] = b
        b.rf_set_complex_weights(conns)   # (c-opt) builds the sparse complex weights FRESH each op -> replaces; no reset needed
        b.rf_kick(kick, period=self.period, lam=0.0)
        b.rf_resonate_steps(self.period + 8)   # (c-opt) fast RF dynamics loop -- skips the full-step machinery
        return np.asarray(b.rf_read_phases())

    @staticmethod
    def _to_phasor(phases):
        return np.exp(2j * np.pi * np.asarray(phases))

    def _bind(self, role_phases, filler_phases):
        """bound = role_phasor (x) filler_phasor, via a diagonal complex synapse (filler pre -> bound post,
        weight = the role phasor)."""
        D = self.D
        zf = self._to_phasor(filler_phases)
        zr = self._to_phasor(role_phases)
        conns = [(D + k, k, zr[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zf
        return self._resonate(2 * D, conns, kick)[D:]

    def _bundle(self, phase_list):
        """composite[k] = sum_l phase_list[l][k] via unit complex synapses (NO opponency)."""
        L = len(phase_list)
        D = self.D
        conns = [(L * D + k, l * D + k, 1.0) for l in range(L) for k in range(D)]
        kick = np.zeros((L + 1) * D, dtype=np.complex128)
        for l in range(L):
            kick[l * D:(l + 1) * D] = self._to_phasor(phase_list[l])
        return self._resonate((L + 1) * D, conns, kick)[L * D:]

    def _filler_phases(self, filler):
        """The phasor phases to bind for a filler: a concept's code, OR (recursively) a Clause's bound composite."""
        if _is_clause(filler):
            return self._encode({"agent": filler.agent, "action": filler.action, "patient": filler.patient})
        return self.concepts[filler]

    def _encode(self, fact):
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ROLES if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _render(self, comp_phases, role, stored, order_fn=None):
        """Render `role`'s filler from a composite, FROM THE RF UNBIND. `stored` (a word or Clause) ROUTES
        flat-cleanup vs recursive clause-decode; the content is decoded from the substrate, not the stored labels.
        `order_fn` (opt-in, default None = the host f-string): when set, the inner clause's SVO word order is
        produced by the de-risked spiking serial-order generator instead of the host literal (the generation path
        passes it; the Q&A path leaves it None)."""
        rec = self._unbind_phases(comp_phases, role)
        if _is_clause(stored):
            a = self._cleanup(self._unbind_phases(rec, "agent"))
            ac = self._cleanup(self._unbind_phases(rec, "action"))
            pt = self._cleanup(self._unbind_phases(rec, "patient"))
            words = [a, ac, pt]
            if order_fn is not None:
                return " ".join(words[i] for i in order_fn(len(words)))   # neural serial-order (inner clause)
            return f"{a} {ac} {pt}"
        return self._cleanup(rec)

    def _unbind_phases(self, composite_phases, role):
        """recovered = conj(role_phasor) (x) composite, via a conj diagonal complex synapse."""
        D = self.D
        zc = self._to_phasor(composite_phases)
        zr_conj = np.conj(self._to_phasor(self.roles[role]))
        conns = [(D + k, k, zr_conj[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zc
        return self._resonate(2 * D, conns, kick)[D:]

    def _izh_bank(self, V):
        """A cached Izhikevich concept bank of V neurons (no wiring; driven by external current) -- the Stage-2 WTA."""
        bank = self._izh_bank_cache.get(V)
        if bank is None:
            cfg = CoreSimConfig()
            cfg.num_neurons = int(V)
            cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
            cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
            cfg.seed = self.seed
            cfg.dt_ms = 1.0
            cfg.connections_per_neuron = 0
            cfg.num_traits = 1
            for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                      "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                      "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
                if hasattr(cfg, f):
                    setattr(cfg, f, False)
            cfg.ou_std_current_pA = 0.0
            bank = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                    runtime_state=RuntimeState(), gpu_config=GPUConfig())
            bank._initialize_simulation_data(called_from_playback_init=False)
            # snapshot the resting state so each cleanup starts clean (a cached bank's v/u persist across calls,
            # which would let a recently-fired neuron's adapted state bias the next cleanup's WTA)
            bank._cleanup_v0 = bank.cp_membrane_potential_v.copy()
            bank._cleanup_u0 = bank.cp_recovery_variable_u.copy()
            self._izh_bank_cache[V] = bank
        return bank

    def _spiking_cleanup(self, rec_phases, words):
        """Fully on-bridge cleanup (clears cheat B). Stage 1 -- the matched FILTER is the bridge's complex-synapse
        matvec (the SAME op as unbind): install conj(codebook) synapses (rec -> concept), kick rec, one matvec step,
        read each concept neuron's |c_k| = |S* rec| off the membrane (cp_membrane_potential_v / cp_recovery_variable_u
        = the RF re/im). Stage 2 -- the SELECTION is a spiking Izhikevich WTA driven by the input-normalized scores;
        winner = argmax-over-FIRING (a readout of spiking output, as the NEF cleanup's final argmax). The only numpy
        is the membrane readout + the firing-argmax readout -- NO numpy COMPUTATION of the match or the selection.
        Validated == numpy argmax multi-seed: research/findings/2026-06-05-phase1-tpam-cleanup-derisk-GO.md."""
        D = self.D
        V = len(words)
        # Stage 1: matched filter on the complex synapse (concept k = index D+k receives rec via conj(code_k)).
        conns = []
        for k in range(V):
            cc = np.conj(self._to_phasor(self.concepts[words[k]]))
            for d in range(D):
                conns.append((D + k, d, cc[d]))
        b = self._bridge_cache.get(D + V)
        if b is None:
            b = _build_rf_bridge(D + V, self.seed)
            self._bridge_cache[D + V] = b
        b.rf_set_complex_weights(conns)
        kick = np.zeros(D + V, dtype=np.complex128)
        kick[:D] = self._to_phasor(rec_phases)
        b.rf_kick(kick, period=self.period, lam=0.0)
        b.rf_resonate_steps(1)
        # The matched-filter score is Re(c_k) = the concept neuron's membrane (re) = exactly the numpy cos score
        # (mean cos = Re(c_k)/D). Rectified so off-target concepts (Re~0 / negative) emit ZERO drive -> silent ->
        # a clean WTA (the NEF cleanup's "off-target emits zero spikes"). |c_k| would leave off-targets driven.
        re = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[D:D + V]
        scores = np.maximum(re, 0.0)
        peak = float(scores.max())
        if peak <= 1e-9:
            return words[int(np.argmax(scores))]
        # Stage 2: spiking WTA (input-normalized drive -> firing -> argmax-over-firing).
        drive = (scores / peak) * self._cleanup_drive_pA
        bank = self._izh_bank(V)
        bank.cp_membrane_potential_v[:] = bank._cleanup_v0   # reset to resting -> each cleanup is independent
        bank.cp_recovery_variable_u[:] = bank._cleanup_u0
        import sim.backend as _b
        xp, _ = _b.get_backend()
        bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(V)
        for _ in range(self._cleanup_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        if float(firing.max()) <= 0.0:
            return words[int(np.argmax(scores))]
        return words[int(np.argmax(firing))]

    def _cleanup(self, rec_phases, words=None):
        words = words if words is not None else self.words
        if self.enable_spiking_cleanup:
            return self._spiking_cleanup(rec_phases, words)
        sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - self.concepts[w])))) for w in words]
        return words[int(np.argmax(sims))]

    def unbind(self, composite_phases, role, words=None):
        return self._cleanup(self._unbind_phases(composite_phases, role), words)

    # --- conversational API (mirrors CoreSimComposer; the no-confab moat preserved) ---
    def store(self, agent, action, patient, polarity=None):
        fact = {"agent": agent, "action": action}
        if _is_clause(patient):                    # a recursive clause filler (check BEFORE tuple: a Clause IS a tuple)
            fact["patient"] = patient
        elif isinstance(patient, tuple):           # (adj(s), noun) -- an attributed entity ('big apple' or 'big hot apple')
            adjs, noun = patient
            adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
            fact["patient"] = noun
            fact["attribute"] = adjs[0]
            if len(adjs) > 1:
                fact["attribute2"] = adjs[1]       # 2-attribute (the +-1 scheme's K=5 boundary -- does FHRR lift it?)
        else:
            fact["patient"] = patient
        if polarity is not None:
            fact["polarity"] = polarity      # a bound AFFIRM/NEGATE tag (extra binding -> more load)
        comp = self._encode(fact)
        self.kb.append((fact, self._store_substrate(comp) if self.enable_substrate_store else comp))

    # --- reconsolidation: prediction-error-gated in-place fact update (Option A; additive, store/query unchanged) ---
    def _find_cued_fact(self, agent, action):
        """Reactivation: the FIRST stored fact whose CUE roles (agent+action) match, by the substrate unbind +
        cleanup. Returns (kb_index, fact, composite) or None (no trace to reactivate -> abstain)."""
        for i, (fact, handle) in enumerate(self.kb):
            comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                return i, fact, comp
        return None

    def _patient_prediction_error(self, comp, patient_word):
        """PE = 1 - phase-cos(recovered patient phasor, the asserted patient's code). ~0 when the asserted filler
        matches the stored one (a re-statement); ~1 on a mismatch (a correction)."""
        rec = self._unbind_phases(comp, "patient")
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.concepts[patient_word]))))

    def _calibrate_pe_labile(self):
        """Frozen labilization gate = the midpoint of the measured same-vs-different prediction-error distributions
        over the CURRENT facts (each fact's PE against its OWN stored patient = 'same'; against other facts'
        patients = 'different'). The data's own separation point -- NOT tuned to a downstream probe (the
        calibrate_threshold rule). 0.5 fallback when too few distinct facts exist to calibrate."""
        facts = []
        for fact, handle in self.kb:
            p = fact.get("patient")
            if isinstance(p, str):
                comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
                facts.append((comp, p))
        same, diff = [], []
        for comp, p in facts:
            same.append(self._patient_prediction_error(comp, p))
            for _comp2, p2 in facts:
                if p2 != p:
                    diff.append(self._patient_prediction_error(comp, p2))
        if not same or not diff:
            return 0.5
        return 0.5 * (float(np.mean(same)) + float(np.mean(diff)))

    def update_on_mismatch(self, agent, action, new_patient, pe_labile=None):
        """RECONSOLIDATION: a corrective utterance ('actually, <agent> <action> <new_patient>') reactivates the
        cued fact and -- ONLY if the new filler carries a prediction error above the labilization gate -- rewrites
        that fact's patient IN PLACE (no contradictory duplicate). A fully-predicted re-statement re-stabilizes
        unchanged; a NEVER-stored cue ABSTAINS (the no-confab moat: a reactivated trace is updated, a missing one
        is not fabricated). ADDITIVE -- store()/query_*() are unchanged, so any caller that never invokes this
        keeps the append-only path byte-for-byte; the agent-level opt-in is where 'default-off' lives.

        pe_labile=None -> auto-calibrate the gate from the current facts (the validated midpoint rule); else use
        the supplied gate. Returns {action: abstain|rewrite|restabilize, wrote: bool, pe: float|None}. Nader 2000;
        Osan-Tort-Amaral 2011 mismatch-gated attractor update; Sevenster 2013 prediction-error necessity. De-risked
        6/6 multi-seed: research/findings/2026-06-17-reconsolidation-update-derisk-GO.md."""
        found = self._find_cued_fact(agent, action)
        if found is None:
            return {"action": "abstain", "wrote": False, "pe": None}     # no trace -> no update, no fabrication
        idx, fact, comp = found
        gate = self._calibrate_pe_labile() if pe_labile is None else float(pe_labile)
        pe = self._patient_prediction_error(comp, new_patient)
        if pe >= gate:
            f2 = dict(fact); f2["patient"] = new_patient
            comp2 = self._encode(f2)
            self.kb[idx] = (f2, self._store_substrate(comp2) if self.enable_substrate_store else comp2)
            return {"action": "rewrite", "wrote": True, "pe": pe}
        return {"action": "restabilize", "wrote": False, "pe": pe}        # PE below the gate -> re-stabilize

    def count_facts(self, agent, action):
        """Number of stored facts whose cue roles (agent+action) match -- 1 after a reconsolidation update, 2 if a
        correction was naively appended. Used by the reconsolidation tests + the correction-turn hook."""
        return sum(1 for fact, handle in self.kb
                   if self.unbind(self._retrieve_substrate(handle) if self.enable_substrate_store else handle,
                                  "agent") == agent
                   and self.unbind(self._retrieve_substrate(handle) if self.enable_substrate_store else handle,
                                   "action") == action)

    def _store_substrate(self, comp_phases):
        """Hold the bound composite in the SUBSTRATE: a persistent (1+D) RF bridge whose trigger(neuron 0) ->
        readout(1..D) complex weights carry the composite phasor. The composite lives in the synaptic weights
        (cp_rf_w_re/im), NOT a numpy array -- the Crawford-Eliasmith weight-store (Hebb memory-in-weights). The kb
        holds this bridge handle, not the composite. Validated == numpy store at parity (Phase-2 de-risk GO)."""
        D = self.D
        zc = self._to_phasor(comp_phases)
        conns = [(1 + k, 0, zc[k]) for k in range(D)]
        b = _build_rf_bridge(1 + D, self.seed)
        b.rf_set_complex_weights(conns)
        return b

    def _retrieve_substrate(self, b):
        """Read a substrate-held composite back: fire the trigger (unit phasor) -> the readout neurons reconstruct
        the composite IN PHASE (the magnitude-invariant RF phase readout)."""
        D = self.D
        kick = np.zeros(1 + D, dtype=np.complex128)
        kick[0] = 1.0
        b.rf_kick(kick, period=self.period, lam=0.0)
        b.rf_resonate_steps(self.period + 8)
        return np.asarray(b.rf_read_phases())[1:1 + D]

    def _iter_facts(self):
        """Yield (fact_dict, composite_phases) per stored fact. With the substrate store, the composite is read back
        from its substrate weight-bridge (fire the trigger); else it's the numpy array in kb. Lazy -> an early-return
        query only retrieves the facts it actually checks."""
        for fact, handle in self.kb:
            yield fact, (self._retrieve_substrate(handle) if self.enable_substrate_store else handle)

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> the agent of the matching fact; None if no fact matches (abstention)."""
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "action") == action and self.unbind(comp, "patient") == patient:
                return self.unbind(comp, "agent")
        return None

    def query_patient(self, agent, action, order_fn=None):
        """'what does <agent> <action>?' -> the patient of the matching fact (an attributed entity 'big apple' if
        the fact bound an ATTRIBUTE); None if no match (abstention). The stored structure only routes the rendering;
        the words are decoded from the RF unbind. `order_fn` (opt-in, default None = host f-string): when set, an
        inner CLAUSE patient's SVO order is produced by the de-risked spiking serial-order generator. The moat is
        unaffected: abstention (return None) happens BEFORE any rendering."""
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                noun = self._render(comp, "patient", fact["patient"], order_fn=order_fn)   # word OR recursive Clause
                adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    return " ".join(adjs + [noun])    # 'big apple' / 'big hot apple'
                return noun
        return None

    def query_chain(self, cue, actions):
        """Multi-hop relational reasoning: follow a chain of stored facts. Each hop matches the current concept as
        the AGENT under the hop's action and reads the PATIENT, which becomes the next hop's cue --
        query_chain('dog', ['eat', 'eat']) over {dog eat cat, cat eat mouse} -> 'mouse'. Returns the terminal
        concept, or None (abstain) the moment any hop has no matching fact -- so the no-confab moat holds at EVERY
        hop and a broken or over-run chain never confabulates. The cleanup re-discretizes the intermediate concept
        each hop, so retrieval error does NOT compound across hops. De-risked GO 3 seeds x 3 D (controls -- leaky
        spreading, permuted-relation, between-hop re-cue lesion -- all collapse): 2026-06-17-multihop-query-chain-GO.md."""
        x = cue
        for action in actions:
            x = self.query_patient(x, action)
            if x is None:
                return None
        return x

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown' via the bound AFFIRM/NEGATE polarity tag.
        Matches the full SVO; 'unknown' (abstention) when no stored fact matches."""
        for fact, comp in self._iter_facts():
            if (self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action
                    and self.unbind(comp, "patient") == patient):
                return "yes" if self.unbind(comp, "polarity", self.pol_words) == "AFFIRM" else "no"
        return "unknown"

    def render_fact(self, agent, order_fn=None):
        """Generation: render a full stored sentence whose agent matches `agent` -- e.g. 'dog go north' (an
        attributed patient 'big apple' or a nested clause renders too). The action + patient are DECODED from the
        RF unbind (not the stored labels); None if no fact's agent matches (the no-confab moat -- no invented
        sentence about an unknown subject).

        `order_fn` (opt-in, default None = the host f-string): a callable n -> a permutation of range(n) that
        produces the word ORDER. When set, the slot order comes from the de-risked spiking competitive-queuing
        serial-order generator (NeuralSerialOrderRenderer) instead of the host literal -- the cognitive ordering
        is then neural; only the final join (the body's emission) is host. The moat is unaffected: abstention
        (return None) happens BEFORE any ordering."""
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "agent") == agent:
                ac = self.unbind(comp, "action")
                pt = self._render(comp, "patient", fact["patient"], order_fn=order_fn)   # inner clause neural too
                adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    pt = " ".join(adjs + [pt])
                words = [agent, ac, pt]
                if order_fn is not None:
                    return " ".join(words[i] for i in order_fn(len(words)))   # neural serial-order (outer SVO)
                return f"{agent} {ac} {pt}"
        return None

    # --- dialogue planning (the dlPFC content-selection Control; architecture-independent: operates on the graph) ---
    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) from the stored facts (agent/action/patient co-occur;
        clause patients are skipped -- their inner concepts are structural). The graph the dlPFC spreads over."""
        graph = {}
        for fact, _ in self.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient") if isinstance(fact.get(r), str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning: the next on-topic concept about `topic`, chosen by the dlPFC spiking content-selection
        Control (loop-attractor working memory + spreading activation) over the agent's own association graph -- the
        same validated SpikingSpreadingController the rate-coded agent uses (it operates on the GRAPH, so it is
        substrate-independent). None if `topic` is unconnected."""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        graph = self._assoc_graph()
        if topic not in graph:
            return None
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc is None or self._dlpfc_key != key:
            self._dlpfc = SpikingSpreadingController(graph, seed=self.seed)
            self._dlpfc_key = key
        return self._dlpfc.turn_latency([topic])
