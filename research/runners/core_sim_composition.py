"""Core-sim composition — the conversational composition primitives realized ON the core SimulationBridge
(the brain), promoted from the validated `_insubstrate_*` probes into ONE clean, importable, tested module.

This is Phase 1 of the consolidation (`docs/plans/2026-06-04-consolidate-conversational-pipeline-onto-core-sim-design.md`):
the role-filler VSA composition runs as a genuine spiking computation on a `SimulationBridge` of Izhikevich
neurons, NOT on the bolted-on numpy phasor simulators. Mechanism (validated, pillar n=111): a +-1 Hadamard
computed by COINCIDENCE neurons -- role_ON/OFF + fill_ON/OFF source banks synapse into 4 AND banks A/B/C/D
(A=AND(role_ON,fill_ON), B=AND(role_OFF,fill_OFF) -> bound_ON; C,D -> bound_OFF); each coincidence neuron fires
only when BOTH inputs are active (threshold + a tonic hyperpolarizing bias). bound = bound_ON - bound_OFF =
role (x) filler in spiking rates. UNBIND reuses the SAME layer with role := query. Cleanup = nearest concept by
dot product. Facts are stored SEPARATELY (each its own bound vector). Negation = a 4th POLARITY role bound to an
AFFIRM/NEGATE filler. Abstention (the no-confab moat) = the relational query returns None when no stored fact's
agent matches the cue.

Concept codes are the substrate's OWN concept-pool activity (the `denoise64` cache = captured + denoised V=16
concept-pool firing) -- i.e. grounded in the brain, not random. Operating point (validated multi-seed): bias=-500
(higher coincidence firing rate = more dynamic range), readout window 150 steps.

FROZEN bars (carried unchanged from the probes): spiking single-fact role recovery >= 0.80; relational Q&A
(find-by-agent, read-patient) >= 0.80; yes/no >= 0.80; the absent-cue control gives no false match. A pure-numpy
reference at the same D is the algebra ceiling (isolates projection/cleanup from spiking loss).

Provenance (ported faithfully; a regression test pins parity to the probe):
  `research/findings/raw/_insubstrate_bind_unbind_probe.py` (bind/unbind/build/hadamard)
  `research/findings/raw/_insubstrate_relational_memory_probe.py` (KB/Q&A)
  `research/findings/raw/_insubstrate_negation_probe.py` (4-role polarity)
  `research/runners/abstention_gate.py` (the moat threshold; here abstention is the in-loop no-false-match)
"""
from __future__ import annotations
import os
from collections import namedtuple
import numpy as np

# An embedded clause used as a filler (recursive role-filler structure): "dog look (cat go south)".
Clause = namedtuple("Clause", ["agent", "action", "patient"])

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.backend import get_backend, to_host

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"

# coincidence operating point (validated in the _insubstrate probes)
W_COINC = 320.0
DEFAULT_BIAS = -500.0       # production: higher firing rate -> more dynamic range -> robust multi-seed KB/Q&A/negation
ROLE_DRIVE = 2500.0
FILL_DRIVE = 2500.0
RESET_STEPS = 20
DEFAULT_RUN_STEPS = 150


def _center(v):
    v = np.asarray(v, dtype=np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed, proj_dim, rng):
    """The substrate's OWN concept codes (denoise64 cache = captured concept-pool activity). Average the samples,
    optionally project to proj_dim (random Gaussian, preserves cosines), mean-center + unit-normalize. Returns
    (words, codes[V, Deff]). Grounded in the brain's concept pools, not random."""
    d = np.load(CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)   # [V, 3200]
    if proj_dim and proj_dim > 0:
        P = rng.standard_normal((raw.shape[1], proj_dim)) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = np.stack([_center(raw[i]) for i in range(raw.shape[0])])
    return ws, codes


def onoff(vec):
    """signed vector -> (ON, OFF) non-negative parts (the substrate's ON/OFF opponent channels)."""
    return np.maximum(vec, 0.0), np.maximum(-vec, 0.0)


def _scale_to_current(on, off, drive):
    m = max(on.max(), off.max(), 1e-9)
    return on / m * drive, off / m * drive


def build_bind_bridge(seed, D):
    """Build the 8D-neuron coincidence bridge: role_ON/OFF + fill_ON/OFF sources -> 4 AND banks A/B/C/D, wired so
    A=AND(role_ON,fill_ON), B=AND(role_OFF,fill_OFF), C=AND(role_ON,fill_OFF), D=AND(role_OFF,fill_ON). Plasticity
    OFF (a fixed-wiring computation). Returns (bridge, idx)."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 8 * D
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed); cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0; cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz"):
        setattr(cfg, f, False)
    cfg.ou_std_current_pA = 20.0

    role_on = np.arange(0, D); role_off = np.arange(D, 2 * D)
    fill_on = np.arange(2 * D, 3 * D); fill_off = np.arange(3 * D, 4 * D)
    A = np.arange(4 * D, 5 * D); B = np.arange(5 * D, 6 * D)
    C = np.arange(6 * D, 7 * D); Dd = np.arange(7 * D, 8 * D)
    pre, post = [], []
    for src1, src2, dst in ((role_on, fill_on, A), (role_off, fill_off, B),
                            (role_on, fill_off, C), (role_off, fill_on, Dd)):
        for i in range(D):
            pre.append(int(src1[i])); post.append(int(dst[i]))
            pre.append(int(src2[i])); post.append(int(dst[i]))
    w = np.full(len(pre), W_COINC, dtype=np.float32)
    plan = {"bind": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                     "plastic": False, "conn_type": "E_TO_E", "count": len(pre)}}
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.inject_explicit_wiring(plan)
    xp, _ = get_backend()
    idx = dict(role_on=role_on, role_off=role_off, fill_on=fill_on, fill_off=fill_off, A=A, B=B, C=C, D=Dd)
    return bridge, {k: xp.asarray(v, dtype=xp.int64) for k, v in idx.items()}


def hadamard_spiking(bridge, idx, role_vec, fill_on_cur, fill_off_cur, D, run_steps, coinc_bias):
    """One spiking (x): drive role (binary +-1 -> ON/OFF) + fill (graded ON/OFF currents); read the coincidence
    banks over `run_steps`. Returns (out_on, out_off) D-vectors of coincidence firing rates."""
    xp, _ = get_backend()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
    cur = xp.zeros(8 * D, dtype=xp.float32)
    cur[idx["role_on"]] = xp.asarray((role_vec > 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["role_off"]] = xp.asarray((role_vec < 0).astype(np.float32) * ROLE_DRIVE)
    cur[idx["fill_on"]] = xp.asarray(fill_on_cur.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(fill_off_cur.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = coinc_bias
    bridge.cp_external_input_current[:] = cur
    acc = {b: xp.zeros(D, dtype=xp.float64) for b in ("A", "B", "C", "D")}
    for _ in range(run_steps):
        bridge._run_one_simulation_step()
        for b in ("A", "B", "C", "D"):
            acc[b] += bridge.cp_firing_states[idx[b]].astype(xp.float64)
    bridge.cp_external_input_current[:] = 0.0
    rates = {b: to_host(acc[b]) / run_steps for b in ("A", "B", "C", "D")}
    return rates["A"] + rates["B"], rates["C"] + rates["D"]      # bound_ON / est_ON , bound_OFF / est_OFF


class CoreSimComposer:
    """Role-filler VSA composition + a queryable SVO fact-memory, realized ON the core SimulationBridge. Holds ONE
    coincidence bridge; binds/unbinds in spiking; stores facts separately; answers who/what + yes/no; abstains on
    the unknown. Concept codes are the substrate's own (grounded). The brain analogue of the role-filler half of
    the unified agent -- no bolted-on numpy simulator in the path."""

    ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2")

    def __init__(self, seed=42, proj_dim=800, coinc_bias=DEFAULT_BIAS, run_steps=DEFAULT_RUN_STEPS, concepts=None):
        if concepts is None:
            if not os.path.exists(CACHE % seed):
                raise FileNotFoundError(f"concept cache missing: {CACHE % seed}")
            rng0 = None if (not proj_dim or proj_dim <= 0) else np.random.default_rng(seed)
            self.words, codes = load_concepts(seed, proj_dim, rng0)
        else:
            self.words = sorted(concepts)
            codes = np.stack([_center(concepts[w]) for w in self.words])
        self.D = codes.shape[1]
        self.concepts = {w: codes[i] for i, w in enumerate(self.words)}
        self.coinc_bias = float(coinc_bias)
        self.run_steps = int(run_steps)
        rng = np.random.default_rng(seed)
        # distinct AFFIRM/NEGATE polarity fillers (added to the codebook so unbind can clean up to them)
        for tag in ("AFFIRM", "NEGATE"):
            self.concepts[tag] = _center(rng.standard_normal(self.D))
        self.pol_words = ["AFFIRM", "NEGATE"]
        # +-1 distributed roles (ON/OFF realizable); unit-normalized
        self.roles = {r: rng.choice([-1.0, 1.0], size=self.D) for r in self.ROLES}
        self.roles = {r: v / np.linalg.norm(v) for r, v in self.roles.items()}
        self.bridge, self.idx = build_bind_bridge(seed, self.D)
        self.kb = []   # list of (fact_dict, bound_onoff)

    # --- low-level spiking ops ---
    def _op(self, role_vec, fill_on_cur, fill_off_cur):
        return hadamard_spiking(self.bridge, self.idx, role_vec, fill_on_cur, fill_off_cur,
                                self.D, self.run_steps, self.coinc_bias)

    def _filler_signed(self, filler):
        """The signed code to bind for a filler: a concept's code, OR (recursively) the bound vector of an embedded
        Clause -- so a clause can be a filler (recursive role-filler nesting)."""
        if isinstance(filler, Clause):
            cb = self.bind_fact({"agent": filler.agent, "action": filler.action, "patient": filler.patient})
            return cb[0] - cb[1]
        return self.concepts[filler]

    def bind_fact(self, fact):
        """Bind sum_role role (x) filler[fact[role]] in spiking; return canonical (ON, OFF). Each present role's
        filler is a concept word OR an embedded Clause (recursively bound)."""
        bon = np.zeros(self.D); boff = np.zeros(self.D)
        for role in self.ROLES:
            if role not in fact:
                continue
            c_on, c_off = onoff(self._filler_signed(fact[role]))
            fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
            o, f = self._op(self.roles[role], fon, foff)
            bon += o; boff += f
        return onoff(bon - boff)      # ON/OFF opponency (common-mode removal) before storage

    def _unbind_onoff(self, bound_onoff, role):
        """Spiking-unbind `role` from a bound (ON, OFF) -> the recovered filler's (est_ON, est_OFF)."""
        fon, foff = _scale_to_current(bound_onoff[0], bound_onoff[1], FILL_DRIVE)
        return self._op(self.roles[role], fon, foff)

    def unbind(self, bound_onoff, role, codebook=None):
        """Spiking-unbind `role`; clean up to the nearest code in `codebook` (default: all concept words)."""
        words = codebook if codebook is not None else self.words
        e_on, e_off = self._unbind_onoff(bound_onoff, role)
        est = e_on - e_off
        return words[int(np.argmax([self.concepts[w] @ est for w in words]))]

    def _render_filler(self, bound_onoff, role, stored):
        """Decode the filler of `role` from a bound structure, FROM THE SPIKING UNBIND. `stored` is the agent's
        memory of the filler's structure (a word or a Clause) -- used only to ROUTE flat-cleanup vs recursive
        clause-decode; the CONTENT is decoded from the substrate. Returns a rendered string."""
        e_on, e_off = self._unbind_onoff(bound_onoff, role)
        if isinstance(stored, Clause):
            rec = onoff(e_on - e_off)                      # the recovered clause-bound vector
            a = self._render_filler(rec, "agent", stored.agent)
            ac = self._render_filler(rec, "action", stored.action)
            pt = self._render_filler(rec, "patient", stored.patient)
            return f"{a} {ac} {pt}"
        est = e_on - e_off
        return self.words[int(np.argmax([self.concepts[w] @ est for w in self.words]))]

    # --- conversational API ---
    def store(self, agent, action, patient, polarity=None):
        """Learn an SVO fact. `patient` may be:
          - a concept word ('apple'); or
          - an ATTRIBUTED entity ('big apple') as a tuple (adj, noun) or ((adj1, adj2), noun) -- the adjective(s)
            are bound to dedicated ATTRIBUTE role(s) (feature binding), the noun to the patient role; or
          - an embedded Clause (recursive role-filler).
        `polarity` (AFFIRM/NEGATE) is OPTIONAL (adds a binding -> more load), only for yes/no facts."""
        fact = {"agent": agent, "action": action}
        if isinstance(patient, Clause):
            fact["patient"] = patient
        elif isinstance(patient, tuple):                       # (adj(s), noun) -- an attributed entity
            adjs, noun = patient
            adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
            fact["patient"] = noun
            fact["attribute"] = adjs[0]
            if len(adjs) > 1:
                fact["attribute2"] = adjs[1]
        else:
            fact["patient"] = patient
        if polarity is not None:
            fact["polarity"] = polarity
        self.kb.append((fact, self.bind_fact(fact)))

    def query_patient(self, agent, action):
        """'what does <agent> <action>?' -> the patient of the matching fact: a concept word, an ATTRIBUTED entity
        ('big apple' / 'big red ball'), or a rendered embedded clause. None if no fact's agent matches the cue
        (abstention -- the no-confab moat). The noun + each adjective are decoded from the spiking unbind; the
        stored structure only routes the rendering."""
        for fact, bound in self.kb:
            if self.unbind(bound, "agent") == agent and self.unbind(bound, "action") == action:
                noun = self._render_filler(bound, "patient", fact["patient"])
                adjs = [self.unbind(bound, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    adjs = sorted(adjs, key=self.words.index)     # canonical vocabulary order (set, not sequence)
                    return " ".join(adjs + [noun])
                return noun
        return None

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> the agent of the matching fact; None if no match."""
        for fact, bound in self.kb:
            if self.unbind(bound, "action") == action and self.unbind(bound, "patient") == patient:
                return self.unbind(bound, "agent")
        return None

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown' via the bound POLARITY tag."""
        for fact, bound in self.kb:
            if (self.unbind(bound, "agent") == agent and self.unbind(bound, "action") == action
                    and self.unbind(bound, "patient") == patient):
                return "yes" if self.unbind(bound, "polarity", self.pol_words) == "AFFIRM" else "no"
        return "unknown"

    def render_fact(self, agent):
        """Generation: render a full stored sentence whose agent matches `agent` -- e.g. 'dog go north' -- with the
        action + patient DECODED from the spiking unbind (not the stored labels). None if no fact's agent matches
        (the no-confab moat: the agent does not invent a sentence about an unknown subject)."""
        for fact, bound in self.kb:
            if self.unbind(bound, "agent") == agent:
                action = self.unbind(bound, "action")
                patient = self._render_filler(bound, "patient", fact["patient"])
                return f"{agent} {action} {patient}"
        return None
