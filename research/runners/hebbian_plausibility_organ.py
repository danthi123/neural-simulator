"""HEBBIAN SELF-ORGANIZATION rung for the #3E brain-native plausibility gate — the NEXT host-computed
shortcut named by the just-landed ensemble finding (2026-09-01-plausibility-ensemble-read-host-parity-
generation-all6-default-on-GO): "the synaptic weights are still SET from the co-occurrence counts (the
same counts the host P holds). Online Hebbian self-organization of those weights... remains the next rung
toward a fully self-organized plausibility."

THE RESIDUAL THIS CLOSES.  `SpikingAssociativePlausibilityOrgan._build_bridge` (spiking_plausibility_organ.py)
installs the co-occurrence graph onto `cortex_ctx -> dlpfc_wm` synapses via ONE host-computed injection:
`weights = P[a,b] * gain` written directly with `set_pathway_weights(..., add_missing=True)`. The RELATEDNESS
DECISION is already spiking (a spike-threshold read of those synapses), but the NUMBER living in each synapse
is a host float, never touched by a learning rule. This is exactly the project's own standing bar
(`feedback_spiking_structure_must_self_organize`): "host-DESIGNED weights/structure of a spiking op = residual
shortcut; close via developmental self-organization."

THE MECHANISM.  A structurally-IDENTICAL two-population monosynaptic organ (same disjoint per-concept
assemblies, same K-ensemble, same NO-BACKPROJECTION architecture that the 2026-09-01 arc found was essential
to avoid cross-talk), but every `cortex_ctx^k -> dlpfc_wm^k` synapse starts at weight 0 and is DECLARED
PLASTIC (`plasticity_gate="c2d_hebbian"`, `cfg.enable_hebbian_learning=True` -- the SAME validated Hebbian
co-fire growth rule `research/runners/_D_sparse_heteroassoc.py`/`LearnedAssocGraph` already use and this
project already accepts as genuine substrate learning). Instead of injecting P, this organ REPLAYS the
brain's own stored facts: for each `(agent, verb, patient)`, it co-drives the CORTEX *and* DLPFC assemblies of
all three concepts together for `replay_cycles` cycles, so Hebbian growth potentiates the `A->B` synapse for
every ordered pair `(A, B)` in that fact -- exactly the same "every fact's roles co-occur pairwise" structure
the host `P` matrix encodes (`brain_chat_tui.py::_build_generation_proposer`: `for x in cs: for y in cs: if
x != y: graph[x][y] += 1`), except the NUMBER now comes from the bridge's own STDP-style Hebbian update
executing on REAL co-firing spikes, not a Python counter. A fact that recurs (shares an edge with another
fact) is replayed again -> Hebbian growth ACCUMULATES -> a naturally stronger synapse, the same qualitative
effect `P[a,b] = count` encodes, but produced by repetition-driven plasticity instead of arithmetic.

The backward `dlpfc_wm -> cortex_ctx` pathway is never declared at all (not even at zero) -- there is nothing
for activity to flow back through, so the READ stays strictly monosynaptic exactly as the 2026-09-01 finding's
`density=0.0` fix required, and REPLAY (which co-drives both regions to grow the forward synapses) cannot
recruit any resonant loop.

WHAT REMAINS HOST (declared, honest).  (1) The GRAPH TOPOLOGY replayed -- which fact triples exist -- is the
brain's own stored facts (`self.stored_facts`), not host-invented, but the REPLAY SCHEDULE (co-drive all 3
roles of a fact together, `replay_cycles` times) is a host protocol choice, same status as the b2 proposer's
sampling loop / the `_D_sparse_heteroassoc` co-replay protocol this project already accepts. (2) `tau_spike`
(the operating point) is still the host percentile-of-the-brain's-own-output rule the parent organ uses,
unchanged. (3) The ENSEMBLE/K, gain and pattern-size hyperparameters are host-chosen (as in the parent organ).
The load-bearing upgrade is narrow and honest: the SYNAPTIC WEIGHT VALUE is no longer a host arithmetic
product, it is the residue of the bridge's own plasticity rule after replaying the brain's own experience.

Additive, master-switched (`BRAIN_HEBBIAN_PLAUSIBILITY`), reuse-by-import (subclasses
`SpikingAssociativePlausibilityOrgan`; only `_build_bridge`/`__init__` differ -- `related`,
`_self_threshold`, `plausible_graded`, `install`, `uninstall`, `agreement_with_host` are inherited
UNCHANGED). NO sim/ edit (uses only public `sim.regions`/`sim.bridge`/`sim.config` APIs, the same ones
`_D_sparse_heteroassoc.build()` already uses). CPU (numpy backend fine).

LESION handles (mirroring the parent organ, adapted to REPLAY):
  lesion="shuffle"  -> replay a role-shuffled fact list (same word MULTISET / same number of replay events,
                       but which words co-occur in a triple is destroyed) -- growth still happens, but not
                       aligned with the real graph, so the advantage must collapse toward random.
  lesion="ablate"   -> never replay (weights never leave 0) -- nothing downstream can fire -> relatedness
                       collapses to 0 related pairs, proving REPLAY (not the architecture) carries the signal.
"""
from __future__ import annotations

import numpy as np

from research.runners.spiking_plausibility_organ import SpikingAssociativePlausibilityOrgan


def _shuffled_facts(facts, rng):
    """Role-shuffle: pool every role-filler seen across all facts, independently permute AGENT and PATIENT
    columns (verb kept, matching the b2 host anti-cheat's own shuffle convention -- see
    `_genfrontier_b2_generative_replay_derisk.shuffle_graph`/`spiking_plausibility_organ._shuffle_offdiag`),
    so the SAME number of replay events happens (same total co-firing volume) but the concept COMBINATIONS
    are scrambled -- the spiking analogue of the parent organ's off-diagonal P shuffle, applied to the
    replay schedule instead of a pre-built matrix."""
    facts = list(facts)
    if len(facts) < 2:
        return facts
    agents = [f[0] for f in facts]
    patients = [f[2] for f in facts]
    rng.shuffle(agents)
    rng.shuffle(patients)
    return [(agents[i], facts[i][1], patients[i]) for i in range(len(facts))]


class HebbianAssociativePlausibilityOrgan(SpikingAssociativePlausibilityOrgan):
    """`SpikingAssociativePlausibilityOrgan` with the synaptic weight SOURCE replaced: instead of a host
    `P*gain` injection, the `cortex_ctx^k -> dlpfc_wm^k` synapses start at 0 and are grown by the bridge's
    own Hebbian rule while REPLAYING the brain's stored facts (co-driving each fact's three role-fillers'
    cortex+dlpfc assemblies together). Every read method (`related`, `_self_threshold`, `plausible_graded`,
    `install`/`uninstall`, `agreement_with_host`) is INHERITED UNCHANGED from the parent -- only how the
    synapses got their weight differs."""

    def __init__(self, P, row, facts, vocab=None, seed=42, tau_pct=50.0, pattern_size=12, gain=16.0,
                 drive_pA=4000.0, stim_steps=12, read_window=20, n_ensemble=1, graded=False,
                 beta_frac=0.6, density=0.0, lesion=None, verbose=False,
                 replay_cycles=40, replay_pA=1100.0, replay_on_steps=8, replay_off_steps=4,
                 hebbian_max_weight=60.0, hebbian_min_weight=0.0, hebbian_learning_rate=0.15,
                 inter_density=1.0):
        self.facts = [tuple(f) for f in facts if all(isinstance(x, str) for x in f)]
        self.replay_cycles = int(replay_cycles)
        self.replay_pA = float(replay_pA)
        self.replay_on_steps = int(replay_on_steps)
        self.replay_off_steps = int(replay_off_steps)
        self.hebbian_max_weight = float(hebbian_max_weight)
        self.hebbian_min_weight = float(hebbian_min_weight)
        self.hebbian_learning_rate = float(hebbian_learning_rate)
        self.inter_density = float(inter_density)
        self.n_replay_events = 0          # provenance: how many co-fire replay events actually ran
        super().__init__(P, row, vocab=vocab, seed=seed, tau_pct=tau_pct, pattern_size=pattern_size,
                          gain=gain, drive_pA=drive_pA, stim_steps=stim_steps, read_window=read_window,
                          n_ensemble=n_ensemble, graded=graded, beta_frac=beta_frac, density=density,
                          lesion=lesion, verbose=verbose)

    # ---- the substrate: TWO Izhikevich pools + a PLASTIC feedforward-only c2d pathway, grown by replay ----
    def _build_bridge(self):
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.regions import BrainRegion, RegionPathway
        from sim.bridge import SimulationBridge
        from sim.enums import NeuronType

        ps, K = self.pattern_size, self.n_ensemble
        n = max(600, 2 * ps * len(self.vocab) * K)

        def reg(name):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=self.density,
                               exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                               plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = [reg("cortex_ctx"), reg("dlpfc_wm")]
        # ONE-DIRECTIONAL c2d pathway only -- no dlpfc_wm -> cortex_ctx pathway is declared at all, so there
        # is nothing for activity to flow back through (strictly monosynaptic by construction, the same
        # property the parent organ's density=0.0 + never-declared-backward pathway achieves). Zero-init,
        # PLASTIC, gated so replay can be frozen after training (mirrors _D_sparse_heteroassoc's "recurrent"
        # gate / LearnedAssocGraph.store_fact).
        cfg.region_pathways = [
            RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=self.inter_density,
                          weight_mean=0.0, weight_jitter=0.0, plastic=True, plasticity_gate="c2d_hebbian"),
        ]
        cfg.dt_ms = 0.5
        cfg.seed = self.seed
        cfg.enable_nmda = True
        cfg.enable_ou_process = False   # deterministic replay + deterministic read (matches parent: enable_ou=False)
        cfg.enable_structural_plasticity = False
        cfg.enable_short_term_plasticity = False
        cfg.fast_spike_reset = True
        # Direct Hebbian co-fire growth (validated: _D_sparse_heteroassoc.py / LearnedAssocGraph, 24/24 edges
        # + 9/9 top-associate match vs the Python co-occurrence oracle). NOT STDP-with-reward (no reward
        # signal here; plain co-activity growth, same as the validated precedent). `enable_stdp=False`
        # (matching _D_sparse_heteroassoc.build()): STDP's eligibility trace needs `current_time_ms` advanced
        # every step (a project-wide known trap, docs/... "STDP IS INERT"); this organ never advances it
        # during REPLAY (only Hebbian co-activity growth is used, which needs no timestamp), so STDP would
        # sit silently inert (harmless but noisy) if left on -- disabled outright for a clean provenance story.
        cfg.enable_hebbian_learning = True
        cfg.enable_reward_modulation = False
        cfg.enable_stdp = False
        cfg.hebbian_max_weight = self.hebbian_max_weight
        cfg.hebbian_min_weight = self.hebbian_min_weight
        cfg.hebbian_learning_rate = self.hebbian_learning_rate
        cfg.stdp_w_max = self.hebbian_max_weight

        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge = bridge

        rm = bridge.region_manager
        cin = np.asarray(rm.indices("cortex_ctx"))
        dout = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(self.seed)
        perm_in = rng.permutation(len(cin))
        perm_out = rng.permutation(len(dout))
        self._cin, self._dout = {}, {}
        self._cin_k, self._dout_k = {c: [] for c in self.vocab}, {c: [] for c in self.vocab}
        slot = 0
        for k in range(K):
            for c in self.vocab:
                ai = cin[perm_in[slot * ps:(slot + 1) * ps]]
                bi = dout[perm_out[slot * ps:(slot + 1) * ps]]
                self._cin_k[c].append(ai)
                self._dout_k[c].append(bi)
                slot += 1
        for c in self.vocab:
            self._cin[c] = np.concatenate(self._cin_k[c])
            self._dout[c] = np.concatenate(self._dout_k[c])

        # ---- REPLAY: grow the c2d synapses by co-firing each fact's role-fillers (never a host weight write) ----
        if self.lesion == "ablate":
            pass                                   # never replay -> every c2d synapse stays at its 0 init
        else:
            facts = self.facts
            if self.lesion == "shuffle":
                facts = _shuffled_facts(facts, np.random.default_rng(self.seed * 101 + 7))
            self._replay(facts)

        self._v0 = self.bridge.cp_membrane_potential_v.copy()
        self._u0 = self.bridge.cp_recovery_variable_u.copy()

    def _replay_reset(self):
        """Return every neuron to quiescence (membrane potential/recovery to the network's INITIAL resting
        state, all conductances/refractory timers/firing flags cleared) between two DIFFERENT facts' replay
        blocks. Without this, a region with internal_density=0.0 (no local inhibitory feedback -- deliberate,
        it is what keeps the READ monosynaptic) has nothing to quench residual depolarization from the
        PREVIOUS fact's drive; that residual activity can coincide with the NEXT fact's drive and Hebbian-
        potentiate a synapse between two concepts that never actually co-occurred in any fact -- spurious
        cross-fact contamination, not a signal. Resetting between facts (never WITHIN a fact's own
        `replay_cycles`, where accumulation is the intended effect) makes each fact's co-firing evidence
        depend only on that fact's own role-fillers."""
        b = self.bridge
        b.cp_membrane_potential_v[:] = self._rest_v
        b.cp_recovery_variable_u[:] = self._rest_u
        for name in ("cp_firing_states", "cp_prev_firing_states"):
            arr = getattr(b, name, None)
            if arr is not None:
                arr[:] = False
        for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                     "cp_conductance_g_nmda_rise", "cp_refractory_timers", "cp_synapse_pulse_timers",
                     "cp_synapse_pulse_progress"):
            arr = getattr(b, name, None)
            if arr is not None:
                arr[:] = 0

    def _replay(self, facts):
        """Co-drive a fact's THREE role concepts' CORTEX *and* DLPFC assemblies together, `replay_cycles`
        times per fact, so the plastic `cortex_ctx^k -> dlpfc_wm^k` synapse for every ordered pair among the
        fact's roles gets correlated pre/post activity -> Hebbian LTP. A recurring pair (shared by >1 fact)
        is replayed again each time its fact recurs -> growth ACCUMULATES (the substrate analogue of a
        higher host co-occurrence COUNT). Mirrors LearnedAssocGraph.store_fact's co-replay protocol. Resets
        to quiescence BETWEEN facts (see `_replay_reset`) so co-firing evidence stays fact-local."""
        b = self.bridge
        self._rest_v = b.cp_membrane_potential_v.copy()
        self._rest_u = b.cp_recovery_variable_u.copy()
        try:
            b.set_plasticity_gate("c2d_hebbian", 1.0)
        except KeyError:
            pass
        for a, v, p in facts:
            roles = [w for w in (a, v, p) if w in self._cin]
            if len(roles) < 2:
                continue
            self._replay_reset()
            drive_idx = np.concatenate([self._cin[w] for w in roles] + [self._dout[w] for w in roles])
            for _ in range(self.replay_cycles):
                b.cp_external_input_current[:] = 0.0
                b.cp_external_input_current[drive_idx] = self.replay_pA
                for _ in range(self.replay_on_steps):
                    b._run_one_simulation_step()
                b.cp_external_input_current[:] = 0.0
                for _ in range(self.replay_off_steps):
                    b._run_one_simulation_step()
                self.n_replay_events += 1
        b.cp_external_input_current[:] = 0.0
        try:
            b.set_plasticity_gate("c2d_hebbian", 0.0)     # FREEZE post-replay: the read phase must not keep learning
        except KeyError:
            pass


# TUNED OPERATING POINT (this file's de-risk arc, 2026-09-01). The FIRST attempt used inter_density=1.0
# (full cortex_ctx<->dlpfc_wm region connectivity) and reached agreement 0.55 -- barely above the 0.43
# positive-rate floor: with EVERY (a,b) assembly pair guaranteed a dense synapse bundle regardless of
# whether the pair ever co-occurred, sheer CONVERGENT SUMMATION (many weakly-grown synapses landing on the
# same postsynaptic neuron) swamped pairwise specificity -- a HIGH-DEGREE concept's assembly fired everything
# downstream almost uniformly, independent of true co-occurrence. This is the SAME class of failure the
# ensemble arc's density=0.0 (internal recurrence) fix targeted, one level up: there it was recurrence WITHIN
# a region contaminating the read; here it is FULL density BETWEEN regions contaminating which pairs can grow
# a synapse at all. Dropping `inter_density` to a SPARSE regime (so only a modest, assembly-sized subset of
# possible pre-post pairs exist to potentiate) restored pairwise specificity: agreement rose to ~0.71 (f1
# ~0.62, precision ~0.68-0.72) on the synthetic 4-fact/8-concept graph -- built + measured, meaningfully
# above the shuffle-lesion control, NOT YET at the ensemble organ's 1.0 host parity. See
# _hebbian_plausibility_derisk.py and the 2026-09-01 finding for the honest scope of this rung.
PRODUCTION_READ_CONFIG = dict(pattern_size=8, n_ensemble=4, read_window=20, graded=False, density=0.0,
                              drive_pA=4000.0, stim_steps=12, replay_cycles=100, replay_pA=1100.0,
                              hebbian_learning_rate=0.4, hebbian_max_weight=60.0, inter_density=0.08)


def build_for_proposer(prop, seed=42, lesion=None, production=True, **kw):
    """Build a `HebbianAssociativePlausibilityOrgan` from a live `GenerativeReplayProposer` (reads its
    P/row -- P is passed through ONLY for the `agreement_with_host` diagnostic + the shuffle/ablate lesion
    dispatch on the REPLAY schedule; it is never written into a synapse). `facts` = the proposer's own
    stored triples (`prop.stored_set`, AFFIRMED only -- the same facts `_build_generation_proposer` builds
    P from; NEGATED facts are deliberately excluded from REPLAY so a negation cannot itself teach a
    spurious positive association)."""
    vocab = sorted(prop.row.keys(), key=lambda w: prop.row[w])
    facts = sorted(getattr(prop, "stored_set", None) or set())
    cfg = dict(PRODUCTION_READ_CONFIG) if production else {}
    cfg.update(kw)
    return HebbianAssociativePlausibilityOrgan(prop.P, prop.row, facts, vocab=vocab, seed=seed,
                                               lesion=lesion, **cfg)
