"""Content-selection Milestone 2 (Approach 3), cheap-first load-bearing test: can a SPIKING dlPFC
working-memory region hold a fading multi-concept conversational context the way the structured
ContextBuffer (Milestone 1) does?

Reuses the project's dlPFC working-memory configuration (recurrent self-excitation + NMDA bistability
for persistent activity; see g11_bg_runner.py dlpfc_wm) via a minimal one-region bridge. Drives a
sequence of concept patterns into the dlPFC region, reads the sustained firing after each turn, and
decodes which concepts are active (cosine of the firing pattern to each concept pattern). The
load-bearing question: is the read-out context a FADING SUPERPOSITION -- the most recently driven
concept strongest, earlier ones present but faded -- which is exactly what the structured context
buffer provides. If yes, the spiking dlPFC can replace the structured buffer (Milestone 2 proceeds);
if no, characterize the limit honestly.

Reuse-by-import only; no protected-module edits. GPU (CuPy) when available, else NumPy.

  python -m research.runners.content_selection_spiking --seed 42
"""
from __future__ import annotations
import argparse
import numpy as np


def generate_concept_patterns(n_concepts, n_pfc, pattern_size, seed=42):
    """Each concept = a distinct random subset of `pattern_size` dlPFC neuron indices (sparse,
    near-orthogonal codes -- distinct concepts overlap little)."""
    rng = np.random.default_rng(seed)
    return {i: rng.choice(n_pfc, size=pattern_size, replace=False) for i in range(n_concepts)}


def build_dlpfc_context_bridge(n_pfc=500, pfc_density=0.2, seed=42, plastic_recurrence=False,
                               hebbian=False, exc_weight=2.0, verbose=True):
    """Minimal region-framework bridge with a single recurrent NMDA-enabled dlPFC working-memory region,
    using the project's validated dlpfc_wm configuration. plastic_recurrence + hebbian enable attractor
    formation (shaping the recurrence so concept patterns self-sustain)."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronType
    region = BrainRegion(
        name="dlpfc_wm", n_neurons=n_pfc, exc_fraction=0.8, internal_density=pfc_density,
        exc_weight_mean=exc_weight, inh_weight_mean=4.0, weight_jitter=0.2,
        plastic_internal=plastic_recurrence,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [region]
    cfg.region_pathways = []
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True            # global NMDA on; only dlPFC (enable_nmda=True) gets bistability
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = bool(hebbian)
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 10.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        print(f"[dlpfc context bridge] {n_pfc} dlPFC neurons, recurrent density {pfc_density}, NMDA on",
              flush=True)
    return bridge


def build_loop_wm_bridge(n=400, density=0.1, loop_weight=4.0, loop_density=0.15, seed=42,
                         plastic_loop=False, hebbian=False, enable_ou=True, verbose=True):
    """Two mutually-exciting regions forming a cortico-PFC LOOP (cortex_ctx <-> dlpfc_wm), both NMDA-
    enabled. The hypothesis (from the Milestone-2 standalone-region negative): persistent activity is
    sustained by reverberation around the loop, which a single recurrent region cannot do. With
    plastic_loop + hebbian, the loop connections can be SHAPED into pattern-specific attractors."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    def reg(name):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=density,
                           exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                           plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [reg("cortex_ctx"), reg("dlpfc_wm")]
    cfg.region_pathways = [
        RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=plastic_loop),
        RegionPathway(from_region="dlpfc_wm", to_region="cortex_ctx", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=plastic_loop),
    ]
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    # OU background noise tips over-eager bistable concept attractors into spurious ON states once the
    # network is excited by holding >=2 concepts (Hopfield spurious-state behavior; see finding doc
    # 2026-06-03). enable_ou=False gives a quiet, noise-robust multi-concept hold. Default True keeps
    # biological realism for single-concept / characterization use.
    cfg.enable_ou_process = bool(enable_ou)
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = bool(hebbian)
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 30.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        print(f"[loop WM bridge] cortex_ctx<->dlpfc_wm loop, {n} neurons each, loop weight {loop_weight}, "
              f"NMDA on, OU={'on' if enable_ou else 'off'}", flush=True)
    return bridge


class SpikingContextBuffer:
    """Spiking analogue of the Milestone-1 ContextBuffer: the dlPFC region's sustained firing IS the
    discourse context. drive() injects a concept pattern and lets NMDA recurrence sustain it; read()
    samples the sustained firing; decode() maps it back to active concepts via cosine to the patterns."""

    def __init__(self, bridge, patterns, drive_pA=2500.0, stim_steps=50, settle_steps=20):
        import sim.backend as B
        self.B = B
        self.xp, _ = B.get_backend()
        self.bridge = bridge
        self.patterns = patterns
        self.drive_pA = drive_pA
        self.stim_steps = stim_steps
        self.settle_steps = settle_steps
        idx = bridge.region_manager.indices("dlpfc_wm")
        self.idx = self.xp.asarray(idx)
        self.n = len(idx)

    def drive(self, concept_id):
        xp = self.xp
        full = self.bridge.cp_external_input_current
        pat = self.idx[xp.asarray(self.patterns[concept_id])]
        for _ in range(self.stim_steps):
            full[:] = 0.0
            full[pat] = self.drive_pA
            self.bridge._run_one_simulation_step()
        full[:] = 0.0
        for _ in range(self.settle_steps):   # let the drive stop and NMDA recurrence take over
            self.bridge._run_one_simulation_step()

    def read(self, window=20):
        xp = self.xp
        acc = xp.zeros(self.n, dtype=xp.float32)
        for _ in range(window):
            self.bridge.cp_external_input_current[:] = 0.0
            self.bridge._run_one_simulation_step()
            acc += self.bridge.cp_firing_states[self.idx].astype(xp.float32)
        return self.B.to_host(acc)

    def decode(self, activity):
        a = activity / (np.linalg.norm(activity) + 1e-9)
        out = {}
        for cid, pat in self.patterns.items():
            v = np.zeros(self.n, dtype=np.float32)
            v[pat] = 1.0
            v /= (np.linalg.norm(v) + 1e-9)
            out[cid] = float(a @ v)
        return out


class SpikingLoopContextBuffer:
    """Spiking working-memory context buffer (Milestone 2): a cortico-PFC loop with a pattern-specific
    attractor installed per concept. update() drives concepts (each held by its attractor); read()
    decodes the held SET (which concepts are currently active). The spiking analogue of the structured
    ContextBuffer -- the discourse context held as concept attractors reverberating in the loop.

    Validated mechanism (this module's findings): the loop holds the specific driven concept (220x
    specificity) and a SET of >=3 concepts at once (the WM span). Attractor weights are SET here
    (outer-product); learning them with the correct rule is the documented next step."""

    def __init__(self, concepts, n=600, pattern_size=50, attractor_weight=50.0, loop_weight=0.0,
                 internal_density=0.1, seed=42, enable_ou=True, verbose=False):
        import sim.backend as B
        self.B = B
        self.xp, _ = B.get_backend()
        self.concepts = list(concepts)
        # loop_weight=0 -> the installed concept attractors are the ONLY loop connections (no generic
        # random reverberation to bleed driven patterns into undriven ones -> less cross-talk).
        # internal_density=0 -> no random within-region recurrence coupling separate attractors.
        # enable_ou=False -> quiet hold; prevents OU noise from tipping bistable attractors into spurious
        # states when holding a multi-concept set (the validated clean-WM config; see finding doc).
        self.bridge = build_loop_wm_bridge(n=n, density=internal_density, loop_weight=loop_weight,
                                           loop_density=0.05, seed=seed, enable_ou=enable_ou,
                                           verbose=verbose)
        rm = self.bridge.region_manager
        cidx = np.asarray(rm.indices("cortex_ctx"))
        didx = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        self._cpat = {}
        self._dpat = {}   # dlPFC assembly indices per concept (for Milestone-3 spiking spreading-activation)
        for i, c in enumerate(self.concepts):
            p = perm[i * pattern_size:(i + 1) * pattern_size]
            cpat, dpat = cidx[p], didx[p]
            self._cpat[c] = self.xp.asarray(cpat)
            self._dpat[c] = self.xp.asarray(dpat)
            pre1 = np.repeat(cpat, pattern_size).astype(np.int64)
            post1 = np.tile(dpat, pattern_size).astype(np.int64)
            pre2 = np.repeat(dpat, pattern_size).astype(np.int64)
            post2 = np.tile(cpat, pattern_size).astype(np.int64)
            ww = np.full(pattern_size * pattern_size, attractor_weight, np.float32)
            self.bridge.set_pathway_weights("c2d", pre_indices=pre1, post_indices=post1, weights=ww,
                                            add_missing=True)
            self.bridge.set_pathway_weights("d2c", pre_indices=pre2, post_indices=post2, weights=ww,
                                            add_missing=True)
        self._psize = pattern_size

    def update(self, concepts, drive_pA=2500.0, stim=40, settle=15):
        for c in concepts:
            if c not in self._cpat:
                continue
            drv = self._cpat[c]
            for _ in range(stim):
                self.bridge.cp_external_input_current[:] = 0.0
                self.bridge.cp_external_input_current[drv] = drive_pA
                self.bridge._run_one_simulation_step()
            self.bridge.cp_external_input_current[:] = 0.0
            for _ in range(settle):
                self.bridge._run_one_simulation_step()

    def read(self, window=20):
        """Decode the held set: per-neuron firing of each concept's attractor over a no-drive window."""
        acc = {c: 0.0 for c in self.concepts}
        for _ in range(window):
            self.bridge.cp_external_input_current[:] = 0.0
            self.bridge._run_one_simulation_step()
            fs = self.bridge.cp_firing_states
            for c in self.concepts:
                acc[c] += float(self.B.to_host(fs[self._cpat[c]]).sum())
        return {c: acc[c] / (self._psize * window) for c in self.concepts}


class SpikingController:
    """Milestone-2 end-to-end (Approach 3): content-selection Control whose discourse CONTEXT is held in
    the spiking cortico-PFC loop. Per turn: drive the input into the spiking context; read the held set;
    select the most relevant unsaid associate (reusing the validated relevance + inhibition-of-return);
    drive the selection into the spiking context. The faithful spiking analogue of the Milestone-1
    ContentSelectionController -- same selection logic, but the context is real spiking working memory."""

    def __init__(self, graph, seed=42, lam=1.0, said_decay=0.6, internal_density=0.0, enable_ou=False,
                 verbose=False):
        from research.runners.content_selection import SaidTrace
        self.graph = graph
        self._vocab = sorted(set(graph) | {a for v in graph.values() for a in v})
        # VALIDATED multi-seed config (2026-06-03): internal_density=0.0 + enable_ou=False -> 6/6 seeds.
        # Root cause of the earlier 2/3-seed fragility (now SOLVED): holding >=2 concepts raises global
        # excitability enough that the seeded OU background noise tips OTHER concepts' over-eager bistable
        # attractors into spurious ON states (classic Hopfield spurious-state/capacity behavior); those
        # then hijack the relevance-based selection, seed-dependently. The random recurrence
        # (internal_density>0) adds further spurious cross-talk. Removing BOTH (clean within-concept
        # attractors + quiet hold) yields an EXACT multi-concept WM -> robust coherent selection across
        # all seeds. (The earlier "density=0 alone -> 3/6 WORSE" note was confounded: density=0 still
        # leaves OU-driven spurious; it is density=0 AND OU-off TOGETHER that resolves it.)
        n = max(600, 60 * len(self._vocab))
        self.ctx = SpikingLoopContextBuffer(self._vocab, n=n, internal_density=internal_density,
                                            seed=seed, enable_ou=enable_ou, verbose=verbose)
        self.said = SaidTrace(decay=said_decay)
        self.lam = lam

    def turn(self, user_concepts):
        from research.runners.content_selection import select_candidate
        self.ctx.update(list(user_concepts))            # spiking working memory holds the input
        self.said.step()
        held = self.ctx.read()                          # the held set = the discourse context
        candidates = [c for c in self._vocab if c not in set(user_concepts)
                      and self.said.activation(c) < 0.5]   # hard inhibition-of-return
        said_now = {c: self.said.activation(c) for c in candidates}
        choice = select_candidate(candidates, held, self.graph, said_now, lam=self.lam)
        if choice is not None:
            self.said.mark(choice)
            self.ctx.update([choice])
        return choice


class SpikingSpreadingController:
    """Milestone 3: the RELEVANCE computation is itself spiking. The association graph is embodied as
    inter-assembly synapses (cortex_A -> dlpfc_B at weight proportional to graph[A][B]); driving the
    discourse context into the spiking working memory SPREADS activation along those synapses to the
    associated concept assemblies, and the most-active candidate assembly IS the selection. This is the
    faithful spiking analogue of the structured relevance sum (Sum_c context[c]*graph[c][candidate]) in
    Milestone 1/2 -- there the relevance was computed in numpy over the graph; here it is computed by
    spreading spikes through learned-style associative synapses (cortico-cortical spreading activation).
    Inhibition-of-return is the structured SaidTrace (making it spiking, e.g. spike-frequency adaptation
    on the selected assembly, is the documented Milestone-3b step).

    Validated cheap-first (2026-06-03): driving 'apple' lights apple's cluster (big/cat/hot ~0.32) while
    the unrelated dog-cluster stays at 0.00 -- the spreading reproduces the relevance ranking cleanly and
    seed-robustly. Uses the same clean-dynamics config as the validated SpikingController
    (internal_density=0 + enable_ou=False) so the multi-concept hold stays exact."""

    def __init__(self, graph, seed=42, said_decay=0.9, edge_scale=60.0, internal_density=0.0,
                 verbose=False):
        # said_decay 0.9 (was 0.6): the inhibition-of-return must keep a just-said concept excluded for
        # SEVERAL turns so the dialogue PROGRESSES through the topic instead of alternating two direct
        # neighbours (decay 0.6 excluded only ~1 turn -> cloud/storm/cloud/storm). At 0.9 the decisive
        # coherence eval RESOLVES: the spiking Control beats the no-control baseline on on_topic + turn_to
        # _turn AND progression=1.00 (rain -> cloud,storm,wind,sky,sun); validated 2026-06-03.
        # edge_scale 60 (was 20): the spread must be strong enough that EVERY designed associate latches.
        # At edge_scale=20 some seeds lit only the first associate (the within-cluster `None` blemish:
        # apple -> pear only, plum/grape stayed at 0.0); edge_scale=60 lights all in-cluster associates
        # (validated 2026-06-03: 8-concept 6/6 + 16-concept 12/12 STRICT, no off-topic risk since there
        # are no cross-cluster edges). See finding doc scaling section.
        from research.runners.content_selection import SaidTrace
        self.graph = graph
        self._vocab = sorted(set(graph) | {a for v in graph.values() for a in v})
        n = max(600, 60 * len(self._vocab))
        self.ctx = SpikingLoopContextBuffer(self._vocab, n=n, internal_density=internal_density,
                                            seed=seed, enable_ou=False, verbose=verbose)
        self._install_graph_edges(edge_scale)
        self.said = SaidTrace(decay=said_decay)

    def _install_graph_edges(self, scale):
        """Embody each association A->B as cortex_A -> dlpfc_B synapses (weight = graph[A][B]*scale), so
        that A firing drives B's assembly (B's within-concept attractor then sustains it). Only DESIGNED
        associations get a synaptic path -> spreading stays on the association graph, never into unrelated
        concepts (clean by construction)."""
        ps = self.ctx._psize
        for A in self.graph:
            for B, w in self.graph[A].items():
                cA = self.ctx.B.to_host(self.ctx._cpat[A])
                dB = self.ctx.B.to_host(self.ctx._dpat[B])
                pre = np.repeat(cA, ps).astype(np.int64)
                post = np.tile(dB, ps).astype(np.int64)
                ww = np.full(ps * ps, float(w) * scale, np.float32)
                self.ctx.bridge.set_pathway_weights("c2d", pre_indices=pre, post_indices=post,
                                                    weights=ww, add_missing=True)

    def turn(self, user_concepts):
        """Drive the discourse context -> spreading activation lights associated assemblies -> the
        most-active unsaid candidate assembly IS the selection (relevance computed in spikes)."""
        self.ctx.update(list(user_concepts))               # drive context; spreading lights associates
        self.said.step()
        held = self.ctx.read()                             # per-assembly firing = spiking relevance
        uc = set(user_concepts)
        cands = [(c, held[c]) for c in self._vocab
                 if c not in uc and self.said.activation(c) < 0.5 and held[c] > 0.02]
        if not cands:
            return None
        choice = max(cands, key=lambda kv: kv[1])[0]       # winner-take-all over spread activation
        self.said.mark(choice)
        return choice

    def _reset_wm(self):
        """Active WM reset (inter-probe interval): return the network to rest so a fresh probe is not
        contaminated by the previous probe. Clears membrane v + recovery u + firing + conductances AND the
        in-flight cross-step state that the earlier best-effort version missed: `cp_prev_firing_states`,
        `cp_refractory_timers`, and the synaptic pulse timers/progress that carry DELAYED transmission
        between steps. With these cleared, repeated latency probes on one bridge are clean -- validated
        2026-06-03: multi-turn latency selection on a richly-connected graph stays on-topic 6/6 (vs the
        best-effort reset which drifted off-topic on turns 2-3). This substantially resolves M3b obstacle 3
        (clean inter-probe reset)."""
        b = self.ctx.bridge
        for a in ("cp_firing_states", "cp_prev_firing_states"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                  "cp_conductance_g_nmda_rise", "cp_refractory_timers", "cp_synapse_pulse_timers",
                  "cp_synapse_pulse_progress"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0
        if getattr(b, "cp_izh_vr", None) is not None and b.cp_membrane_potential_v is not None:
            b.cp_membrane_potential_v[:] = b.cp_izh_vr
        if getattr(b, "cp_recovery_variable_u", None) is not None:
            b.cp_recovery_variable_u[:] = 0.0

    def turn_latency(self, user_concepts):
        """Latency-based selection: relevance = first-spike LATENCY of the spread (focused 1-hop), robust
        on richly-connected graphs where the sustained-rate `turn()` over-spreads multi-hop and loses
        topic focus. Each turn is a fresh latency probe (fuller `_reset_wm`), so there is no cross-turn
        accumulation; inhibition-of-return is the structured SaidTrace. Validated 2026-06-03: on the
        connected multi-topic graph the earliest-latency pick is a DIRECT neighbour 6/6, and 3-turn chains
        stay within the 2-hop topic region 6/6 -- where the rate read wandered off-topic (rain -> dog)."""
        self.said.step()
        lat = self.relevance_by_latency(user_concepts[0])
        uc = set(user_concepts)
        cands = [(c, lat[c]) for c in self._vocab
                 if c not in uc and self.said.activation(c) < 0.5 and lat[c] is not None]
        if not cands:
            return None
        choice = min(cands, key=lambda kv: kv[1])[0]       # earliest first-spike = most relevant (1-hop)
        self.said.mark(choice)
        return choice

    def relevance_by_latency(self, context_concept, steps=60, drive_pA=2500.0, thresh=0.15):
        """Spiking relevance as first-spike LATENCY of the spread -- richer than the rate read: it encodes
        graph DISTANCE (direct associates fire earlier than indirect ones; unrelated concepts never fire).
        Resets the WM, drives the context concept, and returns {concept: first-spike step} (None = never
        fired = unrelated). Validated seed-robust (2026-06-03): driving 'apple' -> direct big/cat earliest
        (~8-19), 2-hop hot later (~16-20), unrelated dog-cluster never. This is a faithful spiking analogue
        of graph spreading-activation with distance encoded in spike timing (latency/rank-order coding).

        NOTE the honest boundary (M3b): latency naturally ranks DIRECT associates before INDIRECT ones, and
        an indirect concept (reached via a direct one) can never out-race its own upstream -> latency-based
        inhibition-of-return cannot reach indirect associates by delay alone; full-cluster non-repetition
        still needs exclusion (the structured SaidTrace). See finding doc."""
        self._reset_wm()
        drv = self.ctx._cpat[context_concept]
        first = {c: None for c in self._vocab}
        for t in range(steps):
            cur = self.ctx.bridge.cp_external_input_current
            cur[:] = 0.0
            cur[drv] = drive_pA
            self.ctx.bridge._run_one_simulation_step()
            fs = self.ctx.bridge.cp_firing_states
            for c in self._vocab:
                if first[c] is None and \
                        float(self.ctx.B.to_host(fs[self.ctx._cpat[c]]).sum()) / self.ctx._psize > thresh:
                    first[c] = t
        return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pfc", type=int, default=500)
    ap.add_argument("--pattern-size", type=int, default=50)
    a = ap.parse_args()

    bridge = build_dlpfc_context_bridge(n_pfc=a.n_pfc, seed=a.seed)
    patterns = generate_concept_patterns(3, a.n_pfc, a.pattern_size, seed=a.seed)
    scb = SpikingContextBuffer(bridge, patterns)

    print("Driving concepts c0 -> c1 -> c2 in sequence; reading the dlPFC context after each turn.")
    print("Expect a FADING SUPERPOSITION: the just-driven concept strongest, earlier ones present but")
    print("decaying.")
    history = []
    for c in [0, 1, 2]:
        scb.drive(c)
        act = scb.read()
        ctx = scb.decode(act)
        history.append(ctx)
        print(f"  after driving c{c}:  raw_firing_sum={act.sum():.0f}  " +
              "  ".join(f"c{k}={ctx[k]:+.2f}" for k in sorted(ctx)))

    # Load-bearing checks on the final state (after c0,c1,c2):
    final = history[-1]
    recent_strongest = final[2] >= final[1] >= final[0] - 1e-6   # c2 >= c1 >= c0 (fading order)
    recent_active = final[2] > 0.15                              # the just-driven concept is clearly present
    print("\nload-bearing checks (final context after c0,c1,c2):")
    print(f"  fading order c2>=c1>=c0 : {recent_strongest}   (c2={final[2]:.2f} c1={final[1]:.2f} c0={final[0]:.2f})")
    print(f"  most-recent clearly present (c2>0.15): {recent_active}")
    if recent_strongest and recent_active:
        print("\nVERDICT: RESOLVES -- the spiking dlPFC holds a fading multi-concept context like the "
              "structured buffer -> wire it into the controller (replace ContextBuffer) and re-run the "
              "Milestone-1 coherence eval.")
    else:
        print("\nVERDICT: does-not-cleanly-hold -- characterize: the spiking dlPFC does not reproduce the "
              "fading-superposition behavior at these params; tune drive/density/decay or report the limit.")


if __name__ == "__main__":
    main()
