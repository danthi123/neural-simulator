"""BiasedCompetitionContextBuffer — production multi-referent pronoun disambiguation via WTA biased competition.

This is the validated mechanism from `_phaseB_biased_competition_derisk.py`, promoted to importable production code
(de-risk GO: 2026-06-19-multireferent-biased-competition-derisk.md — 5/6 seeds on the strict GO-arm, all anti-cheat
controls 6/6; the single miss is an ABSTENTION, moat-preserving). The de-risk runner imports these symbols, so it
keeps working byte-faithfully; `MultiTurnAgent(enable_biased_competition=True)` routes its pronoun resolution through
this buffer.

THE MECHANISM. When the spiking working memory holds several discourse referents, which one does a bare pronoun
("it") bind to? The plain SpikingLoopContextBuffer holds each referent in an INDEPENDENT attractor with NO
cross-referent coupling, so neither recency (no position signal in the rate read) nor a salience boost (only ADDS
activity to an independent attractor; cannot SUPPRESS the competitor) can pick the right one — both were prior
NEGATIVEs (2026-06-17-multireferent-disambiguation-NEGATIVE.md). THE FIX (Desimone-Duncan 1995 biased competition;
Wong-Wang 2006 attractor WTA): MUTUAL INHIBITION between the held referent assemblies (each referent's assembly
drives a dedicated FS inhibitory pool that suppresses the OTHER referents' assemblies — the Rutishauser
selective-inhibition motif the navigation sel_X/sel_FS_X read-out already uses) + a small CONTENT-based top-down BIAS
(a feed-forward current into the referent whose features — animacy/number agreement with the pronoun + selectional
compatibility with the query verb — match). The crux: the bias is a CONTENT signal (NOT position=recency, NOT
magnitude=boost — the two already-disproven signals); the recurrence amplifies the small content asymmetry into a
SUPPRESSIVE winner.

THE WIRING (additive; NO sim/ edit). BiasedCompetitionContextBuffer wraps the validated loop bridge: it builds the
cortex_ctx<->dlpfc_wm loop PLUS, per referent, a Wong-Wang accumulator pool (sel_X) + a selective inhibitory pool
(sel_FS_X, exc_fraction=0.0 -> inhibitory traits -> out-synapses route to g_i), then installs, via
set_pathway_weights(add_missing):
  * cortex_assembly[X] -> sel_X               (EXCITATORY feed-forward EVIDENCE — read-only tap, no sel_X->cortex)
  * sel_X              -> sel_FS_X            (EXCITATORY: a winning accumulator recruits its interneuron)
  * sel_FS_X           -> sel_Y!=X           (INHIBITORY: that interneuron suppresses the OTHER referents)
plus a bias(concept, pA) injector that adds a SMALL feed-forward current into the favored sel_X during the
competitive read window. The held attractors + the holding update() are reused verbatim from
SpikingLoopContextBuffer.

>>> HOST-SCAFFOLD SHORTCUT (FLAGGED for conversion, BRAIN-BASED-ONLY) <<<
`content_bias_target` (+ the ANIMACY / VERB_SELECTS feature lexicons) is HOST-SIDE: given the pronoun's features and
the query verb's selectional restriction, it returns WHICH held referent receives the bias current. The WIN is
brain-based (spiking competition + suppression + the recurrence amplifying the small content asymmetry); the content
SCORING is host in this scaffold. The follow-on neuralizes it into a LEARNED SYNAPTIC FEATURE-COMPATIBILITY MAP
(pronoun-feature population x candidate-feature population -> bias current), so the bias itself is computed by
neurons/synapses. See `2026-06-19-multireferent-integration-multiturnagent.md` for the boundaries.

Two substrate facts found + handled in the original build (both diagnosed against `sim/bridge.py`, not assumed):
  1. The synapse E/I sign is the PRE-neuron's inhibitory trait (NOT the weight sign) -> the FS pools are
     exc_fraction=0.0 framework regions (every neuron inhibitory -> their out-synapses route to g_i).
  2. The plain loop does NOT hold >=2 referents as a coexisting set (the stronger intrinsic attractor dominates and
     the other collapses) -> the competitive read RE-PRESENTS the held discourse-referent registry as co-active
     competitors (a retrieval cue gently re-drives their assemblies — the biology of biased competition, where the
     competing stimuli are simultaneously present). The moat reads the held assembly (a winner must be
     re-presentable above a floor) + abstains when the content is silent.
"""
from __future__ import annotations

import numpy as np

from research.runners.content_selection_spiking import SpikingLoopContextBuffer


# ---------------------------------------------------------------------------
# Content-bias helper (HOST scaffold — flagged for conversion to a learned
# synaptic feature-compatibility map per BRAIN-BASED-ONLY; see module docstring).
# A bare pronoun's features filter candidate antecedents; the query verb's
# selectional restriction biases toward the compatible referent. Both CONTENT
# signals, not position (recency) or magnitude (boost).
# ---------------------------------------------------------------------------
ANIMACY = {  # per-concept feature tag (the small-world feature lexicon)
    "cat": "animate", "dog": "animate", "bird": "animate", "fox": "animate",
    "fish": "animate", "worm": "animate",
    "ball": "inanimate", "apple": "inanimate", "river": "inanimate",
    "rock": "inanimate", "book": "inanimate",
}
# selectional restriction: which animacy a verb's THEME/argument prefers as an antecedent for "it".
VERB_SELECTS = {
    "eat": "animate",     # "what does it eat?" -> the eater is animate
    "chase": "animate",   # an agentive verb -> animate
    "roll": "inanimate",  # "where did it roll?" -> the roller is the ball (inanimate)
    "float": "inanimate", # "did it float?" -> inanimate theme
}


def content_bias_target(candidates, query_verb):
    """Return the single held referent that the pronoun+verb content selects for, or None if the content
    does not disambiguate (no match, or >1 equally-compatible candidate -> a TIE the moat must abstain on)."""
    want = VERB_SELECTS.get(query_verb)
    if want is None:
        return None
    matches = [c for c in candidates if ANIMACY.get(c) == want]
    if len(matches) == 1:
        return matches[0]
    return None  # 0 matches or a tie -> content is silent; abstain


# ---------------------------------------------------------------------------
# The biased-competition buffer: the held referents as COMPETING assemblies
# with mutual inhibition (per-referent FS pool) + a small content bias injector.
# Subclasses the validated SpikingLoopContextBuffer's holding mechanism.
# ---------------------------------------------------------------------------
class BiasedCompetitionContextBuffer:
    """Adds WTA biased competition to the held referents, as a READ-OUT TAP layer (faithful reuse of the
    navigation sel_X/sel_FS_X Wong-Wang accumulator WTA; g11_bg_runner.py). The held referents stay in the
    cortex_ctx<->dlpfc_wm loop unchanged; the competition is between per-referent ACCUMULATOR pools:

      cortex_assembly[X] --(ff evidence)--> sel_X (NMDA-slow recurrent, alpha<1) --> sel_FS_X --(inh)--> sel_Y!=X

    The accumulator integrates each referent's held firing; the selective inhibition makes the leader suppress
    the others (the SUPPRESSION a salience boost could never produce); the small CONTENT bias is injected into
    the favored sel_X so the recurrence amplifies the small content asymmetry into a clean winner. The sel
    layer is a pure tap — no sel_X -> cortex projection — so the held attractors are unperturbed (exactly the
    navigation design where sel reads thal but never projects back). The winner is READ from the sel pools.

    competition=False -> no sel layer (== the plain SpikingLoopContextBuffer holding; the salience baseline
    substrate). competition=True -> the accumulator WTA is wired; bias() steers it.
    """

    def __init__(self, concepts, n=600, pattern_size=40, attractor_weight=50.0,
                 n_sel=20, n_sel_fs=10, ref_to_sel_weight=12.0, sel_recurrent_weight=0.35,
                 sel_recurrent_density=0.5, sel_to_fs_weight=20.0, fs_to_sel_weight=5.0,
                 seed=42, enable_ou=False, competition=True, verbose=False):
        import sim.backend as B
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronType
        self.B = B
        self.xp, _ = B.get_backend()
        self.concepts = list(concepts)
        self.competition = bool(competition)
        self._psize = pattern_size
        self._held = []   # discourse-referent registry (which referents were introduced via update())

        def loop_reg(name):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=0.0,
                               exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                               plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)

        regions = [loop_reg("cortex_ctx"), loop_reg("dlpfc_wm")]
        if self.competition:
            for c in self.concepts:
                # ACCUMULATE pool: excitatory, NMDA-slow recurrent self-excitation (Wang-2002 integrator);
                # soft-WTA gain alpha<1 (Rutishauser) -> ramps/holds under evidence+bias, never self-ignites.
                regions.append(BrainRegion(
                    name=f"sel_{c}", n_neurons=n_sel, exc_fraction=1.0,
                    internal_density=sel_recurrent_density, exc_weight_mean=sel_recurrent_weight,
                    inh_weight_mean=0.0, weight_jitter=0.2, plastic_internal=False, enable_nmda=True,
                    izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name))
                # Selective inhibitory interneuron: driven only by sel_X, inhibits only sel_Y!=X (Rutishauser).
                regions.append(BrainRegion(
                    name=f"sel_FS_{c}", n_neurons=n_sel_fs, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions
        pathways = [
            # loop pathways at weight 0 (matching build_loop_wm_bridge with loop_weight=0): SEED a non-empty
            # CSR so the per-concept attractors + the sel wiring can be installed via set_pathway_weights.
            RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=0.05,
                          weight_mean=0.0, weight_jitter=0.2, plastic=False),
            RegionPathway(from_region="dlpfc_wm", to_region="cortex_ctx", density=0.05,
                          weight_mean=0.0, weight_jitter=0.2, plastic=False),
        ]
        if self.competition:
            # sel_X -> sel_FS_X (exc: the winning accumulator recruits its interneuron).
            for c in self.concepts:
                pathways.append(RegionPathway(from_region=f"sel_{c}", to_region=f"sel_FS_{c}",
                                              density=1.0, weight_mean=sel_to_fs_weight, weight_jitter=0.2,
                                              plastic=False))
            # sel_FS_X -> sel_Y!=X (inh: gentle cross-pool suppression; symmetric over-inhibition is unstable).
            for X in self.concepts:
                for Y in self.concepts:
                    if X == Y:
                        continue
                    pathways.append(RegionPathway(from_region=f"sel_FS_{X}", to_region=f"sel_{Y}",
                                                  density=1.0, weight_mean=fs_to_sel_weight, weight_jitter=0.2,
                                                  plastic=False))
        cfg.region_pathways = pathways
        cfg.dt_ms = 0.5
        cfg.seed = seed
        cfg.enable_nmda = True
        cfg.enable_ou_process = bool(enable_ou)
        cfg.enable_structural_plasticity = False
        cfg.enable_hebbian_learning = False
        cfg.enable_short_term_plasticity = False
        cfg.stdp_w_max = 60.0
        cfg.fast_spike_reset = True
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        self.bridge = bridge

        rm = bridge.region_manager
        cidx = np.asarray(rm.indices("cortex_ctx"))
        didx = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        self._cpat = {}
        self._dpat = {}
        # BRAIN-LOAD SPEEDUP (2026-06-24): batch ALL concepts' attractor edges into ONE c2d + ONE d2c
        # set_pathway_weights call (was one-call-per-concept). Each call rebuilds the WHOLE sparse CSR, so the
        # original 2*len(concepts) calls cost 2*len(concepts) full CSR rebuilds; batching => exactly 2. The
        # edges are identical (the per-concept attractors partition the indices via disjoint perm slices, all
        # at attractor_weight; set_pathway_weights is order-independent for distinct edges) -> byte-identical
        # CSR. Mirrors the same fix in content_selection_spiking.SpikingLoopContextBuffer.
        c2d_pre, c2d_post, d2c_pre, d2c_post = [], [], [], []
        for i, c in enumerate(self.concepts):
            p = perm[i * pattern_size:(i + 1) * pattern_size]
            cpat, dpat = cidx[p], didx[p]
            self._cpat[c] = self.xp.asarray(cpat)
            self._dpat[c] = self.xp.asarray(dpat)
            # per-concept outer-product attractor (the SpikingLoopContextBuffer mechanism)
            c2d_pre.append(np.repeat(cpat, pattern_size))
            c2d_post.append(np.tile(dpat, pattern_size))
            d2c_pre.append(np.repeat(dpat, pattern_size))
            d2c_post.append(np.tile(cpat, pattern_size))
        if self.concepts:
            c2d_pre = np.concatenate(c2d_pre).astype(np.int64)
            c2d_post = np.concatenate(c2d_post).astype(np.int64)
            d2c_pre = np.concatenate(d2c_pre).astype(np.int64)
            d2c_post = np.concatenate(d2c_post).astype(np.int64)
            ww_attr = np.full(c2d_pre.size, attractor_weight, np.float32)
            self.bridge.set_pathway_weights("c2d", pre_indices=c2d_pre, post_indices=c2d_post, weights=ww_attr,
                                            add_missing=True)
            self.bridge.set_pathway_weights("d2c", pre_indices=d2c_pre, post_indices=d2c_post, weights=ww_attr,
                                            add_missing=True)

        self._sel_idx = {}
        if self.competition:
            self._sel_idx = {c: np.asarray(rm.indices(f"sel_{c}"), dtype=np.int64) for c in self.concepts}
            self._wire_assembly_to_sel(rm, cpat_by_concept={c: self.B.to_host(self._cpat[c]).astype(np.int64)
                                                            for c in self.concepts},
                                       ref_to_sel_weight=ref_to_sel_weight)
            self._n_sel = n_sel

        if verbose:
            print(f"[biased-competition buffer] {len(self.concepts)} referents, competition={self.competition}, "
                  f"sel/referent={n_sel}", flush=True)

    def _wire_assembly_to_sel(self, rm, cpat_by_concept, ref_to_sel_weight):
        """cortex_assembly[X] -> sel_X (all-to-all, excitatory FEED-FORWARD EVIDENCE: the held referent's
        firing drives its accumulator). Read-only: there is NO sel_X -> cortex projection, so the held
        attractors are byte-unperturbed by the competition."""
        # BRAIN-LOAD SPEEDUP (2026-06-24): batch all referents' assembly->sel edges into ONE call (was
        # one-call-per-concept; each is a full CSR rebuild). Identical edge set -> byte-identical CSR.
        pre_all, post_all = [], []
        for X in self.concepts:
            aX = cpat_by_concept[X]
            sX = self._sel_idx[X]
            pre_all.append(np.repeat(aX, sX.size))
            post_all.append(np.tile(sX, aX.size))
        if self.concepts:
            pre = np.concatenate(pre_all).astype(np.int64)
            post = np.concatenate(post_all).astype(np.int64)
            w = np.full(pre.size, np.float32(ref_to_sel_weight), np.float32)
            self.bridge.set_pathway_weights("ref2sel", pre_indices=pre, post_indices=post, weights=w,
                                            add_missing=True)

    # ---- holding (reused from SpikingLoopContextBuffer) ----
    def update(self, concepts, drive_pA=2500.0, stim=40, settle=15):
        for c in concepts:
            if c not in self._cpat:
                continue
            self._held.append(c)   # discourse-referent registry (which referents were introduced)
            drv = self._cpat[c]
            for _ in range(stim):
                self.bridge.cp_external_input_current[:] = 0.0
                self.bridge.cp_external_input_current[drv] = drive_pA
                self.bridge._run_one_simulation_step()
            self.bridge.cp_external_input_current[:] = 0.0
            for _ in range(settle):
                self.bridge._run_one_simulation_step()

    def read(self, window=20, bias_concept=None, bias_pA=0.0, redrive_pA=2200.0):
        """Read the pronoun-resolution competition over a window.

        WITHOUT competition (the salience-baseline substrate): the plain per-assembly cortex firing read,
        no re-presentation — exactly the SpikingLoopContextBuffer behavior.

        WITH competition (biased competition): the held discourse referents (the registry) are RE-PRESENTED
        as co-active competitors during the read (a retrieval cue gently re-drives their cortex assemblies —
        the biology of biased competition, where the competing stimuli are simultaneously present; without
        re-presentation the substrate's destructive single-winner hold means only the strongest intrinsic
        attractor is active and there is nothing to arbitrate). The co-active assemblies feed their sel_X
        accumulators; a SMALL content bias is injected into the favored sel_X; the selective inhibition +
        recurrence amplify the small content asymmetry into a SUPPRESSIVE winner. The per-sel firing IS the
        competition read; the per-assembly firing is the moat gate (a referent must be re-presentable above
        held_floor — empty registry -> no re-presentation -> nothing held -> abstain).

        Returns {"sel": {c: rate}, "held": {c: assembly_rate}}."""
        if not self.competition:
            acc = {c: 0.0 for c in self.concepts}
            bias_idx = (self._cpat[bias_concept] if bias_concept in self._cpat and bias_pA > 0.0 else None)
            for _ in range(window):
                self.bridge.cp_external_input_current[:] = 0.0
                if bias_idx is not None:
                    self.bridge.cp_external_input_current[bias_idx] = np.float32(bias_pA)
                self.bridge._run_one_simulation_step()
                fs = self.bridge.cp_firing_states
                for c in self.concepts:
                    acc[c] += float(self.B.to_host(fs[self._cpat[c]]).sum())
            held = {c: acc[c] / (self._psize * window) for c in self.concepts}
            return {"sel": dict(held), "held": held}
        # competition: re-present the held referents (co-active competitors) + bias the favored sel pool.
        held_set = sorted(set(self._held))
        re_idx = None
        if held_set:
            re_idx = self.xp.asarray(np.concatenate(
                [self.B.to_host(self._cpat[c]).astype(np.int64) for c in held_set]))
        bias_sel = (self._sel_idx[bias_concept] if (bias_concept in self._sel_idx and bias_pA > 0.0) else None)
        sel_acc = {c: 0.0 for c in self.concepts}
        held_acc = {c: 0.0 for c in self.concepts}
        for _ in range(window):
            self.bridge.cp_external_input_current[:] = 0.0
            if re_idx is not None:
                self.bridge.cp_external_input_current[re_idx] = np.float32(redrive_pA)
            if bias_sel is not None:
                self.bridge.cp_external_input_current[bias_sel] = np.float32(bias_pA)
            self.bridge._run_one_simulation_step()
            fs = self.bridge.cp_firing_states
            for c in self.concepts:
                sel_acc[c] += float(self.B.to_host(fs[self._sel_idx[c]]).sum())
                held_acc[c] += float(self.B.to_host(fs[self._cpat[c]]).sum())
        return {"sel": {c: sel_acc[c] / (self._n_sel * window) for c in self.concepts},
                "held": {c: held_acc[c] / (self._psize * window) for c in self.concepts}}


def resolve_referent(read, spec_threshold=1.3, held_floor=0.08):
    """Resolve the pronoun to the winning referent IFF (1) the competition (sel) winner leads the runner-up
    by >= spec_threshold AND (2) that winner is ACTUALLY HELD in WM (its cortex assembly fires above
    held_floor). Else None = abstain — the no-confab moat: an empty WM (nothing held -> the bias cannot
    confabulate an antecedent) or a tie (no clear sel winner) produces None.

    Accepts either the structured read ({"sel":..., "held":...}) or a bare rates dict (treated as both)."""
    if not read:
        return None
    if "sel" in read and "held" in read:
        sel, held = read["sel"], read["held"]
    else:
        sel = held = read
    items = sorted(sel.items(), key=lambda kv: kv[1], reverse=True)
    top_c, top_r = items[0]
    if top_r <= 1e-6:
        return None
    runner = items[1][1] if len(items) > 1 else 0.0
    if runner > 1e-9 and top_r < spec_threshold * runner:
        return None  # no decisive competition winner -> tie -> abstain
    # moat gate: the winner must be a referent actually held in WM (assembly active).
    if held.get(top_c, 0.0) < held_floor:
        return None
    return top_c
