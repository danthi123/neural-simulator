"""Multi-referent disambiguation via WTA BIASED COMPETITION — the mechanism the two converging NEGATIVEs
(2026-06-17-multireferent-disambiguation-NEGATIVE.md: recency 0/3 + salience-boost even-4x-fails) named as the
fix, and scoped in 2026-06-19-multireferent-wta-biased-competition-scoping.md.

THE QUESTION. When the spiking working memory holds several discourse referents, which one does a bare pronoun
("it") bind to? The plain SpikingLoopContextBuffer holds each referent in an INDEPENDENT attractor with NO
cross-referent coupling (loop_weight=0, internal_density=0 by construction), so neither recency (no position
signal in the rate read) nor a salience boost (only ADDS activity to an independent attractor; cannot SUPPRESS
the competitor) can pick the right one. THE FIX (Desimone-Duncan 1995 biased competition; Wong-Wang 2006 attractor
WTA): MUTUAL INHIBITION between the held referent assemblies (each referent's assembly drives a dedicated FS
inhibitory pool that suppresses the OTHER referents' assemblies — the Rutishauser selective-inhibition motif the
navigation sel_X/sel_FS_X read-out already uses) + a small CONTENT-based top-down BIAS (a feed-forward current
into the referent whose features — animacy/number agreement with the pronoun + selectional compatibility with the
query verb — match). The crux: the bias is a CONTENT signal (NOT position=recency, NOT magnitude=boost — the two
already-disproven signals); the recurrence amplifies the small content asymmetry into a SUPPRESSIVE winner.

THE NEW WIRING (additive; NO sim/ edit). BiasedCompetitionContextBuffer wraps the validated loop bridge: it builds
the cortex_ctx<->dlpfc_wm loop PLUS one all-inhibitory FS region per referent (exc_fraction=0.0 -> the FS neurons
get inhibitory traits, so their out-synapses route to g_i), then installs, via set_pathway_weights(add_missing):
  * cortex_assembly[X] -> ref_FS_X[all]        (EXCITATORY: a referent recruits its own interneuron)
  * ref_FS_X[all]      -> cortex_assembly[Y!=X] (INHIBITORY: that interneuron suppresses the OTHER referents)
plus a bias(concept, pA) injector that adds a SMALL feed-forward current into a concept's cortex assembly during
the competitive read window. The held attractors + read() are reused verbatim from SpikingLoopContextBuffer.

THE CONTENT-BIAS HELPER is host-side (a teaching scaffold, FLAGGED for conversion to a learned synaptic
feature-compatibility map per BRAIN-BASED-ONLY): given the pronoun's features + the query verb's selectional
restriction, it returns WHICH held referent receives the bias current. The win is brain-based (spiking competition
+ suppression); the content SCORING is host in this probe (the follow-on neuralizes it).

GO BAR (pre-registered, FROZEN; >=5/6 seeds):
  1. the content-bias-favored referent WINS the WTA (the pronoun resolves to it), in BOTH write-orders, AND the
     feature-flip FLIPS the winner (proves it's content, not position/magnitude).
  2. the recency baseline FAILS on the identical {cat, ball} setup (re-run in-probe).
  3. the salience-boost baseline FAILS on the identical setup (re-run in-probe, even 4x).
  4. bias-LESION (remove the content bias, keep the competition) -> the WTA picks at chance / wrong (THE decisive
     control proving genuine content-steered competition, not a relabelled boost).
  5. the no-confab MOAT intact: empty WM or a TIE -> abstain (None), 0 breaches.
A bias-lesion that does NOT break resolution = the bias wasn't load-bearing = NOT a GO.

Run: SIM_BACKEND=numpy python -m research.runners._phaseB_biased_competition_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

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
        ww_attr = np.full(pattern_size * pattern_size, attractor_weight, np.float32)
        for i, c in enumerate(self.concepts):
            p = perm[i * pattern_size:(i + 1) * pattern_size]
            cpat, dpat = cidx[p], didx[p]
            self._cpat[c] = self.xp.asarray(cpat)
            self._dpat[c] = self.xp.asarray(dpat)
            # per-concept outer-product attractor (the SpikingLoopContextBuffer mechanism)
            pre1 = np.repeat(cpat, pattern_size).astype(np.int64)
            post1 = np.tile(dpat, pattern_size).astype(np.int64)
            self.bridge.set_pathway_weights("c2d", pre_indices=pre1, post_indices=post1, weights=ww_attr,
                                            add_missing=True)
            self.bridge.set_pathway_weights("d2c", pre_indices=np.repeat(dpat, pattern_size).astype(np.int64),
                                            post_indices=np.tile(cpat, pattern_size).astype(np.int64),
                                            weights=ww_attr, add_missing=True)

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
        for X in self.concepts:
            aX = cpat_by_concept[X]
            sX = self._sel_idx[X]
            pre = np.repeat(aX, sX.size).astype(np.int64)
            post = np.tile(sX, aX.size).astype(np.int64)
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


# ---------------------------------------------------------------------------
# The de-risk
# ---------------------------------------------------------------------------
# Two referents of OPPOSING content features (so the bias has a content handle and the test is NOT
# recency-solvable): cat (animate) vs ball (inanimate). Pronoun "it" + a query verb whose selectional
# restriction picks one of them.
PAIR = ("cat", "ball")            # (animate, inanimate)
DISTRACTORS = ["fish", "worm", "fox", "dog"]   # extra referents for the >=2 setup / 3-referent scale check


def _favored(candidates, verb):
    return content_bias_target(candidates, verb)


def _disp(read):
    """Compact display of a structured read: {concept: (sel_rate, held_rate)}."""
    if isinstance(read, dict) and "sel" in read:
        return {c: [round(read["sel"][c], 4), round(read["held"][c], 4)] for c in read["sel"]}
    return {c: round(v, 4) for c, v in read.items()}


def resolve_pronoun(w, verb, candidates, bias_pA, spec_threshold, window, lesion=False):
    """The full pronoun-resolution decision. (1) The CONTENT bias selects which held referent matches the
    pronoun+verb; if content is SILENT (no/ambiguous match) -> abstain (None) — the no-confab moat refuses
    to pick by intrinsic strength. (2) Else run the biased competition (re-present + bias the favored sel)
    and resolve the WTA winner (gated on the moat held-floor). lesion=True keeps the competition but DROPS
    the bias (bias_pA=0) -> the WTA reverts to the intrinsic winner -> the content control is broken."""
    fav = content_bias_target(candidates, verb)
    if fav is None:
        return None, None, {}   # content silent -> abstain (moat)
    rates = w.read(window=window, bias_concept=(None if lesion else fav),
                   bias_pA=(0.0 if lesion else bias_pA))
    resolved = resolve_referent(rates, spec_threshold)
    return resolved, fav, rates


def run_seed(seed, bias_pA, spec_threshold, window, verbose=False):
    cat, ball = PAIR

    def buf(concepts, competition=True):
        return BiasedCompetitionContextBuffer(concepts, n=600, pattern_size=40, seed=seed,
                                              enable_ou=False, competition=competition, verbose=verbose)

    out = {"seed": seed}

    # --- GO arm: biased competition, BOTH write-orders, on the {cat, ball} pair ---
    # query "eat" selects animate -> favored = cat ; query "roll" selects inanimate -> favored = ball.
    def trial(order, verb, lesion=False):
        cands = [cat, ball]
        w = buf(cands, competition=True)
        w.update([order[0]]); w.update([order[1]])
        resolved, fav, rates = resolve_pronoun(w, verb, cands, bias_pA, spec_threshold, window, lesion=lesion)
        return {"order": list(order), "verb": verb, "favored": fav, "rates": _disp(rates),
                "resolved": resolved, "correct": bool(resolved == fav and fav is not None)}

    out["bc_cat_first_eat"] = trial((cat, ball), "eat")    # favored cat, cat written first
    out["bc_ball_first_eat"] = trial((ball, cat), "eat")   # favored cat, ball written first -> if recency, ball would win
    out["bc_cat_first_roll"] = trial((cat, ball), "roll")  # FEATURE-FLIP: favored ball, cat written first
    out["bc_ball_first_roll"] = trial((ball, cat), "roll") # FEATURE-FLIP: favored ball, ball written first
    out["go_arm"] = bool(out["bc_cat_first_eat"]["correct"] and out["bc_ball_first_eat"]["correct"]
                         and out["bc_cat_first_roll"]["correct"] and out["bc_ball_first_roll"]["correct"])

    # --- LESION: competition present, bias REMOVED. For the SAME held WM {cat,ball}, the unbiased WTA picks
    # the SAME intrinsic winner regardless of verb -> it cannot match BOTH opposite favoreds (eat->cat,
    # roll->ball) -> >=1 is wrong -> the bias is load-bearing. ---
    les_a = trial((cat, ball), "eat", lesion=True)
    les_b = trial((cat, ball), "roll", lesion=True)
    out["lesion_eat"] = les_a
    out["lesion_roll"] = les_b
    out["lesion_breaks"] = bool(not (les_a["correct"] and les_b["correct"]))

    # --- MOAT: (a) empty WM -> abstain ; (b) content-silent query (verb with no selectional restriction, OR
    # two same-feature candidates) -> abstain (the agent refuses to pick by intrinsic strength). ---
    w_empty = buf([cat, ball], competition=True)  # nothing written (empty registry)
    er, ef, erates = resolve_pronoun(w_empty, "eat", [cat, ball], bias_pA, spec_threshold, window)
    out["moat_empty"] = {"rates": _disp(erates), "resolved": er}
    # content-silent: a verb with no selectional restriction -> favored None -> abstain
    w_sil = buf([cat, ball], competition=True); w_sil.update([cat]); w_sil.update([ball])
    sr, sf, srates = resolve_pronoun(w_sil, "see", [cat, ball], bias_pA, spec_threshold, window)  # 'see' not in VERB_SELECTS
    out["moat_silent"] = {"rates": _disp(srates), "resolved": sr, "favored": sf}
    out["moat_intact"] = bool(er is None and sr is None)

    # --- 3-referent scale check (one compatible + two incompatible): {cat(animate), ball, river(inanimate)} ---
    three = [cat, ball, "river"]
    w3 = buf(three, competition=True)
    w3.update([ball]); w3.update(["river"]); w3.update([cat])   # cat written last but bias is content, not recency
    res3, fav3, rates3 = resolve_pronoun(w3, "eat", three, bias_pA, spec_threshold, window)  # animate -> cat
    out["three_ref"] = {"favored": fav3, "rates": _disp(rates3),
                        "resolved": res3, "correct": bool(res3 == fav3 and fav3 is not None)}

    return out


def run_baselines_on_pair(seed, window, spec_threshold):
    """Re-run the recency + salience baselines on the IDENTICAL {cat, ball} setup (no competition substrate),
    proving the setup is genuinely ambiguous without the new mechanism."""
    cat, ball = PAIR

    def plain(concepts):
        return SpikingLoopContextBuffer(concepts, n=600, pattern_size=40, seed=seed, enable_ou=False)

    # recency: write cat then ball (ball recent). Does the read carry a usable order gradient either way?
    w = plain([cat, ball]); w.update([cat]); w.update([ball])
    r_nat = w.read(window=window)
    nat_res = resolve_referent(r_nat, spec_threshold)
    w2 = plain([cat, ball]); w2.update([ball]); w2.update([cat])
    r_ord = w2.read(window=window)
    ord_res = resolve_referent(r_ord, spec_threshold)
    # recency PASSES only if the read flips with order (recent always wins). It FAILS (the documented NEGATIVE)
    # if it does not produce a recency-aligned, order-flipping winner.
    recency_resolves = bool(nat_res == ball and ord_res == cat)

    # salience: write cat normal, ball boosted 4x. Does the boost win? order-control: ball normal, cat boosted.
    def boosted(w, c, f=4.0):
        w.update([c], drive_pA=2500.0 * f, stim=int(40 * f), settle=15)
    ws = plain([cat, ball]); ws.update([cat]); boosted(ws, ball)
    s_nat = ws.read(window=window)
    s_nat_res = resolve_referent(s_nat, spec_threshold)
    ws2 = plain([cat, ball]); ws2.update([ball]); boosted(ws2, cat)
    s_ord = ws2.read(window=window)
    s_ord_res = resolve_referent(s_ord, spec_threshold)
    salience_resolves = bool(s_nat_res == ball and s_ord_res == cat)

    return {
        "recency": {"natural_resolved": nat_res, "order_resolved": ord_res,
                    "nat_rates": {k: round(v, 4) for k, v in r_nat.items()},
                    "ord_rates": {k: round(v, 4) for k, v in r_ord.items()},
                    "resolves": recency_resolves},
        "salience_4x": {"natural_resolved": s_nat_res, "order_resolved": s_ord_res,
                        "nat_rates": {k: round(v, 4) for k, v in s_nat.items()},
                        "ord_rates": {k: round(v, 4) for k, v in s_ord.items()},
                        "resolves": salience_resolves},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--bias-pA", type=float, default=2500.0,
                    help="content-bias feed-forward current (~1x the per-assembly drive scale 2500 — SMALL on "
                         "purpose, the magnitude a uniform boost already FAILED at, so any win is from the "
                         "competition amplifying a small content asymmetry).")
    ap.add_argument("--spec-threshold", type=float, default=1.3)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--out", default="research/findings/raw/_phaseB_biased_competition.json")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    print("[biased-competition de-risk] does WTA biased competition (mutual inhibition + small CONTENT bias) "
          "bind a bare pronoun\n  to the correct one of >=2 held referents, where recency + salience cannot?\n"
          f"  bias_pA={a.bias_pA} (~1x drive; small on purpose), spec_threshold={a.spec_threshold}\n", flush=True)

    results = []
    for seed in a.seeds:
        r = run_seed(seed, a.bias_pA, a.spec_threshold, a.window, verbose=a.verbose)
        r["baselines"] = run_baselines_on_pair(seed, a.window, a.spec_threshold)
        results.append(r)
        ea = r["bc_cat_first_eat"]; eb = r["bc_ball_first_eat"]
        ra = r["bc_cat_first_roll"]; rb = r["bc_ball_first_roll"]
        bl = r["baselines"]
        print(f"  [seed {seed}] GO-arm: eat(cat-1st) ->{ea['resolved']}({'OK' if ea['correct'] else 'X'}) "
              f"eat(ball-1st)->{eb['resolved']}({'OK' if eb['correct'] else 'X'}) "
              f"roll(cat-1st)->{ra['resolved']}({'OK' if ra['correct'] else 'X'}) "
              f"roll(ball-1st)->{rb['resolved']}({'OK' if rb['correct'] else 'X'}) || go_arm={r['go_arm']}",
              flush=True)
        print(f"            lesion: eat->{r['lesion_eat']['resolved']} roll->{r['lesion_roll']['resolved']} "
              f"(breaks={r['lesion_breaks']}) | moat empty->{r['moat_empty']['resolved']} "
              f"silent->{r['moat_silent']['resolved']} (intact={r['moat_intact']}) | "
              f"3ref->{r['three_ref']['resolved']}({'OK' if r['three_ref']['correct'] else 'X'})", flush=True)
        print(f"            baselines on {PAIR}: recency_resolves={bl['recency']['resolves']} "
              f"salience4x_resolves={bl['salience_4x']['resolves']}", flush=True)

    n = len(results)
    go_seeds = sum(r["go_arm"] for r in results)
    lesion_seeds = sum(r["lesion_breaks"] for r in results)
    moat_seeds = sum(r["moat_intact"] for r in results)
    three_seeds = sum(r["three_ref"]["correct"] for r in results)
    recency_fail = sum(not r["baselines"]["recency"]["resolves"] for r in results)
    salience_fail = sum(not r["baselines"]["salience_4x"]["resolves"] for r in results)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "summary": {
            "n": n, "go_arm_seeds": go_seeds, "lesion_breaks_seeds": lesion_seeds,
            "moat_intact_seeds": moat_seeds, "three_ref_seeds": three_seeds,
            "recency_fail_seeds": recency_fail, "salience_fail_seeds": salience_fail,
            "bias_pA": a.bias_pA, "spec_threshold": a.spec_threshold}}, fh, indent=2, default=str)

    print(f"\n{'='*100}", flush=True)
    print(f"  GO-arm (favored wins both orders + feature-flip): {go_seeds}/{n}", flush=True)
    print(f"  bias-LESION breaks resolution (load-bearing):     {lesion_seeds}/{n}", flush=True)
    print(f"  no-confab MOAT intact (empty/tie abstain):        {moat_seeds}/{n}", flush=True)
    print(f"  recency baseline FAILS (identical setup):         {recency_fail}/{n}", flush=True)
    print(f"  salience-4x baseline FAILS (identical setup):     {salience_fail}/{n}", flush=True)
    print(f"  3-referent scale (in-probe):                      {three_seeds}/{n}", flush=True)
    bar = 5 if n >= 6 else n
    GO = (go_seeds >= bar and lesion_seeds >= bar and moat_seeds == n
          and recency_fail == n and salience_fail == n)
    if GO:
        print(f"\n  ==> GO: WTA biased competition resolves multi-referent pronouns where recency + salience "
              "CANNOT. The favored\n  referent wins (both write-orders + feature-flip), the bias is load-bearing "
              "(lesion breaks it), the moat holds.\n  ==> recommend wiring into MultiTurnAgent behind a "
              "default-OFF enable_biased_competition flag (follow-on).", flush=True)
    elif go_seeds >= bar and lesion_seeds >= bar and moat_seeds == n and three_seeds < bar:
        print(f"\n  ==> BOUNDARY: the 2-referent case resolves (lesion+moat+baselines hold) but the 3-referent "
              "case degrades\n  -> localizes competition-strength-vs-N as the next tuning sub-problem (within the "
              "alpha<1 envelope).", flush=True)
    else:
        print(f"\n  ==> NEGATIVE: even with mutual inhibition + a small content bias the intrinsic-attractor "
              "asymmetry dominates\n  (or the lesion did not break resolution = the bias was not load-bearing). "
              "Honest rate-attractor substrate boundary\n  -> re-scope to gamma-cycle (N.19) phase segregation; "
              "do NOT escalate into a config search.", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
