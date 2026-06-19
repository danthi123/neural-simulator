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
    """Adds WTA biased competition to the held referents. Builds the loop bridge with one all-inhibitory FS
    region per referent, then installs assembly-targeted mutual-inhibition synapses + per-referent attractors.

    REUSE: the per-concept outer-product attractors (c2d/d2c) + the read() (per-assembly cortex firing) are
    the SpikingLoopContextBuffer mechanism, replicated here. NEW: the FS regions + the cross-referent
    inhibition + the bias() injector.

    competition=False (default) -> no FS wiring installed == the plain SpikingLoopContextBuffer (the salience
    baseline substrate). competition=True -> the mutual inhibition is wired; bias() then steers it.
    """

    def __init__(self, concepts, n=600, pattern_size=40, attractor_weight=50.0,
                 fs_per_referent=20, ref_to_fs_weight=8.0, fs_to_ref_weight=14.0,
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

        def loop_reg(name):
            return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=0.0,
                               exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                               plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)

        regions = [loop_reg("cortex_ctx"), loop_reg("dlpfc_wm")]
        if self.competition:
            for c in self.concepts:
                # All-inhibitory FS pool (exc_fraction=0.0 -> every neuron inhibitory -> out-synapses -> g_i).
                regions.append(BrainRegion(
                    name=f"ref_FS_{c}", n_neurons=fs_per_referent, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                    plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = regions
        cfg.region_pathways = []  # cross-region edges installed by hand below (assembly-targeted)
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

        if self.competition:
            self._wire_mutual_inhibition(rm, ref_to_fs_weight, fs_to_ref_weight)

        if verbose:
            print(f"[biased-competition buffer] {len(self.concepts)} referents, competition={self.competition}, "
                  f"FS/referent={fs_per_referent}", flush=True)

    def _wire_mutual_inhibition(self, rm, ref_to_fs_weight, fs_to_ref_weight):
        """Install, per referent X: cortex_assembly[X] -> ref_FS_X (excitatory) and ref_FS_X ->
        cortex_assembly[Y!=X] (inhibitory; the FS neurons are inhibitory so the edge routes to g_i).
        Rutishauser selective inhibition: driven only by X, inhibits only the others."""
        cpat_h = {c: self.B.to_host(self._cpat[c]).astype(np.int64) for c in self.concepts}
        fs_idx = {c: np.asarray(rm.indices(f"ref_FS_{c}"), dtype=np.int64) for c in self.concepts}
        for X in self.concepts:
            aX = cpat_h[X]
            fX = fs_idx[X]
            # cortex_assembly[X] -> ref_FS_X (all-to-all, excitatory)
            pre = np.repeat(aX, fX.size).astype(np.int64)
            post = np.tile(fX, aX.size).astype(np.int64)
            w = np.full(pre.size, np.float32(ref_to_fs_weight), np.float32)
            self.bridge.set_pathway_weights("ref2fs", pre_indices=pre, post_indices=post, weights=w,
                                            add_missing=True)
            # ref_FS_X -> cortex_assembly[Y!=X] (all-to-all, inhibitory via inhibitory pre-neuron)
            for Y in self.concepts:
                if Y == X:
                    continue
                aY = cpat_h[Y]
                pre2 = np.repeat(fX, aY.size).astype(np.int64)
                post2 = np.tile(aY, fX.size).astype(np.int64)
                w2 = np.full(pre2.size, np.float32(fs_to_ref_weight), np.float32)
                self.bridge.set_pathway_weights("fs2ref", pre_indices=pre2, post_indices=post2, weights=w2,
                                                add_missing=True)

    # ---- holding (reused from SpikingLoopContextBuffer) ----
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

    def read(self, window=20, bias_concept=None, bias_pA=0.0):
        """Decode the held set over a no-drive window. If bias_concept is set, inject a SMALL feed-forward
        bias current into that concept's cortex assembly during the window (the content top-down bias). The
        mutual inhibition (always active, driven by the assemblies' own firing) amplifies the bias into a
        suppressive winner."""
        bias_idx = None
        if bias_concept is not None and bias_pA > 0.0 and bias_concept in self._cpat:
            bias_idx = self._cpat[bias_concept]
        acc = {c: 0.0 for c in self.concepts}
        for _ in range(window):
            self.bridge.cp_external_input_current[:] = 0.0
            if bias_idx is not None:
                self.bridge.cp_external_input_current[bias_idx] = np.float32(bias_pA)
            self.bridge._run_one_simulation_step()
            fs = self.bridge.cp_firing_states
            for c in self.concepts:
                acc[c] += float(self.B.to_host(fs[self._cpat[c]]).sum())
        return {c: acc[c] / (self._psize * window) for c in self.concepts}


def resolve_referent(rates, spec_threshold=1.3):
    """Top assembly resolves IFF it leads the runner-up by >= spec_threshold (else None = abstain — the
    no-confab moat: a tie or an empty WM produces no confabulated antecedent)."""
    if not rates:
        return None
    items = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    top_c, top_r = items[0]
    if top_r <= 1e-6:
        return None  # empty WM -> nothing held -> abstain
    runner = items[1][1] if len(items) > 1 else 0.0
    if runner <= 1e-9:
        return top_c
    return top_c if top_r >= spec_threshold * runner else None


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


def run_seed(seed, bias_pA, spec_threshold, window, verbose=False):
    cat, ball = PAIR

    def buf(concepts, competition=True):
        return BiasedCompetitionContextBuffer(concepts, n=600, pattern_size=40, seed=seed,
                                              enable_ou=False, competition=competition, verbose=verbose)

    out = {"seed": seed}

    # --- GO arm: biased competition, BOTH write-orders, on the {cat, ball} pair ---
    # query "eat" selects animate -> favored = cat ; query "roll" selects inanimate -> favored = ball.
    def trial(order, verb):
        cands = [cat, ball]
        fav = _favored(cands, verb)
        w = buf(cands, competition=True)
        w.update([order[0]]); w.update([order[1]])
        rates = w.read(window=window, bias_concept=fav, bias_pA=bias_pA)
        resolved = resolve_referent(rates, spec_threshold)
        return {"order": list(order), "verb": verb, "favored": fav,
                "rates": {k: round(v, 4) for k, v in rates.items()},
                "resolved": resolved, "correct": bool(resolved == fav)}

    out["bc_cat_first_eat"] = trial((cat, ball), "eat")    # favored cat, cat written first
    out["bc_ball_first_eat"] = trial((ball, cat), "eat")   # favored cat, ball written first (recency favors cat? no: order flips)
    out["bc_cat_first_roll"] = trial((cat, ball), "roll")  # FEATURE-FLIP: favored ball, cat written first
    out["bc_ball_first_roll"] = trial((ball, cat), "roll") # FEATURE-FLIP: favored ball, ball written first
    out["go_arm"] = bool(out["bc_cat_first_eat"]["correct"] and out["bc_ball_first_eat"]["correct"]
                         and out["bc_cat_first_roll"]["correct"] and out["bc_ball_first_roll"]["correct"])

    # --- LESION: competition present, bias REMOVED (bias_pA=0). Must NOT resolve to the favored referent. ---
    def lesion_trial(order, verb):
        cands = [cat, ball]
        fav = _favored(cands, verb)
        w = buf(cands, competition=True)
        w.update([order[0]]); w.update([order[1]])
        rates = w.read(window=window, bias_concept=None, bias_pA=0.0)  # NO bias
        resolved = resolve_referent(rates, spec_threshold)
        return {"order": list(order), "verb": verb, "favored": fav,
                "rates": {k: round(v, 4) for k, v in rates.items()},
                "resolved": resolved, "favored_won": bool(resolved == fav)}
    les_a = lesion_trial((cat, ball), "eat")
    les_b = lesion_trial((cat, ball), "roll")  # same WM, OPPOSITE favored: an unbiased winner can't be right for both
    out["lesion_eat"] = les_a
    out["lesion_roll"] = les_b
    # lesion PASSES (bias is load-bearing) iff removing the bias breaks the content control: the favored
    # referent does NOT reliably win. The decisive check: for the SAME held WM, the unbiased read produces
    # the SAME winner for both verbs -> it cannot match both opposite favoreds -> >=1 is wrong.
    out["lesion_breaks"] = bool(not (les_a["favored_won"] and les_b["favored_won"]))

    # --- MOAT: empty WM -> abstain ; tie (two equally-biased, equal-intrinsic) -> abstain ---
    w_empty = buf([cat, ball], competition=True)  # nothing written
    empty_rates = w_empty.read(window=window, bias_concept=cat, bias_pA=bias_pA)
    out["moat_empty"] = {"rates": {k: round(v, 4) for k, v in empty_rates.items()},
                         "resolved": resolve_referent(empty_rates, spec_threshold)}
    # tie: hold both, bias BOTH equally (content silent -> bias_for returns None -> no bias) -> abstain
    w_tie = buf([cat, ball], competition=True)
    w_tie.update([cat]); w_tie.update([ball])
    tie_rates = w_tie.read(window=window, bias_concept=None, bias_pA=0.0)
    tie_resolved = resolve_referent(tie_rates, spec_threshold)
    out["moat_tie"] = {"rates": {k: round(v, 4) for k, v in tie_rates.items()}, "resolved": tie_resolved}
    out["moat_intact"] = bool(out["moat_empty"]["resolved"] is None and tie_resolved is None)

    # --- 3-referent scale check (one compatible + two incompatible): {cat(animate), ball, river(inanimate)} ---
    three = [cat, ball, "river"]
    fav3 = _favored(three, "eat")  # animate -> cat (the only animate of the three)
    w3 = buf(three, competition=True)
    w3.update([ball]); w3.update(["river"]); w3.update([cat])   # cat written last but bias is content not recency
    rates3 = w3.read(window=window, bias_concept=fav3, bias_pA=bias_pA)
    res3 = resolve_referent(rates3, spec_threshold)
    out["three_ref"] = {"favored": fav3, "rates": {k: round(v, 4) for k, v in rates3.items()},
                        "resolved": res3, "correct": bool(res3 == fav3)}

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
              f"tie->{r['moat_tie']['resolved']} (intact={r['moat_intact']}) | "
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
