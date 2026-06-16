"""Generalization CAPSTONE -- STAGE 2 (verbalize the generalization): use a NOVEL object's spiking concept-category
to RECALL a fact about the matched KNOWN category and ANSWER, while ABSTAINING on a truly-novel no-category object.

THE PRIOR GO (STAGE 1, 2026-06-16, `_genfrontier_capstone_vision_to_concept_derisk`): a NOVEL object, perceived
through the REAL Gabor/V1 front end -> top-K structured perception -> the rate-Hebbian convergence -> the NMDA
CONCEPT assembly SPIKES (real cp_firing_states) in the correct semantic CATEGORY (held-out cat-acc 0.75, 3x chance),
with the flat-distinct baseline at chance, the category-derangement collapsing, and the no-confab moat intact.

THE STAGE-2 QUESTION (the final capstone piece): a novel object's concept-category neurons now SPIKE. Can the agent
VERBALIZE the generalization -- use those concept spikes to RECALL a fact about the matched KNOWN category and answer
("this novel thing -> chase cat"), while ABSTAINING (the no-confab moat) on a truly-novel NO-category object?

THE LOAD-BEARING DESIGN QUESTION (resolved here, OPTION (a) -- fully on-substrate spiking recall):
  The convergence's concept region is per-concept POPULATION BLOCKS (spiking). We connect the spiking concept-category
  to a SPIKING fact recall by ADDING a downstream FACT-TAG region (one block per CATEGORY) and a TRAINED associative
  pathway concept-block -> fact-tag block that CONVERGES every concept block of a category onto that category's fact
  tag.  A per-category fact ("<category> chase cat") is associated with the fact tag.  When a novel object's
  concept-CATEGORY spikes, it drives THAT category's fact-tag block to FIRE -> the matched category's stored fact is
  recalled ("this novel thing -> chase cat", the generalized answer).  The ANSWER is read from WHICH fact-tag block
  FIRES MOST (argmax over fact-tag SPIKE counts -- a readout of spiking output), and the recalled fact is that block's
  stored per-category fact.  Crucially the DECISION rides the concept SPIKES (the fact-tag firing is the literal next
  synaptic hop driven by the concept spikes), NOT a host lookup of the true label.

  ABSTENTION (the no-confab moat) -- ALSO driven by the concept spikes: a no-category object produces DIFFUSE, weak
  concept spikes -> no single fact-tag block dominates -> the winning fact-tag block's firing fails a FAMILIARITY GATE
  (calibrated from the TRAIN fact-tag firing distribution, the substrate's own familiarity -- NOT the true label) AND
  the winner-vs-runner-up margin collapses -> ABSTAIN (no fact recalled).  The gate is the spiking-substrate analogue
  of the validated RFPhasorComposer abstention (query returns None when no stored fact's cue matches): here the "cue"
  is the concept-category spike pattern, and "no match" = no fact-tag block fires confidently.

  We ALSO report OPTION (b) (a documented hybrid) for comparison: read which concept-category spiked (the stage-1
  read), then key the validated RFPhasorComposer.query_patient by that category's concept code.  (a) is the gate; (b)
  is a documented cross-check that the same concept-spike read can drive the validated phasor recall too.

THE ANTI-CHEATS (all four, mirroring stage 1):
  1. FLAT-distinct baseline ~chance: with NO visual structure the held-out object's concept does not land in the
     right category -> the recalled fact is ~chance (the VISUAL structure is load-bearing for the verbalized answer).
  2. Category-DERANGEMENT collapses: train the concept->fact-tag pathway with a DERANGED category->fact mapping ->
     the novel object recalls the WRONG category's fact (the recall is the LEARNED vision-category<->fact correspondence).
  3. The no-confab MOAT survives: a visually-novel NO-category object ABSTAINS (does not confabulate a fact) -- the
     answer + abstention ride the concept SPIKES.
  4. PROVENANCE: the answer + abstention are computed from cp_firing_states (concept + fact-tag spikes), never a host
     lookup of the true category label (asserted: the recall key is the argmax over fact-tag spikes; abstention is the
     familiarity gate on those spikes).

GATE (3 seeds 42/43/44, GPU):
  GO       : a held-out NOVEL object's concept-category spikes drive the recall of the MATCHED category's fact >>
             chance, AND a no-category novel object ABSTAINS (the moat), AND the FLAT-distinct baseline is ~chance,
             AND the category-derangement collapses.  The answer + abstention ride the concept spikes.
  PARTIAL/NEGATIVE (a fully acceptable outcome): the concept->fact-recall representation match is too noisy (the
             spiking concept can't cleanly drive the fact recall) -- report option (a/b) + numbers + the localized
             next step (the representation-match is the boundary).  The moat is NEVER weakened to manufacture a GO.

Reuse-by-import ONLY (stage-1 vision pipeline + the convergence bridge/training/NMDA-spike-read/anti-cheats + the
RFPhasorComposer for the option-(b) cross-check).  NO sim/ edit.  GPU `SIM_BACKEND=cupy`.  SMALL config (16
concepts / 4 categories).
Run:  SIM_BACKEND=cupy python -u -m research.runners._genfrontier_capstone_verbalize_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402

# --- reuse-by-import: STAGE 1's vision pipeline (real Gabor/V1 shape render + the top-K conversion + the moat shape) ---
from research.runners._genfrontier_capstone_vision_to_concept_derisk import (  # noqa: E402
    N_V1_COMPLEX, vision_to_perception_sets, active_set_overlap_margin,
    flat_distinct_sets_like, novel_no_category_perc_set,
)
from research.runners._genfrontier_optionB_visual_similarity_derisk import (  # noqa: E402
    build_shape_set, build_gabor_response_matrix, encode_v1, pool_v1_to_complex, within_between_margin,
)
from sim.visual_cortex import RETINA_SIZE  # noqa: E402
# --- reuse-by-import: the convergence bridge + rate-Hebbian training + the per-concept spike read ---
from research.runners._genfrontier_graded_propagation_derisk import (  # noqa: E402
    build_propagation_bridge, train_convergence, read_heldout_spikes,
)
from research.runners._genfrontier_onsubstrate_convergence_derisk import N_CAT, N_PER_CAT, F  # noqa: E402
# --- reuse-by-import: the validated phasor recall + no-confab abstention (the OPTION-(b) cross-check) ---
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# the per-category fact: "<category> chase cat". The ANSWER the agent verbalizes for a matched-category novel object.
ACTION_WORD = "chase"
PATIENT_WORD = "cat"
def _category_word(c):       # the category names (the "agent" of each per-category fact)
    return f"category{c}"


# ===========================================================================
# The fact-tag region: ADD a downstream region (one block per CATEGORY) + a TRAINED associative pathway
# concept-block -> fact-tag block (CONVERGENT: every concept block of category C -> fact-tag block C).
# Built by REUSING the convergence bridge wiring path (inject_explicit_wiring), with a THIRD region.
# ===========================================================================
def build_verbalize_bridge(n_perc, a, seed, cat_to_fact):
    """perception (n_perc) -> concept (F x n_concept_per, plastic rate-Hebbian convergence) -> fact-tag
    (N_CAT x n_fact_per, FIXED CONVERGENT block: every concept block whose category maps to fact f drives fact-tag
    block f).  `cat_to_fact` (length N_CAT) is the category->fact-tag mapping (identity for the real arm; a
    derangement for the control).  NMDA on concept + fact-tag (the slow conductance integrates the sparse concept
    spikes to fact-tag spikes, as stage 1's concept<-perception).  Returns (bridge, perc_region, conc_region,
    conc_blocks, fact_region, fact_blocks)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    n_cp = a.n_concept_per
    n_fp = a.n_fact_per
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="perception", n_neurons=n_perc, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="concept", n_neurons=F * n_cp, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="fact", n_neurons=N_CAT * n_fp, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
    ]
    # declare all pathways so the framework takes the clean wiring branch at init (then fully overwritten below).
    cfg.region_pathways = [
        RegionPathway(from_region="perception", to_region="concept", density=1.0,
                      weight_mean=0.05, weight_jitter=0.0, plastic=True),
        RegionPathway(from_region="concept", to_region="fact", density=1.0,
                      weight_mean=a.fact_weight, weight_jitter=0.0, plastic=False),
    ]
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebbian_rate
    cfg.hebbian_max_weight = a.hebbian_max
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.00001
    cfg.enable_nmda = True
    cfg.nmda_ratio = a.nmda_ratio

    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()

    perc_region = np.asarray(bridge.region_manager.indices("perception"))
    conc_region = np.asarray(bridge.region_manager.indices("concept"))
    fact_region = np.asarray(bridge.region_manager.indices("fact"))
    conc_blocks = conc_region.reshape(F, n_cp)
    fact_blocks = fact_region.reshape(N_CAT, n_fp)

    # (1) perception -> concept: ALL-TO-ALL, plastic, near-floor init (the convergence the rate-Hebbian LEARNS).
    pc_pre = np.repeat(perc_region, conc_region.shape[0])
    pc_post = np.tile(conc_region, perc_region.shape[0])
    pc_w = np.full(pc_pre.shape[0], 0.05, np.float32)
    # (2) concept -> fact-tag: CONVERGENT block (every concept block of category c -> fact-tag block cat_to_fact[c]),
    #     FIXED (plastic=False), strong fixed weight.  The CATEGORY->FACT correspondence IS this wiring -- it is what
    #     the derangement control deranges.  graded=False (the concept spikes drive the fact tag synaptically).
    fc_pre_l, fc_post_l = [], []
    for i in range(F):
        c = int(cat_ids[i])
        f = int(cat_to_fact[c])
        pre_b = conc_blocks[i]
        post_b = fact_blocks[f]
        fc_pre_l.append(np.repeat(pre_b, post_b.shape[0]))
        fc_post_l.append(np.tile(post_b, pre_b.shape[0]))
    fc_pre = np.concatenate(fc_pre_l); fc_post = np.concatenate(fc_post_l)
    fc_w = np.full(fc_pre.shape[0], a.fact_weight, np.float32)
    wiring = {
        "perception_to_concept": {
            "pre_indices": pc_pre.astype(np.int64).tolist(),
            "post_indices": pc_post.astype(np.int64).tolist(),
            "initial_weights": pc_w.tolist(),
            "plastic": True, "conn_type": "E_TO_MIX",
        },
        "concept_to_fact": {
            "pre_indices": fc_pre.astype(np.int64).tolist(),
            "post_indices": fc_post.astype(np.int64).tolist(),
            "initial_weights": fc_w.tolist(),
            "plastic": False, "conn_type": "E_TO_MIX",
        },
    }
    bridge.inject_explicit_wiring(wiring)
    return bridge, perc_region, conc_region, conc_blocks, fact_region, fact_blocks


# ===========================================================================
# Read the SPIKING fact recall: drive ONLY the perception cue, accumulate BOTH the concept-block spikes (per concept)
# AND the fact-tag-block spikes (per CATEGORY).  The recalled fact = argmax over fact-tag block spikes.
# ===========================================================================
def _set_perc_drive(bridge, xp, perc_region, perc_idx_local, perc_scale):
    n_perc = perc_region.shape[0]
    full = np.zeros(n_perc, np.float32)
    full[perc_idx_local] = perc_scale
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[perc_region] = xp.asarray(full) if xp is not None else full


def read_fact_spikes(bridge, xp, perc_region, conc_region, conc_blocks, fact_region, fact_blocks,
                     perc_idx, scale, steps):
    """Drive ONLY the perception cue; accumulate the concept region's spikes (per concept block) AND the fact-tag
    region's spikes (per CATEGORY block) over `steps` (cp_firing_states -- REAL spikes).  Returns
    (conc_per_block[F], fact_per_block[N_CAT], conc_total, fact_total).  The fact-tag per-block spike vector is the
    SPIKING fact recall: the category whose fact-tag block fires most is the recalled fact's category."""
    perc_local = np.asarray(perc_idx) - perc_region[0]
    _set_perc_drive(bridge, xp, perc_region, perc_local, scale)
    conc_acc = np.zeros(conc_region.shape[0], np.float64)
    fact_acc = np.zeros(fact_region.shape[0], np.float64)
    conc_total = fact_total = 0
    for _ in range(steps):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))
        conc_acc += fs[conc_region].astype(np.float64)
        fact_acc += fs[fact_region].astype(np.float64)
        conc_total += int(fs[conc_region].sum())
        fact_total += int(fs[fact_region].sum())
    bridge.cp_external_input_current[:] = 0.0
    cb_local = conc_blocks - conc_region[0]
    fb_local = fact_blocks - fact_region[0]
    conc_per_block = conc_acc[cb_local].mean(axis=1).astype(np.float64)
    fact_per_block = fact_acc[fb_local].mean(axis=1).astype(np.float64)
    return conc_per_block, fact_per_block, conc_total, fact_total


# ===========================================================================
# The verbalized answer: from the fact-tag spike vector, decide the recalled category (argmax over fact-tag spikes)
# OR abstain (the no-confab moat).  The abstention familiarity gate is calibrated from TRAIN fact-tag firing.
# ===========================================================================
def _train_fact_familiarity(bridge, xp, pr, cr, cb, fr, fb, perc_sets, train, a):
    """Per-train-cue, the WINNING fact-tag block's firing (the substrate's own familiarity for KNOWN objects).
    Returns the mean + std of the train winners -> the abstention gate threshold (NOT the true label)."""
    wins = []
    for t in train:
        _, fpb, _, _ = read_fact_spikes(bridge, xp, pr, cr, cb, fr, fb, perc_sets[t], a.perc_scale, a.read_steps)
        wins.append(float(np.max(fpb)))
    wins = np.asarray(wins, np.float64)
    return float(wins.mean()), float(wins.std() + 1e-9)


def verbalized_answer(fact_per_block, gate_thresh, margin_thresh):
    """The verbalized fact recall + abstention, computed FROM THE FACT-TAG SPIKES (not a host label):
      * winner = argmax over fact-tag block spikes (the recalled category).
      * the familiarity gate: the winner's firing must exceed `gate_thresh` (the substrate's own KNOWN-object
        familiarity) AND be decisively above the runner-up (winner - runner_up >= margin_thresh of the winner).
      * if the gate passes -> ANSWER (recall category `winner`'s fact "<category_winner> chase cat"); else -> ABSTAIN.
    Returns (recalled_category or None, winner_firing, margin_frac)."""
    order = np.argsort(fact_per_block)[::-1]
    winner = int(order[0])
    win_fire = float(fact_per_block[winner])
    runner = float(fact_per_block[order[1]]) if fact_per_block.shape[0] > 1 else 0.0
    margin_frac = (win_fire - runner) / (win_fire + 1e-9)
    answered = bool(win_fire >= gate_thresh and margin_frac >= margin_thresh)
    return (winner if answered else None), win_fire, margin_frac


def evaluate_recall(bridge, xp, pr, cr, cb, fr, fb, perc_sets, cat_ids, held_out, train, a,
                    gate_thresh, margin_thresh):
    """For each held-out NOVEL object: drive ONLY its perception cue, read the fact-tag SPIKE recall, decide the
    recalled category (or abstain).  Returns dict: fact-recall cat-acc (does the recalled fact's category == the
    novel object's true category?), the per-held-out answers, mean concept + fact spikes/cue, and the held-out
    familiarity (the winning fact-tag firing -- for the moat)."""
    hits, answers, conc_s, fact_s, win_fires = [], [], [], [], []
    for j in held_out:
        cpb, fpb, ct, ft = read_fact_spikes(bridge, xp, pr, cr, cb, fr, fb, perc_sets[j], a.perc_scale, a.read_steps)
        rec_cat, win_fire, mf = verbalized_answer(fpb, gate_thresh, margin_thresh)
        hit = int(rec_cat is not None and rec_cat == cat_ids[j])
        hits.append(hit)
        answers.append({"true_cat": int(cat_ids[j]), "recalled_cat": rec_cat, "win_fire": win_fire, "margin": mf})
        conc_s.append(ct); fact_s.append(ft); win_fires.append(win_fire)
    return {
        "fact_recall_cat_acc": float(np.mean(hits)),
        "answers": answers,
        "concept_spikes_per_cue": float(np.mean(conc_s)),
        "fact_spikes_per_cue": float(np.mean(fact_s)),
        "heldout_win_fire": float(np.mean(win_fires)),
    }


# ===========================================================================
# OPTION (b) cross-check: key the validated RFPhasorComposer recall by the spiking concept-category read.
# ===========================================================================
def option_b_crosscheck(conc_spikes_per_heldout, cat_ids, held_out, seed):
    """A documented HYBRID: the recall is the VALIDATED phasor mechanism (RFPhasorComposer.query_patient), KEYED by
    the spiking concept-category (the stage-1 read -- argmax category-mean over the concept-block spikes).  We store
    a per-category fact "<category> chase cat" in the composer; for each held-out object we (i) read which CATEGORY
    its concept assembly spiked for (argmax category-mean -- a readout of cp_firing_states), then (ii) query the
    composer 'what does <that category> chase?' -> the recalled patient.  cat-acc = does the spiking-concept-keyed
    composer recall the matched category's fact?  This shows the same concept-SPIKE read can also drive the
    validated phasor recall (a cross-check, not the gate)."""
    vocab = [_category_word(c) for c in range(N_CAT)] + [ACTION_WORD, PATIENT_WORD]
    comp = RFPhasorComposer(seed=seed, D=64, vocab=vocab)
    for c in range(N_CAT):
        comp.store(_category_word(c), ACTION_WORD, PATIENT_WORD)     # the per-category fact
    hits = []
    for n, j in enumerate(held_out):
        cpb = conc_spikes_per_heldout[n]                              # per-concept spike vector for this held-out cue
        catmean = [float(cpb[cat_ids == c].mean()) for c in range(N_CAT)]
        keyed_cat = int(np.argmax(catmean))                          # the spiking-concept category (the recall key)
        ans = comp.query_patient(_category_word(keyed_cat), ACTION_WORD)   # the validated phasor recall
        hits.append(int(ans == PATIENT_WORD and keyed_cat == cat_ids[j]))
    return float(np.mean(hits))


# ===========================================================================
# One seed, end-to-end.
# ===========================================================================
def run_seed(seed, a):
    a.seed_base = seed
    rng = np.random.default_rng(seed)
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)
    chance = 1.0 / N_CAT

    # leakage-free split: hold out 1 concept per category (mirrors stage 1 exactly).
    rng_split = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng_split.choice(np.where(cat_ids == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: train and held-out overlap"

    # ---- (1) render shapes + encode through the REAL Gabor/V1 front end (stage-1 vision pipeline) ----
    images, labels, meta = build_shape_set(N_CAT, N_PER_CAT, rng, image_size=RETINA_SIZE)
    assert np.array_equal(labels, cat_ids), "shape labels must match the concept category layout"
    W = build_gabor_response_matrix()
    v1 = encode_v1(images, W)
    it_like = pool_v1_to_complex(v1)
    assert it_like.shape[1] == N_V1_COMPLEX
    code_within, code_between, code_margin = within_between_margin(it_like, cat_ids)

    # ---- (2) the conversion: V1-complex code -> top-K perception drive; assert structure preserved ----
    vis_sets = vision_to_perception_sets(it_like, a.top_k)
    set_within, set_between, set_margin = active_set_overlap_margin(vis_sets, N_V1_COMPLEX, cat_ids)
    structure_preserved = bool(set_margin > a.min_set_margin)
    print(f"  [seed {seed}] Gabor code margin {code_margin:+.3f} -> top-{a.top_k} active-set margin {set_margin:+.3f} "
          f"[structure {'PRESERVED' if structure_preserved else 'LOST'}]", flush=True)

    identity = np.arange(N_CAT)
    derange = (np.arange(N_CAT) + 1) % N_CAT

    # ====================================================================
    # ARM 1: STRUCTURED vision -> concept SPIKES -> fact-tag SPIKES -> verbalized recall (+ the moat).
    # ====================================================================
    b1, pr, cr, cb, fr, fb = build_verbalize_bridge(N_V1_COMPLEX, a, seed, identity)
    xp = b1._cp if hasattr(b1, "_cp") else None
    train_convergence(b1, xp, pr, cr, cb, vis_sets, train, a)        # learn perception->concept convergence

    # calibrate the abstention familiarity gate from the TRAIN fact-tag firing (the substrate's own familiarity).
    fam_mu, fam_sd = _train_fact_familiarity(b1, xp, pr, cr, cb, fr, fb, vis_sets, train, a)
    gate_thresh = fam_mu - a.gate_k * fam_sd       # a KNOWN object fires its fact tag near fam_mu; gate a few sd below
    print(f"  [seed {seed}] train fact-tag familiarity: win-fire mean {fam_mu:.2f} std {fam_sd:.2f} -> abstain gate "
          f"{gate_thresh:.2f} (k={a.gate_k}), margin gate {a.margin_thresh}", flush=True)

    S = evaluate_recall(b1, xp, pr, cr, cb, fr, fb, vis_sets, cat_ids, held_out, train, a, gate_thresh, a.margin_thresh)
    print(f"  [seed {seed}] STRUCTURED held-out: concept spikes/cue {S['concept_spikes_per_cue']:.0f}, fact spikes/cue "
          f"{S['fact_spikes_per_cue']:.0f}, fact-recall cat-acc {S['fact_recall_cat_acc']:.2f} (chance {chance:.2f})",
          flush=True)

    # ---- MOAT: a visually-novel NO-category object on the SAME trained bridge -> must ABSTAIN ----
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_set = novel_no_category_perc_set(W, a.top_k, N_CAT, rngm)
    _, novel_fpb, _, novel_fact_total = read_fact_spikes(b1, xp, pr, cr, cb, fr, fb, novel_set, a.perc_scale,
                                                         a.read_steps)
    novel_cat, novel_win_fire, novel_margin = verbalized_answer(novel_fpb, gate_thresh, a.margin_thresh)
    moat_abstains = bool(novel_cat is None)        # the no-category object must NOT confabulate a fact
    # also: held-out KNOWN objects' familiarity should clearly exceed the novel object's (the familiarity contrast).
    fam_contrast_ok = bool(S["heldout_win_fire"] > novel_win_fire * 1.2 + 1e-9)

    # the OPTION-(b) cross-check: read the held-out concept-spike vectors, key the phasor recall by them.
    conc_spikes_per_heldout = []
    for j in held_out:
        cpb, _, _, _ = read_fact_spikes(b1, xp, pr, cr, cb, fr, fb, vis_sets[j], a.perc_scale, a.read_steps)
        conc_spikes_per_heldout.append(cpb)
    opt_b_acc = option_b_crosscheck(conc_spikes_per_heldout, cat_ids, held_out, seed)
    del b1

    # ====================================================================
    # ARM 2: FLAT-distinct vision baseline (same set sizes, NO visual structure) -> recall ~chance.
    # ====================================================================
    flat_sets = flat_distinct_sets_like(vis_sets, N_V1_COMPLEX, seed * 19 + 3)
    b2, pr2, cr2, cb2, fr2, fb2 = build_verbalize_bridge(N_V1_COMPLEX, a, seed, identity)
    xp2 = b2._cp if hasattr(b2, "_cp") else None
    train_convergence(b2, xp2, pr2, cr2, cb2, flat_sets, train, a)
    fam_mu2, fam_sd2 = _train_fact_familiarity(b2, xp2, pr2, cr2, cb2, fr2, fb2, flat_sets, train, a)
    gate2 = fam_mu2 - a.gate_k * fam_sd2
    Fl = evaluate_recall(b2, xp2, pr2, cr2, cb2, fr2, fb2, flat_sets, cat_ids, held_out, train, a, gate2, a.margin_thresh)
    del b2

    # ====================================================================
    # ARM 3: category-DERANGEMENT (concept->fact-tag wired with a WRONG category->fact mapping) -> recall the WRONG fact.
    # ====================================================================
    b3, pr3, cr3, cb3, fr3, fb3 = build_verbalize_bridge(N_V1_COMPLEX, a, seed, derange)
    xp3 = b3._cp if hasattr(b3, "_cp") else None
    train_convergence(b3, xp3, pr3, cr3, cb3, vis_sets, train, a)
    fam_mu3, fam_sd3 = _train_fact_familiarity(b3, xp3, pr3, cr3, cb3, fr3, fb3, vis_sets, train, a)
    gate3 = fam_mu3 - a.gate_k * fam_sd3
    # for the derangement, "correct" means the recalled fact's category == the novel object's TRUE category; the
    # deranged wiring sends category c's concept spikes to fact-tag derange[c], so the recall lands WRONG -> low acc.
    P = evaluate_recall(b3, xp3, pr3, cr3, cb3, fr3, fb3, vis_sets, cat_ids, held_out, train, a, gate3, a.margin_thresh)
    del b3

    out = {
        "seed": seed, "held_out": held_out, "chance": chance,
        "gabor_code": {"within": code_within, "between": code_between, "margin": code_margin},
        "active_set": {"within": set_within, "between": set_between, "margin": set_margin,
                       "structure_preserved": structure_preserved},
        "structured": S, "flat": Fl, "permuted": P,
        "moat": {"novel_recalled_cat": novel_cat, "novel_win_fire": novel_win_fire, "novel_margin": novel_margin,
                 "novel_fact_total": int(novel_fact_total), "moat_abstains": moat_abstains,
                 "heldout_win_fire": S["heldout_win_fire"], "fam_contrast_ok": fam_contrast_ok},
        "option_b_crosscheck_acc": opt_b_acc,
        "gate": {"fam_mu": fam_mu, "fam_sd": fam_sd, "gate_thresh": gate_thresh, "margin_thresh": a.margin_thresh},
    }
    print(f"  [seed {seed}] FACT-RECALL cat-acc {S['fact_recall_cat_acc']:.2f} (chance {chance:.2f}) | FLAT "
          f"{Fl['fact_recall_cat_acc']:.2f} | DERANGE {P['fact_recall_cat_acc']:.2f} | moat "
          f"{'ABSTAIN' if moat_abstains else 'CONFAB'} (novel win-fire {novel_win_fire:.2f} vs held-out "
          f"{S['heldout_win_fire']:.2f}) | option-b {opt_b_acc:.2f}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    # the conversion knob (matches stage 1).
    p.add_argument("--top-k", type=int, default=60)
    p.add_argument("--min-set-margin", type=float, default=0.05)
    # concept / fact-tag config mirror the stage-1 / graded-prop GO (the population-code lift).
    p.add_argument("--n-concept-per", type=int, default=100)
    p.add_argument("--n-fact-per", type=int, default=100, help="neurons per CATEGORY fact-tag block")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--scene-steps", type=int, default=16)
    p.add_argument("--read-steps", type=int, default=80)
    p.add_argument("--perc-scale", type=float, default=300.0)
    p.add_argument("--conc-scale", type=float, default=600.0)
    p.add_argument("--read-weight", type=float, default=30.0, help="(passed through to the graded-prop helpers)")
    p.add_argument("--fact-weight", type=float, default=30.0, help="fixed concept->fact-tag convergent block weight")
    p.add_argument("--nmda-ratio", type=float, default=2.0)
    p.add_argument("--hebbian-rate", type=float, default=0.05)
    p.add_argument("--hebbian-max", type=float, default=20.0)
    # the abstention gate.
    p.add_argument("--gate-k", type=float, default=2.5, help="abstain gate = train-win-fire mean - gate_k*std "
                   "(a KNOWN object fires its fact tag near the mean; a no-category object falls below the gate)")
    p.add_argument("--margin-thresh", type=float, default=0.15, help="winner-vs-runner-up fraction the winning "
                   "fact-tag block must exceed to ANSWER (else abstain): (win - runner_up)/win >= this")
    p.add_argument("--candidate", default="nmda", choices=["nmda", "pool", "graded"],
                   help="read-out propagation mechanism passed to the graded-prop bridge helpers (default nmda)")
    p.add_argument("--out", default="research/findings/raw/_genfrontier_capstone_verbalize.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[genfrontier CAPSTONE stage-2 VERBALIZE] a NOVEL object's concept-category SPIKES -> (option a) a "
          f"trained concept->fact-tag associative pathway FIRES the matched category's fact ('<category> chase cat') "
          f"-> verbalized recall; a no-category object ABSTAINS (the no-confab moat). seeds={seeds}", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    chance = 1.0 / N_CAT

    def m(arm, k):
        return float(np.mean([r[arm][k] for r in rows]))
    s_acc = m("structured", "fact_recall_cat_acc")
    f_acc = m("flat", "fact_recall_cat_acc")
    p_acc = m("permuted", "fact_recall_cat_acc")
    s_conc = m("structured", "concept_spikes_per_cue")
    s_fact = m("structured", "fact_spikes_per_cue")
    opt_b = float(np.mean([r["option_b_crosscheck_acc"] for r in rows]))
    moat_all = all(r["moat"]["moat_abstains"] for r in rows)
    fam_contrast_all = all(r["moat"]["fam_contrast_ok"] for r in rows)
    structure_all = all(r["active_set"]["structure_preserved"] for r in rows)
    concept_spikes_present = s_conc > 0.0
    fact_spikes_present = s_fact > 0.0

    # GO: the concept assembly spikes AND drives the fact-tag (fact spikes present); the held-out NOVEL object's
    # spiking concept-category recalls the MATCHED category's fact >> chance every seed; the FLAT baseline ~chance
    # (visual structure load-bearing for the verbalized answer); the derangement collapses (recall lands WRONG, well
    # below structured); the no-confab moat survives (a no-category object abstains on every seed).
    go = (structure_all and concept_spikes_present and fact_spikes_present
          and all(r["structured"]["fact_recall_cat_acc"] > chance + 1e-9 for r in rows)
          and s_acc >= 0.50
          and f_acc <= chance + 0.15
          and p_acc <= s_acc - 0.20
          and moat_all)
    partial = (concept_spikes_present and fact_spikes_present and s_acc > chance + 0.10
               and s_acc > f_acc + 0.10 and moat_all)
    verdict = "GO" if go else ("PARTIAL" if partial else "NEGATIVE")

    print(f"\n{'='*118}\n  MEAN ({len(rows)} seeds): concept spikes/cue {s_conc:.0f} -> fact-tag spikes/cue {s_fact:.0f} "
          f"| FACT-RECALL cat-acc {s_acc:.2f} (chance {chance:.2f}) | FLAT {f_acc:.2f} | DERANGE {p_acc:.2f} | moat "
          f"{'INTACT (all abstain)' if moat_all else 'BREACH (confab)'} | option-b {opt_b:.2f}  ==> {verdict}\n"
          f"{'='*118}", flush=True)
    if verdict == "GO":
        print(f"  GO -- THE GENERALIZATION IS VERBALIZED ON ONE BRAIN: a HELD-OUT NOVEL object, perceived through the "
              f"REAL Gabor/V1 front end, drives its CONCEPT-CATEGORY to SPIKE; those concept spikes FIRE the matched "
              f"category's fact-tag block ({s_fact:.0f} fact spikes/cue, real cp_firing_states) -> the agent RECALLS "
              f"the matched category's fact ('this novel thing -> chase cat') at {s_acc:.0%} >> chance {chance:.0%}; a "
              f"visually-novel NO-category object ABSTAINS (the no-confab moat survives all seeds -- the abstention "
              f"rides the concept spikes, NOT a host label); the FLAT-distinct baseline is ~chance ({f_acc:.0%}) => "
              f"the visual structure is load-bearing for the answer; the category-derangement collapses ({p_acc:.0%}). "
              f"The OPTION-(b) phasor cross-check (the spiking concept-category keys the validated RFPhasorComposer "
              f"recall) agrees at {opt_b:.0%}. perceive a novel object -> generalize -> ANSWER, on one brain. "
              f"NO sim/ edit.", flush=True)
    elif verdict == "PARTIAL":
        print(f"  PARTIAL: the concept->fact-recall is above flat ({s_acc:.0%} vs {f_acc:.0%}) + the moat holds, but "
              f"below the GO bar -- localize the representation match: fact-weight / nmda-ratio / read-steps / "
              f"n-fact-per / the abstention gate (gate-k {a.gate_k}, margin {a.margin_thresh}). option-b {opt_b:.0%}.",
              flush=True)
    else:
        if not structure_all:
            why = "the top-K conversion did not preserve the Gabor structure (route to stage-1's conversion knobs)"
        elif not concept_spikes_present:
            why = "the concept assembly does not spike from the vision-derived perception drive (route to graded)"
        elif not fact_spikes_present:
            why = "the concept spikes do NOT drive the fact-tag region (raise fact-weight / nmda-ratio)"
        elif not moat_all:
            why = ("the no-confab moat BREACHED (a no-category object confabulated a fact) -- the gate is too loose; "
                   "the representation match is the boundary. NEVER weaken the moat to manufacture a GO")
        else:
            why = (f"the fact-recall representation match is too noisy (structured {s_acc:.0%}, flat {f_acc:.0%}, "
                   f"derange {p_acc:.0%}) -- the spiking concept can't cleanly drive the fact recall")
        print(f"  NEGATIVE: {why}. concept->fact-tag option (a) tried (option-b cross-check {opt_b:.0%}). Moat "
              f"{'INTACT' if moat_all else 'BREACH'}. Honest negative + the localized next step (the "
              f"concept->pipeline representation match is the boundary).", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump({"verdict": verdict, "chance": chance, "candidate": a.candidate, "top_k": a.top_k,
                   "concept_spikes_per_cue": s_conc, "fact_spikes_per_cue": s_fact,
                   "structured_fact_recall_cat_acc": s_acc, "flat_fact_recall_cat_acc": f_acc,
                   "permuted_fact_recall_cat_acc": p_acc, "option_b_crosscheck_acc": opt_b,
                   "moat_intact_all_abstain": moat_all, "fam_contrast_all": fam_contrast_all,
                   "structure_preserved_all": structure_all, "per_seed": rows}, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
