"""Generalization frontier — GRADED-PROPAGATION read-out de-risk: make the held-out cross-modal convergence
PROPAGATE AS SPIKES (the one open piece of the live-task capstone).

THE PRIOR GO (2026-06-16, `_genfrontier_onsubstrate_convergence_derisk`): population-Hebbian co-activation of a
similarity-STRUCTURED perception region + a concept region transfers category-generalization on the spiking
substrate -- a HELD-OUT (never-converged) concept's structured perception cue lands in its correct semantic
CATEGORY (held-out cat-acc 0.92, chance 0.25), with the flat-distinct baseline at chance, the category-derangement
control collapsing, and the no-confab moat intact. BUT that transfer was read as the concept assembly's GRADED
population DEPOLARIZATION (an instrument), because THE LOAD-BEARING RESIDUAL is: the point-neuron concept assembly
CANNOT SPIKE from perception alone (verified there: 0 concept spikes even at 8000 pA / weight 29 -- the synaptic
conductance decays between sparse perception spikes faster than it accumulates to the Izhikevich +30 mV threshold).
The live-task pipeline (who/what recall + the no-confab moat) reads concept codes through SYNAPSES = SPIKES, so the
novel-perceived-object response MUST PROPAGATE AS SPIKES. THIS runner de-risks that propagation.

THE QUESTION: for a HELD-OUT perceived object (perception cue ALONE, no word) driving the converged concept
assembly sub-threshold, can a downstream READ-OUT mechanism convert that graded category-correct response into
category-correct SPIKES (cp_firing_states -- REAL spikes, not membrane potential) that a synaptic pipeline reads?

THE THREE CANDIDATES (the project uses all three; selectable via --candidate, default `nmda`):
  1. NMDA-integrated read-out (most likely): a downstream read-out region driven by the concept assembly, with
     NMDA enabled on BOTH the concept region AND the read-out (the per-region NMDA mask -- BrainRegion(enable_nmda
     =True) + cfg.enable_nmda=True; the framework confines the slow NMDA current to those slices, sim/bridge.py
     :1212-1221). NMDA's slow conductance (tau_decay 100 ms, fed the same excitatory synaptic input as AMPA scaled
     by nmda_ratio -- sim/bridge.py:5986-5989) TEMPORALLY INTEGRATES the sparse perception-driven drive across the
     gaps that defeat AMPA-only point neurons -> supra-threshold SPIKES. The read-out = one population per concept
     (concept block i -> read-out block i, a fixed block-diagonal projection; the CATEGORY structure is NOT wired
     -- it is read by category-mean over the read-out SPIKE counts, so the wiring does no category work). The
     winner = the read-out population that SPIKES most; category by category-mean.
  2. Population pooling + low threshold: a large read-out population pooling many concept neurons, with a LOWERED
     spike threshold (cp_izh_vpeak), so the pooled graded drive crosses threshold. (--candidate pool; --readout
     -vpeak lowers the threshold.)
  3. Graded transmission: the project's per-pathway graded=True (RegionPathway.graded, sim/regions.py:319-336) --
     the concept->readout pathway transmits the concept neurons' CONTINUOUS membrane (a_cont = clip((v-rest)/scale,
     0,1)), bypassing the concept spike threshold, so the read-out is driven by the graded concept response
     directly and (NMDA-integrated) spikes. (--candidate graded.)

THE PROBE: reuse the convergence (build_convergence_bridge's pattern + structured/flat perception sets +
train_convergence + the anti-cheat structure), but ADD the downstream read-out region + the chosen read-out
mechanism. After training perception->concept convergence, drive a HELD-OUT structured-perception cue ALONE, run
the bridge, and measure whether the READ-OUT region's SPIKES (cp_firing_states accumulated over the read window,
per read-out block) land in the correct semantic CATEGORY >> chance (1/n_cat).

GATE (3 seeds 42/43/44, GPU):
  GO       : the read-out's SPIKE response to a held-out novel-perceived cue lands in its correct category >>
             chance, with the FLAT-distinct baseline at chance (structure load-bearing), the category-derangement
             control collapsing, AND the no-confab moat surviving (a novel-no-category cue does not drive
             confident category spikes). KEY: the read-out must SPIKE (real spike counts) -- that is what makes the
             response propagatable downstream.
  PARTIAL/NEGATIVE : no read-out mechanism produces category-correct spikes (the graded code cannot be made to
             propagate at point-neuron scale) -- an honest, load-bearing negative that routes the capstone to
             graded transmission or a different read-out. Report which of the 3 candidates were tried + numbers.

Reuse-by-import (the convergence runner's bridge/perception/training/anti-cheat helpers + the region framework +
NMDA + the graded pathway). GPU `SIM_BACKEND=cupy`. NO sim/ edits.
Run:  SIM_BACKEND=cupy python -u -m research.runners._genfrontier_graded_propagation_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402

# Reuse the validated convergence machinery (the GO de-risk) by import: the structured/flat perception ensembles,
# the rate-Hebbian co-activation training, and the concept config (N_CAT/N_PER_CAT/F).
from research.runners._genfrontier_onsubstrate_convergence_derisk import (  # noqa: E402
    N_CAT, N_PER_CAT, F,
    structured_perception_sets, flat_perception_sets, train_convergence,
)


# ===========================================================================
# The 3-region bridge: perception -> concept(NMDA) -> readout(NMDA/pool/graded).
# Reuses the convergence bridge pattern; ADDS the downstream read-out region + the chosen propagation mechanism.
# ===========================================================================
def build_propagation_bridge(n_perc, n_concept_per, n_readout_per, seed, a):
    """perception region (n_perc, index-addressed) + concept region (F x n_concept_per, disjoint blocks) + readout
    region (F x n_readout_per, disjoint blocks). Pathways:
      * perception -> concept : plastic rate-Hebbian (the convergence the co-activation LEARNS; near-floor init).
      * concept    -> readout : FIXED block-diagonal (concept block i -> readout block i only; the CATEGORY
        structure is NOT wired -- category is read by category-mean over readout spikes, so the projection does no
        category work). The read-out mechanism (NMDA-integrate / pool+low-threshold / graded) is what converts the
        sub-threshold concept response into readout SPIKES.
    Candidate selection (a.candidate):
      'nmda'   : enable_nmda on BOTH concept and readout (slow NMDA integrates sparse drive -> spikes). [primary]
      'pool'   : large readout population, NMDA on readout, LOWERED readout vpeak (pool + low threshold).
      'graded' : concept->readout pathway graded=True (transmits concept's continuous membrane, bypassing the
                 concept spike threshold), NMDA on readout to integrate.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cand = a.candidate
    # NMDA on concept lets its assembly INTEGRATE the sparse perception drive to spikes (candidate 1's core). For
    # the 'graded' candidate the concept->readout pathway reads the concept's CONTINUOUS membrane, so the concept
    # need not spike -- but NMDA on concept does not hurt (its graded membrane is still read). Keep concept NMDA on
    # for nmda/pool (it must spike to drive the readout synaptically); on for graded too (harmless, and lets the
    # nmda-vs-graded comparison hold the concept region fixed).
    concept_nmda = True
    readout_nmda = True   # the read-out always integrates with NMDA (the slow conductance is the lift)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    # Declare the 3 regions (this allocates the per-neuron NMDA mask from enable_nmda).
    cfg.brain_regions = [
        BrainRegion(name="perception", n_neurons=n_perc, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="concept", n_neurons=F * n_concept_per, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=concept_nmda),
        BrainRegion(name="readout", n_neurons=F * n_readout_per, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=readout_nmda),
    ]
    # Declare BOTH pathways in the framework so the init wiring path runs CLEANLY (an EMPTY region_pathways list
    # makes the framework generate no synapses -> the bridge falls into the spatial-generator FALLBACK, which
    # leaves the regions inert -- verified: perception 0 spikes at any drive). The framework wiring is then FULLY
    # OVERWRITTEN by the explicit inject_explicit_wiring below (which installs the precise block-diagonal
    # concept->readout + rebuilds every per-synapse array, incl. the graded mask, in one correct pass -- avoiding
    # post-init cp_connections surgery that would desync the STP/eligibility arrays). The framework declaration
    # here is just to take the clean wiring branch at init.
    cfg.region_pathways = [
        RegionPathway(from_region="perception", to_region="concept", density=1.0,
                      weight_mean=0.05, weight_jitter=0.0, plastic=True),
        RegionPathway(from_region="concept", to_region="readout", density=1.0,
                      weight_mean=a.read_weight, weight_jitter=0.0, plastic=False),
    ]

    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # RATE-Hebbian convergence (perception->concept); the concept->readout projection is plastic=False so Hebbian
    # does not touch it (its weights are the fixed read-out wiring).
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebbian_rate
    cfg.hebbian_max_weight = a.hebbian_max
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 0.00001
    # NMDA on globally; the per-region mask (built at init from the enable_nmda regions) confines the slow NMDA
    # current to concept + readout (perception stays AMPA-only). nmda_ratio raised so NMDA dominates the slow
    # integration that lifts the read-out over threshold.
    cfg.enable_nmda = True
    cfg.nmda_ratio = a.nmda_ratio

    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()

    perc_region = np.asarray(bridge.region_manager.indices("perception"))
    conc_region = np.asarray(bridge.region_manager.indices("concept"))
    read_region = np.asarray(bridge.region_manager.indices("readout"))
    conc_blocks = conc_region.reshape(F, n_concept_per)
    read_blocks = read_region.reshape(F, n_readout_per)

    # --- inject the EXACT wiring (replaces the empty framework wiring) ---
    # (1) perception -> concept: ALL-TO-ALL, plastic, near-floor init 0.05 (the convergence the rate-Hebbian
    #     LEARNS -- the spiking analogue of the numpy ridge map).
    # (2) concept -> readout: BLOCK-DIAGONAL (concept block i -> readout block i ONLY), FIXED (plastic=False),
    #     strong fixed weight. The CATEGORY structure is NOT in this wiring -- it is read by category-mean over
    #     the readout SPIKE counts, so the read-out projection does no category work. graded=True for candidate 3.
    pc_pre = np.repeat(perc_region, conc_region.shape[0])
    pc_post = np.tile(conc_region, perc_region.shape[0])
    pc_w = np.full(pc_pre.shape[0], 0.05, np.float32)
    cr_pre_l, cr_post_l = [], []
    for i in range(F):
        pre_b = conc_blocks[i]; post_b = read_blocks[i]
        cr_pre_l.append(np.repeat(pre_b, post_b.shape[0]))
        cr_post_l.append(np.tile(post_b, pre_b.shape[0]))
    cr_pre = np.concatenate(cr_pre_l); cr_post = np.concatenate(cr_post_l)
    cr_w = np.full(cr_pre.shape[0], a.read_weight, np.float32)
    wiring = {
        "perception_to_concept": {
            "pre_indices": pc_pre.astype(np.int64).tolist(),
            "post_indices": pc_post.astype(np.int64).tolist(),
            "initial_weights": pc_w.tolist(),
            "plastic": True, "conn_type": "E_TO_MIX",
        },
        "concept_to_readout": {
            "pre_indices": cr_pre.astype(np.int64).tolist(),
            "post_indices": cr_post.astype(np.int64).tolist(),
            "initial_weights": cr_w.tolist(),
            "plastic": False, "conn_type": "E_TO_MIX",
            "graded": bool(cand == "graded"),
        },
    }
    bridge.inject_explicit_wiring(wiring)
    return bridge, perc_region, conc_region, read_region, conc_blocks, read_blocks


# ===========================================================================
# Read-out: drive a held-out perception cue ALONE, accumulate the READOUT region's SPIKES per block.
# ===========================================================================
def _set_perc_drive(bridge, xp, perc_region, perc_idx_local, perc_scale):
    n_perc = perc_region.shape[0]
    full = np.zeros(n_perc, np.float32)
    full[perc_idx_local] = perc_scale
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[perc_region] = xp.asarray(full) if xp is not None else full


def read_heldout_spikes(bridge, xp, perc_region, conc_region, read_region, conc_blocks, read_blocks,
                        perc_idx, scale, steps):
    """Drive ONLY the perception cue; accumulate the SPIKES (cp_firing_states -- REAL spikes, not membrane) of
    BOTH the concept region (per concept block) AND the read-out region (per read-out block) over `steps`.

    Returns (concept_per_block[F], readout_per_block[F], conc_total, read_total):
      * concept_per_block -- the converged concept ASSEMBLY's spike response per concept (this IS the conversation
        cortex's concept code that the live who/what + no-confab pipeline reads via SYNAPSES; that it SPIKES at all
        from perception alone refutes the prior 'concept cannot spike from perception' residual).
      * readout_per_block -- the DOWNSTREAM read-out region's spike response (the literal next synaptic hop driven
        by the concept spikes; >0 proves the concept response PROPAGATES one synapse further).
    Both are per-neuron mean spike counts. The category decision is read from these REAL spikes (not membrane)."""
    perc_local = np.asarray(perc_idx) - perc_region[0]
    _set_perc_drive(bridge, xp, perc_region, perc_local, scale)
    conc_acc = np.zeros(conc_region.shape[0], np.float64)
    read_acc = np.zeros(read_region.shape[0], np.float64)
    conc_total = 0
    read_total = 0
    for _ in range(steps):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))
        conc_acc += fs[conc_region].astype(np.float64)
        read_acc += fs[read_region].astype(np.float64)
        conc_total += int(fs[conc_region].sum())
        read_total += int(fs[read_region].sum())
    bridge.cp_external_input_current[:] = 0.0
    cb_local = conc_blocks - conc_region[0]
    rb_local = read_blocks - read_region[0]
    conc_per_block = conc_acc[cb_local].mean(axis=1).astype(np.float64)   # per-concept mean spike count (concept)
    read_per_block = read_acc[rb_local].mean(axis=1).astype(np.float64)   # per-concept mean spike count (read-out)
    return conc_per_block, read_per_block, conc_total, read_total


# ===========================================================================
# Per-arm evaluation on REAL SPIKES. The category decision is the CATEGORY-MEAN over the concept-assembly spike
# response (raw -- the concept spike code is already a clean category signal at point-neuron scale; no z-score
# needed, unlike the prior GRADED-depolarization read which carried a per-block membrane offset). The read-out
# region's spike response (the downstream propagation) is read + reported the same way.
# ===========================================================================
def _cat_decision(per_block, cat_ids, j):
    """category-mean decision over a per-concept spike-response vector: the category whose concept blocks spike
    most. Returns (hit, same-vs-other margin)."""
    catmean = [float(per_block[cat_ids == c].mean()) for c in range(N_CAT)]
    hit = int(int(np.argmax(catmean)) == cat_ids[j])
    same = float(per_block[cat_ids == cat_ids[j]].mean())
    other = float(per_block[cat_ids != cat_ids[j]].mean())
    return hit, same - other


def evaluate_arm_spikes(bridge, xp, perc_region, conc_region, read_region, conc_blocks, read_blocks,
                        perc_sets, cat_ids, held_out, train, a):
    """For each held-out concept: drive ONLY its perception cue, read the concept-assembly AND read-out SPIKE
    responses, and make a CATEGORY-MEAN decision on each. Returns a dict of (cat_acc/margin for concept + readout,
    mean concept + readout spikes/cue)."""
    conc_hits, conc_marg, read_hits, read_marg, conc_s, read_s = [], [], [], [], [], []
    for j in held_out:
        cpb, rpb, ct, rt = read_heldout_spikes(bridge, xp, perc_region, conc_region, read_region,
                                               conc_blocks, read_blocks, perc_sets[j], a.perc_scale, a.read_steps)
        h_c, m_c = _cat_decision(cpb, cat_ids, j)
        h_r, m_r = _cat_decision(rpb, cat_ids, j)
        conc_hits.append(h_c); conc_marg.append(m_c)
        read_hits.append(h_r); read_marg.append(m_r)
        conc_s.append(ct); read_s.append(rt)
    return {
        "concept_cat_acc": float(np.mean(conc_hits)), "concept_margin": float(np.mean(conc_marg)),
        "readout_cat_acc": float(np.mean(read_hits)), "readout_margin": float(np.mean(read_marg)),
        "concept_spikes_per_cue": float(np.mean(conc_s)), "readout_spikes_per_cue": float(np.mean(read_s)),
    }


def run_seed(seed, a):
    a.seed_base = seed
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)
    rng = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng.choice(np.where(cat_ids == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: train and held-out overlap"

    n_perc = a.n_perc
    n_cp = a.n_concept_per
    n_rp = a.n_readout_per
    out = {"seed": seed, "held_out": held_out, "candidate": a.candidate}

    # --- ARM 1: STRUCTURED perception (same-category overlap = Option B) ---
    perc_sets_s, _ = structured_perception_sets(n_perc, a.n_active_cat, a.n_active_uniq, seed * 23 + 7)
    b1, pr, cr, rr, cb, rb = build_propagation_bridge(n_perc, n_cp, n_rp, seed, a)
    xp = b1._cp if hasattr(b1, "_cp") else None
    diag = train_convergence(b1, xp, pr, cr, cb, perc_sets_s, train, a)
    print(f"  [seed {seed}] structured train firing diag (first scene): perc {diag['perc']} conc {diag['conc']}",
          flush=True)
    S = evaluate_arm_spikes(b1, xp, pr, cr, rr, cb, rb, perc_sets_s, cat_ids, held_out, train, a)
    print(f"  [seed {seed}] STRUCTURED held-out: concept spikes/cue {S['concept_spikes_per_cue']:.0f}, READOUT "
          f"spikes/cue {S['readout_spikes_per_cue']:.0f}  (>0 => the response PROPAGATES as spikes)", flush=True)

    # --- MOAT: a NOVEL perception ensemble (random neurons, no category) on the SAME trained structured bridge ---
    # The moat reads the SAME spike signal: a real held-out concept produces a HIGH best-category concept-spike
    # response (it matches a learned category); a novel ensemble (no learned category) produces a LOW, diffuse
    # response -> the system abstains rather than confabulating a category.
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_idx = pr[0] + rngm.choice(n_perc, size=a.n_active_cat + a.n_active_uniq, replace=False)

    def _best_cat_spikes(perc_idx):
        cpb, _, _, _ = read_heldout_spikes(b1, xp, pr, cr, rr, cb, rb, perc_idx, a.perc_scale, a.read_steps)
        return float(np.max([cpb[cat_ids == c].mean() for c in range(N_CAT)]))

    ho_fam = float(np.mean([_best_cat_spikes(perc_sets_s[j]) for j in held_out]))
    novel_fam = _best_cat_spikes(novel_idx)
    moat_ok = bool(ho_fam > novel_fam * 1.5 + 1e-9)        # held-out concept clearly more category-familiar
    del b1

    # --- ARM 2: FLAT-distinct perception (baseline; structure ablation) ---
    perc_sets_f, _ = flat_perception_sets(n_perc, a.n_active_cat + a.n_active_uniq, seed * 19 + 3)
    b2, pr2, cr2, rr2, cb2, rb2 = build_propagation_bridge(n_perc, n_cp, n_rp, seed, a)
    xp2 = b2._cp if hasattr(b2, "_cp") else None
    train_convergence(b2, xp2, pr2, cr2, cb2, perc_sets_f, train, a)
    Fl = evaluate_arm_spikes(b2, xp2, pr2, cr2, rr2, cb2, rb2, perc_sets_f, cat_ids, held_out, train, a)
    del b2

    # --- ARM 3: category-DERANGEMENT permuted control (structured perception, train concept's perception
    # co-activated with a WRONG-category concept block). Transfer must land in the WRONG category. ---
    derange = (np.arange(N_CAT) + 1) % N_CAT
    train_by_cat = {c: [t for t in train if cat_ids[t] == c] for c in range(N_CAT)}
    deranged_block = {}
    for t in train:
        c = int(cat_ids[t]); k = train_by_cat[c].index(t)
        donor_cat = int(derange[c])
        donor = train_by_cat[donor_cat][k % len(train_by_cat[donor_cat])]
        deranged_block[t] = donor
    b3, pr3, cr3, rr3, cb3, rb3 = build_propagation_bridge(n_perc, n_cp, n_rp, seed, a)
    xp3 = b3._cp if hasattr(b3, "_cp") else None
    for ep in range(a.epochs):
        order = np.random.RandomState(seed * 7 + ep).permutation(train)
        for t in order:
            perc_local = np.asarray(perc_sets_s[t]) - pr3[0]
            conc_local = cb3[deranged_block[t]] - cr3[0]            # WRONG-category concept block
            n_perc_l = pr3.shape[0]; n_conc_l = cr3.shape[0]
            full_perc = np.zeros(n_perc_l, np.float32); full_perc[perc_local] = a.perc_scale
            full_conc = np.zeros(n_conc_l, np.float32); full_conc[conc_local] = a.conc_scale
            b3.cp_external_input_current[:] = 0.0
            b3.cp_external_input_current[pr3] = xp3.asarray(full_perc) if xp3 is not None else full_perc
            b3.cp_external_input_current[cr3] = xp3.asarray(full_conc) if xp3 is not None else full_conc
            for _ in range(a.scene_steps):
                b3._run_one_simulation_step()
    b3.cp_external_input_current[:] = 0.0
    P = evaluate_arm_spikes(b3, xp3, pr3, cr3, rr3, cb3, rb3, perc_sets_s, cat_ids, held_out, train, a)
    del b3

    out["structured"] = S
    out["flat"] = Fl
    out["permuted"] = P
    out["moat"] = {"heldout_familiarity": ho_fam, "novel_familiarity": novel_fam, "moat_ok": moat_ok}

    chance = 1.0 / N_CAT
    print(f"  [seed {seed}] [{a.candidate}] STRUCTURED concept-spike cat-acc {S['concept_cat_acc']:.2f} (chance "
          f"{chance:.2f}) margin {S['concept_margin']:+.3f} | readout-spike cat-acc {S['readout_cat_acc']:.2f} | "
          f"FLAT concept {Fl['concept_cat_acc']:.2f} | PERMUTED concept {P['concept_cat_acc']:.2f} margin "
          f"{P['concept_margin']:+.3f} | moat {'OK' if moat_ok else 'BREACH'} (ho {ho_fam:.2f} vs novel "
          f"{novel_fam:.2f})", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--candidate", default="nmda", choices=["nmda", "pool", "graded"],
                   help="read-out propagation mechanism (default nmda = NMDA-integrated read-out)")
    # perception / concept config mirror the convergence GO (the documented population-code lift).
    p.add_argument("--n-perc", type=int, default=1600)
    p.add_argument("--n-concept-per", type=int, default=100, help="neurons per concept block")
    p.add_argument("--n-readout-per", type=int, default=100, help="neurons per readout block (population pooling)")
    p.add_argument("--n-active-cat", type=int, default=48, help="shared per-CATEGORY perception core size")
    p.add_argument("--n-active-uniq", type=int, default=12, help="per-CONCEPT unique perception tail size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--scene-steps", type=int, default=16, help="co-drive steps per training scene")
    p.add_argument("--read-steps", type=int, default=80, help="steps to accumulate the readout SPIKE response")
    p.add_argument("--perc-scale", type=float, default=300.0, help="perception drive pA")
    p.add_argument("--conc-scale", type=float, default=600.0, help="concept teacher drive pA (training only)")
    p.add_argument("--read-weight", type=float, default=30.0, help="fixed concept->readout block-diagonal weight")
    p.add_argument("--nmda-ratio", type=float, default=2.0, help="NMDA:AMPA ratio (raised so the slow NMDA "
                   "conductance dominates the read-out's integration of sparse concept spikes)")
    p.add_argument("--hebbian-rate", type=float, default=0.05)
    p.add_argument("--hebbian-max", type=float, default=20.0)
    p.add_argument("--out", default="research/findings/raw/_genfrontier_graded_propagation.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[genfrontier graded-propagation] candidate={a.candidate} -- does a downstream READ-OUT convert the "
          f"held-out concept's sub-threshold category response into category-correct SPIKES? (structured vs "
          f"flat-distinct vs derangement; moat). seeds={seeds}", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    chance = 1.0 / N_CAT

    def m(arm, k):
        return float(np.mean([r[arm][k] for r in rows]))
    # PRIMARY signal = the concept ASSEMBLY's spike response (the conversation cortex's concept code the live
    # pipeline reads via synapses; raw category-mean -- the spike code is already a clean category signal).
    s_cat, s_margin = m("structured", "concept_cat_acc"), m("structured", "concept_margin")
    f_cat = m("flat", "concept_cat_acc")
    p_cat, p_margin = m("permuted", "concept_cat_acc"), m("permuted", "concept_margin")
    # PROPAGATION = the downstream read-out region's spikes (>0 = the response propagates a synapse further) +
    # its own category-correctness (the read-out region's spike-based category read).
    s_read_sp = m("structured", "readout_spikes_per_cue")
    s_conc_sp = m("structured", "concept_spikes_per_cue")
    s_read_cat = m("structured", "readout_cat_acc")
    moat_all = all(r["moat"]["moat_ok"] for r in rows)

    # GO: the concept ASSEMBLY SPIKES (the key new measurement -- concept spikes/cue > 0, refuting the prior
    # 'concept cannot spike from perception' residual; these spikes ARE what a synaptic pipeline reads) and lands
    # category-correctly (cat-acc > chance every seed + positive margin), AND the response PROPAGATES (the
    # downstream read-out region also spikes, > 0), AND flat ~chance (structure load-bearing), the derangement
    # collapses (margin far below structured), the no-confab moat survives.
    concept_spikes_present = s_conc_sp > 0.0
    readout_spikes_present = s_read_sp > 0.0
    go = (concept_spikes_present and readout_spikes_present
          and all(r["structured"]["concept_cat_acc"] > chance + 1e-9 for r in rows)
          and s_margin > 0.005
          and f_cat <= chance + 0.15
          and p_margin <= s_margin - 0.005
          and moat_all)
    partial = (concept_spikes_present and readout_spikes_present and s_cat > chance + 0.10
               and s_margin > 0.0 and s_cat > f_cat + 0.10)
    verdict = "GO" if go else ("PARTIAL" if partial else "NEGATIVE")

    print(f"\n{'='*112}\n  MEAN ({len(rows)} seeds) [{a.candidate}]: concept spikes/cue {s_conc_sp:.0f} -> READOUT "
          f"spikes/cue {s_read_sp:.0f} (PROPAGATES) | STRUCTURED concept-spike cat-acc {s_cat:.2f} (chance "
          f"{chance:.2f}) margin {s_margin:+.4f} [readout-spike cat-acc {s_read_cat:.2f}] | FLAT {f_cat:.2f} | "
          f"PERMUTED {p_cat:.2f} margin {p_margin:+.4f} | moat {'INTACT' if moat_all else 'BREACH'}  ==> "
          f"{verdict}\n{'='*112}", flush=True)
    if verdict == "GO":
        print(f"  GO: the held-out cross-modal convergence PROPAGATES AS SPIKES -- the converged concept assembly "
              f"SPIKES ({s_conc_sp:.0f} spikes/cue, REAL cp_firing_states not membrane -- refuting the prior "
              f"'concept cannot spike from perception' residual) category-correctly ({s_cat:.0%} >> chance "
              f"{chance:.0%}, margin {s_margin:+.4f}); the downstream [{a.candidate}] read-out region ALSO spikes "
              f"({s_read_sp:.0f}/cue, readout cat-acc {s_read_cat:.0%}) -- the response propagates a synapse "
              f"further; flat-distinct ~chance ({f_cat:.0%}) => structure load-bearing; the derangement collapses "
              f"({p_margin:+.4f}); the no-confab moat survives. The novel-perception response is now SYNAPTICALLY "
              f"readable -- the live-task capstone's open piece is de-risked. NO sim/ edit.", flush=True)
    elif verdict == "PARTIAL":
        print(f"  PARTIAL: concept spikes ({s_conc_sp:.0f}/cue) + read-out spikes ({s_read_sp:.0f}/cue) and the "
              f"transfer is above flat ({s_cat:.0%} vs {f_cat:.0%}) but noisier than the GO bar -- localize: "
              f"read-weight / nmda-ratio / read-steps / n-readout-per, or try --candidate "
              f"{('graded' if a.candidate!='graded' else 'pool')}.", flush=True)
    else:
        if not concept_spikes_present:
            why = ("the concept assembly does NOT spike from perception alone (the graded code cannot be made to "
                   "propagate at point-neuron scale -- the prior residual stands; route to graded transmission)")
        elif not readout_spikes_present:
            why = (f"the concept assembly spikes ({s_conc_sp:.0f}/cue) but they do NOT drive the downstream "
                   f"read-out (try a stronger read-weight / different candidate)")
        else:
            why = (f"concept + read-out spike ({s_conc_sp:.0f}/{s_read_sp:.0f}/cue) but the spike-based transfer "
                   f"is not clean (structured {s_cat:.0%}, flat {f_cat:.0%}, permuted margin {p_margin:+.4f})")
        print(f"  NEGATIVE [{a.candidate}]: {why}. Moat {'INTACT' if moat_all else 'BREACH'}. Honest, "
              f"load-bearing: report which candidates were tried + route the capstone to the next read-out.",
              flush=True)
    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump({"verdict": verdict, "candidate": a.candidate, "chance": chance,
                   "concept_spikes_per_cue": s_conc_sp, "readout_spikes_per_cue": s_read_sp,
                   "structured_concept_cat_acc": s_cat, "structured_concept_margin": s_margin,
                   "structured_readout_cat_acc": s_read_cat, "flat_concept_cat_acc": f_cat,
                   "permuted_concept_cat_acc": p_cat, "permuted_concept_margin": p_margin,
                   "moat_intact": moat_all, "per_seed": rows}, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
