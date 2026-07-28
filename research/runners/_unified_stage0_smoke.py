"""Unified embodied agent — STAGE 0: the cheapest-first CPU/numpy integration smoke (§3 Stage 0 + §5 GATE of
`research/findings/2026-06-16-unified-embodied-agent-scoping.md`).

THE QUESTION (Stage 0): do the NEW integration pieces co-exist without regression on ONE small bridge —
  (H5) the generalization read (structured perception -> NMDA concept assembly SPIKES for the right category),
  (H6) the option-(b) HYBRID recall (read which concept-category SPIKED -> key the validated RFPhasorComposer
       recall of THAT category's stored fact),
  the no-confab MOAT (a no-category cue must ABSTAIN),
  AND a co-resident PARSER whose comprehension read is BYTE-STABLE beside the new regions
(the 5a co-residence discipline, in miniature)?

This de-risks the merged-bridge GPU wiring (Stage 1) on CPU before any heavy build. An honest NEGATIVE/PARTIAL
that LOCALIZES is acceptable; a MOAT BREACH is a HARD STOP (never weaken the moat to manufacture a GO).

CO-RESIDENCE (how the two stacks share ONE SimulationBridge):
  * brain-region framework ON; FIVE disjoint regions on one bridge / one step loop:
      perception, concept(NMDA), fact(NMDA)  -- the generalization stack (H5 read + the fact-tag recall scaffold)
      parse_conj(6), parse_role(3*R)         -- the co-resident PARSER (reuse `parser_regions_pathways`)
  * the convergence pathway (perception->concept) is RATE-Hebbian (CYCLE-95: STDP is the WRONG rule for
    symmetric co-occurrence) and is TRAINED in a setup pass; the parser is trained the same pass (its own
    teacher-driven Hebbian co-firing). Both share the global Hebbian rule -- to keep them from drifting each
    other we (i) read the parser BEFORE vs AFTER the generalization training/regions and ASSERT byte-stability
    (the smoke's load-bearing co-residence check), and (ii) raise hebbian_max_weight above any frozen weight
    (the established discipline). The parser's conj->role edges are on DISJOINT slices from the convergence's
    perception->concept edges, so the per-synapse Hebbian update touches each independently.

REUSE-BY-IMPORT ONLY (no sim/ edit):
  * the generalization machinery (the build_verbalize_bridge wiring pattern + train_convergence + the per-cue
    spike read) is taken VERBATIM in spirit from `_genfrontier_capstone_verbalize_derisk` /
    `_genfrontier_graded_propagation_derisk` -- re-expressed here with the parser regions appended and at a
    SMALL CPU scale (this runner owns the small dims so it stays a few-minutes numpy smoke).
  * the parser ports (`parser_regions_pathways`, `train_parser_on_slices`, `role_of_on_slices`,
    `parse_on_slices`, `_parser_index_arrays`, `_GT`, `PARSER_R`) come from `nav_conv_merged_bridge`.
  * the H6 hybrid recall + the no-confab moat are the validated `RFPhasorComposer` (`store` / `query_patient`).

Run:  SIM_BACKEND=numpy python -u -m research.runners._unified_stage0_smoke --seed 42
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

# --- reuse-by-import: the parser ports (framework-region parser; non-zero-offset slice arithmetic = the
#     co-residence exercise the merged bridge needs) ---
from research.runners.nav_conv_merged_bridge import (  # noqa: E402
    parser_regions_pathways, train_parser_on_slices, role_of_on_slices, parse_on_slices,
    _parser_index_arrays, PARSER_R, PARSER_GATE,
)
# --- reuse-by-import: the validated phasor recall + the no-confab abstention (the H6 hybrid) ---
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# ---- SMALL config (owned here so the smoke runs in a few minutes on numpy/CPU) ----
# 4 categories x 2 exemplars = 8 concepts; tiny perception region + small population blocks; few epochs.
N_CAT = 4
N_PER_CAT = 2
F = N_CAT * N_PER_CAT          # 8 concepts
CAT_IDS = np.repeat(np.arange(N_CAT), N_PER_CAT)

# the per-category fact: "<category> chase cat" (the answer the agent verbalizes for a matched-category cue).
ACTION_WORD = "chase"
PATIENT_WORD = "cat"
def _category_word(c):
    return f"category{c}"


# ===========================================================================
# Synthetic structured-perception ensembles (same-category OVERLAP) -- the controlled given (Option-B output),
# so we need NO heavy Gabor/V1 front end (per §5: "if Gabor/V1 is too heavy for numpy in minutes, use the
# synthetic structured-perception variant"). Same shape as the de-risks' structured_perception_sets, sized small.
# ===========================================================================
def structured_perception_sets(n_perc, n_active_cat, n_active_uniq, seed):
    """Each concept = a SHARED per-CATEGORY core (same for all concepts in a category -> same-category OVERLAP)
    + a per-concept UNIQUE tail (disjoint). Scattered across the whole region (random permutation) so neither
    cores nor tails sit in a low-index block (which would create a spurious monotonic index bias in the read)."""
    rng = np.random.default_rng(seed)
    n_need = N_CAT * n_active_cat + F * n_active_uniq
    assert n_need <= n_perc, f"perception region too small: need {n_need}, have {n_perc}"
    perm = rng.permutation(n_perc)[:n_need]
    cat_core = [perm[c * n_active_cat:(c + 1) * n_active_cat] for c in range(N_CAT)]
    base = N_CAT * n_active_cat
    sets = []
    for i in range(F):
        uniq = perm[base + i * n_active_uniq: base + (i + 1) * n_active_uniq]
        sets.append(np.concatenate([cat_core[CAT_IDS[i]], uniq]))
    return sets


def novel_no_category_perc_set(n_perc, n_active, seed):
    """A visually-novel NO-category perception cue: random scattered neurons with no learned category core. The
    moat must ABSTAIN on this (no concept-category spikes confidently)."""
    rng = np.random.default_rng(seed)
    return rng.choice(n_perc, size=n_active, replace=False)


# ===========================================================================
# The ONE co-resident bridge: perception -> concept(NMDA) -> fact(NMDA)  +  parse_conj / parse_role (the parser).
# Reuses the verbalize-bridge wiring pattern (inject_explicit_wiring) AND appends the parser framework regions.
# ===========================================================================
def build_coresident_bridge(n_perc, a, seed):
    """Build the small merged-style bridge holding BOTH stacks. Returns
    (bridge, perc_region, conc_region, conc_blocks, fact_region, fact_blocks, conj_arr, role_arr).

    Generalization wiring (identity category->fact map for the real arm):
      perception -> concept : ALL-TO-ALL, plastic, near-floor init (the rate-Hebbian convergence the setup LEARNS).
      concept    -> fact    : CONVERGENT block (every concept block of category c -> fact-tag block c), FIXED.
    Parser wiring: the framework `parse_conj -> parse_role` all-to-all plastic pathway (tagged PARSER_GATE),
    appended to the region/pathway lists so ONE init-time injection wires everything; then the explicit
    generalization wiring is injected (the framework parser pathway survives -- it is on disjoint slices and the
    explicit wiring dict only names the two generalization populations)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    n_cp = a.n_concept_per
    n_fp = a.n_fact_per

    parser_regions, parser_pathways = parser_regions_pathways(PARSER_R)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="perception", n_neurons=n_perc, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="concept", n_neurons=F * n_cp, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="fact", n_neurons=N_CAT * n_fp, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
    ] + parser_regions
    # declare all pathways so the framework takes the clean wiring branch at init.
    cfg.region_pathways = [
        RegionPathway(from_region="perception", to_region="concept", density=1.0,
                      weight_mean=0.05, weight_jitter=0.0, plastic=True),
        RegionPathway(from_region="concept", to_region="fact", density=1.0,
                      weight_mean=a.fact_weight, weight_jitter=0.0, plastic=False),
    ] + parser_pathways
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebbian_rate
    cfg.hebbian_max_weight = a.hebbian_max          # raised above any frozen weight (the 5a clip discipline)
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

    # (1) perception -> concept: all-to-all plastic near-floor.
    pc_pre = np.repeat(perc_region, conc_region.shape[0])
    pc_post = np.tile(conc_region, perc_region.shape[0])
    pc_w = np.full(pc_pre.shape[0], 0.05, np.float32)
    # (2) concept -> fact: convergent block (identity cat->fact), fixed.
    fc_pre_l, fc_post_l = [], []
    for i in range(F):
        c = int(CAT_IDS[i])
        pre_b = conc_blocks[i]
        post_b = fact_blocks[c]
        fc_pre_l.append(np.repeat(pre_b, post_b.shape[0]))
        fc_post_l.append(np.tile(post_b, pre_b.shape[0]))
    fc_pre = np.concatenate(fc_pre_l); fc_post = np.concatenate(fc_post_l)
    fc_w = np.full(fc_pre.shape[0], a.fact_weight, np.float32)

    # (3) the PARSER edges MUST be included in THIS explicit injection: inject_explicit_wiring REPLACES
    #     cp_connections entirely (sim/bridge.py:2289 "rebuilds self.cp_connections from the explicit edges"),
    #     so the framework-injected parse_conj->parse_role pathway would be WIPED if we omit it. We rebuild it
    #     here exactly as the framework declared it (all-to-all, init 0.5, plastic, gated PARSER_GATE) -> the
    #     parser + generalization populations are wired in ONE replacing injection, on disjoint slices.
    conj_arr0, role_arr0 = _parser_index_arrays(bridge, PARSER_R)   # framework slices (any offset)
    conj_h = [int(x) for x in to_host(conj_arr0)]
    role_h = {r: [int(x) for x in to_host(v)] for r, v in role_arr0.items()}
    par_pre, par_post, par_w = [], [], []
    for k in conj_h:
        for r in ["agent", "action", "patient"]:
            for j in role_h[r]:
                par_pre.append(k); par_post.append(j); par_w.append(0.5)

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
        "parse": {
            "pre_indices": par_pre,
            "post_indices": par_post,
            "initial_weights": np.array(par_w, dtype=np.float32),
            "plastic": True, "plasticity_gate": PARSER_GATE, "conn_type": "E_TO_E",
        },
    }
    bridge.inject_explicit_wiring(wiring)

    conj_arr, role_arr = _parser_index_arrays(bridge, PARSER_R)
    return bridge, perc_region, conc_region, conc_blocks, fact_region, fact_blocks, conj_arr, role_arr


# ===========================================================================
# Training the perception->concept convergence (rate-Hebbian co-activation). Tiny inline port of train_convergence.
# ===========================================================================
def _set_drive(bridge, xp, perc_region, conc_region, perc_idx_local, perc_scale,
               conc_block_local=None, conc_scale=0.0):
    n_perc = perc_region.shape[0]
    full_perc = np.zeros(n_perc, np.float32)
    full_perc[perc_idx_local] = perc_scale
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[perc_region] = xp.asarray(full_perc) if xp is not None else full_perc
    if conc_block_local is not None and conc_scale > 0.0:
        n_conc = conc_region.shape[0]
        full_conc = np.zeros(n_conc, np.float32)
        full_conc[conc_block_local] = conc_scale
        bridge.cp_external_input_current[conc_region] = xp.asarray(full_conc) if xp is not None else full_conc


def train_convergence(bridge, xp, perc_region, conc_region, conc_blocks, perc_sets, train, a):
    """Co-activate (perception ensemble + its concept block) for each TRAIN concept, repeated, so rate-Hebbian
    potentiates the perception->concept synapses (the convergence)."""
    for ep in range(a.epochs):
        order = np.random.RandomState(a.seed_base * 7 + ep).permutation(train)
        for t in order:
            perc_local = np.asarray(perc_sets[t]) - perc_region[0]
            conc_local = conc_blocks[t] - conc_region[0]
            _set_drive(bridge, xp, perc_region, conc_region, perc_local, a.perc_scale,
                       conc_block_local=conc_local, conc_scale=a.conc_scale)
            for _ in range(a.scene_steps):
                bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


# ===========================================================================
# The per-cue spike read (drive ONLY the perception cue; accumulate concept + fact spikes per block).
# ===========================================================================
def read_spikes(bridge, xp, perc_region, conc_region, conc_blocks, fact_region, fact_blocks,
                perc_idx, scale, steps):
    perc_local = np.asarray(perc_idx) - perc_region[0]
    n_perc = perc_region.shape[0]
    full = np.zeros(n_perc, np.float32)
    full[perc_local] = scale
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[perc_region] = xp.asarray(full) if xp is not None else full
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


def _category_of_concept_spikes(conc_per_block):
    """H5 read: the category whose concept blocks SPIKE most (category-mean over the concept-block spike vector)."""
    catmean = [float(conc_per_block[CAT_IDS == c].mean()) for c in range(N_CAT)]
    return int(np.argmax(catmean)), catmean


# ===========================================================================
# H6 HYBRID recall: read which concept-category SPIKED -> key the validated RFPhasorComposer recall by that
# category's fact. A no-category cue must ABSTAIN (the familiarity gate on the concept spikes returns None ->
# no recall keyed).
# ===========================================================================
def build_composer(seed):
    vocab = [_category_word(c) for c in range(N_CAT)] + [ACTION_WORD, PATIENT_WORD]
    comp = RFPhasorComposer(seed=seed, D=48, vocab=vocab)
    for c in range(N_CAT):
        comp.store(_category_word(c), ACTION_WORD, PATIENT_WORD)   # the per-category fact
    return comp


def h6_recall(comp, keyed_cat):
    """The H6 hybrid: key the validated composer recall by the spiking-concept category. Returns the recalled
    patient ('cat') for the matched category's stored fact, via the no-confab-moat-preserving query_patient."""
    return comp.query_patient(_category_word(keyed_cat), ACTION_WORD)


# ===========================================================================
# One seed, end-to-end.
# ===========================================================================
def run_seed(seed, a):
    a.seed_base = seed
    chance = 1.0 / N_CAT
    rng_split = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng_split.choice(np.where(CAT_IDS == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: train and held-out overlap"

    n_perc = a.n_perc
    perc_sets = structured_perception_sets(n_perc, a.n_active_cat, a.n_active_uniq, seed * 23 + 7)

    # --- build the ONE co-resident bridge ---
    b, pr, cr, cb, fr, fb, conj_arr, role_arr = build_coresident_bridge(n_perc, a, seed)
    xp = b._cp if hasattr(b, "_cp") else None

    # ===== CO-RESIDENCE CHECK (5a, miniature) — read the parser BEFORE generalization train, AFTER training the
    # parser itself; then AFTER the generalization train; assert the role reads are BYTE-STABLE. =====
    # 1) train the parser (its own teacher-driven Hebbian co-firing on the disjoint parse_conj/parse_role slices).
    train_parser_on_slices(b, conj_arr, role_arr, n_epochs=a.parser_epochs, train_steps=a.parser_steps)
    # freeze the parser gate so a later re-read / the generalization training cannot drift it.
    b.set_plasticity_gate(PARSER_GATE, 0.0)
    # 2) parse read BEFORE the generalization training (the baseline comprehension read).
    sentence = ["dog", "chase", "cat"]
    parse_before = parse_on_slices(b, conj_arr, role_arr, sentence)
    roles_before = [role_of_on_slices(b, conj_arr, role_arr, p) for p in range(3)]

    # ===== train the perception->concept convergence (the generalization stack) ON THE SAME BRIDGE =====
    train_convergence(b, xp, pr, cr, cb, perc_sets, train, a)

    # 3) parse read AFTER the generalization training — the co-residence assert: byte-stable role reads?
    parse_after = parse_on_slices(b, conj_arr, role_arr, sentence)
    roles_after = [role_of_on_slices(b, conj_arr, role_arr, p) for p in range(3)]
    parser_byte_stable = bool(roles_before == roles_after and parse_before == parse_after)
    # the parser must also be CORRECT (voice-invariant SVO): position 0 agent, 1 action, 2 patient.
    parser_correct = bool(parse_after.get("agent") == "dog" and parse_after.get("action") == "chase"
                          and parse_after.get("patient") == "cat")

    # ===== H5: held-out structured-perception cue -> concept-category SPIKES =====
    comp = build_composer(seed)
    h5_hits, h6_hits, conc_s, fact_s, win_fires = [], [], [], [], []
    answers = []
    for j in held_out:
        cpb, fpb, ct, ft = read_spikes(b, xp, pr, cr, cb, fr, fb, perc_sets[j], a.perc_scale, a.read_steps)
        keyed_cat, catmean = _category_of_concept_spikes(cpb)
        h5_hits.append(int(keyed_cat == CAT_IDS[j]))
        # H6 hybrid recall keyed by the spiking concept-category.
        rec = h6_recall(comp, keyed_cat)
        h6_hits.append(int(rec == PATIENT_WORD and keyed_cat == CAT_IDS[j]))
        conc_s.append(ct); fact_s.append(ft); win_fires.append(float(np.max(catmean)))
        answers.append({"true_cat": int(CAT_IDS[j]), "keyed_cat": keyed_cat, "recall": rec,
                        "concept_cat_means": catmean})
    h5_acc = float(np.mean(h5_hits))
    h6_acc = float(np.mean(h6_hits))
    heldout_win_fire = float(np.mean(win_fires))

    # ===== MOAT: a no-category cue must ABSTAIN. The familiarity gate: a known held-out cue drives a HIGH
    # best-category concept-spike response; a no-category cue drives a LOW, diffuse response -> below the gate ->
    # NO recall keyed (abstain). We calibrate the gate from the held-out familiarity and require the no-category
    # cue to fall clearly below it; when below, the agent does NOT key the composer recall (returns None). =====
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_set = novel_no_category_perc_set(n_perc, a.n_active_cat + a.n_active_uniq, seed * 41 + 9)
    ncpb, nfpb, nct, nft = read_spikes(b, xp, pr, cr, cb, fr, fb, novel_set, a.perc_scale, a.read_steps)
    novel_cat, novel_catmean = _category_of_concept_spikes(ncpb)
    novel_win_fire = float(np.max(novel_catmean))
    # gate: the no-category cue's best-category response must be clearly below the held-out familiarity band.
    gate_thresh = heldout_win_fire * a.moat_gate_frac
    novel_familiar = bool(novel_win_fire >= gate_thresh)
    # the agent abstains when the cue is NOT familiar (below the gate) -> no recall keyed.
    novel_recall = h6_recall(comp, novel_cat) if novel_familiar else None
    moat_abstains = bool(novel_recall is None)
    fam_contrast_ok = bool(heldout_win_fire > novel_win_fire * 1.2 + 1e-9)

    del b
    return {
        "seed": seed, "held_out": held_out, "chance": chance,
        "parser": {"parse_before": parse_before, "parse_after": parse_after,
                   "roles_before": roles_before, "roles_after": roles_after,
                   "byte_stable": parser_byte_stable, "correct": parser_correct},
        "h5_concept_cat_acc": h5_acc, "h6_hybrid_recall_acc": h6_acc,
        "concept_spikes_per_cue": float(np.mean(conc_s)), "fact_spikes_per_cue": float(np.mean(fact_s)),
        "heldout_win_fire": heldout_win_fire,
        "moat": {"novel_keyed_cat": novel_cat, "novel_win_fire": novel_win_fire, "gate_thresh": gate_thresh,
                 "novel_familiar": novel_familiar, "novel_recall": novel_recall,
                 "moat_abstains": moat_abstains, "fam_contrast_ok": fam_contrast_ok},
        "answers": answers,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    # SMALL config for a few-minutes numpy/CPU run.
    p.add_argument("--n-perc", type=int, default=400, help="perception region size (small)")
    p.add_argument("--n-concept-per", type=int, default=40, help="neurons per concept block")
    p.add_argument("--n-fact-per", type=int, default=40, help="neurons per CATEGORY fact-tag block")
    p.add_argument("--n-active-cat", type=int, default=24, help="shared per-CATEGORY perception core size")
    p.add_argument("--n-active-uniq", type=int, default=8, help="per-CONCEPT unique perception tail size")
    p.add_argument("--epochs", type=int, default=12, help="convergence training epochs")
    p.add_argument("--scene-steps", type=int, default=12, help="co-drive steps per training scene")
    p.add_argument("--read-steps", type=int, default=60, help="steps to accumulate the spike read")
    p.add_argument("--perc-scale", type=float, default=300.0)
    p.add_argument("--conc-scale", type=float, default=600.0)
    p.add_argument("--fact-weight", type=float, default=30.0)
    p.add_argument("--nmda-ratio", type=float, default=2.0)
    p.add_argument("--hebbian-rate", type=float, default=0.05)
    # 400 = the merged-bridge value (nav_conv_merged_bridge / BridgeParser both use hebbian_max_weight=400). The
    # parser's conj->role readout needs its learned weights ABOVE the role-ensemble spike threshold: capping the
    # GLOBAL Hebbian max at 20 (the convergence's default) caps the parser too LOW -> the role ensemble never fires
    # under conj-alone drive -> the parser read collapses (every position reads the default first role). The
    # convergence's category-MEAN-over-spikes read is robust to the higher cap (more concept spikes, still
    # category-correct), so one global cap of 400 serves both stacks. (Measured: hebbian_max=20 -> parser correct
    # FALSE / H5 chance; hebbian_max=400 -> parser correct TRUE + byte-stable / H5 0.75 / moat intact = GO.)
    p.add_argument("--hebbian-max", type=float, default=400.0)
    p.add_argument("--parser-epochs", type=int, default=20)
    p.add_argument("--parser-steps", type=int, default=100)
    p.add_argument("--moat-gate-frac", type=float, default=0.6, help="a no-category cue's best-category "
                   "concept-spike response must be >= this fraction of the held-out familiarity to be 'familiar' "
                   "(else abstain -> no recall keyed)")
    p.add_argument("--out", default="research/findings/raw/_unified_stage0_smoke.json")
    a = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[unified STAGE-0 smoke] generalization read (H5) + H6-hybrid recall + no-confab moat + a CO-RESIDENT "
          f"parser whose comprehension read must stay BYTE-STABLE -- on ONE small CPU bridge. seed={a.seed}",
          flush=True)
    r = run_seed(a.seed, a)
    chance = r["chance"]

    par = r["parser"]
    print(f"  [seed {a.seed}] PARSER: parse before {par['parse_before']} | after {par['parse_after']} | "
          f"byte-stable {par['byte_stable']} | correct {par['correct']}", flush=True)
    print(f"  [seed {a.seed}] H5 concept-category spike acc {r['h5_concept_cat_acc']:.2f} (chance {chance:.2f}) | "
          f"concept spikes/cue {r['concept_spikes_per_cue']:.0f} | fact spikes/cue {r['fact_spikes_per_cue']:.0f}",
          flush=True)
    print(f"  [seed {a.seed}] H6 hybrid recall acc {r['h6_hybrid_recall_acc']:.2f}", flush=True)
    m = r["moat"]
    print(f"  [seed {a.seed}] MOAT: held-out win-fire {r['heldout_win_fire']:.2f} vs novel {m['novel_win_fire']:.2f} "
          f"(gate {m['gate_thresh']:.2f}) -> {'ABSTAIN' if m['moat_abstains'] else 'CONFAB'} "
          f"(novel_recall={m['novel_recall']})", flush=True)

    # ---- GATE (single-seed CPU, per §5) ----
    h5 = r["h5_concept_cat_acc"]
    h6 = r["h6_hybrid_recall_acc"]
    moat_ok = bool(m["moat_abstains"])                              # HARD: a breach FAILS outright
    parser_ok = bool(par["byte_stable"] and par["correct"])
    h5_over_chance = bool(h5 > chance + 1e-9)
    h6_ok = bool(h6 >= 0.50)
    spikes_present = bool(r["concept_spikes_per_cue"] > 0.0)

    if not moat_ok:
        verdict = "MOAT_BREACH"           # HARD STOP
    elif h5_over_chance and h6_ok and parser_ok and spikes_present:
        verdict = "GO"
    elif spikes_present and (h5 > chance or h6 > 0.0) and parser_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    print(f"\n{'='*112}\n  STAGE-0 (seed {a.seed}): H5 concept-spike acc {h5:.2f} (chance {chance:.2f}) | "
          f"H6 hybrid recall {h6:.2f} | moat {'INTACT (abstain)' if moat_ok else 'BREACH (CONFAB)'} | parser "
          f"{'BYTE-STABLE+CORRECT' if parser_ok else 'REGRESSED'}  ==> {verdict}\n{'='*112}", flush=True)

    if verdict == "GO":
        print(f"  GO -- the NEW integration pieces CO-EXIST without regression on ONE small CPU bridge: a held-out "
              f"structured-perception cue drives the concept-category to SPIKE ({r['concept_spikes_per_cue']:.0f} "
              f"concept spikes/cue) and lands in the right category ({h5:.0%} > chance {chance:.0%}); the H6 hybrid "
              f"reads that spiking category and keys the VALIDATED RFPhasorComposer recall of the category's fact "
              f"({h6:.0%}); a no-category cue ABSTAINS (the no-confab moat survives); AND the co-resident parser's "
              f"comprehension read is BYTE-STABLE + correct beside the new regions (the 5a co-residence discipline). "
              f"==> promote to Stage 1 (the merged-bridge region addition + the single-seed GPU gate). NO sim/ edit.",
              flush=True)
    elif verdict == "MOAT_BREACH":
        print(f"  MOAT_BREACH -- HARD STOP: a no-category cue CONFABULATED a fact (novel_recall="
              f"{m['novel_recall']}). The gate is too loose at this small CPU scale. Do NOT proceed; do NOT loosen "
              f"the gate to manufacture a GO. Localize: raise --moat-gate-frac / tighten the familiarity contrast "
              f"(held-out win-fire {r['heldout_win_fire']:.2f} vs novel {m['novel_win_fire']:.2f}).", flush=True)
    elif verdict == "PARTIAL":
        why = []
        if not h5_over_chance:
            why.append(f"H5 concept-spike acc {h5:.0%} not above chance (raise epochs / n-active-cat / perc-scale)")
        if not h6_ok:
            why.append(f"H6 recall {h6:.0%} below 0.50 (inherits the H5 noise at small CPU scale)")
        print(f"  PARTIAL: the route closes + the moat holds + the parser is byte-stable, but below the GO bar -- "
              f"{'; '.join(why) if why else 'noisy at small CPU scale'}. These are bounded knobs (epochs / "
              f"n-active-cat / read-steps / n-concept-per), not walls. The GPU Stage-1 build uses the de-risks' "
              f"validated larger dims (the convergence GO config).", flush=True)
    else:
        if not parser_ok:
            why = ("the CO-RESIDENT PARSER regressed (byte-stable=%s, correct=%s) -- the generalization regions/"
                   "training corrupted the parser read. Localize: the parser gate freeze, the read settle window, "
                   "or the Hebbian max-weight clip vs the frozen parser weights"
                   % (par["byte_stable"], par["correct"]))
        elif not spikes_present:
            why = "the concept assembly does not SPIKE from the perception drive (raise perc-scale / nmda-ratio)"
        else:
            why = (f"the spiking-concept read is too noisy (H5 {h5:.0%}, H6 {h6:.0%}) at this small CPU scale")
        print(f"  NEGATIVE: {why}. Moat {'INTACT' if moat_ok else 'BREACH'}. Honest negative + the localized next "
              f"step (this is a CPU smoke; the GPU Stage-1 uses the validated convergence dims).", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, a.out)), exist_ok=True)
    with open(os.path.join(_REPO, a.out), "w") as fh:
        json.dump({"verdict": verdict, "seed": a.seed, "chance": chance,
                   "h5_concept_cat_acc": h5, "h6_hybrid_recall_acc": h6,
                   "concept_spikes_per_cue": r["concept_spikes_per_cue"],
                   "fact_spikes_per_cue": r["fact_spikes_per_cue"],
                   "parser_byte_stable": par["byte_stable"], "parser_correct": par["correct"],
                   "moat_abstains": moat_ok, "result": r}, fh, indent=2, default=str)
    print(f"  [saved] {a.out}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    # exit codes: 0 GO, 2 PARTIAL, 3 MOAT_BREACH (hard stop), 1 NEGATIVE
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else (3 if verdict == "MOAT_BREACH" else 1)))


if __name__ == "__main__":
    main()
