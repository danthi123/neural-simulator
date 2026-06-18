"""Navigation + Conversational single-instance merge (roadmap step 2) — builder + ports.

Per `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md`. The brain-region framework IS a
wrapper around `inject_explicit_wiring` (`sim/bridge.py:1514-1526`), so the merge appends the parser +
dlPFC as framework `BrainRegion`/`RegionPathway` to the navigation lists; everything is wired by one
init-time injection of `region_manager.build_wiring_plan(...)`.

This file starts with the PARSER PORT (the highest implementation risk 4.1): `BridgeParser` cannot be a
drop-in `shared_bridge=` because its merge path re-injects and would clobber navigation, and it assumes a
contiguous parser block. So we re-express the parser as framework regions and PORT its drive/train/read to
raw framework slice indices. `--microcheck` builds a parser-only framework bridge (behind a navigation-stub
region that forces a non-zero offset, exercising the slice arithmetic) and validates that the parser learns
the voice-invariant role map on framework slices under the merged config — retiring risk 4.1 before the full
STEP 2a build.

Reuse-by-import; no `sim/` edit.
"""
from __future__ import annotations

import argparse

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host
# Module-level (the MergedRFComposer base class needs it at class-definition time). Lightweight: rf_phasor_composer
# only imports numpy + the already-imported sim.* substrate, so this does not slow the parser-only --microcheck path.
from research.runners.rf_phasor_composer import RFPhasorComposer

# parser ground truth (from brain_conversational_agent): conjunction index k = position*2 + voice
# (voice 0=active, 1=passive); each k teacher-binds to one role.
PARSER_R = 40
ROLES = ["agent", "action", "patient"]
_GT = {0: "agent", 1: "patient", 2: "action", 3: "action", 4: "patient", 5: "agent"}

# the parser's gate name on the merged bridge (frozen to 0.0 after the train pass)
PARSER_GATE = "parser_fixed"

# dlPFC dialogue-planning loop constants. These mirror unified_brain_bridge.py:117-120 exactly so the
# merged dlPFC slices reproduce the validated UnifiedBrainBridge dialogue-planning behaviour:
#   DLPFC_PATTERN_SIZE      per-concept assembly size (cells per word in each dlPFC region).
#   DLPFC_ATTRACTOR_WEIGHT  the cortex_ctx<->dlpfc_wm self-attractor weight. 30.0 is the genuinely
#                           NMDA-dependent Wang-2002 regime (50 would be trivial AMPA ping-pong).
#   DLPFC_FIXED_GATE        one plasticity gate over ALL dlPFC loop+graph edges, held at 0.0 so neither
#                           the parser train pass nor a later navigation episode can drift them.
DLPFC_PATTERN_SIZE = 50
DLPFC_ATTRACTOR_WEIGHT = 30.0
#   DLPFC_EDGE_SCALE        the graph-edge scale `elaborate` installs over the pre-allocated c2d slots. 60.0 is the
#                           validated SpikingSpreadingController default (unified_brain_bridge.py:119) — the spread must
#                           be strong enough that every designed associate latches.
DLPFC_EDGE_SCALE = 60.0
DLPFC_FIXED_GATE = "dlpfc_fixed"

# ── generalization stack constants (STAGE 1, additive default-off) ────────────────────────────────────────
# The generalization stack appended LAST (after rf + cortex_it) so the nav/parser/dlPFC/rf/cortex_it index
# bases stay BYTE-UNCHANGED. Reuse-by-import the validated de-risk machinery (Gabor/V1 vision → top-K → the
# rate-Hebbian convergence → the NMDA concept assembly → the convergent concept→fact tags). N_CAT/N_PER_CAT/F
# are the de-risk's 4-category × 4-exemplar = 16-concept layout; the perception region is the V1-complex
# feature dimension (one perception neuron per V1-complex cell). Distinct region names (`gen_*`) so they never
# collide with the navigation `cortex_it` perception region or anything else on the merged bridge.
GEN_PERCEPTION = "gen_perception"
GEN_CONCEPT = "gen_concept"
GEN_FACT = "gen_fact"
# the generalization convergence gate (perception→concept edges); frozen after the convergence train pass, the
# same trained-then-frozen discipline as the parser. The convergent concept→fact edges are plastic=False.
GEN_CONV_GATE = "gen_convergence_fixed"


# ── parser as framework regions/pathways ─────────────────────────────────────────────────────────────────
def parser_regions_pathways(R: int = PARSER_R):
    """The parser's two framework regions (separate slices) + the all-to-all plastic conj->role pathway.

    parse_conj : 6 conjunction units. parse_role : 3*R role-ensemble neurons (agent|action|patient blocks).
    The conj->role pathway is all-to-all (density 1.0), init weight 0.5 exactly (jitter 0), plastic, tagged
    `parser_fixed` so it can be frozen after the Hebbian train pass.
    """
    regions = [
        BrainRegion(name="parse_conj", n_neurons=6, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="parse_role", n_neurons=3 * R, exc_fraction=1.0, internal_density=0.0),
    ]
    pathways = [
        RegionPathway(from_region="parse_conj", to_region="parse_role",
                      density=1.0, weight_mean=0.5, weight_jitter=0.0,
                      plastic=True, plasticity_gate=PARSER_GATE),
    ]
    return regions, pathways


def _parser_index_arrays(bridge, R: int = PARSER_R):
    """Resolve the parser's conjunction + per-role index arrays from the FRAMEWORK slices (any offset)."""
    rm = bridge.region_manager
    conj = list(rm.indices("parse_conj"))            # 6 contiguous global indices
    role_base_list = list(rm.indices("parse_role"))  # 3*R contiguous global indices
    role_idx = {r: role_base_list[i * R:(i + 1) * R] for i, r in enumerate(ROLES)}
    xp, _ = get_backend()
    conj_arr = xp.asarray(conj, dtype=xp.int64)
    role_arr = {r: xp.asarray(v, dtype=xp.int64) for r, v in role_idx.items()}
    return conj_arr, role_arr


def _step_reset(bridge, reset: int = 20):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(reset):
        bridge._run_one_simulation_step()


def train_parser_on_slices(bridge, conj_arr, role_arr, n_epochs: int = 30, train_steps: int = 120,
                           drive: float = 2500.0):
    """Port of BridgeParser._train onto framework slices: for each conjunction k, co-drive conj[k] + the
    teacher role ensemble _GT[k]; Hebbian co-firing strengthens conj[k]->correct-role. Caller must have
    Hebbian ON + STDP/reward OFF for this pass."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    for _ in range(n_epochs):
        for k in range(6):
            _step_reset(bridge)
            cur = xp.zeros(n, dtype=xp.float32)
            cur[conj_arr[k]] = drive
            cur[role_arr[_GT[k]]] = drive
            bridge.cp_external_input_current[:] = cur
            for _ in range(train_steps):
                bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


# The parser READ-path quiescence window (design risk 4.3 / 5a). On a FRESH merged bridge the parser reads
# byte-stably with reset=20; but when a read follows a prior heavy drive of the SAME slices (e.g. the agent's
# `hear` ran the parser just before), the merged bridge's larger neuron population has not fully relaxed and the
# FIRST read drifts (a single position mis-reads its role -> a degenerate parse). A longer pre-read settle restores
# it. Measured on the merged bridge (seed 42, after a prior `hear`): reset=20 -> 3/5 stable, reset=60/120/200 ->
# 5/5. 60 is the stable floor with margin (the 5a "longer settle restores byte-identity" mitigation), used for the
# read ports only (training drives continuously -> its inter-epoch reset stays 20).
PARSER_READ_SETTLE = 60


def _rates_on_slices(bridge, conj_arr, role_arr, position, voice, test_steps, drive, reset):
    """Drive the (position, voice) conjunction ALONE and return the per-role accumulated firing rates. `reset` is the
    pre-read quiescence window (design risk 4.3): a longer settle self-quiesces the merged bridge before the read."""
    xp, _ = get_backend()
    n = bridge.core_config.num_neurons
    k = position * 2 + (0 if voice in (0, "active") else 1)
    _step_reset(bridge, reset)
    cur = xp.zeros(n, dtype=xp.float32)
    cur[conj_arr[k]] = drive
    bridge.cp_external_input_current[:] = cur
    rates = {r: 0.0 for r in ROLES}
    for _ in range(test_steps):
        bridge._run_one_simulation_step()
        for r in ROLES:
            rates[r] += float(to_host(bridge.cp_firing_states[role_arr[r]].astype(xp.float64).mean()))
    bridge.cp_external_input_current[:] = 0.0
    return rates


def role_of_on_slices(bridge, conj_arr, role_arr, position: int, voice="active",
                      test_steps: int = 80, drive: float = 2500.0, reset: int = PARSER_READ_SETTLE):
    """Port of BridgeParser.role_of: drive the (position, voice) conjunction ALONE; the role ensemble that fires most
    is the learned role. (Single-position argmax read; `parse_on_slices` uses the distinct-assignment read below.)"""
    rates = _rates_on_slices(bridge, conj_arr, role_arr, position, voice, test_steps, drive, reset)
    return max(rates, key=rates.get)


def parse_on_slices(bridge, conj_arr, role_arr, words, voice="active", test_steps: int = 80, drive: float = 2500.0,
                    reset: int = PARSER_READ_SETTLE):
    """Robust 3-word SVO parse: read EACH position's full per-role rate vector, then assign positions to DISTINCT
    roles (greedy by rate). This GUARANTEES all three roles (agent/action/patient) appear -- eliminating the dt=1.0
    WTA read-tie where two positions would otherwise decode to the SAME role (dropping a key, crashing the unsafe
    roles[...] access; the Stage-3 seeds 43/102 failure mode). For a clean read this is identical to the
    per-position argmax; under a tie it picks the distinct assignment maximizing total rate."""
    assert len(words) == 3, "this minimal parser handles 3-word SVO sentences"
    vecs = [_rates_on_slices(bridge, conj_arr, role_arr, pos, voice, test_steps, drive, reset) for pos in range(3)]
    # greedy distinct assignment: take the highest (position, role) rate, fix it, remove that position + role, repeat.
    triples = sorted(((vecs[p][r], p, r) for p in range(3) for r in ROLES), key=lambda t: t[0], reverse=True)
    pos_role, used_pos, used_role = {}, set(), set()
    for _rate, p, r in triples:
        if p in used_pos or r in used_role:
            continue
        pos_role[p] = r
        used_pos.add(p)
        used_role.add(r)
    return {pos_role[pos]: words[pos] for pos in range(3)}


# ── dlPFC loop population on the framework slices ────────────────────────────────────────────────────────
def _build_dlpfc_loop_population(ctx, words):
    """Build the hand-wired `dlpfc_loop` synapse population on the merged-bridge dlPFC slices.

    Ported VERBATIM (logic) from UnifiedBrainBridge._wire_dlpfc (unified_brain_bridge.py:641-661). The
    framework's RegionPathway can only build a uniform region->region projection; the dlPFC needs
    per-word-pair assembly-to-assembly BLOCK structure, so the loop edges are hand-built and injected as one
    extra population alongside the framework plan.

      c2d : cortex_ctx_A -> dlpfc_wm_B for ALL ordered (A, B) word pairs. The DIAGONAL (A==B) is the forward
            self-attractor (weight DLPFC_ATTRACTOR_WEIGHT); the off-diagonal (A!=B) pairs are the
            association-graph edge SLOTS, pre-allocated at weight 0 and overwritten in place at elaborate time.
      d2c : dlpfc_wm_C -> cortex_ctx_C for all C (the backward self-attractor, weight DLPFC_ATTRACTOR_WEIGHT).

    `ctx` is a `_SharedDlpfcContext` over the framework cortex_ctx/dlpfc_wm slices; its `_cpat_host`/`_dpat_host`
    hold the per-word assembly indices already shifted to the framework slice bases. All edges are tagged
    `plasticity_gate=DLPFC_FIXED_GATE` and `plastic=False`. Returns the population spec dict (one entry).
    """
    ps = DLPFC_PATTERN_SIZE
    cpat = {w: ctx._cpat_host[w] for w in words}   # cortex_ctx assembly indices per word (framework slice)
    dpat = {w: ctx._dpat_host[w] for w in words}   # dlpfc_wm  assembly indices per word (framework slice)

    c2d_pre = []; c2d_post = []; c2d_w = []
    for a in words:                        # cortex_A -> dlpfc_B for ALL (A, B): diagonal = self-attractor
        preA = np.repeat(cpat[a], ps)      # (ps*ps,) each cortex_A neuron -> every dlpfc_B neuron
        for b in words:
            c2d_pre.append(preA)
            c2d_post.append(np.tile(dpat[b], ps))
            w0 = DLPFC_ATTRACTOR_WEIGHT if a == b else 0.0     # self-attractor on the diagonal, slot elsewhere
            c2d_w.append(np.full(ps * ps, w0, dtype=np.float32))
    d2c_pre = []; d2c_post = []; d2c_w = []
    for c in words:                        # dlpfc_C -> cortex_C (backward self-attractor only)
        d2c_pre.append(np.repeat(dpat[c], ps))
        d2c_post.append(np.tile(cpat[c], ps))
        d2c_w.append(np.full(ps * ps, DLPFC_ATTRACTOR_WEIGHT, dtype=np.float32))

    pre = np.concatenate(c2d_pre + d2c_pre).astype(np.int64)
    post = np.concatenate(c2d_post + d2c_post).astype(np.int64)
    ww = np.concatenate(c2d_w + d2c_w).astype(np.float32)
    return {
        "pre_indices": pre, "post_indices": post, "initial_weights": ww,
        "plastic": False, "plasticity_gate": DLPFC_FIXED_GATE, "conn_type": "E_TO_E", "count": int(pre.size),
    }


# ── generalization stack: regions, exact edges, and the trained-then-frozen convergence (STAGE 1) ──────────
def _generalization_regions_pathways(gen_n_concept_per: int, gen_n_fact_per: int):
    """The generalization stack's three framework regions + two pathways (declared so the framework takes the
    clean wiring branch at init; their exact all-to-all / convergent block-diagonal edges are OVERWRITTEN in the
    union_plan before the combined injection, the same pattern dlpfc_loop uses).

      gen_perception : N_V1_COMPLEX cells (one perception neuron per V1-complex feature) — the structured-perception
                       region that receives the Gabor/V1 top-K drive. internal_density=0, NMDA off.
      gen_concept    : F × gen_n_concept_per population blocks, enable_nmda=True (the slow NMDA conductance
                       integrates the sparse perception drive to SPIKES — the documented rate-code-wall lift).
      gen_fact       : N_CAT × gen_n_fact_per fact-tag blocks, enable_nmda=True (the concept spikes drive the
                       fact tags synaptically; the H6-hybrid reads which concept-category spiked).
    Both pathways are declared so build_wiring_plan emits clean entries; those entries are then replaced.
    """
    from research.runners._genfrontier_capstone_vision_to_concept_derisk import N_V1_COMPLEX
    from research.runners._genfrontier_onsubstrate_convergence_derisk import N_CAT, F
    regions = [
        BrainRegion(name=GEN_PERCEPTION, n_neurons=int(N_V1_COMPLEX), exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name=GEN_CONCEPT, n_neurons=F * int(gen_n_concept_per), exc_fraction=1.0,
                    internal_density=0.0, enable_nmda=True),
        BrainRegion(name=GEN_FACT, n_neurons=N_CAT * int(gen_n_fact_per), exc_fraction=1.0,
                    internal_density=0.0, enable_nmda=True),
    ]
    pathways = [
        # perception → concept: all-to-all, plastic, near-floor init (the rate-Hebbian convergence the LEARN pass
        # grows). Tagged GEN_CONV_GATE so it can be isolated/frozen (the trained-then-frozen discipline).
        RegionPathway(from_region=GEN_PERCEPTION, to_region=GEN_CONCEPT, density=1.0,
                      weight_mean=0.05, weight_jitter=0.0, plastic=True, plasticity_gate=GEN_CONV_GATE),
        # concept → fact: convergent block (every concept block of category c → fact-tag block c), FIXED.
        RegionPathway(from_region=GEN_CONCEPT, to_region=GEN_FACT, density=1.0,
                      weight_mean=30.0, weight_jitter=0.0, plastic=False),
    ]
    return regions, pathways


def _build_generalization_edges(rm, gen_n_concept_per: int, gen_n_fact_per: int, fact_weight: float = 30.0):
    """The EXACT generalization edges, keyed by the build_wiring_plan entry names so they OVERWRITE the framework's
    uniform versions in the union_plan (the dlpfc_loop insertion pattern). Returns
    (edge_overrides, gen_handles) where edge_overrides maps the two plan keys to their precise edge dicts and
    gen_handles holds the resolved region index arrays + per-block reshapes (for training + the spike read)."""
    from research.runners._genfrontier_capstone_vision_to_concept_derisk import N_V1_COMPLEX
    from research.runners._genfrontier_onsubstrate_convergence_derisk import N_CAT, N_PER_CAT, F

    perc_region = np.asarray(rm.indices(GEN_PERCEPTION), dtype=np.int64)
    conc_region = np.asarray(rm.indices(GEN_CONCEPT), dtype=np.int64)
    fact_region = np.asarray(rm.indices(GEN_FACT), dtype=np.int64)
    conc_blocks = conc_region.reshape(F, int(gen_n_concept_per))
    fact_blocks = fact_region.reshape(N_CAT, int(gen_n_fact_per))
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)

    # (1) gen_perception → gen_concept: all-to-all, plastic, near-floor init (the convergence the LEARN pass grows).
    pc_pre = np.repeat(perc_region, conc_region.shape[0])
    pc_post = np.tile(conc_region, perc_region.shape[0])
    pc_w = np.full(pc_pre.shape[0], 0.05, np.float32)
    # (2) gen_concept → gen_fact: convergent block (every concept block of category c → fact-tag block c), FIXED.
    fc_pre_l, fc_post_l = [], []
    for i in range(F):
        c = int(cat_ids[i])
        pre_b = conc_blocks[i]
        post_b = fact_blocks[c]
        fc_pre_l.append(np.repeat(pre_b, post_b.shape[0]))
        fc_post_l.append(np.tile(post_b, pre_b.shape[0]))
    fc_pre = np.concatenate(fc_pre_l)
    fc_post = np.concatenate(fc_post_l)
    fc_w = np.full(fc_pre.shape[0], float(fact_weight), np.float32)

    edge_overrides = {
        f"pathway_{GEN_PERCEPTION}_to_{GEN_CONCEPT}": {
            "pre_indices": pc_pre.astype(np.int64).tolist(),
            "post_indices": pc_post.astype(np.int64).tolist(),
            "initial_weights": pc_w.tolist(),
            "plastic": True, "plasticity_gate": GEN_CONV_GATE, "conn_type": "E_TO_MIX",
        },
        f"pathway_{GEN_CONCEPT}_to_{GEN_FACT}": {
            "pre_indices": fc_pre.astype(np.int64).tolist(),
            "post_indices": fc_post.astype(np.int64).tolist(),
            "initial_weights": fc_w.tolist(),
            "plastic": False, "conn_type": "E_TO_MIX",
        },
    }
    gen_handles = {
        "perc_region": perc_region, "conc_region": conc_region, "fact_region": fact_region,
        "conc_blocks": conc_blocks, "fact_blocks": fact_blocks, "cat_ids": cat_ids,
        "n_concept_per": int(gen_n_concept_per), "n_fact_per": int(gen_n_fact_per),
        "n_v1_complex": int(N_V1_COMPLEX), "N_CAT": int(N_CAT), "N_PER_CAT": int(N_PER_CAT), "F": int(F),
        "perc_base": int(perc_region[0]), "conc_base": int(conc_region[0]), "fact_base": int(fact_region[0]),
        "perc_last": int(perc_region[-1]), "fact_last": int(fact_region[-1]),
    }
    return edge_overrides, gen_handles


def _gen_vision_sets_and_split(seed: int):
    """Render the de-risk's object SHAPES, encode through the REAL Gabor/V1 front end, convert to top-K structured
    perception sets, and produce the leakage-free held-out/train split — VERBATIM from the vision→concept de-risk
    (reuse-by-import). Returns (vis_sets, held_out, train, cat_ids, structure_margin, structure_preserved)."""
    from research.runners._genfrontier_optionB_visual_similarity_derisk import (
        build_shape_set, build_gabor_response_matrix, encode_v1, pool_v1_to_complex, within_between_margin,
    )
    from research.runners._genfrontier_capstone_vision_to_concept_derisk import (
        vision_to_perception_sets, active_set_overlap_margin, N_V1_COMPLEX,
    )
    from research.runners._genfrontier_onsubstrate_convergence_derisk import N_CAT, N_PER_CAT, F
    from sim.visual_cortex import RETINA_SIZE

    GEN_TOP_K = 60
    GEN_MIN_SET_MARGIN = 0.05
    cat_ids = np.repeat(np.arange(N_CAT), N_PER_CAT)
    rng = np.random.default_rng(seed)
    rng_split = np.random.default_rng(seed * 31 + 5)
    held_out = [int(rng_split.choice(np.where(cat_ids == c)[0])) for c in range(N_CAT)]
    train = [i for i in range(F) if i not in held_out]
    assert not (set(train) & set(held_out)), "leakage: gen train and held-out overlap"

    images, labels, _meta = build_shape_set(N_CAT, N_PER_CAT, rng, image_size=RETINA_SIZE)
    assert np.array_equal(labels, cat_ids), "shape labels must match the concept category layout"
    W = build_gabor_response_matrix()
    v1 = encode_v1(images, W)
    it_like = pool_v1_to_complex(v1)
    assert it_like.shape[1] == N_V1_COMPLEX
    vis_sets = vision_to_perception_sets(it_like, GEN_TOP_K)
    _, _, set_margin = active_set_overlap_margin(vis_sets, N_V1_COMPLEX, cat_ids)
    structure_preserved = bool(set_margin > GEN_MIN_SET_MARGIN)
    return vis_sets, held_out, train, cat_ids, float(set_margin), structure_preserved, W, GEN_TOP_K


def _train_merged_convergence(bridge, gen_handles, vis_sets, train, *, epochs=20, scene_steps=16,
                              perc_scale=300.0, conc_scale=600.0, hebbian_rate=0.05, hebbian_max=20.0,
                              nmda_ratio=2.0, seed=42):
    """Train the perception→concept rate-Hebbian convergence ON THE MERGED BRIDGE, ISOLATED from the navigation
    reward-STDP + the global dopamine (scope="all") + the parser, via the cp_plasticity_rate_gain INDEX MASK (the
    finalize_conv_for_nav_gate discipline): only the gen_perception→gen_concept synapses are plastic (gain 1);
    EVERYTHING else (nav, parser, dlPFC, concept→fact) is frozen (gain 0) so the Hebbian decay cannot erode it.

    Runs after the parser train pass (so the parser weights are final) and freezes the convergence afterward
    (gain 0). The NMDA-ratio / Hebbian knobs are the de-risk's validated convergence values (the convergence GO
    config); nav keeps Hebbian OFF in episodes, so these are consulted ONLY during this pass.
    """
    xp, _ = get_backend()
    cc = bridge.core_config
    rm = bridge.region_manager
    csr = bridge.cp_connections
    # the per-data-position (pre, post), guaranteed aligned with cp_connections.data / cp_plasticity_rate_gain
    # (computed on the HOST from indptr/indices — the finalize_conv_for_nav_gate pattern, backend-agnostic).
    indptr_h = to_host(csr.indptr)
    post_h = to_host(csr.indices).astype(np.int64)
    nnz = int(post_h.shape[0])
    pre_h = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre_h[int(indptr_h[r]):int(indptr_h[r + 1])] = r
    perc = gen_handles["perc_region"]
    conc = gen_handles["conc_region"]
    conv_mask = xp.asarray(np.isin(pre_h, perc) & np.isin(post_h, conc))

    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    gain = bridge.cp_plasticity_rate_gain

    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation, cc.enable_ou_process,
             cc.hebbian_learning_rate, cc.hebbian_max_weight, cc.hebbian_min_weight, cc.hebbian_weight_decay,
             cc.nmda_ratio)
    saved_gain = gain.copy()
    # Snapshot ALL weights so the FROZEN (non-convergence) pathways can be byte-restored afterward. The Hebbian
    # weight DECAY is gain-gated (sim/bridge.py:6505 -> gain 0 = no decay), BUT the [hebbian_min,hebbian_max]
    # CLIP (sim/bridge.py:6509) is UNGATED: it runs every Hebbian step on EVERY weight. Because this pass lowers
    # hebbian_max_weight to `hebbian_max` (=20) for the gen edges, that ungated clip would crush the frozen
    # parser's load-bearing conj->role edges (legit ~40-60, the strong edges that drive the role ensembles) down
    # to <=20 -- silencing parse_role on the merged bridge (the Stage-1 parser-silence bug; diag2 measured the
    # gen-ON conj->role max at exactly 20.0). Only the gen_perception->gen_concept edges (conv_mask) legitimately
    # train here; everything else (parser, nav, dlPFC, concept->fact) is restored verbatim in the finally below.
    saved_weights = bridge.cp_connections.data.copy()
    # convergence train: ONLY the perception→concept edges plastic; nav/parser/dlPFC/concept→fact frozen (gain 0).
    gain[:] = 0.0
    gain[conv_mask] = 1.0
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_ou_process = False                  # the convergence de-risk trains OU-off (controlled co-activation)
    cc.hebbian_learning_rate = float(hebbian_rate)
    cc.hebbian_max_weight = float(hebbian_max)
    cc.hebbian_min_weight = 0.0
    cc.hebbian_weight_decay = 0.00001
    cc.nmda_ratio = float(nmda_ratio)

    class _A:                                     # the de-risk's train_convergence reads these off `a`
        pass
    a = _A()
    a.epochs = int(epochs); a.scene_steps = int(scene_steps)
    a.perc_scale = float(perc_scale); a.conc_scale = float(conc_scale); a.seed_base = int(seed)
    from research.runners._genfrontier_onsubstrate_convergence_derisk import train_convergence
    try:
        # train_convergence rebases perception indices by `- perc_region[0]` (it expects GLOBAL perc indices;
        # standalone the perception region sat at base 0 so it was a no-op). On the MERGED bridge gen_perception is
        # appended LAST (base > 0) and vis_sets are built LOCAL (0..N_V1_COMPLEX-1) → globalize so the rebase recovers
        # the correct local indices. (conc_blocks are already GLOBAL: conc_region.reshape.)
        perc_base = int(np.asarray(perc)[0])
        vis_sets_g = [np.asarray(vs, dtype=np.int64) + perc_base for vs in vis_sets]
        diag = train_convergence(bridge, xp, perc, conc, gen_handles["conc_blocks"], vis_sets_g, train, a)
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation, cc.enable_ou_process,
         cc.hebbian_learning_rate, cc.hebbian_max_weight, cc.hebbian_min_weight, cc.hebbian_weight_decay,
         cc.nmda_ratio) = saved
        # freeze the convergence: restore the prior gain everywhere, then hold the gen_perception→gen_concept
        # edges at gain 0 (the trained-then-frozen discipline — nav episodes must not erode the convergence).
        gain[:] = saved_gain
        gain[conv_mask] = 0.0
        # Undo the UNGATED-clip damage (sim/bridge.py:6509) to every FROZEN pathway: restore all non-convergence
        # synapses to their pre-pass values. The convergence legitimately changed ONLY the gen_perception->gen_concept
        # edges (conv_mask); the parser's strong conj->role edges (and nav/dlPFC/concept->fact) must be byte-identical
        # to before this pass, so parse_role keeps its firing margin on the merged bridge.
        _restore = ~conv_mask
        bridge.cp_connections.data[_restore] = saved_weights[_restore]
    return {"conv_mask": conv_mask, "train_diag": diag}


# ── the merged nav + parser + dlPFC bridge builder (design §2.5 FINAL FORM) ───────────────────────────────
def build_merged_nav_conv_bridge(seed: int = 42, vocab=None, n_cortex: int = 100,
                                 co_resident_rf: bool = False, rf_D: int = 128,
                                 co_resident_perception: bool = False,
                                 enable_spiking_wta_readout: bool = False,
                                 co_resident_generalization: bool = False,
                                 gen_n_concept_per: int = 100, gen_n_fact_per: int = 100,
                                 co_resident_limbic: bool = False,
                                 _global_het_test: bool = False):
    """Build ONE brain-region-framework `SimulationBridge` holding navigation + the conversational parser +
    the dlPFC dialogue-planning loop, per `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md`
    §2.5 FINAL FORM.

    Reconciliation decision (A): the framework path IS a wrapper around `inject_explicit_wiring`
    (sim/bridge.py:1514-1526 builds `region_manager.build_wiring_plan(seed)` then injects it), so the parser
    and dlPFC are appended as framework regions/pathways onto the navigation lists. The dlPFC loop/graph edges
    (per-word-pair block structure the framework cannot express) go in via ONE combined injection.

    Sequence:
      1. Union the region/pathway lists: nav (build_bg_brain_regions, DEFAULT kwargs — this is the construction
         smoke, not the flagship) + parser (parse_conj 6, parse_role 3*PARSER_R) + dlPFC (cortex_ctx, dlpfc_wm,
         both enable_nmda=True, n_dlpfc=max(600,60*V)). Append the parse_conj->parse_role plastic pathway.
      2. Config the merged cfg (framework on; dt=1; Izhikevich; the 5a clip mitigation stdp_w_max=400 /
         hebbian_max_weight=400; nav-resident learning flags; homeostasis OFF and ASSERTED so the synaptic-
         scaling clip at sim/bridge.py:6758/6760 can never slam the frozen conversational weights; NMDA on with
         ratio 0.5). Build the bridge and _initialize_simulation_data (sets region_manager, auto-injects the
         framework plan, builds the per-region NMDA mask confined to the dlPFC slices).
      3. Resolve the dlPFC slice bases from the framework, build a _SharedDlpfcContext over them, and build the
         hand-wired dlpfc_loop population on those slices.
      4. Combined injection (the gate-safe sequence): re-inject build_wiring_plan(seed) + dlpfc_loop ONCE so the
         rebuilt _plasticity_gate_to_synapses map INCLUDES the dlpfc_loop edges under DLPFC_FIXED_GATE. Re-apply
         BOTH gate zeros (the parser gate is zeroed only AFTER its train pass, step 5).
      5. Parser train pass (Hebbian temporarily on; STDP/reward off; OU=20 already on from build — see the OU
         note below), THEN set the resting OU-off nav config and freeze parser_fixed.

    Returns (bridge, handles) where handles = {region bases, the _SharedDlpfcContext, the parser index arrays,
    the dlpfc_loop edge count, the resolved vocab} the later increments (agent shim, episode/test gates) need.
    """
    # Heavy imports are deferred to here so `--microcheck` (parser-only) stays fast and import-light.
    from research.runners.g11_bg_runner import build_bg_brain_regions
    from research.runners.unified_brain_bridge import _SharedDlpfcContext

    xp, _ = get_backend()

    # Vocab: default to the 16-word probe vocab (every conversational capability is validated there). The
    # composer's DEFAULT_VOCAB is that exact 16-word probe set; sort it so the dlPFC assembly order is canonical.
    if vocab is None:
        from research.runners.rf_phasor_composer import DEFAULT_VOCAB
        vocab = DEFAULT_VOCAB
    words = sorted(vocab)
    V = len(words)
    n_dlpfc = max(600, 60 * V)

    # 1) Region / pathway union.
    # enable_spiking_wta_readout (additive, default False = byte-preserved): forward into build_bg_brain_regions so
    # the merged cascade can OPTIONALLY include the spiking-WTA selection layer `sel_{N,E,S,W}` (+ sel_FS) that
    # navigate_to_see_then_answer.build_navsee_bridge uses (line 167-168). When False (the default for every existing
    # gate), the cascade is the DEFAULT motor_X-only readout — STEP-2a/2b byte-identity preserved (the index bases of
    # nav/parser/dlPFC/rf/cortex_it are unchanged because the sel_X regions are only appended when this is True). The
    # STEP-3 behavioral runner sets it True so `_cascade_select_move` reads the validated sel_X winner (matching the
    # navsee selection quality); the default motor_X fallback also selects moves (Step-1 de-risk: 4/4 clear winners),
    # so this kwarg is a selection-quality upgrade, not a requirement.
    nav_regions, nav_pathways = build_bg_brain_regions(
        n_cortex=n_cortex, enable_spiking_wta_readout=enable_spiking_wta_readout)
    parser_regions, parser_pathways = parser_regions_pathways(PARSER_R)
    dlpfc_regions = [
        # Both dlPFC regions opt into NMDA so the framework confines the slow NMDA current to the dlPFC slice
        # (sim/bridge.py:1180-1189): if ANY region sets enable_nmda, the auto-mask is the union of those regions
        # and every other neuron is NMDA-free even with the global flag on. No internal edges; the loop/graph
        # edges are the hand-wired dlpfc_loop population (step 3/4).
        BrainRegion(name="cortex_ctx", n_neurons=n_dlpfc, exc_fraction=1.0,
                    internal_density=0.0, enable_nmda=True),
        BrainRegion(name="dlpfc_wm", n_neurons=n_dlpfc, exc_fraction=1.0,
                    internal_density=0.0, enable_nmda=True),
    ]
    # STEP 2b: reserve a contiguous `rf` slice for the resonate-and-fire composer to run co-resident on the one
    # bridge (the strict single-instance unification). 7*rf_D covers the largest single RF op (a 6-role bundle =
    # (6+1)*D). internal_density=0 and NO pathways => no cp_connections out-edges into navigation, so the rf
    # neurons' incidental Izhikevich firing between composer ops injects NOTHING into the nav cascade (the Task-1
    # anti-cheat). enable_nmda=False keeps the slow NMDA current confined to the dlPFC slices. Appended LAST so the
    # navigation/parser/dlPFC index bases are unchanged (the nav byte-identity is preserved).
    rf_regions = []
    if co_resident_rf:
        rf_regions = [BrainRegion(name="rf", n_neurons=7 * int(rf_D), exc_fraction=1.0,
                                  internal_density=0.0, enable_nmda=False)]
    # STEP-3 (compose perceived content): an optional BARE `cortex_it` perception region so the navigation
    # perception's live spiking rate code can be read OFF the merged bridge (co-resident with the whole stack) and
    # grounded into a composer concept code. internal_density=0 + NO pathways => no cp_connections out-edges into
    # navigation (like `rf`), and appended AFTER `rf` so the navigation/parser/dlPFC/rf index bases are byte-unchanged
    # (the STEP-2a/2b byte-identity is preserved). Default False = STEP-2b byte-preserved. 256 neurons + exc_fraction
    # 0.8 match the de-risk's validated cortex_it (`funcint_perception_to_memory_probe.build_probe_bridge`).
    perception_regions = []
    if co_resident_perception:
        perception_regions = [BrainRegion(name="cortex_it", n_neurons=256, exc_fraction=0.8,
                                          internal_density=0.0, enable_nmda=False)]
    # STAGE 1 (co_resident_generalization, additive default-off): the GENERALIZATION STACK — a structured-perception
    # region (Gabor/V1 top-K), an NMDA `gen_concept` region, an NMDA `gen_fact` tag region, the plastic rate-Hebbian
    # gen_perception→gen_concept convergence pathway, and the FIXED convergent gen_concept→gen_fact pathway. Appended
    # LAST (after rf + cortex_it) so the navigation/parser/dlPFC/rf/cortex_it index bases are BYTE-UNCHANGED (the
    # STEP-2a/2b byte-identity is preserved). The gen regions have internal_density=0 and NO cp_connections out-edges
    # into navigation (only gen_perception→gen_concept→gen_fact, all within the stack), so they are nav-inert. The
    # NMDA per-region mask auto-expands to include gen_concept + gen_fact (so the slow NMDA conductance lets the
    # concept assembly SPIKE), while the navigation/parser/rf slices stay NMDA-free. The convergence is trained
    # then frozen (gain-masked) after the parser pass (step 5b below). Default False = STEP-2b byte-preserved.
    generalization_regions, generalization_pathways = [], []
    if co_resident_generalization:
        generalization_regions, generalization_pathways = _generalization_regions_pathways(
            gen_n_concept_per, gen_n_fact_per)
    # SHARED LIMBIC CORE (co_resident_limbic, additive default-off): the validated reward/value/dopamine
    # organ (finding 2026-06-18-limbic-core-rpe-battery-GO.md, the Schultz RPE battery 6/6) as a co-resident
    # slice — the highest-leverage TRUE-ONE-BRAIN consolidation step (the merged bridge otherwise has NO limbic
    # core: build_bg_brain_regions is called with default kwargs). limbic_reward_us (PPN-like US afferent) ->
    # limbic_snc (DOPAMINE) <- limbic_striosome (GABAergic MSN-D1 value critic; -V via the GABA_B/GIRK K+
    # conductance); delta=r-V is the limbic_snc FIRING. limbic_cue is the generic state input (a later increment
    # wires it to the nav place code / conversational salience). ALL limbic_-prefixed (zero name collision with
    # the nav cascade) + internal_density=0; the only out-edges are WITHIN the slice (limbic_cue->striosome,
    # reward_us->snc, striosome->snc) so the slice is nav-inert (no cp_connections edges into navigation, like
    # rf/cortex_it). Appended LAST so the nav/parser/dlPFC/rf/cortex_it/gen index bases are BYTE-UNCHANGED (the
    # STEP-2a/2b byte-identity is preserved). Default False = byte-preserved.
    limbic_regions, limbic_pathways = [], []
    if co_resident_limbic:
        from sim.enums import NeuronType as _NT
        _RS = _NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name
        limbic_regions = [
            BrainRegion(name="limbic_cue", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_RS, enable_nmda=False),
            BrainRegion(name="limbic_striosome", n_neurons=60, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_STRIATAL_MSN_D1.name,
                        syn_reversal_potential_i_override=-60.0, enable_nmda=False),
            BrainRegion(name="limbic_reward_us", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_RS, enable_nmda=False),
            BrainRegion(name="limbic_snc", n_neurons=30, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_DOPAMINE.name,
                        syn_reversal_potential_i_override=-55.0, enable_nmda=False),
        ]
        # The de-risk-validated weights (the standalone organ's het-on operating point, GO 6/6). The MERGED bridge
        # runs heterogeneity OFF for nav/conv determinism, which makes the point-neuron limbic dynamics all-or-
        # nothing/chaotic (a razor-steep f-I + a finicky GABA_B operating point — finding
        # 2026-06-18-merged-limbic-core-lift.md): at these weights the value subtraction works co-resident (the diag:
        # cue+US snc ≪ US-alone snc) but the FULL multi-gate arithmetic (burst≥3× AND subtraction together) does not
        # robustly reproduce het-off. Raising the cue→striosome weight to fire the cold MSN harder BREAKS the
        # subtraction (the cue over-drives → pred≫unpred). ⇒ keep the VALIDATED 10/10/10; the clean fix is
        # INCREMENT #2 = per-region heterogeneity for the limbic slice (a small additive sim/ analogue of the
        # per-region NMDA mask) → restores the het-on operating point WITHOUT touching nav/conv determinism.
        limbic_pathways = [
            RegionPathway(from_region="limbic_cue", to_region="limbic_striosome", density=0.6,
                          weight_mean=10.0, weight_jitter=0.5, plastic=True, plasticity_gate="limbic_value"),
            RegionPathway(from_region="limbic_reward_us", to_region="limbic_snc", density=0.6,
                          weight_mean=10.0, weight_jitter=0.2, plastic=False),
            RegionPathway(from_region="limbic_striosome", to_region="limbic_snc", density=0.5,
                          weight_mean=10.0, weight_jitter=0.2, plastic=False, receptor="gaba_b"),
        ]
    union_regions = (list(nav_regions) + list(parser_regions) + list(dlpfc_regions)
                     + list(rf_regions) + list(perception_regions) + list(generalization_regions)
                     + list(limbic_regions))
    union_pathways = (list(nav_pathways) + list(parser_pathways)
                      + list(generalization_pathways) + list(limbic_pathways))   # dlPFC loop is hand-built, NOT a pathway

    # 2) Merged config.
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = union_regions
    cfg.region_pathways = union_pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    # 5a clip mitigation: raise the rule clip bounds ABOVE the max frozen conversational real-valued weight
    # (~300 parser role-route) so the ungated reward/Hebbian clips (sim/bridge.py:6200,6505) cannot move them.
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    # Navigation-resident learning state (the nav cascade runs reward-STDP during episodes).
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.enable_hebbian_learning = False           # global Hebbian OFF during nav (parser is trained separately)
    # The parser's VALIDATED Hebbian learning rate (brain_conversational_agent.py:75, microcheck reference). In
    # effect ONLY during the parser train pass (nav keeps Hebbian off so this rate is never consulted in episodes).
    # The CoreSimConfig default (0.0005) is 10x too low: the parse_conj->parse_role weights barely grow (most
    # conjunctions stay near init 0.49), the role ensembles never fire on readout, and the parse degenerates.
    cfg.hebbian_learning_rate = 0.005
    cfg.enable_homeostasis = False                # FOOT-GUN: the synaptic-scaling clip would slam frozen weights
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    # OU noise: the parser train pass REQUIRES real OU=20 (with OU off the WTA role readout is degenerate —
    # finding 2026-06-10-merge-parser-on-framework-slices-PASS-needs-OU.md). The OU per-neuron state is allocated
    # at _initialize_simulation_data ONLY when enable_ou_process is True at BUILD time; a later runtime toggle
    # does NOT allocate it (verified: build-OU-off then toggle-on trains only 2/6 conjunctions; build-OU-on
    # trains 6/6). So BUILD with OU on (state allocated), then set enable_ou_process=False for the resting
    # navigation config AFTER the parser train pass (step 5). The OU flag is read dynamically each step, so nav
    # episodes get OU-off and the read-time toggles in the smoke / later increments cleanly re-enable it.
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = 20.0
    # Default OFF for nav/conv determinism. _global_het_test=True is a DE-RISK hook ONLY (does heterogeneity
    # restore the merged limbic arithmetic? — if so, the per-region-het increment is the right fix); it perturbs
    # nav/conv determinism so it is never a production path.
    cfg.enable_parameter_heterogeneity = bool(_global_het_test)
    # NMDA on globally; the per-region mask (built at init from the enable_nmda regions) confines it to dlPFC.
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5

    # SHARED LIMBIC CORE config (co_resident_limbic): (a) enable the GABA_B/GIRK conductance (the already-shipped
    # owner-approved edit; ONLY the limbic_striosome->limbic_snc pathway is tagged receptor="gaba_b", so this is
    # additive/zero-effect for every other synapse) at the validated operating point (prop 0.22); (b) register the
    # `dopamine` signed-firing neuromodulator over [limbic_snc] = the SHARED dopamine broadcast. THRESHOLD 0.0 makes
    # it NEUTRAL-AT-REST: da_signal = sensitivity*(rate_ema - 0) >= 0, so a quiescent limbic_snc (during the parser
    # train pass + conversational ops, when the limbic slice is undriven) gives da_signal=0 -> da=baseline ->
    # plasticity-rate multiplier ~1.0 -> it CANNOT suppress the parser/conversational/nav plasticity (a positive
    # threshold would: a silent SNc -> negative da_signal -> LTD on everything). The LTD/DA-omission-dip half (the
    # signed teaching signal for the critic's V-unlearning) is an increment-#2 concern (the on-merge critic-learning
    # + nav-reward routing), calibrated there with a tonic-driven limbic_snc. Both ONLY when co_resident_limbic ->
    # default-off byte-preserved (the merge otherwise has enable_gabab=False + zero neuromodulators).
    if co_resident_limbic:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 150.0
        cfg.gabab_propagation_strength = 0.22
        cfg.gabab_conductance_max = 0.0
        from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=8.0,
                                             threshold=0.0, window_ms=200.0, source_regions=["limbic_snc"])])]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # The homeostasis foot-gun guard: build_bg_brain_regions / its config path must not have toggled the global
    # flag on (only the global flag enables the [0.01,5.0] synaptic-scaling clip that would crush weight ~30/300).
    assert cfg.enable_homeostasis is False, "global homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager

    # 3) Resolve dlPFC slice bases and build the loop population on those slices.
    cortex_base = rm.indices("cortex_ctx")[0]
    dlpfc_base = rm.indices("dlpfc_wm")[0]
    dlpfc_ctx = _SharedDlpfcContext(bridge, words, cortex_base, dlpfc_base, n_dlpfc, DLPFC_PATTERN_SIZE, seed)
    dlpfc_loop = _build_dlpfc_loop_population(dlpfc_ctx, words)
    n_dlpfc_loop_edges = int(dlpfc_loop["count"])

    # 4) Combined injection (design §2.5 final form). Rebuild the union framework plan + ADD dlpfc_loop, then
    #    inject ONCE. inject_explicit_wiring rebuilds cp_connections + the gate maps from scratch, so this single
    #    re-injection produces a _plasticity_gate_to_synapses map that includes dlpfc_loop under DLPFC_FIXED_GATE.
    #    Pass the SAME output_inhibitory_indices the auto-injection used (sim/bridge.py:1520-1525) so the nav
    #    inhibitory neurons keep their inhibitory trait (D2 MSNs, GPe/GPi, FS pools, ...).
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    assert "dlpfc_loop" not in union_plan, "dlpfc_loop name collides with a framework population"
    union_plan["dlpfc_loop"] = dlpfc_loop
    # STAGE 1: OVERWRITE the generalization pathways' framework-uniform entries with the EXACT all-to-all
    # (perception→concept) + convergent block-diagonal (concept→fact) edges (the dlpfc_loop insertion pattern).
    # The framework declared both pathways (so the clean wiring branch ran + the plan keys exist); the precise
    # structure is installed here in the SAME single inject_explicit_wiring below.
    gen_handles = None
    if co_resident_generalization:
        gen_edges, gen_handles = _build_generalization_edges(rm, gen_n_concept_per, gen_n_fact_per)
        for plan_key, edge in gen_edges.items():
            assert plan_key in union_plan, \
                f"FAIL: generalization plan key {plan_key!r} not in the union plan ({sorted(union_plan)[:12]}...)"
            union_plan[plan_key] = edge
    inh_indices_concat = []
    for region in rm.regions():
        inh_indices_concat.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh_indices_concat or None)

    # Re-apply both gate zeros (the gate maps were rebuilt -> default gain 1.0). The dlPFC gate is zeroed NOW
    # (before the parser train pass) so the parser pass cannot drift the dlPFC edges either. The parser gate is
    # zeroed only AFTER its train pass (step 5).
    bridge.set_plasticity_gate(DLPFC_FIXED_GATE, 0.0)

    # 5) Parser train pass on the framework slices (after the FINAL injection — a later injection would reset the
    #    trained weights). Temporarily Hebbian ON + STDP/reward OFF; OU=20 is already ON from build (the validated
    #    condition — with OU off the WTA readout is degenerate, and OU state must be allocated at build, see above).
    conj_arr, role_arr = _parser_index_arrays(bridge, PARSER_R)
    cc = bridge.core_config
    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation)
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    # OU is already ON from build (state allocated); the train pass uses it directly (ou_std_current_pA=20).
    try:
        train_parser_on_slices(bridge, conj_arr, role_arr)
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation) = saved
    # Now set the RESTING navigation config: OU OFF (nav episodes run OU-off; the OU state stays allocated so the
    # smoke/later increments can re-enable it per-read). The flag is read dynamically each step.
    cc.enable_ou_process = False
    bridge.set_plasticity_gate(PARSER_GATE, 0.0)

    # 5b) STAGE 1 generalization convergence train pass (after the parser pass — a later injection would reset
    #     the trained weights, and the parser must already be final). The perception→concept rate-Hebbian
    #     convergence is trained ISOLATED from nav/parser/dlPFC via the cp_plasticity_rate_gain index mask, then
    #     FROZEN (gain 0). The vision sets + leakage-free split are the de-risk's exact Gabor/V1 pipeline.
    gen_extra = {}
    if co_resident_generalization:
        (vis_sets, gen_held_out, gen_train, gen_cat_ids, gen_set_margin, gen_structure_preserved,
         gen_W, gen_top_k) = _gen_vision_sets_and_split(int(seed))
        train_meta = _train_merged_convergence(bridge, gen_handles, vis_sets, gen_train, seed=int(seed))
        gen_extra = {
            "vis_sets": vis_sets, "gen_held_out": gen_held_out, "gen_train": gen_train,
            "gen_cat_ids": gen_cat_ids, "gen_set_margin": gen_set_margin,
            "gen_structure_preserved": gen_structure_preserved, "gen_W": gen_W, "gen_top_k": gen_top_k,
            "gen_conv_mask": train_meta["conv_mask"], "gen_train_diag": train_meta["train_diag"],
        }

    handles = {
        "seed": int(seed),
        "vocab": words,
        "n_dlpfc": int(n_dlpfc),
        "cortex_base": int(cortex_base),
        "dlpfc_base": int(dlpfc_base),
        "dlpfc_ctx": dlpfc_ctx,
        "dlpfc_loop_edges": n_dlpfc_loop_edges,
        "conj_arr": conj_arr,
        "role_arr": role_arr,
    }
    if co_resident_rf:
        handles["rf_base"] = int(rm.indices("rf")[0])
        handles["rf_size"] = 7 * int(rf_D)
        handles["rf_D"] = int(rf_D)
    if co_resident_generalization:
        handles["gen"] = dict(gen_handles, **gen_extra)
    if co_resident_limbic:
        # The limbic-slice base indices (for the on-merge RPE-battery validation + the later increments that
        # drive limbic_cue/limbic_reward_us and read limbic_snc/limbic_striosome).
        handles["limbic"] = {n: {"base": int(rm.indices(n)[0]), "size": len(list(rm.indices(n)))}
                             for n in ("limbic_cue", "limbic_striosome", "limbic_reward_us", "limbic_snc")}
    return bridge, handles


# ── the EPISODE-path conv finalization (nav gate (a): nav episode runs on the merged bridge) ─────────────────
def conv_extra_regions_pathways(vocab=None, co_resident_rf=False, rf_D=128):
    """The conversational regions/pathways to APPEND to the navigation lists for the episode-path merge: the
    parser (parse_conj 6, parse_role 3*PARSER_R) + the dlPFC regions (cortex_ctx, dlpfc_wm, both enable_nmda).
    For the NAV GATE the dlPFC regions are present but EDGELESS (the dlpfc_loop is only for `elaborate`, not
    needed for nav-not-regressed), so they are silent during the nav episode. Returns (extra_regions,
    extra_pathways) for `run_moving_goal_episode(extra_regions=, extra_pathways=)`.

    co_resident_rf (STEP 2b): also append the `rf` composer region (7*rf_D neurons, no pathways, NMDA-off) so the
    nav-not-regressed gate can be re-run with the rf slice present. The rf region has NO cp_connections out-edges
    into navigation (the Task-1 anti-cheat) and is idle during the nav episode (no composer ops run mid-episode),
    so it is provably nav-inert — this gate just confirms that empirically."""
    if vocab is None:
        from research.runners.rf_phasor_composer import DEFAULT_VOCAB
        vocab = DEFAULT_VOCAB
    V = len(set(vocab))
    n_dlpfc = max(600, 60 * V)
    parser_regions, parser_pathways = parser_regions_pathways(PARSER_R)
    dlpfc_regions = [
        BrainRegion(name="cortex_ctx", n_neurons=n_dlpfc, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="dlpfc_wm", n_neurons=n_dlpfc, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
    ]
    rf_regions = []
    if co_resident_rf:
        rf_regions = [BrainRegion(name="rf", n_neurons=7 * int(rf_D), exc_fraction=1.0,
                                  internal_density=0.0, enable_nmda=False)]
    return list(parser_regions) + list(dlpfc_regions) + list(rf_regions), list(parser_pathways)


def finalize_conv_for_nav_gate(bridge, seed=42, R=PARSER_R, n_epochs=30, train_steps=120):
    """The `prebuilt_post_init_hook` `run_moving_goal_episode` calls AFTER its Gabor/SC post-init wiring and
    BEFORE the episode loop. Trains the parser on the merged bridge's framework slices and freezes it, so the
    navigation episode runs with the conversational populations frozen (the conv neurons must not perturb nav).

    INDEX/MASK-BASED (the load-bearing point): the navigation post-init helpers `apply_v1_gabor_weights` +
    `install_spiking_sc_wiring` call `set_pathway_weights(add_missing=True)`, which `tocsr()+sum_duplicates()`
    RE-SORTS the synapse data (`sim/bridge.py:2851-2853`) and does NOT re-derive the plasticity-gate index maps
    — so the framework-registered `parser_fixed` gate is STALE here. We therefore compute the parser synapse
    mask DIRECTLY from the FINAL CSR (guaranteed cp_connections.data-aligned via indptr/indices) and manage
    `cp_plasticity_rate_gain` by that mask, NOT by the stale gate name. The gain is masked so the parser's
    Hebbian train pass (gain 1 on the parser only) cannot decay the FIXED Gabor/SC perception + navigation
    edges (gain 0) via the ungated Hebbian decay (~1e-6/step). After training, the parser is frozen (gain 0)
    and navigation is plastic (gain 1) for the reward-STDP episode. NO dlpfc_loop (that is for `elaborate`, a
    follow-on; the dlPFC regions are present but edgeless → silent during nav). Returns handles for anti-cheat.
    """
    xp, _ = get_backend()
    rm = bridge.region_manager
    csr = bridge.cp_connections
    # per-data-position (pre, post), guaranteed aligned with cp_connections.data / cp_plasticity_rate_gain.
    # Computed on the HOST from indptr/indices (backend-agnostic — cupy.repeat rejects an array `repeats`),
    # then the boolean mask is pushed to the device. The indptr loop is over n_rows (a few thousand) = fast.
    indptr_h = to_host(csr.indptr)
    post_h = to_host(csr.indices).astype(np.int64)
    nnz = int(post_h.shape[0])
    pre_h = np.zeros(nnz, dtype=np.int64)
    for r in range(int(csr.shape[0])):
        pre_h[int(indptr_h[r]):int(indptr_h[r + 1])] = r
    pc = np.asarray(rm.indices("parse_conj"), dtype=np.int64)
    prr = np.asarray(rm.indices("parse_role"), dtype=np.int64)
    parser_mask = xp.asarray(np.isin(pre_h, pc) & np.isin(post_h, prr))

    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    gain = bridge.cp_plasticity_rate_gain

    conj_arr, role_arr = _parser_index_arrays(bridge, R)
    cc = bridge.core_config
    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
             cc.enable_ou_process, cc.hebbian_max_weight, cc.hebbian_learning_rate)
    # parser train: ONLY the parser plastic; Gabor/SC + nav frozen (gain 0) so the Hebbian decay can't erode
    # them. hebbian_max_weight=400 + lr=0.005 are the validated parser-pass values; OU=20 (state allocated at
    # build via build_with_ou) for the WTA role readout.
    gain[:] = 0.0
    gain[parser_mask] = 1.0
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.enable_ou_process = True
    cc.hebbian_max_weight = 400.0
    cc.hebbian_learning_rate = 0.005
    try:
        train_parser_on_slices(bridge, conj_arr, role_arr, n_epochs=n_epochs, train_steps=train_steps)
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
         cc.enable_ou_process, cc.hebbian_max_weight, cc.hebbian_learning_rate) = saved
    # restore for the episode: parser FROZEN (gain 0), navigation plastic (gain 1), OU off (nav default).
    gain[:] = 1.0
    gain[parser_mask] = 0.0
    cc.enable_ou_process = False
    return {"conj_arr": conj_arr, "role_arr": role_arr, "parser_mask": parser_mask}


# ── the parser adapter (the BrainConversationalAgent `.parser.parse(...)` surface on framework slices) ───────
class _MergedParserAdapter:
    """Exposes the `BridgeParser` READ surface (`parse(words, voice)`) on the merged bridge's framework parser
    slices, so the agent shim's `self.parser.parse(...)` call sites are byte-for-byte the BrainConversationalAgent's.

    There is no separate parser bridge: `parse` drives `parse_conj`/`parse_role` slices of the MERGED bridge via the
    ported `parse_on_slices`. The parser readout needs OU (finding 2026-06-10-merge-parser-on-framework-slices-PASS-
    needs-OU.md): the merged config rests with OU off, so this adapter toggles OU on (20 pA) for the read and restores
    the resting flag afterward — mirroring how `construction_smoke` and `UnifiedBrainBridge.elaborate` toggle their
    reads. The OU per-neuron state was allocated at build (build-OU-on), so the runtime toggle is clean."""

    ROLES = ROLES

    def __init__(self, bridge, conj_arr, role_arr):
        self._bridge = bridge
        self._conj_arr = conj_arr
        self._role_arr = role_arr

    def parse(self, words, voice="active"):
        cc = self._bridge.core_config
        prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
        cc.enable_ou_process = True
        cc.ou_std_current_pA = 20.0
        try:
            return parse_on_slices(self._bridge, self._conj_arr, self._role_arr, words, voice)
        finally:
            cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std

    def role_of(self, position, voice=0):
        cc = self._bridge.core_config
        prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
        cc.enable_ou_process = True
        cc.ou_std_current_pA = 20.0
        try:
            return role_of_on_slices(self._bridge, self._conj_arr, self._role_arr, position, voice)
        finally:
            cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std


# ── the agent shim (design §3 STEP 2a: the BrainConversationalAgent surface on the merged bridge) ────────────
class MergedRFComposer(RFPhasorComposer):
    """STEP 2b: an `RFPhasorComposer` whose RF binding ops run on a SLICE of the shared merged bridge (the strict
    single-instance co-residence) instead of on its own per-op bridges.

    Only `_resonate` is overridden — it shifts the op's local indices into the bridge's `rf` region, builds a
    full-N complex kick (zeros off the slice), installs the full-`(N,N)` complex weights (the bind/unbind diagonals
    are O(D) sparse, so the size is fine), kicks with `neuron_mask=rf_mask` (the owner-approved sliced `rf_kick`),
    runs the resonate loop (which auto-respects the persisted mask via `self._rf_neuron_mask`, so the loop touches
    only the rf slice), and reads the slice's phases. `_bind`/`_bundle`/`_unbind_phases`/`_encode`/`_render` are
    inherited unchanged — they call this `_resonate`.

    Why this is sound (the de-risks): the composer is stateless per op (re-kicks each op) and stores fact memory in
    numpy (`self.kb`), so a navigation Izhikevich step between ops harmlessly clobbers the idle rf slice's v/u
    (re-kicked next op); the masked write-back leaves the navigation neurons' v/u byte-untouched (the 5b coexistence
    guarantee). The complex weights `cp_rf_w_re/im` are array-disjoint from `cp_connections`, so the navigation step
    never corrupts the fact-binding synapses. NO further `sim/` edit. enable_spiking_cleanup co-residence is out of
    STEP-2b scope (the numpy argmax cleanup is the validated default readout, a pure-numpy op with no bridge)."""

    def __init__(self, merged_bridge, rf_base, rf_size, **kwargs):
        if kwargs.get("enable_spiking_cleanup"):
            raise NotImplementedError(
                "co-resident spiking cleanup is out of STEP-2b scope; use the numpy argmax cleanup (default)")
        super().__init__(**kwargs)
        self._merged = merged_bridge
        self._rf_base = int(rf_base)
        self._rf_size = int(rf_size)
        xp, _ = get_backend()
        mask = xp.zeros(int(merged_bridge.core_config.num_neurons), dtype=bool)
        mask[self._rf_base:self._rf_base + self._rf_size] = True
        self._rf_mask = mask

    def _resonate(self, n, conns, kick):
        n = int(n)
        if n > self._rf_size:
            raise ValueError(
                f"RF op needs {n} neurons but the merged rf region is {self._rf_size} "
                f"(raise rf_D so 7*rf_D >= {n})")
        b = self._merged
        N = int(b.core_config.num_neurons)
        base = self._rf_base
        shifted = [(base + int(post), base + int(pre), w) for (post, pre, w) in conns]
        b.rf_set_complex_weights(shifted)
        full_kick = np.zeros(N, dtype=np.complex128)
        kk = np.asarray(kick, dtype=np.complex128).reshape(-1)
        full_kick[base:base + n] = kk[:n]
        b.rf_kick(full_kick, period=self.period, lam=0.0, neuron_mask=self._rf_mask)
        b.rf_resonate_steps(self.period + 8)
        phases = np.asarray(b.rf_read_phases())
        return phases[base:base + n]


class MergedNavConvAgent:
    """STEP 2a agent shim: the `BrainConversationalAgent` surface where comprehension (the PARSER) and dialogue
    planning (the dlPFC `elaborate`) run on the MERGED nav+conv `SimulationBridge`, while fact storage/retrieval
    (`store`/`query_*`/`render_fact`/`ask_yes_no`) delegate to a SEPARATE-bridge production `RFPhasorComposer`
    (STEP 2a keeps the RF composer on its own per-op bridges; STEP 2b co-residence is gated on the owner byte-review).

    Per `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md` §3 STEP 2a. The method signatures + semantics
    MATCH `brain_conversational_agent.BrainConversationalAgent` exactly so `tests/test_brain_conversational_agent.py`'s
    assertions pass VERBATIM against this shim (incl. the three `is None` no-confab moat assertions).

    Anti-cheat (asserted in __init__, design §3): the parser+dlPFC actually run on the merged bridge (a silent fallback
    to a standalone parser/dlPFC bridge fails loudly):
      * `"parse_conj"` AND `"dlpfc_wm"` are regions of `self._merged_bridge.region_manager` (the parser+dlPFC slices
        live on the merged bridge), AND
      * the dlPFC context `elaborate` drives is the merged bridge's (`self._dlpfc_ctx.bridge is self._merged_bridge`).
    """

    def __init__(self, seed=42, vocab=None, co_resident_composer=False, co_resident_limbic=False):
        """Build the merged nav+parser+dlPFC bridge + the composer (same seed + vocab). The composer's vocab is the
        merged dlPFC vocab (the sorted probe vocab) so the dialogue-planning assemblies and the fact-memory codebook
        share one word set.

        co_resident_composer (STEP 2b): when True, the fact-binding composer runs CO-RESIDENT on the merged bridge's
        own `rf` slice (the strict single-instance unification, via the owner-approved sliced `rf_kick`); when False
        (STEP 2a default) it runs on its own separate per-op bridges."""
        self.seed = int(seed)
        self.co_resident_composer = bool(co_resident_composer)
        # co_resident_limbic (TRUE ONE BRAIN item #1): also lift the shared reward/value/dopamine limbic core
        # onto the merged bridge (the limbic_ slice + the `dopamine` modulator). Default False = byte-preserved
        # (the conversational gate b is unaffected). When True, the moat-no-regression check verifies the shared
        # DA modulator (threshold-0, neutral-at-rest) does not perturb the parser/conversational comprehension.
        self.co_resident_limbic = bool(co_resident_limbic)
        _D = 128
        self._merged_bridge, self._handles = build_merged_nav_conv_bridge(
            seed=seed, vocab=vocab, co_resident_rf=self.co_resident_composer, rf_D=_D,
            co_resident_limbic=self.co_resident_limbic)
        words = self._handles["vocab"]   # the sorted merged vocab (the dlPFC + parser word set)

        # The composer. STEP 2a default: on its OWN per-op bridges (separate). STEP 2b (co_resident_composer=True):
        # the MergedRFComposer runs the RF binding ops on the merged bridge's own `rf` slice. Same seed + vocab as
        # the merged dlPFC either way.
        if self.co_resident_composer:
            self.composer = MergedRFComposer(
                self._merged_bridge, self._handles["rf_base"], self._handles["rf_size"],
                seed=seed, D=_D, vocab=words, period=200)
        else:
            self.composer = RFPhasorComposer(seed=seed, D=_D, vocab=words, period=200)

        # The parser READ surface on the merged framework slices (so `self.parser.parse(...)` matches the agent's).
        self.parser = _MergedParserAdapter(self._merged_bridge, self._handles["conj_arr"], self._handles["role_arr"])

        # The dlPFC context (`_SharedDlpfcContext` over the merged cortex_ctx/dlpfc_wm slices) that `elaborate` drives.
        self._dlpfc_ctx = self._handles["dlpfc_ctx"]
        self._dlpfc_controller = None
        self._dlpfc_graph_key = None

        # --- ANTI-CHEAT asserts (design §3): the parser+dlPFC run on the MERGED bridge, not a standalone fallback. ---
        region_names = self._merged_bridge.region_manager.region_indices_dict()
        assert "parse_conj" in region_names, \
            f"FAIL anti-cheat: 'parse_conj' not on the merged bridge (regions: {sorted(region_names)[:8]}...)"
        assert "dlpfc_wm" in region_names, \
            f"FAIL anti-cheat: 'dlpfc_wm' not on the merged bridge (regions: {sorted(region_names)[:8]}...)"
        assert self._dlpfc_ctx.bridge is self._merged_bridge, \
            "FAIL anti-cheat: elaborate's dlPFC context is NOT the merged bridge (silent standalone-dlPFC fallback)"
        if self.co_resident_composer:
            assert "rf" in region_names, \
                "FAIL anti-cheat: co_resident_composer set but no 'rf' region on the merged bridge"
            assert self.composer._merged is self._merged_bridge, \
                "FAIL anti-cheat: the co-resident composer is not bound to the merged bridge"

    # --- comprehend / store / recall (mirror BrainConversationalAgent exactly) ---
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO statement and store it. `sentence` is 'agent action patient' (or its passive frame)."""
        roles = self.parser.parse(sentence.split(), voice)
        self.composer.store(roles["agent"], roles["action"], roles["patient"], polarity=polarity)
        return roles

    def hear_clause_fact(self, agent, action, clause, polarity=None):
        """Store a fact whose patient is an embedded clause (the parser handles flat SVO; nested input parsing is
        future work, so the clause is provided structurally here)."""
        self.composer.store(agent, action, clause, polarity=polarity)

    def what_does(self, agent, action):
        """'what does <agent> <action>?' -> patient (concept or rendered clause) or None (abstain)."""
        return self.composer.query_patient(agent, action)

    def who_does(self, action, patient):
        return self.composer.query_agent(action, patient)

    def is_it_true(self, agent, action, patient):
        return self.composer.ask_yes_no(agent, action, patient)

    def describe(self, agent):
        """Generation: produce a sentence about `agent` from the spiking memory ('dog go north'), or None if the agent
        knows no fact about it (no confabulation)."""
        return self.composer.render_fact(agent)

    # --- dialogue planning (the dlPFC `elaborate`, ported from UnifiedBrainBridge.elaborate onto the merged slices) ---
    def _assoc_graph(self):
        """The association graph (concept -> {concept: weight}) built from the composer's OWN stored facts — identical
        to `BrainConversationalAgent._assoc_graph` / `UnifiedBrainBridge._assoc_graph` (agent/action/patient of each
        fact co-occur; clause patients skipped)."""
        graph = {}
        for fact, _ in self.composer.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient")]
            cs = [c for c in cs if isinstance(c, str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning on the MERGED bridge's dlPFC slice: bring up the next on-topic concept about `topic`,
        chosen by the dlPFC spiking spreading-activation Control over the composer's own association graph. Returns an
        associate concept, or None if `topic` is unconnected (the abstention / no-confab moat). Matches
        `BrainConversationalAgent.elaborate` semantics and PORTS `UnifiedBrainBridge.elaborate`
        (unified_brain_bridge.py:684-732) onto the merged-bridge dlPFC slices.

        The Control is a `SpikingSpreadingController` built WITHOUT its __init__ (which would build a throwaway bridge):
        its attributes are set by hand against the shared-slice context (`self._dlpfc_ctx`), and its OWN validated
        `_install_graph_edges` overwrites the pre-allocated `dlpfc_loop` c2d SLOTS in place (no CSR rebuild → the
        parser/nav gate maps are preserved). Cached, rebuilt only when the graph CONTENT changes. OU is toggled OFF for
        the spreading-activation read (the validated dlPFC regime; OU tips the bistable attractors into spurious ON
        states), then restored — exactly as UnifiedBrainBridge.elaborate."""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        from research.runners.content_selection import SaidTrace

        graph = self._assoc_graph()
        if topic not in graph:
            return None
        # cache key = the graph CONTENT (not kb length: different fact sets can share a length -> stale Control).
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc_controller is None or self._dlpfc_graph_key != key:
            ctrl = object.__new__(SpikingSpreadingController)
            ctrl.graph = graph
            ctrl._vocab = sorted(set(graph) | {a for v in graph.values() for a in v})
            ctrl.ctx = self._dlpfc_ctx
            ctrl.said = SaidTrace(decay=0.9)                      # the validated SpikingSpreadingController default
            ctrl._install_graph_edges(DLPFC_EDGE_SCALE)           # overwrites the pre-allocated c2d slots IN PLACE
            self._dlpfc_controller = ctrl
            self._dlpfc_graph_key = key
        prev_ou = self._merged_bridge.core_config.enable_ou_process
        self._merged_bridge.core_config.enable_ou_process = False
        try:
            return self._dlpfc_controller.turn_latency([topic])
        finally:
            self._merged_bridge.core_config.enable_ou_process = prev_ou


# ── the construction smoke (this increment's acceptance) ─────────────────────────────────────────────────
def construction_smoke(seed: int = 42, n_cortex: int = 100, vocab=None):
    """Build the merged bridge and assert it is structurally correct: nav + parser + dlPFC co-reside on ONE
    bridge, both fixed gates resolve and are actually zeroed, the dlpfc_loop edge count matches, the neuron
    count is the exact union, and the parser parses (voice-invariantly) on the FULL merged bridge.

    Returns True on PASS. Raises AssertionError (caught by the CLI -> exit 1) on any failure.
    """
    xp, backend = get_backend()
    print(f"[construction-smoke] backend={backend} seed={seed} n_cortex={n_cortex}")
    bridge, h = build_merged_nav_conv_bridge(seed=seed, vocab=vocab, n_cortex=n_cortex)
    rm = bridge.region_manager
    cfg = bridge.core_config

    n_regions = len(cfg.brain_regions)
    n_neurons = int(cfg.num_neurons)
    nnz = int(bridge.cp_connections.nnz)
    print(f"[construction-smoke] {n_regions} regions, {n_neurons} neurons, {nnz} synapses "
          f"(dlpfc_loop edges={h['dlpfc_loop_edges']}, n_dlpfc={h['n_dlpfc']})")

    # (a) the homeostasis foot-gun stayed off.
    assert cfg.enable_homeostasis is False, \
        "FAIL: cfg.enable_homeostasis is True — the synaptic-scaling clip would crush the frozen conv weights"

    # (b) both fixed gates exist in the gate map, and the dlpfc gate covers exactly the dlpfc_loop edges built.
    gate_map = bridge._plasticity_gate_to_synapses
    assert PARSER_GATE in gate_map, f"FAIL: '{PARSER_GATE}' not a key of _plasticity_gate_to_synapses ({list(gate_map)})"
    assert DLPFC_FIXED_GATE in gate_map, f"FAIL: '{DLPFC_FIXED_GATE}' not a key of _plasticity_gate_to_synapses ({list(gate_map)})"
    dlpfc_gate_size = int(bridge._plasticity_gate_indices_gpu[DLPFC_FIXED_GATE].size)
    assert dlpfc_gate_size == h["dlpfc_loop_edges"], \
        f"FAIL: dlpfc_fixed gate covers {dlpfc_gate_size} synapses, expected {h['dlpfc_loop_edges']} (the dlpfc_loop edges)"

    # (c) the gains are ACTUALLY 0 at both gates' synapses (freeze took effect, not just registered).
    dlpfc_idx = bridge._plasticity_gate_indices_gpu[DLPFC_FIXED_GATE]
    parser_idx = bridge._plasticity_gate_indices_gpu[PARSER_GATE]
    dlpfc_gains = to_host(bridge.cp_plasticity_rate_gain[dlpfc_idx])
    parser_gains = to_host(bridge.cp_plasticity_rate_gain[parser_idx])
    assert bool((dlpfc_gains == 0).all()), "FAIL: dlpfc_fixed plasticity gains are not all 0 after freeze"
    assert bool((parser_gains == 0).all()), "FAIL: parser_fixed plasticity gains are not all 0 after freeze"

    # (d) nav AND conv regions co-reside (proves one bridge holds both brains).
    region_names = rm.region_indices_dict()
    for required in ("cortex_N", "parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm"):
        assert required in region_names, \
            f"FAIL: region '{required}' missing from the merged bridge (regions: {sorted(region_names)[:8]}...)"

    # (e) the neuron count is the EXACT union: nav + parser (6 + 3*R = 126) + 2*n_dlpfc.
    parser_total = 6 + 3 * PARSER_R
    region_sum = sum(int(r.n_neurons) for r in cfg.brain_regions)
    nav_total = region_sum - parser_total - 2 * h["n_dlpfc"]
    assert n_neurons == region_sum, f"FAIL: num_neurons {n_neurons} != sum of region sizes {region_sum}"
    assert region_sum == nav_total + parser_total + 2 * h["n_dlpfc"], "FAIL: union neuron-count arithmetic"
    print(f"[construction-smoke] neuron union: nav={nav_total} + parser={parser_total} + "
          f"2*dlpfc={2 * h['n_dlpfc']} == {region_sum}")

    # (f) the parser parses on the FULL merged bridge (the parser readout needs OU; toggle it on for the read,
    #     then restore the merged config's OU-off). Voice-invariance confirmed too (cheap).
    conj_arr, role_arr = h["conj_arr"], h["role_arr"]
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        active = parse_on_slices(bridge, conj_arr, role_arr, ["dog", "go", "north"], voice="active")
        passive = parse_on_slices(bridge, conj_arr, role_arr, ["north", "go", "dog"], voice="passive")
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    print(f"[construction-smoke] active  parse: {active}")
    print(f"[construction-smoke] passive parse: {passive}")
    assert active.get("agent") == "dog", f"FAIL: active 'dog go north' agent != dog ({active})"
    assert passive.get("agent") == "dog", f"FAIL: voice-invariance broke ('north go dog' passive agent != dog) ({passive})"

    print("\n[construction-smoke] PASS - nav + parser + dlPFC co-reside on ONE framework bridge; both fixed "
          "gates resolve and are zeroed; the parser parses voice-invariantly on the merged bridge.")
    return True


# ── the parser-on-framework-slices micro-check (risk 4.1) ────────────────────────────────────────────────
def _build_parser_microcheck_bridge(seed: int, R: int, nav_stub: int, ou: float):
    """A framework bridge with [nav_stub, parse_conj, parse_role] under the MERGED config. The nav_stub forces
    a non-zero offset so the parser's slice arithmetic is exercised (the real merged condition)."""
    regions, pathways = parser_regions_pathways(R)
    if nav_stub > 0:
        regions = [BrainRegion(name="nav_stub", n_neurons=nav_stub, exc_fraction=1.0,
                               internal_density=0.0)] + regions
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    # MERGED config: Hebbian for the parser pass, STDP/reward OFF, the merged clip bounds (5a mitigation).
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.005
    cfg.hebbian_max_weight = 400.0
    cfg.stdp_w_max = 400.0
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    if ou > 0:
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = float(ou)
    else:
        cfg.enable_ou_process = False
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def microcheck(seed: int = 42, R: int = PARSER_R, nav_stub: int = 50, ou: float = 0.0,
               n_epochs: int = 30, train_steps: int = 120):
    xp, backend = get_backend()
    print(f"[parser-microcheck] backend={backend} seed={seed} nav_stub={nav_stub} ou={ou}")
    bridge = _build_parser_microcheck_bridge(seed, R, nav_stub, ou)
    rm = bridge.region_manager
    conj_base = rm.indices("parse_conj")[0]
    role_base = rm.indices("parse_role")[0]
    print(f"[parser-microcheck] {len(bridge.core_config.brain_regions)} regions, "
          f"{bridge.core_config.num_neurons} neurons, {int(bridge.cp_connections.nnz)} synapses; "
          f"conj_base={conj_base} role_base={role_base}")

    conj_arr, role_arr = _parser_index_arrays(bridge, R)

    # train pass on the framework slices, then FREEZE the parser gate (5a isolation)
    train_parser_on_slices(bridge, conj_arr, role_arr, n_epochs=n_epochs, train_steps=train_steps)
    bridge.set_plasticity_gate(PARSER_GATE, 0.0)

    # comprehension: active "dog go north" and the passive frame must both call dog the agent
    active = parse_on_slices(bridge, conj_arr, role_arr, ["dog", "go", "north"], voice="active")
    passive = parse_on_slices(bridge, conj_arr, role_arr, ["north", "go", "dog"], voice="passive")
    print(f"[parser-microcheck] active  parse: {active}")
    print(f"[parser-microcheck] passive parse: {passive}")

    ok_active = active.get("agent") == "dog" and active.get("action") == "go" and active.get("patient") == "north"
    ok_voice_inv = passive.get("agent") == "dog"   # voice-invariant agent assignment
    passed = ok_active and ok_voice_inv

    print(f"\n[parser-microcheck] active SVO correct      : {ok_active}")
    print(f"[parser-microcheck] voice-invariant agent    : {ok_voice_inv}  (passive 'north go dog' -> agent=dog)")
    print(f"[parser-microcheck] {'PASS' if passed else 'FAIL'} — the parser "
          f"{'learns the voice-invariant role map on framework slices (risk 4.1 retired)' if passed else 'did NOT port cleanly; see parses above'}")
    return passed


# ── STEP 2a nav gate (a): run the nav episode on the MERGED bridge (single-seed smoke) ───────────────────────
def nav_on_merged_smoke(seed=42, n_steps=400, grid_size=8, vocab=None,
                        out="research/findings/raw/nav_gate_2a/nav_on_merged_smoke.json"):
    """Run the navigation episode on the MERGED nav+conv bridge (via the hybrid run_moving_goal_episode hook)
    and assert: (A1) nav + conv regions co-reside on ONE bridge; (A2) the conversational (parser) weights stay
    BYTE-IDENTICAL across the episode (frozen under the LIVE nav reward-STDP + dopamine stressor — the 5a
    isolation, now in vivo) + the gains are still 0; (A3) the parser still parses on the merged bridge after
    the episode. enable_visual_cortex=True so the Gabor post-init `set_pathway_weights(add_missing=True)` CSR
    rebuild runs (exercising the index-based finalize against the rebuilt CSR). NOT the full 6-seed gate."""
    import os
    import numpy as np
    from sim.backend import to_host
    from research.runners.g11_bg_runner import run_moving_goal_episode

    os.makedirs(os.path.dirname(out), exist_ok=True)
    extra_regions, extra_pathways = conv_extra_regions_pathways(vocab)
    box = {}

    def hook(bridge):
        h = finalize_conv_for_nav_gate(bridge, seed=seed)
        box["bridge"] = bridge
        box["parser_mask"] = h["parser_mask"]
        box["conj_arr"] = h["conj_arr"]
        box["role_arr"] = h["role_arr"]
        box["pre_nnz"] = int(bridge.cp_connections.nnz)
        box["pre_conv"] = to_host(bridge.cp_connections.data[h["parser_mask"]]).copy()
        box["n_conv"] = int(h["parser_mask"].sum())

    print(f"[nav-on-merged-smoke] seed={seed} grid={grid_size} n_steps={n_steps} (enable_visual_cortex => Gabor rebuild)")
    run_moving_goal_episode(
        out_path=out, seed=seed, n_steps=n_steps, grid_size=grid_size,
        enable_visual_cortex=True, visual_cortex_action_warmup_steps=min(100, max(1, n_steps // 2)),
        stdp_w_max_override=400.0,
        extra_regions=extra_regions, extra_pathways=extra_pathways,
        build_with_ou=True, prebuilt_post_init_hook=hook,
    )

    bridge = box["bridge"]
    rm = bridge.region_manager
    region_names = rm.region_indices_dict()
    print(f"[nav-on-merged-smoke] merged bridge: {len(bridge.core_config.brain_regions)} regions, "
          f"{int(bridge.core_config.num_neurons)} neurons, {box['n_conv']} frozen parser synapses")

    a1 = all(r in region_names for r in ("cortex_N", "parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm"))
    nnz_same = int(bridge.cp_connections.nnz) == box["pre_nnz"]
    if nnz_same:
        post_conv = to_host(bridge.cp_connections.data[box["parser_mask"]])
        post_gain = to_host(bridge.cp_plasticity_rate_gain[box["parser_mask"]])
        a2_weights = bool(np.array_equal(box["pre_conv"], post_conv))
        a2_gains = bool((post_gain == 0).all())
    else:
        a2_weights = a2_gains = False  # structural plasticity changed the synapse set (should be OFF in nav)
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        parse = parse_on_slices(bridge, box["conj_arr"], box["role_arr"], ["dog", "go", "north"], "active")
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    a3 = parse.get("agent") == "dog"

    passed = a1 and a2_weights and a2_gains and a3
    print(f"[nav-on-merged-smoke] (A1) nav+conv regions co-reside : {a1}")
    print(f"[nav-on-merged-smoke] (A2) parser weights frozen      : {a2_weights}  gains==0: {a2_gains}  (nnz_same={nnz_same})")
    print(f"[nav-on-merged-smoke] (A3) parser parses post-episode : {a3}  (active 'dog go north' -> {parse})")
    print(f"\n[nav-on-merged-smoke] {'PASS' if passed else 'FAIL'} - the merged bridge navigates AND the "
          f"conversational populations stay byte-frozen under the live nav reward-STDP stressor.")
    return passed


def main():
    ap = argparse.ArgumentParser(description="Nav+Conv merge builder (parser microcheck + construction smoke)")
    ap.add_argument("--microcheck", action="store_true",
                    help="parser-only framework bridge: validate the parser ports onto framework slices (risk 4.1)")
    ap.add_argument("--construction-smoke", action="store_true",
                    help="build the FULL merged nav+parser+dlPFC bridge and assert it is structurally correct")
    ap.add_argument("--nav-on-merged-smoke", action="store_true",
                    help="STEP 2a gate (a): run the nav episode on the merged bridge; assert nav navigates + conv frozen")
    ap.add_argument("--n-steps", type=int, default=400, help="nav-on-merged-smoke episode length")
    ap.add_argument("--grid-size", type=int, default=8, help="nav-on-merged-smoke grid size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cortex", type=int, default=100, help="build_bg_brain_regions n_cortex (construction smoke)")
    ap.add_argument("--nav-stub", type=int, default=50)
    ap.add_argument("--ou", type=float, default=20.0,
                    help="OU noise pA for the parser train pass (validated: 20 PASSES, 0=off FAILS — degenerate "
                         "WTA readout; the merge enables OU only for the pass, then restores OU-off for nav)")
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--train-steps", type=int, default=120)
    args = ap.parse_args()
    if args.nav_on_merged_smoke:
        ok = nav_on_merged_smoke(seed=args.seed, n_steps=args.n_steps, grid_size=args.grid_size)
        raise SystemExit(0 if ok else 1)
    if args.construction_smoke:
        ok = construction_smoke(seed=args.seed, n_cortex=args.n_cortex)
        raise SystemExit(0 if ok else 1)
    if args.microcheck:
        ok = microcheck(seed=args.seed, nav_stub=args.nav_stub, ou=args.ou,
                        n_epochs=args.n_epochs, train_steps=args.train_steps)
        raise SystemExit(0 if ok else 1)
    ap.error("pass --construction-smoke (full merged bridge) or --microcheck (parser-only)")


if __name__ == "__main__":
    main()
