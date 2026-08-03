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
# Module-level too (the CoResidentOneBrainComposer base class needs it at class-definition time -- the consolidation
# port, option A). one_brain_composer imports rf_phasor_composer + the same sim.* substrate (no new heavy deps), so
# this does not slow the parser-only --microcheck path either.
from research.runners.one_brain_composer import OneBrainComposer

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

# Minimal grounded speech-action slice.  The visual association is plastic and
# trained by the developmental runner; the need/cue coincidence and the
# request-vs-silence competition are fixed neural pathways on this bridge.
SPEECH_FOOD_CUE = "speech_food_cue"
SPEECH_REQUEST = "speech_request"
SPEECH_SILENCE = "speech_silence"
SPEECH_WTA_FS = "speech_wta_fs"
SPEECH_GROUNDING_GATE = "speech_grounding"

# ── command-route (route A: language->action) constants — ported from spoken_instruction_nav.py (GO 3-seed) ────
# The CONSOLIDATION (FOLLOW-ON #2) lifts the LEARNED `language_input -> cortex_X` route + its `command_route`
# transmission gate onto the merged bridge so MergedNavConvAgent.command_move() steers nav from a PARSED command.
# These mirror spoken_instruction_nav.py's constants exactly (the validated values).
COMMAND_GATE = "command_route"                     # the ONE transmission gate on language_input -> cortex_X
CMD_LANG_GATE = "language_input_to_cortex"         # the LEARNED route's plasticity gate (open to train, then freeze)
CMD_N_LANG_INPUT = 256                             # language_input region size (g11 text-IO default)
CMD_LANG_TO_CORTEX_DENSITY = 0.20                  # the route's density (g11 text_input_to_cortex_density)
CMD_LANG_TO_CORTEX_INIT_W = 2.0                    # non-zero init (g11 canon: dense-then-prune, not grow-from-zero)
CMD_LANG_TO_CORTEX_JITTER = 0.5
CMD_LANG_DRIVE_PA = 2500.0                         # per-active-neuron language_input drive (composer ROLE_DRIVE scale)
CMD_LANG_SPARSITY = 0.1
CMD_ROUTE_TRAIN_EPOCHS = 30
CMD_ROUTE_TRAIN_STEPS = 60                          # per (word, teacher) co-drive window
CMD_ROUTE_TEACHER_PA = 600.0                       # cortex_d teacher current (supervised co-fire label)
CMD_ROUTE_HEBBIAN_LR = 0.02                        # the LEARNED-route Hebbian rate (the route grows in 30 epochs)
CMD_ROUTE_SETTLE_STEPS = 30                         # the per-epoch inter-drive settle (matches the standalone)


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


def _grounded_speech_regions_pathways(n_acc: int = 40, n_fs: int = 20,
                                      drive_weight: float = 8.0,
                                      cue_weight: float = 60.0):
    """Regions for a minimal grounded request action on the shared bridge.

    The learned ``gen_perception -> speech_food_cue`` route associates the
    spiking visual feature ensemble with food. Hunger and that cue converge on ``speech_request``; either
    input alone is intended to remain below the competing silence population.
    ``drive_pomc`` adds satiety evidence to silence.  The runner tunes only the
    operating point and trains the visual association; inference is synaptic.
    """
    from sim.enums import NeuronType as _NT

    _RS = _NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    _FS = _NT.IZH2007_FS_CORTICAL_INTERNEURON.name
    regions = [
        BrainRegion(name=SPEECH_FOOD_CUE, n_neurons=int(n_acc), exc_fraction=1.0,
                    internal_density=0.15, exc_weight_mean=0.3, inh_weight_mean=0.0,
                    weight_jitter=0.05, plastic_internal=False, izh_neuron_type=_RS,
                    enable_nmda=True, enable_homeostasis=True),
        BrainRegion(name=SPEECH_REQUEST, n_neurons=int(n_acc), exc_fraction=1.0,
                    internal_density=0.35, exc_weight_mean=0.3, inh_weight_mean=0.0,
                    weight_jitter=0.05, plastic_internal=False, izh_neuron_type=_RS,
                    enable_nmda=True, enable_homeostasis=True),
        BrainRegion(name=SPEECH_SILENCE, n_neurons=int(n_acc), exc_fraction=1.0,
                    internal_density=0.35, exc_weight_mean=0.3, inh_weight_mean=0.0,
                    weight_jitter=0.05, plastic_internal=False, izh_neuron_type=_RS,
                    enable_nmda=True, enable_homeostasis=True),
        BrainRegion(name=SPEECH_WTA_FS, n_neurons=int(n_fs), exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=_FS,
                    enable_nmda=False, enable_homeostasis=True),
    ]
    pathways = [
        RegionPathway(from_region=GEN_PERCEPTION, to_region=SPEECH_FOOD_CUE,
                      density=0.25, weight_mean=0.05, weight_jitter=0.0,
                      plastic=True, plasticity_gate=SPEECH_GROUNDING_GATE),
        RegionPathway(from_region="drive_agrp", to_region=SPEECH_REQUEST,
                      density=0.60, weight_mean=float(drive_weight), weight_jitter=0.10,
                      plastic=False),
        RegionPathway(from_region=SPEECH_FOOD_CUE, to_region=SPEECH_REQUEST,
                      density=0.60, weight_mean=float(cue_weight), weight_jitter=0.10,
                      plastic=False),
        RegionPathway(from_region="drive_pomc", to_region=SPEECH_SILENCE,
                      density=0.60, weight_mean=float(drive_weight), weight_jitter=0.10,
                      plastic=False),
        RegionPathway(from_region=SPEECH_REQUEST, to_region=SPEECH_WTA_FS,
                      density=0.50, weight_mean=8.0, weight_jitter=0.10, plastic=False),
        RegionPathway(from_region=SPEECH_SILENCE, to_region=SPEECH_WTA_FS,
                      density=0.50, weight_mean=8.0, weight_jitter=0.10, plastic=False),
        RegionPathway(from_region=SPEECH_WTA_FS, to_region=SPEECH_REQUEST,
                      density=0.60, weight_mean=6.0, weight_jitter=0.10,
                      plastic=False, receptor="gaba_a"),
        RegionPathway(from_region=SPEECH_WTA_FS, to_region=SPEECH_SILENCE,
                      density=0.60, weight_mean=6.0, weight_jitter=0.10,
                      plastic=False, receptor="gaba_a"),
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


# ── command-route training (route A: language->action) — ported from spoken_instruction_nav._train_learned_route ─
def _command_band_excitatory(rm, region_name, cue_idx, n_cues, sparsity):
    """The EXCITATORY global indices of cue_idx's orthogonal band within `region_name` (the word code must hit the
    band's EXCITATORY neurons; including inhibitory ones makes the word drive partly suppressive). Layout MUST match
    orthogonal_drive_pattern exactly. Ported VERBATIM from spoken_instruction_nav._orthogonal_band_excitatory."""
    idx_h = np.asarray(list(rm.indices(region_name)), dtype=np.int64)
    n = int(idx_h.size)
    n_active = max(1, int(round(sparsity * n)))
    stride = n // n_cues
    if n_active > stride:
        raise ValueError(f"band overlap: n_active={n_active} > stride={stride}")
    start = cue_idx * stride
    band = idx_h[start:start + n_active]
    inh = set(int(i) for i in rm.inhibitory_indices(region_name))
    return np.asarray([int(p) for p in band if int(p) not in inh], dtype=np.int64)


def _train_command_route_on_merged(bridge, seed):
    """Grow the direction-selectivity of `language_input -> cortex_X` by BRAIN-BASED co-firing (Pulvermüller action-
    word somatotopy), then FREEZE it. Ported from spoken_instruction_nav._train_learned_route: for each direction d,
    co-drive language_input(d)'s orthogonal band + a teacher current on cortex_d with Hebbian ON, so the simultaneously
    active pre (the word code) and post (cortex_d) strengthen their connection. ISOLATED to the route via the global
    plasticity gain (0 everywhere, 1 on the CMD_LANG_GATE), so the nav cascade's own plastic pathways do not drift.
    After training: route frozen (CMD_LANG_GATE 0), global gain restored to 1, the command_route transmission gate
    CLOSED (the agent reopens it per-decision via the parser-firing coupling)."""
    from research.runners.g11_bg_runner import ACTION_NAMES as _CR_ACTIONS, N_ACTIONS as _CR_N
    xp, _ = get_backend()
    rm = bridge.region_manager
    n = int(bridge.core_config.num_neurons)
    cc = bridge.core_config

    band_exc = {a: _command_band_excitatory(rm, "language_input", i, _CR_N, CMD_LANG_SPARSITY)
                for i, a in enumerate(_CR_ACTIONS)}
    cortex_idx = {a: np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64) for a in _CR_ACTIONS}

    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    saved_gain = bridge.cp_plasticity_rate_gain.copy()       # restore the parser-frozen / nav-plastic gain afterward
    bridge.set_global_plasticity_gain(0.0)                   # only the route's gate-1 synapses learn this pass
    bridge.set_plasticity_gate(CMD_LANG_GATE, 1.0)
    bridge.set_transmission_gate(COMMAND_GATE, 1.0)          # open the route's current during training

    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
             cc.hebbian_learning_rate, cc.enable_ou_process, cc.ou_std_current_pA)
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.hebbian_learning_rate = CMD_ROUTE_HEBBIAN_LR
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        for _ in range(CMD_ROUTE_TRAIN_EPOCHS):
            for a in _CR_ACTIONS:
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(CMD_ROUTE_SETTLE_STEPS):
                    bridge._run_one_simulation_step()
                cur = xp.zeros(n, dtype=xp.float32)
                cur[xp.asarray(band_exc[a])] = CMD_LANG_DRIVE_PA          # the word code (pre)
                cur[xp.asarray(cortex_idx[a])] = CMD_ROUTE_TEACHER_PA      # the cortex_d teacher (post label)
                bridge.cp_external_input_current[:] = cur
                for _ in range(CMD_ROUTE_TRAIN_STEPS):
                    bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
         cc.hebbian_learning_rate, cc.enable_ou_process, cc.ou_std_current_pA) = saved
    # FREEZE the route + restore the parser-frozen / nav-plastic gain; close the command gate (reopened per-decision).
    bridge.set_plasticity_gate(CMD_LANG_GATE, 0.0)
    bridge.cp_plasticity_rate_gain[:] = saved_gain
    bridge.set_transmission_gate(COMMAND_GATE, 0.0)
    cc.enable_ou_process = False                              # the resting nav config


# ── the merged nav + parser + dlPFC bridge builder (design §2.5 FINAL FORM) ───────────────────────────────
def build_merged_nav_conv_bridge(seed: int = 42, vocab=None, n_cortex: int = 100,
                                 co_resident_rf: bool = False, rf_D: int = 128,
                                 onebrain_rf_size: int = 0,
                                 co_resident_perception: bool = False,
                                 co_resident_command_route: bool = False,
                                 enable_spiking_wta_readout: bool = False,
                                 co_resident_generalization: bool = False,
                                 gen_n_concept_per: int = 100, gen_n_fact_per: int = 100,
                                 co_resident_limbic: bool = False,
                                 co_resident_drive: bool = False,
                                 drive_n_pool: int = 60,
                                 drive_to_da: bool = False,
                                 drive_da_sensitivity: float = 8.0,
                                 co_resident_grounded_speech: bool = False,
                                 speech_n_acc: int = 40, speech_n_fs: int = 20,
                                 speech_drive_weight: float = 8.0,
                                 speech_cue_weight: float = 60.0,
                                 co_resident_nav_critic: bool = False,
                                 nav_critic_convergent_upstate: bool = False,
                                 nav_critic_homeostasis_mask: str = "all3",
                                 nav_critic_spiking_sc: bool = False,
                                 nav_critic_place_selforg: bool = False,
                                 nav_critic_grid_frontend: bool = False,
                                 co_resident_td_cueshift: bool = False,
                                 td_csc_n: int = 8, td_csc_n_per: int = 25,
                                 td_csc_to_strio_weight: float = 14.0,
                                 td_to_fs_weight: float = 16.0, td_fs_to_strio_weight: float = 10.0,
                                 td_strio_to_snc_weight: float = 1.5,
                                 td_gabab_prop: float = 0.105, td_gabab_conductance_max: float = 0.0,
                                 td_stdp_w_max: float = 0.0,
                                 td_derivative_gain: float = 1.0, td_slow_tau_ms: float = 130.0,
                                 co_resident_hippo_memory: bool = False,
                                 hippo_n_ca3: int = 500, hippo_n_ca1: int = 120, hippo_k_thresh: float = 66.0,
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
    # co_resident_nav_critic (CYCLE 209, the FULL nav reward/critic limbic organ — the real consolidation
    # target, de-risked 2026-06-18-organ-lift-homeo-generalize-derisk.md): lift build_bg_brain_regions' validated
    # spiking critic (striosome_value MSN-D1 + GABA_B->snc) + the reward_us US-afferent onto the merged bridge.
    # The CYCLE-208 per-region-homeostasis enabler restores the SNc f-I (the existing enable_critic_homeostasis
    # masks only the afferent+critic, so snc+reward_us are masked POST-HOC below). Mutually exclusive with the
    # 4-region minimal co_resident_limbic organ (two DA pools + two scope=all broadcasts would double-count).
    assert not (co_resident_limbic and co_resident_nav_critic), \
        "co_resident_limbic (minimal organ) and co_resident_nav_critic (full nav critic) are mutually exclusive"
    # co_resident_td_cueshift (TRUE ONE BRAIN roadmap #3, the A-CSC TD cue-shift CONSOLIDATION onto the one brain
    # — finding 2026-06-10-N9-TD-cue-shift-A-CSC-GO.md, migration r -0.80/-0.77/-0.89): lift the validated
    # complete-serial-compound TD machinery (the tapped-delay cue chain + the multi-channel reward relay + the
    # B-2 conductance-derivative + the FS-clamp) onto the merged bridge as a co-resident td_-prefixed slice, so the
    # one brain computes delta = r + gamma*V(s') - V(s) (TEMPORAL-difference, the burst MIGRATES onto the cue) on
    # top of the R-W delta=r-V the limbic/nav-critic organs already supply. It registers its OWN `dopamine`
    # signed-firing modulator over [td_snc], so it is MUTUALLY EXCLUSIVE with co_resident_limbic / co_resident_nav_critic
    # (two scope=all DA broadcasts off two SNc pools would double-count the plasticity-rate modulation).
    assert not (co_resident_td_cueshift and (co_resident_limbic or co_resident_nav_critic)), \
        ("co_resident_td_cueshift (A-CSC TD cue-shift) registers its own dopamine modulator over [td_snc] and is "
         "mutually exclusive with co_resident_limbic / co_resident_nav_critic (one DA broadcast per merged bridge)")
    # nav_critic_place_selforg (TRUE ONE BRAIN roadmap #5 — retire the host-Gaussian vs_place_context afferent for
    # the SELF-ORGANIZED spiking `place` code): forward neural_place_selforg=True into build_bg_brain_regions so the
    # critic's position afferent becomes the self-org hippocampal `place` pool (place_sensors -> place threshold-WTA
    # + place_fs FS-PING -> the PLASTIC place->striosome_value coincidence_detector pathway, g11_bg_runner.py:1175/
    # :1783) instead of the host-Gaussian vs_place_context (g11_bg_runner.py:1841 `elif enable_neural_critic:`).
    # DUAL VALUE: (a) retires a host shortcut (the position code becomes self-organized spiking place cells); (b) the
    # coincidence-volley fires the MSN-D1 critic from the LEARNED code WITHOUT the position-BLIND up-state bootstrap
    # that capped the CYCLE-211/214 value-train delta ~1.3 -> so it CAN lift the delta. MUTUALLY EXCLUSIVE with
    # nav_critic_convergent_upstate: the up-state arm is vs_place_context-specific (the self-org branch has no
    # vs_place_drive region/pathway), and g11_bg_runner.py:3853 HARD-GATES enable_convergent_upstate OFF whenever
    # neural_place_selforg is on (the position-blind A1 floor caps grading) -- so co-requesting both is a config error.
    # Default False = byte-identical to the vs_place_context value-train build.
    assert not (nav_critic_place_selforg and nav_critic_convergent_upstate), \
        ("nav_critic_place_selforg (self-org place afferent) and nav_critic_convergent_upstate (vs_place_context "
         "up-state arm) are mutually exclusive: the up-state arm has no vs_place_drive in the self-org branch and "
         "g11_bg_runner hard-gates enable_convergent_upstate OFF under neural_place_selforg.")
    # nav_critic_grid_frontend (#5b R1 SURPASS, production-wiring nav chunk item 2): the place_sensors afferent is
    # the DECORRELATED spatial-phase grid metric instead of the locally-degenerate landmark render -> the self-org
    # place pool carves SELECTIVE fields (place value V n/f 4.5-12.3x, R1 GO 3/3,
    # research/findings/2026-06-22-shortcut5b-R1-grid-frontend-derisk.md). REQUIRES nav_critic_place_selforg (it IS
    # the place_sensors afferent). This LOW-LEVEL builder default STAYS False (conservative -- the research runners
    # that compose their own critic config keep the assert protecting a genuine double-request); the PRODUCTION
    # default-ON lives in MergedNavConvAgent (the `None` sentinel below). #5b CLOSED (2026-06-22-shortcut5b-td-read-
    # derisk.md): the host-Gaussian vs_place_context retires on R1 grounds (the grid front end produces a genuinely-
    # neural, value-gradable place code -- selectivity + learned near/far value, both 3/3); the residual value-READ
    # structural/learned separation (the graded-plateau read conflates the place code's structural near/far magnitude
    # with learned value) is the CHARACTERIZED DENDRITIC FRONTIER (a point-neuron limit, NOT a host shortcut, NOT a
    # blocker -- the existing graded-plateau read stays; the close does NOT depend on the TD read).
    assert not (nav_critic_grid_frontend and not nav_critic_place_selforg), \
        ("nav_critic_grid_frontend (the grid-cell place afferent) requires nav_critic_place_selforg (it IS the "
         "self-org place_sensors afferent; without the self-org place pool there is nothing to drive).")
    if co_resident_grounded_speech and not (co_resident_generalization and co_resident_drive):
        raise ValueError("co_resident_grounded_speech requires co_resident_generalization and co_resident_drive")
    if co_resident_nav_critic:
        # nav_critic_convergent_upstate (the value-train A1 up-state arm, CYCLE 209 value-train build): forward
        # enable_convergent_upstate to build_bg_brain_regions so the dense NON-plastic vs_place_drive->striosome_value
        # arm is wired (sums past the MSN-D1 rheobase at the goal from init -> breaks the LTP bootstrap), and the
        # PLASTIC vs_place_context->striosome_value STDP learns V on top. Default False = byte-identical to the
        # op-map build (no vs_place_drive region/pathway). The op-map de-risk drove vs_place_context DIRECTLY at the
        # MSN rheobase (the trained-V proxy); the value-train needs the up-state arm to fire the critic so the STDP
        # has a post-spike to pair with at the INIT weight 0.20.
        # nav_critic_spiking_sc (TRUE ONE BRAIN roadmap #2 — the NEURAL reward `r` SOURCE, default-off):
        # also build the spiking superior colliculus chain (sc_retina -> sc_map -> sc_fs Mexican-hat ->
        # sc_rostral) + the sc_rostral->reward_us pathway (g11_bg_runner.py:2488/2535/2541), so the reward `r`
        # the SNc bursts on is produced SYNAPTICALLY by the SC bump's proximity/goal-salience (the N5 approach
        # reward), retiring the host sign(Manhattan) formula that currently drives reward_us (g11_bg_runner.py:
        # 7148). The sc_map->sc_rostral foveal-centre proximity readout + retina->sc_map retinotopy are wired
        # POST-INIT by install_spiking_sc_wiring (it uses set_pathway_weights, add_missing); the de-risk /
        # episode hook calls it after build. The SC chain is self-contained on its OWN sc_retina (does NOT
        # require enable_visual_cortex for the region build — that flag is for the deployed nav-loop retina).
        # Default False = byte-identical to the value-train build (no sc_* regions). N5 validated standalone by
        # sc_n5_rpe_probe.py (corr(distance, SNc)=-0.99, omission dip, lesion sc_rostral->reward_us collapses).
        # SCOPE NOTE (roadmap #2): the SC region build (g11_bg_runner.py:2488 `if enable_spiking_sc:`) is NESTED
        # inside `if enable_visual_cortex:` (:2428) — so the SC chain (sc_retina/sc_map/sc_fs/sc_rostral) ONLY
        # builds when enable_visual_cortex=True. ⇒ nav_critic_spiking_sc forwards BOTH flags: the full vision
        # hierarchy (retina/V1/V2/cortex_it) AND the SC chain. (cortex_it is also the critic's perceived-state
        # afferent, so this is consistent with the value-train critic.)
        nav_regions, nav_pathways = build_bg_brain_regions(
            n_cortex=n_cortex, enable_spiking_wta_readout=enable_spiking_wta_readout,
            enable_neural_critic=True, spiking_reward_us=True, enable_critic_homeostasis=True,
            enable_convergent_upstate=bool(nav_critic_convergent_upstate),
            neural_place_selforg=bool(nav_critic_place_selforg),
            nav_critic_grid_frontend=bool(nav_critic_grid_frontend),
            enable_visual_cortex=bool(nav_critic_spiking_sc),
            enable_spiking_sc=bool(nav_critic_spiking_sc),
            enable_spiking_sc_approach=bool(nav_critic_spiking_sc))
        # POST-HOC homeostasis mask: snc + reward_us are built WITHOUT enable_homeostasis (g11_bg_runner.py:
        # 1133-1142/:1158-1163). The "all3" set (the original op-map default) ALSO masks snc + reward_us -> the SNc
        # SATURATES (~435 Hz, no GABA_B headroom). The op-map de-risk (2026-06-18-navcritic-valuetrain-opmap-derisk.md
        # §1) found the RECOMMENDED op-point is the "critic_only" mask: ONLY striosome_value gets the per-region
        # homeostasis boost; snc + reward_us stay at vpeak (non-saturated ~97 Hz burst, gap ~19). reward_us STILL
        # bursts the SNc from vpeak (the US->DA reflex doesn't need the f-I boost), and the homeostasis mask is
        # f-I-IRRELEVANT for the MSN-D1 critic anyway (its rheobase is vt=-25mV, not threshold-set, §2) — so
        # "critic_only" ~= no critic homeostasis at all + the SNc at its native Stage-B f-I.
        _mask_names = (("striosome_value",) if nav_critic_homeostasis_mask == "critic_only"
                       else ("snc", "reward_us", "striosome_value"))
        for _r in nav_regions:
            if _r.name in _mask_names:
                _r.enable_homeostasis = True
    else:
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
    #
    # rf region size: default `7*rf_D` covers the largest SINGLE MergedRFComposer op (a 6-role bundle = (6+1)*D). The
    # co-resident OneBrainComposer (option A) needs a MUCH larger span (work registers + k_max store blocks + per-block
    # + batched Q+cleanup); the agent passes `onebrain_rf_size` = CoResidentOneBrainComposer.n_total_for(...). When >0,
    # it OVERRIDES `7*rf_D` (and must be >= it, since the same region still hosts the largest single RF op). Default 0
    # = the byte-unchanged MergedRFComposer sizing (the `--composer onebrain` path is the only caller that sets it).
    _rf_n = int(onebrain_rf_size) if int(onebrain_rf_size) > 0 else 7 * int(rf_D)
    if int(onebrain_rf_size) > 0 and int(onebrain_rf_size) < 7 * int(rf_D):
        raise ValueError(f"onebrain_rf_size={onebrain_rf_size} < 7*rf_D={7 * int(rf_D)} (the rf region must still hold "
                         f"the largest single RF op)")
    rf_regions = []
    if co_resident_rf:
        rf_regions = [BrainRegion(name="rf", n_neurons=_rf_n, exc_fraction=1.0,
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
    # COMMAND-ROUTE SLICE (co_resident_command_route, route A: language->action, additive default-off): the
    # `language_input` region + the LEARNED `language_input -> cortex_X` route (transmission_gate=command_route,
    # plasticity_gate=language_input_to_cortex), ported VERBATIM from the GO standalone spoken_instruction_nav.py
    # (3-seed GO: COUPLED 1.0, LESION ~0.1). The route is plastic at build (the LEARNED word->cortex selectivity is
    # GROWN by co-firing in step 5c below), then frozen; its CURRENT is scaled by the `command_route` transmission
    # gate, which the agent couples to the parser's action-role FIRING (so a comprehended verb opens the route).
    # language_input is appended AFTER cortex_it (so the nav/parser/dlPFC/rf/cortex_it index bases are byte-unchanged);
    # the route pathways enter the EXISTING nav cortex_{N,E,S,W} (they are real cp_connections edges into navigation —
    # held CLOSED by the gate at rest, so nav-inert until a parsed command opens them). NMDA off. Default False =
    # byte-preserved. enable_spiking_wta_readout is forced on by the agent for this route (the sel_X readout the
    # standalone validated). Uses N_ACTIONS cortex_X pools, which only exist when the nav cascade is present.
    command_route_regions, command_route_pathways = [], []
    if co_resident_command_route:
        from research.runners.g11_bg_runner import ACTION_NAMES as _CR_ACTIONS
        from sim.enums import NeuronType as _CR_NT
        command_route_regions = [BrainRegion(
            name="language_input", n_neurons=CMD_N_LANG_INPUT, exc_fraction=0.8, internal_density=0.05,
            exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2, plastic_internal=True,
            izh_neuron_type=_CR_NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        for _a in _CR_ACTIONS:
            command_route_pathways.append(RegionPathway(
                from_region="language_input", to_region=f"cortex_{_a}",
                density=CMD_LANG_TO_CORTEX_DENSITY, weight_mean=CMD_LANG_TO_CORTEX_INIT_W,
                weight_jitter=CMD_LANG_TO_CORTEX_JITTER, plastic=True,
                plasticity_gate=CMD_LANG_GATE, transmission_gate=COMMAND_GATE))
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
        # enable_homeostasis=True PER-REGION (the merged-config operating-point fix, root-caused 2026-06-18 by the
        # deep-research subagent): the standalone limbic organ was pinned with the CoreSimConfig DEFAULT
        # enable_homeostasis=True, which lowers the spike threshold from vpeak (+35mV) to the homeostatic threshold
        # (~-42mV) — a large f-I gain. The merged bridge keeps GLOBAL enable_homeostasis OFF (the synaptic-scaling
        # foot-gun would crush the frozen conversational weights), so a standalone-tuned organ fires ~6-10x weaker
        # co-resident and its GABA_B arithmetic collapses. The fix uses the ALREADY-SHIPPED per-region homeostasis
        # mask (sim/bridge.py:1227-1245 builds cp_homeostasis_neuron_mask from regions with enable_homeostasis=True;
        # :6320-6323 the threshold-select gives the masked neurons the adapted threshold + everyone else vpeak) —
        # so ONLY the limbic slice gets the low threshold, the conversational/nav slices stay at vpeak (byte-
        # unchanged), and the synaptic-scaling clip (gated by the SEPARATE cfg.enable_synaptic_scaling, OFF here)
        # never runs → the frozen-weight foot-gun is NOT triggered. NO sim/ edit (the scaffold exists). SYSTEMIC:
        # this is the general fix for lifting any standalone-tuned spiking organ onto the het-off merged bridge.
        limbic_regions = [
            BrainRegion(name="limbic_cue", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
            BrainRegion(name="limbic_striosome", n_neurons=60, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_STRIATAL_MSN_D1.name,
                        syn_reversal_potential_i_override=-60.0, enable_nmda=False, enable_homeostasis=True),
            BrainRegion(name="limbic_reward_us", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
            BrainRegion(name="limbic_snc", n_neurons=30, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_DOPAMINE.name,
                        syn_reversal_potential_i_override=-55.0, enable_nmda=False, enable_homeostasis=True),
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

    # INTEROCEPTIVE DRIVE SLICE (co_resident_drive, additive default-off): the validated 2-pool SPIKING hunger drive
    # (hypothalamic AgRP=hunger / POMC=satiety; catalog O.05/O.06; finding 2026-06-17-homeostatic-spiking-drive-
    # mechanism-GO.md, GO 6/6 corr(deficit,AgRP)>=0.9) lifted as actual CO-RESIDENT neurons on the merged bridge — the
    # Tier-3 living-loop primitive (the rate-proxy persistent living loop is GO 6/6, 2026-06-20-tier3-persistent-living-
    # loop-derisk.md; this realizes its DRIVE in spikes on the shared substrate). drive_agrp ∝ the body's energy DEFICIT
    # (an interoceptive current — the legitimate body→sensory boundary), drive_pomc ∝ the surplus; the SPIKING hunger =
    # drive_agrp firing rate read off `cp_firing_states` (NOT a host deficit value), which gates the validated BG-cascade
    # reward via run_moving_goal_episode(homeostatic_hook=...) → an INTRINSIC drive-reduction reward (Keramati-Gutkin).
    # The pools are DRIVEN by external interoceptive current + READ by firing rate, so they need NO internal pathways and
    # NO neuromodulator (reading the firing directly is fully brain-based — the firing IS the drive signal — and avoids a
    # second scope=all plasticity-rate modulator colliding with the nav DA broadcast). ALL drive_-prefixed (zero name
    # collision with the nav cascade / limbic / td / rf) + internal_density=0. With grounded speech OFF there are zero
    # out-edges; with it ON, AgRP projects only to the appended speech-request pool. Neither mode adds edges into
    # navigation, so the slice remains nav-inert (like rf/cortex_it). enable_
    # homeostasis=True PER-REGION (the merged-config operating-point fix, the limbic-core-lift lesson): the standalone
    # drive was tuned with the CoreSimConfig default enable_homeostasis=True (vpeak->~-42mV threshold, the f-I gain that
    # makes the deficit current cross threshold); the merged bridge keeps GLOBAL homeostasis OFF (the synaptic-scaling
    # foot-gun), so the already-shipped per-region homeostasis mask (sim/bridge.py:1227-1245/:6320) gives ONLY the drive
    # slice the low threshold while nav/conv stay at vpeak (byte-unchanged) and the synaptic-scaling clip (gated by the
    # SEPARATE cfg.enable_synaptic_scaling, OFF here) never runs. Appended LAST so the nav/parser/dlPFC/rf/cortex_it/gen/
    # limbic index bases are BYTE-UNCHANGED. Default False = byte-preserved. Compatible with co_resident_nav_critic /
    # co_resident_limbic / co_resident_td_cueshift (the drive adds NO DA broadcast — it only reads firing — so it cannot
    # double-count any of their dopamine modulators).
    drive_regions, drive_pathways = [], []
    if co_resident_drive:
        from sim.enums import NeuronType as _NT
        _RS = _NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name
        drive_regions = [
            BrainRegion(name="drive_agrp", n_neurons=int(drive_n_pool), exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
            BrainRegion(name="drive_pomc", n_neurons=int(drive_n_pool), exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
        ]
        drive_pathways = []  # The opt-in grounded-speech slice adds its own AgRP output pathway.

    speech_regions, speech_pathways = [], []
    if co_resident_grounded_speech:
        speech_regions, speech_pathways = _grounded_speech_regions_pathways(
            n_acc=speech_n_acc, n_fs=speech_n_fs,
            drive_weight=speech_drive_weight, cue_weight=speech_cue_weight)

    # A-CSC TD CUE-SHIFT SLICE (co_resident_td_cueshift, additive default-off): the validated complete-serial-compound
    # TD machinery (snc_stageb_critic_probe.py --td-csc, GO 3/3 r=-0.80/-0.77/-0.89) as a td_-prefixed co-resident slice.
    # Regions: td_csc_0..td_csc_{K-1} (the tapped-delay cue, each its OWN plastic synapse onto the critic) + td_striosome
    # (the GABAergic MSN-D1 value critic) + td_fs (the production FS-clamp, holds the critic sparse as the per-tap
    # weights grow) + td_reward_us (the excitatory reward relay the critic inhibits => r-V at td_snc, localizing -V to
    # the reward window) + td_snc (DOPAMINE). Pathways: td_csc_k->td_striosome plastic (the tap value w_k, gated
    # `td_value` so it can be frozen for the byte-isolation checks); td_csc_k->td_fs + td_fs->td_striosome (the clamp);
    # td_reward_us->td_snc exc + td_striosome->td_reward_us inhib (the relay r-V); td_striosome->td_snc GABA_B (the -V
    # level + the conductance-derivative source). ALL td_-prefixed (zero name collision) + internal_density=0; the only
    # out-edges are WITHIN the slice, so the slice is nav-inert (no cp_connections edges into navigation, like rf/limbic).
    # Per-region enable_homeostasis=True is the merged-config operating-point fix (the limbic-core-lift lesson): the
    # standalone A-CSC organ was tuned with the CoreSimConfig default enable_homeostasis=True (vpeak->~-42mV threshold,
    # a large f-I gain); the merged bridge keeps GLOBAL homeostasis OFF (the synaptic-scaling foot-gun), so the already-
    # shipped per-region homeostasis mask (sim/bridge.py:1227-1245/:6320) gives ONLY the td slice the low threshold while
    # nav/conv stay at vpeak (byte-unchanged) and the synaptic-scaling clip (gated by the SEPARATE enable_synaptic_scaling,
    # OFF here) never runs. Appended LAST so the nav/parser/dlPFC/rf/cortex_it/gen/limbic index bases are BYTE-UNCHANGED.
    td_regions, td_pathways = [], []
    if co_resident_td_cueshift:
        from sim.enums import NeuronType as _NT
        _RS = _NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name
        K = int(td_csc_n)
        # enable_heterogeneity=True is the merged-config OPERATING-POINT fix for roadmap #3 (2026-06-18,
        # the BOUNDARY root-cause). The standalone A-CSC organ was tuned with the CoreSimConfig DEFAULT
        # enable_parameter_heterogeneity=True (per-neuron jittered izh_a/b/d/C => a GRADED MSN-D1 f-I band);
        # the merged bridge keeps GLOBAL het OFF (nav/conv determinism), which — combined with the 5a
        # stdp_w_max=400 clip + the per-region homeostasis low threshold — drives the HOMOGENEOUS td_striosome
        # critic ~6x hotter than the standalone => V SATURATES (clamps the dopamine -V flat) => the TD peak
        # stays stuck @ reward (migration r=-0.43, not the r<-0.7 GO bar). The --global-het-test diagnostic
        # CONFIRMED that het-ON restores the graded critic + value-growth + reward-shrink + dip. The already-
        # shipped per-region HETEROGENEITY mask (sim/bridge.py: cp_heterogeneity_neuron_mask, built from
        # regions with enable_heterogeneity=True; the cp.where in _apply_parameter_heterogeneity) gives ONLY
        # the td slice the het-ON graded band while nav/conv stay deterministic (cfg.enable_parameter_
        # heterogeneity stays False => the mask restricts the jitter to these neurons; everyone else keeps
        # their deterministic per-region presets => the nav/conv builds are byte-unchanged). See
        # research/findings/2026-06-18-merged-TD-cueshift-consolidation-BOUNDARY.md.
        td_regions = [BrainRegion(name=f"td_csc_{k}", n_neurons=int(td_csc_n_per), exc_fraction=1.0,
                                  internal_density=0.0, plastic_internal=False, izh_neuron_type=_RS,
                                  enable_nmda=False, enable_homeostasis=True, enable_heterogeneity=True)
                      for k in range(K)]
        td_regions += [
            BrainRegion(name="td_striosome", n_neurons=60, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_STRIATAL_MSN_D1.name,
                        syn_reversal_potential_i_override=-60.0, enable_nmda=False,
                        enable_homeostasis=True, enable_heterogeneity=True),
            BrainRegion(name="td_fs", n_neurons=24, exc_fraction=0.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_FS_CORTICAL_INTERNEURON.name,
                        syn_reversal_potential_i_override=-60.0, enable_nmda=False,
                        enable_homeostasis=True, enable_heterogeneity=True),
            BrainRegion(name="td_reward_us", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_RS, enable_nmda=False,
                        enable_homeostasis=True, enable_heterogeneity=True),
            BrainRegion(name="td_snc", n_neurons=30, exc_fraction=1.0, internal_density=0.0,
                        plastic_internal=False, izh_neuron_type=_NT.IZH2007_DOPAMINE.name,
                        syn_reversal_potential_i_override=-55.0, enable_nmda=False,
                        enable_homeostasis=True, enable_heterogeneity=True),
        ]
        # The locked production-recipe weights (the standalone GO config, finding §production recipe):
        # csc_to_strio 14.0 (the recipe TEXT says "--csc-to-strio-weight 6.0", but _run_td_csc_mode resolves
        # `args.csc_to_strio_weight if != 6.0 else 14.0` -> the documented 6.0 is the default-sentinel and the GO
        # run actually used 14.0; VERIFIED by re-deriving the arg logic), strio_to_snc(GABA_B -V level) 1.5,
        # reward_us_to_snc 8, strio_to_reward_us 10, to_fs 16, fs_to_strio 10. (The drives + the conductance-
        # derivative gain are the runner's, not wiring.)
        for k in range(K):
            td_pathways.append(RegionPathway(from_region=f"td_csc_{k}", to_region="td_striosome",
                                             density=0.6, weight_mean=float(td_csc_to_strio_weight), weight_jitter=0.5,
                                             plastic=True, plasticity_gate="td_value"))
            td_pathways.append(RegionPathway(from_region=f"td_csc_{k}", to_region="td_fs",
                                             density=0.6, weight_mean=float(td_to_fs_weight), weight_jitter=0.2,
                                             plastic=False))
        td_pathways += [
            RegionPathway(from_region="td_fs", to_region="td_striosome", density=0.7,
                          weight_mean=float(td_fs_to_strio_weight), weight_jitter=0.2, plastic=False),
            RegionPathway(from_region="td_reward_us", to_region="td_snc", density=0.6,
                          weight_mean=8.0, weight_jitter=0.2, plastic=False),
            RegionPathway(from_region="td_striosome", to_region="td_reward_us", density=0.6,
                          weight_mean=10.0, weight_jitter=0.2, plastic=False),
            RegionPathway(from_region="td_striosome", to_region="td_snc", density=0.6,
                          weight_mean=float(td_strio_to_snc_weight), weight_jitter=0.2, plastic=False,
                          receptor="gaba_b"),
        ]
    # HIPPOCAMPAL PATTERN-COMPLETION SLICE (co_resident_hippo_memory, additive default-off): the validated CA3 recurrent
    # autoassociator + dendritic-coincidence dAP-plateau completion (R-iii; _riii_ca3_coincidence_completion_derisk)
    # lifted as a co-resident slice on the merged one-brain. The DIRECT-drive formation bypasses ec/dg, so only ca3 (the
    # recurrent ATTRACTOR = the completion mechanism), ca1 (the Schaffer read-out), and ca3_pv_basket (the FS feedback
    # sparsifier that caps CA3 active-cell count so an attractor can form — the CYCLE-1072 fix) are added — zero name
    # collision with the nav cascade / parser / dlPFC / rf / perception / command_route / generalization / limbic / td /
    # drive slices. The ca3->ca3 recurrent pathway is flipped to a dendritic-COINCIDENCE detector (coincidence_detector=
    # True), so ONLY those synapses are routed through the two-compartment dAP plateau (cp_coincidence_synapse_mask) —
    # nav/conv synapses are untouched. internal_density on ca3 is 0.0 (the recurrent self-loop is the explicit SWR-gated
    # pathway, so plasticity can be gated ON during ripple bursts, OFF otherwise). Appended LAST so the nav/parser/dlPFC/
    # rf/cortex_it/gen/limbic/td/drive index bases are BYTE-UNCHANGED. Default False = byte-preserved (the cfg coincidence
    # / two-compartment-dAP fields below are ALSO guarded by this flag, so the default path keeps the CoreSimConfig
    # defaults enable_coincidence_detection=False / enable_two_compartment_dap=False).
    hippo_regions, hippo_pathways = [], []
    if co_resident_hippo_memory:
        from sim.enums import NeuronType as _NT
        import os as _os_bask
        _HIPPO_PYR = _NT.IZH2007_HIPPO_PYRAMIDAL.name
        _FS = _NT.IZH2007_FS_CORTICAL_INTERNEURON.name
        _n_basket = max(8, int(0.25 * int(hippo_n_ca3)))
        # E%-max feedback-inhibition set-point (CYCLE 1093 mechanism test, de Almeida-Idiart-Lisman 2009): the basket
        # can set the FRACTION of ca3 that fires robust to the excitatory regime -> a PORTABLE sparsifier that does NOT
        # depend on cell-type diversity. Default 5.0/120.0 = byte-identical; env knobs to test a stronger E%-max.
        _bask_drive_w = float(_os_bask.environ.get("CA3_BASKET_DRIVE_W", "5.0"))   # ca3->basket (E->I); higher -> basket fires proportionally
        _bask_inh_w = float(_os_bask.environ.get("CA3_BASKET_INH_W", "120.0"))     # basket->ca3 (I->E); the sparsity strength
        hippo_regions = [
            # NOTE (CYCLE 1093): the co-resident FORMATION+COMPLETION does NOT transplant here by config-matching. ~17
            # exhaustive single-variable tests (AUTONOMOUS_STATE CYCLE 1093) show the merged bridge's dynamical regime
            # has NO sparse-specific window for this ca3 completion: the DEFAULT saturates (all ca3 fire 30/60, within==
            # cross==118), and EVERY standalone-config match (num_traits=5, enable_nmda-on-ca3, enable_heterogeneity,
            # vt/adaptation diversity, per_type_stp=False, OU-on) collapses to a DEAD attractor (within~7, held=0). The
            # standalone completion's working regime rests on a fragile interaction (STP facilitation + the num_traits>1
            # type-MIX happening to assign FIRING cell-types to the assembly + its 6-region context) that is not
            # portable. So ca3/ca1 are kept MINIMAL (no NMDA/heterogeneity -- they didn't open a sparse regime). The
            # STRUCTURE co-resides (disjoint + byte-identical off); the co-resident FUNCTION is a characterized boundary
            # awaiting a research-gated mechanism (memory-as-co-state with its OWN dynamics phase -- the biological
            # theta-encoding / SWR-retrieval state-switch, hippocampus runs a distinct regime from neocortex). Byte-
            # preserved when co_resident_hippo_memory=False (these regions don't exist).
            BrainRegion(name="ca3", n_neurons=int(hippo_n_ca3), exc_fraction=0.85, internal_density=0.0,
                        exc_weight_mean=1.5, inh_weight_mean=2.0, weight_jitter=0.2, plastic_internal=True,
                        izh_neuron_type=_HIPPO_PYR),
            BrainRegion(name="ca1", n_neurons=int(hippo_n_ca1), exc_fraction=0.85, internal_density=0.05,
                        exc_weight_mean=0.3, inh_weight_mean=0.8, weight_jitter=0.2, plastic_internal=False,
                        izh_neuron_type=_HIPPO_PYR),
            BrainRegion(name="ca3_pv_basket", n_neurons=_n_basket, exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=_FS),
        ]
        hippo_pathways = [
            # the recurrent ATTRACTOR (the COMPLETION mechanism); coincidence_detector=True routes it through the dAP plateau.
            # density=0.5 matches the CI-guarded CYCLE-1076 completion GO (test_riii_emergent_ca3_completion: n_ca3=500,
            # n_assembly=12, k=20, ca3_recurrent_density=0.5); the CYCLE-1081 sparse-recurrent 0.2 was the big-assembly
            # (n_assembly=40) SWR-consolidation regime, NOT the small-sparse-assembly completion this slice reproduces.
            RegionPathway(from_region="ca3", to_region="ca3", density=0.5, weight_mean=6.0, weight_jitter=0.2,
                          plastic=True, plasticity_gate="ca3_swr_burst", coincidence_detector=True),
            RegionPathway(from_region="ca3", to_region="ca1", density=0.30, weight_mean=4.0, weight_jitter=0.2,
                          plastic=True, plasticity_gate="ca3_to_ca1"),                      # Schaffer read-out
            RegionPathway(from_region="ca3", to_region="ca3_pv_basket", density=0.40, weight_mean=_bask_drive_w,
                          weight_jitter=0.2, plastic=False),                                # E->I (feedback drive)
            RegionPathway(from_region="ca3_pv_basket", to_region="ca3", density=1.0, weight_mean=_bask_inh_w,
                          weight_jitter=0.2, plastic=False),                                # I->E (feedback sparsifier / E%-max set-point)
        ]

    union_regions = (list(nav_regions) + list(parser_regions) + list(dlpfc_regions)
                     + list(rf_regions) + list(perception_regions) + list(command_route_regions)
                     + list(generalization_regions)
                     + list(limbic_regions) + list(td_regions) + list(drive_regions)
                     + list(speech_regions)
                     + list(hippo_regions))
    union_pathways = (list(nav_pathways) + list(parser_pathways) + list(command_route_pathways)
                      + list(generalization_pathways) + list(limbic_pathways)
                      + list(td_pathways) + list(drive_pathways) + list(speech_pathways)
                      + list(hippo_pathways))   # dlPFC loop is hand-built, NOT a pathway

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
    import os as _os_nt
    if _os_nt.environ.get("MERGED_NUM_TRAITS") is not None:   # CYCLE 1093 diagnostic ONLY: confirm the ca3 completion
        cfg.num_traits = int(_os_nt.environ["MERGED_NUM_TRAITS"])   # needs the calibrated num_traits=5 diversity (unset=1, byte-identical)
    if _os_nt.environ.get("MERGED_NO_STP") is not None:      # CYCLE 1093 diagnostic: standalone has per_type_stp=False;
        cfg.enable_per_type_stp = False                       # STP facilitation on ca3->ca3 could amplify the recurrent -> broad firing
    if _os_nt.environ.get("MERGED_OU_ON") is not None:       # CYCLE 1093 diagnostic: standalone has OU on
        cfg.enable_ou_process = True
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
    # (Also applies to co_resident_nav_critic — the FULL nav critic — with the DA source re-pointed to the nav SNc
    # `snc` instead of the minimal organ's `limbic_snc`, per the CYCLE-209 integration plan.)
    if co_resident_limbic or co_resident_nav_critic:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 150.0
        cfg.gabab_propagation_strength = 0.22
        cfg.gabab_conductance_max = 0.0
        _da_source = ["snc"] if co_resident_nav_critic else ["limbic_snc"]
        from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=8.0,
                                             threshold=0.0, window_ms=200.0, source_regions=_da_source)])]

    # A-CSC TD CUE-SHIFT config (co_resident_td_cueshift): (a) the GABA_B/GIRK conductance (already-shipped edit; ONLY
    # td_striosome->td_snc is tagged receptor="gaba_b") with the SHORT per-tap tau (40ms, so g_gabab tracks each tap's
    # value); (b) the B-2 PROTECTED conductance-derivative edit (enable_td_value_derivative; byte-identical when OFF,
    # COMBO e728d7f1...) at slow-EMA tau 130ms — the bootstrap +dV/dt source; (c) the SHORT eligibility tau 40ms so TD
    # back-propagates one tap per trial (the CoreSimConfig default 1000ms smears credit across the chain -> no
    # migration); (d) the `dopamine` signed-firing modulator over [td_snc] = the SHARED DA broadcast (threshold 0 =>
    # neutral-at-rest, so a quiescent td_snc during the parser train pass + conversational ops gives da_signal=0 ->
    # plasticity-rate ~1.0 -> it CANNOT suppress parser/conversational/nav plasticity). All ONLY when
    # co_resident_td_cueshift -> default-off byte-preserved (the merge otherwise has enable_gabab=False, eligibility
    # tau 1000ms, enable_td_value_derivative False, zero neuromodulators).
    if co_resident_td_cueshift:
        cfg.enable_gabab = True
        cfg.gabab_reversal_potential = -90.0
        cfg.gabab_tau_decay = 40.0                     # SHORT: -V tracks the per-tap value (the standalone csc_gabab_tau_decay)
        cfg.gabab_propagation_strength = float(td_gabab_prop)         # the standalone csc default 0.105 (tunable co-resident)
        # GIRK saturation cap (the owner-approved guardrail): bound g_gabab so a HOT critic cannot FULLY CLAMP td_snc
        # to silence (the B-2 tonic-death wall). The merged config pins stdp_w_max=400 (the 5a conversational-weight
        # clip mitigation), which REMOVES the per-tap weight cap (40) the standalone CSC bridge used -> the critic can
        # run away, its GABA_B -V saturates, and td_snc dies. The GIRK cap is the co-resident fix: a bounded -V is a
        # GRADED shift at any critic rate, so the reward burst shrinks WITHOUT killing the tonic. 0 = no cap (off).
        cfg.gabab_conductance_max = float(td_gabab_conductance_max)
        cfg.enable_td_value_derivative = True          # the B-2 protected edit (byte-identical when OFF)
        cfg.td_slow_tau_ms = float(td_slow_tau_ms)     # the locked --csc-td-slow-tau-ms (130; tunable co-resident)
        cfg.td_derivative_gain = float(td_derivative_gain)   # the locked --csc-td-derivative-gain (1.0; raise co-resident to lift the cue burst)
        cfg.reward_eligibility_tau_ms = 40.0           # SHORT (tap-local credit; the locked --csc-eligibility-tau-ms)
        from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [NeuromodulatorConfig(
            name="dopamine", baseline=0.5, decay_tau_ms=200.0, concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="plasticity_rate", scope="all", sensitivity=+1.0)],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed", sensitivity=8.0,
                                             threshold=0.30, window_ms=200.0, source_regions=["td_snc"])])]

    # HUNGER -> DA link (drive_to_da, additive default-off; Tier-3 Option 3 "cross-modal one animal"): the shared
    # spiking hunger drive raises the shared `dopamine` broadcast, so a HUNGRY brain's DA rises and the moat-safe
    # `_da_confidence_gate` TIGHTENS conversational abstention -- the SAME limbic drive touches BOTH the acting (nav)
    # and conversing (composer) halves. Appends a `from_region_firing` rule reading `drive_agrp` firing to the
    # `dopamine` modulator's production_rules; a modulator's rules are SUMMED per step (sim/neuromodulators.py:264),
    # so DA = the SNc term (unchanged) + a hunger term. `from_region_firing` is a shipped rule type (NOT the reserved
    # from_novelty stub); this is a RUNNER-layer edit (NO `sim/` edit). Biology: O.10 incentive motivation
    # (deprivation amplifies reward VALUE; Berridge/Toates). MOAT-SAFE: DA only ever RAISES the gate above its g0
    # floor (da_to_gate clamps), so hunger can ONLY TIGHTEN the no-confab moat, never loosen it. Requires a `dopamine`
    # modulator (a limbic/critic/td slice) + the drive slice; default False = byte-unchanged.
    if drive_to_da and co_resident_drive and getattr(cfg, "neuromodulators", None):
        from sim.neuromodulators import ProductionRule as _HungerPR
        _da_cfg = next((m for m in cfg.neuromodulators if m.name == "dopamine"), None)
        assert _da_cfg is not None, \
            "drive_to_da requires a `dopamine` modulator (co_resident_nav_critic / co_resident_limbic / td_cueshift)"
        _da_cfg.production_rules.append(
            _HungerPR(rule_type="from_region_firing", sensitivity=float(drive_da_sensitivity),
                      threshold=0.0, window_ms=200.0, source_regions=["drive_agrp"]))

    # HIPPOCAMPAL COINCIDENCE-PLATEAU config (co_resident_hippo_memory): route the ca3->ca3 recurrent synapses (the ONLY
    # pathway with coincidence_detector=True) through the two-compartment dendritic-dAP plateau, per _riii_ca3_
    # coincidence_completion_derisk._build. EVERY field is guarded by the flag -> default-off keeps the CoreSimConfig
    # defaults (enable_coincidence_detection=False / enable_two_compartment_dap=False / coincidence_weighted_drive=False /
    # k_thr=6.0 / strength=80.0 / apical_R=0.15) so the merged build is BYTE-IDENTICAL when off. Because only ca3->ca3
    # carries the coincidence mask (cp_coincidence_synapse_mask), these GLOBAL cfg flags act ONLY on those synapses ->
    # nav/conv synapses are unaffected. k_thresh=20 / plateau_strength=300 / apical_R=50 are the R-iii operating point
    # (the controller multi-seed-tunes the formation+completion). The merged cfg already has enable_nmda=True (the plateau
    # needs NMDA) + stdp_w_max=400 / hebbian_max_weight=400 (>= the ca3 design weights 6/120, so no spurious clipping).
    if co_resident_hippo_memory:
        cfg.enable_coincidence_detection = True
        cfg.coincidence_weighted_drive = True          # WEIGHTED-DRIVE plateau read (c_drive = sum of effective weight over coincident inputs)
        cfg.coincidence_k_threshold = float(hippo_k_thresh)  # scaled UP vs the CYCLE-1076 k=20: the merged cfg's hebbian_max_weight=400 (the nav/conv clip mitigation) over-potentiates the ca3->ca3 vs the completion's tuned-at-120 ceiling, so the within-drive is ~3.3x -> k must scale ~proportionally (default 66 ~ 20*400/120) or the completion over-fires (non-specific)
        cfg.coincidence_plateau_strength = 300.0
        cfg.enable_two_compartment_dap = True
        cfg.apical_R = 50.0                            # apical params exactly as _build does (apical_g_couple stays at its default)
        cfg.hebbian_rate_window = True                 # ALLOCATE the co-activity trace at BUILD (cp_hebb_coactivity_trace is init-time-gated on this flag); the rate-window Hebbian only RUNS when enable_hebbian_learning is toggled True during the scoped ca3->ca3 formation (the validate runner's mid-run switch), so nav/conv (enable_hebbian_learning=False) are UNAFFECTED. Default-off keeps this False -> byte-identical.
        cfg.hebbian_coactivity_thresh = 0.001

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
    if co_resident_grounded_speech:
        bridge.set_plasticity_gate(SPEECH_GROUNDING_GATE, 0.0)

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

    # 5c) COMMAND-ROUTE train pass (route A: language->action). GROW the LEARNED `language_input -> cortex_X`
    #     direction-selectivity by brain-based co-firing (Pulvermüller action-word somatotopy), then FREEZE the route
    #     and hold the command_route transmission gate CLOSED at rest. Ported from spoken_instruction_nav._train_learned_route
    #     (the GO standalone) — isolated to the route via the global plasticity gain (0 everywhere, 1 on the route gate)
    #     so the nav cascade's own plastic pathways do not drift. Runs after the parser pass; nav stays plastic for the
    #     episode (gain restored to 1, the route gate frozen to 0). The transmission gate is registered by the framework
    #     from the route pathway's transmission_gate field; assert it exists before training.
    if co_resident_command_route:
        assert COMMAND_GATE in bridge._transmission_gate_to_synapses, \
            f"FAIL: '{COMMAND_GATE}' transmission gate not registered (known: " \
            f"{list(bridge._transmission_gate_to_synapses.keys())})"
        _train_command_route_on_merged(bridge, int(seed))

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
        handles["rf_size"] = int(_rf_n)        # the ACTUAL rf region size (7*rf_D, or onebrain_rf_size when given)
        handles["rf_D"] = int(rf_D)
    if co_resident_perception:
        # the cortex_it perception slice indices (for perceive_and_ground: the live-rate read + grounded code).
        handles["cortex_it_indices"] = np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)
    if co_resident_command_route:
        # the command-route handles (for MergedNavConvAgent.command_move): language_input indices, the parser
        # action-role block (the gate's control ensemble), and the cortex_X / readout / tonic indices.
        from research.runners.g11_bg_runner import ACTION_NAMES as _CR_ACTIONS
        handles["lang_indices"] = xp.asarray(np.asarray(list(rm.indices("language_input")), dtype=np.int64))
        handles["action_block_idx"] = role_arr["action"]
        handles["cmd_cortex_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64))
                                     for a in _CR_ACTIONS}
        _rn = set(rm.region_indices_dict())
        if all(f"sel_{a}" in _rn for a in _CR_ACTIONS):
            handles["cmd_readout_region"] = "sel"
            handles["cmd_readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                                          for a in _CR_ACTIONS}
        else:
            handles["cmd_readout_region"] = "motor"
            handles["cmd_readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"motor_{a}")), dtype=np.int64))
                                          for a in _CR_ACTIONS}

        def _cr_ridx(name):
            return xp.asarray(np.asarray(list(rm.indices(name)), dtype=np.int64)) if name in _rn else None
        handles["cmd_cascade_tonic"] = []
        for a in _CR_ACTIONS:
            for name, pa in ((f"gpe_{a}", 150.0), (f"gpe_arky_{a}", 120.0), (f"gpi_{a}", 110.0),
                             (f"thal_{a}", 300.0)):
                ii = _cr_ridx(name)
                if ii is not None:
                    handles["cmd_cascade_tonic"].append((ii, float(pa)))
        for name, pa in (("stn", 150.0), ("snc", 150.0)):
            ii = _cr_ridx(name)
            if ii is not None:
                handles["cmd_cascade_tonic"].append((ii, float(pa)))
    if co_resident_generalization:
        handles["gen"] = dict(gen_handles, **gen_extra)
    if co_resident_limbic:
        # The limbic-slice base indices (for the on-merge RPE-battery validation + the later increments that
        # drive limbic_cue/limbic_reward_us and read limbic_snc/limbic_striosome).
        handles["limbic"] = {n: {"base": int(rm.indices(n)[0]), "size": len(list(rm.indices(n)))}
                             for n in ("limbic_cue", "limbic_striosome", "limbic_reward_us", "limbic_snc")}
    if co_resident_drive:
        # The drive-slice base indices (for the on-merge living loop: drive drive_agrp/drive_pomc ∝ the body
        # deficit/surplus each step, read drive_agrp firing as the spiking hunger that gates the reward).
        handles["drive"] = {n: {"base": int(rm.indices(n)[0]), "size": len(list(rm.indices(n)))}
                            for n in ("drive_agrp", "drive_pomc")}
    if co_resident_grounded_speech:
        handles["grounded_speech"] = {
            n: np.asarray(list(rm.indices(n)), dtype=np.int64)
            for n in (SPEECH_FOOD_CUE, SPEECH_REQUEST, SPEECH_SILENCE, SPEECH_WTA_FS)
        }
    if co_resident_td_cueshift:
        # The TD-slice base indices (for the on-merge A-CSC cue-shift battery: drive td_csc_k / td_reward_us, read
        # td_snc / td_striosome per bin for the migration time-course).
        _td_names = tuple(f"td_csc_{k}" for k in range(int(td_csc_n))) + (
            "td_striosome", "td_fs", "td_reward_us", "td_snc")
        handles["td"] = {n: {"base": int(rm.indices(n)[0]), "size": len(list(rm.indices(n)))}
                         for n in _td_names}
        handles["td_csc_n"] = int(td_csc_n)
        # The per-tap weight cap the runner enforces on the td_value-gated synapses (the standalone CSC bridge used
        # stdp_w_max=40 to keep the critic SPARSE; the merged bridge pins the GLOBAL stdp_w_max=400 for the
        # conversational weights, so the runner re-clips ONLY the td_value synapses to this LOCAL cap per trial — a
        # weight-BOUND, not a host computation of value/reward/delta, so the cue-shift stays 100% neural). 0 = no clip.
        handles["td_stdp_w_max"] = float(td_stdp_w_max)
    return bridge, handles


# ── the EPISODE-path conv finalization (nav gate (a): nav episode runs on the merged bridge) ─────────────────
def conv_extra_regions_pathways(vocab=None, co_resident_rf=False, rf_D=128,
                                co_resident_drive=False, drive_n_pool=60):
    """The conversational regions/pathways to APPEND to the navigation lists for the episode-path merge: the
    parser (parse_conj 6, parse_role 3*PARSER_R) + the dlPFC regions (cortex_ctx, dlpfc_wm, both enable_nmda).
    For the NAV GATE the dlPFC regions are present but EDGELESS (the dlpfc_loop is only for `elaborate`, not
    needed for nav-not-regressed), so they are silent during the nav episode. Returns (extra_regions,
    extra_pathways) for `run_moving_goal_episode(extra_regions=, extra_pathways=)`.

    co_resident_rf (STEP 2b): also append the `rf` composer region (7*rf_D neurons, no pathways, NMDA-off) so the
    nav-not-regressed gate can be re-run with the rf slice present. The rf region has NO cp_connections out-edges
    into navigation (the Task-1 anti-cheat) and is idle during the nav episode (no composer ops run mid-episode),
    so it is provably nav-inert — this gate just confirms that empirically.

    co_resident_drive (Tier-3 SPIKING living loop): also append the 2-pool SPIKING interoceptive drive slice
    (`drive_agrp`=hunger / `drive_pomc`=satiety; hypothalamic AgRP/POMC, catalog O.05/O.06; mechanism GO
    2026-06-17-homeostatic-spiking-drive-mechanism-GO.md) so it is genuinely CO-RESIDENT on the SAME bridge the
    agent navigates. This episode-path helper gives the pools zero out-edges (they are driven by interoceptive current
    and read by firing rate), so the
    slice is maximally nav-inert (like rf). enable_homeostasis=True PER-REGION = the merged-config operating-point
    fix (the limbic-core-lift lesson): the already-shipped per-region homeostasis mask (sim/bridge.py:1227-1245/
    :6320) gives ONLY the drive slice the low spike threshold while nav/conv stay at vpeak (byte-unchanged), and
    the synaptic-scaling clip (gated by the SEPARATE enable_synaptic_scaling, OFF here) never runs. Default
    False = byte-preserved (the nav gate's extra-region list is unchanged)."""
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
    drive_regions = []
    if co_resident_drive:
        from sim.enums import NeuronType as _NT
        _RS = _NT.IZH2007_RS_CORTICAL_PYRAMIDAL.name
        drive_regions = [
            BrainRegion(name="drive_agrp", n_neurons=int(drive_n_pool), exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
            BrainRegion(name="drive_pomc", n_neurons=int(drive_n_pool), exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                        izh_neuron_type=_RS, enable_nmda=False, enable_homeostasis=True),
        ]
    return (list(parser_regions) + list(dlpfc_regions) + list(rf_regions) + list(drive_regions),
            list(parser_pathways))


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


# ── CONSOLIDATION (option A, Probe 2/3): the co-resident `OneBrainComposer` on the merged `rf` slice ──────────────
# The persistent-loop composer (the whole who/what pipeline: synaptic multi-fact store + spiking cleanup + the
# no-confab moat + the encoding_gain_fn WRITE-side hook) run as a SLICE of the merged bridge -- the same index-shift
# port `MergedRFComposer` performs for `RFPhasorComposer`, applied ONE LEVEL UP. Byte-identical to a standalone
# `OneBrainComposer` (Probe 1 GO, research/findings/raw/_consolidation_probe1_byteident.json: full who/what matrix +
# every is-None/unknown abstention identical to atol 1e-9, moat preserved, nav slice byte-isolated). This is the
# importable home for the class first proven in research/runners/_consolidation_probe1_byteident.py (the probe now
# imports it from here). NO `sim/` edit (reuse-by-import: OneBrainComposer + BridgeParser(index_offset=) + the masked
# rf_kick + the _rf_reset_mask co-residence guarantee). The scoping rationale: §1.2/§1.3 of
# research/findings/raw/_consolidation_onebrain_limbic_scoping.md.
class CoResidentOneBrainComposer(OneBrainComposer):
    """`OneBrainComposer` whose RF/store/cleanup ops run on a SLICE of a shared (merged) bridge instead of on its own
    private bridge -- the consolidation port.

    `OneBrainComposer.__init__` builds the layout from `self.P` and a PRIVATE bridge at `[0:n_total]`. This subclass
    redirects the bridge handle to `merged_bridge` and REBASES every absolute index by `rf_base`:
      * the parser slice moves to `[rf_base : rf_base + P_local]` (BridgeParser(index_offset=rf_base));
      * every base (P, store_base, q_base, c_base, bat_q_base, bat_c_base) is shifted += rf_base, so every downstream
        `self.<base> + i*...` lands inside the slice;
      * `self.n_total` is REDEFINED to the merged bridge's N (so every `np.zeros(self.n_total)` kick + every
        `_build_complex_csr(self.n_total, ...)` is full-N, the merged-bridge size);
      * `self.rf_mask` AND `self._rf_reset_mask` cover exactly the composer's layout span on the merged bridge
        ([rf_base : rf_base + span]); the per-op `v/u <- 0` reset is thus restricted to the rf slice, so a composer op
        leaves a co-resident Izhikevich (nav) slice's v/u BYTE-untouched (the masked-rf-kick 5b coexistence guarantee).

    Because the resonate loop is pure complex dynamics (no OU) + masked + the CSR is block-local, the RF ops reproduce
    the standalone composer's results bit-for-bit (Probe 1). On the merged agent the parser is built on the slice but
    comprehension is driven via `store()` (the agent's merged `parse_conj`/`parse_role` slices supply the roles --
    LAYOUT decision 2b), so the composer's own parser is idle (it is constructed for layout completeness only)."""

    def __init__(self, merged_bridge, rf_base, build_parser=True, **kwargs):
        # Reproduce OneBrainComposer.__init__'s feature/layout computation WITHOUT building the private bridge/parser
        # (which __init__ does at one_brain_composer.py:282-283). We mirror the relevant body, then rebase. Keeping
        # this in the subclass leaves one_brain_composer.py byte-untouched (a NEW opt-in alongside MergedRFComposer).
        #
        # build_parser (DEFAULT True = the Probe-1 / standalone-parity path, byte-unchanged): whether to construct the
        # idle layout-only `BridgeParser(shared_bridge=merged_bridge)`. On a FRAMEWORK-WIRED merged bridge (the
        # MergedNavConvAgent path) that parser is REDUNDANT — comprehension goes through the agent's
        # `_MergedParserAdapter` (nav_conv_merged_bridge.py:1887, reading the framework `parse_conj` slice), so the
        # composer's own parser is NEVER used (LAYOUT decision 2b). Worse, `BridgeParser(shared_bridge=...)` calls
        # `merge_population_into_shared_bridge`, which re-injects from the (empty on a framework-wired bridge)
        # `_unified_wiring_plan` -> it WIPES the framework wiring (the nav cascade) + the COMMAND_GATE
        # transmission-gate registration -> the MergedNavConvAgent COMMAND_GATE anti-cheat assert fails at
        # construction (the traced close-out bug, _closure1_optionA_gate3_flip.json). So the merged agent passes
        # build_parser=False: skip the destructive merge, set self.parser=None. The RF ops are reset-isolated from the
        # parser (every op `_zero_rf_v_u()`-resets the rf slice before kicking, and the complex `cp_rf_w_*` synapses are
        # array-disjoint from `cp_connections`), so dropping the idle parser leaves the composer's RF numerical OUTPUT
        # byte-identical to the standalone/Probe-1 oracle (verified: _closure1_buildfix_answer_identity.py + Probe-1
        # re-run, which uses the default build_parser=True and is therefore UNCHANGED).
        from research.runners.brain_conversational_agent import BridgeParser
        seed = int(kwargs.get("seed", 42)); D = int(kwargs.get("D", 128))
        vocab = kwargs.get("vocab", None); period = int(kwargs.get("period", 200))
        k_max = int(kwargs.get("k_max", 32))
        grounded_codes = kwargs.get("grounded_codes", None)
        # --- the flag fields (defaults match OneBrainComposer.__init__ signature) ---
        self.seed = seed; self.D = D; self.period = period
        self.persistent_store = bool(kwargs.get("persistent_store", False))
        self._persistent_dirty = True
        self.trace = bool(kwargs.get("trace", False)); self.last_trace = None
        self.integrated_loop = bool(kwargs.get("integrated_loop", False))
        self.persistent_loop = bool(kwargs.get("persistent_loop", False))
        self.sequencer_match_thresh = float(kwargs.get("sequencer_match_thresh", 0.06))
        self.sequencer_gain = float(kwargs.get("sequencer_gain", 0.11))
        self.sequencer_sigma = float(kwargs.get("sequencer_sigma", 1.0))
        self.sequencer_input_gain = float(kwargs.get("sequencer_input_gain", 1.0))
        self._seq = None; self._seq_score = None; self._seq_K = None; self._seq_drives = None; self._seq_dirty = True
        self.enable_seq_vocab_shrink = bool(kwargs.get("enable_seq_vocab_shrink", True))
        self._seq_mapA = None; self._seq_mapX = None; self._seq_cuevocab_sig = None; self._seq_cleanup_conns_cache = None
        self.local_reciprocal_unbind = bool(kwargs.get("local_reciprocal_unbind", True))
        self.encoding_gain_fn = kwargs.get("encoding_gain_fn", None)
        self.confidence_gate = float(kwargs.get("confidence_gate", 0.0))
        self.enable_spiking_cleanup = bool(kwargs.get("enable_spiking_cleanup", True))
        self.enable_multiframe = bool(kwargs.get("enable_multiframe", False)); self._frame_parser = None
        self.enable_batched = bool(kwargs.get("enable_batched", True))
        self.enable_rf_cudagraph = bool(kwargs.get("enable_rf_cudagraph", False))   # numpy path => no megakernel
        self.enable_csr_cache = bool(kwargs.get("enable_csr_cache", True))
        self._csr_cache = {}; self._store_csr = None; self._store_dirty = True
        self.comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, period=period, grounded_codes=grounded_codes,
                                     local_reciprocal_unbind=self.local_reciprocal_unbind)
        self.words = list(self.comp.words); self.V = len(self.words)
        self.R = 40; self.P = 6 + 3 * self.R; self.k_max = int(k_max)
        self.pol_words = list(self.comp.pol_words); self.NP = len(self.pol_words)
        self.enable_attributed = bool(kwargs.get("enable_attributed", False))
        self.bind_roles = (["agent", "action", "patient", "attribute", "polarity"] if self.enable_attributed
                           else ["agent", "action", "patient", "polarity"])
        self.n_roles = len(self.bind_roles)
        self.main_roles = [r for r in self.bind_roles if r != "polarity"]; self.n_main = len(self.main_roles)
        # the standalone (pre-offset) layout:
        self.store_base = self.P + (2 * self.n_roles + 1) * D
        self.block = 1 + D
        self.q_base = self.store_base + self.k_max * self.block
        self.c_base = self.q_base + self.n_roles * D
        self.cb = self.n_main * self.V + self.NP
        self.bat_q_base = self.c_base + self.cb
        self.bat_c_base = self.bat_q_base + self.k_max * self.n_roles * D
        layout_span = self.bat_c_base + self.k_max * self.cb     # == standalone n_total = the slice span on the merged bridge

        # --- THE REBASE: shift every base by rf_base; n_total becomes the merged bridge N; rf_mask = the slice. ---
        self._rf_base = int(rf_base)
        N = int(merged_bridge.core_config.num_neurons)
        if self._rf_base + layout_span > N:
            raise ValueError(f"co-resident OneBrainComposer needs {layout_span} rf neurons at base {self._rf_base} "
                             f"but the merged bridge has only {N} (raise the rf region size via onebrain_rf_size)")
        self.P += self._rf_base
        self.store_base += self._rf_base
        self.q_base += self._rf_base
        self.c_base += self._rf_base
        self.bat_q_base += self._rf_base
        self.bat_c_base += self._rf_base
        self.n_total = N                                         # array-sizing is full merged-bridge N
        self.b = merged_bridge
        # the parser slice lives at [rf_base : rf_base + P_local] on the merged bridge (same relative wiring). On the
        # merged agent it is idle (comprehension goes through the merged parser -> store()); built for layout completeness
        # on the Probe-1 / standalone-parity path (build_parser=True). On a framework-wired merged bridge the agent
        # passes build_parser=False (the parser would be redundant AND its shared-bridge merge would wipe the framework
        # wiring + COMMAND_GATE -- see the __init__ docstring); the RF ops are reset-isolated from the parser so the
        # composer's output is byte-identical with self.parser=None.
        if build_parser:
            self.parser = BridgeParser(seed=seed, R=self.R, shared_bridge=self.b, index_offset=self._rf_base)
        else:
            self.parser = None
        self.rf_mask = np.zeros(self.n_total, dtype=bool)
        self.rf_mask[self._rf_base:self._rf_base + layout_span] = True
        # the per-op `v/u <- 0` reset is restricted to the rf slice (so a co-resident Izhikevich/nav slice's v/u is
        # byte-untouched across a composer op) -- the masked-rf-kick co-residence guarantee.
        self._rf_reset_mask = self.rf_mask
        self.kb = []
        self.store_conns = []
        self._word_index = {w: i for i, w in enumerate(self.words)}
        self._layout_span = int(layout_span)
        self._merged = merged_bridge                             # parity with MergedRFComposer's anti-cheat attribute

    @staticmethod
    def n_total_for(D=128, vocab=None, k_max=32, enable_attributed=False):
        """The full layout span (= standalone `OneBrainComposer.n_total`) the composer needs on the merged `rf` slice,
        as a function of (D, |vocab|, k_max, attribute role). Used by `build_merged_nav_conv_bridge` to size the merged
        `rf` region. Mirrors OneBrainComposer.__init__'s layout math EXACTLY (a drift here would over/under-size the
        slice and Probe-1 byte-identity would fail loudly via the rebase bounds check)."""
        D = int(D); k_max = int(k_max)
        # the vocab the composer actually uses == RFPhasorComposer(vocab=...).words
        comp = RFPhasorComposer(seed=42, D=D, vocab=vocab)
        V = len(comp.words); NP = len(comp.pol_words)
        P = 6 + 3 * 40
        n_roles = 5 if enable_attributed else 4
        n_main = n_roles - 1
        store_base = P + (2 * n_roles + 1) * D
        block = 1 + D
        q_base = store_base + k_max * block
        cb = n_main * V + NP
        c_base = q_base + n_roles * D
        bat_q_base = c_base + cb
        bat_c_base = bat_q_base + k_max * n_roles * D
        return int(bat_c_base + k_max * cb)


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

    def __init__(self, seed=42, vocab=None, co_resident_composer=False, co_resident_composer_kind="onebrain",
                 co_resident_limbic=False,
                 co_resident_nav_critic=None, nav_critic_spiking_sc=False,
                 nav_critic_place_selforg=None, nav_critic_grid_frontend=None,
                 co_resident_td_cueshift=False,
                 co_resident_perception=False, co_resident_generalization=False,
                 perception_grounding="gen_spikes", perception_device_resident=False,
                 co_resident_drive=False, drive_n_pool=60, drive_to_da=False, drive_da_sensitivity=8.0,
                 co_resident_grounded_speech=False, speech_n_acc=40, speech_n_fs=20,
                 speech_drive_weight=8.0, speech_cue_weight=60.0,
                 co_resident_command_route=None,
                 enable_da_salience_gate=True, da_gate_g0=0.06, da_gate_k=2.0, da_gate_cap=0.25,
                 enable_da_encoding_gain=True, da_encoding_k=2.0,
                 da_encoding_g_min=0.5, da_encoding_g_max=3.0,
                 enable_da_recall_vigor=False, da_recall_beta=8.0, da_recall_value_default=1.0,
                 onebrain_k_max=32):
        """Build the merged nav+parser+dlPFC bridge + the composer (same seed + vocab). The composer's vocab is the
        merged dlPFC vocab (the sorted probe vocab) so the dialogue-planning assemblies and the fact-memory codebook
        share one word set.

        co_resident_composer (STEP 2b): when True, the fact-binding composer runs CO-RESIDENT on the merged bridge's
        own `rf` slice (the strict single-instance unification, via the owner-approved sliced `rf_kick`); when False
        (STEP 2a default) it runs on its own separate per-op bridges.

        enable_da_salience_gate (TRUE ONE BRAIN roadmap #6 — the spiking-DA -> composer precision gate, de-risked
        6/6 GO: 2026-06-18-DA-composer-precision-derisk-GO.md): when True, before each conversational READ op the
        agent reads the SHARED spiking-SNc dopamine off ITS OWN merged bridge
        (`self._merged_bridge.neuromodulator_manager.get_concentration("dopamine")` — the same DA the BG actor /
        limbic core learns from, produced by the spiking SNc on the merged bridge via `co_resident_limbic` /
        `co_resident_nav_critic` / `co_resident_td_cueshift`), maps it CLAMPED-TO-SHARPEN onto the composer's
        cue-role CONFIDENCE GATE (`g_eff = clip(g0, g_cap, g0 + k*(DA - DA_baseline))`, the de-risk's `da_to_gate`),
        and ABSTAINS on a noise-dominated cue read (`min(margin(agent), margin(action)) < g_eff`). A higher gate =>
        STRICTER abstention, so this can ONLY TIGHTEN the no-confab moat, never loosen it (moat-safe by
        construction).

        PRODUCTION DEFAULT = ON (2026-06-24, burndown I-4-a — the merged DEFAULT now INTERACTS, closing the
        co-located-not-interacting gap I-4 / functional one-brain scoping I-7-a). The DA SOURCE is already
        default-ON (`co_resident_nav_critic` resolves True by default => the shared spiking SNc + `dopamine`
        modulator on the merged bridge), so the limbic core now reaches the conversational cortex by default
        (the read-side salience gate). MOAT-SAFE + NAV-NEUTRAL by construction: (i) a higher gate only TIGHTENS
        abstention (never loosens — the 6/6-GO `da_to_gate` clamps at the `g0` floor, so the moat can only get
        STRICTER); (ii) the gate is a composer-read-side scalar that reads DA and NEVER writes any nav drive, and
        the place/critic arrays are array-disjoint from the composer's complex `cp_rf_w_*` synapses, so the
        navigation score is unaffected. BYTE-IDENTICAL AT REST: with no salient/nav-driven turn the SNc is tonic
        => DA == baseline => `g_eff = g0` => the gate floor => the read path is unchanged (verified on the merged
        bridge: DA-at-rest 0.5 == baseline, `g_eff_rest == g0`, conversational reads identical OFF vs ON, moat
        0-FA); the interaction ENGAGES (g_eff rises above g0) only on a SALIENT/high-DA turn. Set False for the
        legacy byte-identical-everywhere read path (the revertible escape). NO `sim/` edit (composer-runner-layer
        read of a spike-derived scalar; the gate reuses the composer's own cleanup primitives + `OneBrainComposer._margin`).
        `da_gate_g0`/`da_gate_k`/`da_gate_cap` = the de-risk's validated 0.06/2.0/0.25 (the inverted-U ceiling).
        The hook is meaningful only when a `dopamine` modulator is present on the merged bridge (a limbic/critic/TD
        slice is co-resident — the production default); without one DA reads as baseline => g_eff = g0 => no-op."""
        self.seed = int(seed)
        self.co_resident_composer = bool(co_resident_composer)
        # --- CONSOLIDATION (option A, Closure 1): which co-resident composer runs on the merged `rf` slice. ---
        # "onebrain" (PRODUCTION DEFAULT, 2026-06-25, Closure 1 flip via owner-default (b)): CoResidentOneBrainComposer
        #            -- the persistent-loop composer (synaptic multi-fact store + spiking cleanup + the no-confab moat) on
        #            the merged `rf` slice (Probe-1 byte-identical to a standalone OneBrainComposer). This ACTIVATES the
        #            limbic encoding-gain WRITE side (OneBrainComposer._write_block scales the stored complex-weight
        #            MAGNITUDE by encoding_gain_fn() -> LOAD-BEARING under read damage), and routes the DA salience READ
        #            gate through the composer's NATIVE `confidence_gate` (the same `min(margin(agent),margin(action)) < g`
        #            abstention, moat-safe by construction). The merged conversation is now FULLY-SPIKING-ONE-BRAIN.
        # "rf"       (TEST ORACLE / numpy-CPU path): MergedRFComposer (an RFPhasorComposer storing PHASES in numpy `kb`) --
        #            the byte-identity ORACLE; the encoding_gain_fn WRITE-side hook is wired+DA-correct but INERT here
        #            (phases are magnitude-invariant), and the DA salience READ gate uses the composer-external `_gated_out`.
        #            Retained as the oracle the onebrain path is validated against (`test_*_merged_agent[rf]` parametrization).
        # FLIP RATIONALE (Closure 1, owner-default (b)): the literal cross-N byte-identity nav gate is UNREACHABLE on the
        # GPU without a sim/ FP rewrite (the residual is GPU floating-point NON-ASSOCIATIVITY: the rf-region SIZE changes
        # the total N -> the episode-loop readout/render reduction tiles differently -> ~1e-5 FP32 round-off in the SAME
        # nav-slice neurons, chaotically amplified by the spiking-WTA decision over 200 steps). That is NOT a composer
        # functional effect: the `rf` slice has ZERO cp_connections out-edges into nav + every composer op _zero_rf_v_u()-
        # resets it, so it cannot leak into nav (Probe-1 op-isolation, atol exact; gate 3A same-N functional neutrality,
        # Δ=0). The documented STANDALONE nav benchmark (--readout-source motor, the CLI default) NEVER builds the
        # conversational rf region at all -> totally unaffected by the merged composer kind. So the flip is on the proven
        # FUNCTIONAL-neutrality argument, NOT cross-N byte-identity. See research/findings/raw/_closure1_optionA_DONE_optionB.json.
        # Only meaningful when co_resident_composer=True (it selects WHICH co-resident composer). NO `sim/` edit.
        self.co_resident_composer_kind = str(co_resident_composer_kind)
        if self.co_resident_composer_kind not in ("rf", "onebrain"):
            raise ValueError(f"co_resident_composer_kind must be 'rf' or 'onebrain', got {co_resident_composer_kind!r}")
        self._onebrain_k_max = int(onebrain_k_max)
        # --- DA salience-gate (roadmap #6 / burndown I-4-a) state. PRODUCTION DEFAULT = ON (the merged DEFAULT now
        # INTERACTS: the shared spiking-SNc dopamine reaches the conversational composer's read-side precision gate).
        # Byte-identical at rest (DA == baseline => g_eff = g0 floor => no-op); moat-safe + nav-neutral by
        # construction (see the __init__ docstring). False = the legacy byte-identical-everywhere read path. ---
        self.enable_da_salience_gate = bool(enable_da_salience_gate)
        self._da_gate_g0 = float(da_gate_g0)
        self._da_gate_k = float(da_gate_k)
        self._da_gate_cap = float(da_gate_cap)
        # --- DA ENCODING-gain hook (TRUE ONE BRAIN roadmap #6 WRITE side / burndown I-7-b): the WRITE-side mirror of
        # the read-side salience gate. The shared spiking-SNc dopamine modulates the composer's fact ENCODING STRENGTH
        # AT STORE TIME (Lisman-Grace hippocampal-VTA loop; Kandel D.16 — dopamine gates entry into LONG-TERM memory: a
        # rewarded trace stays STABLE, an un-rewarded one degrades). When True, the composer's `encoding_gain_fn` is
        # wired to read the SAME shared `dopamine` `_da_confidence_gate` reads (off THIS agent's merged bridge) and map
        # it to a per-fact encoding gain `g = clip(g_min, g_max, 1 + k*(DA - DA_baseline))` (the de-risk's
        # `da_to_encoding_gain`), so a fact heard while the spiking SNc is bursting (a salient/rewarded utterance) is
        # encoded STRONGER than one heard at DA baseline. Default OFF = byte-identical (g=1 for every fact, the unit-mag
        # write). MOAT-SAFE by construction: the gain scales the stored complex-weight MAGNITUDE only; the cue-match
        # abstention + the cleanup winner-pick read relative scores (magnitude-invariant for the argmax), so the moat is
        # unchanged. LOAD-BEARING only when the composer stores MAGNITUDE in synapses — i.e. the production
        # `OneBrainComposer` (`_write_block` -> `store_conns`) or `RFPhasorComposer(enable_substrate_store=True)`; the
        # default merged `RFPhasorComposer`/`MergedRFComposer` numpy-kb path stores PHASES (magnitude-invariant), so the
        # hook is wired + DA-correct there but behaviorally INERT until a substrate-store composer is used (the
        # characterized consolidation step — the OneBrainComposer co-residence). De-risked GO on the production
        # OneBrainComposer (the stored block magnitude == g, lesion-confirmed, moat 0-FA, regression byte-identical):
        # research/findings/raw/_burndown_I7_limbic_encoding_hook.json + the prior oracle GO
        # 2026-06-19-dopamine-encoding-gain-derisk.md. NO `sim/` edit (composer-runner-layer read of a spike-derived
        # scalar -> the composer's already-shipped default-OFF `encoding_gain_fn`).
        self.enable_da_encoding_gain = bool(enable_da_encoding_gain)
        self._da_encoding_k = float(da_encoding_k)
        self._da_encoding_g_min = float(da_encoding_g_min)
        self._da_encoding_g_max = float(da_encoding_g_max)
        # --- DA RECALL-VIGOR hook (TRUE ONE BRAIN roadmap #6 READ side, the higher-leverage lever the Route-A/B
        #     findings both point to — deep-research scoping 2026-06-30-tier2-6-limbic-to-composer-scoping.md Option 1;
        #     validated GPU 6/6 GO on THIS deployed merged agent + the real spiking SNc,
        #     research/runners/_tier2_6_da_recall_vigor_derisk.py). The shared spiking-SNc dopamine carries a
        #     value/salience PRIOR that RE-RANKS *which* familiarity-cleared stored fact is RETRIEVED by a who/what
        #     recall (Niv-2007 tonic-DA response vigor; Kandel/catalog O.19 + G.16 "value scales the accumulator drift
        #     rate"; Lisman-Grace salience-gated retrieval): `score'_i = match_i + beta*(DA - DA_tonic)*value_i`, the
        #     argmax wins. When True, the agent's `what_does`/`who_does` re-rank the cue-gated candidate set by this
        #     value prior (via the validated `DARecallVigorComposer`, reuse-by-import) instead of taking the composer's
        #     plain first-match. Default OFF = byte-identical (the composer's plain query_*; value tags ignored).
        #     MOAT-SAFE BY CONSTRUCTION: the prior re-ranks ONLY within the familiarity-gated (exact-cue-decode-match)
        #     candidate set — an UNSTORED cue yields an EMPTY set -> abstain (None) at every DA level, so the prior can
        #     NEVER manufacture a false-accept (the no-confab moat is a HARD structural gate, not a tunable). The DA is
        #     read from the SAME shared `dopamine` `_da_confidence_gate`/`_da_encoding_gain` read (off THIS agent's
        #     merged bridge); the reference (DA_tonic) is the `dopamine` modulator baseline, so the prior is exactly OFF
        #     at tonic DA (byte-identical at rest). NO `sim/` edit (composer-runner-layer read-side value prior; reuses
        #     the composer's own on-substrate unbind/cleanup as the familiarity gate + match scores). The VALUE TAG per
        #     fact is the fact's salience at ENCODING TIME = the live shared dopamine relative to baseline (a salient/
        #     high-DA utterance encodes with higher value — the SAME Lisman-Grace salience the encoding-gain WRITE side
        #     reads); facts heard at tonic DA carry value 0. `da_recall_value_default` is the value used only when the
        #     bridge has NO `dopamine` modulator (no limbic/critic/TD slice co-resident) so the prior is otherwise a
        #     no-op; the hook is meaningful only with a `dopamine` modulator present (the production default).
        self.enable_da_recall_vigor = bool(enable_da_recall_vigor)
        self._da_recall_beta = float(da_recall_beta)
        self._da_recall_value_default = float(da_recall_value_default)
        self._da_recall_baseline = 0.5            # resolved from the `dopamine` modulator baseline in __init__ (below)
        self._fact_values = []                    # per-kb-index stored value/salience tag (parallel to composer.kb)
        self._da_recall_view = None               # lazily-built DARecallVigorComposer bound to self.composer
        # co_resident_td_cueshift (TRUE ONE BRAIN roadmap #3): also lift the A-CSC TD cue-shift slice onto the merged
        # bridge (the td_ slice + the `dopamine`-over-td_snc modulator). Default False = byte-preserved. When True the
        # moat-no-regression check verifies the shared TD DA broadcast does not perturb conversational comprehension.
        self.co_resident_td_cueshift = bool(co_resident_td_cueshift)
        # co_resident_limbic (TRUE ONE BRAIN item #1): also lift the shared reward/value/dopamine limbic core
        # onto the merged bridge (the limbic_ slice + the `dopamine` modulator). Default False = byte-preserved
        # (the conversational gate b is unaffected). When True, the moat-no-regression check verifies the shared
        # DA modulator (threshold-0, neutral-at-rest) does not perturb the parser/conversational comprehension.
        self.co_resident_limbic = bool(co_resident_limbic)
        # co_resident_nav_critic (CYCLE 209): lift the FULL nav reward/critic (vs the minimal limbic organ) onto the
        # merged bridge -- the spiking limbic core (US->SNc reward burst + striosome_value MSN-D1 value critic +
        # the scope=all `dopamine` modulator over [snc]); the moat check verifies the DA-over-snc modulator does
        # NOT perturb the frozen conversational comprehension (validated 15/15: tests/test_nav_conv_merged_agent
        # 8 + tests/test_nav_conv_step2b_coresident 7 all pass with this resident, incl. the is-None no-confab
        # assertions).
        #
        # PRODUCTION DEFAULT = ON (2026-06-21, production-wiring nav chunk item 5): the production "one brain"
        # agent brings up the spiking reward/value/dopamine limbic core by default, so the merged-nav cognition
        # (reward, value, RPE) is spiking-by-default (brain-based purity). GREEN_INERT CAVEAT (documented, NOT
        # hidden): the nav value/RPE is BEHAVIORALLY INERT on the orient-solvable immediate-reward gridworld (the
        # #9 lesson / the merged-nav-critic BOUNDARY finding) -- flipping this is a brain-based-purity default, not
        # a navigation behavior win; the limbic core is validated spiking but its δ does not change the navigation
        # score. The default is `None` = "production default ON unless a MUTUALLY-EXCLUSIVE critic was explicitly
        # requested": co_resident_limbic (the 4-region minimal organ) and co_resident_td_cueshift (the A-CSC TD
        # cue-shift slice) each register their OWN scope=all DA broadcast, so only ONE critic can be co-resident
        # (the builder asserts this). An EXPLICIT co_resident_limbic=True / co_resident_td_cueshift=True therefore
        # YIELDS the production default (the explicit research-config request wins; no mutual-exclusivity crash);
        # an EXPLICIT co_resident_nav_critic=False also opts out (the legacy no-critic config). The low-level
        # build_merged_nav_conv_bridge default stays False (conservative -- the many research runners that compose
        # their own critic config keep the assert protecting a genuine double-request).
        if co_resident_nav_critic is None:
            # production default: ON, unless a mutually-exclusive critic was explicitly requested.
            self.co_resident_nav_critic = not (bool(co_resident_limbic) or bool(co_resident_td_cueshift))
        else:
            self.co_resident_nav_critic = bool(co_resident_nav_critic)
        # nav_critic_spiking_sc (TRUE ONE BRAIN roadmap #2): also lift the spiking SC chain so the nav reward `r`
        # is the SYNAPTIC SC-proximity (sc_rostral->reward_us), retiring the host Manhattan formula. Default
        # False = byte-preserved. Only meaningful with co_resident_nav_critic (it builds reward_us + the critic).
        self.nav_critic_spiking_sc = bool(nav_critic_spiking_sc)
        # nav_critic_place_selforg (TRUE ONE BRAIN roadmap #5) + nav_critic_grid_frontend (#5b R1, nav chunk item 2):
        # swap the critic's host-Gaussian vs_place_context position afferent for the SELF-ORGANIZED spiking `place`
        # code (place_sensors -> place threshold-WTA + place_fs FS-PING -> plastic coincidence place->striosome_value),
        # with the place_sensors afferent fed by the DECORRELATED spatial-phase grid metric (the grid front end)
        # instead of the locally-degenerate landmark render.
        #
        # PRODUCTION DEFAULT = ON (2026-06-21, #5b CLOSED): the production "one brain" agent retires the host-Gaussian
        # `vs_place_context` place-code scaffold by default — the place code is genuinely NEURAL (the self-org place
        # pool carves locally-SELECTIVE fields off the grid metric, place value V n/f 4.5-12.3x vs the render's 1.0x
        # R1-cap; R1 GO 3/3, research/findings/2026-06-22-shortcut5b-R1-grid-frontend-derisk.md +
        # -deltabar-3of3-close.md). The TD-read de-risk (2026-06-22-shortcut5b-td-read-derisk.md) CLOSED #5b on R1
        # grounds: the host-Gaussian retires because the grid front end produces a genuinely-neural, value-gradable
        # place code (afferent selectivity + learned near/far value, both 3/3). The residual value-READ structural/
        # learned separation (the graded-plateau read conflates the place code's structural near/far magnitude with
        # learned value) is the CHARACTERIZED DENDRITIC FRONTIER -- a point-neuron-substrate limit (a two-compartment
        # neuron would route the structural drive away from the learned-value read-out; a point neuron cannot), NOT a
        # host shortcut and NOT a blocker for the close (the existing graded-plateau read stays; the close does NOT
        # depend on the TD read).
        #
        # The default is `None` = "production default ON, but ONLY when the spiking critic is actually co-resident
        # (it builds the place->striosome_value arm the self-org/grid afferent feeds). When the critic is NOT resident
        # (an explicit co_resident_nav_critic=False, OR a MUTUALLY-EXCLUSIVE critic was requested so the nav critic
        # auto-yields OFF), the self-org/grid afferent is meaningless (there is no vs_place_context to retire and no
        # critic to drive) -> resolve OFF." An EXPLICIT True forces it ON (asserts the critic is present); an EXPLICIT
        # False opts out (the legacy host-Gaussian vs_place_context afferent, the revertible escape). grid REQUIRES
        # place_selforg (the builder asserts it), so the grid default mirrors place_selforg.
        # The moat check verifies the self-org place afferent + the grid metric + the DA-over-snc modulator do NOT
        # perturb the parser/conversational comprehension (the place/critic arrays are array-disjoint from the
        # composer's complex cp_rf_w_* synapses -> the no-confab moat is preserved by construction).
        if nav_critic_place_selforg is None:
            # production default: ON when the critic is resident, else OFF (the afferent needs the critic).
            self.nav_critic_place_selforg = bool(self.co_resident_nav_critic)
        else:
            self.nav_critic_place_selforg = bool(nav_critic_place_selforg)
        if nav_critic_grid_frontend is None:
            # production default mirrors place_selforg (the grid IS the place_sensors afferent; requires the self-org pool).
            self.nav_critic_grid_frontend = bool(self.nav_critic_place_selforg)
        else:
            self.nav_critic_grid_frontend = bool(nav_critic_grid_frontend)
        # --- CONSOLIDATION (FOLLOW-ON #2, 2026-06-24): wire the two GO cross-region routes (which until now lived in
        # the standalone behavioral-task runners spoken_instruction_nav.py / navigate_to_compose_then_answer.py) ONTO
        # the deployed MergedNavConvAgent itself, so the merged agent's OWN methods carry the functional integration
        # (scoping 2026-06-23-functional-one-brain-integration-scoping.md §I-4-c / §I-5-b). BOTH are opt-in (default
        # False = byte-preserved); each is validated separately (route engages + nav Δ~0 + moat 0-FA + lesion-collapses)
        # before any default flip. Reuse-by-import (the standalone runners' primitives), NO `sim/` edit. ---
        # co_resident_perception (route B, perception->memory/compose): bring the bare `cortex_it` perception region +
        # the co-resident `rf` composer onto the merged bridge so the agent's PERCEPTION (the navigating body reading a
        # rendered object's live cortex_it rate) writes a grounded code into the co-resident composer's codebook, which
        # the conversational composer then binds/queries. REQUIRES co_resident_composer (the grounded code feeds the
        # `rf` composer's FHRR algebra). When True, the agent gains `perceive_and_ground(obj_word)` (the in-episode
        # perception->codebook grounding) so a perceived object becomes composable/recallable on the one brain.
        self.co_resident_perception = bool(co_resident_perception)
        if self.co_resident_perception and not self.co_resident_composer:
            raise ValueError("co_resident_perception requires co_resident_composer (the grounded percept code feeds "
                             "the co-resident `rf` composer's bind/unbind algebra)")
        # ROUTE-B GROUNDING MODE (purity #1 Route-B Option-1 wire-in, 2026-06-25). Mirrors the standalone behavioral
        # runner navigate_to_compose_then_answer's GROUNDING_DEFAULT/_codebook/_algebra wiring:
        #   "gen_spikes" (DEFAULT, spikes-only — the cross-region host-`M` CLOSURE): the percept is rendered into the
        #     generalization stack's structured-perception region `gen_perception`; the LEARNED (self-organized)
        #     rate-Hebbian `gen_perception->gen_concept` CONVERGENCE fires the NMDA-integrated `gen_concept` assembly; the
        #     grounded code is a FIXED read-projection of `gen_concept`'s `cp_firing_states` (REAL spikes). The
        #     load-bearing percept->concept transform is SYNAPTIC+LEARNED, NOT a host-designed matrix — so NO host
        #     quantity crosses regions (feedback_spiking_structure_must_self_organize). REQUIRES the co-resident trained
        #     generalization stack -> forces co_resident_generalization=True. WORKS on the onebrain composer because the
        #     standalone `_perceive_and_ground` writes the grounded code to `_codebook(composer)` (= `composer.comp.concepts`
        #     for the OneBrainComposer, the codebook the binds/cleanups actually read), so there is no stray-attr problem
        #     — the prior onebrain guard is RETIRED (6-seed GO: research/findings/raw/_route_b_6seed_and_agent_wirein.json).
        #   "host_m" (LEGACY, revertible A/B escape): the bare `cortex_it` region + a host-DESIGNED random projection
        #     `composer.concepts[o] = angle(M @ live_cortex_it_rate)`. Retired as the default because `M` carries
        #     host-designed (not self-organized) structure; kept ONLY for the A/B comparison.
        self.perception_grounding = str(perception_grounding)
        if self.perception_grounding not in ("gen_spikes", "host_m"):
            raise ValueError(f"perception_grounding must be 'gen_spikes' or 'host_m', got {perception_grounding!r}")
        # R4 close (default False = the validated host grounding path): run the gen_spikes perception->compose grounding
        # hand-off DEVICE-RESIDENT (no host gen_proj@rate matmul, no to_host of the gen_concept spike VECTOR; only the
        # final phases cross host, the R5 body-read). gen_spikes only; the fixed cortico-cortical fan-in runs on-device.
        self.perception_device_resident = bool(perception_device_resident)
        # co_resident_generalization (the gen stack — structured-perception gen_perception -> NMDA gen_concept ->
        # gen_fact + the trained-then-frozen rate-Hebbian convergence). gen_spikes grounding REQUIRES it (the learned
        # convergence does the percept->concept grounding); it is also independently usable for the generalization
        # checks. Appended LAST in the build so the nav/parser/dlPFC/rf/cortex_it index bases are byte-unchanged.
        self.co_resident_generalization = bool(co_resident_generalization)
        if self.co_resident_perception and self.perception_grounding == "gen_spikes":
            self.co_resident_generalization = True   # gen_spikes needs the co-resident trained gen stack
        # co_resident_command_route (route A, language->action COMMAND_GATE): bring the `language_input` region + the
        # LEARNED `language_input -> cortex_X` route (transmission_gate=command_route) onto the merged bridge so a
        # PARSED spoken command (the parser's action-role FIRING) opens the route and steers the BG cascade. When on,
        # the agent gains `command_move(direction)` (the parser comprehends -> gate opens -> the commanded word's
        # learned route biases the action cascade -> the body picks a move).
        #
        # PRODUCTION DEFAULT = ON (2026-06-24, TRUE ONE BRAIN roadmap #6 / cross-region persistent-loop build A): the
        # merged DEFAULT now carries the language->action functional integration (the "spoken command steers the body"
        # one-brain story, GO 6-seed: 2026-06-10-spoken-instruction-nav-GO.md). The route is ALREADY fully spiking -- the
        # COMMAND_GATE is a 0/1 transmission-gate STATE coupled to the parser's action-role FIRING (no host value crosses
        # regions; the parser supplies the WHEN, the word supplies the WHICH via its LEARNED route), provenance asserted
        # (spoken_instruction_nav.provenance_facts: no parser-derived value written to any nav drive), lesion-load-bearing
        # (lesion_command_route collapses command-following to chance even with the parser firing). NAV-NEUTRAL by
        # construction: the route's `language_input -> cortex_X` edges are held CLOSED by COMMAND_GATE at rest (no current,
        # no STDP cold-start) -> nav-inert until a parsed command opens them, so an UNCOMMANDED nav episode is byte-identical
        # (Δ=0). MOAT-SAFE: the route is array-disjoint from the composer's complex `cp_rf_w_*` synapses + the parser slice,
        # so it cannot perturb the conversational no-confab abstention. The `language_input` region is appended AFTER
        # cortex_it/rf (the nav/parser/dlPFC/rf/cortex_it index bases stay byte-unchanged). Following the DA-route precedent
        # (co_resident_nav_critic=None-sentinel default-ON at :1573): the default is `None` = ON; an EXPLICIT False opts out
        # (the legacy byte-identical-everywhere build, the revertible escape). NO `sim/` edit (reuse-by-import of the GO
        # standalone spoken_instruction_nav primitives). enable_spiking_wta_readout is forced on for this route (the sel_X
        # readout the standalone validated).
        if co_resident_command_route is None:
            self.co_resident_command_route = True            # production default: ON (the deployed merged agent INTERACTS)
        else:
            self.co_resident_command_route = bool(co_resident_command_route)

        _D = 128
        # CONSOLIDATION (option A): when the co-resident composer is the persistent-loop OneBrainComposer, size the
        # merged `rf` region for its FULL layout span (work registers + k_max store blocks + per-block + batched
        # Q+cleanup) -- a function of (D, |merged vocab|, k_max, attribute role). Computed BEFORE the build (it sizes
        # the region). The "rf" oracle path passes 0 (the byte-unchanged 7*rf_D sizing). The merged vocab the composer
        # gets is build_merged_nav_conv_bridge's SORTED probe vocab; n_total_for builds an RFPhasorComposer(vocab=...)
        # internally so it sees the SAME word set the composer will -> the size is exact.
        _onebrain_rf_size = 0
        if self.co_resident_composer and self.co_resident_composer_kind == "onebrain":
            _onebrain_rf_size = CoResidentOneBrainComposer.n_total_for(
                D=_D, vocab=vocab, k_max=self._onebrain_k_max, enable_attributed=False)
        # Tier-3 living-loop seam (additive, default-off, byte-preserving): the co-resident 2-pool SPIKING hunger
        # drive (drive_agrp/drive_pomc; O.05/O.06). Appended LAST in the build so all other index bases are
        # byte-unchanged. Grounded speech adds only the opt-in AgRP/POMC speech outputs; no mode adds an edge into
        # navigation or conversation. The drive is read by firing rate off cp_firing_states.
        self.co_resident_drive = bool(co_resident_drive)
        self._drive_n_pool = int(drive_n_pool)
        self._drive_to_da = bool(drive_to_da)                  # Tier-3 Option 3: hunger raises the shared DA
        self._drive_da_sensitivity = float(drive_da_sensitivity)
        self.co_resident_grounded_speech = bool(co_resident_grounded_speech)
        self._merged_bridge, self._handles = build_merged_nav_conv_bridge(
            seed=seed, vocab=vocab, co_resident_rf=self.co_resident_composer, rf_D=_D,
            onebrain_rf_size=_onebrain_rf_size,
            co_resident_drive=self.co_resident_drive, drive_n_pool=self._drive_n_pool,
            drive_to_da=self._drive_to_da, drive_da_sensitivity=self._drive_da_sensitivity,
            co_resident_grounded_speech=self.co_resident_grounded_speech,
            speech_n_acc=int(speech_n_acc), speech_n_fs=int(speech_n_fs),
            speech_drive_weight=float(speech_drive_weight), speech_cue_weight=float(speech_cue_weight),
            co_resident_perception=self.co_resident_perception,
            co_resident_generalization=self.co_resident_generalization,
            enable_spiking_wta_readout=(self.co_resident_perception or self.co_resident_command_route),
            co_resident_command_route=self.co_resident_command_route,
            co_resident_limbic=self.co_resident_limbic,
            co_resident_nav_critic=self.co_resident_nav_critic,
            nav_critic_spiking_sc=self.nav_critic_spiking_sc,
            nav_critic_place_selforg=self.nav_critic_place_selforg,
            nav_critic_grid_frontend=self.nav_critic_grid_frontend,
            co_resident_td_cueshift=self.co_resident_td_cueshift)
        words = self._handles["vocab"]   # the sorted merged vocab (the dlPFC + parser word set)

        # The composer. STEP 2a default: on its OWN per-op bridges (separate). STEP 2b (co_resident_composer=True):
        # a CO-RESIDENT composer runs the RF binding ops on the merged bridge's own `rf` slice. co_resident_composer_kind
        # selects which: "rf" (MergedRFComposer, the byte-identity oracle, numpy-kb phases) or "onebrain"
        # (CoResidentOneBrainComposer, the persistent-loop composer with the synaptic store + the LOAD-BEARING limbic
        # encoding-gain WRITE side). Same seed + vocab as the merged dlPFC either way.
        if self.co_resident_composer:
            if self.co_resident_composer_kind == "onebrain":
                # build_parser=False: the merged agent comprehends via _MergedParserAdapter (the framework parse_conj
                # slice), so the composer's own idle parser is redundant; building it would re-inject from the empty
                # _unified_wiring_plan and WIPE the framework wiring + COMMAND_GATE (the traced close-out bug). The RF
                # ops are reset-isolated from the parser -> the composer's output is byte-identical without it.
                self.composer = CoResidentOneBrainComposer(
                    self._merged_bridge, self._handles["rf_base"], build_parser=False,
                    seed=seed, D=_D, vocab=words, period=200, k_max=self._onebrain_k_max,
                    persistent_loop=True, enable_rf_cudagraph=False)
            else:
                self.composer = MergedRFComposer(
                    self._merged_bridge, self._handles["rf_base"], self._handles["rf_size"],
                    seed=seed, D=_D, vocab=words, period=200)
        else:
            self.composer = RFPhasorComposer(seed=seed, D=_D, vocab=words, period=200)

        # WRITE-side limbic->composer hook (burndown I-7-b): wire the composer's encoding_gain_fn to read the SHARED
        # `dopamine` (the same DA `_da_confidence_gate` reads) so a fact heard during a salient/high-DA turn encodes
        # stronger (Lisman-Grace/Kandel D.16). PRODUCTION DEFAULT = ON (2026-06-25, Closure 3 flip): the LOAD-BEARING
        # limbic WRITE side is engaged on the (now-default) onebrain composer (OneBrainComposer._write_block scales the
        # stored complex-weight MAGNITUDE by encoding_gain_fn()). Byte-identical AT REST (DA == baseline => g=1 => no-op);
        # the gain rises above 1 only on a salient/high-DA encode. Set enable_da_encoding_gain=False for the legacy
        # byte-identical-everywhere write path. The hook is inert on the rf-oracle composer (numpy-kb phases are
        # magnitude-invariant); see the __init__ docstring for the inert-on-numpy-kb caveat.
        if self.enable_da_encoding_gain:
            self.composer.encoding_gain_fn = self._da_encoding_gain

        # READ-side limbic->composer recall-vigor hook (roadmap #6 READ side): resolve the DA reference (DA_tonic) from
        # the shared `dopamine` modulator baseline so the value prior is exactly OFF at tonic DA (byte-identical at
        # rest). If no `dopamine` modulator is present, the default 0.5 is harmless (the prior is a no-op anyway —
        # _da_recall_dopamine returns the baseline -> da_term == 0). NO composer mutation when the hook is OFF.
        _nm = getattr(self._merged_bridge, "neuromodulator_manager", None)
        if _nm is not None:
            try:
                self._da_recall_baseline = float(_nm._config_by_name("dopamine").baseline)
            except (KeyError, AttributeError):
                pass                                          # no `dopamine` modulator -> keep the harmless default

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

        # --- ROUTE A (language->action) setup (FOLLOW-ON #2): the command-route grounding into the agent. The route +
        #     gate live on the merged bridge; couple `command_route` to the parser's action-role FIRING here so a
        #     comprehended verb opens the route (the in-substrate gate-from-firing, NOT a Python value copy). ---
        self._grounded_proj = None
        if self.co_resident_command_route:
            assert "language_input" in region_names, \
                "FAIL anti-cheat: co_resident_command_route set but no 'language_input' region on the merged bridge"
            assert COMMAND_GATE in self._merged_bridge._transmission_gate_to_synapses, \
                "FAIL anti-cheat: the command_route transmission gate is not registered on the merged bridge"
            self._couple_command_gate()
        # --- ROUTE B (perception->memory/compose) setup: the fixed grounded-code read-projection. For gen_spikes the
        #     projection FORMATS the spiking `gen_concept` response into a phasor (the LEARNED convergence does the
        #     grounding); for host_m it is the legacy host-`M` projection of the bare cortex_it live rate. Mirrors the
        #     standalone navigate_to_compose_then_answer.build_compose_bridge's exact construction (reuse-by-import). ---
        self._grounded_objects = []
        if self.co_resident_perception:
            import numpy as _np
            from sim.backend import get_backend as _gb
            from research.runners._step3_grounded_codes_production_composer_derisk import _projection
            from research.runners.navigate_to_compose_then_answer import _gen_read_projection
            _xp, _ = _gb()
            it_h = self._handles["cortex_it_indices"]
            self._handles["cortex_it_indices_xp"] = _xp.asarray(_np.asarray(it_h, dtype=_np.int64))
            if self.perception_grounding == "gen_spikes":
                assert "gen" in self._handles, \
                    "gen_spikes grounding requires the co-resident generalization stack (handles['gen'])"
                _n_conc = int(_np.asarray(self._handles["gen"]["conc_region"], dtype=_np.int64).size)
                # the FIXED read-projection of the gen_concept SPIKES (length n_concept) -> D-dim phasor. The
                # standalone runner's `_gen_read_projection` (distinct seed offset from host_m's `M` so the two modes'
                # projections never coincide). Stored on handles["gen_proj"] (where the standalone `_perceive_and_ground`
                # reads it). `_grounded_proj` stays the legacy cortex_it `M` for the host_m A/B path.
                self._handles["gen_proj"] = _gen_read_projection(_D, _n_conc, seed)
            self._grounded_proj = _projection(_D, int(_np.asarray(it_h).size), seed)

    # --- ROUTE A (language->action): the parser action-ensemble firing opens the command_route + one commanded move ---
    def _couple_command_gate(self):
        """Couple `command_route` to the parser's ACTION-role ensemble firing (the in-substrate primitive — a 0/1 gate
        STATE from the parser SPIKING, not a value). Ported from spoken_instruction_nav.couple_command_gate."""
        from research.runners.unified_brain_bridge import couple_gate_to_indices
        from research.runners.spoken_instruction_nav import COMMAND_GATE_THRESHOLD, COMMAND_GATE_ALPHA
        couple_gate_to_indices(self._merged_bridge, COMMAND_GATE, to_host(self._handles["action_block_idx"]),
                               threshold=COMMAND_GATE_THRESHOLD, alpha=COMMAND_GATE_ALPHA)

    def command_move(self, direction, parse_first=True):
        """Route A — ONE navigation decision driven by a PARSED spoken command. The parser comprehends the action verb
        (its action ensemble FIRES) -> `command_route` opens (via the firing coupling) -> the commanded direction word's
        LEARNED `language_input -> cortex_{direction}` current biases the action cascade -> the body picks the move (the
        cascade's disinhibited sel/motor winner). Returns (chosen_action, per-pool counts). Delegates to the GO
        standalone `spoken_instruction_nav.decide_move` against THIS agent's merged-bridge handles, so the in-episode
        mechanism is byte-for-byte the validated one (reuse-by-import).

        parse_first=False is the ISOLATED-NAV control (no parser drive -> the gate stays closed -> the word's route
        current never reaches cortex -> chance)."""
        assert self.co_resident_command_route, "command_move requires co_resident_command_route=True"
        from research.runners.spoken_instruction_nav import decide_move
        h = {
            "conj_arr": self._handles["conj_arr"],
            "lang_indices": self._handles["lang_indices"],
            "readout_idx": self._handles["cmd_readout_idx"],
            "readout_region": self._handles["cmd_readout_region"],
            "cascade_tonic": self._handles["cmd_cascade_tonic"],
            "sel_all_idx": (list(self._handles["cmd_readout_idx"].values())
                            if self._handles["cmd_readout_region"] == "sel" else []),
        }
        return decide_move(self._merged_bridge, h, direction, parse_first=parse_first)

    def lesion_command_route(self):
        """Cut the command_route (zero its synapses) — the route A lesion control (the behavior must collapse to chance,
        proving it rides the SYNAPTIC route). Returns the number of synapses zeroed."""
        assert self.co_resident_command_route, "lesion_command_route requires co_resident_command_route=True"
        from research.runners.spoken_instruction_nav import _lesion_command_route
        return _lesion_command_route(self._merged_bridge)

    # --- ROUTE B (perception->memory/compose): ground a perceived object into the composer codebook from live cortex_it ---
    def perceive_and_ground(self, obj_word):
        """Route B — the agent has PERCEIVED object `obj_word`: read the percept's LIVE SPIKING response OFF THE MERGED
        BRIDGE and SET the composer codebook entry for `obj_word` = the grounded phasor code. The percept thus becomes a
        phasor the co-resident composer's FHRR algebra can bind/query — so a perceived object is composable on the one
        brain. Delegates to the GO standalone `navigate_to_compose_then_answer._perceive_and_ground` against THIS agent's
        merged bridge + co-resident composer (reuse-by-import), which writes the grounded code to `_codebook(composer)`
        (= `composer.comp.concepts` for the OneBrainComposer, the codebook the binds/cleanups actually read; =
        `composer.concepts` for the rf composer). Returns (source_vec, phases).

        GROUNDING (self.perception_grounding):
          "gen_spikes" (DEFAULT — the cross-region host-`M` CLOSURE, spikes-only): the object is RENDERED as its
            structured-perception set into `gen_perception` (the sensory render); the LEARNED rate-Hebbian
            `gen_perception->gen_concept` CONVERGENCE fires the NMDA-integrated `gen_concept` assembly; the grounded code
            is a FIXED read-projection of `gen_concept`'s `cp_firing_states` (REAL spikes). The load-bearing
            percept->concept transform is SYNAPTIC+LEARNED — NO host quantity crosses regions (in particular NO
            `composer.concepts[o]=host_fn(cortex_it_rate)`, the retired host-`M` round-trip). Validated through the AGENT
            on the onebrain composer (the prior onebrain guard is retired): the held-out compose recovers the perceived
            object >> the memorization floor, lesioning the convergence collapses it, the moat abstains
            (research/findings/raw/_route_b_6seed_and_agent_wirein.json).
          "host_m" (LEGACY, revertible A/B escape): the bare cortex_it live rate -> the host-DESIGNED random projection
            `self._grounded_proj` (`composer.concepts[o] = angle(M @ live_cortex_it_rate)`).

        Requires co_resident_perception (+ co_resident_composer). For gen_spikes, co_resident_generalization is forced on
        in __init__ so the gen stack + the gen read-projection (handles['gen'], handles['gen_proj']) are present."""
        assert self.co_resident_perception, "perceive_and_ground requires co_resident_perception=True"
        from research.runners.navigate_to_compose_then_answer import _perceive_and_ground
        h = {
            "grounding": self.perception_grounding,        # gen_spikes (DEFAULT, the host-`M` closure) | host_m (A/B)
            "composer_kind": self.co_resident_composer_kind,
            "it_indices": self._handles["cortex_it_indices_xp"],
            "grounded_objects": self._grounded_objects,
            # R4 close (default False): the gen_spikes grounding hand-off runs device-resident (no host gen_proj@rate
            # matmul, no to_host of the gen_concept spike VECTOR; only the final phases cross host, the R5 body-read).
            "device_resident_grounding": self.perception_device_resident,
        }
        if self.perception_grounding == "gen_spikes":
            # the gen handles the standalone `_perceive_and_ground`/`read_gen_concept_spikes` consult for the
            # SPIKES-ONLY grounding (the trained convergence + the gen_concept read-projection), surfaced from the build.
            h["gen"] = self._handles["gen"]
            h["gen_proj"] = self._handles["gen_proj"]
        return _perceive_and_ground(self._merged_bridge, self.composer, h, self._grounded_proj, obj_word)

    # --- DA salience-gate (roadmap #6): the spiking-SNc dopamine -> composer cue-role confidence gate ---
    def _da_confidence_gate(self):
        """Read the SHARED spiking-SNc dopamine off the merged bridge and map it (CLAMPED-TO-SHARPEN, the de-risk's
        `da_to_gate`) onto the composer's cue-role confidence gate `g_eff`. SAFE: if no `dopamine` modulator is on
        the merged bridge (no limbic/critic/TD slice co-resident), DA reads as baseline => `g_eff = g0` (the gate
        floor = a no-op). The map can ONLY raise the gate above `g0` (DA above baseline) -> the moat can only tighten
        (moat-safe by construction). Returns `g_eff`."""
        from research.runners._da_composer_salience_cleanup_derisk import da_to_gate   # reuse the de-risk map VERBATIM
        g0 = self._da_gate_g0
        nm = getattr(self._merged_bridge, "neuromodulator_manager", None)
        if nm is None:
            return g0                                    # no neuromodulator subsystem -> the gate floor (no-op)
        try:
            da = float(nm.get_concentration("dopamine"))
            da_baseline = float(nm._config_by_name("dopamine").baseline)
        except (KeyError, AttributeError):
            return g0                                    # no `dopamine` modulator present -> the gate floor (no-op)
        return da_to_gate(da, da_baseline, g0, self._da_gate_k, g_cap=self._da_gate_cap)

    # --- DA ENCODING-gain hook (roadmap #6 WRITE side / burndown I-7-b): the spiking-SNc dopamine -> composer fact
    #     ENCODING strength at store time (the write-side mirror of _da_confidence_gate) ---
    def _da_encoding_gain(self):
        """Read the SHARED spiking-SNc dopamine off the merged bridge and map it (the de-risk's `da_to_encoding_gain`)
        onto a per-fact ENCODING gain `g = clip(g_min, g_max, 1 + k*(DA - DA_baseline))`, read AT STORE TIME by the
        composer's `encoding_gain_fn`. SAFE: if no `dopamine` modulator is on the merged bridge (no limbic/critic/TD
        slice co-resident), DA reads as baseline => `g = 1.0` (the byte-identical unit-mag write = a no-op), exactly
        like `_da_confidence_gate`'s gate-floor fallback. A salient (high-DA) turn => g > 1 => a stronger, more-stable
        encoding (Lisman-Grace/Kandel D.16). MOAT-SAFE: the gain scales stored magnitude only; the cue-match abstention
        is unchanged. Returns `g`."""
        from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import da_to_encoding_gain  # the de-risk map
        nm = getattr(self._merged_bridge, "neuromodulator_manager", None)
        if nm is None:
            return 1.0                                   # no neuromodulator subsystem -> unit gain (byte-identical)
        try:
            da = float(nm.get_concentration("dopamine"))
            da_baseline = float(nm._config_by_name("dopamine").baseline)
        except (KeyError, AttributeError):
            return 1.0                                   # no `dopamine` modulator present -> unit gain (byte-identical)
        return da_to_encoding_gain(da, da_baseline, self._da_encoding_k,
                                   g_min=self._da_encoding_g_min, g_max=self._da_encoding_g_max)

    # --- DA RECALL-VIGOR hook (roadmap #6 READ side): the shared spiking-SNc dopamine carries a value/salience prior
    #     that re-ranks WHICH familiarity-cleared stored fact is RETRIEVED (the validated DARecallVigorComposer) ---
    def _da_recall_dopamine(self):
        """Read the SHARED spiking-SNc dopamine off the merged bridge (the SAME read `_da_confidence_gate` /
        `_da_encoding_gain` use). Returns the live concentration, or the resolved tonic reference if no `dopamine`
        modulator is present (=> da_term == 0 => the value prior is a no-op). The GPU agent reads the live SNc here; a
        CPU guard test overrides this with a scalar."""
        nm = getattr(self._merged_bridge, "neuromodulator_manager", None)
        if nm is None:
            return float(self._da_recall_baseline)
        try:
            return float(nm.get_concentration("dopamine"))
        except (KeyError, AttributeError):
            return float(self._da_recall_baseline)

    def _store_fact_value(self, agent, action, patient, value=None, polarity=None):
        """Store a fact via the composer (the unchanged on-substrate encode) AND record its per-fact VALUE/SALIENCE tag
        (parallel to `composer.kb`) so the recall-vigor prior can re-rank by it. `value`=None (the live-store path) =>
        the fact's salience at ENCODING TIME = the live shared dopamine relative to baseline, clipped to >= 0 (a
        salient/high-DA utterance encodes with higher value — the SAME Lisman-Grace salience the encoding-gain WRITE
        side reads; a tonic-DA fact carries value 0). When the bridge has no `dopamine` modulator the salience is the
        `da_recall_value_default` (so the prior is otherwise a documented no-op). The list stays in lock-step with the
        kb even when the recall-vigor hook is OFF (cheap; keeps the value tags valid if the hook is toggled on)."""
        if value is None:
            nm = getattr(self._merged_bridge, "neuromodulator_manager", None)
            has_da = False
            if nm is not None:
                try:
                    nm.get_concentration("dopamine"); has_da = True
                except (KeyError, AttributeError):
                    has_da = False
            if has_da:
                value = max(0.0, self._da_recall_dopamine() - float(self._da_recall_baseline))
            else:
                value = float(self._da_recall_value_default)
        self.composer.store(agent, action, patient, polarity=polarity)
        self._fact_values.append(float(value))

    def _da_recall_vigor_view(self):
        """The validated `DARecallVigorComposer` (reuse-by-import) bound to THIS agent's composer + value tags + the
        live shared-DA read. Lazily built (and rebound if the composer object changed). The view holds NO state beyond
        a reference to `self.composer` + `self._fact_values`, so it tracks the agent's live kb."""
        from research.runners._tier2_6_da_recall_vigor_derisk import DARecallVigorComposer
        v = self._da_recall_view
        if v is None or v.comp is not self.composer:
            v = DARecallVigorComposer(self.composer, da_fn=self._da_recall_dopamine,
                                      beta=self._da_recall_beta, da_baseline=self._da_recall_baseline)
            v.values = self._fact_values            # SHARE the agent's value list (not a copy) so it tracks new stores
            self._da_recall_view = v
        else:
            v.da_baseline = float(self._da_recall_baseline)
            v.beta = float(self._da_recall_beta)
        return v

    def _da_recall_select(self, **cue_roles):
        """The DA-gated value-prior winner over the cue-gated candidate set, generalized to a MULTI-ROLE who/what cue
        (the de-risk's `DARecallVigorComposer` cues on a single role; a production who/what recall gates on the JOINT
        cue — agent AND action for `what_does`, action AND patient for `who_does`). Returns the kb index of the
        value-prior winner among the facts whose cue roles ALL decode-match `cue_roles`, or None (abstain) if the gated
        set is empty (the no-confab moat: an unstored/again-unmatched cue has nothing to re-rank).

        The re-rank is the de-risk's EXACT formula `score'_i = match_i + beta*(DA - DA_tonic)*value_i` (match_i = the
        composer's OWN matched-filter cleanup confidence; the value prior re-ranks ONLY within the gated set), reusing
        the composer's on-substrate `_unbind_all_phases` + `_cleanup_all_scored` (the SAME primitives the
        DARecallVigorComposer view uses) as the familiarity gate + the per-role match scores. Ties resolve to the
        FIRST candidate (kb order) so the prior-off / value-independent baseline is the composer's first-match."""
        comp = self.composer
        comps = [c for _f, c in comp.kb]
        if not comps:
            return None
        # decode every cue role for every stored fact (batched on-substrate unbind + matched-filter cleanup -- the
        # composer's OWN op result; this IS the familiarity gate the no-confab moat rides on).
        decoded = {}
        scored = {}
        for role, want in cue_roles.items():
            rec = comp._unbind_all_phases(comps, role)
            words, scores = comp._cleanup_all_scored(rec)
            decoded[role] = words
            scored[role] = scores
        cand = [i for i in range(len(comps))
                if all(decoded[role][i] == want for role, want in cue_roles.items())]
        if not cand:
            return None                              # empty gated set -> abstain (the moat)
        da = float(self._da_recall_dopamine())
        da_term = self._da_recall_beta * (da - float(self._da_recall_baseline))   # the DA gate on the value prior
        # match_i = the mean of the cue roles' cleanup confidences (the de-risk uses the single cue role's score; for a
        # joint cue the mean is the natural matched-filter confidence over the gated roles). Value-weighted score'.
        best_i, best_s = None, None
        for i in cand:
            match_i = float(np.mean([scored[role][i] for role in cue_roles]))
            val = self._fact_values[i] if i < len(self._fact_values) else 0.0
            sprime = match_i + da_term * float(val)
            # strict '>' so an exact tie keeps the EARLIER (lower-index) candidate = the first-match baseline
            if best_s is None or sprime > best_s + 1e-12:
                best_i, best_s = i, sprime
        return best_i

    def _gated_out(self, match_fn, g_eff):
        """The de-risk's gate, applied to THIS agent's composer reads (no `sim/`/composer edit, reuse-by-import): is
        the FIRST stored block matching the cue (`match_fn(agent, action, patient)`) NOISE-DOMINATED at the operating
        gate `g_eff`? Returns True iff a matching block exists but its CUE-ROLE cleanup is below the gate
        (`min(margin(agent_scores), margin(action_scores)) < g_eff`, the EXACT quantity the de-risk gates on) -> the
        agent abstains (the salience-gated precision tail). Returns False (do NOT gate) when `g_eff <= g0` (the
        no-modulation floor: BYTE-IDENTICAL), when no block matches (the composer abstains anyway -> the moat is
        unchanged), or when the matched read is decisive. Reuses the composer's OWN cleanup primitives
        (`_unbind_phases` + the matched filter == `_cleanup`'s cosine scores) + `OneBrainComposer._margin`.

        CONSOLIDATION (option A): the co-resident OneBrainComposer stores facts in SYNAPSES (no numpy `_iter_facts`
        with a host composite to scan), and it already has a NATIVE `confidence_gate` applying the SAME
        `min(margin(agent),margin(action)) < g` abstention inside its read path (`_read_block`/`_read_blocks`). So for
        the onebrain composer the gate is delegated: set `composer.confidence_gate = g_eff` (or 0.0 at the floor =
        byte-identical-at-rest) and return False (the native gate inside the subsequent query_* call handles the
        abstention). Moat-safe by construction (a higher gate only TIGHTENS abstention)."""
        from research.runners.one_brain_composer import OneBrainComposer    # the EXACT margin function under de-risk
        comp = self.composer
        if isinstance(comp, OneBrainComposer):
            # route the DA gate through the composer's NATIVE confidence_gate; the floor maps to 0.0 (off) so an
            # at-rest (DA==baseline) read is byte-identical to the no-gate default.
            comp.confidence_gate = 0.0 if g_eff <= self._da_gate_g0 + 1e-12 else float(g_eff)
            return False
        if g_eff <= self._da_gate_g0 + 1e-12:
            return False                                 # the gate floor -> no modulation -> byte-identical read path
        for fact, composite in comp._iter_facts():       # the same cue-matching scan the composer's query_* uses
            sa = self._role_cleanup_scores(composite, "agent")
            sv = self._role_cleanup_scores(composite, "action")
            sp = self._role_cleanup_scores(composite, "patient")
            wa = comp.words[int(np.argmax(sa))]
            wv = comp.words[int(np.argmax(sv))]
            wp = comp.words[int(np.argmax(sp))]
            if match_fn(wa, wv, wp):                     # the FIRST matching block (first-match semantics preserved)
                return bool(min(OneBrainComposer._margin(sa), OneBrainComposer._margin(sv)) < g_eff)
        return False                                     # no match -> the composer abstains; the moat is unchanged

    def _role_cleanup_scores(self, composite, role):
        """The composer's matched-filter cleanup scores for `role` from a stored composite -- IDENTICAL (up to the
        argmax-irrelevant /D) to `RFPhasorComposer._cleanup`'s `cos(rec - concepts[w])` over the vocab. Reuses the
        composer's `_unbind_phases` (the on-substrate RF unbind); rectified (the NEF-cleanup off-target-zero
        convention) so the margin is the de-risk's `(peak - runner_up)/peak`."""
        comp = self.composer
        rec = comp._unbind_phases(composite, role)
        sims = np.array([float(np.mean(np.cos(2.0 * np.pi * (rec - comp.concepts[w])))) for w in comp.words])
        return np.maximum(sims, 0.0)

    # --- comprehend / store / recall (mirror BrainConversationalAgent exactly) ---
    def hear(self, sentence, voice="active", polarity=None):
        """Comprehend an SVO statement and store it. `sentence` is 'agent action patient' (or its passive frame). The
        fact's VALUE/SALIENCE tag (the live shared dopamine relative to baseline) is recorded for the recall-vigor
        prior -- via `_store_fact_value` (which calls `composer.store`), so the store is unchanged and the value list
        stays in lock-step with the kb whether or not the recall-vigor hook is on."""
        roles = self.parser.parse(sentence.split(), voice)
        self._store_fact_value(roles["agent"], roles["action"], roles["patient"], polarity=polarity)
        return roles

    def hear_clause_fact(self, agent, action, clause, polarity=None):
        """Store a fact whose patient is an embedded clause (the parser handles flat SVO; nested input parsing is
        future work, so the clause is provided structurally here). Records the fact's value/salience tag (live shared
        dopamine) for the recall-vigor prior."""
        self._store_fact_value(agent, action, clause, polarity=polarity)

    def what_does(self, agent, action):
        """'what does <agent> <action>?' -> patient (concept or rendered clause) or None (abstain). When the DA
        salience gate is on, a high-DA (salient) turn ABSTAINS on a noise-dominated cue read (moat-safe sharpening).
        When the DA RECALL-VIGOR hook is on (roadmap #6 READ side), the (agent, action)-gated candidate set is
        RE-RANKED by the shared-dopamine value prior so the high-value fact's patient is recalled (default OFF =>
        the composer's plain first-match query). MOAT-SAFE: the prior re-ranks ONLY within the familiarity-gated set,
        so an unstored/unmatched cue still abstains (the selector returns None -> None)."""
        if self.enable_da_salience_gate and self._gated_out(
                lambda wa, wv, _wp: wa == agent and wv == action, self._da_confidence_gate()):
            return None
        if self.enable_da_recall_vigor:
            return self._render_patient_at(self._da_recall_select(agent=agent, action=action))
        return self.composer.query_patient(agent, action)

    def who_does(self, action, patient):
        """'who <action> <patient>?' -> agent or None (abstain). The DA salience gate + the DA recall-vigor prior apply
        symmetrically to what_does (the value prior re-ranks the (action, patient)-gated set; default OFF =>
        the composer's plain first-match)."""
        if self.enable_da_salience_gate and self._gated_out(
                lambda _wa, wv, wp: wv == action and wp == patient, self._da_confidence_gate()):
            return None
        if self.enable_da_recall_vigor:
            i = self._da_recall_select(action=action, patient=patient)
            return None if i is None else self.composer.unbind(self.composer.kb[i][1], "agent")
        return self.composer.query_agent(action, patient)

    def _render_patient_at(self, idx):
        """Render the patient of stored fact `idx` (the recall-vigor winner) EXACTLY as the composer's plain
        `query_patient` does for a matched fact -- the decoded patient noun plus any bound attributes -- so the
        value-prior path produces the same answer SHAPE as the plain query (only WHICH fact differs). None idx ->
        None (abstain)."""
        if idx is None:
            return None
        comp = self.composer
        fact, composite = comp.kb[idx]
        noun = comp._render(composite, "patient", fact.get("patient"))
        adjs = [comp.unbind(composite, r) for r in ("attribute", "attribute2") if r in fact]
        return " ".join(adjs + [noun]) if adjs else noun

    def is_it_true(self, agent, action, patient):
        if self.enable_da_salience_gate and self._gated_out(
                lambda wa, wv, wp: wa == agent and wv == action and wp == patient, self._da_confidence_gate()):
            return "unknown"
        return self.composer.ask_yes_no(agent, action, patient)

    def describe(self, agent):
        """Generation: produce a sentence about `agent` from the spiking memory ('dog go north'), or None if the agent
        knows no fact about it (no confabulation). The DA salience gate abstains on a noise-dominated subject read."""
        if self.enable_da_salience_gate and self._gated_out(
                lambda wa, _wv, _wp: wa == agent, self._da_confidence_gate()):
            return None
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
    # NOTE (production-wiring nav chunk item 1, 2026-06-21): the merged-nav episode now INHERITS the
    # `log_polar_retina=True` library default of run_moving_goal_episode (the #6 biology-faithful SC
    # retina, 5/6-GO). It is BYTE-INERT in THIS gate because the STEP-2a byte-identity smoke runs the
    # host-argmax nav cascade WITHOUT the spiking SC (the SC eye-drive render that consumes the flag is
    # gated on enable_spiking_sc, off here) -> the GREEN_INERT nav byte-identity is preserved. It takes
    # effect on the spiking-SC merged path (MergedNavConvAgent(nav_critic_spiking_sc=True)), making the
    # log-polar render the merged-nav default there.
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
