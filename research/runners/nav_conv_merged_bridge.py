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


def role_of_on_slices(bridge, conj_arr, role_arr, position: int, voice="active",
                      test_steps: int = 80, drive: float = 2500.0, reset: int = PARSER_READ_SETTLE):
    """Port of BridgeParser.role_of: drive the (position, voice) conjunction ALONE; the role ensemble that
    fires most is the learned role. `reset` is the pre-read quiescence window (design risk 4.3): a longer settle
    self-quiesces the merged bridge from any prior drive before the read, so the WTA readout is stable."""
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
    return max(rates, key=rates.get)


def parse_on_slices(bridge, conj_arr, role_arr, words, voice="active", test_steps: int = 80, drive: float = 2500.0,
                    reset: int = PARSER_READ_SETTLE):
    assert len(words) == 3, "this minimal parser handles 3-word SVO sentences"
    return {role_of_on_slices(bridge, conj_arr, role_arr, pos, voice, test_steps, drive, reset): words[pos]
            for pos in range(3)}


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


# ── the merged nav + parser + dlPFC bridge builder (design §2.5 FINAL FORM) ───────────────────────────────
def build_merged_nav_conv_bridge(seed: int = 42, vocab=None, n_cortex: int = 100):
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
    nav_regions, nav_pathways = build_bg_brain_regions(n_cortex=n_cortex)   # DEFAULT kwargs (construction smoke)
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
    union_regions = list(nav_regions) + list(parser_regions) + list(dlpfc_regions)
    union_pathways = list(nav_pathways) + list(parser_pathways)   # dlPFC loop is hand-built, NOT a pathway

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
    cfg.enable_parameter_heterogeneity = False
    # NMDA on globally; the per-region mask (built at init from the enable_nmda regions) confines it to dlPFC.
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5

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
    return bridge, handles


# ── the EPISODE-path conv finalization (nav gate (a): nav episode runs on the merged bridge) ─────────────────
def conv_extra_regions_pathways(vocab=None):
    """The conversational regions/pathways to APPEND to the navigation lists for the episode-path merge: the
    parser (parse_conj 6, parse_role 3*PARSER_R) + the dlPFC regions (cortex_ctx, dlpfc_wm, both enable_nmda).
    For the NAV GATE the dlPFC regions are present but EDGELESS (the dlpfc_loop is only for `elaborate`, not
    needed for nav-not-regressed), so they are silent during the nav episode. Returns (extra_regions,
    extra_pathways) for `run_moving_goal_episode(extra_regions=, extra_pathways=)`."""
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
    return list(parser_regions) + list(dlpfc_regions), list(parser_pathways)


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
    # per-data-position (pre, post), guaranteed aligned with cp_connections.data / cp_plasticity_rate_gain
    counts = xp.diff(csr.indptr)
    pre = xp.repeat(xp.arange(csr.shape[0], dtype=csr.indices.dtype), counts)
    post = csr.indices
    pc = xp.asarray(rm.indices("parse_conj"), dtype=pre.dtype)
    prr = xp.asarray(rm.indices("parse_role"), dtype=pre.dtype)
    parser_mask = xp.isin(pre, pc) & xp.isin(post, prr)

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

    def __init__(self, seed=42, vocab=None):
        """Build the merged nav+parser+dlPFC bridge + the separate RFPhasorComposer (same seed + vocab). The composer's
        vocab is the merged dlPFC vocab (the sorted probe vocab) so the dialogue-planning assemblies and the fact-memory
        codebook share one word set."""
        self.seed = int(seed)
        self._merged_bridge, self._handles = build_merged_nav_conv_bridge(seed=seed, vocab=vocab)
        words = self._handles["vocab"]   # the sorted merged vocab (the dlPFC + parser word set)

        # The production composer on its OWN bridge(s) (STEP 2a: separate). Same seed + vocab as the merged dlPFC.
        from research.runners.rf_phasor_composer import RFPhasorComposer
        self.composer = RFPhasorComposer(seed=seed, D=128, vocab=words, period=200)

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


def main():
    ap = argparse.ArgumentParser(description="Nav+Conv merge builder (parser microcheck + construction smoke)")
    ap.add_argument("--microcheck", action="store_true",
                    help="parser-only framework bridge: validate the parser ports onto framework slices (risk 4.1)")
    ap.add_argument("--construction-smoke", action="store_true",
                    help="build the FULL merged nav+parser+dlPFC bridge and assert it is structurally correct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-cortex", type=int, default=100, help="build_bg_brain_regions n_cortex (construction smoke)")
    ap.add_argument("--nav-stub", type=int, default=50)
    ap.add_argument("--ou", type=float, default=20.0,
                    help="OU noise pA for the parser train pass (validated: 20 PASSES, 0=off FAILS — degenerate "
                         "WTA readout; the merge enables OU only for the pass, then restores OU-off for nav)")
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--train-steps", type=int, default=120)
    args = ap.parse_args()
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
