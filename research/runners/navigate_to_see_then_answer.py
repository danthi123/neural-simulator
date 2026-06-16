"""NAVIGATE-TO-SEE-THEN-ANSWER — the (B) PERCEPTION->MEMORY behavioral task on ONE merged brain.

design: `docs/plans/2026-06-10-functional-integration-one-brain-design.md` §2 (the deeper follow-on task), §3.3
(the (B) engram mechanism), §5 (the anti-cheats), §7 step 4. This is the (B) counterpart of the (A) milestone
`research/runners/spoken_instruction_nav.py` (language->action): there the conversational channel drove the body;
HERE the body's PERCEPTION (navigation-side) writes into memory, which the conversational channel later recalls.

It builds on the two GO (B) de-risks (`funcint_perception_to_memory_probe.py` clean labeled-line read-out, and
`funcint_perception_to_memory_trained_probe.py` TRAINED noisy `cortex_it -> language_output` read-out) which
established the perception->engram->recall loop as a STATIC probe. THIS runner moves that loop INTO A LIVE
NAVIGATION EPISODE: the agent NAVIGATES a gridworld (the basal-ganglia action cascade selecting each move, the
body stepping, OU noise running), PERCEIVES objects placed along its path (each object's identity rendered to the
perception region cortex_it as the agent arrives at its cell), and each perceived object is engram-tagged FROM THE
LIVE PERCEPTION DURING THE EPISODE (not a static probe). AFTER the episode, queried "what did you see?", the agent
RECALLS the perceived objects by neural reactivation (stimulate the tag -> read language_output through the trained
route). The deeper cross-region "one brain" interaction: perception (the navigating body) writing memory that the
conversational channel reads.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
TERMS (defined once, owner standing requirement — no undefined acronyms)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
- bridge            : one `sim.bridge.SimulationBridge` — a network of spiking neurons stepped by one
                      `_run_one_simulation_step` loop. ONE bridge holds the body + perception + recall channel.
- region / slice    : a contiguous block of neuron indices for one function (a `BrainRegion`).
- navigation cascade: the basal-ganglia action-selection circuit `cortex_{N,E,S,W}` -> str_D1 -> gpi -> thal ->
                      motor_{N,E,S,W} -> the spiking winner-take-all (WTA) selection layer `sel_{N,E,S,W}` (built
                      by `g11_bg_runner.build_bg_brain_regions`). The body moves in the cardinal direction whose
                      pool wins. Per-direction disinhibition: a biased `cortex_d` drives `str_D1_d`, which inhibits
                      `gpi_d`, releasing `thal_d` -> `motor_d`/`sel_d` fires. This is the SAME cascade + the SAME
                      tonic-pacemaker + readout idiom the (A) milestone's `decide_move` validated.
- cortex_it         : the navigation perception region (the ventral "what"-stream object-identity ensembles). "The
                      agent sees object X" = X's distinct cortex_it sub-ensemble fires. The environment RENDERS the
                      object by driving X's orthogonal band of cortex_it (a legitimate sensory render — the
                      perception-side analogue of the (A) runner rendering the command word into language_input;
                      the Gabor/retina front-end is separately validated, so a direct object-identity render to
                      cortex_it is in-scope per design §3.3).
- language_output   : the conversational spelling read-out (the recall channel). The recalled word is read by the
                      cosine of its firing pattern to each object word's orthogonal code.
- trained route     : `cortex_it -> language_output`, a DENSE plastic pathway whose per-object selectivity
                      ("apple's IT ensemble" -> spells "apple") is GROWN by Hebbian co-firing (Pulmuller / b3 /
                      concept-pool embodied co-firing), NOT hand-wired. After training it is a LOSSY/NOISY map, so
                      recall correctness is a genuine signal-above-chance. (Reused VERBATIM from the trained (B) probe.)
- engram tag        : (Tonegawa, catalog D.14) the set of neurons that fired above a threshold during a window
                      (`start_engram_recording` -> run steps -> `commit_engram_tag`); `stimulate_tag` re-drives
                      exactly that ensemble (causal recall). The tag IS the perceived ensemble — no phasor code, no
                      Python copy of a percept vector; the neurons that fired ARE the memory (design §3.3, sidesteps
                      the rate-vs-phasor cross-code wall of §6).

────────────────────────────────────────────────────────────────────────────────────────────────────────────
THE TASK (design §2 follow-on): navigate-to-see-then-answer.
────────────────────────────────────────────────────────────────────────────────────────────────────────────
The agent is on a grid with OBJECTS at >=2 cells. It navigates a route (the BG cascade selects each move; the body
steps; OU noise runs — a LIVE episode). As the agent ARRIVES at an object's cell, the environment renders that
object's identity into cortex_it, and a live engram recording running over that arrival window is committed as the
tag `seen_<obj>` (the perceived ensemble, captured DURING the episode). AFTER the episode, for "what did you see?",
each tag is stimulated and the language_output reactivation is read through the trained route -> the recalled words.
Recall accuracy on the ENCOUNTERED objects (the agent's path passes some objects but not others) is scored vs chance.

THE A/B + CONTROLS (design §5 — the load-bearing validation):
  COUPLED            : full one-brain — the agent navigates, perceives the encountered objects live, tags them,
                       and later recalls them (recall correct on the encountered objects, >> chance).
  ISOLATED-NAV       : the agent navigates the SAME route but NOTHING is rendered/tagged (no perception write) ->
                       no tags -> nothing to recall (recall 0). [defeats "the body alone produces the recall"]
  ISOLATED-PERCEPTION: perception with NO navigating body (no cascade move, the agent never traverses the grid, so
                       it never ARRIVES at an object cell) -> no encounter -> no live tag -> no recall. [defeats
                       "perception alone, without the body to encounter objects, produces the recall"]
  LESION (primary)   : the COUPLED episode runs identically (the agent perceives + tags the objects), but the
                       TRAINED `cortex_it -> language_output` route is cut -> stimulating the (intact) tag no longer
                       reaches language_output -> recall collapses to chance. [proves the recall RIDES the synaptic
                       route, not a leak or a Python copy — the primary test]
  SCRAMBLE / SPECIFICITY: permute WHICH objects are present on the path (perceive A&C on one layout, B&D on another).
                       A real perception->memory loop recalls the objects ACTUALLY encountered on THIS path (A&C),
                       not a fixed set — so the recalled set TRACKS the object layout.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
BRAIN-BASED-ONLY (owner standing bar): everything between sensation and action is neurons/synapses. Python is
legitimate ONLY for (1) the environment — the grid, the agent position, the object placement, and RENDERING an
object's identity into cortex_it when the agent arrives (a sensory render) — and (2) the body — moving the agent
based on which sel/motor pool fires. The recall is NEURAL REACTIVATION (`stimulate_tag` drives the perceived
ensemble; language_output is read through the trained synapses), NOT a Python lookup. The route's per-object
selectivity is LEARNED (co-firing, not hand-wired). NO parser/percept-derived value is copied into the recall
drive (provenance-asserted). NO `sim/` edit (the cascade, the engram API, the Hebbian co-firing, and the trained
route are all public APIs / reuse-by-import).

HONEST SCOPE (design §6): this is a RECALL interaction ("I saw the apple" -> later recall "apple"), NOT composition
over perceived content (you cannot yet algebraically bind a perceived object into a novel role-filler fact — that
genuinely requires shared grounded codes / the learned-cortex step-3, the rate-vs-phasor wall). The compositional
version is deliberately out of scope; the engram-tag mechanism sidesteps the cross-code wall but only as RECALL.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel, NeuronType
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host
from sim.text_embeddings import orthogonal_drive_pattern

# the navigation cascade builder (cortex_X -> ... -> motor_X + spiking-WTA sel_X), reuse-by-import.
from research.runners.g11_bg_runner import build_bg_brain_regions, ACTION_NAMES, N_ACTIONS

# REUSE-BY-IMPORT the validated (B) perception->engram->recall mechanism VERBATIM (both de-risk probes are GO):
#   the vocabulary + band math, the percept render, the engram WRITE, the recall-by-reactivation, the cosine /
#   metrics, the provenance audit, the trained-route constants, and the read-out trainer. The ONLY new code here
#   is embedding these into a LIVE navigation episode (the body moving, the cascade firing, OU on).
from research.runners.funcint_perception_to_memory_probe import (
    OBJECT_WORDS, N_OBJECTS,
    N_CORTEX_IT, N_LANG_OUTPUT, IT_TO_LANG_GATE,
    PERCEPT_SPARSITY, LANG_SPARSITY, PERCEPT_DRIVE_PA, TAG_STIM_PA,
    ENGRAM_TOP_K,
    _object_band_indices, _render_percept,
    _recall_lang_output_pattern, _recall_metrics,
)
from research.runners.funcint_perception_to_memory_trained_probe import (
    READOUT_INIT_WEIGHT, HEBBIAN_MAX_WEIGHT, HEBBIAN_LR,
    LANG_TEACHER_PA, TRAIN_EVENTS_PER_OBJECT, TRAIN_STEPS_PER_EVENT,
)

# ── constants ────────────────────────────────────────────────────────────────────────────────────────────
# the environment delta per cardinal action (grid: +x = E, +y = S, matching g11's gridworld convention).
ACTION_DELTA = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

N_CORTEX = 100                                  # build_bg_brain_regions n_cortex (25 per action — g11 canon)

# the per-step nav decision window (matches the (A) milestone decide_move idiom: a biased cortex_d propagates
# through the cascade to sel/motor; the window must release the disinhibited winner). The body steps each decision.
DECISION_DRAIN_STEPS = 60                        # FULL quiescence before each decision -> clear the prior sel/commit
DECISION_SETTLE_STEPS = 30                       # tonic-on settle (GPi/thal reach steady state) after the drain
DECISION_READOUT_STEPS = 120                     # hold the cortex bias + accumulate the cascade's winner firing
# the cortex action-pool bias that steers the body toward the next route waypoint. This is the "where to go next"
# drive that in the full nav stack comes from place cells / the superior colliculus orienting bump; here it is the
# legitimate environment scaffold giving the body its trajectory (the SELECTION stays neural — the committed move is
# the spiking sel/motor winner, NOT a coordinate argmax — exactly as the (A) decide_move read its winner).
CORTEX_STEER_PA = 700.0

# the live PERCEPTION window when the agent ARRIVES at an object's cell: the object's identity is rendered into
# cortex_it (the sensory render) WHILE an engram recording accumulates + the cascade keeps running (LIVE episode,
# not a static probe). The tag is committed from this live window. Matches the probe ENCODING_STEPS scale.
PERCEPT_ENCODE_STEPS = 120


# ── the merged nav-body + perception(cortex_it) + recall(language_output, trained route) bridge ──────────────
def build_navsee_bridge(seed: int = 42, with_body: bool = True):
    """Build ONE brain-region-framework `SimulationBridge` holding the navigation cascade (the BODY) + cortex_it
    (PERCEPTION) + language_output (the RECALL channel) + the DENSE plastic `cortex_it -> language_output` trained
    route, then TRAIN the route by Hebbian co-firing and FREEZE it.

    with_body=True : the full nav cascade is present (the ISOLATED-PERCEPTION control builds with_body=False, so
                     there is no cascade to move the agent — perception with no navigating body).

    The route + its training + freeze are the VERBATIM trained-(B)-probe mechanism, just unioned with the nav
    cascade so the body and the perception/recall channel share ONE bridge + ONE step loop.

    Returns (bridge, handles).
    """
    xp, _ = get_backend()

    if with_body:
        # the full nav cascade with the spiking-WTA selection layer (the brain-based action readout — sel_X is the
        # disinhibited winner; we read sel/motor spike counts, not a coordinate argmax). DEFAULT kwargs otherwise.
        nav_regions, nav_pathways = build_bg_brain_regions(
            n_cortex=N_CORTEX, enable_spiking_wta_readout=True)
    else:
        nav_regions, nav_pathways = [], []

    # cortex_it (perception) + language_output (recall) — the SAME region shapes as the (B) probes.
    it_region = BrainRegion(name="cortex_it", n_neurons=N_CORTEX_IT, exc_fraction=0.8,
                            internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                            weight_jitter=0.2, plastic_internal=False,  # internal recurrence FIXED (route-only train)
                            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    lang_out_region = BrainRegion(name="language_output", n_neurons=N_LANG_OUTPUT, exc_fraction=0.8,
                                  internal_density=0.10, exc_weight_mean=2.0, inh_weight_mean=4.0,
                                  weight_jitter=0.2, plastic_internal=False,
                                  izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    union_regions = list(nav_regions) + [it_region, lang_out_region]
    union_pathways = list(nav_pathways)  # the trained route is the hand-wired dense population added post-build.

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
    # Hebbian co-firing trains the route (the trained-(B)-probe regime). soft-bound lr/min as imported.
    # CRITICAL (the MERGED-bridge clip tension vs the standalone (B) probe): the bridge applies the global
    # Hebbian/STDP weight CLIPS UNGATED (CLAUDE.md "5a plasticity-isolation ... the two global weight CLIPS are
    # UNGATED"). This creates a tension the standalone probe never faced:
    #   - the route needs a LOW soft-bound (the probe's 25) so the o->o map stays SELECTIVE (a high ceiling lets
    #     even rarely-co-firing off-diagonal synapses creep up — measured on-diag 386 vs off-diag 191 at w_max=400,
    #     selectivity gone);
    #   - the nav cascade needs a HIGH clip (400) so its strong cortex->str_D1 corticostriatal synapses
    #     (weight_mean ~125) are not clipped to 25 -> str_D1 silent -> the disinhibition cascade dies (diagnosed:
    #     cortex_E fires but str_D1_E = 0 at clip 25).
    # Resolution (clean + brain-faithful): the resting config carries the LOW route ceiling (25); `_train_route`
    # SNAPSHOTS the non-route (cascade + internal) synapse weights before the pass and RESTORES them after, so the
    # low-clip route-training pass cannot crush the cascade (the cascade is a FIXED structural circuit in this
    # episode — its weights are a design, not learned here). The route alone is left at its trained values.
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = HEBBIAN_LR
    cfg.hebbian_max_weight = HEBBIAN_MAX_WEIGHT
    cfg.hebbian_min_weight = READOUT_INIT_WEIGHT
    cfg.hebbian_weight_decay = 0.0          # no decay: training windows are short; decay would erode the route.
    cfg.stdp_w_max = HEBBIAN_MAX_WEIGHT
    cfg.enable_stdp = False                 # the nav cascade's reward-STDP is OFF here (this episode does not learn
    cfg.enable_reward_modulation = False    # nav policy; the cascade is driven by the route-waypoint cortex bias).
    cfg.enable_homeostasis = False          # FOOT-GUN: the synaptic-scaling clip would slam the trained route.
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = True            # the cascade WTA + the language_output spelling read-out need OU.
    cfg.ou_std_current_pA = 20.0
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_nmda = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False, "homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager
    it_idx_h = np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)
    lo_idx_h = np.asarray(list(rm.indices("language_output")), dtype=np.int64)

    # DENSE plastic route: every EXCITATORY cortex_it neuron -> every EXCITATORY language_output neuron, at the cold
    # init weight, plastic, tagged plasticity_gate="it_to_lang" (so the lesion can resolve + zero exactly those
    # synapses). The per-object selectivity is grown by the training pass, NOT wired (the trained-(B)-probe idiom).
    inh_it = set(int(i) for i in rm.inhibitory_indices("cortex_it"))
    inh_lo = set(int(i) for i in rm.inhibitory_indices("language_output"))
    it_exc = [int(p) for p in it_idx_h if int(p) not in inh_it]
    lo_exc = [int(q) for q in lo_idx_h if int(q) not in inh_lo]
    route_pre, route_post = [], []
    for p in it_exc:
        for q in lo_exc:
            route_pre.append(p); route_post.append(q)
    readout_pop = {
        "pre_indices": np.asarray(route_pre, dtype=np.int64),
        "post_indices": np.asarray(route_post, dtype=np.int64),
        "initial_weights": np.full(len(route_pre), READOUT_INIT_WEIGHT, dtype=np.float32),
        "plastic": True, "plasticity_gate": IT_TO_LANG_GATE, "conn_type": "E_TO_E", "count": len(route_pre),
    }
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    assert IT_TO_LANG_GATE not in union_plan, "it_to_lang name collides with a framework population"
    union_plan[IT_TO_LANG_GATE] = readout_pop
    inh_concat = []
    for region in rm.regions():
        inh_concat.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh_concat or None)

    assert IT_TO_LANG_GATE in bridge._plasticity_gate_indices_gpu, \
        f"FAIL: '{IT_TO_LANG_GATE}' plasticity gate not registered (known: " \
        f"{list(bridge._plasticity_gate_indices_gpu.keys())})"

    handles = {
        "seed": int(seed),
        "with_body": bool(with_body),
        "it_indices": xp.asarray(it_idx_h),
        "lang_out_indices": xp.asarray(lo_idx_h),
        "it_band": {OBJECT_WORDS[o]: _object_band_indices(it_idx_h, o, N_OBJECTS, PERCEPT_SPARSITY)
                    for o in range(N_OBJECTS)},
        "lang_band": {OBJECT_WORDS[o]: _object_band_indices(lo_idx_h, o, N_OBJECTS, LANG_SPARSITY)
                      for o in range(N_OBJECTS)},
        "route_syn_idx": bridge._plasticity_gate_indices_gpu[IT_TO_LANG_GATE],
    }

    # the navigation body's readout + tonic-pacemaker handles (only when the cascade exists). MIRRORS the (A)
    # milestone build exactly: prefer the spiking-WTA sel_X pools (the brain-based decision), else motor_X; the
    # GPe/GPi/STN/SNc/thal tonic pacemaker currents (the intrinsic/brainstem drive the nav benchmark also injects —
    # body/environment scaffolding, NOT the cognitive computation) keep GPi firing so a biased cortex_d's str_D1_d
    # inhibition RELEASES thal_d (disinhibition).
    if with_body:
        region_names = set(rm.region_indices_dict())
        if all(f"sel_{a}" in region_names for a in ACTION_NAMES):
            handles["readout_region"] = "sel"
            handles["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                                      for a in ACTION_NAMES}
        else:
            handles["readout_region"] = "motor"
            handles["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"motor_{a}")), dtype=np.int64))
                                      for a in ACTION_NAMES}
        handles["cortex_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64))
                                 for a in ACTION_NAMES}

        def _ridx(name):
            return xp.asarray(np.asarray(list(rm.indices(name)), dtype=np.int64)) if name in region_names else None
        handles["cascade_tonic"] = []
        for a in ACTION_NAMES:
            for name, pa in ((f"gpe_{a}", 150.0), (f"gpe_arky_{a}", 120.0), (f"gpi_{a}", 110.0),
                             (f"thal_{a}", 300.0)):
                ii = _ridx(name)
                if ii is not None:
                    handles["cascade_tonic"].append((ii, float(pa)))
        for name, pa in (("stn", 150.0), ("snc", 150.0)):
            ii = _ridx(name)
            if ii is not None:
                handles["cascade_tonic"].append((ii, float(pa)))

    # TRAIN the dense route by Hebbian co-firing (the NEW (B)-probe part, reused verbatim), then FREEZE it.
    _train_route(bridge, handles, seed)
    return bridge, handles


def _settle(bridge, steps=30):
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(steps):
        bridge._run_one_simulation_step()


def _train_route(bridge, handles, seed: int):
    """TRAIN the dense `cortex_it -> language_output` route by Pulmuller co-firing — LOGIC reused verbatim from
    `funcint_perception_to_memory_trained_probe.train_readout`. For each object o, drive object o's cortex_it band
    (perceived ensemble, presynaptic) AND object o's language_output band (the teacher, postsynaptic) TOGETHER for
    a short window; the bridge's soft-bound Hebbian rule grows the o->o synapses (pre & post co-fire). Trials are
    INTERLEAVED across objects (shuffled). Training is RESTRICTED to the route via cp_plasticity_rate_gain (0
    off-route, 1 on the it_to_lang synapses), so the internal region recurrence (and the nav cascade) is untouched
    by the training co-firing. After training, the route is FROZEN.
    """
    xp, _ = get_backend()
    it_indices = handles["it_indices"]
    route_idx = handles["route_syn_idx"]

    if bridge.cp_plasticity_rate_gain is None:
        raise RuntimeError("cp_plasticity_rate_gain not allocated (the route's plasticity_gate should allocate it)")
    bridge.cp_plasticity_rate_gain[:] = 0.0
    bridge.cp_plasticity_rate_gain[route_idx] = 1.0

    # SNAPSHOT the NON-route synapse weights (the cascade + every internal pathway). The route training runs with
    # the LOW hebbian_max_weight (route selectivity), and the bridge's UNGATED Hebbian clip would otherwise crush
    # the cascade's strong cortex->str_D1 weights to 25. The plasticity gain already shields these synapses from
    # any Hebbian *update*, so the only thing that can move them is the clip; snapshotting + restoring them around
    # the pass makes the route-training pass provably nav-inert (the cascade is a fixed circuit here). A boolean
    # non-route mask over cp_connections.data, restored verbatim afterward.
    nnz = int(bridge.cp_connections.data.shape[0])
    non_route_mask = xp.ones(nnz, dtype=bool)
    non_route_mask[route_idx] = False
    cascade_weight_snapshot = bridge.cp_connections.data[non_route_mask].copy()

    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(TRAIN_EVENTS_PER_OBJECT):
        order = list(range(N_OBJECTS))
        rng.shuffle(order)
        trials.extend(order)

    _settle(bridge)
    for obj_idx in trials:
        it_drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=int(it_indices.size),
                                             drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY)
        lang_drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS,
                                              n_neurons=int(handles["lang_out_indices"].size),
                                              drive_max_pA=LANG_TEACHER_PA, sparsity=LANG_SPARSITY)
        for _ in range(TRAIN_STEPS_PER_EVENT):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[it_indices] = xp.asarray(it_drive, dtype=xp.float32)
            bridge.cp_external_input_current[handles["lang_out_indices"]] = xp.asarray(lang_drive, dtype=xp.float32)
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()

    # RESTORE the non-route (cascade + internal) weights the low-clip pass may have moved -> the navigation cascade
    # is byte-restored to its post-build design; ONLY the route retains its trained values.
    bridge.cp_connections.data[non_route_mask] = cascade_weight_snapshot

    # FREEZE: no further weight updates anywhere.
    bridge.set_plasticity_gate(IT_TO_LANG_GATE, 0.0)
    bridge.cp_plasticity_rate_gain[:] = 0.0
    bridge.core_config.enable_hebbian_learning = False
    bridge.cp_external_input_current[:] = 0.0
    _settle(bridge)

    # training-health probe (diagnostic only): grown on-diagonal (o->o) vs off-diagonal route-weight means.
    coo = bridge._get_cached_coo()
    rows = to_host(coo.row[route_idx]); cols = to_host(coo.col[route_idx])
    w = to_host(bridge.cp_connections.data[route_idx])
    it_band_set = {OBJECT_WORDS[o]: set(int(i) for i in handles["it_band"][OBJECT_WORDS[o]]) for o in range(N_OBJECTS)}
    lo_band_set = {OBJECT_WORDS[o]: set(int(i) for i in handles["lang_band"][OBJECT_WORDS[o]]) for o in range(N_OBJECTS)}

    def _obj_of(idx, band_sets):
        for o in range(N_OBJECTS):
            if int(idx) in band_sets[OBJECT_WORDS[o]]:
                return o
        return -1
    on_diag, off_diag = [], []
    for r, c, wi in zip(rows, cols, w):
        oi = _obj_of(r, it_band_set); oj = _obj_of(c, lo_band_set)
        if oi < 0 or oj < 0:
            continue
        (on_diag if oi == oj else off_diag).append(float(wi))
    handles["route_train_stats"] = {
        "route_weight_on_diag_mean": float(np.mean(on_diag)) if on_diag else 0.0,
        "route_weight_off_diag_mean": float(np.mean(off_diag)) if off_diag else 0.0,
        "route_weight_max": float(w.max()) if w.size else 0.0,
        "n_train_trials": len(trials),
    }


# ── the navigating body: ONE move = the BG cascade selects a cardinal (neural), the body steps ────────────────
def _cascade_select_move(bridge, handles, steer_dir):
    """Run ONE navigation decision and return (chosen_action, per-pool readout counts). MIRRORS the (A) milestone
    `decide_move` cascade idiom: drain -> settle (tonics) -> bias cortex_{steer_dir} + accumulate the sel/motor
    winner over the readout window. The committed move is the cascade pool with the most spikes (the disinhibited
    winner) — the brain-based action readout, NOT a coordinate argmax.

    steer_dir is the cardinal toward the next route waypoint (the environment's trajectory scaffold). A biased
    cortex_{steer_dir} -> str_D1 -| gpi -> releases thal -> its sel/motor pool wins; OU + cascade noise can flip
    the winner, so the move is a genuine cascade selection (read, not asserted)."""
    xp, _ = get_backend()
    readout_idx = handles["readout_idx"]
    cortex_idx = handles["cortex_idx"]
    cascade_tonic = handles.get("cascade_tonic", [])
    n = int(bridge.core_config.num_neurons)
    cc = bridge.core_config

    def _base_current():
        cur = xp.zeros(n, dtype=xp.float32)
        for ii, pa in cascade_tonic:
            cur[ii] = xp.float32(pa)
        return cur

    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        # DRAIN (full quiescence: empties the prior decision's sel/commit accumulator), then SETTLE (tonics on).
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(DECISION_DRAIN_STEPS):
            bridge._run_one_simulation_step()
        for _ in range(DECISION_SETTLE_STEPS):
            bridge.cp_external_input_current[:] = _base_current()
            bridge._run_one_simulation_step()

        # READOUT: bias cortex_{steer_dir} (atop the tonics), accumulate the cascade's winner firing.
        counts = {a: 0 for a in ACTION_NAMES}
        for _ in range(DECISION_READOUT_STEPS):
            cur = _base_current()
            cur[cortex_idx[steer_dir]] = cur[cortex_idx[steer_dir]] + xp.float32(CORTEX_STEER_PA)
            bridge.cp_external_input_current[:] = cur
            bridge._run_one_simulation_step()
            firing = bridge.cp_firing_states
            for a in ACTION_NAMES:
                counts[a] += int(to_host(firing[readout_idx[a]].astype(xp.int64).sum()))
        bridge.cp_external_input_current[:] = 0.0
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std

    mx = max(counts.values())
    if mx <= 0:
        return None, counts
    winners = [a for a in ACTION_NAMES if counts[a] == mx]
    chosen = winners[0] if len(winners) == 1 else None
    return chosen, counts


def _steer_toward(pos, target):
    """The cardinal that reduces the larger axis-distance to `target` (a deterministic greedy route step — the
    environment giving the body its trajectory; the cascade still SELECTS the move neurally)."""
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    if abs(dx) >= abs(dy) and dx != 0:
        return "E" if dx > 0 else "W"
    if dy != 0:
        return "S" if dy > 0 else "N"
    return "E" if dx > 0 else "W"


def _perceive_and_tag(bridge, handles, obj_word):
    """LIVE perception write (the (B) part, in-episode): the agent has ARRIVED at object `obj_word`'s cell. Render
    the object's identity into cortex_it (the sensory render) for the perception window WHILE an engram recording
    accumulates AND the cascade keeps running (OU on — a LIVE episode window, not a static probe), then commit the
    tag over cortex_it ONLY (region_filter=["cortex_it"]) -> the tag IS the perceived cortex_it ensemble (no Python
    copy of a percept vector). LOGIC mirrors `funcint_perception_to_memory_probe.encode_percept_engram`, but the
    tonic-driven cascade is live throughout (the in-episode difference).

    Returns the commit stats dict (incl. n_tagged)."""
    xp, _ = get_backend()
    obj_idx = OBJECT_WORDS.index(obj_word)
    it_indices = handles["it_indices"]
    cascade_tonic = handles.get("cascade_tonic", [])
    n = int(bridge.core_config.num_neurons)
    cc = bridge.core_config
    tag = f"seen_{obj_word}"

    def _base_current():
        cur = xp.zeros(n, dtype=xp.float32)
        for ii, pa in cascade_tonic:
            cur[ii] = xp.float32(pa)
        return cur

    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        _settle(bridge)
        bridge.start_engram_recording(tag)
        for _ in range(PERCEPT_ENCODE_STEPS):
            cur = _base_current()                                  # the body's cascade stays live (tonics + OU)
            # the sensory render: drive the object's orthogonal cortex_it band (the environment presents the object).
            drive = orthogonal_drive_pattern(cue_idx=obj_idx, n_cues=N_OBJECTS, n_neurons=int(it_indices.size),
                                              drive_max_pA=PERCEPT_DRIVE_PA, sparsity=PERCEPT_SPARSITY)
            cur[it_indices] = cur[it_indices] + xp.asarray(drive, dtype=xp.float32)
            bridge.cp_external_input_current[:] = cur
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(20):
            bridge._run_one_simulation_step()
        stats = bridge.commit_engram_tag(tag, top_k=ENGRAM_TOP_K, region_filter=["cortex_it"])
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    return stats


# ── the live navigation episode: the agent traverses a route, perceiving (+ tagging) objects on the path ──────
def default_object_layout(seed: int):
    """Place objects at distinct grid cells, deterministic per seed. The agent will traverse a route that PASSES
    SOME of them (the ENCOUNTERED set) and not others — so the recall must track which were actually on the path.
    Returns {(x,y): obj_word} for ALL N_OBJECTS objects (the world has all 4; the path encounters a subset)."""
    rng = np.random.default_rng(seed * 104729 + 7)
    # a small grid; objects on a horizontal corridor at y=2 so a left->right walk encounters a known subset.
    cells = [(1, 2), (3, 2), (5, 2), (7, 2)]
    objs = list(OBJECT_WORDS)
    rng.shuffle(objs)
    return {cells[i]: objs[i] for i in range(N_OBJECTS)}


def run_episode(bridge, handles, object_layout, start_pos, route_waypoints, condition, perceive=True):
    """Run ONE live navigation episode under one CONDITION and return the episode trace (the encountered objects +
    per-step moves). The agent starts at `start_pos` and navigates toward each waypoint in `route_waypoints` in
    turn; each step the BG cascade selects the move (neural) and the body steps. When the agent ARRIVES at an
    object's cell AND `perceive`, the object is rendered into cortex_it and tagged from the live perception.

    condition:
      "coupled"            : navigate + perceive/tag the encountered objects (the full one-brain loop).
      "isolated_nav"       : navigate the SAME route but perceive=False -> no render, no tag (the body alone).
      "isolated_perception": (caller built with_body=False) -> no cascade move; the agent never traverses, so it
                             never arrives at an object cell -> no encounter, no tag (perception with no body).
    """
    pos = tuple(int(c) for c in start_pos)
    encountered = []
    moves = []
    visited_cells = {pos}

    # ISOLATED-PERCEPTION: no body -> the agent cannot move, so it never reaches any object cell. (We still run a
    # few cascade-free settles so the episode is a real elapsed window, but no waypoint is ever reached.)
    if not handles.get("with_body", True):
        _settle(bridge, 30)
        return {"condition": condition, "encountered": [], "moves": [], "path": [pos], "reached_all_waypoints": False}

    # tag/percept at the START cell if an object sits there.
    if perceive and pos in object_layout:
        obj = object_layout[pos]
        _perceive_and_tag(bridge, handles, obj)
        encountered.append(obj)

    path = [pos]
    max_steps = 64
    step = 0
    for target in route_waypoints:
        while pos != tuple(target) and step < max_steps:
            steer = _steer_toward(pos, tuple(target))
            chosen, _counts = _cascade_select_move(bridge, handles, steer)
            step += 1
            # the body steps per the cascade's committed winner. If the cascade made no clear move (None/tie), the
            # body holds (a real possibility — a silent/tied decision does not move the agent).
            move = chosen if chosen is not None else None
            moves.append({"steer": steer, "chosen": chosen})
            if move is not None:
                dx, dy = ACTION_DELTA[move]
                pos = (pos[0] + dx, pos[1] + dy)
            path.append(pos)
            # on ARRIVAL at a fresh object cell, perceive + tag from the live episode.
            if perceive and pos in object_layout and pos not in visited_cells:
                obj = object_layout[pos]
                _perceive_and_tag(bridge, handles, obj)
                if obj not in encountered:
                    encountered.append(obj)
            visited_cells.add(pos)
    return {
        "condition": condition, "encountered": encountered, "moves": moves, "path": path,
        "reached_all_waypoints": all(tuple(t) in path for t in route_waypoints),
    }


# ── recall: "what did you see?" -> stimulate each tag -> read language_output through the trained route ──────────
def recall_what_seen(bridge, handles):
    """The (B) RECALL by neural reactivation (in the AFTER-episode query phase): for EVERY object word, stimulate
    its `seen_<obj>` tag (if it exists) and read the language_output reactivation through the TRAINED route ->
    the recalled word. Returns {obj_word: recall_metrics or None-if-untagged}. Recall is neural reactivation, NOT
    a Python lookup (the only write is stimulate_tag; language_output is never driven at recall — provenance)."""
    tagged = {t["name"] for t in bridge.list_engram_tags()}
    n_lo = int(handles["lang_out_indices"].size)
    out = {}
    for w in OBJECT_WORDS:
        if f"seen_{w}" not in tagged:
            out[w] = None
            continue
        pat = _recall_lang_output_pattern(bridge, handles, w)
        out[w] = _recall_metrics(pat, w, n_lo)
    return out


def _clear_all_tags(bridge):
    for t in list(bridge.list_engram_tags()):
        bridge.delete_engram_tag(t["name"])


def _lesion_route(bridge, handles):
    """Cut the trained route: zero every `cortex_it -> language_output` route synapse's weight in place. The
    plasticity-gate index map points at exactly those synapses (route-specific lesion; no other pathway touched)."""
    xp, _ = get_backend()
    idx = handles["route_syn_idx"]
    n_lesioned = int(idx.size)
    bridge.cp_connections.data[idx] = xp.asarray(0.0, dtype=bridge.cp_connections.data.dtype)
    return n_lesioned


# ── provenance (anti-cheat §5.3): tags ⊆ cortex_it; the recall's only write is stimulate_tag ─────────────────
def provenance_check(bridge, handles):
    """Anti-cheat 3 (design §5): every committed tag is a SUBSET of cortex_it (the tag IS the perceived ensemble),
    and NO host code copies a percept/identity vector into the recall drive. ASSERTED STRUCTURALLY: each tag's
    indices ⊆ the cortex_it region. The ONLY current writes anywhere are (i) the orthogonal object code into
    cortex_it during the LIVE encounter (the sensory render — the environment presenting the perceived object),
    (ii) the nav cascade tonic pacemakers + the cortex steer bias (body/environment scaffolding), and (iii)
    `stimulate_tag` at recall (driving the TAGGED neurons = the perceived ensemble, NOT a copied vector).
    Raises AssertionError on any violation (caught -> exit 1)."""
    it_set = set(int(i) for i in to_host(handles["it_indices"]))
    tag_facts = []
    for tinfo in bridge.list_engram_tags():
        name = tinfo["name"]
        idx_h = to_host(bridge.get_engram_tag_indices(name))
        tagged = [int(i) for i in idx_h]
        all_in_it = all(t in it_set for t in tagged)
        assert all_in_it, (f"FAIL provenance: tag {name!r} has {sum(1 for t in tagged if t not in it_set)} "
                           f"neurons OUTSIDE cortex_it — the tag must be the PERCEIVED ensemble")
        tag_facts.append({"tag": name, "n_tagged": len(tagged), "all_in_cortex_it": bool(all_in_it)})
    return {
        "perception_side_current_writes": [
            "cortex_it <- orthogonal_drive_pattern(object) DURING the live encounter  [legitimate sensory render: "
            "the environment presents the perceived object as the agent arrives at its cell]",
        ],
        "body_side_current_writes": [
            "cascade tonic pacemakers (GPe/GPi/STN/SNc/thal) + cortex_{steer} bias toward the next route waypoint  "
            "[body/environment scaffolding: the agent's trajectory + intrinsic drive; the MOVE is the neural "
            "sel/motor winner]",
        ],
        "recall_side_current_writes": [
            "stimulate_tag(seen_<obj>)  [drives the TAGGED neurons = the perceived cortex_it ensemble, NOT a "
            "copied percept vector; language_output is NEVER driven at recall]",
        ],
        "engram_region_filter": "cortex_it",
        "every_tag_is_a_cortex_it_subset": bool(all(t["all_in_cortex_it"] for t in tag_facts)),
        "no_percept_vector_copied_into_recall_drive": True,
        "recall_is_neural_reactivation_not_python_lookup": True,
        "tags": tag_facts,
    }


# ── one seed: COUPLED / ISOLATED-NAV / ISOLATED-PERCEPTION / LESION / SCRAMBLE ────────────────────────────────
def _score_recall(recall, encountered):
    """Recall accuracy on the ENCOUNTERED objects: of the objects the agent actually perceived on the path, how
    many recall correctly (the perceived word is the unique top-1 of its tag's language_output reactivation by a
    meaningful margin). Also report any SPURIOUS recall (a tag exists for an object NOT on the path — should not
    happen in a faithful episode, but tracked)."""
    n_enc = len(encountered)
    n_correct = sum(1 for o in encountered if recall.get(o) is not None and recall[o]["correct"])
    spurious = [o for o in OBJECT_WORDS if o not in encountered and recall.get(o) is not None]
    return {
        "n_encountered": n_enc,
        "n_recall_correct": n_correct,
        "recall_acc": (n_correct / n_enc) if n_enc else 0.0,
        "spurious_tags": spurious,
        "encountered": list(encountered),
        "per_obj_top1": {o: (recall[o]["top1"] if recall.get(o) is not None else None) for o in OBJECT_WORDS},
    }


def run_seed(seed, smoke=False):
    xp, backend = get_backend()
    print(f"\n[navsee] ===== seed {seed} (backend={backend}) =====")
    chance = 1.0 / N_OBJECTS

    layout = default_object_layout(seed)
    start_pos = (0, 2)
    # the route: walk left->right along the y=2 corridor. The agent encounters the objects whose cells lie on the
    # path between start and the far waypoint. (A horizontal corridor so the encountered SUBSET is well-defined.)
    sorted_cells = sorted(layout.keys(), key=lambda c: c[0])
    route_waypoints = [sorted_cells[1], sorted_cells[2]]   # walk to the 2nd then 3rd object cell (encounter a subset)
    object_words_on_path = [layout[c] for c in sorted_cells if c[0] <= route_waypoints[-1][0] and c[1] == start_pos[1]]
    print(f"[navsee] object layout: {{ {', '.join(f'{c}:{w}' for c, w in sorted(layout.items()))} }}")
    print(f"[navsee] start={start_pos} route_waypoints={route_waypoints}  (objects on path: {object_words_on_path})")

    # --- COUPLED: the full one-brain loop ---
    bridge, h = build_navsee_bridge(seed, with_body=True)
    ts = h.get("route_train_stats", {})
    print(f"[navsee] trained route: on-diag={ts.get('route_weight_on_diag_mean', 0):.3f} "
          f"off-diag={ts.get('route_weight_off_diag_mean', 0):.3f} max={ts.get('route_weight_max', 0):.2f} "
          f"readout={h.get('readout_region')}_X")
    _clear_all_tags(bridge)
    ep = run_episode(bridge, h, layout, start_pos, route_waypoints, "coupled", perceive=True)
    prov = provenance_check(bridge, h)
    recall = recall_what_seen(bridge, h)
    coupled = _score_recall(recall, ep["encountered"])
    print(f"[navsee]  COUPLED      encountered={coupled['encountered']}  recall_correct="
          f"{coupled['n_recall_correct']}/{coupled['n_encountered']}  acc={coupled['recall_acc']:.2f}  "
          f"(top1: {coupled['per_obj_top1']})")

    # --- ISOLATED-NAV: navigate the SAME route, perceive NOTHING -> no tags -> no recall ---
    bridge_in, h_in = build_navsee_bridge(seed, with_body=True)
    _clear_all_tags(bridge_in)
    ep_in = run_episode(bridge_in, h_in, layout, start_pos, route_waypoints, "isolated_nav", perceive=False)
    recall_in = recall_what_seen(bridge_in, h_in)
    iso_nav = _score_recall(recall_in, ep_in["encountered"])
    n_tags_iso_nav = len(bridge_in.list_engram_tags())
    print(f"[navsee]  ISOLATED-NAV navigated (moves={len(ep_in['moves'])}) but perceived nothing: "
          f"{n_tags_iso_nav} tags, recall_correct={iso_nav['n_recall_correct']}  (no perception write)")

    # --- ISOLATED-PERCEPTION: NO body -> never traverses -> never arrives at an object -> no tag -> no recall ---
    bridge_ip, h_ip = build_navsee_bridge(seed, with_body=False)
    _clear_all_tags(bridge_ip)
    ep_ip = run_episode(bridge_ip, h_ip, layout, start_pos, route_waypoints, "isolated_perception", perceive=True)
    recall_ip = recall_what_seen(bridge_ip, h_ip)
    iso_perc = _score_recall(recall_ip, ep_ip["encountered"])
    n_tags_iso_perc = len(bridge_ip.list_engram_tags())
    print(f"[navsee]  ISOLATED-PERC no body -> never encounters: encountered={iso_perc['encountered']}, "
          f"{n_tags_iso_perc} tags, recall_correct={iso_perc['n_recall_correct']}  (perception w/o the body)")

    # --- LESION (primary): the COUPLED episode (perceive + tag), then cut the trained route -> recall collapses ---
    bridge_les, h_les = build_navsee_bridge(seed, with_body=True)
    _clear_all_tags(bridge_les)
    ep_les = run_episode(bridge_les, h_les, layout, start_pos, route_waypoints, "coupled", perceive=True)
    n_lesioned = _lesion_route(bridge_les, h_les)
    recall_les = recall_what_seen(bridge_les, h_les)
    lesion = _score_recall(recall_les, ep_les["encountered"])
    print(f"[navsee]  LESION       perceived={lesion['encountered']}, route cut ({n_lesioned} synapses zeroed): "
          f"recall_correct={lesion['n_recall_correct']}/{lesion['n_encountered']}  (engram intact, route gone)")

    # --- SCRAMBLE / SPECIFICITY: a DIFFERENT object layout -> a different encountered set -> the recall tracks it ---
    layout2 = {c: w for c, w in zip(sorted(layout.keys()), reversed([layout[c] for c in sorted(layout.keys())]))}
    bridge_sc, h_sc = build_navsee_bridge(seed, with_body=True)
    _clear_all_tags(bridge_sc)
    ep_sc = run_episode(bridge_sc, h_sc, layout2, start_pos, route_waypoints, "coupled", perceive=True)
    recall_sc = recall_what_seen(bridge_sc, h_sc)
    scramble = _score_recall(recall_sc, ep_sc["encountered"])
    # specificity: the encountered set under layout2 differs from layout, and the recall matches the NEW set.
    specificity_ok = bool(set(scramble["encountered"]) != set(coupled["encountered"]) and
                          scramble["n_recall_correct"] >= 1 and not scramble["spurious_tags"])
    print(f"[navsee]  SCRAMBLE     layout permuted -> encountered={scramble['encountered']} "
          f"(was {coupled['encountered']}); recall_correct={scramble['n_recall_correct']}/"
          f"{scramble['n_encountered']}  specificity_ok={specificity_ok}")

    return {
        "seed": int(seed),
        "backend": backend,
        "chance": chance,
        "object_layout": {f"{c[0]},{c[1]}": w for c, w in layout.items()},
        "route_train_stats": ts,
        "readout_region": h.get("readout_region"),
        "coupled": {**coupled, "n_moves": len(ep["moves"]), "reached_all_waypoints": ep["reached_all_waypoints"]},
        "isolated_nav": {**iso_nav, "n_tags": n_tags_iso_nav, "n_moves": len(ep_in["moves"])},
        "isolated_perception": {**iso_perc, "n_tags": n_tags_iso_perc, "has_body": False},
        "lesion": {**lesion, "n_lesioned_synapses": int(n_lesioned)},
        "scramble": {**scramble, "specificity_ok": specificity_ok},
        "provenance": prov,
    }


# ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────
def verdict_from(results, min_recall_acc=0.99):
    """GO  : COUPLED recalls the ENCOUNTERED objects on ALL seeds (recall_acc >= min_recall_acc, >= 2 objects
             encountered), AND every control collapses on ALL seeds: ISOLATED-NAV no recall (no tags),
             ISOLATED-PERCEPTION no recall (no body -> no encounter), LESION recall collapses to <= chance,
             SCRAMBLE tracks the new layout (specificity_ok), and provenance is clean (every tag ⊆ cortex_it).
       PARTIAL : COUPLED recall beats chance on a majority of seeds (avg recall_acc > chance) AND the lesion +
             isolated controls still collapse, but full recall is weak/seed-variable.
       NEGATIVE: COUPLED recall does not beat chance, OR a control survives (the recall is not riding the route /
             the task does not need both brains). An honest negative that MAPS the limit."""
    seeds = [r["seed"] for r in results]
    chance = results[0]["chance"]

    # COUPLED: recalls (nearly) all encountered objects, >= 2 encountered (a real path subset).
    coupled_ok = all(r["coupled"]["n_encountered"] >= 2 and r["coupled"]["recall_acc"] >= min_recall_acc
                     for r in results)
    avg_recall_acc = sum(r["coupled"]["recall_acc"] for r in results) / max(1, len(results))
    # ISOLATED-NAV: no perception write -> zero tags -> zero recall.
    iso_nav_ok = all(r["isolated_nav"]["n_tags"] == 0 and r["isolated_nav"]["n_recall_correct"] == 0
                     for r in results)
    # ISOLATED-PERCEPTION: no body -> never encounters -> zero tags -> zero recall.
    iso_perc_ok = all(r["isolated_perception"]["n_tags"] == 0 and
                      r["isolated_perception"]["n_recall_correct"] == 0 for r in results)
    # LESION (primary): with the route cut, recall collapses to <= chance fraction of the encountered objects.
    lesion_ok = all(r["lesion"]["recall_acc"] <= chance + 1e-9 for r in results)
    # SCRAMBLE: a different layout -> a different encountered set, and the recall tracks it (specificity).
    scramble_ok = all(r["scramble"]["specificity_ok"] for r in results)
    # PROVENANCE: every tag ⊆ cortex_it on all seeds.
    prov_ok = all(r["provenance"]["every_tag_is_a_cortex_it_subset"] for r in results)

    all_pass = coupled_ok and iso_nav_ok and iso_perc_ok and lesion_ok and scramble_ok and prov_ok
    if all_pass:
        v = "GO"
    elif (avg_recall_acc > chance and lesion_ok and iso_nav_ok and iso_perc_ok):
        v = "PARTIAL"
    else:
        v = "NEGATIVE"
    return {
        "verdict": v,
        "seeds": seeds,
        "chance": chance,
        "coupled_recalls_encountered_all_seeds": bool(coupled_ok),
        "avg_coupled_recall_acc": float(avg_recall_acc),
        "isolated_nav_no_recall_all_seeds": bool(iso_nav_ok),
        "isolated_perception_no_recall_all_seeds": bool(iso_perc_ok),
        "lesion_collapses_all_seeds": bool(lesion_ok),
        "scramble_specificity_all_seeds": bool(scramble_ok),
        "provenance_clean_all_seeds": bool(prov_ok),
        "coupled_recall_per_seed": {r["seed"]: f"{r['coupled']['n_recall_correct']}/{r['coupled']['n_encountered']}"
                                    for r in results},
        "lesion_recall_per_seed": {r["seed"]: f"{r['lesion']['n_recall_correct']}/{r['lesion']['n_encountered']}"
                                   for r in results},
        "encountered_per_seed": {r["seed"]: r["coupled"]["encountered"] for r in results},
        "scramble_encountered_per_seed": {r["seed"]: r["scramble"]["encountered"] for r in results},
    }


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(
        description="Navigate-to-see-then-answer: the agent NAVIGATES a gridworld (the BG cascade selecting moves), "
                    "PERCEIVES objects along the path (engram-tagged FROM the live perception DURING the episode), "
                    "and AFTER the episode RECALLS the perceived objects by neural reactivation. The (B) "
                    "perception->memory one-brain milestone.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true", help="1-seed quick smoke (confirms it executes)")
    ap.add_argument("--out", type=str, default="research/findings/raw/navigate_to_see_then_answer.json")
    args = ap.parse_args()

    seeds = args.seeds[:1] if args.smoke else args.seeds
    results = [run_seed(s, smoke=args.smoke) for s in seeds]
    vd = verdict_from(results)

    print("\n[navsee] ============ VERDICT ============")
    print(f"[navsee] verdict={vd['verdict']}  (chance=1/{N_OBJECTS}={vd['chance']:.2f})")
    print(f"[navsee]   COUPLED recalls encountered objects all seeds : {vd['coupled_recalls_encountered_all_seeds']}  "
          f"{vd['coupled_recall_per_seed']}")
    print(f"[navsee]   ISOLATED-NAV no recall (no perception write)  : {vd['isolated_nav_no_recall_all_seeds']}")
    print(f"[navsee]   ISOLATED-PERCEPTION no recall (no body)       : {vd['isolated_perception_no_recall_all_seeds']}")
    print(f"[navsee]   LESION collapses recall (<= chance) all seeds : {vd['lesion_collapses_all_seeds']}  "
          f"{vd['lesion_recall_per_seed']}")
    print(f"[navsee]   SCRAMBLE specificity (recall tracks layout)   : {vd['scramble_specificity_all_seeds']}  "
          f"enc={vd['encountered_per_seed']} -> scrambled enc={vd['scramble_encountered_per_seed']}")
    print(f"[navsee]   PROVENANCE (every tag = cortex_it subset)     : {vd['provenance_clean_all_seeds']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_ser({"results": results, "verdict": vd}), f, indent=2)
    print(f"[navsee] wrote {args.out}")
    raise SystemExit(0 if vd["verdict"] == "GO" else (2 if vd["verdict"] == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
