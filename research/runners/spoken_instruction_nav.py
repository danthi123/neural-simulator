"""SPOKEN-INSTRUCTION NAVIGATION — the real one-brain functional-integration milestone.

design: `docs/plans/2026-06-10-functional-integration-one-brain-design.md` §2 (the task), §3 (the synaptic
mechanism), §5 (the anti-cheat controls). Builds on the GO de-risk
`research/runners/funcint_lang_to_action_probe.py` (the GATING mechanism validated: a `command_route`
transmission gate on `language_input -> cortex_{N,E,S,W}`, opened by the parser's action-role FIRING via the
in-substrate `couple_gate_to_indices`). THIS runner replaces the de-risk's hand-wired topographic labelled-line
STAND-IN with the GENUINE LEARNED word->action mapping (brain-based co-firing, the Pulvermüller action-word
somatotopy the project validated as the `language_input -> cortex_X` route), combined with that validated gating,
and runs the full BEHAVIORAL task.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
TERMS (defined once, owner standing requirement — no undefined acronyms)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
- bridge            : one `sim.bridge.SimulationBridge` — a network of spiking neurons stepped by one
                      `_run_one_simulation_step` loop.
- region / slice    : a contiguous block of neuron indices for one function (a `BrainRegion`).
- navigation cascade: the basal-ganglia action-selection circuit `cortex_{N,E,S,W}` -> str_D1 -> gpi -> thal ->
                      motor_{N,E,S,W} -> the spiking winner-take-all (WTA) selection layer `sel_{N,E,S,W}`. The
                      body moves in the cardinal direction whose pool wins (built by
                      `g11_bg_runner.build_bg_brain_regions`). Per-direction disinhibition: a biased `cortex_d`
                      drives `str_D1_d`, which inhibits `gpi_d`, releasing `thal_d` -> `motor_d` fires.
- parser            : the conversational comprehension network (`parse_conj` 6 conjunction units + `parse_role`
                      3*40 role-ensemble neurons split agent|action|patient). It assigns each word of an SVO
                      command ("dog go north") to its grammatical role; for the verb ("go") the ACTION role
                      ensemble fires. (from `nav_conv_merged_bridge`, reuse-by-import.)
- transmission gate : a per-synapse multiplier in [0,1] on a pathway's synaptic CURRENT, set at runtime by
                      `bridge.set_transmission_gate(name, value)`. Pre-wire a route, hold it closed (gate 0 ->
                      no current), open it on command (Logiaco-Abbott-Escola 2021 thalamocortical gating).
- gate-from-firing  : `couple_gate_to_indices(bridge, gate, control_idx, threshold)`: each step the gate opens
                      iff a control population's smoothed firing rate (EMA) exceeds a threshold. IN-SUBSTRATE —
                      no Python reads a value; the firing of one region opens a route into another
                      (`_apply_gate_couplings`, `sim/bridge.py`).
- command_route     : the ONE transmission gate on `language_input -> cortex_{N,E,S,W}`. Closed at rest; coupled
                      to the parser's action-role ensemble firing.
- LEARNED route     : `language_input -> cortex_X`, a real plastic pathway whose direction-selectivity
                      ("north" -> cortex_N) is GROWN by brain-based co-firing training (NOT hand-wired). After
                      training, driving `language_input` with a direction word biases that direction's cortex pool.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
THE TASK (design §2): commanded-goal gridworld.
────────────────────────────────────────────────────────────────────────────────────────────────────────────
The agent is on a grid. The goal direction is NOT rendered into any retina and NOT given as coordinates. Each
PHASE a 3-word instruction ("dog go <direction>") is presented to the conversational channel. The ONLY way the
body can know which way to move is: the parser comprehends the instruction (its action ensemble FIRES) ->
`command_route` opens -> the commanded direction word's LEARNED `language_input -> cortex_{direction}` current
biases the action cascade -> the agent steps. The commanded direction CHANGES across a multi-phase schedule. The
metric is command-following accuracy: the fraction of steps the agent moves in the commanded direction (chance =
0.25 for 4 directions).

THE A/B + CONTROLS (design §5 — the load-bearing validation):
  COUPLED       : gating on, LEARNED route on -> the agent follows the commands (>> chance, tracks the schedule).
  ISOLATED-NAV  : no command / gate held CLOSED -> no goal cue -> the agent does NOT systematically follow any
                  direction (chance). [defeats "the cascade alone solves it"]
  ISOLATED-CONV : the conversational half alone (no cascade / no body) -> it PARSES but produces no movement.
                  [defeats "conversation alone moves the body"]
  LESION        : `command_route` synapses cut -> the agent stops following even with the parser firing (chance).
                  [proves the behavior rides the SYNAPTIC route, not a leak or a Python copy — the primary test,
                   and the one that resolves the standing nav-reward residual: the route must be NECESSARY]
  SCRAMBLE      : the instruction's word->direction mapping is permuted (say "north", the agent should still
                  follow what it COMPREHENDS, i.e. the parser+route move it toward the spoken word). A real
                  instruction-follower's accuracy-vs-commanded tracks the SPOKEN word, not a fixed structural bias.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
BRAIN-BASED-ONLY (owner standing bar): everything between sensation and action is neurons/synapses. Python is
legitimate ONLY for (1) the environment — the grid, the agent position, and presenting the instruction TEXT to
`language_input` (a sensory render) — and (2) the body — moving the agent based on which motor/sel pool fires.
The word->action selectivity is LEARNED (co-firing STDP/Hebbian, not hand-wired). The gate is opened by parser
SPIKES (`couple_gate_to_indices`, a 0/1 gate state from firing, not a value). NO parser-derived value is copied
into any nav drive. NO `sim/` edit (the route, the gate-from-firing coupling, and the episode pieces are all
public APIs / reuse-by-import).
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
# the parser ports (slices/train/read) live on the merge builder; reuse them verbatim.
from research.runners.nav_conv_merged_bridge import (
    PARSER_R, ROLES, PARSER_GATE,
    parser_regions_pathways, _parser_index_arrays, train_parser_on_slices, parse_on_slices,
)
# the in-substrate gate-from-firing coupling (raw-index variant) + the validated pre-warm cap.
from research.runners.unified_brain_bridge import couple_gate_to_indices, ROLE_GATE_PREWARM_CAP_STEPS

# ── constants ────────────────────────────────────────────────────────────────────────────────────────────
DIRECTION_WORDS = {"N": "north", "E": "east", "S": "south", "W": "west"}
# the environment delta per cardinal action (grid: +x = E, +y = S, matching g11's gridworld convention).
ACTION_DELTA = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

COMMAND_GATE = "command_route"                      # the ONE transmission gate on language_input -> cortex_X
LANG_GATE = "language_input_to_cortex"              # the LEARNED route's plasticity gate (open to train, then freeze)

N_LANG_INPUT = 256                                  # language_input region size (matches g11 text-IO default)
N_CORTEX = 100                                      # build_bg_brain_regions n_cortex (25 per action — g11 canon)
LANG_TO_CORTEX_DENSITY = 0.20                       # the route's density (matches g11 text_input_to_cortex_density)
LANG_TO_CORTEX_INIT_W = 2.0                         # non-zero init (g11 canon: dense-then-prune, NOT grow-from-zero)
LANG_TO_CORTEX_JITTER = 0.5

# the direction-word drive (orthogonal codes -> non-overlapping bands per word, maximally separable). This is the
# sensory render: the environment presents the command word as text to language_input.
LANG_DRIVE_PA = 2500.0                              # per-active-neuron language_input drive (composer ROLE_DRIVE scale)
LANG_SPARSITY = 0.1
# the parser-conjunction drive (the command verb at position 1 -> the ACTION role ensemble).
PARSER_DRIVE_PA = 2500.0
ACTION_VERB_CONJ_K = 2                              # position 1 (verb), active voice -> k = 1*2+0 = 2 -> "action" role

# the gate-from-firing coupling (the validated GO-probe values): the parser action ensemble fires burstily
# (sustained mean ~0.017 of the 40 ensemble neurons under continuous verb drive); the default threshold 0.05 sits
# ABOVE that, so we set it BELOW the sustained rate (0.008) + a stickier EMA (alpha 0.2) so the parser's continuous
# comprehension HOLDS the gate open across the readout (comprehend->latch->act). The gate still opens ONLY because
# the parser SPIKES (firing-driven, not hand-set).
COMMAND_GATE_THRESHOLD = 0.008
COMMAND_GATE_ALPHA = 0.2

# the LEARNED-route training (brain-based co-firing — the embodied / Tier-1 / b3 word->action recipe): for each
# direction d, drive language_input(d) + a teacher on cortex_d simultaneously with Hebbian ON; the co-firing grows
# the selective language_input(d)->cortex_d synapses (Pulvermüller action-word somatotopy). Mirrors the parser
# train-pass mechanism (proven on this exact framework bridge).
ROUTE_TRAIN_EPOCHS = 30
ROUTE_TRAIN_STEPS = 60                               # per (word, teacher) co-drive window
ROUTE_TEACHER_PA = 600.0                             # cortex_d teacher current (supervised co-fire label)
ROUTE_HEBBIAN_LR = 0.02                              # the LEARNED-route Hebbian rate (the route grows in 30 epochs)

# the per-decision readout (one nav decision = a stim/readout window like g11's). cortex bias must propagate through
# the cascade to motor/sel; the window must be long enough for the disinhibition to release the winner.
DECISION_PREWARM_STEPS = ROLE_GATE_PREWARM_CAP_STEPS  # comprehend->latch: drive the verb until the gate opens
DECISION_READOUT_STEPS = 120                          # act: hold gate open + drive the word, accumulate cascade firing
# DRAIN: a FULL-QUIESCENCE window (NO drive at all, not even the cascade tonics) BEFORE each decision, to clear the
# sel_X / commit_X accumulator state from the PRIOR decision. WITHOUT this, the accumulator's residual activity
# carries between sequential decisions and the prior winner leaks into the next decision (the diagnosed cause of
# direction-dependent reliability: a fresh-state decision follows the command 6/6, a carried-state one does not).
# Validated sweep: drain 60 + the tonic settle -> 24/24 command-following at seed 42; drain 0 -> the carry-over bug.
DECISION_DRAIN_STEPS = 60
DECISION_SETTLE_STEPS = 30                            # tonic-on settle after the drain (GPi/thal reach steady state)
# the Cisek urgency ramp (g11 N6): a growing action-INDEPENDENT baseline into ALL sel_X. With a CLEAN drained
# accumulator the genuine GPi->thal disinhibition releases the commanded thal -> its sel pool fires WITHOUT urgency
# (the validated sweep: drain 60 + urgency 0 -> 24/24). Urgency added per-pool cross-talk that HURT reliability
# (drain 60 + urgency 180 -> 22/24, drain 120 + urgency 180 -> 11/24), so it is OFF here (the disinhibition alone
# selects). Kept as a tunable for harder/larger cascades. 0 = off.
URGENCY_MAX_PA = 0.0


# ── the merged nav + (language_input LEARNED route) + parser bridge builder ──────────────────────────────────
def _orthogonal_band_excitatory(rm, region_name, cue_idx, n_cues, sparsity):
    """The EXCITATORY global indices of cue_idx's orthogonal band within `region_name`. The band->cortex training
    teacher / drive must hit the band's excitatory neurons (the region is 20% inhibitory; including inhibitory
    neurons would make the word DRIVE partly suppressive). Layout MUST match orthogonal_drive_pattern exactly."""
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


def build_spoken_instruction_bridge(seed: int = 42, isolated_conv: bool = False):
    """Build ONE brain-region-framework `SimulationBridge` holding the navigation cascade + the LEARNED
    `language_input -> cortex_X` route (with the `command_route` transmission gate) + the parser slices.

    isolated_conv=True: build the conversational half WITHOUT the navigation cascade (only language_input + the
    parser), so the ISOLATED-CONV control has a brain that parses but has no body — there is no motor pool to move.

    Sequence (mirrors the validated merge builder + the g11 text-IO block):
      1. nav regions/pathways (build_bg_brain_regions, spiking-WTA readout) + language_input + the
         language_input->cortex_X route (plasticity_gate=language_input_to_cortex, transmission_gate=command_route)
         + the parser regions/pathway. (isolated_conv: nav + the route omitted.)
      2. config (framework on, dt=1, Izhikevich, the 5a clip mitigation stdp_w_max/hebbian_max_weight=400, OU on
         at build for the parser WTA readout, STDP/reward/STP/homeostasis/structural OFF, NMDA OFF). build.
      3. (the transmission gate is registered by the framework from the route pathway's transmission_gate field.)
      4. train the LEARNED route (Hebbian co-firing, gate language_input_to_cortex open), then FREEZE it.
      5. train the parser on the framework slices (Hebbian, OU on), then FREEZE it; set the resting OU-off config.

    Returns (bridge, handles).
    """
    xp, _ = get_backend()

    if isolated_conv:
        nav_regions, nav_pathways = [], []
    else:
        # the full nav cascade with the spiking-WTA selection layer (the brain-based action readout — sel_X is a
        # Wang-2002 NMDA-free accumulator here; we read sel/motor spike counts, not a coordinate argmax). DEFAULT
        # kwargs otherwise (this is the integration substrate, not the flagship nav benchmark).
        nav_regions, nav_pathways = build_bg_brain_regions(
            n_cortex=N_CORTEX, enable_spiking_wta_readout=True)

    # language_input + the LEARNED route language_input -> cortex_X (only when the nav cortex exists).
    lang_region = BrainRegion(name="language_input", n_neurons=N_LANG_INPUT, exc_fraction=0.8,
                              internal_density=0.05, exc_weight_mean=2.0, inh_weight_mean=4.0,
                              weight_jitter=0.2, plastic_internal=True,
                              izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    route_pathways = []
    if not isolated_conv:
        for a in ACTION_NAMES:
            route_pathways.append(RegionPathway(
                from_region="language_input", to_region=f"cortex_{a}",
                density=LANG_TO_CORTEX_DENSITY, weight_mean=LANG_TO_CORTEX_INIT_W,
                weight_jitter=LANG_TO_CORTEX_JITTER, plastic=True,
                plasticity_gate=LANG_GATE,            # opened to TRAIN the mapping, then frozen
                transmission_gate=COMMAND_GATE,       # the parser-firing gate scales this route's CURRENT
            ))

    parser_regions, parser_pathways = parser_regions_pathways(PARSER_R)
    union_regions = list(nav_regions) + [lang_region] + list(parser_regions)
    union_pathways = list(nav_pathways) + list(route_pathways) + list(parser_pathways)

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
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False           # global Hebbian OFF; toggled ON only for the train passes
    cfg.hebbian_learning_rate = 0.005
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_ou_process = True                   # allocate OU state at build (parser + cascade need it)
    cfg.ou_std_current_pA = 20.0
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_nmda = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False, "homeostasis must stay OFF (synaptic-scaling clip foot-gun)"

    rm = bridge.region_manager

    if not isolated_conv:
        assert COMMAND_GATE in bridge._transmission_gate_to_synapses, \
            f"FAIL: '{COMMAND_GATE}' transmission gate not registered (known: " \
            f"{list(bridge._transmission_gate_to_synapses.keys())})"

    # 4) Train the LEARNED route (brain-based co-firing) — only when the nav cortex exists.
    if not isolated_conv:
        _train_learned_route(bridge, seed)

    # 5) Parser train pass on the framework slices (Hebbian ON, STDP/reward OFF, OU=20 already on), then FREEZE.
    conj_arr, role_arr = _parser_index_arrays(bridge, PARSER_R)
    cc = bridge.core_config
    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
             cc.hebbian_learning_rate)
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.hebbian_learning_rate = 0.005
    try:
        train_parser_on_slices(bridge, conj_arr, role_arr)
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
         cc.hebbian_learning_rate) = saved
    cc.enable_ou_process = False                    # resting nav config: OU off (re-enabled per-read below)
    bridge.set_plasticity_gate(PARSER_GATE, 0.0)

    action_block_idx = role_arr["action"]
    handles = {
        "seed": int(seed),
        "isolated_conv": bool(isolated_conv),
        "conj_arr": conj_arr,
        "role_arr": role_arr,
        "action_block_idx": action_block_idx,
        "lang_indices": xp.asarray(np.asarray(list(rm.indices("language_input")), dtype=np.int64)),
    }
    if not isolated_conv:
        handles["cortex_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64))
                                 for a in ACTION_NAMES}
        # the readout pools: prefer the spiking-WTA selection layer sel_X (the brain-based decision), fall back to
        # motor_X. Both are populated by build_bg_brain_regions(enable_spiking_wta_readout=True).
        region_names = set(rm.region_indices_dict())
        if all(f"sel_{a}" in region_names for a in ACTION_NAMES):
            handles["readout_region"] = "sel"
            handles["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                                      for a in ACTION_NAMES}
        else:
            handles["readout_region"] = "motor"
            handles["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"motor_{a}")), dtype=np.int64))
                                      for a in ACTION_NAMES}
        # the cascade pacemaker indices: GPe/GPi/STN/SNc/thal need standing tonic drive each step so the
        # disinhibition readout works (a biased cortex_d -> str_D1_d -| gpi_d releases thal_d -> motor/sel_d).
        # These tonic pacemaker currents are the intrinsic/brainstem drive the nav benchmark also injects
        # (run_moving_goal_episode lines ~4748-4759); they are body/environment scaffolding (NOT the cognitive
        # computation, which is the neural action selection itself). We resolve the pool indices once here.
        region_names = set(rm.region_indices_dict())

        def _ridx(name):
            return xp.asarray(np.asarray(list(rm.indices(name)), dtype=np.int64)) if name in region_names else None
        handles["cascade_tonic"] = []   # list of (indices, tonic_pA) injected EVERY decision step
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
        # the sel_X accumulator pools (for the Cisek urgency ramp that collapses the commit bound over the window).
        handles["sel_all_idx"] = ([xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                                   for a in ACTION_NAMES] if handles["readout_region"] == "sel" else [])
    return bridge, handles


def _train_learned_route(bridge, seed):
    """Grow the direction-selectivity of `language_input -> cortex_X` by BRAIN-BASED co-firing (NOT hand-wired).

    For each direction d, co-drive language_input(d)'s orthogonal band + a teacher current on cortex_d with Hebbian
    ON, so the simultaneously-active pre (the word code) and post (cortex_d) strengthen their connection (the
    Pulvermüller action-word somatotopy / the project's embodied / b3 / Tier-1 word->action recipe). After training,
    driving language_input(d) ALONE biases cortex_d. This is the SAME train-pass mechanism the parser uses (proven
    on this framework bridge). Only the `language_input_to_cortex` gate is open during the pass; the cascade's own
    plastic pathways (cortex->D1 etc.) are held frozen so the route training does not perturb the nav substrate.
    """
    xp, _ = get_backend()
    rm = bridge.region_manager
    n = int(bridge.core_config.num_neurons)
    cc = bridge.core_config

    # the orthogonal band (excitatory) for each direction word + cortex_d targets.
    band_exc = {a: _orthogonal_band_excitatory(rm, "language_input", i, N_ACTIONS, LANG_SPARSITY)
                for i, a in enumerate(ACTION_NAMES)}
    cortex_idx = {a: np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64) for a in ACTION_NAMES}

    # FREEZE everything except the LEARNED route while training it (so cortex->D1 etc. don't drift). The route's
    # gate is `language_input_to_cortex`; the cascade pathways are on their own gates. We open ONLY the route gate
    # and zero a global gain elsewhere via set_plasticity_gate per-gate — but the simplest robust freeze is: set
    # the global plasticity gain to 0, then the route gate (which multiplies on TOP of the global gain) to 1. The
    # route gate is applied as a per-synapse gain; with global gain 0 only the route's synapses learn. (We restore
    # global gain 1 after.)
    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    bridge.set_global_plasticity_gain(0.0)
    bridge.set_plasticity_gate(LANG_GATE, 1.0)
    # also hold the command_route transmission gate OPEN during training so the route's current reaches cortex_d
    # (the teacher drives cortex_d directly, but opening the route lets the word's pre-drive contribute too).
    bridge.set_transmission_gate(COMMAND_GATE, 1.0)

    saved = (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
             cc.hebbian_learning_rate, cc.enable_ou_process, cc.ou_std_current_pA)
    cc.enable_hebbian_learning = True
    cc.enable_stdp = False
    cc.enable_reward_modulation = False
    cc.hebbian_learning_rate = ROUTE_HEBBIAN_LR
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        for _ in range(ROUTE_TRAIN_EPOCHS):
            for a in ACTION_NAMES:
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(DECISION_SETTLE_STEPS):
                    bridge._run_one_simulation_step()
                cur = xp.zeros(n, dtype=xp.float32)
                cur[xp.asarray(band_exc[a])] = LANG_DRIVE_PA          # the word code (pre)
                cur[xp.asarray(cortex_idx[a])] = ROUTE_TEACHER_PA      # the cortex_d teacher (post label)
                bridge.cp_external_input_current[:] = cur
                for _ in range(ROUTE_TRAIN_STEPS):
                    bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
    finally:
        (cc.enable_hebbian_learning, cc.enable_stdp, cc.enable_reward_modulation,
         cc.hebbian_learning_rate, cc.enable_ou_process, cc.ou_std_current_pA) = saved
    # FREEZE the route + restore global plasticity gain; close the command gate (reopened per-decision by the parser).
    bridge.set_plasticity_gate(LANG_GATE, 0.0)
    bridge.set_global_plasticity_gain(1.0)
    bridge.set_transmission_gate(COMMAND_GATE, 0.0)


# ── the command-route gate coupling (the parser action-ensemble firing opens the route) ──────────────────────
def couple_command_gate(bridge, action_block_idx):
    """Couple `command_route` to the parser's ACTION-role ensemble firing (the in-substrate primitive). The action
    block is a sub-range of the `parse_role` region (not its own region), so the index-based `couple_gate_to_indices`
    is used (design §3.2 option ii; the IDENTICAL dict `couple_gate_to_pool` builds). The gate opens ONLY because
    the parser action ensemble SPIKES."""
    # action_block_idx may be a backend (cupy) array; couple_gate_to_indices does np.asarray on it, which fails
    # on cupy. Pass host indices (the coupling re-wraps them in the active backend internally).
    couple_gate_to_indices(bridge, COMMAND_GATE, to_host(action_block_idx),
                           threshold=COMMAND_GATE_THRESHOLD, alpha=COMMAND_GATE_ALPHA)


# ── one nav decision (the parser comprehends -> the LEARNED route biases the cascade -> the body picks a move) ──
def decide_move(bridge, handles, command_dir, parse_first=True, lesion=False):
    """Run ONE navigation decision and return (chosen_action, per-pool readout counts).

    command_dir: the direction the instruction commands (its word is presented to language_input as text).
    parse_first=True  : COMPREHEND->LATCH->ACT — drive the action-verb parser conjunction for a PRE-WINDOW so the
                        parser fires and (via the coupling) OPENS command_route, THEN run the readout window holding
                        the parser conjunction (gate latched) AND the direction word's language_input drive. The
                        cascade's biased winner is the chosen move. The gate opens because the parser SPIKES.
    parse_first=False : the ISOLATED-NAV control — no parser conjunction drive, so the action ensemble never fires,
                        command_route stays CLOSED, the word's route current never reaches cortex. Only the direction
                        word is presented. Expect no systematic following (chance).
    lesion=True       : (caller has zeroed the command_route weights) — the route is cut, so even with the parser
                        firing the word cannot bias cortex. Expect chance.

    The body's move = the cascade pool (sel_X if the spiking-WTA layer exists, else motor_X) with the most spikes
    over the readout window — the brain-based action readout (the disinhibited winner), NOT a coordinate argmax.
    """
    xp, _ = get_backend()
    conj_arr = handles["conj_arr"]
    lang_indices = handles["lang_indices"]
    readout_idx = handles["readout_idx"]
    cascade_tonic = handles.get("cascade_tonic", [])
    sel_all_idx = handles.get("sel_all_idx", [])
    n = int(bridge.core_config.num_neurons)
    cc = bridge.core_config

    def _base_current():
        """A fresh per-step current vector pre-loaded with the cascade pacemaker tonics (GPe/GPi/STN/SNc/thal).
        These keep GPi firing tonically so a biased cortex_d's str_D1_d inhibition RELEASES thal_d (disinhibition)
        — the standing intrinsic drive the nav benchmark also injects (body/environment scaffolding, not cognition)."""
        cur = xp.zeros(n, dtype=xp.float32)
        for ii, pa in cascade_tonic:
            cur[ii] = xp.float32(pa)
        return cur

    # OU on for the decision (the cascade + parser WTA need it; matches g11 / the merge per-read toggle).
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process = True
    cc.ou_std_current_pA = 20.0
    try:
        # DRAIN: FULL quiescence (NO drive, not even the tonics) so the sel_X / commit_X accumulator from the PRIOR
        # decision empties — otherwise its residual activity leaks the prior winner into this decision (the
        # diagnosed direction-dependent-reliability bug). Then SETTLE with the tonics so GPi/thal reach steady state.
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(DECISION_DRAIN_STEPS):
            bridge._run_one_simulation_step()
        for _ in range(DECISION_SETTLE_STEPS):
            bridge.cp_external_input_current[:] = _base_current()
            bridge._run_one_simulation_step()

        # PRE-WINDOW (comprehend->latch): drive the action-verb conjunction (atop the cascade tonics) so the parser
        # fires and the coupling opens command_route. No word drive yet (the gate opening is purely the parser's doing).
        if parse_first:
            for _ in range(DECISION_PREWARM_STEPS):
                cur = _base_current()
                cur[conj_arr[ACTION_VERB_CONJ_K]] = PARSER_DRIVE_PA
                bridge.cp_external_input_current[:] = cur
                bridge._run_one_simulation_step()

        # READOUT WINDOW (act): hold the parser conjunction (keeps the gate latched via the coupling EMA) + the
        # cascade tonics + the Cisek urgency ramp into all sel pools, AND present the direction word's code;
        # accumulate the readout-pool firing (the cascade's biased, disinhibited winner).
        counts = {a: 0 for a in ACTION_NAMES}
        cue_idx = ACTION_NAMES.index(command_dir)
        word_drive = orthogonal_drive_pattern(cue_idx=cue_idx, n_cues=N_ACTIONS, n_neurons=int(lang_indices.size),
                                              drive_max_pA=LANG_DRIVE_PA, sparsity=LANG_SPARSITY)
        word_drive = xp.asarray(word_drive, dtype=xp.float32)
        for s in range(DECISION_READOUT_STEPS):
            cur = _base_current()
            if URGENCY_MAX_PA > 0.0 and sel_all_idx:
                u = xp.float32(URGENCY_MAX_PA * (s / max(1, DECISION_READOUT_STEPS - 1)))
                for sii in sel_all_idx:
                    cur[sii] = cur[sii] + u
            if parse_first:
                cur[conj_arr[ACTION_VERB_CONJ_K]] = PARSER_DRIVE_PA   # keep the action ensemble firing
            cur[lang_indices] = cur[lang_indices] + word_drive        # the command-word code (sensory render)
            bridge.cp_external_input_current[:] = cur
            bridge._run_one_simulation_step()
            firing = bridge.cp_firing_states
            for a in ACTION_NAMES:
                counts[a] += int(to_host(firing[readout_idx[a]].astype(xp.int64).sum()))
        bridge.cp_external_input_current[:] = 0.0
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std

    # the body picks the most-firing direction pool; ties / fully-silent -> a deterministic "no clear move" (None)
    # so a silent decision is NOT scored as a (lucky) correct move.
    mx = max(counts.values())
    if mx <= 0:
        return None, counts
    winners = [a for a in ACTION_NAMES if counts[a] == mx]
    chosen = winners[0] if len(winners) == 1 else None      # a tie is not a clear command-following move
    return chosen, counts


# ── the multi-phase instruction schedule + the per-condition episode ─────────────────────────────────────────
def default_schedule(n_phases=4, seed=42):
    """A multi-phase instruction schedule that CHANGES the commanded direction across phases (so the metric must
    TRACK the changes, not exploit a fixed bias). Deterministic per seed; covers all 4 cardinals when n_phases>=4."""
    rng = np.random.default_rng(seed * 7919 + 13)
    base = list(ACTION_NAMES)
    rng.shuffle(base)
    return [base[i % len(base)] for i in range(n_phases)]


def run_condition(bridge, handles, schedule, decisions_per_phase, condition, scramble_map=None):
    """Run the instruction schedule under one CONDITION and return the per-phase + overall command-following
    accuracy (the fraction of decisions whose chosen move == the COMMANDED direction).

    condition:
      "coupled"      : parser fires (gate opens via coupling) + LEARNED route on -> follow the commands.
      "isolated_nav" : gate held CLOSED (parse_first=False) -> no goal cue -> chance.
      "lesion"       : command_route weights zeroed (caller) + parser fires -> the route is cut -> chance.
      "scramble"     : like coupled, but the WORD presented for commanded direction d is scramble_map[d]'s word
                       (the instruction is permuted). A real instruction-follower moves toward the SPOKEN word, so
                       its accuracy-vs-commanded should COLLAPSE while accuracy-vs-spoken stays high.
    """
    parse_first = condition in ("coupled", "lesion", "scramble")
    per_phase = []
    n_correct_cmd = 0
    n_correct_spoken = 0
    n_total = 0
    for phase_i, command_dir in enumerate(schedule):
        # the word the environment PRESENTS to language_input. "scramble" presents a permuted word.
        spoken_dir = scramble_map[command_dir] if (condition == "scramble" and scramble_map) else command_dir
        phase_correct = 0
        phase_moves = {a: 0 for a in ACTION_NAMES}
        phase_none = 0
        for _ in range(decisions_per_phase):
            chosen, _counts = decide_move(bridge, handles, spoken_dir, parse_first=parse_first)
            n_total += 1
            if chosen is None:
                phase_none += 1
            else:
                phase_moves[chosen] += 1
                if chosen == command_dir:
                    n_correct_cmd += 1
                    phase_correct += 1
                if chosen == spoken_dir:
                    n_correct_spoken += 1
        per_phase.append({
            "phase": phase_i, "commanded": command_dir, "spoken": spoken_dir,
            "correct_vs_commanded": phase_correct, "decisions": decisions_per_phase,
            "moves": phase_moves, "no_clear_move": phase_none,
            "acc_vs_commanded": phase_correct / max(1, decisions_per_phase),
        })
    return {
        "condition": condition,
        "n_total": n_total,
        "acc_vs_commanded": n_correct_cmd / max(1, n_total),
        "acc_vs_spoken": n_correct_spoken / max(1, n_total),
        "per_phase": per_phase,
    }


def _lesion_command_route(bridge):
    """Cut the command_route: zero every command_route synapse's weight in place. The transmission-gate index map
    points at exactly those synapses, so this is a clean, route-specific lesion (no other pathway touched)."""
    xp, _ = get_backend()
    idx = bridge._transmission_gate_indices_gpu[COMMAND_GATE]
    n_lesioned = int(idx.size)
    bridge.cp_connections.data[idx] = xp.asarray(0.0, dtype=bridge.cp_connections.data.dtype)
    return n_lesioned


# ── provenance (anti-cheat §5.2: no Python copies the parsed direction into the nav drive) ───────────────────
def provenance_facts():
    """The structural facts the BRAIN-BASED-ONLY provenance audit records (design §5 anti-cheat 2): the ONLY
    nav-side current write is the orthogonal direction code into language_input (a legitimate sensory render); the
    parser conjunction drive is conversational-side; no parser-DERIVED quantity (a parsed {role: word}) is written
    into any cortex/striatum/motor drive. The cross-region coupling is the firing-driven command_route gate (a 0/1
    gate state, NOT a value) + the LEARNED (trained, not hand-wired) language_input->cortex_X synapses."""
    return {
        "nav_side_current_writes": [
            "language_input <- orthogonal_drive_pattern(command word)  [sensory render: the environment presents "
            "the instruction word as text]",
        ],
        "conv_side_current_writes": [
            "parse_conj[k=2] <- PARSER_DRIVE_PA  [conv-side: the parser comprehending the verb]",
        ],
        "cross_region_coupling": [
            "couple_gate_to_indices(command_route, action_block_idx)  [a 0/1 GATE STATE from the parser action "
            "ensemble FIRING, not a value]",
            "language_input -> cortex_X synapses (LEARNED by co-firing training, transmission_gate=command_route)  "
            "[the word->action selectivity is GROWN, not hand-wired; gated by the firing-driven gate]",
        ],
        "word_to_action_mapping_is_learned_not_handwired": True,
        "no_parser_derived_value_written_to_nav_drive": True,
    }


# ── one seed: COUPLED / ISOLATED-NAV / ISOLATED-CONV / LESION / SCRAMBLE ─────────────────────────────────────
def run_seed(seed, n_phases=4, decisions_per_phase=8, smoke=False):
    xp, backend = get_backend()
    if smoke:
        n_phases, decisions_per_phase = 2, 3
    print(f"\n[spoken-nav] ===== seed {seed} (backend={backend}, phases={n_phases}, "
          f"decisions/phase={decisions_per_phase}) =====")

    schedule = default_schedule(n_phases=n_phases, seed=seed)
    # the scramble map: a derangement of the 4 cardinals (no direction maps to itself), deterministic per seed.
    derange = {"N": "E", "E": "S", "S": "W", "W": "N"}

    # --- the COUPLED brain (full one-brain): nav cascade + LEARNED route + parser ---
    bridge, h = build_spoken_instruction_bridge(seed, isolated_conv=False)
    couple_command_gate(bridge, h["action_block_idx"])
    # sanity: the trained parser actually calls the action verb the action (so the gate's control ensemble is right).
    cc = bridge.core_config
    prev = (cc.enable_ou_process, cc.ou_std_current_pA)
    cc.enable_ou_process, cc.ou_std_current_pA = True, 20.0
    try:
        parse = parse_on_slices(bridge, h["conj_arr"], h["role_arr"], ["dog", "go", "north"], "active")
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev
    parser_ok = parse.get("action") == "go" and parse.get("agent") == "dog"
    print(f"[spoken-nav] parser 'dog go north' -> {parse}  (action==go & agent==dog: {parser_ok})")
    print(f"[spoken-nav] readout layer = {h['readout_region']}_X ; schedule = {schedule}")

    coupled = run_condition(bridge, h, schedule, decisions_per_phase, "coupled")
    print(f"[spoken-nav]  COUPLED      acc-vs-commanded = {coupled['acc_vs_commanded']:.3f}  "
          f"(per-phase: {[round(p['acc_vs_commanded'], 2) for p in coupled['per_phase']]})")
    isolated_nav = run_condition(bridge, h, schedule, decisions_per_phase, "isolated_nav")
    print(f"[spoken-nav]  ISOLATED-NAV acc-vs-commanded = {isolated_nav['acc_vs_commanded']:.3f}  (gate CLOSED)")

    # --- the LESION control: a FRESH brain (clean), command_route weights zeroed, parser fires ---
    bridge_les, h_les = build_spoken_instruction_bridge(seed, isolated_conv=False)
    n_lesioned = _lesion_command_route(bridge_les)
    couple_command_gate(bridge_les, h_les["action_block_idx"])
    lesion = run_condition(bridge_les, h_les, schedule, decisions_per_phase, "lesion")
    print(f"[spoken-nav]  LESION       acc-vs-commanded = {lesion['acc_vs_commanded']:.3f}  "
          f"(command_route cut, {n_lesioned} synapses zeroed, parser firing)")

    # --- the SCRAMBLE control: the COUPLED brain, but the presented word is permuted ---
    scramble = run_condition(bridge, h, schedule, decisions_per_phase, "scramble", scramble_map=derange)
    print(f"[spoken-nav]  SCRAMBLE     acc-vs-commanded = {scramble['acc_vs_commanded']:.3f}  "
          f"acc-vs-SPOKEN = {scramble['acc_vs_spoken']:.3f}  (word permuted; follower tracks the SPOKEN word)")

    # --- the ISOLATED-CONV control: a brain with NO body (no nav cascade) — it parses but cannot move ---
    bridge_conv, h_conv = build_spoken_instruction_bridge(seed, isolated_conv=True)
    conv_region_names = set(bridge_conv.region_manager.region_indices_dict())
    has_body = any(f"motor_{a}" in conv_region_names or f"sel_{a}" in conv_region_names for a in ACTION_NAMES)
    cc2 = bridge_conv.core_config
    prev2 = (cc2.enable_ou_process, cc2.ou_std_current_pA)
    cc2.enable_ou_process, cc2.ou_std_current_pA = True, 20.0
    try:
        conv_parse = parse_on_slices(bridge_conv, h_conv["conj_arr"], h_conv["role_arr"],
                                     ["dog", "go", "north"], "active")
    finally:
        cc2.enable_ou_process, cc2.ou_std_current_pA = prev2
    conv_parses = conv_parse.get("agent") == "dog"
    print(f"[spoken-nav]  ISOLATED-CONV: parses 'dog go north' -> {conv_parse} (agent==dog: {conv_parses}); "
          f"has motor/sel body: {has_body}  (a brain that comprehends but cannot move)")

    chance = 1.0 / N_ACTIONS
    return {
        "seed": int(seed),
        "backend": backend,
        "parser_ok": bool(parser_ok),
        "readout_region": h["readout_region"],
        "schedule": schedule,
        "chance": chance,
        "coupled": coupled,
        "isolated_nav": isolated_nav,
        "lesion": lesion,
        "scramble": scramble,
        "isolated_conv": {
            "parses": bool(conv_parses), "parse": conv_parse, "has_body": bool(has_body),
            "n_motor_or_sel_pools": int(sum(1 for a in ACTION_NAMES
                                            if f"motor_{a}" in conv_region_names or f"sel_{a}" in conv_region_names)),
        },
        "n_lesioned_synapses": int(n_lesioned),
    }


# ── verdict ──────────────────────────────────────────────────────────────────────────────────────────────
def verdict_from(results, coupled_min=0.50, control_max=0.40):
    """GO  : COUPLED follows commands WELL above chance on ALL seeds (acc-vs-commanded >= coupled_min, and clearly
             above each seed's ISOLATED-NAV and LESION), AND every control fails as predicted on ALL seeds:
             ISOLATED-NAV at chance (<= control_max), LESION at chance (<= control_max), ISOLATED-CONV cannot move
             (no body), SCRAMBLE collapses vs-commanded (<= control_max) while tracking the SPOKEN word, parser OK.
       PARTIAL : COUPLED beats the controls on a majority of seeds but the margin is thin or a control leaks on some.
       NEGATIVE: COUPLED does not reliably follow, or a control does not fail (the route is not load-bearing).
    """
    seeds = [r["seed"] for r in results]
    chance = results[0]["chance"]

    coupled_ok = all(r["coupled"]["acc_vs_commanded"] >= coupled_min for r in results)
    coupled_beats_controls = all(
        r["coupled"]["acc_vs_commanded"] >= r["isolated_nav"]["acc_vs_commanded"] + 0.20 and
        r["coupled"]["acc_vs_commanded"] >= r["lesion"]["acc_vs_commanded"] + 0.20
        for r in results)
    isonav_ok = all(r["isolated_nav"]["acc_vs_commanded"] <= control_max for r in results)
    lesion_ok = all(r["lesion"]["acc_vs_commanded"] <= control_max for r in results)
    isoconv_ok = all(not r["isolated_conv"]["has_body"] and r["isolated_conv"]["parses"] for r in results)
    scramble_ok = all(r["scramble"]["acc_vs_commanded"] <= control_max for r in results)
    parser_ok = all(r["parser_ok"] for r in results)

    all_pass = (coupled_ok and coupled_beats_controls and isonav_ok and lesion_ok and
                isoconv_ok and scramble_ok and parser_ok)
    if all_pass:
        v = "GO"
    elif (coupled_beats_controls and (lesion_ok or isonav_ok)):
        v = "PARTIAL"
    else:
        v = "NEGATIVE"
    return {
        "verdict": v,
        "seeds": seeds,
        "chance": chance,
        "coupled_follows_all_seeds": bool(coupled_ok),
        "coupled_beats_controls_all_seeds": bool(coupled_beats_controls),
        "isolated_nav_at_chance_all_seeds": bool(isonav_ok),
        "lesion_at_chance_all_seeds": bool(lesion_ok),
        "isolated_conv_no_body_all_seeds": bool(isoconv_ok),
        "scramble_collapses_vs_commanded_all_seeds": bool(scramble_ok),
        "parser_ok_all_seeds": bool(parser_ok),
        "coupled_acc_per_seed": {r["seed"]: round(r["coupled"]["acc_vs_commanded"], 3) for r in results},
        "isolated_nav_acc_per_seed": {r["seed"]: round(r["isolated_nav"]["acc_vs_commanded"], 3) for r in results},
        "lesion_acc_per_seed": {r["seed"]: round(r["lesion"]["acc_vs_commanded"], 3) for r in results},
        "scramble_acc_vs_commanded_per_seed": {r["seed"]: round(r["scramble"]["acc_vs_commanded"], 3)
                                               for r in results},
        "scramble_acc_vs_spoken_per_seed": {r["seed"]: round(r["scramble"]["acc_vs_spoken"], 3) for r in results},
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
        description="Spoken-instruction navigation: the agent follows a PARSED command to navigate, where the "
                    "only goal signal is the parser's output routed SYNAPTICALLY (the LEARNED language_input-> "
                    "cortex_X route gated by parser firing) into the nav cascade. The one-brain milestone.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-phases", type=int, default=4, help="instruction phases (commanded direction CHANGES each)")
    ap.add_argument("--decisions-per-phase", type=int, default=8, help="nav decisions per instruction phase")
    ap.add_argument("--smoke", action="store_true", help="1-seed short episode (2 phases x 3 decisions)")
    ap.add_argument("--out", type=str, default="research/findings/raw/spoken_instruction_nav.json")
    args = ap.parse_args()

    seeds = args.seeds[:1] if args.smoke else args.seeds
    results = [run_seed(s, n_phases=args.n_phases, decisions_per_phase=args.decisions_per_phase, smoke=args.smoke)
               for s in seeds]
    prov = provenance_facts()
    vd = verdict_from(results)

    print("\n[spoken-nav] ============ VERDICT ============")
    print(f"[spoken-nav] verdict={vd['verdict']}  (chance={vd['chance']:.2f})")
    print(f"[spoken-nav]   COUPLED follows commands (>= {0.50}) all seeds : {vd['coupled_follows_all_seeds']}  "
          f"{vd['coupled_acc_per_seed']}")
    print(f"[spoken-nav]   COUPLED >> ISOLATED-NAV & LESION (margin .20)  : {vd['coupled_beats_controls_all_seeds']}")
    print(f"[spoken-nav]   ISOLATED-NAV at chance (<= .40) all seeds      : {vd['isolated_nav_at_chance_all_seeds']}  "
          f"{vd['isolated_nav_acc_per_seed']}")
    print(f"[spoken-nav]   LESION at chance (<= .40) all seeds            : {vd['lesion_at_chance_all_seeds']}  "
          f"{vd['lesion_acc_per_seed']}")
    print(f"[spoken-nav]   ISOLATED-CONV parses but has NO body all seeds : {vd['isolated_conv_no_body_all_seeds']}")
    print(f"[spoken-nav]   SCRAMBLE collapses vs-commanded all seeds      : {vd['scramble_collapses_vs_commanded_all_seeds']}  "
          f"vs-commanded={vd['scramble_acc_vs_commanded_per_seed']} vs-spoken={vd['scramble_acc_vs_spoken_per_seed']}")
    print(f"[spoken-nav]   provenance: word->action mapping is LEARNED, no Python value copy : "
          f"{prov['word_to_action_mapping_is_learned_not_handwired'] and prov['no_parser_derived_value_written_to_nav_drive']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_ser({"results": results, "provenance": prov, "verdict": vd}), f, indent=2)
    print(f"[spoken-nav] wrote {args.out}")
    raise SystemExit(0 if vd["verdict"] == "GO" else (2 if vd["verdict"] == "PARTIAL" else 1))


if __name__ == "__main__":
    main()
