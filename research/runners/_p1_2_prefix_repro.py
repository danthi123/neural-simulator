"""P1.2 GNW workspace + deliberation-loop de-risk — the REASONING integrator.

Consolidate GNW rungs 1-4 into ONE persistent Global Neuronal Workspace region and add a re-entrant
PROPOSE -> EVALUATE -> COMMIT loop, so a never-told multi-hop conclusion EMERGES from the workspace iterating on
ITSELF (spiking re-entry), NOT from a host for-loop calling query_chain. Roadmap §2.7 / Stage-3 / Track C.

MECHANISM (Dehaene GNW ignition + re-entry; Wang-attractor accumulation; reuse-by-import, NO `sim/` edit):
  ONE persistent `workspace` region = N candidate ASSEMBLIES (dense self-recurrent NMDA loops at attractor_weight=30,
  the validated rung-1 ignition recipe) sharing ONE inhibitory `workspace_fs` pool (rung-2 mutual inhibition =
  single-content access, Baars "one spotlight"). The deliberation loop, per hop:
    PROPOSE  — the currently-held content x (the spiking read of the last committed winner) cues the composer's
               LEARNED relational read query_patient(x, action) -> the candidate content t (the modular processor;
               distractors are competing candidates).
    EVALUATE — t (strong, salience-biased) + distractors (weak) are DRIVEN into the workspace slots; mutual-inhibition
               WTA + the ignition threshold select ONE winner and SUSTAIN it (Wang-2002 biased competition).
    COMMIT   — the winner IGNITES (all-or-none, sustained); its identity is READ FROM SPIKES (argmax over the
               late-window per-slot rates) and BROADCAST BACK as the next hop's cue. The moat abstains if nothing
               crosses the ignition threshold (or the relational read misses).
  So the intermediate concept of a multi-hop chain is CARRIED across the deliberation by the ignited spiking assembly
  (the loop cursor), not by a python variable — the whole point of the GNW integrator vs. a host for-loop.

CHEAP-FIRST = PER-HOP-RESET (the rung this file GATES on). Between hops the workspace is washed out to the exact
quiescent snapshot (a COMPLETE `_full_snapshot`/`_full_restore` of every `cp_*` dynamical array — rung-1's 7-array
snapshot is insufficient for a RECURRENT-NMDA workspace, see the wash-out note below) so each hop's ignition starts
from a clean slate with NO limit-cycle phase carry-over — exactly how rung-3 chained report==reasoning. This
SIDESTEPS the open risk (below).
The FULLY-CONTINUOUS (no reset between hops = a genuine self-sustaining "train of thought") form is the FOLLOW-ON
rung and is GATED on Rung-2b (an ASYNC rate attractor via heterogeneity+OU-noise + spike-frequency-adaptation
EVICTION), which is NOT yet built — see OPEN RISK.

OPEN RISK (from the rung-2 finding, localized): the ignited state is a SYNCHRONOUS period-3 LIMIT CYCLE, so a
challenger pulse landing on an arbitrary phase of the incumbent's cycle makes CONTINUOUS (no-reset) per-hop
selection PHASE-ERRATIC (clean on only 1/6 seeds in rung-2). The per-hop snapshot-restore wash-out here removes the
phase carry-over -> the cheap-first form is robust; the continuous form needs Rung-2b first.

GO GATE (6-seed 42/43/44/100/101/102): reentrant_3hop_acc >= query_chain_3hop_baseline AND both >> the
spreading-activation memorization floor AND the moat abstains at EACH hop (unstored cue + past-chain-end) AND all
anti-cheats collapse — i.e. a never-told 3-hop conclusion reached by the workspace ITERATING >= the one-shot host
orchestrator, with the no-confab moat preserved, on all 6 seeds.

ANTI-CHEATS (the anti-cheats ARE the result):
  - DOMAIN DISSOCIATION (the keystone): workspace-silence LESION (assembly self-recurrence -> 0) makes the workspace
    unable to ignite/sustain -> the 3-hop workspace-routed DELIBERATION collapses to ~chance, WHILE a single-step
    REFLEX (direct query_patient, a peripheral path that never routes state through the workspace) SURVIVES. Multi-step
    dies, single-step lives.
  - MULTI-CYCLE not one-shot: cap the loop at 1 cycle -> only the 1-hop neighbour is reached -> the 3-hop conclusion
    is unreachable (scored vs the 3-hop want -> collapse). Proves the conclusion emerges from ITERATED re-entry.
  - RE-CUE LESION: replace each committed winner with a RANDOM concept before re-cueing -> chain collapses to chance
    (the broadcast-back re-entry is load-bearing).
  - PERMUTED-PREMISES: re-store facts under scrambled patient assignment -> the 3-hop chase collapses (role structure,
    not co-occurrence).
  - SPREADING-ACTIVATION FLOOR (the mechanism the 2026-05-14 transitive-inference RETRACTION rode): undirected
    co-occurrence diffusion stays at chance; the re-entrant chase must BEAT it.
  - AFFECT-directedness (cheap-first salience-scalar STAND-IN; FULL form gated on P0.3 affect-state region): at a
    BRANCH (a concept with a value-relevant eat-relation AND a non-value play-relation), a host-set salience scalar
    biases the value candidate's drive -> deliberation preferentially pursues the value chain (DIRECTED). Removing the
    salience (affect-lesion) leaves deliberation RUNNING (still ignites A conclusion) but NON-directed (~chance which
    branch) — biases WHICH chain, not WHETHER it deliberates.

Run (CPU cheap-first):
  SIM_BACKEND=numpy python -u -m research.runners._p1_2_workspace_deliberation_loop_derisk \
      --seeds 42 43 44 100 101 102 --D 256 --backend numpy \
      --json research/findings/raw/_p1_2_workspace_deliberation/summary.json
  # 1-seed primitive smoke first:
  SIM_BACKEND=numpy python -u -m research.runners._p1_2_workspace_deliberation_loop_derisk --smoke --seed 42 --D 256
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

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import get_backend, to_host

# reuse-by-import: the rung-1 assembly-loop builder + protocol constants
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS, WS_LOOP_GATE,
)


# ── COMPLETE quiescent wash-out (per-hop snapshot-restore) ───────────────────────────────────────────────────
# rung-1's `_snapshot_state` captured only 7 arrays (v/u/firing + fast conductances). A workspace with RECURRENT
# NMDA (`cp_conductance_g_nmda_recurrent`/`_rise`) + per-synapse pulse timers (`cp_synapse_pulse_timers`/`_progress`)
# has MORE dynamical state, and a mid-flight limit cycle leaks those un-restored buffers into the next trial ->
# intermittent basin over-drive (identical drive -> non-deterministic ignition). Snapshotting EVERY `cp_*` device
# array (33 of them; connectivity/weight/mask arrays are constant here so restoring them is a safe no-op) makes the
# per-hop reset a BYTE-IDENTICAL return to the fresh quiescent substrate == an order-invariant fresh bridge. Verified:
# 8/8 repeated identical drives -> identical rates. This is a runner-side wash-out (NO `sim/` edit).
def _full_snapshot(bridge):
    snap = {}
    for k, v in vars(bridge).items():
        if k.startswith("cp_") and hasattr(v, "copy") and hasattr(v, "shape"):
            snap[k] = v.copy()
    return snap


def _full_restore(bridge, snap):
    for k, arr in snap.items():
        getattr(bridge, k)[:] = arr
# reuse-by-import: the held-out CHAINS fixtures + fact store + spreading-activation floor control + relations
from research.runners._phaseB_multihop_query_chain_derisk import (
    CHAINS, EAT, PLAY, build_vocab, store_facts, spreading_predict,
)
from research.runners.rf_phasor_composer import RFPhasorComposer


# ── workspace geometry: K candidate assemblies sharing one FS pool (single-content mutual exclusion) ─────────
K_SLOTS = 4                    # candidate slots per hop (WTA chance 1/K = 0.25; false-belief K_LOC precedent)
ASSEMBLY_SIZE = 80             # per slot (rung-1 constant)
WORKSPACE_N = K_SLOTS * ASSEMBLY_SIZE + 20
WORKSPACE_FS_N = 50
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0         # rung-2 mutual-inhibition strength (WTA between simultaneously-driven slots)

IGNITE_PA = 2500.0             # target drive (ignites its slot; rung-2 drive_inc)
DISTRACTOR_FRAC = 0.30         # competing-distractor drive as a fraction of the target drive (loses the WTA)

IGNITE_FRAC = 0.5              # a slot is "ignited" iff its late-window rate >= IGNITE_FRAC * SOLO_PLATEAU
SOLO_PLATEAU = 1.0 / 3.0       # the rung-1 ignited period-3 limit-cycle rate


def build_workspace_bridge(seed: int, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                           lesion: bool = False, n_slots: int = K_SLOTS):
    """ONE persistent GNW `workspace` region: `n_slots` dense self-recurrent NMDA assemblies sharing one inhibitory
    `workspace_fs` pool (mutual inhibition = single-content access). `lesion=True` zeroes the assembly self-recurrence
    (the workspace-silence anti-cheat: it can no longer ignite/sustain). Follows the validated build_competitive_bridge
    pattern (rung-2). Returns (bridge, xp, slots_dev, snapshot)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    workspace_fs = BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0,
                               internal_density=0.0, enable_nmda=False)
    regions = [workspace, workspace_fs]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                       # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False             # FOOT-GUN: synaptic-scaling clip slams the frozen attractor weights
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    # cheap-first per-hop-reset: deterministic (like rung-1) — the snapshot-restore wash-out gives a clean ignition
    # each hop, so the async-attractor levers (heterogeneity + OU noise) are NOT needed here (they are the Rung-2b
    # prerequisite for the CONTINUOUS no-reset form).
    cfg.enable_ou_process = False
    cfg.enable_parameter_heterogeneity = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = rm.indices("workspace")
    slots = [np.asarray(ws[i * ASSEMBLY_SIZE:(i + 1) * ASSEMBLY_SIZE], dtype=np.int64) for i in range(n_slots)]

    eff_weight = 0.0 if lesion else float(attractor_weight)
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    for i, s in enumerate(slots):
        key = f"workspace_loop_{i}"
        assert key not in union_plan
        union_plan[key] = _build_assembly_loop_population(s, eff_weight)   # gated by WS_LOOP_GATE (frozen below)

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _full_snapshot(bridge)

    slots_dev = [xp.asarray(s) for s in slots]
    return bridge, xp, slots_dev, snap


def _ignite_and_read(bridge, xp, slots_dev, snap, drives):
    """One EVALUATE/COMMIT: restore quiescence (wash-out) -> drive each slot at `drives[i]` pA for DRIVE_STEPS ->
    free run FREE_STEPS -> return the late-window per-slot mean firing rate (the sustained-ignition read)."""
    bridge.cp_external_input_current[:] = 0.0
    _full_restore(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        for s_dev, d in zip(slots_dev, drives):
            if d > 0.0:
                bridge.cp_external_input_current[s_dev] = xp.float32(d)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    counts = [0] * len(slots_dev)
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            for i, s_dev in enumerate(slots_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[s_dev].astype(xp.float64).sum()))
    denom = float((FREE_STEPS - late_start) * ASSEMBLY_SIZE)
    return [c / denom for c in counts]


def _deliberate_hop(bridge, xp, slots_dev, snap, target, distractors):
    """PROPOSE-result `target` (from query_patient) -> slot 0 (strong); `distractors` -> slots 1.. (weak). EVALUATE
    via WTA ignition; COMMIT = the spiking winner (argmax late-rate) IF it crossed the ignition threshold, else
    None (abstain). Returns (committed_concept|None, assignment, rates, winner, n_ignited)."""
    n = len(slots_dev)
    assignment = {0: target}
    drives = [0.0] * n
    drives[0] = IGNITE_PA
    slot = 1
    for d in distractors:
        if slot >= n:
            break
        assignment[slot] = d
        drives[slot] = IGNITE_PA * DISTRACTOR_FRAC
        slot += 1

    rates = _ignite_and_read(bridge, xp, slots_dev, snap, drives)
    winner = int(np.argmax(rates))
    thr = IGNITE_FRAC * SOLO_PLATEAU
    ignited = rates[winner] >= thr
    n_ignited = int(sum(1 for r in rates if r >= thr))
    committed = assignment.get(winner) if ignited else None
    return committed, assignment, rates, winner, n_ignited


def _pick_distractors(all_concepts, exclude, k, rng):
    pool = [c for c in all_concepts if c not in exclude]
    if k <= 0 or not pool:
        return []
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in idx]


def reentrant_chase(bridge, xp, slots_dev, snap, composer, cue, actions, all_concepts, rng,
                    recue_lesion_rng=None, max_cycles=None, return_trace=False):
    """The workspace-carried multi-hop DELIBERATION (the P1.2 loop). x starts at `cue`; each hop PROPOSE (relational
    read on the composer) -> EVALUATE/COMMIT (workspace WTA ignition) -> BROADCAST BACK (x_next = the SPIKING read of
    the committed winner, NOT the python target). Returns the terminal concept (or None = abstain), and optionally a
    per-hop trace. `recue_lesion_rng`: replace each committed winner with a random concept before re-cueing (anti-cheat).
    `max_cycles`: cap the number of deliberation cycles (the multi-cycle anti-cheat)."""
    x = cue
    trace = []
    n_hops = len(actions) if max_cycles is None else min(int(max_cycles), len(actions))
    for h in range(n_hops):
        action = actions[h]
        target = composer.query_patient(x, action)                    # PROPOSE (modular relational read)
        if target is None:                                            # no relational neighbour -> moat abstains
            trace.append({"hop": h, "x": x, "target": None, "committed": None, "n_ignited": 0})
            return (None, trace) if return_trace else None
        distractors = _pick_distractors(all_concepts, exclude={target, x}, k=len(slots_dev) - 1, rng=rng)
        committed, assignment, rates, winner, n_ign = _deliberate_hop(
            bridge, xp, slots_dev, snap, target, distractors)
        x_next = committed                                            # BROADCAST BACK: the spike-derived re-cue
        trace.append({"hop": h, "x": x, "target": target, "committed": committed,
                      "winner": int(winner), "n_ignited": int(n_ign), "winner_rate": float(rates[winner])})
        if x_next is None:                                            # workspace failed to ignite -> abstain (lesion)
            return (None, trace) if return_trace else None
        if recue_lesion_rng is not None:                             # sever the broadcast-back re-cue (anti-cheat)
            x_next = all_concepts[int(recue_lesion_rng.integers(len(all_concepts)))]
        x = x_next
    return (x, trace) if return_trace else x


# ── the per-seed experiment ─────────────────────────────────────────────────────────────────────────────────
def run_seed(seed: int, D: int, verbose: bool = True):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    edges, cooc = store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]
    n_concepts = len(all_concepts)
    chance = 1.0 / n_concepts
    HOPS = 3

    # persistent workspace bridges (built ONCE per seed): intact + workspace-silence lesion
    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)

    def dist_rng():                                   # fresh, reproducible distractor stream per condition
        return np.random.default_rng(seed * 991 + 7)

    chains3 = [ch for ch in CHAINS if len(ch) > HOPS]  # chains long enough for a 3-hop conclusion
    tot = len(chains3)

    # ── the re-entrant workspace deliberation (INTACT) + the one-shot host baseline + spreading floor ──────────
    reent_ok = qc_ok = spread_ok = 0
    moat_each_hop_ok = True
    for ch in chains3:
        cue, want = ch[0], ch[HOPS]
        term = reentrant_chase(b_i, xp, slots_i, snap_i, composer, cue, [EAT] * HOPS, all_concepts, dist_rng())
        reent_ok += int(term == want)
        qc_ok += int(composer.query_chain(cue, [EAT] * HOPS) == want)
        spread_ok += int(spreading_predict(cooc, cue, HOPS, all_concepts) == want)
    reent_acc = reent_ok / tot
    qc_acc = qc_ok / tot
    spread_floor = spread_ok / tot

    # ── ANTI-CHEAT: workspace-silence LESION -> the 3-hop deliberation collapses ───────────────────────────────
    lesion_ok = 0
    for ch in chains3:
        term = reentrant_chase(b_l, xp_l, slots_l, snap_l, composer, ch[0], [EAT] * HOPS, all_concepts, dist_rng())
        lesion_ok += int(term == ch[HOPS])
    lesion_acc = lesion_ok / tot

    # ── ANTI-CHEAT (the DISSOCIATION keystone): single-step REFLEX survives the lesion (direct query_patient, ────
    #     a peripheral path that never routes state through the workspace) ──────────────────────────────────────
    reflex_ok = 0
    for ch in chains3:
        reflex_ok += int(composer.query_patient(ch[0], EAT) == ch[1])   # 1-hop reflex; workspace-independent
    reflex_acc = reflex_ok / tot

    # ── ANTI-CHEAT: MULTI-CYCLE not one-shot — cap the loop at 1 cycle, score vs the 3-hop want -> collapse ─────
    onecycle_ok = 0
    for ch in chains3:
        term = reentrant_chase(b_i, xp, slots_i, snap_i, composer, ch[0], [EAT] * HOPS, all_concepts, dist_rng(),
                               max_cycles=1)
        onecycle_ok += int(term == ch[HOPS])
    onecycle_acc = onecycle_ok / tot

    # ── ANTI-CHEAT: RE-CUE LESION — random concept substituted before each re-cue -> collapse ──────────────────
    recue_ok = 0
    for ch in chains3:
        term = reentrant_chase(b_i, xp, slots_i, snap_i, composer, ch[0], [EAT] * HOPS, all_concepts, dist_rng(),
                               recue_lesion_rng=np.random.default_rng(seed * 7 + HOPS))
        recue_ok += int(term == ch[HOPS])
    recue_acc = recue_ok / tot

    # ── ANTI-CHEAT: PERMUTED-PREMISES — re-store facts under scrambled patient assignment -> collapse ──────────
    comp_perm = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(comp_perm, CHAINS, permute_relation=True, rng=np.random.default_rng(seed * 101 + 5),
                distractor_rng=np.random.default_rng(seed * 53 + 1))
    perm_ok = 0
    for ch in chains3:
        term = reentrant_chase(b_i, xp, slots_i, snap_i, comp_perm, ch[0], [EAT] * HOPS, all_concepts, dist_rng())
        perm_ok += int(term == ch[HOPS])
    perm_acc = perm_ok / tot

    # ── ANTI-CHEAT: MOAT — unstored cue + past-chain-end must abstain (None) at each hop ──────────────────────
    moat_unstored = reentrant_chase(b_i, xp, slots_i, snap_i, composer, "ball", [EAT] * HOPS, all_concepts, dist_rng())
    overrun_actions = [EAT] * (len(CHAINS[0]) + 2)
    moat_overrun = reentrant_chase(b_i, xp, slots_i, snap_i, composer, CHAINS[0][0], overrun_actions,
                                   all_concepts, dist_rng())
    moat_unstored_abstains = moat_unstored is None
    moat_overrun_abstains = moat_overrun is None
    moat_ok = bool(moat_unstored_abstains and moat_overrun_abstains)

    # ── mutual-exclusion (single-content access) diagnostic: over the intact 3-hop chases, was exactly ONE slot ─
    #     ignited at each committed hop? ──────────────────────────────────────────────────────────────────────
    me_single = me_total = 0
    for ch in chains3:
        _t, tr = reentrant_chase(b_i, xp, slots_i, snap_i, composer, ch[0], [EAT] * HOPS, all_concepts, dist_rng(),
                                 return_trace=True)
        for step in tr:
            if step.get("committed") is not None:
                me_total += 1
                me_single += int(step["n_ignited"] == 1)
    mutual_exclusion_frac = (me_single / me_total) if me_total else 0.0

    # ── ANTI-CHEAT: AFFECT-directedness (cheap-first salience-scalar STAND-IN; full form gated on P0.3) ────────
    #     At each cue there is a BRANCH: a value-relevant candidate (eat-target, down the chain) vs a non-value
    #     candidate (play-target). WITH affect, the salience scalar routes STRONG drive to the value candidate and
    #     weak drive to the non-value -> the deliberation preferentially ignites the value candidate (DIRECTED).
    #     AFFECT-LESION removes the salience differential (BOTH get equal strong drive) -> the WTA STILL commits a
    #     winner (deliberation RUNS) but no longer preferentially the value one (non-directed). Slot assignment is
    #     RANDOMIZED per branch so a slot-position bias can't masquerade as directedness. This biases WHICH chain,
    #     not WHETHER it deliberates. FULL affect wiring (a spiking affect-state region biasing the drive) is the
    #     follow-on, gated on P0.3 _affect_state_region_derisk.py.
    def _branch(value, nonvalue, salient, arng):
        drives = [0.0] * len(slots_i)
        vslot, nslot = (0, 1) if arng.random() < 0.5 else (1, 0)   # randomize positions
        drives[vslot] = IGNITE_PA if salient else IGNITE_PA        # value: strong either way
        drives[nslot] = IGNITE_PA * DISTRACTOR_FRAC if salient else IGNITE_PA   # non-value: weak (salient) / equal (lesion)
        rates = _ignite_and_read(b_i, xp, slots_i, snap_i, drives)
        w = int(np.argmax(rates))
        assign = {vslot: value, nslot: nonvalue}
        won = assign[w] if rates[w] >= IGNITE_FRAC * SOLO_PLATEAU else None
        return won, (rates[w] >= IGNITE_FRAC * SOLO_PLATEAU)
    arng = np.random.default_rng(seed * 313 + 11)
    aff_directed = aff_lesion_directed = aff_lesion_deliberates = 0
    aff_tot = 0
    for ch in chains3:
        cue = ch[0]
        val_t = composer.query_patient(cue, EAT)
        non_t = composer.query_patient(cue, PLAY)
        if val_t is None or non_t is None or val_t == non_t:
            continue
        aff_tot += 1
        won_on, _ = _branch(val_t, non_t, salient=True, arng=arng)
        aff_directed += int(won_on == val_t)
        won_off, delib_off = _branch(val_t, non_t, salient=False, arng=arng)
        aff_lesion_directed += int(won_off == val_t)
        aff_lesion_deliberates += int(delib_off)
    aff_directed_frac = (aff_directed / aff_tot) if aff_tot else float("nan")
    aff_lesion_directed_frac = (aff_lesion_directed / aff_tot) if aff_tot else float("nan")
    aff_lesion_deliberates_frac = (aff_lesion_deliberates / aff_tot) if aff_tot else float("nan")

    result = {
        "seed": int(seed), "D": int(D), "hops": HOPS, "n_concepts": n_concepts, "chance": chance,
        "n_chains": tot, "K_slots": K_SLOTS,
        "reentrant_3hop_acc": reent_acc,
        "query_chain_3hop_acc": qc_acc,
        "spreading_floor": spread_floor,
        "lesion_3hop_acc": lesion_acc,
        "single_hop_reflex_acc": reflex_acc,
        "onecycle_3hop_acc": onecycle_acc,
        "recue_lesion_3hop_acc": recue_acc,
        "permuted_3hop_acc": perm_acc,
        "moat_unstored_abstains": moat_unstored_abstains,
        "moat_overrun_abstains": moat_overrun_abstains,
        "moat_ok": moat_ok,
        "mutual_exclusion_frac": mutual_exclusion_frac,
        "affect_directed_frac": aff_directed_frac,
        "affect_lesion_directed_frac": aff_lesion_directed_frac,
        "affect_lesion_deliberates_frac": aff_lesion_deliberates_frac,
        "affect_n": aff_tot,
    }

    # ── per-seed GO gate ──────────────────────────────────────────────────────────────────────────────────────
    two_chance = 2.0 * chance
    seed_go = bool(
        reent_acc >= qc_acc and                        # workspace iterating >= the one-shot host orchestrator
        reent_acc >= spread_floor + 0.5 and            # and >> the spreading memorization floor
        reent_acc >= 0.75 and                          # a real multi-hop conclusion (not chance)
        lesion_acc <= max(two_chance, 0.10) and        # workspace-silence collapses the deliberation
        reflex_acc >= 0.85 and                         # the single-step reflex survives (the dissociation keystone)
        onecycle_acc <= max(two_chance, 0.10) and      # 1 cycle can't reach the 3-hop conclusion (multi-cycle needed)
        recue_acc <= max(two_chance, 0.10) and         # the broadcast-back re-cue is load-bearing
        perm_acc <= max(two_chance, 0.10) and          # role structure, not co-occurrence
        moat_ok                                        # no-confab preserved at each hop
    )
    result["seed_go"] = seed_go

    if verbose:
        print(f"[p1.2 seed={seed} D={D}] reentrant_3hop={reent_acc:.3f} vs query_chain={qc_acc:.3f} "
              f"(spread_floor={spread_floor:.3f}, chance={chance:.3f})", flush=True)
        print(f"    DISSOCIATION: lesion_3hop={lesion_acc:.3f} (collapse) | single_hop_reflex={reflex_acc:.3f} (survives)",
              flush=True)
        print(f"    anti-cheats: 1cycle={onecycle_acc:.3f} recue_lesion={recue_acc:.3f} permuted={perm_acc:.3f} "
              f"| moat unstored={moat_unstored_abstains} overrun={moat_overrun_abstains}", flush=True)
        print(f"    mutual_exclusion(single-slot)={mutual_exclusion_frac:.3f} | affect directed={aff_directed_frac:.3f} "
              f"lesion_directed={aff_lesion_directed_frac:.3f} lesion_still_deliberates={aff_lesion_deliberates_frac:.3f} "
              f"(n={aff_tot})", flush=True)
        print(f"    seed_GO={seed_go}", flush=True)
    return result


def run_primitive_smoke(seed: int, D: int):
    """The cheapest-first re-entry-PRIMITIVE smoke (1 cycle): ignite the cue's hop-1 target computed by query_patient,
    verify it IGNITES with single-content mutual-exclusion, and verify the workspace-silence lesion kills the re-cue."""
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    all_concepts = [c for ch in CHAINS for c in ch]

    ch = CHAINS[0]
    cue = ch[0]
    hop1 = composer.query_patient(cue, EAT)               # the relational target (should be ch[1])
    print(f"[smoke] cue={cue!r} -> query_patient -> {hop1!r} (want {ch[1]!r})", flush=True)

    # INTACT: drive the hop-1 target (+ distractors) -> it should ignite, single-content mutual-exclusion
    b_i, xp, slots_i, snap_i = build_workspace_bridge(seed, lesion=False)
    rng = np.random.default_rng(seed * 991 + 7)
    distractors = _pick_distractors(all_concepts, exclude={hop1, cue}, k=K_SLOTS - 1, rng=rng)
    committed, assign, rates, winner, n_ign = _deliberate_hop(b_i, xp, slots_i, snap_i, hop1, distractors)
    print(f"[smoke] INTACT rates={[round(r,3) for r in rates]} winner=slot{winner} n_ignited={n_ign} "
          f"committed={committed!r}", flush=True)
    intact_ok = bool(committed == hop1 and n_ign == 1)

    # LESION: same drive on the silenced workspace -> the target must NOT ignite (re-cue killed)
    b_l, xp_l, slots_l, snap_l = build_workspace_bridge(seed, lesion=True)
    c_l, _a, rates_l, w_l, n_l = _deliberate_hop(b_l, xp_l, slots_l, snap_l, hop1, distractors)
    print(f"[smoke] LESION rates={[round(r,3) for r in rates_l]} winner=slot{w_l} n_ignited={n_l} committed={c_l!r}",
          flush=True)
    lesion_kills = bool(c_l is None)

    # REFLEX survives the lesion (direct read, no workspace)
    reflex_ok = bool(composer.query_patient(cue, EAT) == ch[1])

    ok = bool(intact_ok and lesion_kills and reflex_ok and hop1 == ch[1])
    print(f"\n[smoke] PRIMITIVE {'HOLDS' if ok else 'FAILS'}: relational_read_ok={hop1 == ch[1]} "
          f"intact_ignites_single={intact_ok} lesion_kills_recue={lesion_kills} reflex_survives={reflex_ok}",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="P1.2 GNW workspace + deliberation-loop de-risk (per-hop-reset cheap-first).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42, help="single seed (smoke)")
    ap.add_argument("--D", type=int, default=256, help="composer phasor dimension (cheap-first CPU)")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--smoke", action="store_true", help="run the 1-seed re-entry-primitive smoke only")
    ap.add_argument("--json", type=str, default="research/findings/raw/_p1_2_workspace_deliberation/summary.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    n_concepts = len({c for ch in CHAINS for c in ch})
    print(f"[p1.2 workspace-deliberation] {len(CHAINS)} chains | {n_concepts} concepts | chance {1.0/n_concepts:.3f} | "
          f"K_slots={K_SLOTS} D={args.D} backend={args.backend}\n"
          "  cheap-first = PER-HOP-RESET (snapshot-restore wash-out); continuous no-reset form is the follow-on rung "
          "(gated on Rung-2b async attractor).\n", flush=True)

    if args.smoke:
        ok = run_primitive_smoke(args.seed, args.D)
        return 0 if ok else 1

    results = []
    for seed in args.seeds:
        results.append(run_seed(seed, args.D))

    all_go = all(r["seed_go"] for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "runner": "_p1_2_workspace_deliberation_loop_derisk",
        "form": "per-hop-reset (cheap-first)",
        "seeds": list(args.seeds), "D": int(args.D), "backend": args.backend,
        "all_go": all_go, "n_go": n_go, "n_seeds": len(results),
        "mean_reentrant_3hop_acc": mean("reentrant_3hop_acc"),
        "mean_query_chain_3hop_acc": mean("query_chain_3hop_acc"),
        "mean_spreading_floor": mean("spreading_floor"),
        "mean_lesion_3hop_acc": mean("lesion_3hop_acc"),
        "mean_single_hop_reflex_acc": mean("single_hop_reflex_acc"),
        "mean_onecycle_3hop_acc": mean("onecycle_3hop_acc"),
        "mean_recue_lesion_3hop_acc": mean("recue_lesion_3hop_acc"),
        "mean_permuted_3hop_acc": mean("permuted_3hop_acc"),
        "mean_mutual_exclusion_frac": mean("mutual_exclusion_frac"),
        "mean_affect_directed_frac": mean("affect_directed_frac"),
        "mean_affect_lesion_directed_frac": mean("affect_lesion_directed_frac"),
        "mean_affect_lesion_deliberates_frac": mean("affect_lesion_deliberates_frac"),
        "all_moat_ok": all(r["moat_ok"] for r in results),
        "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    verdict = "GO" if all_go else ("PARTIAL" if n_go >= 1 else "NEGATIVE")
    print(f"\n{'='*100}", flush=True)
    print(f"  P1.2 (per-hop-reset) VERDICT: {verdict}  ({n_go}/{len(results)} seeds GO)", flush=True)
    print(f"    reentrant_3hop={summary['mean_reentrant_3hop_acc']:.3f} vs query_chain="
          f"{summary['mean_query_chain_3hop_acc']:.3f} (spread_floor={summary['mean_spreading_floor']:.3f})", flush=True)
    print(f"    lesion={summary['mean_lesion_3hop_acc']:.3f}(collapse) reflex="
          f"{summary['mean_single_hop_reflex_acc']:.3f}(survive) 1cycle={summary['mean_onecycle_3hop_acc']:.3f} "
          f"recue={summary['mean_recue_lesion_3hop_acc']:.3f} perm={summary['mean_permuted_3hop_acc']:.3f}", flush=True)
    print(f"    mutual_exclusion={summary['mean_mutual_exclusion_frac']:.3f} affect_directed="
          f"{summary['mean_affect_directed_frac']:.3f} affect_lesion_directed="
          f"{summary['mean_affect_lesion_directed_frac']:.3f} affect_lesion_deliberates="
          f"{summary['mean_affect_lesion_deliberates_frac']:.3f} moat_all={summary['all_moat_ok']}", flush=True)
    print(f"    [saved] {args.json}\n{'='*100}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
