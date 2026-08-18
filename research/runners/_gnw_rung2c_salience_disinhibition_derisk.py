"""GNW Rung-2c de-risk: SALIENCE-GATED DIS-INHIBITION PULSE eviction on a competitive workspace.

Rung-2b (`_gnw_rung2b_sfa_workspace_eviction_derisk.py`, 6/6 BOUNDARY) proved INTRINSIC Izhikevich
spike-frequency adaptation is NOT the workspace-eviction effector: the fatigue that would evict the dense
recurrent attractor equals the fatigue that kills it (un-evictable hold with a CO-igniting strong challenger, or
SELF-extinction) — no window gives ignite-hold-AND-evictable. It DID prove the continuous no-reset protocol works
(0 `_restore_state` calls). The BOUNDARY named the next mechanism: a TRANSIENT SALIENCE-GATED DIS-INHIBITION PULSE
— a phasic attention-shift RELEASE of the inhibition, driven from OUTSIDE the fatiguing assembly (thalamic-reticular
/ pulvinar attention-gating; Dehaene & Changeux 2011, Neuron 70(2):200-227, metastability: an ignited workspace
state must be able to be "destabilized" and "spontaneously replaced by another" — destabilization driven from
OUTSIDE the incumbent's own recovery current, NOT intrinsic fatigue).

WHY A GLOBAL RELEASE IS INSUFFICIENT (this session's first result, banked): a single SHARED FS pool + a GLOBAL
dis-inhibition release lets the challenger IGNITE but cannot REMOVE the incumbent — both lock at the period-3
plateau (CO-ignition), because symmetric shared inhibition sets a common activity level both tolerate and the
dense weight-30 recurrence overpowers any inhibition still weak enough to permit ignition (fs 16->70 all give
`AAAA222`, never a clean evict; salience_gain barely moves the winner). This reproduces the Rung-2b wall.

THE EFFECTOR (biased-competition dis-inhibition; brain-based, NO `sim/` edit, explicit-wiring only): the workspace
carries two assemblies A, B; each has its OWN inhibitory interneuron pool (fs_A -| A, fs_B -| B), and BOTH
assemblies excite BOTH pools -> CROSS-inhibition (A -> fs_B -| B and B -> fs_A -| A), the Wang-2002 two-attractor
WTA motif that gives real mutual exclusion (unlike one shared pool). A SALIENCE-gated VIP-like disinhibitory pool
per slot (dis_A -| fs_A, dis_B -| fs_B) is driven from OUTSIDE the assemblies by the challenger's salience
(`vip_current = salience_gain * drive_chal`, a phasic pulse of `pulse_duration` steps into the CHALLENGER's dis
pool). Releasing the challenger's slot shields the challenger from inhibition while its now-vigorous firing drives
the INCUMBENT's pool (fs_incumbent) HARD -> the undriven incumbent is suppressed below its self-sustaining
threshold and collapses; the pulse ends, inhibition restores, and the challenger (still driven) holds the workspace
alone. The eviction effector is a current/conductance change gated by an EXTERNAL salience signal, timed as a
PULSE — NOT a host state reset, NOT the incumbent's own recovery current (banked negatives: intrinsic SFA
self-extinguishes; GABA_B killed; STP annihilates; the active-clear FS quench works but is a HOST shortcut — this
is its competitive, salience-DRIVEN, brain-based analogue).

GO GATE — 6 seeds 42/43/44/100/101/102 at ONE FROZEN (salience_gain, dis_to_fs, pulse_duration, fs_to_ws, ou), NO
per-seed tuning, ALL of: mutual exclusion preserved (no co-ignition anywhere in the sweep; must NOT regress);
incumbent HOLDS a sub-crossover weak challenger; MONOTONE salience-graded takeover (single crossover, no reversal);
causal-swap membership (swapping the salient role FLIPS the ignited content, attributable >= ~0.9); post-takeover
n_ignited == 1 (challenger; anti-annihilation, never 0); continuous run with ZERO mid-competition `_restore_state`;
the pulse is salience-driven (`vip_current` scales with salience & is 0 at zero salience). Controls: PULSE-OFF
(salience_gain=0) reproduces the non-eviction negative (the pulse is the effector); WTA-lesion (fs_to_ws=0)
co-ignites (mutual exclusion is inhibition-caused).

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2c_salience_disinhibition_derisk --seed 42 --smoke \
      --json research/findings/raw/_gnw_rung2c_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2c_salience_disinhibition_derisk --seed 42 \
      --salience-gain 1.0 --dis-to-fs 30 --pulse-duration 25 --fs-to-ws 20 --ou-noise 40 \
      --json research/findings/raw/_gnw_rung2c_seed42.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS,
)
from research.runners._gnw_rung2_competitive_access_derisk import (
    _ignited, IGNITE_FRAC, SOLO_PLATEAU, WORKSPACE_N, ASSEMBLY_SIZE, WS_LOOP_A, WS_LOOP_B, WS_LOOP_GATE,
)
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import (
    _late_rate, _winner, _is_monotone, _threshold_hash,
)

# ── geometry: two assemblies + per-slot inhibitory pools (cross-inhibition) + per-slot disinhibitory pools ─────
FS_N = 40                      # per-slot inhibitory interneuron pool (fs_A -| A, fs_B -| B)
DIS_N = 30                     # per-slot VIP-like disinhibitory pool (dis_A -| fs_A, dis_B -| fs_B)
A2FS_WEIGHT = 3.0              # assembly -> fs excitation (BOTH assemblies -> BOTH pools = cross-inhibition)
FS_TO_WS_WEIGHT = 20.0         # fs -> assembly inhibition (the mutual-inhibition strength; the "fs_baseline" knob)
DIS_TO_FS_WEIGHT = 30.0        # dis -| fs (how strongly the VIP pulse suppresses a slot's inhibition)
SALIENCE_GAIN = 1.0            # vip_current = salience_gain * drive_chal (the salience->pulse amplitude gain)
PULSE_DURATION = 25            # steps the challenger's VIP pool is driven (the phasic pulse; < DRIVE_STEPS=35)

# restore-call accounting: the continuous headline MUST make ZERO restore calls (anti-cheat).
_RESTORE_CALLS = {"n": 0}


def _restore_counted(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _restore_state(bridge, snap)


def _dense_pop(pre_idx, post_idx, weight, conn_type, gate=None):
    """A dense population: every pre -> every post at `weight` (no self-pairs when pre==post sets overlap).
    Polarity is set by whether the PRE neurons are in `output_inhibitory_indices` at inject time (conn_type is
    informational). plastic=False, frozen."""
    pre = np.asarray(pre_idx, dtype=np.int64)
    post = np.asarray(post_idx, dtype=np.int64)
    P = np.repeat(pre, post.shape[0])
    Q = np.tile(post, pre.shape[0])
    keep = P != Q
    P = P[keep]; Q = Q[keep]
    ww = np.full(P.shape[0], float(weight), dtype=np.float32)
    spec = {"pre_indices": P.astype(np.int64), "post_indices": Q.astype(np.int64), "initial_weights": ww,
            "plastic": False, "conn_type": conn_type, "count": int(P.size)}
    if gate is not None:
        spec["plasticity_gate"] = gate
    return spec


# ── build: two assemblies + per-slot cross-inhibition + per-slot dis-inhibition (NO sim/ edit; explicit wiring) ─
def build_disinhibition_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                               fs_lesion: bool = False, dis_lesion: bool = False,
                               fs_to_ws: float = FS_TO_WS_WEIGHT, dis_to_fs: float = DIS_TO_FS_WEIGHT,
                               a2fs: float = A2FS_WEIGHT, ou_noise_pA: float = 40.0, heterogeneity: bool = True):
    """One `workspace` region (two dense self-recurrent assemblies A, B), per-slot inhibitory pools fs_A/fs_B
    (each driven by BOTH assemblies -> cross-inhibition; fs_A -| A, fs_B -| B), and per-slot disinhibitory pools
    dis_A/dis_B (dis_A -| fs_A, dis_B -| fs_B). Driving a dis pool externally suppresses that slot's inhibition ->
    RELEASES that slot (dis-inhibition).
      fs_lesion=True  -> fs->assembly weight 0 (WTA anti-cheat; both assemblies co-ignite);
      dis_lesion=True -> dis->fs weight 0 (the VIP pools are wired but decoupled; a structural PULSE-OFF).
    Returns (bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    fs_A = BrainRegion(name="fs_A", n_neurons=FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    fs_B = BrainRegion(name="fs_B", n_neurons=FS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    dis_A = BrainRegion(name="dis_A", n_neurons=DIS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    dis_B = BrainRegion(name="dis_B", n_neurons=DIS_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False)
    regions = [workspace, fs_A, fs_B, dis_A, dis_B]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []      # ALL inter-region wiring is explicit (sub-slice precision for cross-inhibition)
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = rm.indices("workspace")
    A_idx = np.asarray(ws[:ASSEMBLY_SIZE], dtype=np.int64)
    B_idx = np.asarray(ws[ASSEMBLY_SIZE:2 * ASSEMBLY_SIZE], dtype=np.int64)
    fsA = np.asarray(rm.indices("fs_A"), dtype=np.int64)
    fsB = np.asarray(rm.indices("fs_B"), dtype=np.int64)
    disA = np.asarray(rm.indices("dis_A"), dtype=np.int64)
    disB = np.asarray(rm.indices("dis_B"), dtype=np.int64)

    eff_weight = float(attractor_weight)
    fs2ws_eff = 0.0 if fs_lesion else float(fs_to_ws)
    dis2fs_eff = 0.0 if dis_lesion else float(dis_to_fs)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    # the two self-recurrent assemblies (the ignitable attractors).
    union_plan[WS_LOOP_A] = _build_assembly_loop_population(A_idx, eff_weight)
    union_plan[WS_LOOP_B] = _build_assembly_loop_population(B_idx, eff_weight)
    # PURE CROSS-inhibition (Wang-2002 two-attractor WTA): each assembly excites ONLY the OTHER slot's inhibitory
    # pool (A -> fs_B -| B, B -> fs_A -| A). Critically NO self-drive (A -/-> fs_A): if the challenger drove its
    # OWN pool it would swamp the dis-inhibition (measured this session: fs_B rebounds to 0.4 under the pulse when
    # B->fs_B exists), so releasing fs_B could not free B. With pure cross, fs_B is driven by A ALONE, so
    # suppressing fs_B genuinely frees B, and B's firing then drives fs_A -> suppresses (evicts) the incumbent.
    union_plan["A2fsB"] = _dense_pop(A_idx, fsB, a2fs, "E_TO_I")
    union_plan["B2fsA"] = _dense_pop(B_idx, fsA, a2fs, "E_TO_I")
    union_plan["fsA2A"] = _dense_pop(fsA, A_idx, fs2ws_eff, "I_TO_E")
    union_plan["fsB2B"] = _dense_pop(fsB, B_idx, fs2ws_eff, "I_TO_E")
    # per-slot DIS-inhibition: the VIP pool suppresses its slot's inhibitory pool.
    union_plan["disA2fsA"] = _dense_pop(disA, fsA, dis2fs_eff, "I_TO_I")
    union_plan["disB2fsB"] = _dense_pop(disB, fsB, dis2fs_eff, "I_TO_I")

    # polarity: fs_A/fs_B/dis_A/dis_B neurons are inhibitory (their outgoing synapses are GABA); assemblies excite.
    inh = list(fsA) + list(fsB) + list(disA) + list(disB)
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    handles = {"seed": int(seed), "fs_lesion": bool(fs_lesion), "dis_lesion": bool(dis_lesion),
               "attractor_weight": eff_weight, "fs_to_ws": float(fs2ws_eff), "dis_to_fs": float(dis2fs_eff),
               "a2fs": float(a2fs), "ou_noise_pA": float(ou_noise_pA), "heterogeneity": bool(heterogeneity),
               "A_idx": A_idx, "B_idx": B_idx, "n_fs": int(fsA.size), "n_dis": int(disA.size)}
    return bridge, xp, xp.asarray(A_idx), xp.asarray(B_idx), xp.asarray(disA), xp.asarray(disB), snap, handles


def run_incumbency_pulse(bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, drive_inc: float, drive_chal: float,
                         salience_gain: float, pulse_duration: int, challenger_is_B: bool = True,
                         incumbent_settle: int = 120, isolate: bool = True):
    """ONE incumbency competition trial. CONTINUOUS within the competition: NO `_restore_state` between the
    incumbent settling and the challenger arriving. When the challenger arrives, a SALIENCE-GATED external pulse
    `vip_current = salience_gain * drive_chal` is injected INTO the CHALLENGER's dis pool for `pulse_duration`
    steps -> releases the challenger's slot -> the challenger fires hard, drives the incumbent's inhibitory pool,
    the incumbent (no external drive) collapses; the pulse ends, inhibition restores, the challenger holds alone.
      isolate=True  -> `_restore_state(snap)` ONCE at the very start (independent-trial isolation for the sweep);
      isolate=False -> NO restore at all (the fully-continuous headline).
    Returns (late_rate_A, late_rate_B, vip_current)."""
    inc_dev = A_dev if challenger_is_B else B_dev
    chal_dev = B_dev if challenger_is_B else A_dev
    chal_dis_dev = disB_dev if challenger_is_B else disA_dev    # the CHALLENGER's slot dis pool (salience routing)
    vip_current = float(salience_gain) * float(drive_chal)      # the SALIENCE-GATED pulse amplitude (anti-cheat 1)
    pdur = int(pulse_duration)

    bridge.cp_external_input_current[:] = 0.0
    if isolate:
        _restore_counted(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    for _ in range(DRIVE_STEPS):                              # (1) ignite the incumbent alone
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[inc_dev] = xp.float32(drive_inc)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(incumbent_settle):                         # (2) the incumbent holds (no drive, no pulse)
        bridge._run_one_simulation_step()
    for t in range(DRIVE_STEPS):                              # (3) challenger arrives WITH the dis-inhibition pulse
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[chal_dev] = xp.float32(drive_chal)
        if t < pdur:
            bridge.cp_external_input_current[chal_dis_dev] = xp.float32(vip_current)   # phasic VIP -> dis-inhibition
        bridge._run_one_simulation_step()
    for t in range(DRIVE_STEPS, pdur):                        # (3b) pulse tail if pulse_duration > challenger drive
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[chal_dis_dev] = xp.float32(vip_current)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0                 # (4) free (inhibition RESTORED) -> settled winner
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    ra, rb = _late_rate(bridge, xp, A_dev, B_dev, FREE_STEPS, late_start)
    return ra, rb, vip_current


# ── the properties at a fixed operating point ──────────────────────────────────────────────────────────────
def evaluate_operating_point(seed, salience_gain, dis_to_fs, pulse_duration, fs_to_ws, ou_noise,
                             incumbent_settle, heterogeneity, drive_inc, chal_max, n_chal,
                             a2fs=A2FS_WEIGHT, attractor_weight=DEFAULT_ATTRACTOR_WEIGHT, verbose=True):
    """Build the disinhibition bridge and measure: the challenger sweep (mutual exclusion, a_holds_weak, monotone
    crossover, anti-annihilation, salience-gating of the pulse), the causal-swap membership test, and the
    CONTINUOUS (no-restore) headline takeover."""
    drive_chals = list(np.linspace(0.0, float(chal_max), int(n_chal)))

    bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, handles = build_disinhibition_bridge(
        seed=seed, fs_to_ws=fs_to_ws, dis_to_fs=dis_to_fs, a2fs=a2fs, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, dis_lesion=False, attractor_weight=attractor_weight)

    winners, a_rates, b_rates, vip_currents, both, none = [], [], [], [], [], []
    for dC in drive_chals:
        ra, rb, vc = run_incumbency_pulse(bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, drive_inc, float(dC),
                                          salience_gain=salience_gain, pulse_duration=pulse_duration,
                                          challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
        w, ia, ib = _winner(ra, rb)
        winners.append(w); a_rates.append(ra); b_rates.append(rb); vip_currents.append(vc)
        both.append(bool(ia and ib)); none.append(bool((not ia) and (not ib)))
        if verbose:
            print(f"  [sweep] chal={dC:8.1f} vip={vc:8.1f}  A={ra:.4f}{'*' if ia else ' '}  "
                  f"B={rb:.4f}{'*' if ib else ' '}  -> {w}", flush=True)

    co_ignition_any = any(both)
    weak_idx = 1 if len(winners) > 1 else 0
    a_holds_weak = bool(winners[weak_idx] == "A")
    b_takes_strong = bool(winners[-1] == "B")
    monotone, crossover = _is_monotone(winners)
    annihilation_any = any(none[weak_idx:])
    pulse_zero_at_zero_salience = bool(abs(vip_currents[0]) < 1e-9)
    pulse_scales_with_salience = bool(all(vip_currents[i + 1] >= vip_currents[i] for i in range(len(vip_currents) - 1))
                                      and (vip_currents[-1] > vip_currents[0] if salience_gain > 0 else True))

    # ── causal-swap membership (the salient role is the ONLY swapped variable; pulse magnitude IDENTICAL) ──────
    strong = float(chal_max)
    swap_trials = []
    for strong_drive in (strong, 0.85 * strong):
        raB, rbB, _ = run_incumbency_pulse(bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, drive_inc,
                                           strong_drive, salience_gain=salience_gain, pulse_duration=pulse_duration,
                                           challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
        wB, _, _ = _winner(raB, rbB)
        raA, rbA, _ = run_incumbency_pulse(bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, drive_inc,
                                           strong_drive, salience_gain=salience_gain, pulse_duration=pulse_duration,
                                           challenger_is_B=False, incumbent_settle=incumbent_settle, isolate=True)
        wA, _, _ = _winner(raA, rbA)
        swap_trials.append({"strong_drive": float(strong_drive),
                            "B_challenges": {"A": raB, "B": rbB, "winner": wB, "follows_salience": wB == "B"},
                            "A_challenges": {"A": raA, "B": rbA, "winner": wA, "follows_salience": wA == "A"}})
    follows = [t["B_challenges"]["follows_salience"] for t in swap_trials] + \
              [t["A_challenges"]["follows_salience"] for t in swap_trials]
    swap_attribution = float(np.mean(follows)) if follows else 0.0
    causal_swap = bool(swap_attribution >= 0.9)

    # ── CONTINUOUS headline (anti-cheat): NO restore anywhere in the competition ───────────────────────────────
    bridge_c, xp_c, A_c, B_c, disA_c, disB_c, snap_c, _ = build_disinhibition_bridge(
        seed=seed, fs_to_ws=fs_to_ws, dis_to_fs=dis_to_fs, a2fs=a2fs, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, dis_lesion=False, attractor_weight=attractor_weight)
    restore_before = _RESTORE_CALLS["n"]
    bridge_c.cp_external_input_current[:] = 0.0
    for _ in range(DRIVE_STEPS):
        bridge_c.cp_external_input_current[:] = 0.0
        bridge_c.cp_external_input_current[A_c] = xp_c.float32(drive_inc)
        bridge_c._run_one_simulation_step()
    bridge_c.cp_external_input_current[:] = 0.0
    for _ in range(incumbent_settle):
        bridge_c._run_one_simulation_step()
    a_pre, b_pre = _late_rate(bridge_c, xp_c, A_c, B_c, FREE_STEPS, FREE_STEPS - 1)
    incumbent_ignited_pre = bool(_ignited(a_pre) and not _ignited(b_pre))
    vip_strong = float(salience_gain) * strong
    pdur = int(pulse_duration)
    for t in range(DRIVE_STEPS):                              # B challenges strong WITH the pulse into dis_B
        bridge_c.cp_external_input_current[:] = 0.0
        bridge_c.cp_external_input_current[B_c] = xp_c.float32(strong)
        if t < pdur:
            bridge_c.cp_external_input_current[disB_c] = xp_c.float32(vip_strong)
        bridge_c._run_one_simulation_step()
    for t in range(DRIVE_STEPS, pdur):
        bridge_c.cp_external_input_current[:] = 0.0
        bridge_c.cp_external_input_current[disB_c] = xp_c.float32(vip_strong)
        bridge_c._run_one_simulation_step()
    bridge_c.cp_external_input_current[:] = 0.0
    ls = FREE_STEPS - max(1, FREE_STEPS // 3)
    a_post, b_post = _late_rate(bridge_c, xp_c, A_c, B_c, FREE_STEPS, ls)
    w_post, ia_post, ib_post = _winner(a_post, b_post)
    n_ignited_post = int(ia_post) + int(ib_post)
    restore_after = _RESTORE_CALLS["n"]
    continuous_no_restore = bool(restore_after == restore_before)
    continuous_takeover = bool(w_post == "B")
    anti_annihilation = bool(n_ignited_post == 1)

    # ── determinism: build twice, hash the seed-derived params (cfg.seed, NOT actual_seed_used) ────────────────
    h1 = _threshold_hash(bridge, xp)
    bridge2, xp2, _, _, _, _, _, _ = build_disinhibition_bridge(
        seed=seed, fs_to_ws=fs_to_ws, dis_to_fs=dis_to_fs, a2fs=a2fs, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, dis_lesion=False, attractor_weight=attractor_weight)
    h2 = _threshold_hash(bridge2, xp2)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    op_go = bool(
        (not co_ignition_any) and a_holds_weak and b_takes_strong and monotone and crossover
        and causal_swap and (not annihilation_any)
        and continuous_no_restore and continuous_takeover and anti_annihilation and seed_deterministic
        and pulse_zero_at_zero_salience and pulse_scales_with_salience)

    result = {
        "seed": int(seed),
        "operating_point": {"salience_gain": float(salience_gain), "dis_to_fs": float(dis_to_fs),
                            "pulse_duration": int(pulse_duration), "fs_to_ws": float(fs_to_ws), "a2fs": float(a2fs),
                            "ou_noise_pA": float(ou_noise), "incumbent_settle": int(incumbent_settle),
                            "heterogeneity": bool(heterogeneity), "drive_inc": float(drive_inc),
                            "chal_max": float(chal_max), "n_chal": int(n_chal)},
        "drive_challengers": [float(x) for x in drive_chals],
        "vip_currents": [float(x) for x in vip_currents],
        "a_rates": [float(x) for x in a_rates], "b_rates": [float(x) for x in b_rates],
        "winner_per_challenger": winners,
        "mutual_exclusion": bool(not co_ignition_any),
        "a_holds_weak": a_holds_weak, "b_takes_strong": b_takes_strong,
        "monotone": monotone, "crossover": crossover, "annihilation_on_sweep": annihilation_any,
        "pulse_zero_at_zero_salience": pulse_zero_at_zero_salience,
        "pulse_scales_with_salience": pulse_scales_with_salience,
        "causal_swap": {"attribution": swap_attribution, "pass": causal_swap, "trials": swap_trials},
        "continuous_headline": {
            "no_restore_calls": continuous_no_restore, "takeover": continuous_takeover,
            "incumbent_ignited_pre_challenge": incumbent_ignited_pre, "vip_current_strong": vip_strong,
            "n_ignited_post": n_ignited_post, "anti_annihilation": anti_annihilation,
            "A_pre": a_pre, "B_pre": b_pre, "A_post": a_post, "B_post": b_post, "winner_post": w_post},
        "seed_deterministic": seed_deterministic, "threshold_hash": h1,
        "op_go": op_go,
    }
    if verbose:
        print(f"  [op seed={seed} g={salience_gain} dfs={dis_to_fs} pd={pulse_duration} fs={fs_to_ws} ou={ou_noise}] "
              f"go={op_go} | mutual_excl={not co_ignition_any} holds_weak={a_holds_weak} "
              f"takes_strong={b_takes_strong} monotone={monotone} crossover={crossover} "
              f"causal_swap={swap_attribution:.2f} contin_takeover={continuous_takeover} "
              f"n_ign_post={n_ignited_post} sal_gated={pulse_zero_at_zero_salience and pulse_scales_with_salience} "
              f"det={seed_deterministic}", flush=True)
    return result


# ── anti-cheat controls ────────────────────────────────────────────────────────────────────────────────────
def control_pulse_off(seed, dis_to_fs, pulse_duration, fs_to_ws, ou_noise, incumbent_settle, heterogeneity,
                      drive_inc, chal_max, n_chal, a2fs=A2FS_WEIGHT):
    """PULSE-OFF (salience_gain=0 -> vip_current=0 everywhere): the clean monotone takeover + causal-swap must FAIL
    (reproduce the non-eviction negative) -> the eviction is CAUSED by the salience-gated dis-inhibition pulse."""
    r = evaluate_operating_point(seed, 0.0, dis_to_fs, pulse_duration, fs_to_ws, ou_noise, incumbent_settle,
                                 heterogeneity, drive_inc, chal_max, n_chal, a2fs=a2fs, verbose=False)
    clean = bool(r["monotone"] and r["crossover"] and r["causal_swap"]["pass"]
                 and r["continuous_headline"]["takeover"] and r["continuous_headline"]["anti_annihilation"]
                 and r["mutual_exclusion"])
    return {"reproduces_negative": (not clean), "monotone": r["monotone"], "crossover": r["crossover"],
            "causal_swap_pass": r["causal_swap"]["pass"], "mutual_exclusion": r["mutual_exclusion"],
            "continuous_takeover": r["continuous_headline"]["takeover"],
            "winner_per_challenger": r["winner_per_challenger"]}


def control_wta_lesion(seed, salience_gain, dis_to_fs, pulse_duration, fs_to_ws, ou_noise, incumbent_settle,
                       heterogeneity, drive_inc, chal_max, a2fs=A2FS_WEIGHT, drive_chal=None):
    """WTA lesion (fs_to_ws=0) WITH the pulse on: BOTH assemblies co-ignite -> mutual exclusion comes from the
    (cross-)inhibition, not from the pulse silencing one assembly. `drive_chal` defaults to chal_max (strong)."""
    dC = float(chal_max) if drive_chal is None else float(drive_chal)
    bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, _ = build_disinhibition_bridge(
        seed=seed, fs_to_ws=fs_to_ws, dis_to_fs=dis_to_fs, a2fs=a2fs, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=True, dis_lesion=False)
    ra, rb, _ = run_incumbency_pulse(bridge, xp, A_dev, B_dev, disA_dev, disB_dev, snap, drive_inc, dC,
                                     salience_gain=salience_gain, pulse_duration=pulse_duration,
                                     challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
    both = bool(_ignited(ra) and _ignited(rb))
    return {"both_ignite": both, "A": ra, "B": rb, "drive_chal": dC}


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-2c salience-gated dis-inhibition-pulse eviction de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung2c_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--salience-gain", type=float, default=SALIENCE_GAIN,
                    help="vip_current = salience_gain * drive_chal (the salience->pulse amplitude gain)")
    ap.add_argument("--dis-to-fs", type=float, default=DIS_TO_FS_WEIGHT, help="dis -| fs structural weight")
    ap.add_argument("--pulse-duration", type=int, default=PULSE_DURATION, help="steps the VIP pool is driven")
    ap.add_argument("--fs-to-ws", type=float, default=FS_TO_WS_WEIGHT, help="fs -> assembly inhibition baseline")
    ap.add_argument("--a2fs", type=float, default=A2FS_WEIGHT, help="assembly -> fs excitation (cross-inhibition)")
    ap.add_argument("--ou-noise", type=float, default=40.0, help="OU noise std (pA)")
    ap.add_argument("--incumbent-settle", type=int, default=120, help="free steps the incumbent holds")
    ap.add_argument("--no-heterogeneity", action="store_true", help="disable parameter heterogeneity")
    ap.add_argument("--drive-inc", type=float, default=5000.0)
    ap.add_argument("--chal-max", type=float, default=8000.0)
    ap.add_argument("--n-chal", type=int, default=9)
    ap.add_argument("--smoke", action="store_true",
                    help="grid-scan (salience_gain, dis_to_fs, pulse_duration) on ONE seed to find the window")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)
    het = not args.no_heterogeneity

    if args.smoke:
        print(f"[rung2c-smoke] seed={args.seed} fs={args.fs_to_ws} a2fs={args.a2fs} ou={args.ou_noise} "
              f"settle={args.incumbent_settle} het={het} — scanning (salience_gain, dis_to_fs, pulse_duration)",
              flush=True)
        grid = []
        for sg in (0.5, 1.0, 1.5):
            for dfs in (20.0, 40.0):
                for pd in (18, 25):
                    r = evaluate_operating_point(args.seed, sg, dfs, pd, args.fs_to_ws, args.ou_noise,
                                                 args.incumbent_settle, het, args.drive_inc, args.chal_max,
                                                 args.n_chal, a2fs=args.a2fs, verbose=True)
                    grid.append(r)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_rung2c_salience_disinhibition_derisk", "mode": "smoke", "grid": grid}, f, indent=2)
        any_go = any(g["op_go"] for g in grid)
        print(f"\n[rung2c-smoke] wrote {args.json}  any_op_go={any_go}", flush=True)
        return 0 if any_go else 1

    print(f"[rung2c] seed={args.seed} g={args.salience_gain} dfs={args.dis_to_fs} pd={args.pulse_duration} "
          f"fs={args.fs_to_ws} a2fs={args.a2fs} ou={args.ou_noise} settle={args.incumbent_settle} het={het}", flush=True)
    r = evaluate_operating_point(args.seed, args.salience_gain, args.dis_to_fs, args.pulse_duration, args.fs_to_ws,
                                 args.ou_noise, args.incumbent_settle, het, args.drive_inc, args.chal_max,
                                 args.n_chal, a2fs=args.a2fs, verbose=True)
    off = control_pulse_off(args.seed, args.dis_to_fs, args.pulse_duration, args.fs_to_ws, args.ou_noise,
                            args.incumbent_settle, het, args.drive_inc, args.chal_max, args.n_chal, a2fs=args.a2fs)
    # WTA-lesion + ATTRIBUTION at the DISCRIMINATING drive = the STRONGEST challenger the incumbent still HOLDS
    # (intact winner == "A"). There the intact keeps B suppressed (single content) while the lesion (fs_to_ws=0)
    # lets B ignite (co-ignition) -> the mutual exclusion in the hold region is INHIBITION-caused. (A weak-drive
    # comparison is uninformative: a weak challenger stays off even WITHOUT inhibition.) Attribute the A-B rate gap
    # intact vs lesion at that same drive (measuring both arms is not the same as asking whose the difference is).
    held = [i for i, w in enumerate(r["winner_per_challenger"])
            if w == "A" and r["drive_challengers"][i] > 0.0]
    hold_idx = held[-1] if held else (1 if len(r["a_rates"]) > 1 else 0)
    hold_drive = r["drive_challengers"][hold_idx]
    lesion = control_wta_lesion(args.seed, args.salience_gain, args.dis_to_fs, args.pulse_duration, args.fs_to_ws,
                                args.ou_noise, args.incumbent_settle, het, args.drive_inc, args.chal_max,
                                a2fs=args.a2fs, drive_chal=hold_drive)
    intact_gap_hold = float(r["a_rates"][hold_idx] - r["b_rates"][hold_idx])
    lesion_gap_hold = float(lesion["A"] - lesion["B"])
    mutual_excl_attribution = attributable_to(
        "workspace single-content (A-B gap @ strongest-held challenger) via FS inhibition",
        intact_gap_hold, lesion_gap_hold, warn_below=0.8)
    print(f"[rung2c] PULSE-OFF reproduces_negative={off['reproduces_negative']} (monotone={off['monotone']} "
          f"causal_swap={off['causal_swap_pass']} takeover={off['continuous_takeover']}) | "
          f"WTA-lesion @ held-drive {hold_drive:.0f} both_ignite={lesion['both_ignite']} | "
          f"mutual_excl_attribution={mutual_excl_attribution}", flush=True)

    go = bool(r["op_go"] and off["reproduces_negative"] and lesion["both_ignite"])

    v = Verdict("rung2c salience-gated dis-inhibition eviction @ frozen operating point (seed %d)" % args.seed)
    v.require("incumbent ignites & holds a weak challenger", r["a_holds_weak"], expect=True)
    v.require("pulse is salience-gated (0 at zero salience, scales with salience)",
              r["pulse_zero_at_zero_salience"] and r["pulse_scales_with_salience"], expect=True)
    v.knob("VIP dis-inhibition pulse current (strong challenger)", requested=args.salience_gain * args.chal_max,
           applied=r["continuous_headline"]["vip_current_strong"])
    v.require("WTA inhibition load-bearing (fs=0 lesion co-ignites)", lesion["both_ignite"], expect=True)
    v.require("PULSE-OFF reproduces the non-eviction negative", off["reproduces_negative"], expect=True)
    v.require("continuous headline: zero _restore_state calls", r["continuous_headline"]["no_restore_calls"], expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", r["seed_deterministic"], expect=True)
    # NOTE: the clean-eviction outcome (n_ignited==1 post-takeover, monotone salience-graded crossover, causal-swap)
    # is the MEASURED result, NOT a validity precondition — its FAILURE (co-ignition n_ignited=2, or hold-out) IS
    # the negative (go=False), not an instrument failure. Encoding it as a require would wrongly mark a valid
    # negative UNDEFINED (Rung-2b keeps the same split).
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("short_term_plasticity", why="STP banked as annihilating for eviction (2026-08-01)")
    v.disabled("intrinsic_SFA", why="intrinsic Izhikevich SFA banked as self-extinguishing for eviction (Rung-2b)")
    vd = v.decide(go=go)

    result = {"runner": "_gnw_rung2c_salience_disinhibition_derisk", "mode": "single", "go": go,
              "verdict": vd["status"], "preconditions": vd["preconditions"],
              "disabled_processes": vd["disabled_processes"], "undefined_reasons": vd["undefined_reasons"],
              "backend": args.backend, "operating_point": r["operating_point"], "eval": r,
              "control_pulse_off": off, "control_wta_lesion": lesion,
              "attribution": {"label": "workspace single-content (A-B gap @ strongest-held challenger) via FS inhibition",
                              "intact_gap_hold": intact_gap_hold, "lesion_gap_hold": lesion_gap_hold,
                              "fraction_attributable_to_inhibition": mutual_excl_attribution,
                              "held_challenger_drive": hold_drive}}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[rung2c] seed={args.seed} GO={go}  wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
