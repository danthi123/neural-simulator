"""GNW QUENCH-EVICT OVERWRITE — evict a HELD workspace coalition with a TRANSIENT, OPEN-LOOP feedback-inhibition
pulse (the affect active-clear mechanism transferred to the Global-Neuronal-Workspace), so a NEW thought can take
the freed slot. The missing HALF of the thought-swap: a basal-ganglia IN-gate already lets a NEW coalition in
(_gnw_bg_thalamus_gate_overwrite_derisk: admit 6/6); the UNSOLVED half is EVICTING the OLD held representation.

WHY THIS IS A NEW MECHANISM (not a re-run of a mapped NO-GO). Every prior eviction lever on this workspace failed
on a SUPRA-CRITICAL (self-sufficient) incumbent, and all belong to the STANDING-BRAKE / GATE class:
  - STN stop-veto (2026-08-18-gnw-stn-stop-veto-NOGO): a standing GPi brake either too weak (n_post=2, co-ignition
    survives) or, past g_i~200, reverses inhibitory current to depolarizing -> post-inhibitory REBOUND re-ignites
    weak slots (n_post=4). No strength reaches n=0.
  - BG-thalamus gate-overwrite (2026-08-18-gnw-bg-thalamus-gate-overwrite-NOGO): the gate ADMITS the challenger
    (lockout solved 6/6) but its per-slot inhibition cannot evict a THALAMUS-INDEPENDENT incumbent -> n_post=2 all 6.
  - Active-overwrite (2026-08-18-gnw-active-overwrite-NOGO): rate-competition + WTA + targeted STD -> HOLD or
    CO-IGNITION (clean swap 0/6).
  - Rung-2b SFA (2026-08-14-...-BOUNDARY): the fatigue that would evict equals the fatigue that self-extinguishes.
This is EXACTLY the pattern the affect ratchet showed: every OUTWARD/STANDING brake fails structurally
(2026-08-01-affect-ratchet-STP-annihilates...; -affect-eviction-slow-GABAB-KILLED...). The ONE thing that worked
for affect was an ACTIVE, TRANSIENT, OPEN-LOOP CLEAR: a dedicated spiking quench_fs FS pool fires strong GABA_A onto
the held pools for a window that EXCEEDS the drain threshold, the reverberation collapses, the OFF fixed point then
HOLDS with ZERO standing force, and re-ignition of a NEW state survives (synapses left recovered). Biology: Compte,
Brunel, Goldman-Rakic & Wang 2000 — persistent-activity termination by "nonspecific excitatory input recruiting
feedback inhibition" drives a selective population back to its down state. The quench_fs GABA_A pulse is the spiking
realization. It is a genuine BASIN-SWITCH, not current subtraction: bistability (which KILLED every brake) is the
ASSET that makes the cleared state HOLD.

HYPOTHESIS de-risked here: workspace eviction needs the same active-transient-open-loop quench, NOT a gate/veto. A
quench_fs pulse collapses the currently-ignited coalition for a window; then the new coalition (driven in by the
already-working IN-gate) ignites into the cleared workspace. The OLD state drops to baseline, the NEW state ignites,
and the co-ignition PARTIAL (new in, old still lingering, n=2) is BEATEN.

SUBSTRATE: build_swap_bridge's distributed divisively-normalized workspace (the EXACT substrate of the
active-overwrite / distributed-overwrite NO-GOs), forked here as build_quench_bridge, run in its SUPRA-CRITICAL
disjoint headline configuration (overlap=0, uniform recurrence w=34, NO WTA) + a dedicated additive quench_fs pool.
All spiking/synaptic. NO sim/ edit; explicit dense frozen wiring; native STP/homeostasis OFF.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed) — per seed, ALL of:
  SWAP        — win_pre=A & n_pre=1, then A drops to baseline (NOT ignited) & B ignites & win_post=B & n_post=1.
  OPEN-LOOP   — quench_fs fires HARD during the clear (rate high) and is SILENT at the read (rate ~0): a transient
                basin-switch, not a standing current subtraction.
  LOAD-BEARING— LESION the quench (GABA_A weight 0, timing/drive identical) -> NO swap (A persists / co-ignites).
  STANDING-FAILS — the STANDING-brake contrast arm (quench HELD ON through the B-drive + read) does NOT clean-swap
                (it suppresses B too / is not silent at read), reproducing the NO-GO class.
  NO-QUENCH-FAILS — drive B on top of A with NO clear -> co-ignition / lockout (the PARTIAL we beat).
  REIGNITE    — the NEW coalition ignites AND HOLDS through an extended free tail (the quench did not poison it).
  DETERMINISM — build twice at one seed -> identical substrate hash (heterogeneity from cfg.seed).
  ADDITIVE    — the quench pool is purely additive: the workspace/norm/thal threshold slice is byte-identical
                with vs without the quench pool (best-effort; caveated if the RNG prefix property does not hold).

NOT-A-WALL: if the quench does not achieve a clean swap, the residual is QUANTIFIED (old-state rate, empty-window
length, whether B ignites) so the next mechanism is mapped.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_quench_evict_overwrite_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_quench_evict_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_quench_evict_overwrite_derisk --six-seed \
      --json research/findings/raw/_gnw_quench_evict_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict

# reuse-by-import: the workspace geometry/recurrence, the stepping+read instruments, the ignition criterion,
# validated constants, and the snapshot/restore — all from the active-overwrite / rung chain (the NO-GO substrate).
from research.runners._gnw_rung1_ignition_curve_derisk import (
    DRIVE_STEPS, SETTLE_STEPS, FREE_STEPS, WS_LOOP_GATE,
)
from research.runners._gnw_rung2_competitive_access_derisk import _ignited, IGNITE_FRAC, SOLO_PLATEAU
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
from research.runners._gnw_active_overwrite_derisk import (
    _pattern_geometry, _rec_population_split, _ws_step, _drive, _read_private_rates,
    _instant_private_rate, _margin, _verdict_label, READ_FREE_STEPS,
    N_PATTERNS, PATTERN_SIZE, WORKSPACE_N, NORM_N, THAL_N,
    W_SHARED, WS_NORM_W, NORM_WS_W, THAL_TONIC_PA, THAL_WS_W, STRONG_PA, OU_NOISE_PA,
)

# ── the quench pool (the affect active-clear mechanism, transferred) ────────────────────────────────────────────
QUENCH_N = 30                 # dedicated FS quench interneurons (PV-basket-like), mirrors the affect quench_fs pool
QUENCH_W = 8.0                # quench_fs -> ws_used GABA_A weight (all-to-all). CALIBRATED clear strength (swept):
                              # too low -> A holds; too high -> overshoot/rebound (the affect w>=25 tip / the STN
                              # g_i>200 reversal). E_i=-75 mV (cfg default).
QUENCH_DRIVE_PA = 2500.0      # external drive into quench_fs during the clear (recruits the pool to fire hard).
QUENCH_WINDOW = 120           # clear-window length (steps=ms). Must EXCEED the drain threshold (> the NMDA decay +
                              # recovery) or the loop re-ignites from OU noise. Swept.
REIGNITE_HOLD = 120           # extra free-run after the identity read: the NEW coalition must still be ignited.

# headline substrate = the SUPRA-CRITICAL disjoint workspace (the active-overwrite NO-GO configuration):
HEADLINE_W_REC = W_SHARED     # 34.0, uniform (w_priv=w_shared) -> each disjoint pattern self-sustains (supra-critical)
IGNITED_BASELINE_MAX = IGNITE_FRAC * SOLO_PLATEAU   # a "dropped to baseline" old state reads BELOW this (not ignited)

_BASE_N = WORKSPACE_N + NORM_N + THAL_N   # the base substrate size (before the appended quench pool)


# ── build: fork of build_swap_bridge (disjoint, uniform, NO WTA) + an ADDITIVE quench_fs pool ──────────────────
def build_quench_bridge(seed=42, quench_n=QUENCH_N, quench_w=QUENCH_W, quench_thal_w=0.0, quench_lesion=False,
                        w_rec=HEADLINE_W_REC, heterogeneity=True, ou_noise_pA=OU_NOISE_PA, with_quench=True):
    """workspace (exc, NMDA; K=3 DISJOINT self-sustaining patterns, uniform recurrence w_rec) + norm_pool (inh;
    divisive normalization) + thal (exc; tonic shared support) + [quench_fs] (inh FS; the ADDITIVE active-clear
    pool, all-to-all GABA_A onto every workspace-used unit, behind an external drive). quench_lesion=True zeroes the
    quench->ws WEIGHT (the pool is kept + still driven, so timing/firing are identical) = the load-bearing lesion.
    with_quench=False omits the pool entirely (the base substrate, for the additive-hash anti-cheat). ALL wiring
    explicit; native STP/homeostasis OFF; heterogeneity seeded from cfg.seed. Returns (bridge, xp, patterns_dev,
    privates_dev, thal_dev, quench_dev, ws_used, snap, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    norm_pool = BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0,
                            enable_nmda=False)
    thal = BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
    regions = [workspace, norm_pool, thal]
    if with_quench:
        # APPENDED LAST so the workspace/norm/thal indices + their heterogeneity draws are unchanged (additive).
        regions = regions + [BrainRegion(name="quench_fs", n_neurons=int(quench_n), exc_fraction=0.0,
                                         internal_density=0.0, enable_nmda=False)]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(w_rec) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(w_rec) * 4.0)
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
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, PATTERN_SIZE, overlap=0)   # DISJOINT
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["workspace_rec"] = _rec_population_split(patterns, privates, float(w_rec), float(w_rec))  # uniform
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(WS_NORM_W), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, float(NORM_WS_W), "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(THAL_WS_W), "E_TO_E")

    inh = list(norm_idx)
    quench_idx = np.array([], dtype=np.int64)
    if with_quench:
        quench_idx = np.asarray(rm.indices("quench_fs"), dtype=np.int64)
        qw = 0.0 if quench_lesion else float(quench_w)
        # the ACTIVE-CLEAR limb: quench_fs -> every workspace-used unit, GABA_A (I_TO_E). Non-selective feedback
        # inhibition (Compte-Wang termination). Polarity set by quench_idx being in output_inhibitory_indices.
        union_plan["quench2ws"] = _dense_pop(quench_idx, ws_used, qw, "I_TO_E")
        # the LOOP-RESET limb (optional): quench_fs -> thal, GABA_A. Terminating persistent activity means opening
        # the WHOLE thalamocortical loop, not just cortex — else the intact tonic re-ignites the supra-critical loop.
        qtw = 0.0 if quench_lesion else float(quench_thal_w)
        if qtw > 0.0:
            union_plan["quench2thal"] = _dense_pop(quench_idx, thal_idx, qtw, "I_TO_E")
        inh += list(quench_idx)

    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(THAL_TONIC_PA)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "quench_n": int(quench_n), "quench_w": float(quench_w),
               "quench_thal_w": float(quench_thal_w),
               "quench_lesion": bool(quench_lesion), "w_rec": float(w_rec), "with_quench": bool(with_quench),
               "heterogeneity": bool(heterogeneity), "ou_noise_pA": float(ou_noise_pA),
               "n_ws_used": int(ws_used.size), "n_patterns": N_PATTERNS,
               "private_sizes": [int(p.size) for p in privates], "base_n": int(_BASE_N)}
    quench_dev = xp.asarray(quench_idx) if with_quench else None
    return (bridge, xp, [xp.asarray(p) for p in patterns], [xp.asarray(p) for p in privates],
            thal_dev, quench_dev, ws_used, snap, handles)


# ── hashes (determinism + additive-substrate anti-cheats) ──────────────────────────────────────────────────────
def _izh_cat(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return parts


def _full_hash(bridge):
    parts = _izh_cat(bridge)
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


def _slice_hash(bridge, n):
    """Hash the FIRST n neurons' izh params (the base workspace/norm/thal slice) — invariant to appending a pool
    IFF the RNG prefix property holds. Best-effort additive anti-cheat."""
    parts = _izh_cat(bridge)
    if not parts:
        return ""
    return hashlib.sha256(np.concatenate([p[:n] for p in parts]).tobytes()).hexdigest()


# ── quench firing read ─────────────────────────────────────────────────────────────────────────────────────────
def _quench_rate_now(bridge, xp, quench_dev):
    if quench_dev is None:
        return 0.0
    return float(to_host(bridge.cp_firing_states[quench_dev].astype(xp.float64).mean()))


def _read_private_and_quench(bridge, xp, thal_dev, privates_dev, quench_dev, *, quench_on=False,
                             quench_drive_pA=QUENCH_DRIVE_PA, n_free=READ_FREE_STEPS):
    """Free-run n_free steps (tonic on, no pattern drive; quench driven only if quench_on) and return the
    LATE-window per-pattern private-core rate AND the whole-window quench_fs rate. quench_on=True is the STANDING
    arm's read (the quench is still driven -> its rate is HIGH here, failing the open-loop anti-cheat)."""
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(privates_dev)
    q_spikes = 0
    q_n = int(quench_dev.shape[0]) if quench_dev is not None else 0
    for t in range(n_free):
        dmap = [(quench_dev, quench_drive_pA)] if (quench_on and quench_dev is not None) else None
        _ws_step(bridge, xp, thal_dev, THAL_TONIC_PA, None, drive_map=dmap)
        if quench_dev is not None:
            q_spikes += int(to_host(bridge.cp_firing_states[quench_dev].astype(xp.float64).sum()))
        if t >= late_start:
            for i, p in enumerate(privates_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[p].astype(xp.float64).sum()))
    rates = []
    for i, p in enumerate(privates_dev):
        denom = float((n_free - late_start) * int(p.shape[0]))
        rates.append(counts[i] / denom if denom > 0 else 0.0)
    q_rate = q_spikes / float(max(1, n_free) * q_n) if q_n else 0.0
    return rates, q_rate


# ── the quench-evict swap (modes: transient / standing / no_quench) ───────────────────────────────────────────
def run_quench_swap(bridge, xp, patterns_dev, privates_dev, thal_dev, quench_dev, snap, *,
                    incumbent=0, challenger=1, mode="transient", quench_drive_pA=QUENCH_DRIVE_PA,
                    quench_window=QUENCH_WINDOW, b_overlap=0, reignite_hold=REIGNITE_HOLD):
    """Ignite incumbent A (hold); CLEAR (quench window); drive challenger B (the IN-gate); read; hold.
      mode="transient" — the headline: quench fires for quench_window, RELEASES, then B is driven (open-loop clear).
      mode="standing"  — the STANDING-brake contrast: the quench is HELD ON through the B-drive AND the read (never
                         released) -> it suppresses B too / is not silent at read -> no clean swap (the NO-GO class).
      mode="no_quench" — drive B directly on top of the held A, NO clear -> the co-ignition/lockout PARTIAL.
    b_overlap>0: the IN-gate's challenger volley arrives DURING the last b_overlap steps of the clear (concurrent with
    the waning quench) so B — driven — wins the post-clear recovery race against the un-driven, tonic-primed incumbent.
    This is the affect mechanism's "drive the new state as inhibition wanes" made explicit for a tonic-driven
    supra-critical loop. The load-bearing lesion is a SEPARATE build (quench_lesion=True) run identically: timing/drive
    identical, the GABA_A weight removed -> A is not collapsed. Reports A-residual, B-rate, the empty window, q rates."""
    _full_restore(bridge, snap)

    # (1) ignite A alone
    _drive(bridge, xp, thal_dev, THAL_TONIC_PA, None, [(patterns_dev[incumbent], STRONG_PA)], n=DRIVE_STEPS)
    pre = _read_private_rates(bridge, xp, thal_dev, THAL_TONIC_PA, privates_dev, None)
    win_pre, margin_pre, n_pre = _margin(pre)

    # (2) CLEAR window: fire the quench pool (non-selective GABA_A). During the last b_overlap steps the challenger's
    #     drive is ALSO present (the IN-gate volley arriving as the quench wanes) — A is never re-driven.
    empty_steps, q_clear_spikes = 0, 0
    q_n = int(quench_dev.shape[0]) if quench_dev is not None else 0
    ov_start = max(0, quench_window - int(b_overlap))
    # windowed rate over the last third of the PURE-quench phase (before any b_overlap drive) — the clean read of
    # whether the quench actually COLLAPSED the incumbent DURING the clear (distinct from whether it HOLDS after).
    mid_start = max(0, min(ov_start, quench_window) - max(1, quench_window // 3))
    a_mid_spikes, a_mid_steps, gi_mid_sum, gi_rest = 0, 0, 0.0, -1.0
    inc_p = privates_dev[incumbent]
    has_gi = hasattr(bridge, "cp_conductance_g_i")
    if has_gi:
        gi_rest = float(to_host(bridge.cp_conductance_g_i[inc_p].astype(xp.float64).mean()))
    if mode != "no_quench" and quench_dev is not None:
        for t in range(quench_window):
            dmap = [(quench_dev, quench_drive_pA)]
            if b_overlap > 0 and t >= ov_start:
                dmap.append((patterns_dev[challenger], STRONG_PA))
            _ws_step(bridge, xp, thal_dev, THAL_TONIC_PA, None, drive_map=dmap)
            q_clear_spikes += int(to_host(bridge.cp_firing_states[quench_dev].astype(xp.float64).sum()))
            if mid_start <= t < min(ov_start, quench_window):
                a_mid_spikes += int(to_host(bridge.cp_firing_states[inc_p].astype(xp.float64).sum()))
                if has_gi:
                    gi_mid_sum += float(to_host(bridge.cp_conductance_g_i[inc_p].astype(xp.float64).mean()))
                a_mid_steps += 1
            na = _ignited(_instant_private_rate(bridge, xp, privates_dev, incumbent))
            nb = _ignited(_instant_private_rate(bridge, xp, privates_dev, challenger))
            empty_steps += int((not na) and (not nb))
    q_clear_rate = q_clear_spikes / float(max(1, quench_window) * q_n) if (q_n and mode != "no_quench") else 0.0
    old_rate_midclear = a_mid_spikes / float(a_mid_steps * int(inc_p.shape[0])) if a_mid_steps else -1.0
    gi_midclear = gi_mid_sum / float(a_mid_steps) if (has_gi and a_mid_steps) else -1.0  # inhibition DELIVERED read

    # (3) drive challenger B (the IN-gate). standing arm: keep the quench ON during the B-drive too.
    for _ in range(DRIVE_STEPS):
        dmap = [(patterns_dev[challenger], STRONG_PA)]
        if mode == "standing" and quench_dev is not None:
            dmap.append((quench_dev, quench_drive_pA))
        _ws_step(bridge, xp, thal_dev, THAL_TONIC_PA, None, drive_map=dmap)
        na = _ignited(_instant_private_rate(bridge, xp, privates_dev, incumbent))
        nb = _ignited(_instant_private_rate(bridge, xp, privates_dev, challenger))
        empty_steps += int((not na) and (not nb))

    # (4) identity read (+ quench-at-read). standing arm reads WITH the quench still driven (open-loop anti-cheat).
    post, q_read_rate = _read_private_and_quench(bridge, xp, thal_dev, privates_dev, quench_dev,
                                                 quench_on=(mode == "standing"), quench_drive_pA=quench_drive_pA,
                                                 n_free=READ_FREE_STEPS)
    win_post, margin_post, n_post = _margin(post)

    # (5) reignition-survival hold: extended free-run (quench OFF for every arm) — B must still be ignited.
    hold, q_hold_rate = _read_private_and_quench(bridge, xp, thal_dev, privates_dev, quench_dev,
                                                 quench_on=False, n_free=reignite_hold)
    win_hold, margin_hold, n_hold = _margin(hold)

    a_res = float(post[incumbent]); b_rate = float(post[challenger])
    a_res_hold = float(hold[incumbent]); b_hold = float(hold[challenger])
    v_pre, _ = _verdict_label(pre); v_post, _ = _verdict_label(post)

    swapped = bool(win_pre == incumbent and n_pre == 1 and (not _ignited(a_res))
                   and _ignited(b_rate) and win_post == challenger and n_post == 1)
    reignite_ok = bool(_ignited(b_hold) and win_hold == challenger and n_hold == 1 and (not _ignited(a_res_hold)))
    return {
        "mode": mode, "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
        "hold_rates": [float(r) for r in hold], "winner_pre": int(win_pre), "winner_post": int(win_post),
        "winner_hold": int(win_hold), "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post),
        "n_ignited_hold": int(n_hold), "delivered_pre": v_pre, "delivered_post": v_post,
        "old_residual_post": a_res, "old_residual_hold": a_res_hold, "new_rate_post": b_rate, "new_rate_hold": b_hold,
        "old_rate_midclear": float(old_rate_midclear), "gi_rest": float(gi_rest), "gi_midclear": float(gi_midclear),
        "swapped": swapped, "reignite_ok": reignite_ok, "empty_steps": int(empty_steps),
        "quench_clear_rate": float(q_clear_rate), "quench_read_rate": float(q_read_rate),
        "quench_hold_rate": float(q_hold_rate),
        "co_ignition": bool(n_pre == 1 and n_post >= 2), "went_empty": bool(n_pre >= 1 and n_post == 0),
        "incumbent_held": bool(win_post == incumbent and n_post >= 1 and not _ignited(b_rate)),
    }


# ── one seed: the four arms + anti-cheats + GO gate ───────────────────────────────────────────────────────────
def evaluate_seed(seed, *, quench_w=QUENCH_W, quench_window=QUENCH_WINDOW, quench_drive_pA=QUENCH_DRIVE_PA,
                  quench_n=QUENCH_N, quench_thal_w=0.0, b_overlap=0, w_rec=HEADLINE_W_REC,
                  heterogeneity=True, verbose=True):
    def _build(**kw):
        p = dict(seed=seed, quench_n=quench_n, quench_w=quench_w, quench_thal_w=quench_thal_w,
                 w_rec=w_rec, heterogeneity=heterogeneity)
        p.update(kw)
        return build_quench_bridge(**p)

    b, xp, pats, privs, thal, quench, ws_used, snap, hh = _build()

    # HEADLINE: transient open-loop quench clear + the IN-gate volley as it wanes, then B holds.
    headline = run_quench_swap(b, xp, pats, privs, thal, quench, snap, mode="transient",
                               quench_drive_pA=quench_drive_pA, quench_window=quench_window, b_overlap=b_overlap)
    # STANDING-brake contrast (same build, same b_overlap): quench held ON through B-drive + read.
    standing = run_quench_swap(b, xp, pats, privs, thal, quench, snap, mode="standing",
                               quench_drive_pA=quench_drive_pA, quench_window=quench_window, b_overlap=b_overlap)
    # NO-QUENCH baseline (same build): drive B on top of A, no clear -> the co-ignition/lockout PARTIAL.
    noq = run_quench_swap(b, xp, pats, privs, thal, quench, snap, mode="no_quench",
                          quench_drive_pA=quench_drive_pA, quench_window=quench_window, b_overlap=b_overlap)

    # LESION (separate build, quench weight 0; timing/drive identical) -> the load-bearing anti-cheat.
    bl, xpl, patsl, privsl, thall, quenchl, _wl, snapl, _ = _build(quench_lesion=True)
    lesion = run_quench_swap(bl, xpl, patsl, privsl, thall, quenchl, snapl, mode="transient",
                             quench_drive_pA=quench_drive_pA, quench_window=quench_window, b_overlap=b_overlap)

    # ── anti-cheats ──
    swap_ok = bool(headline["swapped"])
    reignite_ok = bool(headline["reignite_ok"])
    open_loop = bool(headline["quench_clear_rate"] > 0.05 and headline["quench_read_rate"] < 0.01
                     and headline["quench_hold_rate"] < 0.01)
    load_bearing = bool(swap_ok and not lesion["swapped"])
    standing_fails = bool(not standing["swapped"])          # the standing brake cannot clean-swap (NO-GO class)
    no_quench_fails = bool(not noq["swapped"])              # the PARTIAL we beat (co-ignition / lockout)

    # DETERMINISM: build twice at this seed -> identical full hash.
    h_a = _full_hash(b)
    b2, xp2, *_2 = _build()
    seed_deterministic = bool(_full_hash(b2) == h_a and h_a != "")
    # ADDITIVE: the base slice is byte-identical with vs without the quench pool (best-effort).
    b_no, xp_no, *_no = _build(with_quench=False)
    additive_substrate = bool(_slice_hash(b, _BASE_N) == _slice_hash(b_no, _BASE_N) and _slice_hash(b, _BASE_N) != "")

    seed_go = bool(swap_ok and open_loop and load_bearing and standing_fails and no_quench_fails
                   and reignite_ok and seed_deterministic)

    v = Verdict("GNW quench-evict OVERWRITE (seed %d)" % seed)
    v.require("incumbent ignites confidently (n_pre==1) [precondition]",
              bool(headline["n_ignited_pre"] == 1 and headline["winner_pre"] == 0), expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash) [precondition]",
              seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity", why="the clear is an EXTERNAL FS pulse, not recurrence depletion")
    vd = v.decide(go=bool(swap_ok and open_loop and load_bearing and standing_fails and no_quench_fails
                          and reignite_ok), verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"quench_w": float(quench_w), "quench_window": int(quench_window),
                            "quench_drive_pA": float(quench_drive_pA), "quench_n": int(quench_n),
                            "quench_thal_w": float(quench_thal_w), "b_overlap": int(b_overlap),
                            "w_rec": float(w_rec), "heterogeneity": bool(heterogeneity),
                            "reignite_hold": int(REIGNITE_HOLD)},
        "go_gate": {"swap_ok": swap_ok, "open_loop": open_loop, "load_bearing": load_bearing,
                    "standing_fails": standing_fails, "no_quench_fails": no_quench_fails,
                    "reignite_ok": reignite_ok},
        "anti_cheats": {"quench_load_bearing": load_bearing, "quench_open_loop": open_loop,
                        "standing_brake_fails": standing_fails, "no_quench_partial_reproduced": no_quench_fails,
                        "reignition_survives": reignite_ok, "seed_deterministic": seed_deterministic,
                        "additive_substrate": additive_substrate},
        "residual": {
            "headline": {"winner_pre": headline["winner_pre"], "winner_post": headline["winner_post"],
                         "n_pre": headline["n_ignited_pre"], "n_post": headline["n_ignited_post"],
                         "old_residual_post": headline["old_residual_post"], "new_rate_post": headline["new_rate_post"],
                         "old_residual_hold": headline["old_residual_hold"], "new_rate_hold": headline["new_rate_hold"],
                         "old_rate_midclear": headline["old_rate_midclear"], "gi_rest": headline["gi_rest"],
                         "gi_midclear": headline["gi_midclear"],
                         "empty_steps": headline["empty_steps"], "quench_clear_rate": headline["quench_clear_rate"],
                         "quench_read_rate": headline["quench_read_rate"]},
            "standing": {"winner_post": standing["winner_post"], "n_post": standing["n_ignited_post"],
                         "old_residual_post": standing["old_residual_post"], "new_rate_post": standing["new_rate_post"],
                         "quench_read_rate": standing["quench_read_rate"], "went_empty": standing["went_empty"],
                         "swapped": standing["swapped"]},
            "no_quench": {"winner_post": noq["winner_post"], "n_post": noq["n_ignited_post"],
                          "old_residual_post": noq["old_residual_post"], "new_rate_post": noq["new_rate_post"],
                          "co_ignition": noq["co_ignition"], "incumbent_held": noq["incumbent_held"],
                          "swapped": noq["swapped"]},
            "lesion": {"winner_post": lesion["winner_post"], "n_post": lesion["n_ignited_post"],
                       "old_residual_post": lesion["old_residual_post"], "new_rate_post": lesion["new_rate_post"],
                       "co_ignition": lesion["co_ignition"], "incumbent_held": lesion["incumbent_held"],
                       "swapped": lesion["swapped"]},
        },
        "measurements": {"headline": headline, "standing": standing, "no_quench": noq, "lesion": lesion,
                         "substrate_hash": h_a},
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        hd = headline
        print(f"[quench-evict seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    HEADLINE(transient): win {hd['winner_pre']}->{hd['winner_post']} n {hd['n_ignited_pre']}->"
              f"{hd['n_ignited_post']} | A_midclear={hd['old_rate_midclear']:.3f} (g_i {hd['gi_rest']:.0f}->"
              f"{hd['gi_midclear']:.0f}) old_res={hd['old_residual_post']:.3f} new={hd['new_rate_post']:.3f} "
              f"empty={hd['empty_steps']} q_read={hd['quench_read_rate']:.3f} swapped={hd['swapped']} "
              f"reignite={hd['reignite_ok']}", flush=True)
        print(f"    STANDING : win->{standing['winner_post']} n->{standing['n_ignited_post']} "
              f"new={standing['new_rate_post']:.3f} q_read={standing['quench_read_rate']:.3f} "
              f"swapped={standing['swapped']}  | NO_QUENCH: win->{noq['winner_post']} n->{noq['n_ignited_post']} "
              f"old_res={noq['old_residual_post']:.3f} co_ign={noq['co_ignition']} swapped={noq['swapped']}", flush=True)
        print(f"    LESION   : win->{lesion['winner_post']} n->{lesion['n_ignited_post']} "
              f"old_res={lesion['old_residual_post']:.3f} swapped={lesion['swapped']}  | "
              f"open_loop={open_loop} load_bearing={load_bearing} standing_fails={standing_fails} "
              f"no_quench_fails={no_quench_fails} det={seed_deterministic} additive={additive_substrate}", flush=True)
    return result


# ── smoke: an operating-point grid on one seed (find a swap + open-loop point) ─────────────────────────────────
def run_smoke(seed, args):
    print(f"[quench-evict smoke] seed={seed} — operating-point grid", flush=True)
    w_grid = args.w_grid if args.w_grid else [args.quench_w]
    win_grid = args.window_grid if args.window_grid else [args.quench_window]
    ov_grid = args.overlap_grid if args.overlap_grid else [args.b_overlap]
    thal_grid = args.thal_grid if args.thal_grid else [args.quench_thal_w]
    grid = []
    for w in w_grid:
        for win in win_grid:
            for ov in ov_grid:
                for qtw in thal_grid:
                    r = evaluate_seed(seed, quench_w=float(w), quench_window=int(win), b_overlap=int(ov),
                                      quench_thal_w=float(qtw), quench_drive_pA=args.quench_drive_pA,
                                      quench_n=args.quench_n, heterogeneity=not args.no_heterogeneity, verbose=True)
                    grid.append({"quench_w": float(w), "quench_window": int(win), "b_overlap": int(ov),
                                 "quench_thal_w": float(qtw), "seed_go": r["seed_go"], "go_gate": r["go_gate"],
                                 "anti_cheats": r["anti_cheats"], "residual": r["residual"]})
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_quench_evict_overwrite_derisk", "mode": "smoke", "seed": seed, "grid": grid}, f,
                  indent=2, default=str)
    print(f"\n[quench-evict smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[quench-evict six-seed] seeds={seeds} @ w={args.quench_w} window={args.quench_window} "
          f"drive={args.quench_drive_pA}", flush=True)
    per_seed = []
    for s in seeds:
        r = evaluate_seed(s, quench_w=args.quench_w, quench_window=args.quench_window, b_overlap=args.b_overlap,
                          quench_thal_w=args.quench_thal_w, quench_drive_pA=args.quench_drive_pA,
                          quench_n=args.quench_n, heterogeneity=not args.no_heterogeneity, verbose=True)
        per_seed.append(r)
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swap_ok"])
    n_open = sum(1 for r in per_seed if r["go_gate"]["open_loop"])
    n_lb = sum(1 for r in per_seed if r["go_gate"]["load_bearing"])
    n_stand = sum(1 for r in per_seed if r["go_gate"]["standing_fails"])
    n_noq = sum(1 for r in per_seed if r["go_gate"]["no_quench_fails"])
    n_reig = sum(1 for r in per_seed if r["go_gate"]["reignite_ok"])
    n_add = sum(1 for r in per_seed if r["anti_cheats"]["additive_substrate"])
    pooled_go = bool(n_go >= 5 and n_swap >= 5 and n_open == 6 and n_lb >= 5 and n_stand == 6 and n_noq == 6
                     and n_reig >= 5)
    summary = {"runner": "_gnw_quench_evict_overwrite_derisk", "mode": "six_seed", "seeds": seeds,
               "operating_point": per_seed[0]["operating_point"],
               "counts": {"seed_go": n_go, "swap_ok": n_swap, "open_loop": n_open, "load_bearing": n_lb,
                          "standing_fails": n_stand, "no_quench_fails": n_noq, "reignite_ok": n_reig,
                          "additive_substrate": n_add, "n_seeds": len(seeds)},
               "pooled_go": pooled_go, "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[quench-evict six-seed] seed_go {n_go}/6 swap {n_swap}/6 open_loop {n_open}/6 load_bearing {n_lb}/6 "
          f"standing_fails {n_stand}/6 no_quench_fails {n_noq}/6 reignite {n_reig}/6 additive {n_add}/6 "
          f"-> POOLED_GO={pooled_go}", flush=True)
    print(f"[quench-evict six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW quench-evict overwrite de-risk (affect active-clear transfer)")
    ap.add_argument("--smoke", action="store_true", help="operating-point grid on one seed")
    ap.add_argument("--six-seed", action="store_true", help="42/43/44/100/101/102 at the chosen operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quench-w", type=float, default=QUENCH_W)
    ap.add_argument("--quench-window", type=int, default=QUENCH_WINDOW)
    ap.add_argument("--quench-drive-pA", type=float, default=QUENCH_DRIVE_PA)
    ap.add_argument("--quench-n", type=int, default=QUENCH_N)
    ap.add_argument("--quench-thal-w", type=float, default=0.0, help="quench_fs -> thal GABA_A (loop-reset limb)")
    ap.add_argument("--w-rec", type=float, default=HEADLINE_W_REC, help="workspace recurrence weight (supra vs sub-crit)")
    ap.add_argument("--b-overlap", type=int, default=0, help="steps of concurrent B-drive during the quench tail")
    ap.add_argument("--w-grid", type=float, nargs="*", default=None, help="smoke: quench weights")
    ap.add_argument("--window-grid", type=int, nargs="*", default=None, help="smoke: clear windows")
    ap.add_argument("--overlap-grid", type=int, nargs="*", default=None, help="smoke: b_overlap values")
    ap.add_argument("--thal-grid", type=float, nargs="*", default=None, help="smoke: quench_thal_w values")
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_quench_evict.json")
    args = ap.parse_args()

    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    # default: a single-seed evaluation at the chosen point
    r = evaluate_seed(args.seed, quench_w=args.quench_w, quench_window=args.quench_window, b_overlap=args.b_overlap,
                      quench_thal_w=args.quench_thal_w, quench_drive_pA=args.quench_drive_pA, quench_n=args.quench_n,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_quench_evict_overwrite_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[quench-evict] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
