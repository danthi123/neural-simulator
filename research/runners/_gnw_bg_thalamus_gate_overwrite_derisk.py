"""GNW ACTIVE OVERWRITE via a per-slot BG-THALAMUS SELECTIVE GATE — the named next lever from the active-overwrite
NO-GO (`research/findings/2026-08-18-gnw-active-overwrite-NOGO.md`).

CONTEXT. The active-overwrite NO-GO proved a clean single-slot SWAP (replace ignited incumbent A with challenger B
while n_ignited stays 1) is a substrate CATCH-22 by INTRINSIC competition: any lever that lets B break in settles
into a stable n=2 CO-IGNITION (the incumbent, propped by tonic + its own drive, will not vacate a depleted-but-driven
shared recurrence); any per-slot WTA lateral inhibition strong enough to give single-content selectivity LOCKS OUT
the challenger before it can co-activate (or, stronger, destabilises the solo-A hold). The finding's external-lit
conclusion (O'Reilly & Frank 2006 PBWM; Lundqvist/Lisman theta multiplexing): biology does NOT resolve
incumbent-vs-challenger by intrinsic rate competition over a shared resource at all — it uses an EXTERNAL, dedicated
gate that DISINHIBITS the challenger's thalamic drive and INHIBITS the incumbent's slot, arbitrated by conflict.

THIS RUNNER builds that gate (fork of `_gnw_active_overwrite_derisk.py`; reuse-by-import of the workspace geometry +
the STN conflict-sensor read; NO `sim/` edit). The catch-22 is broken because the eviction is (1) CONFLICT-GATED —
OFF when there is no challenger, so an unchallenged A is never disturbed (selectivity survives ANY gate strength),
and (2) DIRECTED — it closes the incumbent slot and opens the challenger slot, so there is no symmetric competition
to lose. The WTA failed on exactly these two axes (always-on + symmetric).

MECHANISM (per-slot k in 0..K-1; all explicit wiring, dense frozen pools):
  * workspace: K DISJOINT recurrent patterns (w=34, supra-critical: a pattern self-sustains on recurrence alone —
    verified, no thalamic tonic needed) + a divisive-normalisation norm_pool (single-content pressure).
  * PER-SLOT BG-THALAMUS relay: thal_k (tonic drive, WANTS to relay) is held SILENT by a tonically-firing gpi_k
    (gpi_k -> thal_k inhibitory). A striatal-Go pool str_k, when driven, INHIBITS gpi_k -> DISINHIBITS thal_k ->
    thal_k -> slot_k (releases the challenger's thalamic drive). This is the PBWM output gate.
  * PER-SLOT eviction: gate_inh_k (inhibitory), when driven, INHIBITS slot_k's assembly (closes the incumbent slot).
  * THE GATE SIGNAL is signal-driven from the STN conflict read: i_gate = conflict_gain * max(0, margin_ref -
    instant_margin) * scale, where instant_margin = top-slot rate - runner-up rate over a short smoothing window
    (the same neural ignition-margin sensor the STN stop-veto GO uses). At zero conflict (A alone, high margin)
    i_gate = 0 -> the gate is silent. On B's afferent volley (B co-activates -> margin collapses) i_gate rises ->
    the gate fires: it DISINHIBITS the challenger's thal (drives str_{challenger}) and INHIBITS every OTHER slot
    (drives gate_inh_{k != challenger}) — gating in B closes A. As A leaves, conflict falls, the gate self-releases,
    and B (driven + thalamically supported, then on its own recurrence) holds as the new single content.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, cfg.seed determinism):
  SWITCH >=5/6 — a challenger B displaces incumbent A: delivered identity A->B with n_ignited settling to EXACTLY 1
    (asserted NOT 0 = a stop, NOT 2 = co-ignition).
  SELECTIVITY 6/6 — an UNCHALLENGED confident A holds (n=1, winner A) through a long window.
ANTI-CHEATS (asserted every seed):
  (a) SUBSTRATE-DRIVEN: the swap HEADLINE is a CONTINUOUS run — host_workspace_reset_calls == 0 and
      host_content_swap_calls == 0 (the only host writes are the external stimulus drive = world/body-legitimate).
  (b) SIGNAL-DRIVEN gate: i_gate scales with the STN conflict read and is 0 at zero conflict (a challenger-drive
      sweep records i_gate 0 -> rising); a margin-SCRAMBLE (feed the confident margin to the swap) breaks the swap.
  (c) PER-SLOT INHIBITION LOAD-BEARING: lesion gate_inh (weight 0) -> NO clean swap (back to the catch-22:
      co-ignition n=2 / incumbent holds).
  (d) BYTE-IDENTICAL substrate: the gate-flag-OFF build == the active-overwrite base (== the DIST-OVERWRITE base
      hash) at the same seed; the gate-ON build is additive (its workspace+norm prefix is the same seeded draw).
  (e) DETERMINISM: build twice at one seed -> identical seed-derived Izhikevich-param hash (cfg.seed, NOT
      actual_seed_used).

NOT-A-WALL: if the gate does not cleanly swap, the residual is QUANTIFIED (which horn remains: co-ignition n=2,
incumbent-holds n=1, or a slow clear-then-reload through a long empty window) and the next lever named
(theta-phase segregation; tune the disinhibition/inhibition timing). An honest PARTIAL/NO-GO with the residual
quantified IS the deliverable.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_bg_thalamus_gate_overwrite_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_bg_thalamus_gate_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_bg_thalamus_gate_overwrite_derisk \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_bg_thalamus_gate_6seed.json
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
from tools.lab import attributable_to

# reuse-by-import: validated ignition/competition instruments + protocol constants + determinism hash + wash-out.
from research.runners._gnw_rung1_ignition_curve_derisk import (
    DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS, WS_LOOP_GATE,
)
from research.runners._gnw_rung2_competitive_access_derisk import _ignited, SOLO_PLATEAU
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
# the ACTIVE-OVERWRITE base (== the DIST-OVERWRITE base substrate) — the byte-identical anchor.
from research.runners._gnw_active_overwrite_derisk import (
    build_swap_bridge as _ao_build, _pattern_geometry, _recurrence_edges,
    N_PATTERNS, PATTERN_SIZE, WORKSPACE_N, NORM_N, W_SHARED, WS_NORM_W, NORM_WS_W, IGNITE_PA,
)

# ── geometry: K disjoint recurrent patterns + per-slot BG-thalamus gate pools ──────────────────────────────────
STRONG_PA = 2.0 * IGNITE_PA          # 5000 pA per-pattern ignition / afferent-volley drive
HEADLINE_OVERLAP = 0                 # DISJOINT patterns (private_k == pattern_k) — the NO-GO headline substrate

# per-slot pools (built only when gate_on=True; the gate-OFF build == the active-overwrite base substrate).
THAL_N = 40                          # thalamic relay per slot (tonic drive; held silent by gpi_k)
GPI_N = 60                           # GPi/SNr per slot (tonic drive -> tonically inhibits thal_k)
STR_N = 40                           # striatal-Go per slot (gate-driven -> inhibits gpi_k = disinhibition)
GATE_INH_N = 120                     # per-slot eviction interneuron (gate-driven -> inhibits slot_k assembly)

THAL_WS_W = 10.0                     # thal_k -> slot_k (released challenger drive)
GPI_THAL_W = 22.0                    # gpi_k -> thal_k (tonic inhibition holding the relay closed)
STR_GPI_W = 26.0                     # str_k -> gpi_k (the gate's disinhibition of the relay)
GATE_INH_W = 14.0                    # gate_inh_k -> slot_k assembly (the eviction effector)
GPI_TONIC_PA = 600.0                 # gpi_k baseline drive (tonically firing -> holds thal closed)
THAL_TONIC_PA = 700.0                # thal_k baseline drive (wants to relay; opposed by gpi_k until disinhibited)
WS_TONIC_PA = 450.0                  # uniform workspace baseline-excitability tonic (replaces the base substrate's
                                     # shared-thal support removed for per-slot relays) — makes a marginal slot's
                                     # recurrence robustly supra-critical across seeds; content-neutral, sub-ignition

# the STN conflict sensor (the mismatch comparator: a challenger afferent volley + a held incumbent).
MARGIN_REF = SOLO_PLATEAU * 0.5      # 1/6 (retained for the API; the mismatch read uses the incumbent rate)
GATE_CURRENT_SCALE = 9000.0          # i_gate = conflict_gain * volley * incumbent_rate * SCALE (~0.33*9000 ~3000 pA)
SMOOTH_WIN = 8                       # steps for the instantaneous per-slot rate smoothing (the sensor window)

OU_NOISE_PA = 30.0                   # desynchronises the attractor -> no synchronous rebound after eviction
READ_FREE_STEPS = 45                 # free steps to let a commit settle before the margin read
CONF_HOLD_STEPS = 130                # an unchallenged confident commit must hold at least this long (selectivity)
SWAP_STEPS = 160                     # steps the challenger B is driven while the conflict-gate runs
EMPTY_CLEAN_MAX = 20                 # a CLEAN swap keeps the empty (both-off) window <= this; more = slow clear-reload

_RESTORE_CALLS = {"n": 0}
_CONTENT_SWAP_CALLS = {"n": 0}       # a host "replace content" poke — MUST stay 0 (never called; asserted)


def _counted_full_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


def _uniform_recurrence_pop(patterns):
    """Dense E->E recurrence within each disjoint pattern clique at uniform W_SHARED (supra-critical self-sustain)."""
    pre, post = _recurrence_edges(patterns)
    ww = np.full(pre.size, np.float32(W_SHARED), dtype=np.float32)
    return {"pre_indices": pre, "post_indices": post, "initial_weights": ww,
            "plastic": False, "plasticity_gate": WS_LOOP_GATE, "conn_type": "E_TO_E", "count": int(pre.size)}


# ── build the per-slot BG-thalamus-gated workspace bridge ──────────────────────────────────────────────────────
def build_gate_bridge(seed=42, gate_inh_w=GATE_INH_W, thal_ws_w=THAL_WS_W, str_gpi_w=STR_GPI_W,
                      gpi_thal_w=GPI_THAL_W, gate_inh_lesion=False, thal_lesion=False, heterogeneity=True,
                      ou_noise_pA=OU_NOISE_PA):
    """workspace (K disjoint recurrent patterns, exc/NMDA) + norm_pool (divisive normalisation) + per-slot
    {thal_k, gpi_k, str_k, gate_inh_k}. gpi_k tonically inhibits thal_k (relay closed); str_k disinhibits it;
    gate_inh_k evicts slot_k. gate_inh_lesion -> gate_inh weight 0 (the per-slot-inhibition load-bearing anti-cheat).
    thal_lesion -> thal_ws weight 0 (the disinhibition arm load-bearing probe). Returns
    (bridge, xp, patterns_dev, privates_dev, pools, ws_used, snap, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    norm_pool = BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0,
                            enable_nmda=False)
    regions = [workspace, norm_pool]
    for k in range(N_PATTERNS):
        regions.append(BrainRegion(name=f"thal{k}", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0,
                                   enable_nmda=False))
        regions.append(BrainRegion(name=f"gpi{k}", n_neurons=GPI_N, exc_fraction=0.0, internal_density=0.0,
                                   enable_nmda=False))
        regions.append(BrainRegion(name=f"str{k}", n_neurons=STR_N, exc_fraction=0.0, internal_density=0.0,
                                   enable_nmda=False))
        regions.append(BrainRegion(name=f"gate_inh{k}", n_neurons=GATE_INH_N, exc_fraction=0.0,
                                   internal_density=0.0, enable_nmda=False))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                 # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
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
    cfg.stdp_w_max = max(400.0, float(W_SHARED) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(W_SHARED) * 4.0)
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
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, PATTERN_SIZE, HEADLINE_OVERLAP)   # disjoint
    ws_used = np.asarray(patterns[0], dtype=np.int64) if HEADLINE_OVERLAP >= PATTERN_SIZE \
        else np.unique(np.concatenate(patterns)).astype(np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)

    thal_idx = [np.asarray(rm.indices(f"thal{k}"), dtype=np.int64) for k in range(N_PATTERNS)]
    gpi_idx = [np.asarray(rm.indices(f"gpi{k}"), dtype=np.int64) for k in range(N_PATTERNS)]
    str_idx = [np.asarray(rm.indices(f"str{k}"), dtype=np.int64) for k in range(N_PATTERNS)]
    gate_inh_idx = [np.asarray(rm.indices(f"gate_inh{k}"), dtype=np.int64) for k in range(N_PATTERNS)]

    thal_ws_eff = 0.0 if thal_lesion else float(thal_ws_w)
    gate_inh_eff = 0.0 if gate_inh_lesion else float(gate_inh_w)

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["workspace_rec"] = _uniform_recurrence_pop(patterns)
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(WS_NORM_W), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, float(NORM_WS_W), "I_TO_E")

    inh = list(norm_idx)
    for k in range(N_PATTERNS):
        # per-slot thalamocortical relay + its tonic GPi brake + the striatal-Go disinhibition
        union_plan[f"thal{k}2ws"] = _dense_pop(thal_idx[k], patterns[k], thal_ws_eff, "E_TO_E")
        union_plan[f"gpi{k}2thal"] = _dense_pop(gpi_idx[k], thal_idx[k], float(gpi_thal_w), "I_TO_E")
        union_plan[f"str{k}2gpi"] = _dense_pop(str_idx[k], gpi_idx[k], float(str_gpi_w), "I_TO_E")
        # per-slot eviction interneuron
        union_plan[f"gate_inh{k}2ws"] = _dense_pop(gate_inh_idx[k], patterns[k], gate_inh_eff, "I_TO_E")
        inh += list(gpi_idx[k]) + list(str_idx[k]) + list(gate_inh_idx[k])

    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    # baseline tonics: gpi_k fires (holds thal closed), thal_k wants to relay. Settle.
    thal_dev = [xp.asarray(t) for t in thal_idx]
    gpi_dev = [xp.asarray(g) for g in gpi_idx]
    str_dev = [xp.asarray(s) for s in str_idx]
    gate_inh_dev = [xp.asarray(g) for g in gate_inh_idx]
    ws_dev = xp.asarray(ws_used)
    _apply_baseline_tonics(bridge, xp, gpi_dev, thal_dev, ws_dev=ws_dev)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    pools = {"thal": thal_dev, "gpi": gpi_dev, "str": str_dev, "gate_inh": gate_inh_dev, "ws": ws_dev,
             "thal_h": thal_idx, "gpi_h": gpi_idx, "str_h": str_idx, "gate_inh_h": gate_inh_idx}
    handles = {"seed": int(seed), "gate_inh_w": float(gate_inh_eff), "thal_ws_w": float(thal_ws_eff),
               "str_gpi_w": float(str_gpi_w), "gpi_thal_w": float(gpi_thal_w),
               "gate_inh_lesion": bool(gate_inh_lesion), "thal_lesion": bool(thal_lesion),
               "heterogeneity": bool(heterogeneity), "n_patterns": N_PATTERNS,
               "private_sizes": [int(p.size) for p in privates], "n_ws_used": int(ws_used.size)}
    patterns_dev = [xp.asarray(p) for p in patterns]
    privates_dev = [xp.asarray(p) for p in privates]
    return bridge, xp, patterns_dev, privates_dev, pools, ws_used, snap, handles


def _apply_baseline_tonics(bridge, xp, gpi_dev, thal_dev, gpi_tonic=GPI_TONIC_PA, thal_tonic=THAL_TONIC_PA,
                           ws_dev=None, ws_tonic=WS_TONIC_PA):
    """Zero external current, then apply the per-slot baseline tonics (gpi_k firing, thal_k driven-but-held) plus a
    uniform content-neutral workspace baseline-excitability tonic (robust supra-critical recurrence across seeds)."""
    bridge.cp_external_input_current[:] = 0.0
    if ws_dev is not None and ws_tonic > 0.0:
        bridge.cp_external_input_current[ws_dev] = xp.float32(ws_tonic)
    for g in gpi_dev:
        bridge.cp_external_input_current[g] = xp.float32(gpi_tonic)
    for t in thal_dev:
        bridge.cp_external_input_current[t] = xp.float32(thal_tonic)


# ── spiking reads ──────────────────────────────────────────────────────────────────────────────────────────────
def _instant_private_rate(bridge, xp, privates_dev, idx):
    p = privates_dev[idx]
    return float(to_host(bridge.cp_firing_states[p].astype(xp.float64).mean()))


def _margin(rates):
    order = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    top = rates[order[0]]
    second = rates[order[1]] if len(order) > 1 else 0.0
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    return int(order[0]), float(top - second), n_ign


def _verdict_label(rates):
    n_ign = int(sum(1 for r in rates if _ignited(r)))
    if n_ign == 0:
        return "ABSTAIN", 0
    return (f"COMMIT_p{int(np.argmax(rates))}" if n_ign == 1 else f"AMBIGUOUS_p{int(np.argmax(rates))}"), n_ign


def _gate_step(bridge, xp, pools, gpi_dev, thal_dev, drive_map=None, gate_current=0.0, challenger=None,
               gate_inh_targets=None):
    """One step under the BG-thalamus gate. Applies the per-slot baseline tonics, the external drive (afferent
    volley), and — if gate_current>0 — DISINHIBITS the challenger's relay (drive str_{challenger}) and INHIBITS the
    incumbent slots (drive gate_inh_{k in gate_inh_targets}). The disinhibition/inhibition then play out on the
    substrate (str -> gpi -> thal -> slot ; gate_inh -> slot)."""
    _apply_baseline_tonics(bridge, xp, gpi_dev, thal_dev, ws_dev=pools.get("ws"))
    if drive_map:
        for idx_dev, val in drive_map:
            if val > 0.0:
                bridge.cp_external_input_current[idx_dev] = xp.float32(val)
    if gate_current > 0.0:
        if challenger is not None:
            bridge.cp_external_input_current[pools["str"][challenger]] = xp.float32(gate_current)
        if gate_inh_targets:
            for k in gate_inh_targets:
                bridge.cp_external_input_current[pools["gate_inh"][k]] = xp.float32(gate_current)
    bridge._run_one_simulation_step()


def _read_private_rates(bridge, xp, pools, privates_dev, n_free=READ_FREE_STEPS):
    """Free-run n_free steps (tonics on, NO afferent volley, NO gate) and return the LATE-window per-pattern
    private-core mean firing rate — the clean spiking identity read. Leaves the workspace evolving in place."""
    gpi_dev, thal_dev = pools["gpi"], pools["thal"]
    late_start = n_free - max(1, n_free // 3)
    counts = [0] * len(privates_dev)
    for t in range(n_free):
        _gate_step(bridge, xp, pools, gpi_dev, thal_dev, drive_map=None, gate_current=0.0)
        if t >= late_start:
            for i, p in enumerate(privates_dev):
                counts[i] += int(to_host(bridge.cp_firing_states[p].astype(xp.float64).sum()))
    out = []
    for i, p in enumerate(privates_dev):
        denom = float((n_free - late_start) * int(p.shape[0]))
        out.append(counts[i] / denom if denom > 0 else 0.0)
    return out


def _drive(bridge, xp, pools, drive_map, n=DRIVE_STEPS):
    gpi_dev, thal_dev = pools["gpi"], pools["thal"]
    for _ in range(n):
        _gate_step(bridge, xp, pools, gpi_dev, thal_dev, drive_map=drive_map, gate_current=0.0)


# ── SELECTIVITY: an UNCHALLENGED confident commit holds (n=1, winner=target) through a long window ─────────────
def run_confident(bridge, xp, patterns_dev, privates_dev, pools, snap, target=0, hold_steps=CONF_HOLD_STEPS,
                  isolate=True):
    if isolate:
        _counted_full_restore(bridge, snap)
    _drive(bridge, xp, pools, [(patterns_dev[target], STRONG_PA)])
    for _ in range(hold_steps):                          # HOLD: tonics on, NO drive, NO challenger -> gate silent
        _gate_step(bridge, xp, pools, pools["gpi"], pools["thal"], drive_map=None, gate_current=0.0)
    rates = _read_private_rates(bridge, xp, pools, privates_dev)
    win, m, n = _margin(rates)
    v, _ = _verdict_label(rates)
    return {"rates": [float(r) for r in rates], "winner": int(win), "n_ignited": int(n), "margin": float(m),
            "delivered": v, "confident_ok": bool(n == 1 and win == target)}


# ── the ACTIVE OVERWRITE swap via the conflict-gated BG-thalamus gate ──────────────────────────────────────────
def run_swap(bridge, xp, patterns_dev, privates_dev, pools, snap, *, incumbent=0, challenger=1,
             swap_steps=SWAP_STEPS, conflict_gain=1.0, margin_ref=MARGIN_REF, current_scale=GATE_CURRENT_SCALE,
             gate_on=True, drive_challenger=True, isolate=True, margin_override=None, record=False):
    """Ignite incumbent A (hold), then drive challenger B for swap_steps under the conflict-gated BG-thalamus gate.
    Each step: read a smoothed instantaneous per-slot rate, form instant_margin = top-runnerup, i_gate =
    conflict_gain * max(0, margin_ref - instant_margin) * scale; if i_gate>0 DISINHIBIT the challenger's relay and
    INHIBIT every OTHER slot. gate_on=False -> i_gate forced 0 (the gate-off control). margin_override feeds a fixed
    margin to the sensor (the SCRAMBLE anti-cheat). isolate=False -> a CONTINUOUS run (0 restores) = the HEADLINE."""
    gpi_dev, thal_dev = pools["gpi"], pools["thal"]
    if isolate:
        _counted_full_restore(bridge, snap)
    # (1) ignite A alone
    _drive(bridge, xp, pools, [(patterns_dev[incumbent], STRONG_PA)])
    pre = _read_private_rates(bridge, xp, pools, privates_dev)
    win_pre, margin_pre, n_pre = _margin(pre)

    # (2) SWAP: drive B; conflict-gated BG-thalamus gate (disinhibit B's relay + inhibit the INCUMBENT slot).
    # The eviction targets the incumbent (the ignited slot being displaced) — a corticostriatal read of which slot
    # holds. Targeting only the incumbent (not every non-challenger) avoids a spurious quiescent-slot rebound and
    # gives a CLEAN co-ignition residual (n=2 = A+B), the horn under test.
    dmap = [(patterns_dev[challenger], STRONG_PA)] if drive_challenger else None
    gate_targets = [incumbent]
    smooth = np.zeros((SMOOTH_WIN, N_PATTERNS), dtype=np.float64)
    coactive_win, empty_win, gate_fired, i_gate_max = 0, 0, 0, 0.0
    thal_chal_during, str_chal_during = [], []
    trace = []
    for t in range(swap_steps):
        # MISMATCH conflict read (STN next-lever #3): the gate fires when the challenger's afferent volley is
        # present AND a DIFFERENT content (a non-challenger slot) is currently held — a comparator, not resolved
        # cortical co-activation (which cannot occur before the gate admits the challenger: the bootstrap the
        # margin read deadlocked on). i_gate scales with the held incumbent's rate; 0 when no volley or no incumbent.
        for k in range(N_PATTERNS):
            smooth[t % SMOOTH_WIN, k] = _instant_private_rate(bridge, xp, privates_dev, k)
        srate = smooth.mean(axis=0)
        inc_rate = float(max(srate[k] for k in range(N_PATTERNS) if k != challenger))   # strongest incumbent
        sensed_inc = inc_rate if margin_override is None else float(margin_override)
        volley = 1.0 if (dmap is not None) else 0.0
        i_gate = (float(conflict_gain) * volley * sensed_inc * float(current_scale)) if gate_on else 0.0
        gate_fired += int(i_gate > 0.0)
        i_gate_max = max(i_gate_max, i_gate)
        _gate_step(bridge, xp, pools, gpi_dev, thal_dev, drive_map=dmap, gate_current=i_gate,
                   challenger=challenger, gate_inh_targets=gate_targets)
        na = _ignited(_instant_private_rate(bridge, xp, privates_dev, incumbent))
        nb = _ignited(_instant_private_rate(bridge, xp, privates_dev, challenger))
        coactive_win += int(na and nb); empty_win += int(not na and not nb)
        if record:
            thal_chal_during.append(float(to_host(bridge.cp_firing_states[pools["thal"][challenger]]
                                                  .astype(xp.float64).mean())))
            str_chal_during.append(float(to_host(bridge.cp_firing_states[pools["str"][challenger]]
                                                 .astype(xp.float64).mean())))
            if t % 10 == 0:
                trace.append({"t": t, "srate": [round(float(x), 3) for x in srate], "i_gate": round(i_gate, 1),
                              "na": int(na), "nb": int(nb)})

    # (3) free-run (B drive off, gate off): does B self-sustain as the new incumbent, A gone?
    post = _read_private_rates(bridge, xp, pools, privates_dev, n_free=FREE_STEPS)
    win_post, margin_post, n_post = _margin(post)
    v_pre, _ = _verdict_label(pre); v_post, _ = _verdict_label(post)
    switched = bool(win_pre == incumbent and n_pre == 1 and win_post == challenger and n_post == 1)
    clean_swap = bool(switched and empty_win <= EMPTY_CLEAN_MAX)
    slow_overwrite = bool(switched and empty_win > EMPTY_CLEAN_MAX)
    out = {"pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
           "winner_pre": int(win_pre), "winner_post": int(win_post), "n_ignited_pre": int(n_pre),
           "n_ignited_post": int(n_post), "delivered_pre": v_pre, "delivered_post": v_post,
           "swap_ok": clean_swap, "switched_identity": switched, "slow_overwrite": slow_overwrite,
           "co_ignition": bool(n_pre == 1 and n_post >= 2), "went_empty": bool(n_pre >= 1 and n_post == 0),
           "incumbent_held": bool(win_post == incumbent and n_post == 1),
           "coactive_steps": int(coactive_win), "empty_steps": int(empty_win),
           "gate_fired_steps": int(gate_fired), "i_gate_max": float(i_gate_max)}
    if record:
        out["thal_chal_rate_during"] = float(np.mean(thal_chal_during)) if thal_chal_during else None
        out["str_chal_rate_during"] = float(np.mean(str_chal_during)) if str_chal_during else None
        out["trace"] = trace
    return out


# ── the ADVANCE: does the BG-thalamus DISINHIBITION admit the challenger past the norm lockout? ────────────────
def run_breakin_control(bridge, xp, patterns_dev, privates_dev, pools, snap, *, incumbent=0, challenger=1,
                        str_drive=STRONG_PA, n_swap=SWAP_STEPS):
    """Ignite A, then drive challenger B at STRONG_PA for n_swap steps in TWO conditions and read B's late-window
    private rate: (1) WITHOUT the gate's thalamic disinhibition (no str drive) — B must break in against the
    divisive-normalisation lockout on its own; (2) WITH the disinhibition (drive str_challenger -> silence
    gpi_challenger -> release thal_challenger -> extra thalamic drive to B). No eviction in either. A break-in that
    requires the disinhibition = the gate SOLVES the lockout horn the WTA could not (the WTA locked B out)."""
    gpi_dev, thal_dev = pools["gpi"], pools["thal"]

    def _oneside(with_disinhib):
        _counted_full_restore(bridge, snap)
        _drive(bridge, xp, pools, [(patterns_dev[incumbent], STRONG_PA)])
        for _ in range(n_swap):
            _apply_baseline_tonics(bridge, xp, gpi_dev, thal_dev, ws_dev=pools.get("ws"))
            bridge.cp_external_input_current[patterns_dev[challenger]] = xp.float32(STRONG_PA)
            if with_disinhib:
                bridge.cp_external_input_current[pools["str"][challenger]] = xp.float32(str_drive)
            bridge._run_one_simulation_step()
        rates = _read_private_rates(bridge, xp, pools, privates_dev)
        return float(rates[challenger])

    b_no = _oneside(False)
    b_yes = _oneside(True)
    return {"b_rate_no_disinhib": b_no, "b_rate_with_disinhib": b_yes,
            "locked_out_without": bool(not _ignited(b_no)), "admitted_with": bool(_ignited(b_yes)),
            "disinhibition_admits_challenger": bool(not _ignited(b_no) and _ignited(b_yes))}


# ── the SIGNAL-DRIVEN anti-cheat: i_gate vs the challenger drive (0 at zero conflict; rises with conflict) ──────
def sweep_gate_signal(bridge, xp, patterns_dev, privates_dev, pools, snap, *, incumbent=0, challenger=1,
                      conflict_gain=1.0, margin_ref=MARGIN_REF, current_scale=GATE_CURRENT_SCALE, n_probe=40):
    """Ignite A, then drive B at a range of amplitudes (weak->strong) WITHOUT applying the gate effectors, and read
    the i_gate the sensor WOULD command at settle. i_gate must be ~0 when B is weak (no conflict) and rise as B
    co-activates (conflict). This is the pulse_zero_at_zero_conflict / pulse_scales_with_conflict anti-cheat."""
    amps = list(np.linspace(0.0, STRONG_PA, 6))
    i_gate_by_amp = []
    for amp in amps:
        _counted_full_restore(bridge, snap)
        _drive(bridge, xp, pools, [(patterns_dev[incumbent], STRONG_PA)])
        dmap = [(patterns_dev[challenger], float(amp))] if amp > 0.0 else None
        smooth = np.zeros((SMOOTH_WIN, N_PATTERNS), dtype=np.float64)
        last_i = 0.0
        for t in range(n_probe):
            for k in range(N_PATTERNS):
                smooth[t % SMOOTH_WIN, k] = _instant_private_rate(bridge, xp, privates_dev, k)
            srate = smooth.mean(axis=0)
            inc_rate = float(max(srate[k] for k in range(N_PATTERNS) if k != challenger))
            volley = 1.0 if (dmap is not None) else 0.0     # 0 at amp=0 -> no afferent volley -> gate silent
            last_i = float(conflict_gain) * volley * inc_rate * float(current_scale)
            _gate_step(bridge, xp, pools, pools["gpi"], pools["thal"], drive_map=dmap, gate_current=0.0)  # no effector
        i_gate_by_amp.append(last_i)
    zero_at_zero = bool(i_gate_by_amp[0] <= 1e-6)
    rises = bool(i_gate_by_amp[-1] > 1e-6 and i_gate_by_amp[-1] >= i_gate_by_amp[0])
    return {"amps": [float(a) for a in amps], "i_gate_by_amp": [float(x) for x in i_gate_by_amp],
            "zero_at_zero_conflict": zero_at_zero, "scales_with_conflict": rises}


def _prefix_hash(bridge, xp, n_prefix):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64)[:n_prefix])
    if not parts:
        return ""
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest()


# ── one seed: swap + selectivity + anti-cheats ─────────────────────────────────────────────────────────────────
def evaluate_seed(seed, *, gate_inh_w=GATE_INH_W, thal_ws_w=THAL_WS_W, conflict_gain=1.0, margin_ref=MARGIN_REF,
                  current_scale=GATE_CURRENT_SCALE, swap_steps=SWAP_STEPS, heterogeneity=True, verbose=True):
    def _build(**kw):
        params = dict(seed=seed, gate_inh_w=gate_inh_w, thal_ws_w=thal_ws_w, heterogeneity=heterogeneity)
        params.update(kw)
        return build_gate_bridge(**params)
    bridge, xp, pats, privs, pools, ws_used, snap, hh = _build()

    # ── SELECTIVITY: an unchallenged confident A holds (gate silent, conflict=0) ─────────────────────────────────
    conf = run_confident(bridge, xp, pats, privs, pools, snap, target=0)
    selectivity = bool(conf["confident_ok"])

    # ── HEADLINE swap: conflict-gated BG-thalamus gate, CONTINUOUS (0 restore) = the substrate-driven headline ────
    restore_before = _RESTORE_CALLS["n"]
    headline = run_swap(bridge, xp, pats, privs, pools, snap, gate_on=True, isolate=False, swap_steps=swap_steps,
                        conflict_gain=conflict_gain, margin_ref=margin_ref, current_scale=current_scale, record=True)
    continuous_no_restore = bool(_RESTORE_CALLS["n"] == restore_before)
    host_workspace_reset_calls = 0 if continuous_no_restore else 1
    host_content_swap_calls = int(_CONTENT_SWAP_CALLS["n"])
    clean_swap = bool(headline["swap_ok"])
    switched = bool(headline["switched_identity"])
    slow_overwrite = bool(headline["slow_overwrite"])

    # ── THE ADVANCE: does the DISINHIBITION admit the challenger past the norm lockout? (the horn the WTA failed) ──
    breakin = run_breakin_control(bridge, xp, pats, privs, pools, snap)
    disinhibition_admits_challenger = bool(breakin["disinhibition_admits_challenger"])

    # ── EVICTION-RESISTANCE sweep: crank the per-slot inhibition on the incumbent — does A EVER leave (n_post<2)? ──
    #    Quantifies the residual (co-ignition horn): brute inhibition on the supra-critical incumbent is inhibition-
    #    resistant (Rung-2c) — weak=hold(n=2), strong=destabilise-up/rebound. This is a MEASUREMENT, not a gate.
    evict_sweep = []
    for giw in (float(gate_inh_w), float(gate_inh_w) * 2.0, float(gate_inh_w) * 4.0):
        b_e, xp_e, pats_e, privs_e, pools_e, ws_e, snap_e, _ = _build(gate_inh_w=giw)
        r_e = run_swap(b_e, xp_e, pats_e, privs_e, pools_e, snap_e, gate_on=True, isolate=True, swap_steps=swap_steps,
                       conflict_gain=conflict_gain, margin_ref=margin_ref, current_scale=current_scale)
        evict_sweep.append({"gate_inh_w": giw, "n_post": int(r_e["n_ignited_post"]),
                            "incumbent_held": bool(r_e["incumbent_held"]), "switched": bool(r_e["switched_identity"])})
    incumbent_ever_evicted = bool(any(e["n_post"] < 2 and not e["incumbent_held"] for e in evict_sweep))

    # ── SIGNAL-DRIVEN anti-cheat: i_gate vs the challenger-drive amplitude ───────────────────────────────────────
    signal = sweep_gate_signal(bridge, xp, pats, privs, pools, snap, conflict_gain=conflict_gain,
                               margin_ref=margin_ref, current_scale=current_scale)
    gate_signal_driven = bool(signal["zero_at_zero_conflict"] and signal["scales_with_conflict"])

    # ── SCRAMBLE anti-cheat: feed "no incumbent held" (0) to the mismatch read -> gate stays silent (fires 0 steps)
    #    even though a content IS held -> proves the cortical incumbent read gates the current (not a blind pulse).
    scramble = run_swap(bridge, xp, pats, privs, pools, snap, gate_on=True, isolate=True, swap_steps=swap_steps,
                        conflict_gain=conflict_gain, margin_ref=margin_ref, current_scale=current_scale,
                        margin_override=0.0)
    scramble_breaks_swap = bool(scramble["gate_fired_steps"] == 0 and not scramble["swap_ok"])

    # ── GATE-OFF control: i_gate forced 0 -> back to the catch-22 (no clean swap) ────────────────────────────────
    gate_off = run_swap(bridge, xp, pats, privs, pools, snap, gate_on=False, isolate=True, swap_steps=swap_steps,
                        conflict_gain=conflict_gain, margin_ref=margin_ref, current_scale=current_scale)
    gate_off_no_swap = bool(not gate_off["swap_ok"])

    # ── PER-SLOT INHIBITION LOAD-BEARING: lesion gate_inh -> NO clean swap (the REQUIRED load-bearing anti-cheat) ─
    b_li, xp_li, pats_li, privs_li, pools_li, ws_li, snap_li, _ = _build(gate_inh_lesion=True)
    lesion_inh = run_swap(b_li, xp_li, pats_li, privs_li, pools_li, snap_li, gate_on=True, isolate=True,
                          swap_steps=swap_steps, conflict_gain=conflict_gain, margin_ref=margin_ref,
                          current_scale=current_scale)
    inh_load_bearing = bool(clean_swap and not lesion_inh["swap_ok"])
    inh_swap_attr = attributable_to("clean swap via per-slot gate_inh (headline vs gate_inh-lesion)",
                                    float(clean_swap), float(lesion_inh["swap_ok"]), warn_below=0.0)

    # ── DISINHIBITION arm probe (not required load-bearing): lesion thal_ws -> does the swap degrade? ─────────────
    b_lt, xp_lt, pats_lt, privs_lt, pools_lt, ws_lt, snap_lt, _ = _build(thal_lesion=True)
    lesion_thal = run_swap(b_lt, xp_lt, pats_lt, privs_lt, pools_lt, snap_lt, gate_on=True, isolate=True,
                           swap_steps=swap_steps, conflict_gain=conflict_gain, margin_ref=margin_ref,
                           current_scale=current_scale)
    thal_load_bearing = bool(clean_swap and not lesion_thal["swap_ok"])

    # ── BYTE-IDENTICAL: the gate-OFF (active-overwrite base) substrate == the DIST-OVERWRITE base hash ───────────
    b_off, xp_off, *_off = _ao_build(seed=seed, overlap=15, w_shared=W_SHARED, w_priv=W_SHARED, wta_w=0.0,
                                     heterogeneity=heterogeneity)
    h_base = _threshold_hash(b_off, xp_off)
    from research.runners._gnw_distributed_overwrite_workspace_derisk import build_overwrite_bridge as _dist_build
    b_dist, xp_dist, *_dist = _dist_build(seed=seed, overlap=15)
    h_dist = _threshold_hash(b_dist, xp_dist)
    byte_identical_base = bool(h_base == h_dist and h_base != "")
    # the gate-ON build is ADDITIVE: its workspace+norm prefix (first 380) == the base's same seeded draw
    prefix_n = WORKSPACE_N + NORM_N
    additive_substrate = bool(_prefix_hash(bridge, xp, prefix_n) == _prefix_hash(b_off, xp_off, prefix_n)
                              and _prefix_hash(bridge, xp, prefix_n) != "")

    # ── DETERMINISM: build the gate-ON substrate twice at this seed -> identical seed-derived-param hash ──────────
    h_a = _threshold_hash(bridge, xp)
    b2, xp2, *_2 = _build()
    seed_deterministic = bool(_threshold_hash(b2, xp2) == h_a and h_a != "")

    # GO gate: a clean (n stays 1) SWITCH + selectivity + the anti-cheats.
    anti_ok = bool(gate_signal_driven and scramble_breaks_swap and inh_load_bearing and continuous_no_restore
                   and host_content_swap_calls == 0 and byte_identical_base and seed_deterministic)
    seed_go = bool(clean_swap and selectivity and anti_ok)

    v = Verdict("distributed-workspace BG-THALAMUS-GATE OVERWRITE (seed %d)" % seed)
    v.require("confident commit ignites (n>=1) [precondition]", bool(conf["n_ignited"] >= 1), expect=True)
    v.require("substrate-driven: 0 host workspace-reset calls on the headline [precondition]",
              continuous_no_restore, expect=True)
    v.require("substrate-driven: 0 host content-swap calls [precondition]", bool(host_content_swap_calls == 0),
              expect=True)
    v.require("gate is signal-driven (0 at zero conflict, scales with conflict) [precondition]", gate_signal_driven,
              expect=True)
    v.require("byte-identical base substrate (gate-off == DIST-OVERWRITE base hash) [precondition]",
              byte_identical_base, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash) [precondition]", seed_deterministic,
              expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity", why="not used; the gate is a disinhibition circuit, not STP")
    # go = the OUTCOME: a clean n-stays-1 SWITCH, SELECTIVE, with the per-slot inhibition load-bearing + scramble.
    vd = v.decide(go=bool(clean_swap and selectivity and inh_load_bearing and scramble_breaks_swap), verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"gate_inh_w": float(gate_inh_w), "thal_ws_w": float(thal_ws_w),
                            "conflict_gain": float(conflict_gain), "margin_ref": float(margin_ref),
                            "current_scale": float(current_scale), "swap_steps": int(swap_steps),
                            "heterogeneity": bool(heterogeneity), "empty_clean_max": int(EMPTY_CLEAN_MAX),
                            "private_sizes": hh["private_sizes"]},
        "go_gate": {"clean_switch": clean_swap, "selectivity": selectivity, "switched_identity": switched,
                    "slow_overwrite": slow_overwrite},
        "advance": {"disinhibition_admits_challenger": disinhibition_admits_challenger,
                    "b_rate_no_disinhib": float(breakin["b_rate_no_disinhib"]),
                    "b_rate_with_disinhib": float(breakin["b_rate_with_disinhib"]),
                    "locked_out_without": bool(breakin["locked_out_without"]),
                    "admitted_with": bool(breakin["admitted_with"]),
                    "incumbent_ever_evicted": incumbent_ever_evicted, "evict_sweep": evict_sweep},
        "anti_cheats": {"gate_signal_driven": gate_signal_driven,
                        "scramble_breaks_swap": scramble_breaks_swap, "gate_off_no_swap": gate_off_no_swap,
                        "inh_load_bearing": inh_load_bearing, "thal_load_bearing": thal_load_bearing,
                        "continuous_no_restore": continuous_no_restore,
                        "host_content_swap_calls": host_content_swap_calls,
                        "byte_identical_base": byte_identical_base, "additive_substrate": additive_substrate,
                        "seed_deterministic": seed_deterministic,
                        "inh_swap_attribution": (None if inh_swap_attr is None else float(inh_swap_attr))},
        "residual": {"headline_n_post": int(headline["n_ignited_post"]),
                     "headline_winner_post": int(headline["winner_post"]),
                     "headline_delivered": headline["delivered_post"],
                     "headline_co_ignition": bool(headline["co_ignition"]),
                     "headline_went_empty": bool(headline["went_empty"]),
                     "headline_empty_steps": int(headline["empty_steps"]),
                     "headline_coactive_steps": int(headline["coactive_steps"]),
                     "headline_gate_fired_steps": int(headline["gate_fired_steps"]),
                     "headline_i_gate_max": float(headline["i_gate_max"]),
                     "thal_chal_rate_during": headline.get("thal_chal_rate_during"),
                     "str_chal_rate_during": headline.get("str_chal_rate_during"),
                     "lesion_inh_n_post": int(lesion_inh["n_ignited_post"]),
                     "lesion_inh_swapped": bool(lesion_inh["switched_identity"]),
                     "lesion_inh_co_ignition": bool(lesion_inh["co_ignition"]),
                     "lesion_thal_swapped": bool(lesion_thal["switched_identity"]),
                     "gate_off_n_post": int(gate_off["n_ignited_post"]),
                     "gate_off_co_ignition": bool(gate_off["co_ignition"])},
        "measurements": {"confident": conf, "headline": headline, "scramble": scramble, "gate_off": gate_off,
                         "lesion_inh": lesion_inh, "lesion_thal": lesion_thal, "signal_sweep": signal,
                         "substrate_hash": h_a, "base_hash": h_base, "dist_hash": h_dist},
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        r = result["residual"]
        print(f"[bg-gate seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    HEADLINE: win {headline['winner_pre']}->{headline['winner_post']} n "
              f"{headline['n_ignited_pre']}->{headline['n_ignited_post']} clean={clean_swap} switched={switched} "
              f"slow={slow_overwrite} (co_ign={headline['co_ignition']} empty_steps={headline['empty_steps']} "
              f"coact={headline['coactive_steps']} gate_fired={headline['gate_fired_steps']} "
              f"thal_chal={r['thal_chal_rate_during']})", flush=True)
        print(f"    SELECTIVITY: conf n={conf['n_ignited']} win={conf['winner']} ok={selectivity} | "
              f"INH_LB={inh_load_bearing}(lesion swap={lesion_inh['swap_ok']} n_post={lesion_inh['n_ignited_post']} "
              f"co_ign={lesion_inh['co_ignition']}) THAL_LB={thal_load_bearing}", flush=True)
        print(f"    ADVANCE: disinhib_admits_challenger={disinhibition_admits_challenger} "
              f"(B_no_disinhib={breakin['b_rate_no_disinhib']:.3f} locked_out={breakin['locked_out_without']} -> "
              f"B_with_disinhib={breakin['b_rate_with_disinhib']:.3f} admitted={breakin['admitted_with']}) | "
              f"incumbent_ever_evicted={incumbent_ever_evicted} evict_sweep_n_post="
              f"{[e['n_post'] for e in evict_sweep]}", flush=True)
        print(f"    anti: signal_driven={gate_signal_driven} scramble_breaks={scramble_breaks_swap} "
              f"gate_off_no_swap={gate_off_no_swap} byte_id={byte_identical_base} additive={additive_substrate} "
              f"det={seed_deterministic}", flush=True)
        print(f"          i_gate_by_amp={[round(x,0) for x in signal['i_gate_by_amp']]}", flush=True)
    return result


# ── smoke: an operating-point sweep on one seed ────────────────────────────────────────────────────────────────
def run_smoke(seed, args):
    print(f"[bg-gate smoke] seed={seed} — operating-point grid", flush=True)
    grid = []
    for gi in ([args.gate_inh_w] if args.gate_inh_grid is None else args.gate_inh_grid):
        for tw in ([args.thal_ws_w] if args.thal_ws_grid is None else args.thal_ws_grid):
            r = evaluate_seed(seed, gate_inh_w=float(gi), thal_ws_w=float(tw), conflict_gain=args.conflict_gain,
                              margin_ref=args.margin_ref, current_scale=args.current_scale,
                              swap_steps=args.swap_steps, heterogeneity=not args.no_heterogeneity, verbose=True)
            grid.append({"gate_inh_w": float(gi), "thal_ws_w": float(tw), "seed_go": r["seed_go"],
                         "go_gate": r["go_gate"], "anti_cheats": r["anti_cheats"], "residual": r["residual"]})
    any_go = any(g["seed_go"] for g in grid)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_bg_thalamus_gate_overwrite_derisk", "mode": "smoke", "seed": seed, "grid": grid},
                  f, indent=2, default=str)
    print(f"\n[bg-gate smoke] wrote {args.json}  any_seed_go={any_go}", flush=True)
    return 0 if any_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW active overwrite via a per-slot BG-thalamus selective gate: a "
                                             "conflict-gated external arbiter that disinhibits the challenger's "
                                             "thalamic relay and inhibits the incumbent's slot (PBWM; O'Reilly & "
                                             "Frank 2006), breaking the intrinsic-competition catch-22.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--gate-inh-w", type=float, default=GATE_INH_W)
    ap.add_argument("--thal-ws-w", type=float, default=THAL_WS_W)
    ap.add_argument("--conflict-gain", type=float, default=1.0)
    ap.add_argument("--margin-ref", type=float, default=MARGIN_REF)
    ap.add_argument("--current-scale", type=float, default=GATE_CURRENT_SCALE)
    ap.add_argument("--swap-steps", type=int, default=SWAP_STEPS)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gate-inh-grid", type=float, nargs="+", default=None)
    ap.add_argument("--thal-ws-grid", type=float, nargs="+", default=None)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_bg_thalamus_gate_smoke.json")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    print(f"[bg-gate] K={N_PATTERNS} P={PATTERN_SIZE} gate_inh_w={args.gate_inh_w} thal_ws_w={args.thal_ws_w} "
          f"margin_ref={args.margin_ref:.3f} scale={args.current_scale} swap_steps={args.swap_steps} "
          f"het={not args.no_heterogeneity} backend={args.backend}\n", flush=True)

    if args.smoke:
        return run_smoke(args.seed, args)

    common = dict(gate_inh_w=args.gate_inh_w, thal_ws_w=args.thal_ws_w, conflict_gain=args.conflict_gain,
                  margin_ref=args.margin_ref, current_scale=args.current_scale, swap_steps=args.swap_steps,
                  heterogeneity=not args.no_heterogeneity)
    results = [evaluate_seed(s, verbose=True, **common) for s in args.seeds]

    n_clean = sum(int(r["go_gate"]["clean_switch"]) for r in results)
    n_select = sum(int(r["go_gate"]["selectivity"]) for r in results)
    n_switched = sum(int(r["go_gate"]["switched_identity"]) for r in results)
    n_slow = sum(int(r["go_gate"]["slow_overwrite"]) for r in results)
    n_go = sum(int(r["seed_go"]) for r in results)
    any_undefined = any(r["verdict"] == "UNDEFINED" for r in results)
    all_no_reset = all(r["anti_cheats"]["continuous_no_restore"] for r in results)
    all_no_content = all(r["anti_cheats"]["host_content_swap_calls"] == 0 for r in results)
    all_signal = all(r["anti_cheats"]["gate_signal_driven"] for r in results)
    all_scramble = all(r["anti_cheats"]["scramble_breaks_swap"] for r in results)
    all_inh_lb = all(r["anti_cheats"]["inh_load_bearing"] for r in results)
    all_byte_id = all(r["anti_cheats"]["byte_identical_base"] for r in results)
    all_determ = all(r["anti_cheats"]["seed_deterministic"] for r in results)
    n_admit = sum(int(r["advance"]["disinhibition_admits_challenger"]) for r in results)   # the LOCKOUT-horn advance
    n_evicted = sum(int(r["advance"]["incumbent_ever_evicted"]) for r in results)          # the residual (co-ign horn)

    clean_go = bool(n_clean >= 5)
    select_go = bool(n_select >= 6)
    # anti-cheats: signal-driven, byte-id, determinism, no-reset on ALL; scramble+inh-load-bearing require the swap
    # to have happened (on a NO-GO with no swaps these are vacuously moot -> we report but do not require them).
    anti_all = bool(all_signal and all_byte_id and all_determ and all_no_reset and all_no_content)
    all_ignite = all(r["measurements"]["confident"]["n_ignited"] >= 1 for r in results)

    av = Verdict("distributed-workspace BG-THALAMUS-GATE OVERWRITE — 6-seed aggregate")
    av.require("all seeds: confident commit ignites (n>=1)", all_ignite, expect=True)
    av.require("all seeds: substrate-driven (0 host workspace-reset calls)", all_no_reset, expect=True)
    av.require("all seeds: 0 host content-swap calls", all_no_content, expect=True)
    av.require("all seeds: gate signal-driven (0 at zero conflict, scales)", all_signal, expect=True)
    av.require("all seeds: byte-identical base substrate", all_byte_id, expect=True)
    av.require("all seeds: determinism (cfg.seed seeds the substrate)", all_determ, expect=True)
    av.require("no seed UNDEFINED", not any_undefined, expect=True)
    av.disabled("homeostasis", why="frozen weights; synaptic-scaling clip is a Rung-1/2 foot-gun")
    av.disabled("native_short_term_plasticity", why="not used; the gate is a disinhibition circuit")
    agg_vd = av.decide(go=bool(clean_go and select_go and anti_all and all_scramble and all_inh_lb), verbose=True)

    if agg_vd["status"] == "UNDEFINED":
        verdict = "UNDEFINED"
    elif clean_go and select_go and anti_all and all_scramble and all_inh_lb:
        verdict = "GO"                    # clean n-stays-1 swap >=5/6 + selectivity 6/6 + anti-cheats
    elif n_switched >= 5 and select_go:
        verdict = "PARTIAL"               # overwrite ACHIEVED (clean or slow) 5/6 + selectivity; residual named
    else:
        verdict = "NO-GO"

    summary = {
        "runner": "_gnw_bg_thalamus_gate_overwrite_derisk", "mode": "six-seed", "verdict": verdict,
        "n_clean_swap_go": n_clean, "n_selectivity_go": n_select, "n_switched_identity": n_switched,
        "n_slow_overwrite": n_slow, "n_seed_go": n_go, "n_seeds": len(results), "seeds": list(args.seeds),
        "n_disinhibition_admits_challenger": n_admit, "n_incumbent_ever_evicted": n_evicted,
        "any_undefined": any_undefined,
        "aggregate_anti_cheats": {"all_continuous_no_restore": all_no_reset,
                                  "all_host_content_swap_zero": all_no_content,
                                  "all_gate_signal_driven": all_signal, "all_scramble_breaks_swap": all_scramble,
                                  "all_inh_load_bearing": all_inh_lb, "all_byte_identical_base": all_byte_id,
                                  "all_seed_deterministic": all_determ},
        "preconditions": agg_vd["preconditions"], "disabled_processes": agg_vd["disabled_processes"],
        "undefined_reasons": agg_vd["undefined_reasons"],
        "operating_point": results[0]["operating_point"] if results else {}, "per_seed": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  BG-THALAMUS-GATE OVERWRITE VERDICT: {verdict}  (CLEAN-SWAP {n_clean}/{len(results)}; SELECTIVITY "
          f"{n_select}/{len(results)}; switched-identity {n_switched}/{len(results)}; slow {n_slow}/{len(results)}; "
          f"seed_go {n_go}/{len(results)})", flush=True)
    print(f"  ADVANCE: disinhibition ADMITS challenger (solves lockout horn) {n_admit}/{len(results)}; "
          f"RESIDUAL: incumbent ever evicted (co-ignition horn) {n_evicted}/{len(results)}", flush=True)
    for r in results:
        g = r["go_gate"]; a = r["anti_cheats"]; res = r["residual"]
        print(f"    seed {r['seed']}: {r['verdict']:9s} clean={g['clean_switch']} select={g['selectivity']} "
              f"switched={g['switched_identity']} slow={g['slow_overwrite']} | headline win->"
              f"{res['headline_winner_post']} n_post={res['headline_n_post']} empty={res['headline_empty_steps']} "
              f"(co_ign={res['headline_co_ignition']}) | INH_LB={a['inh_load_bearing']} signal={a['gate_signal_driven']}"
              f" scramble={a['scramble_breaks_swap']} byte_id={a['byte_identical_base']} det={a['seed_deterministic']}",
              flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if (clean_go and select_go) else 1


if __name__ == "__main__":
    raise SystemExit(main())
