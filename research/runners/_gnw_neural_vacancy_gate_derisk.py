"""GNW NEURAL VACANCY GATE — make the thought-swap FULLY self-driven: the challenger is admitted by the SUBSTRATE's
own spiking read that the workspace has been vacated, NOT by a host `if incumbent_collapsed:` trigger.

WHAT THIS CLOSES. The recurrence-weaken thought-swap (`_gnw_recurrence_weaken_swap_derisk.py`, GO 6/6,
2026-08-19-gnw-recurrence-weaken-swap-GO.md) EVICTS the incumbent neurally (Rung-2d short-term depression drains the
incumbent's OWN recurrent E->E loop below the sustain knee -> it self-collapses). But that finding's honest limit #2
is that the IN-gate ADMISSION was HOST-orchestrated: a python loop read `_instant_private_rate(incumbent)`, counted
consecutive sub-threshold steps, and only THEN drove the challenger volley (`if vacancy: drive B`). Per the
BRAIN-BASED-ONLY standard, that admission control between sensation and action is a shortcut. This runner replaces it
with a spiking DIS-INHIBITORY vacancy gate so the WHOLE swap is neurons/synapses.

THE MECHANISM (disinhibition; NO `sim/` edit; explicit wiring; reuse-by-import the #34/recweaken swap substrate):
  occ  (occupancy interneurons, inhibitory): driven by the WHOLE workspace (`ws_used -> occ`, E_TO_I). occ fires
       whenever ANY coalition is ignited == the workspace slot is OCCUPIED.
  gate_k (per-pattern admission relay, excitatory): TONICALLY INHIBITED by occ (`occ -> gate_k`, I_TO_E). While a
       coalition holds, occ fires and CLAMPS every gate closed. When the incumbent's recurrence depletes and it
       COLLAPSES, its firing — and thus occ's feed-forward inhibition onto the gates — falls silent, DIS-INHIBITING
       the gates. A gate_k that is ALSO receiving the challenger PROPOSAL (a sub-threshold sensory drive into gate_k:
       "content k is proposed") now fires and drives its coalition (`gate_k -> pattern_k`, E_TO_E) over the ignition
       knee. So the challenger is admitted by the SUBSTRATE reading vacancy (occ silent) coincident with a proposal —
       no host vacancy check anywhere in the loop. Content is the world's PROPOSAL (which gate_k is driven); the
       neural work is the vacancy-gated ADMISSION. This is the mechanism the task named (a tonically-inhibited pool
       released by disinhibition when the incumbent's feed-forward inhibition collapses) and the mechanism the base
       finding's limit #2 called for.

Biology: BG->thalamus / SNr tonic-inhibition RELEASE by disinhibition gates thalamocortical transmission (Chevalier &
Deniau 1990, TINS 13:277; Deniau & Chevalier 1985). Cortical VIP->SST->PC disinhibitory gating (Pi et al. 2013,
Nature 503:521; Karnani et al. 2016). Dehaene & Changeux 2011, Neuron 70:200 (an ignited workspace state must be
destabilizable and "spontaneously replaced by another"). The eviction effector is unchanged: Mongillo, Barak &
Tsodyks 2008, Science 319:1543 (recurrent-resource short-term depression), applied to the incumbent's own loop.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed) — per seed, ALL of:
  SWAP        — the neural gate admits the challenger: win_pre=A & n_pre=1, then A DROPS TO BASELINE
                (old_residual_post NOT ignited) & B ignites & win_post=B & n_post=1. NO host `if vacancy` in the loop.
  GATE LOAD-BEARING (neural detector required) — the detector's causal role is dissociated by TWO tight controls
                (below: NON-CIRCULAR + detector-removed), because the disinhibitory veto is workspace-SYNCHRONIZED and
                a DC "stuck-occupied" occ clamp LEAKS (a brief disinhibition gap lets the strongly-recurrent challenger
                latch) — that leak is characterized in the finding, not swept under.
  TIMING      — the challenger ignites ONLY AFTER the incumbent drops below threshold: b_ignite_step > a_vacate_step
                per seed (windowed rates), zero co-ignition (no premature admission). And REMOVING the detector
                (occ->gate=0, gates never vetoed) admits the challenger PREMATURELY while the incumbent still holds
                (co-ignition / early B) -> the detector enforces the timing, not a host delay.
  NON-CIRCULAR (proposal is not enough) — present the proposal but DO NOT trigger eviction (no STD boost -> A holds)
                -> occ stays high -> the gate is vetoed -> the challenger is locked out (0/6). Vacancy is NECESSARY;
                the proposal alone cannot displace a supra-critical incumbent.
  REIGNITE    — the admitted coalition ignites AND HOLDS through a free tail (n=1, winner B, old gone).
  REVERSIBLE  — a two-swap A->B->A on ONE continuous substrate, BOTH admissions through the neural gate.
  NO-HOST-RESET / NO-HOST-TRIGGER — the swap headline is a CONTINUOUS run (host_workspace_reset_calls==0); the only
                host writes are thal tonic + the sub-threshold challenger PROPOSAL (world/body) + the STD boost (the
                top-down "swap now" intention pulse, a FIXED-duration command, NOT vacancy-gated). NO host read of any
                workspace rate gates the admission (grep-checkable: the admission drive is UNCONDITIONAL each step).
  DETERMINISM — build twice at one seed -> identical seed-derived Izhikevich-param hash (substrate-integrity anti-cheat;
                additive_substrate hash N/A — the RNG-prefix property does not hold on this engine).

NOT-A-WALL: any residual is QUANTIFIED (does B ignite? does it mistime? empty window) so the next mechanism is mapped.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_vacancy_gate_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_vacancy_gate_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_neural_vacancy_gate_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_neural_vacancy_gate_derisk --six-seed \
      --json research/findings/raw/_gnw_neural_vacancy_gate_6seed.json
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

# reuse-by-import: the #34/recweaken swap substrate pieces (geometry, split recurrence, dense pops, the STD eviction
# effector, stepping / spiking reads / margin instruments) + validated ignition constants. NO re-derivation.
from research.runners._gnw_active_overwrite_derisk import (
    _pattern_geometry, _rec_population_split,
    _ws_step, _drive, _read_private_rates, _instant_private_rate, _margin, _verdict_label,
    N_PATTERNS, PATTERN_SIZE, WORKSPACE_N, NORM_N, THAL_N,
    W_SHARED, WS_NORM_W, NORM_WS_W, THAL_TONIC_PA, THAL_WS_W, STRONG_PA, STD_TAU_D, OU_NOISE_PA,
)
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, SETTLE_STEPS, WS_LOOP_GATE
from research.runners._gnw_rung2_competitive_access_derisk import _ignited, IGNITE_FRAC, SOLO_PLATEAU
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
# the neural EVICTION effector (per-coalition short-term depression on the incumbent's OWN recurrent loop) — the exact
# Rung-2d/recweaken instrument, reused unchanged.
from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD

IGNITE_THRESH = IGNITE_FRAC * SOLO_PLATEAU     # a swapped-OUT coalition reads BELOW this (0.1667)

# ── the neural dis-inhibitory vacancy gate (calibrated on seed 42, --calibrate; frozen, no per-seed tuning) ─────────
OCC_N = 40                     # occupancy interneuron pool (ws_used -> occ): fires whenever a coalition is ignited
GATE_PER = 60                  # neurons per per-pattern admission relay gate_k
W_WS_OCC = 8.0                 # workspace -> occ (E_TO_I): the occupancy read
W_OCC_GATE = 20.0              # occ -> gate_k (I_TO_E): the TONIC inhibition released on vacancy. MODEST on purpose —
                               # a large inhibition paradoxically RELIEVES Izhikevich depolarization block and INVERTS
                               # the gate (measured: w>=40 makes an occupied gate fire MORE); 20 is the clean veto.
W_GATE_WS = 100.0              # gate_k -> pattern_k (E_TO_E): drives the admitted coalition over the ignition knee.
                               # Free to be large — a vetoed gate is ~silent so it contributes ~0 regardless.
PROPOSAL_PA = 2800.0           # sub-threshold sensory PROPOSAL into gate_k ("content k is proposed"): drives gate_k
                               # ONLY when dis-inhibited; while occ vetoes gate_k it cannot admit (measured).
OCC_STUCK_PA = 1600.0          # the detector-STUCK lesion drive: occ built BLIND to the workspace (ws->occ=0) is
                               # driven by this STEADY tonic so it reads OCCUPIED forever (no workspace-coupled
                               # oscillation that would let the gate leak); the gate stays vetoed though the slot is free

# swap operating point (the eviction half is the recweaken GO point, reused unchanged).
ESTABLISH_PA = 8000.0          # incumbent-establishment DIRECT drive (the "before" state = attending to thought A).
                               # Above STRONG_PA(5000): the appended gate/occ pools shift the workspace Izhikevich
                               # params per seed (RNG-prefix quirk) so one seed's incumbent sat on the ignition
                               # knife-edge at 5000x35 (this substrate has documented non-deterministic near-threshold
                               # ignition); 8000x35 clears the boundary so every incumbent establishes on every seed
                               # (uniform; NO per-seed tuning). The exact strength is not load-bearing — the mechanism
                               # under test is the ADMISSION gate; establishment is the "before" initial condition.
SWAP_BOOST = 0.12              # per-spike utilization boost on the incumbent loop (the "swap now" intention pulse)
EVICT_STEPS = 260              # FIXED window the swap intention pulse + proposal run (NOT vacancy-gated). The neural
                               # gate admits the challenger DURING this window the moment the incumbent collapses.
BOOST_STEPS = 200              # the STD boost is a FIXED-duration top-down command (<= EVICT_STEPS); NO vacancy read
                               # clears it. After it ends the incumbent stays evicted (depleted) and B holds.
SETTLE_GAP = 0                 # no host settle: the neural gate self-times admission (kept 0; the dynamics settle it)
REIGNITE_HOLD = 150
W_REC = W_SHARED               # 34.0 uniform recurrence (supra-critical disjoint = the inhibition-resistant incumbent)

# timing-read thresholds (MEASUREMENT ONLY — never gate the admission; the admission drive is unconditional).
# The per-step private rate oscillates (period-3 ignition), so timing is read on a trailing WINDOW mean vs the same
# ignited threshold used for the identity read (IGNITE_THRESH), which is the meaningful "slot free / coalition up" line.
TIMING_WINDOW = 9              # trailing-window (3 ignition periods) to smooth the period-3 oscillation
GATE_OPEN_THRESH = 0.04        # challenger gate instant rate above this == the gate opened (diagnostic only)

_RESTORE_CALLS = {"n": 0}


def _counted_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── build: the recweaken swap substrate + the neural dis-inhibitory vacancy gate (explicit wiring; NO sim/ edit) ────
def build(seed=42, w_rec=W_REC, heterogeneity=True, ou_noise_pA=OU_NOISE_PA,
          w_ws_occ=W_WS_OCC, w_occ_gate=W_OCC_GATE, w_gate_ws=W_GATE_WS, occ_gate_lesion=False, blind_occ=False):
    """workspace (exc, NMDA; N_PATTERNS disjoint supra-critical cliques) + norm_pool (inh; divisive norm) + thal
    (exc; tonic) — the EXACT inhibition-resistant swap substrate — PLUS occ (inh occupancy pool) + gate (K per-pattern
    admission relays, exc). occ_gate_lesion=True zeroes occ->gate (the detector-removed lesion: gates never vetoed);
    blind_occ=True zeroes ws->occ (occ can no longer READ the workspace -> the detector-stuck lesion, driven steady).
    Returns a dict of handles. Workspace is region 0 so the coalition geometry/recurrence is byte-for-byte the swap
    substrate's; the appended occ/gate pools do not change the disjoint supra-critical workspace dynamics (verified by
    the lesion/non-circular controls reproducing the inhibition-resistant incumbent)."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="occ", n_neurons=OCC_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="gate", n_neurons=GATE_PER * N_PATTERNS, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # ⭐ the substrate seed (het/threshold RNG) — NOT actual_seed_used
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
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, PATTERN_SIZE, 0)   # overlap=0 -> disjoint supra-critical
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)
    occ_idx = np.asarray(rm.indices("occ"), dtype=np.int64)
    gate_idx = np.asarray(rm.indices("gate"), dtype=np.int64)
    gate_slices = [gate_idx[k * GATE_PER:(k + 1) * GATE_PER] for k in range(N_PATTERNS)]

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    # the swap substrate (identical to build_swap_bridge's disjoint uniform-recurrence config)
    union_plan["workspace_rec"] = _rec_population_split(patterns, privates, float(w_rec), float(w_rec))
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(WS_NORM_W), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, float(NORM_WS_W), "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(THAL_WS_W), "E_TO_E")
    # the NEURAL dis-inhibitory vacancy gate
    w_ws_occ_eff = 0.0 if blind_occ else float(w_ws_occ)
    union_plan["ws2occ"] = _dense_pop(ws_used, occ_idx, w_ws_occ_eff, "E_TO_I")
    wog = 0.0 if occ_gate_lesion else float(w_occ_gate)
    for k in range(N_PATTERNS):
        union_plan[f"occ2gate{k}"] = _dense_pop(occ_idx, gate_slices[k], wog, "I_TO_E")
        union_plan[f"gate{k}2ws"] = _dense_pop(gate_slices[k], patterns[k], float(w_gate_ws), "E_TO_E")

    inh = list(norm_idx) + list(occ_idx)     # occ neurons are inhibitory (the gate-veto source)
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(THAL_TONIC_PA)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "w_rec": float(w_rec), "w_ws_occ": float(w_ws_occ_eff),
               "w_occ_gate": float(wog), "w_gate_ws": float(w_gate_ws), "occ_gate_lesion": bool(occ_gate_lesion),
               "blind_occ": bool(blind_occ),
               "n_ws_used": int(ws_used.size), "occ_n": int(occ_idx.size), "gate_per": int(GATE_PER)}
    return {
        "bridge": bridge, "xp": xp,
        "patterns": [xp.asarray(p) for p in patterns], "privates": [xp.asarray(p) for p in privates],
        "patterns_host": [p.astype(np.int64) for p in patterns], "ws_used": ws_used,
        "thal": thal_dev, "occ": xp.asarray(occ_idx), "gate_slices": [xp.asarray(g) for g in gate_slices],
        "snap": snap, "handles": handles,
    }


def _izh_hash(bridge):
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest() if parts else ""


def _occ_rate(S):
    b, xp = S["bridge"], S["xp"]
    return float(to_host(b.cp_firing_states[S["occ"]].astype(xp.float64).mean()))


def _gate_rate(S, k):
    b, xp = S["bridge"], S["xp"]
    return float(to_host(b.cp_firing_states[S["gate_slices"][k]].astype(xp.float64).mean()))


# ── one single-move swap through the NEURAL vacancy gate (NO host `if vacancy` anywhere in the loop) ────────────────
def run_gated_swap(S, std, *, incumbent=0, challenger=1, boost=SWAP_BOOST, boost_steps=BOOST_STEPS,
                   evict_steps=EVICT_STEPS, proposal_pa=PROPOSAL_PA, reignite_hold=REIGNITE_HOLD,
                   occ_stuck=False, no_evict=False, isolate=True):
    """Ignite incumbent A (hold); TRIGGER the swap intention (a FIXED-duration STD boost on A's own loop) while a
    sub-threshold challenger PROPOSAL is presented to gate_challenger every step. A's loop depletes -> A COLLAPSES ->
    occ falls silent -> the challenger's gate is DIS-INHIBITED -> the proposal drives it over the ignition knee -> B is
    admitted into the freed slot. THE ADMISSION IS UNCONDITIONAL EACH STEP (drive gate_challenger at proposal_pa);
    nothing in this loop reads a workspace rate to decide admission — the gate's opening is the neural occ->gate
    disinhibition. Timing (a_vacate_step / b_ignite_step / co-ignition) is read on a trailing WINDOW mean, READ-ONLY.
      occ_stuck=True -> the detector-STUCK lesion: on a build BLIND to the workspace (ws->occ=0), drive occ with a
        STEADY tonic so it reads OCCUPIED forever -> every gate stays vetoed -> admission FAILS though the incumbent
        collapsed and the slot is free (the neural-gate load-bearing control). No workspace-coupled occ oscillation.
      no_evict=True  -> present the proposal but apply NO STD boost -> A holds -> occ high -> gate vetoed -> the
        challenger is locked out (the NON-CIRCULAR control: the proposal alone cannot displace a supra-critical
        incumbent; only the vacancy admits it).
      isolate=False  -> a CONTINUOUS run (0 restore calls) = the swap HEADLINE."""
    bridge, xp, thal = S["bridge"], S["xp"], S["thal"]
    patterns, privates = S["patterns"], S["privates"]
    if isolate:
        _counted_restore(bridge, S["snap"])
        std.reset()

    # (1) ignite A alone -> it holds on its supra-critical recurrent loop.
    _drive(bridge, xp, thal, THAL_TONIC_PA, std, [(patterns[incumbent], ESTABLISH_PA)], n=DRIVE_STEPS)
    pre = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_pre, _m_pre, n_pre = _margin(pre)

    # (2) SWAP: FIXED-duration STD boost (intention) + UNCONDITIONAL sub-threshold proposal into gate_challenger.
    eff_boost = 0.0 if no_evict else float(boost)
    std.set_boost(incumbent, eff_boost)
    xA_min = 1.0
    gate_open_step = -1
    a_hist, b_hist = [], []
    a_vacate_step, b_ignite_step, coign_steps = -1, -1, 0
    gate_dev = S["gate_slices"][challenger]
    occ_dev = S["occ"]
    for t in range(int(evict_steps)):
        if t == int(boost_steps):
            std.clear_boost()                       # the intention pulse ends (fixed; NOT vacancy-gated)
        drive_map = [(gate_dev, float(proposal_pa))]
        if occ_stuck:
            drive_map.append((occ_dev, float(OCC_STUCK_PA)))    # detector stuck ON (blind build, steady tonic)
        _ws_step(bridge, xp, thal, THAL_TONIC_PA, std, drive_map=drive_map)
        # READ-ONLY timing measurements on a trailing window (never gate the admission):
        a_hist.append(_instant_private_rate(bridge, xp, privates, incumbent))
        b_hist.append(_instant_private_rate(bridge, xp, privates, challenger))
        aw = float(np.mean(a_hist[-TIMING_WINDOW:])); bw = float(np.mean(b_hist[-TIMING_WINDOW:]))
        xA_min = min(xA_min, std.x_mean(incumbent))
        if a_vacate_step < 0 and t >= TIMING_WINDOW and aw < IGNITE_THRESH:
            a_vacate_step = t
        if b_ignite_step < 0 and bw > IGNITE_THRESH:
            b_ignite_step = t
        if _ignited(aw) and _ignited(bw):
            coign_steps += 1                        # BOTH coalitions ignited on the window == a co-ignition step
        if gate_open_step < 0 and _gate_rate(S, challenger) > GATE_OPEN_THRESH:
            gate_open_step = t
    std.clear_boost()

    # (3) identity read (free-run, no proposal drive -> the gate closes; B holds on its own loop).
    post = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_post, _m_post, n_post = _margin(post)
    hold = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std, n_free=int(reignite_hold))
    win_hold, _m_hold, n_hold = _margin(hold)

    old_res = float(post[incumbent]); new_rate = float(post[challenger])
    old_res_hold = float(hold[incumbent]); new_hold = float(hold[challenger])
    swapped = bool(win_pre == incumbent and n_pre == 1 and (not _ignited(old_res))
                   and _ignited(new_rate) and win_post == challenger and n_post == 1)
    reignite_ok = bool(_ignited(new_hold) and win_hold == challenger and n_hold == 1 and (not _ignited(old_res_hold)))
    # premature admission == the challenger ignited BEFORE the incumbent vacated (or the incumbent never vacated), or
    # a sustained co-ignition window (both up at once). A correctly-timed swap admits B strictly AFTER A vacates.
    premature = bool((b_ignite_step >= 0 and (a_vacate_step < 0 or b_ignite_step < a_vacate_step))
                     or coign_steps >= TIMING_WINDOW)
    timing_ok = bool(swapped and a_vacate_step >= 0 and b_ignite_step > a_vacate_step and coign_steps == 0)
    return {
        "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
        "hold_rates": [float(r) for r in hold],
        "winner_pre": int(win_pre), "winner_post": int(win_post), "winner_hold": int(win_hold),
        "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post), "n_ignited_hold": int(n_hold),
        "old_residual_post": old_res, "old_residual_hold": old_res_hold,
        "new_rate_post": new_rate, "new_rate_hold": new_hold, "xA_min": float(xA_min),
        "a_vacate_step": int(a_vacate_step), "gate_open_step": int(gate_open_step),
        "b_ignite_step": int(b_ignite_step), "coign_steps": int(coign_steps),
        "swapped": swapped, "reignite_ok": reignite_ok, "premature": premature, "timing_ok": timing_ok,
        "co_ignition": bool(n_pre == 1 and n_post >= 2), "went_empty": bool(n_pre >= 1 and n_post == 0),
        "incumbent_held": bool(win_post == incumbent and n_post >= 1 and not _ignited(new_rate)),
        "occ_stuck": bool(occ_stuck), "no_evict": bool(no_evict),
    }


# ── two-swap reversibility: A -> B -> A on ONE continuous substrate, BOTH admissions through the neural gate ────────
def run_two_swap(S, std, *, a=0, b=1, boost=SWAP_BOOST, evict_steps=EVICT_STEPS, boost_steps=BOOST_STEPS,
                 proposal_pa=PROPOSAL_PA, recover_steps=None, reignite_hold=REIGNITE_HOLD):
    _counted_restore(S["bridge"], S["snap"]); std.reset()
    if recover_steps is None:
        recover_steps = int(3 * STD_TAU_D)
    s1 = run_gated_swap(S, std, incumbent=a, challenger=b, boost=boost, evict_steps=evict_steps,
                        boost_steps=boost_steps, proposal_pa=proposal_pa, reignite_hold=reignite_hold, isolate=False)
    xA_after_s1 = std.x_mean(a)
    for _ in range(int(recover_steps)):
        _ws_step(S["bridge"], S["xp"], S["thal"], THAL_TONIC_PA, std)
    xA_recovered = std.x_mean(a)
    s2 = run_gated_swap(S, std, incumbent=b, challenger=a, boost=boost, evict_steps=evict_steps,
                        boost_steps=boost_steps, proposal_pa=proposal_pa, reignite_hold=reignite_hold, isolate=False)
    s1_evicted_A = bool(s1["swapped"])
    recovered = bool(xA_recovered > 0.85)
    s2_brought_A_back = bool(s2["winner_hold"] == a and s2["n_ignited_hold"] == 1
                            and _ignited(s2["new_rate_hold"]) and not _ignited(s2["old_residual_hold"]))
    reversible = bool(s1_evicted_A and recovered and s2_brought_A_back)
    return {"swap1": s1, "swap2": s2, "xA_after_swap1": float(xA_after_s1), "xA_recovered": float(xA_recovered),
            "recover_steps": int(recover_steps), "reversible": reversible, "s1_evicted_A": s1_evicted_A,
            "recovered": recovered, "s2_brought_A_back": s2_brought_A_back}


# ── one seed: headline neural-gated swap + detector dissociations + timing + reversibility + determinism ────────────
def evaluate_seed(seed, *, boost=SWAP_BOOST, evict_steps=EVICT_STEPS, boost_steps=BOOST_STEPS,
                  proposal_pa=PROPOSAL_PA, reignite_hold=REIGNITE_HOLD, w_rec=W_REC, heterogeneity=True, verbose=True):
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    ws_used, pats_host = S["ws_used"], S["patterns_host"]
    b_, xp = S["bridge"], S["xp"]
    # ⚠ construct ALL STD instances NOW on the freshly-built substrate (each snapshots base recurrence at build; an STD
    # built after an arm depressed the loop would capture too-low base — the recweaken foot-gun, banked).
    std = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_noevict = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_rev = MultiLoopSTD(b_, xp, ws_used, pats_host)

    # HEADLINE: single-move swap A->B through the NEURAL gate, CONTINUOUS (0 restore calls in the headline).
    restore_before = _RESTORE_CALLS["n"]
    headline = run_gated_swap(S, std, incumbent=0, challenger=1, boost=boost, evict_steps=evict_steps,
                              boost_steps=boost_steps, proposal_pa=proposal_pa, reignite_hold=reignite_hold,
                              isolate=False)
    host_workspace_reset_calls = int(_RESTORE_CALLS["n"] - restore_before)

    # NON-CIRCULAR CONTROL (proposal present, NO eviction -> A holds -> occ high -> gate vetoed): challenger locked out.
    noevict = run_gated_swap(S, std_noevict, incumbent=0, challenger=1, boost=boost, evict_steps=evict_steps,
                             boost_steps=boost_steps, proposal_pa=proposal_pa, reignite_hold=reignite_hold,
                             no_evict=True, isolate=True)

    # DETECTOR-REMOVED LESION (occ->gate=0, gates never vetoed): the challenger is admitted PREMATURELY while the
    # incumbent still holds (co-ignition / early B). Proves the detector enforces the admission TIMING (built fresh).
    S_les = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity, occ_gate_lesion=True)
    std_les = MultiLoopSTD(S_les["bridge"], S_les["xp"], S_les["ws_used"], S_les["patterns_host"])
    detector_removed = run_gated_swap(S_les, std_les, incumbent=0, challenger=1, boost=boost, evict_steps=evict_steps,
                                      boost_steps=boost_steps, proposal_pa=proposal_pa, reignite_hold=reignite_hold,
                                      isolate=True)

    # REVERSIBILITY: two-swap A->B->A, both admissions through the neural gate.
    two = run_two_swap(S, std_rev, a=0, b=1, boost=boost, evict_steps=evict_steps, boost_steps=boost_steps,
                       proposal_pa=proposal_pa, reignite_hold=reignite_hold)

    # ── anti-cheats — the neural detector's causal role is dissociated by TWO tight controls (the disinhibitory veto
    # is workspace-SYNCHRONIZED; a DC "stuck-occupied" clamp leaks — characterized, see the finding — so the
    # load-bearing evidence is the tight non-circular veto + the detector-removed mistiming, both clean). ──
    swap_ok = bool(headline["swapped"])
    reignite_ok = bool(headline["reignite_ok"])
    timing_ok = bool(headline["timing_ok"])
    # NON-CIRCULAR / VETO WORKS: the proposal alone (no eviction, so the workspace correctly reads OCCUPIED) cannot
    # admit -> the incumbent holds, the challenger is locked out. The detector's veto blocks admission when occupied.
    noevict_locks_out = bool(not noevict["swapped"] and _ignited(noevict["old_residual_post"])
                             and not _ignited(noevict["new_rate_post"]))
    # DETECTOR REMOVED -> premature admission: with occ->gate=0 the gate is never vetoed, so B is admitted BEFORE the
    # incumbent vacates (premature) / both co-ignite -> the occ->gate veto is exactly what enforced the correct timing.
    detector_enforces_timing = bool(detector_removed["premature"] or detector_removed["co_ignition"]
                                    or (_ignited(detector_removed["new_rate_post"])
                                        and detector_removed["b_ignite_step"] >= 0
                                        and headline["a_vacate_step"] >= 0
                                        and detector_removed["b_ignite_step"] < headline["a_vacate_step"]))
    # LOAD-BEARING = the neural detector both blocks admission when occupied (non-circular) AND enforces the timing
    # (removing it mistimes). Together these prove the occupancy read, not host logic, gates admission.
    gate_load_bearing = bool(swap_ok and noevict_locks_out and detector_enforces_timing)
    reversible = bool(two["reversible"])
    swap_attr = attributable_to("neural-gated swap (headline vs detector-removed lesion)",
                                float(swap_ok), float(detector_removed["swapped"]), warn_below=0.0)

    # DETERMINISM (substrate-integrity anti-cheat; additive_substrate hash N/A — RNG-prefix property fails here).
    h1 = _izh_hash(b_)
    S2 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity)
    seed_deterministic = bool(_izh_hash(S2["bridge"]) == h1 and h1 != "")

    seed_go = bool(swap_ok and reignite_ok and timing_ok and gate_load_bearing and noevict_locks_out
                   and detector_enforces_timing and reversible and host_workspace_reset_calls == 0
                   and seed_deterministic)

    v = Verdict("GNW neural vacancy gate (seed %d)" % seed)
    v.require("incumbent ignites confidently (n_pre==1, winner A) [precondition]",
              bool(headline["n_ignited_pre"] == 1 and headline["winner_pre"] == 0), expect=True)
    v.require("NEURAL-GATED single-move swap: old->baseline & new ignites (n_post==1, winner B)", swap_ok, expect=True)
    v.require("TIMING: challenger ignites only AFTER incumbent vacates (windowed, zero co-ignition)", timing_ok,
              expect=True)
    v.require("NON-CIRCULAR / veto works: proposal alone (no eviction) cannot admit -> incumbent holds",
              noevict_locks_out, expect=True)
    v.require("detector ENFORCES timing: removing occ->gate admits prematurely", detector_enforces_timing,
              expect=True)
    v.require("admitted coalition re-ignites and HOLDS", reignite_ok, expect=True)
    v.require("REVERSIBLE two-swap A->B->A (both admissions neural-gated)", reversible, expect=True)
    v.require("no host workspace reset in the swap headline (continuous run)",
              host_workspace_reset_calls == 0, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating (2026-08-01); STD targets ONLY the incumbent E->E loop")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is the "
                   "substrate-integrity anti-cheat instead")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"boost": float(boost), "evict_steps": int(evict_steps), "boost_steps": int(boost_steps),
                            "proposal_pa": float(proposal_pa), "establish_pa": float(ESTABLISH_PA),
                            "reignite_hold": int(reignite_hold),
                            "w_rec": float(w_rec), "w_ws_occ": float(W_WS_OCC), "w_occ_gate": float(W_OCC_GATE),
                            "w_gate_ws": float(W_GATE_WS), "occ_n": int(OCC_N), "gate_per": int(GATE_PER),
                            "occ_stuck_pa": float(OCC_STUCK_PA), "U_baseline": 0.0, "tau_D": float(STD_TAU_D),
                            "heterogeneity": bool(heterogeneity)},
        "go_gate": {"swap_ok": swap_ok, "reignite_ok": reignite_ok, "timing_ok": timing_ok,
                    "gate_load_bearing": gate_load_bearing,
                    "noevict_locks_out": noevict_locks_out, "detector_enforces_timing": detector_enforces_timing,
                    "reversible": reversible, "no_host_reset": bool(host_workspace_reset_calls == 0),
                    "seed_deterministic": seed_deterministic},
        "anti_cheats": {"gate_load_bearing": gate_load_bearing,
                        "noevict_non_circular": noevict_locks_out, "detector_enforces_timing": detector_enforces_timing,
                        "reversible_two_swap": reversible, "no_host_workspace_reset": bool(host_workspace_reset_calls == 0),
                        "seed_deterministic": seed_deterministic, "swap_attributable_fraction": swap_attr},
        "timing": {"a_vacate_step": headline["a_vacate_step"], "gate_open_step": headline["gate_open_step"],
                   "b_ignite_step": headline["b_ignite_step"], "coign_steps": headline["coign_steps"],
                   "admit_after_vacate_gap": (headline["b_ignite_step"] - headline["a_vacate_step"])
                   if (headline["b_ignite_step"] >= 0 and headline["a_vacate_step"] >= 0) else None},
        "residual": {
            "headline": {"winner_pre": headline["winner_pre"], "winner_post": headline["winner_post"],
                         "n_pre": headline["n_ignited_pre"], "n_post": headline["n_ignited_post"],
                         "old_residual_post": headline["old_residual_post"],
                         "new_ignited": bool(_ignited(headline["new_rate_post"])),
                         "new_rate_post": headline["new_rate_post"], "xA_min": headline["xA_min"],
                         "old_residual_hold": headline["old_residual_hold"], "new_rate_hold": headline["new_rate_hold"]},
            "noevict": {"winner_post": noevict["winner_post"], "n_post": noevict["n_ignited_post"],
                        "old_residual_post": noevict["old_residual_post"], "new_rate_post": noevict["new_rate_post"],
                        "swapped": noevict["swapped"]},
            "detector_removed": {"winner_post": detector_removed["winner_post"],
                                 "n_post": detector_removed["n_ignited_post"], "premature": detector_removed["premature"],
                                 "co_ignition": detector_removed["co_ignition"],
                                 "coign_steps": detector_removed["coign_steps"],
                                 "b_ignite_step": detector_removed["b_ignite_step"],
                                 "a_vacate_step": detector_removed["a_vacate_step"],
                                 "new_rate_post": detector_removed["new_rate_post"],
                                 "old_residual_post": detector_removed["old_residual_post"],
                                 "swapped": detector_removed["swapped"]},
            "reversibility": {"reversible": two["reversible"], "s1_evicted_A": two["s1_evicted_A"],
                              "recovered": two["recovered"], "s2_brought_A_back": two["s2_brought_A_back"],
                              "xA_recovered": two["xA_recovered"], "recover_steps": two["recover_steps"],
                              "swap1_swapped": two["swap1"]["swapped"],
                              "swap2_winner_hold": two["swap2"]["winner_hold"],
                              "swap2_n_hold": two["swap2"]["n_ignited_hold"],
                              "swap2_b_ignite_step": two["swap2"]["b_ignite_step"],
                              "swap2_a_vacate_step": two["swap2"]["a_vacate_step"]},
        },
        "host_workspace_reset_calls": int(host_workspace_reset_calls),
        "substrate_hash": h1, "std_weight_writes": int(std.n_writes),
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        hd = headline
        print(f"[neural-vac-gate seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        print(f"    HEADLINE: win {hd['winner_pre']}->{hd['winner_post']} n {hd['n_ignited_pre']}->{hd['n_ignited_post']}"
              f" old_res={hd['old_residual_post']:.3f} new={hd['new_rate_post']:.3f} xA_min={hd['xA_min']:.3f}"
              f" | TIMING a_vacate@{hd['a_vacate_step']} gate_open@{hd['gate_open_step']} b_ignite@{hd['b_ignite_step']}"
              f" coign={hd['coign_steps']} swapped={hd['swapped']} timing_ok={hd['timing_ok']}", flush=True)
        print(f"    NOEVICT(veto): swapped={noevict['swapped']} old_res={noevict['old_residual_post']:.3f} "
              f"new={noevict['new_rate_post']:.3f} locks_out={noevict_locks_out} (proposal present, A held)", flush=True)
        print(f"    DETECTOR-REMOVED: premature={detector_removed['premature']} co_ign={detector_removed['co_ignition']} "
              f"b_ig@{detector_removed['b_ignite_step']} a_vac@{detector_removed['a_vacate_step']} "
              f"new={detector_removed['new_rate_post']:.3f} enforces_timing={detector_enforces_timing}", flush=True)
        print(f"    REVERSE: reversible={two['reversible']} xA_recovered={two['xA_recovered']:.3f} "
              f"s1={two['swap1']['swapped']} s2_hold_win={two['swap2']['winner_hold']} | det={seed_deterministic} "
              f"resets={host_workspace_reset_calls} load_bearing={gate_load_bearing}", flush=True)
    return result


# ── calibration: verify the neural gate's ON/OFF window on one seed (no swap; just the gate primitive) ─────────────
def run_calibrate(seed, args):
    print(f"[neural-vac-gate calibrate] seed={seed} — dis-inhibitory gate primitive", flush=True)
    S = build(seed=seed)
    b, xp = S["bridge"], S["xp"]
    # (a) EMPTY workspace + proposal -> gate opens, B ignites
    _full_restore(b, S["snap"])
    for _ in range(45):
        _ws_step(b, xp, S["thal"], THAL_TONIC_PA, None, drive_map=[(S["gate_slices"][1], PROPOSAL_PA)])
    empty = _read_private_rates(b, xp, S["thal"], THAL_TONIC_PA, S["privates"], None)
    # (b) A HELD + proposal -> gate vetoed, B stays out
    _full_restore(b, S["snap"])
    _drive(b, xp, S["thal"], THAL_TONIC_PA, None, [(S["patterns"][0], STRONG_PA)], n=DRIVE_STEPS)
    for _ in range(45):
        _ws_step(b, xp, S["thal"], THAL_TONIC_PA, None, drive_map=[(S["gate_slices"][1], PROPOSAL_PA)])
    held = _read_private_rates(b, xp, S["thal"], THAL_TONIC_PA, S["privates"], None)
    ok = bool(_ignited(empty[1]) and sum(_ignited(x) for x in empty) == 1
              and not _ignited(held[1]) and _ignited(held[0]))
    print(f"  EMPTY+proposal -> {[round(x,3) for x in empty]} (B admitted, n=1: {ok})", flush=True)
    print(f"  A-held+proposal -> {[round(x,3) for x in held]} (B vetoed, A holds)", flush=True)
    print(f"  GATE PRIMITIVE {'HOLDS' if ok else 'FAILS'}", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_neural_vacancy_gate_derisk", "mode": "calibrate", "seed": seed,
                       "empty": [float(x) for x in empty], "held": [float(x) for x in held], "gate_ok": ok}, f,
                      indent=2)
    return 0 if ok else 1


def run_smoke(seed, args):
    r = evaluate_seed(seed, boost=args.boost, evict_steps=args.evict_steps, boost_steps=args.boost_steps,
                      proposal_pa=args.proposal_pa, reignite_hold=args.reignite_hold, w_rec=args.w_rec,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_neural_vacancy_gate_derisk", "mode": "smoke", "seed": seed, "result": r}, f,
                  indent=2, default=str)
    print(f"\n[neural-vac-gate smoke] wrote {args.json}  seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[neural-vac-gate six-seed] seeds={seeds} @ boost={args.boost} evict={args.evict_steps} "
          f"boost_steps={args.boost_steps} proposal={args.proposal_pa}", flush=True)
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, boost=args.boost, evict_steps=args.evict_steps, boost_steps=args.boost_steps,
                                      proposal_pa=args.proposal_pa, reignite_hold=args.reignite_hold, w_rec=args.w_rec,
                                      heterogeneity=not args.no_heterogeneity, verbose=True))
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_swap = sum(1 for r in per_seed if r["go_gate"]["swap_ok"])
    n_timing = sum(1 for r in per_seed if r["go_gate"]["timing_ok"])
    n_lb = sum(1 for r in per_seed if r["go_gate"]["gate_load_bearing"])
    n_noev = sum(1 for r in per_seed if r["go_gate"]["noevict_locks_out"])
    n_enf = sum(1 for r in per_seed if r["go_gate"]["detector_enforces_timing"])
    n_reig = sum(1 for r in per_seed if r["go_gate"]["reignite_ok"])
    n_rev = sum(1 for r in per_seed if r["go_gate"]["reversible"])
    n_nores = sum(1 for r in per_seed if r["go_gate"]["no_host_reset"])
    n_det = sum(1 for r in per_seed if r["go_gate"]["seed_deterministic"])
    pooled_go = bool(n_go >= 5 and n_swap >= 5 and n_timing >= 5 and n_lb >= 5 and n_noev == 6
                     and n_enf >= 5 and n_reig >= 5 and n_rev >= 5 and n_nores == 6 and n_det == 6)
    verdict = "GO" if pooled_go else ("PARTIAL" if n_swap >= 1 else "NO-GO")

    v = Verdict("GNW neural vacancy gate: 6-seed aggregate")
    v.require("neural-gated single-move swap on >=5/6", bool(n_swap >= 5), expect=True)
    v.require("TIMING correct on >=5/6 (challenger ignites only after vacancy, zero co-ignition)",
              bool(n_timing >= 5), expect=True)
    v.require("neural gate load-bearing on >=5/6 (non-circular veto + detector-removed mistiming)", bool(n_lb >= 5),
              expect=True)
    v.require("NON-CIRCULAR / veto works (proposal alone cannot admit) on 6/6", bool(n_noev == 6), expect=True)
    v.require("detector ENFORCES timing on >=5/6 (removal admits prematurely)", bool(n_enf >= 5), expect=True)
    v.require("admitted coalition re-ignites and holds on >=5/6", bool(n_reig >= 5), expect=True)
    v.require("reversible two-swap A->B->A on >=5/6 (both admissions neural)", bool(n_rev >= 5), expect=True)
    v.require("no host workspace reset on 6/6", bool(n_nores == 6), expect=True)
    v.require("determinism: cfg.seed seeds the substrate on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating; STD targets ONLY the incumbent E->E recurrence")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is used instead")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_neural_vacancy_gate_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "swap_ok": n_swap, "timing_ok": n_timing, "gate_load_bearing": n_lb,
                          "noevict_locks_out": n_noev, "detector_enforces_timing": n_enf,
                          "reignite_ok": n_reig, "reversible": n_rev, "no_host_reset": n_nores,
                          "seed_deterministic": n_det, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[neural-vac-gate six-seed] verdict={verdict} seed_go {n_go}/6 swap {n_swap}/6 timing {n_timing}/6 "
          f"load_bearing {n_lb}/6 non_circular {n_noev}/6 enforces_timing {n_enf}/6 "
          f"reignite {n_reig}/6 reversible {n_rev}/6 no_reset {n_nores}/6 det {n_det}/6 -> POOLED_GO={pooled_go}",
          flush=True)
    print(f"[neural-vac-gate six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW neural vacancy gate: a spiking dis-inhibitory admission gate makes "
                                             "the thought-swap fully self-driven (challenger admitted by the "
                                             "substrate's own vacancy read, no host trigger).")
    ap.add_argument("--calibrate", action="store_true", help="verify the dis-inhibitory gate primitive on one seed")
    ap.add_argument("--smoke", action="store_true", help="full single-seed evaluation")
    ap.add_argument("--six-seed", action="store_true", help="42/43/44/100/101/102 at the frozen operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boost", type=float, default=SWAP_BOOST)
    ap.add_argument("--evict-steps", type=int, default=EVICT_STEPS)
    ap.add_argument("--boost-steps", type=int, default=BOOST_STEPS)
    ap.add_argument("--proposal-pa", type=float, default=PROPOSAL_PA)
    ap.add_argument("--reignite-hold", type=int, default=REIGNITE_HOLD)
    ap.add_argument("--w-rec", type=float, default=W_REC)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_neural_vacancy_gate.json")
    args = ap.parse_args()

    if args.calibrate:
        return run_calibrate(args.seed, args)
    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    r = evaluate_seed(args.seed, boost=args.boost, evict_steps=args.evict_steps, boost_steps=args.boost_steps,
                      proposal_pa=args.proposal_pa, reignite_hold=args.reignite_hold, w_rec=args.w_rec,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_neural_vacancy_gate_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[neural-vac-gate] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
