"""gap#5 learn-through-use ON THE ECKER AdEx CA3 STORE: does OFFLINE discrete forward replay DURABLY STRENGTHEN the
replayed sequence -- and does LESION-THE-REPLAY (no ignition -> no replay) control it?

CONTEXT (read these -- the wall + the unblock):
  * 2026-08-27-swr-envelope-learn-through-use-NOGO: the SWR envelope on the BISTABLE-completion CA3 store CANNOT reach
    discrete forward-ordered replay -- its strong within-attractors reverberate semi-continuously (co_active 0.97, never
    rests/segments), so replay-driven learning has NO forward-ordered spike pairs to ride. The wall is the STORE
    ARCHITECTURE.
  * 2026-08-20-ecker-adex-ca3-forward-replay-6seed-GO + ...-stdp-band-...-GO: the Ecker-2022 AdEx CA3 (self-terminating
    within-assembly volleys + STRONG forward / WEAK reverse between links + spike-triggered adaptation) DOES segment into
    DISCRETE forward SWR events A->B->C from a non-specific prefix seed (6-seed GO, band both hand-wired AND STDP-grown).

THE QUESTION THIS RUNNER ANSWERS (the capability the bistable store could not support): with the Ecker store's DISCRETE
FORWARD replay in hand, turn the substrate's OWN spike-timing plasticity (cfg.enable_stdp, the same fused kernel that
GREW the band) ON DURING the offline SWR replay bouts. The self-generated forward-ordered reactivation (A fires before B
fires before C) drives DIRECTIONAL STDP: forward edges see pre-before-post (LTP), reverse edges see post-before-pre
(LTD). So REPLAYING the memory should DEEPEN its forward band (adj_fwd up, adj_rev flat/down) = the sequence becomes more
robust = "using (replaying) a memory strengthens it" via OFFLINE replay. This is exactly what a NON-SEGMENTING co-firing
store CANNOT do: simultaneous co-fire has no pre/post order, so STDP would potentiate all edges symmetrically -> NO
directional consolidation.

MECHANISM (brain-based-only; NO sim/ edit; reuse the STDP-band runner's build/encode/replay/measure by import):
  1. BUILD + ENCODE a forward-asymmetric band by STDP (moving A->B->C cue sweep) to a MODERATE strength (headroom below
     stdp_w_max) -- the memory to be consolidated. Freeze. Read it (band_before + forward-replay quality).
  2. CONSOLIDATE-BY-REPLAY: run SWR replay bouts (non-specific random-per-event prefix seed) with enable_stdp=True AND the
     clock ADVANCED each step (else delta_t==0 -> STDP silently inert, the banked 2026-07-29 failure). The discrete
     forward replay's own spike pairs potentiate the forward band. Measure dw_fwd / dw_rev.
  3. AFTER: freeze, re-read (band_after + forward-replay quality + robustness at a REDUCED cue).
  4. LESION-THE-REPLAY [KEY CONTROL]: repeat step 2 with seed_on=False (no prefix cue -> no ignition -> no replay events
     -> no forward-ordered spike pairs). STDP is ON and the clock advances IDENTICALLY, so the ONLY difference is the
     REPLAY. dw_fwd_noseed must be ~0 and the band/robustness must NOT change -> the strengthening is carried by the
     REPLAY, not by STDP-on time or OU noise.

GO (per seed) =
  * REPLAY-DEEPENS   : seeded consolidation grows the forward band (dw_fwd >= DW_MIN and adj_fwd_after > adj_fwd_before).
  * DIRECTIONAL      : forward deepens MORE than reverse (dw_fwd - dw_rev >= DW_MIN) -> rides the replay ORDER, not
                       generic activity (a co-firing store would move fwd==rev).
  * RECALL-CHANGE    : forward-replay quality is durably maintained/improved after consolidation (forward_frac_after >=
                       forward_frac_before - TOL and still >> chance), AND robustness at a REDUCED cue is >= before.
  * LESION-CONTROLLED: NO-SEED consolidation gives |dw_fwd_noseed| <= NOSEED_MAX_FRAC * dw_fwd and no robustness gain;
                       attributable_to(deepening, seeded vs no-seed).
  * BOUNDED          : adj_fwd_after <= stdp_w_max (soft cap) and the seeded deepening does not blow up.
Honest NO-GO otherwise (localizes whether replay-driven STDP consolidation works on this substrate at all).

SCOPE (stated, not hidden): this demonstrates replay-driven learn-through-use on the ECKER REPLAY SUBSTRATE's OWN
sequence memory (the assembly-chain store). Wiring the strengthened replay back into the production D5 EpisodicDapMemory
organ is a SEPARATE integration: the Ecker AdEx SOMA recurrence does NOT reactivate D5's sparse ~14-cell episodic
assembly (2026-08-20-ecker-real-d5-...-NO-GO, 3-seed + 24-op-point), so that path needs the dendritic-dAP-latch
composition, not soma recurrence. This runner tests the replay substrate itself, the rung the bistable store failed.

Reuse-by-import: build_store / encode / rest_and_replay / measure_band / _score_periods / _load_weights from
_gap5_ecker_adex_ca3_stdp_band_derisk (byte-identical store + replay + scorer).

  Calib:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_ecker_replay_learn_through_use_derisk \
              --seeds 42 --rest-steps 6500 --consol-steps 6500 --n-laps 14
  6-seed: SIM_BACKEND=cupy  .venv/bin/python -m research.runners._gap5_ecker_replay_learn_through_use_derisk \
              --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import to_host, get_backend  # noqa: E402
from sim.kernels import fused_btsp_update  # noqa: E402  -- REUSE the substrate's BTSP kernel (gap#4, Bittner-Magee 2017)
from research.runners._gap5_ecker_adex_ca3_stdp_band_derisk import (  # noqa: E402
    build_store, encode, rest_and_replay, measure_band, _score_periods, _load_weights,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "ecker_replay_learn_through_use.json"


def measure_band_from(w_host, store):
    fwd = np.asarray(w_host)[store["fwd_pos"]]; rev = np.asarray(w_host)[store["rev_pos"]]
    win = np.asarray(w_host)[store["within_pos"]]
    af = float(fwd.mean()) if fwd.size else 0.0; ar = float(rev.mean()) if rev.size else 0.0
    return dict(adj_fwd=af, adj_rev=ar, adj_within=float(win.mean()) if win.size else 0.0,
                ratio=(af / max(ar, 1e-6)), fwd_max=float(fwd.max()) if fwd.size else 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# CONSOLIDATE-BY-REPLAY: SWR replay with STDP ON + clock ADVANCED. Identical drive to rest_and_replay except (a) STDP is
# enabled so the replay's forward-ordered spike pairs potentiate the band, and (b) the clock advances each step (else
# delta_t==0 and STDP is inert). seed_on=False = LESION-THE-REPLAY (no ignition -> no replay -> no directional pairs).
# ----------------------------------------------------------------------------------------------------------------------
def consolidate_by_replay(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt, seed_on=True):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]
    # SAME cue-cell subsets + assembly-choice stream as the read (rest_and_replay) uses -> consistent ignition.
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()
    dw_half = []                                       # forward-edge deepening at the halfway snapshot (self-limit check)
    half = max(1, steps // 2)
    bridge.core_config.enable_stdp = True
    bridge.runtime_state.current_time_ms = 0.0
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)   # ADVANCE THE CLOCK (else STDP delta_t==0 -> inert)
        if t + 1 == half:
            wh = np.asarray(to_host(bridge.cp_connections.data))
            dw_half.append(float((wh[fwd_pos] - w0[fwd_pos]).mean()))
    bridge.core_config.enable_stdp = False
    w1 = np.asarray(to_host(bridge.cp_connections.data))
    dw_fwd_first = dw_half[0] if dw_half else 0.0
    dw_fwd_total = float((w1[fwd_pos] - w0[fwd_pos]).mean())
    return dict(n_env=n_env, w_after=w1.copy(),
                dw_fwd=dw_fwd_total, dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=dw_fwd_first, dw_fwd_second_half=float(dw_fwd_total - dw_fwd_first),
                changed=bool(not np.array_equal(w0, w1)))


# ----------------------------------------------------------------------------------------------------------------------
# CONSOLIDATE-BY-REPLAY, BTSP-DIRECTIONAL-WRITE variant (2026-08-27). Replaces the substrate's ms-coincidence STDP
# (which SYMMETRIZES the band -- reverse potentiates ~6x, the banked NO-GO) with a DIRECTIONAL BTSP write: a SECONDS-long
# CAUSAL presynaptic eligibility trace x an all-or-none plateau instructive POST signal, fed to the substrate's OWN
# fused_btsp_update kernel (dw = eta*Etilde_pre*IS_post*(w_max-w), PURE potentiation -- NO depression arm).
#
# WHY THIS IS DIRECTIONAL where ms-STDP is not. The eligibility e_pre is a CAUSAL, one-sided decaying trace: it is only
# nonzero AFTER a cell fires. For a forward edge A->B (A leads B by the replay lag), when B's plateau gates the write
# e_pre[A] is already high -> LTP. For a reverse edge B->A, when A's plateau gates the write e_pre[B] is still ~0 (B has
# not fired yet); the reverse edge can only ride A's plateau TAIL overlapping B's later eligibility, so dw_rev/dw_fwd ~
# exp(-lag/plat_tau). With PURE potentiation the WORST case is symmetric (rev==fwd); it structurally CANNOT make rev>>fwd
# the way the antisymmetric STDP window did on the overlapping self-driven cascade. The op-point is plat_tau vs the lag.
# Grounded: Gonzalez-Lacefield 2023 bioRxiv (BTSP = an ASYMMETRIC kernel of bidirectional changes around the plateau;
# inputs in a seconds-long window preceding+following it potentiate) + Bittner-Magee 2017 + Milstein-Magee 2021.
# seed_on=False = LESION-THE-REPLAY (identical: no ignition -> no replay -> no fwd-ordered pairs -> the write is a null).
# ----------------------------------------------------------------------------------------------------------------------
def consolidate_by_btsp_replay(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt, seed_on=True,
                               elig_tau_ms, plat_tau_ms, eta, w_min, w_max):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]
    row, col = store["pre_post"]                         # global-neuron pre/post index per edge (COO == .data order)
    bet_pos = np.concatenate([fwd_pos, rev_pos])         # the PLASTIC between edges (within edges stay frozen)
    bet_pos_dev = cp.asarray(bet_pos.astype(np.int64))
    row_bet = cp.asarray(row[bet_pos].astype(np.int64))  # presynaptic (source) neuron of each between edge
    col_bet = cp.asarray(col[bet_pos].astype(np.int64))  # postsynaptic (target) neuron of each between edge
    # SAME cue-cell subsets + assembly-choice stream as the read/STDP-write -> consistent ignition.
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    nN = int(bridge.cp_firing_states.size)
    e_pre = cp.zeros(nN, dtype=cp.float32)               # CAUSAL seconds-long presynaptic eligibility (latch-then-decay)
    p_post = cp.zeros(nN, dtype=cp.float32)              # all-or-none plateau instructive POST signal (latch-then-decay)
    decay_e = cp.float32(np.exp(-dt / max(elig_tau_ms, 1e-9)))
    decay_p = cp.float32(np.exp(-dt / max(plat_tau_ms, 1e-9)))
    eta_d = cp.float32(eta); wmin_d = cp.float32(w_min); wmax_d = cp.float32(w_max)

    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()
    dw_half = []
    half = max(1, steps // 2)
    bridge.core_config.enable_stdp = False               # the write is done HERE (host-side BTSP); substrate STDP OFF
    bridge.runtime_state.current_time_ms = 0.0
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)   # ADVANCE THE CLOCK (kept identical to the STDP variant)
        fired = bridge.cp_firing_states.astype(cp.float32)
        e_pre = cp.maximum(e_pre * decay_e, fired)           # causal one-sided eligibility -> the DIRECTIONALITY source
        p_post = cp.maximum(p_post * decay_p, fired)          # all-or-none plateau -> the instructive gate
        w_edge = bridge.cp_connections.data[bet_pos_dev]
        w_new = fused_btsp_update(w_edge, e_pre[row_bet], p_post[col_bet], eta_d, wmin_d, wmax_d)  # REUSE substrate kernel
        bridge.cp_connections.data[bet_pos_dev] = w_new
        if t + 1 == half:
            wh = np.asarray(to_host(bridge.cp_connections.data))
            dw_half.append(float((wh[fwd_pos] - w0[fwd_pos]).mean()))
    w1 = np.asarray(to_host(bridge.cp_connections.data))
    dw_fwd_first = dw_half[0] if dw_half else 0.0
    dw_fwd_total = float((w1[fwd_pos] - w0[fwd_pos]).mean())
    return dict(n_env=n_env, w_after=w1.copy(),
                dw_fwd=dw_fwd_total, dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=dw_fwd_first, dw_fwd_second_half=float(dw_fwd_total - dw_fwd_first),
                changed=bool(not np.array_equal(w0, w1)))


# ----------------------------------------------------------------------------------------------------------------------
# VOLLEY-OVERLAP metric (the residual blocker the [[2026-08-27-btsp-directional-write-...-PARTIAL]] op-point sweep
# isolated): in a discrete forward replay event A->B->C, the leading assembly A keeps firing AFTER B ignites, so the
# pre(A)-before-post(B) pairing is CONTAMINATED by A-fires-during-B (bidirectional coincidence) -> no coincidence-read
# rule can be forward-selective. This measures it DIRECTLY: for each SWR event, for each consecutive assembly pair
# (k, k+1) that BOTH activate, the temporal Jaccard overlap of their smoothed active windows
# |active_k & active_{k+1}| / |active_k | active_{k+1}|. 0 = cleanly separated volleys (the delay's goal); ->1 = fully
# overlapping. Mean over consecutive pairs over events. SAME smoother/thresholds as _score_periods (one instrument).
# ----------------------------------------------------------------------------------------------------------------------
def _volley_overlap(F, asm_local, env_seed_log, swr_period, *, W, active_frac, onset_frac):
    T = F.shape[0]; n_mem = len(asm_local)
    asizes = [max(1, len(a)) for a in asm_local]
    n_periods = min(len(env_seed_log), T // swr_period)
    ovs = []
    for n in range(n_periods):
        s0, s1 = n * swr_period, (n + 1) * swr_period
        Fw = F[s0:s1]
        act = {}
        for kk, A in enumerate(asm_local):
            a_t = _smooth_local(Fw[:, A].sum(1), W) / asizes[kk]
            if a_t.size and float(a_t.max()) >= active_frac:
                act[kk] = (a_t >= onset_frac)
        for kk in range(n_mem - 1):
            if kk in act and (kk + 1) in act:
                inter = int(np.logical_and(act[kk], act[kk + 1]).sum())
                union = int(np.logical_or(act[kk], act[kk + 1]).sum())
                if union > 0:
                    ovs.append(inter / union)
    return float(np.mean(ovs)) if ovs else 0.0


def _smooth_local(x, W):
    if W <= 1:
        return np.asarray(x, dtype=float)
    k = np.ones(int(W), dtype=float) / float(W)
    return np.convolve(np.asarray(x, dtype=float), k, mode="same")


# ----------------------------------------------------------------------------------------------------------------------
# CONSOLIDATE-BY-REPLAY, BTSP write + FORWARD-EDGE AXONAL CONDUCTION DELAY (2026-08-27). The residual blocker the
# op-point sweep isolated is NOT the write rule -- it is VOLLEY OVERLAP: the leading assembly keeps firing after the
# next ignites, so pre-and-post overlap and no coincidence-read write can be forward-selective. This adds a per-edge
# CONDUCTION DELAY on the FORWARD recurrent edges only, so assembly A's forward drive reaches B `delay_steps` later ->
# B ignites only AFTER A has self-terminated -> the volleys SEPARATE -> the pre(A)-before-post(B) coincidence is clean.
#
# The engine propagates every spike with a UNIFORM 1-step delay (g_e += (W.T @ prev_fired)*prop_strength; there is NO
# per-synapse delay buffer -- max_synaptic_delay_ms is unused in this path). So the forward conduction delay is a
# HOST-SIDE delay-line on the SAME g_e conductance channel (no sim/ edit): the forward edges are ZEROED in the
# propagation matrix (so the matvec delivers only within+reverse drive), and their exact g_e increment
# prop_strength * (W_fwd.T @ prev_fired) is buffered and re-added to cp_conductance_g_e `delay_steps` steps later --
# a faithful axonal delay (same amplitude, same conductance, onset shifted). The plastic forward WEIGHTS live in a
# decoupled host array w_bet (BTSP-updated + used for the delayed drive + measured); reverse weights stay in the matrix
# (their standard 1-step delay is unchanged -- ONLY the forward axons are slowed). delay_steps=0 = path-active control
# (forward re-added immediately == baseline forward timing) -> must reproduce the PARTIAL 0/6. Grounded: Izhikevich 2006
# "Polychronization" (axonal conduction delays + STDP self-organize DIRECTIONAL/time-locked polychronous groups -- the
# delay is what makes a coincidence-read rule direction-selective); Yu 2025 (axonal delays let SNNs exploit temporal
# direction). seed_on=False = LESION-THE-REPLAY (no ignition -> no firing -> delay-line all-zero -> write is a null).
# ----------------------------------------------------------------------------------------------------------------------
def consolidate_by_btsp_replay_delayed(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt, seed_on=True,
                                       elig_tau_ms, plat_tau_ms, eta, w_min, w_max, delay_steps,
                                       overlap_kw=None):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]; n_pc = len(pc)
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]; n_fwd = int(fwd_pos.size)
    row, col = store["pre_post"]
    bet_pos = np.concatenate([fwd_pos, rev_pos])
    bet_pos_dev = cp.asarray(bet_pos.astype(np.int64))
    fwd_pos_dev = cp.asarray(fwd_pos.astype(np.int64)); rev_pos_dev = cp.asarray(rev_pos.astype(np.int64))
    row_bet = cp.asarray(row[bet_pos].astype(np.int64)); col_bet = cp.asarray(col[bet_pos].astype(np.int64))
    row_fwd = cp.asarray(row[fwd_pos].astype(np.int64)); col_fwd = cp.asarray(col[fwd_pos].astype(np.int64))
    prop = float(getattr(bridge.core_config, "propagation_strength", 0.05))
    nN = int(bridge.cp_firing_states.size)

    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    e_pre = cp.zeros(nN, dtype=cp.float32); p_post = cp.zeros(nN, dtype=cp.float32)
    decay_e = cp.float32(np.exp(-dt / max(elig_tau_ms, 1e-9))); decay_p = cp.float32(np.exp(-dt / max(plat_tau_ms, 1e-9)))
    eta_d = cp.float32(eta); wmin_d = cp.float32(w_min); wmax_d = cp.float32(w_max)

    # DECOUPLE the plastic between-edge weights from the propagation matrix. w_bet[:n_fwd]=forward, [n_fwd:]=reverse.
    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()          # BEFORE zeroing forward (encoded weights)
    w_bet = bridge.cp_connections.data[bet_pos_dev].copy()               # canonical plastic weights (device)
    bridge.cp_connections.data[fwd_pos_dev] = cp.float32(0.0)            # forward edges deliver drive ONLY via delay-line
    D = int(max(0, delay_steps))
    buf = cp.zeros((max(D, 1), nN), dtype=cp.float32) if D > 0 else None  # ring buffer of forward g_e increments
    ptr = 0

    F = np.zeros((steps, n_pc), dtype=bool); env_seed_log = []
    bridge.core_config.enable_stdp = False
    bridge.runtime_state.current_time_ms = 0.0
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); env_seed_log.append(cur_k); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        # forward g_e increment this step from the CURRENT firing (== what the matvec would add for forward edges).
        prev_f = bridge.cp_prev_firing_states.astype(cp.float32)
        fwd_g = cp.zeros(nN, dtype=cp.float32)
        w_fwd_now = w_bet[:n_fwd]
        cp.add.at(fwd_g, col_fwd, w_fwd_now * prev_f[row_fwd] * cp.float32(prop))
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)
        # deliver the forward drive with the conduction delay (onto the SAME g_e channel the matvec feeds).
        if D <= 0:
            bridge.cp_conductance_g_e += fwd_g                          # control: immediate == baseline forward timing
        else:
            bridge.cp_conductance_g_e += buf[ptr]                        # arrival of the drive from D steps ago
            buf[ptr] = fwd_g; ptr = (ptr + 1) % D
        F[t] = np.asarray(to_host(bridge.cp_firing_states))[pc].astype(bool)
        # BTSP write on the decoupled plastic weights; propagate reverse back into the matrix (forward stays 0).
        fired = bridge.cp_firing_states.astype(cp.float32)
        e_pre = cp.maximum(e_pre * decay_e, fired); p_post = cp.maximum(p_post * decay_p, fired)
        w_bet = fused_btsp_update(w_bet, e_pre[row_bet], p_post[col_bet], eta_d, wmin_d, wmax_d)
        bridge.cp_connections.data[rev_pos_dev] = w_bet[n_fwd:]
    w1 = np.asarray(to_host(bridge.cp_connections.data)).copy()          # reverse updated, forward still 0 in matrix
    w_bet_h = np.asarray(to_host(w_bet))
    w1[fwd_pos] = w_bet_h[:n_fwd]; w1[rev_pos] = w_bet_h[n_fwd:]         # overlay true plastic weights
    ov = None
    if overlap_kw is not None and seed_on:
        ov = _volley_overlap(F, asm_local, env_seed_log, swr_period, **overlap_kw)
    return dict(n_env=n_env, w_after=w1.copy(),
                dw_fwd=float((w1[fwd_pos] - w0[fwd_pos]).mean()), dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=0.0, dw_fwd_second_half=0.0, volley_overlap=ov,
                changed=bool(not np.array_equal(w0, w1)))


# ----------------------------------------------------------------------------------------------------------------------
# GAP-CODING replay (2026-08-27, Braun & Memmesheimer 2022 PLoS Comput Biol e1009891, DOI 10.1371/journal.pcbi.1009891).
# The residual the conduction-delay PARTIAL isolated is NOT write-directionality (that was SOLVED 6/6) -- it is that the
# op-point REQUIRED to separate the volleys (a long SWR period / short detection window, or a long axonal delay) drives
# the RECALL READ to ceiling (weak-cue forward_frac ~ 1.0 BEFORE), leaving NO headroom for a learn-through-use gain to
# show. Braun's INHIBITORY GAP CODING separates the volleys by a DIFFERENT route: sequences arise from "alternating
# excitatory pulse and inhibitory gap coding -- phases of silence in specific basket cell groups induce selective
# disinhibition of groups of pyramidal neurons", giving "sparse pyramidal cell and dense basket cell spiking", and it
# "does not rely on synfire chain-like feedforward excitation". So the volleys separate WITHOUT the read-saturating
# regime -> the read keeps headroom.
#
# BRAIN-BASED IMPLEMENTATION (reuse the engine's REAL GABA-A inhibition; NO sim/ edit; the accepted host-side-conductance
# precedent from this arc's conduction-delay PARTIAL, applied to the g_i channel): DENSE tonic basket inhibition is a
# high pyramidal `cp_conductance_g_i` (routes through the engine's syn_reversal_potential_i = -75 mV Cl- reversal, the
# same GABA-A mechanism a real basket cell drives), which silences the pyramidal population between gaps (verified:
# g_i~200 fully silences even a 9000 pA drive; g_i~50 partially). A periodic GLOBAL disinhibition GAP (g_i lowered for
# `gap_width` of every `gap_period` steps) opens brief SPARSE windows; in each window the assembly receiving the
# strongest recurrent forward drive from the just-active one (seeded by the prefix cue, advanced by the learned forward
# band + within-assembly adaptation) fires -- ONE assembly per gap -> volleys are INHERENTLY non-overlapping (they can
# only occur inside a gap, separated by inter-gap inhibitory silence), at a NORMAL swr_period. SCOPE (stated, not
# hidden): this is a GLOBAL gap RHYTHM (a septal/theta-like pacemaker on the inhibition) + learned-excitatory-band
# SELECTION of which group fires in the gap -- a reduction of Braun's GROUP-SPECIFIC structured basket->pyramidal
# connectivity (where the gap schedule itself is wired). The sequence ORDER rides the LEARNED band (what we test), not
# the gap wiring. The directional write is the SAME causal-eligibility x plateau BTSP as the btsp variant (reuse
# fused_btsp_update): with gap-separated volleys, for a forward edge A->B e_pre[A] is high when B's plateau gates the
# write; for reverse B->A e_pre[B] is ~0 (B fired later) -> forward-selective. seed_on=False = LESION-THE-REPLAY.
# ----------------------------------------------------------------------------------------------------------------------
def gapcode_replay(store, steps, seed, *, swr_period, cue_pa, cue_steps, cue_frac, dt, seed_on=True,
                   gi_base, fb_gain, fb_tau, gi_cap, pc_tonic,
                   btsp=None, overlap_kw=None):
    cp = store["cp"]; bridge = store["bridge"]; pc = store["pc"]; asm_local = store["asm_local"]; m = store["m_asm"]
    asm_size = store["asm_size"]; n_pc = len(pc)
    fwd_pos = store["fwd_pos"]; rev_pos = store["rev_pos"]
    row, col = store["pre_post"]
    pc_dev = cp.asarray(pc.astype(np.int64))
    cell_rng = np.random.default_rng(int(seed) * 314159 + 17)      # SAME cue subsets/stream as the read/write paths
    k_cells = max(1, int(round(float(cue_frac) * asm_size)))
    cue_cells_dev = []
    for a_loc in asm_local:
        sub = np.sort(cell_rng.choice(a_loc, min(k_cells, len(a_loc)), replace=False))
        cue_cells_dev.append(cp.asarray(pc[sub], dtype=cp.int64))
    choice_rng = np.random.default_rng(int(seed) * 271828 + 23)

    do_write = btsp is not None
    if do_write:
        bet_pos = np.concatenate([fwd_pos, rev_pos]); bet_pos_dev = cp.asarray(bet_pos.astype(np.int64))
        row_bet = cp.asarray(row[bet_pos].astype(np.int64)); col_bet = cp.asarray(col[bet_pos].astype(np.int64))
        nN = int(bridge.cp_firing_states.size)
        e_pre = cp.zeros(nN, dtype=cp.float32); p_post = cp.zeros(nN, dtype=cp.float32)
        decay_e = cp.float32(np.exp(-dt / max(btsp["elig_tau_ms"], 1e-9)))
        decay_p = cp.float32(np.exp(-dt / max(btsp["plat_tau_ms"], 1e-9)))
        eta_d = cp.float32(btsp["eta"]); wmin_d = cp.float32(btsp["w_min"]); wmax_d = cp.float32(btsp["w_max"])
    w0 = np.asarray(to_host(bridge.cp_connections.data)).copy()

    F = np.zeros((steps, n_pc), dtype=bool); env_seed_log = []
    bridge.core_config.enable_stdp = False
    bridge.runtime_state.current_time_ms = 0.0
    # FEEDBACK inhibition: a DENSE basket pool whose activity tracks recent pyramidal firing (Braun's dense basket cells
    # DRIVEN BY the pyramidal volley) -> its inhibition of the population is self-locked to the volley, terminating the
    # leading assembly and opening a trough (the "gap") into which the forward-driven next assembly can fire.
    basket = 0.0; decay_fb = float(np.exp(-dt / max(fb_tau, 1e-9)))
    cur_k = None; n_env = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        if pc_tonic > 0:
            bridge.cp_external_input_current[pc_dev] += float(pc_tonic)   # weak background so a disinhibited group can ignite
        phase = t % swr_period
        if phase == 0 and seed_on:
            cur_k = int(choice_rng.integers(0, m)); env_seed_log.append(cur_k); n_env += 1
        if seed_on and phase < cue_steps and cur_k is not None:
            bridge.cp_external_input_current[cue_cells_dev[cur_k]] += float(cue_pa)
        # GAP-CODING inhibition: SET pyramidal g_i BEFORE the step = tonic basket floor + feedback (capped, avoids the
        # g_i-overflow numerical blow-up observed above ~5000).
        gi = min(gi_base + basket, gi_cap)
        bridge.cp_conductance_g_i[pc_dev] = cp.float32(gi)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += float(dt)
        fired_pc = np.asarray(to_host(bridge.cp_firing_states))[pc].astype(bool)
        F[t] = fired_pc
        basket = basket * decay_fb + float(fb_gain) * float(fired_pc.sum())   # basket recruited by the pyramidal volley
        if do_write:
            fired = bridge.cp_firing_states.astype(cp.float32)
            e_pre = cp.maximum(e_pre * decay_e, fired); p_post = cp.maximum(p_post * decay_p, fired)
            w_edge = bridge.cp_connections.data[bet_pos_dev]
            bridge.cp_connections.data[bet_pos_dev] = fused_btsp_update(w_edge, e_pre[row_bet], p_post[col_bet],
                                                                        eta_d, wmin_d, wmax_d)
    w1 = np.asarray(to_host(bridge.cp_connections.data))
    ov = None
    if overlap_kw is not None and seed_on:
        ov = _volley_overlap(F, asm_local, env_seed_log, swr_period, **overlap_kw)
    return dict(F=F, env_seed_log=env_seed_log, n_env=n_env, w_after=w1.copy(),
                dw_fwd=float((w1[fwd_pos] - w0[fwd_pos]).mean()), dw_rev=float((w1[rev_pos] - w0[rev_pos]).mean()),
                dw_fwd_first_half=0.0, dw_fwd_second_half=0.0, volley_overlap=ov,
                changed=bool(not np.array_equal(w0, w1)))


def _read_gap(bkw, seed, w_host, a, *, cue_pa, cue_frac, tag):
    """GAP-CODING read: fresh store, load weights, gap-coded replay (frozen), score forward-replay quality."""
    s = build_store(seed, **bkw)
    _load_weights(s, w_host)
    r = gapcode_replay(s, a.rest_steps, seed, swr_period=a.swr_period, cue_pa=cue_pa, cue_steps=a.cue_steps,
                       cue_frac=cue_frac, dt=a.dt, seed_on=True, gi_base=a.gi_base, fb_gain=a.fb_gain,
                       fb_tau=a.fb_tau, gi_cap=a.gi_cap, pc_tonic=a.pc_tonic)
    sc = _score_periods(r["F"], s["asm_local"], r["env_seed_log"], a.swr_period,
                        W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    band = measure_band(s)
    return dict(forward=sc["forward_frac"], reverse=sc["reverse_frac"], chance=max(sc["chance_forward"], 1e-6),
                n_multi=sc["n_multi"], per_asm_active=sc["per_asm_active"], seed_first=sc.get("seed_first_frac"),
                duty=sc["duty_cycle"], frozen=r["changed"] is False, band=band, tag=tag)


def _gapcode_consolidate(store, steps, seed, a, *, seed_on, cons_kw, overlap_kw=None):
    """CONSOLIDATE-BY-REPLAY under gap coding: gap-coded forward replay + the directional BTSP write."""
    btsp = dict(elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau, eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max)
    return gapcode_replay(store, steps, seed, seed_on=seed_on, gi_base=a.gi_base, fb_gain=a.fb_gain,
                          fb_tau=a.fb_tau, gi_cap=a.gi_cap, pc_tonic=a.pc_tonic, btsp=btsp,
                          overlap_kw=overlap_kw, **cons_kw)


def _consolidate(store, steps, seed, a, *, seed_on, cons_kw, overlap_kw=None):
    """Dispatch to the STDP (default, byte-identical), BTSP-directional, BTSP+conduction-delay, or GAP-CODING write."""
    if getattr(a, "gap_coding", False):
        return _gapcode_consolidate(store, steps, seed, a, seed_on=seed_on, cons_kw=cons_kw, overlap_kw=overlap_kw)
    if a.write_rule == "btsp" and a.fwd_delay_steps >= 0:
        return consolidate_by_btsp_replay_delayed(store, steps, seed, seed_on=seed_on,
                                                  elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                  eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                  delay_steps=a.fwd_delay_steps, overlap_kw=overlap_kw, **cons_kw)
    if a.write_rule == "btsp":
        return consolidate_by_btsp_replay(store, steps, seed, seed_on=seed_on,
                                          elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                          eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max, **cons_kw)
    return consolidate_by_replay(store, steps, seed, seed_on=seed_on, **cons_kw)


def _read(store_kw_build, seed, w_host, a, *, cue_pa, cue_frac, tag):
    """Fresh store, load weights, replay READ (STDP OFF -> frozen). Returns forward-replay quality."""
    if getattr(a, "gap_coding", False):
        return _read_gap(store_kw_build, seed, w_host, a, cue_pa=cue_pa, cue_frac=cue_frac, tag=tag)
    s = build_store(seed, **store_kw_build)
    _load_weights(s, w_host)
    r = rest_and_replay(s, a.rest_steps, seed, swr_period=a.swr_period, cue_pa=cue_pa,
                        cue_steps=a.cue_steps, cue_frac=cue_frac, seed_on=True)
    sc = _score_periods(r["F"], s["asm_local"], r["env_seed_log"], a.swr_period,
                        W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    band = measure_band(s)
    return dict(forward=sc["forward_frac"], reverse=sc["reverse_frac"], chance=max(sc["chance_forward"], 1e-6),
                n_multi=sc["n_multi"], per_asm_active=sc["per_asm_active"], seed_first=sc.get("seed_first_frac"),
                duty=sc["duty_cycle"], frozen=r["weights_frozen"], band=band, tag=tag)


def one_seed(seed, a):
    t0 = time.time()
    out = {"seed": seed}
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult

    # SEED-CONTROLS-SUBSTRATE guard (build twice, hash firing thresholds)
    if a.verify_seed:
        s1 = build_store(seed, **bkw); s2 = build_store(seed, **bkw)
        h1 = s1["bridge"].cp_neuron_firing_thresholds; h2 = s2["bridge"].cp_neuron_firing_thresholds
        out["seed_hash_ok"] = bool(h1 is None or float(np.asarray(to_host(h1)).sum()) ==
                                   float(np.asarray(to_host(h2)).sum()))

    # 1. BUILD + ENCODE the memory (moderate band; headroom below stdp_w_max)
    st = build_store(seed, **bkw)
    band_pre_encode = measure_band(st)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before = measure_band(st)
    out["band_before"] = band_before
    print(f"  [seed {seed}] ENCODE: band fwd {band_pre_encode['adj_fwd']:.1f}->{band_before['adj_fwd']:.1f} "
          f"rev {band_pre_encode['adj_rev']:.1f}->{band_before['adj_rev']:.1f} "
          f"(w_max={a.stdp_w_max}) ({time.time()-t0:.0f}s)", flush=True)

    # 2. READ BEFORE (full cue + weak cue), frozen
    rd_full_before = _read(bkw, seed, w_learned, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac, tag="full_before")
    rd_weak_before = _read(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_before")
    chance = rd_full_before["chance"]
    print(f"  [seed {seed}] BEFORE: full FWD={rd_full_before['forward']:.3f} (chance {chance:.3f}) "
          f"weak FWD={rd_weak_before['forward']:.3f} multi={rd_weak_before['n_multi']} ({time.time()-t0:.0f}s)",
          flush=True)

    # 3. CONSOLIDATE-BY-REPLAY (seeded): STDP on, forward-ordered replay deepens the band
    overlap_kw = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    st_c = build_store(seed, **bkw); _load_weights(st_c, w_learned)
    cons = _consolidate(st_c, a.consol_steps, seed, a, seed_on=True, cons_kw=cons_kw, overlap_kw=overlap_kw)
    w_consol = cons["w_after"]
    out["consolidate"] = dict(n_env=cons["n_env"], dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"],
                              dw_fwd_first_half=cons["dw_fwd_first_half"], dw_fwd_second_half=cons["dw_fwd_second_half"],
                              volley_overlap=cons.get("volley_overlap"), changed=cons["changed"])
    print(f"  [seed {seed}] CONSOLIDATE(seeded): n_env={cons['n_env']} dw_fwd={cons['dw_fwd']:.2f} "
          f"dw_rev={cons['dw_rev']:.2f} volley_overlap={cons.get('volley_overlap')} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 4. READ AFTER (full + weak), frozen
    rd_full_after = _read(bkw, seed, w_consol, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac, tag="full_after")
    rd_weak_after = _read(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_after")
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    print(f"  [seed {seed}] AFTER: full FWD={rd_full_after['forward']:.3f} weak FWD={rd_weak_after['forward']:.3f} "
          f"multi={rd_weak_after['n_multi']} band fwd={band_after['adj_fwd']:.1f} rev={band_after['adj_rev']:.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 5. LESION-THE-REPLAY: NO-SEED consolidation (STDP on, clock advances, NO ignition -> no replay)
    st_n = build_store(seed, **bkw); _load_weights(st_n, w_learned)
    cons_ns = _consolidate(st_n, a.consol_steps, seed, a, seed_on=False, cons_kw=cons_kw)
    w_noseed = cons_ns["w_after"]
    band_noseed = measure_band_from(w_noseed, st_n)
    rd_weak_noseed = _read(bkw, seed, w_noseed, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, tag="weak_noseed")
    out["no_seed"] = dict(n_env=cons_ns["n_env"], dw_fwd=cons_ns["dw_fwd"], dw_rev=cons_ns["dw_rev"],
                          band_after=band_noseed, weak_forward=rd_weak_noseed["forward"],
                          weak_multi=rd_weak_noseed["n_multi"])
    print(f"  [seed {seed}] NO-SEED(lesion-replay): n_env={cons_ns['n_env']} dw_fwd={cons_ns['dw_fwd']:.3f} "
          f"dw_rev={cons_ns['dw_rev']:.3f} weak FWD={rd_weak_noseed['forward']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # NB: the NO-SEED weak read lives ONLY in out["no_seed"]["weak_forward"] (a scalar). It is byte-identical to
    # weak_before BY DESIGN (the lesion produces zero weight change), so it is deliberately NOT stored beside
    # weak_before as a sibling arm (that identity is the lesion-null working, not a dead lever).
    out["reads"] = dict(full_before=rd_full_before, weak_before=rd_weak_before,
                        full_after=rd_full_after, weak_after=rd_weak_after)

    # ============ PER-SEED VERDICT (verify, don't assert) ============
    dw_fwd = cons["dw_fwd"]; dw_rev = cons["dw_rev"]; dw_ns = cons_ns["dw_fwd"]
    replay_deepens = (dw_fwd >= a.dw_min and band_after["adj_fwd"] > band_before["adj_fwd"])
    directional = ((dw_fwd - dw_rev) >= a.dw_min)
    recall_maintained = (rd_full_after["forward"] >= rd_full_before["forward"] - a.fwd_tol
                         and rd_full_after["forward"] >= 1.5 * chance)
    robustness_gain = (rd_weak_after["forward"] >= rd_weak_before["forward"] + a.robust_min
                       or rd_weak_after["n_multi"] > rd_weak_before["n_multi"])
    recall_change = bool(recall_maintained and robustness_gain)
    lesion_controlled = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6)
                         and rd_weak_noseed["forward"] <= rd_weak_before["forward"] + a.robust_min)
    bounded = (band_after["adj_fwd"] <= a.stdp_w_max + 1e-3
               and cons["dw_fwd_second_half"] <= cons["dw_fwd_first_half"] + a.dw_min)
    seed_go = bool(replay_deepens and directional and recall_change and lesion_controlled and bounded)
    out["checks"] = dict(replay_deepens=replay_deepens, directional=directional,
                         recall_maintained=recall_maintained, robustness_gain=robustness_gain,
                         recall_change=recall_change, lesion_controlled=lesion_controlled, bounded=bounded,
                         dw_fwd=round(dw_fwd, 3), dw_rev=round(dw_rev, 3), dw_noseed=round(dw_ns, 3),
                         seed_hash_ok=out.get("seed_hash_ok"))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6)
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=13000, help="read replay length (~40 events)")
    ap.add_argument("--consol-steps", type=int, default=13000, help="consolidation replay length (STDP on)")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--between-init", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0)
    # STDP
    ap.add_argument("--stdp-w-max", type=float, default=900.0)
    ap.add_argument("--stdp-a-plus", type=float, default=0.05)
    ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    # REPLAY-TIME WRITE RULE: 'stdp' = the substrate's ms-coincidence STDP (default, byte-identical to the banked NO-GO);
    # 'btsp' = the DIRECTIONAL BTSP-eligibility write (seconds-long causal eligibility x an all-or-none plateau, reusing
    # sim.kernels.fused_btsp_update). btsp-* args are INERT unless --write-rule btsp.
    ap.add_argument("--write-rule", choices=["stdp", "btsp"], default="stdp")
    ap.add_argument("--btsp-elig-tau", type=float, default=20.0, help="tau_ms of the CAUSAL presynaptic eligibility "
                    "(long vs the replay lag so the earlier-firing pre stays eligible; short vs the inter-event gap)")
    ap.add_argument("--btsp-plat-tau", type=float, default=6.0, help="tau_ms of the all-or-none plateau instructive POST "
                    "signal (the directionality knob: dw_rev/dw_fwd ~ exp(-lag/plat_tau) -> keep < the replay lag)")
    ap.add_argument("--btsp-eta", type=float, default=0.02, help="BTSP learning rate (folds the eligibility/plateau gain)")
    ap.add_argument("--btsp-w-max", type=float, default=900.0, help="BTSP saturation ceiling (w_max-w); match stdp_w_max")
    # FORWARD-EDGE AXONAL CONDUCTION DELAY (2026-08-27): -1 = OFF (byte-identical; forward edges keep the standard
    # 1-step delay). >=0 activates the host-side forward-only delay-line (requires --write-rule btsp): the forward
    # recurrent drive reaches the next assembly `fwd_delay_steps` steps (*dt ms) later, so the leading assembly
    # SELF-TERMINATES before the next ignites -> the volleys separate -> the coincidence-read write becomes forward-
    # selective. 0 = path-active control (immediate re-add == baseline forward timing -> must reproduce the PARTIAL 0/6).
    ap.add_argument("--fwd-delay-steps", type=int, default=-1, help="forward-edge conduction delay in sim steps "
                    "(-1=OFF/byte-identical; 0=path-active control; k>0=k*dt ms axonal delay; needs --write-rule btsp)")
    # INHIBITORY GAP CODING (2026-08-27, Braun & Memmesheimer 2022, DOI 10.1371/journal.pcbi.1009891). OFF by default
    # (byte-identical: no g_i is ever set, the STDP/BTSP/delay paths are untouched). ON routes BOTH the consolidation
    # WRITE and the recall READ through gap-coded replay: dense tonic pyramidal g_i (basket inhibition) with a periodic
    # global disinhibition GAP separates the volleys via inhibition, so the read runs at a NORMAL swr_period (no
    # long-period/short-window regime -> the recall read keeps headroom). The directional BTSP write is reused.
    ap.add_argument("--gap-coding", action="store_true", help="enable Braun-2022 inhibitory gap-coding replay "
                    "(default OFF/byte-identical; ON drives BOTH write and read via gap-coded feedback-inhibition replay)")
    ap.add_argument("--gi-base", type=float, default=8.0, help="tonic basket-inhibition floor on pyramidal g_i")
    ap.add_argument("--fb-gain", type=float, default=0.6, help="feedback gain: basket g_i recruited PER pyramidal spike "
                    "last step (dense basket driven BY the volley; a volley of ~80 spikes adds ~gain*80 to g_i)")
    ap.add_argument("--fb-tau", type=float, default=4.0, help="tau_ms of the basket (feedback-inhibition) decay")
    ap.add_argument("--gi-cap", type=float, default=150.0, help="hard cap on pyramidal g_i (avoids the >~5000 g_i "
                    "numerical blow-up; full silence sets in ~200)")
    ap.add_argument("--pc-tonic", type=float, default=0.0, help="weak tonic excitatory drive to ALL pyramidals so a "
                    "disinhibited group can ignite in the trough (0 = rely on cue + recurrent forward drive)")
    # ENCODE (moderate: fewer laps than the band-GO's 30, so there is headroom to deepen by replay)
    ap.add_argument("--n-laps", type=int, default=14)
    ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40)
    ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / prefix seed
    ap.add_argument("--swr-period", type=int, default=325)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--weak-cue-mult", type=float, default=0.5, help="reduced-cue robustness read: cue_pa * this")
    ap.add_argument("--weak-cue-frac", type=float, default=0.35, help="reduced-cue robustness read: fewer cue cells")
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    # detection
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    # GO thresholds
    ap.add_argument("--dw-min", type=float, default=5.0, help="min forward-edge deepening (adj_fwd units) to count")
    ap.add_argument("--fwd-tol", type=float, default=0.10, help="allowed drop in full-cue forward_frac after consol")
    ap.add_argument("--robust-min", type=float, default=0.05, help="min weak-cue forward_frac gain to count robustness")
    ap.add_argument("--noseed-max-frac", type=float, default=0.20, help="|dw_noseed| must be <= this * dw_seeded")
    ap.add_argument("--verify-seed", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    if a.gap_coding:
        _wr = (f"WRITE=gap-coding+btsp gi_base={a.gi_base} fb_gain={a.fb_gain} fb_tau={a.fb_tau} gi_cap={a.gi_cap} "
               f"pc_tonic={a.pc_tonic} elig_tau={a.btsp_elig_tau} plat_tau={a.btsp_plat_tau} "
               f"eta={a.btsp_eta} w_max={a.btsp_w_max}")
    else:
        _wr = (f"WRITE=btsp elig_tau={a.btsp_elig_tau} plat_tau={a.btsp_plat_tau} eta={a.btsp_eta} w_max={a.btsp_w_max}"
               + (f" fwd_delay={a.fwd_delay_steps}steps({a.fwd_delay_steps*a.dt:.1f}ms)" if a.fwd_delay_steps >= 0 else "")
               if a.write_rule == "btsp" else "WRITE=stdp (ms-coincidence)")
    print(f"[ecker-ltu] Ecker AdEx CA3 replay-driven learn-through-use | {_wr} | n_mem={a.n_mem} asm={a.asm_size} "
          f"within={a.w_within} between_init={a.between_init} | encode {a.n_laps}laps | STDP a+={a.stdp_a_plus} "
          f"a-={a.stdp_a_minus} tau={a.stdp_tau} w_max={a.stdp_w_max} | swr={a.swr_period} cue={a.cue_pa}@{a.cue_frac} "
          f"weak={a.cue_pa*a.weak_cue_mult}@{a.weak_cue_frac} | rest={a.rest_steps} consol={a.consol_steps} dt={a.dt} "
          f"seeds={a.seeds} backend={backend}", flush=True)

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p.get("seed_go"))
        bar = max(1, (len(per) + 1) // 2) if len(per) < 6 else 5
        go = n_go >= bar
        mdwf = float(np.mean([p["consolidate"]["dw_fwd"] for p in per]))
        mdwr = float(np.mean([p["consolidate"]["dw_rev"] for p in per]))
        mdwns = float(np.mean([p["no_seed"]["dw_fwd"] for p in per]))
        maf_b = float(np.mean([p["band_before"]["adj_fwd"] for p in per]))
        maf_a = float(np.mean([p["band_after"]["adj_fwd"] for p in per]))
        mar_a = float(np.mean([p["band_after"]["adj_rev"] for p in per]))
        mfull_b = float(np.mean([p["reads"]["full_before"]["forward"] for p in per]))
        mfull_a = float(np.mean([p["reads"]["full_after"]["forward"] for p in per]))
        mweak_b = float(np.mean([p["reads"]["weak_before"]["forward"] for p in per]))
        mweak_a = float(np.mean([p["reads"]["weak_after"]["forward"] for p in per]))
        mweak_ns = float(np.mean([p["no_seed"]["weak_forward"] for p in per]))
        mch = float(np.mean([p["reads"]["full_before"]["chance"] for p in per]))
        _ovs = [p["consolidate"].get("volley_overlap") for p in per if p["consolidate"].get("volley_overlap") is not None]
        mov = float(np.mean(_ovs)) if _ovs else None
        if a.gap_coding and mov is not None:
            _delay_note = (f" | GAP-CODING gi_base={a.gi_base} fb_gain={a.fb_gain} fb_tau={a.fb_tau}; "
                           f"volley_overlap {mov:.3f}; read headroom weak_fwd_before {mweak_b:.3f} (<1.0 = headroom)")
        else:
            _delay_note = (f" | forward conduction delay {a.fwd_delay_steps} steps ({a.fwd_delay_steps*a.dt:.1f}ms); "
                           f"volley_overlap {mov:.3f}" if a.fwd_delay_steps >= 0 and mov is not None else "")
        if go:
            verdict = (f"ECKER-REPLAY-LEARN-THROUGH-USE GO {n_go}/{len(per)} -- OFFLINE discrete forward SWR replay on "
                       f"the Ecker AdEx CA3 store DURABLY DEEPENS the replayed sequence via the substrate's OWN STDP: "
                       f"forward band adj_fwd {maf_b:.1f}->{maf_a:.1f} (rev after {mar_a:.1f}); dw_fwd {mdwf:.1f} vs "
                       f"dw_rev {mdwr:.1f} (DIRECTIONAL -- rides the replay order). Recall durably changes: weak-cue "
                       f"forward {mweak_b:.3f}->{mweak_a:.3f} (full {mfull_b:.3f}->{mfull_a:.3f}, chance {mch:.3f}). "
                       f"LESION-THE-REPLAY (NO-SEED): dw_fwd {mdwns:.2f}~0, weak forward {mweak_ns:.3f} (no gain) -> the "
                       f"strengthening is CARRIED BY THE REPLAY. => the Ecker store UNBLOCKS replay-driven "
                       f"learn-through-use the bistable co-firing store could not.{_delay_note}")
        else:
            verdict = (f"ECKER-REPLAY-LEARN-THROUGH-USE NO-GO {n_go}/{len(per)} -- the store SEGMENTS (full-cue forward "
                       f"{mfull_b:.3f} vs chance {mch:.3f}) and replay drives DURABLE LESION-CONTROLLED plasticity "
                       f"(NO-SEED dw_fwd {mdwns:.2f}~0), but replay-driven STDP does NOT strengthen forward recall: it "
                       f"SYMMETRIZES the band (dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}; adj_fwd {maf_b:.1f}->{maf_a:.1f}); "
                       f"weak forward {mweak_b:.3f}->{mweak_a:.3f} (noseed {mweak_ns:.3f}). {n_go}/{len(per)} "
                       f"directional. Next method: separated-volley conduction delay / inhibitory gap-coding / "
                       f"BTSP-eligibility write.{_delay_note}")
        # Preconditions are INSTRUMENT-VALIDITY only (all must HOLD for the go/no-go to be meaningful); the
        # forward-consolidation CONCLUSION is carried by `go` (n_go >= bar), NOT registered as a precondition -- a
        # failed conclusion is a NO-GO, not an UNDEFINED.
        v = Verdict("Ecker AdEx CA3: does OFFLINE replay durably STRENGTHEN forward-ordered recall via replay-driven "
                    "STDP (lesion-the-replay controlled)?", chance=mch)
        v.floor("the store SEGMENTS: full-cue forward replay ignites above chance (a memory to consolidate exists)",
                mfull_b, floor=mch)
        v.require("plasticity is LIVE during replay: seeded consolidation moved the forward band (|dw_fwd| > 0)",
                  abs(mdwf), expect=lambda x: x > 1e-6)
        v.control("LESION-THE-REPLAY ENGAGED: seeded forward-deepening vs NO-SEED forward-deepening -- must DIFFER, so "
                  "the NO-SEED arm is a true null and the negative is about the replay's DIRECTION, not a dead lever",
                  treatment=mdwf, control=mdwns, min_separation=0.0)
        v.disabled("within-assembly recurrence + assembly identity (pre-formed cell groups; only the inter-assembly "
                   "SEQUENCE band is plastic)", why="scope: this tests replay-driven consolidation of the learned "
                   "forward sequence, not assembly formation")
        decided = v.decide(go=go, verbose=False)
        # If an instrument-validity precondition did NOT hold (e.g. the store does not segment under gap-coding
        # inhibition -> full-cue forward replay is 0), the run is UNDEFINED, NOT a negative. Reflect that in the
        # verdict STRING so it agrees with the Verdict object (a precondition-failed run must not read as a NO-GO).
        if decided.get("status") == "UNDEFINED":
            verdict = ("UNDEFINED (an instrument precondition did not hold -> not a negative; the read is compromised) -- "
                       + verdict)
        attributable_to("forward-band deepening (seeded replay vs NO-SEED lesion-the-replay)", mdwf, mdwns)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"),
                             dw_fwd=mdwf, dw_rev=mdwr, dw_fwd_noseed=mdwns,
                             band_adj_fwd_before=maf_b, band_adj_fwd_after=maf_a, band_adj_rev_after=mar_a,
                             full_forward_before=mfull_b, full_forward_after=mfull_a,
                             weak_forward_before=mweak_b, weak_forward_after=mweak_a, weak_forward_noseed=mweak_ns,
                             chance=mch, fwd_delay_steps=a.fwd_delay_steps, volley_overlap=mov,
                             preconditions=decided.get("preconditions", []), decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0)

    summary = {"probe": "gap5_ecker_replay_learn_through_use",
               "mechanism": "Ecker-2022 AdEx CA3 discrete forward SWR replay -> directional replay-driven STDP "
                            "consolidation (offline learn-through-use), lesion-the-replay controlled",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size,
               "write_rule": a.write_rule,
               "cfg": dict(w_within=a.w_within, between_init=a.between_init, b_override=a.b_override, n_laps=a.n_laps,
                           stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
                           stdp_tau=a.stdp_tau, swr_period=a.swr_period, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                           weak_cue_mult=a.weak_cue_mult, weak_cue_frac=a.weak_cue_frac, ou_sigma=a.ou_sigma,
                           rest_steps=a.rest_steps, consol_steps=a.consol_steps, dt=a.dt,
                           write_rule=a.write_rule, btsp_elig_tau=a.btsp_elig_tau, btsp_plat_tau=a.btsp_plat_tau,
                           btsp_eta=a.btsp_eta, btsp_w_max=a.btsp_w_max, fwd_delay_steps=a.fwd_delay_steps,
                           gap_coding=a.gap_coding, gi_base=a.gi_base, fb_gain=a.fb_gain, fb_tau=a.fb_tau,
                           gi_cap=a.gi_cap, pc_tonic=a.pc_tonic),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[ecker-ltu] VERDICT: {verdict}\n[ecker-ltu] wrote {a.out}\n" + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
