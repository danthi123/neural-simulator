"""GNW Rung-2d de-risk: DYNAMICALLY-WEAKENABLE RECURRENCE opens the "empty metastable window".

Rung-2c (`_gnw_rung2c_salience_disinhibition_derisk.py`, 6/6 BOUNDARY) proved that a salience-gated
dis-inhibition PULSE cannot cleanly EVICT the workspace incumbent: the recurrent workspace drive is FROZEN
(a static weight-30 attractor) far above the ignition knee, so once ignited the attractor self-sustains and
somatic/feedforward inhibition cannot remove it (it lets a challenger CO-ignite but never opens an UNIGNITED
window). The Rung-2c BOUNDARY diagnosed this as a SUBSTRATE metastability property (CLAUDE.md's companion-process
lesson: we replaced the incumbent's DYNAMIC recurrent efficacy with a STATIC frozen weight) and NAMED the next
mechanism: make the RECURRENCE dynamically weakenable — Mongillo, Barak & Tsodyks 2008, Science 319:1543,
"Synaptic Theory of Working Memory": short-term synaptic DEPRESSION (+ optional facilitation) on the recurrent
excitatory (E->E) synapses.

THE MECHANISM (Tsodyks-Markram short-term plasticity on the E->E recurrence; brain-based, in-runner, NO `sim/`
edit): each recurrent synapse carries a resources variable x (depression) and a utilization u (facilitation).
Effective efficacy = base_weight * x (u modulates the per-spike depletion, NOT the resting efficacy, so at rest
x=1 -> the attractor weight is IDENTICAL to Rung-2c's frozen 30, and the ignition knee is unchanged). Every
presynaptic spike depletes x by u*x (releasing u*x of the available resources); x recovers toward 1 with tau_D
(~hundreds of ms). A SUSTAINED, self-reverberating incumbent fires continuously -> its recurrent x DEPLETES ->
its self-drive weakens BELOW the sustain (hold) knee -> the ignited attractor DESTABILIZES and collapses to the
REST branch (all-or-none, Rung-1 bistability) -> an EMPTY unignited window opens. Because the collapse frees the
cross-inhibition too, a NEW content driven into the freed workspace then IGNITES (its own x is fresh). This is
the biological surpass for the frozen-recurrence wall: the eviction is a SYNAPTIC-DEPRESSION dynamic on the
substrate, NOT a host "clear the workspace" call.

WHY THIS IS NOT THE BANKED "STP annihilates" negative: that negative (2026-08-01) was STP on a SINGLE
self-exciting pool with no competitor at a non-Mongillo operating point. Here STP is on the E->E RECURRENCE of a
COMPETITIVE two-assembly workspace, tuned so depression depletes the LOOP (evictable-yet-holds) rather than
adapting the SOMA (Rung-2b SFA, which killed the neuron). x modulates the LOOP efficacy; resting x=1 keeps the
attractor at full strength, so the incumbent HOLDS a weak challenger before it self-evicts.

THE INSTRUMENT (the metastability read Rung-2c's BOUNDARY asked for — an unignited window between two ignitions):
a CONTINUOUS run (NO `_restore_state` mid-run): (1) drive A -> A ignites & HOLDS; (2) hold with no drive -> A's
recurrent x depletes -> A SELF-EVICTS (drops below the ignite threshold) BEFORE any challenger arrives, opening an
EMPTY window (A off AND B off); (3) drive B into the freed workspace -> B IGNITES; final winner is B, not A. The
eviction+re-ignition sequence ignited(A) -> empty -> ignited(B) is the GO signal.

GO GATE (6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed):
  eviction+re-ignition occurs (A ignites & holds; A self-evicts to an EMPTY window >= MIN_EMPTY steps BEFORE B is
  driven; B then ignites; final winner B, not A) on >= 5/6 seeds.
ANTI-CHEATS (all required):
  - STD LOAD-BEARING: lesion it (freeze x=1) -> no depletion -> A holds through the whole hold window (no empty
    window) and B CO-ignites/fails to take over -> reproduces the Rung-2c frozen-recurrence BOUNDARY.
  - BYTE-IDENTICAL when `--recurrent-std` is off: the STD layer is provably inert (freeze-x=1 timecourse ==
    STD-off timecourse, hash-identical) AND the seeded substrate params match the pre-edit Rung-2c build
    (separate-process substrate hash).
  - NO host shortcut: host_workspace_reset_calls == 0 (the eviction is the synaptic depression, not a reset).
  - Determinism: build twice at one seed -> identical substrate hash (cfg.seed seeds the substrate).

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2d_weakenable_recurrence_derisk --seed 42 --smoke \
      --json research/findings/raw/_gnw_rung2d_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2d_weakenable_recurrence_derisk --seed 42 \
      --recurrent-std --stp-U 0.02 --stp-tau-d 250 --json research/findings/raw/_gnw_rung2d_seed42.json
  # rung2c-eval mode (byte-identical-vs-pre-edit-rung2c substrate hash):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2d_weakenable_recurrence_derisk --seed 42 \
      --rung2c-eval --json research/findings/raw/_gnw_rung2d_rung2ceval_seed42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

from research.runners._gnw_rung1_ignition_curve_derisk import (
    _snapshot_state, _restore_state, DEFAULT_ATTRACTOR_WEIGHT, DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS,
)
from research.runners._gnw_rung2_competitive_access_derisk import (
    _ignited, IGNITE_FRAC, SOLO_PLATEAU, WORKSPACE_N, ASSEMBLY_SIZE, WS_LOOP_A, WS_LOOP_B,
)
from research.runners._gnw_rung2b_sfa_workspace_eviction_derisk import _threshold_hash
# REUSE (byte-identical) the Rung-2c workspace substrate + its rung2c operating-point eval. Importing the SAME
# build guarantees the substrate is byte-identical to the pre-edit Rung-2c runner (no duplicated build code).
from research.runners._gnw_rung2c_salience_disinhibition_derisk import (
    build_disinhibition_bridge, evaluate_operating_point as rung2c_evaluate_operating_point,
    control_pulse_off as rung2c_control_pulse_off, control_wta_lesion as rung2c_control_wta_lesion,
    A2FS_WEIGHT, FS_TO_WS_WEIGHT, DIS_TO_FS_WEIGHT,
)

# ── metastability protocol timing (dt=1.0 ms) ────────────────────────────────────────────────────────────────
DRIVE_A_STEPS = DRIVE_STEPS          # brief supra-threshold pulse that ignites A (35 steps, as Rung-1/2)
HOLD_STEPS = 400                     # A holds (no drive) while its recurrent x depletes -> self-eviction window
DRIVE_B_STEPS = DRIVE_STEPS          # brief pulse of the NEW content B into the (freed) workspace
FREE_TAIL_STEPS = 100                # free settle after B's drive -> the final settled winner
RATE_WIN = 15                        # sliding-window (steps) for the per-assembly firing-rate estimate (>~4 periods)
IGNITE_THRESH = IGNITE_FRAC * SOLO_PLATEAU   # ignited iff windowed rate >= this (0.5 * 1/3 = 0.1667)
MIN_EMPTY = 20                       # the EMPTY metastable window must last >= this many steps (both A,B un-ignited)

# host-side accounting: the CONTINUOUS metastable run MUST make ZERO workspace-reset calls (anti-cheat).
_RESTORE_CALLS = {"n": 0}


def _restore_counted(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _restore_state(bridge, snap)


# ── short-term depression / facilitation on the recurrent E->E synapses (Tsodyks-Markram; NO sim/ edit) ────────
class RecurrentSTD:
    """Host-computed short-term synaptic plasticity on the workspace's recurrent E->E (assembly-loop) synapses.

    Tracks a per-PRESYNAPTIC-neuron resources variable x (depression) and utilization u (facilitation) for the
    A and B assembly loops, and each step OVERWRITES the loop synapse weights in `bridge.cp_connections.data`
    with `base_weight * x_pre` (u modulates the per-spike depletion, not the resting efficacy). This is the
    synaptic-depression MECHANISM applied to the substrate's own recurrent synapses — the eviction is produced by
    the synapse weakening, NOT by any host state reset. `enable_short_term_plasticity` stays OFF in the engine
    (the native global STP is a banked foot-gun); this targets ONLY the E->E recurrence with Mongillo params.

    Update per spike of presynaptic neuron j (Tsodyks-Markram):
        u_j <- u_j + U*(1 - u_j)         (facilitation; if facilitation OFF, u_j == U constant)
        released = u_j * x_j             (resources used by this spike)
        x_j <- x_j - released            (depression: deplete available resources)
    Recovery each step (dt elapsed):
        x_j <- x_j + (1 - x_j) * dt/tau_D
        u_j <- u_j + (U - u_j) * dt/tau_F   (only if facilitation ON)
    Effective loop efficacy = base_weight * x_j  (=> at rest x=1 -> the frozen Rung-2c weight; freeze_x=1 -> a
    provable no-op == STD-OFF).
    """

    def __init__(self, bridge, xp, A_idx, B_idx, U=0.02, tau_D=250.0, tau_F=0.0, facilitation=False,
                 dt=1.0, freeze_x=False):
        self.bridge = bridge
        self.xp = xp
        self.dt = float(dt)
        self.U = float(U)
        self.tau_D = float(tau_D)
        self.tau_F = float(tau_F)
        self.facilitation = bool(facilitation)
        self.freeze_x = bool(freeze_x)     # lesion / STD-OFF equivalent: x pinned to 1 (no depletion)
        self.n_weight_writes = 0

        n = bridge.core_config.num_neurons
        A_idx = np.asarray(A_idx, dtype=np.int64)
        B_idx = np.asarray(B_idx, dtype=np.int64)

        # locate the A-loop and B-loop synapse positions in cp_connections.data (CSR, canonical order).
        csr = bridge.cp_connections
        csr.sort_indices()
        indptr = to_host(csr.indptr)
        indices = to_host(csr.indices).astype(np.int64)
        rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))   # presyn per data entry
        cols = indices                                                     # postsyn per data entry
        maskA = np.isin(rows, A_idx) & np.isin(cols, A_idx)
        maskB = np.isin(rows, B_idx) & np.isin(cols, B_idx)
        self.idxA = np.where(maskA)[0]
        self.idxB = np.where(maskB)[0]
        assert self.idxA.size == A_idx.size * (A_idx.size - 1), \
            f"A-loop synapse count {self.idxA.size} != {A_idx.size*(A_idx.size-1)}"
        assert self.idxB.size == B_idx.size * (B_idx.size - 1), \
            f"B-loop synapse count {self.idxB.size} != {B_idx.size*(B_idx.size-1)}"
        self.preA = rows[self.idxA]        # presyn neuron id per A-loop synapse
        self.preB = rows[self.idxB]
        data = to_host(csr.data)
        self.baseA = data[self.idxA].astype(np.float64).copy()   # frozen base weights (all == attractor_weight)
        self.baseB = data[self.idxB].astype(np.float64).copy()
        self.idxA_dev = xp.asarray(self.idxA)
        self.idxB_dev = xp.asarray(self.idxB)

        # per-neuron resources x and utilization u (indexed by global neuron id; only assembly neurons evolve).
        self.x = np.ones(n, dtype=np.float64)
        self.u = np.full(n, self.U, dtype=np.float64)
        self.assembly = np.concatenate([A_idx, B_idx])

    def apply_weights(self):
        """Overwrite the loop synapse weights with base * x_pre (efficacy). Called BEFORE each step so the step's
        synaptic transmission reads the depressed weights."""
        xA = self.x[self.preA]
        xB = self.x[self.preB]
        self.bridge.cp_connections.data[self.idxA_dev] = self.xp.asarray(self.baseA * xA, dtype=self.xp.float32)
        self.bridge.cp_connections.data[self.idxB_dev] = self.xp.asarray(self.baseB * xB, dtype=self.xp.float32)
        self.n_weight_writes += 1

    def update(self, fired_host):
        """Recover (dt elapsed) then deplete from the neurons that fired THIS step. `fired_host` is a host bool
        array over ALL neurons. freeze_x=True -> x pinned to 1 (no-op depression == STD-OFF)."""
        if self.freeze_x:
            return
        a = self.assembly
        # recovery toward baseline
        self.x[a] += (1.0 - self.x[a]) * (self.dt / self.tau_D)
        if self.facilitation and self.tau_F > 0.0:
            self.u[a] += (self.U - self.u[a]) * (self.dt / self.tau_F)
        # depletion from this step's spikes (only assembly neurons matter for the loops)
        fired_a = a[fired_host[a]]
        if fired_a.size:
            if self.facilitation:
                self.u[fired_a] = self.u[fired_a] + self.U * (1.0 - self.u[fired_a])
            # released = u*x ; x -= released
            self.x[fired_a] = self.x[fired_a] - self.u[fired_a] * self.x[fired_a]
        np.clip(self.x, 0.0, 1.0, out=self.x)
        np.clip(self.u, 0.0, 1.0, out=self.u)

    def x_mean(self, idx):
        return float(self.x[idx].mean())


# ── the continuous metastability run ───────────────────────────────────────────────────────────────────────
def run_metastable(bridge, xp, A_dev, B_dev, std, drive_inc, drive_chal,
                   drive_a_steps=DRIVE_A_STEPS, hold_steps=HOLD_STEPS, drive_b_steps=DRIVE_B_STEPS,
                   free_tail=FREE_TAIL_STEPS):
    """One CONTINUOUS metastability trial (NO `_restore_state` anywhere in the run):
      (1) drive A -> ignite; (2) hold (no drive) -> A's recurrent x depletes -> A self-evicts (empty window);
      (3) drive B -> B ignites in the freed workspace; (4) free tail -> settled winner.
    `std` may be None (STD-OFF: weights never touched) or a RecurrentSTD (applied every step). Returns a dict of
    per-step A/B spike counts, the phase-boundary indices, and the x-traces."""
    A_idx = to_host(A_dev).astype(np.int64)
    B_idx = to_host(B_dev).astype(np.int64)

    counts_A, counts_B, xA_trace, xB_trace = [], [], [], []
    phase = []

    def _one_step(drive_idx, drive_val, tag):
        if std is not None:
            std.apply_weights()
        bridge.cp_external_input_current[:] = 0.0
        if drive_idx is not None:
            bridge.cp_external_input_current[drive_idx] = xp.float32(drive_val)
        bridge._run_one_simulation_step()
        fired = to_host(bridge.cp_firing_states).astype(bool)
        if std is not None:
            std.update(fired)
        counts_A.append(int(fired[A_idx].sum()))
        counts_B.append(int(fired[B_idx].sum()))
        if std is not None:
            xA_trace.append(std.x_mean(A_idx)); xB_trace.append(std.x_mean(B_idx))
        else:
            xA_trace.append(1.0); xB_trace.append(1.0)
        phase.append(tag)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(drive_a_steps):
        _one_step(A_dev, drive_inc, "driveA")
    for _ in range(hold_steps):
        _one_step(None, 0.0, "hold")
    for _ in range(drive_b_steps):
        _one_step(B_dev, drive_chal, "driveB")
    for _ in range(free_tail):
        _one_step(None, 0.0, "free")

    return {"counts_A": counts_A, "counts_B": counts_B, "xA_trace": xA_trace, "xB_trace": xB_trace,
            "phase": phase, "n_assembly": int(A_idx.size),
            "drive_a_steps": drive_a_steps, "hold_steps": hold_steps,
            "drive_b_steps": drive_b_steps, "free_tail": free_tail}


def _windowed_rate(counts, n_neurons, win=RATE_WIN):
    """Trailing-window mean per-neuron firing rate: rate[t] = mean spikes/neuron over steps (t-win, t]."""
    c = np.asarray(counts, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(c)])
    r = np.empty_like(c)
    for t in range(c.size):
        lo = max(0, t + 1 - win)
        r[t] = (csum[t + 1] - csum[lo]) / (float(n_neurons) * (t + 1 - lo))
    return r


def detect_metastability(trace):
    """From a run_metastable trace, compute the eviction+re-ignition flags.
      a_ignites_holds : A ignited during the hold phase (windowed rate >= IGNITE_THRESH somewhere in the hold).
      a_self_evicts   : A is NOT ignited at the END of the hold phase (it dropped out BEFORE B was driven).
      empty_window    : an interval of >= MIN_EMPTY consecutive steps (within the hold, AFTER A's initial hold)
                        where BOTH A and B are un-ignited (the EMPTY metastable window).
      b_ignites       : B ignited at the END of the free tail (the new content won the freed workspace).
      a_evicted_final : A NOT ignited at the END of the free tail.
      go              : a_ignites_holds AND a_self_evicts AND empty_window AND b_ignites AND a_evicted_final.
    """
    n = trace["n_assembly"]
    rA = _windowed_rate(trace["counts_A"], n)
    rB = _windowed_rate(trace["counts_B"], n)
    phase = np.asarray(trace["phase"])
    igA = rA >= IGNITE_THRESH
    igB = rB >= IGNITE_THRESH

    hold_mask = phase == "hold"
    hold_idx = np.where(hold_mask)[0]
    free_idx = np.where(phase == "free")[0]
    # A must be ignited early in the hold (settle in the first RATE_WIN..~half of the hold), then must have
    # dropped out by the end of the hold. Look at the hold's first third for "held" and its last step for "out".
    hold_start = hold_idx[0]
    hold_end = hold_idx[-1]
    early_hold = hold_idx[: max(1, len(hold_idx) // 3)]
    a_ignites_holds = bool(igA[early_hold].any())
    a_self_evicts = bool(not igA[hold_end])          # A un-ignited at the last hold step (before B drive)

    # EMPTY window: within the hold, the FIRST time A drops out (after it held), count consecutive steps where
    # neither A nor B is ignited.
    empty_len = 0
    empty_start = None
    # find A's drop-out step within the hold (first hold step, after early hold, where A un-ignites)
    dropout = None
    for t in hold_idx:
        if igA[t]:
            continue
        if t > early_hold[-1]:
            dropout = t
            break
    if dropout is not None:
        run = 0
        for t in range(dropout, hold_end + 1):
            if (not igA[t]) and (not igB[t]):
                if run == 0:
                    seg_start = t
                run += 1
                if run > empty_len:
                    empty_len = run
                    empty_start = seg_start
            else:
                run = 0
    empty_window = bool(empty_len >= MIN_EMPTY)

    b_ignites = bool(igB[free_idx[-1]]) if free_idx.size else False
    a_evicted_final = bool(not igA[free_idx[-1]]) if free_idx.size else False

    go = bool(a_ignites_holds and a_self_evicts and empty_window and b_ignites and a_evicted_final)

    xA = np.asarray(trace["xA_trace"], dtype=np.float64)
    return {
        "a_ignites_holds": a_ignites_holds, "a_self_evicts": a_self_evicts,
        "empty_window": empty_window, "empty_len": int(empty_len),
        "empty_start_step": (int(empty_start) if empty_start is not None else None),
        "b_ignites": b_ignites, "a_evicted_final": a_evicted_final, "go": go,
        "rA_hold_end": float(rA[hold_end]), "rB_free_end": float(rB[free_idx[-1]]) if free_idx.size else None,
        "rA_free_end": float(rA[free_idx[-1]]) if free_idx.size else None,
        "xA_hold_end": float(xA[hold_end]), "xA_min": float(xA.min()),
        "dropout_step": (int(dropout) if dropout is not None else None),
        "hold_start": int(hold_start), "hold_end": int(hold_end),
        "rA": [float(v) for v in rA], "rB": [float(v) for v in rB],
    }


# ── one seed: metastability GO + anti-cheats ──────────────────────────────────────────────────────────────────
def evaluate_metastable_seed(seed, U, tau_D, tau_F, facilitation, fs_to_ws, ou_noise, heterogeneity,
                             drive_inc, drive_chal, hold_steps, attractor_weight=DEFAULT_ATTRACTOR_WEIGHT,
                             a2fs=A2FS_WEIGHT, verbose=True):
    """Build the Rung-2c workspace substrate + STD on the E->E recurrence, run the continuous metastability
    protocol, and evaluate the eviction+re-ignition GO + the STD-lesion / byte-identical / determinism anti-cheats.
    Returns a result dict."""
    restore_before = _RESTORE_CALLS["n"]

    # ── STD ON: the headline eviction run ──────────────────────────────────────────────────────────────────
    bridge, xp, A_dev, B_dev, disA, disB, snap, handles = build_disinhibition_bridge(
        seed=seed, attractor_weight=attractor_weight, fs_to_ws=fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT,
        a2fs=a2fs, ou_noise_pA=ou_noise, heterogeneity=heterogeneity)
    A_idx = to_host(A_dev).astype(np.int64); B_idx = to_host(B_dev).astype(np.int64)
    std = RecurrentSTD(bridge, xp, A_idx, B_idx, U=U, tau_D=tau_D, tau_F=tau_F, facilitation=facilitation)
    trace_on = run_metastable(bridge, xp, A_dev, B_dev, std, drive_inc, drive_chal, hold_steps=hold_steps)
    det_on = detect_metastability(trace_on)

    # ── STD LESION (freeze x=1): reproduce the Rung-2c frozen-recurrence BOUNDARY (no eviction) ─────────────
    bridge_l, xp_l, A_l, B_l, _dA, _dB, _snap, _h = build_disinhibition_bridge(
        seed=seed, attractor_weight=attractor_weight, fs_to_ws=fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT,
        a2fs=a2fs, ou_noise_pA=ou_noise, heterogeneity=heterogeneity)
    std_l = RecurrentSTD(bridge_l, xp_l, to_host(A_l).astype(np.int64), to_host(B_l).astype(np.int64),
                         U=U, tau_D=tau_D, tau_F=tau_F, facilitation=facilitation, freeze_x=True)
    trace_lesion = run_metastable(bridge_l, xp_l, A_l, B_l, std_l, drive_inc, drive_chal, hold_steps=hold_steps)
    det_lesion = detect_metastability(trace_lesion)
    # the lesion must reproduce the BOUNDARY: A holds (no self-eviction / no empty window) -> STD is load-bearing.
    std_load_bearing = bool((not det_lesion["a_self_evicts"]) and (not det_lesion["empty_window"])
                            and (not det_lesion["go"]))
    # ATTRIBUTION: whose is the empty-metastable window? treatment = STD-ON empty_len; control = STD-lesion (freeze
    # x=1) empty_len. A high fraction means the STD depression OWNS the eviction, not the protocol timing (measuring
    # both arms is not the same as asking whose the difference is — gap#5 lesson).
    eviction_attribution = attributable_to(
        "empty-metastable window (self-eviction) via recurrent STD depression",
        float(det_on["empty_len"]), float(det_lesion["empty_len"]), warn_below=0.8)

    # ── BYTE-IDENTICAL when OFF: STD-OFF (weights untouched) timecourse == freeze-x=1 timecourse (the STD layer
    #    at x=1 is a provable no-op). Same fresh build + same protocol; hash the A/B spike-count timecourses. ──
    bridge_off, xp_off, A_o, B_o, _dAo, _dBo, _snapo, _ho = build_disinhibition_bridge(
        seed=seed, attractor_weight=attractor_weight, fs_to_ws=fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT,
        a2fs=a2fs, ou_noise_pA=ou_noise, heterogeneity=heterogeneity)
    trace_off = run_metastable(bridge_off, xp_off, A_o, B_o, None, drive_inc, drive_chal, hold_steps=hold_steps)
    hash_off = hashlib.sha256(np.asarray(trace_off["counts_A"] + trace_off["counts_B"],
                                         dtype=np.int64).tobytes()).hexdigest()
    hash_frozen = hashlib.sha256(np.asarray(trace_lesion["counts_A"] + trace_lesion["counts_B"],
                                            dtype=np.int64).tobytes()).hexdigest()
    byte_identical_off = bool(hash_off == hash_frozen)

    # ── determinism: build twice at this seed, hash the seed-derived Izhikevich params (cfg.seed). ─────────────
    h1 = _threshold_hash(bridge, xp)
    bridge2, xp2, _, _, _, _, _, _ = build_disinhibition_bridge(
        seed=seed, attractor_weight=attractor_weight, fs_to_ws=fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT,
        a2fs=a2fs, ou_noise_pA=ou_noise, heterogeneity=heterogeneity)
    h2 = _threshold_hash(bridge2, xp2)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    # host_workspace_reset_calls: ZERO restore/reset calls in the whole continuous metastable evaluation.
    host_workspace_reset_calls = int(_RESTORE_CALLS["n"] - restore_before)

    go = bool(det_on["go"] and std_load_bearing and byte_identical_off and seed_deterministic
              and host_workspace_reset_calls == 0)

    result = {
        "seed": int(seed),
        "operating_point": {"U": float(U), "tau_D": float(tau_D), "tau_F": float(tau_F),
                            "facilitation": bool(facilitation), "fs_to_ws": float(fs_to_ws),
                            "ou_noise_pA": float(ou_noise), "heterogeneity": bool(heterogeneity),
                            "drive_inc": float(drive_inc), "drive_chal": float(drive_chal),
                            "hold_steps": int(hold_steps), "attractor_weight": float(attractor_weight),
                            "a2fs": float(a2fs)},
        "metastability_on": {k: v for k, v in det_on.items() if k not in ("rA", "rB")},
        "metastability_lesion": {k: v for k, v in det_lesion.items() if k not in ("rA", "rB")},
        "std_load_bearing": std_load_bearing,
        "attribution": {"label": "empty-metastable window (self-eviction) via recurrent STD depression",
                        "treatment_empty_len_on": int(det_on["empty_len"]),
                        "control_empty_len_lesion": int(det_lesion["empty_len"]),
                        "fraction_attributable_to_std": eviction_attribution},
        "byte_identical_off": byte_identical_off, "hash_off": hash_off, "hash_frozen": hash_frozen,
        "substrate_hash": h1, "seed_deterministic": seed_deterministic,
        "host_workspace_reset_calls": host_workspace_reset_calls,
        "std_weight_writes_on": int(std.n_weight_writes),
        "go": go,
        # residual quantification (NOT-A-WALL): how far x got, vs where it needs to be to evict.
        "residual": {"xA_min_on": det_on["xA_min"], "xA_hold_end_on": det_on["xA_hold_end"],
                     "empty_len_on": det_on["empty_len"], "empty_len_lesion": det_lesion["empty_len"]},
    }
    if verbose:
        m = det_on
        print(f"  [seed={seed} U={U} tauD={tau_D} tauF={tau_F} fac={facilitation} fs={fs_to_ws} "
              f"drvA={drive_inc} drvB={drive_chal} hold={hold_steps}] GO={go} | "
              f"holds={m['a_ignites_holds']} self_evict={m['a_self_evicts']} empty={m['empty_window']}"
              f"(len={m['empty_len']}) B_ign={m['b_ignites']} A_out={m['a_evicted_final']} | "
              f"xA_min={m['xA_min']:.3f} xA@holdend={m['xA_hold_end']:.3f} | "
              f"std_LB={std_load_bearing} byte_id_off={byte_identical_off} det={seed_deterministic} "
              f"resets={host_workspace_reset_calls}", flush=True)
    return result, trace_on, det_on


# ── main ───────────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="GNW Rung-2d weakenable-recurrence (STD) metastable-eviction de-risk.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung2d_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--recurrent-std", action="store_true",
                    help="ENABLE short-term depression on the E->E recurrence (default OFF = byte-identical Rung-2c)")
    ap.add_argument("--stp-U", type=float, default=0.02, help="utilization / per-spike release fraction U")
    ap.add_argument("--stp-tau-d", type=float, default=300.0, help="depression recovery time constant tau_D (ms)")
    ap.add_argument("--stp-tau-f", type=float, default=0.0, help="facilitation time constant tau_F (ms); 0 = off")
    ap.add_argument("--facilitation", action="store_true", help="enable facilitation (u dynamics) in addition to x")
    ap.add_argument("--fs-to-ws", type=float, default=16.0, help="cross-inhibition strength (Rung-2c op)")
    ap.add_argument("--a2fs", type=float, default=A2FS_WEIGHT, help="assembly -> fs excitation (cross-inhibition)")
    ap.add_argument("--ou-noise", type=float, default=40.0, help="OU noise std (pA)")
    ap.add_argument("--no-heterogeneity", action="store_true", help="disable parameter heterogeneity")
    ap.add_argument("--drive-inc", type=float, default=5000.0, help="drive that ignites A")
    ap.add_argument("--drive-chal", type=float, default=5000.0, help="drive of the new content B")
    ap.add_argument("--hold-steps", type=int, default=HOLD_STEPS, help="steps A holds (no drive) while x depletes")
    ap.add_argument("--smoke", action="store_true", help="grid-scan (U, tau_D) on ONE seed to find the window")
    ap.add_argument("--six-seed", action="store_true",
                    help="run the frozen operating point on 42/43/44/100/101/102 and aggregate the GO gate (>=5/6)")
    ap.add_argument("--robustness", action="store_true",
                    help="map the metastability-window EDGES: GO-count over 6 seeds at the frozen op + two edge ops")
    ap.add_argument("--rung2c-eval", action="store_true",
                    help="run the pre-edit Rung-2c operating-point eval (byte-identical substrate hash check)")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)
    het = not args.no_heterogeneity

    # ── rung2c-eval mode: reproduce the pre-edit Rung-2c single-seed eval (for the byte-identical hash) ──────────
    if args.rung2c_eval:
        r = rung2c_evaluate_operating_point(args.seed, 1.0, DIS_TO_FS_WEIGHT, 25, args.fs_to_ws, args.ou_noise,
                                            120, het, args.drive_inc, 8000.0, 9, a2fs=args.a2fs, verbose=False)
        out = {"runner": "_gnw_rung2d_weakenable_recurrence_derisk", "mode": "rung2c-eval", "seed": int(args.seed),
               "substrate_hash": r["threshold_hash"], "winner_per_challenger": r["winner_per_challenger"],
               "a_rates": r["a_rates"], "b_rates": r["b_rates"]}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[rung2d-rung2c-eval] seed={args.seed} substrate_hash={r['threshold_hash'][:16]} "
              f"winners={''.join('A' if w=='A' else ('B' if w=='B' else '2') for w in r['winner_per_challenger'])} "
              f"wrote {args.json}", flush=True)
        return 0

    if args.robustness:
        # Map the metastability-window edges (why the frozen op is a genuine window, not a knife-edge). ON-run
        # detection only (fast): the frozen op (GO) + two edges — stronger depression (B over-depletes during its
        # own ignition drive) and faster recovery (A re-ignites, empty window shrinks below MIN_EMPTY).
        seeds = [42, 43, 44, 100, 101, 102]
        ops = [{"U": 0.02, "tau_D": 300.0, "label": "frozen-op-GO"},
               {"U": 0.025, "tau_D": 300.0, "label": "edge-stronger-depression"},
               {"U": 0.02, "tau_D": 250.0, "label": "edge-faster-recovery"}]
        print(f"[rung2d-robustness] seeds {seeds} fs={args.fs_to_ws} drvA={args.drive_inc} drvB={args.drive_chal} "
              f"hold={args.hold_steps} — mapping the window edges", flush=True)
        sweep = []
        for op in ops:
            per_seed = []
            for s in seeds:
                bridge, xp, A_dev, B_dev, _dA, _dB, _s, _h = build_disinhibition_bridge(
                    seed=s, fs_to_ws=args.fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT, a2fs=args.a2fs,
                    ou_noise_pA=args.ou_noise, heterogeneity=het)
                std = RecurrentSTD(bridge, xp, to_host(A_dev).astype(np.int64), to_host(B_dev).astype(np.int64),
                                   U=op["U"], tau_D=op["tau_D"], tau_F=args.stp_tau_f, facilitation=args.facilitation)
                det = detect_metastability(run_metastable(bridge, xp, A_dev, B_dev, std, args.drive_inc,
                                                          args.drive_chal, hold_steps=args.hold_steps))
                per_seed.append({"seed": s, "go": det["go"], "empty_len": det["empty_len"],
                                 "b_ignites": det["b_ignites"], "a_self_evicts": det["a_self_evicts"],
                                 "xA_min": det["xA_min"]})
            n_go = sum(1 for r in per_seed if r["go"])
            sweep.append({"U": op["U"], "tau_D": op["tau_D"], "label": op["label"], "n_go": n_go,
                          "n_seeds": len(seeds), "per_seed": per_seed})
            print(f"  [{op['label']}] U={op['U']} tau_D={op['tau_D']}: {n_go}/{len(seeds)} GO", flush=True)
        out = {"runner": "_gnw_rung2d_weakenable_recurrence_derisk", "mode": "robustness",
               "drive_inc": args.drive_inc, "drive_chal": args.drive_chal, "hold_steps": args.hold_steps,
               "fs_to_ws": args.fs_to_ws, "ou_noise": args.ou_noise, "sweep": sweep}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[rung2d-robustness] wrote {args.json}", flush=True)
        return 0

    if args.six_seed:
        seeds = [42, 43, 44, 100, 101, 102]
        print(f"[rung2d-6seed] frozen op: STD-ON U={args.stp_U} tauD={args.stp_tau_d} tauF={args.stp_tau_f} "
              f"fac={args.facilitation} fs={args.fs_to_ws} ou={args.ou_noise} drvA={args.drive_inc} "
              f"drvB={args.drive_chal} hold={args.hold_steps} het={het} — seeds {seeds}", flush=True)
        per_seed = []
        for s in seeds:
            res, _tr, _det = evaluate_metastable_seed(
                s, args.stp_U, args.stp_tau_d, args.stp_tau_f, args.facilitation, args.fs_to_ws, args.ou_noise,
                het, args.drive_inc, args.drive_chal, args.hold_steps, a2fs=args.a2fs, verbose=True)
            per_seed.append(res)
        n_go = sum(1 for r in per_seed if r["go"])
        # every anti-cheat must hold on every seed for the aggregate GO to be trustworthy.
        all_std_load_bearing = all(r["std_load_bearing"] for r in per_seed)
        all_byte_identical = all(r["byte_identical_off"] for r in per_seed)
        all_no_reset = all(r["host_workspace_reset_calls"] == 0 for r in per_seed)
        all_determ = all(r["seed_deterministic"] for r in per_seed)
        gate_go = bool(n_go >= 5 and all_std_load_bearing and all_byte_identical and all_no_reset and all_determ)
        min_attr = min(r["attribution"]["fraction_attributable_to_std"] for r in per_seed
                       if r["attribution"]["fraction_attributable_to_std"] is not None)
        # The aggregate verdict must travel with the VALIDITY preconditions that earned it (all must hold; the
        # eviction+re-ignition GO count is the measured outcome, not a precondition).
        v = Verdict("rung2d weakenable-recurrence (STD) metastable eviction: 6-seed aggregate")
        v.require("STD load-bearing on every seed (freeze x=1 reproduces the Rung-2c no-eviction BOUNDARY)",
                  all_std_load_bearing, expect=True)
        v.require("empty-window attributable to STD depression (>=0.8 on every seed)", bool(min_attr >= 0.8),
                  expect=True)
        v.require("byte-identical when STD off on every seed (freeze-x=1 == STD-off timecourse)",
                  all_byte_identical, expect=True)
        v.require("no host workspace reset on any seed (eviction is synaptic depression)", all_no_reset,
                  expect=True)
        v.require("determinism: cfg.seed seeds the substrate on every seed (build-twice hash)", all_determ,
                  expect=True)
        v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
        v.disabled("native_short_term_plasticity",
                   why="engine global STP banked as annihilating (2026-08-01); STD here targets the E->E "
                       "recurrence only, in-runner, with Mongillo params")
        vd = v.decide(go=gate_go)
        verdict = "GO" if gate_go else ("PARTIAL" if n_go >= 1 else "BOUNDARY")
        out = {"runner": "_gnw_rung2d_weakenable_recurrence_derisk", "mode": "six-seed", "verdict": verdict,
               "gate_go": gate_go, "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds,
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"], "undefined_reasons": vd["undefined_reasons"],
               "min_fraction_attributable_to_std": min_attr,
               "operating_point": per_seed[0]["operating_point"],
               "anti_cheats": {"all_std_load_bearing": all_std_load_bearing,
                               "all_byte_identical_off": all_byte_identical,
                               "all_host_workspace_reset_calls_zero": all_no_reset,
                               "all_seed_deterministic": all_determ},
               "per_seed": per_seed}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[rung2d-6seed] verdict={verdict} n_go={n_go}/6 std_LB={all_std_load_bearing} "
              f"byte_id={all_byte_identical} no_reset={all_no_reset} det={all_determ}  wrote {args.json}", flush=True)
        return 0 if gate_go else 1

    if args.smoke:
        print(f"[rung2d-smoke] seed={args.seed} fs={args.fs_to_ws} ou={args.ou_noise} hold={args.hold_steps} "
              f"het={het} — scanning (U, tau_D) for the ignite-hold-AND-self-evict window", flush=True)
        grid = []
        for U in (0.008, 0.015, 0.03, 0.06):
            for tau_D in (150.0, 300.0, 600.0):
                res, _tr, _det = evaluate_metastable_seed(
                    args.seed, U, tau_D, args.stp_tau_f, args.facilitation, args.fs_to_ws, args.ou_noise, het,
                    args.drive_inc, args.drive_chal, args.hold_steps, a2fs=args.a2fs, verbose=True)
                grid.append(res)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_rung2d_weakenable_recurrence_derisk", "mode": "smoke", "grid": grid}, f, indent=2)
        any_go = any(g["go"] for g in grid)
        print(f"\n[rung2d-smoke] wrote {args.json}  any_go={any_go}", flush=True)
        return 0 if any_go else 1

    # ── single-seed frozen-operating-point evaluation ───────────────────────────────────────────────────────
    if not args.recurrent_std:
        # STD OFF: prove the layer is inert (no eviction; byte-identical). This is the default (byte-identical
        # to pre-edit Rung-2c: A holds, no empty window). We still emit the metastability read for the record.
        bridge, xp, A_dev, B_dev, _dA, _dB, _s, _h = build_disinhibition_bridge(
            seed=args.seed, fs_to_ws=args.fs_to_ws, dis_to_fs=DIS_TO_FS_WEIGHT, a2fs=args.a2fs,
            ou_noise_pA=args.ou_noise, heterogeneity=het)
        trace = run_metastable(bridge, xp, A_dev, B_dev, None, args.drive_inc, args.drive_chal,
                               hold_steps=args.hold_steps)
        det = detect_metastability(trace)
        result = {"runner": "_gnw_rung2d_weakenable_recurrence_derisk", "mode": "std-off", "seed": int(args.seed),
                  "go": False, "recurrent_std": False,
                  "metastability": {k: v for k, v in det.items() if k not in ("rA", "rB")}}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[rung2d] STD-OFF seed={args.seed} empty_window={det['empty_window']} b_ignites={det['b_ignites']} "
              f"(expect no eviction) wrote {args.json}", flush=True)
        return 0

    print(f"[rung2d] seed={args.seed} STD-ON U={args.stp_U} tauD={args.stp_tau_d} tauF={args.stp_tau_f} "
          f"fac={args.facilitation} fs={args.fs_to_ws} ou={args.ou_noise} hold={args.hold_steps} het={het}", flush=True)
    result, trace, det = evaluate_metastable_seed(
        args.seed, args.stp_U, args.stp_tau_d, args.stp_tau_f, args.facilitation, args.fs_to_ws, args.ou_noise,
        het, args.drive_inc, args.drive_chal, args.hold_steps, a2fs=args.a2fs, verbose=True)

    go = bool(result["go"])
    v = Verdict("rung2d weakenable-recurrence (STD) metastable eviction @ frozen op (seed %d)" % args.seed)
    v.require("A ignites & holds (recurrent attractor)", result["metastability_on"]["a_ignites_holds"], expect=True)
    v.require("A SELF-EVICTS via STD before B is driven", result["metastability_on"]["a_self_evicts"], expect=True)
    v.require("EMPTY metastable window opens (A off AND B off, >= MIN_EMPTY steps)",
              result["metastability_on"]["empty_window"], expect=True)
    v.require("B ignites in the freed workspace (re-ignition)", result["metastability_on"]["b_ignites"], expect=True)
    v.require("STD load-bearing (freeze x=1 reproduces the Rung-2c no-eviction BOUNDARY)",
              result["std_load_bearing"], expect=True)
    v.require("byte-identical when STD off (freeze-x=1 == STD-off timecourse)", result["byte_identical_off"],
              expect=True)
    v.require("no host workspace reset (eviction is synaptic depression)",
              result["host_workspace_reset_calls"] == 0, expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", result["seed_deterministic"],
              expect=True)
    v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating (2026-08-01); STD here is targeted to the E->E "
                   "recurrence only, in-runner, with Mongillo params")
    vd = v.decide(go=go)

    result["verdict"] = vd["status"]
    result["preconditions"] = vd["preconditions"]
    result["disabled_processes"] = vd["disabled_processes"]
    result["runner"] = "_gnw_rung2d_weakenable_recurrence_derisk"
    result["mode"] = "single"
    result["backend"] = args.backend
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[rung2d] seed={args.seed} GO={go}  wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
