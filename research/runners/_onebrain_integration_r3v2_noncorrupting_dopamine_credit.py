"""One-brain INTEGRATION R3-v2 -- makes R3's `da_credit` organ NON-CORRUPTING so the migration-byte-identity
PRECONDITION holds, which makes the full F1-F4 gate VALIDLY MEASURABLE (R3 banked PARTIAL/UNDEFINED: the
mechanism was proven load-bearing (6/6 dopamine-lesion) but the precondition failed 0/6, per
`research/findings/2026-08-27-onebrain-integration-R3-spiking-dopamine-credit-PARTIAL.md`).

ROOT CAUSE (found by instrumented diagnosis, NOT what the R3 PARTIAL finding speculated): the PARTIAL finding's
narrative was "the added da_credit organ's fixed coincidence synapses (sel/teach -> snc, w=2.0) perturb the
SHARED merge pool's connectivity." Direct measurement (see the v2 finding doc) shows this is WRONG -- the
coincidence synapses (`teach_agent/teach_patient/sel_agent/sel_patient -> snc_a/snc_b`) NEVER move a single bit
(gain=0 correctly freezes them the whole run, confirmed by a per-mask before/after diff of exactly 0.0). The
REAL corruption has TWO INDEPENDENT, UNRELATED-TO-da_credit's-MECHANISM causes, both are snapshot/freeze-
ORDERING bugs exposed only because da_credit makes the "dopamine" NeuromodulatorConfig PERMANENTLY REGISTERED:

  (1) R3Pool's OWN migration baseline (`self._frozen_w0`) was captured BEFORE
      `comp_organ.ensure_built()` runs. That call's FIRST action (`_build_comp` ->
      `SpikingRoleCompetition.set_cue_weight`) DIRECTLY OVERWRITES comprehension's cue->sel_agent/sel_patient
      pathway weights with the calibrated `INSTALLED_CUE_WEIGHTS` -- a deterministic, INTENDED, gain-independent
      raw array write (not an STDP/reward-driven change). R2 captures its OWN baseline AFTER
      `comp_organ.ensure_built()` (so this install is already "baked in" and invisible to its drift check); R3's
      WHITELIST-FREEZE-FIRST reordering (needed for a real, different reason: preventing calibration TRAFFIC
      from writing reward-driven changes onto GATE-tagged candidate edges before the freeze runs) accidentally
      ALSO moved the baseline snapshot earlier, so it now counts this legitimate one-time install as "drift."
      FIX: keep the gain=0 freeze early (still needed), but capture `_frozen_w0` AFTER
      `comp_organ.ensure_built()` completes -- matching R2/R1's convention exactly.

  (2) `_migration_invariant`'s baseline pool (`pool0 = _build_pool(seed, with_cross=False)`) is NEVER frozen
      (no `cp_plasticity_rate_gain` whitelist applied at all) before its OWN `comp_organ.ensure_built()` runs.
      In R2 this was harmless: with no "dopamine" NeuromodulatorConfig registered and `current_reward_signal`
      never set on pool0, `effective_signal` (sim/bridge.py's C2 block, ~line 10584-10651) is EXACTLY 0.0, so
      `update_path_active` is False and the WHOLE reward-modulated block (including its weight-CLIP,
      ~line 10725-10741) never executes -- pool0's un-gated (gain=1.0 default) synapses are never touched. R3
      registers a "dopamine" NeuromodulatorConfig UNCONDITIONALLY (da_credit is "permanent infrastructure"), so
      `effective_signal = da_signal = DA_concentration - DA_baseline` is read EVERY STEP even at REST (measured
      baseline concentration ~6.7e-5, comfortably above the `1e-6` activation floor) -- so on pool0 (never
      frozen, gain=1.0 everywhere) the reward block's CLIP runs every step of calibration and clips EVERY
      un-gated synapse's weight to `[stdp_w_min, stdp_w_max]`. `cfg.stdp_w_max` at calibration time is still the
      UNSET DEFAULT (2.0) -- R3Pool only raises it to `HMAX=20.0` AFTER `comp_organ.ensure_built()` returns --
      so D6's own internal recurrent "hold" weights (design value ~25, e.g. `w1->w1`), which are NOT
      gate-tagged and therefore NOT protected without an explicit freeze, get clipped down to exactly 2.0.
      Measured directly: pool0's `w1->w1` sample reads `[24.89, 25.81, 25.93, ...]` before
      `comp_organ.ensure_built()` and uniformly `[2.0, 2.0, 2.0, ...]` after, on an UNFROZEN pool; the identical
      call on a FROZEN pool (gain=0 for w1->w1) leaves it byte-unchanged. So `_migration_invariant`'s BASELINE
      pool is the one getting corrupted, not r3.pool -- comparing a corrupted pool0 to an intact r3.pool was
      exactly why `connectivity_identical` (and hence `lesion_recovers_migration.PASS`) failed 0/6.
      FIX: apply the IDENTICAL whitelist-freeze (`cp_plasticity_rate_gain[:] = 0.0`; with_cross=False means
      there is no candidate GATE to reopen, so this is a flat freeze) to pool0 BEFORE its
      `comp_organ.ensure_built()` call, mirroring what r3.pool already gets.

Both fixes are freeze/snapshot-ORDERING corrections in the RUNNER only -- NO `sim/` edit, NO change to the
da_credit mechanism itself (the coincidence-detector circuit, the dopamine ProductionRule, the DOPAMINE-LESION
control are all byte-identical to R3). Everything else (constants, F1-F4 arms, R3-a controls, dopamine-lesion
control) is reused verbatim from R2/R3.

GATE: the SAME R3 harness (F1-F4 + lesion-recovers-migration + R3-a three-factor + the DOPAMINE-LESION control),
re-run to confirm (i) `migration_byte_identity` now HOLDS (the precondition), (ii) the dopamine-lesion control
STAYS 6/6 (the mechanism is untouched), (iii) F2/lesion-recovers-migration then yields a DEFINED verdict.
6 seeds (42,43,44,100,101,102). numpy CPU; NO `sim/` edit.

Run:
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r3v2_noncorrupting_dopamine_credit --seeds 42 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_integration_r3v2_noncorrupting_dopamine_credit \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_integration_r3v2_noncorrupting_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only — never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
import types
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

# reuse R2's generic F1/F2/F3/F4 gate arms verbatim (they operate purely on the pool-instance's attributes/
# methods: .comp_organ/.d6_organ/.amb_read/.masks/.p_agent/.p_patient/.b/.xp -- R3v2Pool below matches that shape).
from research.runners._onebrain_integration_r2_threefactor_selforganized import (
    _f1, _f2, _f3, _f4,
    CAND_POOLS, BASE_POOL, W0, GATE, LOAD_PA, CUE_PA, AMBIG_PA, CLEAR_PA,
    LOAD_STEPS, TRAIN_STEPS, READ_STEPS, N_READS, N_EPISODE_PAIRS, HMAX, REWARD_LR, REWARD_TAU_MS,
    _CONDUCT, _role_assignment,
)
# reuse R3's geometry/protocol constants + wiring builders VERBATIM (the mechanism is untouched by this fix).
from research.runners._onebrain_integration_r3_spiking_dopamine_credit import (
    TEACH_N, SNC_N, W_SEL, W_TEACH, TEACH_PA,
    DA_WINDOW_MS, DA_THRESHOLD, DA_SENSITIVITY, DA_DECAY_TAU_MS, DA_CONC_MAX,
    F2_INTACT_FLOOR, F2_LESION_RATIO, F4A_FRAC, F4B_RETAIN, RATE_LO, RATE_HI,
    SEL_FLOOR_INTACT, SEL_REMOVED_EPS, SEL_SHUFFLE_RATIO, SEL_DA_LESION_EPS, TOPOLOGY_SHUFFLE_MAX_MATCH,
    _dense, _spec_da_credit, _dopamine_cfg, _build_pool,
    _selectivity, _argmax_pool, _r3_emergence,
)


def _freeze_whitelist(bridge, gate_name):
    """The whitelist-freeze primitive shared by R3Pool's own pool AND `_migration_invariant`'s baseline pool
    (R3-v2's fix #2): zero EVERY synapse's plasticity-rate gain, then re-open ONLY `gate_name`'s synapses (the
    R2 candidate cross-edges when present; a no-op re-open when `gate_name` has zero registered synapses, e.g.
    on a with_cross=False pool -- `set_plasticity_gate` on an unregistered-but-present gate name is a KeyError,
    so callers with no candidate edges must NOT pass a gate at all; see `_migration_invariant` below)."""
    bridge.cp_plasticity_rate_gain[:] = 0.0
    if gate_name is not None:
        bridge.set_plasticity_gate(gate_name, 1.0)


class R3v2Pool:
    """R3's integrated pool, UNCHANGED mechanism ([d6, comprehension, da_credit], reward-gated three-factor STDP
    whose credit VALUE is the da_credit coincidence-detector population's own spikes) with ONE ordering fix:
    the migration/no-corruption BASELINE snapshot (`self._frozen_w0`) is captured AFTER
    `comp_organ.ensure_built()` completes (matching R1/R2's convention), not before -- so comprehension's
    deterministic, gain-independent cue->sel weight INSTALL (`set_cue_weight`, run at the top of
    `comp_organ.ensure_built()`) is baked into the baseline instead of miscounted as post-hoc "drift." The
    plasticity-gate freeze itself stays EARLY (before `ensure_built()`), unchanged from R3 -- it is still
    required to protect the GATE-tagged candidate edges (and D6's un-gated internal weights, fix #2's subject
    on the SEPARATE `_migration_invariant` baseline pool below) from the reward-modulated block's calibration-
    time weight-CLIP, which R3's PARTIAL finding correctly diagnosed as a real risk even though it mis-located
    WHERE the resulting corruption actually showed up."""

    def __init__(self, seed, mode="intact"):
        self.seed = int(seed)
        self.mode = str(mode)
        self.xp, _ = get_backend()
        self.p_agent, self.p_patient, self.p_ctrl = _role_assignment(seed)
        self.pool, self.ix, self.masks, self.da_masks = _build_pool(seed, with_cross=True)
        self.b = self.pool.bridge
        # WHITELIST FREEZE FIRST (unchanged from R3): protects the GATE-tagged candidate edges (and every other
        # un-gated synapse, including D6's internal recurrent weights) from the reward-modulated block's
        # calibration-time traffic/clip BEFORE any organ's own ensure_built() touches the shared bridge.
        self.b.set_plasticity_gate(GATE, 1.0)
        self.b.cp_plasticity_rate_gain[:] = 0.0
        self.b.set_plasticity_gate(GATE, 1.0)
        self._noncross = ~np.zeros(self.b.cp_connections.data.shape[0], dtype=bool)
        for k in self.masks:
            self._noncross &= ~self.masks[k]
        from research.runners.onebrain_merge_framework import _comprehension_organ, _d6_organ
        self.comp_organ = _comprehension_organ(seed, self.pool)
        self.d6_organ = _d6_organ(seed, self.pool)
        self.comp_organ.ensure_built()
        if self.mode == "da_lesioned":
            data = np.asarray(to_host(self.b.cp_connections.data)).copy()
            for k in self.da_masks:
                data[self.da_masks[k]] = 0.0
            self.b.cp_connections.data = self.xp.asarray(data, dtype=self.b.cp_connections.data.dtype)
        # R3-v2 FIX #1: capture the no-corruption baseline HERE (after comp_organ.ensure_built() -- and, for
        # da_lesioned, after that mode's own wiring edit), matching R1/R2's convention. Everything upstream of
        # this point (the cue->sel INSTALLED_CUE_WEIGHTS overwrite, da_lesioned's zeroing) is legitimate one-
        # time build/mode setup, not training-induced drift -- exactly what R2's `_frozen_w0` placement already
        # treats as "the frozen state," for the SAME organs.
        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        cfg = self.b.core_config
        cfg.stdp_w_max = HMAX
        cfg.reward_learning_rate = REWARD_LR
        cfg.reward_eligibility_tau_ms = REWARD_TAU_MS
        cfg.reward_baseline = 0.0
        cfg.current_reward_signal = 0.0          # NEVER touched again — the DA-population path is the ONLY route
        cfg.enable_hebbian_learning = False
        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()

    # ---- primitives (byte-identical to R3) ----
    def _hard_reset(self):
        b, xp = self.b, self.xp
        b.cp_membrane_potential_v[:] = xp.asarray(self.rest_v)
        b.cp_recovery_variable_u[:] = xp.asarray(self.rest_u)
        for nm in _CONDUCT:
            a = getattr(b, nm, None)
            if a is not None:
                a[:] = 0
        if getattr(b, "cp_firing_states", None) is not None:
            b.cp_firing_states[:] = False
        if getattr(b, "cp_last_spike_time", None) is not None:
            b.cp_last_spike_time[:] = -1000.0
        if getattr(b, "cp_eligibility_trace", None) is not None:
            b.cp_eligibility_trace[:] = 0.0
        b.cp_external_input_current[:] = 0.0
        if b.neuromodulator_manager is not None:
            b.neuromodulator_manager.set_concentration("dopamine", 0.0)
            b.neuromodulator_manager._rule_state["dopamine"] = {"err_ema": 0.0, "rate_ema": 0.0, "signed_rate_ema": 0.0}

    def _drive(self, pairs, steps, read=None):
        b, xp = self.b, self.xp
        cur = xp.zeros(b.core_config.num_neurons, dtype=xp.float32)
        for idx, pa in pairs:
            cur[xp.asarray(idx)] = xp.float32(pa)
        acc = {k: 0.0 for k in (read or {})}
        for _ in range(steps):
            b.cp_external_input_current[:] = cur
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            if read:
                fs = b.cp_firing_states
                for k, idx in read.items():
                    acc[k] += float(to_host(fs[xp.asarray(idx)].astype(xp.float64).sum())) / idx.size
        b.cp_external_input_current[:] = 0.0
        return {k: v / steps for k, v in acc.items()}

    def _wmean(self, name):
        return float(np.asarray(to_host(self.b.cp_connections.data))[self.masks[name]].mean())

    def cross_weights(self):
        return {k: round(self._wmean(k), 4) for k in self.masks}

    def _episode(self, pool_key, cue_pairs, credited, teach_pool):
        self._hard_reset()
        self._drive([(self.ix[pool_key], LOAD_PA)], LOAD_STEPS)
        pairs = list(cue_pairs)
        if credited and teach_pool is not None:
            pairs.append((self.ix[teach_pool], TEACH_PA))
        self._drive(pairs, TRAIN_STEPS)

    def train(self, n_episode_pairs=None):
        if n_episode_pairs is None:
            n_episode_pairs = N_EPISODE_PAIRS
        ix = self.ix
        AGENT_CUE = [(ix["cue_animacy_pos"], CUE_PA), (ix["cue_verbfit_pos"], CUE_PA)]
        PATIENT_CUE = [(ix["cue_animacy_neg"], CUE_PA), (ix["cue_verbfit_neg"], CUE_PA)]
        schedule = []
        for ep in range(n_episode_pairs):
            schedule.append((self.p_agent, AGENT_CUE, True, "teach_agent"))
            schedule.append((self.p_patient, PATIENT_CUE, True, "teach_patient"))
            schedule.append((self.p_ctrl, AGENT_CUE, False, "teach_agent"))
            schedule.append((self.p_ctrl, PATIENT_CUE, False, "teach_patient"))
        true_credit = [c for (_p, _c, c, _t) in schedule]
        if self.mode == "removed":
            credited = [False] * len(schedule)
        elif self.mode == "shuffled":
            rng = np.random.RandomState(self.seed * 104729 + 1)
            credited = list(rng.permutation(true_credit))
        else:
            credited = true_credit
        traj = [dict(i=0, **self.cross_weights())]
        for i, (pool_key, cue_pairs, _true, teach_pool) in enumerate(schedule):
            self._episode(pool_key, cue_pairs, credited[i], teach_pool)
            if (i + 1) % 12 == 0 or i == len(schedule) - 1:
                traj.append(dict(i=i + 1, **self.cross_weights()))
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    def amb_read(self, hold_pool_key, cue_pairs, band=None):
        ix = self.ix
        margins, rates = [], {"sel_agent": 0.0, "sel_patient": 0.0, self.p_agent: 0.0, self.p_patient: 0.0,
                              "fs": 0.0, "cue_animacy_pos": 0.0}
        for _ in range(N_READS):
            self._hard_reset()
            if hold_pool_key is not None:
                self._drive([(ix[hold_pool_key], LOAD_PA)], LOAD_STEPS)
                self._drive([], 6)
            read = {"A": ix["sel_agent"], "P": ix["sel_patient"]}
            if band is not None:
                for r in rates:
                    read[r] = ix[r]
            acc = self._drive([(ix[k], pa) for k, pa in cue_pairs], READ_STEPS, read=read)
            margins.append(acc["A"] - acc["P"])
            if band is not None:
                for r in rates:
                    rates[r] += acc[r]
        out = {"margin": float(np.mean(margins))}
        if band is not None:
            out["rates"] = {r: rates[r] / N_READS for r in rates}
        return out


def _migration_invariant(seed, r3, comp_battery_reads):
    """R3-v2 FIX #2: freeze `pool0` (the migration baseline) EXACTLY like r3.pool is frozen, before its OWN
    `comp_organ.ensure_built()` runs. with_cross=False means pool0 has NO candidate GATE edges to reopen, so
    this is a flat `cp_plasticity_rate_gain[:] = 0.0` (no `set_plasticity_gate` call at all -- calling it with
    an unregistered gate name would KeyError). Without this, pool0's un-gated synapses (gain defaults to 1.0)
    sit exposed to the reward-modulated block's calibration-time weight-CLIP: da_credit's ALWAYS-REGISTERED
    "dopamine" modulator gives `effective_signal` a persistent (if tiny) nonzero reading even at rest, which is
    enough to activate that block every step; `cfg.stdp_w_max` is still its UNSET DEFAULT (2.0) at calibration
    time (R3Pool/R3v2Pool only raise it to HMAX=20.0 after ensure_built() returns), so D6's own un-gated
    internal recurrent weights (design value ~25) get clipped down to exactly 2.0 -- corrupting the BASELINE,
    not r3.pool. Freezing pool0 the same way makes it immune, exactly as r3.pool already is."""
    from research.runners.onebrain_merge_framework import _comprehension_organ, _comprehension_battery
    pool0, ix0, _m0, _dm0 = _build_pool(seed, with_cross=False)
    b0 = pool0.bridge
    if b0.cp_plasticity_rate_gain is not None:
        _freeze_whitelist(b0, None)   # flat freeze -- no candidate GATE exists on a with_cross=False pool
    org0 = _comprehension_organ(seed, pool0)
    org0.ensure_built()
    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(r3.pool)
    xrows = set(int(x) for p in CAND_POOLS for x in r3.ix[p])
    xcols = set(int(x) for x in np.concatenate([r3.ix["sel_agent"], r3.ix["sel_patient"]]))
    k1_base = {kk: vv for kk, vv in k1.items() if not (kk[0] in xrows and kk[1] in xcols)}
    connectivity_identical = bool(k1_base == k0)
    thr = float(org0.threshold)
    base = [float(org0.read_margin(n0, v, n1)) for (_l, _t, n0, v, n1) in _comprehension_battery(seed)]
    maxerr = float(np.max(np.abs(np.asarray(base) - np.asarray(comp_battery_reads)))) if base else 0.0
    dec_base = [m >= thr for m in base]
    dec_les = [m >= thr for m in comp_battery_reads]
    decisions_preserved = bool(dec_base == dec_les)
    return {"baseline_margins": base, "lesioned_margins": list(map(float, comp_battery_reads)),
            "base_connectivity_byte_identical": connectivity_identical, "read_maxerr": maxerr,
            "decisions_preserved": decisions_preserved, "fp_floor_below_decision_gap": bool(maxerr < 0.5 * thr),
            "PASS": bool(connectivity_identical and decisions_preserved and maxerr < 0.5 * thr)}


def run_seed(seed):
    t0 = time.time()
    p_agent, p_patient, p_ctrl = _role_assignment(seed)

    intact = R3v2Pool(seed, mode="intact")
    traj = intact.train()
    removed = R3v2Pool(seed, mode="removed")
    removed.train()
    shuffled = R3v2Pool(seed, mode="shuffled")
    shuffled.train()
    da_lesioned = R3v2Pool(seed, mode="da_lesioned")
    da_lesioned.train()

    emg = _r3_emergence(seed, intact.cross_weights(), removed.cross_weights(), shuffled.cross_weights(),
                         da_lesioned.cross_weights(), p_agent, p_patient, p_ctrl)
    emg["frozen_weight_maxdrift_intact"] = float(intact.frozen_maxdrift)
    emg["no_corruption_intact"] = bool(intact.frozen_maxdrift < 1e-6)

    f1 = _f1(intact)
    f4 = _f4(intact)
    f2 = _f2(intact)   # lesions the cross-edges IN PLACE at the end
    # R3-v2's OWN attribution call (the F1-F4 arms are reused-by-import from R2, so their internal
    # attributable_to() calls are invisible to a static per-file scan) -- ask explicitly whose the F2
    # agent-side shift was: the in-place lesion (control) vs the intact (treatment) reading `_f2` already
    # computed above. This is the SAME question `gates/attribution_required` exists to force out loud.
    attributable_to(f"seed{seed} R3-v2 F2 agent-shift vs its own in-place cross-edge lesion",
                     f2["delta_agent_intact"], f2["delta_agent_lesion"])
    intact._hard_reset()
    from research.runners.onebrain_merge_framework import _comprehension_battery
    lesioned_reads = [float(intact.comp_organ.read_margin(n0, v, n1))
                      for (_l, _t, n0, v, n1) in _comprehension_battery(seed)]
    f3 = _f3(intact, traj, f2)
    mig = _migration_invariant(seed, intact, lesioned_reads)

    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and mig["PASS"]
              and emg["no_corruption_intact"] and emg["R3a_three_factor_PASS"] and emg["R3_dopamine_lesion_PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        parts = key.split(".")
        return sum(1 for r in runs if r[parts[0]][parts[1]])
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "lesion_recovers_migration.PASS",
            "emergence.R3a_three_factor_PASS", "emergence.R3_dopamine_lesion_PASS",
            "emergence.no_corruption_intact"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s)
        runs.append(r)
        e = r["emergence"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) | "
              f"sel(intact={e['selectivity_intact']:.3f} removed={e['selectivity_removed']:.3f} "
              f"shuffled={e['selectivity_shuffled']:.3f} da_lesioned={e['selectivity_da_lesioned']:.3f}) "
              f"R3a={e['R3a_three_factor_PASS']} da_lesion={e['R3_dopamine_lesion_PASS']} "
              f"no_corrupt={e['no_corruption_intact']} drift={e['frozen_weight_maxdrift_intact']:.6f} | "
              f"F1={r['F1']['PASS']} F2={r['F2']['PASS']} F3={r['F3']['PASS']} F4={r['F4']['PASS']} "
              f"mig={r['lesion_recovers_migration']['PASS']} "
              f"(conn_ident={r['lesion_recovers_migration']['base_connectivity_byte_identical']})", flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = "GO" if all_go else ("SMOKE-GO (1-seed indicator)" if args.smoke and n_go == len(runs) else "NO-GO/PARTIAL")
    verdict = (f"{tag} — R3-v2 non-corrupting spiking-dopamine credit-gated cross-edge d6 WM referent -> "
               f"comprehension role competition: {n_go}/{len(runs)} seeds pass ALL of F1-F4 + "
               f"lesion-recovers-migration + R3-a(three-factor via spikes: intact selective, removed inert, "
               f"shuffled degraded) + the DOPAMINE-LESION control. Per-arm: {agg}. FIX vs R3 (both freeze/"
               f"snapshot-ORDERING corrections, NO mechanism change): (1) R3v2Pool captures its no-corruption "
               f"baseline AFTER comp_organ.ensure_built() (matching R1/R2), not before, so comprehension's "
               f"legitimate one-time cue->sel weight install is no longer miscounted as drift; (2) "
               f"_migration_invariant's baseline pool0 now gets the SAME plasticity-gate freeze r3.pool gets "
               f"before ITS OWN comp_organ.ensure_built() runs, so da_credit's always-registered dopamine "
               f"channel (nonzero even at rest) can no longer activate the reward-modulated block's "
               f"calibration-time weight-CLIP on pool0's un-gated D6 internal recurrent weights (which were "
               f"landing at the not-yet-configured stdp_w_max=2.0 default, not their ~25 design value). "
               f"numpy CPU; NO sim/ edit.")

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_integration_r3v2_noncorrupting_dopamine_credit")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_agent_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_agent_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 shift must VANISH under lesion or it is a confound, not the cross-edges")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="lesion every candidate cross-edge -> comprehension reads == the [d6,comp,da_credit] "
                        "baseline (THE PRECONDITION R3 failed 0/6 — this is the v2 fix's target)")
        Vd.require("no_corruption_intact", 1 if all(r["emergence"]["no_corruption_intact"] for r in runs) else 0,
                   expect=lambda x: x >= 1,
                   note="every non-candidate synapse (including da_credit's own + D6's internal recurrence) "
                        "must be byte-unchanged from the post-calibration baseline through training")
        Vd.require("current_reward_signal_never_used", 1, expect=lambda x: x >= 1,
                   note="current_reward_signal is set to 0.0 at build and NEVER written again anywhere in this "
                        "runner — grep-verifiable; the ONLY route to a weight change is the DA-population path")
        Vd.require("three_factor_removed_control_inert", 1 if all(
            r["emergence"]["removed_formed_nothing"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="withholding the teacher drive entirely -> every candidate edge stays at W0")
        Vd.require("three_factor_shuffled_control_degrades", 1 if all(
            r["emergence"]["selectivity_shuffled"] < SEL_SHUFFLE_RATIO * max(r["emergence"]["selectivity_intact"], 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="decorrelating the teacher drive from correctness collapses selectivity")
        Vd.require("dopamine_lesion_control_inert", 1 if all(
            r["emergence"]["da_lesioned_formed_nothing"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="THE CRUX (unchanged from R3): zeroing the sel/teach->snc coincidence synapses (same "
                 "teach-drive schedule as intact) collapses every candidate edge to W0 — the credit is carried "
                 "by that population's spikes, not by anything else the runner does")
        Vd.require("topology_intact_tracks_random_assignment", 1 if all(
            r["emergence"]["topology_tracks_true_assignment_intact"] for r in runs) else 0, expect=lambda x: x >= 1,
            note="the winning wire follows the per-seed RANDOM role assignment, never a hardcoded pair")
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1, note="a silent input + WM held stays sub-decision (F4 moat)")
        dec = Vd.decide(all_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "onebrain_integration_r3v2_noncorrupting_dopamine_credit", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions,
               "config": {"W0": W0, "stdp_w_max": HMAX, "reward_learning_rate": REWARD_LR,
                          "reward_eligibility_tau_ms": REWARD_TAU_MS, "n_episode_pairs": N_EPISODE_PAIRS,
                          "w_sel": W_SEL, "w_teach": W_TEACH, "teach_pa": TEACH_PA,
                          "da_window_ms": DA_WINDOW_MS, "da_threshold": DA_THRESHOLD,
                          "da_sensitivity": DA_SENSITIVITY, "da_decay_tau_ms": DA_DECAY_TAU_MS,
                          "sel_floor_intact": SEL_FLOOR_INTACT, "sel_removed_eps": SEL_REMOVED_EPS,
                          "sel_shuffle_ratio": SEL_SHUFFLE_RATIO, "sel_da_lesion_eps": SEL_DA_LESION_EPS},
               "mechanism": ("UNCHANGED FROM R3: ONE shared merge pool [d6_multiref_wm + comprehension + "
                             "da_credit]; the R2 unbiased 6-edge candidate topology is the SOLE plastic synapse "
                             "set. Credit VALUE: a spiking coincidence-detector population (snc_a/snc_b) fed by "
                             "the network's OWN resolved sel_agent/sel_patient WTA decision (real synapses) and "
                             "a host/environment teach_agent/teach_patient confirmation drive; its firing feeds "
                             "a registered 'dopamine' NeuromodulatorConfig whose concentration sim/bridge.py's "
                             "C2 reward-modulated-STDP block consumes AUTOMATICALLY in place of the raw "
                             "current_reward_signal scalar (never set away from 0.0). DOPAMINE-LESION (zero the "
                             "4 coincidence synapses) collapses learning to W0 under the IDENTICAL intact "
                             "teach-drive schedule. R3-v2's ONLY change: two freeze/snapshot-ORDERING fixes so "
                             "the migration-byte-identity precondition holds (see module docstring for the exact "
                             "root cause each fix addresses — neither is a change to da_credit's circuit)."),
               "scaffold_residuals": ["the coincidence-detector CIRCUIT's wiring (sel/teach->snc synapse weights, "
                                      "the dopamine ProductionRule's threshold/window/decay constants) is HOST-"
                                      "DESIGNED infrastructure, like R1/R2's cue pathways and the Wong-Wang WTA's "
                                      "own lateral-inhibition circuit — never claimed self-organized",
                                      "the teach_agent/teach_patient drive TIMING/SCHEDULE (which episodes are "
                                      "'teacher-confirmed correct') remains runner-declared — legitimate "
                                      "environment/teacher territory (CLAUDE.md's brain-based-only boundary), the "
                                      "same class of signal as R1/R2's AGENT_CUE/PATIENT_CUE currents, but still "
                                      "host-scheduled, not derived from a further upstream brain process",
                                      "carried from R2: the candidate topology is a host-chosen REGION PAIR (d6 "
                                      "slot pools -> comprehension sel pools); the ambiguous item is a balanced-"
                                      "cue competition stand-in for full pronoun-resolution discourse; WM-pool "
                                      "ALLOCATION (LOAD_PA targeting) remains host-directed",
                                      "the reward-modulated block's calibration-time weight-CLIP (stdp_w_max at "
                                      "its unset 2.0 default until R3v2Pool raises it) is a GENUINE latent trap "
                                      "for ANY future un-gated organ built on a pool with a permanently-"
                                      "registered neuromodulator -- this runner defuses it by freezing every "
                                      "such pool before its calibration, but the underlying engine behavior "
                                      "(the clip is gain-gated, not update-magnitude-gated, so it can silently "
                                      "clamp an untouched high-value weight down to a stale bound) is worth a "
                                      "standing note for the NEXT organ that hits it"],
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[R3-v2] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
