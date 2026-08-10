"""A5 REAL-SUBSTRATE de-risk — NE / locus-coeruleus MULTIPLICATIVE gain sharpens
weak-signal detection, delivered by the sim's NEUROMODULATOR SUBSYSTEM (not a host multiply).

PROMOTES the idealized 200-LIF probe (research/runners/_ne_lc_gain_vigilance_derisk.py;
finding 2026-08-10-parallel-push-results-...-NE-gain-positive-..., section 2) to the
REAL SimulationBridge. The idealized probe multiplied the afferent drive in a private numpy
loop; here the gain is applied by `NeuromodulatorManager.compute_synaptic_gain_multiplier()`
scaling `effective_synaptic_strength` (the actual weight matrix) inside
`sim/bridge.py:8167` — the SAME hook a real NE modulator uses. That is the brain-based
requirement that makes this a real-substrate test rather than a re-run of the probe.

BIOLOGY: Aston-Jones & Cohen 2005 ("An integrative theory of locus coeruleus-norepinephrine
function", Annu. Rev. Neurosci. 28:403-450) — LC-NE multiplicatively scales target GAIN to
improve SNR / signal detection during vigilance.

TASK (weak-signal detection on the real substrate):
  * INPUT population (n_in Izhikevich RS) driven by a constant external current so it fires
    regularly -> provides the afferent SYNAPTIC drive to the target via input->target E->E
    synapses. On SIGNAL-PRESENT trials the input drive is raised by a small `sig_drive` pA
    (the weak signal). OU background noise is MASKED OFF on the input pop so its synaptic
    output is clean/near-constant trial-to-trial.
  * TARGET population (n_tgt Izhikevich RS) receives ONLY the input->target synapses PLUS the
    substrate's own OU background-noise process (cfg.enable_ou_process, cp_ou_neuron_mask
    restricts OU to the target pop). The OU noise is the FIXED intrinsic spike-generation
    floor -- added SEPARATELY from synaptic current (sim/bridge.py:7581, ou_current is summed
    AFTER the synaptic matvec), so `synaptic_gain` does NOT scale it. That asymmetry (gain
    scales the signal-carrying synaptic drive, NOT the noise floor) is exactly why a
    multiplicative gain lifts d' where an additive DC does not.
  * READOUT: target population spike COUNT per trial. d-prime between signal-present and
    signal-absent trials, computed from the ACTUAL spike counts.

NE GAIN via the subsystem: an "NE" NeuromodulatorConfig with a single
ModulatorTarget(target_type="synaptic_gain", scope="all", sensitivity=1.0), baseline=0.
gain multiplier = 1 + 1.0*(conc - 0) = 1 + conc, so set_concentration(g-1) yields gain g.
decay_tau_ms is set huge so the manually-set concentration holds across the trial.

MANDATORY CONTROLS (teeth):
  (1) BYTE-IDENTICAL WHEN OFF. gain=1.0 (NE conc==baseline -> multiplier==1.0, guarded no-op
      at bridge.py:8168) must reproduce a subsystem-DISABLED bridge bit-for-bit: identical
      per-trial target spike counts (hashed). Proves g=1.0 is a true no-op.
  (2) MULTIPLICATIVE not ADDITIVE. A rate-matched ADDITIVE offset -- delivered by the SAME
      subsystem as ModulatorTarget(target_type="excitability_drive", scope="group:target")
      (adds DC pA to the target pop, bridge.py:8488) -- tuned to reproduce the g=2.0 mean
      target rate must NOT reproduce the d' lift. At a matched operating point only the
      multiplicative gain amplifies signal-vs-noise CONTRAST against the fixed OU floor.
  (3) d' from the actual spike counts, signal vs noise trials (not a proxy).

GO = d' rises monotonically with gain on >=5/6 seeds AND multiplicative beats the
rate-matched additive by a margin (mult d'(g=2) - add d' > MARGIN) on >=5/6 seeds.

Run (GPU, 1-seed smoke to find the operating point):
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._ne_lc_gain_vigilance_realbridge_derisk \
      --smoke --out research/findings/raw/lanes/ne_realbridge_smoke.json
Run (GPU, 6-seed GO):
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._ne_lc_gain_vigilance_realbridge_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/lanes/ne_realbridge_6seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from tools.lab import attributable_to


# ----------------------------------------------------------------------------
# Bridge construction
# ----------------------------------------------------------------------------

def build_bridge(seed, n_in, n_tgt, weight, ou_std, enable_nm):
    """Build a 2-population spiking net: input -> target E->E, OU noise on target only.

    enable_nm=True declares the NE (synaptic_gain) + ADD (excitability_drive group:target)
    modulators and enables the neuromodulator subsystem. enable_nm=False is the
    byte-identical baseline (no subsystem at all).
    """
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from sim.backend import get_backend
    cp, _ = get_backend()

    N = n_in + n_tgt
    cfg = CoreSimConfig()
    cfg.num_neurons = N
    cfg.connections_per_neuron = 0            # inject_explicit_wiring overwrites connectivity
    cfg.dt_ms = 1.0
    cfg.seed = seed                            # ⛔ MUST set cfg.seed (actual_seed_used seeds NOTHING)
    # Freeze all plasticity so the swept gain is the only thing that changes transmission.
    cfg.enable_hebbian_learning = False
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_short_term_plasticity = False
    # OU background is the fixed intrinsic noise floor (masked to the target pop below).
    cfg.enable_ou_process = True
    cfg.ou_mean_current_pA = 0.0
    cfg.ou_std_current_pA = float(ou_std)
    cfg.ou_tau_ms = 15.0
    cfg.enable_conductance_noise = False       # HH-only; off for a clean Izhikevich RNG stream

    if enable_nm:
        from sim.neuromodulators import (
            NeuromodulatorConfig, ModulatorTarget, ProductionRule,
        )
        cfg.enable_neuromodulator_subsystem = True
        cfg.neuromodulators = [
            # NE = Aston-Jones-Cohen multiplicative gain: synaptic_gain scope=all.
            #   multiplier = 1 + 1.0*(conc - 0) = 1 + conc  ->  set_concentration(g-1) => gain g.
            NeuromodulatorConfig(
                name="NE", baseline=0.0, decay_tau_ms=1e12,
                concentration_min=0.0, concentration_max=20.0,
                targets=[ModulatorTarget(target_type="synaptic_gain", scope="all",
                                         sensitivity=1.0)],
                production_rules=[],           # manual: concentration set externally
            ),
            # ADD = the additive control: excitability_drive scope=group:target adds DC pA
            #   to the target pop only. value = 1.0*(conc - 0) = conc (pA).
            NeuromodulatorConfig(
                name="ADD", baseline=0.0, decay_tau_ms=1e12,
                concentration_min=0.0, concentration_max=1.0e6,
                targets=[ModulatorTarget(target_type="excitability_drive",
                                         scope="group:target", sensitivity=1.0)],
                production_rules=[],
            ),
        ]

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert bridge.is_initialized

    # Dense input -> target excitatory synapses (the gain-scaled synaptic drive).
    pre, post, w = [], [], []
    for i in range(n_in):
        for t in range(n_in, N):
            pre.append(i)
            post.append(t)
            w.append(weight)
    bridge.inject_explicit_wiring({
        "in2tgt": {
            "pre_indices": pre,
            "post_indices": post,
            "initial_weights": np.asarray(w, dtype=np.float32),
            "plastic": False,
            "conn_type": "E_TO_E",
        },
    })

    # OU background noise ONLY on the target pop -> input stays clean/deterministic.
    ou_mask = cp.zeros(N, dtype=cp.bool_)
    ou_mask[n_in:] = True
    bridge.cp_ou_neuron_mask = ou_mask

    if enable_nm:
        bridge.neuromodulator_manager.set_group_indices({"target": list(range(n_in, N))})

    return bridge, cp


# ----------------------------------------------------------------------------
# State snapshot / restore (identical initial conditions across gain levels)
# ----------------------------------------------------------------------------

# Every per-neuron state array that EVOLVES during a step. cp_neuron_firing_thresholds is
# ADAPTIVE (threshold homeostasis) and cp_neuron_activity_ema drives it -- if not reset,
# trials contaminate each other via the adapted threshold (measured 2026-08-10: this was the
# sole cause of a <=1-spike ON-vs-OFF byte-ident residual). Reset ALL of them so every trial
# starts from the identical pristine state -> trials are independent and byte-ident is exact.
_RESET_ARRAYS = (
    "cp_membrane_potential_v",
    "cp_recovery_variable_u",
    "cp_conductance_g_e",
    "cp_conductance_g_i",
    "cp_refractory_timers",
    "cp_ou_current",
    "cp_last_spike_time",
    "cp_neuron_activity_ema",
    "cp_neuron_firing_thresholds",
    "cp_viz_activity_timers",
)


def snapshot_state(bridge):
    s = {}
    for name in _RESET_ARRAYS:
        arr = getattr(bridge, name, None)
        s[name] = None if arr is None else arr.copy()
    return s


def restore_state(bridge, s):
    for name, arr in s.items():
        if arr is not None:
            getattr(bridge, name)[:] = arr


# ----------------------------------------------------------------------------
# One trial
# ----------------------------------------------------------------------------

def _trial_seed(base_seed, present, trial_idx):
    # distinct present/absent samples, but paired across gain levels for a fair d'.
    return int(base_seed) * 1_000_000 + (100_000 if present else 0) + int(trial_idx)


def run_trial(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive, present,
              base_seed, trial_idx, T):
    """One trial: reset state, re-seed OU RNG (paired across gains), drive, step, count."""
    cp.random.seed(_trial_seed(base_seed, present, trial_idx))
    restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0
    drive = float(base_drive) + (float(sig_drive) if present else 0.0)
    bridge.cp_external_input_current[in_dev] = cp.float32(drive)

    spikes_dev = cp.zeros((), dtype=cp.float64)
    for _ in range(T):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        spikes_dev += cp.sum(bridge.cp_firing_states[tgt_dev].astype(cp.float64))
    return float(spikes_dev.get())


def dprime(present, absent):
    present = np.asarray(present, dtype=np.float64)
    absent = np.asarray(absent, dtype=np.float64)
    mp, ma = present.mean(), absent.mean()
    vp, va = present.var(ddof=1), absent.var(ddof=1)
    denom = math.sqrt(0.5 * (vp + va))
    if denom < 1e-12:
        return float("nan")
    return float((mp - ma) / denom)


def collect_counts(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive,
                   base_seed, n_trials, T, set_gain=None, set_add=None):
    """Run n_trials present + n_trials absent; return (present_counts, absent_counts).

    set_gain: NE concentration to set each trial (gain = 1 + conc). None => leave as is.
    set_add:  ADD concentration (pA additive DC on target) to set each trial. None => leave.
    """
    mgr = getattr(bridge, "neuromodulator_manager", None)
    present, absent = [], []
    for present_flag, bucket in ((True, present), (False, absent)):
        for k in range(n_trials):
            if mgr is not None:
                if set_gain is not None:
                    mgr.set_concentration("NE", float(set_gain))
                if set_add is not None:
                    mgr.set_concentration("ADD", float(set_add))
            c = run_trial(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive,
                          present_flag, base_seed, k, T)
            bucket.append(c)
    return np.asarray(present, np.float64), np.asarray(absent, np.float64)


def _mean_rate(present, absent):
    return 0.5 * (float(np.mean(present)) + float(np.mean(absent)))


def match_additive_to_rate(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive,
                           base_seed, target_rate, T, n_search_trials, iters=8,
                           lo=0.0, hi=None):
    """Bisect the ADD excitability_drive (pA on target) so mean target rate ~= target_rate."""
    if hi is None:
        hi = 2000.0
    # widen hi until it overshoots (or give up)
    for _ in range(6):
        p, a = collect_counts(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive,
                              base_seed, n_search_trials, T, set_gain=0.0, set_add=hi)
        if _mean_rate(p, a) >= target_rate:
            break
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p, a = collect_counts(bridge, cp, snap, in_dev, tgt_dev, base_drive, sig_drive,
                              base_seed, n_search_trials, T, set_gain=0.0, set_add=mid)
        if _mean_rate(p, a) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------------
# One seed
# ----------------------------------------------------------------------------

def run_seed(seed, args):
    from sim.backend import to_host  # noqa: F401  (kept for parity w/ other runners)
    gains = [float(g) for g in args.gains]

    # ON bridge (subsystem enabled) — used for the gain sweep + additive control.
    b_on, cp = build_bridge(seed, args.n_in, args.n_tgt, args.weight, args.ou_std,
                            enable_nm=True)
    in_dev = cp.asarray(list(range(args.n_in)), dtype=cp.int64)
    tgt_dev = cp.asarray(list(range(args.n_in, args.n_in + args.n_tgt)), dtype=cp.int64)
    snap_on = snapshot_state(b_on)

    # ---- gain sweep ----
    sweep = {}
    counts_by_gain = {}
    for g in gains:
        p, a = collect_counts(b_on, cp, snap_on, in_dev, tgt_dev, args.base_drive,
                              args.sig_drive, seed, args.n_trials, args.T,
                              set_gain=g - 1.0, set_add=0.0)
        d = dprime(p, a)
        r = _mean_rate(p, a)
        sweep[g] = {"dprime": d, "mean_rate": r,
                    "mean_present": float(np.mean(p)), "mean_absent": float(np.mean(a))}
        counts_by_gain[g] = (p, a)

    g_lo = min(gains)
    g_hi = max(gains)
    d_lo = sweep[g_lo]["dprime"]
    d_hi = sweep[g_hi]["dprime"]

    # monotone non-decreasing d' across the sorted gain sweep
    ds = [sweep[g]["dprime"] for g in sorted(gains)]
    monotone = all(ds[i] <= ds[i + 1] + args.mono_tol for i in range(len(ds) - 1))

    # ---- CONTROL 2: rate-matched additive offset (excitability_drive on target) ----
    target_rate = sweep[g_hi]["mean_rate"]
    off = match_additive_to_rate(b_on, cp, snap_on, in_dev, tgt_dev, args.base_drive,
                                 args.sig_drive, seed, target_rate, args.T,
                                 max(20, args.n_trials // 3), iters=args.match_iters)
    p_add, a_add = collect_counts(b_on, cp, snap_on, in_dev, tgt_dev, args.base_drive,
                                  args.sig_drive, seed, args.n_trials, args.T,
                                  set_gain=0.0, set_add=off)
    d_add = dprime(p_add, a_add)
    r_add = _mean_rate(p_add, a_add)
    mult_beats_add = (d_hi - d_add) > args.add_margin

    # ---- CONTROL 1: byte-identical when off (gain=1.0 == subsystem disabled) ----
    b_off, cp2 = build_bridge(seed, args.n_in, args.n_tgt, args.weight, args.ou_std,
                              enable_nm=False)
    in_dev2 = cp2.asarray(list(range(args.n_in)), dtype=cp2.int64)
    tgt_dev2 = cp2.asarray(list(range(args.n_in, args.n_in + args.n_tgt)), dtype=cp2.int64)
    snap_off = snapshot_state(b_off)
    n_bi = min(args.n_trials, args.byte_ident_trials)
    p_off, a_off = collect_counts(b_off, cp2, snap_off, in_dev2, tgt_dev2, args.base_drive,
                                  args.sig_drive, seed, n_bi, args.T,
                                  set_gain=None, set_add=None)
    # matched gain=1.0 counts on the ON bridge (recompute at n_bi for a fair hash compare)
    p_on1, a_on1 = collect_counts(b_on, cp, snap_on, in_dev, tgt_dev, args.base_drive,
                                  args.sig_drive, seed, n_bi, args.T,
                                  set_gain=0.0, set_add=0.0)
    off_vec = np.concatenate([p_off, a_off])
    on_vec = np.concatenate([p_on1, a_on1])
    byte_ident = bool(np.array_equal(off_vec, on_vec))
    h_off = hashlib.sha256(off_vec.tobytes()).hexdigest()[:16]
    h_on = hashlib.sha256(on_vec.tobytes()).hexdigest()[:16]
    max_abs_diff = float(np.max(np.abs(off_vec - on_vec))) if off_vec.size else 0.0

    # attribution: the d' lift attributable to the NE multiplicative gain (high vs off).
    attrib = attributable_to(
        "NE multiplicative gain d-prime: gain-high vs gain-off", d_hi, d_lo, warn_below=-1.0)

    seed_go = bool(monotone and mult_beats_add and (d_hi - d_lo) > args.lift_margin)

    return {
        "seed": seed,
        "gains": gains,
        "sweep": {str(g): sweep[g] for g in gains},
        "dprime_low": d_lo,
        "dprime_high": d_hi,
        "dprime_lift_high_minus_low": float(d_hi - d_lo),
        "monotone": bool(monotone),
        "attributable_gain_lift": attrib,
        "additive_control": {
            "matched_offset_pA": float(off),
            "dprime_additive": d_add,
            "mean_rate_additive": r_add,
            "mean_rate_gain_high": target_rate,
            "mult_minus_add_dprime": float(d_hi - d_add),
            "mult_beats_add": bool(mult_beats_add),
        },
        "byte_identical_control": {
            "byte_identical": byte_ident,
            "n_trials_compared": int(n_bi),
            "hash_off": h_off,
            "hash_on_gain1": h_on,
            "max_abs_count_diff": max_abs_diff,
        },
        "GO": seed_go,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="1 seed (42), fewer trials")
    ap.add_argument("--gains", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--n-in", dest="n_in", type=int, default=40)
    ap.add_argument("--n-tgt", dest="n_tgt", type=int, default=160)
    # operating point chosen 2026-08-10 by a weight/drive/ou sweep for a clean monotone d'
    # rise with headroom (rates 96->207 across g=1..2.5, un-saturated): w=3, bd=320, sd=18, ou=60.
    ap.add_argument("--weight", type=float, default=3.0, help="input->target synaptic weight")
    ap.add_argument("--base-drive", dest="base_drive", type=float, default=320.0,
                    help="constant external current (pA) to the input pop (background)")
    ap.add_argument("--sig-drive", dest="sig_drive", type=float, default=18.0,
                    help="extra input-pop current (pA) on signal-present trials (weak signal)")
    ap.add_argument("--ou-std", dest="ou_std", type=float, default=60.0,
                    help="OU noise std (pA) on the target pop = the fixed spike-gen floor")
    ap.add_argument("--T", type=int, default=120, help="steps per trial")
    ap.add_argument("--n-trials", dest="n_trials", type=int, default=120,
                    help="trials per condition (present and absent each)")
    ap.add_argument("--byte-ident-trials", dest="byte_ident_trials", type=int, default=40)
    ap.add_argument("--match-iters", dest="match_iters", type=int, default=8)
    ap.add_argument("--mono-tol", dest="mono_tol", type=float, default=0.05,
                    help="slack allowed in the monotone-d' check")
    ap.add_argument("--lift-margin", dest="lift_margin", type=float, default=0.30,
                    help="min d'(g_hi) - d'(g_lo) for a seed GO")
    ap.add_argument("--add-margin", dest="add_margin", type=float, default=0.30,
                    help="min d'(g_hi) - d'(additive rate-matched) for a seed GO")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [42]
        args.n_trials = min(args.n_trials, 60)
        args.byte_ident_trials = min(args.byte_ident_trials, 30)

    t0 = time.time()
    print("=" * 78)
    print("A5 NE/LC multiplicative-gain vigilance — REAL SUBSTRATE (neuromodulator subsystem)")
    print(f"  n_in={args.n_in} n_tgt={args.n_tgt} weight={args.weight} base_drive={args.base_drive} "
          f"sig_drive={args.sig_drive} ou_std={args.ou_std} T={args.T} n_trials={args.n_trials}")
    print(f"  gains={args.gains}  seeds={args.seeds}")
    print("=" * 78, flush=True)

    per_seed = []
    for s in args.seeds:
        r = run_seed(s, args)
        per_seed.append(r)
        sw = r["sweep"]
        gstr = "  ".join(f"g={g}:d'={sw[str(g)]['dprime']:.3f}(r={sw[str(g)]['mean_rate']:.0f})"
                         for g in r["gains"])
        ac = r["additive_control"]
        bi = r["byte_identical_control"]
        print(f"[seed {s}] {gstr}")
        print(f"   monotone={r['monotone']}  lift(hi-lo)={r['dprime_lift_high_minus_low']:.3f}")
        print(f"   ADDITIVE rate-matched: offset={ac['matched_offset_pA']:.1f}pA "
              f"d'_add={ac['dprime_additive']:.3f}  mult-add={ac['mult_minus_add_dprime']:.3f} "
              f"beats_add={ac['mult_beats_add']}")
        print(f"   BYTE-IDENT g=1.0 vs subsystem-off: {bi['byte_identical']} "
              f"(max|diff|={bi['max_abs_count_diff']:.0f}, hoff={bi['hash_off']} hon={bi['hash_on_gain1']})")
        print(f"   seed GO={r['GO']}", flush=True)

    n_go = sum(1 for r in per_seed if r["GO"])
    n_mono = sum(1 for r in per_seed if r["monotone"])
    n_beats = sum(1 for r in per_seed if r["additive_control"]["mult_beats_add"])
    n_bi = sum(1 for r in per_seed if r["byte_identical_control"]["byte_identical"])
    overall_go = bool(n_go >= 5 and n_bi == len(args.seeds))

    overall = {
        "runner": "_ne_lc_gain_vigilance_realbridge_derisk",
        "biology": "Aston-Jones & Cohen 2005 LC-NE adaptive gain (Annu.Rev.Neurosci.28:403-450)",
        "gain_delivery": "sim NeuromodulatorManager synaptic_gain scope=all (bridge.py:8167) — "
                         "REAL substrate, not a host multiply",
        "additive_control_delivery": "sim NeuromodulatorManager excitability_drive scope=group:target "
                                     "(bridge.py:8488)",
        "noise_floor": "cfg.enable_ou_process OU background on target pop only (cp_ou_neuron_mask); "
                       "summed AFTER synaptic matvec so synaptic_gain does not scale it",
        "params": {
            "n_in": args.n_in, "n_tgt": args.n_tgt, "weight": args.weight,
            "base_drive": args.base_drive, "sig_drive": args.sig_drive,
            "ou_std": args.ou_std, "T": args.T, "n_trials": args.n_trials,
            "gains": args.gains,
        },
        "seeds": args.seeds,
        "per_seed": per_seed,
        "n_seeds": len(args.seeds),
        "n_go": n_go,
        "n_monotone": n_mono,
        "n_mult_beats_add": n_beats,
        "n_byte_identical": n_bi,
        "GO_ALL": overall_go,
        "gate": "d' monotone-up with gain & mult beats rate-matched additive on >=5/6 seeds "
                "AND byte-identical-when-off on all seeds",
        "elapsed_s": round(time.time() - t0, 1),
    }

    print("-" * 78)
    print(f"[NE-realbridge] GO {n_go}/{len(args.seeds)}  monotone {n_mono}/{len(args.seeds)}  "
          f"mult>add {n_beats}/{len(args.seeds)}  byte-ident {n_bi}/{len(args.seeds)}  "
          f"=> {'GO' if overall_go else 'NO-GO / see numbers'}  ({overall['elapsed_s']}s)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(overall, f, indent=2, default=float)
        print(f"[NE-realbridge] wrote {args.out}")


if __name__ == "__main__":
    main()
