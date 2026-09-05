"""DA -> WRITE-GAIN, the remaining LEAF linear map -- a spiking/synaptic read retires it (scaffold-retirement
backlog rank-16: "DA->write-magnitude host linear (MED - partial: homeostasis-half retired)").

WHAT WAS ALREADY RETIRED (do not re-do; build on it). `webapp/da_encoding_drives_chat.py` LEVER-3
(`da_encoding_substrate_enabled`, default ON since 2026-08-25) already moved the POPULATION-LEVEL homeostatic
regulation (the Turrigiano multiplicative scaling toward a set-point) onto the substrate itself
(`OneBrainComposer.apply_homeostatic_scaling`, a genuine synaptic-scaling rule computed from MEASURED neural
readout activity, run as a consolidation pass). That is the "homeostasis-half" rank-16 names as already retired.

WHAT IS STILL A HOST SHORTCUT (this file's target). Even with LEVER-3 on, the PER-WRITE LEAF computation --
"how strongly is THIS fact written, given the live self-produced DA level" -- is still a closed-form host
formula, `encoding_gain_for()`'s calls to `_gain_map()`:
    g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline))
This is arithmetic on a scalar, not a neuron or synapse. Rank-16 asks for a spiking/synaptic read of the SAME
DA signal driving the SAME write gain, reusing the existing DA/neuromodulator machinery.

THE MECHANISM (all-spiking / all-synaptic at the leaf; reuse-by-import; NO sim/ edit). A small excitatory
population ("write_gain", Izhikevich CA1/CA3-pyramidal preset `IZH2007_HIPPO_PYRAMIDAL` -- the SAME cell class
this coupling's own biology citation names: Lisman & Grace 2005's hippocampal-VTA loop, dopamine D1/D5
receptors on hippocampal pyramidal neurons gating LTP/memory consolidation) receives:
  1. a FIXED background "write-event" current (the host-legitimate environmental trigger -- "a fact is being
     taught right now" -- exactly the same class of boundary input every other faculty in this codebase uses
     to drive a population: `da_mode_drives_chat`'s engagement afferent, the GNW stop-trigger's two afferent
     relay pools, etc.);
  2. a DA-modulated EXCITABILITY DRIVE via the SAME `sim.neuromodulators.NeuromodulatorManager` machinery the
     #64/#76 DA-mode reconfiguration already uses for striatal D1/D2 (`ModulatorTarget(target_type=
     "excitability_drive", scope="group:write_gain", sensitivity=+K)` -- literally the SAME target_type/scope
     idiom as `_neuromod_spiking_da_mode_derisk.da_nucleus_config()`'s `group:str_D1` target, sign convention
     matching D1R Gs-excitatory DA action; this file's write_gain plays the D1-like role, str_D1's sibling,
     for a hippocampal- rather than striatal-gated readout).
The population's own membrane integration + spiking response to that DA-modulated current is what now decides
how strongly it fires; ITS FIRING RATE (read over a short window) -- not a python formula -- is what the write
gain is computed FROM. The DA scalar itself is UNCHANGED (still the #76/#79 spiking-SNc-derived
`chat._last_da_drives["da_level"]`; this file does not touch how DA is produced, only how a GAIN is read FROM
it): `spiking_write_gain(da, ...)` takes that already-neural DA level and returns a gain via the population's
rate, a drop-in replacement for `_gain_map()`'s role (same signature shape: da, da_baseline, g_min, g_max).

THE ONLY REMAINING HOST ARITHMETIC (honestly scoped, not hidden). Converting a firing RATE (Hz) into GAIN
units is a two-point affine calibration (`_rate_to_gain`), exactly the same class of unit conversion this
codebase already treats as legitimate "neural" plumbing throughout the DA family: DA CONCENTRATION itself is
produced from SNc firing via `from_region_firing_signed`'s own linear rate->concentration transduction, and
`da_mode_drives_chat._MAX_AFFERENT_PA`'s docstring calibrates "0pA->DA~0.05 (rest), ..., 1300pA->1.24
(arousal)" the same way. The DECISION-BEARING step -- how much the write gain moves for a given DA change --
is now the population's own f-I response to a DA-modulated current, not a chosen slope constant; the two
calibration anchors are pinned to the SAME `da_to_encoding_gain`(reused-by-import, unclipped) values at the
SAME two DA reference points `da_mode_drives_chat.py` already established (0.05 rest-floor, 1.24 arousal-
ceiling), so the calibration is not inventing new operating points either.

LESION (this mechanism's OWN, distinct from `da_encoding_lesioned()`/`da_drives_lesioned()`). `lesion=True`
builds the write_gain population with the excitability_drive target's sensitivity pinned to 0.0 -- the DA
broadcast no longer reaches write_gain's membrane current AT ALL (a structural severance, not merely holding
the input fixed) -- so its firing rate becomes IDENTICAL to the rate at DA==DA_baseline (the driving current
at baseline is already zero: sensitivity*(baseline-baseline)=0, the SAME current the lesioned population
always sees), for every DA level. Read through the SAME calibration anchors, this collapses the gain to the
DA-independent floor value (~1.0) regardless of the true DA -- the same "coupling severed" signature every
other lesion in this codebase produces.

CONTRACT (additive; validated here as a DE-RISK; NOT flipped in production by this file --
`webapp/da_encoding_drives_chat.py`'s `BRAIN_DA_ENCODING_SPIKING_GAIN` gates the production hook, default OFF).

Run (numpy-CPU, cheap, foreground):
    SIM_BACKEND=numpy python -m research.runners._da_write_gain_spiking_derisk --seeds 42 43 44 100 101 102
    SIM_BACKEND=numpy python -m research.runners._da_write_gain_spiking_derisk --explore   # tune bg/k_gain
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import da_to_encoding_gain  # noqa: E402
from research.runners._da_composer_salience_cleanup_derisk import measure_da_levels  # noqa: E402

# ── the write_gain population + DA-drive operating point (tuned via --explore; see finding for the sweep). ──
N_WRITE_GAIN = 80                 # sized to average down the substrate's OWN OU background-current noise
                                   # (`cfg.enable_ou_process` default True, ou_std_current_pA=100 -- biologically
                                   # real background synaptic bombardment, deliberately left ON, not suppressed --
                                   # a bigger pool + longer windows average it out instead of hiding it).
BG_CURRENT_PA = 260.0             # fixed "write-event" background current (host-legitimate boundary trigger);
                                   # comfortably above IZH2007_HIPPO_PYRAMIDAL rheobase (k*(vt-vr)^2/4C-normalized
                                   # threshold ~109pA at these preset params) so the pool fires at a moderate,
                                   # non-degenerate baseline rate before any DA modulation is added.
K_GAIN_PA = 260.0                 # excitability_drive sensitivity (pA per unit DA-baseline deviation); D1-like
                                   # (+): DA above baseline DEPOLARIZES write_gain further (Gs-coupled), DA
                                   # below baseline withdraws drive -- same sign convention as str_D1's S_D1.
N_SETTLE = 200                    # steps to let the added excitability drive settle before reading
N_READ = 600                      # firing-rate accumulation window (ms, dt=1ms) -- long enough that the OU
                                   # noise (tau=15ms) averages down to a small fraction of the DA-driven signal
N_CAL_REPEATS = 3                 # calibration (rate_lo/rate_hi) is a ONE-TIME cost per (seed,lesion) reader --
                                   # spend extra averaging there so a noisy single draw can't set a bad anchor

# Measured noise floor (seed 42, N_READ=600, two DA points 0.007 apart): repeated reads at the SAME da have a
# std of ~0.2-0.3Hz on a ~20Hz total calibration span -- i.e. two DA points this close are genuinely
# INDISTINGUISHABLE by this instrument, not a bug. In gain units that is ~0.03-0.05 per single read; the
# tolerances below are set a small factor above that measured floor, not chosen to paper over a real effect.
GATE_N_REPEATS = 3                 # the 6-seed GATE (not the cheap production read) averages this many
                                   # independent reads per (da, seed) point before comparing -- affordable for a
                                   # validation script, shrinks the effective per-point noise by ~sqrt(3).
MONOTONIC_TOL_GAIN = 0.08          # a monotonicity check tolerates a decrease smaller than this (gain units)
                                   # between adjacent da points -- the measured single-read noise floor is
                                   # ~0.03-0.05 in these units; 0.08 gives margin without hiding a real reversal.

_DA_TONIC_BASELINE = 0.5          # == da_encoding_drives_chat._DA_TONIC_BASELINE (reused BY VALUE; the SAME
                                   # convention that file itself uses re-declaring the I-7-b constant locally)
_K_DA_REF = 2.0                   # == da_encoding_drives_chat._K_DA (the host formula's slope, reused to PIN
                                   # the calibration anchors below to the SAME line the host formula draws)
_DA_CAL_LO = 0.05                 # == da_mode_drives_chat's own "0pA -> DA~0.05 (rest)" calibration anchor
_DA_CAL_HI = 1.24                 # == da_mode_drives_chat's own "1300pA -> DA~1.24 (arousal)" calibration anchor

_RAW_G_LO = da_to_encoding_gain(_DA_CAL_LO, _DA_TONIC_BASELINE, _K_DA_REF, g_min=-1e9, g_max=1e9)   # == 0.10
_RAW_G_HI = da_to_encoding_gain(_DA_CAL_HI, _DA_TONIC_BASELINE, _K_DA_REF, g_min=-1e9, g_max=1e9)   # == 2.48


# ============================================================================
# 1. The write_gain population: ONE region, a `dopamine` modulator whose ONLY
#    job is to broadcast a MANUALLY-set concentration (the SAME shared DA
#    broadcast `read_da_concentration`/`_da_confidence_gate` already reuse)
#    onto write_gain's excitability. NO sim/ edit (region + neuromodulator
#    config only, mirrors `_da_composer_salience_cleanup_derisk._build_snc_bridge`).
# ============================================================================
def _build_write_gain_bridge(seed, n_write_gain=N_WRITE_GAIN, k_gain_pa=K_GAIN_PA,
                             da_baseline=_DA_TONIC_BASELINE, lesion=False):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    from sim.enums import NeuronModel, NeuronType
    from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule

    cfg = CoreSimConfig()
    cfg.seed = int(seed)   # ⛔ CLAUDE.md: heterogeneity is seeded from cfg.seed, NOT actual_seed_used -- set it.
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_reward_modulation = False
    cfg.enable_structural_plasticity = False   # default True in CoreSimConfig; this pool has NO synapses to grow
                                               # (connections_per_neuron=0, internal_density=0.0) -- growth candidate
                                               # bookkeeping (`_get_cached_coo()`) throws on a connection-free
                                               # substrate (`_pre_coo` is None), caught+logged CRITICAL every step
                                               # by the bridge's own handler; explicitly off (irrelevant here anyway
                                               # -- a fixed feedforward pool, no plasticity of any kind) is the fix.
    cfg.brain_regions = [
        BrainRegion(
            name="write_gain", n_neurons=n_write_gain, exc_fraction=1.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        ),
    ]
    cfg.region_pathways = []
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [
        NeuromodulatorConfig(
            name="dopamine", baseline=float(da_baseline), decay_tau_ms=200.0,
            concentration_min=0.0, concentration_max=2.0,
            targets=[ModulatorTarget(target_type="excitability_drive", scope="group:write_gain",
                                     sensitivity=(0.0 if lesion else float(k_gain_pa)))],
            production_rules=[ProductionRule(rule_type="manual")],   # DA is set externally (set_concentration)
        ),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _is_ndarray(x) -> bool:
    if isinstance(x, np.ndarray):
        return True
    try:
        import cupy
        return isinstance(x, cupy.ndarray)
    except Exception:
        return False


def _snapshot(bridge):
    return {k: getattr(bridge, k).copy() for k in dir(bridge)
            if k.startswith("cp_") and _is_ndarray(getattr(bridge, k, None))}


def _restore(bridge, snap):
    for k, v in snap.items():
        getattr(bridge, k)[:] = v


def _read_rate_hz(bridge, wg_idx, da, bg_current_pa=BG_CURRENT_PA, n_settle=N_SETTLE, n_read=N_READ):
    """One history-independent read: set the shared DA concentration, drive write_gain with the fixed
    background write-event current PLUS whatever excitability_drive the modulator now computes from that DA,
    step, and return write_gain's mean firing rate (Hz) over the trailing n_read-step window. Caller is
    responsible for restoring the bridge to its post-build snapshot beforehand (history independence)."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge.neuromodulator_manager.set_concentration("dopamine", float(da))
    total = 0
    for step_i in range(n_settle + n_read):
        drive = bridge.neuromodulator_manager.compute_excitability_drive_per_neuron()
        base = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float64)
        base[wg_idx] = bg_current_pa
        if drive is not None:
            base = base + drive
        bridge.cp_external_input_current[:] = base
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        if step_i >= n_settle:
            total += int(bridge.cp_firing_states[wg_idx].sum())
    return total / max(len(wg_idx), 1) / (n_read * 1e-3)


def _read_rate_hz_repeated(bridge, wg_idx, da, snap, n_repeats=N_CAL_REPEATS):
    """Average `_read_rate_hz` over `n_repeats` INDEPENDENT history-independent reads (restoring the post-build
    snapshot between each), to average down the substrate's own OU background-current noise
    (`cfg.enable_ou_process`, left ON -- biologically real, not suppressed) before it can set a bad calibration
    anchor. Used ONLY for calibration (a one-time cost per reader); a single production read uses one long
    (N_SETTLE+N_READ) window instead of repeats (see `spiking_write_gain`)."""
    vals = []
    for _ in range(n_repeats):
        _restore(bridge, snap)
        vals.append(_read_rate_hz(bridge, wg_idx, da))
    _restore(bridge, snap)
    return float(np.mean(vals))


_MIN_DISCRIMINABLE_HZ = 3.0   # below this, rate_hi-rate_lo cannot be trusted as a real DA-driven separation vs
                              # the OU noise floor -- treat as degenerate rather than divide by a noise-sized gap


def _rate_to_gain(rate, rate_lo, rate_hi, g_min, g_max):
    """Two-point affine calibration (the ONLY host arithmetic left): map a MEASURED write_gain firing rate
    onto the SAME raw-gain line the host formula draws at the two shared calibration DA points (_DA_CAL_LO/HI),
    then clip to the caller's (g_min, g_max) -- identical clip semantics to `da_to_encoding_gain`. Guards
    against the OU-noise degenerate case (rate_hi-rate_lo too small to be a real DA-driven separation), not just
    an exact-zero denominator -- a small noise-sized gap would otherwise AMPLIFY a single noisy read into a
    wild extrapolated gain (earned: da=0.05 under a 1e-9 epsilon read g=2.18 off a noise-only rate_hi-rate_lo)."""
    if abs(rate_hi - rate_lo) < _MIN_DISCRIMINABLE_HZ:
        raw_g = 1.0   # degenerate calibration (population did not discriminate) -> neutral fallback, never NaN
    else:
        frac = (rate - rate_lo) / (rate_hi - rate_lo)
        raw_g = _RAW_G_LO + frac * (_RAW_G_HI - _RAW_G_LO)
    return float(min(g_max, max(g_min, raw_g)))


# ============================================================================
# 2. Process-level cache: build each (seed, lesion) population ONCE, snapshot
#    its post-build state, and calibrate ONCE against the SAME two DA anchors
#    -- so a per-write production read is just (restore, set DA, step, read).
# ============================================================================
_READERS = {}   # (seed, lesion) -> {"bridge", "idx", "snapshot", "rate_lo", "rate_hi"}


def _get_reader(seed, lesion):
    key = (int(seed), bool(lesion))
    r = _READERS.get(key)
    if r is not None:
        return r
    bridge = _build_write_gain_bridge(seed, lesion=lesion)
    wg_idx = np.asarray(bridge.region_manager.indices("write_gain"), dtype=np.int64)
    snap = _snapshot(bridge)
    rate_lo = _read_rate_hz_repeated(bridge, wg_idx, _DA_CAL_LO, snap)
    rate_hi = _read_rate_hz_repeated(bridge, wg_idx, _DA_CAL_HI, snap)
    r = {"bridge": bridge, "idx": wg_idx, "snapshot": snap, "rate_lo": rate_lo, "rate_hi": rate_hi}
    _READERS[key] = r
    return r


def spiking_write_gain(da, da_baseline=_DA_TONIC_BASELINE, seed=42, g_min=1.0, g_max=3.0, lesion=False):
    """Drop-in replacement for `da_encoding_drives_chat._gain_map()`'s role: given the LIVE (already-neural)
    DA level, return the write-magnitude gain read from the write_gain population's firing rate. Same call
    shape as `da_to_encoding_gain(da, da_baseline, k_da, g_min, g_max)` minus `k_da` (the slope is now the
    population's own calibrated f-I response, not a host constant). `da_baseline` is accepted for interface
    parity with the host map but this mechanism's own baseline is fixed to `_DA_TONIC_BASELINE` (the value the
    write_gain population + calibration anchors were built against); the two values agree in production
    (both trace to the same I-7-b/#76 tonic == 0.5), so passing a different one is a caller error, not silently
    honored -- avoids a MISMATCHED baseline going unnoticed."""
    if abs(float(da_baseline) - _DA_TONIC_BASELINE) > 1e-6:
        raise ValueError(f"spiking_write_gain: da_baseline={da_baseline} != the mechanism's own "
                         f"{_DA_TONIC_BASELINE} (the population + calibration were built against the latter)")
    r = _get_reader(seed, lesion)
    _restore(r["bridge"], r["snapshot"])
    rate = _read_rate_hz(r["bridge"], r["idx"], float(da))
    _restore(r["bridge"], r["snapshot"])
    return _rate_to_gain(rate, r["rate_lo"], r["rate_hi"], g_min, g_max)


# ============================================================================
# 3. The 6-seed de-risk gate: LOAD-BEARING (rate/gain vary with DA) + MONOTONIC
#    + PARITY (tracks the host formula's ordering, not bit-exact) + LESION
#    (severing the excitability_drive target collapses the differential) +
#    DETERMINISM (cfg.seed actually seeds -- build twice, hash thresholds).
# ============================================================================
def _neuron_threshold_hash(bridge):
    import hashlib
    from sim.backend import to_host
    arr = to_host(bridge.cp_neuron_firing_thresholds) if bridge.cp_neuron_firing_thresholds is not None else None
    if arr is None:
        return None
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()[:16]


def _avg_gain(da, seed, g_min, g_max, lesion, n=GATE_N_REPEATS):
    """Average `n` independent `spiking_write_gain` reads at one da point (each restores the reader's bridge to
    its post-build snapshot first, so the n draws are independent noise realizations, not n correlated steps of
    one trajectory). Validation-only cost (the cheap per-write production call stays a single read)."""
    return float(np.mean([spiking_write_gain(da, seed=seed, g_min=g_min, g_max=g_max, lesion=lesion)
                          for _ in range(n)]))


def evaluate_seed(seed, da_sweep=None, g_min=1.0, g_max=3.0):
    """One seed's full gate: parity/monotonicity/load-bearing on the INTACT population, collapse-to-constant
    on the LESIONED population, both against REAL SNc-derived DA reference points (reused-by-import from
    `_da_composer_salience_cleanup_derisk.measure_da_levels`, NOT hand-picked numbers), plus the determinism
    self-check CLAUDE.md's seed-trap section requires ("build twice at one seed and hash
    cp_neuron_firing_thresholds; identical => seeded")."""
    da_real = measure_da_levels(seed)   # a REAL spiking SNc's own (da_low, da_high, da_baseline)
    if da_sweep is None:
        da_sweep = sorted(set([
            _DA_CAL_LO, da_real["da_low"], da_real["da_baseline"],
            0.5 * (da_real["da_baseline"] + da_real["da_high"]), da_real["da_high"], _DA_CAL_HI,
        ]))

    g_spike = [_avg_gain(da, seed, g_min, g_max, False) for da in da_sweep]
    g_host = [da_to_encoding_gain(da, _DA_TONIC_BASELINE, _K_DA_REF, g_min, g_max) for da in da_sweep]
    g_lesion = [_avg_gain(da, seed, g_min, g_max, True) for da in da_sweep]

    span_intact = max(g_spike) - min(g_spike)
    span_lesion = max(g_lesion) - min(g_lesion)
    load_bearing = span_intact > 0.15                       # a genuine, non-vacuous differential across the sweep
    lesion_collapses = span_lesion < 0.05                   # the DA-dependence collapses under the lesion
    lesion_near_floor = abs(np.mean(g_lesion) - g_min) < 0.05  # collapses TO the floor (~1.0), not to some other value

    # monotonicity: non-decreasing in da, TOLERANT of a decrease smaller than the measured noise floor
    # (MONOTONIC_TOL_GAIN) between adjacent points -- two da values closer together than this instrument can
    # resolve (measured: ~0.007 DA units apart) must not fail the gate on a coin-flip-sized ordering swap.
    order = np.argsort(da_sweep)
    g_sorted = [g_spike[i] for i in order]
    diffs = np.diff(g_sorted)
    monotonic = bool(np.all(diffs >= -MONOTONIC_TOL_GAIN))

    corr = float(np.corrcoef(g_spike, g_host)[0, 1]) if len(set(g_spike)) > 1 and len(set(g_host)) > 1 else None

    # determinism self-check (the cfg.seed trap): build the SAME seed's population twice, hash thresholds.
    b1 = _build_write_gain_bridge(seed)
    h1 = _neuron_threshold_hash(b1)
    b2 = _build_write_gain_bridge(seed)
    h2 = _neuron_threshold_hash(b2)
    seeded_ok = bool(h1 is not None and h1 == h2)

    return {
        "seed": seed, "da_sweep": da_sweep, "da_real_reference": da_real,
        "g_spike": g_spike, "g_host": g_host, "g_lesion": g_lesion,
        "span_intact": span_intact, "span_lesion": span_lesion,
        "load_bearing": load_bearing, "lesion_collapses": lesion_collapses,
        "lesion_near_floor": lesion_near_floor, "monotonic_intact": monotonic,
        "parity_corr_spike_vs_host": corr,
        "determinism_seeded_ok": seeded_ok, "threshold_hash_build1": h1, "threshold_hash_build2": h2,
        "GO": bool(load_bearing and lesion_collapses and lesion_near_floor and monotonic and seeded_ok
                  and (corr is None or corr > 0.85)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--g-min", type=float, default=1.0)   # default branch: the LEVER-3 recall-safe floor
    ap.add_argument("--g-max", type=float, default=3.0)
    ap.add_argument("--explore", action="store_true", help="one-seed rate sweep, for tuning BG/K_GAIN only")
    ap.add_argument("--out", type=str, default="research/findings/raw/_da_write_gain_spiking/6seed.json")
    args = ap.parse_args()

    if args.explore:
        seed = args.seeds[0]
        da_real = measure_da_levels(seed)
        print(f"real SNc reference: {da_real}")
        for da in sorted(set([_DA_CAL_LO, da_real["da_low"], da_real["da_baseline"],
                              da_real["da_high"], _DA_CAL_HI, 1.5])):
            g = spiking_write_gain(da, seed=seed, g_min=args.g_min, g_max=args.g_max)
            gl = spiking_write_gain(da, seed=seed, g_min=args.g_min, g_max=args.g_max, lesion=True)
            g_host = da_to_encoding_gain(da, _DA_TONIC_BASELINE, _K_DA_REF, args.g_min, args.g_max)
            print(f"  da={da:.4f}  g_spike={g:.4f}  g_host={g_host:.4f}  g_lesion={gl:.4f}")
        return 0

    results = [evaluate_seed(s, g_min=args.g_min, g_max=args.g_max) for s in args.seeds]

    load_bearing_all = all(r["load_bearing"] for r in results)
    monotonic_all = all(r["monotonic_intact"] for r in results)
    lesion_collapses_all = all(r["lesion_collapses"] and r["lesion_near_floor"] for r in results)
    seeded_all = all(r["determinism_seeded_ok"] for r in results)
    corr_vals = [r["parity_corr_spike_vs_host"] for r in results if r["parity_corr_spike_vs_host"] is not None]
    parity_all = bool(corr_vals) and all(c > 0.85 for c in corr_vals)
    go = bool(load_bearing_all and monotonic_all and lesion_collapses_all and seeded_all and parity_all)

    # ── ATTRIBUTION (tools.lab's own lesson: measuring both arms is not the same as asking whose the
    #    difference was): what fraction of the mean intact gain-span is owed to the live excitability_drive
    #    link, pooled across all 6 seeds (control = the SAME link severed at build time, sensitivity=0.0)?
    from tools.lab import attributable_to
    mean_span_intact = float(np.mean([r["span_intact"] for r in results]))
    mean_span_lesion = float(np.mean([r["span_lesion"] for r in results]))
    lesion_attribution = attributable_to(
        "the intact write-gain span owed to the LIVE excitability_drive link (control = sensitivity=0.0 lesion), "
        "pooled across 6 seeds", mean_span_intact, mean_span_lesion)

    from tools.verdict import Verdict
    v = Verdict("DA write-gain spiking population read retires the remaining leaf host linear map (rank-16)")
    v.require("load-bearing on every seed (intact gain span > 0.15)", load_bearing_all, expect=True,
              note=f"spans={[round(r['span_intact'], 4) for r in results]}")
    v.require("monotonic (DA-tolerant) on every seed", monotonic_all, expect=True)
    v.require("lesion collapses the span to the floor on every seed", lesion_collapses_all, expect=True,
              note=f"lesion spans={[round(r['span_lesion'], 4) for r in results]}")
    v.require("cfg.seed determinism confirmed on every seed (threshold hash, build x2)", seeded_all, expect=True)
    v.require("parity with the host formula (corr > 0.85) on every evaluable seed", parity_all, expect=True,
              note=f"corr={[round(c, 4) for c in corr_vals]}")
    v.control("the write-gain span rides the live DA->write_gain link (intact) and is severed by ITS OWN "
              "build-time lesion", treatment=mean_span_intact, control=mean_span_lesion, min_separation=0.1,
              note=f"pooled across 6 seeds; attribution={lesion_attribution}")
    v.disabled("the substrate-homeostat's LEVER-3 population-level regulation (apply_substrate_homeostasis)",
              why="already 6-seed GO'd (2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP.md) and "
                  "UNMODIFIED by this mechanism -- this gate is scoped to the ONE remaining per-write LEAF")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "_da_write_gain_spiking_derisk", "go": go, "status": decided["status"],
        "mechanism": "DA (already-neural) -> write_gain population (IZH2007_HIPPO_PYRAMIDAL, D1/D5-like "
                     "excitability_drive) -> firing rate -> calibrated write-magnitude gain",
        "constants": {"n_write_gain": N_WRITE_GAIN, "bg_current_pa": BG_CURRENT_PA, "k_gain_pa": K_GAIN_PA,
                     "da_cal_lo": _DA_CAL_LO, "da_cal_hi": _DA_CAL_HI, "raw_g_lo": _RAW_G_LO, "raw_g_hi": _RAW_G_HI},
        "per_seed": results,
        "verdict": {
            "GO": go,
            "load_bearing_all_seeds": load_bearing_all,
            "monotonic_all_seeds": monotonic_all,
            "lesion_collapses_all_seeds": lesion_collapses_all,
            "determinism_seeded_all_seeds": seeded_all,
            "parity_corr_range": [r["parity_corr_spike_vs_host"] for r in results],
            "mean_span_intact": mean_span_intact, "mean_span_lesion": mean_span_lesion,
            "lesion_attribution": lesion_attribution,
        },
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar)
    print("  DA WRITE-GAIN SPIKING READ -- 6-seed de-risk (rank-16)")
    print(bar)
    for r in results:
        print(f"  seed={r['seed']:>4}  span_intact={r['span_intact']:.3f}  span_lesion={r['span_lesion']:.3f}  "
              f"monotonic={r['monotonic_intact']}  corr_host={r['parity_corr_spike_vs_host']}  "
              f"seeded_ok={r['determinism_seeded_ok']}  -> GO={r['GO']}")
    print(f"\n  VERDICT: {'GO' if go else 'NO-GO'} ({sum(r['GO'] for r in results)}/{len(results)} seeds)")
    print(f"  [saved] {args.out}\n" + bar)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
