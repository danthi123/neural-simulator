"""Cluster B.2 biology probe — verify striatal FSI cross-action inhibition.

Updated for R1.2 rewire (2026-04-29): FSIs now target cross-action MSNs
ONLY — FS_X does NOT inhibit str_D1_X / str_D2_X. Biology grounding:
TK-2017 pp 161–163 + Tepper-2018 pp 8–9 — paired-recording studies show
MSN-MSN collaterals are functionally weak (<0.5 mV unitary IPSPs, ~20%
connection probability, high failure rates) while FSI→MSN feedforward
inhibition is significantly larger and reliable. The FSI cross-action
projection is the biologically dominant WTA substrate.

Builds matched-seed pairs of minimal BG cascade bridges (baseline vs +FSIs),
applies strong cortex_N drive plus weaker cortex_E drive plus direct MSN
drive, and records str_D1_N / str_D1_E firing rates in 10 ms bins across
5 seeds. The cross-action signature is then directly testable.

Expected biological signature (R1.2 cross-action wiring):
- Without FSIs, both str_D1 pools fire at their drive-determined rates;
  there is no fast circuit to suppress them.
- With FSIs, cortex_N drive recruits str_FS_N. Because FSIs target
  CROSS-action MSNs only, FS_N inhibits str_D1_E (loser pool) but NOT
  str_D1_N (its own action channel). The signature is therefore
  asymmetric: str_D1_E is suppressed more than str_D1_N — the opposite
  of the previous broadcast wiring (R1.1) where str_D1_N took the
  strongest hit.

The signature is millisecond-scale and TRANSIENT — FS_N fires briefly at
the start of the run, suppresses cross-action MSNs during that window,
then mean rates re-equilibrate by the end. PASS criteria:
  1. str_FS_N fires (pathway engaged).
  2. str_D1_E peak rate drops by >= 5 Hz with FSIs on (cross-action hit).
  3. str_D1_E peak drop > str_D1_N peak drop (cross > own; the WTA signature).

Run:
    python -m research.probes.striatal_fsi_probe

Outputs:
- stdout: human-readable summary + verdict
- research/findings/raw/striatal_fsi_probe/probe_results.json: structured
  data including per-seed traces.

Plan: docs/plans/2026-04-29-catalog-remediation-pass.md (R1.2 row).
Original plan: docs/plans/2026-04-28-cluster-b2-striatal-fsis-implementation.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the repo root importable when run as a module or as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cupy as cp  # noqa: E402

from research.runners.g11_bg_runner import build_bg_brain_regions  # noqa: E402
from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)


# ---- Probe configuration ---------------------------------------------------

OUT_DIR = _REPO_ROOT / "research" / "findings" / "raw" / "striatal_fsi_probe"
OUT_JSON = OUT_DIR / "probe_results.json"

# Small network, short sim — designed to run alongside other GPU work.
N_CORTEX = 20  # smallest the runner builder supports
N_STEPS = 200  # 200 steps × 1 ms = 200 ms total
DT_MS = 1.0
BIN_MS = 10.0
STEPS_PER_BIN = int(BIN_MS / DT_MS)
N_BINS = N_STEPS // STEPS_PER_BIN

# Drive levels.
#
# Cortex pyramidals (IZH2007_RS_CORTICAL_PYRAMIDAL: C=100, k=0.7, vt-vr=20)
# need ~150 pA to spike. We drive cortex_N at 800 pA (winner — recruits
# str_FS_N when FSIs are on) and cortex_E weakly (won't recruit str_FS_E).
#
# We ALSO drive str_D1_N and str_D1_E directly with enough current to
# start firing — striatal MSNs (k=1.0, vt-vr=55, b=-20) sit deep in the
# down-state and need ~750 pA just to escape. Cortex→D1 alone doesn't
# get them there in 200 ms with this small a network. Direct MSN drive
# means both pools start firing at comparable rates, so we can isolate
# the FSI effect: with FSIs on, the str_FS_N broadcast should pull
# str_D1_E down even though str_D1_E is still receiving direct drive.
CORTEX_N_DRIVE_PA = 800.0
CORTEX_E_DRIVE_PA = 200.0  # weak — recruits its own MSN_E only weakly
# Direct MSN drive — set just above MSN down-state escape threshold
# (~750 pA) so the pool fires sustainably but FS broadcast inhibition
# can actually push individual cells back below threshold. Above ~900 pA
# the direct drive swamps the inhibition budget and the FSI signal
# disappears.
MSN_DRIVE_PA = 780.0

# BG tonic drives copied from run_moving_goal_episode trial setup. Without
# these, GPe / GPi / thalamus sit silent and the cascade can't operate at
# all, masking any FSI effect.
GPE_TONIC_PA = 150.0
GPI_TONIC_PA = 110.0
STN_DRIVE_PA = 150.0
DOPAMINE_DRIVE_PA = 150.0
THAL_DRIVE_PA = 300.0

# Suppression threshold: first bin where loser rate < 50% of its observed
# peak across the run. Higher than 50% would be under-strict; lower would
# trip on transient dips.
SUPPRESSION_FRACTION = 0.5

# Average across multiple seeds — single-seed dynamics are noisy enough
# that one realization can fail or pass spuriously. With 5 seeds the
# loser-pool peak/mean rate becomes a robust ensemble metric. Each seed
# is paired between conditions (off / on) so the same noise samples and
# heterogeneity are drawn — every comparison is matched. Cost: ~2 sec
# per seed-pair on the 3090, so ~10 sec total — still well under 2 min.
SEEDS = [42, 43, 44, 45, 46]

ACTION_NAMES = ["N", "E", "S", "W"]


# ---- Bridge construction ---------------------------------------------------

def _build_minimal_bridge(enable_fsis: bool, seed: int) -> SimulationBridge:
    """Smallest BG cascade we can stand up.

    Plasticity is fully off so we measure pure circuit dynamics — no STDP /
    Hebbian / homeostasis / structural / synaptic-scaling / reward
    confounding the loser-pool firing rate. Other Cluster B inhibitory
    mechanisms (D1/D2 asymmetry, BG lateral inhibition) are explicitly
    OFF so the only knob is the FSI broadcast pathway.
    """
    regions, pathways = build_bg_brain_regions(
        n_cortex=N_CORTEX,
        enable_striatal_fsis=enable_fsis,
        # Hold the v3 MSN-MSN lateral inhibition off so the FSI broadcast
        # effect is isolated, not summed with the slower MSN-collateral
        # mechanism. (D1/D2 asymmetry lives on CoreSimConfig, not here —
        # we leave it at its default off below.)
        enable_bg_lateral_inhibition=False,
    )

    cfg = CoreSimConfig(
        num_neurons=1,  # placeholder; region_manager overrides in init
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=pathways,
        # Belt-and-braces: keep D1/D2 asymmetry off so the only inhibitory
        # mechanism in play is the FSI broadcast pathway we're testing.
        enable_d1_d2_asymmetry=False,
    )

    # Disable every plasticity mechanism — we want stable circuit dynamics,
    # not learning. Mirrors the strategy in d1_d2_asymmetry_probe.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_reward_modulation = False

    cfg.dt_ms = DT_MS

    # Pin every RNG to the caller-provided seed so the probe is
    # deterministic per-seed. Within a seed, both bridges (FSI off /
    # FSI on) draw IDENTICAL cortex→MSN initial weights, OU noise
    # samples, and heterogeneity assignments — so any rate difference
    # is attributable purely to the FSI circuitry.
    cfg.seed = seed
    cfg.heterogeneity_seed = seed
    cfg.ou_seed = seed

    # STDP_w_max gotcha — even though STDP is off, the bridge clips weights
    # to this bound during init in some paths. Push it well above any
    # design weight so cortex→D1 (mean=25) and FS→MSN (mean=8) survive.
    cfg.stdp_w_max = 100.0
    cfg.hebbian_max_weight = 100.0

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _region_index_dict(bridge: SimulationBridge) -> dict[str, cp.ndarray]:
    """Build name -> cupy index array map for every populated region."""
    out: dict[str, cp.ndarray] = {}
    for region in bridge.core_config.brain_regions:
        idx = list(bridge.region_manager.indices(region.name))
        if idx:
            out[region.name] = cp.asarray(idx, dtype=cp.int64)
    return out


def _set_drives(bridge: SimulationBridge,
                region_idx: dict[str, cp.ndarray]) -> None:
    """Stamp the drive vector: BG tonic + cortex_N strong + cortex_E weak.

    We zero everything first so leftover drive from prior trials can't
    leak in (relevant when a single bridge is reused for multiple probes,
    not strictly needed here but cheap insurance).
    """
    bridge.cp_external_input_current[:] = 0.0

    # BG tonic drives — without these the cascade is silent.
    for action in ACTION_NAMES:
        bridge.cp_external_input_current[region_idx[f"gpe_{action}"]] = (
            cp.float32(GPE_TONIC_PA)
        )
        bridge.cp_external_input_current[region_idx[f"gpi_{action}"]] = (
            cp.float32(GPI_TONIC_PA)
        )
        bridge.cp_external_input_current[region_idx[f"thal_{action}"]] = (
            cp.float32(THAL_DRIVE_PA)
        )
    if "stn" in region_idx:
        bridge.cp_external_input_current[region_idx["stn"]] = (
            cp.float32(STN_DRIVE_PA)
        )
    if "dopamine" in region_idx:
        bridge.cp_external_input_current[region_idx["dopamine"]] = (
            cp.float32(DOPAMINE_DRIVE_PA)
        )

    # Cortex drive — the experimental manipulation.
    # Strong cortex_N → recruits str_FS_N (when FSIs are on) → broadcasts.
    # Weak cortex_E → keeps str_FS_E quiet so its only suppression comes
    # from the str_FS_N broadcast across action pools.
    bridge.cp_external_input_current[region_idx["cortex_N"]] = (
        cp.float32(CORTEX_N_DRIVE_PA)
    )
    bridge.cp_external_input_current[region_idx["cortex_E"]] = (
        cp.float32(CORTEX_E_DRIVE_PA)
    )
    # cortex_S and cortex_W get no drive — they're the silent baseline.

    # Direct MSN drive for both N and E so both pools fire at comparable
    # rates absent any inhibition. Without this, MSN down-state dominates
    # and we measure 0 Hz on both — the FSI broadcast effect is invisible.
    # MSNs in BOTH pools get the same drive, so any rate difference between
    # str_D1_N and str_D1_E is attributable to the inhibitory circuitry.
    for action in ("N", "E"):
        bridge.cp_external_input_current[region_idx[f"str_D1_{action}"]] = (
            cp.float32(MSN_DRIVE_PA)
        )
        bridge.cp_external_input_current[region_idx[f"str_D2_{action}"]] = (
            cp.float32(MSN_DRIVE_PA)
        )


# ---- Simulation + measurement ---------------------------------------------

def _run_and_record(bridge: SimulationBridge,
                    region_idx: dict[str, cp.ndarray],
                    track_fs: bool) -> dict[str, list[float]]:
    """Step bridge for N_STEPS; record per-bin firing rates (Hz) for the
    pools we care about.

    Returns a dict mapping pool name → list of firing rates (one per bin).
    Always records str_D1_N and str_D1_E (the two we're contrasting).
    Optionally records str_FS_N when FSIs are enabled — handy for sanity
    checking that FSIs are actually firing.
    """
    pools = ["str_D1_N", "str_D1_E", "str_D2_N", "str_D2_E"]
    if track_fs and "str_FS_N" in region_idx:
        pools.extend(["str_FS_N", "str_FS_E"])

    # Per-bin spike accumulator. We tally spikes inside each 10 ms window,
    # then convert to Hz at the end.
    rates: dict[str, list[float]] = {p: [] for p in pools}
    n_per_pool = {p: int(region_idx[p].size) for p in pools}

    _set_drives(bridge, region_idx)

    bin_spike_counts = {p: 0 for p in pools}
    for step in range(N_STEPS):
        bridge._run_one_simulation_step()
        # cp_firing_states is a bool vector; sum over each pool's indices.
        for p in pools:
            bin_spike_counts[p] += int(
                bridge.cp_firing_states[region_idx[p]].sum().get()
            )
        if (step + 1) % STEPS_PER_BIN == 0:
            for p in pools:
                # Hz = spikes / (bin_ms * 1e-3) / n_neurons
                rate_hz = (
                    bin_spike_counts[p]
                    / (BIN_MS * 1e-3)
                    / max(1, n_per_pool[p])
                )
                rates[p].append(rate_hz)
                bin_spike_counts[p] = 0

    return rates


def _time_to_suppression(rates: list[float],
                         fraction: float = SUPPRESSION_FRACTION) -> float | None:
    """Index of first bin where rate < fraction × peak rate, multiplied by
    BIN_MS. Returns None if peak is zero or no bin ever drops that low.
    """
    if not rates:
        return None
    peak = max(rates)
    if peak <= 0.0:
        return None
    threshold = peak * fraction
    # Find first bin AFTER the peak where rate dips below threshold.
    peak_idx = rates.index(peak)
    for i in range(peak_idx + 1, len(rates)):
        if rates[i] < threshold:
            # Time at the END of bin i (suppression observed by then).
            return (i + 1) * BIN_MS
    return None  # never suppressed within window


# ---- Top-level orchestration ----------------------------------------------

def _run_one_seed(seed: int, enable_fsis: bool) -> dict[str, list[float]]:
    """Build a fresh bridge for the given seed/condition, run, return rates."""
    bridge = _build_minimal_bridge(enable_fsis=enable_fsis, seed=seed)
    region_idx = _region_index_dict(bridge)
    rates = _run_and_record(bridge, region_idx, track_fs=enable_fsis)
    bridge.clear_simulation_state_and_gpu_memory()
    return rates


def _avg_rates(per_seed: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """Average per-bin rates across seeds, per pool."""
    if not per_seed:
        return {}
    pools = list(per_seed[0].keys())
    n_seeds = len(per_seed)
    avg: dict[str, list[float]] = {}
    for p in pools:
        # All runs use the same N_BINS so list length is constant.
        vals_per_bin = list(zip(*(s[p] for s in per_seed)))
        avg[p] = [sum(v) / n_seeds for v in vals_per_bin]
    return avg


def main() -> int:
    print("=== Striatal FSI Broadcast Inhibition Probe ===")
    print(
        f"Config: n_cortex={N_CORTEX}, sim_steps={N_STEPS}, dt={DT_MS} ms, "
        f"bin={BIN_MS} ms ({N_BINS} bins), seeds={SEEDS}"
    )
    print(
        f"        cortex_N drive = +{CORTEX_N_DRIVE_PA:.0f} pA (strong, "
        f"\"winner\")"
    )
    print(
        f"        cortex_E drive = +{CORTEX_E_DRIVE_PA:.0f} pA (weak, "
        f"\"competitor\")"
    )
    print(
        f"        cortex_S, cortex_W drive = 0 pA (silent baseline)"
    )

    # ---- Run baseline (FSIs off) for each seed ----------------------------
    print(f"\n--- Running baseline (FSIs off) across {len(SEEDS)} seeds ---")
    per_seed_off = []
    for seed in SEEDS:
        per_seed_off.append(_run_one_seed(seed, enable_fsis=False))
        print(f"  seed {seed}: done")

    # ---- Run +FSIs (FSIs on) for each seed --------------------------------
    print(f"\n--- Running +FSIs (FSIs on) across {len(SEEDS)} seeds ---")
    per_seed_on = []
    for seed in SEEDS:
        per_seed_on.append(_run_one_seed(seed, enable_fsis=True))
        print(f"  seed {seed}: done")

    # Seed-averaged per-bin rates. Per-bin averaging gives the cleanest
    # ensemble metric — a noisy bin in one seed gets smoothed out by the
    # other seeds, but a systematic FSI effect (e.g. lower str_D1_E rates
    # at every bin) survives intact.
    rates_off = _avg_rates(per_seed_off)
    rates_on = _avg_rates(per_seed_on)

    # ---- Analyze ----------------------------------------------------------
    def _stat(rates: list[float]) -> dict:
        if not rates:
            return {"peak_hz": 0.0, "mean_hz": 0.0, "final_hz": 0.0}
        return {
            "peak_hz": float(max(rates)),
            "mean_hz": float(sum(rates) / len(rates)),
            "final_hz": float(rates[-1]),
        }

    # Time-to-suppression: ms (or None if never suppressed within window).
    t_supp_off = _time_to_suppression(rates_off["str_D1_E"])
    t_supp_on = _time_to_suppression(rates_on["str_D1_E"])

    # ---- Print summary ----------------------------------------------------
    print(f"\n--- Without FSIs (baseline, n={len(SEEDS)} seeds averaged) ---")
    n_stat = _stat(rates_off["str_D1_N"])
    e_stat = _stat(rates_off["str_D1_E"])
    print(f"  str_D1_N peak rate: {n_stat['peak_hz']:.1f} Hz  "
          f"(mean {n_stat['mean_hz']:.1f}, final {n_stat['final_hz']:.1f})")
    print(f"  str_D1_E peak rate: {e_stat['peak_hz']:.1f} Hz  "
          f"(mean {e_stat['mean_hz']:.1f}, final {e_stat['final_hz']:.1f})")
    if t_supp_off is None:
        print(f"  Time to E-suppression "
              f"({int(SUPPRESSION_FRACTION*100)}% of peak): "
              f">{N_BINS*BIN_MS:.0f} ms (no suppression observed)")
    else:
        print(f"  Time to E-suppression "
              f"({int(SUPPRESSION_FRACTION*100)}% of peak): {t_supp_off:.0f} ms")

    print(f"\n--- With FSIs (--enable-striatal-fsis, n={len(SEEDS)} seeds averaged) ---")
    n_stat_on = _stat(rates_on["str_D1_N"])
    e_stat_on = _stat(rates_on["str_D1_E"])
    print(f"  str_D1_N peak rate: {n_stat_on['peak_hz']:.1f} Hz  "
          f"(mean {n_stat_on['mean_hz']:.1f}, final {n_stat_on['final_hz']:.1f})")
    print(f"  str_D1_E peak rate: {e_stat_on['peak_hz']:.1f} Hz  "
          f"(mean {e_stat_on['mean_hz']:.1f}, final {e_stat_on['final_hz']:.1f})")
    if "str_FS_N" in rates_on:
        fs_n_stat = _stat(rates_on["str_FS_N"])
        print(f"  (sanity) str_FS_N peak: {fs_n_stat['peak_hz']:.1f} Hz "
              f"— FSIs firing means the broadcast is engaged")
    if t_supp_on is None:
        print(f"  Time to E-suppression: "
              f">{N_BINS*BIN_MS:.0f} ms (no suppression observed)")
    else:
        print(f"  Time to E-suppression: {t_supp_on:.0f} ms")

    # ---- Verdict ----------------------------------------------------------
    # R1.2 cross-action wiring: FSIs target other-action MSNs only. The
    # signature is therefore ASYMMETRIC: FS_N recruits during cortex_N
    # drive and inhibits str_D1_E (cross), but does NOT directly inhibit
    # str_D1_N (its own action channel). The expected pattern is:
    #
    #   - str_D1_E peak rate drops noticeably (>= 5 Hz) with FSIs on,
    #     because FS_N broadcasts directly into str_D1_E (cross-action).
    #   - str_D1_N peak rate is much less affected — there is no FS→MSN_N
    #     pathway, so any change comes from indirect cascade routing
    #     (D2→GPe→STN→GPi→thal→cortex), not the direct FSI hit.
    #
    # The PASS gate therefore requires the cross-action signature:
    #  1. str_FS_N fires (pathway engaged).
    #  2. str_D1_E peak rate drops by >= 5 Hz with FSIs on.
    #  3. str_D1_E peak drop > str_D1_N peak drop (cross > own).
    def _stats_for_pool(rates: dict, pool: str) -> tuple[float, float]:
        """Return (peak_hz, mean_hz) for the given pool, or (0,0)."""
        if pool not in rates or not rates[pool]:
            return 0.0, 0.0
        return max(rates[pool]), sum(rates[pool]) / len(rates[pool])

    n_peak_off, n_mean_off = _stats_for_pool(rates_off, "str_D1_N")
    n_peak_on, n_mean_on = _stats_for_pool(rates_on, "str_D1_N")
    e_peak_off, e_mean_off = _stats_for_pool(rates_off, "str_D1_E")
    e_peak_on, e_mean_on = _stats_for_pool(rates_on, "str_D1_E")
    fs_n_peak = (
        max(rates_on["str_FS_N"])
        if "str_FS_N" in rates_on and rates_on["str_FS_N"]
        else 0.0
    )

    n_peak_drop = n_peak_off - n_peak_on
    n_mean_drop = n_mean_off - n_mean_on
    e_peak_drop = e_peak_off - e_peak_on
    e_mean_drop = e_mean_off - e_mean_on
    # Cross-action signature: E (cross-action target of FS_N) should drop more
    # than N (own-action, which has NO direct FSI input under R1.2).
    cross_minus_own_peak_drop = e_peak_drop - n_peak_drop
    peak_drop_pct = (
        100.0 * e_peak_drop / e_peak_off
        if e_peak_off > 0 else 0.0
    )

    print()
    print(f"  str_D1_N peak: {n_peak_off:.1f} Hz (no FSI) -> "
          f"{n_peak_on:.1f} Hz (+FSI), delta = {n_peak_drop:+.1f} Hz "
          f"(own-action; FS_N has NO direct projection here under R1.2)")
    print(f"  str_D1_N mean: {n_mean_off:.1f} Hz (no FSI) -> "
          f"{n_mean_on:.1f} Hz (+FSI), delta = {n_mean_drop:+.1f} Hz")
    print(f"  str_D1_E peak: {e_peak_off:.1f} Hz (no FSI) -> "
          f"{e_peak_on:.1f} Hz (+FSI), delta = {e_peak_drop:+.1f} Hz "
          f"(cross-action; FS_N projects directly here)")
    print(f"  str_D1_E mean: {e_mean_off:.1f} Hz (no FSI) -> "
          f"{e_mean_on:.1f} Hz (+FSI), delta = {e_mean_drop:+.1f} Hz")
    print(f"  cross-vs-own peak drop delta (E - N): "
          f"{cross_minus_own_peak_drop:+.1f} Hz "
          f"(positive = cross-action WTA signature)")
    print(f"  (sanity) str_FS_N peak rate: {fs_n_peak:.1f} Hz "
          "(must be >0 — cortex→FS pathway must be active)")

    # PASS criteria (R1.2 cross-action wiring):
    #  1. str_FS_N must actually fire — sanity check that the cortex→FS
    #     pathway is engaged.
    #  2. str_D1_E peak rate drops by >= 5 Hz with FSIs on (cross hit).
    #  3. str_D1_E peak drop strictly exceeds str_D1_N peak drop — the
    #     cross-action signature distinguishes the new wiring from the
    #     old broadcast wiring (which would have suppressed N more).
    PASS_PEAK_DROP_HZ = 5.0
    PASS_MEAN_DROP_HZ = 1.0  # informational only, not in PASS gate
    fs_engaged = fs_n_peak > 0.0
    e_passes = e_peak_drop >= PASS_PEAK_DROP_HZ
    cross_gt_own = cross_minus_own_peak_drop > 0.0
    passed = fs_engaged and e_passes and cross_gt_own
    print()
    if passed:
        print(
            f"VERDICT: PASS - FSIs suppress cross-action MSN firing "
            f"(str_D1_E peak -{e_peak_drop:.1f} Hz / {peak_drop_pct:.0f}%; "
            f"cross-vs-own delta {cross_minus_own_peak_drop:+.1f} Hz; "
            f"str_FS_N fired at {fs_n_peak:.0f} Hz peak)"
        )
    elif not fs_engaged:
        print(
            "VERDICT: FAIL - str_FS_N never fired (cortex→FS pathway not engaged); "
            "investigate cortex→FS wiring or FS neuron parameters"
        )
    elif not e_passes:
        print(
            f"VERDICT: FAIL - insufficient cross-action suppression "
            f"(str_D1_E peak delta {e_peak_drop:+.1f} Hz; "
            f"need >= {PASS_PEAK_DROP_HZ} Hz on str_D1_E)"
        )
    else:
        print(
            f"VERDICT: FAIL - cross-action signature not present "
            f"(cross_E_drop - own_N_drop = {cross_minus_own_peak_drop:+.1f} Hz; "
            f"under R1.2, cross-action target should drop MORE than own-action). "
            f"This may indicate the FSI is still wired same-action, or that "
            f"indirect cascade routing dominates over the direct FSI hit."
        )

    # ---- Persist JSON -----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "probe": "striatal_fsi_cross_action_inhibition",
        "wiring_version": "R1.2 (2026-04-29) — cross-action only",
        "plan": (
            "docs/plans/2026-04-29-catalog-remediation-pass.md"
        ),
        "task": 3,
        "verdict": "PASS" if passed else "FAIL",
        "config": {
            "n_cortex": N_CORTEX,
            "n_steps": N_STEPS,
            "dt_ms": DT_MS,
            "bin_ms": BIN_MS,
            "n_bins": N_BINS,
            "seeds": SEEDS,
            "cortex_N_drive_pA": CORTEX_N_DRIVE_PA,
            "cortex_E_drive_pA": CORTEX_E_DRIVE_PA,
            "msn_drive_pA": MSN_DRIVE_PA,
            "suppression_fraction": SUPPRESSION_FRACTION,
            "isolated": [
                "enable_d1_d2_asymmetry=False",
                "enable_bg_lateral_inhibition=False",
                "enable_stdp=False",
                "enable_hebbian_learning=False",
                "enable_homeostasis=False",
                "enable_structural_plasticity=False",
                "enable_synaptic_scaling=False",
                "enable_reward_modulation=False",
            ],
        },
        "without_fsis": {
            "rates_per_bin_hz_avg": rates_off,
            "rates_per_seed": per_seed_off,
            "stats": {
                pool: _stat(rates_off[pool])
                for pool in rates_off
            },
            "time_to_suppression_ms": t_supp_off,
        },
        "with_fsis": {
            "rates_per_bin_hz_avg": rates_on,
            "rates_per_seed": per_seed_on,
            "stats": {
                pool: _stat(rates_on[pool])
                for pool in rates_on
            },
            "time_to_suppression_ms": t_supp_on,
        },
        "delta_suppression_ms": (
            None
            if (t_supp_on is None or t_supp_off is None)
            else (t_supp_off - t_supp_on)
        ),
        "msn_suppression": {
            "str_D1_N_peak_drop_hz": n_peak_drop,  # own-action (no direct FS input)
            "str_D1_N_mean_drop_hz": n_mean_drop,
            "str_D1_E_peak_drop_hz": e_peak_drop,  # cross-action (FS_N projects here)
            "str_D1_E_mean_drop_hz": e_mean_drop,
            "cross_minus_own_peak_drop_hz": cross_minus_own_peak_drop,
            "cross_action_peak_drop_pct": peak_drop_pct,
            "fs_n_peak_hz": fs_n_peak,
            "fs_engaged": fs_engaged,
        },
        "pass_thresholds": {
            "e_peak_drop_hz": PASS_PEAK_DROP_HZ,
            "fs_engaged_required": True,
            "cross_gt_own_required": True,
        },
    }
    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"\nJSON written to: {OUT_JSON}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
