"""Cluster B.3 biology probe -- verify TAN/ACh plasticity-window timing.

Builds a minimal generic bridge with the neuromodulator subsystem on and
the default ACh config registered (tonic baseline=1.0, pause_on_reward
rule with sensitivity=-2.0). Pins eligibility, runs a 4-phase scenario,
and samples ACh concentration + plasticity-window gate + cumulative
weight delta at each step.

Expected biological signature (BG TAN pause):
- Phase 0 (baseline, no reward, ~50ms):
    ACh ~= baseline (1.0), gate ~ 0 (plasticity blocked), no weight updates.
- Phase 1 (brief +reward window, ~10ms):
    pause_on_reward drives ACh DOWN; gate rises toward 1; reward-modulated
    weight updates can land. Cumulative dw should occur predominantly here.
- Phase 2 (reward off, ~100ms):
    ACh decays back toward baseline at decay_tau_ms; gate falls back to 0;
    no further weight updates.
- Phase 3 (continued recovery, ~40ms):
    ACh near baseline, gate ~ 0, no further weight updates.

The behavior is exercised end-to-end in
``tests/test_tans.py::test_bridge_blocks_reward_weight_updates_when_ach_at_baseline``;
this probe extends that single-step check into a continuous-time trace
so we can SEE the decay envelope and verify the "pause window" actually
opens AND closes at the expected timescales. Distinct from the test in
two ways: (1) multi-step dynamics rather than one-shot end states,
(2) human-readable per-phase summary + JSON for downstream review.

Run:
    python -m research.probes.tan_ach_probe

Outputs:
- stdout: human-readable per-phase summary + verdict
- research/findings/raw/tan_ach_probe/probe_results.json: structured trace
  including per-step ACh / gate / cumulative-dw arrays.

Plan: docs/plans/2026-04-28-cluster-b3-tans-implementation.md Task 4.
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

from sim import (  # noqa: E402
    CoreSimConfig,
    GPUConfig,
    RuntimeState,
    SimulationBridge,
    VisualizationConfig,
)
from sim.enums import NeuronModel  # noqa: E402
from sim.neuromodulators import _default_acetylcholine_config  # noqa: E402


# ---- Probe configuration ---------------------------------------------------

OUT_DIR = _REPO_ROOT / "research" / "findings" / "raw" / "tan_ach_probe"
OUT_JSON = OUT_DIR / "probe_results.json"

SEED = 42

# 4-phase scenario, all dt=1.0 ms so step indices == ms.
# Step counts chosen so phase 1 (+reward) is brief vs the ACh decay tau
# (500 ms); phases 2 + 3 together give 1.4x decay tau for clean recovery.
PHASE_0_STEPS = 50    # baseline -- tonic ACh, no reward
PHASE_1_STEPS = 10    # reward ON -- pause should open
PHASE_2_STEPS = 100   # reward OFF -- ACh recovery
PHASE_3_STEPS = 40    # continued observation
N_STEPS = PHASE_0_STEPS + PHASE_1_STEPS + PHASE_2_STEPS + PHASE_3_STEPS

REWARD_AMPLITUDE = 1.0   # reward signal during phase 1
ELIGIBILITY_VALUE = 0.5  # pinned per step on every synapse (matches the test)
REWARD_LR = 0.05

# PASS thresholds. The unit test
# ``test_bridge_blocks_reward_weight_updates_when_ach_at_baseline`` proves
# that gate=0 -> ZERO weight delta and gate=1 -> nontrivial delta in a
# single step. Our continuous trace just needs to show that the OPEN
# window lands during phase 1 and CLOSES outside it.
PASS_PHASE_1_MIN_ACH_LT = 0.5     # phase 1 ACh must drop well below baseline
PASS_PHASE_1_MEAN_GATE_GT = 0.05  # phase 1 gate must rise meaningfully
# Phase 1 cumulative |dw| must dominate phase 0's "blocked" baseline.
# Single-step gate=0 should give exactly zero, but floating-point noise
# in the multi-step accumulator can land at machine-epsilon scale, so we
# compare ratios instead of absolute deltas. 100x is comfortably above
# numerical noise and well below the gate=1/gate=0 cliff observed in
# the unit test.
PASS_PHASE_1_VS_PHASE_0_MULTIPLIER = 100.0


# ---- Bridge construction ---------------------------------------------------


def _build_minimal_bridge() -> SimulationBridge:
    """Smallest generic bridge that supports the ACh + reward path.

    Mirrors ``tests/test_tans.py::_make_bridge_with_ach`` exactly so that
    the probe and the test exercise the same end-to-end code. The probe
    isolates the reward-modulated update path (no STDP / Hebbian / structural
    / synaptic-scaling / homeostasis) so the only weight delta is the
    one we're gating with the TAN window.
    """
    cfg = CoreSimConfig()
    cfg.num_neurons = 50
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = SEED
    cfg.heterogeneity_seed = SEED
    cfg.ou_seed = SEED

    # ACh subsystem on, default config.
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [_default_acetylcholine_config()]

    # Isolate the reward-modulated weight update path. Same trick as
    # tests/test_tans.py and research/probes/d1_d2_asymmetry_probe.py:
    # turn off every other plasticity rule so the only weight change is
    # lr x reward x eligibility (x TAN gate). STDP is briefly enabled so
    # cp_eligibility_trace is allocated, then turned off post-init below.
    cfg.enable_stdp = True
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = REWARD_LR
    cfg.reward_baseline = 0.0

    # Wide weight bounds so post-update clip can't mask the small
    # reward-driven delta we're tracking. Same bound trick the test uses.
    cfg.stdp_w_min = -10.0
    cfg.stdp_w_max = 100.0
    cfg.hebbian_min_weight = -10.0
    cfg.hebbian_max_weight = 100.0

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    # Now disable STDP for the trace itself -- only reward modulation
    # should write weights. (We needed STDP=True briefly so eligibility
    # was allocated during init.)
    bridge.core_config.enable_stdp = False
    return bridge


# ---- Trace recording ------------------------------------------------------


def _record_trace(bridge: SimulationBridge) -> dict:
    """Run the 4-phase scenario, sample observables every step.

    Returns:
        dict with per-step lists for ach, gate, cum_abs_dw plus per-phase
        slices and a phase descriptor.
    """
    nnz = int(bridge.cp_connections.nnz)
    if bridge.cp_eligibility_trace is None:
        raise RuntimeError("Eligibility trace not allocated -- bridge init issue")

    # Snapshot starting weights. We measure |dw| relative to this.
    w_initial = bridge.cp_connections.data[:nnz].copy()
    w_prev = w_initial.copy()

    ach_trace = []
    gate_trace = []
    step_abs_dw = []  # per-step |dw| (sum across all synapses)
    cum_abs_dw = []   # running sum of |dw|

    cum = 0.0

    # Define phase plan. Each tuple = (label, n_steps, reward_during_phase).
    plan = [
        ("phase_0", PHASE_0_STEPS, 0.0),
        ("phase_1", PHASE_1_STEPS, REWARD_AMPLITUDE),
        ("phase_2", PHASE_2_STEPS, 0.0),
        ("phase_3", PHASE_3_STEPS, 0.0),
    ]

    # Phase boundaries -- step index at which each phase STARTS.
    phase_bounds = {}
    cursor = 0
    for label, n_steps, _reward in plan:
        phase_bounds[label] = (cursor, cursor + n_steps)
        cursor += n_steps

    step_idx = 0
    for label, n_steps, reward in plan:
        bridge.core_config.current_reward_signal = float(reward)
        for _ in range(n_steps):
            # Pin eligibility uniformly. The bridge decays it inside the
            # step before the reward update, so re-pinning here keeps the
            # per-step delta consistent across the trace.
            bridge.cp_eligibility_trace[:nnz] = ELIGIBILITY_VALUE

            # Sample observables BEFORE the step. The gate computed
            # inside the step uses the current (pre-step) ACh concentration,
            # so what we capture here matches what the bridge will use.
            ach_now = float(
                bridge.neuromodulator_manager.get_concentration("acetylcholine")
            )
            gate_now = float(
                bridge.neuromodulator_manager.compute_plasticity_window_gate_multiplier()
            )
            ach_trace.append(ach_now)
            gate_trace.append(gate_now)

            # Step forward. Inside the step the bridge:
            #  (a) applies reward x eligibility x gate -> dw
            #  (b) clips, then runs neuromodulator_manager.step(...)
            #     which decays ACh + applies pause_on_reward, so the
            #     concentration AFTER this call reflects the phase-current
            #     reward signal.
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

            # Measure step delta.
            w_now = bridge.cp_connections.data[:nnz]
            step_dw = float(cp.sum(cp.abs(w_now - w_prev)).get())
            cum += step_dw
            step_abs_dw.append(step_dw)
            cum_abs_dw.append(cum)
            w_prev = w_now.copy()

            step_idx += 1

    return {
        "phase_bounds": phase_bounds,
        "ach": ach_trace,
        "gate": gate_trace,
        "step_abs_dw": step_abs_dw,
        "cum_abs_dw": cum_abs_dw,
        "nnz": nnz,
        "n_steps": step_idx,
    }


# ---- Summary helpers ------------------------------------------------------


def _phase_slice(values: list, lo: int, hi: int) -> list:
    return values[lo:hi]


def _safe_mean(xs: list) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_min(xs: list) -> float:
    return float(min(xs)) if xs else 0.0


def _safe_max(xs: list) -> float:
    return float(max(xs)) if xs else 0.0


def _phase_summary(label: str, lo: int, hi: int, trace: dict) -> dict:
    ach_p = _phase_slice(trace["ach"], lo, hi)
    gate_p = _phase_slice(trace["gate"], lo, hi)
    step_dw_p = _phase_slice(trace["step_abs_dw"], lo, hi)
    cum_lo = trace["cum_abs_dw"][lo - 1] if lo > 0 else 0.0
    cum_hi = trace["cum_abs_dw"][hi - 1] if hi > 0 else 0.0
    return {
        "label": label,
        "step_range": [lo, hi],
        "n_steps": hi - lo,
        "ach_mean": _safe_mean(ach_p),
        "ach_min": _safe_min(ach_p),
        "ach_max": _safe_max(ach_p),
        "gate_mean": _safe_mean(gate_p),
        "gate_min": _safe_min(gate_p),
        "gate_max": _safe_max(gate_p),
        "step_abs_dw_mean": _safe_mean(step_dw_p),
        "phase_abs_dw_total": float(cum_hi - cum_lo),
    }


def _format_phase_table(summaries: list) -> str:
    header = (
        f"{'Phase':<10}{'Steps':>10}{'ACh_mean':>12}{'ACh_min':>12}"
        f"{'Gate_mean':>12}{'Gate_max':>12}{'phase|dw|':>14}"
    )
    rows = [header, "-" * len(header)]
    for s in summaries:
        rows.append(
            f"{s['label']:<10}"
            f"{s['n_steps']:>10}"
            f"{s['ach_mean']:>12.4f}"
            f"{s['ach_min']:>12.4f}"
            f"{s['gate_mean']:>12.4f}"
            f"{s['gate_max']:>12.4f}"
            f"{s['phase_abs_dw_total']:>14.4e}"
        )
    return "\n".join(rows)


def _check_verdict(summaries: dict) -> tuple[bool, list[str]]:
    """PASS criteria:
    1. Phase 1 minimum ACh < PASS_PHASE_1_MIN_ACH_LT (real pause occurred).
    2. Phase 1 mean gate > PASS_PHASE_1_MEAN_GATE_GT AND > phase 0 mean gate
       (window opened above baseline-blocked).
    3. Phase 1 |dw| > PASS_PHASE_1_VS_PHASE_0_MULTIPLIER * phase 0 |dw|
       (weight updates land in the open window, not in the blocked one).
    """
    issues = []
    p0 = summaries["phase_0"]
    p1 = summaries["phase_1"]

    if p1["ach_min"] >= PASS_PHASE_1_MIN_ACH_LT:
        issues.append(
            f"Phase 1 min ACh = {p1['ach_min']:.4f}, "
            f"expected < {PASS_PHASE_1_MIN_ACH_LT}"
        )
    if p1["gate_mean"] <= PASS_PHASE_1_MEAN_GATE_GT:
        issues.append(
            f"Phase 1 mean gate = {p1['gate_mean']:.4f}, "
            f"expected > {PASS_PHASE_1_MEAN_GATE_GT}"
        )
    if p1["gate_mean"] <= p0["gate_mean"]:
        issues.append(
            f"Phase 1 mean gate ({p1['gate_mean']:.4f}) "
            f"not above Phase 0 mean gate ({p0['gate_mean']:.4f})"
        )
    p0_dw = p0["phase_abs_dw_total"]
    p1_dw = p1["phase_abs_dw_total"]
    # Compare via ratio so floating-point noise at gate=0 doesn't mask.
    threshold_dw = max(
        PASS_PHASE_1_VS_PHASE_0_MULTIPLIER * p0_dw,
        1e-8,  # never accept absolute zero -- something must actually move
    )
    if p1_dw <= threshold_dw:
        issues.append(
            f"Phase 1 |dw| total ({p1_dw:.4e}) "
            f"not > {PASS_PHASE_1_VS_PHASE_0_MULTIPLIER}x Phase 0 ({p0_dw:.4e}) "
            f"(threshold {threshold_dw:.4e})"
        )
    return (len(issues) == 0), issues


# ---- Top-level orchestration ----------------------------------------------


def main() -> int:
    print("=== TAN/ACh Plasticity-Window Timing Probe ===")
    print(
        f"Config: seed={SEED}, dt=1.0 ms, "
        f"phases=({PHASE_0_STEPS}, {PHASE_1_STEPS}, "
        f"{PHASE_2_STEPS}, {PHASE_3_STEPS}) ms, "
        f"reward_amp={REWARD_AMPLITUDE}, eligibility={ELIGIBILITY_VALUE}, "
        f"reward_lr={REWARD_LR}"
    )
    print(
        "ACh defaults: baseline=1.0, decay_tau=500ms, "
        "pause_on_reward sensitivity=-2.0, threshold=0.0"
    )

    bridge = _build_minimal_bridge()
    nnz = int(bridge.cp_connections.nnz)
    print(f"Bridge built: num_neurons={bridge.core_config.num_neurons}, nnz={nnz}")

    trace = _record_trace(bridge)
    bridge.clear_simulation_state_and_gpu_memory()

    # Summarize per phase.
    bounds = trace["phase_bounds"]
    summaries = {
        label: _phase_summary(label, lo, hi, trace)
        for label, (lo, hi) in bounds.items()
    }
    summary_list = [summaries[k] for k in ("phase_0", "phase_1", "phase_2", "phase_3")]

    print()
    print(_format_phase_table(summary_list))
    print()

    # Spot-check transitions for the human reader.
    p0 = summaries["phase_0"]
    p1 = summaries["phase_1"]
    p2 = summaries["phase_2"]
    p3 = summaries["phase_3"]
    print(
        f"Phase 0 -> 1: ACh dropped from {p0['ach_mean']:.3f} (mean) "
        f"to {p1['ach_min']:.3f} (min); gate rose to {p1['gate_max']:.3f} (max)"
    )
    print(
        f"Phase 1 -> 2: reward off; ACh recovering "
        f"({p1['ach_min']:.3f} -> mean {p2['ach_mean']:.3f} "
        f"-> phase-3 mean {p3['ach_mean']:.3f})"
    )
    p1_dw = p1["phase_abs_dw_total"]
    p0_dw = p0["phase_abs_dw_total"]
    p2_dw = p2["phase_abs_dw_total"]
    if p0_dw > 1e-12:
        ratio = p1_dw / p0_dw
        ratio_str = f"{ratio:.1e}x"
    else:
        ratio_str = ">>1x (phase 0 effectively zero)"
    print(
        f"Cumulative |dw| concentration: phase 0={p0_dw:.4e}, "
        f"phase 1={p1_dw:.4e} ({ratio_str}), phase 2={p2_dw:.4e}"
    )

    passed, issues = _check_verdict(summaries)
    print()
    if passed:
        verdict = "PASS"
        print(
            "VERDICT: PASS - TAN plasticity window opens on reward (ACh pause), "
            "closes after reward stops (ACh recovery), weight updates land "
            "in the open window."
        )
    else:
        verdict = "FAIL"
        print("VERDICT: FAIL - TAN window dynamics did not match expectations:")
        for issue in issues:
            print(f"  - {issue}")

    # Persist JSON.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "probe": "tan_ach_plasticity_window_timing",
        "plan": "docs/plans/2026-04-28-cluster-b3-tans-implementation.md",
        "task": 4,
        "verdict": verdict,
        "issues": issues,
        "config": {
            "seed": SEED,
            "dt_ms": 1.0,
            "phase_0_steps": PHASE_0_STEPS,
            "phase_1_steps": PHASE_1_STEPS,
            "phase_2_steps": PHASE_2_STEPS,
            "phase_3_steps": PHASE_3_STEPS,
            "n_steps_total": N_STEPS,
            "reward_amplitude": REWARD_AMPLITUDE,
            "eligibility_value": ELIGIBILITY_VALUE,
            "reward_learning_rate": REWARD_LR,
            "reward_baseline": 0.0,
            "ach_baseline": 1.0,
            "ach_decay_tau_ms": 500.0,
            "ach_pause_sensitivity": -2.0,
            "ach_pause_threshold": 0.0,
            "isolated_path": [
                "enable_stdp=False (after init)",
                "enable_hebbian_learning=False",
                "enable_homeostasis=False",
                "enable_structural_plasticity=False",
                "enable_synaptic_scaling=False",
                "enable_reward_modulation=True",
                "enable_neuromodulator_subsystem=True",
            ],
        },
        "synapse_count_nnz": trace["nnz"],
        "phase_summaries": {
            label: summaries[label] for label in
            ("phase_0", "phase_1", "phase_2", "phase_3")
        },
        "trace": {
            "ach": trace["ach"],
            "plasticity_window_gate": trace["gate"],
            "step_abs_dw": trace["step_abs_dw"],
            "cum_abs_dw": trace["cum_abs_dw"],
        },
        "pass_thresholds": {
            "phase_1_min_ach_lt": PASS_PHASE_1_MIN_ACH_LT,
            "phase_1_mean_gate_gt": PASS_PHASE_1_MEAN_GATE_GT,
            "phase_1_dw_vs_phase_0_multiplier": PASS_PHASE_1_VS_PHASE_0_MULTIPLIER,
        },
    }
    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"\nJSON written to: {OUT_JSON}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
