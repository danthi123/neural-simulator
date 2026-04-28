"""Cluster B.1 biology probe — verify D1/D2 plasticity asymmetry.

Builds a minimal BG cascade bridge (n_cortex=20, the smallest the runner
builder supports), pins eligibility=+1.0 on every synapse, and steps the
sim under +reward and -reward conditions. Measures dw distributions on
synapses that terminate in str_D1_*, str_D2_*, and the rest.

Expected biological signature:
- Phase 1 (+reward): D1 weights ↑, D2 weights ↓
- Phase 2 (-reward): D1 weights ↓, D2 weights ↑
- "Other" synapses move with reward direction (sign=+1) like D1.

Run:
    python -m research.probes.d1_d2_asymmetry_probe

Outputs:
- stdout: human-readable distribution summary + verdict
- research/findings/raw/d1_d2_probe/probe_results.json: structured data

Plan: docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md Task 4.
"""

from __future__ import annotations

import json
import os
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


# Output destinations
OUT_DIR = _REPO_ROOT / "research" / "findings" / "raw" / "d1_d2_probe"
OUT_JSON = OUT_DIR / "probe_results.json"

N_STEPS_PER_PHASE = 50
REWARD_LR = 0.01
ELIGIBILITY_VALUE = 1.0


def _build_minimal_bridge() -> SimulationBridge:
    """Smallest BG cascade the builder supports (n_cortex=20, ~25 cortex/action)."""
    regions, pathways = build_bg_brain_regions(n_cortex=20)
    cfg = CoreSimConfig(
        num_neurons=1,  # placeholder; region_manager overrides
        enable_brain_region_framework=True,
        brain_regions=regions,
        region_pathways=pathways,
        enable_d1_d2_asymmetry=True,
    )
    # Isolate the reward-modulated weight update path. Mirrors the test
    # in tests/test_d1_d2_asymmetry.py — disable every other plasticity
    # rule so the only weight change is lr × RPE × eligibility × sign.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_structural_plasticity = False
    cfg.enable_synaptic_scaling = False
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = REWARD_LR
    cfg.reward_baseline = 0.0
    # Cortex→D1 weights start near 25; with default stdp_w_max=2 the
    # reward delta gets clipped immediately. Push bounds well above any
    # actual weight so we measure the raw update.
    cfg.stdp_w_max = 100.0
    cfg.hebbian_max_weight = 100.0

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _classify_synapses(bridge: SimulationBridge):
    """Return (d1_mask, d2_mask, other_mask) — boolean cupy arrays over nnz."""
    post = bridge.cp_connections.indices
    d1_neurons = []
    d2_neurons = []
    for action in ("N", "E", "S", "W"):
        d1_neurons.extend(bridge.region_manager.indices(f"str_D1_{action}"))
        d2_neurons.extend(bridge.region_manager.indices(f"str_D2_{action}"))
    d1_set = cp.asarray(d1_neurons, dtype=cp.int64)
    d2_set = cp.asarray(d2_neurons, dtype=cp.int64)
    d1_mask = cp.isin(post, d1_set)
    d2_mask = cp.isin(post, d2_set)
    other_mask = ~(d1_mask | d2_mask)
    return d1_mask, d2_mask, other_mask


def _summarize(delta: cp.ndarray, mask: cp.ndarray) -> dict:
    """Return summary stats {n, mean, std, min, max} on delta[mask]."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sel = delta[mask]
    return {
        "n": n,
        "mean": float(sel.mean().get()),
        "std": float(sel.std().get()),
        "min": float(sel.min().get()),
        "max": float(sel.max().get()),
    }


def _run_phase(
    bridge: SimulationBridge,
    reward: float,
    n_steps: int,
    nnz: int,
) -> cp.ndarray:
    """Pin eligibility to ELIGIBILITY_VALUE each step, set reward, step n_steps.

    Returns total dweights (cupy float32, shape (nnz,)).
    """
    bridge.core_config.current_reward_signal = reward
    w_before = bridge.cp_connections.data.copy()
    for _ in range(n_steps):
        # Pin eligibility to a uniform positive value before each step.
        # The bridge applies decay before the update inside the step,
        # so re-pinning here keeps the per-step delta nearly constant
        # (lr × reward × ELIGIBILITY_VALUE × sign).
        bridge.cp_eligibility_trace[:nnz] = ELIGIBILITY_VALUE
        bridge._run_one_simulation_step()
    w_after = bridge.cp_connections.data
    return w_after - w_before


def _format_phase(label: str, reward: float, summaries: dict) -> str:
    lines = [f"\nPhase {label}: {N_STEPS_PER_PHASE} steps with reward = {reward:+.1f}"]
    for name, expected in (("D1", "+" if reward > 0 else "-"),
                            ("D2", "-" if reward > 0 else "+"),
                            ("Other", "+" if reward > 0 else "-")):
        s = summaries[name]
        if s["n"] == 0:
            lines.append(f"  {name} synapses (N=0): no synapses found")
            continue
        lines.append(
            f"  {name} synapses (N={s['n']}): "
            f"mean dw={s['mean']:+.5f}, "
            f"std={s['std']:.5f}, "
            f"range=[{s['min']:+.5f}, {s['max']:+.5f}]  "
            f"<- expected {expected}"
        )
    return "\n".join(lines)


def _check_verdict(p1: dict, p2: dict) -> tuple[bool, list[str]]:
    """Returns (passed, list_of_issues)."""
    issues = []
    # Phase 1 (+reward): D1 mean > 0, D2 mean < 0
    if p1["D1"]["n"] > 0 and p1["D1"]["mean"] <= 0:
        issues.append(f"Phase 1: D1 mean dw = {p1['D1']['mean']:+.5f}, expected > 0")
    if p1["D2"]["n"] > 0 and p1["D2"]["mean"] >= 0:
        issues.append(f"Phase 1: D2 mean dw = {p1['D2']['mean']:+.5f}, expected < 0")
    # Phase 2 (-reward): D1 mean < 0, D2 mean > 0
    if p2["D1"]["n"] > 0 and p2["D1"]["mean"] >= 0:
        issues.append(f"Phase 2: D1 mean dw = {p2['D1']['mean']:+.5f}, expected < 0")
    if p2["D2"]["n"] > 0 and p2["D2"]["mean"] <= 0:
        issues.append(f"Phase 2: D2 mean dw = {p2['D2']['mean']:+.5f}, expected > 0")
    # Sign opposition between D1 and D2 within each phase
    if p1["D1"]["n"] > 0 and p1["D2"]["n"] > 0:
        if (p1["D1"]["mean"] > 0) == (p1["D2"]["mean"] > 0):
            issues.append("Phase 1: D1 and D2 deltas have the SAME sign (asymmetry failed)")
    if p2["D1"]["n"] > 0 and p2["D2"]["n"] > 0:
        if (p2["D1"]["mean"] > 0) == (p2["D2"]["mean"] > 0):
            issues.append("Phase 2: D1 and D2 deltas have the SAME sign (asymmetry failed)")
    return (len(issues) == 0), issues


def main() -> int:
    print("=== D1/D2 Plasticity Asymmetry Biology Probe ===")
    print(
        f"Config: n_cortex=20, n_steps_per_phase={N_STEPS_PER_PHASE}, "
        f"reward_lr={REWARD_LR}, eligibility={ELIGIBILITY_VALUE}, "
        f"stdp_w_max=100.0, reward_baseline=0.0"
    )

    bridge = _build_minimal_bridge()
    nnz = int(bridge.cp_connections.nnz)
    print(f"Bridge built: total neurons={bridge.core_config.num_neurons}, nnz={nnz}")

    d1_mask, d2_mask, other_mask = _classify_synapses(bridge)
    n_d1 = int(d1_mask.sum())
    n_d2 = int(d2_mask.sum())
    n_other = int(other_mask.sum())
    print(f"Synapse classification: D1={n_d1}, D2={n_d2}, other={n_other}")

    if n_d1 == 0 or n_d2 == 0:
        print("\n[ERROR] No D1 or D2 synapses found — cannot validate asymmetry.")
        bridge.clear_simulation_state_and_gpu_memory()
        return 2

    # Phase 1: +reward
    delta_pos = _run_phase(bridge, reward=+1.0, n_steps=N_STEPS_PER_PHASE, nnz=nnz)
    p1 = {
        "D1": _summarize(delta_pos, d1_mask),
        "D2": _summarize(delta_pos, d2_mask),
        "Other": _summarize(delta_pos, other_mask),
    }
    print(_format_phase("1", +1.0, p1))

    # Phase 2: -reward
    delta_neg = _run_phase(bridge, reward=-1.0, n_steps=N_STEPS_PER_PHASE, nnz=nnz)
    p2 = {
        "D1": _summarize(delta_neg, d1_mask),
        "D2": _summarize(delta_neg, d2_mask),
        "Other": _summarize(delta_neg, other_mask),
    }
    print(_format_phase("2", -1.0, p2))

    passed, issues = _check_verdict(p1, p2)
    if passed:
        verdict = "PASS"
        print(
            "\nVERDICT: PASS - asymmetry verified "
            "(D1/D2 deltas have opposite signs vs reward direction)"
        )
    else:
        verdict = "FAIL"
        print("\nVERDICT: FAIL - asymmetry not verified")
        for issue in issues:
            print(f"  - {issue}")

    # Write JSON record
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "probe": "d1_d2_asymmetry",
        "plan": "docs/plans/2026-04-28-cluster-b1-d1d2-asymmetry-implementation.md",
        "task": 4,
        "verdict": verdict,
        "issues": issues,
        "config": {
            "n_cortex": 20,
            "n_steps_per_phase": N_STEPS_PER_PHASE,
            "reward_learning_rate": REWARD_LR,
            "reward_baseline": 0.0,
            "eligibility_value": ELIGIBILITY_VALUE,
            "stdp_w_max": 100.0,
            "hebbian_max_weight": 100.0,
            "enable_d1_d2_asymmetry": True,
            "isolated_path": [
                "enable_stdp=False",
                "enable_hebbian_learning=False",
                "enable_homeostasis=False",
                "enable_structural_plasticity=False",
                "enable_synaptic_scaling=False",
                "enable_reward_modulation=True",
            ],
        },
        "synapse_counts": {"d1": n_d1, "d2": n_d2, "other": n_other, "total_nnz": nnz},
        "phase_1_positive_reward": {"reward": +1.0, "summaries": p1},
        "phase_2_negative_reward": {"reward": -1.0, "summaries": p2},
    }
    OUT_JSON.write_text(json.dumps(record, indent=2))
    print(f"\nJSON written to: {OUT_JSON}")

    bridge.clear_simulation_state_and_gpu_memory()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
