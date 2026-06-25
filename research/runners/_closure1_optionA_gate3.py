"""CLOSE-OUT (option A gate 3) — the HARD prerequisite for flipping the merged composer default to `onebrain`.

Per `research/findings/raw/_closeout_to_full_capacity_audit.md` Closure 1: gates 1+2 (15/15) + gate 4 (limbic,
GPU) are GREEN; the only OUTSTANDING gate is **nav-not-regressed Δ=0** with the onebrain co-resident composer.

THE GATE (the same null/inertness logic as step-2b's "nav-not-regressed = 2.0 byte-identical"):

  The co-resident composer runs on the merged bridge's `rf` slice, which is array-DISJOINT from the spiking nav
  read-out (the `rf` region has ZERO `cp_connections` out-edges into the nav cascade -- the Task-1 anti-cheat),
  and the composer is never stepped DURING a nav episode. So the composer KIND ('onebrain' vs 'rf') can only
  change the navigable bridge through ONE structural variable: the SIZE of that disjoint, out-edge-free `rf`
  region ('onebrain' needs CoResidentOneBrainComposer.n_total_for(...) neurons -- the k_max store blocks +
  batched Q/cleanup; 'rf' needs 7*rf_D). A disjoint, silent, out-edge-free index block CANNOT perturb the nav
  score -> Δ=0 is the expected GO.

This runner verifies that in TWO parts:

  PART 1 (construct both REAL agents): build MergedNavConvAgent(co_resident_composer=True,
    co_resident_composer_kind={'onebrain','rf'}) on GPU and assert the ONLY structural difference in the
    navigable bridge is the rf-region size (every nav region byte-identical in name + size). This honors
    "construct the merged agent BOTH ways" and proves the onebrain path is real (CoResidentOneBrainComposer +
    persistent_loop + the larger rf region all activate).

  PART 2 (the nav score, Δ=0): run the VALIDATED gate-2a navigation episode (run_moving_goal_episode +
    conv_extra_regions_pathways(co_resident_rf=True) + finalize_conv_for_nav_gate -- the exact step-2b harness)
    twice per seed, once with the `rf` region sized at the ONEBRAIN size and once at the RF size, and assert the
    per-phase-sum nav score (the gate-2a metric) is BYTE-IDENTICAL (Δ=0). This isolates exactly the variable the
    composer kind changes on a navigable bridge, through the score-producing harness.

GO bar: max |Δ| over seeds == 0.0 (byte-identical; the disjoint slice is inert). If any seed has Δ != 0, that
is a REAL regression (the onebrain rf slice perturbing the nav read-out) -> propagate the honest NEGATIVE, DO
NOT flip.

GPU-only (SIM_BACKEND=cupy). Reuse-by-import; NO sim/ edit.

Run: SIM_BACKEND=cupy python -m research.runners._closure1_optionA_gate3 --seeds 42,43,44 --grid-size 16 --n-steps 900
"""
from __future__ import annotations

import os

# MUST precede any CuPy import so both rf-size arms are reproducible (matches the standalone flagship --deterministic).
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import tempfile

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_closure1_optionA_gate3_flip.json")
RF_D = 128
K_MAX = 32


def _score_from_file(path):
    """The gate-2a nav score: sum over the 4 goal phases of final_quarter_mean_distance (lower = closer to goal)."""
    with open(path) as f:
        data = json.load(f)
    phases = data.get("phase_stats")
    if not phases:
        raise ValueError(f"{path}: no phase_stats")
    return sum(float(p["final_quarter_mean_distance"]) for p in phases), [
        float(p["final_quarter_mean_distance"]) for p in phases]


def _goal_schedule(gs):
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    return [(0, far), (450, far_west), (900, sw), (1350, far_se)]


def _run_nav_with_rf_size(rf_n, seed, grid_size, n_steps, out_path):
    """Run the validated gate-2a navigation episode with the conversational `rf` region sized at `rf_n` neurons
    (everything else -- the full nav cascade, the parser+dlPFC, the finalize hook -- identical). The composer kind
    differs from the agent ONLY by this rf-region size, so this is the exact bridge-level delta, run through the
    score-producing harness."""
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate,
    )
    from sim.regions import BrainRegion

    extra_regions, extra_pathways = conv_extra_regions_pathways(co_resident_rf=True, rf_D=RF_D)
    # conv_extra_regions_pathways(co_resident_rf=True) appends an `rf` region of 7*rf_D; resize it to rf_n so the
    # two arms differ ONLY in the disjoint, out-edge-free rf slice (the onebrain vs rf composer-kind delta).
    resized = []
    for r in extra_regions:
        if r.name == "rf":
            resized.append(BrainRegion(name="rf", n_neurons=int(rf_n), exc_fraction=1.0,
                                       internal_density=0.0, enable_nmda=False))
        else:
            resized.append(r)
    assert any(r.name == "rf" and r.n_neurons == int(rf_n) for r in resized), "rf region not resized"

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)

    run_moving_goal_episode(
        out_path=out_path, seed=seed, n_steps=n_steps, grid_size=grid_size,
        goal_schedule=_goal_schedule(grid_size),
        enable_d1_d2_asymmetry=True, enable_striatal_fsis=True,
        enable_cluster_a_closed_loop=True, enable_cluster_e_topography=True,
        enable_pfc_nmda=True, enable_visual_cortex=True, visual_cortex_action_warmup_steps=600,
        stdp_w_max_override=400.0,
        extra_regions=resized, extra_pathways=extra_pathways,
        build_with_ou=True, prebuilt_post_init_hook=hook,
    )
    return _score_from_file(out_path)


def part1_construct_both_agents(seed=42):
    """Construct BOTH real agents on GPU; assert the ONLY navigable-bridge difference is the rf-region size."""
    import cupy as cp
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent, CoResidentOneBrainComposer, MergedRFComposer

    def _region_sizes(agent):
        rm = agent._merged_bridge.region_manager
        return {name: len(list(rm.indices(name))) for name in rm.region_indices_dict()}

    a_rf = MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="rf")
    rf_sizes = _region_sizes(a_rf)
    rf_N = int(a_rf._merged_bridge.core_config.num_neurons)
    rf_composer_ok = isinstance(a_rf.composer, MergedRFComposer)
    rf_rf_size = rf_sizes.get("rf")
    del a_rf
    cp.get_default_memory_pool().free_all_blocks()

    # The onebrain arm may FAIL TO CONSTRUCT under the production defaults (co_resident_command_route=True): the
    # idle layout-only BridgeParser the CoResidentOneBrainComposer builds re-injects via
    # merge_population_into_shared_bridge, which wipes the framework wiring + the COMMAND_GATE transmission-gate
    # registration -> the agent's __init__ COMMAND_GATE anti-cheat assert fails. Capture that as the BLOCKED finding
    # (gate 3 cannot run: one arm does not build) rather than dying.
    import traceback as _tb
    onebrain_construct_error = None
    ob_sizes = ob_N = ob_composer_ok = ob_persistent_loop = ob_rf_size = vocab_used = None
    try:
        a_ob = MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="onebrain")
        ob_sizes = _region_sizes(a_ob)
        ob_N = int(a_ob._merged_bridge.core_config.num_neurons)
        ob_composer_ok = isinstance(a_ob.composer, CoResidentOneBrainComposer)
        ob_persistent_loop = bool(getattr(a_ob.composer, "persistent_loop", False))
        ob_rf_size = ob_sizes.get("rf")
        vocab_used = a_ob._handles.get("vocab")
        del a_ob
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as e:
        onebrain_construct_error = f"{type(e).__name__}: {e}"
        print(f"[gate3] ONEBRAIN AGENT FAILED TO CONSTRUCT: {onebrain_construct_error}", flush=True)
        _tb.print_exc()
        cp.get_default_memory_pool().free_all_blocks()
        return {
            "rf_composer_is_MergedRFComposer": rf_composer_ok,
            "rf_kind_rf_region_size": rf_rf_size,
            "merged_N_rf_kind": rf_N,
            "onebrain_constructs": False,
            "onebrain_construct_error": onebrain_construct_error,
            "nav_regions_identical": None,
        }

    # the ONLY region that may differ in size is `rf`; every other region must be byte-identical in name + size.
    nav_regions_identical = True
    diffs = []
    all_names = set(rf_sizes) | set(ob_sizes)
    for name in sorted(all_names):
        if name == "rf":
            continue
        if rf_sizes.get(name) != ob_sizes.get(name):
            nav_regions_identical = False
            diffs.append({"region": name, "rf": rf_sizes.get(name), "onebrain": ob_sizes.get(name)})

    return {
        "rf_composer_is_MergedRFComposer": rf_composer_ok,
        "onebrain_constructs": True,
        "onebrain_construct_error": None,
        "onebrain_composer_is_CoResidentOneBrainComposer": ob_composer_ok,
        "onebrain_persistent_loop_on": ob_persistent_loop,
        "rf_kind_rf_region_size": rf_rf_size,
        "onebrain_kind_rf_region_size": ob_rf_size,
        "expected_onebrain_rf_size": int(CoResidentOneBrainComposer.n_total_for(
            D=RF_D, vocab=vocab_used, k_max=K_MAX, enable_attributed=False)),
        "expected_rf_kind_rf_size": 7 * RF_D,
        "merged_N_rf_kind": rf_N,
        "merged_N_onebrain_kind": ob_N,
        "structural_delta_N": ob_N - rf_N,
        "rf_region_size_delta": (ob_rf_size - rf_rf_size) if (ob_rf_size and rf_rf_size) else None,
        "nav_regions_identical": nav_regions_identical,
        "nav_region_diffs": diffs,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--grid-size", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=900)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        raise SystemExit("gate 3 needs the CuPy/GPU backend (SIM_BACKEND=cupy)")

    # PART 1: construct both real agents, assert the navigable bridge differs ONLY by the rf-region size.
    print("[gate3] PART 1: constructing BOTH agents (co_resident_composer=True, kind=rf / onebrain)...", flush=True)
    part1 = part1_construct_both_agents(seed=seeds[0])
    print(f"[gate3] PART 1: {json.dumps(part1, indent=2)}", flush=True)

    # BLOCKED: if the onebrain arm does not construct (the COMMAND_GATE wipe under co_resident_command_route=True),
    # gate 3 CANNOT RUN -- one arm has no nav score. Write the BLOCKED verdict + STOP (do NOT flip).
    if not part1.get("onebrain_constructs", False):
        result = {
            "gate": "option A gate 3 -- nav-not-regressed Δ=0 (onebrain co-resident composer)",
            "backend": "cupy", "grid_size": args.grid_size, "n_steps": args.n_steps, "seeds": seeds,
            "part1_construct_both_agents": part1,
            "gate3_pass": False,
            "verdict": ("BLOCKED-BY-CONSTRUCTION -- the onebrain co-resident composer does NOT construct under the "
                        "production default co_resident_command_route=True (the idle layout-only BridgeParser the "
                        "CoResidentOneBrainComposer builds re-injects via merge_population_into_shared_bridge, wiping "
                        "the framework wiring + the COMMAND_GATE transmission-gate registration -> the agent __init__ "
                        "COMMAND_GATE anti-cheat assert fails). Gate 3 cannot produce a nav score for the onebrain arm. "
                        "DO NOT FLIP co_resident_composer_kind to 'onebrain' -- it would crash test_nav_conv_step2b_"
                        "coresident.py's default fixture. Fix = a runner-level build (skip the idle parser merge / "
                        "preserve the framework wiring), NOT a flip."),
        }
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[gate3] VERDICT: {result['verdict']}")
        print(f"[gate3] wrote {os.path.normpath(OUT)}")
        return 2

    rf_n = part1["onebrain_kind_rf_region_size"]      # the ACTUAL onebrain rf size from the real agent
    rf_rf_n = part1["rf_kind_rf_region_size"]

    # PART 2: run the nav score for the onebrain rf-size vs the rf rf-size, assert Δ=0 per seed.
    print(f"[gate3] PART 2: nav score (onebrain rf={rf_n} vs rf rf={rf_rf_n}) over seeds {seeds} "
          f"grid={args.grid_size} n_steps={args.n_steps}...", flush=True)
    rows = []
    tmpdir = tempfile.mkdtemp(prefix="gate3_nav_")
    max_abs_delta = 0.0
    for seed in seeds:
        ob_out = os.path.join(tmpdir, f"onebrain_seed{seed}.json")
        rf_out = os.path.join(tmpdir, f"rf_seed{seed}.json")
        ob_score, ob_phases = _run_nav_with_rf_size(rf_n, seed, args.grid_size, args.n_steps, ob_out)
        rf_score, rf_phases = _run_nav_with_rf_size(rf_rf_n, seed, args.grid_size, args.n_steps, rf_out)
        delta = ob_score - rf_score
        max_abs_delta = max(max_abs_delta, abs(delta))
        row = {"seed": seed, "onebrain_score": ob_score, "rf_score": rf_score, "delta": delta,
               "byte_identical": (delta == 0.0), "onebrain_phases": ob_phases, "rf_phases": rf_phases}
        rows.append(row)
        print(f"[gate3]   seed {seed}: onebrain={ob_score:.6f} rf={rf_score:.6f} Δ={delta:+.6e} "
              f"{'byte-identical' if delta == 0.0 else 'DELTA!=0'}", flush=True)

    gate3_pass = (max_abs_delta == 0.0)
    result = {
        "gate": "option A gate 3 -- nav-not-regressed Δ=0 (onebrain co-resident composer)",
        "backend": "cupy",
        "logic": ("the co-resident composer's `rf` slice is array-disjoint from the spiking nav read-out (zero "
                  "out-edges into nav) and idle during a nav episode -> the composer KIND changes the navigable "
                  "bridge ONLY by the rf-region size; a disjoint silent out-edge-free block cannot perturb the nav "
                  "score -> Δ=0 expected (same null-gate logic as step-2b nav-not-regressed = 2.0 byte-identical)."),
        "grid_size": args.grid_size, "n_steps": args.n_steps, "seeds": seeds,
        "part1_construct_both_agents": part1,
        "onebrain_rf_region_size": rf_n, "rf_kind_rf_region_size": rf_rf_n,
        "rows": rows,
        "max_abs_delta": max_abs_delta,
        "n_byte_identical": sum(1 for r in rows if r["byte_identical"]),
        "n_seeds": len(rows),
        "gate3_pass": gate3_pass,
        "verdict": ("DELTA0_GO -- flip co_resident_composer_kind default rf->onebrain"
                    if gate3_pass else "REGRESS -- DO NOT FLIP (onebrain rf slice perturbs nav; report the NEGATIVE)"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[gate3] VERDICT: {result['verdict']}")
    print(f"[gate3] max|Δ|={max_abs_delta:.6e}  byte-identical {result['n_byte_identical']}/{result['n_seeds']} seeds")
    print(f"[gate3] wrote {os.path.normpath(OUT)}")
    return 0 if gate3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
