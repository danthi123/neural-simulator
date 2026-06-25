"""ONE-STEP root-cause probe (option A): is the residual GPU floating-point NON-ASSOCIATIVITY (size-dependent
sparse matvec), i.e. UNFIXABLE runner-side?

The full-stack probe confirmed the EPISODE-START state (all per-neuron arrays incl. thresholds + the global cupy RNG
state) is BYTE-IDENTICAL across N after finalize_conv_for_nav_gate's resets. Yet the full-stack episode STILL diverges
(seed 42 step-100 positions differ). With identical start-state + identical RNG + OU OFF, the ONLY remaining source is
the per-step compute itself producing size-dependent results — i.e. the synaptic-current sparse matvec / reductions over
a DIFFERENT total-N bridge use a different cuSPARSE/reduction tiling -> tiny (~1e-6) differences in the SAME nav-slice
neurons, which the chaotic spiking-WTA dynamics amplify over 200 steps. CUBLAS_WORKSPACE_CONFIG enforces cuBLAS
determinism for the SAME problem but NOT across different matrix sizes (cuSPARSE).

THIS PROBE: build BOTH arms (rf=896 vs rf=24051), run the full reset (finalize_conv_for_nav_gate), then run EXACTLY ONE
`_run_one_simulation_step()` (OU off, no per-step RNG) and compare the NAV-slice cp_membrane_potential_v. If it diverges
at step 1 from a byte-identical start, the residual is intrinsic GPU non-associativity (size-dependent matvec) -> NOT
fixable by any runner-side state reset -> the honest boundary. If it is byte-identical at step 1 but diverges later, the
residual is some slower-accumulating state (re-investigate).

GPU-only. NO sim/ edit. Read-only diagnosis.

Run: SIM_BACKEND=cupy python -m research.runners._optionA_onestep_probe
"""
from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

_CONV = frozenset({"parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm", "rf", "drive_agrp", "drive_pomc"})


def _is_conv(name):
    return (name in _CONV) or name.startswith("gen_")


class _StopAfterNSteps(Exception):
    def __init__(self, vsnap):
        self.vsnap = vsnap


def _build_and_step(rf_n, n_extra_steps, seed=42, grid_size=16):
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate)
    from sim.regions import BrainRegion
    from sim.backend import to_host

    extra_regions, extra_pathways = conv_extra_regions_pathways(co_resident_rf=True, rf_D=128)
    resized = [BrainRegion(name="rf", n_neurons=int(rf_n), exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
               if r.name == "rf" else r for r in extra_regions]

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)
        rm = bridge.region_manager
        nav_names = [nm for nm in rm.region_indices_dict() if not _is_conv(nm)]
        nav_idx = np.concatenate([np.asarray(rm.indices(nm), dtype=np.int64) for nm in nav_names])
        # snapshot v at step 0 (post-reset), then run n_extra_steps deterministic steps (OU off), snapshot again.
        v0 = np.asarray(to_host(bridge.cp_membrane_potential_v[nav_idx])).copy()
        for _ in range(int(n_extra_steps)):
            bridge._run_one_simulation_step()
        vN = np.asarray(to_host(bridge.cp_membrane_potential_v[nav_idx])).copy()
        raise _StopAfterNSteps((nav_idx, v0, vN))

    def _sched(gs):
        return [(0, (gs - 2, gs - 2)), (50, (1, gs - 2)), (100, (1, 1)), (150, (gs - 2, 1))]

    try:
        run_moving_goal_episode(
            out_path=os.path.join(tempfile.mkdtemp(), "x.json"), seed=seed, n_steps=200, grid_size=grid_size,
            goal_schedule=_sched(grid_size),
            enable_d1_d2_asymmetry=True, enable_striatal_fsis=True,
            enable_cluster_a_closed_loop=True, enable_cluster_e_topography=True,
            enable_pfc_nmda=True, enable_visual_cortex=True, visual_cortex_action_warmup_steps=60,
            stdp_w_max_override=400.0,
            extra_regions=resized, extra_pathways=extra_pathways,
            build_with_ou=True, prebuilt_post_init_hook=hook,
        )
    except _StopAfterNSteps as e:
        return e.vsnap


def main():
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        raise SystemExit("needs CuPy/GPU")
    import cupy as cp

    results = {}
    for nsteps in (1, 5):
        idx1, v0_1, vN_1 = _build_and_step(896, nsteps, seed=42)
        cp.get_default_memory_pool().free_all_blocks()
        idx2, v0_2, vN_2 = _build_and_step(24051, nsteps, seed=42)
        cp.get_default_memory_pool().free_all_blocks()
        start_identical = bool(np.array_equal(v0_1, v0_2))
        max_abs_start = float(np.max(np.abs(v0_1 - v0_2))) if v0_1.shape == v0_2.shape else float("inf")
        after_identical = bool(np.array_equal(vN_1, vN_2))
        max_abs_after = float(np.max(np.abs(vN_1 - vN_2))) if vN_1.shape == vN_2.shape else float("inf")
        results[f"{nsteps}_steps"] = {
            "nav_start_v_byte_identical": start_identical, "max_abs_start_dv": max_abs_start,
            "nav_v_byte_identical_after": after_identical, "max_abs_after_dv": max_abs_after}
        print(f"[onestep] {nsteps} step(s): start_identical={start_identical} (max|dv0|={max_abs_start:.2e}) -> "
              f"after_identical={after_identical} (max|dvN|={max_abs_after:.2e})")

    one = results["1_steps"]
    verdict = (
        "INTRINSIC GPU NON-ASSOCIATIVITY -- start-state byte-identical but ONE deterministic step (OU off, no RNG) "
        "diverges -> the size-dependent sparse-matvec/reduction over a different total-N bridge perturbs the SAME "
        "nav-slice neurons at FP-noise level; NOT fixable by any runner-side state reset (the honest boundary)."
        if (one["nav_start_v_byte_identical"] and not one["nav_v_byte_identical_after"]) else
        ("start-state NOT identical -> a per-neuron residual remains (re-investigate the reset)"
         if not one["nav_start_v_byte_identical"] else
         "byte-identical after 1 step -- the divergence accumulates later; re-investigate slower state"))
    out = {"probe": "optionA_onestep_probe", "results": results, "verdict": verdict}
    path = "research/findings/raw/_optionA_onestep_probe.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[onestep] VERDICT: {verdict}")
    print(f"[onestep] wrote {path}")


if __name__ == "__main__":
    main()
