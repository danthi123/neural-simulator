"""DISTINGUISHING probe (option A): is the episode-LOOP per-step residual a GLOBAL cupy RNG draw (runner-fixable by
a per-step N-independent re-seed) OR a deterministic N-dependent FP reduction (sim/-only, unfixable runner-side)?

The bracket probe showed: episode-START state byte-identical (full-stack probe) + bare `_run_one_simulation_step`
byte-identical at 5 steps (1-step probe), BUT the FULL EPISODE LOOP diverges by step ~10 (~1e-5, chaotic growth). So
the residual is in the episode-LOOP per-step compute (the parts of run_moving_goal_episode OUTSIDE the bare bridge
step: retina render / sensory injection / action readout / reward / reward-STDP), and it's N-dependent.

THIS PROBE runs the episode loop TWICE per arm:
  (A) UNMODIFIED (the bracket condition) — expect divergence.
  (B) with the GLOBAL cupy RNG re-seeded to a FIXED N-independent value (`seed`) BEFORE every bridge step (monkeypatch
      _run_one_simulation_step to cp.random.seed(seed) first). If (B) makes the two arms byte-identical at step 50,
      the residual is a per-step GLOBAL-RNG draw (whose consumption desyncs across N) -> a RUNNER-SIDE per-step re-seed
      could fix it (no sim/ edit). If (B) is STILL divergent, the residual is a deterministic N-dependent FP reduction/
      matvec in the readout/render -> sim/-only (GPU non-associativity), NOT fixable runner-side.

Compares the nav-slice v at step 50 across arms, for both (A) and (B).

GPU-only. NO sim/ edit (the per-step re-seed is a PROBE monkeypatch, not a shipped change). Read-only diagnosis.

Run: SIM_BACKEND=cupy python -m research.runners._optionA_rng_vs_fp_probe
"""
from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

_CONV = frozenset({"parse_conj", "parse_role", "cortex_ctx", "dlpfc_wm", "rf", "drive_agrp", "drive_pomc"})
_CHECK = 50


def _is_conv(name):
    return (name in _CONV) or name.startswith("gen_")


class _Done(Exception):
    def __init__(self, v):
        self.v = v


def _run(rf_n, reseed_each_step, seed=42, grid_size=16):
    from research.runners.g11_bg_runner import run_moving_goal_episode
    from research.runners.nav_conv_merged_bridge import (
        conv_extra_regions_pathways, finalize_conv_for_nav_gate)
    from sim.regions import BrainRegion
    from sim.backend import to_host
    import cupy as cp

    extra_regions, extra_pathways = conv_extra_regions_pathways(co_resident_rf=True, rf_D=128)
    resized = [BrainRegion(name="rf", n_neurons=int(rf_n), exc_fraction=1.0, internal_density=0.0, enable_nmda=False)
               if r.name == "rf" else r for r in extra_regions]
    state = {"count": 0}

    def hook(bridge):
        finalize_conv_for_nav_gate(bridge, seed=seed)
        rm = bridge.region_manager
        nav_names = [nm for nm in rm.region_indices_dict() if not _is_conv(nm)]
        nav_idx = np.concatenate([np.asarray(rm.indices(nm), dtype=np.int64) for nm in nav_names])
        orig = bridge._run_one_simulation_step

        def wrapped(*a, **k):
            if reseed_each_step:
                cp.random.seed(int(seed))   # force the global cupy stream to an N-INDEPENDENT state each step
            r = orig(*a, **k)
            state["count"] += 1
            if state["count"] >= _CHECK:
                raise _Done(np.asarray(to_host(bridge.cp_membrane_potential_v[nav_idx])).copy())
            return r
        bridge._run_one_simulation_step = wrapped

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
    except _Done as e:
        return e.v


def main():
    from sim.backend import is_gpu_backend
    if not is_gpu_backend():
        raise SystemExit("needs CuPy/GPU")
    import cupy as cp

    res = {}
    for label, reseed in (("A_unmodified", False), ("B_reseed_each_step", True)):
        v1 = _run(896, reseed, seed=42); cp.get_default_memory_pool().free_all_blocks()
        v2 = _run(24051, reseed, seed=42); cp.get_default_memory_pool().free_all_blocks()
        d = float(np.max(np.abs(v2 - v1))) if v1.shape == v2.shape else float("inf")
        res[label] = {"max_abs_dv_at_step50": d, "identical": bool(d == 0.0)}
        print(f"[rng_vs_fp] {label}: max|dv|@step{_CHECK}={d:.3e} identical={d == 0.0}")

    a = res["A_unmodified"]; b = res["B_reseed_each_step"]
    if a["identical"]:
        verdict = "INCONCLUSIVE — even the unmodified arm is identical at step 50 (re-check)."
    elif b["identical"]:
        verdict = ("RESIDUAL = a per-step GLOBAL cupy RNG draw (N-dependent consumption): forcing an N-independent "
                   "per-step re-seed makes the arms byte-identical -> a RUNNER-SIDE per-step re-seed could close it "
                   "(NO sim/ edit). BUT a blanket per-step re-seed would change the episode's stochasticity globally; "
                   "the proper fix is to make THAT specific draw N-independent — likely a small sim/ change. FLAG.")
    else:
        verdict = ("RESIDUAL = a DETERMINISTIC N-dependent FP reduction/matvec in the episode-loop readout/render "
                   "(GPU non-associativity): re-seeding the global RNG does NOT help -> NOT fixable by any runner-side "
                   "RNG/state reset. Closing it needs an N-independent reduction (a sim/ edit). The episode-start-state "
                   "reset (recommendation a) cannot achieve byte-identity here. DO NOT FLIP; report the boundary.")
    out = {"probe": "optionA_rng_vs_fp_probe", "check_step": _CHECK, "results": res, "verdict": verdict}
    path = "research/findings/raw/_optionA_rng_vs_fp_probe.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[rng_vs_fp] VERDICT: {verdict}")
    print(f"[rng_vs_fp] wrote {path}")


if __name__ == "__main__":
    main()
