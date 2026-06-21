"""CLOSE A de-risk for shortcut #5b (2026-06-22).

Per the scoping `research/findings/2026-06-21-shortcut5b-sparse-place-fields-scoping.md`:
the self-org place code's value-train δ is FLAT at nav scale. The prior sparsify sweep
(2026-06-19-place-code-sparsify-default-BOUNDARY.md) FIXED value-LEARNING (LEARNS-V 1.01→1.91x
at place_sensors_to_place_weight=10) but the all-or-none coincidence-plateau READ-OUT is binary
(under-discriminates at low weights → flat δ~1.04; over-clamps at high → δ→0.0). The scoping's
ONE genuinely unexploited cheap close (CLOSE A): wire the already-SHIPPED, validated GRADED
dendritic plateau read-out (`enable_graded_dendritic_plateau`, commit d69cc0ab — gave a clean
monotone ~9× near/far V on the host-Gaussian afferent, 3/3 seeds) onto the SPARSE self-org afferent.

The runner GATES the graded plateau OFF under self-org (`g11_bg_runner.py:4496`:
`if dendrite_critic and enable_neural_critic and not _neural_place_selforg`). This STANDALONE probe
bypasses that gate by flipping the cfg flags DIRECTLY (NO `sim/` edit, NO `g11_bg_runner.py` edit) via
an init-time monkeypatch on `SimulationBridge._initialize_simulation_data` that sets, on `self.core_config`,
just before the real init (so the array alloc + per-step block see them):
    enable_graded_dendritic_plateau = True
    coincidence_plateau_strength    = 0.0   # all-or-none plateau OFF → graded form carries V
    graded_plateau_center/slope/strength/tau = the VALIDATED deploy values (1.5/1.0/80.0/80ms/2ms)
The graded-plateau per-step block (sim/bridge.py:6399) routes on the EXISTING `coincidence_detector`
mask (the `place→striosome_value` pathway, already allocated under self-org) — NO new wiring — and is
INDEPENDENT of `coincidence_weighted_drive` (it always does the WEIGHTED matvec, which is the point of
a VALUE read-out). So the only change vs the documented W=10 flat-δ baseline is the READ-OUT transfer
(graded logistic vs all-or-none switch).

THE FORK (the decisive measurement):
  GENUINE CLOSE: δ > 1.3 + monotone near/far separation (critic@near ≫ critic@far) → CLOSE A works;
                 the binary read-out was the residual (R2); the host-Gaussian scaffold RETIRES.
  R1-LIMIT     : the overlapping self-org afferent (R1) → overlapping V → flat δ DESPITE the graded
                 read-out → afferent selectivity is the genuine substrate-limit residual; the graded
                 read-out solved R2 but R1 caps it.

ANTI-CHEAT arms (all on the SAME stage-B-smoke δ pipeline):
  * test     : W=10 sparse self-org + graded read-out          (the test)
  * lesion   : test + graded_plateau_strength=0                (graded read-out lesioned → δ collapses)
  * no_learn : test + value_train_trials=0                     (no value-train → V flat, no near>far)
  * dense    : W=28 dense self-org + graded read-out           (no-sparsification control → stay flat;
                                                                 isolates that graded needs the sparse code)
  The HOST-GAUSSIAN positive control (must reproduce ~9× near/far V) is the already-validated
  `_dendrite_stage1_onbridge_graded_plateau.py` runner — run it separately + cite it here.

Usage (GPU faithful; CPU smoke first):
  SIM_BACKEND=numpy python -m research.runners._n5_closeA_graded_on_selforg_probe --seed 42 --arm test --value-train-trials 12
  SIM_BACKEND=cupy  python -m research.runners._n5_closeA_graded_on_selforg_probe --seed 42 --arm test
  SIM_BACKEND=cupy  python -m research.runners._n5_closeA_graded_on_selforg_probe --seed 42 --all-arms --out <out.json>
"""
import os
import sys
import json
import argparse

import research.runners.g11_bg_runner as g
from sim.bridge import SimulationBridge


# ── the CLOSE-A cfg-flip: install the graded plateau on the self-org afferent ────────────────
# A module-level holder the init-monkeypatch reads (the graded params for THIS run) and writes
# (a reference to the constructed bridge, for an optional post-run graded-V diagnostic read).
_CLOSEA = {
    "enable": False,          # flip the graded plateau on (the test); off = a clean control rerun
    "center": 1.5,            # validated logistic center (c_w / WEIGHT units) — deploy default
    "slope": 1.0,             # validated smooth (non-saturating) slope
    "strength": 80.0,         # validated per-step plateau conductance scale (0 = lesion arm)
    "tau_decay_ms": 80.0,     # the slow NMDA-plateau tau (deploy default)
    "tau_rise_ms": 2.0,
    "bridge": None,           # captured for diagnostics
}

_orig_init = SimulationBridge._initialize_simulation_data


def _patched_init(self, *a, **kw):
    if _CLOSEA["enable"]:
        c = self.core_config
        # the CLOSE-A read-out swap (the deploy block g11_bg_runner.py:4510-4521, but UNDER self-org):
        c.enable_graded_dendritic_plateau = True
        c.coincidence_plateau_strength = 0.0     # all-or-none plateau OFF; graded form carries V
        c.graded_plateau_center = float(_CLOSEA["center"])
        c.graded_plateau_slope = float(_CLOSEA["slope"])
        c.graded_plateau_strength = float(_CLOSEA["strength"])
        c.graded_plateau_tau_decay_ms = float(_CLOSEA["tau_decay_ms"])
        c.graded_plateau_tau_rise_ms = float(_CLOSEA["tau_rise_ms"])
        # enable_nmda already set True by the self-org build block (per-region mask → critic only).
    out = _orig_init(self, *a, **kw)
    _CLOSEA["bridge"] = self
    return out


def _capture_deployed_kwargs(seed, value_train_trials, *, stage_b=True):
    """Run g.main() with the NEGATIVE-repro argv, intercepting run_moving_goal_episode so nothing
    heavy runs, to capture the EXACT deployed kwargs (the _n5_place_sparsify_probe pattern)."""
    captured = {}
    real_fn = g.run_moving_goal_episode

    def _intercept(*args, **kwargs):
        captured.update(kwargs)
        return {"_intercepted": True}

    g.run_moving_goal_episode = _intercept
    argv = [
        "g11", "--moving-goal", "--goal-schedule", "multi", "--deterministic",
        "--enable-neural-critic", "--spiking-reward-us", "--enable-critic-homeostasis",
        "--enable-critic-fs-inhibition", "--critic-fs-weight", "16",
        "--neural-place-selforg", "--deterministic-selforg",
        ("--stage-b-smoke" if stage_b else "--stage-a-smoke"),
        "--value-train-trials", str(value_train_trials),
        "--seed", str(seed),
        "--no-emit-webapp-sidecar",
    ]
    saved_argv = sys.argv
    try:
        sys.argv = argv
        g.main()
    finally:
        sys.argv = saved_argv
        g.run_moving_goal_episode = real_fn
    if not captured:
        raise RuntimeError("failed to capture deployed kwargs")
    return captured, real_fn


# arm → (graded_enabled, override-dict applied on the captured kwargs).
# `single_goal=True` (the default) trains ONLY the first scheduled goal (critic_warmup_all_goals=False)
# — the finding's CLEAN capability test (multi-goal makes far=(1,1) ALSO a goal → a probe-contrast
# artifact that inflates w_far). All arms share the single-vs-multi choice so the comparison is fair.
def _arm_spec(arm, w_sparse, w_dense, value_train_trials, single_goal):
    base = {"place_sensors_to_place_weight": w_sparse,
            "critic_warmup_all_goals": (not single_goal)}
    if arm == "test":
        return True, {**base}
    if arm == "lesion":          # graded read-out lesioned (strength=0) → delta must collapse
        return True, {**base}    # (strength override handled via _CLOSEA below)
    if arm == "no_learn":        # no value-train → V flat (no near>far)
        return True, {**base, "value_train_trials": 0}
    if arm == "dense":           # no-sparsification control (W=28) → graded grades nothing → flat
        return True, {**base, "place_sensors_to_place_weight": w_dense}
    if arm == "allnone":         # the documented baseline (all-or-none read-out, graded OFF) for contrast
        return False, {**base}
    raise ValueError(f"unknown arm {arm}")


def _run_arm(arm, seed, *, w_sparse, w_dense, value_train_trials,
             center, slope, strength, single_goal):
    graded, overrides = _arm_spec(arm, w_sparse, w_dense, value_train_trials, single_goal)
    _CLOSEA["enable"] = bool(graded)
    _CLOSEA["center"] = float(center)
    _CLOSEA["slope"] = float(slope)
    _CLOSEA["strength"] = (0.0 if arm == "lesion" else float(strength))
    _CLOSEA["bridge"] = None

    captured, real_fn = _capture_deployed_kwargs(
        seed, value_train_trials, stage_b=True)
    for k, v in overrides.items():
        captured[k] = v

    print("=" * 72, flush=True)
    print(f"[closeA] seed={seed} arm={arm} graded={graded} "
          f"(center={center} slope={slope} strength={_CLOSEA['strength']}) "
          f"overrides={overrides}", flush=True)
    print("=" * 72, flush=True)

    result = real_fn(**captured)

    # diagnostic: the on-bridge graded-V conductance near/far over striosome_value (the SAME analog
    # quantity the host-Gaussian positive control reports as ~9×). Read cleanly post-run.
    grv = _read_graded_v_near_far(captured, result)
    return {"arm": arm, "graded": graded, "overrides": overrides,
            "strength_used": _CLOSEA["strength"],
            "graded_v": grv,
            "stage_b": (result or {}).get("stage_b_smoke"),
            "selforg": (result or {}).get("selforg")}


def _read_graded_v_near_far(captured, result):
    """Read mean(cp_conductance_g_graded_plateau over striosome_value) at near vs far, driving
    place_sensors with the runner's render (re-derived from the deployed kwargs). Best-effort:
    returns None if the captured bridge / arrays are absent (e.g. graded OFF arm)."""
    br = _CLOSEA["bridge"]
    if br is None or getattr(br, "cp_conductance_g_graded_plateau", None) is None:
        return None
    try:
        import numpy as np
        from sim.backend import get_backend
        xp, _ = get_backend()
        ri = _build_region_indices(br)
        if "striosome_value" not in ri or "place_sensors" not in ri:
            return None
        c_idx = ri["striosome_value"]
        ps_idx = ri["place_sensors"]
        render = _make_render(captured)
        if render is None:
            return None
        sb = (result or {}).get("stage_b_smoke") or {}
        near = sb.get("near") or [6.0, 6.0]
        far = sb.get("far") or [1.0, 1.0]

        def _v_at(px, py, *, n_meas=120, warmup=40):
            saved = br.core_config.reward_learning_rate
            br.core_config.reward_learning_rate = 0.0
            # clean the slow plateau + GABA_B + critic membrane before the read.
            if getattr(br, "cp_conductance_g_graded_plateau", None) is not None:
                br.cp_conductance_g_graded_plateau[:] = xp.float32(0.0)
                br.cp_conductance_g_graded_plateau_rise[:] = xp.float32(0.0)
            act = xp.asarray(render(float(px), float(py)), dtype=xp.float32)
            br.cp_external_input_current[:] = xp.float32(0.0)
            br.cp_external_input_current[ps_idx] = act
            vsum = 0.0; m = 0
            for t in range(int(n_meas)):
                br._run_one_simulation_step()
                br.runtime_state.current_time_step += 1
                if t >= warmup:
                    gv = br.cp_conductance_g_graded_plateau[c_idx]
                    vsum += float(gv.mean()); m += 1
            br.core_config.reward_learning_rate = saved
            br.cp_external_input_current[:] = xp.float32(0.0)
            return vsum / max(m, 1)

        v_near = _v_at(near[0], near[1])
        v_far = _v_at(far[0], far[1])
        return {"v_near": float(v_near), "v_far": float(v_far),
                "v_near_over_far": float(v_near / max(v_far, 1e-9))}
    except Exception as e:  # diagnostic only — never fail the arm on it
        return {"error": repr(e)}


def _build_region_indices(br):
    """Map region name → GPU index array from the bridge's region manager."""
    rm = getattr(br, "region_manager", None)
    if rm is None:
        return {}
    try:
        from sim.backend import get_backend
        xp, _ = get_backend()
        import numpy as np
        d = rm.region_indices_dict()
        return {name: xp.asarray(np.asarray(idx, dtype=np.int64)) for name, idx in d.items()}
    except Exception:
        return {}


def _make_render(captured):
    """Re-derive the runner's place-sensor render from the captured deployed kwargs (the same
    `_n9_place_sensor_act` + the deterministically-laid landmarks the runner builds)."""
    try:
        import numpy as np
        grid_size = int(captured.get("grid_size", 32))
        n_bearing = int(captured.get("n_place_bearing", 8))
        n_dist = int(captured.get("n_place_dist", 4))
        max_int = float(captured.get("place_sensor_max_intensity", 1500.0))
        falloff = float(captured.get("place_sensor_falloff", 1.0))
        dist_sigma = float(captured.get("place_sensor_dist_sigma", 3.0))
        bexp = float(captured.get("place_sensor_bexp", 4.0))
        # the runner lays the landmarks via g._n9_place_landmarks; re-derive the SAME layout.
        landmarks = g._n9_place_landmarks(grid_size) if hasattr(g, "_n9_place_landmarks") else None
        if landmarks is None:
            return None
        dist_max = float(np.hypot(grid_size, grid_size))

        def render(px, py):
            return g._n9_place_sensor_act(
                px, py, landmarks, n_bearing, n_dist, max_int, falloff,
                dist_sigma, dist_max, bexp)
        return render
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", type=str, default="test",
                    choices=["test", "lesion", "no_learn", "dense", "allnone"])
    ap.add_argument("--all-arms", action="store_true",
                    help="run test + lesion + no_learn + dense + allnone in sequence")
    ap.add_argument("--w-sparse", type=float, default=10.0,
                    help="place_sensors_to_place_weight for the sparse arms (the W=10 sweet spot)")
    ap.add_argument("--w-dense", type=float, default=28.0,
                    help="place_sensors_to_place_weight for the dense no-sparsification control")
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--multi-goal", action="store_true",
                    help="train ALL scheduled goals (critic_warmup_all_goals=True); default is the "
                         "single-goal CLEAN capability test (only the first goal trained)")
    ap.add_argument("--graded-center", type=float, default=1.5)
    ap.add_argument("--graded-slope", type=float, default=1.0)
    ap.add_argument("--graded-strength", type=float, default=80.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # install the init-monkeypatch for the whole run.
    SimulationBridge._initialize_simulation_data = _patched_init

    arms = (["test", "lesion", "no_learn", "dense", "allnone"]
            if args.all_arms else [args.arm])
    results = {}
    for arm in arms:
        r = _run_arm(arm, args.seed,
                     w_sparse=args.w_sparse, w_dense=args.w_dense,
                     value_train_trials=args.value_train_trials,
                     center=args.graded_center, slope=args.graded_slope,
                     strength=args.graded_strength,
                     single_goal=(not args.multi_goal))
        results[arm] = r
        # honest per-arm one-liner.
        sb = r.get("stage_b") or {}
        grv = r.get("graded_v") or {}
        print(f"[closeA RESULT seed={args.seed} arm={arm}] "
              f"LEARNS-V={sb.get('learns_v')} (w_n/w_f={sb.get('w_near_over_far')}) "
              f"critic@near={sb.get('crit_near_hz')}Hz @far={sb.get('crit_far_hz')}Hz "
              f"grade={sb.get('crit_grade_ratio')} | "
              f"delta(gap)={sb.get('snc_gap_ratio')} gabab_gap={sb.get('gabab_gap')} "
              f"lesion-collapses={sb.get('lesion_collapses')} | "
              f"V_near={grv.get('v_near')} V_far={grv.get('v_far')} "
              f"V_n/f={grv.get('v_near_over_far')}", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        def _clean(o):
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()
                        if not (isinstance(k, str) and k.startswith("_"))}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            return o
        with open(args.out, "w") as f:
            json.dump({"seed": args.seed, "w_sparse": args.w_sparse,
                       "w_dense": args.w_dense,
                       "value_train_trials": args.value_train_trials,
                       "graded": {"center": args.graded_center,
                                  "slope": args.graded_slope,
                                  "strength": args.graded_strength},
                       "arms": _clean(results)}, f, indent=2, default=str)
        print(f"[closeA] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
