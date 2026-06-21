"""On-bridge grid-cell front end de-risk for shortcut #5b R1 (2026-06-22) — Steps 2 + 3.

Per the SURPASS scoping `research/findings/2026-06-22-shortcut5b-R1-selective-afferent-surpass.md`
(RANK 1) and the CPU smoke (`_n5_grid_frontend_selectivity_smoke.py`, GATE PASS — grid afferent adj cos
0.58 vs render 0.99 -> k-WTA place 0.29 vs render 0.93): the #5b residual R1 (the egocentric
`place_sensors` render is LOCALLY DEGENERATE, so the self-org place code is not location-SELECTIVE -> the
graded value read-out grades only ~1.18x near/far V -> delta flat) is SURPASSED by a spatial-phase
grid-cell metric (catalog D.07, the missing medial-EC piece). This probe carries the grid front end onto
the REAL spiking nav bridge.

THE BUILD (reuse-by-import, NO sim/ edit, NO g11_bg_runner.py edit):
  * the `place_sensors` region (an EXC stub driven externally each step by `_n9_render(px,py)` =
    `g._n9_place_sensor_act(...)`) is RE-POINTED at the grid code by a module-level monkeypatch of
    `g._n9_place_sensor_act`: when grid mode is ON it IGNORES the landmark args and returns the
    spatial-phase grid activations at (x,y). Because the runner drives `place_sensors` with `_n9_render`
    EVERYWHERE (self-org STEP-1, value-train STEP-2, the stage-B reads), the grid replaces the render
    consistently through the WHOLE pipeline -> the existing competitive `place_sensors -> place`
    threshold-WTA self-org carves locally-selective `place` fields off the decorrelated grid metric.
  * `place_sensors` is sized to the grid dim by setting n_place_bearing/n_place_dist so
    N_PLACE_LANDMARKS*(b+d) == n_grid (default 44+22 -> 3*66 == 198 == 6 modules x 33 grid cells).
  * the value read-out is the already-SHIPPED, byte-reviewed GRADED dendritic plateau
    (`enable_graded_dendritic_plateau`, commit d69cc0ab) — wired EXACTLY as the CLOSE A probe
    (`_n5_closeA_graded_on_selforg_probe.py`): an init-monkeypatch flips the cfg flags + the validated
    deploy params; `--readout-only` holds the graded strength 0 through STEP-1/STEP-2 (so the place code
    + learned V are the canonical regime) and swaps it on only for the stage-B reads.

ANTI-CHEAT (the grid reads ONLY (x,y) self-position): `make_grid_code(x,y)` takes ONLY (x,y) — the
agent's own legitimate self-position, the SAME channel the egocentric render reads. The GOAL coordinates
NEVER enter the grid. The grid phases are drawn ONCE from rng(seed) (a genome-style developmental draw,
the accepted self-organized bar — B1 dev-random precedent) and FIXED.

ARMS:
  * grid       : grid afferent + graded plateau read-out             (the TEST)
  * render     : the CURRENT egocentric render + graded plateau       (the R1-LIMIT NEGATIVE control —
                 reproduces CLOSE A's V n/f ~1.18x, delta flat; isolates that the lift is the grid INPUT)
  * scramble   : grid-SCRAMBLE lesion (per-cell phase permutation) + graded (selectivity collapses ->
                 delta collapses; the periodic metric is load-bearing, not a generic expansion)
  * no_learn   : grid + graded + value_train_trials=0                 (no value-train -> V flat; the lift
                 must come from the value-train ON a selective afferent)
  * lesion     : grid + graded_plateau_strength=0                     (the graded read-out is load-bearing)
The HOST-GAUSSIAN positive control (must give ~9x V + delta 1.33) is the already-validated
`_dendrite_stage1_onbridge_graded_plateau.py` runner — run separately + cited (CLOSE A: 3/3 seeds delta 1.33).

GO = grid arm: V n/f > the render's ~1.18x AND delta >= 1.3, with the render arm staying flat, the
grid-scramble collapsing delta, the no-learning floor flat, and the moat intact -> #5b R1 SURPASSED.

Usage:
  # Step 2 (on-bridge place selectivity on real spikes; near-neighbour read cos < 0.3):
  SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 42 --step2-selectivity
  # Step 3 (the delta verdict; grid vs render-negative vs scramble vs no-learning):
  SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 42 --arm grid --readout-only --multi-goal
  SIM_BACKEND=cupy python -m research.runners._n5_grid_frontend_onbridge_probe --seed 42 --all-arms --readout-only --multi-goal --out <out.json>
"""
import os
import sys
import json
import argparse

import numpy as np

import research.runners.g11_bg_runner as g
from sim.bridge import SimulationBridge
from research.runners._n5_grid_frontend_selectivity_smoke import (
    make_grid_code, make_grid_code_scrambled)


# ── grid-render monkeypatch state ────────────────────────────────────────────────────────────
_GRID = {
    "enable": False,        # ON => replace the egocentric render with the grid code
    "scramble": False,      # the grid-SCRAMBLE lesion (per-cell phase permutation)
    "grid_size": 32,
    "n_modules": 6,
    "n_per_module": 33,
    "lambda_min": 2.0,
    "lambda_max": 24.0,
    "seed": 42,
    "_code": None,          # the active grid_code(x,y) closure (built lazily at first use)
    "_n_grid": None,
    "_asserted_goal_free": False,
}
_orig_place_sensor_act = g._n9_place_sensor_act


def _build_grid_code():
    if _GRID["scramble"]:
        code, n = make_grid_code_scrambled(
            _GRID["grid_size"], n_modules=_GRID["n_modules"], n_per_module=_GRID["n_per_module"],
            lambda_min=_GRID["lambda_min"], lambda_max=_GRID["lambda_max"], seed=_GRID["seed"])
    else:
        code, n = make_grid_code(
            _GRID["grid_size"], n_modules=_GRID["n_modules"], n_per_module=_GRID["n_per_module"],
            lambda_min=_GRID["lambda_min"], lambda_max=_GRID["lambda_max"], seed=_GRID["seed"])
    _GRID["_code"] = code
    _GRID["_n_grid"] = int(n)


def _grid_place_sensor_act(x, y, landmarks, n_bearing, n_dist, max_int, falloff,
                           dist_sigma, dist_max, bexp):
    """Drop-in for g._n9_place_sensor_act: when grid mode is ON, return the spatial-phase grid code at
    (x,y) (IGNORING the landmark/render args entirely — anti-cheat: ONLY (x,y) is read), scaled to the
    render's max_int so the downstream place-pool current is in the SAME operating range as the render
    (so the self-org WTA + the critic see comparable drive magnitudes)."""
    if not _GRID["enable"]:
        return _orig_place_sensor_act(x, y, landmarks, n_bearing, n_dist, max_int, falloff,
                                      dist_sigma, dist_max, bexp)
    if _GRID["_code"] is None:
        _build_grid_code()
    # the grid code is in [0,1]; scale to the render's max_intensity (same current operating range).
    act = (_GRID["_code"](float(x), float(y)) * float(max_int)).astype(np.float32)
    # size guard: place_sensors is sized N_PLACE_LANDMARKS*(b+d); grid must match.
    expect = int(g.N_PLACE_LANDMARKS) * (int(n_bearing) + int(n_dist))
    if act.size != expect:
        raise RuntimeError(
            f"grid code size {act.size} != place_sensors size {expect} "
            f"(set n_place_bearing/n_place_dist so {g.N_PLACE_LANDMARKS}*(b+d) == n_grid {_GRID['_n_grid']})")
    return act


# ── graded-plateau install (VERBATIM from the CLOSE A probe) ──────────────────────────────────
_CLOSEA = {
    "enable": False, "center": 1.5, "slope": 1.0, "strength": 80.0,
    "tau_decay_ms": 80.0, "tau_rise_ms": 2.0, "bridge": None,
    "readout_only": False, "_armed": False,
}
_orig_init = SimulationBridge._initialize_simulation_data
_orig_set_gate = SimulationBridge.set_plasticity_gate


def _patched_init(self, *a, **kw):
    if _CLOSEA["enable"]:
        c = self.core_config
        c.enable_graded_dendritic_plateau = True
        c.graded_plateau_center = float(_CLOSEA["center"])
        c.graded_plateau_slope = float(_CLOSEA["slope"])
        c.graded_plateau_tau_decay_ms = float(_CLOSEA["tau_decay_ms"])
        c.graded_plateau_tau_rise_ms = float(_CLOSEA["tau_rise_ms"])
        if _CLOSEA["readout_only"]:
            c.graded_plateau_strength = 0.0   # graded OFF through training; swapped on at the V-train freeze
        else:
            c.coincidence_plateau_strength = 0.0
            c.graded_plateau_strength = float(_CLOSEA["strength"])
        _CLOSEA["_armed"] = False
    out = _orig_init(self, *a, **kw)
    _CLOSEA["bridge"] = self
    return out


def _patched_set_gate(self, name, value):
    if _CLOSEA["enable"] and _CLOSEA["readout_only"] and name == "value_input":
        if float(value) >= 1.0:
            _CLOSEA["_armed"] = True
        elif float(value) <= 0.0 and _CLOSEA["_armed"]:
            self.core_config.coincidence_plateau_strength = 0.0
            self.core_config.graded_plateau_strength = float(_CLOSEA["strength"])
            _CLOSEA["_armed"] = False
    return _orig_set_gate(self, name, value)


# ── capture the deployed kwargs (VERBATIM from CLOSE A; same NEGATIVE-repro argv) ─────────────
def _capture_deployed_kwargs(seed, value_train_trials, *, stage_b=True):
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


# grid dim = N_PLACE_LANDMARKS*(b+d): pick b/d so 3*(b+d) == n_modules*n_per_module.
def _grid_place_sensor_dims():
    n_grid = int(_GRID["n_modules"]) * int(_GRID["n_per_module"])
    bd = n_grid // int(g.N_PLACE_LANDMARKS)
    if bd * int(g.N_PLACE_LANDMARKS) != n_grid:
        raise ValueError(
            f"n_grid {n_grid} not divisible by N_PLACE_LANDMARKS {g.N_PLACE_LANDMARKS}; "
            f"choose n_modules*n_per_module divisible by {g.N_PLACE_LANDMARKS}")
    # split bd into (bearing, dist) ~2:1 (arbitrary; both feed the SAME place_sensors stub).
    n_dist = max(1, bd // 3)
    n_bearing = bd - n_dist
    return n_bearing, n_dist, n_grid


def _set_grid_mode(enable, scramble, seed):
    _GRID["enable"] = bool(enable)
    _GRID["scramble"] = bool(scramble)
    _GRID["seed"] = int(seed)
    _GRID["grid_size"] = 32
    _GRID["_code"] = None
    _GRID["_n_grid"] = None
    if enable:
        _build_grid_code()
        # the goal-free anti-cheat assertion (the grid_code signature takes ONLY (x,y) — structural).
        import inspect
        params = list(inspect.signature(_GRID["_code"]).parameters)
        assert params == ["x", "y"], f"grid_code reads {params}, must read ONLY (x,y) — goal-free!"
        _GRID["_asserted_goal_free"] = True


# ── Step 2: on-bridge place selectivity on REAL spikes (near-neighbour read cos) ──────────────
def _read_onbridge_place_selectivity(br, captured, pairs=((13, 13, 14, 13), (6, 6, 7, 6), (20, 20, 21, 20))):
    """Drive `place_sensors` (grid code) at adjacent cells, read `cp_firing_states[place]` over a window,
    cos. The FROZEN place code's near-neighbour read cos (< 0.3 = locally selective on real spikes; the
    render gives ~0.99 locally). Reads the bridge captured by _patched_init."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = getattr(br, "region_manager", None)
    if rm is None:
        return None
    d = rm.region_indices_dict()
    if "place" not in d or "place_sensors" not in d:
        return None
    p_idx = xp.asarray(np.asarray(d["place"], dtype=np.int64))
    ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
    code = _GRID["_code"]
    max_int = float(captured.get("place_sensor_max_intensity", 450.0))

    def _ensemble(px, py, *, n_meas=80):
        act = (code(float(px), float(py)) * max_int).astype(np.float32)
        br.cp_external_input_current[:] = xp.float32(0.0)
        br.cp_external_input_current[ps_idx] = xp.asarray(act, dtype=xp.float32)
        saved = br.core_config.reward_learning_rate
        br.core_config.reward_learning_rate = 0.0
        n = int(p_idx.size) if hasattr(p_idx, "size") else len(p_idx)
        counts = xp.zeros(n, dtype=xp.float32)
        for _ in range(int(n_meas)):
            br._run_one_simulation_step()
            br.runtime_state.current_time_step += 1
            counts += br.cp_firing_states[p_idx].astype(xp.float32)
        br.core_config.reward_learning_rate = saved
        br.cp_external_input_current[:] = xp.float32(0.0)
        return counts.get() if hasattr(counts, "get") else np.asarray(counts)

    coss = []
    for (ax, ay, bx, by) in pairs:
        ea = _ensemble(ax, ay); eb = _ensemble(bx, by)
        na = float(np.linalg.norm(ea)); nb = float(np.linalg.norm(eb))
        c = float(np.dot(ea, eb) / (na * nb)) if (na > 0 and nb > 0) else 1.0
        coss.append({"pair": [ax, ay, bx, by], "cos": c,
                     "sparsity": float(0.5 * (np.mean(ea > 0) + np.mean(eb > 0)))})
    mean_cos = float(np.mean([d["cos"] for d in coss]))
    return {"pairs": coss, "mean_adjacent_place_cos": mean_cos,
            "locally_selective": bool(mean_cos < 0.30)}


def _run_step2_selectivity(seed, *, value_train_trials=0):
    """Build the grid bridge, run STEP-1 self-org ONLY (value_train_trials=0 -> the value-train is a
    no-op), then read the FROZEN place code's near-neighbour cos on real spikes. Uses --stage-a-smoke
    so the run exits right after self-org (no value-train, no nav)."""
    _set_grid_mode(enable=True, scramble=False, seed=seed)
    _CLOSEA["enable"] = False
    n_bearing, n_dist, n_grid = _grid_place_sensor_dims()
    captured, real_fn = _capture_deployed_kwargs(seed, value_train_trials, stage_b=False)
    captured["n_place_bearing"] = n_bearing
    captured["n_place_dist"] = n_dist
    print("=" * 72, flush=True)
    print(f"[grid-onbridge STEP-2] seed={seed} grid n_grid={n_grid} "
          f"(place_sensors b={n_bearing}+d={n_dist} = {g.N_PLACE_LANDMARKS}*{n_bearing+n_dist})", flush=True)
    print("=" * 72, flush=True)
    result = real_fn(**captured)
    sel = _read_onbridge_place_selectivity(_CLOSEA["bridge"], captured)
    # ALSO compare the render baseline's near-neighbour read on the SAME bridge build (grid OFF) for
    # contrast — re-point the stub at the render and read.
    sel_render = None
    try:
        _GRID["enable"] = False
        sel_render = _read_onbridge_place_render_baseline(_CLOSEA["bridge"], captured)
    finally:
        _GRID["enable"] = True
    out = {"seed": seed, "n_grid": n_grid, "grid": sel,
           "selforg_diff_cos": (result or {}).get("selforg", {}).get("diff_cos")
           if isinstance((result or {}).get("selforg"), dict) else None,
           "render_baseline_on_same_bridge": sel_render}
    print(f"[grid-onbridge STEP-2 RESULT seed={seed}] grid mean adjacent-place cos="
          f"{(sel or {}).get('mean_adjacent_place_cos')} (locally_selective="
          f"{(sel or {}).get('locally_selective')}) | render-on-same-bridge="
          f"{(sel_render or {}).get('mean_adjacent_place_cos')}", flush=True)
    return out


def _read_onbridge_place_render_baseline(br, captured,
                                         pairs=((13, 13, 14, 13), (6, 6, 7, 6), (20, 20, 21, 20))):
    """Same near-neighbour place read but driving the stub with the ORIGINAL egocentric render (the R1
    cap on the SAME trained bridge — a within-bridge contrast). The place fields were carved off the
    GRID, so this reads how those grid-carved fields respond to render input; informational contrast."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = getattr(br, "region_manager", None)
    if rm is None:
        return None
    d = rm.region_indices_dict()
    if "place" not in d or "place_sensors" not in d:
        return None
    p_idx = xp.asarray(np.asarray(d["place"], dtype=np.int64))
    ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
    grid_size = int(captured.get("grid_size", 32))
    n_bearing = int(captured.get("n_place_bearing", 12))
    n_dist = int(captured.get("n_place_dist", 8))
    max_int = float(captured.get("place_sensor_max_intensity", 450.0))
    falloff = float(captured.get("place_sensor_falloff", 0.03))
    dist_sigma = float(captured.get("place_sensor_dist_sigma", 4.0))
    bexp = float(captured.get("place_sensor_bexp", 4.0))
    landmarks = g._n9_place_landmarks(grid_size)
    dist_max = float(grid_size) * 1.42

    def _ens(px, py, *, n_meas=80):
        act = _orig_place_sensor_act(px, py, landmarks, n_bearing, n_dist, max_int, falloff,
                                     dist_sigma, dist_max, bexp)
        br.cp_external_input_current[:] = xp.float32(0.0)
        br.cp_external_input_current[ps_idx] = xp.asarray(act, dtype=xp.float32)
        saved = br.core_config.reward_learning_rate
        br.core_config.reward_learning_rate = 0.0
        n = int(p_idx.size) if hasattr(p_idx, "size") else len(p_idx)
        counts = xp.zeros(n, dtype=xp.float32)
        for _ in range(int(n_meas)):
            br._run_one_simulation_step()
            br.runtime_state.current_time_step += 1
            counts += br.cp_firing_states[p_idx].astype(xp.float32)
        br.core_config.reward_learning_rate = saved
        br.cp_external_input_current[:] = xp.float32(0.0)
        return counts.get() if hasattr(counts, "get") else np.asarray(counts)

    coss = []
    for (ax, ay, bx, by) in pairs:
        ea = _ens(ax, ay); eb = _ens(bx, by)
        na = float(np.linalg.norm(ea)); nb = float(np.linalg.norm(eb))
        coss.append(float(np.dot(ea, eb) / (na * nb)) if (na > 0 and nb > 0) else 1.0)
    return {"mean_adjacent_place_cos": float(np.mean(coss)), "pairs": coss}


# ── Step 3: the delta verdict ────────────────────────────────────────────────────────────────
def _read_graded_v_near_far(captured, result):
    """Mean(cp_conductance_g_graded_plateau over striosome_value) at near vs far (grid-driven). The
    on-bridge analog the host-Gaussian control reports as ~9x. Best-effort."""
    br = _CLOSEA["bridge"]
    if br is None or getattr(br, "cp_conductance_g_graded_plateau", None) is None:
        return None
    try:
        from sim.backend import get_backend
        xp, _ = get_backend()
        rm = getattr(br, "region_manager", None)
        if rm is None:
            return None
        d = rm.region_indices_dict()
        if "striosome_value" not in d or "place_sensors" not in d:
            return None
        c_idx = xp.asarray(np.asarray(d["striosome_value"], dtype=np.int64))
        ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
        max_int = float(captured.get("place_sensor_max_intensity", 450.0))
        sb = (result or {}).get("stage_b_smoke") or {}
        near = sb.get("near") or [6.0, 6.0]
        far = sb.get("far") or [1.0, 1.0]
        # use whatever afferent the run used: grid (if enabled) else render.
        if _GRID["enable"]:
            if _GRID["_code"] is None:
                _build_grid_code()
            def _act(px, py):
                return (_GRID["_code"](float(px), float(py)) * max_int).astype(np.float32)
        else:
            grid_size = int(captured.get("grid_size", 32))
            landmarks = g._n9_place_landmarks(grid_size)
            dist_max = float(grid_size) * 1.42
            def _act(px, py):
                return _orig_place_sensor_act(
                    px, py, landmarks, int(captured.get("n_place_bearing", 12)),
                    int(captured.get("n_place_dist", 8)), max_int,
                    float(captured.get("place_sensor_falloff", 0.03)),
                    float(captured.get("place_sensor_dist_sigma", 4.0)), dist_max,
                    float(captured.get("place_sensor_bexp", 4.0)))

        def _v_at(px, py, *, n_meas=120, warmup=40):
            saved = br.core_config.reward_learning_rate
            br.core_config.reward_learning_rate = 0.0
            if getattr(br, "cp_conductance_g_graded_plateau", None) is not None:
                br.cp_conductance_g_graded_plateau[:] = xp.float32(0.0)
                br.cp_conductance_g_graded_plateau_rise[:] = xp.float32(0.0)
            br.cp_external_input_current[:] = xp.float32(0.0)
            br.cp_external_input_current[ps_idx] = xp.asarray(_act(px, py), dtype=xp.float32)
            vsum = 0.0; m = 0
            for t in range(int(n_meas)):
                br._run_one_simulation_step()
                br.runtime_state.current_time_step += 1
                if t >= warmup:
                    vsum += float(br.cp_conductance_g_graded_plateau[c_idx].mean()); m += 1
            br.core_config.reward_learning_rate = saved
            br.cp_external_input_current[:] = xp.float32(0.0)
            return vsum / max(m, 1)

        v_near = _v_at(near[0], near[1]); v_far = _v_at(far[0], far[1])
        return {"v_near": float(v_near), "v_far": float(v_far),
                "v_near_over_far": float(v_near / max(v_far, 1e-9))}
    except Exception as e:
        return {"error": repr(e)}


def _run_delta_arm(arm, seed, *, value_train_trials, single_goal, readout_only,
                   center, slope, strength):
    """One delta-verdict arm. arm in {grid, render, scramble, no_learn, lesion}."""
    grid_on = arm in ("grid", "scramble", "no_learn", "lesion")
    scramble = (arm == "scramble")
    _set_grid_mode(enable=grid_on, scramble=scramble, seed=seed)
    _CLOSEA["enable"] = True   # graded plateau on for every arm (render arm = CLOSE A R1-LIMIT contrast)
    _CLOSEA["center"] = float(center)
    _CLOSEA["slope"] = float(slope)
    _CLOSEA["strength"] = (0.0 if arm == "lesion" else float(strength))
    _CLOSEA["readout_only"] = bool(readout_only) and arm != "lesion"
    _CLOSEA["bridge"] = None
    _CLOSEA["_armed"] = False

    vtt = 0 if arm == "no_learn" else int(value_train_trials)
    captured, real_fn = _capture_deployed_kwargs(seed, vtt, stage_b=True)
    captured["critic_warmup_all_goals"] = (not single_goal)
    captured["place_sensors_to_place_weight"] = 10.0   # the W=10 sparse sweet spot (CLOSE A)
    if arm == "no_learn":
        captured["value_train_trials"] = 0
    if grid_on:
        n_bearing, n_dist, n_grid = _grid_place_sensor_dims()
        captured["n_place_bearing"] = n_bearing
        captured["n_place_dist"] = n_dist
    else:
        n_grid = None

    print("=" * 72, flush=True)
    print(f"[grid-onbridge DELTA] seed={seed} arm={arm} grid_on={grid_on} scramble={scramble} "
          f"(graded center={center} slope={slope} strength={_CLOSEA['strength']} "
          f"readout_only={_CLOSEA['readout_only']}) n_grid={n_grid}", flush=True)
    print("=" * 72, flush=True)
    result = real_fn(**captured)
    grv = _read_graded_v_near_far(captured, result)
    sb = (result or {}).get("stage_b_smoke") or {}
    return {"arm": arm, "grid_on": grid_on, "scramble": scramble, "n_grid": n_grid,
            "graded_v": grv, "stage_b": sb,
            "goal_free_asserted": _GRID.get("_asserted_goal_free", False) if grid_on else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step2-selectivity", action="store_true",
                    help="Step 2: on-bridge place selectivity (near-neighbour read cos on real spikes)")
    ap.add_argument("--arm", type=str, default="grid",
                    choices=["grid", "render", "scramble", "no_learn", "lesion"])
    ap.add_argument("--all-arms", action="store_true",
                    help="Step 3: run grid + render + scramble + no_learn + lesion (the delta battery)")
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--multi-goal", action="store_true",
                    help="train ALL scheduled goals (critic_warmup_all_goals=True)")
    ap.add_argument("--readout-only", action="store_true",
                    help="hold the graded plateau strength 0 through STEP-1/STEP-2 (canonical place "
                         "code + learned V), enable it only for the stage-B reads (apples-to-apples)")
    ap.add_argument("--graded-center", type=float, default=1.5)
    ap.add_argument("--graded-slope", type=float, default=1.0)
    ap.add_argument("--graded-strength", type=float, default=80.0)
    ap.add_argument("--n-modules", type=int, default=6)
    ap.add_argument("--n-per-module", type=int, default=33)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    _GRID["n_modules"] = int(args.n_modules)
    _GRID["n_per_module"] = int(args.n_per_module)

    # install the monkeypatches (the grid render + the graded plateau init/gate hooks).
    g._n9_place_sensor_act = _grid_place_sensor_act
    SimulationBridge._initialize_simulation_data = _patched_init
    SimulationBridge.set_plasticity_gate = _patched_set_gate

    out_obj = {"seed": args.seed, "n_modules": args.n_modules, "n_per_module": args.n_per_module}

    if args.step2_selectivity:
        out_obj["step2"] = _run_step2_selectivity(args.seed)
    else:
        arms = (["grid", "render", "scramble", "no_learn", "lesion"]
                if args.all_arms else [args.arm])
        results = {}
        for arm in arms:
            r = _run_delta_arm(arm, args.seed,
                               value_train_trials=args.value_train_trials,
                               single_goal=(not args.multi_goal),
                               readout_only=args.readout_only,
                               center=args.graded_center, slope=args.graded_slope,
                               strength=args.graded_strength)
            results[arm] = r
            sb = r.get("stage_b") or {}; grv = r.get("graded_v") or {}
            print(f"[grid-onbridge DELTA RESULT seed={args.seed} arm={arm}] "
                  f"LEARNS-V={sb.get('learns_v')} (w_n/w_f={sb.get('w_near_over_far')}) "
                  f"critic@near={sb.get('crit_near_hz')}Hz @far={sb.get('crit_far_hz')}Hz "
                  f"grade={sb.get('crit_grade_ratio')} | V_near={grv.get('v_near')} "
                  f"V_far={grv.get('v_far')} V_n/f={grv.get('v_near_over_far')} | "
                  f"delta(gap)={sb.get('snc_gap_ratio')} gabab_gap={sb.get('gabab_gap')}", flush=True)
        out_obj["arms"] = results

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
            json.dump(_clean(out_obj), f, indent=2, default=str)
        print(f"[grid-onbridge] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
