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
    "drive_scale": 2.5,     # gain on max_int so the grid's TOTAL drive matches the render's operating
                            # range (the grid code is in [0,1] cos^3-rectified -> sparse+small; the
                            # render mean drive at the goal is ~67 pA, the grid's ~27 at scale 1 ->
                            # ~2.5x matches the mean so the place pool + critic see comparable drive.
                            # A pure GAIN (NOT a per-location renorm) -> the grid's magnitude structure
                            # is preserved; it only sets the operating current, exactly as max_int does
                            # for the render).
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
    # the grid code is in [0,1]; scale to the render's max_intensity * drive_scale (so the grid's TOTAL
    # drive matches the render's operating current range; see _GRID["drive_scale"]).
    act = (_GRID["_code"](float(x), float(y)) * float(max_int) * float(_GRID["drive_scale"])).astype(np.float32)
    # size guard: place_sensors is sized N_PLACE_LANDMARKS*(b+d); grid must match.
    expect = int(g.N_PLACE_LANDMARKS) * (int(n_bearing) + int(n_dist))
    if act.size != expect:
        raise RuntimeError(
            f"grid code size {act.size} != place_sensors size {expect} "
            f"(set n_place_bearing/n_place_dist so {g.N_PLACE_LANDMARKS}*(b+d) == n_grid {_GRID['_n_grid']})")
    return act


# ── critic-homeostasis adapt-rate override (the δ-readout STABILIZATION lever) ────────────────
# The self-org place code's volley STRENGTH is CuPy-non-deterministic (the transpose-SpMV atomic scatter;
# 28-118 Hz critic spread), so a fixed value-train soft-bound lands the critic at very different rates
# across seeds (17/65/257 Hz) -> the SNc subtraction is clean at ~17 Hz but over-clamps at 257 Hz -> the
# δ-gap is seed-variable. The critic ALREADY runs --enable-critic-homeostasis (intrinsic threshold
# adaptation that defends a TARGET rate against volley draws), but its global adapt rate (0.0005, ~0.5
# mV/sec) is too slow to converge within the short value-train/read windows. This override speeds the
# threshold adaptation so the critic settles to ~target Hz on EVERY seed regardless of the volley. A pure
# DYNAMICS knob (the cell still defends the biological target rate; only the convergence timescale changes).
_HOMEO = {"adapt_rate": 0.0, "target_rate": 0.0, "ema_alpha": 0.0}  # 0 = no override (the runner default)


# ── deterministic-READ lever (the #5b deterministic-scatter δ close, 2026-06-22) ──────────────
# RANK-1a per `research/findings/2026-06-22-shortcut5b-deterministic-scatter-scoping.md` (a48ad76f):
# the secondary SNc-burst δ holds only 2/3 because the read-time critic rate is seed-variable (17/65/254
# Hz), root-caused to CuPy's transpose-SpMV atomic-scatter non-determinism on the place→critic path. The
# deterministic-scatter SpMV ALREADY EXISTS (.tocsr()-materialize, numerically-allclose) and is wired at
# all five critic-path matvec sites in sim/bridge.py, gated on cfg.deterministic_transpose_matvec. The
# runner (g11_bg_runner.py:5510-5548) only toggles it ON for STEP-1 self-org and RESTORES it OFF before
# the value-train + δ-read. The fix is to set the cfg field to True in _patched_init BEFORE the runner's
# `_saved_detmv = getattr(...)` capture — so the runner's restore (`= _saved_detmv`) keeps it ON through
# the value-train + δ-read, giving a seed-stable volley strength → a single critic-rate regime → the
# SNc-burst δ holds 3/3 under ONE config. NO sim/ edit (the deterministic branch ships); probe-only.
# 0 = off (the runner default; STEP-1-only determinism, byte-identical).
_DETMV = {"read": False}


# ── synaptic-scaling VOLLEY-NORMALIZATION lever (the #5b deferred-item-1 close, 2026-06-21) ────
# Per the close doc `2026-06-22-shortcut5b-determinism-deltabar-close.md` (08d24a61): determinism ALONE
# holds the SNc-burst δ only 2/3 — seed 44's residual is NOT a non-determinism artifact (its 255.8 Hz
# critic rate is reproducible with the flag ON) but a GENUINELY STRONG learned place→value volley
# (w_near grew to 2.475 on seed 44 vs 0.40/0.57 on seeds 42/43) OVER-DRIVING the weighted-plateau read →
# the critic fires hard even at FAR (136.5 Hz) → the SNc GABA_B subtraction over-clamps at BOTH near and
# far → gabab_gap=False. A FLAT value-train soft-bound CAP (0.8) fails — it STARVES the gentle seeds
# (their critic needs the higher weight to fire the read at all → 0–1.4 Hz). The genuine fix is to
# NORMALIZE the seed-variable learned-volley strength: Turrigiano 2008 synaptic scaling
# (cfg.enable_synaptic_scaling, sim/bridge.py:7402) multiplicatively scales the critic's afferent weights
# toward a TARGET critic firing rate. Because it scales ALL afferents to a postsynaptic neuron by ONE
# factor, the near/far RATIO (the R1 selectivity, set by STDP) is PRESERVED while the ABSOLUTE volley
# level is driven to one seed-STABLE operating point → the strong seed (44) is scaled DOWN to the gentle
# seeds' regime without starving them (unlike the flat cap). Point-neuron, biology-grounded, on the EXISTING
# sim/ machinery (NO sim/ edit). 0 = off (byte-identical). The target_rate/scaling_rate set the operating
# point + convergence speed; the scaling is held ON through value-train + the reads so the volley settles
# to target and stays there (scale_factor→1 at target → no read-time drift).
_SYNSCALE = {"enable": False, "target_rate": 0.0, "scaling_rate": 0.0, "ema_alpha": 0.0}


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
    # the #5b deterministic-READ lever (RANK-1a): set cfg.deterministic_transpose_matvec=True so the
    # runner's STEP-1 toggle captures `_saved_detmv=True` and RESTORES it back to True (not OFF) after
    # self-org → the deterministic-scatter SpMV stays ON through the value-train + δ-read → seed-stable
    # place→critic volley strength → a single critic-rate regime across seeds → the SNc-burst δ holds
    # 3/3. NO sim/ edit (the deterministic branch already ships at all five critic-path matvec sites).
    if _DETMV["read"]:
        self.core_config.deterministic_transpose_matvec = True
    # the synaptic-scaling VOLLEY-NORMALIZATION lever (#5b deferred-item-1): enable Turrigiano synaptic
    # scaling so the critic's afferent weights are driven toward a seed-STABLE target firing rate. The
    # strong-volley seed (44, w_near 2.475) is scaled DOWN to the gentle-seed regime; the gentle seeds
    # (42/43) are NOT starved (multiplicative per-postsynaptic scaling preserves the near/far RATIO).
    # Held ON through value-train + the reads (the runner does NOT toggle this flag, so setting it here
    # keeps it live for the whole pipeline). NO sim/ edit — the existing cfg.enable_synaptic_scaling path.
    if _SYNSCALE["enable"]:
        self.core_config.enable_synaptic_scaling = True
        if _SYNSCALE["scaling_rate"] and _SYNSCALE["scaling_rate"] > 0:
            self.core_config.synaptic_scaling_rate = float(_SYNSCALE["scaling_rate"])
        if _SYNSCALE["target_rate"] and _SYNSCALE["target_rate"] > 0:
            self.core_config.homeostasis_target_rate = float(_SYNSCALE["target_rate"])
        if _SYNSCALE["ema_alpha"] and _SYNSCALE["ema_alpha"] > 0:
            self.core_config.homeostasis_ema_alpha = float(_SYNSCALE["ema_alpha"])
    # the δ-readout STABILIZATION lever: speed the critic-homeostasis threshold adaptation (+ optional
    # target rate) so the critic converges to the biological target on every place-code draw.
    if _HOMEO["adapt_rate"] and _HOMEO["adapt_rate"] > 0:
        self.core_config.homeostasis_threshold_adapt_rate = float(_HOMEO["adapt_rate"])
    if _HOMEO["target_rate"] and _HOMEO["target_rate"] > 0:
        self.core_config.homeostasis_target_rate = float(_HOMEO["target_rate"])
    if _HOMEO["ema_alpha"] and _HOMEO["ema_alpha"] > 0:
        # speed the homeostasis RATE-ESTIMATE EMA so the critic over-firing is registered (and the
        # threshold adapted) within the short value-train window (default tau~5000 steps is too slow).
        self.core_config.homeostasis_ema_alpha = float(_HOMEO["ema_alpha"])
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
        act = (code(float(px), float(py)) * max_int * float(_GRID["drive_scale"])).astype(np.float32)
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
                return (_GRID["_code"](float(px), float(py)) * max_int * float(_GRID["drive_scale"])).astype(np.float32)
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


def _read_graded_v_delta(captured, result, *, settle_steps=0):
    """The GRADED-V-only δ read (the SURPASS ISOLATE move (a)) — decouple the δ residual from the
    SNc-burst over-clamp.

    The stage-B `snc_gap_ratio` (the WEIGHTED-plateau read) over-fires the critic on high-volley draws
    (seed 44: critic ~260 Hz) -> the SNc over-clamps (GABA_B silences the SNc at BOTH near AND far) ->
    that δ inverts to 0.0 even though the GRADED VALUE is robustly selective (v_near/v_far 4.5-12.3x on
    every seed). The fix is to read δ from the already-3/3 graded value V, NOT the over-driving somatic
    burst. Two faithful graded-V δ forms (both on the captured bridge, the place plateau supplying the
    differential V; NO weighted-plateau toggle -> no over-clamp):

      (a1) delta_vnf       : the graded plateau conductance near/far RATIO itself (the doc's "host-Gaussian
                             exact analog read") — the direct R1-fix quantity. >= the δ bar trivially when
                             the value is selective.
      (a2) delta_snc_graded: the GENUINE r-V RPE gap (snc_unpred_FAR / snc_pred_NEAR), read in the SETTLED
                             count-plateau critic regime (coincidence_weighted_drive=FALSE — the regime the
                             value-train + homeostasis ran in, ~30 Hz critic), so the differential V comes
                             from the graded plateau WITHOUT the weighted-plateau somatic over-drive. Keeps
                             the RPE shape but does not saturate the SNc.

    `settle_steps` (move (b)) runs a brief homeostasis SETTLING window (place@near drive, learning off)
    before the gap read so an over-fired critic relaxes toward its target before δ is read. settle_steps=0
    = move (a) only (no settle).
    """
    br = _CLOSEA["bridge"]
    if br is None:
        return None
    try:
        from sim.backend import get_backend
        xp, _ = get_backend()
        rm = getattr(br, "region_manager", None)
        if rm is None:
            return None
        d = rm.region_indices_dict()
        for need in ("striosome_value", "snc", "place_sensors"):
            if need not in d:
                return {"error": f"missing region {need}"}
        c_idx = xp.asarray(np.asarray(d["striosome_value"], dtype=np.int64))
        s_idx = xp.asarray(np.asarray(d["snc"], dtype=np.int64))
        ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
        n_snc = int(s_idx.size) if hasattr(s_idx, "size") else len(s_idx)
        has_reward_us = "reward_us" in d
        ru_idx = xp.asarray(np.asarray(d["reward_us"], dtype=np.int64)) if has_reward_us else None
        max_int = float(captured.get("place_sensor_max_intensity", 450.0))
        snc_tonic = float(captured.get("snc_tonic_pa", 220.0))
        snc_reward_gain = float(captured.get("snc_reward_gain", 400.0))
        reward_us_drive = float(captured.get("reward_us_drive_pa", 250.0))
        spiking_reward_us = bool(captured.get("spiking_reward_us", True))
        lead_steps = int(captured.get("critic_lead_steps", 120))
        hold_steps = int(captured.get("value_train_hold_steps", 40))

        sb = (result or {}).get("stage_b_smoke") or {}
        near = sb.get("near") or [6.0, 6.0]
        far = sb.get("far") or [1.0, 1.0]

        if _GRID["enable"]:
            if _GRID["_code"] is None:
                _build_grid_code()
            def _act(px, py):
                return (_GRID["_code"](float(px), float(py)) * max_int * float(_GRID["drive_scale"])).astype(np.float32)
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

        def _reset():
            # clean the slow plateau + GABA_B + SNc/critic membrane between reads (mirror
            # _n9_reset_critic_read_state: contamination across near<->far reads = false grading).
            for nm in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise",
                       "cp_conductance_g_graded_plateau", "cp_conductance_g_graded_plateau_rise",
                       "cp_conductance_g_gabab"):
                arr = getattr(br, nm, None)
                if arr is not None:
                    arr[:] = xp.float32(0.0)
            if (getattr(br, "cp_membrane_potential_v", None) is not None
                    and getattr(br, "cp_izh_vr", None) is not None):
                br.cp_membrane_potential_v[c_idx] = br.cp_izh_vr[c_idx]
                br.cp_membrane_potential_v[s_idx] = br.cp_izh_vr[s_idx]
                if getattr(br, "cp_recovery_variable_u", None) is not None:
                    br.cp_recovery_variable_u[c_idx] = xp.float32(0.0)
                    br.cp_recovery_variable_u[s_idx] = xp.float32(0.0)
            br.cp_external_input_current[:] = xp.float32(0.0)
            for _ in range(60):
                br._run_one_simulation_step(); br.runtime_state.current_time_step += 1

        def _snc_burst_rate_graded(px, py):
            """The r-V SNc burst rate in the SETTLED (non-over-driving) count-plateau regime: the graded
            plateau supplies the differential V (it is already on, strength set by the readout-only swap);
            coincidence_weighted_drive STAYS at its converged value (no weighted-plateau over-drive)."""
            _reset()
            act = xp.asarray(_act(px, py), dtype=xp.float32)
            # optional SETTLING (move b): let an over-fired critic relax toward its homeostatic target
            # under place drive before the gap read (learning off; SNc tonic only).
            if settle_steps and settle_steps > 0:
                saved = br.core_config.reward_learning_rate
                br.core_config.reward_learning_rate = 0.0
                br.cp_external_input_current[:] = xp.float32(0.0)
                br.cp_external_input_current[ps_idx] = act
                br.cp_external_input_current[s_idx] = xp.float32(snc_tonic)
                for _ in range(int(settle_steps)):
                    br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
                br.core_config.reward_learning_rate = saved
            # LEAD: place drive -> critic fires -> GABA_B builds onto SNc BEFORE the reward burst.
            br.cp_external_input_current[:] = xp.float32(0.0)
            br.cp_external_input_current[ps_idx] = act
            br.cp_external_input_current[s_idx] = xp.float32(snc_tonic)
            saved = br.core_config.reward_learning_rate
            br.core_config.reward_learning_rate = 0.0
            # accumulate the mean GABA_B conductance ON the SNc during the LEAD — the DIRECT V-subtraction
            # term (the "-V" in δ=r-V). It grades with the learned V (near >> far) and, unlike the SNc
            # SOMATIC burst, does NOT saturate when V is large (a clamped SNc still has a large g_gabab) ->
            # the most over-clamp-DECOUPLED RPE-shaped read. None if GABA_B is not enabled on this bridge.
            gabab_arr = getattr(br, "cp_conductance_g_gabab", None)
            gb_sum = 0.0; gb_m = 0
            for _ in range(int(lead_steps)):
                br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
                if gabab_arr is not None:
                    gb_sum += float(gabab_arr[s_idx].mean()); gb_m += 1
            gabab_lead = (gb_sum / gb_m) if (gabab_arr is not None and gb_m > 0) else None
            # REWARD burst (place still on) — the spiking US afferent fires into the SNc (fully-spiking r).
            if spiking_reward_us and ru_idx is not None:
                br.cp_external_input_current[s_idx] = xp.float32(snc_tonic)
                br.cp_external_input_current[ru_idx] = xp.float32(reward_us_drive)
            else:
                br.cp_external_input_current[s_idx] = xp.float32(snc_tonic + snc_reward_gain)
            spk = 0
            for _ in range(int(hold_steps)):
                br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
                spk += int(br.cp_firing_states[s_idx].sum())
            br.core_config.reward_learning_rate = saved
            br.cp_external_input_current[:] = xp.float32(0.0)
            return (spk / max(n_snc, 1) / max(int(hold_steps) * 1e-3, 1e-9), gabab_lead)

        # (a2) the genuine r-V SNc-burst gap in the settled regime; (a3) the GABA_B-conductance gap.
        snc_pred, gb_near = _snc_burst_rate_graded(near[0], near[1])    # NEAR: V subtracts
        snc_unpred, gb_far = _snc_burst_rate_graded(far[0], far[1])     # FAR: no V
        delta_snc_graded = snc_unpred / max(snc_pred, 1e-6)
        gabab_gap_graded = bool(snc_unpred > 1.30 * max(snc_pred, 1e-6))
        # (a3) the GABA_B-conductance δ: near >> far V-subtraction = the prediction graded with V; immune
        # to the SNc somatic over-clamp. δ_gabab = g_gabab(near) / g_gabab(far).
        delta_gabab = (float(gb_near / max(gb_far, 1e-12))
                       if (gb_near is not None and gb_far is not None) else None)
        gabab_cond_gap = bool(delta_gabab is not None and delta_gabab >= 1.30)

        # (a1) the pure graded-V near/far ratio (the direct R1-fix read) — reuse the same-bridge V read.
        grv = _read_graded_v_near_far(captured, result) or {}
        delta_vnf = float(grv.get("v_near_over_far", 0.0))

        return {
            "settle_steps": int(settle_steps),
            "delta_vnf": delta_vnf,                                    # (a1) graded V near/far ratio
            "vnf_gap": bool(delta_vnf >= 1.30),
            "snc_pred_near_hz": float(snc_pred), "snc_unpred_far_hz": float(snc_unpred),
            "delta_snc_graded": float(delta_snc_graded),               # (a2) settled-regime SNc r-V gap
            "gabab_gap_graded": gabab_gap_graded,
            "gabab_near": (float(gb_near) if gb_near is not None else None),   # (a3) the -V subtraction term
            "gabab_far": (float(gb_far) if gb_far is not None else None),
            "delta_gabab": delta_gabab,                                # (a3) g_gabab(near)/g_gabab(far)
            "gabab_cond_gap": gabab_cond_gap,
        }
    except Exception as e:
        return {"error": repr(e)}


def _run_delta_arm(arm, seed, *, value_train_trials, single_goal, readout_only,
                   center, slope, strength, value_train_w_max=0.0, settle_steps=0,
                   critic_gabab_max=0.0):
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
    if value_train_w_max and value_train_w_max > 0:
        # cap the place->value soft-bound DURING value-train so the critic settles in the GRADED
        # ~3-6 range (the de-risk's stdp_w_max=40 regime) instead of over-firing/over-clamping the SNc.
        captured["value_train_stdp_w_max"] = float(value_train_w_max)
    if critic_gabab_max and critic_gabab_max > 0:
        # the principled GIRK-saturation regime fix: cap g_gabab so a hot critic cannot fully clamp the
        # SNc -> the genuine SNc-burst δ stays GRADED on high-volley draws (the over-clamp seed-44 fix).
        captured["critic_gabab_max"] = float(critic_gabab_max)
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
    # the SURPASS ISOLATE move (a)/(b): the GRADED-V-only δ read (decoupled from the SNc-burst
    # over-clamp). delta_vnf = (a1) the graded V near/far ratio; delta_snc_graded = (a2) the genuine
    # r-V gap in the settled count-plateau regime (+ optional move-(b) settling window).
    graded_delta = _read_graded_v_delta(captured, result, settle_steps=settle_steps)
    sb = (result or {}).get("stage_b_smoke") or {}
    return {"arm": arm, "grid_on": grid_on, "scramble": scramble, "n_grid": n_grid,
            "graded_v": grv, "graded_delta": graded_delta, "stage_b": sb,
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
    ap.add_argument("--grid-drive-scale", type=float, default=2.5,
                    help="gain on max_int for the grid code so its total drive matches the render's "
                         "operating current range (the grid is sparse+small in [0,1]); see _GRID")
    ap.add_argument("--value-train-w-max", type=float, default=0.0,
                    help="cap place->value soft-bound DURING value-train (de-risk 40 -> critic in the "
                         "graded ~3-6 range; 0=no override). Use to avoid critic over-clamp.")
    ap.add_argument("--critic-homeo-adapt-rate", type=float, default=0.0,
                    help="the δ-readout STABILIZATION lever: speed the critic-homeostasis threshold "
                         "adaptation (runner default 0.0005 is too slow to converge in the value-train "
                         "window -> seed-variable critic rate); 0=no override.")
    ap.add_argument("--critic-homeo-target-rate", type=float, default=0.0,
                    help="critic-homeostasis target firing rate (fraction; runner default 0.02=20Hz); "
                         "0=no override.")
    ap.add_argument("--critic-homeo-ema-alpha", type=float, default=0.0,
                    help="speed the homeostasis rate-estimate EMA (runner default 0.0002 tau~5000 steps "
                         "is too slow to register over-firing in the value-train window); 0=no override.")
    ap.add_argument("--settle-steps", type=int, default=0,
                    help="the SURPASS move (b): homeostasis SETTLING window (steps) under place drive "
                         "before the graded-V SNc gap read, so an over-fired critic relaxes toward its "
                         "target before δ is read; 0=move (a) only (no settle).")
    ap.add_argument("--critic-gabab-max", type=float, default=0.0,
                    help="the principled GIRK-saturation regime fix (already in sim/, cfg field "
                         "gabab_conductance_max, default 0=off=byte-identical): cap g_gabab so a hot "
                         "critic cannot fully CLAMP the SNc (graded subtraction at any rate) -> the "
                         "genuine SNc-burst δ stays graded instead of over-clamping on high-volley draws.")
    ap.add_argument("--deterministic-read", action="store_true",
                    help="the #5b deterministic-scatter δ close (RANK-1a, 2026-06-22): hold the EXISTING "
                         "cfg.deterministic_transpose_matvec ON through the value-train + δ-read (the runner "
                         "only toggles it for STEP-1 self-org + restores it OFF). Pins the place->critic "
                         "transpose-SpMV atomic-scatter order -> seed-stable critic rate -> the SNc-burst δ "
                         "holds 3/3 under one config. NO sim/ edit (the deterministic branch ships); "
                         "default off = the runner's STEP-1-only determinism (byte-identical).")
    ap.add_argument("--synaptic-scaling", action="store_true",
                    help="the #5b VOLLEY-NORMALIZATION close (deferred-item-1, 2026-06-21): enable the "
                         "EXISTING Turrigiano synaptic scaling (cfg.enable_synaptic_scaling) so the critic's "
                         "afferent weights are driven toward a seed-STABLE target rate. Normalizes the "
                         "seed-variable learned volley (strong seed 44 w_near 2.475 scaled DOWN to the gentle "
                         "42/43 regime) WITHOUT starving the gentle seeds (multiplicative per-post scaling "
                         "preserves the near/far ratio = R1). Hold with --deterministic-read. NO sim/ edit.")
    ap.add_argument("--synscale-target-rate", type=float, default=0.0,
                    help="synaptic-scaling target firing rate (fraction of steps; runner default 0.02=20Hz). "
                         "Sets the seed-stable critic operating point. 0=no override (uses the cfg default).")
    ap.add_argument("--synscale-rate", type=float, default=0.0,
                    help="synaptic-scaling rate (cfg.synaptic_scaling_rate, default 0.001/step, clipped to "
                         "0.95-1.05 per step). Higher = faster convergence to the target within the "
                         "value-train window. 0=no override.")
    ap.add_argument("--synscale-ema-alpha", type=float, default=0.0,
                    help="speed the synaptic-scaling rate-estimate EMA (cfg.homeostasis_ema_alpha, default "
                         "0.0002 tau~5000 steps is too slow to register the critic rate in the short "
                         "value-train window); 0=no override.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    _GRID["n_modules"] = int(args.n_modules)
    _GRID["n_per_module"] = int(args.n_per_module)
    _GRID["drive_scale"] = float(args.grid_drive_scale)
    _HOMEO["adapt_rate"] = float(args.critic_homeo_adapt_rate)
    _HOMEO["target_rate"] = float(args.critic_homeo_target_rate)
    _HOMEO["ema_alpha"] = float(args.critic_homeo_ema_alpha)
    _DETMV["read"] = bool(args.deterministic_read)
    _SYNSCALE["enable"] = bool(args.synaptic_scaling)
    _SYNSCALE["target_rate"] = float(args.synscale_target_rate)
    _SYNSCALE["scaling_rate"] = float(args.synscale_rate)
    _SYNSCALE["ema_alpha"] = float(args.synscale_ema_alpha)

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
                               strength=args.graded_strength,
                               value_train_w_max=args.value_train_w_max,
                               settle_steps=args.settle_steps,
                               critic_gabab_max=args.critic_gabab_max)
            results[arm] = r
            sb = r.get("stage_b") or {}; grv = r.get("graded_v") or {}
            gd = r.get("graded_delta") or {}
            print(f"[grid-onbridge DELTA RESULT seed={args.seed} arm={arm}] "
                  f"LEARNS-V={sb.get('learns_v')} (w_n/w_f={sb.get('w_near_over_far')}) "
                  f"critic@near={sb.get('crit_near_hz')}Hz @far={sb.get('crit_far_hz')}Hz "
                  f"grade={sb.get('crit_grade_ratio')} | V_near={grv.get('v_near')} "
                  f"V_far={grv.get('v_far')} V_n/f={grv.get('v_near_over_far')} | "
                  f"snc-burst-delta(gap)={sb.get('snc_gap_ratio')} gabab_gap={sb.get('gabab_gap')} | "
                  f"GRADED-V delta_vnf={gd.get('delta_vnf')} (vnf_gap={gd.get('vnf_gap')}) "
                  f"delta_snc_graded={gd.get('delta_snc_graded')} "
                  f"(snc_pred={gd.get('snc_pred_near_hz')} snc_unpred={gd.get('snc_unpred_far_hz')}) | "
                  f"GABA_B-cond delta_gabab={gd.get('delta_gabab')} (gap={gd.get('gabab_cond_gap')}) "
                  f"[gb_near={gd.get('gabab_near')} gb_far={gd.get('gabab_far')}]", flush=True)
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
