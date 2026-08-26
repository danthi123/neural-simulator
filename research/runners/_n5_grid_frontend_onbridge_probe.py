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
    "normalize_drive": False,  # #5b NEXT MOVE: divisive (per-location L1) normalization of the grid drive to
                            # a constant total → removes the structural near/far MAGNITUDE asymmetry the
                            # graded plateau reads as a (non-learned) V → the only near/far V left is the
                            # LEARNED weight ratio. Applied EVERYWHERE place_sensors is driven.
    "_norm_ref_sum": 0.0,   # the reference per-location total (mean over the grid; lazily computed)
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


def _compute_grid_mean_location_sum(grid_size=None):
    """The mean per-location L1 sum of the (unscaled) grid code over the whole grid — the reference total
    for place-drive normalization (so a per-location renorm to this value leaves the AVERAGE operating
    current unchanged; it only equalizes the per-location structural magnitude asymmetry)."""
    if _GRID["_code"] is None:
        _build_grid_code()
    gs = int(grid_size if grid_size is not None else _GRID["grid_size"])
    code = _GRID["_code"]
    tot = 0.0; n = 0
    for iy in range(gs):
        for ix in range(gs):
            tot += float(np.asarray(code(float(ix), float(iy))).sum()); n += 1
    return tot / max(n, 1)


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
    raw = _GRID["_code"](float(x), float(y)).astype(np.float32)   # [0,1] grid activations at (x,y)
    # OPTIONAL place-drive normalization (#5b residual NEXT MOVE, 2026-06-21): the grid code's TOTAL
    # activation differs by location (some phase draws fire more cells at one place than another) — a
    # structural near/far MAGNITUDE asymmetry that the graded plateau reads as a near/far V INDEPENDENT of
    # the learned weight ratio (the magnitude-matched shuffle_v finding). Divisive normalization
    # (Carandini-Heeger; biology-grounded, point-neuron) to a CONSTANT total per location removes that
    # structural asymmetry → the only near/far V left is the LEARNED weight ratio. Preserves the activation
    # PATTERN (which cells fire) so the place self-org still carves selective fields. Applied EVERYWHERE the
    # runner drives place_sensors (STEP-1 self-org + value-train + reads) for a consistent normalized regime.
    raw = _grid_norm(raw)
    # the grid code is in [0,1]; scale to the render's max_intensity * drive_scale (so the grid's TOTAL
    # drive matches the render's operating current range; see _GRID["drive_scale"]).
    act = (raw * float(max_int) * float(_GRID["drive_scale"])).astype(np.float32)
    # size guard: place_sensors is sized N_PLACE_LANDMARKS*(b+d); grid must match.
    expect = int(g.N_PLACE_LANDMARKS) * (int(n_bearing) + int(n_dist))
    if act.size != expect:
        raise RuntimeError(
            f"grid code size {act.size} != place_sensors size {expect} "
            f"(set n_place_bearing/n_place_dist so {g.N_PLACE_LANDMARKS}*(b+d) == n_grid {_GRID['_n_grid']})")
    return act


def _grid_norm(raw):
    """Apply the optional per-location divisive (L1) normalization to a raw grid activation vector (the
    #5b NEXT MOVE: remove the structural per-location magnitude asymmetry). No-op when normalize_drive off."""
    if not _GRID.get("normalize_drive", False):
        return raw
    s = float(np.asarray(raw).sum())
    if s <= 1e-9:
        return raw
    ref = float(_GRID.get("_norm_ref_sum", 0.0))
    if ref <= 0.0:
        ref = _compute_grid_mean_location_sum()
        _GRID["_norm_ref_sum"] = ref
    return (np.asarray(raw, dtype=np.float32) * (ref / s)).astype(np.float32)


def _grid_act(px, py, max_int):
    """The normalized grid place-drive at (px,py) scaled to max_int*drive_scale — the SINGLE source used by
    every read closure so the place-drive normalization is applied CONSISTENTLY (reads == self-org/value-
    train regime)."""
    if _GRID["_code"] is None:
        _build_grid_code()
    raw = _grid_norm(_GRID["_code"](float(px), float(py)).astype(np.float32))
    return (raw * float(max_int) * float(_GRID["drive_scale"])).astype(np.float32)


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
#
# MODE "continuous" (the stock cfg flag): enable cfg.enable_synaptic_scaling for the whole pipeline. This
# measures the VALUE-TRAIN firing (inflated by the critic_teacher_pa=300 teacher) → over-suppresses the
# READ regime (seed-44 t03: w_near 2.475 → 0.0, critic starves). The teacher-driven value-train regime is
# the WRONG rate to normalize against.
# MODE "freeze_seam" (the corrected form): a brief Turrigiano synaptic-scaling CALIBRATION applied at the
# value-train→read FREEZE seam (the `value_input` 1.0→0.0 transition in _patched_set_gate). It measures the
# READ-regime critic@near rate (WEIGHTED-plateau, no teacher — the SAME regime stage-B reads) and
# multiplicatively scales the place→value weights toward a target read rate, iterating to convergence.
# Uniform per-postsynaptic scaling → the near/far RATIO (R1 selectivity) is PRESERVED; the ABSOLUTE volley
# is normalized to one seed-STABLE read operating point. Targets the regime that actually matters (the
# read), so it lands the strong seed in the gentle-seed band WITHOUT the teacher-regime over-suppression.
# NO sim/ edit — a multiplicative scale on cp_connections.data (the same op cfg.enable_synaptic_scaling does
# per-step, just measured in the read regime + applied once at the freeze).
_SYNSCALE = {"enable": False, "mode": "continuous", "target_rate": 0.0, "scaling_rate": 0.0,
             "ema_alpha": 0.0,
             # freeze_seam params:
             "fs_target_wnear": 0.0,  # WEIGHT-TARGET form (robust): scale w_near to this target weight
                                      # (the gentle-seed band ~0.4–0.6) in one shot — no rate measurement,
                                      # no critic-threshold-homeostasis interaction. >0 = use this instead
                                      # of the rate-target loop.
             "fs_target_hz": 40.0,    # target READ-regime critic@near rate (the gentle-seed band 17–64 Hz)
             "fs_iters": 12,          # calibration iterations (measure→scale→repeat)
             "fs_gain": 0.5,          # fractional step toward target per iter (log-domain); <1 = damped
             "fs_tol": 0.15,          # relative tolerance band around fs_target_hz (stop when within)
             "fs_down_only": False,   # homeostatic-CEILING form: only scale DOWN over-firing seeds (never
                                      # UP) → passing gentle seeds (below target) are left UNTOUCHED; only
                                      # the strong seed (over the ceiling) is normalized down.
             "fs_freeze_critic_threshold": False,  # rate-target: pin the critic threshold homeostasis
                                      # (adapt_rate=0) from the freeze on, so the calibrated read rate is
                                      # stable into stage-B (the 50-vs-273 Hz mismatch fix).
             "fs_applied": False,     # one-shot guard (only at the first 1.0→0.0 transition)
             "_log": []}


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
    # MODE "continuous" enables the stock per-step cfg flag (measures the teacher-driven value-train rate →
    # over-suppresses). MODE "freeze_seam" does NOT enable the per-step flag; it applies the read-regime
    # calibration ONCE at the value-train→read freeze in _patched_set_gate (the corrected form).
    if _SYNSCALE["enable"] and _SYNSCALE["mode"] == "continuous":
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


def _freeze_seam_normalize(br):
    """Turrigiano synaptic-scaling CALIBRATION at the value-train→read FREEZE seam (#5b deferred-item-1).

    Measures the READ-regime critic@near rate (WEIGHTED-plateau, no teacher — the SAME regime stage-B reads)
    and multiplicatively scales the place→value (place→striosome_value) weights toward fs_target_hz,
    iterating measure→scale→repeat. Uniform per-postsynaptic scaling preserves the near/far RATIO (R1
    selectivity) while normalizing the ABSOLUTE volley to a seed-STABLE read operating point. The strong
    seed (44, critic ~256 Hz) is scaled DOWN into the gentle-seed band; the gentle seeds (42/43, already in
    band) are left near-unchanged (scale ≈ 1). NO sim/ edit (a multiplicative scale on cp_connections.data,
    the same op cfg.enable_synaptic_scaling does per-step, applied once in the read regime).
    """
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = getattr(br, "region_manager", None)
    if rm is None:
        return
    d = rm.region_indices_dict()
    if "place" not in d or "striosome_value" not in d or "place_sensors" not in d:
        return
    ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
    c_idx = xp.asarray(np.asarray(d["striosome_value"], dtype=np.int64))
    place_post = np.asarray(d["striosome_value"], dtype=np.int64)
    place_pre = np.asarray(d["place"], dtype=np.int64)
    n_crit = int(c_idx.size) if hasattr(c_idx, "size") else len(c_idx)

    # the near location: the first scheduled goal (multi-goal) — derive via g._n9 helpers off the bridge's
    # captured grid_size. The grid code is FROZEN; we read at near only (a single read operating point).
    near = _SYNSCALE.get("_near") or [6.0, 6.0]
    max_int = float(_SYNSCALE.get("_max_int", 450.0))
    if _GRID["enable"]:
        if _GRID["_code"] is None:
            _build_grid_code()
        def _act():
            return _grid_act(near[0], near[1], max_int)
    else:
        grid_size = int(_SYNSCALE.get("_grid_size", 32))
        landmarks = g._n9_place_landmarks(grid_size)
        dist_max = float(grid_size) * 1.42
        def _act():
            return _orig_place_sensor_act(
                near[0], near[1], landmarks, int(_SYNSCALE.get("_n_bearing", 12)),
                int(_SYNSCALE.get("_n_dist", 8)), max_int, 0.03, 4.0, dist_max, 4.0)
    act = xp.asarray(_act(), dtype=xp.float32)

    # the place→value synapse mask on cp_connections (rows=pre place, cols=post critic), for the scale.
    coo = br.cp_connections.tocoo()
    rows = coo.row.get() if hasattr(coo.row, "get") else np.asarray(coo.row)
    cols = coo.col.get() if hasattr(coo.col, "get") else np.asarray(coo.col)
    pv_mask = np.isin(rows, place_pre) & np.isin(cols, place_post)
    if not pv_mask.any():
        pv_mask = np.isin(rows, place_post) & np.isin(cols, place_pre)
    if not pv_mask.any():
        return
    pv_mask_xp = xp.asarray(pv_mask)
    p_idx = xp.asarray(place_pre)

    cfg = br.core_config

    # ── WEIGHT-TARGET mode (the robust form) ──────────────────────────────────────────────────
    # Scale w_near to a TARGET WEIGHT directly (the gentle-seed band ~0.4–0.6), ONE shot, no
    # rate measurement → no interaction with the critic's threshold homeostasis (which moves the
    # threshold during a rate-calibration and isn't stable into stage-B). A set-point form of
    # Turrigiano scaling: normalize the place→value volley STRENGTH (the weight) to a common set
    # point. Uniform multiplicative scale → the near/far RATIO (R1) is preserved; the strong seed
    # (44, w_near 2.475) lands at the set point matching the gentle seeds (42/43, ~0.4–0.6) so the
    # WHOLE read regime (incl. stage-B's threshold dynamics) behaves like the gentle seeds → crit@far
    # falls sub-threshold → the SNc-burst δ holds. With fs_down_only, leave seeds already at/below the
    # set point UNTOUCHED.
    target_wnear = float(_SYNSCALE.get("fs_target_wnear", 0.0))
    if target_wnear and target_wnear > 0:
        down_only = bool(_SYNSCALE.get("fs_down_only", False))
        # measure the near-ACTIVE place set (which place cells fire at near) on the FROZEN grid code,
        # then mean w over their place→value synapses — the SAME w_near the stage-B LEARNS-V gate reads.
        _saved_lr0 = cfg.reward_learning_rate
        cfg.reward_learning_rate = 0.0
        br.cp_external_input_current[:] = xp.float32(0.0)
        br.cp_external_input_current[ps_idx] = act
        n_p = int(p_idx.size) if hasattr(p_idx, "size") else len(p_idx)
        ens = xp.zeros(n_p, dtype=xp.float32)
        for _ in range(80):
            br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
            ens += br.cp_firing_states[p_idx].astype(xp.float32)
        cfg.reward_learning_rate = _saved_lr0
        br.cp_external_input_current[:] = xp.float32(0.0)
        ens_h = ens.get() if hasattr(ens, "get") else np.asarray(ens)
        near_active = place_pre[ens_h > 0]
        if near_active.size == 0:
            near_active = place_pre  # fallback: all place cells
        near_rows_mask = np.isin(rows, near_active) & np.isin(cols, place_post)
        if not near_rows_mask.any():
            near_rows_mask = pv_mask
        data_h = (br.cp_connections.data.get() if hasattr(br.cp_connections.data, "get")
                  else np.asarray(br.cp_connections.data))
        w_near_now = float(data_h[near_rows_mask].mean()) if near_rows_mask.any() else 0.0
        scale = 1.0 if w_near_now <= 1e-9 else float(target_wnear / w_near_now)
        if down_only and scale >= 1.0:
            scale = 1.0   # already at/below the set point → leave the gentle seed untouched
        if abs(scale - 1.0) > 1e-6:
            br.cp_connections.data[pv_mask_xp] = br.cp_connections.data[pv_mask_xp] * xp.float32(scale)
        w_near_after = w_near_now * scale
        _SYNSCALE["_log"] = [{"mode": "weight_target", "n_near_active": int(near_active.size),
                              "w_near_before": float(w_near_now), "scale": float(scale),
                              "w_near_after": float(w_near_after), "target_wnear": float(target_wnear),
                              "down_only": down_only}]
        print(f"[freeze-seam synaptic-scaling WEIGHT-TARGET] w_near {w_near_now:.3f} -> {w_near_after:.3f} "
              f"(target {target_wnear:.3f}, scale {scale:.3f}, {int(near_active.size)} near-active place cells)",
              flush=True)
        return

    # ── RATE-TARGET mode (the original; interacts with critic threshold homeostasis) ───────────
    # WEIGHTED-plateau read regime (matches stage-B's _critic_rate); restore after.
    _saved_wd = bool(getattr(cfg, "coincidence_weighted_drive", False))
    _saved_kth = float(getattr(cfg, "coincidence_k_threshold", 4.0))
    _saved_lr = cfg.reward_learning_rate
    cfg.coincidence_weighted_drive = True
    cfg.coincidence_k_threshold = float(_SYNSCALE.get("_coin_thr", _saved_kth))
    cfg.reward_learning_rate = 0.0

    def _reset_read():
        for nm in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise",
                   "cp_conductance_g_graded_plateau", "cp_conductance_g_graded_plateau_rise",
                   "cp_conductance_g_gabab"):
            arr = getattr(br, nm, None)
            if arr is not None:
                arr[:] = xp.float32(0.0)
        if (getattr(br, "cp_membrane_potential_v", None) is not None
                and getattr(br, "cp_izh_vr", None) is not None):
            br.cp_membrane_potential_v[c_idx] = br.cp_izh_vr[c_idx]
            if getattr(br, "cp_recovery_variable_u", None) is not None:
                br.cp_recovery_variable_u[c_idx] = xp.float32(0.0)
        br.cp_external_input_current[:] = xp.float32(0.0)

    def _crit_rate(*, n_meas=120, warmup=30):
        _reset_read()
        br.cp_external_input_current[:] = xp.float32(0.0)
        br.cp_external_input_current[ps_idx] = act
        spk = 0; m = 0
        for t in range(int(n_meas)):
            br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
            if t >= warmup:
                spk += int(br.cp_firing_states[c_idx].sum()); m += 1
        br.cp_external_input_current[:] = xp.float32(0.0)
        return spk / max(n_crit, 1) / max(m * 1e-3, 1e-9)

    target = float(_SYNSCALE["fs_target_hz"])
    gain = float(_SYNSCALE["fs_gain"])
    tol = float(_SYNSCALE["fs_tol"])
    down_only = bool(_SYNSCALE.get("fs_down_only", False))
    # optionally FREEZE the critic's intrinsic threshold homeostasis from here on (NOT restored): the
    # rate-calibration fires the critic hard → its homeostatic threshold drifts up during calibration and
    # is not stable into stage-B (the 50-vs-273 Hz mismatch). Pinning the threshold-adapt rate to 0 makes
    # the critic threshold STATIC at its post-value-train value → my calibrated read rate persists into the
    # stage-B read. The threshold stays the cell's own (intrinsic, neural); only its ADAPTATION is frozen.
    if bool(_SYNSCALE.get("fs_freeze_critic_threshold", False)):
        cfg.homeostasis_threshold_adapt_rate = 0.0
    log = []
    r0 = _crit_rate()
    log.append({"iter": -1, "rate_hz": float(r0), "scale": 1.0})
    for it in range(int(_SYNSCALE["fs_iters"])):
        r = _crit_rate()
        if down_only and r <= target * (1.0 + tol):
            # homeostatic-CEILING: at/below the ceiling already → leave this (gentle) seed UNTOUCHED.
            log.append({"iter": it, "rate_hz": float(r), "scale": 1.0, "below_ceiling": True})
            break
        if r <= 1e-6:
            # critic silent — scale UP (toward target) so we don't get stuck at the floor.
            scale = 1.0 + gain
        elif abs(r - target) <= tol * target:
            log.append({"iter": it, "rate_hz": float(r), "scale": 1.0, "converged": True})
            break
        else:
            # log-domain damped step toward target: scale = (target/r)^gain, clipped per-iter for stability.
            ratio = max(target, 1e-6) / max(r, 1e-6)
            scale = float(np.clip(ratio ** gain, 0.5, 2.0))
        # apply the uniform multiplicative scale to the place→value synapses (preserves near/far ratio).
        br.cp_connections.data[pv_mask_xp] = br.cp_connections.data[pv_mask_xp] * xp.float32(scale)
        log.append({"iter": it, "rate_hz": float(r), "scale": float(scale)})
    r_final = _crit_rate()
    log.append({"iter": "final", "rate_hz": float(r_final), "scale": 1.0})

    cfg.coincidence_weighted_drive = _saved_wd
    cfg.coincidence_k_threshold = _saved_kth
    cfg.reward_learning_rate = _saved_lr
    _reset_read()
    _SYNSCALE["_log"] = log
    print(f"[freeze-seam synaptic-scaling] read-regime critic@near {r0:.1f}Hz -> {r_final:.1f}Hz "
          f"(target {target:.0f}Hz, {len(log)-2} scale steps)", flush=True)


# ── V-location-shuffle lesion (the CLEAN metric-lesion anti-cheat, 2026-06-21) ────────────────
# The `scramble` arm permutes grid PHASES but leaves a DECORRELATED, location-DISCRIMINABLE code, so the
# place self-org carves selective fields + the value-train learns a genuine near/far V on it → the δ does
# NOT collapse (the scramble lesions the periodic METRIC but NOT the learnability of a near/far V; the
# determinism-close doc's "no spatially-selective V" framing for scramble was wrong — its baseline collapse
# was an over-clamp artifact). The CLEAN lesion that MUST collapse the LEARNED δ: at the freeze, randomly
# PERMUTE the learned place→value weights ACROSS the place presynaptic neurons. This destroys WHERE V is
# high (the learned near/far spatial correspondence) while preserving the weight DISTRIBUTION → if the δ
# requires the LEARNED spatial V (it does), the shuffled V no longer maps near>>far → the δ collapses. The
# direct anti-cheat for "the δ is the genuine learned spatial RPE, not a graded-V structural artifact".
_LESION_SHUFFLE_V = {"enable": False, "_applied": False}


def _lesion_shuffle_v(br):
    """Permute the learned place→value weights across the place presynaptic neurons (the CLEAN
    metric-lesion). Destroys the learned near/far spatial V structure; preserves the weight distribution."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    rm = getattr(br, "region_manager", None)
    if rm is None:
        return
    d = rm.region_indices_dict()
    if "place" not in d or "striosome_value" not in d:
        return
    place_pre = np.asarray(d["place"], dtype=np.int64)
    place_post = np.asarray(d["striosome_value"], dtype=np.int64)
    coo = br.cp_connections.tocoo()
    rows = coo.row.get() if hasattr(coo.row, "get") else np.asarray(coo.row)
    cols = coo.col.get() if hasattr(coo.col, "get") else np.asarray(coo.col)
    pv = np.isin(rows, place_pre) & np.isin(cols, place_post)
    if not pv.any():
        pv = np.isin(rows, place_post) & np.isin(cols, place_pre)
    if not pv.any():
        return
    data_h = (br.cp_connections.data.get() if hasattr(br.cp_connections.data, "get")
              else np.asarray(br.cp_connections.data))
    idx = np.where(pv)[0]
    rng = np.random.default_rng(20260621)
    shuffled = data_h[idx].copy()
    rng.shuffle(shuffled)              # permute the learned weights across the place→value synapses
    new_data = data_h.copy()
    new_data[idx] = shuffled
    br.cp_connections.data[:] = xp.asarray(new_data, dtype=br.cp_connections.data.dtype)
    print(f"[lesion shuffle-V] permuted {idx.size} place->value weights "
          f"(destroyed learned near/far spatial V; weight dist preserved)", flush=True)


def _patched_set_gate(self, name, value):
    if _CLOSEA["enable"] and _CLOSEA["readout_only"] and name == "value_input":
        if float(value) >= 1.0:
            _CLOSEA["_armed"] = True
        elif float(value) <= 0.0 and _CLOSEA["_armed"]:
            self.core_config.coincidence_plateau_strength = 0.0
            self.core_config.graded_plateau_strength = float(_CLOSEA["strength"])
            _CLOSEA["_armed"] = False
            # the CLEAN metric-lesion anti-cheat: shuffle the learned place→value V FIRST (on the raw
            # value-trained weights), THEN normalize — so the shuffled (flat-ratio) lesion is
            # MAGNITUDE-MATCHED to the grid arm (both land at w_near→target) and the ONLY difference is the
            # learned near/far RATIO. If the δ then collapses, it is the genuine LEARNED ratio; if it
            # survives at flat ratio + matched magnitude, it is the graded-V structural contamination.
            if _LESION_SHUFFLE_V["enable"] and not _LESION_SHUFFLE_V["_applied"]:
                _LESION_SHUFFLE_V["_applied"] = True
                try:
                    _lesion_shuffle_v(self)
                except Exception as e:
                    print(f"[lesion shuffle-V] failed: {e!r}", flush=True)
            # the freeze-seam volley-normalization (#5b deferred-item-1): the value_input 1.0→0.0
            # transition IS the value-train→read freeze. Run the read-regime synaptic-scaling
            # calibration on the place→value weights here (one-shot), BEFORE the stage-B reads. (Runs
            # AFTER any shuffle so the shuffled lesion is magnitude-matched to the grid arm.)
            if (_SYNSCALE["enable"] and _SYNSCALE["mode"] == "freeze_seam"
                    and not _SYNSCALE["fs_applied"]):
                _SYNSCALE["fs_applied"] = True
                try:
                    _freeze_seam_normalize(self)
                except Exception as e:
                    print(f"[freeze-seam] normalize failed: {e!r}", flush=True)
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
    _GRID["_norm_ref_sum"] = 0.0   # recompute per (seed/scramble) code draw
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
                return _grid_act(px, py, max_int)
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
                return _grid_act(px, py, max_int)
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


def _read_td_delta(captured, result, *, settle_steps=0):
    """RANK-1 (#5b value-read cleanup, 2026-06-22): the TEMPORAL-DIFFERENCE dopamine read.

    The raw single-state read in `_read_graded_v_delta` is the `no_bootstrap` form
    `delta ~= r - V(s)` evaluated separately at NEAR and FAR (the `gabab_gap` ratio
    snc_unpred(FAR) / snc_pred(NEAR)) -> it reads TOTAL afferent magnitude (structural place-code
    baseline + learned increment), which is why it SURVIVES the magnitude-matched `shuffle_v`
    control (the bug: it reads the place code's structural geometry, not the learned value).

    The biologically-correct phasic-dopamine signal (Schultz-Dayan-Montague 1997; Schultz 1998;
    catalog C.28/C.30/C.31; `sim/td_value_critic.py` `delta = r + GAMMA*v_tp1 - v_t`) is a
    TEMPORAL-DIFFERENCE error -- a DIFFERENCE between successive-state values, which cancels any
    baseline that is consistent across the two states. We read it across the FAR->NEAR transition
    (the agent approaches the goal): FAR is the state being left (s), NEAR the state entered (s'):

        delta_TD = r + GAMMA * V(near) - V(far)

    using the SAME per-location reads the raw read uses. Three faithful estimators of V are reported
    (all reuse-by-import, NO sim/ edit); GAMMA is the td_value_critic default (0.95):

      (td1) graded-V TD   : V(loc) = the graded-plateau conductance read (the critic's learned value
                            estimate, `_read_graded_v_near_far`). r set to the V-scale reference
                            r_ref = V(far) (the unpredicted-reward magnitude on the same scale).
      (td2) snc-burst TD  : V(loc) estimated from the SNc burst RPE: at FAR the burst is unsuppressed
                            (the reward, ~r), at NEAR it is suppressed by V -> V(near) ~= burst(FAR) -
                            burst(NEAR); V(far) ~= 0; r ~= burst(FAR). The bootstrapped difference of
                            the burst-derived RPE.
      (td3) adjacent TD   : the cleanest structural-baseline-cancellation test -- read V at NEAR and
                            at an ADJACENT near-neighbour MID location; the structural baseline is
                            ~common across adjacent states, so delta_TD = r + GAMMA*V(near) - V(mid)
                            isolates the LEARNED near-vs-neighbour value step.

    The MAKE-OR-BREAK anti-cheat: under the magnitude-matched `shuffle_v` the learned near/far V
    DIFFERENCE is destroyed (w_n/f -> ~1.0) while the structural magnitude is matched. The raw read
    survives this; the TD read MUST collapse it (delta_TD -> ~its unlearned value) iff the TD
    difference genuinely reads the LEARNED gradient and not the structural baseline. If the TD read
    ALSO survives shuffle_v, the structural/learned inseparability is the honest point-neuron
    boundary (the scoping MOVE-4 fallback) -- report it, do not force a GO.

    `settle_steps` (the SURPASS move (b)) is forwarded to the per-location burst reads.
    """
    br = _CLOSEA["bridge"]
    if br is None:
        return None
    try:
        from sim.backend import get_backend
        from sim.td_value_critic import GAMMA
        xp, _ = get_backend()
        rm = getattr(br, "region_manager", None)
        if rm is None:
            return None
        d = rm.region_indices_dict()
        for need in ("striosome_value", "place_sensors"):
            if need not in d:
                return {"error": f"missing region {need}"}
        c_idx = xp.asarray(np.asarray(d["striosome_value"], dtype=np.int64))
        ps_idx = xp.asarray(np.asarray(d["place_sensors"], dtype=np.int64))
        max_int = float(captured.get("place_sensor_max_intensity", 450.0))
        sb = (result or {}).get("stage_b_smoke") or {}
        near = sb.get("near") or [6.0, 6.0]
        far = sb.get("far") or [1.0, 1.0]
        # the ADJACENT near-neighbour MID state (one grid-step toward far from near) for td3.
        mid = [float(near[0]) - 1.0, float(near[1]) - 1.0]

        if _GRID["enable"]:
            if _GRID["_code"] is None:
                _build_grid_code()
            def _act(px, py):
                return _grid_act(px, py, max_int)
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

        # (td1) graded-V learned value at each location (the SAME plateau read the raw read's V uses).
        def _v_graded_at(px, py, *, n_meas=120, warmup=40):
            saved = br.core_config.reward_learning_rate
            br.core_config.reward_learning_rate = 0.0
            if getattr(br, "cp_conductance_g_graded_plateau", None) is not None:
                br.cp_conductance_g_graded_plateau[:] = xp.float32(0.0)
                br.cp_conductance_g_graded_plateau_rise[:] = xp.float32(0.0)
            br.cp_external_input_current[:] = xp.float32(0.0)
            br.cp_external_input_current[ps_idx] = xp.asarray(_act(px, py), dtype=xp.float32)
            vsum = 0.0; m = 0
            for t in range(int(n_meas)):
                br._run_one_simulation_step(); br.runtime_state.current_time_step += 1
                if t >= warmup:
                    vsum += float(br.cp_conductance_g_graded_plateau[c_idx].mean()); m += 1
            br.core_config.reward_learning_rate = saved
            br.cp_external_input_current[:] = xp.float32(0.0)
            return vsum / max(m, 1)

        v_near = _v_graded_at(near[0], near[1])
        v_far = _v_graded_at(far[0], far[1])
        v_mid = _v_graded_at(mid[0], mid[1])

        # (td2) the per-location SNc burst RPE reads (reuse the raw read's per-location burst fn).
        rawd = _read_graded_v_delta(captured, result, settle_steps=settle_steps) or {}
        burst_near = float(rawd.get("snc_pred_near_hz", 0.0))   # r - V(near)  (suppressed)
        burst_far = float(rawd.get("snc_unpred_far_hz", 0.0))   # r - V(far) ~= r (unsuppressed)

        # ── TD forms (delta = r + GAMMA*V(s') - V(s); FAR=s, NEAR=s') ────────────────────────────
        # (td1) graded-V TD: r_ref on the V scale = V(far) (the unpredicted-reward magnitude). The
        # learned-value step is the term that must vanish under shuffle_v.
        r_ref_graded = v_far
        delta_td_graded = r_ref_graded + GAMMA * v_near - v_far  # = GAMMA*v_near + (1-GAMMA)*... -> learned step
        # the discriminating ratio: bootstrapped value at NEAR vs the value baseline at FAR. A learned
        # near>>far step gives a ratio >> 1; a destroyed step (shuffle_v) -> ~1.
        td_graded_ratio = (GAMMA * v_near + r_ref_graded) / max(v_far + r_ref_graded, 1e-9)
        td_graded_gap = bool(td_graded_ratio >= 1.30)

        # (td2) snc-burst TD: V(near) = burst_far - burst_near; V(far) = 0; r = burst_far.
        v_near_burst = max(burst_far - burst_near, 0.0)
        r_burst = burst_far
        delta_td_burst = r_burst + GAMMA * v_near_burst - 0.0
        # the discriminating ratio: with the learned suppression at near gone (shuffle_v leaves near
        # suppressed by STRUCTURE), V(near)_burst stays large -> survives; with the learned suppression
        # the ONLY source of the near/far burst gap, a destroyed learned ratio -> burst_near ~= burst_far
        # -> V(near)_burst -> 0 -> delta_td_burst -> r (no TD lift). Ratio = delta_td_burst / r.
        td_burst_ratio = delta_td_burst / max(r_burst, 1e-9)
        td_burst_gap = bool(td_burst_ratio >= 1.30)

        # (td3) adjacent-state TD: NEAR vs its near-neighbour MID. The structural baseline is ~common
        # across adjacent states -> the difference isolates the LEARNED local value step. THIS is the
        # cleanest cancellation test the scoping flags.
        r_ref_adj = v_mid
        delta_td_adjacent = r_ref_adj + GAMMA * v_near - v_mid
        td_adjacent_ratio = (GAMMA * v_near + r_ref_adj) / max(v_mid + r_ref_adj, 1e-9)
        td_adjacent_gap = bool(td_adjacent_ratio >= 1.30)

        return {
            "settle_steps": int(settle_steps), "gamma": float(GAMMA),
            "near": [float(near[0]), float(near[1])],
            "far": [float(far[0]), float(far[1])],
            "mid": [float(mid[0]), float(mid[1])],
            "v_near": float(v_near), "v_far": float(v_far), "v_mid": float(v_mid),
            "v_near_over_far": float(v_near / max(v_far, 1e-9)),
            "burst_near_hz": float(burst_near), "burst_far_hz": float(burst_far),
            # (td1) graded-V TD
            "delta_td_graded": float(delta_td_graded),
            "td_graded_ratio": float(td_graded_ratio), "td_graded_gap": td_graded_gap,
            # (td2) snc-burst TD
            "v_near_burst": float(v_near_burst), "r_burst": float(r_burst),
            "delta_td_burst": float(delta_td_burst),
            "td_burst_ratio": float(td_burst_ratio), "td_burst_gap": td_burst_gap,
            # (td3) adjacent-state TD (the cleanest baseline-cancellation test)
            "delta_td_adjacent": float(delta_td_adjacent),
            "td_adjacent_ratio": float(td_adjacent_ratio), "td_adjacent_gap": td_adjacent_gap,
        }
    except Exception as e:
        return {"error": repr(e)}


def _run_delta_arm(arm, seed, *, value_train_trials, single_goal, readout_only,
                   center, slope, strength, value_train_w_max=0.0, settle_steps=0,
                   critic_gabab_max=0.0, td_read=False):
    """One delta-verdict arm. arm in {grid, render, scramble, no_learn, lesion, shuffle_v}."""
    grid_on = arm in ("grid", "scramble", "no_learn", "lesion", "shuffle_v")
    scramble = (arm == "scramble")
    _set_grid_mode(enable=grid_on, scramble=scramble, seed=seed)
    _CLOSEA["enable"] = True   # graded plateau on for every arm (render arm = CLOSE A R1-LIMIT contrast)
    _CLOSEA["center"] = float(center)
    _CLOSEA["slope"] = float(slope)
    _CLOSEA["strength"] = (0.0 if arm == "lesion" else float(strength))
    _CLOSEA["readout_only"] = bool(readout_only) and arm != "lesion"
    _CLOSEA["bridge"] = None
    _CLOSEA["_armed"] = False
    # shuffle_v = the CLEAN metric-lesion: grid + graded + value-train, then PERMUTE the learned place→value
    # V across place neurons at the freeze (destroys the learned spatial near/far V) → the δ MUST collapse.
    _LESION_SHUFFLE_V["enable"] = (arm == "shuffle_v")
    _LESION_SHUFFLE_V["_applied"] = False

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

    # freeze-seam synaptic-scaling setup: tell the normalize fn the READ operating point (near=[6,6], the
    # multi-goal first scheduled goal; the weighted-plateau k = coincidence_threshold=12) + reset the
    # one-shot guard so it runs at THIS arm's value-train→read freeze.
    if _SYNSCALE["enable"] and _SYNSCALE["mode"] == "freeze_seam":
        _SYNSCALE["fs_applied"] = False
        _SYNSCALE["_log"] = []
        _SYNSCALE["_near"] = [6.0, 6.0]
        _SYNSCALE["_max_int"] = float(captured.get("place_sensor_max_intensity", 450.0))
        _SYNSCALE["_grid_size"] = int(captured.get("grid_size", 32))
        _SYNSCALE["_coin_thr"] = float(captured.get("coincidence_threshold", 12))
        if grid_on:
            _SYNSCALE["_n_bearing"] = int(n_bearing)
            _SYNSCALE["_n_dist"] = int(n_dist)

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
    # RANK-1 (#5b value-read cleanup): the TD-difference dopamine read (opt-in). Reuses the same
    # per-location V / burst reads; the magnitude-matched shuffle_v is the make-or-break (the raw
    # read SURVIVES it, the TD read must COLLAPSE it iff it reads learned value not structure).
    td_delta = _read_td_delta(captured, result, settle_steps=settle_steps) if td_read else None
    sb = (result or {}).get("stage_b_smoke") or {}
    synscale_log = (list(_SYNSCALE.get("_log") or [])
                    if (_SYNSCALE["enable"] and _SYNSCALE["mode"] == "freeze_seam") else None)
    return {"arm": arm, "grid_on": grid_on, "scramble": scramble, "n_grid": n_grid,
            "graded_v": grv, "graded_delta": graded_delta, "td_delta": td_delta, "stage_b": sb,
            "synscale_freeze_seam_log": synscale_log,
            "goal_free_asserted": _GRID.get("_asserted_goal_free", False) if grid_on else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step2-selectivity", action="store_true",
                    help="Step 2: on-bridge place selectivity (near-neighbour read cos on real spikes)")
    ap.add_argument("--arm", type=str, default="grid",
                    choices=["grid", "render", "scramble", "no_learn", "lesion", "shuffle_v"])
    ap.add_argument("--all-arms", action="store_true",
                    help="Step 3: run grid + render + scramble + no_learn + lesion (the delta battery)")
    ap.add_argument("--with-shuffle-v", action="store_true",
                    help="add the CLEAN metric-lesion 'shuffle_v' arm to --all-arms (the learned place->value "
                         "V is permuted across place neurons at the freeze → the δ MUST collapse; the "
                         "anti-cheat that the δ is the genuine LEARNED spatial RPE, not a graded-V artifact).")
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
    ap.add_argument("--normalize-place-drive", action="store_true",
                    help="the #5b residual NEXT MOVE: per-location DIVISIVE (L1) normalization of the grid "
                         "place drive to a CONSTANT total (Carandini-Heeger; point-neuron). Removes the "
                         "structural per-location MAGNITUDE asymmetry the graded plateau reads as a "
                         "non-learned near/far V -> the only near/far V left is the LEARNED weight ratio. "
                         "Applied EVERYWHERE place_sensors is driven (self-org + value-train + reads). With "
                         "this ON, the magnitude-matched shuffle_v lesion should COLLAPSE (proving the grid "
                         "δ is then genuinely learned).")
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
    ap.add_argument("--td-read", action="store_true",
                    help="RANK-1 (#5b value-read cleanup, 2026-06-22): add the biologically-correct "
                         "TEMPORAL-DIFFERENCE dopamine read delta = r + GAMMA*V(near) - V(far) (a "
                         "DIFFERENCE between successive states; td_value_critic GAMMA=0.95) alongside the "
                         "raw single-state read, so the A/B is one process. The raw `gabab_gap` reads TOTAL "
                         "afferent magnitude (structural + learned) and SURVIVES the magnitude-matched "
                         "shuffle_v (the bug); the TD difference must COLLAPSE shuffle_v iff it reads the "
                         "LEARNED gradient not the structural place-code geometry. NO sim/ edit "
                         "(reuse-by-import of td_value_critic). Default off = the raw read only.")
    ap.add_argument("--synaptic-scaling", action="store_true",
                    help="the #5b VOLLEY-NORMALIZATION close (deferred-item-1, 2026-06-21): Turrigiano "
                         "synaptic scaling on the place->value path so the critic's afferent weights are "
                         "driven toward a seed-STABLE target rate. Normalizes the seed-variable learned "
                         "volley (strong seed 44 w_near 2.475 scaled DOWN to the gentle 42/43 regime) "
                         "WITHOUT starving the gentle seeds (multiplicative per-post scaling preserves the "
                         "near/far ratio = R1). Hold with --deterministic-read. NO sim/ edit. See "
                         "--synscale-mode.")
    ap.add_argument("--synscale-mode", type=str, default="freeze_seam",
                    choices=["continuous", "freeze_seam"],
                    help="continuous = the stock per-step cfg.enable_synaptic_scaling (measures the "
                         "teacher-driven VALUE-TRAIN rate -> over-suppresses). freeze_seam (default) = a "
                         "read-regime calibration applied ONCE at the value-train->read freeze (measures "
                         "the WEIGHTED-plateau critic@near, the regime stage-B reads -> normalizes "
                         "correctly).")
    ap.add_argument("--synscale-fs-target-wnear", type=float, default=0.0,
                    help="freeze_seam WEIGHT-TARGET (robust): scale w_near to this TARGET WEIGHT (the "
                         "gentle-seed band ~0.4-0.6) in ONE shot — no rate measurement, no interaction "
                         "with the critic threshold homeostasis. >0 = use instead of the rate-target loop. "
                         "The recommended form (the rate-target loop interacts with the critic's own "
                         "threshold homeostasis, which is not stable into stage-B).")
    ap.add_argument("--synscale-fs-target-hz", type=float, default=40.0,
                    help="freeze_seam RATE-TARGET: target READ-regime critic@near rate (Hz; the "
                         "gentle-seed band 17-64Hz). Default 40. (Use --synscale-fs-target-wnear for the "
                         "robust weight-target form.)")
    ap.add_argument("--synscale-fs-iters", type=int, default=12,
                    help="freeze_seam: calibration iterations (measure->scale->repeat). Default 12.")
    ap.add_argument("--synscale-fs-gain", type=float, default=0.5,
                    help="freeze_seam: log-domain step fraction toward target per iter (<1 = damped). "
                         "Default 0.5.")
    ap.add_argument("--synscale-fs-tol", type=float, default=0.15,
                    help="freeze_seam: relative tolerance band around the target (stop when within). "
                         "Default 0.15.")
    ap.add_argument("--synscale-fs-down-only", action="store_true",
                    help="freeze_seam: homeostatic-CEILING form — only scale DOWN over-firing seeds (never "
                         "UP). Passing gentle seeds (below the target ceiling) are left UNTOUCHED; only the "
                         "strong over-firing seed is normalized down. The cleanest form (perturbs no "
                         "passing seed).")
    ap.add_argument("--synscale-fs-freeze-critic-threshold", action="store_true",
                    help="freeze_seam RATE-TARGET: pin the critic threshold homeostasis (adapt_rate=0) from "
                         "the freeze on, so the calibrated read rate is stable into stage-B (fixes the "
                         "rate-target 50-vs-273Hz mismatch where the critic's own threshold homeostasis "
                         "drifts during calibration).")
    ap.add_argument("--synscale-target-rate", type=float, default=0.0,
                    help="continuous: synaptic-scaling target firing rate (fraction of steps; runner "
                         "default 0.02=20Hz). 0=no override (uses the cfg default).")
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
    _GRID["normalize_drive"] = bool(args.normalize_place_drive)
    _HOMEO["adapt_rate"] = float(args.critic_homeo_adapt_rate)
    _HOMEO["target_rate"] = float(args.critic_homeo_target_rate)
    _HOMEO["ema_alpha"] = float(args.critic_homeo_ema_alpha)
    _DETMV["read"] = bool(args.deterministic_read)
    _SYNSCALE["enable"] = bool(args.synaptic_scaling)
    _SYNSCALE["mode"] = str(args.synscale_mode)
    _SYNSCALE["target_rate"] = float(args.synscale_target_rate)
    _SYNSCALE["scaling_rate"] = float(args.synscale_rate)
    _SYNSCALE["ema_alpha"] = float(args.synscale_ema_alpha)
    _SYNSCALE["fs_target_wnear"] = float(args.synscale_fs_target_wnear)
    _SYNSCALE["fs_target_hz"] = float(args.synscale_fs_target_hz)
    _SYNSCALE["fs_iters"] = int(args.synscale_fs_iters)
    _SYNSCALE["fs_gain"] = float(args.synscale_fs_gain)
    _SYNSCALE["fs_tol"] = float(args.synscale_fs_tol)
    _SYNSCALE["fs_down_only"] = bool(args.synscale_fs_down_only)
    _SYNSCALE["fs_freeze_critic_threshold"] = bool(args.synscale_fs_freeze_critic_threshold)

    # install the monkeypatches (the grid render + the graded plateau init/gate hooks).
    g._n9_place_sensor_act = _grid_place_sensor_act
    SimulationBridge._initialize_simulation_data = _patched_init
    SimulationBridge.set_plasticity_gate = _patched_set_gate

    out_obj = {"seed": args.seed, "n_modules": args.n_modules, "n_per_module": args.n_per_module}

    if args.step2_selectivity:
        out_obj["step2"] = _run_step2_selectivity(args.seed)
    else:
        arms = (["grid", "render", "scramble", "no_learn", "lesion"]
                + (["shuffle_v"] if args.with_shuffle_v else [])
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
                               critic_gabab_max=args.critic_gabab_max,
                               td_read=args.td_read)
            results[arm] = r
            sb = r.get("stage_b") or {}; grv = r.get("graded_v") or {}
            gd = r.get("graded_delta") or {}
            td = r.get("td_delta") or {}
            if td and not td.get("error"):
                print(f"[grid-onbridge TD-READ seed={args.seed} arm={arm}] "
                      f"V_near={td.get('v_near')} V_far={td.get('v_far')} V_mid={td.get('v_mid')} "
                      f"V_n/f={td.get('v_near_over_far')} | "
                      f"(td1 graded-V) delta_td={td.get('delta_td_graded')} "
                      f"ratio={td.get('td_graded_ratio')} GAP={td.get('td_graded_gap')} | "
                      f"(td2 snc-burst) burst_n={td.get('burst_near_hz')} burst_f={td.get('burst_far_hz')} "
                      f"delta_td={td.get('delta_td_burst')} ratio={td.get('td_burst_ratio')} "
                      f"GAP={td.get('td_burst_gap')} | "
                      f"(td3 adjacent) delta_td={td.get('delta_td_adjacent')} "
                      f"ratio={td.get('td_adjacent_ratio')} GAP={td.get('td_adjacent_gap')}", flush=True)
            elif td and td.get("error"):
                print(f"[grid-onbridge TD-READ seed={args.seed} arm={arm}] ERROR {td.get('error')}", flush=True)
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
