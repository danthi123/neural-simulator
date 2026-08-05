"""Vectorized Stage B candidate metrics that preserve filed estimator semantics.

This module is deliberately separate from the filed evidence analyzers.  It is
for candidate screening only: the authority for a reported deactivation time
constant remains :func:`tools.v14_stageB_fast_channel_clamp_analysis._fit_tail`.
"""

from __future__ import annotations

from typing import Any

import numpy as np


TAU_MIN_MS = 1.0e-5
TAU_MAX_MS = 1000.0
_GRID_POINTS = 257
_REFINEMENT_STEPS = 32
_LOG_TAU_MIN = float(np.log(TAU_MIN_MS))
_LOG_TAU_MAX = float(np.log(TAU_MAX_MS))
_GOLDEN_LOWER = 0.3819660112501051
_GOLDEN_UPPER = 0.6180339887498949
_FLOAT64_EPSILON = np.finfo(np.float64).eps

__all__ = ["TAU_MAX_MS", "TAU_MIN_MS", "fit_deactivation_tails"]


def fit_deactivation_tails(elapsed_ms: Any, traces: Any, xp: Any) -> Any:
    """Fit candidate-first single-exponential deactivation tails.

    ``traces`` must have shape ``(candidate, command, time)`` and the result
    has shape ``(candidate, command)``.  The first sample is deliberately
    omitted, matching the filed SciPy authority.  For each trial tau, the
    asymptote and amplitude are solved by a batched two-column least-squares
    fit.  A fixed global log-tau grid finds the local basin associated with
    the filed authority's 1 ms tau seed, then a deterministic vectorized
    golden-section search refines that bracket.

    The function is intentionally fail-closed.  A non-finite input, malformed
    grid, degenerate tail, or numerically unidentifiable fit raises instead of
    returning a host-side fallback or an arbitrary tau.
    """

    _validate_xp(xp)
    time = _elapsed_array(elapsed_ms, xp)
    values = _trace_array(traces, time.size, xp)
    fit_time = time[1:]
    fit_values = values[..., 1:]
    _validate_identifiable_tails(fit_values, xp)

    grid_log_tau = xp.asarray(
        np.linspace(_LOG_TAU_MIN, _LOG_TAU_MAX, _GRID_POINTS, dtype=np.float64),
        dtype=xp.float64,
    )
    grid_sse = _grid_sse(fit_time, fit_values, grid_log_tau, xp)
    if _scalar_bool(xp.any(~xp.isfinite(xp.min(grid_sse, axis=-1)))):
        raise FloatingPointError("deactivation tail fit is numerically unidentifiable")

    grid_index = _authority_seeded_grid_index(grid_sse, grid_log_tau, xp)
    lower_index = xp.maximum(grid_index - 1, 0)
    upper_index = xp.minimum(grid_index + 1, _GRID_POINTS - 1)
    lower = grid_log_tau[lower_index]
    upper = grid_log_tau[upper_index]

    # This fixed-count search is intentionally independent of candidate values.
    # It never introduces a candidate/command Python loop or data-dependent
    # convergence behavior that could break CPU/GPU reproducibility.
    for _ in range(_REFINEMENT_STEPS):
        width = upper - lower
        first = lower + _GOLDEN_LOWER * width
        second = lower + _GOLDEN_UPPER * width
        first_sse = _local_sse(fit_time, fit_values, first, xp)
        second_sse = _local_sse(fit_time, fit_values, second, xp)
        first_is_better = first_sse <= second_sse
        upper = xp.where(first_is_better, second, upper)
        lower = xp.where(first_is_better, lower, first)

    refined = (lower + upper) * xp.float64(0.5)
    # Preserve a true constrained solution at either filed tau bound instead
    # of returning an interior value merely because refinement is open-ended.
    candidates = xp.stack(
        (
            refined,
            xp.full_like(refined, _LOG_TAU_MIN),
            xp.full_like(refined, _LOG_TAU_MAX),
        ),
        axis=-1,
    )
    candidate_sse = _candidate_sse(fit_time, fit_values, candidates, xp)
    best = xp.argmin(candidate_sse, axis=-1)
    selected_log_tau = xp.take_along_axis(candidates, best[..., None], axis=-1)[..., 0]
    tau = xp.exp(selected_log_tau)
    if _scalar_bool(xp.any(~xp.isfinite(tau))) or _scalar_bool(
        xp.any((tau < TAU_MIN_MS) | (tau > TAU_MAX_MS))
    ):
        raise FloatingPointError("deactivation tail fit produced an invalid tau")
    return tau


def _grid_sse(time: Any, values: Any, log_tau: Any, xp: Any) -> Any:
    """Return residual sums of squares with shape ``(candidate, command, grid)``."""

    basis = xp.exp(-time[:, None] / xp.exp(log_tau)[None, :])
    sample_count = xp.float64(time.size)
    sum_x = xp.sum(basis, axis=0)
    sum_xx = xp.sum(basis * basis, axis=0)
    sum_y = xp.sum(values, axis=-1)
    sum_yy = xp.sum(values * values, axis=-1)
    sum_xy = xp.einsum("ckt,tg->ckg", values, basis)
    determinant = sample_count * sum_xx - sum_x * sum_x
    safe_determinant = _safe_determinant(determinant, sample_count, xp)
    asymptote = (
        sum_y[..., None] * sum_xx - sum_x * sum_xy
    ) / safe_determinant
    amplitude = (
        sample_count * sum_xy - sum_x * sum_y[..., None]
    ) / safe_determinant
    sse = sum_yy[..., None] - asymptote * sum_y[..., None] - amplitude * sum_xy
    return _bounded_sse(sse, determinant, sample_count, xp)


def _local_sse(time: Any, values: Any, log_tau: Any, xp: Any) -> Any:
    """Return residual sums of squares for one tau per candidate and command."""

    basis = xp.exp(-time[None, None, :] / xp.exp(log_tau)[..., None])
    sample_count = xp.float64(time.size)
    sum_x = xp.sum(basis, axis=-1)
    sum_xx = xp.sum(basis * basis, axis=-1)
    sum_y = xp.sum(values, axis=-1)
    sum_yy = xp.sum(values * values, axis=-1)
    sum_xy = xp.sum(values * basis, axis=-1)
    determinant = sample_count * sum_xx - sum_x * sum_x
    safe_determinant = _safe_determinant(determinant, sample_count, xp)
    asymptote = (sum_y * sum_xx - sum_x * sum_xy) / safe_determinant
    amplitude = (sample_count * sum_xy - sum_x * sum_y) / safe_determinant
    sse = sum_yy - asymptote * sum_y - amplitude * sum_xy
    return _bounded_sse(sse, determinant, sample_count, xp)


def _candidate_sse(time: Any, values: Any, log_tau: Any, xp: Any) -> Any:
    """Return residual sums of squares for a short last-axis tau candidate set."""

    basis = xp.exp(-time[None, None, None, :] / xp.exp(log_tau)[..., None])
    sample_count = xp.float64(time.size)
    sum_x = xp.sum(basis, axis=-1)
    sum_xx = xp.sum(basis * basis, axis=-1)
    sum_y = xp.sum(values, axis=-1)[..., None]
    sum_yy = xp.sum(values * values, axis=-1)[..., None]
    sum_xy = xp.sum(values[..., None, :] * basis, axis=-1)
    determinant = sample_count * sum_xx - sum_x * sum_x
    safe_determinant = _safe_determinant(determinant, sample_count, xp)
    asymptote = (sum_y * sum_xx - sum_x * sum_xy) / safe_determinant
    amplitude = (sample_count * sum_xy - sum_x * sum_y) / safe_determinant
    sse = sum_yy - asymptote * sum_y - amplitude * sum_xy
    return _bounded_sse(sse, determinant, sample_count, xp)


def _bounded_sse(sse: Any, determinant: Any, sample_count: Any, xp: Any) -> Any:
    """Reject rank-deficient regressions and suppress round-off-only negatives."""

    # With a two-column [1, exp(-t/tau)] design, a nearly zero determinant
    # means tau has no identifiable amplitude component on this sampled grid.
    valid = determinant > _FLOAT64_EPSILON * sample_count
    return xp.where(valid, xp.maximum(sse, xp.float64(0.0)), xp.float64(np.inf))


def _safe_determinant(determinant: Any, sample_count: Any, xp: Any) -> Any:
    valid = determinant > _FLOAT64_EPSILON * sample_count
    return xp.where(valid, determinant, xp.float64(1.0))


def _authority_seeded_grid_index(grid_sse: Any, grid_log_tau: Any, xp: Any) -> Any:
    """Choose the profile basin reached from the authority's 1 ms tau seed.

    ``curve_fit`` is seeded at tau=1 ms.  Some multi-timescale source tails
    have more than one profile minimum, so choosing the global one can change
    the filed metric.  The nearest discrete local basin to log(1 ms) preserves
    that initialization policy without a host-side candidate/command loop.
    A monotonic profile has no interior basin and falls back to its global
    constrained minimum.
    """

    interior = grid_sse[..., 1:-1]
    local = (interior <= grid_sse[..., :-2]) & (interior <= grid_sse[..., 2:])
    seed_distance = xp.abs(grid_log_tau[1:-1])
    local_distance = xp.where(local, seed_distance, xp.float64(np.inf))
    nearest_local = xp.argmin(local_distance, axis=-1) + 1
    has_local = xp.any(local, axis=-1)
    global_minimum = xp.argmin(grid_sse, axis=-1)
    return xp.where(has_local, nearest_local, global_minimum)


def _elapsed_array(value: Any, xp: Any) -> Any:
    time = xp.asarray(value, dtype=xp.float64)
    if time.ndim != 1 or time.size < 4:
        raise ValueError("elapsed_ms must be one-dimensional with at least four samples")
    if _scalar_bool(xp.any(~xp.isfinite(time))):
        raise ValueError("elapsed_ms must contain only finite values")
    if _scalar_bool(xp.any(time < 0.0)) or _scalar_bool(xp.any(xp.diff(time) <= 0.0)):
        raise ValueError("elapsed_ms must be nonnegative and strictly increasing")
    return time


def _trace_array(value: Any, time_count: int, xp: Any) -> Any:
    traces = xp.asarray(value, dtype=xp.float64)
    if traces.ndim != 3 or traces.shape[0] == 0 or traces.shape[1] == 0:
        raise ValueError("traces must have nonempty shape (candidate, command, time)")
    if traces.shape[2] != time_count:
        raise ValueError("traces time axis must match elapsed_ms")
    if _scalar_bool(xp.any(~xp.isfinite(traces))):
        raise ValueError("traces must contain only finite values")
    return traces


def _validate_identifiable_tails(values: Any, xp: Any) -> None:
    scale = xp.maximum(xp.max(xp.abs(values), axis=-1), xp.float64(1.0))
    span = xp.max(values, axis=-1) - xp.min(values, axis=-1)
    if _scalar_bool(xp.any(span <= _FLOAT64_EPSILON * scale)):
        raise ValueError("traces contain a numerically unidentifiable constant tail")


def _validate_xp(xp: Any) -> None:
    required = (
        "all",
        "abs",
        "any",
        "argmin",
        "asarray",
        "diff",
        "einsum",
        "exp",
        "float64",
        "full_like",
        "isfinite",
        "max",
        "maximum",
        "min",
        "minimum",
        "stack",
        "sum",
        "take_along_axis",
        "where",
    )
    if xp is None or any(not hasattr(xp, name) for name in required):
        raise TypeError("xp must be a NumPy/CuPy-compatible array module")


def _scalar_bool(value: Any) -> bool:
    return bool(value)
