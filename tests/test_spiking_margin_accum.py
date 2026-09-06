"""Unit tests for `RFPhasorComposer._spiking_margin_accum` (2026-09-05, wall-reframe follow-on to the rank-9
recall-margin PARTIAL, `research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`).

The PARTIAL's own residual: a FIXED-ENDPOINT snapshot margin (`_spiking_margin`) agrees with the host confidence
formula only 50% of the time in the ambiguous middle band, and a drive/window-size sweep did not resolve it. The
accumulation-to-bound read (`_spiking_margin_accum`) reads the SAME Izhikevich winner-vs-runner-up competition's
RUNNING margin at every step of the SAME deliberation window instead of only at the end, deriving:
  - `mean_trajectory_margin` -- the time-INTEGRATED evidence (a competition that separates early and stays
    separated spends the whole window near its final value; one that only just arrives by the last step spends
    most of the window near 0, even with a similar final number).
  - `steps_to_bound` / `bounded` -- the TIME-TO-BOUND read (a fixed criterion, `_margin_accum_bound`); fast
    crossing = the confident/high-drift signature, late-or-never crossing = the genuinely-ambiguous signature.

These tests pin the MECHANICS directly (hand-picked synthetic candidate-score vectors), independent of whatever
the composer-level CPU smoke run's own numbers turn out to be -- ADDITIVE, unwired: `_spiking_margin` itself is
untouched, so this file exercises only the new method. SIM_BACKEND=numpy (no GPU needed; this Izhikevich bank is
tiny, V<=5 neurons).

`steps_to_bound` is deliberately defined on the RAW spike-count DIFFERENCE, not the normalized ratio
`_spiking_margin` itself reports -- an instrument fix found WHILE building this (see
`_margin_accum_count_bound`'s docstring in rf_phasor_composer.py): the normalized ratio's first non-degenerate
value is trivially 1.0 the instant the very first spike lands and nothing else has fired yet, so a naive
first-crossing test on the RATIO spuriously "bounds" a nearly-tied (ambiguous) competition just as fast as a
genuinely decisive one, purely from firing-order noise -- measured directly below (`test_ambiguous...`) before
the fix. The raw count difference is non-decreasing once a lead opens and is not renormalized by a small, noisy
denominator, matching the accumulator a real race/DDM model puts a threshold on (Usher & McClelland 2001).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest

from research.runners.rf_phasor_composer import RFPhasorComposer

# a clearly-separable ("confident") candidate distribution: one dominant score, others near zero.
CONFIDENT_SCORES = np.array([10.0, 0.5, 0.4, 0.3])
# a nearly-tied ("ambiguous") candidate distribution: nothing dominates.
AMBIGUOUS_SCORES = np.array([1.01, 1.0, 1.0, 0.99])


@pytest.mark.parametrize("seed", [42, 43])
def test_final_margin_matches_spiking_margin_exactly(seed):
    """`_spiking_margin_accum`'s `final_margin` must equal `_spiking_margin`'s return on the SAME scores/lesion --
    both read off the identical cumulative firing counts at the identical last step of the identical
    deterministic simulation, so this is a bit-exact consistency check tying the new read to the already-
    validated one (2026-09-05 PARTIAL, Pearson r=0.959/Spearman rho=0.954 vs. the host formula)."""
    comp = RFPhasorComposer(seed=seed, D=64)
    for lesion in (False, True):
        expected = comp._spiking_margin(CONFIDENT_SCORES, lesion=lesion)
        got = comp._spiking_margin_accum(CONFIDENT_SCORES, lesion=lesion)["final_margin"]
        assert got == pytest.approx(expected, abs=1e-12)


def test_confident_high_drift_bounds_fast_and_sustains():
    """A clearly-separable competition (one dominant candidate) must reach the `_margin_accum_count_bound`
    criterion EARLY (small `steps_to_bound`) and sustain a high margin for most of the window
    (`mean_trajectory_margin` close to `final_margin`, not pulled far below it) -- the "fast high-drift bound"
    signature the mission frame names as the confident-correct case."""
    comp = RFPhasorComposer(seed=42, D=64)
    r = comp._spiking_margin_accum(CONFIDENT_SCORES)
    assert r["bounded"] is True
    assert r["steps_to_bound"] is not None
    # reaches the criterion well before the window ends (120 steps) -- a real "fast" bound crossing, not a
    # last-instant one indistinguishable from "never".
    assert r["steps_to_bound"] < comp._cleanup_window // 2
    assert r["final_margin"] > 0.9
    # sustained: the trajectory mean is not collapsed far below the endpoint (a "just barely arrived" competition
    # would have a mean well under half its final value; a genuinely fast, sustained one should not).
    assert r["mean_trajectory_margin"] >= 0.5 * r["final_margin"]


def test_ambiguous_low_drift_slow_or_never_bounds():
    """A nearly-tied competition (no dominant candidate, all driven within ~2% of each other) must NEVER reach
    the raw count-difference bound within the window -- the "slow / no bound" signature the mission frame names
    as the genuinely-uncertain case -- and its accumulated evidence must be materially lower than the confident
    case's. (Measured directly before the raw-count-difference fix: a NORMALIZED-ratio first-crossing test
    spuriously bounded this exact competition at step 12, indistinguishable from the confident case's own
    ratio-based crossing -- see the module docstring.)"""
    comp = RFPhasorComposer(seed=42, D=64)
    ambiguous = comp._spiking_margin_accum(AMBIGUOUS_SCORES)
    confident = comp._spiking_margin_accum(CONFIDENT_SCORES)
    assert confident["bounded"] is True
    assert ambiguous["bounded"] is False
    assert ambiguous["steps_to_bound"] is None
    assert ambiguous["mean_trajectory_margin"] < confident["mean_trajectory_margin"]


def test_lesion_collapses_a_confident_competition_to_the_ambiguous_signature():
    """The load-bearing lesion (mirrors the PARTIAL finding's own lesion test on `_spiking_margin`): removing the
    recall circuit's OWN discrimination (uniform drive, no differential) on an otherwise clearly-separable
    competition must collapse BOTH the accumulated evidence and the bound-crossing (never reached within the
    window) toward the ambiguous signature -- confirming the accumulation read genuinely tracks the
    competition's discrimination, not something riding along independent of it."""
    comp = RFPhasorComposer(seed=42, D=64)
    intact = comp._spiking_margin_accum(CONFIDENT_SCORES, lesion=False)
    lesioned = comp._spiking_margin_accum(CONFIDENT_SCORES, lesion=True)
    assert intact["bounded"] is True
    assert lesioned["bounded"] is False
    assert lesioned["mean_trajectory_margin"] < intact["mean_trajectory_margin"]


@pytest.mark.parametrize("degenerate", [np.array([0.0]), np.array([]), np.array([0.0, 0.0, 0.0])])
def test_degenerate_inputs_match_spiking_margins_zero_verdict(degenerate):
    """V<2 or an all-zero (uninformative) score vector must read the SAME uninformative verdict
    `_spiking_margin` already gives (0.0) -- `final_margin` 0.0, no bound ever reached."""
    comp = RFPhasorComposer(seed=42, D=64)
    expected = comp._spiking_margin(degenerate)
    r = comp._spiking_margin_accum(degenerate)
    assert r["final_margin"] == pytest.approx(expected, abs=1e-12)
    assert r["bounded"] is False
    assert r["steps_to_bound"] is None


def test_default_off_by_construction_nothing_calls_the_new_method():
    """`_spiking_margin_accum` is unwired by construction: the ONLY way it can affect a query is if something
    calls it, and nothing in `_cleanup_all_score_stats` / `OneBrainComposer._block_role_scores` / any
    production/preference-chain code does (grepped, not merely asserted) -- a normal query's answer and every
    existing trace field are untouched regardless of this method's existence."""
    import inspect
    import research.runners.rf_phasor_composer as rfp
    src = inspect.getsource(rfp.RFPhasorComposer._cleanup_all_score_stats)
    assert "_spiking_margin_accum" not in src
