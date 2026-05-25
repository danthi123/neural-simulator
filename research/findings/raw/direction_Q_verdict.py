"""Direction Q verdict module - pre-registered frozen-threshold scoring.

Computes the Direction Q test verdict ONLY from recorded per-seed
data. Does not re-run, does not retune, does not import any
project module. Standard library + typing only.

Pre-registered thresholds frozen 2026-05-25 per design doc
docs/plans/2026-05-25-direction-Q-dlpfc-scale-up-design.md:
- _Q_RATE_RATIO_MIN = 2.0   (delay-period rate / baseline rate)
- _Q_DELAY_MIN_SEC = 3.0    (minimum sustained-elevation duration, sec)
- _Q_MIN_SEEDS_PASS = 3     (number of seeds that must individually pass)

Verdict tags (distinct categories):
- Q_BISTABILITY_PASS              : >=_Q_MIN_SEEDS_PASS test seeds meet
                                    BOTH ratio AND sustained bars, AND
                                    NO control seed meets the bar.
- Q_BISTABILITY_PARTIAL           : Some test signal present (at least
                                    one seed meets at least one
                                    sub-criterion, but full bar not met
                                    across enough seeds).
- Q_BISTABILITY_NEGATIVE          : 0 test seeds meet EITHER bar.
- Q_VOID_CONTROL_ALSO_PASSED      : Any control seed meets the full bar
                                    (persistence not NMDA-driven; result
                                    is invalid as a Wang-2002 claim).
- Q_VOID_MALFORMED_INPUT          : Instrument-validity failure (None,
                                    empty, non-numeric, NaN, Inf, or
                                    not a (ratio, sustained_sec) tuple).

Discipline (load-bearing):
- Standard library + typing imports only.
- Thresholds are MODULE-LEVEL CONSTANTS set ONCE; no runtime override path.
- Instrument-validity check FIRST (fail-closed). Malformed input never
  raises; it returns Q_VOID_MALFORMED_INPUT.
- VOID branches are strictly distinct tags from PASS/NEGATIVE.
- Does NOT import or modify any existing verdict module.
- A control seed passing the same bar voids the result; this is
  pre-registered, not a post-hoc judgment call.
"""
from __future__ import annotations
from typing import Any, List, Optional, Tuple
import math


# -------------------------------------------------------------------
# Pre-registered frozen thresholds (set ONCE at module load time).
# These values come from the design doc and the implementation plan
# Task 3 specification. They are NOT modifiable at runtime: there is
# no setter, no env-var override, no config injection. A future PR
# that silently lowers any of these is caught by:
#   tests/test_direction_Q_verdict.py::test_thresholds_frozen_at_design_values
#   tests/test_direction_Q_verdict.py::test_threshold_tamper_detection
#   tests/test_direction_Q_grounding.py::test_direction_Q_verdict_thresholds_frozen
# -------------------------------------------------------------------
_Q_RATE_RATIO_MIN: float = 2.0
_Q_DELAY_MIN_SEC: float = 3.0
_Q_MIN_SEEDS_PASS: int = 3


# -------------------------------------------------------------------
# Verdict tag constants (for external callers that want to compare
# without typos)
# -------------------------------------------------------------------
Q_BISTABILITY_PASS: str = "Q_BISTABILITY_PASS"
Q_BISTABILITY_PARTIAL: str = "Q_BISTABILITY_PARTIAL"
Q_BISTABILITY_NEGATIVE: str = "Q_BISTABILITY_NEGATIVE"
Q_VOID_CONTROL_ALSO_PASSED: str = "Q_VOID_CONTROL_ALSO_PASSED"
Q_VOID_MALFORMED_INPUT: str = "Q_VOID_MALFORMED_INPUT"


# -------------------------------------------------------------------
# Instrument-validity helpers (fail-closed: malformed -> False)
# -------------------------------------------------------------------
def _is_valid_seed_entry(seed_entry: Any) -> bool:
    """Returns True iff seed_entry is a 2-element sequence-like
    (rate_ratio, sustained_sec) with finite, non-NaN, non-Inf numeric
    values. Any other shape, type, or numeric pathology -> False.

    A defensive try/except around the entire body ensures that any
    unusual object passed in (custom __len__ raising, custom __iter__
    misbehaving, etc.) is treated as malformed rather than crashing
    the verdict computation.
    """
    if seed_entry is None:
        return False
    try:
        # Reject strings (they have len but are not tuple-like data here)
        if isinstance(seed_entry, (str, bytes)):
            return False
        # Require length-2 sequence
        if len(seed_entry) != 2:
            return False
        ratio_raw, sec_raw = seed_entry[0], seed_entry[1]
        # Bool is a subclass of int in Python; allow it (it casts to 0/1)
        # but reject other non-numeric types explicitly to avoid silent
        # coercions (e.g. "2.5" string -> 2.5 float would mask a bug).
        if not isinstance(ratio_raw, (int, float)):
            return False
        if not isinstance(sec_raw, (int, float)):
            return False
        ratio = float(ratio_raw)
        sec = float(sec_raw)
        if math.isnan(ratio) or math.isnan(sec):
            return False
        if math.isinf(ratio) or math.isinf(sec):
            return False
        return True
    except (TypeError, ValueError, AttributeError, IndexError):
        return False


def _seed_meets_full_bar(seed_entry: Tuple[float, float]) -> bool:
    """A seed meets the FULL Direction Q bar iff:
        rate_ratio >= _Q_RATE_RATIO_MIN  AND
        sustained_sec >= _Q_DELAY_MIN_SEC
    Caller MUST have already verified _is_valid_seed_entry(seed_entry).
    """
    ratio, sec = float(seed_entry[0]), float(seed_entry[1])
    return (ratio >= _Q_RATE_RATIO_MIN) and (sec >= _Q_DELAY_MIN_SEC)


def _seed_meets_any_subcriterion(seed_entry: Tuple[float, float]) -> bool:
    """A seed has 'partial signal' iff at least one of the two
    sub-criteria is met. Used to distinguish PARTIAL from NEGATIVE.
    Caller MUST have already verified _is_valid_seed_entry(seed_entry).
    """
    ratio, sec = float(seed_entry[0]), float(seed_entry[1])
    return (ratio >= _Q_RATE_RATIO_MIN) or (sec >= _Q_DELAY_MIN_SEC)


# -------------------------------------------------------------------
# Public verdict function
# -------------------------------------------------------------------
def compute_verdict(
    per_seed: Optional[List[Tuple[float, float]]],
    control_per_seed: Optional[List[Tuple[float, float]]],
) -> str:
    """Compute the Direction Q verdict tag from recorded per-seed data.

    Args:
        per_seed: list of (rate_ratio, sustained_sec) tuples, one per
                  test seed (NMDA-on). rate_ratio is the delay-period
                  population firing rate divided by the baseline rate;
                  sustained_sec is the maximum continuous duration in
                  seconds for which the delay rate stayed elevated.
        control_per_seed: same shape, but from the AMPA-only control
                  sweep (NMDA disabled). Used to detect non-NMDA-driven
                  persistence (bug or substrate artifact).

    Returns:
        One of the Q_* tag constants defined at module top.

    Discipline:
        - Instrument-validity checked FIRST. Malformed input -> VOID.
        - Control-also-passes detection is pre-registered: any control
          seed meeting the full bar voids the result regardless of
          test-seed counts. This is the canonical guard against a
          substrate bug that drives persistence without NMDA.
        - Function never raises; all paths return a verdict tag.
    """
    # --- Instrument-validity check (fail-closed) ---
    if per_seed is None or control_per_seed is None:
        return Q_VOID_MALFORMED_INPUT

    # Require list/tuple containers (reject dict, set, generator, etc.,
    # to keep the input contract crisp).
    if not isinstance(per_seed, (list, tuple)):
        return Q_VOID_MALFORMED_INPUT
    if not isinstance(control_per_seed, (list, tuple)):
        return Q_VOID_MALFORMED_INPUT

    if len(per_seed) == 0 or len(control_per_seed) == 0:
        return Q_VOID_MALFORMED_INPUT

    for entry in per_seed:
        if not _is_valid_seed_entry(entry):
            return Q_VOID_MALFORMED_INPUT
    for entry in control_per_seed:
        if not _is_valid_seed_entry(entry):
            return Q_VOID_MALFORMED_INPUT

    # --- Control gate (pre-registered: any control pass -> VOID) ---
    # If the AMPA-only control also exhibits the full Wang-2002
    # bistability signature, then whatever is driving persistence
    # in the test sweep is not the NMDA mechanism we set out to test.
    # The result is invalid as a Q claim.
    for entry in control_per_seed:
        if _seed_meets_full_bar(entry):
            return Q_VOID_CONTROL_ALSO_PASSED

    # --- Test-seed scoring ---
    n_full_pass = sum(1 for s in per_seed if _seed_meets_full_bar(s))
    n_any_signal = sum(
        1 for s in per_seed if _seed_meets_any_subcriterion(s)
    )

    if n_full_pass >= _Q_MIN_SEEDS_PASS:
        return Q_BISTABILITY_PASS

    if n_any_signal == 0:
        # 0 test seeds meet EITHER sub-criterion -> truly nothing
        # happened across the test sweep.
        return Q_BISTABILITY_NEGATIVE

    # Some signal present but not the full bar across enough seeds.
    return Q_BISTABILITY_PARTIAL
