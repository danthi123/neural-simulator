"""Direction 3 V=32 verdict module - pre-registered frozen-threshold scoring.

Computes the Direction 3 V=32 test verdict ONLY from recorded per-seed
parallel-matching mode-unification accuracy. Does not re-run, does not
retune, does not import any project module. Standard library + typing
only.

Pre-registered thresholds frozen 2026-05-25 per design doc
docs/plans/2026-05-25-direction-3-vocab-scaling-bio_brain_regions-design.md:
- _DIRECTION_3_V32_OB_MIN = 0.80   (order-bearing multi-seed-mean min;
                                       same as pillars n=93+ + Direction Q
                                       FROZEN 0.80 multi-seed strict bar)
- _DIRECTION_3_V32_OI_MIN = 0.80   (order-invariant multi-seed-mean min)
- _DIRECTION_3_V32_LOADS = [2, 3, 5] (pre-registered load ladder; same as
                                          the validated OPTION 3 V=16 probe)
- _DIRECTION_3_V32_MIN_SEEDS = 3   (minimum number of seeds for verdict)

Verdict tags (distinct categories):
- DIRECTION_3_V32_PASS            : multi-seed-mean OB AND OI both clear
                                    the 0.80 bar at EVERY load in the
                                    pre-registered ladder. The validated
                                    parallel-matching mode-unification
                                    extends to V=32 on the bio_brain_-
                                    regions substrate.
- DIRECTION_3_V32_PARTIAL         : at least one cell above bar but not
                                    all cells; characterize precisely
                                    which loads / readouts pass.
- DIRECTION_3_V32_NEGATIVE        : NO load-cell on EITHER readout
                                    clears the bar; the substrate's
                                    concept-pool architecture does NOT
                                    cleanly extend to V=32 (more
                                    substantial finding).
- DIRECTION_3_V32_VOID_MALFORMED  : instrument-validity failure (None,
                                    empty, non-numeric, NaN, Inf, wrong
                                    shape, etc.).

Discipline (load-bearing):
- Standard library + typing imports only.
- Thresholds are MODULE-LEVEL CONSTANTS set ONCE; no runtime override path.
- Instrument-validity check FIRST (fail-closed). Malformed input never
  raises; it returns DIRECTION_3_V32_VOID_MALFORMED.
- VOID branch is strictly distinct from PASS / PARTIAL / NEGATIVE.
- Does NOT import or modify any existing verdict module.
- Returns tag in {DIRECTION_3_V32_PASS, DIRECTION_3_V32_PARTIAL,
  DIRECTION_3_V32_NEGATIVE, DIRECTION_3_V32_VOID_MALFORMED}.

Input shape (recorded by the multi-seed runner):
    per_seed: list[dict] - one entry per seed, each with shape:
        {
          "L=2":  {"OB": float, "OI": float},
          "L=3":  {"OB": float, "OI": float},
          "L=5":  {"OB": float, "OI": float},
        }
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import math


# -------------------------------------------------------------------
# Pre-registered frozen thresholds (set ONCE at module load time).
# These values come from the design doc. They are NOT modifiable at
# runtime: there is no setter, no env-var override, no config injection.
# A future PR that silently lowers any of these is caught by:
#   tests/test_direction_3_verdict.py::test_thresholds_frozen_at_design_values
#   tests/test_direction_3_verdict.py::test_threshold_tamper_detection
#   tests/test_direction_3_grounding.py::test_direction_3_verdict_thresholds_frozen
# -------------------------------------------------------------------
_DIRECTION_3_V32_OB_MIN: float = 0.80
_DIRECTION_3_V32_OI_MIN: float = 0.80
_DIRECTION_3_V32_LOADS: Tuple[int, ...] = (2, 3, 5)
_DIRECTION_3_V32_MIN_SEEDS: int = 3


# -------------------------------------------------------------------
# Verdict tag constants (for external callers that want to compare
# without typos).
# -------------------------------------------------------------------
DIRECTION_3_V32_PASS: str = "DIRECTION_3_V32_PASS"
DIRECTION_3_V32_PARTIAL: str = "DIRECTION_3_V32_PARTIAL"
DIRECTION_3_V32_NEGATIVE: str = "DIRECTION_3_V32_NEGATIVE"
DIRECTION_3_V32_VOID_MALFORMED: str = "DIRECTION_3_V32_VOID_MALFORMED"


# -------------------------------------------------------------------
# Instrument-validity helpers (fail-closed: malformed -> False).
# -------------------------------------------------------------------
def _is_finite_number(x: Any) -> bool:
    """True iff x is a finite numeric (int/float, not bool-only-trap
    rejected because bool is a subclass of int and we DO want to accept
    0/1 as numeric)."""
    if x is None:
        return False
    if not isinstance(x, (int, float)):
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    if math.isnan(xf) or math.isinf(xf):
        return False
    return True


def _is_valid_load_entry(entry: Any) -> bool:
    """True iff entry is a dict with 'OB' and 'OI' keys, each mapping
    to a finite number."""
    if entry is None or not isinstance(entry, dict):
        return False
    if "OB" not in entry or "OI" not in entry:
        return False
    return _is_finite_number(entry["OB"]) and _is_finite_number(entry["OI"])


def _is_valid_seed_entry(seed_entry: Any,
                          loads: Tuple[int, ...]) -> bool:
    """True iff seed_entry is a dict containing 'L=<l>' keys for every
    l in loads, each valid as a load entry. Any other shape/type/numeric
    pathology -> False.

    Defensive try/except around the entire body so that any unusual
    object (with custom __getitem__ etc.) is treated as malformed rather
    than crashing the verdict computation.
    """
    if seed_entry is None:
        return False
    if isinstance(seed_entry, (str, bytes)):
        return False
    if not isinstance(seed_entry, dict):
        return False
    try:
        for load in loads:
            key = "L=" + str(load)
            if key not in seed_entry:
                return False
            if not _is_valid_load_entry(seed_entry[key]):
                return False
        return True
    except (TypeError, KeyError, AttributeError):
        return False


def _compute_multi_seed_means(
    per_seed: List[Dict[str, Dict[str, float]]],
    loads: Tuple[int, ...],
) -> Dict[int, Dict[str, float]]:
    """Compute multi-seed-mean OB and OI per load. Caller MUST have
    already verified each seed_entry via _is_valid_seed_entry.

    Returns: {load: {"OB": mean_ob, "OI": mean_oi}, ...}
    """
    means: Dict[int, Dict[str, float]] = {}
    n_seeds = len(per_seed)
    for load in loads:
        key = "L=" + str(load)
        ob_sum = 0.0
        oi_sum = 0.0
        for s in per_seed:
            ob_sum += float(s[key]["OB"])
            oi_sum += float(s[key]["OI"])
        means[load] = {
            "OB": ob_sum / n_seeds,
            "OI": oi_sum / n_seeds,
        }
    return means


# -------------------------------------------------------------------
# Public verdict function.
# -------------------------------------------------------------------
def compute_verdict(
    per_seed: Optional[List[Dict[str, Dict[str, float]]]],
) -> str:
    """Compute the Direction 3 V=32 verdict tag from recorded per-seed
    accuracy.

    Args:
        per_seed: list of seed-level result dicts. Each dict must have
                  shape {"L=2": {"OB": float, "OI": float},
                         "L=3": {"OB": float, "OI": float},
                         "L=5": {"OB": float, "OI": float}}
                  with finite floats in [0.0, 1.0].

    Returns:
        One of the DIRECTION_3_V32_* tag constants defined at module top.

    Discipline:
        - Instrument-validity checked FIRST. Malformed input -> VOID.
        - Below-MIN-SEEDS input is treated as malformed (the bar is
          multi-seed-mean; 1 or 2 seeds is not multi-seed).
        - Function never raises; all paths return a verdict tag.
    """
    # --- Instrument-validity check (fail-closed) ---
    if per_seed is None:
        return DIRECTION_3_V32_VOID_MALFORMED

    if not isinstance(per_seed, (list, tuple)):
        return DIRECTION_3_V32_VOID_MALFORMED

    if len(per_seed) < _DIRECTION_3_V32_MIN_SEEDS:
        # Multi-seed verdict requires >= _DIRECTION_3_V32_MIN_SEEDS seeds.
        # Fewer is treated as instrument-validity failure (too few seeds
        # is not a NEGATIVE; it is an under-determined instrument).
        return DIRECTION_3_V32_VOID_MALFORMED

    for entry in per_seed:
        if not _is_valid_seed_entry(entry, _DIRECTION_3_V32_LOADS):
            return DIRECTION_3_V32_VOID_MALFORMED

    # --- Compute multi-seed means ---
    means = _compute_multi_seed_means(
        list(per_seed), _DIRECTION_3_V32_LOADS,
    )

    # --- Cell-level pass/fail ---
    # A "cell" is one (load, readout) combination. There are
    # len(loads) * 2 cells (OB + OI per load = 6 cells at loads {2,3,5}).
    n_cells_total = 0
    n_cells_passed = 0
    all_ob_pass = True
    all_oi_pass = True
    for load in _DIRECTION_3_V32_LOADS:
        ob_mean = means[load]["OB"]
        oi_mean = means[load]["OI"]
        n_cells_total += 2
        if ob_mean >= _DIRECTION_3_V32_OB_MIN:
            n_cells_passed += 1
        else:
            all_ob_pass = False
        if oi_mean >= _DIRECTION_3_V32_OI_MIN:
            n_cells_passed += 1
        else:
            all_oi_pass = False

    # --- Verdict ---
    if all_ob_pass and all_oi_pass:
        return DIRECTION_3_V32_PASS

    if n_cells_passed == 0:
        # NO cell on EITHER readout clears the bar; substrate doesn't
        # extend to V=32 in any measurable form.
        return DIRECTION_3_V32_NEGATIVE

    # Some cells pass, but not all. Honest per-load breakdown captured
    # by the runner; verdict is PARTIAL.
    return DIRECTION_3_V32_PARTIAL
