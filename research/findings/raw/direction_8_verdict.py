"""Direction 8 verdict module - pre-registered frozen-threshold scoring.

Computes the Direction 8 cross-bridge-on-bio_brain_regions test verdict
(2x-vocab extension of Direction 6; V=128 per bridge x 5 bridges = 320
cross-bridge concepts; matches the Direction M G.20 sparse production
deliverable vocabulary on a biology-faithful substrate) ONLY from
recorded per-seed cross-bridge parallel-matching accuracy. Does not
re-run, does not retune, does not import any project module. Standard
library + typing only.

Pre-registered thresholds frozen per the binding D7 discipline:
- _DIRECTION_8_OB_MIN = 0.80     (order-bearing multi-seed-mean min;
                                   same as pillars n=93+ + Directions Q,
                                   3, 4, 6 FROZEN 0.80 multi-seed strict bar)
- _DIRECTION_8_OI_MIN = 0.80     (order-invariant multi-seed-mean min)
- _DIRECTION_8_LOADS  = [2, 3, 5] (pre-registered load ladder; same as
                                   the validated OPTION 3 V=16 probe,
                                   Direction 3 V=32 probe, Direction 4
                                   V=80 probe, Direction 6 V=160 probe)
- _DIRECTION_8_MIN_SEEDS = 3     (minimum number of seeds for verdict)

Verdict tags (distinct categories):
- DIRECTION_8_PASS           : multi-seed-mean OB AND OI both clear the
                                0.80 bar at EVERY load in the pre-
                                registered ladder. Cross-bridge
                                parallel-matching mode-unification
                                extends to 5x128=640 cross-bridge concepts
                                on bio_brain_regions substrates -- the
                                Direction M G.20 sparse production
                                deliverable vocabulary on a biology-
                                faithful substrate. Unifies the user-
                                facing chat capability tier with the
                                biology-faithful architecture (pillars
                                n=98/n=105/n=108/n=109).
- DIRECTION_8_PARTIAL        : at least one cell above bar but not all
                                cells; characterise precisely which
                                loads / readouts pass; biology-
                                translatable comparison to D6 V=160 PASS.
- DIRECTION_8_NEGATIVE       : NO load-cell on EITHER readout clears
                                the bar; D6's dedicated-pool composition
                                does not scale from V=32 -> V=128 per
                                bridge (a substantial finding;
                                characterises the substrate's
                                vocab-per-bridge ceiling).
- DIRECTION_8_VOID_MALFORMED : instrument-validity failure (None, empty,
                                non-numeric, NaN, Inf, wrong shape, etc.).

Discipline (load-bearing):
- Standard library + typing imports only.
- Thresholds are MODULE-LEVEL CONSTANTS set ONCE; no runtime override path.
- Instrument-validity check FIRST (fail-closed). Malformed input never
  raises; it returns DIRECTION_8_VOID_MALFORMED.
- VOID branch is strictly distinct from PASS / PARTIAL / NEGATIVE.
- Does NOT import or modify any existing verdict module.
- Returns tag in {DIRECTION_8_PASS, DIRECTION_8_PARTIAL,
  DIRECTION_8_NEGATIVE, DIRECTION_8_VOID_MALFORMED}.

Input shape (recorded by the cross-bridge probe runner, Task 4):
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
# These values come from the D7 discipline binding. They are NOT
# modifiable at runtime: there is no setter, no env-var override, no
# config injection. A future PR that silently lowers any of these is
# caught by:
#   tests/test_direction_8_grounding.py::test_direction_8_verdict_thresholds_frozen
# -------------------------------------------------------------------
_DIRECTION_8_OB_MIN: float = 0.80
_DIRECTION_8_OI_MIN: float = 0.80
_DIRECTION_8_LOADS: Tuple[int, ...] = (2, 3, 5)
_DIRECTION_8_MIN_SEEDS: int = 3


# -------------------------------------------------------------------
# Verdict tag constants (for external callers that want to compare
# without typos).
# -------------------------------------------------------------------
DIRECTION_8_PASS: str = "DIRECTION_8_PASS"
DIRECTION_8_PARTIAL: str = "DIRECTION_8_PARTIAL"
DIRECTION_8_NEGATIVE: str = "DIRECTION_8_NEGATIVE"
DIRECTION_8_VOID_MALFORMED: str = "DIRECTION_8_VOID_MALFORMED"


# -------------------------------------------------------------------
# Instrument-validity helpers (fail-closed: malformed -> False).
# -------------------------------------------------------------------
def _is_finite_number(x: Any) -> bool:
    """True iff x is a finite numeric (int/float, NaN/Inf rejected).

    Defensive: bool is a subclass of int in Python; we want to REJECT
    bool here because OB/OI values are accuracies in [0.0, 1.0] and a
    True/False slipping into the slot indicates malformed JSON, not
    a real 0/1 measurement.
    """
    if x is None:
        return False
    if isinstance(x, bool):
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
    to a finite numeric."""
    if entry is None or not isinstance(entry, dict):
        return False
    if "OB" not in entry or "OI" not in entry:
        return False
    return _is_finite_number(entry["OB"]) and _is_finite_number(entry["OI"])


def _is_valid_seed_entry(seed_entry: Any,
                          loads: Tuple[int, ...]) -> bool:
    """True iff seed_entry is a dict containing 'L=<l>' keys for every
    l in loads, each valid as a load entry.

    Defensive try/except around the entire body so that any unusual
    object (with custom __getitem__ etc.) is treated as malformed
    rather than crashing the verdict computation.
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
    """Compute the Direction 8 verdict tag from recorded per-seed
    cross-bridge accuracy.

    Args:
        per_seed: list of seed-level result dicts. Each dict must have
                  shape {"L=2": {"OB": float, "OI": float},
                         "L=3": {"OB": float, "OI": float},
                         "L=5": {"OB": float, "OI": float}}
                  with finite floats in [0.0, 1.0].

    Returns:
        One of the DIRECTION_8_* tag constants defined at module top.

    Discipline:
        - Instrument-validity checked FIRST. Malformed input -> VOID.
        - Below-MIN-SEEDS input is treated as malformed (the bar is
          multi-seed-mean; 1 or 2 seeds is not multi-seed).
        - Function never raises; all paths return a verdict tag.
    """
    # --- Instrument-validity check (fail-closed) ---
    if per_seed is None:
        return DIRECTION_8_VOID_MALFORMED

    if not isinstance(per_seed, (list, tuple)):
        return DIRECTION_8_VOID_MALFORMED

    if len(per_seed) < _DIRECTION_8_MIN_SEEDS:
        # Multi-seed verdict requires >= _DIRECTION_8_MIN_SEEDS seeds.
        # Fewer is treated as instrument-validity failure (too few
        # seeds is not a NEGATIVE; it is an under-determined
        # instrument).
        return DIRECTION_8_VOID_MALFORMED

    for entry in per_seed:
        if not _is_valid_seed_entry(entry, _DIRECTION_8_LOADS):
            return DIRECTION_8_VOID_MALFORMED

    # --- Compute multi-seed means ---
    means = _compute_multi_seed_means(
        list(per_seed), _DIRECTION_8_LOADS,
    )

    # --- Cell-level pass/fail ---
    # A "cell" is one (load, readout) combination. There are
    # len(loads) * 2 cells (OB + OI per load = 6 cells at loads
    # {2, 3, 5}).
    n_cells_total = 0
    n_cells_passed = 0
    all_ob_pass = True
    all_oi_pass = True
    for load in _DIRECTION_8_LOADS:
        ob_mean = means[load]["OB"]
        oi_mean = means[load]["OI"]
        n_cells_total += 2
        if ob_mean >= _DIRECTION_8_OB_MIN:
            n_cells_passed += 1
        else:
            all_ob_pass = False
        if oi_mean >= _DIRECTION_8_OI_MIN:
            n_cells_passed += 1
        else:
            all_oi_pass = False

    # --- Verdict ---
    if all_ob_pass and all_oi_pass:
        return DIRECTION_8_PASS

    if n_cells_passed == 0:
        # NO cell on EITHER readout clears the bar; substrate doesn't
        # extend to cross-bridge composition in any measurable form.
        return DIRECTION_8_NEGATIVE

    # Some cells pass, but not all. Honest per-load breakdown captured
    # by the runner; verdict is PARTIAL.
    return DIRECTION_8_PARTIAL
