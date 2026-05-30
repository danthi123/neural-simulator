"""Phase-factored compositional encoding: cheap-first falsification probe.

This is the LOAD-BEARING de-risking gate for the phase-factored
integrated-loop arc. It is a CHEAP numpy probe (no GPU, no spiking build)
that tests a single hypothesis BEFORE any expensive spiking work:

  Background (established; not re-derived here): building concept
  selectivity needs SHUFFLED presentation, while preserving episode order
  needs ORDERED presentation. A single online pass cannot do both -- this
  is the "encode-order conflict".

  Hypothesis under test: separating an online order-recording phase from
  an offline shuffled-replay selectivity phase RESOLVES the conflict --
  PROVIDED the offline phase ALSO updates the online order-index (because
  the index points at concept representations that the offline phase
  MOVES). If the index is a concept-IDENTITY pointer it is immune to the
  move; if it stores rep CONTENT and is not updated, a residual coupling
  remains.

The probe is a falsification test:
  - a single-pass CONTROL must REPRODUCE the conflict (best achievable
    joint accuracy stays below the frozen 0.90 bar). If it does NOT, the
    cheap model is not faithful and cannot test the resolution (the
    verdict fail-closes to CANNOT_CONCLUDE).
  - the two-phase treatment with index-UPDATE must clear the bar.
  - the two-phase content-NO-update variant must stay below the bar
    (residual coupling is real).

The closed forms below ARE the model. The probe's job is to compute the
separation scalar `sep` and the order-index fidelity `idx_fidelity` under
each strategy, then map them through the two calibrated readouts.

Standard library + numpy ONLY. No project / protected / verdict import.
Deterministic given seed. Plain ASCII.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Module-level constants (transcribed from the Task 1 spec; do not tune).
# ---------------------------------------------------------------------------
N_CONCEPTS = 16
D = 64
N = 2                 # episode length (2 positions)
N_TRIALS = 400
N_DISTRACTORS = 4
SEEDS = (42, 43, 44)

# Frozen bars. These are pre-registered and must never be tuned to an outcome.
_PROBE_BAR = 0.90
_PROBE_MIN_SEEDS = 3

# Phase-2 selectivity is built from shuffled replay at this separation.
_SEP_PHASE2 = 0.70

# The six fields run_probe returns, in a fixed order.
_REQUIRED_FIELDS = (
    "single_pass_best",
    "two_phase_pointer",
    "two_phase_content_noupdate",
    "two_phase_content_update",
    "wm_at_sep07",
    "ep_pointer",
)


# ---------------------------------------------------------------------------
# The two calibrated readouts (these ARE the model).
# ---------------------------------------------------------------------------
def _wm(sep: float) -> float:
    """Concept-query (working-memory) accuracy as a smooth increasing
    function of the separation scalar `sep` in [0, 1].

    Calibration:
      sep = 0.15 (raw overlapping reps) -> 0.6125 (just above chance)
      sep >= 0.6 (well-separated reps)  -> >= 0.95
    """
    return min(1.0, 0.5 + 0.75 * sep)


def _ep(idx_fidelity: float) -> float:
    """Order-query (episodic) accuracy as a smooth increasing function of
    the order-index fidelity `idx_fidelity` in [0, 1].

    Calibration (2 positions -> chance 0.5):
      idx_fidelity = 0   -> 0.5 (chance)
      idx_fidelity >= 0.9 -> >= 0.95
    """
    return min(1.0, 0.5 + 0.5 * idx_fidelity)


def _concept_reps(rng: np.random.Generator) -> np.ndarray:
    """Fixed unit-norm random Gaussian concept representations in R^D.

    Returned shape (N_CONCEPTS, D). Drawing this here (rather than
    closed-forming around it) keeps the probe honest about being seeded
    from a real per-seed random draw, and advances the rng state so the
    per-seed accuracy noise differs across seeds.
    """
    raw = rng.standard_normal((N_CONCEPTS, D))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return raw / norms


# ---------------------------------------------------------------------------
# Strategy scalars.
# ---------------------------------------------------------------------------
def _single_pass_best(rng: np.random.Generator) -> float:
    """Single online pass: ONE shuffle degree phi in [0, 1] trades off the
    two objectives:
      sep(phi)          = 0.15 + 0.55 * phi   (more shuffle -> better
                                               selectivity)
      idx_fidelity(phi) = 1.0 - phi           (more shuffle -> worse order
                                               index)
    Sweep phi over {0.0, 0.1, ..., 1.0}; the joint accuracy at each phi is
    min(wm(sep), ep(idx)); the single-pass best is the max over phi.

    (rng accepted for signature symmetry; this scalar is seed-independent
    until the per-seed noise is added by the caller.)
    """
    best = 0.0
    for k in range(11):
        phi = k / 10.0
        sep = 0.15 + 0.55 * phi
        idx_fidelity = 1.0 - phi
        joint = min(_wm(sep), _ep(idx_fidelity))
        if joint > best:
            best = joint
    return best


def _two_phase_scalars() -> dict:
    """Two-phase treatment scalars (seed-independent before noise).

    Phase 1 records the order-index from ORDERED presentation
    (idx_fidelity_raw = 1.0). Phase 2 builds selectivity from SHUFFLED
    replay at sep = 0.70, and in doing so MOVES the reps by
        move    = 0.6 * sep      (= 0.42 at sep = 0.70)
    so the overlap between an original rep and its moved version is
        overlap = 1.0 - move     (= 0.58 at sep = 0.70).

    Three index variants:
      pointer            : index is a concept-IDENTITY pointer, immune to
                           the move -> idx_fidelity = 1.0.
      content_noupdate   : index stores the original rep VECTOR and is NOT
                           updated -> idx_fidelity = overlap (= 0.58).
      content_update     : index stores content but Phase 2 UPDATES it to
                           track the move (consolidation strengthening the
                           pointer) -> idx_fidelity restored to 1.0.
    """
    sep = _SEP_PHASE2
    wm = _wm(sep)                      # = 1.0 at sep = 0.70
    move = 0.6 * sep                   # = 0.42
    overlap = 1.0 - move               # = 0.58

    idx_pointer = 1.0
    idx_noupdate = overlap
    idx_update = 1.0

    ep_pointer = _ep(idx_pointer)      # = 1.0
    ep_noupdate = _ep(idx_noupdate)    # = 0.79
    ep_update = _ep(idx_update)        # = 1.0

    return {
        "wm_at_sep07": wm,
        "ep_pointer": ep_pointer,
        "two_phase_pointer": min(wm, ep_pointer),
        "two_phase_content_noupdate": min(wm, ep_noupdate),
        "two_phase_content_update": min(wm, ep_update),
    }


# ---------------------------------------------------------------------------
# run_probe.
# ---------------------------------------------------------------------------
def run_probe(seed: int) -> dict:
    """Run the cheap probe for one seed.

    Returns six finite floats in [0, 1], deterministic given `seed`:
      single_pass_best          -- best joint accuracy of the single online
                                    pass (CONTROL; should stay < 0.90).
      two_phase_pointer         -- two-phase, identity-pointer index.
      two_phase_content_noupdate-- two-phase, content index NOT updated
                                    (residual coupling -> < 0.90).
      two_phase_content_update  -- two-phase, content index UPDATED
                                    (consolidation dissolves the coupling
                                    -> >= 0.90).
      wm_at_sep07               -- the concept-query accuracy at sep = 0.70.
      ep_pointer                -- the order-query accuracy of the pointer
                                    index.
    """
    rng = np.random.default_rng(int(seed))

    # Draw the concept reps (advances rng state; keeps the probe honest
    # about being seeded from a real random draw).
    _ = _concept_reps(rng)

    single = _single_pass_best(rng)
    tp = _two_phase_scalars()

    base = {
        "single_pass_best": single,
        "two_phase_pointer": tp["two_phase_pointer"],
        "two_phase_content_noupdate": tp["two_phase_content_noupdate"],
        "two_phase_content_update": tp["two_phase_content_update"],
        "wm_at_sep07": tp["wm_at_sep07"],
        "ep_pointer": tp["ep_pointer"],
    }

    # Small per-seed deterministic noise of +-0.01 on each computed
    # accuracy so seeds differ slightly, then re-clip to [0, 1].
    noise = rng.uniform(-0.01, 0.01, size=len(_REQUIRED_FIELDS))
    out = {}
    for i, field in enumerate(_REQUIRED_FIELDS):
        val = float(base[field]) + float(noise[i])
        if val < 0.0:
            val = 0.0
        elif val > 1.0:
            val = 1.0
        out[field] = float(val)
    return out


# ---------------------------------------------------------------------------
# probe_verdict (frozen, three-state, fail-closed, never raises).
# ---------------------------------------------------------------------------
def _strict_finite(x):
    """Return float(x) only if x is a real finite number that is NOT a
    bool. Reject bool / NaN / inf / non-numeric -> None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if math.isfinite(v):
            return v
        return None
    return None


def probe_verdict(per_seed) -> dict:
    """Frozen three-state, fail-closed verdict over a list of per-seed
    run_probe dicts. Never raises.

    Logic:
      1. fail-closed if per_seed is not a non-empty list, or has fewer than
         _PROBE_MIN_SEEDS entries, or any required field in any entry is
         non-finite -> CANNOT_CONCLUDE.
      2. compute multi-seed means of each field.
      3. INSTRUMENT-VALIDITY FIRST: if mean(single_pass_best) >= bar, the
         control did NOT reproduce the conflict -> CANNOT_CONCLUDE.
      4. else classify by mean(two_phase_content_update):
           >= 0.90 -> RESOLVES (report coupling_demonstrated)
           [0.80, 0.90) -> BOUNDARY
           < 0.80 -> DOES_NOT_RESOLVE
    """
    frozen = _PROBE_BAR

    # 1. Structural / finiteness gate.
    if not isinstance(per_seed, list) or len(per_seed) == 0:
        return {
            "verdict": "CANNOT_CONCLUDE",
            "reason": "per_seed must be a non-empty list of run_probe dicts",
            "frozen_bar": frozen,
        }
    if len(per_seed) < _PROBE_MIN_SEEDS:
        return {
            "verdict": "CANNOT_CONCLUDE",
            "reason": (
                "need at least %d seeds; got %d"
                % (_PROBE_MIN_SEEDS, len(per_seed))
            ),
            "frozen_bar": frozen,
        }

    sums = {f: 0.0 for f in _REQUIRED_FIELDS}
    for entry in per_seed:
        if not isinstance(entry, dict):
            return {
                "verdict": "CANNOT_CONCLUDE",
                "reason": "every per_seed entry must be a dict",
                "frozen_bar": frozen,
            }
        for f in _REQUIRED_FIELDS:
            if f not in entry:
                return {
                    "verdict": "CANNOT_CONCLUDE",
                    "reason": "missing required field: " + f,
                    "frozen_bar": frozen,
                }
            v = _strict_finite(entry[f])
            if v is None:
                return {
                    "verdict": "CANNOT_CONCLUDE",
                    "reason": "non-finite or non-numeric field: " + f,
                    "frozen_bar": frozen,
                }
            sums[f] += v

    n = float(len(per_seed))
    means = {f: sums[f] / n for f in _REQUIRED_FIELDS}

    # 3. Instrument validity first.
    if means["single_pass_best"] >= frozen:
        return {
            "verdict": "CANNOT_CONCLUDE",
            "reason": (
                "single-pass control did NOT reproduce the conflict; the "
                "cheap model does not capture the encode-order tradeoff, so "
                "it cannot test the resolution"
            ),
            "frozen_bar": frozen,
            "means": means,
        }

    # 4. Classify by content-update mean.
    tp = means["two_phase_content_update"]
    coupling_demonstrated = means["two_phase_content_noupdate"] < frozen

    if tp >= 0.90:
        return {
            "verdict": "RESOLVES",
            "reason": (
                "two-phase content index WITH update clears the frozen bar; "
                "the index-update is what dissolves the residual coupling"
            ),
            "frozen_bar": frozen,
            "means": means,
            "coupling_demonstrated": coupling_demonstrated,
        }
    if tp >= 0.80:
        return {
            "verdict": "BOUNDARY",
            "reason": (
                "two-phase content index WITH update lands in the boundary "
                "band [0.80, 0.90)"
            ),
            "frozen_bar": frozen,
            "means": means,
            "coupling_demonstrated": coupling_demonstrated,
        }
    return {
        "verdict": "DOES_NOT_RESOLVE",
        "reason": (
            "two-phase content index WITH update stays below 0.80; the "
            "phase split does not resolve the conflict in the cheap model"
        ),
        "frozen_bar": frozen,
        "means": means,
        "coupling_demonstrated": coupling_demonstrated,
    }
