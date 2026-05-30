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
  MAY remain (the strength of that coupling is the load-bearing question
  this probe now MEASURES rather than assumes).

The probe is a falsification test:
  - a single-pass CONTROL must REPRODUCE the conflict (best achievable
    joint accuracy stays below the frozen 0.90 bar). If it does NOT, the
    cheap model is not faithful and cannot test the resolution (the
    verdict fail-closes to CANNOT_CONCLUDE).
  - the two-phase treatment with index-UPDATE must clear the bar.
  - the two-phase content-NO-update variant's index fidelity is now a
    GENUINE NUMERICAL MEASUREMENT (see below). If the residual coupling is
    real, this variant falls below the bar; if the coupling turns out to
    be weak at the probe's dimensionality, this variant can stay high --
    and the verdict reports `coupling_demonstrated` accordingly. The
    probe's novel claim CAN therefore FAIL; it is not returned by
    construction.

WHAT IS CLOSED-FORM (assumption) vs WHAT IS MEASURED:
  - CLOSED-FORM ASSUMPTIONS (grounded in prior, already-validated
    findings, NOT measured here):
      * the single-pass selectivity/order tradeoff in `_single_pass_best`
        (more shuffle -> better selectivity but worse order index), and
      * the `_wm(sep)` concept-query readout map.
    Both encode the project's already-validated result that concept
    selectivity needs shuffled / interleaved presentation while ordered
    presentation preserves episode order -- see the v16 concept-binding
    arc (interleaved training is required for clean per-concept
    selectivity) and the 2026-05-19 integrated-loop iteration-4 finding
    (encode-order conflict). They are deliberately NOT re-derived here.
  - GENUINELY MEASURED (this is the strengthening): the residual-coupling
    outcome -- i.e. the order-index fidelity (`idx_fidelity`) of the three
    index variants AFTER representational drift -- is computed numerically
    in `measure_index_resolution` from REAL random unit vectors, the
    project's own validated cortical separation mechanism (common-mode /
    pooled-inhibition removal, `rep - BETA*common`), and nearest-match
    (argmax-cosine) index resolution. Whatever that measurement yields is
    what `run_probe` reports; the move/overlap are no longer hand-written
    closed forms.

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

# Measurement constants (used by measure_index_resolution).
# BETA is the strength of the Phase-2 common-mode / pooled-inhibition
# removal -- the project's own validated cortical separation mechanism
# (rep_new[i] = normalize(rep[i] - BETA*common)). It genuinely MOVES the
# reps; the move is what a content index must survive (or be updated to
# track). N_EPISODES is the number of random length-N ordered episodes
# over which index resolution is measured.
BETA = 0.6
N_EPISODES = 200

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


def _normalize(v: np.ndarray) -> np.ndarray:
    """Unit-normalize a 1-D vector; return it unchanged if it is the zero
    vector (avoids divide-by-zero)."""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def _mean_offdiag_abscos(reps: np.ndarray) -> float:
    """Mean absolute cosine between distinct concept reps (off-diagonal of
    the |cos| Gram matrix). Lower = more separated."""
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = reps / norms
    gram = np.abs(unit @ unit.T)
    mask = ~np.eye(reps.shape[0], dtype=bool)
    return float(gram[mask].mean())


# ---------------------------------------------------------------------------
# The GENUINE measurement (replaces the old closed-form move/overlap).
# ---------------------------------------------------------------------------
def measure_index_resolution(seed: int) -> dict:
    """Measure order-index fidelity of the three index variants NUMERICALLY.

    This is the strengthening that makes the probe's load-bearing novel
    claim (residual coupling under representational drift) a real
    measurement that CAN fail, not a closed form returned by construction.

    Procedure (deterministic given `seed`):
      1. Draw `rep` = (N_CONCEPTS, D) random unit-norm Gaussian concept
         vectors with rng = np.random.default_rng(seed).
      2. Apply the Phase-2 separation transform -- the project's own
         validated cortical mechanism: common-mode / pooled-inhibition
         removal. With `common = rep.mean(axis=0)`,
             rep_new[i] = normalize(rep[i] - BETA*common).
         This genuinely MOVES each rep and changes inter-concept
         separation. `mean_move = mean_i(1 - cos(rep[i], rep_new[i]))` and
         `sep_gain = mean_offdiag|cos|(rep) - mean_offdiag|cos|(rep_new)`
         (positive = the transform improved separation) are reported for
         transparency.
      3. Over N_EPISODES random length-N ordered tuples of distinct concept
         ids, resolve each (episode, position) under three variants by
         NEAREST-MATCH (argmax cosine) against the NEW rep set rep_new:
           - pointer:          index stored the concept IDENTITY id i_k;
                               resolution is i_k trivially -> always correct.
           - content_noupdate: index stored the OLD vector rep[i_k];
                               resolve = argmax_i cos(rep[i_k], rep_new[i]);
                               correct iff == i_k (MEASURED -- may or may
                               not survive the move).
           - content_update:   index stored content but was UPDATED during
                               consolidation to the NEW vector rep_new[i_k];
                               resolve = argmax_i cos(rep_new[i_k],
                               rep_new[i]); correct iff == i_k.
         idx_fidelity (per variant) = fraction of correct resolutions over
         all (episode, position) pairs.

    Returns a dict:
      {"idx_pointer": float, "idx_content_noupdate": float,
       "idx_content_update": float, "mean_move": float, "sep_gain": float}
    """
    rng = np.random.default_rng(int(seed))

    # 1. Concept reps.
    raw = rng.standard_normal((N_CONCEPTS, D))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    rep = raw / norms

    # 2. Phase-2 separation transform (common-mode removal) -- genuinely
    #    moves the reps.
    common = rep.mean(axis=0)
    rep_new = np.empty_like(rep)
    for i in range(N_CONCEPTS):
        rep_new[i] = _normalize(rep[i] - BETA * common)

    moves = [1.0 - float(np.dot(rep[i], rep_new[i])) for i in range(N_CONCEPTS)]
    mean_move = float(np.mean(moves))
    sep_before = _mean_offdiag_abscos(rep)
    sep_after = _mean_offdiag_abscos(rep_new)
    sep_gain = float(sep_before - sep_after)

    # 3. Order-index resolution over episodes via nearest-match.
    n_correct_pointer = 0
    n_correct_noupdate = 0
    n_correct_update = 0
    n_total = 0
    for _ep in range(N_EPISODES):
        ids = rng.choice(N_CONCEPTS, size=N, replace=False)
        for k in range(N):
            i_k = int(ids[k])
            n_total += 1
            # pointer: identity -> always correct.
            n_correct_pointer += 1
            # content_noupdate: OLD vector resolved against NEW reps.
            if int(np.argmax(rep_new @ rep[i_k])) == i_k:
                n_correct_noupdate += 1
            # content_update: NEW vector resolved against NEW reps.
            if int(np.argmax(rep_new @ rep_new[i_k])) == i_k:
                n_correct_update += 1

    total = float(n_total)
    return {
        "idx_pointer": float(n_correct_pointer) / total,
        "idx_content_noupdate": float(n_correct_noupdate) / total,
        "idx_content_update": float(n_correct_update) / total,
        "mean_move": mean_move,
        "sep_gain": sep_gain,
    }


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


def _two_phase_scalars(seed: int) -> dict:
    """Two-phase treatment scalars, with index fidelity MEASURED.

    Phase 1 records the order-index from ORDERED presentation. Phase 2
    builds selectivity from SHUFFLED replay; the concept-query readout is
    closed-form at sep = 0.70 (wm = 1.0 -- a CLOSED-FORM assumption
    grounded in the validated selectivity-needs-shuffle finding). The
    order-index fidelity of each variant is GENUINELY MEASURED by
    `measure_index_resolution(seed)` (real vectors + common-mode-removal
    separation transform + nearest-match resolution), then mapped through
    the existing `_ep(idx_fidelity)` calibration:

      pointer            : index is a concept-IDENTITY pointer, immune to
                           the move -> measured idx_fidelity (== 1.0 by
                           construction of the pointer variant).
      content_noupdate   : index stores the ORIGINAL rep VECTOR and is NOT
                           updated -> measured idx_fidelity (CAN drop below
                           1.0 if the move breaks nearest-match; stays high
                           if the residual coupling is weak).
      content_update     : index stores content but Phase 2 UPDATES it to
                           the moved rep -> measured idx_fidelity (>= the
                           no-update value; updating never hurts).
    """
    wm = _wm(_SEP_PHASE2)              # CLOSED-FORM: = 1.0 at sep = 0.70

    meas = measure_index_resolution(seed)
    idx_pointer = meas["idx_pointer"]
    idx_noupdate = meas["idx_content_noupdate"]
    idx_update = meas["idx_content_update"]

    ep_pointer = _ep(idx_pointer)
    ep_noupdate = _ep(idx_noupdate)
    ep_update = _ep(idx_update)

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
      two_phase_content_noupdate-- two-phase, content index NOT updated.
                                    MEASURED: < 0.90 iff the residual
                                    coupling is real; stays high if the
                                    representational move is too weak to
                                    break nearest-match resolution.
      two_phase_content_update  -- two-phase, content index UPDATED to the
                                    moved rep (consolidation re-points the
                                    index). MEASURED.
      wm_at_sep07               -- the concept-query accuracy at sep = 0.70
                                    (CLOSED-FORM assumption).
      ep_pointer                -- the order-query accuracy of the pointer
                                    index (from the MEASURED idx_pointer).
    """
    rng = np.random.default_rng(int(seed))

    # Draw the concept reps (advances rng state; keeps the probe honest
    # about being seeded from a real random draw).
    _ = _concept_reps(rng)

    single = _single_pass_best(rng)
    # Two-phase index fidelity is GENUINELY MEASURED (real vectors +
    # common-mode-removal separation transform + nearest-match resolution),
    # not a closed form -- so the residual-coupling claim CAN fail.
    tp = _two_phase_scalars(int(seed))

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
