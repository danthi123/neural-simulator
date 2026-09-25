"""Pure, no-GPU regression tests for `research/runners/_affect_marker_settle_gpu_timing.py`'s Amendment-3
within-process crossover (`--xo-run` / `--xo-score`), added in the 2026-09-25 fix round (opus review of
`research/settle-a3-amendment3`, `.claude/worktrees/_review_settle_r4_2026-09-25.txt`). Everything exercised
here (`_bad_warmup_reads`, `_xo_proc`/`_xo_quad`, `check_process_xo`, `decide_xo`, `_xo_go_rate`) is pure
Python/numpy/scipy -- no cupy, no torch, no subprocess, sub-second except the calibration test (~4 s).

Each test is built to FAIL in its failing direction (removing the fix it pins reproduces the exact silent
failure mode the paired review comment described), not just to pass by construction:

  - `test_bad_warmup_reads_*`: the b21140758-smoke-adjacent Addendum-item-6 fix -- `_worker_xo` must fail fast
    (not burn the rest of the process's GPU time) when a warm-up turn made != 2 WTA reads (one axis's reader
    bridge left unbuilt, exactly what A3 saw at one OFF warm index). `_worker_xo` itself needs a live GPU/model
    to run end-to-end, so the fail-fast condition was extracted into the pure `_bad_warmup_reads` helper that
    `_worker_xo` now calls -- this file pins THAT helper directly.

  - `test_check_process_xo_flags_zero_qwen_renders_as_undefined` / `..._does_not_false_positive_on_a_real_run`:
    the orchestrator's "M4_render reads exactly 0.0 (se 0, resid 0) on every turn" finding. Root cause (verified
    by reading the smoke's raw per-turn records): every turn of the b21140758 smoke read `abstained: True`, and
    an abstain's reply is the HOST-composed curiosity follow-up (`curiosity_production_organ.followup_question`)
    -- `MoodConditionedRenderer.render_svo` (and therefore `model.generate`, and the `timed_generate` wrapper
    around it) is only ever reached for a GATE-MATCHED fact, which an abstain has none of. The wrapper itself is
    sound; `check_process_xo` had no precondition catching "renderer required but never actually invoked" (the
    `renderer` field it already checks is a static per-response identity label, not evidence of a real call).

  - `test_calibration_go_rate_is_bounded_both_directions`: review LOW 2026-09-25 -- the pre-existing calibration
    check only had a <=10% ceiling (catches an under-conservative/too-narrow CI) and no floor (an OVER-
    conservative CI, which wastes GPU time chasing an unnecessarily loose bound, passed silently). Tightened to
    n_reps=1000 and a [2.5%, 8%] two-sided band around the nominal 5% (one-sided alpha=0.05).
"""
from __future__ import annotations

from research.runners._affect_marker_settle_gpu_timing import (
    BOUND_S,
    _bad_warmup_reads,
    _xo_go_rate,
    _xo_quad,
    check_process_xo,
    decide_xo,
)


# ─────────────────────────────────────────── _bad_warmup_reads (Addendum item 6) ───────────────────────────────
def test_bad_warmup_reads_flags_an_under_read_axis():
    """A warm-up turn that made only 1 WTA read (the arousal bridge never built) must be flagged -- this is
    the EXACT failure mode A3 saw at one OFF warm index (see the runner's AMENDMENT 3 docstring)."""
    warm = [{"arm": "off", "n_wta_reads": 1}, {"arm": "on", "n_wta_reads": 2}]
    bad = _bad_warmup_reads(warm)
    assert len(bad) == 1 and bad[0]["arm"] == "off"


def test_bad_warmup_reads_flags_either_or_both_turns():
    assert len(_bad_warmup_reads([{"arm": "off", "n_wta_reads": 2}, {"arm": "on", "n_wta_reads": 1}])) == 1
    assert len(_bad_warmup_reads([{"arm": "off", "n_wta_reads": 0}, {"arm": "on", "n_wta_reads": 1}])) == 2
    assert len(_bad_warmup_reads([{"arm": "off", "n_wta_reads": 3}, {"arm": "on", "n_wta_reads": 2}])) == 1


def test_bad_warmup_reads_passes_a_fully_committed_pair():
    """The healthy case (both axes committed, both reader bridges built) must NOT be flagged -- a test that
    always failed regardless of input would be worthless."""
    assert _bad_warmup_reads([{"arm": "off", "n_wta_reads": 2}, {"arm": "on", "n_wta_reads": 2}]) == []


def test_bad_warmup_reads_treats_missing_key_as_zero_reads():
    """A malformed/absent `n_wta_reads` (e.g. a turn that raised before recording it) must be treated as an
    under-read, never silently ignored via a `None != 2` -> True vs a KeyError."""
    assert len(_bad_warmup_reads([{"arm": "off"}, {"arm": "on", "n_wta_reads": 2}])) == 1


# ───────────────────────────── check_process_xo: the renderer was never actually exercised ─────────────────────
def test_check_process_xo_flags_zero_qwen_renders_as_undefined():
    """A process whose every turn recorded zero `model.generate()` calls (the b21140758 smoke's shape: every
    turn abstained) must be rejected as invalid, not silently scored as if M4_render's 0.0 were a measured
    near-zero render cost."""
    procs = _xo_quad(render_zero=True)
    checked = [check_process_xo(p, "qwen") for p in procs]
    assert not any(c["valid"] for c in checked), checked
    assert all(any("zero Qwen" in prob for prob in c["problems"]) for c in checked), checked

    status = decide_xo(procs)["status"]
    assert status == "UNDEFINED", "zero real Qwen renders must never resolve GO or NO-GO"


def test_check_process_xo_does_not_false_positive_on_a_real_run():
    """The SAME quad with genuine (synthetic) `model.generate()` calls recorded must NOT be flagged by the new
    check -- otherwise it would blanket-reject every valid run, not just the degenerate one."""
    procs = _xo_quad()   # render_zero defaults to False: every turn carries n_gen_calls=1
    checked = [check_process_xo(p, "qwen") for p in procs]
    assert all(c["valid"] for c in checked), checked
    assert decide_xo(procs)["status"] == "GO"


def test_check_process_xo_zero_renders_mutation_verify():
    """MUTATION-VERIFY: reproduce, by hand, what removing the new precondition does -- `check_process_xo` with
    the fix must find a problem citing 'Qwen' for a zero-render process; a build with the check deleted (as it
    was before this fix round) reports NO problems for the exact same input, which is what let the zero-render
    smoke pass every OTHER precondition silently."""
    procs = _xo_quad(render_zero=True)
    problems = check_process_xo(procs[0], "qwen")["problems"]
    assert any("qwen" in p.lower() and ("zero" in p.lower() or "never" in p.lower()) for p in problems), (
        "expected a problem naming the missing Qwen render; got %r" % problems)


# ───────────────────────────────────── calibration: two-sided floor + ceiling ────────────────────────────────
def test_calibration_go_rate_is_bounded_both_directions():
    """review LOW 2026-09-25: a true whole-turn cost sitting EXACTLY at the 0.3 s bound must read GO at
    close to its nominal one-sided alpha=0.05 rate -- not routinely (a too-narrow CI, the original ceiling-only
    check's target) and not vanishingly rarely either (a too-loose CI silently wastes GPU time chasing a bound
    looser than the design calls for, and was NOT caught by a ceiling-only check)."""
    rate = _xo_go_rate(n_reps=1000, runs=8, true_cost=BOUND_S, noise=0.3)
    assert 0.025 <= rate <= 0.08, "GO rate %.3f outside the calibrated [0.025, 0.08] band" % rate
