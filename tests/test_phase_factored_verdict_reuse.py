"""Task 3: pin that the phase-factored controller scores via the INHERITED
frozen verdict (research/runners/integrated_loop_core.integrated_loop_verdict)
unchanged -- no new verdict logic, bars not shadowed. Feeds synthetic rungs
end-to-end through the inherited verdict to confirm PASS / VOID / FAIL are
all reachable with the controller's rung shape.

Most of this contract is already pinned by Task 2's reuse tests
(test_imports_parked_frozen_verdict_and_defines_no_own_bars,
test_run_rung_returns_exact_rung_shape) and Task 0 part (d) (frozen bars).
This file adds the end-to-end behavioural pin: synthetic rung dicts shaped
like the controller's run_rung output produce the three pre-registered
classifications via the inherited verdict.

stdlib + the inherited verdict only. No new bars. ASCII.
"""
from __future__ import annotations
import os
import importlib.util

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_verdict():
    path = os.path.join(REPO, "research/runners/integrated_loop_core.py")
    spec = importlib.util.spec_from_file_location("integrated_loop_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The 7 frozen lesion names + their pre-registered collapse responsibility.
# SHARED + HELPER_BOTH collapse BOTH readouts; HELPER_WM collapses wm only;
# HELPER_EP collapses ep only. (Mirrors integrated_loop_core lines 62-67.)
_COLLAPSE_BOTH = ("no_binding", "no_shared_clock", "no_hippo_store",
                  "no_neuromod_timing")
_COLLAPSE_WM = ("no_bg_gate",)
_COLLAPSE_EP = ("no_sequencing", "no_cls_replay")


def _lesions(collapse_both=0.20, wm_only=0.20, ep_only=0.20,
             good=0.85, override=None):
    """Build a lesions block where each lesion collapses the readout it is
    responsible for (below the 0.40 ceiling) and leaves the other healthy.
    `override` (name -> {"wm","ep"}) forces a specific malformed/non-collapsing
    entry for the VOID test."""
    out = {}
    for name in _COLLAPSE_BOTH:
        out[name] = {"wm": collapse_both, "ep": collapse_both}
    for name in _COLLAPSE_WM:
        out[name] = {"wm": wm_only, "ep": good}
    for name in _COLLAPSE_EP:
        out[name] = {"wm": good, "ep": ep_only}
    if override:
        out.update(override)
    return out


def _rung(N, full_wm=0.85, full_ep=0.85, v1=0.95, lesion_override=None):
    return {"N": N, "n_seeds": 3,
            "v1": {"wm": v1, "ep": v1},
            "full": {"wm": full_wm, "ep": full_ep},
            "lesions": _lesions(good=min(full_wm, full_ep),
                                override=lesion_override)}


def test_inherited_verdict_pass_reachable():
    """A sound, discriminating, scaling 3-rung input -> SCALE-CONFIDENT-PASS."""
    v = _load_verdict()
    rungs = [_rung(2), _rung(4), _rung(8)]
    out = v.integrated_loop_verdict(rungs)
    assert out["GATE"] == "PASS", out
    assert out["classification"] == "SCALE-CONFIDENT-PASS", out


def test_inherited_verdict_void_when_shared_lesion_does_not_collapse():
    """A shared-system lesion that does NOT collapse both readouts ->
    non-discriminating instrument -> VOID (not a science PASS/FAIL)."""
    v = _load_verdict()
    # no_binding is SHARED; force it to stay healthy -> must VOID.
    bad = {"no_binding": {"wm": 0.85, "ep": 0.85}}
    rungs = [_rung(2, lesion_override=bad), _rung(4), _rung(8)]
    out = v.integrated_loop_verdict(rungs)
    assert out["GATE"] == "VOID", out
    assert out["instrument_valid"] is False, out


def test_inherited_verdict_fail_when_below_science_bar_at_smallest_load():
    """Sound + discriminating but the full loop is below the science bar
    even at N=2 -> GATE FAIL / classification FAIL."""
    v = _load_verdict()
    rungs = [_rung(2, full_wm=0.50, full_ep=0.50),
             _rung(4, full_wm=0.50, full_ep=0.50),
             _rung(8, full_wm=0.50, full_ep=0.50)]
    out = v.integrated_loop_verdict(rungs)
    assert out["GATE"] == "FAIL", out
    assert out["classification"] == "FAIL", out


def test_inherited_verdict_works_small_no_scale_confidence_reachable():
    """Minimal load passes but a larger load drops below the bar ->
    WORKS-SMALL-NO-SCALE-CONFIDENCE (an honest non-success)."""
    v = _load_verdict()
    rungs = [_rung(2, full_wm=0.85, full_ep=0.85),
             _rung(4, full_wm=0.50, full_ep=0.50),
             _rung(8, full_wm=0.50, full_ep=0.50)]
    out = v.integrated_loop_verdict(rungs)
    assert out["GATE"] == "FAIL", out
    assert out["classification"] == "WORKS-SMALL-NO-SCALE-CONFIDENCE", out


def test_controller_run_rung_shape_flows_through_inherited_verdict():
    """The controller's run_rung output keys are exactly what the inherited
    verdict consumes -- pin the end-to-end contract without a GPU run by
    constructing a rung with the controller's required keys and scoring it."""
    v = _load_verdict()
    # Build a 3-rung PASS-shaped set using the SAME key structure run_rung
    # emits (N, n_seeds, v1{wm,ep}, full{wm,ep}, lesions{7 names}{wm,ep}).
    rungs = [_rung(2), _rung(4), _rung(8)]
    # every required key present + scores without raising
    for r in rungs:
        assert set(r.keys()) == {"N", "n_seeds", "v1", "full", "lesions"}
        assert set(r["lesions"].keys()) == set(
            _COLLAPSE_BOTH + _COLLAPSE_WM + _COLLAPSE_EP)
    out = v.integrated_loop_verdict(rungs)
    assert out["GATE"] in ("PASS", "FAIL", "VOID")
