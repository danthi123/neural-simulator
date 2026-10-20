"""Step 0a of `docs/drafts/bg-tonic-loop/PACKET.md` (RULING bg-tonic-loop-1020): the two DEFAULT-OFF
env switches `BRAIN_BG_PRESET_CELLS` and `BRAIN_BG_BOUNDED_COND`.

What is pinned here:
  * flag OFF (either switch unset) is byte-identical to the pre-switch merged tree: every bridge array
    hash plus the post-build RNG state of all four golden cases (plain build_selector_bridge and full
    NeuralSeamSelector.ensure_built, seeds 9 and 16) must match tests/_bg_loop_identity_golden.json,
    captured on the UNEDITED merged research/bg-tonic-loop tree before this edit existed;
  * flag ON plumbing: each switch sets its CoreSimConfig field through BOTH build paths (a direct
    build_selector_bridge call and NeuralSeamSelector, whose core_config_updates today come only from
    neural_seam_parts).
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tests._capture_bg_loop_identity_golden import CASES, GOLDEN_PATH, snapshot  # noqa: E402

SWITCHES = ("BRAIN_BG_PRESET_CELLS", "BRAIN_BG_BOUNDED_COND")


def _clear_switches() -> None:
    for k in SWITCHES:
        os.environ.pop(k, None)


def _load_golden() -> dict:
    with open(GOLDEN_PATH) as fh:
        return json.load(fh)


def test_flag_off_identical_to_golden(monkeypatch):
    """Both switches unset: every bridge array + the post-build RNG state match the golden exactly,
    through build_selector_bridge and through NeuralSeamSelector.ensure_built (seeds 9 and 16)."""
    for k in SWITCHES:
        monkeypatch.delenv(k, raising=False)
    golden = _load_golden()
    assert set(golden["cases"]) == set(CASES), "golden/case set mismatch"
    for name, builder in CASES.items():
        got = snapshot(builder())
        want = golden["cases"][name]
        assert set(got) == set(want), f"{name}: array set changed (extra/missing bridge arrays)"
        differing = sorted(k for k in want if got[k] != want[k])
        assert not differing, f"{name}: flag-off build is NOT byte-identical, differs in {differing}"


def test_switches_reach_core_config_both_paths(monkeypatch):
    """Each switch ON sets its CoreSimConfig field through both build paths; both unset, both default OFF."""
    from research.runners._vocal_action_selector_gate import build_selector_bridge

    _clear_switches()
    plain = build_selector_bridge(9)
    assert plain.core_config.preset_centred_parameter_heterogeneity is False
    assert plain.core_config.enable_bounded_conductance_integration is False

    monkeypatch.setenv("BRAIN_BG_PRESET_CELLS", "1")
    monkeypatch.setenv("BRAIN_BG_BOUNDED_COND", "1")
    plain_on = build_selector_bridge(9)
    assert plain_on.core_config.preset_centred_parameter_heterogeneity is True
    assert plain_on.core_config.enable_bounded_conductance_integration is True

    from research.runners.bg_action_selection_production_organ import NeuralSeamSelector
    org = NeuralSeamSelector(seed=9)
    org.ensure_built()
    assert org._bridge is not None
    assert org._bridge.core_config.preset_centred_parameter_heterogeneity is True
    assert org._bridge.core_config.enable_bounded_conductance_integration is True
