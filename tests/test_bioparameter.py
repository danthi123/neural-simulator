"""Tests for sim.bioparameter — provenance discipline."""
from __future__ import annotations

import pytest


def test_bioparameter_construction():
    from sim.bioparameter import BioParameter, Certainty
    bp = BioParameter(
        name="test_param", value=1.0, unit="ms",
        certainty=Certainty.HIGH, source="Kandel 2021",
    )
    assert bp.name == "test_param"
    assert bp.value == 1.0
    assert bp.certainty == Certainty.HIGH


def test_source_enum_resolves_to_string():
    """Source enum members get unwrapped to strings via __post_init__."""
    from sim.bioparameter import BioParameter, Certainty, Source
    bp = BioParameter(
        name="x", value=1, unit="ms",
        certainty=Certainty.HIGH, source=Source.BI_POO,
    )
    assert bp.source == Source.BI_POO.value
    assert "Bi & Poo" in bp.source


def test_bioparameter_is_hashable():
    """Frozen dataclass — usable as dict key or set member."""
    from sim.bioparameter import BioParameter, Certainty
    bp1 = BioParameter("x", 1.0, "ms", Certainty.HIGH, "src")
    bp2 = BioParameter("x", 1.0, "ms", Certainty.HIGH, "src")
    s = {bp1, bp2}
    assert len(s) == 1  # equal objects, one set entry


def test_registry_has_critical_params():
    """Core tuned parameters should all be in the registry."""
    from sim.bioparameter import PARAMETER_REGISTRY
    expected = ["dt_ms", "stdp_w_max", "ou_std_current_pA",
                "E_inh_mV", "topographic_bias_factor"]
    for name in expected:
        assert name in PARAMETER_REGISTRY, \
            f"Critical parameter '{name}' missing from registry"


def test_registry_summary_counts_certainty():
    from sim.bioparameter import registry_summary, Certainty
    counts = registry_summary()
    # All certainty levels are keys
    for c in Certainty:
        assert c.value in counts
    # Total = registry size
    from sim.bioparameter import PARAMETER_REGISTRY
    assert sum(counts.values()) == len(PARAMETER_REGISTRY)


def test_registry_no_unflagged_blindguess():
    """Any parameter without a clear citation should be marked BLINDGUESS.
    This catches drift where a 'tuned' param sneaks in claiming MEDIUM
    certainty without a real source."""
    from sim.bioparameter import PARAMETER_REGISTRY, Certainty
    for name, bp in PARAMETER_REGISTRY.items():
        # If certainty is HIGH or MEDIUM, source must NOT be just "BlindGuess"
        if bp.certainty in (Certainty.HIGH, Certainty.MEDIUM):
            assert "BlindGuess" not in bp.source, \
                f"{name}: certainty {bp.certainty.value} but source is BlindGuess"
            assert bp.source, f"{name}: HIGH/MEDIUM certainty must have source"


def test_get_returns_param():
    from sim.bioparameter import get
    bp = get("dt_ms")
    assert bp is not None
    assert bp.name == "dt_ms"
    assert bp.unit == "ms"


def test_get_returns_none_for_unknown():
    from sim.bioparameter import get
    assert get("nonexistent_param_xyz") is None


def test_audit_unregistered_returns_list():
    """audit_unregistered_params should return a list (may be non-empty
    since CoreSimConfig has many fields not yet audited)."""
    from sim.bioparameter import audit_unregistered_params
    out = audit_unregistered_params()
    assert isinstance(out, list)
    # Most fields are not yet registered — that's OK
    assert all(isinstance(x, str) for x in out)


def test_registered_params_not_in_unregistered_list():
    """Sanity: a registered param should NOT appear in unregistered list."""
    from sim.bioparameter import audit_unregistered_params, PARAMETER_REGISTRY
    unregistered = set(audit_unregistered_params())
    registered = set(PARAMETER_REGISTRY.keys())
    overlap = registered & unregistered
    assert not overlap, f"Registered params shown as unregistered: {overlap}"
