"""Process-only guards for the retired two-pool morphology runner."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import _productive_morphology_two_pool_derisk as gate


def test_cli_requires_an_explicit_phase_before_any_execution():
    with pytest.raises(SystemExit):
        gate.build_parser().parse_args([])


@pytest.mark.parametrize(
    ("base_seed", "n_seeds"),
    [(42, 6), (42, 3), (45, 3), (100, 3), (99, 2)],
)
def test_recorded_seed_ranges_are_consumed_and_closed(base_seed, n_seeds):
    with pytest.raises(ValueError, match="consumed by existing evidence and closed"):
        gate.validate_execution_request("historical", base_seed, n_seeds)


def test_unrecorded_seeds_remain_closed_without_a_preregistered_partition():
    assert gate.OPEN_PHASES == ()
    with pytest.raises(ValueError, match="no scientific partition is preregistered"):
        gate.validate_execution_request("calibration", 700, 6)


def test_invalid_request_cannot_reach_simulation(monkeypatch):
    monkeypatch.setattr(
        gate,
        "summarize",
        lambda *args, **kwargs: pytest.fail("closed request reached simulation"),
    )
    with pytest.raises(ValueError, match="consumed by existing evidence and closed"):
        gate.main(["--phase", "historical", "--seed", "42", "--n-seeds", "6"])


def test_direct_entry_points_cannot_bypass_the_closed_phase(monkeypatch):
    monkeypatch.setattr(
        gate,
        "build_two_pool",
        lambda *args, **kwargs: pytest.fail("closed direct call built a bridge"),
    )
    with pytest.raises(ValueError, match="consumed by existing evidence and closed"):
        gate.run(42, verbose=False)
    with pytest.raises(ValueError, match="no scientific partition is preregistered"):
        gate.summarize(700, n_seeds=6, verbose=False)
