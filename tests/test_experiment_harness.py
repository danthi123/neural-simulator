"""The harness must FAIL CLOSED. If these ever pass by warning instead of raising, the harness is decorative.

Every case here is a real incident from the project record, encoded so it cannot recur silently.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from tools.experiment import Experiment, HarnessError

BASE = dict(name="t", lane="X", question="q", hypothesis="h", gate="g", kill="k",
            one_variable="v", corpus_check=False)


def _exp(**kw):
    return Experiment(**{**BASE, **kw})


def test_pre_registration_is_mandatory():
    """Every field exists because omitting it cost a retraction or a day of compute."""
    for field in ("question", "hypothesis", "gate", "kill", "one_variable"):
        with pytest.raises(HarnessError):
            _exp(**{field: ""})


def test_identical_arms_blocked():
    """Two identical arms produce two identical numbers that look like a result (the kp arm, 2026-07-31)."""
    with pytest.raises(HarnessError):
        _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 1}})


def test_confounded_arms_blocked():
    """ONE FLAG != ONE VARIABLE: mean-subtract also changed weight mass AND firing rate."""
    with pytest.raises(HarnessError):
        _exp(one_variable="a", arms={"A": {"a": 1, "b": 1}, "B": {"a": 2, "b": 9}})


def test_verdict_requires_validated_instrument():
    """Most retractions here were correct measurements read through an unverified instrument."""
    e = _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 2}})
    with pytest.raises(HarnessError):
        e.verdict(observed={"x": 1}, passed=True)


def test_bound_trap_blocked():
    """The plasticity bound trap: documented for 4 rules, then hit a 5th (w_max=150 vs W0=250)."""
    e = _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 2}})
    with pytest.raises(HarnessError):
        e.check_bounds(btsp_w_max=(150, 250))
    e.check_bounds(btsp_w_max=(2500, 250))          # bound above weight: fine


def test_powerless_instrument_blocked():
    """A control that agreed with its treatment to 1e-9 in 29/36 runs, printing confident negatives."""
    e = _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 2}})
    with pytest.raises(HarnessError):
        e.validate_instrument(lambda c: 0.5, lambda i: i, lambda i: i, n=10)


def test_crying_wolf_instrument_blocked():
    """A false alarm is as corrosive as a missed one -- it trains the reader to ignore the flag."""
    e = _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 2}})
    with pytest.raises(HarnessError):
        e.validate_instrument(lambda c: 0.001, lambda i: i, lambda i: i, n=10)


def test_happy_path_records_full_provenance(tmp_path):
    """A FILENAME IS NOT PROVENANCE: one run's pool_k was recoverable only by forensics on its synapse count."""
    e = _exp(one_variable="a", arms={"A": {"a": 1}, "B": {"a": 2}})
    e.check_bounds(btsp_w_max=(2500, 250))
    e.validate_instrument(lambda c: 0.001 if c else 0.7, lambda i: True, lambda i: False, n=10)
    out = tmp_path / "a.json"
    rec = e.verdict(observed={"ratio": 5.05}, passed=True, artifact=str(out))
    assert out.exists()
    for k in ("gate", "kill_criterion", "one_variable", "arms", "instrument", "observed"):
        assert k in rec, "artifact must embed %s" % k
    assert rec["instrument"]["power"] >= 0.9
