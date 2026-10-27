"""SIM_STABLE_RECAL (research/runners/stable_recal.py) -- the DEFAULT-OFF re-calibration of the GNW ignition-bus and
thought-swap workspaces for the stable integrator. Unset / "" / "0": the production organ builders are byte-identical
to the code before the switch existed, on the old integrator AND on the stable one (reference fingerprints captured
on that code with tests/_stable_recal_fingerprints.py). "1": requires SIM_STABLE_INTEGRATION=1 (raises without it)
and applies both re-calibrations; any other value raises.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from research.runners import stable_recal as SR  # noqa: E402
from tests import _stable_recal_fingerprints as F  # noqa: E402

STABLE_ENV = "SIM_STABLE_INTEGRATION"
RECAL_ENV = SR.STABLE_RECAL_ENV

# captured on the pre-switch code (base e38904197 + probes), numpy backend, seed 3
_REF_UNSET = {"old": {"swap": "716a7f5a975617d815bce644c45dae3d5ec0d4adaa46ec0a187d41ceed7f44c8",
                      "bus": "389b83cacdbe6d72c4fbc396a4c1c6ce68587cec00c2cdf383a03672bbb122a1"},
              "stable": {"swap": "976e88024cc579e4c49d56f690e8045f4b2aaaa8a07b3c0b0675a68c79226fd8",
                         "bus": "b03708f67c053b40df4e0e458909fc6eff9a670a44902dbbe60c29369f4c1648"}}
# the re-calibrated organs (SIM_STABLE_INTEGRATION=1 + SIM_STABLE_RECAL=1), pinned so a silent drift fails loudly
_REF_RECAL = {"swap": "2065a1aa8d157e53791751f51868086b4e1585b411a47f1f7ca90d36651f0638",
              "bus": "6359db2be1316d3ce98183f55a8aa9cd88b7a7fff3fad7e3cc2077b01145e821"}


def _numpy_backend():
    from sim.backend import get_backend
    return str(get_backend()[1]).lower() == "numpy"


def _env(monkeypatch, stable, recal):
    for name, val in ((STABLE_ENV, stable), (RECAL_ENV, recal)):
        if val is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, val)


# --- the switch itself ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
def test_off_values(monkeypatch, value):
    _env(monkeypatch, None, value)
    assert SR.stable_recal_requested() is False
    _env(monkeypatch, "1", value)
    assert SR.stable_recal_requested() is False


@pytest.mark.parametrize("stable", [None, "", "0"])
def test_on_without_stable_integration_raises(monkeypatch, stable):
    _env(monkeypatch, stable, "1")
    with pytest.raises(RuntimeError, match="SIM_STABLE_INTEGRATION"):
        SR.stable_recal_requested()


@pytest.mark.parametrize("value", ["2", "true", "yes", "on", "1x"])
def test_bad_values_raise(monkeypatch, value):
    _env(monkeypatch, "1", value)
    with pytest.raises(ValueError):
        SR.stable_recal_requested()


def test_on_requires_and_accepts_stable(monkeypatch):
    _env(monkeypatch, "1", "1")
    assert SR.stable_recal_requested() is True


def test_raise_reaches_the_production_builders(monkeypatch):
    """Set without SIM_STABLE_INTEGRATION, building either organ raises instead of silently running production."""
    _env(monkeypatch, None, "1")
    with pytest.raises(RuntimeError):
        F.swap_fingerprint()
    with pytest.raises(RuntimeError):
        F.bus_fingerprint()


# --- accessors: identity when off, the measured values when on ---------------------------------------------------

def test_accessors_identity_when_off(monkeypatch):
    _env(monkeypatch, "1", None)
    assert SR.bus_attractor_weight(30.0) == 30.0 and SR.bus_thr(1 / 6) == 1 / 6
    assert SR.bus_solo_plateau(1 / 3) == 1 / 3
    assert SR.bus_d_sub(3, 1000.0) == 1000.0 and SR.bus_d_sub(2, 1400.0) == 1400.0
    for name, (prod, _recal) in SR.SWAP.items():
        assert SR.swap_value(name, prod) == prod
    assert SR.swap_ignite_thresh(0.5, 1 / 6) == 1 / 6


def test_accessors_recal_values_when_on(monkeypatch):
    _env(monkeypatch, "1", "1")
    assert SR.bus_attractor_weight(30.0) == 5.0
    assert SR.bus_attractor_weight(0.0) == 0.0          # a lesion is kept
    assert SR.bus_attractor_weight(12.0) == 12.0        # an explicit research weight is kept
    assert SR.bus_thr(1 / 6) == pytest.approx(0.0445) and SR.bus_solo_plateau(1 / 3) == pytest.approx(0.089)
    assert SR.bus_d_sub(3, 1000.0) == 500.0 and SR.bus_d_sub(2, 1400.0) == 750.0
    assert SR.bus_d_sub(3, 1234.0) == 1234.0            # a non-production drive is kept
    expect = {"w_rec": 6.5, "norm_ws_w": 4.0, "thal_tonic_pa": 25.0, "solo_plateau": 0.084, "establish_pa": 1280.0,
              "w_ws_occ": 32.0, "w_pat_pred": 32.0, "w_occ_gate": 80.0, "w_pred_mm": 80.0, "w_gate_ws": 4.0,
              "timing_window": 36}
    for name, (prod, _recal) in SR.SWAP.items():
        assert SR.swap_value(name, prod) == expect[name]
    assert isinstance(SR.swap_value("timing_window", 9), int)
    assert SR.swap_value("w_rec", 0.0) == 0.0
    assert SR.swap_ignite_thresh(0.5, 1 / 6) == pytest.approx(0.042)


def test_production_call_sites_read_the_switch(monkeypatch):
    import webapp.gnw_three_organ_bus as TOB
    import webapp.gnw_two_organ_bus as TWO
    from research.runners import _gnw_norgan_bus_derisk as NB
    from research.runners import _gnw_coincidence_integrator_derisk as CI
    from research.runners import _gnw_reentrant_metacog_gated_deliberation_derisk as RE
    from research.runners import _gnw_two_distinct_organs_derisk as TD
    from research.runners import _gnw_rung2_competitive_access_derisk as R2
    _env(monkeypatch, "1", None)
    assert TOB._d_sub_3() == 1000.0 and TWO._d_sub_2() == 1400.0
    assert NB._thr() == NB.THR and CI._thr() == CI.THR and RE._thr() == RE.THR and TD._thr is CI._thr
    _env(monkeypatch, "1", "1")
    assert TOB._d_sub_3() == 500.0 and TWO._d_sub_2() == 750.0
    assert NB._thr() == CI._thr() == RE._thr() == pytest.approx(0.0445)
    assert R2.SOLO_PLATEAU == pytest.approx(1 / 3)       # the shared Rung-2 criterion (global stop) is untouched


# --- byte-identical when unset (both integrators) ---------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["old", "stable"])
@pytest.mark.parametrize("value", [None, "", "0"])
def test_unset_is_byte_identical(monkeypatch, integrator, value):
    _env(monkeypatch, "1" if integrator == "stable" else None, value)
    hs, _ = F.swap_fingerprint()
    hb, _ = F.bus_fingerprint()
    if _numpy_backend():
        assert hs == _REF_UNSET[integrator]["swap"]
        assert hb == _REF_UNSET[integrator]["bus"]


# --- on: the re-calibrated organs work by their own rules on the stable integrator -------------------------------

def test_recal_swap_and_bus_work_on_stable(monkeypatch):
    _env(monkeypatch, "1", "1")
    hs, r = F.swap_fingerprint()
    assert r["swapped"] and r["n_ignited_pre"] == 1 and r["n_ignited_post"] == 1 and r["winner_post"] == 1
    hb, rec = F.bus_fingerprint()
    assert all(v[0] == "fox" for v in rec.values())
    if _numpy_backend():
        assert hs == _REF_RECAL["swap"] and hb == _REF_RECAL["bus"]


def test_recal_bus_quorum_rules(monkeypatch):
    """The bus's own quorum rules at the re-calibrated drives: three votes ignite and two do not (unanimity); two
    votes ignite and one does not (coincidence)."""
    _env(monkeypatch, "1", "1")
    import webapp.gnw_three_organ_bus as TOB
    import webapp.gnw_two_organ_bus as TWO
    from research.runners import _p1_2_workspace_deliberation_loop_derisk as P
    from research.runners import _gnw_norgan_bus_derisk as NB
    from research.runners import _gnw_two_distinct_organs_derisk as TD
    np.random.seed(3)
    b, xp, slots, snap = P.build_workspace_bridge(3)
    three = NB.norgan_hop(b, xp, slots, snap, ["fox", "fox", "fox"], "owl", TOB._d_sub_3(), rng=np.random.default_rng(1))
    two = NB.norgan_hop(b, xp, slots, snap, ["fox", "fox", None], "owl", TOB._d_sub_3(), rng=np.random.default_rng(1))
    assert three[0] == "fox" and three[3] == 1
    assert two[0] is None and two[3] == 0
    pair = TD.coincidence_hop(b, xp, slots, snap, "fox", "fox", "owl", TWO._d_sub_2())
    solo = TD.coincidence_hop(b, xp, slots, snap, "fox", None, "owl", TWO._d_sub_2())
    assert pair[0] == "fox" and pair[3] == 1
    assert solo[0] is None and solo[3] == 0
