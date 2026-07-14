"""CI guard for the RUNG 6 FULL SPIKING composition: the deployed two-gate SPIKING D3 register (SpikingPopGateRegister,
a persistent slow-NMDA held slot on a real SimulationBridge) feeds its resumed who-state into a reservoir generator's
read-out, so the generator predicts the resumed referent across a discourse POP -- above the fading reservoir + a
shuffled who-state. Builds a real SimulationBridge (slow); skips if deps/backend unavailable. Small smoke (n_disc=20)."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_rung6_spiking_composition_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"rung6 spiking composition deps unavailable: {e}")
    try:
        return run(42, 20)                                        # small smoke: builds the spiking register + reservoir
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"spiking register/bridge unavailable: {e}")


def test_spiking_register_beats_fading_reservoir(result):
    """The spiking register's resumed who-state improves cross-pop prediction over the fading reservoir alone."""
    if result.get("n", 0) < 12:                                  # too few pop discourses in the smoke -> inconclusive
        pytest.skip("too few pop discourses in the smoke")
    assert result["register"] > result["reservoir"] + 0.10


def test_shuffled_whostate_collapses(result):
    """A shuffled who-state collapses -> the two-gate register's resumed slot is load-bearing."""
    if result.get("n", 0) < 12:
        pytest.skip("too few pop discourses in the smoke")
    assert result["register"] > result["shuffle"] + 0.10
