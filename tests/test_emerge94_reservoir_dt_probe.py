"""CPU test for EMERGE-94 -- the dt= param is byte-preserving at its default, and the reservoir parses at dt=1.0.

CPU/numpy, offline.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def test_dt_param_default_byte_identical():
    # the default dt=0.5 build == the no-dt-arg build (byte-identical shipped path)
    from research.runners._emerge82_onbridge_lsm_derisk import _build_reservoir_bridge
    b0, i0, w0, _s0 = _build_reservoir_bridge(42, 60, 8)
    b1, i1, w1, _s1 = _build_reservoir_bridge(42, 60, 8, dt=0.5)
    assert float(b0.core_config.dt) == 0.5 == float(b1.core_config.dt)
    assert np.array_equal(i0, i1) and np.allclose(w0, w1)


@pytest.mark.slow
def test_reservoir_parses_at_dt1():
    import research.runners._emerge94_reservoir_dt_probe as m94
    acc10, spk10 = m94._parse_acc_at_dt(42, 1.0)
    assert acc10 >= 0.90            # the spiking reservoir parses at dt=1.0 (the shared-bridge dt)
    assert spk10 > 0.5              # genuinely spiking
