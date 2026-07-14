"""CI guard for the ON-BRIDGE (fully-spiking) coupling: a recurrent spiking reservoir region + the on-bridge selective
channel (cp_ssm_state) co-resident on ONE SimulationBridge -> the selective channel lifts the spiking reservoir past its
long-range conjunction bound, on real spikes, transport-free. Builds a real bridge (needs the sim backend); numpy CPU, one
seed. The bridge stepping is the cost -> one seed only."""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pytest


@pytest.fixture(scope="module")
def result():
    try:
        from research.runners._reslm_onbridge_couple_selssm_reservoir_derisk import run
    except Exception as e:                                        # pragma: no cover
        pytest.skip(f"onbridge-couple deps unavailable: {e}")
    return run(42)


def test_onbridge_selective_channel_lifts_spiking_reservoir(result):
    """res_plus_sel (spiking reservoir + on-bridge selective channel) beats res_only (reservoir alone) AND chance ->
    the on-bridge selective channel holds the distal KEY the fading spiking reservoir cannot."""
    assert result["res_plus_sel"] > result["res_only"] + 0.06
    assert result["res_plus_sel"] > (1.0 / 6) + 0.10
    assert result["GO"]
