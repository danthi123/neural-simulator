"""CI guard for the additive on-bridge GRADED read-out over the SSM state (rung-ii forward, 2026-07-20).

`sim/bridge.py` computes `cp_ssm_readout_out = cp_ssm_readout_w @ cp_ssm_state` in the step loop (the read-out value
carried through the synapse weights, the OUTPUT analogue of M2's synaptic INPUT decode). This pins: (1) byte-identity
when OFF (cp_ssm_readout_w is None -> the block is skipped -> cp_ssm_readout_out stays None); (2) ON-path correctness
(the step-loop matvec equals W @ state exactly). CPU (numpy); skips if the ssm-state builder is unavailable.
"""
from __future__ import annotations
import os
import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
from sim.backend import to_host, get_backend  # noqa: E402

try:
    from research.runners._emerge_wkv_onbridge_derisk import _build_ssm_state_bridge
    _HAVE = True
except Exception:
    _HAVE = False

pytestmark = pytest.mark.skipif(not _HAVE, reason="ssm-state bridge builder unavailable")


def _bridge(D=6):
    b, chan_groups, _cg2, _snap = _build_ssm_state_bridge(D, seed=42, decay=0.8, pop_k=1)
    return b


def test_ssm_readout_off_is_byte_identical():
    """With cp_ssm_readout_w None (default), the read-out block is skipped -> cp_ssm_readout_out stays None."""
    b = _bridge()
    assert b.cp_ssm_readout_w is None and b.cp_ssm_readout_out is None
    b.cp_ssm_inject[:] = 0.0; b.cp_ssm_shunt[:] = 0.0
    b._run_one_simulation_step()
    assert b.cp_ssm_readout_out is None, "read-out block must be skipped when cp_ssm_readout_w is None"


def test_ssm_readout_forward_matches_matvec():
    """ON path: the step-loop computes cp_ssm_readout_out == W @ cp_ssm_state exactly."""
    xp, _ = get_backend()
    b = _bridge()
    N = int(b.cp_membrane_potential_v.size)
    n_out = 4
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_out, N)).astype(np.float32)
    b.cp_ssm_readout_w = xp.asarray(W)
    b.cp_ssm_state[:] = xp.asarray(rng.standard_normal(N).astype(np.float32))
    b.cp_ssm_inject[:] = 0.0
    b.cp_ssm_shunt[:] = -1.0                       # shunt=-1 => lam=1 => state frozen this step (known state)
    state = np.asarray(to_host(b.cp_ssm_state)).copy()
    b._run_one_simulation_step()
    out = np.asarray(to_host(b.cp_ssm_readout_out))
    assert out.shape == (n_out,)
    assert float(np.abs(out - (W @ state)).max()) < 1e-3, "on-bridge read-out must equal W @ state"
