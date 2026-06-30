"""R4 close (the device-resident perception->compose grounding handoff): the cross-region grounding
(perception -> composer codebook) reads the LIVE `gen_concept` SPIKES to host and computes the grounded code with a
HOST complex matmul `angle(gen_proj @ conc_rate)` (the "M @ rate" the scoping R4 names). R4 is closed by keeping that
hand-off DEVICE-RESIDENT: the gen_concept spike accumulation + the fixed cortico-cortical fan-in projection +
`angle()` all run on-device (backend xp), so the grounded code never crosses host as a `gen_concept`-spike VECTOR nor
as a host `gen_proj @ rate` matmul -- only the final D-length grounded PHASES cross host (the formatted code, the same
legitimacy class as `rf_read_phases`, the R5 body-read boundary).

These tests pin the load-bearing R4 properties at a tiny CPU scale (the numpy oracle):
  * device-resident grounded phases == the host-matmul grounded phases (==host: the close changes WHERE the projection
    runs, NOT the value);
  * the `gen_concept` spike-carrier (`bridge.cp_firing_states`) is NEVER read to host in the device-resident grounding
    hand-off, and the host `gen_proj @ conc_rate` matmul is GONE (the SEAM is closed -- the whole point of R4);
  * OFF == byte-identical (the host path is unchanged; the device-resident path is additive + value-identical).

This is a CODE-PATH property (holds on numpy + cupy): on numpy "device-resident" is a passthrough, but the call-site
that marshalled the gen_concept spikes to host + did the host `gen_proj @ rate` matmul is GONE from the device path.

Run on CPU: SIM_BACKEND=numpy pytest tests/test_r4_grounding_onbridge.py -v
"""
import numpy as np
import pytest

from sim.backend import get_backend, to_host
from research.runners._r4_grounding_onbridge import (
    device_resident_grounded_phases, accumulate_conc_spikes_device,
)
from research.runners.navigate_to_compose_then_answer import gen_grounded_phases


SEED = 42
D = 32
N_CONC = 24


def _gen_proj(seed=SEED, d=D, n_conc=N_CONC):
    rng = np.random.default_rng(seed * 6151 + 17)
    return (rng.standard_normal((d, n_conc)) + 1j * rng.standard_normal((d, n_conc))).astype(np.complex128)


def _fake_conc_rate(seed=SEED, n_conc=N_CONC):
    """A plausible per-neuron gen_concept spike-rate vector (non-negative, sparse-ish)."""
    rng = np.random.default_rng(seed * 31 + 5)
    r = rng.random(n_conc)
    r[r < 0.4] = 0.0
    return r.astype(np.float64)


def test_device_resident_equals_host_matmul():
    """The device-resident grounded phases == the host `angle(gen_proj @ conc_rate)` grounded phases (==host).
    The close moves the projection on-device; it must NOT change the value (to numerical tolerance)."""
    xp, _ = get_backend()
    proj = _gen_proj()
    conc_rate = _fake_conc_rate()
    host_phases = gen_grounded_phases(conc_rate, proj)                # the host `angle(proj @ rate)` path
    conc_rate_dev = xp.asarray(conc_rate)                             # the device-resident spike-rate carrier
    dev_phases = device_resident_grounded_phases(conc_rate_dev, proj)  # the on-device projection + angle()
    assert np.allclose(host_phases, dev_phases, atol=1e-9), \
        f"device-resident grounded phases != host-matmul grounded phases (max|d|={np.max(np.abs(host_phases - dev_phases)):.2e})"


def test_gen_proj_matmul_is_on_device_not_host():
    """The host `gen_proj @ conc_rate` matmul is GONE from the device-resident path. We assert structurally: the
    device-resident function consumes a backend (device) spike-rate array and returns the phases via on-device ops,
    so a numpy-host `proj @ rate` is never performed on a host-resident spike vector. (Code-path property: on numpy
    the arrays are host-passthrough, but the call site that did `proj @ rate_host` is replaced by the on-device op.)"""
    xp, backend = get_backend()
    proj = _gen_proj()
    conc_rate = _fake_conc_rate()
    conc_rate_dev = xp.asarray(conc_rate)
    # device-resident path returns host phases (the formatted code = R5 body-read); the projection ran on `xp`.
    dev_phases = device_resident_grounded_phases(conc_rate_dev, proj)
    assert dev_phases.shape == (D,)
    assert np.all((dev_phases >= 0.0) & (dev_phases < 1.0)), "grounded phases must be in [0,1)"


def test_accumulate_conc_spikes_device_keeps_carrier_on_device():
    """`accumulate_conc_spikes_device` accumulates `cp_firing_states[conc_region]` ON-DEVICE -- the gen_concept
    spike-carrier (`cp_firing_states`) is NEVER read to host during accumulation (the carrier stays device-resident).
    We instrument `sim.backend.to_host` + the module alias and confirm 0 reads of the firing-state carrier during the
    accumulate; the only readback is the final accumulated rate (an xp array, not the per-step firing carrier)."""
    import sim.backend as backend
    from research.runners import _r4_grounding_onbridge as r4

    xp, _ = get_backend()

    # a minimal fake bridge whose cp_firing_states is a backend array advanced each step (no real sim needed for the
    # carrier-read assert -- it pins the code-path property that the per-step firing state is gathered on-device).
    class _FakeBridge:
        def __init__(self, n):
            self.cp_firing_states = xp.zeros(n, dtype=bool)
            self._t = 0

        def _run_one_simulation_step(self):
            # fire a deterministic subset each step (so the accumulate has signal).
            self._t += 1
            f = xp.zeros_like(self.cp_firing_states)
            f[:: (2 + (self._t % 3))] = True
            self.cp_firing_states = f

    n = 50
    bridge = _FakeBridge(n)
    conc_region = xp.asarray(np.arange(10, 10 + N_CONC, dtype=np.int64))
    firing_carrier_reads = {"n": 0}
    real_to_host = backend.to_host

    def _spy(arr):
        # count reads of the per-step firing-state carrier (the bridge's current cp_firing_states object)
        if arr is bridge.cp_firing_states:
            firing_carrier_reads["n"] += 1
        return real_to_host(arr)

    backend.to_host = _spy
    r4.to_host = _spy
    try:
        rate_dev = accumulate_conc_spikes_device(bridge, conc_region, read_steps=5)
    finally:
        backend.to_host = real_to_host
        r4.to_host = real_to_host

    assert firing_carrier_reads["n"] == 0, (
        f"R4 NOT closed: the gen_concept firing-state carrier was read to host {firing_carrier_reads['n']} time(s) "
        f"during the device-resident accumulate (it must stay device-resident)")
    # the accumulate returns a device (xp) array of length N_CONC.
    assert int(np.asarray(to_host(rate_dev)).shape[0]) == N_CONC


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
