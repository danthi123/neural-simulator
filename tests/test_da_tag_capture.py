"""Unit tests for webapp/da_tag_capture.py (no brain build): flag gate, lesions, capture windows, decay."""
import math
import os

import numpy as np
import pytest

from webapp import da_tag_capture as TC


class _FakeComp:
    """Just the store fields the ledger touches."""

    def __init__(self, n_blocks=3, D=8):
        self.D = D
        rng = np.random.default_rng(0)
        self.store_conns = []
        for i in range(n_blocks):
            ph = rng.uniform(0, 2 * np.pi, D)
            self.store_conns += [(100 + i * 10 + k + 1, 100 + i * 10, complex(np.exp(1j * ph[k]))) for k in range(D)]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION"):
        monkeypatch.delenv(k, raising=False)


def test_flag_default_off():
    assert TC.tag_capture_enabled() is False
    assert TC.maybe_ledger(42) is None


def test_flag_on(monkeypatch):
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    assert isinstance(TC.maybe_ledger(42), TC.TagCaptureLedger)


def test_threshold_is_the_existing_go_boundary():
    from webapp.da_mode_drives_chat import _DA_NEUTRAL_MAX
    assert TC.prp_threshold() == _DA_NEUTRAL_MAX


@pytest.mark.parametrize("flag", ["BRAIN_DA_ENCODING_LESION", "BRAIN_DA_CAPTURE_LESION"])
def test_lesions_pin_prp_read_to_tonic(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    assert TC.prp_da(1.2) == 0.5
    L = TC.TagCaptureLedger(1, threshold=0.62)
    assert L.observe_da(0.0, 1.2) is False and L.prp_events == []


def test_beta_zero_passthrough_is_bit_exact():
    comp = _FakeComp()
    before = list(comp.store_conns)
    L = TC.TagCaptureLedger(3, beta=0.0)
    L.on_store(comp, 0.0)
    assert comp.store_conns == before


def test_capture_windows_and_decay():
    comp = _FakeComp(n_blocks=3)
    L = TC.TagCaptureLedger(5, beta=1.0, threshold=0.62)
    # PRP event at t=0; block writes at t=0.5 (captured, inside tag window), t=1.9 (PRP 1.9 h before -> outside
    # the 1 h pre-window) ... emulate by registering blocks at different times.
    L.observe_da(0.0, 0.9)
    D = comp.D
    full = list(comp.store_conns)
    comp.store_conns = full[:D]
    L.on_store(comp, 0.5)                       # block 0, PRP 0.5 h before -> captured
    comp.store_conns = comp.store_conns + full[D:2 * D]
    L.on_store(comp, 1.9)                       # block 1, PRP 1.9 h before -> NOT captured
    comp.store_conns = comp.store_conns + full[2 * D:]
    L.on_store(comp, 3.0)                       # block 2, no PRP in window -> NOT captured
    L.advance(comp, 27.0)
    caps = [b["t_c"] for b in L.blocks]
    assert caps[0] == 0.5 and caps[1] is None and caps[2] is None
    f0, f1 = L.factor(L.blocks[0], 27.0), L.factor(L.blocks[1], 27.0)
    assert f0 > 0.9 and f1 == pytest.approx(math.exp(-(27.0 - 1.9) / L.tau_early_h))


def test_retroactive_capture_within_tag_window():
    comp = _FakeComp(n_blocks=1)
    L = TC.TagCaptureLedger(5, beta=1.0, threshold=0.62)
    L.on_store(comp, 0.0)
    L.advance(comp, 0.5)
    assert L.blocks[0]["t_c"] is None
    L.observe_da(1.0, 0.9)                     # a salient event 1 h AFTER the write (behavioural tagging)
    L.advance(comp, 25.0)
    assert L.blocks[0]["t_c"] == 1.0


def test_baseline_seeded_and_identical_across_ledgers():
    a = TC.TagCaptureLedger(9)._baseline(2, 16)
    b = TC.TagCaptureLedger(9)._baseline(2, 16)
    c = TC.TagCaptureLedger(10)._baseline(2, 16)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


# ── v3: SynapticTagCaptureLedger (DA acts through per-synapse tag / PRP / bistable late-phase state) ──────────────
def _syn_ledger(gamma, **kw):
    return TC.SynapticTagCaptureLedger(5, gamma=gamma, beta=1.0, **kw)


def test_v3_calibration_recovers_its_operating_point():
    for a in (0.1, 0.224, 0.4):
        g = TC.calibrate_gamma(a)
        assert TC.critical_activation(g) == pytest.approx(a, rel=1e-3)


def test_v3_no_host_threshold_subboundary_drive_can_capture():
    """The v2 compare (`da >= 0.62`) is gone: a D1 drive at HALF the Go-boundary activation captures if it lasts long
    enough, and the Go-boundary level itself does not capture if it is brief. The time course decides."""
    g = TC.calibrate_gamma(0.224)
    assert TC.kernel_capture_single(0.112, g, dur_h=0.5) > 0.5
    assert TC.kernel_capture_single(0.224, g, dur_h=1.0 / 60.0) < 0.5


def test_v3_prp_before_the_write_captures_but_not_hours_before():
    g = TC.calibrate_gamma(0.224)
    comp = _FakeComp(n_blocks=2)
    D = comp.D
    full = list(comp.store_conns)
    L = _syn_ledger(g)
    L.observe_turn(0.0, 5.0 / 60.0, 0.9, a_override=0.6)     # a salient event, then two plain writes
    comp.store_conns = full[:D]
    L.on_store(comp, 0.5)                                    # 30 min after the event -> captured
    comp.store_conns = comp.store_conns + full[D:]
    L.on_store(comp, 6.0)                                    # 6 h after -> the PRP pool is gone
    L.advance(comp, 30.0)
    z = [float(np.mean(b["z"])) for b in L.blocks]
    assert z[0] > 0.9 and z[1] < 0.1


def test_v3_tag_strength_from_the_write_gain_matters():
    """A DA-boosted write (|inc| = 2.5, a larger early-LTP tag) captures at a drive where a unit write does not."""
    g = TC.calibrate_gamma(0.224)
    comp1, comp2 = _FakeComp(n_blocks=1), _FakeComp(n_blocks=1)
    comp2.store_conns = [(p, q, 2.5 * w) for (p, q, w) in comp2.store_conns]
    out = []
    for comp in (comp1, comp2):
        L = _syn_ledger(g)
        L.observe_turn(0.0, 5.0 / 60.0, 0.9, a_override=0.15)
        L.on_store(comp, 0.0)
        L.advance(comp, 24.0)
        out.append(float(np.mean(L.blocks[0]["z"])))
    assert out[0] < 0.1 and out[1] > 0.9


def test_v3_lesions_act_on_the_edge(monkeypatch):
    g = TC.calibrate_gamma(0.224)
    monkeypatch.setenv("BRAIN_DA_CAPTURE_LESION", "1")
    L = _syn_ledger(g)
    assert L.observe_turn(0.0, 1.0, 1.2, a_override=0.9) == 0.0
    comp = _FakeComp(n_blocks=1)
    L.on_store(comp, 0.0)
    L.advance(comp, 24.0)
    assert L.p_max == 0.0 and float(np.max(L.blocks[0]["z"])) == 0.0


def test_v3_uncaptured_block_reads_as_pure_baseline_at_24h():
    comp = _FakeComp(n_blocks=1)
    L = _syn_ledger(TC.calibrate_gamma(0.224))
    L.on_store(comp, 0.0)
    L.advance(comp, 24.0)
    w = np.array([w for (_p, _q, w) in comp.store_conns])
    assert np.allclose(w, L.blocks[0]["base"], atol=1e-6)
