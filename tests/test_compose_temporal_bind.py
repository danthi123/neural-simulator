import numpy as np
from sim.compose_temporal_bind import run_bind, _N, _GAP


def test_reuses_eligibility_kernel_unmodified():
    import sim.compose_temporal_bind as m
    src = open(m.__file__).read()
    assert "from sim.kernels import fused_eligibility_trace_decay" in src
    assert "fused_eligibility_trace_decay(" in src


def test_no_autograd():
    import sim.compose_temporal_bind as m
    src = open(m.__file__).read()
    assert "autograd" not in src and "torch" not in src


def test_V1_nogap_td_learns_bijection():
    assert run_bind("td", 42, 0) >= 0.90


def test_science_gapped_td_learns_compositional_binding():
    assert run_bind("td", 42, _GAP) >= 0.90


def test_hebbian_no_trace_is_faithful_v16_analog_and_fails():
    acc = run_bind("hebbian_no_trace", 42, _GAP)
    assert acc <= 0.35
    assert abs(acc - 1.0 / _N) < 1e-9


def test_permuted_and_wrongsign_fail():
    assert run_bind("permuted", 42, _GAP) <= 0.35
    assert run_bind("wrongsign", 42, _GAP) <= 0.35


def test_multiseed_decisive_discrimination():
    for s in (42, 43, 44):
        assert run_bind("td", s, _GAP) >= 0.90
        assert run_bind("hebbian_no_trace", s, _GAP) <= 0.35
