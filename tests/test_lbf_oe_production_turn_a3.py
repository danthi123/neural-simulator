"""Amendment-3 pins for research/runners/_lbf_open_ended_production_turn_probe.py (review 2026-09-23).

The amendment-2 GO statistic was DEGENERATE: each arm is a deterministic function of the seed, so under H0 the per-seed
difference is exactly 0 and a 6/6 sign-flip p = 1/64 only says "the modal reply changed on 6 seeds". Amendment 3 runs
M independent sessions per arm, each drawing on its OWN noise stream, so the per-seed difference has a real null
distribution. These tests FAIL on the amendment-2 runner (no a3 scorer / no noise stream) and pass after.
"""
from __future__ import annotations

import os
import random
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import research.runners._lbf_open_ended_production_turn_probe as P  # noqa: E402
from sim.backend import _reset_cache_for_tests  # noqa: E402


def test_deterministic_arms_are_undefined_not_a_pass():
    # the seed-42 amendment-2 shape: every session of an arm identical (the noise never moved the reply)
    I = [["rabbit"] + ["deer"] * 7] * P.A3_SESSIONS
    L = [["deer"] + ["rabbit"] * 7] * P.A3_SESSIONS
    s = P.score_seed_a3(*P._mk_seed(42, I, L))
    assert s["verdict"] == "UNDEFINED"
    assert not s["noise_live"]
    assert "delta" not in s


def test_null_is_real_aa_world_does_not_go():
    rng = random.Random(7)
    n_go = 0
    for _ in range(2000):
        per = {}
        for s in P.A3_SEEDS:
            v = [rng.random() for _ in range(2 * P.A3_SESSIONS)]
            per[str(s)] = {"verdict": "DEFINED",
                           "delta": sum(v[:P.A3_SESSIONS]) / P.A3_SESSIONS - sum(v[P.A3_SESSIONS:]) / P.A3_SESSIONS}
        n_go += P.aggregate_a3(per)["GO"]
    assert n_go / 2000. <= P.ALPHA


def test_gate_can_fail_each_way():
    rec = lambda d: {"verdict": "DEFINED", "delta": d}
    six = {str(s): rec(0.3) for s in P.A3_SEEDS}
    assert P.aggregate_a3(six)["GO"]
    assert not P.aggregate_a3(dict(six, **{"100": rec(-0.2)}))["GO"]            # 5/6 -> p = 7/64
    assert not P.aggregate_a3(dict(six, **{"100": {"verdict": "UNDEFINED"}}))["GO"]
    assert not P.aggregate_a3({str(s): rec(0.02) for s in P.A3_SEEDS})["GO"]      # below the effect floor
    assert P.aggregate_a3(six)["p_sign_test_heldout_excl_seed42"] == 1 / 32.


def test_noise_stream_varies_draws_and_leaves_global_rng_untouched(monkeypatch):
    # `_install_noise_stream` REFUSES on any backend but numpy (its state save/restore is exercised against plain
    # `np.random`, which only tracks the numpy backend's RNG -- see the function's own docstring). Pin it here:
    # on a GPU box `get_backend()` resolves to cupy by default (round-4 review, 2026-09-23 -- measured 18 passed /
    # 1 failed with no pin), and production is UNAFFECTED by pinning the test, because `_worker` (the only real
    # caller, used by every governed a3 session) already does `os.environ.setdefault("SIM_BACKEND", "numpy")`
    # before any run -- so a real a3 session always exercises this exact numpy path regardless of what this test
    # pins. Reset the cached backend after too, so later tests are not left pinned to numpy by this one.
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    _reset_cache_for_tests()
    try:
        _test_noise_stream_varies_draws_and_leaves_global_rng_untouched_body()
    finally:
        _reset_cache_for_tests()


def _test_noise_stream_varies_draws_and_leaves_global_rng_untouched_body():
    class FakeSampler:
        def _compete(self, drive, V):
            return np.random.randn(V)                     # the bank's OU noise comes from the global RNG

    def run(noise_seed):
        F2 = types.SimpleNamespace(SpikingWTASampler=type("S", (FakeSampler,), {}))
        stream = P._install_noise_stream(noise_seed, F2)
        np.random.seed(123)
        before = np.random.get_state()[1].copy()
        out = [F2.SpikingWTASampler()._compete(None, 3) for _ in range(4)]
        after = np.random.get_state()[1]
        assert (before == after).all()                    # the rest of the brain sees an unchanged global RNG
        assert stream["n_competes"] == 4
        return np.concatenate(out)

    a, b, c = run(1000), run(1000), run(1500)
    assert np.array_equal(a, b)                           # same stream -> same draws (the rebuild check)
    assert not np.array_equal(a, c)                       # a different stream -> different draws
    assert len(set(np.round(a, 6))) == len(a)             # the stream is not reset between draws


def test_noise_seed_assignment_prereg():
    assert P.a3_noise_seed(43, "intact", 2) == 43002
    assert P.a3_noise_seed(43, "lesion", 2) == 43502
    assert P.a3_noise_seed(43, "intact_rebuild", 0) == P.a3_noise_seed(43, "intact", 0)
