"""Fast checks for the sleep transitive-inference family (prereg
research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md). No brain is built here: the grader's
registered fixtures, the surrogate rasters' invariants, the weight-budget match, the part merge and the dev
operating-point check."""
from __future__ import annotations

import numpy as np

from research.runners import sleep_transitive_inference as S
from research.runners import sleep_transitive_inference_grade as G


def _toy_store(m=3, size=4, extra=2):
    asm = [np.arange(i * size, (i + 1) * size) for i in range(m)]
    return dict(asm_local=asm, n_pc=m * size + extra, m=m)


def test_grader_selftest_passes():
    assert G.selftest()


def test_scramble_keeps_every_cells_spike_count_and_shifts_whole_periods():
    st = _toy_store()
    c = dict(S.DEFAULTS, swr_period=10)
    rng = np.random.default_rng(0)
    F = rng.random((100, st["n_pc"])) < 0.1
    Gr, shifts = S.scramble_raster(st, F, 7, c)
    assert np.array_equal(Gr.sum(0), F.sum(0))
    assert all(v % 10 == 0 and v > 0 for v in shifts.values())
    for i, al in enumerate(st["asm_local"]):          # an assembly moves as one block
        assert np.array_equal(Gr[:, al], np.roll(F[:, al], shifts[S.NAMES[i]], axis=0))


def test_reverse_keeps_counts_and_coparticipation_and_reverses_order():
    st = _toy_store()
    c = dict(S.DEFAULTS, swr_period=10)
    F = np.zeros((20, st["n_pc"]), bool)
    F[1, 0] = True
    F[5, 4] = True
    Gr = S.reverse_raster(st, F, c)
    assert np.array_equal(Gr.sum(0), F.sum(0))
    assert Gr[8, 0] and Gr[4, 4]                      # inside period 0: t -> 9 - t
    assert np.flatnonzero(Gr[:, 4])[0] < np.flatnonzero(Gr[:, 0])[0]


def test_budget_match_equalises_total_absolute_change_and_clips():
    st = dict(between_pos=np.arange(4))
    w0 = np.array([10.0, 10.0, 10.0, 10.0, 60.0], np.float32)
    w_ref = np.array([20.0, 10.0, 10.0, 10.0, 60.0], np.float32)       # total |dw| 10
    w_sur = np.array([30.0, 30.0, 10.0, 10.0, 60.0], np.float32)       # total |dw| 40
    w, k = S.budget_match(st, w0, w_ref, w_sur, 900.0)
    assert abs(k - 0.25) < 1e-9
    assert abs(float(np.abs(w[:4] - w0[:4]).sum()) - 10.0) < 1e-4
    assert w[4] == 60.0                                                 # a non-plastic synapse is untouched


def _part(seed, arms, h="abc", **kw):
    return dict(seed=seed, cfg=dict(S.DEFAULTS), names=S.NAMES, arms={a: {"x": 1} for a in arms},
                encode=dict(w_hash=h), arms_requested=list(arms), error=None, **kw)


def test_merge_parts_joins_arms_and_flags_disagreement():
    rec = G.merge_parts([_part(7, ["wake", "sleep"]), _part(7, ["off", "dis"])])
    assert rec["error"] is None and set(rec["arms"]) == {"wake", "sleep", "off", "dis"}
    assert "U7" in G.merge_parts([_part(7, ["wake"]), _part(7, ["off"], h="zzz")])["error"]
    assert "U7" in G.merge_parts([_part(7, ["wake"]), _part(7, ["wake"])])["error"]
    assert "U7" in G.merge_parts([_part(7, ["wake"]), _part(8, ["off"])])["error"]


def test_devcheck_reads_no_inference_and_catches_saturation_and_a_leaky_disruption():
    rec = G._fixture()
    rec["cfg"] = dict(S.DEFAULTS)
    rec["arms"]["sleep"]["reads"]["foil_direct_max"] = 0.0
    assert G.devcheck(rec)["all"]
    rec["arms"]["sleep"]["reads"]["DI"] = 0.0                          # DI is not a criterion
    assert G.devcheck(rec)["all"]
    rec["arms"]["sleep"]["reads"]["foil_direct_max"] = 0.2
    assert not G.devcheck(rec)["all"]
    rec["arms"]["sleep"]["reads"]["foil_direct_max"] = 0.0
    rec["arms"]["dis"]["night"]["reach"]["frac_propagated"] = 0.5
    assert not G.devcheck(rec)["all"]


def test_gate_run_refuses_a_non_numpy_backend(monkeypatch, capsys):
    import pytest
    monkeypatch.setenv("SIM_BACKEND", "cupy")
    monkeypatch.setattr("sys.argv", ["x", "--seed", "42"])
    with pytest.raises(SystemExit):
        S.main()


def test_weight_permutation_keeps_the_multiset_of_changes():
    st = dict(between_pos=np.arange(5))
    w0 = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 60.0], np.float32)
    w_ref = np.array([20.0, 5.0, 10.0, 40.0, 10.0, 60.0], np.float32)
    w = S.permute_dw(st, w0, w_ref, 42)
    assert np.allclose(np.sort(w[:5] - w0[:5]), np.sort(w_ref[:5] - w0[:5]))
    assert w[5] == 60.0
