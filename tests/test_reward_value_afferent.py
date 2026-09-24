"""A10 (fix round, 2026-09-24): pins for the prediction-error salience afferent hook in
webapp/da_mode_drives_chat.observe_turn, webapp/reward_value_afferent_chat.py and the LBF row. No brain, no DA
substrate: the workspace and the surprise organ are stubbed. The failing directions are tested first (OFF must not
import the module; an error must not drive the SNc; no affect-valence fallback).
"""
import importlib
import os
import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp

os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import webapp.da_mode_drives_chat as DAD  # noqa: E402

RVA_NAME = "webapp.reward_value_afferent_chat"


class _FakeWS:
    def __init__(self):
        self.calls = []

    def observe(self, message, **kw):
        self.calls.append((message, kw))
        return {"acted": True, "mode": "rest", "lead": ""}


@pytest.fixture
def ws(monkeypatch):
    w = _FakeWS()
    monkeypatch.setattr(DAD, "get_workspace", lambda chat, seed=None: w)
    monkeypatch.delenv("BRAIN_DA_DRIVES_INDUCE", raising=False)
    monkeypatch.delenv("BRAIN_REWARD_VALUE_AFFERENT", raising=False)
    monkeypatch.delenv("BRAIN_REWARD_VALUE_LESION", raising=False)
    return w


class _BlockImport:
    """A meta-path finder that makes importing the A10 module fail loudly."""

    def find_spec(self, name, path=None, target=None):
        if name == RVA_NAME:
            raise ImportError("A10 module imported while the flag is OFF")
        return None


def _chat():
    return types.SimpleNamespace(inner=types.SimpleNamespace(what_does=lambda a, v: "cat"))


@pytest.mark.parametrize("val", [None, "0", "", "off", "false", "no"])
def test_off_never_imports_the_module_and_calls_observe_unchanged(ws, monkeypatch, val):
    if val is not None:
        monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", val)
    import webapp
    monkeypatch.delitem(sys.modules, RVA_NAME, raising=False)
    monkeypatch.delattr(webapp, "reward_value_afferent_chat", raising=False)   # else `from webapp import` skips import
    blocker = _BlockImport()
    sys.meta_path.insert(0, blocker)
    try:
        info = DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    finally:
        sys.meta_path.remove(blocker)
    assert RVA_NAME not in sys.modules
    assert "reward_value" not in info
    assert ws.calls == [("the dog chase the fish", {"lesion": False, "afferent_override": None})]


@pytest.mark.parametrize("val,expect", [("1", True), ("true", True), ("ON", True), ("yes", True), ("0", False),
                                        ("off", False), ("", False)])
def test_hook_and_module_parse_the_flag_the_same_way(ws, monkeypatch, val, expect):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", val)
    called = []
    monkeypatch.setattr(RVA, "spiking_reward_value", lambda chat, message, seed: called.append(1) or None)
    DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    assert bool(called) is expect
    assert RVA.reward_value_enabled() is expect


def test_error_record_does_not_drive_the_snc(ws, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")
    monkeypatch.setattr(RVA, "spiking_reward_value",
                        lambda chat, message, seed: {"on": True, "source": "surprise", "drives": False, "error": "x"})
    info = DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": None}   # the pre-existing call, no override
    assert info["reward_value"]["error"] == "x"


def test_exception_in_the_module_does_not_drive_the_snc(ws, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")

    def boom(chat, message, seed):
        raise RuntimeError("organ exploded")

    monkeypatch.setattr(RVA, "spiking_reward_value", boom)
    info = DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": None}
    assert info["reward_value"]["drives"] is False and "organ exploded" in info["reward_value"]["error"]


def test_a_drive_goes_through_turn_signal_not_afferent_override(ws, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")
    monkeypatch.setattr(RVA, "spiking_reward_value",
                        lambda chat, message, seed: {"source": "surprise", "drives": True, "normalized": 0.7})
    info = DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": None, "turn_signal_override": 0.7}
    assert info["reward_value"]["normalized"] == 0.7


def test_none_falls_through_unchanged(ws, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")
    monkeypatch.setattr(RVA, "spiking_reward_value", lambda chat, message, seed: None)
    info = DAD.observe_turn(_chat(), "", seed=7)
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": None}
    assert "reward_value" not in info


def test_manual_induction_keeps_priority(ws, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")
    monkeypatch.setenv("BRAIN_DA_DRIVES_INDUCE", "800")
    called = []
    monkeypatch.setattr(RVA, "spiking_reward_value", lambda chat, message, seed: called.append(1))
    DAD.observe_turn(_chat(), "the dog chase the fish", seed=7)
    assert called == []
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": 800.0}


# ── the module itself, with a stub organ ─────────────────────────────────────────────────────────────────────────
class _FakeOrgan:
    def __init__(self, hz, thr, les_weight=0.0):
        self.hz, self.thr = hz, thr
        self._block = {}
        self.seed = 7
        n = 8
        self.idx_map = {"patient_expected": np.arange(0, 4), "surprise": np.arange(4, 8)}
        W = np.zeros((n, n), dtype=np.float32)
        W[4:8, 0:4] = 1.0                       # intact pe->surprise (rows = post)
        self.bridge = types.SimpleNamespace(cp_connections=sp.csr_matrix(W))
        L = np.zeros((n, n), dtype=np.float32)
        L[4:8, 0:4] = les_weight
        self._les = {"bridge": types.SimpleNamespace(cp_connections=sp.csr_matrix(L)), "idx_map": self.idx_map}

    def _ensure_les(self):
        return self._les

    def judge(self, a, v, ps, pa, lesion=False):
        self._block.setdefault(ps.lower(), 0)
        if pa.lower() != ps.lower():
            self._block.setdefault(pa.lower(), 8)
        return {"surprise_hz": self.hz, "threshold": self.thr, "surprised": self.hz >= self.thr}


@pytest.fixture
def SO(monkeypatch):
    import research.runners.surprise_production_organ as _SO
    monkeypatch.setattr(_SO, "surprise_enabled", lambda: True)
    monkeypatch.delenv("BRAIN_REWARD_VALUE_LESION", raising=False)
    return _SO


def test_no_affect_valence_fallback(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    chat = _chat()
    chat._affect_drives_workspace = types.SimpleNamespace(ema_valence=0.8, ema_arousal=0.3)
    assert RVA.spiking_reward_value(chat, "hello there", seed=7) is None
    assert RVA.spiking_reward_value(chat, "", seed=7) is None
    assert not hasattr(RVA, "_affect_valence_path")


def test_surprise_path_normalizes_against_the_organ_threshold(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    org = _FakeOrgan(hz=5.0, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the fish", seed=7)
    assert info["drives"] is True and info["source"] == "surprise"
    assert info["normalized"] == pytest.approx(1.0)
    assert (info["stored_block"], info["asserted_block"]) == (0, 8)
    assert "pa" not in info and "lesion_cut" not in info


def test_degenerate_threshold_does_not_drive(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: _FakeOrgan(hz=1.0, thr=0.0))
    info = RVA.spiking_reward_value(_chat(), "the dog chase the fish", seed=7)
    assert info["drives"] is False and "normalized" not in info


def test_organ_failure_does_not_drive(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA

    def broken(seed=42):
        raise RuntimeError("build failed")

    monkeypatch.setattr(SO, "get_organ", broken)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the fish", seed=7)
    assert info["drives"] is False and "normalized" not in info and "build failed" in info["error"]


@pytest.mark.parametrize("les_weight,holds", [(0.0, True), (0.5, False)])
def test_lesion_records_a_read_time_cut_check(SO, monkeypatch, les_weight, holds):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: _FakeOrgan(hz=4.0, thr=2.5, les_weight=les_weight))
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    cut = info["lesion_cut"]
    assert cut["holds"] is holds
    assert cut["intact_reference"]["total"] == pytest.approx(16.0)
    assert cut["lesion_twin"]["total"] == pytest.approx(16.0 * les_weight)


# ── the LBF row ──────────────────────────────────────────────────────────────────────────────────────────────────
def _load_row(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("BRAIN_REWARD_VALUE_AFFERENT", raising=False)
    else:
        monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", flag)
    import research.runners.lbf_rows.reward_value_afferent as ROW
    return importlib.reload(ROW)


def test_row_is_thin_when_the_master_flag_is_off(monkeypatch):
    ROW = _load_row(monkeypatch, None)
    (key, spec), = ROW.EXTRA_LESIONS.items()
    assert spec["kind"] == "thin" and spec["flag"] == "BRAIN_REWARD_VALUE_LESION"
    assert list(ROW.EXTRA_PROBES)[0][3] is True


def test_row_is_a_neural_lesion_when_the_master_flag_is_on(monkeypatch):
    ROW = _load_row(monkeypatch, "1")
    (key, spec), = ROW.EXTRA_LESIONS.items()
    assert spec["kind"] == "neural-lesion"
    assert list(ROW.EXTRA_PROBES)[0][3] is False
    _load_row(monkeypatch, None)


def test_row_probes_confirm_with_decision_fields_only_and_works_as_list_and_dict(monkeypatch):
    ROW = _load_row(monkeypatch, None)
    rows = list(ROW.EXTRA_PROBES)                          # main's documented list interface
    assert len(rows) == 1 and isinstance(rows[0], tuple) and len(rows[0]) == 4
    key, turn, fields, _thin = rows[0]
    assert key in ROW.EXTRA_LESIONS
    assert turn == "confirm"
    assert "da_drives.reward_value.normalized" not in fields
    assert set(fields) == {"da_drives.reward_value.source", "da_drives.reward_value.surprised", "da_drives.mode",
                           "da_drives.lead"}
    # the AG-REG hook's reading (research/lbf-row-registry-hook): `for key, row in EXTRA_PROBES.items()`
    pairs = list(ROW.EXTRA_PROBES.items())
    assert pairs == [(key, rows[0])]
    for k, row in ROW.EXTRA_PROBES.items():
        assert isinstance(row, (tuple, list)) and len(row) == 4 and row[0] == k


def test_row_probe_turn_exists_in_the_battery():
    from research.runners.onebrain_regression_battery import PROBE_TURNS
    labels = {t[0]: t for t in PROBE_TURNS}
    assert labels["confirm"][1] == "the dog chase the cat" and labels["confirm"][2] == "surp"
