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

    def ensure_built(self):
        pass

    def _ensure_les(self):
        return self._les

    def judge(self, a, v, ps, pa, lesion=False):
        self._block.setdefault(ps.lower(), 0)
        if pa.lower() != ps.lower():
            self._block.setdefault(pa.lower(), 8)
        return {"surprise_hz": self.hz, "threshold": self.thr, "surprised": self.hz >= self.thr}


class _StatefulOrgan(_FakeOrgan):
    """A stub organ whose read MUTATES everything a real read on the shared pool can: per-neuron arrays in place, an
    array re-bound to a new object, the sparse weight data, a scalar, a list, the runtime clock, a new attribute, the
    host block bookkeeping and the global numpy RNG -- and whose rate DEPENDS on that history (the first read returns
    `hz`, later reads `hz / 2`), like the production CONFIRM read on the merged pool (0.4051 then 0.3472 Hz)."""

    def __init__(self, hz, thr):
        super().__init__(hz, thr)
        b = self.bridge
        b.cp_neuron_firing_thresholds = np.linspace(-50.0, -40.0, 8).astype(np.float32)
        b.cp_neuron_activity_ema = np.zeros(8, dtype=np.float32)
        b.cp_membrane_potential_v = np.full(8, -65.0, dtype=np.float32)
        b._blk = 24
        b._pending = []
        b.runtime_state = types.SimpleNamespace(current_time_step=0, current_time_ms=0.0)
        self._cue_next, self._novel_next = 0, 8

    def judge(self, a, v, ps, pa, lesion=False):
        b = self.bridge
        first = bool(b.cp_neuron_activity_ema.sum() == 0.0)
        b.cp_neuron_activity_ema += 0.5                                   # in place
        b.cp_neuron_firing_thresholds += 1.0
        b.cp_membrane_potential_v = b.cp_membrane_potential_v + 3.0       # re-bound
        b.cp_connections.data *= 1.5                                      # weights
        b._blk = 12
        b._pending.append("x")
        b.runtime_state.current_time_step += 120
        b.runtime_state.current_time_ms += 120.0
        b._created_by_read = np.ones(2)
        self._cue_next += 1
        np.random.random(5)
        out = super().judge(a, v, ps, pa, lesion)
        out["surprise_hz"] = self.hz if first else self.hz / 2.0
        return out


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


def test_composer_class_is_read_from_the_agent_itself(SO, monkeypatch):
    """chat.inner (BrainConversationalAgent) holds .composer directly; the v2 seed-7 run recorded null because the
    first version looked for chat.inner.agent.composer."""
    import webapp.reward_value_afferent_chat as RVA

    class OneBrainComposer:
        pass

    monkeypatch.setattr(SO, "get_organ", lambda seed=42: _FakeOrgan(hz=1.0, thr=2.5))
    chat = _chat()
    chat.inner.composer = OneBrainComposer()
    info = RVA.spiking_reward_value(chat, "the dog chase the cat", seed=7)
    assert info["composer"] == "OneBrainComposer"
    assert RVA._composer_class(_chat()) is None


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


# ── the read leaves no footprint (fix round 2, review of 7d5c2743d) ────────────────────────────────────────────────
def _bridge_fingerprint(org):
    b = org.bridge
    out = {}
    for k, v in vars(b).items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif sp.issparse(v):
            out[k] = (v.data.copy(), v.indices.copy(), v.indptr.copy())
        elif isinstance(v, list):
            out[k] = list(v)
        elif k == "runtime_state":
            out[k] = dict(vars(v))
        else:
            out[k] = v
    return out, dict(org._block), org._cue_next, org._novel_next


def _fingerprints_equal(f1, f2):
    (b1, blk1, c1, n1), (b2, blk2, c2, n2) = f1, f2
    if set(b1) != set(b2) or (blk1, c1, n1) != (blk2, c2, n2):
        return False
    for k in b1:
        x, y = b1[k], b2[k]
        if isinstance(x, np.ndarray):
            if not np.array_equal(x, y):
                return False
        elif isinstance(x, tuple):
            if not all(np.array_equal(p, q) for p, q in zip(x, y)):
                return False
        elif x != y:
            return False
    return True


def test_raw_read_is_history_dependent_so_the_stub_can_fail(SO, monkeypatch):
    """Sensitivity control for the test below: WITHOUT isolation the stub's read changes its own state and the
    second (production) read differs from the first."""
    org = _StatefulOrgan(hz=0.4, thr=2.5)
    before = _bridge_fingerprint(org)
    first = org.judge("dog", "chase", "cat", "cat")["surprise_hz"]
    assert not _fingerprints_equal(before, _bridge_fingerprint(org))
    assert org.judge("dog", "chase", "cat", "cat")["surprise_hz"] != first


@pytest.mark.parametrize("lesion", [False, True])
def test_the_a10_read_leaves_no_footprint_and_production_reads_as_if_flag_off(SO, monkeypatch, lesion):
    import webapp.reward_value_afferent_chat as RVA
    if lesion:
        monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    org = _StatefulOrgan(hz=0.4, thr=2.5)
    if lesion:   # the lesion read drives the twin bridge: make the twin the stateful one for this case
        org._les = {"bridge": org.bridge, "idx_map": org.idx_map}
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    before = _bridge_fingerprint(org)
    rng_before = np.random.get_state()[1].copy()
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is True
    assert info["surprise_hz"] == pytest.approx(0.4)                  # the A10 read itself is the organ's first read
    assert info["stored_block"] == 0                                   # recorded before the bookkeeping is restored
    fp = info["footprint"]
    assert fp["isolated"] is True and fp["restored_exact"] is True
    assert fp["bridge"] == ("lesion_twin" if lesion else "intact")
    for name in ("cp_neuron_activity_ema", "cp_neuron_firing_thresholds", "cp_membrane_potential_v",
                 "cp_connections", "_blk", "_pending"):
        assert name in fp["changed_during_read"], name
    assert fp["added_during_read"] == ["_created_by_read"]
    assert set(fp["runtime_state_changed"]) == {"current_time_step", "current_time_ms"}
    assert set(fp["organ_bookkeeping_changed"]) == {"_block", "_cue_next"}
    assert _fingerprints_equal(before, _bridge_fingerprint(org))      # every mutated piece of state is back
    assert np.array_equal(np.random.get_state()[1], rng_before)       # global numpy RNG restored
    # the production read that follows sees the flag-OFF state: it is the organ's FIRST read, as with the flag off
    assert org.judge("dog", "chase", "cat", "cat")["surprise_hz"] == pytest.approx(0.4)


def test_no_snapshot_means_no_read(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    org = _StatefulOrgan(hz=0.4, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)

    def broken(*a, **k):
        raise MemoryError("no room for the snapshot")

    monkeypatch.setattr(RVA, "_snapshot_read_state", broken)
    before = _bridge_fingerprint(org)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is False and "snapshot" in info["error"]
    assert _fingerprints_equal(before, _bridge_fingerprint(org))      # the organ was never read


@pytest.mark.skipif(os.environ.get("A10_SKIP_REAL_ORGAN") == "1", reason="opt-out for quick runs")
def test_real_standalone_organ_read_is_restored_exactly(SO, monkeypatch):
    """The real SurpriseProductionOrgan (standalone bridge, numpy): the A10 read mutates the bridge's state (the
    record lists what) and the restore returns every array, sparse weight and scalar to its pre-read value."""
    import webapp.reward_value_afferent_chat as RVA
    from sim.backend import get_backend
    xp, _ = get_backend()
    if xp is not np:
        pytest.skip("real-organ pin runs on the numpy backend (SIM_BACKEND=numpy)")
    org = SO.SurpriseProductionOrgan(seed=7, shared=None)
    org.ensure_built()
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    snap = RVA._snapshot_obj(org.bridge)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is True
    fp = info["footprint"]
    assert fp["restored_exact"] is True
    assert "cp_membrane_potential_v" in fp["changed_during_read"]     # the read really ran on this bridge
    for name, rec in snap.items():
        now = getattr(org.bridge, name)
        if rec[0] == "dense":
            assert now is rec[1] and np.array_equal(now, rec[2], equal_nan=True), name
        elif rec[0] == "sparse":
            assert np.array_equal(now.data, rec[2]), name
        elif rec[0] == "scalar":
            assert RVA._scalar_equal(now, rec[1]), name
    assert set(vars(org.bridge)) == set(snap)


# ── AMENDMENT-2 (D): the side-effect criterion can fail, and fails on the v2 data it was written for ───────────────
_V2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "research", "findings", "raw", "_reward_value_afferent_derisk", "v2")


def _v2_arms(sub=""):
    import json
    base = os.path.join(_V2, sub)
    return {arm: json.load(open(os.path.join(base, "s7_arms_%s.json" % arm))) for arm in ("off_a", "on_a", "les")}


@pytest.mark.parametrize("sub", ["", "rf"])
def test_side_effect_check_fails_on_the_v2_arms(sub):
    from research.runners._reward_value_afferent_derisk import side_effect_check
    side = side_effect_check(_v2_arms(sub))
    assert side["GO"] is False
    conf = side["per_arm"]["on_a"]["confirm"]
    assert conf["surprise_equal"] is False
    assert (conf["surprise_hz_off"], conf["surprise_hz_arm"]) == (0.4050925925925926, 0.3472222222222222)
    assert side["per_arm"]["les"]["confirm"]["surprise_equal"] is True      # the twin read never touched it


def test_side_effect_check_passes_when_the_surprise_block_is_unchanged():
    import copy
    from research.runners._reward_value_afferent_derisk import side_effect_check
    arms = _v2_arms()
    for arm in ("on_a", "les"):
        arms[arm] = copy.deepcopy(arms[arm])
        for t in ("confirm", "contra"):
            arms[arm][t]["surprise"] = copy.deepcopy(arms["off_a"][t]["surprise"])
            arms[arm][t]["reconsolidation"] = copy.deepcopy(arms["off_a"][t].get("reconsolidation"))
    assert side_effect_check(arms)["GO"] is True


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
