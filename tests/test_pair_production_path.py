"""The pair's production-path seams (branch research/pair-production-path-arms; Amendment 7 of the sleep-replay-capture
prereg): the wall-clock injection point, the cupy RNG restore in `_private_rng`, the waking-only DA lesion knob, and the
runner's gates (its selftest drives every gate through its failing directions). No brain is built here."""
from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)

from webapp import da_tag_capture as T                     # noqa: E402
from webapp import da_tag_capture_chat as W                # noqa: E402
from webapp import sleep_replay_capture as S               # noqa: E402

FLAGS = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
         "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION",
         "BRAIN_DA_ENCODING_LESION_SPARE_SWR", "BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE")


class FakeD1:
    def __init__(self, *a, **kw):
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class FakeComposer:
    def __init__(self, D=32, seed=0):
        self.D = D
        self.store_conns = []
        self.patterns = []
        self._rng = np.random.default_rng(seed)

    def store(self, g=1.0):
        u = np.exp(2j * np.pi * self._rng.random(self.D))
        i = len(self.patterns)
        self.patterns.append(u)
        trig = 1000 + i * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(g * u[k])) for k in range(self.D)]

    def _block_role_scores(self, i):
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        c = float(abs(np.mean(np.conj(self.patterns[i]) * w)) / max(1e-12, float(np.mean(np.abs(w)))))
        return {"agent": ("a", 1.0, c, None), "action": ("b", 1.0, c, None), "patient": ("c", 1.0, c, None)}


class Chat:
    def __init__(self, comp):
        self.inner = types.SimpleNamespace(composer=comp)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setattr(W, "SpikingD1Activation", FakeD1)
    monkeypatch.setattr(W, "_WALL_CLOCK", None)
    yield


# ── the wall-clock seam ─────────────────────────────────────────────────────────────────────────────────────────────
def test_wall_clock_default_is_time_time(monkeypatch):
    # replace only the module's own `time` reference (never the global time.time: other importers read it at import)
    clock = [1000.0]
    monkeypatch.setattr(W, "time", types.SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    comp = FakeComposer()
    comp.store()
    cap = W.ChatTagCapture(Chat(comp), 7)
    assert cap.mode == "wall" and cap.t0_wall == 1000.0   # read through time.time when no source is installed
    clock[0] = 2800.0
    assert abs(cap.now_h() - 0.5) < 1e-12


def test_wall_clock_seam_installs_and_restores():
    vt = [0.0]
    prev = W.set_wall_clock(lambda: vt[0])
    try:
        assert prev is None and W._wall_now() == 0.0
        vt[0] = 5400.0
        assert W._wall_now() == 5400.0
    finally:
        assert W.set_wall_clock(prev) is not None
    assert W._WALL_CLOCK is None


def test_a_virtual_wall_day_is_reproducible_and_leaves_no_clock_behind():
    from research.runners._pair_production_path_probe import design_fake
    a = design_fake(None, False)
    b = design_fake(None, False)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert len(a["epochs"]) == 7                     # 3 pauses + the night + 3 idle nights, on the wall clock
    assert W._WALL_CLOCK is None and W._WORLD_OFFSET_H == 0.0


# ── the cupy RNG restore ────────────────────────────────────────────────────────────────────────────────────────────
class _FakeRS:
    def __init__(self, seed=None):
        self.seeded = seed

    def seed(self, s):
        self.seeded = s


def _fake_cupy(monkeypatch):
    rnd = types.SimpleNamespace()
    rnd.state = _FakeRS("global")
    rnd.get_random_state = lambda: rnd.state
    rnd.set_random_state = lambda rs: setattr(rnd, "state", rs)
    rnd.RandomState = _FakeRS
    rnd.seed = lambda s: rnd.state.seed(s)
    xp = types.SimpleNamespace(random=rnd)
    import sim.backend as SB
    monkeypatch.setattr(SB, "get_backend", lambda: (xp, "cupy"))
    return rnd


def test_private_rng_restores_the_cupy_stream(monkeypatch):
    rnd = _fake_cupy(monkeypatch)
    glob = rnd.state
    with W._private_rng(42, 3) as ctx:
        assert rnd.state is not glob and rnd.state.seeded == ctx.s     # a private stream inside
    assert rnd.state is glob and glob.seeded == "global"              # the one other organs draw from, untouched


def test_private_rng_legacy_knob_reproduces_the_reseed(monkeypatch):
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE", "1")
    rnd = _fake_cupy(monkeypatch)
    glob = rnd.state
    with W._private_rng(42, 3) as ctx:
        pass
    assert rnd.state is glob and glob.seeded == ctx.s                 # left reseeded (the pre-fix behaviour)


def test_private_rng_numpy_path_unchanged():
    st = np.random.get_state()[1].copy()
    with W._private_rng(42, 3):
        np.random.random(5)
    assert np.array_equal(st, np.random.get_state()[1])


# ── the waking-only DA lesion ─────────────────────────────────────────────────────────────────────────────────────
def _conversation(monkeypatch, env):
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CLOCK", "turn")
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    comp = FakeComposer()
    comp.store()
    chat = Chat(comp)
    for i, da in enumerate([1.0, 1.1, 0.5, 1.1, 1.0]):
        chat._last_da_drives = {"da_level": da}
        W.observe_chat_turn(chat, seed=7)
        if i == 2:
            comp.store()
        W.after_store_chat(chat)
    W.advance_world_clock_h(24.0)
    W.tick_chat(chat)
    return chat._da_tag_capture


def test_full_da_lesion_pins_the_swr_read(monkeypatch):
    cap = _conversation(monkeypatch, {"BRAIN_DA_ENCODING_LESION": "1"})
    e = cap._src.summary()["epochs"][0]
    assert e["da_seen_by_d1"] == T._DA_TONIC and "da_lesion_spares_swr" not in e
    assert all(t["da_seen_by_d1"] == T._DA_TONIC for t in cap.ledger.turn_log)


def test_waking_only_lesion_keeps_the_swr_edge(monkeypatch):
    cap = _conversation(monkeypatch, {"BRAIN_DA_ENCODING_LESION": "1", "BRAIN_DA_ENCODING_LESION_SPARE_SWR": "1"})
    e = cap._src.summary()["epochs"][0]
    assert e["da_lesion_spares_swr"] is True and e["da_seen_by_d1"] == e["da_swr"] and e["da_swr"] > T._DA_TONIC
    assert all(t["da_seen_by_d1"] == T._DA_TONIC for t in cap.ledger.turn_log)     # waking D1 input still pinned


def test_spare_knob_never_lifts_the_capture_lesion(monkeypatch):
    cap = _conversation(monkeypatch, {"BRAIN_DA_CAPTURE_LESION": "1", "BRAIN_DA_ENCODING_LESION_SPARE_SWR": "1"})
    assert cap._src.summary()["epochs"][0]["da_seen_by_d1"] == T._DA_TONIC


def test_intact_epoch_record_has_no_new_key(monkeypatch):
    cap = _conversation(monkeypatch, {})
    assert all("da_lesion_spares_swr" not in e for e in cap._src.summary()["epochs"])


# ── the runner's gates ────────────────────────────────────────────────────────────────────────────────────────────
def test_pair_probe_selftest_passes():
    from research.runners._pair_production_path_probe import selftest
    assert selftest()
