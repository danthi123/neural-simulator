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


# ── follow-up round (re-review of 4b6a9cf66): a failed restore never drives; the twin build shifts no RNG ─────────
def test_a_restore_that_raises_does_not_drive(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    org = _StatefulOrgan(hz=0.4, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)

    def broken(*a, **k):
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(RVA, "_restore_read_state", broken)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is False and "normalized" not in info
    assert info["footprint"]["restored_exact"] is False
    assert "restore exploded" in info["footprint"]["restore_error"] and info["error"].startswith("restore")


def test_an_inexact_restore_does_not_drive(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    org = _StatefulOrgan(hz=0.4, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    real = RVA._restore_read_state

    def inexact(*a, **k):
        out = real(*a, **k)
        out["exact"] = False
        return out

    monkeypatch.setattr(RVA, "_restore_read_state", inexact)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is False and "normalized" not in info
    assert info["footprint"]["restored_exact"] is False


def test_a_failed_restore_falls_through_to_the_pre_existing_afferent(ws, SO, monkeypatch):
    """End to end through observe_turn: the workspace is called with exactly the pre-existing arguments."""
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_AFFERENT", "1")
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: _StatefulOrgan(hz=0.4, thr=2.5))
    monkeypatch.setattr(RVA, "_restore_read_state", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    info = DAD.observe_turn(_chat(), "the dog chase the cat", seed=7)
    assert ws.calls[-1][1] == {"lesion": False, "afferent_override": None}
    assert info["reward_value"]["drives"] is False


class _ReseedingTwinOrgan(_StatefulOrgan):
    """A stub whose lesion twin's FIRST-USE build does what sim/bridge.py `_initialize_rng` does on every bridge build:
    reseed numpy's and Python's global generators (then draw from them, as a build does)."""

    def __init__(self, hz, thr):
        super().__init__(hz, thr)
        self._twin_built = False
        self._les = {"bridge": self.bridge, "idx_map": self.idx_map}

    def _ensure_les(self):
        if not self._twin_built:
            import random as _r
            np.random.seed(7)
            _r.seed(7)
            np.random.random(3)
            _r.random()
            self._twin_built = True
        return self._les


def _rng_fingerprint():
    import random as _r
    st = np.random.get_state()
    return st[1].copy(), st[2], _r.getstate()


def _seed_host_rngs():
    import random as _r
    np.random.seed(123)
    np.random.random(11)
    _r.seed(99)
    _r.random()


def test_the_lesion_twin_first_use_build_leaves_the_host_rngs_unchanged(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    org = _ReseedingTwinOrgan(hz=0.4, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    _seed_host_rngs()
    k0, pos0, py0 = _rng_fingerprint()
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert org._twin_built is True and info["drives"] is True
    k1, pos1, py1 = _rng_fingerprint()
    assert np.array_equal(k0, k1) and pos0 == pos1 and py0 == py1
    tb = info["footprint"]["twin_build"]
    assert tb["host_rngs_unchanged"] is True and tb["numpy_unchanged"] is True and tb["python_unchanged"] is True


def test_without_the_guard_the_twin_build_shifts_the_host_rngs_so_the_test_can_fail(SO, monkeypatch):
    """Sensitivity control for the test above: with `_global_rngs_untouched` replaced by a no-op, the same stub build
    changes numpy's and Python's global generators."""
    import webapp.reward_value_afferent_chat as RVA
    monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    org = _ReseedingTwinOrgan(hz=0.4, thr=2.5)
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)

    import contextlib

    @contextlib.contextmanager
    def noop(*a, **k):
        yield {"host_rngs_unchanged": True}

    monkeypatch.setattr(RVA, "_global_rngs_untouched", noop)
    _seed_host_rngs()
    k0, pos0, py0 = _rng_fingerprint()
    RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    k1, pos1, py1 = _rng_fingerprint()
    assert not (np.array_equal(k0, k1) and pos0 == pos1) and py0 != py1


def test_a_twin_build_that_changed_a_host_rng_does_not_drive(SO, monkeypatch):
    import webapp.reward_value_afferent_chat as RVA
    import contextlib
    monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: _ReseedingTwinOrgan(hz=0.4, thr=2.5))

    @contextlib.contextmanager
    def leaky(*a, **k):
        yield {"host_rngs_unchanged": False}

    monkeypatch.setattr(RVA, "_global_rngs_untouched", leaky)
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    assert info["drives"] is False and "normalized" not in info and info["error"].startswith("twin build")


class _FakeCupyRandom:
    """cupy.random's relevant surface: a per-device CURRENT generator object; `seed` reseeds that object IN PLACE
    (cupy's RandomState.seed does exactly this), and the object has no get_state/set_state."""

    class RandomState:
        def __init__(self, seed=0):
            self.stream = [int(seed)]

        def seed(self, s):
            self.stream[:] = [int(s)]

        def draw(self):
            self.stream.append(self.stream[-1] * 31 + 7)

    def __init__(self):
        self._cur = self.RandomState(42)

    def get_random_state(self):
        return self._cur

    def set_random_state(self, rs):
        self._cur = rs

    def seed(self, s):
        self._cur.seed(s)


def test_the_cupy_host_generator_object_is_set_aside_not_reseeded():
    import webapp.reward_value_afferent_chat as RVA
    fake = types.SimpleNamespace(random=_FakeCupyRandom())
    host = fake.random.get_random_state()
    host.draw()
    host.draw()
    before = list(host.stream)
    with RVA._global_rngs_untouched(xp=fake, name="cupy") as rec:
        fake.random.seed(7)                          # what `_initialize_rng` does to cupy on a build
        fake.random.get_random_state().draw()
    assert fake.random.get_random_state() is host and host.stream == before
    assert rec["cupy"] == "swapped" and rec["cupy_host_object_restored"] is True and rec["host_rngs_unchanged"] is True
    # sensitivity: the same build WITHOUT the swap reseeds the host object in place
    fake.random.seed(7)
    assert host.stream != before


@pytest.mark.skipif(os.environ.get("A10_SKIP_REAL_ORGAN") == "1", reason="opt-out for quick runs")
def test_real_lesion_twin_first_use_build_leaves_host_rngs_unchanged_and_builds_the_same_twin(SO, monkeypatch):
    """The real SurpriseProductionOrgan (standalone bridge, numpy): the lesion arm's first A10 call builds the twin
    (sim/bridge.py `_initialize_rng` reseeds numpy and Python random on that build). After the call the host's numpy
    and Python global generators are exactly as before, and the twin equals one built with no guard (the build
    reseeds from its own seed)."""
    import webapp.reward_value_afferent_chat as RVA
    from sim.backend import get_backend
    xp, _ = get_backend()
    if xp is not np:
        pytest.skip("real-organ pin runs on the numpy backend (SIM_BACKEND=numpy)")
    monkeypatch.setenv("BRAIN_REWARD_VALUE_LESION", "1")
    org = SO.SurpriseProductionOrgan(seed=7, shared=None)
    org.ensure_built()
    assert org.les is None                                            # the A10 call below is the first use
    monkeypatch.setattr(SO, "get_organ", lambda seed=42: org)
    _seed_host_rngs()
    k0, pos0, py0 = _rng_fingerprint()
    info = RVA.spiking_reward_value(_chat(), "the dog chase the cat", seed=7)
    k1, pos1, py1 = _rng_fingerprint()
    assert org.les is not None and info["drives"] is True
    assert np.array_equal(k0, k1) and pos0 == pos1 and py0 == py1
    assert info["footprint"]["twin_build"]["host_rngs_unchanged"] is True
    ref = SO.SurpriseProductionOrgan(seed=7, shared=None)._ensure_les()   # unguarded twin build, same seed
    for name in ("cp_neuron_firing_thresholds", "cp_membrane_potential_v"):
        assert np.array_equal(getattr(org.les["bridge"], name), getattr(ref["bridge"], name)), name
    a, b = org.les["bridge"].cp_connections, ref["bridge"].cp_connections
    assert np.array_equal(a.data, b.data) and np.array_equal(a.indices, b.indices)


def test_recall_probe_hash_sees_nested_changes_and_survives_cycles():
    """The AMENDMENT-3 recall probe's state hash: equal on an unchanged graph (with a cycle), different after a
    one-element change in a nested array, equal again once it is undone."""
    from research.runners._reward_value_afferent_recall_probe import deep_hash
    o = types.SimpleNamespace(a=np.arange(5.0), s=sp.csr_matrix(np.eye(3)), l=[1, 2, {"x": np.ones(2)}], t=None)
    o.self = o
    h0 = deep_hash(o)[0]
    assert deep_hash(o)[0] == h0
    o.l[2]["x"][1] = 3.0
    assert deep_hash(o)[0] != h0
    o.l[2]["x"][1] = 1.0
    assert deep_hash(o)[0] == h0
    o.s.data[0] = 2.0
    assert deep_hash(o)[0] != h0


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


_RC_BLOCK = {"on": True, "action": "rewrite", "agent": "dog", "action_verb": "chase", "old": "cat", "new": "fish"}


def _surprise_equalized_v2_arms(with_rc=True, rc_on_block=None):
    """v2 arms with the surprise blocks of on_a / les set equal to off_a's; with `with_rc`, an rc pair built from
    off_a / on_a whose CONTRA turn carries a reconsolidation block (off_rc: `_RC_BLOCK`; on_rc: `rc_on_block`, default
    the same)."""
    import copy
    arms = _v2_arms()
    for arm in ("on_a", "les"):
        arms[arm] = copy.deepcopy(arms[arm])
        for t in ("confirm", "contra"):
            arms[arm][t]["surprise"] = copy.deepcopy(arms["off_a"][t]["surprise"])
            arms[arm][t]["reconsolidation"] = copy.deepcopy(arms["off_a"][t].get("reconsolidation"))
    if with_rc:
        arms["off_rc"] = copy.deepcopy(arms["off_a"])
        arms["on_rc"] = copy.deepcopy(arms["on_a"])
        arms["off_rc"]["contra"]["reconsolidation"] = dict(_RC_BLOCK)
        arms["on_rc"]["contra"]["reconsolidation"] = dict(rc_on_block if rc_on_block is not None else _RC_BLOCK)
    return arms


def test_side_effect_check_passes_when_the_surprise_block_is_unchanged():
    from research.runners._reward_value_afferent_derisk import side_effect_check
    side = side_effect_check(_surprise_equalized_v2_arms())
    assert side["GO"] is True
    assert side["reconsolidation_half"]["measured"] is True and side["reconsolidation_half"]["GO"] is True
    assert set(side["surprise_half"]["per_arm"]) == {"on_a", "les", "on_rc"}


def test_reconsolidation_half_is_unmeasured_without_the_rc_pair_so_d_cannot_pass_on_none_vs_none():
    """AMENDMENT-3: in the five core arms reconsolidation is off and both blocks are None. AMENDMENT-2 scored that
    comparison, which could not fail; it is now reported and the half reads unmeasured, so (D) is not GO."""
    from research.runners._reward_value_afferent_derisk import side_effect_check
    arms = _surprise_equalized_v2_arms(with_rc=False)
    assert all(arms[a][t].get("reconsolidation") is None for a in ("off_a", "on_a", "les") for t in ("confirm", "contra"))
    side = side_effect_check(arms)
    assert side["surprise_half"]["GO"] is True
    assert side["reconsolidation_half"]["measured"] is False and side["reconsolidation_half"]["GO"] is None
    assert "no rc arm pair" in side["reconsolidation_half"]["why_unmeasured"]
    assert side["GO"] is False


def test_reconsolidation_half_is_unmeasured_when_reconsolidation_never_ran_in_off_rc():
    from research.runners._reward_value_afferent_derisk import side_effect_check
    arms = _surprise_equalized_v2_arms()
    for t in ("confirm", "contra"):
        arms["off_rc"][t]["reconsolidation"] = None
        arms["on_rc"][t]["reconsolidation"] = None
    side = side_effect_check(arms)
    assert side["reconsolidation_half"]["measured"] is False and side["GO"] is False


def test_reconsolidation_half_fails_when_the_on_rc_block_differs():
    from research.runners._reward_value_afferent_derisk import side_effect_check
    side = side_effect_check(_surprise_equalized_v2_arms(rc_on_block={"on": True, "action": "append"}))
    assert side["reconsolidation_half"]["measured"] is True and side["reconsolidation_half"]["GO"] is False
    assert side["reconsolidation_half"]["per_turn"]["contra"]["reconsolidation_equal"] is False
    assert side["GO"] is False


def _score_synthetic(tmp_path, arms_resp, lesion_equal=True):
    """Run mode_score on the v2 artifact with its per-arm responses replaced by `arms_resp` (and, with
    `lesion_equal`, the lesion arm's normalized reads set equal so (B) attribution and (C) hold)."""
    import copy
    import json
    from research.runners._reward_value_afferent_derisk import mode_score
    A = json.load(open(os.path.join(_V2, "s7_arms.json")))
    for arm, resp in arms_resp.items():
        rec = copy.deepcopy(A["arms"].get(arm) or A["arms"]["off_a"])
        rec["arm"], rec["responses"], rec["returncode"] = arm, copy.deepcopy(resp), 0
        A["arms"][arm] = rec
    for arm in list(A["arms"]):
        if arm not in arms_resp:
            A["arms"].pop(arm)
    if lesion_equal:
        for t in ("confirm", "contra"):
            A["arms"]["les"]["responses"][t]["da_drives"]["reward_value"]["normalized"] = 0.9
    ap = tmp_path / "arms.json"
    ap.write_text(json.dumps(A))
    out = tmp_path / "verdict.json"
    mode_score(str(ap), os.path.join(_V2, "s7_pre_off_main.json"), str(out))
    return json.loads(out.read_text())


def _full_v2(arms_sub):
    import json
    base = _V2
    full = {arm: json.load(open(os.path.join(base, "s7_arms_%s.json" % arm))) for arm in ("off_b", "on_b")}
    full.update(arms_sub)
    return full


@pytest.mark.parametrize("with_rc,rc_on_block,lesion_equal,expect", [
    (True, None, True, "GO"),                                    # every criterion measured and true
    (True, {"on": True, "action": "append"}, True, "NO-GO"),     # the reconsolidation half measured and false
    (False, None, True, "UNDEFINED"),                            # all else GO, reconsolidation half unmeasured
    (False, None, False, "NO-GO"),                               # (C) measured and false: NO-GO stands unmeasured
])
def test_score_reads_d_split_as_amendment_3_says(tmp_path, with_rc, rc_on_block, lesion_equal, expect):
    arms = _full_v2(_surprise_equalized_v2_arms(with_rc=with_rc, rc_on_block=rc_on_block))
    v = _score_synthetic(tmp_path, arms, lesion_equal=lesion_equal)
    assert v["status"] == expect, v["verdict"].get("undefined_reasons")


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
