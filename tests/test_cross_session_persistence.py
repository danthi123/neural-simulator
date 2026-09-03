"""Cross-session persistence (#B8) — the FAST, deterministic, CI-gateable subset of the selftest.

Two properties the mission requires, both pinned here (the full real-substrate D5 teach->save->fresh-reload->recall
cycle is `python -m research.runners.cross_session_persistence` — heavy: it builds the n_ca3=2000 GO organ):

  (1) flag OFF  -> every save/reload entry point is INERT: returns None/0/False and writes NO file. This is the
      byte-identical-off guarantee at the API boundary (no substrate is even reachable from here without a build).
  (2) flag ON   -> the CSR weight round-trip + fingerprint-gated wholesale overwrite RESTORES a learned weight array
      EXACTLY (the byte-level analogue of the D5 within-assembly weight surviving a restart), and REFUSES to restore
      across a structural mismatch (degrade to baseline, never corrupt by positional misalignment).

Pre-fix these fail by import error (the module does not exist); post-fix they pass. The array-restore test is the
teeth: a fresh "pre-teach baseline" matrix, overwritten from a saved "post-teach" matrix, must equal post-teach and
differ from pre-teach — exactly the gap the persistence closes.
"""
import os

import numpy as np
import pytest

from research.runners import cross_session_persistence as CSP


# ── a lightweight CSR stand-in for a substrate connection matrix (same attributes the real save path reads) ──
def _csr(nnz_data, indices, indptr, shape):
    sp = pytest.importorskip("scipy.sparse")
    return sp.csr_matrix((np.asarray(nnz_data, dtype=np.float32),
                          np.asarray(indices, dtype=np.int32),
                          np.asarray(indptr, dtype=np.int32)), shape=shape)


def _example(data):
    # a fixed 4x4 CSR structure; only `data` varies (mirrors "same build, different learned weights")
    indices = [0, 2, 1, 3, 0, 3]
    indptr = [0, 2, 4, 5, 6]
    return _csr(data, indices, indptr, (4, 4))


# ────────────────────────────────────────────────────────────────────────────────────────────
# (2) fingerprint + weight round-trip + fingerprint-gated overwrite.
# ────────────────────────────────────────────────────────────────────────────────────────────
def test_fingerprint_stable_and_structure_sensitive():
    a = _example([1, 2, 3, 4, 5, 6])
    b = _example([9, 9, 9, 9, 9, 9])          # SAME structure, different weights
    fa = CSP._csr_fingerprint(*CSP._sparse_struct(a)[:3])
    fb = CSP._csr_fingerprint(*CSP._sparse_struct(b)[:3])
    assert fa == fb, "fingerprint must ignore weights (same structure -> same fingerprint)"
    # a different structure (different indices) -> different fingerprint
    diff = _csr([1, 2, 3, 4, 5, 6], [0, 1, 1, 2, 0, 2], [0, 2, 4, 5, 6], (4, 4))
    fd = CSP._csr_fingerprint(*CSP._sparse_struct(diff)[:3])
    assert fd != fa


def test_weight_roundtrip_and_overwrite_restores_learned_state(tmp_path):
    """The teeth: a POST-teach weight array survives a save -> fresh-baseline-rebuild -> reload cycle exactly."""
    pre = _example([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])    # "pre-teach baseline"
    post = _example([0.90, 0.85, 0.10, 0.10, 0.70, 0.10])   # "post-teach" (learned)
    stem = tmp_path / "d5" / "sess"
    CSP._save_csr_weights(stem, post, {"seed": 42, "topics": ["cat", "dog"], "formed": [1]})

    # simulate a restart: a FRESH build at the baseline, same structure.
    fresh = _example([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    assert np.allclose(fresh.data, pre.data)

    sidecar = CSP._load_csr_sidecar(stem)
    host = CSP._load_csr_weights(stem)
    assert sidecar is not None and host is not None
    # fingerprint gate must pass (same structure) then overwrite.
    idx, indptr, shape, nnz = CSP._sparse_struct(fresh)
    assert CSP._csr_fingerprint(idx, indptr, shape) == sidecar["fingerprint"]
    CSP._overwrite_csr_data(fresh, host)

    assert np.allclose(fresh.data, post.data), "reload must restore the learned weights exactly"
    assert not np.allclose(fresh.data, pre.data), "reload must NOT leave the fresh baseline (the gap it closes)"
    # sidecar bookkeeping round-trips
    assert sidecar["formed"] == [1] and sidecar["topics"] == ["cat", "dog"]


def test_fingerprint_mismatch_refuses_restore():
    """A structural mismatch (build-to-build drift) must be DETECTABLE so the reload skips rather than corrupts."""
    saved = _example([0.9, 0.8, 0.1, 0.1, 0.7, 0.1])
    saved_fp = CSP._csr_fingerprint(*CSP._sparse_struct(saved)[:3])
    drifted = _csr([0.1] * 6, [0, 1, 1, 2, 0, 2], [0, 2, 4, 5, 6], (4, 4))   # different indices
    drifted_fp = CSP._csr_fingerprint(*CSP._sparse_struct(drifted)[:3])
    assert saved_fp != drifted_fp   # the reload's gate would trip -> skip weight overwrite


# ────────────────────────────────────────────────────────────────────────────────────────────
# (1) flag OFF -> inert (no file, no restore).
# ────────────────────────────────────────────────────────────────────────────────────────────
def test_flag_off_is_default_and_inert(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAIN_PERSIST_LEARNING", raising=False)
    assert CSP.persist_learning_enabled() is False, "default MUST be off (byte-identical-off)"
    # every save/reload entry point returns the inert value and writes nothing
    assert CSP.save_d5_organ(("s", "b", "r"), object(), base=tmp_path) is None
    assert CSP.save_xedge(base=tmp_path, pool=object()) is None
    assert CSP.save_session_facts(("s", "b", "r"), object(), base=tmp_path) is None
    assert CSP.save_session_learning(("s", "b", "r"), object(), object(), base=tmp_path) == {}
    assert CSP.reload_d5_organ(("s", "b", "r"), 42, ["cat"], base=tmp_path) is None
    assert CSP.reload_xedge(object(), base=tmp_path) is False
    assert CSP.reload_session_facts(("s", "b", "r"), object(), base=tmp_path) == 0
    assert CSP.reload_session_learning(("s", "b", "r"), object(), 42, ["cat"], base=tmp_path) == {}
    assert not list(tmp_path.rglob("*")), "OFF path must write no files"


def test_flag_on_recognized():
    for v in ("1", "true", "on", "yes", "TRUE"):
        os.environ["BRAIN_PERSIST_LEARNING"] = v
        try:
            assert CSP.persist_learning_enabled() is True
        finally:
            del os.environ["BRAIN_PERSIST_LEARNING"]
    for v in ("0", "false", "off", "no", ""):
        os.environ["BRAIN_PERSIST_LEARNING"] = v
        try:
            assert CSP.persist_learning_enabled() is False
        finally:
            del os.environ["BRAIN_PERSIST_LEARNING"]


# ────────────────────────────────────────────────────────────────────────────────────────────
# (c) runtime-acquired facts round-trip through developed_brain_io (a lightweight duck-typed composer).
# ────────────────────────────────────────────────────────────────────────────────────────────
class _FakeComposer:
    def __init__(self, facts=()):
        self.kb = [({"agent": a, "action": v, "patient": p}, None) for (a, v, p) in facts]
        self.concepts = {"cat": [0.1, 0.2], "dog": [0.3, 0.4], "sat": [0.5, 0.6], "mat": [0.7, 0.8]}
        self.pol_words = set()

    def store(self, a, v, p, polarity=None):
        rec = {"agent": a, "action": v, "patient": p}
        if polarity is not None:
            rec["polarity"] = polarity
        self.kb.append((rec, None))


class _FakeAgent:
    def __init__(self, comp):
        self.composer = comp


class _FakeChat:
    def __init__(self, comp):
        self.agent = _FakeAgent(comp)


def test_facts_save_and_reload_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    # process 1: a session that acquired two facts at runtime
    chat1 = _FakeChat(_FakeComposer([("cat", "sat", "mat"), ("dog", "sat", "mat")]))
    p = CSP.save_session_facts(("sess", "brain", "qwen"), chat1, base=tmp_path)
    assert p is not None and p.exists()

    # process 2: a FRESH build with ZERO runtime facts -> reload must re-store both.
    chat2 = _FakeChat(_FakeComposer([]))
    added = CSP.reload_session_facts(("sess", "brain", "qwen"), chat2, base=tmp_path)
    assert added == 2
    stored = {(f["agent"], f["action"], f["patient"]) for f, _ in chat2.agent.composer.kb}
    assert ("cat", "sat", "mat") in stored and ("dog", "sat", "mat") in stored

    # idempotent: a second reload adds nothing (dedup against already-present facts).
    assert CSP.reload_session_facts(("sess", "brain", "qwen"), chat2, base=tmp_path) == 0


# ────────────────────────────────────────────────────────────────────────────────────────────
# PRODUCTION LATEST-CHECKPOINT UX (2026-09-02, layered on the mechanism above, same flag):
#   - successive saves ACCUMULATE as versioned checkpoints, not one overwritten file;
#   - a reload with no explicit selector loads the LATEST checkpoint automatically;
#   - `checkpoint=<id>` (or `BRAIN_PERSIST_CHECKPOINT=<id>`) pins an explicit OLDER one;
#   - retention (`BRAIN_PERSIST_KEEP_N`) prunes the oldest checkpoints beyond N so they don't grow unbounded.
# ────────────────────────────────────────────────────────────────────────────────────────────
def test_checkpoint_id_monotonic_and_lexicographically_sortable():
    a = CSP._new_checkpoint_id()
    b = CSP._new_checkpoint_id()
    assert a != b
    assert sorted([b, a]) == [a, b], "checkpoint ids must sort in arrival order (string sort == time order)"


def test_facts_save_accumulates_versioned_checkpoints_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")
    identity_dir = CSP._facts_identity_dir(cache_key, tmp_path)

    p1 = CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("cat", "sat", "mat")])), base=tmp_path)
    p2 = CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("dog", "ran", "park")])), base=tmp_path)

    assert p1 is not None and p2 is not None and p1 != p2, "each save must land in a NEW checkpoint path"
    assert len(CSP.list_checkpoints(identity_dir)) == 2, "two saves must accumulate as two versions"


def test_reload_default_selector_loads_latest_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("cat", "sat", "mat")])), base=tmp_path)
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("dog", "ran", "park")])), base=tmp_path)

    fresh = _FakeChat(_FakeComposer([]))
    added = CSP.reload_session_facts(cache_key, fresh, base=tmp_path)   # no selector -> "latest"
    assert added == 1
    stored = {(f["agent"], f["action"], f["patient"]) for f, _ in fresh.agent.composer.kb}
    assert ("dog", "ran", "park") in stored and ("cat", "sat", "mat") not in stored, (
        "a fresh boot with no override must load the NEWEST checkpoint, not the oldest")


def test_reload_explicit_checkpoint_arg_overrides_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")
    identity_dir = CSP._facts_identity_dir(cache_key, tmp_path)
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("cat", "sat", "mat")])), base=tmp_path)
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("dog", "ran", "park")])), base=tmp_path)
    first_id = CSP.list_checkpoints(identity_dir)[0]

    fresh = _FakeChat(_FakeComposer([]))
    added = CSP.reload_session_facts(cache_key, fresh, base=tmp_path, checkpoint=first_id)
    stored = {(f["agent"], f["action"], f["patient"]) for f, _ in fresh.agent.composer.kb}
    assert added == 1 and ("cat", "sat", "mat") in stored and ("dog", "ran", "park") not in stored


def test_reload_brain_persist_checkpoint_env_overrides_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")
    identity_dir = CSP._facts_identity_dir(cache_key, tmp_path)
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("cat", "sat", "mat")])), base=tmp_path)
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("dog", "ran", "park")])), base=tmp_path)
    first_id = CSP.list_checkpoints(identity_dir)[0]

    monkeypatch.setenv("BRAIN_PERSIST_CHECKPOINT", first_id)
    fresh = _FakeChat(_FakeComposer([]))
    added = CSP.reload_session_facts(cache_key, fresh, base=tmp_path)   # no explicit arg -> reads the env
    stored = {(f["agent"], f["action"], f["patient"]) for f, _ in fresh.agent.composer.kb}
    assert added == 1 and ("cat", "sat", "mat") in stored and ("dog", "ran", "park") not in stored


def test_reload_unknown_explicit_checkpoint_degrades_to_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")
    CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([("cat", "sat", "mat")])), base=tmp_path)
    fresh = _FakeChat(_FakeComposer([]))
    added = CSP.reload_session_facts(cache_key, fresh, base=tmp_path, checkpoint="not-a-real-checkpoint-id")
    assert added == 0, "an unknown explicit checkpoint must degrade to 'nothing loaded', never raise/corrupt"


def test_retention_prunes_oldest_checkpoints_beyond_keep_n(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    monkeypatch.setenv("BRAIN_PERSIST_KEEP_N", "2")
    cache_key = ("sess", "brain", "qwen")
    identity_dir = CSP._facts_identity_dir(cache_key, tmp_path)
    for w in ("a", "b", "c", "d"):
        CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([(w, "v", "p")])), base=tmp_path)
    assert len(CSP.list_checkpoints(identity_dir)) == 2, "retention must prune down to the newest KEEP_N"

    fresh = _FakeChat(_FakeComposer([]))
    CSP.reload_session_facts(cache_key, fresh, base=tmp_path)
    stored = {f["agent"] for f, _ in fresh.agent.composer.kb}
    assert "d" in stored, "the newest checkpoint (last save) must have survived pruning"


def test_retention_keep_n_zero_is_unlimited(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    monkeypatch.setenv("BRAIN_PERSIST_KEEP_N", "0")
    cache_key = ("sess", "brain", "qwen")
    identity_dir = CSP._facts_identity_dir(cache_key, tmp_path)
    for w in ("a", "b", "c"):
        CSP.save_session_facts(cache_key, _FakeChat(_FakeComposer([(w, "v", "p")])), base=tmp_path)
    assert len(CSP.list_checkpoints(identity_dir)) == 3, "KEEP_N<=0 must disable pruning (keep every version)"


def test_save_session_learning_shares_one_checkpoint_id_across_d5_and_facts(tmp_path, monkeypatch):
    """d5 + facts are both keyed by the session's cache_key, so a single `save_session_learning` call mints ONE
    checkpoint id and versions them together as one snapshot."""
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")

    class _Mem:
        def __init__(self):
            self.formed = {1}
            self.store_log = []
            self.bridge = type("B", (), {"cp_connections": _example([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])})()

    class _Organ:
        def __init__(self):
            self.mem = _Mem()
            self.seed = 42
            self.topics = ["cat", "dog"]
            self.sep_bias = 0.0
            self._store_order = []

    chat = _FakeChat(_FakeComposer([("cat", "sat", "mat")]))
    rec = CSP.save_session_learning(cache_key, chat, _Organ(), base=tmp_path)
    assert rec.get("checkpoint"), "the record must report which checkpoint was just written"
    assert rec["checkpoint"] in str(rec["d5"]) and rec["checkpoint"] in str(rec["facts"]), (
        "d5 and facts must land under the SAME checkpoint id from one save_session_learning call")


def test_d5_checkpoint_selection_versions_and_resolves(tmp_path, monkeypatch):
    """Fast unit coverage of the D5 checkpoint SELECTION surface (versioning + latest + explicit override); the
    full teach->save->fresh-substrate-reload round trip is `_selftest_d5` / `python -m
    research.runners.cross_session_persistence` (heavy: builds the real n_ca3=2000 GO organ)."""
    monkeypatch.setenv("BRAIN_PERSIST_LEARNING", "1")
    cache_key = ("sess", "brain", "qwen")

    class _Mem:
        def __init__(self, data):
            self.formed = {1}
            self.store_log = []
            self.bridge = type("B", (), {"cp_connections": _example(data)})()

    class _Organ:
        def __init__(self, data):
            self.mem = _Mem(data)
            self.seed = 42
            self.topics = ["cat", "dog"]
            self.sep_bias = 0.0
            self._store_order = []

    identity_dir = CSP._d5_identity_dir(cache_key, tmp_path)
    p1 = CSP.save_d5_organ(cache_key, _Organ([0.1] * 6), base=tmp_path)
    p2 = CSP.save_d5_organ(cache_key, _Organ([0.9] * 6), base=tmp_path)
    assert p1 != p2
    ids = CSP.list_checkpoints(identity_dir)
    assert len(ids) == 2

    assert CSP._resolve_checkpoint(identity_dir, None) == ids[-1], "default selector must resolve to the latest"
    assert CSP._resolve_checkpoint(identity_dir, ids[0]) == ids[0], "an explicit id must resolve to itself"
    assert CSP._resolve_checkpoint(identity_dir, "bogus-id") is None, "an unknown id must resolve to None"
