"""Gate for the process-shared, read-only cortical LTM (one brain, many conversations, step 1; 2026-10-13).

The failure it guards (research/FAILURE_LOG.md 2026-10-13, "per-conversation 5 GiB LTM copies"): every conversation
the webapp built loaded its own ~5.44 GiB copy of the imported knowledge store, with no limit. Now one store is
loaded per process and each conversation's TieredFactStore wraps a `SharedLTMView` of it.

Brain-free. What each test pins:
  * two conversations get the IDENTICAL store object (helper level and through `_build_chat_brain`, both brain paths);
  * BRAIN_SHARED_LTM=0 gives each conversation its own object, through the exact pre-change loader call;
  * the cache key includes every flag that changes the loaded store;
  * a recall through the views leaves the store's arrays, codebook and RNG state unchanged (mutation guard);
  * per-conversation answers AND traces through views over one store equal those of private stores, interleaved,
    warm caches included (the per-conversation trace state is really per conversation);
  * concurrent recall through two views stays correct (the lock); writes through a view fail loudly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import types

import numpy as np
import pytest

from research.runners.shared_ltm import SharedLTMView, SharedLTMWriteError, unshareable_reason, view_for
from research.runners.sharded_phasor_store import ShardedPhasorStore
from research.runners.tiered_fact_store import TieredFactStore

VOCAB = ["dog", "go", "north", "cat", "run", "south", "river", "look", "apple", "bird", "eat", "seed", "sky", "blue"]
FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple"), ("bird", "eat", "seed"),
         ("north", "look", "sky")]


def _small_store():
    """A real ShardedPhasorStore with the production LTM flags: substrate store + batched scan + codebook cache +
    decode escalation (what `_load_or_build_ltm_store` passes for wikidata_100k)."""
    s = ShardedPhasorStore(n_shards=3, seed=42, D=64, period=200, vocab=list(VOCAB), enable_substrate_store=True,
                           enable_batched_substrate_scan=True, enable_codebook_cache=True,
                           enable_decode_escalation=True)
    for a, v, p in FACTS:
        s.store(a, v, p, polarity="AFFIRM")
    return s


def _views(store, n=2):
    lock = threading.RLock()
    return [SharedLTMView(store, lock) for _ in range(n)]


# A conversation's LTM reads, in the shapes TieredFactStore / the chat path issue them (hits, misses, the moat, a
# render_fact that reads a shard's previous trace, a chain that dead-ends before any hop).
def _script(tag):
    if tag == "A":
        return [("query_patient", ("dog", "go")), ("render_fact", ("cat",)), ("ask_yes_no", ("cat", "run", "south")),
                ("query_patient", ("elephant", "go")), ("chain_of_thought", ("dog",)), ("render_fact", ("dog",)),
                ("query_agent", ("eat", "seed")), ("chain_of_thought", ("apple",)), ("query_chain", ("dog", ["go", "look"]))]
    return [("ask_yes_no", ("dog", "go", "south")), ("query_patient", ("river", "look")), ("render_fact", ("dog",)),
            ("chain_of_thought", ("zebra",)), ("query_agent", ("run", "south")), ("render_fact", ("river",)),
            ("ask_yes_no", ("unicorn", "eat", "seed")), ("query_patient", ("bird", "eat"))]


def _run(target, op):
    name, args = op
    return getattr(target, name)(*args)


# ── the views: equivalence with private stores, mutation guard, lock, write guard ───────────────────────────────
def test_shared_views_answer_and_trace_exactly_like_private_stores():
    """Interleave two conversations' reads on views over ONE store and on two private stores. Every answer and every
    `last_trace` (decoded chips, matched index, RF gauge floats) must be equal, op by op -- so the shared store's
    warm scratch bridges and caches answer like fresh ones, and the trace state stays per conversation."""
    shared = _small_store()
    va, vb = _views(shared)
    pa, pb = _small_store(), _small_store()
    for t in (va, vb, pa, pb):
        t.trace = True                     # the server arms it every turn (TieredFactStore forwards it to the LTM)
    sa, sb = _script("A"), _script("B")
    for i in range(max(len(sa), len(sb))):
        for view, priv, script in ((va, pa, sa), (vb, pb, sb)):
            if i < len(script):
                got, want = _run(view, script[i]), _run(priv, script[i])
                assert got == want, (script[i], got, want)
                assert view.last_trace == priv.last_trace, script[i]
    assert va.last_trace is not vb.last_trace


def test_recall_through_views_does_not_mutate_the_shared_store():
    """THE MUTATION GUARD. Fingerprint every array the store holds (each fact's substrate weights, the codebook,
    the roles, the vocab list) and its RNG state; run every read the chat path can issue, unknown words included
    (a query must never grow the vocabulary); the fingerprint must not change."""
    store = _small_store()

    def fingerprint():
        h = hashlib.sha256()
        for sh in store.shards:
            for fact, handle in sh.kb:
                h.update(json.dumps(fact, sort_keys=True).encode())
                for arr in (handle.cp_rf_w_re.toarray(), handle.cp_rf_w_im.toarray()):
                    h.update(np.ascontiguousarray(np.asarray(arr)).tobytes())
            h.update(json.dumps(sorted(sh.words)).encode())
            for w in sorted(sh.concepts):
                h.update(w.encode())
                h.update(np.ascontiguousarray(np.asarray(sh.concepts[w])).tobytes())
            for r in sorted(sh.roles):
                h.update(np.ascontiguousarray(np.asarray(sh.roles[r])).tobytes())
            h.update(repr(sh._growth_rng.bit_generator.state).encode())
            h.update(repr(sh._retrieve_noise_rng.bit_generator.state).encode())
        return h.hexdigest(), store.total_facts(), len(store.shards[0].words)

    before = fingerprint()
    va, vb = _views(store)
    for v in (va, vb):
        v.trace = True
        for op in _script("A") + _script("B"):
            _run(v, op)
        v._resolve_alias("dog")
    assert fingerprint() == before


def test_concurrent_recall_through_two_views_stays_correct():
    """Two threads, one per conversation, query the same store at once. With the per-store lock every answer is
    right and each view's trace names its own conversation's fact."""
    store = _small_store()
    va, vb = _views(store)
    errors = []

    def worker(view, agent, action, want, n=6):
        view.trace = True
        for _ in range(n):
            got = view.query_patient(agent, action)
            src = (view.last_trace or {}).get("source_fact", {})
            if got != want or src.get("agent") != agent:
                errors.append((agent, got, src))

    ts = [threading.Thread(target=worker, args=(va, "dog", "go", "north")),
          threading.Thread(target=worker, args=(vb, "river", "look", "apple"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []


def test_writes_through_a_view_fail_loudly():
    store = _small_store()
    (v,) = _views(store, 1)
    with pytest.raises(SharedLTMWriteError):
        v.store("emu", "eat", "seed")
    with pytest.raises(SharedLTMWriteError):
        v.update_on_mismatch("dog", "go", "south")
    with pytest.raises(SharedLTMWriteError):
        v.kb = []
    tiered = TieredFactStore(types.SimpleNamespace(kb=[({"agent": "emu", "action": "eat", "patient": "seed"}, None)]),
                             v)
    with pytest.raises(SharedLTMWriteError):
        tiered.promote_buffer_to_ltm()
    assert store.total_facts() == len(FACTS)


def test_tiered_trace_flag_reaches_the_view_not_the_shared_store():
    store = _small_store()
    (v,) = _views(store, 1)
    tiered = TieredFactStore(types.SimpleNamespace(trace=False, last_trace=None), v)
    tiered.trace = True
    assert v.trace is True
    assert store.trace is False and not any(sh.trace for sh in store.shards)


def test_copies_of_a_view_share_the_store():
    store = _small_store()
    (v,) = _views(store, 1)
    assert copy.deepcopy(v).shared_store is store
    assert copy.copy(v).shared_store is store


def test_unshareable_store_is_returned_uncached():
    """A store with a read path not shown pure across calls (here the read-noise RNG) is not shared: the caller gets
    that store itself, as a per-conversation copy, and nothing is cached."""
    store = _small_store()
    assert unshareable_reason(store) is None
    for sh in store.shards:
        sh._retrieve_noise = 0.1
    assert unshareable_reason(store) is not None
    cache = {}
    assert view_for(cache, ("k",), lambda: store) is store
    assert cache == {}


# ── the server: one store per key, the escape, the key ──────────────────────────────────────────────────────────
class _FakeStore:
    def __init__(self, tag):
        self.tag = tag
        self.shards = [types.SimpleNamespace()]

    def total_facts(self):
        return 0


@pytest.fixture()
def S(monkeypatch, tmp_path):
    from webapp import server as srv
    monkeypatch.setattr(srv, "_SHARED_LTM", {})
    for k in ("BRAIN_SHARED_LTM", "BRAIN_LTM_CODEBOOK_CACHE", "BRAIN_LTM_DECODE_ESCALATION",
              "BRAIN_LTM_SUBSTRATE_STORE", "BRAIN_BATCHED_SUBSTRATE_SCAN", "BRAIN_SHARD_SPARSE_INDEX",
              "BRAIN_METACOG_SPIKING_MARGIN"):
        monkeypatch.delenv(k, raising=False)
    calls = []

    def fake_load(ltm_bundle, seed=42, n_shards=None, D=128, enable_codebook_cache=None,
                  enable_decode_escalation=None):
        calls.append(dict(ltm_bundle=ltm_bundle, seed=seed, n_shards=n_shards, D=D,
                          enable_codebook_cache=enable_codebook_cache,
                          enable_decode_escalation=enable_decode_escalation))
        return _FakeStore(len(calls))

    monkeypatch.setattr(srv, "_load_or_build_ltm_store", fake_load)
    srv._test_calls = calls
    srv._test_bundle = str(tmp_path / "bundle")
    (tmp_path / "bundle").mkdir()
    return srv


def test_two_conversations_get_the_identical_store_object(S):
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert isinstance(a, SharedLTMView) and isinstance(b, SharedLTMView)
    assert a is not b                                  # each conversation has its own view (its own trace state)
    assert a.shared_store is b.shared_store            # ... over the ONE store
    assert len(S._test_calls) == 1                     # loaded once


def test_escape_flag_gives_each_conversation_its_own_store(S, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert isinstance(a, _FakeStore) and isinstance(b, _FakeStore) and a is not b
    # the pre-change call, argument for argument: _load_or_build_ltm_store(bundle, seed=_brain_chat_seed())
    assert S._test_calls == [dict(ltm_bundle=S._test_bundle, seed=7, n_shards=None, D=128,
                                  enable_codebook_cache=None, enable_decode_escalation=None)] * 2
    assert S._SHARED_LTM == {}


@pytest.mark.parametrize("val,on", [(None, True), ("", True), ("1", True), ("on", True), ("0", False),
                                    ("off", False), ("false", False), ("no", False)])
def test_shared_ltm_flag_parsing(S, monkeypatch, val, on):
    if val is not None:
        monkeypatch.setenv("BRAIN_SHARED_LTM", val)
    assert S._shared_ltm_on() is on


def test_cache_key_includes_every_flag(S, monkeypatch, tmp_path):
    base_args = (S._test_bundle, 7, 128, None, True, True)
    base = S._shared_ltm_key(*base_args)
    other = tmp_path / "other"
    other.mkdir()
    variants = [(str(other), 7, 128, None, True, True), (S._test_bundle, 8, 128, None, True, True),
                (S._test_bundle, 7, 64, None, True, True), (S._test_bundle, 7, 128, 16, True, True),
                (S._test_bundle, 7, 128, None, False, True), (S._test_bundle, 7, 128, None, True, False)]
    keys = [S._shared_ltm_key(*v) for v in variants]
    for env, val in (("BRAIN_LTM_SUBSTRATE_STORE", "0"), ("BRAIN_BATCHED_SUBSTRATE_SCAN", "0"),
                     ("BRAIN_SHARD_SPARSE_INDEX", "1"), ("BRAIN_METACOG_SPIKING_MARGIN", "1")):
        monkeypatch.setenv(env, val)
        keys.append(S._shared_ltm_key(*base_args))
        monkeypatch.delenv(env)
    assert all(k != base for k in keys)
    assert len(set(keys)) == len(keys)
    assert S._shared_ltm_key(*base_args) == base       # restored env -> the same key again
    # the same bundle through a symlink is the same store
    link = tmp_path / "link"
    link.symlink_to(S._test_bundle)
    assert S._shared_ltm_key(str(link), 7, 128, None, True, True) == base


def test_a_changed_flag_between_conversations_loads_a_second_store(S, monkeypatch):
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    monkeypatch.setenv("BRAIN_LTM_CODEBOOK_CACHE", "0")
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert a.shared_store is not b.shared_store
    assert [c["enable_codebook_cache"] for c in S._test_calls] == [True, False]


def _fake_tiny_demo(seed, use_multiturn=True, enable_neural_render=False, composer_kind=None, integrated_loop=False):
    agent = types.SimpleNamespace(composer=types.SimpleNamespace(kb=[], trace=False, last_trace=None), seed=seed)
    return agent, set(), 0


def test_build_chat_brain_tiny_demo_conversations_share_one_store(S, monkeypatch):
    """Through the real `_build_chat_brain` (tiny-demo branch; the brain build itself stubbed): two conversations'
    TieredFactStores wrap views of the SAME store; the escape gives two stores."""
    import research.runners.brain_chat_tui as tui
    monkeypatch.setattr(tui, "_build_tiny_demo", _fake_tiny_demo)
    monkeypatch.setattr(S, "_resolve_ltm_bundle", lambda: S._test_bundle)
    c1, src1 = S._build_chat_brain("tiny-demo", "stub")
    c2, src2 = S._build_chat_brain("tiny-demo", "stub")
    t1, t2 = c1.inner.composer, c2.inner.composer
    assert isinstance(t1, TieredFactStore) and src1 == src2 == "tiny-demo +LTM"
    assert t1.buffer is not t2.buffer
    assert t1.ltm.shared_store is t2.ltm.shared_store
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    c3, _ = S._build_chat_brain("tiny-demo", "stub")
    c4, _ = S._build_chat_brain("tiny-demo", "stub")
    assert isinstance(c3.inner.composer.ltm, _FakeStore)
    assert c3.inner.composer.ltm is not c4.inner.composer.ltm
    assert c3.inner.composer.ltm is not t1.ltm.shared_store


def test_build_chat_brain_developed_bundle_shares_the_same_cache(S, monkeypatch, tmp_path):
    """The developed-bundle branch hands the shared view in as `ltm_store` (and no `ltm_bundle`); the escape passes
    `ltm_bundle` and no store, the pre-change call. The build-fallback seed is the bundle's own brain.json seed."""
    import research.runners.developed_brain_io as dbio
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "brain.json").write_text(json.dumps({"seed": 11}))
    seen = []

    def fake_load_developed(path, **kw):
        seen.append(kw)
        return types.SimpleNamespace(composer=None), {"self_aliases": []}

    monkeypatch.setattr(dbio, "load_developed_brain", fake_load_developed)
    monkeypatch.setattr(S, "_resolve_ltm_bundle", lambda: S._test_bundle)
    S._build_chat_brain(str(dev), "stub")
    S._build_chat_brain(str(dev), "stub")
    assert seen[0]["ltm_bundle"] is None and seen[1]["ltm_bundle"] is None
    assert seen[0]["ltm_store"].shared_store is seen[1]["ltm_store"].shared_store
    assert S._test_calls[0]["seed"] == 11 and len(S._test_calls) == 1
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    S._build_chat_brain(str(dev), "stub")
    assert seen[2]["ltm_bundle"] == S._test_bundle and seen[2]["ltm_store"] is None
