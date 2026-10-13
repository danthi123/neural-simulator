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
from sim.backend import to_host
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
                    h.update(np.ascontiguousarray(to_host(arr)).tobytes())   # cupy backend: device -> host
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


def test_cached_key_never_waits_behind_another_keys_load():
    """Per-key load locks: while key A's load is in progress, a conversation on an already-cached key B gets its view
    at once, and a second conversation on A waits for A's single load instead of loading a second copy."""
    cache = {}
    store_b = _FakeStore("b")
    view_for(cache, ("B",), lambda: store_b)
    gate, loads = threading.Event(), []

    def slow_loader():
        loads.append(1)
        gate.wait(10)
        return _FakeStore("a")

    t1 = threading.Thread(target=view_for, args=(cache, ("A",), slow_loader))
    t1.start()
    got = []
    t2 = threading.Thread(target=lambda: got.append(view_for(cache, ("B",), lambda: _FakeStore("never"))))
    t2.start()
    t2.join(2.0)
    assert got and got[0].shared_store is store_b      # not blocked by A's load
    t3 = threading.Thread(target=view_for, args=(cache, ("A",), slow_loader))
    t3.start()
    gate.set()
    t1.join(5.0)
    t3.join(5.0)
    assert len(loads) == 1 and cache[("A",)].store.tag == "a"


# ── global-RNG hygiene: a recall must not reseed the process-global streams ─────────────────────────────────────
def _np_state():
    st = np.random.get_state()
    return st[1].copy(), st[2], st[3], st[4]


def _same_np_state(a, b):
    return np.array_equal(a[0], b[0]) and a[1:] == b[1:]


def test_preserved_global_rng_restores_on_raise_and_releases_the_lock():
    from research.runners import global_rng_guard as G
    np.random.seed(3)
    before = _np_state()
    with pytest.raises(ValueError):
        with G.preserved_global_rng():
            np.random.seed(42)
            raise ValueError("boom")
    assert _same_np_state(before, _np_state())
    assert G._GUARD_LOCK.acquire(blocking=False)       # released after the raise
    G._GUARD_LOCK.release()


def test_preserved_global_rng_nests_on_one_thread():
    from research.runners.global_rng_guard import preserved_global_rng
    np.random.seed(4)
    before = _np_state()
    with preserved_global_rng():
        np.random.seed(42)
        inner_before = _np_state()
        with preserved_global_rng():                  # re-entrant: no deadlock
            np.random.seed(43)
        assert _same_np_state(inner_before, _np_state())
    assert _same_np_state(before, _np_state())


def test_preserved_global_rng_serializes_two_threads():
    """A enters, reseeds, holds 0.15 s; B starts 0.05 s later, reseeds, holds 0.3 s. Unserialized, B would save A's
    reseeded state and restore it last, leaving the process on the seed-42 stream. With the guard's lock B waits for
    A, and the process ends on its original stream."""
    import time
    from research.runners.global_rng_guard import preserved_global_rng
    np.random.seed(5)
    before = _np_state()

    def block(seed, hold):
        with preserved_global_rng():
            np.random.seed(seed)
            time.sleep(hold)
            np.random.rand()

    ta = threading.Thread(target=block, args=(42, 0.15))
    tb = threading.Thread(target=block, args=(43, 0.3))
    ta.start()
    time.sleep(0.05)
    tb.start()
    ta.join(5)
    tb.join(5)
    assert _same_np_state(before, _np_state())


def test_preserved_global_rng_keeps_the_cupy_state_object():
    """The cupy branch: cupy.random.seed reseeds the CURRENT RandomState in place, so the guard swaps a temporary one
    in and puts the original object back. Skipped without cupy or a CUDA device."""
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
        cupy.random.get_random_state()
    except Exception as e:                            # noqa: BLE001 -- no usable device
        pytest.skip("no CUDA device: %s" % e)
    from research.runners import global_rng_guard as G
    orig_backend = G._backend_name
    G._backend_name = lambda: "cupy"
    try:
        cupy.random.seed(123)
        want = float(cupy.random.rand())
        cupy.random.seed(123)
        obj = cupy.random.get_random_state()
        with G.preserved_global_rng():
            cupy.random.seed(42)
            cupy.random.rand(3)
        assert cupy.random.get_random_state() is obj
        assert float(cupy.random.rand()) == want
    finally:
        G._backend_name = orig_backend


def test_preserved_global_rng_restores_numpy_and_python_streams():
    import random
    from research.runners.global_rng_guard import preserved_global_rng
    np.random.seed(123)
    random.seed(123)
    before = (np.random.get_state()[1].copy(), np.random.get_state()[2], random.getstate())
    with preserved_global_rng():
        np.random.seed(42)
        random.seed(42)
        np.random.rand(5)
    after = (np.random.get_state()[1], np.random.get_state()[2], random.getstate())
    assert np.array_equal(before[0], after[0]) and before[1] == after[1] and before[2] == after[2]


def test_cold_ltm_recall_reseeds_the_global_rng_unless_isolated():
    """THE DEFECT, then the fix. A cold recall on a shard builds its RF bridges, and each build reseeds np.random to
    seed 42 (sim/bridge.py `_initialize_rng`). With the shard's `_isolate_global_rng` set, the global stream after the
    recall is exactly the one before it."""
    leaky = _small_store()
    np.random.seed(7)
    want = np.random.rand()
    np.random.seed(7)
    leaky.query_patient("cat", "run")              # cold: builds bridges
    leaked = np.random.rand()
    assert leaked != want                          # the pre-existing reseed (the defect this branch isolates)
    clean = _small_store()
    for sh in clean.shards:
        sh._isolate_global_rng = True
    np.random.seed(7)
    assert clean.query_patient("cat", "run") == leaky.query_patient("cat", "run") == "south"
    assert np.random.rand() == want                # isolated: the recall left the global stream alone


def test_shared_warm_and_private_cold_recall_leave_the_same_global_rng():
    """The reviewer's synthetic check. Same prior global state; a recall through a view whose shard another
    conversation already warmed, vs the same recall on a private cold store. Without isolation the next
    `np.random.rand()` differs (the cold build reseeds, the warm one does not); with isolation both equal the draw
    with no recall at all."""
    def run(isolate):
        shared, private = _small_store(), _small_store()
        for st in (shared, private):
            for sh in st.shards:
                sh._isolate_global_rng = isolate
        va, vb = _views(shared)
        va.query_patient("cat", "run")             # conversation A warms the shard
        np.random.seed(11)
        vb.query_patient("cat", "run")             # conversation B: warm
        on = np.random.rand()
        np.random.seed(11)
        private.query_patient("cat", "run")        # a private store: cold
        off = np.random.rand()
        return on, off
    on, off = run(False)
    assert on != off
    np.random.seed(11)
    baseline = np.random.rand()
    on, off = run(True)
    assert on == off == baseline


# ── the server: flags, one store per key, the escape, the key ───────────────────────────────────────────────────
class _FakeStore:
    def __init__(self, tag):
        self.tag = tag
        self.shards = [types.SimpleNamespace()]

    def total_facts(self):
        return 0


_ENV = ("BRAIN_SHARED_LTM", "BRAIN_LTM_RNG_ISOLATE", "BRAIN_LTM_CODEBOOK_CACHE", "BRAIN_LTM_DECODE_ESCALATION",
        "BRAIN_LTM_SUBSTRATE_STORE", "BRAIN_BATCHED_SUBSTRATE_SCAN", "BRAIN_SHARD_SPARSE_INDEX",
        "BRAIN_METACOG_SPIKING_MARGIN")


@pytest.fixture()
def S(monkeypatch, tmp_path):
    from webapp import server as srv
    monkeypatch.setattr(srv, "_SHARED_LTM", {})
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    calls = []

    def fake_load(ltm_bundle, seed=42, n_shards=None, D=128, enable_codebook_cache=None,
                  enable_decode_escalation=None):
        calls.append(dict(ltm_bundle=ltm_bundle, seed=seed, n_shards=n_shards, D=D,
                          enable_codebook_cache=enable_codebook_cache,
                          enable_decode_escalation=enable_decode_escalation))
        np.random.seed(42)                         # what a real load does to the global stream
        return _FakeStore(len(calls))

    monkeypatch.setattr(srv, "_load_or_build_ltm_store", fake_load)
    srv._test_calls = calls
    srv._test_bundle = str(tmp_path / "bundle")
    (tmp_path / "bundle").mkdir()
    return srv


def test_default_is_off_and_makes_the_pre_change_call(S):
    """DEFAULT OFF (design section 5: default-off, then flip): unset env -> no sharing, no RNG isolation, and each
    conversation loads its own store through the pre-change call, argument for argument."""
    assert S._SHARED_LTM_DEFAULT_ON is False and S._LTM_RNG_ISOLATE_DEFAULT_ON is False
    assert S._shared_ltm_on() is False and S._ltm_rng_isolate_on() is False
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert isinstance(a, _FakeStore) and isinstance(b, _FakeStore) and a is not b
    assert S._test_calls == [dict(ltm_bundle=S._test_bundle, seed=7, n_shards=None, D=128,
                                  enable_codebook_cache=None, enable_decode_escalation=None)] * 2
    assert not hasattr(a.shards[0], "_isolate_global_rng")
    assert S._SHARED_LTM == {}


def test_two_conversations_get_the_identical_store_object(S, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "1")
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert isinstance(a, SharedLTMView) and isinstance(b, SharedLTMView)
    assert a is not b                                  # each conversation has its own view (its own trace state)
    assert a.shared_store is b.shared_store            # ... over the ONE store
    assert len(S._test_calls) == 1                     # loaded once
    assert a.shared_store.shards[0]._isolate_global_rng is True   # sharing forces RNG isolation


def test_escape_flag_gives_each_conversation_its_own_store(S, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert isinstance(a, _FakeStore) and isinstance(b, _FakeStore) and a is not b
    assert S._test_calls == [dict(ltm_bundle=S._test_bundle, seed=7, n_shards=None, D=128,
                                  enable_codebook_cache=None, enable_decode_escalation=None)] * 2
    assert S._SHARED_LTM == {}


def test_shared_or_isolated_load_leaves_the_global_rng_alone(S, monkeypatch):
    """A load reseeds np.random (the fake loader does what the real bridge init does). Plain OFF lets it through
    (today's behaviour); OFF+BRAIN_LTM_RNG_ISOLATE=1 and the shared store do not, and flag every shard."""
    np.random.seed(5)
    want = np.random.rand()
    np.random.seed(5)
    S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert np.random.rand() != want                    # plain OFF: the load reseeded (unchanged)
    for env in ({"BRAIN_LTM_RNG_ISOLATE": "1"}, {"BRAIN_SHARED_LTM": "1"}):
        S._SHARED_LTM.clear()
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        np.random.seed(5)
        st = S._ltm_store_for_conversation(S._test_bundle, seed=7)
        assert np.random.rand() == want, env
        assert getattr(st, "shared_store", st).shards[0]._isolate_global_rng is True
        for k in env:
            monkeypatch.delenv(k)


@pytest.mark.parametrize("val,on", [(None, False), ("", False), ("1", True), ("on", True), ("0", False),
                                    ("off", False), ("false", False), ("no", False)])
def test_shared_ltm_flag_parsing(S, monkeypatch, val, on):
    if val is not None:
        monkeypatch.setenv("BRAIN_SHARED_LTM", val)
    assert S._shared_ltm_on() is on
    assert S._ltm_rng_isolate_on() is on               # forced on with sharing; otherwise its own default (off)


def test_rng_isolate_flag_parsing(S, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    for val, on in ((None, False), ("1", True), ("yes", True), ("0", False)):
        if val is None:
            monkeypatch.delenv("BRAIN_LTM_RNG_ISOLATE", raising=False)
        else:
            monkeypatch.setenv("BRAIN_LTM_RNG_ISOLATE", val)
        assert S._ltm_rng_isolate_on() is on, val
    monkeypatch.setenv("BRAIN_SHARED_LTM", "1")
    monkeypatch.setenv("BRAIN_LTM_RNG_ISOLATE", "0")
    assert S._ltm_rng_isolate_on() is True             # cannot be turned off while sharing


_ENV_CALL = r"""(?:environ\.get|getenv|environ\.setdefault|environ\.pop)\(|environ\["""


def _scan_env_reads(src, label="src"):
    """(literal env names read in `src`, the env reads whose name is NOT a string literal -- the scan cannot see
    which variable those read, so the cache key cannot be shown to cover them)."""
    import re
    names, opaque = set(), []
    for m in re.finditer(_ENV_CALL, src):
        rest = src[m.end():m.end() + 80].lstrip()
        lit = re.match(r"""['"]([A-Za-z0-9_]+)['"]""", rest)
        if lit:
            names.add(lit.group(1))
        else:
            opaque.append("%s: %s" % (label, src[m.start():m.end() + 40].splitlines()[0]))
    return names, opaque


def _module_env_reads():
    """Every env read anywhere in rf_phasor_composer.py and sharded_phasor_store.py (whole files: an env read in a
    helper, a method or the module body counts, not only the constructors)."""
    import research.runners.rf_phasor_composer as rf_mod
    import research.runners.sharded_phasor_store as sh_mod
    names, opaque = set(), []
    for mod in (rf_mod, sh_mod):
        n, o = _scan_env_reads(open(mod.__file__, encoding="utf-8").read(), mod.__name__)
        names |= n
        opaque += o
    return names, opaque


def test_cache_key_covers_every_env_read_in_the_store_modules(S):
    names, opaque = _module_env_reads()
    assert names, "the scan found no env reads -- the pattern no longer matches the modules"
    assert not opaque, "env reads with a non-literal name (the key cannot cover them): %s" % opaque
    missing = names - set(S._SHARED_LTM_COMPOSER_ENV)
    assert not missing, "env reads missing from _SHARED_LTM_COMPOSER_ENV: %s" % sorted(missing)


def test_env_scan_catches_helper_and_non_literal_reads():
    """The scan's own discriminating check: a read inside a helper is found, and a non-literal name is opaque."""
    src = ('def _flag(name):\n    return os.environ.get(name, "")\n'
           'def _h():\n    return os.getenv( "BRAIN_X")\nY = os.environ["BRAIN_Y"]\n')
    names, opaque = _scan_env_reads(src)
    assert names == {"BRAIN_X", "BRAIN_Y"}
    assert len(opaque) == 1 and "environ.get(" in opaque[0]


def test_cache_key_includes_every_flag(S, monkeypatch, tmp_path):
    """A facts-only bundle (built at load): seed, D, n_shards, the facts.json stamp, every LTM flag and every
    constructor env value are in the key."""
    (tmp_path / "bundle" / "facts.json").write_text("[]")
    base_args = (S._test_bundle, 7, 128, None, True, True)
    base = S._shared_ltm_key(*base_args)
    other = tmp_path / "other"
    other.mkdir()
    variants = [(str(other), 7, 128, None, True, True), (S._test_bundle, 8, 128, None, True, True),
                (S._test_bundle, 7, 64, None, True, True), (S._test_bundle, 7, 128, 16, True, True),
                (S._test_bundle, 7, 128, None, False, True), (S._test_bundle, 7, 128, None, True, False)]
    keys = [S._shared_ltm_key(*v) for v in variants]
    for env, val in [("BRAIN_LTM_SUBSTRATE_STORE", "0"), ("BRAIN_BATCHED_SUBSTRATE_SCAN", "0")] + \
            [(k, "1") for k in S._SHARED_LTM_COMPOSER_ENV if k != "BRAIN_BATCHED_SUBSTRATE_SCAN"]:
        monkeypatch.setenv(env, val)
        keys.append(S._shared_ltm_key(*base_args))
        monkeypatch.delenv(env)
    (tmp_path / "bundle" / "facts.json").write_text("[1]")          # a rebuilt facts file
    keys.append(S._shared_ltm_key(*base_args))
    assert all(k != base for k in keys)
    assert len(set(keys)) == len(keys)
    link = tmp_path / "link"
    link.symlink_to(S._test_bundle)                   # the same bundle through a symlink is the same store
    assert S._shared_ltm_key(str(link), 7, 128, None, True, True) == S._shared_ltm_key(*base_args)


def test_persisted_bundle_key_is_the_files_not_the_callers_seed(S, tmp_path):
    """A persisted bundle (manifest with n_shards) loads seed / D / n_shards from its own manifest, so the key leaves
    the caller's out (tiny-demo at BRAIN_CHAT_SEED and a developed bundle at its own seed share one copy) and
    carries the files' (mtime, size) stamps instead (a rebuilt bundle is a new key)."""
    b = tmp_path / "bundle"
    (b / "manifest.json").write_text(json.dumps({"n_shards": 2, "seed": 42, "D": 128}))
    (b / "facts.json").write_text("[]")
    k7 = S._shared_ltm_key(str(b), 7, 128, None, True, True)
    assert S._shared_ltm_key(str(b), 11, 64, 9, True, True) == k7
    assert S._shared_ltm_key(str(b), 7, 128, None, False, True) != k7
    (b / "manifest.json").write_text(json.dumps({"n_shards": 3, "seed": 42, "D": 128}))
    assert S._shared_ltm_key(str(b), 7, 128, None, True, True) != k7


def test_a_changed_flag_between_conversations_loads_a_second_store(S, monkeypatch):
    monkeypatch.setenv("BRAIN_SHARED_LTM", "1")
    a = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    monkeypatch.setenv("BRAIN_LTM_CODEBOOK_CACHE", "0")
    b = S._ltm_store_for_conversation(S._test_bundle, seed=7)
    assert a.shared_store is not b.shared_store
    assert [c["enable_codebook_cache"] for c in S._test_calls] == [True, False]


def _fake_tiny_demo(seed, use_multiturn=True, enable_neural_render=False, composer_kind=None, integrated_loop=False):
    agent = types.SimpleNamespace(composer=types.SimpleNamespace(kb=[], trace=False, last_trace=None), seed=seed)
    return agent, set(), 0


def test_build_chat_brain_tiny_demo_conversations_share_one_store(S, monkeypatch):
    """Through the real `_build_chat_brain` (tiny-demo branch; the brain build itself stubbed): with BRAIN_SHARED_LTM=1
    two conversations' TieredFactStores wrap views of the SAME store; the default gives two stores."""
    import research.runners.brain_chat_tui as tui
    monkeypatch.setattr(tui, "_build_tiny_demo", _fake_tiny_demo)
    monkeypatch.setattr(S, "_resolve_ltm_bundle", lambda: S._test_bundle)
    monkeypatch.setenv("BRAIN_SHARED_LTM", "1")
    c1, src1 = S._build_chat_brain("tiny-demo", "stub")
    c2, src2 = S._build_chat_brain("tiny-demo", "stub")
    t1, t2 = c1.inner.composer, c2.inner.composer
    assert isinstance(t1, TieredFactStore) and src1 == src2 == "tiny-demo +LTM"
    assert t1.buffer is not t2.buffer
    assert t1.ltm.shared_store is t2.ltm.shared_store
    monkeypatch.delenv("BRAIN_SHARED_LTM")
    c3, _ = S._build_chat_brain("tiny-demo", "stub")
    c4, _ = S._build_chat_brain("tiny-demo", "stub")
    assert isinstance(c3.inner.composer.ltm, _FakeStore)
    assert c3.inner.composer.ltm is not c4.inner.composer.ltm
    assert c3.inner.composer.ltm is not t1.ltm.shared_store


def test_build_chat_brain_developed_bundle_shares_the_same_cache(S, monkeypatch, tmp_path):
    """The developed-bundle branch: shared on -> the shared view as `ltm_store` (no `ltm_bundle`); OFF+isolate -> a
    private RNG-isolated store as `ltm_store`; the default -> `ltm_bundle` and no store, the pre-change call. The
    build-fallback seed is the bundle's own brain.json seed."""
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
    monkeypatch.setenv("BRAIN_SHARED_LTM", "1")
    S._build_chat_brain(str(dev), "stub")
    S._build_chat_brain(str(dev), "stub")
    assert seen[0]["ltm_bundle"] is None and seen[1]["ltm_bundle"] is None
    assert seen[0]["ltm_store"].shared_store is seen[1]["ltm_store"].shared_store
    assert S._test_calls[0]["seed"] == 11 and len(S._test_calls) == 1
    monkeypatch.setenv("BRAIN_SHARED_LTM", "0")
    monkeypatch.setenv("BRAIN_LTM_RNG_ISOLATE", "1")
    S._build_chat_brain(str(dev), "stub")
    assert seen[2]["ltm_bundle"] is None and isinstance(seen[2]["ltm_store"], _FakeStore)
    assert seen[2]["ltm_store"].shards[0]._isolate_global_rng is True
    monkeypatch.delenv("BRAIN_SHARED_LTM")
    monkeypatch.delenv("BRAIN_LTM_RNG_ISOLATE")
    S._build_chat_brain(str(dev), "stub")
    assert seen[3]["ltm_bundle"] == S._test_bundle and seen[3]["ltm_store"] is None


def test_load_developed_brain_installs_a_given_ltm_store(tmp_path):
    """The REAL `ltm_store=` branch of `load_developed_brain` on a small saved developed bundle (rf composer): the
    composer becomes a TieredFactStore over the given view, an LTM-only fact is recalled through the agent, the
    bundle's own fact still comes from the buffer, and an untaught cue still abstains."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.developed_brain_io import save_developed_brain, load_developed_brain, _inner_agent
    vocab = ["dog", "go", "north", "cat", "eat", "fish"]
    a = BrainConversationalAgent(seed=42, concepts={w: None for w in vocab}, composer_kind="rf",
                                 enable_neural_render=False)
    a.composer.kb = []
    a.composer.store("dog", "go", "north", polarity="AFFIRM")
    bundle = str(tmp_path / "brain")
    save_developed_brain(a, bundle, seed=42, composer_kind="rf")
    (view,) = _views(_small_store(), 1)
    agent, _m = load_developed_brain(bundle, composer_kind="rf", ltm_store=view)
    comp = _inner_agent(agent).composer
    assert isinstance(comp, TieredFactStore) and comp.ltm is view
    assert agent.what_does("dog", "go") == "north"           # the bundle's own fact (buffer)
    assert agent.what_does("river", "look") == "apple"       # an LTM-only fact, through the given store
    assert agent.what_does("cat", "eat") is None              # the moat
