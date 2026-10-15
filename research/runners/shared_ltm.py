"""SharedLTMView -- one process-shared, read-only cortical LTM behind every conversation (STEP 1 of the owner-approved
2026-10-13 "one brain, many conversations" memory plan).

WHY. `webapp/server.py` `_build_chat_brain` runs once per conversation and used to load its OWN copy of the imported
knowledge store (wikidata_100k: 395 shards, 78,857 facts, one substrate bridge per fact, ~5.44 GiB). The store is an
imprinted prior that no chat path writes (docs/drafts/shared-ltm/WORKLOG.md, step 1), so one copy per process serves
every conversation. This module holds that copy and gives each conversation a VIEW of it.

WHAT IS PER CONVERSATION. Only the trace instrumentation: the store's `trace` gate and `last_trace`, and every
shard's `last_trace`. A per-conversation store kept these from that conversation's own previous calls (`render_fact`
and a dead-end `chain_of_thought` read a shard's or the store's previous trace without resetting it). The view keeps
them per conversation and swaps them in and out around each call, so each conversation sees exactly what its own
store would have shown.

WHAT IS SHARED AND WHY IT IS SAFE. The facts (kb), the codebook, the lazy caches (codebook cache, alias index, the
consolidated retrieve bridge) and the per-op RF scratch bridges. Every RF op replaces its weights and `rf_kick`
resets the whole RF state, so an op is a pure function of its inputs and a warm bridge answers like a fresh one.
The stateful read paths (the Izhikevich WTA bank behind spiking cleanup / the spiking recall margin, and the
read-noise RNG) are NOT shown pure, so a store with any of them on is refused (`unshareable_reason`) and the caller
falls back to a per-conversation copy.

GLOBAL RNG (found in review, 2026-10-13). Building a bridge reseeds the process-global np.random / random / cp.random
streams (sim/bridge.py `_initialize_rng`), so a COLD lazy build inside the store had a side effect outside it, and on a
shared store whether a recall builds cold depends on other conversations. The server therefore loads a shared store
under `global_rng_guard.preserved_global_rng()` and sets every shard's `_isolate_global_rng`, so no load or lazy
build touches the global streams (`webapp/server.py` `_load_ltm_isolated`; BRAIN_LTM_RNG_ISOLATE, forced on with
sharing).

DEFAULT. On in the server since 2026-10-15 (`_SHARED_LTM_DEFAULT_ON = True`; BRAIN_SHARED_LTM=0 is the escape),
after the flip battery's Amendment A passed A1 and A2 on 6/6 gate seeds (research/coordination/
2026-10-13-shared-ltm-flip-JOBS.md).

CONCURRENCY. `/api/brain-chat` runs in FastAPI's threadpool and the continuous-state tick runs in an executor, so
two conversations can query at once. One RLock per shared store serializes every call made through a view (the
shard scratch bridges and `_scan_comps_cache` are single-user). Plain reads of `shard.kb` (the chain-route,
multistep and elaboration readers) need no lock: kb never changes.

READ-ONLY, LOUDLY. Writes through a view (`store`, `update_on_mismatch`, any attribute other than the trace pair)
raise `SharedLTMWriteError`: a write would leak into every conversation. No chat path writes (WORKLOG step 1).

Engineering of a read-only prior; NO sim/ edit; the store, its recall and the no-confab moat are unchanged.
"""
from __future__ import annotations

import threading

# The process cache the server owns is a plain dict passed in (so tests can use their own). Locking is per key:
# `_KEYS_GUARD` is held only for dict lookups (never across a load), and each key has its own load lock, so "load
# once" holds for two conversations building the same key at the same moment while a cached key never waits behind
# an unrelated key's load.
_KEYS_GUARD = threading.Lock()
_LOAD_LOCKS = {}   # (id(cache), key) -> threading.Lock

class SharedLTMWriteError(RuntimeError):
    """A write was attempted on the process-shared, read-only LTM."""


def unshareable_reason(store):
    """None if `store` may be shared across conversations, else the reason it may not. Refuses the read paths whose
    purity across calls is not shown: spiking cleanup and the spiking recall margin (an Izhikevich bank stepped by
    the full simulation step, only v/u reset per read) and the read-noise RNG (`_retrieve_noise > 0`)."""
    shards = getattr(store, "shards", None)
    if not shards:
        return "store has no shards (not a ShardedPhasorStore)"
    for sh in shards:
        if getattr(sh, "enable_spiking_cleanup", False):
            return "enable_spiking_cleanup is on (stateful Izhikevich WTA bank)"
        if getattr(sh, "spiking_recall_margin", False):
            return "spiking_recall_margin is on (stateful Izhikevich margin bank)"
        if float(getattr(sh, "_retrieve_noise", 0.0) or 0.0) > 0.0:
            return "_retrieve_noise > 0 (stateful read-noise RNG)"
    return None


class _Entry:
    __slots__ = ("store", "lock")

    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()


def view_for(cache, key, loader, log=None):
    """The LTM one new conversation should wrap.

    `cache`: the process dict (key -> entry). `loader()`: loads a fresh store, or returns None (no LTM). On a cache
    miss the store is loaded ONCE (under that key's own load lock, so a concurrent second conversation on the same
    key waits instead of loading a second copy; other keys are not blocked) and kept for the process lifetime.
    Returns a new `SharedLTMView` over the cached store; None when the loader has no LTM; or, when the loaded store is
    not shareable (`unshareable_reason`), that store itself, uncached: a per-conversation copy, as before this
    module."""
    with _KEYS_GUARD:
        ent = cache.get(key)
        if ent is None:
            load_lock = _LOAD_LOCKS.setdefault((id(cache), key), threading.Lock())
    if ent is None:
        with load_lock:
            with _KEYS_GUARD:
                ent = cache.get(key)          # another conversation may have finished the load while we waited
            if ent is None:
                # Loads of DIFFERENT keys may run here at once; the server's loader runs under
                # global_rng_guard.preserved_global_rng(), whose module RLock serializes the RNG-touching part, so
                # two loads cannot interleave their save/restore of the process-global RNG streams.
                store = loader()
                if store is None:
                    return None
                reason = unshareable_reason(store)
                if reason is not None:
                    if log is not None:
                        log("[shared-ltm] not sharing this LTM (per-conversation copy): %s" % reason)
                    return store
                ent = _Entry(store)
                with _KEYS_GUARD:
                    cache[key] = ent
                if log is not None:
                    n = store.total_facts() if hasattr(store, "total_facts") else "?"
                    log("[shared-ltm] loaded the LTM once for this process: %s facts, key=%r" % (n, key))
    return SharedLTMView(ent.store, ent.lock)


class SharedLTMView:
    """One conversation's view of a process-shared `ShardedPhasorStore`.

    Same read API as the store (`query_patient` / `query_agent` / `ask_yes_no` / `render_fact` / `query_chain` /
    `chain_of_thought` / `_resolve_alias` / `build_alias_index`), each run under the store's lock with this
    conversation's trace state swapped in. `trace` and `last_trace` are this view's own. Every other attribute
    (`shards`, `shard_for`, `n_shards`, `total_facts`, ...) reads through to the shared store."""

    def __init__(self, store, lock):
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_lock", lock)
        object.__setattr__(self, "_trace", False)        # a fresh store's trace gate
        object.__setattr__(self, "last_trace", None)     # a fresh store's last_trace
        object.__setattr__(self, "_shard_traces", {})    # shard index -> that shard's last_trace (non-None only)

    @property
    def shared_store(self):
        """The ONE process-shared store this view reads (identical object across conversations)."""
        return object.__getattribute__(self, "_store")

    @property
    def trace(self):
        return object.__getattribute__(self, "_trace")

    @trace.setter
    def trace(self, value):
        object.__setattr__(self, "_trace", bool(value))

    def __setattr__(self, name, value):
        if name in ("trace", "last_trace"):
            object.__setattr__(self, name, value)
            return
        raise SharedLTMWriteError("the shared LTM is read-only: refusing to set %r (it would reach every "
                                  "conversation)" % (name,))

    # -- the per-conversation trace swap -------------------------------------------------------------------------
    def _enter(self, st):
        st.trace = self._trace                 # the store's own setter arms every shard, as on a private store
        st.last_trace = self.last_trace
        saved = self._shard_traces
        for i, sh in enumerate(st.shards):
            sh.last_trace = saved.get(i)

    def _exit(self, st):
        object.__setattr__(self, "last_trace", st.last_trace)
        traces = {}
        for i, sh in enumerate(st.shards):
            t = sh.last_trace
            if t is not None:
                traces[i] = t
                sh.last_trace = None
        object.__setattr__(self, "_shard_traces", traces)
        st.last_trace = None
        st.trace = False                       # leave the shared object in its fresh state between calls

    def _call(self, name, args, kwargs):
        st = object.__getattribute__(self, "_store")
        with object.__getattribute__(self, "_lock"):
            self._enter(st)
            try:
                return getattr(st, name)(*args, **kwargs)
            finally:
                self._exit(st)

    # -- reads (locked, trace-swapped) ----------------------------------------------------------------------------
    def query_patient(self, *args, **kwargs):
        return self._call("query_patient", args, kwargs)

    def query_agent(self, *args, **kwargs):
        return self._call("query_agent", args, kwargs)

    def ask_yes_no(self, *args, **kwargs):
        return self._call("ask_yes_no", args, kwargs)

    def render_fact(self, *args, **kwargs):
        return self._call("render_fact", args, kwargs)

    def query_chain(self, *args, **kwargs):
        return self._call("query_chain", args, kwargs)

    def chain_of_thought(self, *args, **kwargs):
        return self._call("chain_of_thought", args, kwargs)

    def build_alias_index(self, *args, **kwargs):
        return self._call("build_alias_index", args, kwargs)

    def _resolve_alias(self, *args, **kwargs):
        return self._call("_resolve_alias", args, kwargs)

    def save(self, *args, **kwargs):
        # save() fires each substrate bridge (`_retrieve_substrate`) to read its composite: serialize it too.
        return self._call("save", args, kwargs)

    # -- writes: refused ------------------------------------------------------------------------------------------
    def store(self, *args, **kwargs):
        raise SharedLTMWriteError("the shared LTM is read-only: store() would add a fact to every conversation")

    def update_on_mismatch(self, *args, **kwargs):
        raise SharedLTMWriteError("the shared LTM is read-only: update_on_mismatch() would rewrite a shared fact")

    # -- copies share the store (copying it would re-create the 5 GiB per conversation this module removes) -------
    def __copy__(self):
        v = SharedLTMView(self.shared_store, object.__getattribute__(self, "_lock"))
        object.__setattr__(v, "_trace", self._trace)
        object.__setattr__(v, "last_trace", self.last_trace)
        object.__setattr__(v, "_shard_traces", dict(self._shard_traces))
        return v

    def __deepcopy__(self, memo):
        return self.__copy__()

    # -- everything else reads through to the shared store --------------------------------------------------------
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_store"), name)
