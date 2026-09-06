"""SHARDED FHRR fact-store (knowledge-scale de-risk, 2026-08-20): partition K facts across S independent
`RFPhasorComposer` shards so a routed query touches only ONE shard (~K/S facts) instead of scanning all K.

WHY. `RFPhasorComposer.query_patient` batch-unbinds the cue role from EVERY stored composite (a resonate over a
2*K*D-neuron RF bridge + a matched-filter cleanup over the codebook). Cost is O(K*D) (+ the cue-scan cleanup term
O(K*V*D)) -- linear in the number of stored facts. Measured: ~121 ms at K=60, ~5 s at K=2413, growing linearly.
At LLM-scale (1e5-1e6 facts) a single unrouted store is minutes/query. The REPRESENTATION already scales (each
fact is its own composite; recall is fact-count-independent; the no-confab abstain is architecture-preserved) --
the wall is purely the O(K) SCAN. Sharding removes it: route the cue to its shard, scan only that shard.

ROUTING (the key design choice). We route by the AGENT (subject) of the fact: `shard = hash(agent) mod S`. This
is concept-centric partitioning -- every fact ABOUT a subject lives in one shard, exactly the biological motif
(a cortical concept map / a hippocampal index pointing a cue at its stored trace; the hippocampus/cortex IS
partitioned). It makes every AGENT-CUED query O(1) route + O(K/S) scan with NO change in answer:
  * query_patient(agent, action)      -- agent known  -> one shard
  * ask_yes_no(agent, action, patient)-- agent known  -> one shard
  * render_fact(agent)                -- agent known  -> one shard
  * query_chain / chain_of_thought    -- pivot on the agent each hop -> one shard/hop
Because ALL of a subject's facts are co-located, first-match semantics within the shard == first-match over the
whole store for that subject: the routed answer is byte-identical to the unsharded answer. The one query whose
cue does NOT contain the agent is the REVERSE lookup query_agent(action, patient); it fans out to all shards
(embarrassingly parallel) or would use a second patient-routed index -- handled + costed honestly below.

MOAT. An unknown agent still hashes to SOME shard; that shard holds no matching fact -> returns None/'unknown'.
The no-confab abstain is preserved unchanged (it is the shard composer's own genuine read).

SHARED CODEBOOK. All shards are built at the SAME seed + vocab, so their concept/role phasor codebooks are
byte-identical; we then point every shard at ONE shared `concepts`/`roles`/`words` object so S shards cost ONE
codebook in memory, not S copies (a concept means the same thing in every shard -- a single global vocabulary).
Only the KB (the per-fact composites) is partitioned. Footprint = codebook(V*D) + facts(K*D), NOT S*V*D.

SCAFFOLD FLAG (brain-based-only). The router `hash(agent) mod S` is a HOST computation -- a declared test
scaffold for the capacity de-risk. The faithful version is a LEARNED/SPIKING router (a cue-driven attractor /
hippocampal index that maps a partial cue to the sub-population holding its trace); the recall + no-confab moat
inside each shard remain the genuine spiking/synaptic reads. This module changes NO production default and edits
NO sim/ code; it composes the validated RFPhasorComposer.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from research.runners.rf_phasor_composer import RFPhasorComposer, Clause, ROLES  # noqa: F401


def _stable_hash(s: str) -> int:
    """A deterministic, process-independent hash of a routing key (Python's built-in hash() is salted per
    process, which would move a subject to a different shard on every run -> store/query would disagree across
    restarts). BLAKE2b of the utf-8 key, first 8 bytes as an int."""
    h = hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


# KEY-ROUTING RESIDUAL (2026-08-27, research/findings/2026-08-27-ltm-exempt-production-flip-...md's honest
# residual): `_knowledge_core_curate.py`'s `pick_clean_alias` scores a MULTI-WORD, capitalized raw Wikidata alias
# above a single-word canonical name (`+2` for 2-4 words, `+1.5` for a capital first letter). Wikidata's
# crowd-sourced alias dump for several entities includes Wikipedia-namespace artifacts -- e.g. entity Q16
# (Canada) lists the raw alias "Canada portal" (from the "Portal:Canada" Wikipedia page) alongside the bare
# "Canada" -- and the 2-word capitalized "Canada portal" OUTSCORES the 1-word "Canada" under that rule, so it
# gets picked as the canonical token and sanitized to "canada_portal". Confirmed directly against
# `wikidata5m_entity.txt` (Q16's alias list contains both "Canada" and "Canada portal"). The shipped
# `wikidata_core_15k` bundle has 11 entities keyed this way (10x "_portal", 1x "_core": ballet, berlin,
# brandenburg, cambodia, canada, comic, dorset, lgbt, portugal, schleswig_holstein, ska) -- a user typing the
# bare name gets an honest-but-WRONG abstain (the store holds the fact, keyed differently) rather than a
# genuine "unknown" abstain. This is a curation-time picker bug; the deep fix is re-curating (rejecting a
# Wikipedia-namespace-derived alias the same way `_CRUFT` already rejects "wikiproject"/"template"/etc, then
# rebuilding the bundle -- ~13 CPU-min for 15k facts). The additive, retrieval-time mitigation below fixes the
# SAME-SESSION user-facing symptom without a rebuild or any change to already-shipped keys.
_KNOWN_KEY_SUFFIXES = ("_portal", "_core")


def _strip_known_suffix(key):
    """`'canada_portal' -> 'canada'`; `None` if `key` doesn't end in a known curation-artifact suffix."""
    if not isinstance(key, str):
        return None
    for suf in _KNOWN_KEY_SUFFIXES:
        if key.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)]
    return None


class ShardedPhasorStore:
    """S independent RFPhasorComposer shards with agent-hash routing + a shared codebook.

    Same conversational API as RFPhasorComposer (store / query_patient / query_agent / ask_yes_no / render_fact
    / query_chain / chain_of_thought), so it is a drop-in capacity upgrade behind the same reads.
    """

    def __init__(self, n_shards=64, seed=42, D=128, vocab=None, share_codebook=True, **composer_kwargs):
        self.n_shards = int(n_shards)
        self.seed = int(seed)
        self.D = int(D)
        self._composer_kwargs = dict(composer_kwargs)
        self._alias_index = None   # lazy-built bare-surface-form -> stored-key map (see build_alias_index)
        self._trace = False        # per-query match-trace recording gate (#184 fix); see the `trace` property
        self.last_trace = None     # the answering shard's OWN `last_trace`, captured after the last query
        # Build the shards. share_codebook: build the FULL {word:[D]} codebook ONCE (shard 0); the rest get a
        # 1-word vocab then GRAFT shard 0's codebook. Building all S shards with the full vocab FIRST allocated S
        # transient copies of a large codebook -> 500 shards x a 100k-word dict (~100 MB each) OOM-crashed a 46 GB
        # box (2026-08-21). Sharing the OBJECT is what the design always intended; do it DURING construction so peak
        # memory is ONE codebook, not S. Grafting the SAME dict/list references is safe (read/append only), and a
        # runtime-grown word (RFPhasorComposer._filler_phases) allocated in ANY shard stays visible to ALL.
        if share_codebook and self.n_shards > 0:
            base = RFPhasorComposer(seed=self.seed, D=self.D, vocab=vocab, **composer_kwargs)
            self.shards = [base]
            minimal = list(vocab[:1]) if vocab else None   # a 1-word codebook the graft immediately replaces
            for _ in range(self.n_shards - 1):
                sh = RFPhasorComposer(seed=self.seed, D=self.D, vocab=minimal, **composer_kwargs)
                sh.concepts = base.concepts
                sh.roles = base.roles
                sh.pol_words = base.pol_words
                sh.words = base.words          # SAME list object -> bisect.insort in one shard grows all
                sh._growth_rng = base._growth_rng
                # (#66 knowledge-scale) DG sparse-index GRAFT: a non-base shard delegates index-building to `base`
                # instead of lazily building its OWN DGSparseIndex over the identical shared codebook on first use
                # -- same memory rationale as the codebook graft above (S shards x an independent V-sized index
                # would multiply the RSS budget by S; grafting keeps it at ONE index for the whole store). Only
                # matters when `enable_sparse_index` is actually on (composer_kwargs / BRAIN_SHARD_SPARSE_INDEX);
                # a byte-identical no-op default-off (the attribute exists but `_ensure_dg_index` is never called).
                sh._dg_index_source = base
                self.shards.append(sh)
        else:
            # same seed+vocab -> byte-identical codebooks in every shard (no sharing = S independent copies).
            self.shards = [
                RFPhasorComposer(seed=self.seed, D=self.D, vocab=vocab, **composer_kwargs)
                for _ in range(self.n_shards)
            ]
        self._shared_codebook = bool(share_codebook)

    # --- match-trace instrumentation (#184 fix) -------------------------------------------------------------
    # WHY. Before this, `ShardedPhasorStore` emitted NO `last_trace` at all: each shard is its own independent
    # `RFPhasorComposer` with its OWN `.trace`/`.last_trace` (see that class's `_trace_scan`), and nothing here
    # ever armed a shard's flag or read its result back up to the store level -- so a query answered by the LTM
    # tier left metacog's confidence read (`webapp/server.py`'s `composer.last_trace`) with nothing to see,
    # even though the shard that answered had ALREADY computed a genuine per-role decode-confidence/margin (the
    # identical `_cleanup_all_score_stats` machinery the small conversation buffer uses). Fixed here by (a) a
    # `trace` property that arms every shard's own flag (any of them may be the one a routed query lands on),
    # and (b) `_note_trace` capturing whichever shard actually produced (or attempted) the answer, so
    # `self.last_trace` reports the SAME dict shape (`{"roles": [...], "abstained": ..., ...}`) the plain
    # unsharded composer already does -- no confidence is fabricated, it is the real match the LTM recall
    # already computed. `TieredFactStore._tiered` (research/runners/tiered_fact_store.py) then propagates this
    # up to where the metacog read actually looks.
    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, value):
        self._trace = bool(value)
        for sh in self.shards:
            sh.trace = self._trace

    def _note_trace(self, sh):
        """Capture shard `sh`'s own `last_trace` (whether it matched or abstained) as this store's `last_trace`.
        No-op unless tracing is armed (byte-identical to before when `.trace` is never set, e.g. every caller
        that doesn't opt in -- the default)."""
        if self._trace:
            self.last_trace = getattr(sh, "last_trace", None)

    # --- routing -------------------------------------------------------------------------------------------
    def route(self, agent) -> int:
        """The shard holding (all facts about) `agent`. HOST scaffold: hash(agent) mod S. Deterministic across
        processes so store and query agree. (Faithful version: a learned/spiking cue->sub-population router.)"""
        return _stable_hash(agent) % self.n_shards

    def shard_for(self, agent) -> RFPhasorComposer:
        return self.shards[self.route(agent)]

    # --- key-routing alias fallback (additive; see the module-level comment above _KNOWN_KEY_SUFFIXES) -------
    def build_alias_index(self, force=False):
        """Lazily build (cache) a `bare_surface_form -> stored_key` map, e.g. `'canada' -> 'canada_portal'`, by
        scanning every fact currently in the store for agent/patient values ending in a known curation-artifact
        suffix. MOAT-SAFE by construction:
          * a bare form is mapped ONLY when it is not ITSELF already a stored key (never shadows a real,
            distinctly-keyed entity -- the direct lookup for a genuinely-existing bare-keyed entity always wins
            because callers try the literal key FIRST and only consult this index on a miss);
          * a bare form is mapped ONLY when the suffix-strip is UNAMBIGUOUS (exactly one distinct stored key
            strips to it) -- an ambiguous bare form (two different suffixed entities colliding on the same bare
            name) resolves to NOTHING rather than guessing, so this can never manufacture a wrong answer, only
            recover a right one that was reachable under a different string.
        A genuinely nonexistent entity's bare form is never a suffix-stripped form of anything stored, so it is
        simply absent from this index -- the abstain path is untouched. Cached after the first call (the LTM is
        a static bulk store, never runtime-grown -- see `save`/`load`'s own docstring); pass `force=True` to
        rebuild after a write (bulk `.store()` calls between server restarts are not expected on the LTM tier)."""
        if self._alias_index is not None and not force:
            return self._alias_index
        all_keys = set()
        for sh in self.shards:
            for fact, _handle in sh.kb:
                for role in ("agent", "patient"):
                    v = fact.get(role)
                    if isinstance(v, str):
                        all_keys.add(v)
        candidates = {}   # bare -> set of stored keys that strip to it
        for k in all_keys:
            bare = _strip_known_suffix(k)
            if bare:
                candidates.setdefault(bare, set()).add(k)
        index = {}
        for bare, keys in candidates.items():
            if bare in all_keys:
                continue          # never shadow a real, distinctly-keyed entity
            if len(keys) == 1:
                index[bare] = next(iter(keys))
            # len(keys) > 1: ambiguous -> deliberately left unresolved (moat safety over recall)
        self._alias_index = index
        return index

    def _resolve_alias(self, key):
        """`None` if `key` has no known-suffix alias in the store (the common case -- byte-identical no-op)."""
        return self.build_alias_index().get(key)

    # --- write ---------------------------------------------------------------------------------------------
    def store(self, agent, action, patient, polarity=None):
        """Route by agent -> store in that one shard. A clause/attributed patient is passed through unchanged
        (RFPhasorComposer handles the fact-dict shape); routing only ever reads the top-level agent."""
        self.shard_for(agent).store(agent, action, patient, polarity=polarity)
        self._alias_index = None   # a new fact may change the index; invalidate the cache (rebuilt lazily)

    # --- agent-cued reads (O(1) route + one-shard scan; answer-identical to the unsharded store) -----------
    def query_patient(self, agent, action, order_fn=None):
        sh = self.shard_for(agent)
        ans = sh.query_patient(agent, action, order_fn=order_fn)
        self._note_trace(sh)
        if ans is None:
            alias = self._resolve_alias(agent)
            if alias is not None:
                sh2 = self.shard_for(alias)
                ans = sh2.query_patient(alias, action, order_fn=order_fn)
                self._note_trace(sh2)
        return ans

    def ask_yes_no(self, agent, action, patient):
        sh = self.shard_for(agent)
        ans = sh.ask_yes_no(agent, action, patient)
        self._note_trace(sh)
        if ans == "unknown":
            alias_a = self._resolve_alias(agent)
            alias_p = self._resolve_alias(patient)
            if alias_a is not None or alias_p is not None:
                a2 = alias_a if alias_a is not None else agent
                p2 = alias_p if alias_p is not None else patient
                sh2 = self.shard_for(a2)
                ans = sh2.ask_yes_no(a2, action, p2)
                self._note_trace(sh2)
        return ans

    def render_fact(self, agent, order_fn=None):
        sh = self.shard_for(agent)
        ans = sh.render_fact(agent, order_fn=order_fn)
        self._note_trace(sh)
        if ans is None:
            alias = self._resolve_alias(agent)
            if alias is not None:
                sh2 = self.shard_for(alias)
                ans = sh2.render_fact(alias, order_fn=order_fn)
                self._note_trace(sh2)
        return ans

    def query_chain(self, cue, actions):
        """Multi-hop: each hop pivots on the CURRENT concept as agent, so each hop routes to that concept's
        shard. Abstains (None) the moment any hop has no matching fact (the moat holds at every hop)."""
        x = cue
        for action in actions:
            x = self.query_patient(x, action)
            if x is None:
                return None
        return x

    def chain_of_thought(self, start, goal=None, max_hops=4, lesion=None, lesion_rng=None, return_path=False):
        """Self-cued associative chain-of-thought across shards. The relation-selection statistic
        (_relation_assoc) is LOCAL to a shard -- and since all of a concept's facts are co-located in that
        concept's shard, the concept's available relations are exactly those in its shard, so the per-hop
        self-cue is complete without a cross-shard scan. Each hop then routes to the next concept's shard."""
        x = start
        path = [x]
        terminal = None
        for _ in range(int(max_hops)):
            sh = self.shard_for(x)
            assoc = sh._relation_assoc()
            rel = sh._select_next_relation(x, assoc, lesion=lesion, lesion_rng=lesion_rng)
            if rel is None:
                break
            nxt = self.query_patient(x, rel)
            if nxt is None:
                break
            path.append(nxt)
            x = nxt
            terminal = x
            if goal is not None and x == goal:
                break
        return (terminal, path) if return_path else terminal

    # --- reverse lookup (cue lacks the agent) --------------------------------------------------------------
    def query_agent(self, action, patient, shard_query_fn=None):
        """'who <action> <patient>?' -- the AGENT is unknown, so agent-hash routing cannot pick a shard. Fan
        out to ALL shards and return the first hit (embarrassingly parallel: each shard's scan is independent,
        so wall-clock is one-shard latency with S-way parallelism; here run serially for a correctness/latency
        baseline). HONEST COST: reverse queries do NOT get the routing speedup unless a SECOND patient-routed
        index is maintained (2x write cost + 2x footprint) -- see the finding. `shard_query_fn` lets a caller
        inject a parallel map (e.g. a process/thread/GPU-batched fan-out)."""
        if shard_query_fn is not None:
            for ans in shard_query_fn(self.shards, action, patient):
                if ans is not None:
                    return ans
            return None
        for sh in self.shards:
            if sh.kb:
                ans = sh.query_agent(action, patient)
                self._note_trace(sh)
                if ans is not None:
                    return ans
        # key-routing alias fallback: the cue patient may be a bare surface form of a suffix-keyed entity
        # (e.g. 'canada' when the store holds 'canada_portal' as the patient) -- retry once with the resolved
        # key, only ever reached after every shard has already genuinely missed on the literal cue.
        alias_p = self._resolve_alias(patient)
        if alias_p is not None:
            for sh in self.shards:
                if sh.kb:
                    ans = sh.query_agent(action, alias_p)
                    self._note_trace(sh)
                    if ans is not None:
                        return ans
        return None

    # --- introspection -------------------------------------------------------------------------------------
    def total_facts(self) -> int:
        return sum(len(sh.kb) for sh in self.shards)

    def shard_sizes(self):
        return [len(sh.kb) for sh in self.shards]

    def load_balance(self):
        """Return (min, max, mean, max/mean) shard occupancy -- the routing balance. max/mean near 1.0 means
        the hash spread facts evenly; a large ratio means a hot subject (many facts under one agent)."""
        sizes = self.shard_sizes()
        if not sizes:
            return (0, 0, 0.0, 0.0)
        mean = sum(sizes) / len(sizes)
        return (min(sizes), max(sizes), mean, (max(sizes) / mean if mean else 0.0))

    # --- persistence (build ONCE, reload FAST) -------------------------------------------------------------
    # A 100k-fact LTM is ~30 min to BUILD (one RF resonate per fact at store time), but should reload in seconds
    # for a live chat server. Mirroring developed_brain_io's kb_composites speedup: save each fact's already-bound
    # COMPOSITE, and on load reconstruct the shards (the codebook regenerates byte-identically from seed+vocab --
    # the LTM is a bulk store that is never runtime-grown, so no growth diverges the codes) and set kb DIRECTLY,
    # skipping the resonate.
    #
    # SUBSTRATE-STORE FIX (2026-09-05, rank-6 pickle bug): with `enable_substrate_store=True` a shard's `kb`
    # holds a live `SimulationBridge` HANDLE per fact (`RFPhasorComposer._store_substrate`), not a numpy array.
    # The original code did `np.asarray(handle)` on that bridge object; numpy silently accepts this (it makes a
    # 0-d OBJECT array wrapping the bridge instance instead of raising), so the failure surfaced two steps
    # later, at `np.savez`'s pickling of the object array: `TypeError: cannot pickle 'mappingproxy' object` --
    # `SimulationBridge` sets several instance attributes (`snr_packet_bindings`/`snr_packet_kernel_parameters`/
    # `snr_packet_hh_phi`, see `sim/bridge.py`) to `types.MappingProxyType(...)`, and a mappingproxy has no
    # pickle support. Serializing the WHOLE bridge object graph was never the intent -- the actual per-fact
    # information is the D-dim composite phase vector the bridge's synaptic weights carry (identical in kind to
    # the numpy-kb path's plain array); the bridge is just WHERE that vector lives when the substrate store is
    # on. The fix reads the vector back out with the composer's own `_retrieve_substrate` (the same call every
    # substrate-store query already uses) before it ever reaches numpy, so `composites.npz` keeps holding plain
    # real-valued phase arrays -- byte-identical on-disk shape/dtype to the numpy-kb path, whether or not the
    # substrate store produced them. `load()`'s mirror-image fix rebuilds the substrate handle from that same
    # vector via `_store_substrate` (the identical call `store()` makes on first write) -- deterministic given
    # the manifest's seed, so the reloaded bridge is structurally the one `store()` would have built directly.
    # No `sim/` edit; no change to the numpy-kb on-disk format.
    def save(self, path):
        """Persist to `path/` (manifest.json + facts.json + composites.npz) so `ShardedPhasorStore.load(path)`
        reloads WITHOUT the per-fact RF resonate. Returns the number of facts saved."""
        import json
        import os
        import numpy as np
        os.makedirs(path, exist_ok=True)
        base = self.shards[0]
        manifest = {
            "n_shards": self.n_shards, "seed": self.seed, "D": self.D,
            "share_codebook": self._shared_codebook, "composer_kwargs": self._composer_kwargs,
            "vocab": list(base.words), "n_facts": self.total_facts(),
        }
        facts = []            # [{shard, fact}] preserving the store's shard placement
        comp_arrays = {}
        for i, sh in enumerate(self.shards):
            comps = []
            for fact, handle in sh.kb:
                facts.append({"shard": i, "fact": fact})
                # A substrate handle is a live bridge object (unpicklable -- see the note above); pull the
                # composite phase vector back out. A numpy-kb handle already IS that vector.
                comp = sh._retrieve_substrate(handle) if sh.enable_substrate_store else handle
                comps.append(np.asarray(comp))
            if comps:
                comp_arrays[f"sh{i}"] = np.stack(comps)
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        with open(os.path.join(path, "facts.json"), "w") as f:
            json.dump(facts, f)
        np.savez(os.path.join(path, "composites.npz"), **comp_arrays)
        return manifest["n_facts"]

    @classmethod
    def load(cls, path, extra_kwargs=None):
        """Reconstruct a store saved by `save()` WITHOUT re-resonating (codebook regenerates from seed+vocab; each
        fact's composite is set directly into its shard's kb). `extra_kwargs`: optional dict of additional
        composer kwargs to merge with the manifest's `composer_kwargs` (e.g. `{"enable_codebook_cache": True}`
        to activate the codebook-cache opt-in on a previously-saved bundle whose manifest doesn't record it)."""
        import json
        import os
        import numpy as np
        with open(os.path.join(path, "manifest.json")) as f:
            m = json.load(f)
        store_kwargs = dict(m.get("composer_kwargs") or {})
        if extra_kwargs:
            store_kwargs.update(extra_kwargs)
        store = cls(n_shards=m["n_shards"], seed=m["seed"], D=m["D"], vocab=m["vocab"],
                    share_codebook=m.get("share_codebook", True), **store_kwargs)
        with open(os.path.join(path, "facts.json")) as f:
            facts = json.load(f)
        comps = np.load(os.path.join(path, "composites.npz"))
        # Read each shard's composite array ONCE (npz access re-decompresses a NEW array every call), and store each
        # fact's composite as a COPY of its row -- a bare `arr[j]` VIEW pins the whole shard array alive, so
        # re-reading per-fact + keeping the view retained ~one full-shard array PER FACT (94k x ~200 KB ~= 18 GB,
        # the 2026-08-21 load OOM). One read per shard + a 1-KB row copy per fact -> peak ~a few hundred MB.
        shard_arrays = {}
        per_shard_idx = {}
        for rec in facts:
            i = int(rec["shard"])
            arr = shard_arrays.get(i)
            if arr is None:
                arr = comps[f"sh{i}"]
                shard_arrays[i] = arr
            j = per_shard_idx.get(i, 0)
            sh = store.shards[i]
            comp = np.array(arr[j])
            # Mirror `store()`'s own store-path choice (`RFPhasorComposer.store`): under the substrate store,
            # `kb` holds a bridge HANDLE, not the raw composite -- rebuild it deterministically (same seed) via
            # the composer's own `_store_substrate`, exactly the call the original write made. `sh.
            # enable_substrate_store` comes from the manifest's `composer_kwargs`, so it matches how this bundle
            # was actually built (see the fix note above `save()`).
            handle = sh._store_substrate(comp) if sh.enable_substrate_store else comp
            sh.kb.append((rec["fact"], handle))
            per_shard_idx[i] = j + 1
        return store
