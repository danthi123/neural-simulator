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


def _fact_to_store_args(fact):
    """Reconstruct the (agent, action, patient, polarity) `store()` arguments from a composer's stored fact-dict
    (the shape `RFPhasorComposer.store`/`OneBrainComposer.store` write). Handles a plain-string patient, an
    attributed patient ('big red apple' -> (adjs, noun)), a recursive Clause filler (passed through unchanged),
    and a bound AFFIRM/NEGATE polarity. Used to RE-HOME a source composer's facts into the shards (re-encoded with
    the SHARED codebook -> byte-identical composite)."""
    agent = fact.get("agent")
    action = fact.get("action")
    polarity = fact.get("polarity")
    if "attribute" in fact:
        adjs = [fact["attribute"]]
        if "attribute2" in fact:
            adjs.append(fact["attribute2"])
        patient = (adjs, fact.get("patient"))
    else:
        patient = fact.get("patient")   # a plain string OR a Clause namedtuple -> store() handles both
    return agent, action, patient, polarity


class ShardedPhasorStore:
    """S independent RFPhasorComposer shards with agent-hash routing + a shared codebook.

    Same conversational API as RFPhasorComposer (store / query_patient / query_agent / ask_yes_no / render_fact
    / query_chain / chain_of_thought), so it is a drop-in capacity upgrade behind the same reads.
    """

    def __init__(self, n_shards=64, seed=42, D=128, vocab=None, share_codebook=True,
                 composer_factory=None, **composer_kwargs):
        self.n_shards = int(n_shards)
        self.seed = int(seed)
        self.D = int(D)
        self._composer_kwargs = dict(composer_kwargs)
        # composer_factory (default RFPhasorComposer -- the de-risk is UNCHANGED): the per-shard recall engine. It
        # stays the genuine FHRR/RF composer; only the hash(agent) router is a host scaffold. Kept a parameter so the
        # live-integration path can build shards whose class + codebook match the production composer (see
        # `from_existing_composer`).
        self._composer_factory = composer_factory or RFPhasorComposer
        # Build the shards. Same seed+vocab -> byte-identical codebooks in every shard.
        self.shards = [
            self._composer_factory(seed=self.seed, D=self.D, vocab=vocab, **composer_kwargs)
            for _ in range(self.n_shards)
        ]
        # Point every shard at ONE shared codebook object (memory: 1 codebook, not S). The codebooks are already
        # value-identical (same seed); sharing the OBJECT means a runtime-grown word (see RFPhasorComposer.
        # _filler_phases) allocated in ANY shard is visible to ALL shards -> a concept stays global. Grafting the
        # SAME dict/list references is safe because the composer only ever reads or appends to them.
        if share_codebook and self.n_shards > 0:
            base = self.shards[0]
            for sh in self.shards[1:]:
                sh.concepts = base.concepts
                sh.roles = base.roles
                sh.pol_words = base.pol_words
                sh.words = base.words          # SAME list object -> bisect.insort in one shard grows all
                sh._growth_rng = base._growth_rng
        self._shared_codebook = bool(share_codebook)

    # --- routing -------------------------------------------------------------------------------------------
    def route(self, agent) -> int:
        """The shard holding (all facts about) `agent`. HOST scaffold: hash(agent) mod S. Deterministic across
        processes so store and query agree. (Faithful version: a learned/spiking cue->sub-population router.)"""
        return _stable_hash(agent) % self.n_shards

    def shard_for(self, agent) -> RFPhasorComposer:
        return self.shards[self.route(agent)]

    # --- write ---------------------------------------------------------------------------------------------
    def store(self, agent, action, patient, polarity=None):
        """Route by agent -> store in that one shard. A clause/attributed patient is passed through unchanged
        (RFPhasorComposer handles the fact-dict shape); routing only ever reads the top-level agent."""
        self.shard_for(agent).store(agent, action, patient, polarity=polarity)

    # --- agent-cued reads (O(1) route + one-shard scan; answer-identical to the unsharded store) -----------
    def query_patient(self, agent, action, order_fn=None):
        return self.shard_for(agent).query_patient(agent, action, order_fn=order_fn)

    def ask_yes_no(self, agent, action, patient):
        return self.shard_for(agent).ask_yes_no(agent, action, patient)

    def render_fact(self, agent, order_fn=None):
        return self.shard_for(agent).render_fact(agent, order_fn=order_fn)

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

    # --- drop-in composer surface (read-only) --------------------------------------------------------------
    # The live chat treats its composer as a duck: it reads `.kb` (ChatBrain._refresh_facts, agent.elaborate),
    # `.words` (agent vocab), and the recall/store methods above. Expose those so a ShardedPhasorStore is a drop-in
    # replacement for the single-store composer. `.kb` AGGREGATES every shard's kb (read-only concatenation, in
    # shard order); nothing in the live path mutates `composer.kb` directly (writes go through `store()`), so a
    # computed aggregate is safe.
    @property
    def kb(self):
        out = []
        for sh in self.shards:
            out.extend(sh.kb)
        return out

    def _base(self):
        return self.shards[0] if self.shards else None

    @property
    def words(self):
        b = self._base()
        return getattr(b, "words", []) if b is not None else []

    @property
    def concepts(self):
        b = self._base()
        return getattr(b, "concepts", {}) if b is not None else {}

    @property
    def roles(self):
        b = self._base()
        return getattr(b, "roles", {}) if b is not None else {}

    @property
    def pol_words(self):
        b = self._base()
        return getattr(b, "pol_words", []) if b is not None else []

    # --- live-integration constructor ----------------------------------------------------------------------
    @classmethod
    def from_existing_composer(cls, composer, n_shards=16, composer_factory=None, share_codebook=True):
        """Build a sharded store that is BYTE-IDENTICAL to `composer` for every agent-cued read, by (a) SHARING the
        source composer's codebook object -- so a concept/role/word (incl. any grounded-code override or
        runtime-grown word) means the exact same phasor in every shard -- and (b) RE-HOMING each stored fact into
        the shard its AGENT routes to. First-match WITHIN an agent's shard == first-match over the source store for
        that agent (agent co-location), so the routed answer is byte-identical. Only `hash(agent) mod S` is a host
        scaffold; the in-shard FHRR recall + no-confab moat are the genuine reads.

        Works for a bare `RFPhasorComposer` OR a `OneBrainComposer` (whose codebook lives on its inner `.comp` and
        whose `kb` carries the fact-dicts). Shards default to `RFPhasorComposer` (the de-risked, scale-capable
        numpy fast-path); because RFPhasorComposer is the validated numpy ORACLE the spiking OneBrainComposer is
        itself byte-identical to, routing a OneBrain source's facts through RF shards preserves the source's recall
        answer (verify before flipping) while removing the O(K) scan wall. Pass `composer_factory=type(composer)`
        to shard the exact production class instead (faithful but S-fold build/VRAM cost)."""
        cb = getattr(composer, "comp", composer)   # OneBrainComposer wraps an inner RFPhasorComposer as `.comp`
        D = int(getattr(cb, "D", getattr(composer, "D", 128)))
        seed = int(getattr(cb, "seed", getattr(composer, "seed", 42)))
        vocab = list(getattr(cb, "words", []))
        store = cls(n_shards=n_shards, seed=seed, D=D, vocab=vocab,
                    composer_factory=composer_factory, share_codebook=True)
        # Graft the SOURCE codebook objects into EVERY shard (identical concepts/roles/pol_words/words incl.
        # grounded overrides + any runtime-grown words), so a re-encoded composite is byte-identical to the source.
        if store.shards:
            for sh in store.shards:
                if hasattr(cb, "concepts"):
                    sh.concepts = cb.concepts
                if hasattr(cb, "roles"):
                    sh.roles = cb.roles
                if hasattr(cb, "pol_words"):
                    sh.pol_words = cb.pol_words
                if hasattr(cb, "words"):
                    sh.words = cb.words
                if hasattr(cb, "_growth_rng"):
                    sh._growth_rng = cb._growth_rng
        # Re-home every stored fact into its agent's shard, IN ORDER (preserves per-agent first-match), by replaying
        # store() -> re-encode with the shared codebook -> byte-identical composite.
        for entry in getattr(composer, "kb", []):
            fact = entry[0] if isinstance(entry, (tuple, list)) else entry
            if not isinstance(fact, dict):
                continue
            agent, action, patient, polarity = _fact_to_store_args(fact)
            if agent is None or action is None:
                continue
            store.store(agent, action, patient, polarity=polarity)
        return store
