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
        # Build the shards. Same seed+vocab -> byte-identical codebooks in every shard.
        self.shards = [
            RFPhasorComposer(seed=self.seed, D=self.D, vocab=vocab, **composer_kwargs)
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

    # --- persistence (build ONCE, reload FAST) -------------------------------------------------------------
    # A 100k-fact LTM is ~30 min to BUILD (one RF resonate per fact at store time), but should reload in seconds
    # for a live chat server. Mirroring developed_brain_io's kb_composites speedup: save each fact's already-bound
    # COMPOSITE, and on load reconstruct the shards (the codebook regenerates byte-identically from seed+vocab --
    # the LTM is a bulk store that is never runtime-grown, so no growth diverges the codes) and set kb DIRECTLY,
    # skipping the resonate. Numpy fast path only (enable_substrate_store=False, the LTM default).
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
                comps.append(np.asarray(handle))
            if comps:
                comp_arrays[f"sh{i}"] = np.stack(comps)
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        with open(os.path.join(path, "facts.json"), "w") as f:
            json.dump(facts, f)
        np.savez(os.path.join(path, "composites.npz"), **comp_arrays)
        return manifest["n_facts"]

    @classmethod
    def load(cls, path):
        """Reconstruct a store saved by `save()` WITHOUT re-resonating (codebook regenerates from seed+vocab; each
        fact's composite is set directly into its shard's kb)."""
        import json
        import os
        import numpy as np
        with open(os.path.join(path, "manifest.json")) as f:
            m = json.load(f)
        store = cls(n_shards=m["n_shards"], seed=m["seed"], D=m["D"], vocab=m["vocab"],
                    share_codebook=m.get("share_codebook", True), **(m.get("composer_kwargs") or {}))
        with open(os.path.join(path, "facts.json")) as f:
            facts = json.load(f)
        comps = np.load(os.path.join(path, "composites.npz"))
        per_shard_idx = {}
        for rec in facts:
            i = int(rec["shard"])
            arr = comps[f"sh{i}"]
            j = per_shard_idx.get(i, 0)
            store.shards[i].kb.append((rec["fact"], arr[j]))
            per_shard_idx[i] = j + 1
        return store
