"""TieredFactStore -- the hippocampal-buffer / cortical-LTM split for the live chat's fact store.

WHY. The live conversational agent holds ONE flat fact store on `agent.composer` (an RFPhasorComposer /
OneBrainComposer): every stored fact is scanned per query (O(K.D)), and the co-resident spiking K-way sequencer
that reads it is provisioned at a small K (the `k_max=32` working-set cap). That is exactly right for the handful
of facts a CONVERSATION introduces, but it caps how much KNOWLEDGE the brain can hold and query -- the owner's #1
priority is to load the fundamental knowledge an LLM has (100k-1M facts) and then chat over it.

THE SPLIT (biological: hippocampal working set + cortical semantic store). Keep the small flat composer as the
active-conversation BUFFER (recent, conversation-taught facts -- the k_max working set, unchanged). Add a
`ShardedPhasorStore` as the cortical LONG-TERM store (bulk knowledge; a routed query touches only ONE shard ~K/S
facts, so it is sub-second at ANY K -- de-risked GO, recall + no-confab moat byte-identically preserved). A READ
checks the BUFFER first (the recent working set), and on an ABSTAIN falls through to the routed LTM shard. A WRITE
goes to the buffer (the newest facts); an explicit `promote_buffer_to_ltm()` is the hippocampal->cortical
consolidation hook (sleep-replay analogue; NOT auto-invoked in v1).

DROP-IN. This class implements the exact RFPhasorComposer/OneBrainComposer READ+WRITE API the live path uses
(`store` / `query_patient` / `query_agent` / `ask_yes_no` / `query_chain` / `chain_of_thought` / `render_fact`),
and delegates every other attribute (`concepts`, `kb`, `D`, `seed`, `unbind`, trace, ...) to the buffer -- so
`agent.composer = TieredFactStore(buffer, ltm)` is a transparent substitution. With `ltm=None` it is
byte-identical to the plain buffer (the safe default). The no-confab MOAT is preserved by construction: each tier
abstains on its own; the tiered read only returns an LTM answer when the buffer ABSTAINED, and returns None/unknown
only when BOTH tiers abstained. NO `sim/` edit -- it composes the two validated stores.

Tiers self-consistently encode by WORD (each store binds/unbinds concept STRINGS with its own codebook), so the
buffer and the LTM need not share a codebook for answers to be correct -- a fact is stored in one tier and read
from that same tier's codes.

See research/findings/2026-08-20-sharded-fact-store-removes-the-O-K-query-wall-knowledge-scales-to-LLM-scale.md.
"""
from __future__ import annotations

from typing import Optional


def auto_n_shards(n_facts: int, target_shard_size: int = 200, min_shards: int = 16) -> int:
    """Pick S so a routed shard holds ~target_shard_size facts (the finding's m=125-250 sub-second band)."""
    return max(int(min_shards), (int(n_facts) + target_shard_size - 1) // target_shard_size)


def encode_fast(comp, fact):
    """The CLOSED-FORM bind+bundle a fact's composite = `angle(sum_r exp(2*pi*i*(role_r + filler_r)))/(2*pi) mod 1`
    -- the exact math the RF resonate-and-fire CONVERGES to (bind of unit phasors = phase addition; bundle = the
    sum's phase), computed directly in numpy instead of stepping the bridge dynamics per role.

    WHY (LLM-scale knowledge). The neural `store` runs 3-4 RF resonates per fact (~52-63 ms/fact), so a
    million-fact teacher-load is ~17 h and 20M is ~350 h -- the wall to LLM-scale knowledge. This computes the
    SAME composite (recall-identical to the resonate path -- 120/120 matched answers + moat preserved, measured)
    at ~78 us/fact = ~670x faster: 20M facts drops to ~26 min (single-thread; further parallelizable per shard).

    SCOPE (honest). This is a declared BULK TEACHER-LOAD optimization -- the teacher precomputes the composite the
    neural bind would produce, so the brain holds the identical representation; the QUERY / recall (the cognition)
    stays FULLY neural (resonate unbind + cleanup), unchanged. It uses `comp._filler_phases` for every role, so
    polarity (AFFIRM/NEGATE), attributes, and nested clauses bind identically to `_encode`.
    """
    import numpy as np
    from research.runners.rf_phasor_composer import ROLES
    acc = np.zeros(comp.D, dtype=np.complex128)
    for r in ROLES:
        if r in fact:
            acc += np.exp(2j * np.pi * (comp.roles[r] + comp._filler_phases(fact[r])))
    return np.angle(acc) / (2.0 * np.pi) % 1.0


def build_ltm_from_facts(facts, vocab=None, *, n_shards=None, seed=42, D=128, composer_kwargs=None, fast=False):
    """Build a ShardedPhasorStore LTM from a list of fact-dicts (agent/action/patient[/polarity]).

    `facts`: an iterable of dicts with string `agent`/`action` and a string-or-clause `patient` (the shape
    `developed_brain_io` persists). `vocab`: the concept word set the shared codebook covers (defaults to every
    word seen in the facts). `n_shards`: defaults to `auto_n_shards(len(facts))`.

    `fast=True` uses the closed-form `encode_fast` for str-patient facts (the ~670x bulk-load speedup; recall-
    identical to the neural resonate bind, verified) and falls back to the neural `store` for clause/attributed
    patients. Default False keeps the byte-identical neural-bind path.
    """
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    facts = list(facts)
    if vocab is None:
        vs = set()
        for f in facts:
            for role in ("agent", "action", "patient", "attribute", "attribute2"):
                w = f.get(role)
                if isinstance(w, str):
                    vs.add(w)
                elif isinstance(w, dict) and w.get("__clause__"):
                    vs.update(x for x in (w.get("agent"), w.get("action"), w.get("patient"))
                              if isinstance(x, str))
        vocab = sorted(vs)
    if n_shards is None:
        n_shards = auto_n_shards(len(facts))
    ltm = ShardedPhasorStore(n_shards=int(n_shards), seed=int(seed), D=int(D),
                             vocab=list(vocab), share_codebook=True, **(composer_kwargs or {}))
    for f in facts:
        a, act, p = f.get("agent"), f.get("action"), f.get("patient")
        # a plain declarative KB fact is an AFFIRMation (the live parser passes 'AFFIRM' for a statement), so a
        # fact with no explicit polarity gets the AFFIRM tag -> ask_yes_no answers 'yes' on it (a None polarity
        # binds NO tag and would read as 'no'). An explicit NEGATE is preserved.
        pol = f.get("polarity") or "AFFIRM"
        if not (isinstance(a, str) and isinstance(act, str)):
            continue
        if fast and isinstance(p, str):
            # closed-form bulk teacher-load: route by agent (== store's routing), append the precomputed composite.
            shard = ltm.shard_for(a)
            fd = {"agent": a, "action": act, "patient": p, "polarity": pol}
            shard.kb.append((fd, encode_fast(shard, fd)))
        elif isinstance(p, str) or isinstance(p, dict):
            ltm.store(a, act, p, polarity=pol)
    return ltm


class TieredFactStore:
    """A small conversation BUFFER (flat composer) + a cortical LTM (ShardedPhasorStore) behind the composer API.

    Reads check the buffer first, then fall through to the routed LTM shard on an abstain. Writes go to the
    buffer. With `ltm=None` it degrades to exactly the buffer (byte-identical).
    """

    # per-read-method: does this return value mean the tier ABSTAINED (-> try the LTM)?
    @staticmethod
    def _abstained(name, result):
        if name == "ask_yes_no":
            return result == "unknown"
        # query_patient / query_agent / render_fact / query_chain: None == abstain
        return result is None

    def __init__(self, buffer, ltm=None):
        # set via object.__setattr__ so __getattr__ never sees these as "missing" (no recursion)
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "ltm", ltm)

    def __setattr__(self, name, value):
        """`buffer`/`ltm` are this wrapper's OWN state (set once above via `object.__setattr__`, never
        reassigned elsewhere); every OTHER attribute write is INSTRUMENTATION meant for the underlying
        composer that actually runs the query, per this class's own documented contract ("delegates every
        other attribute ... to the buffer" -- see the module docstring). Without this override, Python's
        default `object.__setattr__` would silently shadow the write as a NEW instance attribute on the
        TieredFactStore WRAPPER itself (`__getattr__` is only consulted on a MISSING attribute, i.e. reads,
        never on assignment) -- so a caller's `composer.trace = True` would never reach `self.buffer.trace`.

        FOUND 2026-08-27 verifying board #94's confidence-forthcomingness production flip:
        `webapp/server.py`'s per-turn activity-trace flip (`_composer.trace = True; _composer.last_trace =
        None`, the B3 read the E1 metacog-monitor's `activity` depends on) has been a silent no-op on every
        `tiny-demo +LTM` turn since the 2026-08-26 knowledge-core default-on flip
        (`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` `tiered-knowledge-ltm`) -- `self.buffer.trace` was never
        actually set True, so `self.buffer.last_trace` was never recorded, so `activity`/`metacog.confident`
        silently read None on the out-of-the-box production brain regardless of the turn's real confidence.
        See `research/findings/2026-08-27-confidence-forthcomingness-production-default-*.md`."""
        if name in ("buffer", "ltm"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.buffer, name, value)
            # ALSO arm the LTM tier's OWN trace flag (#184 fix, 2026-09-01): `webapp/server.py`'s per-turn
            # `composer.trace = True` used to reach ONLY the buffer (the delegate above) -- a `ShardedPhasorStore`
            # LTM's shards are independent `RFPhasorComposer`s with their own `.trace`, so an LTM-sourced answer
            # never recorded a trace at all, regardless of `_tiered()`'s propagation below. Scoped to `trace`
            # only (not a blanket delegate): several runners assign buffer-only bookkeeping through this same
            # `__setattr__` (e.g. `composer.kb = []` to reset a conversation's working set) that must NOT also
            # touch the LTM's actual stored knowledge. Guarded: an `ltm` without a `trace` attribute (a foreign
            # store type) is a silent no-op, never an exception.
            if name == "trace" and self.ltm is not None:
                try:
                    self.ltm.trace = value
                except Exception:
                    pass

    # -- WRITE: conversation-taught facts land in the recent working-set buffer -------------------------------
    def store(self, agent, action, patient, polarity=None):
        return self.buffer.store(agent, action, patient, polarity=polarity)

    # -- READ: buffer first; on an abstain, the routed LTM shard --------------------------------------------
    def _propagate_ltm_trace(self):
        """#184 fix: when the LTM tier is what actually answered, overwrite the buffer's own (abstain) trace
        with the LTM's real match trace, so `composer.last_trace` -- which always reads through to
        `self.buffer.last_trace` via `__getattr__` below, the SAME slot `webapp/server.py`'s metacog read
        consults -- reports what the brain actually did this turn instead of the buffer's own abstain record.
        Fabricates nothing: it is the LTM shard's own `_trace_scan` output (see `sharded_phasor_store.py`'s
        `_note_trace`), the identical decode-confidence/margin machinery the buffer's own trace uses. No-op
        unless tracing is armed (`self.buffer.trace`) or the LTM produced no trace (an LTM without `last_trace`,
        e.g. a raw dict-shaped store predating this fix, degrades to leaving the buffer's abstain trace as-is)."""
        if getattr(self.buffer, "trace", False):
            ltm_trace = getattr(self.ltm, "last_trace", None)
            if ltm_trace is not None:
                self.buffer.last_trace = ltm_trace

    def _tiered(self, name, args, kwargs):
        r = getattr(self.buffer, name)(*args, **kwargs)
        if self.ltm is not None and self._abstained(name, r):
            r2 = getattr(self.ltm, name)(*args, **kwargs)
            self._propagate_ltm_trace()
            return r2
        return r

    def query_patient(self, agent, action, order_fn=None):
        return self._tiered("query_patient", (agent, action), {"order_fn": order_fn})

    def query_patient_source(self, agent, action, order_fn=None):
        """Like `query_patient`, but also reports WHICH tier answered: `"buffer"` (the small recent-conversation
        working set) or `"ltm"` (the query fell through the buffer's abstain to the cortical long-term store), or
        `(None, None)` when both tiers abstained. Read-only, zero side effects beyond the identical per-tier reads
        `query_patient` already performs (same calls, same order) -- this does not change `query_patient`'s own
        behavior or add any extra composer read.

        WHY. Callers that need to tell a STABLE long-term-memory recall apart from a recent conversational one
        (e.g. `webapp/gnw_two_organ_bus.py`'s organ-B LTM-exemption lever, `BRAIN_GNW_ORGANB_LTM_EXEMPT`) cannot
        do so from `query_patient`'s return value alone -- the answer looks identical either way. This is the
        single place that exposes tier provenance for such a caller, so the distinction is made ONCE, off the
        store's own read path, not re-derived ad hoc at each call site."""
        r = self.buffer.query_patient(agent, action, order_fn=order_fn)
        if r is not None:
            return r, "buffer"
        if self.ltm is not None:
            r2 = self.ltm.query_patient(agent, action, order_fn=order_fn)
            if r2 is not None:
                return r2, "ltm"
        return None, None

    def query_agent(self, action, patient):
        return self._tiered("query_agent", (action, patient), {})

    def ask_yes_no(self, agent, action, patient):
        return self._tiered("ask_yes_no", (agent, action, patient), {})

    def render_fact(self, agent, order_fn=None):
        return self._tiered("render_fact", (agent,), {"order_fn": order_fn})

    def query_chain(self, cue, actions):
        # a single-tier chain: buffer first (a few recent facts dead-end fast), else the LTM knowledge graph.
        r = self.buffer.query_chain(cue, actions)
        if self.ltm is not None and r is None:
            r2 = self.ltm.query_chain(cue, actions)
            self._propagate_ltm_trace()
            return r2
        return r

    def chain_of_thought(self, start, goal=None, max_hops=4, lesion=None, lesion_rng=None, return_path=False):
        r = self.buffer.chain_of_thought(start, goal=goal, max_hops=max_hops, lesion=lesion,
                                         lesion_rng=lesion_rng, return_path=return_path)
        terminal = r[0] if isinstance(r, tuple) else r
        if self.ltm is not None and terminal is None:
            r2 = self.ltm.chain_of_thought(start, goal=goal, max_hops=max_hops, lesion=lesion,
                                           lesion_rng=lesion_rng, return_path=return_path)
            self._propagate_ltm_trace()
            return r2
        return r

    # -- CONSOLIDATION hook (hippocampal -> cortical; sleep-replay analogue; NOT auto-invoked in v1) ---------
    def promote_buffer_to_ltm(self):
        """Move every buffer fact into the LTM and clear the buffer (bounding the working set). Returns the
        number of facts promoted. Requires an LTM. The faithful version is replay-driven consolidation; here it
        is an explicit, caller-triggered transfer so the buffer stays a bounded working set."""
        if self.ltm is None:
            return 0
        moved = 0
        for fact, _comp in list(getattr(self.buffer, "kb", [])):
            a, act, p = fact.get("agent"), fact.get("action"), fact.get("patient")
            pol = fact.get("polarity")
            if isinstance(a, str) and isinstance(act, str) and (isinstance(p, str) or isinstance(p, dict)):
                self.ltm.store(a, act, p, polarity=pol)
                moved += 1
        # clear the buffer's fact list (the codebook/concepts stay; only the stored facts drain)
        if hasattr(self.buffer, "kb"):
            self.buffer.kb.clear()
        return moved

    # -- INTROSPECTION --------------------------------------------------------------------------------------
    def total_facts(self) -> int:
        n_buf = len(getattr(self.buffer, "kb", []))
        n_ltm = self.ltm.total_facts() if self.ltm is not None else 0
        return n_buf + n_ltm

    # -- everything else (concepts, kb, D, seed, unbind, _render, trace, pol_words, ...) -> the buffer -------
    def __getattr__(self, name):
        # only reached for attributes NOT defined on the class/instance; delegate to the buffer composer.
        return getattr(object.__getattribute__(self, "buffer"), name)
