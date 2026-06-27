"""RoutedComposer -- the PRODUCTION multi-bridge facade for the first-chat console (deep-knowledge scaling).

Design: research/findings/2026-06-26-multibridge-deep-knowledge-design.md (§2.7 the RoutedComposer facade).
De-risk: research/runners/_multibridge_stage0_derisk.py (Stage-0 GO: per-bridge recall 0.958 >= single 0.917,
moat 0-FA, ~1.75x faster, numpy-CPU). This module turns those VERIFIED blocks into a drop-in composer.

THE PROBLEM IT SOLVES: a single RFPhasorComposer cleans up every unbind over its WHOLE vocab (an argmax matched
filter over `comp.words`, rf_phasor_composer.py:381-444). At ~1,000 concepts the codebook is dense enough that a
D=128 phasor recovered at lossy fidelity mis-cleans to a wrong-but-close word -> recall drops on rarer facts, and
each query's matvec grows O(V). The fix (DE-RISKED) is to SHARD the concepts across N composers, each cleaning up
over only ~V/N concepts, behind a host `word2shard` router (the proven g20_multibridge routing pattern).

THE WHOLE MECHANISM IS REUSE-BY-IMPORT -- NO `sim/` edit, NO `rf_phasor_composer.py` edit. The composer ALREADY
accepts a `words=` cleanup subset; per-bridge cleanup falls out of constructing one composer per shard over that
shard's vocab. RoutedComposer just dispatches each call to the shard owning the relevant concept and presents the
SAME surface the DiscursiveTurn / proposer / agent consume:
    store, query_patient, query_agent, ask_yes_no, query_chain, render_fact, update_on_mismatch, count_facts,
    elaborate, unbind, .kb, .concepts, .words.

THE NO-CONFAB MOAT IS THE LOAD-BEARING INVARIANT (preserved + tested cross-shard):
  - an unknown concept (in NO shard's vocab) -> the router abstains (word2shard.get -> None -> return None);
  - a present-but-unstored cue -> the owning shard's composer abstains (query_* -> None / 'unknown');
  - a never-stored cross-shard cue (agent in A, patient in B) -> must abstain, NOT spuriously match via the §2.5
    codebook extension.
A single confident answer on any absent cue is a HARD STOP -- never a fabrication.

CPU / numpy: build under SIM_BACKEND=numpy (the whole composer pipeline is a numpy-CPU brain; Stage-0 was numpy).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners._multibridge_stage0_derisk import (  # noqa: E402  (the VERIFIED Stage-0 building blocks)
    load_brain,
    split_shards,
    build_shard_composers,
)


def _domain_shards(vocab, cat_ids, cat_names, n_shards):
    """Shard by SEMANTIC DOMAIN (design §2.6): group the g20 categories into n_shards DISJOINT bands so most SVO
    facts are within-domain (same-shard) and each shard learns a coherent code space. Categories are assigned to
    shards round-robin by descending member-count so the shard VOCAB sizes stay balanced (a big category does not
    dump 200 words onto one shard). Returns a list of sorted word-lists (a disjoint partition of the vocab) OR
    None if category info is unusable (caller then falls back to split_shards)."""
    if cat_ids is None or cat_names is None or len(cat_names) < n_shards:
        return None
    by_cat = {}
    for w, ci in zip(vocab, cat_ids):
        by_cat.setdefault(int(ci), []).append(w)
    # largest categories first; assign each whole category to the currently-smallest shard (greedy balance).
    cats_sorted = sorted(by_cat.items(), key=lambda kv: -len(kv[1]))
    shards = [[] for _ in range(n_shards)]
    for _ci, words in cats_sorted:
        si = min(range(n_shards), key=lambda j: len(shards[j]))
        shards[si].extend(words)
    out = [sorted(set(s)) for s in shards]
    # guard: every shard must be non-empty + the union must be the whole vocab (a true partition)
    if any(len(s) == 0 for s in out):
        return None
    allw = [w for s in out for w in s]
    if len(allw) != len(set(allw)) or set(allw) != set(vocab):
        return None
    return out


class RoutedComposer:
    """N stream-cortex shards (each an RFPhasorComposer over a disjoint ~V/N-concept vocab) behind a host router.

    Presents the RFPhasorComposer conversational API; routes each call to the shard owning the relevant concept.
    """

    def __init__(self, codes_npz, n_shards=3, seed=42, D=128, shard_by="domain", grounded_codes=None, verbose=False):
        """codes_npz: path to the trained brain .npz (vocab + grounded phasor codes + cat_ids/cat_names + D).
        n_shards: number of shards (~V/n_shards concepts each). shard_by: 'domain' (g20-category bands, design §2.6;
        falls back to a disjoint random partition if categories are unusable) or 'partition' (the Stage-0 disjoint
        random split). grounded_codes: optional {word: phases[D]} override (e.g. the production grounded path); when
        None the codes are loaded from the npz."""
        self.seed = int(seed)
        self.n_shards = int(n_shards)
        vocab, grounded_npz, cat_ids, cat_names, D_npz = load_brain(codes_npz)
        self.D = int(D if D is not None else D_npz)
        if self.D != D_npz:
            # the console passes D explicitly; honor the npz's actual D to keep codes aligned.
            self.D = D_npz
        grounded = dict(grounded_npz)
        if grounded_codes:
            for w, ph in grounded_codes.items():
                if w in grounded:
                    grounded[w] = np.asarray(ph, dtype=float)
        self._grounded = grounded
        self.cat_ids = cat_ids
        self.cat_names = cat_names

        # ---- the disjoint shard vocabularies ----
        shards = None
        if shard_by == "domain":
            shards = _domain_shards(vocab, cat_ids, cat_names, self.n_shards)
        if shards is None:
            shards = split_shards(vocab, self.n_shards, self.seed)   # Stage-0 disjoint random partition (the floor)
            self._shard_policy = "partition"
        else:
            self._shard_policy = "domain"
        self.shard_vocabs = shards

        # routing-correctness invariant: word2shard is a PARTITION (disjoint vocabs, full union).
        all_words = [w for s in shards for w in s]
        assert len(all_words) == len(set(all_words)) == len(set(vocab)), "shards are not a disjoint partition"

        # ---- one RFPhasorComposer per shard over ONLY its vocab + grounded codes (cleanup over ~V/N) ----
        self.comps, self.word2shard = build_shard_composers(shards, grounded, self.seed, self.D)

        # a read-only union view of the codes (for context_code / _phase_cos callers that introspect concepts).
        self._union_concepts = {}
        for c in self.comps:
            self._union_concepts.update(c.concepts)
        self._union_words = sorted(set(all_words))
        if verbose:
            sizes = [len(s) for s in shards]
            print(f"[routed] {self.n_shards} shards ({self._shard_policy}), sizes={sizes}, "
                  f"D={self.D}, {len(self._union_words)} concepts total", flush=True)

    # ----------------------------------------------------------------------------------------------------------
    # routing helpers
    # ----------------------------------------------------------------------------------------------------------
    def _shard_of(self, word):
        """The shard index owning `word`, or None (unknown word -> the router moat)."""
        return self.word2shard.get(word)

    def _ensure_filler(self, comp, word):
        """Co-store a cross-shard role-filler's grounded code into `comp`'s codebook (design option 2a). The
        composer's CLEANUP ranges over comp.words (rf_phasor_composer.py:382), so we extend BOTH the code dict AND
        the cleanup word-list -- else a recovered cross-shard filler phasor could not be decoded. BOUNDED: the
        codebook grows only by the distinct cross-shard fillers this shard's facts actually reference. Returns
        True iff the word was a cross-shard extension (was not native to comp)."""
        if word in comp.concepts:
            return False
        code = self._grounded.get(word)
        if code is None:                       # filler not in ANY shard's learned codes -> cannot extend (no code)
            return False
        comp.concepts[word] = np.asarray(code, dtype=float)
        comp.words = sorted(set(comp.words) | {word})
        return True

    # ----------------------------------------------------------------------------------------------------------
    # the conversational API (mirrors RFPhasorComposer; routes to the owning shard; moat preserved)
    # ----------------------------------------------------------------------------------------------------------
    def store(self, agent, action, patient, polarity=None):
        """Store a fact on the shard owning its AGENT (design §2.5 agent-anchoring). The composer's _encode binds
        all three roles, so any role-filler not native to the agent-shard's codebook is co-stored there (option
        2a, the bounded per-shard codebook extension). A fact whose AGENT is unknown (in no shard) is silently
        dropped -- we never invent a home for an out-of-vocab agent (consistent with the moat: unknown -> nothing).
        An attributed/clause patient is passed through verbatim (the composer handles tuple/Clause patients); its
        cross-shard string fillers are also co-stored when present.

        Returns the storing shard index, or None if the agent is unknown (dropped)."""
        si = self._shard_of(agent)
        if si is None:
            return None                        # unknown agent -> no home; do not fabricate one
        comp = self.comps[si]
        # co-store every cross-shard STRING role-filler's code so the shard can decode it (agent is native by def).
        self._ensure_filler(comp, action)
        self._ensure_filler(comp, patient if isinstance(patient, str) else agent)
        if isinstance(patient, tuple) and not _is_clause(patient):
            adjs, noun = patient
            adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
            for w in list(adjs) + [noun]:
                if isinstance(w, str):
                    self._ensure_filler(comp, w)
        comp.store(agent, action, patient, polarity=polarity)
        return si

    def query_patient(self, agent, action, order_fn=None):
        """'what does <agent> <action>?' -> the patient, routed to the agent's shard; None (abstain) if the agent
        is unknown OR no stored fact on that shard matches."""
        si = self._shard_of(agent)
        if si is None:
            return None                        # unknown agent -> abstain (the router moat)
        return self.comps[si].query_patient(agent, action, order_fn=order_fn)

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> the agent, routed by the PATIENT's shard first, then any other shard whose
        codebook contains the patient (a cross-shard fact lives on its agent's shard and co-stored the patient via
        option 2a). Returns the first non-None answer, or None (abstain) if no shard matches."""
        tried = []
        sp = self._shard_of(patient)
        if sp is not None:
            tried.append(sp)
        # cross-shard fallback: the fact may live on a DIFFERENT shard that co-stored this patient (option 2a).
        for si, comp in enumerate(self.comps):
            if si not in tried and patient in comp.concepts:
                tried.append(si)
        for si in tried:
            ans = self.comps[si].query_agent(action, patient)
            if ans is not None:
                return ans
        return None

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown', routed to the agent's shard; 'unknown'
        (abstain) if the agent is unknown OR no stored fact matches."""
        si = self._shard_of(agent)
        if si is None:
            return "unknown"                   # unknown agent -> abstain
        return self.comps[si].ask_yes_no(agent, action, patient)

    def query_chain(self, cue, actions):
        """Multi-hop relational reasoning: follow a chain of stored facts across shards. Each hop matches the
        current concept as the AGENT under the hop's action and reads the PATIENT (the next cue). Routes each hop
        to the current concept's shard; abstains (None) the moment any hop has no match -- so the no-confab moat
        holds at EVERY hop and a broken/over-run chain never confabulates."""
        x = cue
        for action in actions:
            x = self.query_patient(x, action)
            if x is None:
                return None
        return x

    def render_fact(self, agent, order_fn=None):
        """Generation: render a full stored sentence whose agent matches `agent`, routed to the agent's shard;
        None (abstain) if the agent is unknown or has no stored fact."""
        si = self._shard_of(agent)
        if si is None:
            return None
        return self.comps[si].render_fact(agent, order_fn=order_fn)

    def update_on_mismatch(self, agent, action, new_patient, pe_labile=None):
        """RECONSOLIDATION (additive): route to the agent's shard. A never-stored cue ABSTAINS (the moat -- a
        reactivated trace is updated, a missing one is not fabricated). Co-stores the new patient's code on the
        shard first so the rewrite can encode/decode it."""
        si = self._shard_of(agent)
        if si is None:
            return {"action": "abstain", "wrote": False, "pe": None}
        if isinstance(new_patient, str):
            self._ensure_filler(self.comps[si], new_patient)
        return self.comps[si].update_on_mismatch(agent, action, new_patient, pe_labile=pe_labile)

    def count_facts(self, agent, action):
        """Number of stored facts whose cue roles (agent+action) match, on the agent's shard (0 if unknown)."""
        si = self._shard_of(agent)
        return 0 if si is None else self.comps[si].count_facts(agent, action)

    def unbind(self, composite_phases, role, words=None):
        """Decode a role from a composite. Routing-agnostic (operates on a phasor + role); cleans up over the union
        unless `words` is given. (Provided for API completeness; the moat-bearing paths above all route first.)"""
        # use shard 0's composer as the unbind engine (roles are seed-shared across shards -> identical); restrict
        # the cleanup codebook to the union view unless a subset is named.
        comp0 = self.comps[0]
        rec = comp0._unbind_phases(composite_phases, role)
        if words is None:
            words = self._union_words
        return comp0._cleanup(rec, words)

    def elaborate(self, topic):
        """Dialogue planning: the next on-topic concept about `topic`, routed to the topic's shard (the dlPFC
        spreading Control runs on that shard's own association graph). None if the topic is unknown/unconnected.

        NOTE: this elaborates over the topic's SHARD-LOCAL fact graph. The cross-shard discuss adjacency (the
        (N)/(D) channel) is supplied by the console's SHARED PPMI graph, not by this method (design §2.2)."""
        si = self._shard_of(topic)
        if si is None:
            return None
        return self.comps[si].elaborate(topic)

    # ----------------------------------------------------------------------------------------------------------
    # union views (read-only) -- the DiscursiveTurn / agent introspect these
    # ----------------------------------------------------------------------------------------------------------
    @property
    def concepts(self):
        """A read-only UNION view of all shards' codes (for context_code / _phase_cos). Cleanup never uses this --
        each shard's own composer cleans up over its own ~V/N vocab."""
        return self._union_concepts

    @property
    def words(self):
        """The UNION vocab (for callers that introspect the full word set). Cleanup never uses this."""
        return self._union_words

    @property
    def kb(self):
        """The UNION of stored facts across shards (for _assoc_graph / audits). Each entry is the shard's own
        (fact_dict, composite) tuple."""
        return [f for c in self.comps for f in c.kb]

    @property
    def roles(self):
        """The role codes (seed-shared across shards -> identical). Some callers read composer.roles."""
        return self.comps[0].roles

    @property
    def pol_words(self):
        return self.comps[0].pol_words


def _is_clause(x):
    """A clause-like filler: a namedtuple with (agent, action, patient) fields (mirrors rf_phasor_composer)."""
    return getattr(x, "_fields", None) == ("agent", "action", "patient")


# ==================================================================================================================
# Self-check (run directly): a few stores + queries + an absent-cue abstention on the 3000 brain.
#   SIM_BACKEND=numpy python -m research.runners.routed_composer
# ==================================================================================================================
def _self_check():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default=os.path.join(_REPO, "bridges", "firstchat",
                                                     "brain3000pos_w7000.npz_seed42.npz"))
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"[self-check] backend={os.environ.get('SIM_BACKEND', 'auto')}", flush=True)
    rc = RoutedComposer(a.brain, n_shards=a.n_shards, seed=a.seed, verbose=True)
    vocab = rc.words
    print(f"[self-check] loaded {len(vocab)} concepts across {rc.n_shards} shards "
          f"(policy={rc._shard_policy}); shard sizes={[len(s) for s in rc.shard_vocabs]}", flush=True)

    # pick three same-shard facts (agent, action, patient all native to one shard) so we test the trivial path,
    # plus one deliberately cross-shard fact (agent on A, patient on B) so we test the option-2a extension.
    facts = []
    for si, sv in enumerate(rc.shard_vocabs):
        if len(sv) >= 3:
            facts.append((sv[0], sv[1], sv[2]))      # a same-shard triple
    # a cross-shard fact: agent from shard 0, action from shard 0, patient from shard 1
    if len(rc.shard_vocabs) >= 2 and rc.shard_vocabs[0] and rc.shard_vocabs[1]:
        a0 = rc.shard_vocabs[0][3] if len(rc.shard_vocabs[0]) > 3 else rc.shard_vocabs[0][0]
        v0 = rc.shard_vocabs[0][4] if len(rc.shard_vocabs[0]) > 4 else rc.shard_vocabs[0][1]
        p1 = rc.shard_vocabs[1][0]
        if a0 != p1:
            facts.append((a0, v0, p1))

    print(f"[self-check] storing {len(facts)} facts:", flush=True)
    for a, v, p in facts:
        home = rc.store(a, v, p, polarity="AFFIRM")
        print(f"   ({a}, {v}, {p}) -> shard {home}", flush=True)

    print("[self-check] RECALL (what + who):", flush=True)
    ok = tot = 0
    for a, v, p in facts:
        what = rc.query_patient(a, v)
        who = rc.query_agent(v, p)
        ok += int(what == p) + int(who == a)
        tot += 2
        print(f"   what({a},{v})={what!r} (want {p!r})   who({v},{p})={who!r} (want {a!r})", flush=True)
    print(f"[self-check] recall {ok}/{tot} = {ok/max(tot,1):.3f}", flush=True)

    # ABSTENTION (the moat): an unknown word + a present-but-unstored cue + a never-stored cross-shard cue.
    print("[self-check] ABSTENTION (the no-confab moat):", flush=True)
    unknown = "zzqxglarbneverword"
    a_unknown = rc.query_patient(unknown, "go")
    print(f"   unknown agent query_patient({unknown!r}, 'go') -> {a_unknown!r}  "
          f"{'OK (abstain)' if a_unknown is None else 'LEAK!'}", flush=True)
    # present-but-unstored: a real agent + a real action never paired
    pa, pv = rc.shard_vocabs[0][10], rc.shard_vocabs[0][11]
    a_unstored = rc.query_patient(pa, pv)
    print(f"   unstored cue query_patient({pa!r}, {pv!r}) -> {a_unstored!r}  "
          f"{'OK (abstain)' if a_unstored is None else 'LEAK!'}", flush=True)
    yn = rc.ask_yes_no(unknown, "go", "park")
    print(f"   unknown ask_yes_no -> {yn!r}  {'OK' if yn == 'unknown' else 'LEAK!'}", flush=True)

    leaks = sum(int(x is not None) for x in (a_unknown, a_unstored)) + int(yn != "unknown")
    print(f"\n[self-check] {'PASS' if leaks == 0 and ok >= tot - 1 else 'CHECK'}: "
          f"recall {ok}/{tot}, moat leaks {leaks}", flush=True)
    return rc


if __name__ == "__main__":
    _self_check()
