"""SlotBinderComposer — the gap-#2 spiking competitive-slot binder as a first-class, selectable conversational
composer (2026-07-17, follow-on (c) of "finish gap #2 fully").

It implements the same store/query contract as `RFPhasorComposer`/`OneBrainComposer`, so it is a drop-in composer
for `BrainConversationalAgent` (`composer=SlotBinderComposer(...)` or `composer_kind="slotbinder"`). Instead of the
hand-designed exact-inverse FHRR/VSA algebra, it stores each fact's (agent, verb, patient, polarity) into its OWN
four spiking slots and recalls by a NEURAL SCAN with a no-confab moat:

  store(a, v, p, polarity):  allocate fact i -> teach slot[4i+0]->code(a), +1->code(v), +2->code(p),
                             +3->code(AFFIRM|NEGATE), each via the validated Hebbian slot->filler write.
  query_patient(a, v):       scan facts; the fact whose agent- AND verb-slots read back (a, v) -> read its patient
                             slot; NO match -> None (abstain = the moat).
  ask_yes_no(a, v, p):       the matching fact's patient + polarity slots -> yes/no; no match -> 'unknown'.

Why this closes gap #2 in the pipeline: (1) separate slots per (fact, role) = the gap-#2 win over the FHRR
superposition cap (validated 6-seed GO, adversarially confirmed: no-teach->chance, scramble-teach->0.00); (2) the
recall is repeated application of the validated single-bind readout (drive a slot -> read its filler); (3) it is
fully spiking (the readout reset is the neuralized D3 CLEAR); (4) the no-confab moat holds by construction (no
matching fact -> abstain). Fillers are concept POOLS (the g20/Pulvermüller distributed-word-ensemble representation);
generalization across SIMILAR concepts is the separate, already-closed cross-modal/PPMI arc, not this binder.

Honest scope: it is an LTM plastic-weight store (the NMDA hold is not load-bearing); flat SVO facts + SINGLE-attribute
patients ('big apple' -- a 5th flat `attribute` role, gap-#2 attribute-slot step, 2026-07-22; embedded-clause patients
are the follow-on pointer/indirection step); capacity = `max_facts` slots (a scale lever, not a wall). CPU/numpy or
GPU/cupy (the underlying bridge is backend-agnostic).
"""

import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge, _idx  # noqa: E402

_DEFAULT_VOCAB = ["dog", "cat", "fish", "bird", "chase", "eat", "see", "hear", "north", "south"]
AFFIRM, NEGATE = "__AFFIRM__", "__NEGATE__"
NOATTR = "__NOATTR__"                 # the "no adjective" filler -> query_attribute returns None (moat by construction)
_ROLES = 5   # agent, verb, patient, polarity, attribute (attribute defaults to NOATTR for an un-attributed fact)


def _CLAUSE_LABEL(j):
    """The dedicated pointer/indirection filler-pool label for fact-group `j` (a POINTER, not a copy: the matrix
    patient slot binds this pool; `query_clause` reads it back and FOLLOWS it to group `j`'s own slots). Appended as
    extra filler pools exactly like the AFFIRM/NEGATE/NOATTR internal fillers. Depth-1 embedded clauses only."""
    return f"__CLAUSE{j}__"


class SlotBinderComposer:
    def __init__(self, seed=42, vocab=None, D=128, max_facts=16, concepts=None, grounded_codes=None,
                 gain=400.0, teach_steps=40, retr_steps=40, max_clauses=None, fanout=None, prewire_facts=None,
                 **_ignored):
        """fanout / prewire_facts (L2 sparsification, 2026-09-04 -- see
        research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md): `fanout=None`
        (default) is byte-identical to the pre-2026-09-04 dense O(K*KF) slot->filler wiring. Passing an int
        `< KF` switches to a fixed small per-slot candidate set (`build_binder_bridge`'s `fanout` -- O(K), not
        O(K*KF)). `prewire_facts`, when given, is the ordered list of flat-SVO fact dicts/tuples this composer
        will be `store()`-d with (the KNOWN-corpus batch-consolidation case this was built for -- e.g. migrating
        an existing `facts.json` bundle) -- it lets `_ensure()` PRE-REGISTER each slot's one true required filler
        so the sparsified wiring is guaranteed to include it (padded to `fanout` with random distractors). Without
        `prewire_facts`, a sparse build is BLIND (no foreknowledge) -- honest about the coverage risk that
        carries. `bind`/`write`/`recall`/`moat` mechanics below (`store_pair`, `read_slot`, `_match`, ...) are
        UNCHANGED either way -- only the WIRING differs."""
        self.seed = int(seed)
        self.D = int(D)                        # API compat (unused: this composer binds pools, not D-dim codes)
        base = list(vocab) if vocab is not None else (list(concepts.keys()) if concepts else list(_DEFAULT_VOCAB))
        self.words = base
        self.max_facts = int(max_facts)
        # one pointer pool per possible fact-group (CLAUSE_j <-> group j, bijective; the pointer literally names the
        # group index, so no host address table is load-bearing -- the pointer identity IS the address). A scale
        # lever (like max_facts), not a wall.
        self.max_clauses = int(max_facts if max_clauses is None else max_clauses)
        self._clause_labels = [_CLAUSE_LABEL(j) for j in range(self.max_clauses)]
        # polarity + noattr + the pointer pools are all appended as extra filler pools
        self._vocab = base + [AFFIRM, NEGATE, NOATTR] + self._clause_labels
        self._w2i = {w: i for i, w in enumerate(self._vocab)}
        self._pol = {"AFFIRM": self._w2i[AFFIRM], "NEGATE": self._w2i[NEGATE]}
        self._noattr = self._w2i[NOATTR]
        self._clause_filler = {j: self._w2i[lab] for j, lab in enumerate(self._clause_labels)}   # group -> filler idx
        self._clause_label_to_group = {lab: j for j, lab in enumerate(self._clause_labels)}       # label -> group
        self.gain, self.teach_steps, self.retr_steps = gain, teach_steps, retr_steps
        self.parser = None                     # no on-bridge parser -> the agent uses its own
        self.concepts = {w: i for i, w in enumerate(self.words)}
        self.facts = []                        # list of dicts: {agent, action, patient, polarity[, ptr_group]}
        self._b = None                         # bridge built lazily on first store
        self.fanout = None if fanout is None else int(fanout)
        self._prewire_facts = prewire_facts

    def _required_fillers_from_prewire(self):
        """Precompute {slot_index: [filler_index]} for every (fact_i, role) this composer will populate, from
        `self._prewire_facts` (ordered flat-SVO fact dicts: 'agent','action','patient','polarity'[,'attribute']).
        This is a WIRING-TIME pre-registration of a KNOWN corpus (not a per-query lookahead) -- see __init__.
        Embedded-clause patients are not supported here (every real fact in the day_33 bundle is flat SVO) and
        raise rather than silently mis-wire, since a wrong required-filler set defeats the point of using fanout."""
        req = {}
        for i, fact in enumerate(self._prewire_facts):
            if isinstance(fact, dict):
                agent, action, patient = fact["agent"], fact["action"], fact["patient"]
                polarity, attribute = fact.get("polarity"), fact.get("attribute")
            else:
                agent, action, patient, polarity, attribute = (list(fact) + [None, None])[:5]
            if self._as_clause(patient) is not None:
                raise ValueError("prewire_facts: embedded-clause patients are not supported by fanout "
                                  "pre-registration (fact %d)" % i)
            noun, tuple_attr = self._resolve_patient(patient)
            if attribute is None:
                attribute = tuple_attr
            pol = "NEGATE" if polarity in ("NEGATE", "neg", False) else "AFFIRM"
            attr_filler = self._w2i[attribute] if attribute is not None else self._noattr
            base = _ROLES * i
            req[base + 0] = [self._w2i[agent]]
            req[base + 1] = [self._w2i[action]]
            req[base + 2] = [self._w2i[noun]]
            req[base + 3] = [self._pol[pol]]
            req[base + 4] = [attr_filler]
        return req

    # ---- lazy bridge + primitives -------------------------------------------------------------------
    def _ensure(self):
        if self._b is not None:
            return
        from sim.backend import to_host, from_host
        KF = len(self._vocab)
        required = self._required_fillers_from_prewire() if (self.fanout is not None and self._prewire_facts) else None
        b = build_binder_bridge(self.seed, K=_ROLES * self.max_facts, KF=KF, fanout=self.fanout,
                                required_fillers=required)
        n = b.core_config.num_neurons
        slot_idx = [_idx(b, f"w{k}") for k in range(b._K_slots)]
        fill_idx = [_idx(b, f"f{f}") for f in range(KF)]
        # L3 wire-in latency de-risk (2026-09-05, research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-
        # perstep-cost-dominates-latency.md):
        # read_slot's per-step readout used to be a Python `for f in range(KF): rate[f] += fir[fill_idx[f]].mean()`
        # -- an O(KF) PYTHON-LEVEL loop every retrieval step, unchanged by L2's fanout sparsification (flagged by
        # the L2 finding's own S4 as "the real gate on production-viability": fanout shrinks wired SYNAPSES, not
        # this readout's KF-iteration count). Every filler pool has the SAME neuron count (`n_fill`, fixed by
        # `build_binder_bridge`), so `fill_idx` stacks into one rectangular (KF, n_fill) index matrix and the whole
        # per-step readout becomes ONE vectorized numpy reduction (`fir[fill_idx_mat].mean(axis=1)`) instead of KF
        # separate Python-dispatched `.mean()` calls -- identical arithmetic (same elements, same axis-wise mean),
        # verified numerically equivalent against the original loop before this replaced it. Falls back to the
        # original loop if a future caller ever builds non-uniform filler pools (not reachable via this module's
        # own `build_binder_bridge`, kept defensive rather than assumed).
        _fill_pool_sizes = {len(x) for x in fill_idx}
        fill_idx_mat = np.stack(fill_idx) if len(_fill_pool_sizes) == 1 else None

        def _reset():
            if getattr(b, "cp_izh_c_reset", None) is not None:
                b.cp_membrane_potential_v[:] = b.cp_izh_c_reset
            else:
                b.cp_membrane_potential_v[:] = -65.0
            b.cp_recovery_variable_u[:] = 0.0
            if getattr(b, "cp_firing_states", None) is not None:
                b.cp_firing_states[:] = False
            for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
                       "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
                _arr = getattr(b, _a, None)
                if _arr is not None:
                    _arr[:] = 0.0

        def store_pair(slot, filler):
            _reset()
            cur = np.zeros(n); cur[slot_idx[slot]] = self.gain; cur[fill_idx[filler]] = self.gain
            b.set_plasticity_gate(f"slot{slot}_to_filler", 1.0)
            dev = from_host(cur.astype(np.float64))
            for _ in range(self.teach_steps):
                b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
            b.set_plasticity_gate(f"slot{slot}_to_filler", 0.0)
            _reset()

        def read_slot(slot):
            _reset()
            cur = np.zeros(n); cur[slot_idx[slot]] = self.gain
            dev = from_host(cur.astype(np.float64)); rate = np.zeros(KF)
            for _ in range(self.retr_steps):
                b.cp_external_input_current[:] = dev; b._run_one_simulation_step()
                fir = np.asarray(to_host(b.cp_firing_states)).astype(float)
                if fill_idx_mat is not None:
                    rate += fir[fill_idx_mat].mean(axis=1)      # vectorized -- one numpy reduction, not KF Python calls
                else:
                    for f in range(KF):
                        rate[f] += fir[fill_idx[f]].mean()
            return int(np.argmax(rate)), float(rate.max())

        self._b, self._store_pair, self._read_slot = b, store_pair, read_slot

    def _read_word(self, fact_i, role):
        idx, _ = self._read_slot(_ROLES * fact_i + role)
        return self._vocab[idx]

    # ---- the composer contract ----------------------------------------------------------------------
    @staticmethod
    def _resolve_patient(patient):
        """Split a patient operand into (noun, attribute). A bare word -> (word, None). An attributed entity
        `(adjs, noun)` / `(adj, noun)` tuple -> (noun, the FIRST adjective) -- SINGLE-attribute only (a 2nd
        adjective is dropped; the 2-attribute case is the FHRR's own ~29% boundary, deliberately out of scope)."""
        if not isinstance(patient, tuple):
            return patient, None
        adjs, noun = patient                                   # (adj(s), noun)
        adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
        return noun, (adjs[0] if adjs else None)

    @staticmethod
    def _as_clause(patient):
        """A depth-1 embedded-clause patient -> its (agent, action, patient) triple; anything else -> None.
        Accepts a `Clause` namedtuple, a {'agent','action','patient'} dict, or a plain 3-tuple of strings. A 2-tuple
        `(adjs, noun)` is an ATTRIBUTE (handled by `_resolve_patient`), not a clause; a bare string is a flat noun."""
        if hasattr(patient, "agent") and hasattr(patient, "action") and hasattr(patient, "patient"):
            return (patient.agent, patient.action, patient.patient)          # Clause namedtuple
        if isinstance(patient, dict) and {"agent", "action", "patient"} <= set(patient):
            return (patient["agent"], patient["action"], patient["patient"])  # dict
        if (isinstance(patient, tuple) and not hasattr(patient, "_fields")
                and len(patient) == 3 and all(isinstance(x, str) for x in patient)):
            return tuple(patient)                                            # plain 3-tuple (a,v,p)
        return None

    def _store_matrix_with_pointer(self, agent, action, ptr_group, polarity=None):
        """Store a MATRIX fact whose patient slot binds the POINTER pool `CLAUSE_{ptr_group}` (indirection: the
        patient is a reference to fact-group `ptr_group`, NOT a copy of its content). `ptr_group` may name a group
        that is not (yet) stored -- `query_clause` then abstains (the dangling-pointer moat). Returns True/False."""
        if not (agent in self._w2i and action in self._w2i and 0 <= ptr_group < self.max_clauses):
            return False
        self._ensure()
        i = len(self.facts)
        if i >= self.max_facts:
            raise RuntimeError(f"SlotBinderComposer capacity {self.max_facts} facts reached (raise max_facts)")
        pol = "NEGATE" if polarity in ("NEGATE", "neg", False) else "AFFIRM"
        self._store_pair(_ROLES * i + 0, self._w2i[agent])
        self._store_pair(_ROLES * i + 1, self._w2i[action])
        self._store_pair(_ROLES * i + 2, self._clause_filler[ptr_group])     # the pointer pool in the patient slot
        self._store_pair(_ROLES * i + 3, self._pol[pol])
        self._store_pair(_ROLES * i + 4, self._noattr)
        self.facts.append({"agent": agent, "action": action, "patient": _CLAUSE_LABEL(ptr_group),
                           "polarity": pol, "attribute": None, "ptr_group": int(ptr_group)})
        return True

    def _store_clause_fact(self, agent, action, clause, polarity=None):
        """Depth-1 recursion by INDIRECTION (point, don't copy): (1) store the inner clause `(ca,cv,cp)` as its OWN
        flat fact at group `j` (the existing 6-seed-GO flat mechanism, its own near-orthogonal slots); (2) store the
        matrix fact `(agent, action, PTR=CLAUSE_j)`. Read = `query_clause` scans the matrix -> reads the pointer ->
        follows it to group `j`. No clause-level superposition (the gap-#2 win is preserved one level down)."""
        ca, cv, cp = clause
        if not all(w in self._w2i for w in (agent, action, ca, cv, cp)):
            return False
        self._ensure()
        j = len(self.facts)                              # the inner clause's group index (== its pointer id)
        if j >= self.max_clauses:
            raise RuntimeError(f"SlotBinderComposer only has {self.max_clauses} pointer pools (raise max_clauses)")
        if self.store(ca, cv, cp) is not True:           # inner clause as a flat fact -> group j
            return False
        return self._store_matrix_with_pointer(agent, action, j, polarity=polarity)

    def store(self, agent, action, patient, polarity=None, attribute=None):
        # A depth-1 embedded-clause patient (Clause / dict / 3-tuple) routes through pointer/indirection; flat SVO +
        # SINGLE attribute keep the byte-identical path below.
        clause = self._as_clause(patient)
        if clause is not None:
            return self._store_clause_fact(agent, action, clause, polarity=polarity)
        # flat SVO + SINGLE attribute. The attribute may be passed via `attribute=` OR inline as a `(adjs, noun)`
        # tuple patient (split here); a bare-string patient keeps the flat path (attribute defaults to NOATTR).
        noun, tuple_attr = self._resolve_patient(patient)
        if attribute is None:
            attribute = tuple_attr
        if not (isinstance(noun, str) and agent in self._w2i and action in self._w2i and noun in self._w2i):
            return False
        if attribute is not None and attribute not in self._w2i:
            return False
        self._ensure()
        i = len(self.facts)
        if i >= self.max_facts:
            raise RuntimeError(f"SlotBinderComposer capacity {self.max_facts} facts reached (raise max_facts)")
        pol = "NEGATE" if polarity in ("NEGATE", "neg", False) else "AFFIRM"
        attr_filler = self._w2i[attribute] if attribute is not None else self._noattr
        self._store_pair(_ROLES * i + 0, self._w2i[agent])
        self._store_pair(_ROLES * i + 1, self._w2i[action])
        self._store_pair(_ROLES * i + 2, self._w2i[noun])
        self._store_pair(_ROLES * i + 3, self._pol[pol])
        self._store_pair(_ROLES * i + 4, attr_filler)          # NOATTR when the fact has no adjective
        self.facts.append({"agent": agent, "action": action, "patient": noun, "polarity": pol,
                           "attribute": attribute})
        return True

    store_fact = None  # (agent uses store(); store_fact is an RF-only convenience)

    def _match(self, cue_a=None, cue_v=None, cue_p=None):
        """Neural scan: FIRST fact whose cued role-slots read back the cue words; else None (abstain)."""
        if self._b is None:
            return None
        for i in range(len(self.facts)):
            if cue_a is not None and self._read_word(i, 0) != cue_a:
                continue
            if cue_v is not None and self._read_word(i, 1) != cue_v:
                continue
            if cue_p is not None and self._read_word(i, 2) != cue_p:
                continue
            return i
        return None

    def query_patient(self, agent, action, order_fn=None):
        i = self._match(cue_a=agent, cue_v=action)
        return None if i is None else self._read_word(i, 2)

    def query_agent(self, action, patient):
        i = self._match(cue_v=action, cue_p=patient)
        return None if i is None else self._read_word(i, 0)

    def ask_yes_no(self, agent, action, patient):
        i = self._match(cue_a=agent, cue_v=action)
        if i is None or self._read_word(i, 2) != patient:
            return "unknown"                                   # the moat
        return "yes" if self._read_word(i, 3) == AFFIRM else "no"

    def query_attribute(self, agent, action):
        """The single adjective bound to the matching fact's patient (e.g. 'big') -> None when the fact stored no
        attribute (the attribute slot reads NOATTR) or the cue matches no fact (abstain = the moat)."""
        i = self._match(cue_a=agent, cue_v=action)
        if i is None:
            return None
        w = self._read_word(i, 4)
        return None if w == NOATTR else w

    def query_clause(self, agent, action):
        """FOLLOW an embedded-clause pointer: scan-match the matrix fact `(agent, action, PTR)`, read its patient
        slot back to the pointer `CLAUSE_j`, then read group `j`'s (agent, action, patient) with the SAME scan and
        return that inner triple. Returns None when: no matrix matches (moat); the matched fact's patient is a plain
        noun (a flat fact, not a clause); or the pointer names no stored group (a dangling pointer -- the moat).
        Depth-1 only (the inner patient is a plain noun; no depth-2 recursion is attempted)."""
        i = self._match(cue_a=agent, cue_v=action)
        if i is None:
            return None                                         # no matrix fact -> abstain (moat)
        ptr_word = self._read_word(i, 2)                        # the patient slot reads back the pointer pool
        j = self._clause_label_to_group.get(ptr_word)
        if j is None:
            return None                                         # flat patient (not a pointer) -> not a clause fact
        if j < 0 or j >= len(self.facts):
            return None                                         # pointer names no stored group -> abstain (moat)
        return (self._read_word(j, 0), self._read_word(j, 1), self._read_word(j, 2))

    def render_fact(self, agent, order_fn=None):
        i = self._match(cue_a=agent)
        if i is None:
            return None
        attr = self._read_word(i, 4)
        patient = self._read_word(i, 2)
        j = self._clause_label_to_group.get(patient)            # a clause pointer? expand it inline
        if j is not None and 0 <= j < len(self.facts):
            inner = f"( {self._read_word(j, 0)} {self._read_word(j, 1)} {self._read_word(j, 2)} )"
            patient_str = inner
        else:
            patient_str = f"{attr} {patient}" if attr != NOATTR else patient   # 'big apple' when attributed
        words = [self._read_word(i, 0), self._read_word(i, 1), patient_str]
        order = order_fn(3) if order_fn is not None else [0, 1, 2]
        return " ".join(words[o] for o in order)

    def query_chain(self, cue, actions):
        current = cue
        for action in actions:
            current = self.query_patient(current, action)
            if current is None:
                return None
        return current


def _selftest():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    c = SlotBinderComposer(seed=42, vocab=["dog", "cat", "fish", "bird", "chase", "eat", "see"], max_facts=6)
    c.store("dog", "chase", "cat")
    c.store("cat", "eat", "fish", polarity="NEGATE")
    c.store("bird", "see", "dog")
    print("query_patient(dog,chase) =", c.query_patient("dog", "chase"), "(exp cat)")
    print("query_agent(see,dog)     =", c.query_agent("see", "dog"), "(exp bird)")
    print("ask_yes_no(dog,chase,cat)=", c.ask_yes_no("dog", "chase", "cat"), "(exp yes)")
    print("ask_yes_no(cat,eat,fish) =", c.ask_yes_no("cat", "eat", "fish"), "(exp no -- negated)")
    print("query_patient(fish,eat)  =", c.query_patient("fish", "eat"), "(exp None -- moat)")
    print("render_fact(bird)        =", c.render_fact("bird"), "(exp 'bird see dog')")
    print("query_chain(dog,[chase]) =", c.query_chain("dog", ["chase"]), "(exp cat)")


if __name__ == "__main__":
    _selftest()
