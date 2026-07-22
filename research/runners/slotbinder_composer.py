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


class SlotBinderComposer:
    def __init__(self, seed=42, vocab=None, D=128, max_facts=16, concepts=None, grounded_codes=None,
                 gain=400.0, teach_steps=40, retr_steps=40, **_ignored):
        self.seed = int(seed)
        self.D = int(D)                        # API compat (unused: this composer binds pools, not D-dim codes)
        base = list(vocab) if vocab is not None else (list(concepts.keys()) if concepts else list(_DEFAULT_VOCAB))
        # the two polarity pools are appended as extra filler pools
        self.words = base
        self._vocab = base + [AFFIRM, NEGATE, NOATTR]
        self._w2i = {w: i for i, w in enumerate(self._vocab)}
        self._pol = {"AFFIRM": self._w2i[AFFIRM], "NEGATE": self._w2i[NEGATE]}
        self._noattr = self._w2i[NOATTR]
        self.max_facts = int(max_facts)
        self.gain, self.teach_steps, self.retr_steps = gain, teach_steps, retr_steps
        self.parser = None                     # no on-bridge parser -> the agent uses its own
        self.concepts = {w: i for i, w in enumerate(self.words)}
        self.facts = []                        # list of dicts: {agent, action, patient, polarity}
        self._b = None                         # bridge built lazily on first store

    # ---- lazy bridge + primitives -------------------------------------------------------------------
    def _ensure(self):
        if self._b is not None:
            return
        from sim.backend import to_host, from_host
        KF = len(self._vocab)
        b = build_binder_bridge(self.seed, K=_ROLES * self.max_facts, KF=KF)
        n = b.core_config.num_neurons
        slot_idx = [_idx(b, f"w{k}") for k in range(b._K_slots)]
        fill_idx = [_idx(b, f"f{f}") for f in range(KF)]

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

    def store(self, agent, action, patient, polarity=None, attribute=None):
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

    def render_fact(self, agent, order_fn=None):
        i = self._match(cue_a=agent)
        if i is None:
            return None
        attr = self._read_word(i, 4)
        patient = self._read_word(i, 2)
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
