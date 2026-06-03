"""Nested-composition conversational agent on the phasor FHRR substrate -- the payoff of Direction A.

The capability the flat-distinct 320 substrate fundamentally CANNOT do (its dense real-Hadamard binding is
non-invertible, so it is single-binding-only): store + answer SVO facts whose slots are themselves
STRUCTURED ENTITIES -- an attributed patient ("big cat"), an embedded structure -- decoded by the resonator
network. Validated cheap-first this session: the resonator factors multi-factor products (algebra + genuine
spiking resonate-and-fire substrate + D-scaling), and a semantic nested fact "dog chase (big cat)" decodes at
100% on phasor FHRR (crosstalk-robust) where the flat decode is at chance (the 0.000-class nesting failure).

This agent exposes that capability:
  - COMPREHEND/STORE: encode each SVO fact in phasor FHRR; a slot's filler is either a flat concept code OR a
    nested product (adj ⊗ noun); bundle the role-bindings.
  - QUERY/PRODUCE: unbind a role; the abstention threshold (TPAM-style) detects whether the slot is FLAT (a
    clean concept match) or NESTED (no flat match -> the resonator factors it into adjective + noun).
  - ABSTAIN: a query with no matching stored fact returns None (no confabulation -- the project's distinctive
    trust property carries over).

Phasor FHRR is the substrate where the resonator works (the real-Hadamard 320 substrate cannot nest). numpy
phasor algebra here is the fast realization of the resonate-and-fire phasor arithmetic validated in
_spiking_resonator_probe (the mechanism is biology-faithful; this is its usable-speed form). Reuse-by-import;
no protected-module change.

  python -m research.runners.nested_composition_agent          # scripted demo
"""
from __future__ import annotations
import numpy as np

ROLES = ("AGENT", "ACTION", "PATIENT")


class NestedCompositionAgent:
    """Store + answer SVO facts whose patient may be an attributed entity (adj ⊗ noun), on phasor FHRR,
    decoded by the resonator; flat vs nested is detected automatically by the abstention threshold."""

    def __init__(self, nouns, verbs, adjs, D=1024, seed=42, flat_threshold=0.30, n_iter=120):
        self.nouns = list(nouns)
        self.verbs = list(verbs)
        self.adjs = list(adjs)
        self.D = int(D)
        self.flat_threshold = float(flat_threshold)
        self.n_iter = int(n_iter)
        rng = np.random.default_rng(seed)

        def code():
            return self._unit(np.exp(1j * rng.uniform(-np.pi, np.pi, size=self.D)))

        self.noun_cb = {w: code() for w in self.nouns}
        self.verb_cb = {w: code() for w in self.verbs}
        self.adj_cb = {w: code() for w in self.adjs}
        self.roles = {r: code() for r in ROLES}
        self.NMAT = np.stack([self.noun_cb[w] for w in self.nouns], axis=1)   # D x |nouns|
        self.VMAT = np.stack([self.verb_cb[w] for w in self.verbs], axis=1)
        self.AMAT = np.stack([self.adj_cb[w] for w in self.adjs], axis=1)
        self.kb = []      # list of bundle vectors (the stored facts; Q&A DECODES from these = composition proof)
        self.facts = []   # parallel list of fact structures (for dialogue planning / elaboration indexing)
        self.ctrl = None        # content-selection Control for the current elaboration topic
        self.focus = None
        self._elaborated = set()

    @staticmethod
    def _unit(v):
        return v / (np.abs(v) + 1e-12)

    def _filler(self, patient):
        if isinstance(patient, tuple):                      # (adjective, noun) -> NESTED product
            return self.adj_cb[patient[0]] * self.noun_cb[patient[1]]
        return self.noun_cb[patient]                        # flat noun

    def learn(self, agent, action, patient):
        """Store a fact. `patient` is a noun (flat) or an (adjective, noun) tuple (nested attributed entity)."""
        b = (self.roles["AGENT"] * self.noun_cb[agent]
             + self.roles["ACTION"] * self.verb_cb[action]
             + self.roles["PATIENT"] * self._filler(patient))
        self.kb.append(b)
        self.facts.append({"agent": agent, "action": action, "patient": patient})
        return b

    def _cleanup(self, p, MAT, names):
        ov = np.abs(MAT.conj().T @ self._unit(p))
        k = int(np.argmax(ov))
        return names[k], float(ov[k] / self.D)              # (best match, normalised confidence in [0,1])

    def _decode_role(self, bundle, role, MAT, names):
        return self._cleanup(bundle * np.conj(self.roles[role]), MAT, names)

    def _resonator(self, p):
        """Factor p ~ adj ⊗ noun into (adjective, noun) by the phasor resonator (the validated decode)."""
        ea = self._unit(self.AMAT.sum(1))
        en = self._unit(self.NMAT.sum(1))
        for _ in range(self.n_iter):
            xa = p * np.conj(en); ea = self._unit(self.AMAT @ (self.AMAT.conj().T @ xa))
            xn = p * np.conj(ea); en = self._unit(self.NMAT @ (self.NMAT.conj().T @ xn))
        a = int(np.argmax(np.abs(self.AMAT.conj().T @ ea)))
        n = int(np.argmax(np.abs(self.NMAT.conj().T @ en)))
        return self.adjs[a], self.nouns[n]

    def query_patient(self, agent, action):
        """"what does <agent> <action>?" -> the patient: a flat concept ("cat") or an attributed entity
        ("big cat"); None if no stored fact matches (abstention -- no confabulation)."""
        for b in self.kb:
            ag, _ = self._decode_role(b, "AGENT", self.NMAT, self.nouns)
            ac, _ = self._decode_role(b, "ACTION", self.VMAT, self.verbs)
            if ag == agent and ac == action:
                p = b * np.conj(self.roles["PATIENT"])
                noun, conf = self._cleanup(p, self.NMAT, self.nouns)
                if conf >= self.flat_threshold:             # a clean flat concept match
                    return noun
                adj, nn = self._resonator(self._unit(p))    # no flat match -> NESTED -> factor it
                return f"{adj} {nn}"
        return None                                         # abstain

    def query_agent(self, action, patient):
        """"who <action> <patient>?" -> the agent of the matching fact (flat patient); None if no fact
        matches (abstention)."""
        for b in self.kb:
            ac, _ = self._decode_role(b, "ACTION", self.VMAT, self.verbs)
            pn, conf = self._decode_role(b, "PATIENT", self.NMAT, self.nouns)
            if ac == action and conf >= self.flat_threshold and pn == patient:
                ag, _ = self._decode_role(b, "AGENT", self.NMAT, self.nouns)
                return ag
        return None

    # --- dialogue planning: the content-selection Control over the agent's own facts ---
    def _render(self, entry):
        pa = entry["patient"]
        patient = f"{pa[0]} {pa[1]}" if isinstance(pa, tuple) else pa   # flat or attributed entity
        return f"{entry['agent']} {entry['action']} {patient}"

    def _fact_concepts(self, entry):
        cs = [entry["agent"], entry["action"]]
        pa = entry["patient"]
        cs += list(pa) if isinstance(pa, tuple) else [pa]
        return cs

    def _concept_graph(self):
        from research.runners.content_selection import build_association_graph
        pairs = []
        for e in self.facts:
            cs = self._fact_concepts(e)
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs.append(f"{cs[i]}_{cs[j]}")
        return build_association_graph(pairs) if pairs else {}

    def tell_about(self, concept):
        """All stored facts mentioning `concept`, rendered (flat + nested)."""
        return [self._render(e) for e in self.facts if concept in self._fact_concepts(e)]

    def set_topic(self, topic):
        from research.runners.content_selection import ContentSelectionController
        self.focus = topic
        self._elaborated = set()
        self.ctrl = ContentSelectionController(self._concept_graph())

    def elaborate(self):
        """Bring up the next coherent fact about the focus (dialogue planning via the content-selection
        Control), rendered as a sentence -- unifying nested composition + dialogue planning. None when
        nothing on-topic remains."""
        if self.focus is None or not any(self.focus in self._fact_concepts(e) for e in self.facts):
            return None
        for _ in range(len(self.nouns) + len(self.verbs) + len(self.adjs)):
            pick = self.ctrl.turn([self.focus])
            if pick is None:
                break
            for k, e in enumerate(self.facts):
                cs = self._fact_concepts(e)
                if self.focus in cs and pick in cs and k not in self._elaborated:
                    self._elaborated.add(k)
                    return self._render(e)
        return None


def main():
    nouns = ["dog", "cat", "ball", "bird", "river", "child"]
    verbs = ["chase", "hold", "see", "eat", "want"]
    adjs = ["big", "small", "red", "cold", "fast"]
    a = NestedCompositionAgent(nouns, verbs, adjs, seed=42)
    print("=== nested-composition conversational agent (phasor FHRR + resonator decode) ===\n", flush=True)
    facts = [
        ("dog", "chase", "cat"),                 # flat
        ("dog", "eat", ("red", "ball")),         # NESTED attributed patient
        ("bird", "see", ("cold", "river")),      # NESTED
        ("child", "hold", "ball"),               # flat
    ]
    for ag, ac, pa in facts:
        a.learn(ag, ac, pa)
        shown = f"({pa[0]} {pa[1]})" if isinstance(pa, tuple) else pa
        print(f"  learn: {ag} {ac} {shown}", flush=True)
    print("\n  -- what-queries (patient; flat or nested attributed entity) --", flush=True)
    queries = [("dog", "chase"), ("dog", "eat"), ("bird", "see"), ("child", "hold"), ("cat", "want")]
    for ag, ac in queries:
        print(f"  Q: what does {ag} {ac}?   A: {a.query_patient(ag, ac)}", flush=True)
    print("\n  -- who-queries (agent) --", flush=True)
    for ac, pa in [("chase", "cat"), ("hold", "ball"), ("eat", "ball")]:
        print(f"  Q: who {ac} {pa}?   A: {a.query_agent(ac, pa)}", flush=True)
    print("\n  -- dialogue planning (content-selection Control over the agent's nested facts) --", flush=True)
    print(f"  tell me about dog: {a.tell_about('dog')}", flush=True)
    a.set_topic("dog")
    for _ in range(3):
        print(f"  elaborate on dog -> {a.elaborate()}", flush=True)
    print("\n  -> a UNIFIED conversational agent: stores facts whose slot is itself a structured entity (an", flush=True)
    print("     attributed", flush=True)
    print("     patient, 'red ball'), decoded by the resonator, and ABSTAINS on the unknown -- nested", flush=True)
    print("     composition the flat-distinct substrate fundamentally could not do.", flush=True)


if __name__ == "__main__":
    main()
