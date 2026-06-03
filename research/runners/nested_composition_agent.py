"""Nested-composition conversational agent on the phasor FHRR substrate -- the payoff of Direction A.

The capability the flat-distinct 320 substrate fundamentally CANNOT do (its dense real-Hadamard binding is
non-invertible, so it is single-binding-only): store + answer SVO facts whose slots are themselves
STRUCTURED ENTITIES -- an attributed patient ("big cat"), an embedded structure -- decoded by the resonator
network. Validated cheap-first this session: the resonator factors multi-factor products (algebra + genuine
spiking resonate-and-fire substrate + D-scaling), and a semantic nested fact "dog chase (big cat)" decodes at
100% on phasor FHRR (crosstalk-robust) where the flat decode is at chance (the 0.000-class nesting failure).

This agent exposes that capability:
  - COMPREHEND/STORE: encode each SVO fact in phasor FHRR; a slot's filler is either a flat concept code OR a
    nested product (adj ⊗ noun, or adj ⊗ adj ⊗ noun for a two-attribute entity "big red ball"); bundle the
    role-bindings.
  - QUERY/PRODUCE: unbind a role; the slot's DEPTH is detected automatically from confidence signals -- the
    flat cleanup confidence (flat vs nested), then the 2-factor reconstruction residual (one attribute vs two).
    A flat slot is a clean concept match; a one-attribute slot is factored by the 2-factor resonator; a
    two-attribute slot (the adjectives share a codebook -> permutation symmetry) is factored by the 3-factor
    resonator with random restarts selected by reconstruction residual.
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

    def __init__(self, nouns, verbs, adjs, D=1024, seed=42, flat_threshold=0.30,
                 n_iter=120, resid_threshold=0.5, n_restarts=16):
        self.nouns = list(nouns)
        self.verbs = list(verbs)
        self.adjs = list(adjs)
        self.D = int(D)
        self.seed = int(seed)
        self.flat_threshold = float(flat_threshold)
        self.resid_threshold = float(resid_threshold)   # 2-factor reconstruction residual splitting single vs multi modifier
        self.n_iter = int(n_iter)
        self.n_restarts = int(n_restarts)                # random restarts for the repeated-codebook (multi-modifier) decode
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

    @staticmethod
    def _mods(patient):
        """The modifier(s) of a tuple patient as a list: ('big','cat')->['big']; (('big','red'),'ball')->['big','red']."""
        mods = patient[0]
        return list(mods) if isinstance(mods, (tuple, list)) else [mods]

    def _filler(self, patient):
        if isinstance(patient, tuple):                      # (adjective(s), noun) -> NESTED product (1 OR 2 modifiers)
            v = self.noun_cb[patient[1]]
            for a in self._mods(patient):                   # bind each modifier in (binding is commutative)
                v = v * self.adj_cb[a]
            return v
        return self.noun_cb[patient]                        # flat noun

    def learn(self, agent, action, patient):
        """Store a fact. `patient` is a noun (flat), an (adjective, noun) tuple (one attribute), or a
        ((adjective, adjective), noun) tuple (two attributes -- 'big red ball')."""
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

    def _resonator2(self, p, K=4):
        """Factor p ~ adj ⊗ noun into (adjective, noun) by the phasor resonator. Returns (adj, noun, residual)
        where residual in [0,1] is the reconstruction quality |<adj⊗noun, p>|/D -- ~1 for a true single-modifier
        product, ~0.1 for a 2-modifier product (the model-selection signal for single vs multi)."""
        rng = np.random.default_rng(self.seed + 7)
        Ma = len(self.adjs); best = None
        for _ in range(K):
            ea = self._unit(self.AMAT.sum(1) + 0.3 * self.AMAT[:, rng.integers(Ma)])
            en = self._unit(self.NMAT.sum(1))
            for _ in range(self.n_iter):
                xa = p * np.conj(en); ea = self._unit(self.AMAT @ (self.AMAT.conj().T @ xa))
                xn = p * np.conj(ea); en = self._unit(self.NMAT @ (self.NMAT.conj().T @ xn))
            resid = float(np.abs(np.vdot(ea * en, p)) / self.D)
            a = int(np.argmax(np.abs(self.AMAT.conj().T @ ea)))
            n = int(np.argmax(np.abs(self.NMAT.conj().T @ en)))
            if best is None or resid > best[0]:
                best = (resid, a, n)
        return self.adjs[best[1]], self.nouns[best[2]], best[0]

    def _resonator3(self, p):
        """Factor p ~ adj ⊗ adj ⊗ noun into ({adjective, adjective}, noun) by the 3-factor phasor resonator with
        random restarts (the two adjective factors share a codebook -> permutation symmetry -> restarts + best
        reconstruction residual break it). Adjectives are recovered as a SET (binding is commutative; order is not
        preserved) and rendered in vocabulary order. Returns (list-of-adjectives, noun)."""
        rng = np.random.default_rng(self.seed + 11)
        Ma = len(self.adjs); n3 = max(self.n_iter, 150); best = None
        cbs = [self.AMAT, self.AMAT, self.NMAT]
        for _ in range(self.n_restarts):
            est = [self._unit(self.AMAT.sum(1) + 0.7 * self.AMAT[:, rng.integers(Ma)]),
                   self._unit(self.AMAT.sum(1) + 0.7 * self.AMAT[:, rng.integers(Ma)]),
                   self._unit(self.NMAT.sum(1))]
            for _ in range(n3):
                new = []
                for i in range(3):
                    o = np.ones(self.D, complex)
                    for j in range(3):
                        if j != i:
                            o = o * est[j]
                    new.append(self._unit(cbs[i] @ (cbs[i].conj().T @ (p * np.conj(o)))))
                est = new
            resid = float(np.abs(np.vdot(est[0] * est[1] * est[2], p)) / self.D)
            a1 = int(np.argmax(np.abs(self.AMAT.conj().T @ est[0])))
            a2 = int(np.argmax(np.abs(self.AMAT.conj().T @ est[1])))
            n = int(np.argmax(np.abs(self.NMAT.conj().T @ est[2])))
            if best is None or resid > best[0]:
                best = (resid, tuple(sorted({a1, a2})), n)
        adjs = [self.adjs[i] for i in best[1]]              # vocabulary order (canonical; order is not recoverable)
        return adjs, self.nouns[best[2]]

    def query_patient(self, agent, action):
        """"what does <agent> <action>?" -> the patient: a flat concept ("cat"), a one-attribute entity
        ("big cat"), or a two-attribute entity ("big red ball"); None if no stored fact matches (abstention --
        no confabulation). The depth (flat / one / two modifiers) is detected automatically: first the flat
        cleanup confidence, then the 2-factor reconstruction residual (high -> one modifier; low -> two)."""
        for b in self.kb:
            ag, _ = self._decode_role(b, "AGENT", self.NMAT, self.nouns)
            ac, _ = self._decode_role(b, "ACTION", self.VMAT, self.verbs)
            if ag == agent and ac == action:
                praw = b * np.conj(self.roles["PATIENT"])
                noun, conf = self._cleanup(praw, self.NMAT, self.nouns)
                if conf >= self.flat_threshold:             # a clean flat concept match
                    return noun
                p = self._unit(praw)                        # no flat match -> NESTED -> factor it
                adj, nn, resid = self._resonator2(p)
                if resid >= self.resid_threshold:           # 2-factor reconstructs well -> ONE attribute
                    return f"{adj} {nn}"
                adjs, nn = self._resonator3(p)              # poor 2-factor fit -> TWO attributes
                return " ".join(adjs + [nn])
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
        if isinstance(pa, tuple):                                      # attributed entity (one or two modifiers)
            mods = sorted(self._mods(pa), key=self.adjs.index)         # canonical vocabulary order
            patient = " ".join(mods + [pa[1]])
        else:
            patient = pa                                              # flat
        return f"{entry['agent']} {entry['action']} {patient}"

    def _fact_concepts(self, entry):
        cs = [entry["agent"], entry["action"]]
        pa = entry["patient"]
        cs += self._mods(pa) + [pa[1]] if isinstance(pa, tuple) else [pa]
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
        ("dog", "chase", "cat"),                       # flat
        ("dog", "eat", ("red", "ball")),               # NESTED: one attribute
        ("bird", "see", ("cold", "river")),            # NESTED: one attribute
        ("cat", "want", (("big", "red"), "ball")),     # NESTED: TWO attributes
        ("child", "hold", "ball"),                     # flat
    ]
    for ag, ac, pa in facts:
        a.learn(ag, ac, pa)
        if isinstance(pa, tuple):
            shown = "(" + " ".join(a._mods(pa) + [pa[1]]) + ")"
        else:
            shown = pa
        print(f"  learn: {ag} {ac} {shown}", flush=True)
    print("\n  -- what-queries (patient; flat / one attribute / two attributes -- depth auto-detected) --", flush=True)
    queries = [("dog", "chase"), ("dog", "eat"), ("bird", "see"), ("cat", "want"), ("child", "hold"), ("cat", "chase")]
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
    print("\n  -> a UNIFIED conversational agent: stores facts whose slot is itself a structured entity -- a", flush=True)
    print("     one- OR two-attribute patient ('red ball', 'big red ball'), decoded by the resonator with the", flush=True)
    print("     depth auto-detected from confidence signals, plans dialogue over those facts, and ABSTAINS on", flush=True)
    print("     the unknown -- nested composition the flat-distinct substrate fundamentally could not do.", flush=True)


if __name__ == "__main__":
    main()
