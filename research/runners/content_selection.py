"""Content-selection / dialogue-control layer (Milestone 1).

A small structured controller that decides *what to say next* so a multi-turn
dialogue stays coherent, sitting on top of the project's validated concept
substrate. It models the brain's prefrontal "Control" role with three functions:

  - a context buffer  (a fading record of which concepts have been discussed),
  - a relevance score  (how strongly a candidate concept is associated with the
    current context, computed from the substrate's LEARNED associations -- not
    cosine similarity of concept codes, which are orthogonal by design and so
    carry no relatedness signal),
  - an inhibition-of-return trace  (a fading record of what was just said, used
    to avoid repetition).

A turn = update context with the user's input, score every candidate concept by
`relevance - lam * inhibition`, pick the best, then record what was said. All
actual content (the concepts and their associations) is reused from the existing
substrate; the controller only chooses among it.

Reuse-by-import only; this module does not modify any sim/ / experiment/ module
or any bridge. Plain ASCII, Python 3 + NumPy.
"""


class ContextBuffer:
    """A fading record of which concepts have been discussed. weight = how 'active' a concept is in
    the current discourse; multiplied by `decay` each update, then incoming concepts add 1.0."""

    def __init__(self, decay: float = 0.7):
        self.decay = float(decay)
        self._w: dict[str, float] = {}

    def update(self, concepts):
        for k in list(self._w):
            self._w[k] *= self.decay
        for c in concepts:
            self._w[c] = self._w.get(c, 0.0) + 1.0

    def weights(self) -> dict:
        return dict(self._w)


def relevance(candidate: str, context: dict, graph: dict) -> float:
    """How strongly `candidate` is associated with the active context. Sum over active context
    concepts c of  context_weight[c] * association_strength(c, candidate), i.e. the strength of the
    edge from each active context concept to the candidate.

    Note: the edge is looked up as graph[c][candidate] (context concept -> candidate). For the
    symmetric graphs produced by build_association_graph this is equivalent to the reverse lookup,
    but the orientation matters for the asymmetric hand-built test graphs, where association edges
    are stored from the context concept outward (e.g. graph['apple']['big'])."""
    total = 0.0
    for c, w in context.items():
        total += w * graph.get(c, {}).get(candidate, 0.0)
    return total


class SaidTrace:
    """A fading record of what was recently said, to penalize repetition (inhibition-of-return)."""

    def __init__(self, decay: float = 0.6):
        self.decay = float(decay)
        self._a: dict[str, float] = {}

    def mark(self, concept: str):
        self._a[concept] = self._a.get(concept, 0.0) + 1.0

    def step(self):
        for k in list(self._a):
            self._a[k] *= self.decay

    def activation(self, concept: str) -> float:
        return self._a.get(concept, 0.0)


def select_candidate(candidates, context, graph, said, lam: float = 1.0):
    """Pick the candidate with the highest  relevance - lam * inhibition.  Deterministic tie-break by
    name so tests are stable."""
    best, best_score = None, float("-inf")
    for cand in candidates:
        score = relevance(cand, context, graph) - lam * said.get(cand, 0.0)
        if score > best_score or (score == best_score and (best is None or cand < best)):
            best, best_score = cand, score
    return best


class ContentSelectionController:
    """Decides what to say each turn: update context with the input, score every known concept by
    relevance-minus-inhibition, pick the best previously-unsaid-ish concept, record it."""

    def __init__(self, graph, decay=0.7, said_decay=0.6, lam=1.0):
        self.graph = graph
        self.ctx = ContextBuffer(decay=decay)
        self.said = SaidTrace(decay=said_decay)
        self.lam = lam
        self._vocab = sorted({k for k in graph} | {a for v in graph.values() for a in v})
        # Hard inhibition-of-return: concepts already emitted this dialogue are removed from the
        # candidate set so the controller never repeats itself. This complements the graded said-trace
        # penalty (which still shapes the soft score) and realizes the strongest form of the design's
        # "avoid repetition" goal. See Task 5 note in the implementation plan: the graded penalty alone
        # is too weak to stop the strongest associate (e.g. 'big') being re-selected every turn.
        self._already_said: set[str] = set()

    def turn(self, user_concepts):
        self.ctx.update(list(user_concepts))
        self.said.step()
        context = self.ctx.weights()
        # candidates = everything except concepts in the active input this turn AND everything we have
        # already said this dialogue (inhibition-of-return).
        excluded = set(user_concepts) | self._already_said
        candidates = [c for c in self._vocab if c not in excluded]
        said_now = {c: self.said.activation(c) for c in candidates}
        choice = select_candidate(candidates, context, self.graph, said_now, lam=self.lam)
        if choice is not None:
            self.said.mark(choice)
            self._already_said.add(choice)
            self.ctx.update([choice])     # what we said becomes part of the context
        return choice
