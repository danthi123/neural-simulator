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
