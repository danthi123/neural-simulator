---
type: plan
status: live
date: 2026-06-03
---

# Content-selection / dialogue-Control layer — Implementation Plan (Milestone 1)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this
> plan task-by-task in this session.

**Goal:** Build a small structured controller that decides *what to say next* so a multi-turn
dialogue stays coherent, sitting on top of the project's validated concept substrate, and prove with
a controlled experiment that it makes dialogue measurably more coherent than no controller.

**Architecture:** One new module holds three functions that together model the brain's prefrontal
"Control" role: a **context buffer** (a fading record of which concepts have been discussed), a
**relevance score** (how strongly a candidate concept is associated with the current context), and an
**inhibition-of-return** trace (a fading record of what was just said, used to avoid repetition). A
turn = update context with the user's input, score every candidate concept by `relevance −
inhibition`, pick the best, then record what was said. All actual content (the concepts and their
associations) is reused from the existing substrate; the controller only chooses among it.

**Tech stack:** Python, NumPy, pytest. Reuse-by-import only. No edits to any `sim/` module or any
protected/frozen module. New files live under `research/runners/` and `tests/`.

**Key design correction (read first):** concept codes in this project are *orthogonal by design*
(each concept's vector is near-disjoint from the others), so cosine similarity between concept codes
carries **no** relatedness signal. Therefore "relevance" is computed from the substrate's **learned
associations** — an association graph `{concept: {associate: strength}}` derived from the stored
engram tags / KB facts (e.g., the tag `apple_big` means apple is associated with big). This is the
faithful reading of PFC relevance-biasing (the frontal system biases retrieval by associative
relatedness), and it keeps the three Control functions exactly as designed.

**Plain-language note:** terms are defined once on first use; no undefined abbreviations.

---

## Definitions (used throughout)

- **Concept:** one vocabulary item (e.g., `apple`, `big`), identified by a string name.
- **Association graph:** a dictionary `{concept: {associate_concept: strength_float}}` capturing which
  concepts the substrate has learned to associate, and how strongly. Strength is a non-negative float;
  higher = stronger association. Built from the substrate's stored associations.
- **Context buffer:** a dictionary `{concept: weight_float}` recording recently-discussed concepts,
  with weights that fade each turn (recent = heavier).
- **Said trace:** a dictionary `{concept: activation_float}` recording recently-expressed concepts,
  with activation that fades each turn. Used to penalize repetition.
- **Turn:** one step of dialogue. Input: the user's concept(s) for this turn (may be empty for an
  "elaborate further" turn). Output: the concept the controller chooses to express.
- **Coherence eval:** a controlled experiment comparing the controller against a no-controller
  baseline on multi-turn dialogues, scored by the metrics in Task 7.

---

## Files

- Create: `research/runners/content_selection.py` (the controller + association-graph builder).
- Create: `research/runners/content_selection_eval.py` (baseline, metrics, the controlled eval).
- Create: `tests/test_content_selection.py` (all unit + smoke tests).

All controller-logic tests use **small hand-built (synthetic) association graphs** so they are fast
and deterministic and need no spiking bridge. Only Task 8–10 touch a real substrate.

---

## Task 1: Context buffer

**Files:** Create `research/runners/content_selection.py`; Test `tests/test_content_selection.py`.

**Step 1: Write the failing test**

```python
from research.runners.content_selection import ContextBuffer

def test_context_buffer_decays_and_adds():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"])                 # apple enters at weight 1.0
    cb.update(["big"])                   # apple fades to 0.5, big enters at 1.0
    w = cb.weights()
    assert abs(w["big"] - 1.0) < 1e-9
    assert abs(w["apple"] - 0.5) < 1e-9

def test_context_buffer_reinforces_repeat():
    cb = ContextBuffer(decay=0.5)
    cb.update(["apple"]); cb.update(["apple"])   # 1.0 -> 0.5 then +1.0 = 1.5
    assert abs(cb.weights()["apple"] - 1.5) < 1e-9
```

**Step 2: Run to verify it fails** — `pytest tests/test_content_selection.py -k context_buffer -v`
→ FAIL (module/class missing).

**Step 3: Minimal implementation**

```python
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
```

**Step 4: Run to verify it passes.** **Step 5: Commit** (`feat: context buffer for dialogue control`).

---

## Task 2: Relevance score (association-to-context)

**Files:** Modify `research/runners/content_selection.py`; Test `tests/test_content_selection.py`.

**Step 1: Failing test**

```python
from research.runners.content_selection import relevance

def test_relevance_sums_association_to_active_context():
    graph = {"apple": {"big": 2.0, "cat": 1.0}, "big": {"apple": 2.0}}
    context = {"apple": 1.0}             # we've been talking about apple
    # 'big' is associated with apple at 2.0 -> relevance 2.0; 'cat' at 1.0 -> 1.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9
    assert abs(relevance("cat", context, graph) - 1.0) < 1e-9
    assert abs(relevance("apple", context, graph) - 0.0) < 1e-9   # not associated w/ itself here

def test_relevance_weights_by_context_strength():
    graph = {"apple": {"big": 2.0}, "river": {"big": 1.0}}
    context = {"apple": 0.5, "river": 1.0}   # river is more active
    # big: 0.5*2.0 + 1.0*1.0 = 2.0
    assert abs(relevance("big", context, graph) - 2.0) < 1e-9
```

**Step 3: Minimal implementation**

```python
def relevance(candidate: str, context: dict, graph: dict) -> float:
    """How strongly `candidate` is associated with the active context. Sum over active context
    concepts c of  context_weight[c] * association_strength(candidate, c)."""
    total = 0.0
    for c, w in context.items():
        total += w * graph.get(candidate, {}).get(c, 0.0)
    return total
```

Failing → pass → **Commit** (`feat: association-based relevance score`).

---

## Task 3: Inhibition-of-return (said trace)

**Files:** Modify both; Test.

**Step 1: Failing test**

```python
from research.runners.content_selection import SaidTrace

def test_said_trace_decays():
    st = SaidTrace(decay=0.5)
    st.mark("big")                       # big -> 1.0
    st.step()                            # fade -> 0.5
    assert abs(st.activation("big") - 0.5) < 1e-9
    assert st.activation("apple") == 0.0
```

**Step 3: Minimal implementation**

```python
class SaidTrace:
    """A fading record of what was recently said, to penalize repetition (inhibition-of-return)."""
    def __init__(self, decay: float = 0.6):
        self.decay = float(decay); self._a: dict[str, float] = {}
    def mark(self, concept: str):
        self._a[concept] = self._a.get(concept, 0.0) + 1.0
    def step(self):
        for k in list(self._a):
            self._a[k] *= self.decay
    def activation(self, concept: str) -> float:
        return self._a.get(concept, 0.0)
```

Failing → pass → **Commit** (`feat: inhibition-of-return said trace`).

---

## Task 4: Candidate selection

**Files:** Modify both; Test.

**Step 1: Failing test**

```python
from research.runners.content_selection import select_candidate

def test_select_prefers_relevant_unsaid():
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    context = {"apple": 1.0}
    said = {}                            # nothing said yet
    # big (2.0) beats cat (1.0)
    assert select_candidate(["big", "cat"], context, graph, said, lam=1.0) == "big"

def test_select_inhibits_repeats():
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    context = {"apple": 1.0}
    said = {"big": 2.0}                  # big was just said, strongly
    # big score 2.0 - 1.0*2.0 = 0.0 ; cat score 1.0 - 0 = 1.0 -> cat wins
    assert select_candidate(["big", "cat"], context, graph, said, lam=1.0) == "cat"
```

**Step 3: Minimal implementation**

```python
def select_candidate(candidates, context, graph, said, lam: float = 1.0):
    """Pick the candidate with the highest  relevance - lam * inhibition.  Deterministic tie-break by
    name so tests are stable."""
    best, best_score = None, float("-inf")
    for cand in candidates:
        score = relevance(cand, context, graph) - lam * said.get(cand, 0.0)
        if score > best_score or (score == best_score and (best is None or cand < best)):
            best, best_score = cand, score
    return best
```

Failing → pass → **Commit** (`feat: relevance-minus-inhibition selection`).

---

## Task 5: ContentSelectionController (one turn ties it together)

**Files:** Modify both; Test.

**Step 1: Failing test**

```python
from research.runners.content_selection import ContentSelectionController

def test_controller_walks_coherently_without_repeating():
    # apple links to big and cat; big links to hot. A coherent walk from apple should visit
    # big/cat (apple's associates) without repeating, staying connected.
    graph = {"apple": {"big": 2.0, "cat": 1.5}, "big": {"hot": 2.0, "apple": 2.0},
             "cat": {"apple": 1.5}, "hot": {"big": 2.0}}
    ctrl = ContentSelectionController(graph, decay=0.7, said_decay=0.6, lam=1.0)
    said = [ctrl.turn(["apple"]) for _ in range(3)]   # 3 elaboration turns from topic 'apple'
    assert said[0] == "big"               # apple's strongest associate
    assert len(set(said)) == 3            # no repeats across the 3 turns
    assert "apple" not in said            # doesn't just echo the topic
```

**Step 3: Minimal implementation**

```python
class ContentSelectionController:
    """Decides what to say each turn: update context with the input, score every known concept by
    relevance-minus-inhibition, pick the best previously-unsaid-ish concept, record it."""
    def __init__(self, graph, decay=0.7, said_decay=0.6, lam=1.0):
        self.graph = graph
        self.ctx = ContextBuffer(decay=decay)
        self.said = SaidTrace(decay=said_decay)
        self.lam = lam
        self._vocab = sorted({k for k in graph} | {a for v in graph.values() for a in v})

    def turn(self, user_concepts):
        self.ctx.update(list(user_concepts))
        self.said.step()
        context = self.ctx.weights()
        # candidates = everything except concepts currently in the active input this turn
        candidates = [c for c in self._vocab if c not in set(user_concepts)]
        said_now = {c: self.said.activation(c) for c in candidates}
        choice = select_candidate(candidates, context, self.graph, said_now, lam=self.lam)
        if choice is not None:
            self.said.mark(choice)
            self.ctx.update([choice])     # what we said becomes part of the context
        return choice
```

Failing → pass → **Commit** (`feat: ContentSelectionController turn loop`).

---

## Task 6: No-Control baseline

**Files:** Create `research/runners/content_selection_eval.py`; Test.

**Step 1: Failing test**

```python
from research.runners.content_selection_eval import BaselineSelector

def test_baseline_ignores_context_and_repetition():
    # Baseline = retrieval-only: pick the globally strongest associate of the *last input*,
    # with NO context buffer and NO inhibition -> it will happily repeat.
    graph = {"apple": {"big": 2.0, "cat": 1.0}}
    b = BaselineSelector(graph, seed=0)
    out = [b.turn(["apple"]) for _ in range(3)]
    assert out == ["big", "big", "big"]   # repeats because no inhibition / context progression
```

**Step 3: Minimal implementation** — a selector that returns the top associate of the latest input
each turn, with no context buffer and no said trace (so it has nothing to drive progression or
prevent repeats). Random tie-break seeded for determinism.

Failing → pass → **Commit** (`feat: no-control baseline selector`).

---

## Task 7: Coherence metrics

**Files:** Modify `content_selection_eval.py`; Test.

Define four metrics over a dialogue transcript (the list of concepts said, plus the association
graph). Each is a function returning a float.

```python
def on_topic(transcript, graph):
    """Mean association-strength of each said concept to the concepts said just before it
    (a running window). Higher = stays on topic."""
def non_repetition(transcript):
    """1 - (repeated_turns / total_turns). Higher = fewer repeats."""
def turn_to_turn_coherence(transcript, graph):
    """Mean association-strength between consecutive said concepts. Higher = adjacent turns relate."""
def topic_progression(transcript):
    """Fraction of turns that introduce a not-yet-said concept. Higher = keeps advancing
    (a controller that parks on one concept scores high coherence but LOW progression -> caught)."""
```

**Step 1: Failing tests** assert exact values on a tiny hand-built transcript + graph (e.g. a
non-repeating coherent walk scores `non_repetition == 1.0` and `topic_progression == 1.0`; a
parked-on-one-concept walk scores high coherence but `topic_progression` near 0). Implement minimally.
**Commit** (`feat: coherence metrics incl. topic-progression honesty guard`).

---

## Task 8: Build the real association graph from a substrate bridge

**Files:** Modify `content_selection.py` (`build_association_graph`); Test.

Reuse-by-import the existing substrate. The simplest faithful graph comes from the substrate's stored
associations: each engram tag of the form `a_b` (e.g. `apple_big`) is a learned association edge
between `a` and `b`. Strengths default to 1.0 (a later refinement can use retrieval cosine scores).

```python
def build_association_graph(tag_names) -> dict:
    """Turn the substrate's stored association tags (['apple_big','dog_river',...]) into a symmetric
    association graph {concept:{associate:strength}}. Unknown/malformed tags are skipped, not crashed."""
```

**Step 1: Failing test** on a hand-built tag list (`["apple_big","dog_river"]` →
`{"apple":{"big":1.0}, "big":{"apple":1.0}, "dog":{"river":1.0}, "river":{"dog":1.0}}`; malformed
`"weirdtag"` skipped). Implement minimally. **Commit** (`feat: association graph from substrate tags`).

> Reuse note: a loaded bridge exposes its tags via `bridge.list_engram_tags()` (see
> `compose_concept_chat.py`). The eval (Task 10) loads a real saved bridge and feeds its tag names in.
> Do **not** modify the bridge or any `sim/` module.

---

## Task 9: Integration smoke (controller on a real graph)

**Files:** Modify `content_selection_eval.py`; Test (marked slow / optional bridge).

A function `run_dialogue(graph, topic, n_turns)` that drives the controller for `n_turns` elaboration
turns from a `topic` concept and returns the transcript. The smoke test builds a graph from a small
hand-built tag list (no bridge needed for the unit test) and asserts the transcript is non-empty,
non-repeating, and stays within the topic's connected component. **Commit**
(`feat: multi-turn dialogue runner + smoke`).

A separate, non-pytest manual smoke (documented in the module docstring) loads a real saved substrate
bridge, builds its graph, runs a dialogue, and prints the transcript for human inspection (the
transcript honesty guard).

---

## Task 10 (CONTROLLER-RUN, NOT a subagent): the decisive multi-seed coherence eval

**Files:** Modify `content_selection_eval.py` (a `main()` CLI).

This is the pre-registered controlled experiment and must be run by the controller (the human-facing
agent), not a subagent, with the mandatory smell-test.

- For each of seeds 42–46: build dialogues (several topics) and run them twice — once with
  `ContentSelectionController`, once with `BaselineSelector` — on the **same** real association graph
  (from a loaded saved substrate bridge; fall back to a synthetic multi-topic graph if no bridge is
  provided, clearly labelled).
- Score every dialogue with the four metrics (Task 7).
- **Pre-registered PASS:** controller beats baseline on `on_topic`, `non_repetition`, and
  `turn_to_turn_coherence`, AND controller's `topic_progression` is not degenerate (e.g. ≥ 0.5),
  across a clear majority of seeds, by a margin outside seed noise.
- Print a per-seed table, the mean deltas, a verdict line (RESOLVES / BOUNDARY / DOES-NOT-RESOLVE),
  and one example transcript per condition for the human to read.
- **Smell-test (mandatory):** scrutinise a PASS harder than a FAIL — confirm the controller isn't
  winning trivially (e.g. by parking on one concept, which `topic_progression` is designed to catch;
  or because the baseline was crippled beyond a fair retrieval-only selector). If the controlled
  comparison is unfair, fix the baseline and re-run.
- **Honest propagation:** write the result (PASS or NEGATIVE) to a findings doc under
  `research/findings/` and commit + push both remotes. A negative coherence result is a real finding.
- Long runs (if a real bridge is loaded) use a bounded background waiter.

---

## Discipline (applies to every task)

- Strict failing-test → minimal-impl → run → commit. Frequent commits.
- Reuse-by-import only; the controller + eval are the only new modules; **no** edits to any `sim/`,
  `experiment/`, or other protected module, and no bridge modification.
- Controller logic is unit-tested on synthetic graphs (fast, deterministic); only Tasks 8–10 touch a
  real substrate.
- Plain ASCII; plain professional language in all docs and messages.
- Honest propagation of every outcome (including a negative eval) to both git remotes.
