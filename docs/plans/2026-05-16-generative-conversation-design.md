# Generative-conversation layer — design (staged hybrid)

> **Status:** APPROVED for autonomous execution (user delegated the
> design decision + authorized design→plan→implement, 2026-05-16).
> Grounded in: the frontier survey (4 options), the FINAL-SYNTHESIS
> substrate state, the reference catalog (D.14 engram, D.13 CA3
> completion, Cluster-D hippocampal replay, G.20 Pulvermüller), and
> the realigned-plan constraints (standalone, NO external LLM,
> biology-grounded, no cheating, local-only).

## Goal & honest framing

Move the sim from *retrieval* toward *generation* — toward
LLM-like conversation — **without cheating**. Honest bound stated
up front: a pure biology-grounded sim will not match a 1–3B-param
LLM's fluency soon (scale gap is ~orders of magnitude; char-BPTT is
falsified at our scale per Phase 2.3a; no cloud/scaling allowed).
What is achievable and high-EV is a **trustworthy grounded
conversational agent** now, plus a **biology-grounded research path
to genuine novel generation** that the agent's own experience feeds.

## Architecture: two stages, one substrate

```
            user utterance
                 │
        ┌────────▼─────────┐   Stage 1 (implement now)
        │  intent parse    │   - reuse g20_multibridge parsing +
        │  (Q / stmt / req)│     the 11 shipped conversational feats
        └────────┬─────────┘
                 │
        ┌────────▼─────────────────────┐
        │ grounded retrieval            │  validated cross-bridge /
        │  + ABSTENTION GATE (AUC .990) │  multi-tag; below threshold
        └────────┬──────────────────────┘  ⇒ "I don't know" (no
                 │                           confabulation = the moat)
        ┌────────▼─────────┐
        │ productive        │  slot-grammar filled from RETRIEVED
        │ concept-grammar   │  concepts (+ negation/conj/pronoun/
        │ generation        │  tense/comparison feats) — not canned
        └────────┬─────────┘
                 │
        ┌────────▼─────────┐
        │ dialogue state    │  recent-concept working memory →
        │ (coref / follow)  │  resolves "its", "what about…"
        └────────┬─────────┘
                 ▼
            response  ──────► (logged as concept-sequence experience)
                                       │
                          ┌────────────▼──────────────┐ Stage 2
                          │ hippocampal SWR/theta      │ (designed
                          │ replay over logged concept │  now, the
                          │ sequences → learns concept │  real
                          │ transition structure       │  generative
                          │ (Cluster-D, biology, no    │  research)
                          │  external LLM, continual,  │
                          │  no catastrophic forgetting)│
                          └────────────────────────────┘
```

## Stage 1 — Grounded generative conversational agent (now)

**Components**
1. **Intent parser** — reuse `g20_multibridge` dispatch +
   `compose_concept_chat` parsing (Q vs statement vs request vs
   role-query). No new NLP; the 11 conversational features exist.
2. **Abstention-gated retrieval** — wrap the validated multi-tag /
   cross-bridge retrieval; compute the answer's confidence; if it
   does not clear the empirically-derived separation threshold
   (from `2026-05-16-G20-320-abstention-benchmark`, encoded≈796 vs
   control max≈584 → gate ≈ 650), respond "I don't know / I haven't
   learned that" rather than emit the top noisy associate. **This
   is the deliberate moat: a small LLM confabulates; this agent
   refuses.**
3. **Productive concept-grammar generator** — a small grammar with
   slots (SUBJ/REL/OBJ/ATTR/POLARITY/QTY) filled from *retrieved*
   concepts, not fixed strings. Composes from the existing feature
   set (negation, conjunction, possessive, pronoun, tense,
   comparison, yes/no). Output is grammatical, grounded, and varies
   with retrieved content — "productive" in the linguistic sense,
   bounded by grammar coverage (honest: not free prose).
4. **Dialogue-state working memory** — a small ring of recent
   (concept, role) tuples; resolves pronoun/elliptical follow-ups
   ("what about its colour?"). Pure Python state over the existing
   tag space; no architecture change.

**Data flow:** utterance → parse → retrieve(+abstain) → fill
grammar → state-update → response; every (input, retrieved-concepts,
response) tuple appended to a session concept-sequence log (Stage 2
fuel).

**Error handling:** retrieval below gate → explicit abstention;
unparseable input → clarifying prompt (not a guess); empty
substrate concept → "I don't know that word yet" + offer to learn
(the existing `:learn` continual path).

**Testing (TDD):** pure-logic units (grammar slot-filling,
abstention gate decision, dialogue-state coref resolution,
intent parse) are CPU-testable without a bridge — the bulk of
Stage 1 is deterministic orchestration over the validated
substrate, mirroring how the benchmark tools were built/tested.
One GPU smoke (scripted multi-turn conversation on an existing 320
bridge) as the integration check.

**Honest success criterion:** a multi-turn conversation where the
agent answers grounded questions, composes varied grammatical
responses from learned concepts, resolves simple follow-ups, and
**reliably abstains on the unknown** — measured, not vibes
(scripted eval + the abstention metric).

## Stage 2 — Biology-grounded concept-sequence learning (designed)

**Mechanism:** catalog Cluster-D — hippocampal sharp-wave-ripple /
theta-compressed **replay** of logged concept sequences drives STDP
over concept→concept pathways, so the ensemble learns transition
structure (what concept tends to follow what, in context). This is
the project's *own sanctioned* biology-grounded sequence-learning
route (NOT the falsified char-BPTT; NOT an external LLM). The
substrate's validated **no-catastrophic-forgetting** is the
enabler: it can accumulate sequence structure across sessions
continually. Stage 1's conversation log is the training experience.

**Why not BPTT:** Phase 2.3a documented char-level BPTT features do
NOT transfer at our scale, and scaling ~1000× is disallowed
(local-only). Concept-level replay sidesteps both: it operates on
the discrete, validated, well-separated concept representations the
substrate already provides, using a biology-grounded local rule.

**Honest risk:** in-sim sequence learning is the project's hardest,
most-NEGATIVE-prone area (composition arc). Stage 2 is therefore
explicitly *research*, scoped as its own arc with a falsifiable
success gate (held-out next-concept prediction above an n-gram
baseline, anti-cheat permuted-sequence control) — NOT promised as a
deliverable of this design. Stage 1 stands alone in value if Stage
2 underdelivers.

## What this is NOT (anti-overclaim)

- Not LLM-fluent free prose. Stage 1 is a productive grammar over
  grounded retrieval; Stage 2 is a research bet on biology-grounded
  generation, not a guarantee.
- Not a scaling play. Local-only; the value is trustworthiness +
  continual learning + groundedness, not parameter count.

## Build order

1. Stage 1 components, TDD, on `main`, over the existing validated
   320 ensemble. Ship the grounded agent + scripted eval +
   abstention-gated honesty.
2. Conversation→concept-sequence logging (Stage 2 fuel) — small,
   ships with Stage 1.
3. Stage 2 as a separate research arc with its own design/plan when
   Stage 1 + a sequence corpus exist.

## Files (anticipated)

- `research/runners/g20_generative_agent.py` (Stage 1 loop)
- `research/runners/concept_grammar.py` (productive slot-grammar)
- `tests/test_concept_grammar.py`, `tests/test_dialogue_state.py`,
  `tests/test_abstention_gate.py` (CPU TDD)
- Stage 2 gets its own `*-stage2-sequence-replay-design.md` later.
- Grounded in: `2026-05-16-generative-conversation-frontier-survey.md`,
  `2026-05-16-G20-failure-mechanism-FINAL-SYNTHESIS.md`.
