# Content-selection / dialogue-Control layer — design (2026-06-03)

**Goal:** add the missing **Control** function — deciding *what to say next* so dialogue stays
**coherent across turns** — on top of the project's validated spiking substrate, moving toward
tiny-LLM-like conversational capability the biology-faithful way. Agreed next arc (option B).

## Why this, and the biology

The current conversational layer is **reactive retrieval**: type a concept, it ranks and returns
associated concepts. What's missing is exactly what Hagoort's **Memory–Unification–Control (MUC)**
model calls Control — the frontal/PFC system that maintains the discourse context, biases retrieval
toward what's relevant, and suppresses what's off-topic or already-said. The substrate already
provides **Memory** (320 concepts + KB facts) and **Unification** (VSA composition). We add Control.

"Coherent across turns" decomposes cleanly into three PFC-Control functions:
- **Context buffer** — what we're talking about (PFC sustained discourse model).
- **Relevance biasing** — favor content related to the context (PFC top-down attention on Memory).
- **Inhibition-of-return** — suppress recently-said content (habituation / inhibitory control).

## Staged plan (cheap-first -> faithful)

| Milestone | Approach | Context buffer | Relevance + inhibition | Faithfulness |
|---|---|---|---|---|
| **1 (this build)** | **2: structured controller over the spiking substrate** | structured (VSA vector) | structured | mechanism-faithful, prototyped |
| 2 (next) | 3: hybrid | **spiking dlPFC region** | structured | partial spiking |
| 3 (last) | 1: fully spiking PFC Control | spiking | spiking (top-down bias + spike-adaptation) | maximum |

Each milestone re-runs the *same* coherence eval, so we know faithfulness didn't cost coherence.
This mirrors the discipline that held all session: validate the operation cheap-first; the spiking
realization is a separate, harder step (the recognizer arc showed why this ordering matters).

## Milestone 1 design (Approach 2)

A small structured `ContentSelectionController` sits **on top of** the validated substrate — it does
not replace any of it. The substrate stays the source of all actual content (concept codes, KB facts,
composed responses, existing retrieval). The controller only decides *which* content to express.

### The three components

- **Context buffer:** a running VSA context vector, a decaying superposition of concept codes
  discussed so far (recent turns heavier): `context <- normalize(gamma * context + new_concepts)`.
- **Relevance biasing:** score each candidate by `cosine(candidate_code, context)`.
- **Inhibition-of-return:** a decaying "said" trace over recently-expressed concepts; penalize a
  candidate by its said-trace activation `inhibition(candidate)`.

### One turn's data flow

1. Parse user input to concept(s) via existing retrieval/parser.
2. Fold input concepts into the context buffer.
3. Pull candidate content from the substrate (KB facts about the active concept, associated
   concepts, composed answers — the validated mechanisms).
4. Score each candidate: `relevance(candidate, context) - lambda * inhibition(candidate)`.
5. Select the top candidate(s) as the response.
6. Fold the response into the context buffer **and** the said-trace.
7. Emit. State persists across turns -- this is what makes coherence possible.

The only new code is the controller (~one module). Everything producing content is reused unchanged.

## The cheap-first test (controlled, pre-registered)

Run multi-turn dialogues on a loaded substrate bridge under two conditions: **with** the controller
vs a **no-Control baseline** (existing retrieval, no context buffer, no inhibition). Three proxy
metrics per dialogue, averaged across turns:

- **On-topic:** mean `cosine(selected_content, running_context)` -- Control higher.
- **Non-repetition:** fraction of turns repeating earlier content -- Control lower.
- **Turn-to-turn coherence:** mean `cosine(turn_t, turn_{t-1})` -- Control higher.

**Pre-registered PASS:** Control beats baseline on all three, multi-seed (3-5 seeds), by a clear
margin (not noise).

### Honesty guard (or the test is circular)

The controller *directly* optimizes relevance and non-repetition, so beating baseline on those alone
is near-tautological. So the test ALSO requires:
- **Topic-progression check:** the controller must keep *advancing* (introducing new on-topic
  content), not collapse onto one concept forever (which would trivially maximize coherence). Measure
  coherence **and** novelty together; a degenerate stay-on-one-concept controller fails.
- **Transcript inspection:** the multi-turn dialogues must actually *read* coherently to a human,
  not merely score well.

The proxies validate that the three Control functions do their job on the real substrate (the
cheap-first goal). Genuine human-judged coherence at scale is honestly downstream.

## Testing & reuse

- **Unit tests** for each Control function: context decay/normalize, relevance scoring, inhibition
  decay, candidate selection -- deterministic given inputs.
- **Integration smoke:** a multi-turn dialogue on a real loaded bridge, end-to-end.
- **Coherence eval:** the controlled comparison above, multi-seed, with the honesty guards.

**Reuse (no new content machinery):** the loaded substrate bridge (`g20_multibridge` /
`compose_concept_chat` infrastructure) for content + concept codes; existing retrieval (multitag,
KB-QA) for candidates; VSA concept codes as the relevance vectors. The controller is the only new
module.

## Discipline

Reuse-by-import; no protected/frozen-module edits; cheap-first before spiking; controlled comparison
vs a no-Control baseline; honest proxies + honesty guards (topic-progression + transcript); honest
propagation of every outcome (including a negative) to both remotes; biology-grounded in MUC/PFC.
