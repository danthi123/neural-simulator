# Fluid conversation — Phase 13 GO: kind vs instance ("the dog" a specific referent vs "dogs" the kind)

**2026-07-01 (autonomous; owner steer = all levers in parallel; the "which dog?" nuance).** Scoping
(`2026-07-01-kind-vs-instance-representation-scoping.md`, subagent): the gap is small + at the agent layer — the
composer matches facts by agent/action VALUE, so an INSTANCE is first-class the moment it has a distinct token+code.
This de-risks the #1 mechanism. Reuse-by-import; **NO `sim/` edit**; CPU (brain-only).

## The mechanism
- An **instance** = a fresh token (dog_one) with its own composer code, linked to its KIND by a stored **`isa`** fact
  (`dog_one isa dog`) + carrying its own episodic facts (`dog_one is brown`).
- **Instance-first / kind-fallback** query: `what_does(instance, verb)`; on a miss, retry against the isa-parent →
  the instance **inherits** the kind's facts.
- **Definite/generic routing:** "the dog" (definite) → the discourse-current instance; "dogs" (generic) → the kind.
- Biology: semantic (kind, cortex) vs episodic (instance, hippocampus); object-file / DRT discourse referents;
  isa-inheritance (Collins-Quillian).

## Result — GO (3 seeds)
The owner's example, delivered: *"I saw a dog. the dog was brown. what is the dog? → brown; what do dogs eat? → meat."*
- **INSTANCE own-fact:** "the dog" (definite) → **brown** (the instance's OWN fact, from its own store, `src=own`) —
  NOT the kind's "mammal".
- **INHERITANCE:** "what does the dog eat?" (instance) → **meat** (`src=inherited`, via the isa link to the kind).
- **GENERIC:** "what do dogs eat?" (kind) → **meat**.
- **DISTINCTNESS:** dog_one is brown, dog_two is black — distinct instances, no cross-leak.
- **ISA-LESION** (remove the isa link) → inheritance FAILS (→ None) — the isa link is load-bearing.
- **MOAT:** an unknown instance ("otter_one") → abstain.

## Honest ceiling
- A lightweight instance token + isa-inheritance + definite/generic routing, at the conversational layer
  (reuse-by-import). Inheritance is via an explicit symbolic **isa** link (like DRT), NOT code-similarity
  generalization (the separate PPMI/dendritic frontier — not reopened).
- A **perceived/consolidated episodic** instance ("the specific dog I saw on my walk") = the engram-tag/hippocampal
  path (composes with the Tier-3 live-and-remember loop) — a follow-on.
- **Multiple co-present** instances (which of two held "dogs" a bare pronoun binds) = the already-mapped
  biased-competition WTA drop-in.

## Where this sits (the owner's "all levers, parallel" batch)
- **Phase-12 (GO):** knowledge-acquisition pipeline (learn a real-fact corpus, staged cumulatively). [breadth]
- **Phase-13 (this, GO):** kind vs instance ("the dog" vs "dogs"). [reference]
- **Broader render fine-tune (parallel track, in flight):** more verbs render → more learned facts render fluently.
- Together with Phase-10 (discussion) + Phase-11 (richness scales) → the grounded-growth conversational path.

**Artifacts:** `research/runners/_fluidconv_phase13_instance_representation_derisk.py`; result
`research/findings/raw/_fluidconv_phase13_instance_representation.json`.
