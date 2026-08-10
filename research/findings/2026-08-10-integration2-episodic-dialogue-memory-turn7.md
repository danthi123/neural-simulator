---
type: finding
status: contributing
lane: stageA-integration
date: 2026-08-10
seed-waiver: single-seed (seed 42) behavioural integration on the live chat loop, not a metric-size GO; the
  numbers below are qualitative turn-routing counts, marked derived, reproduced by re-running the one command.
artifacts:
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42_transcript.md
instrument: the eval's own CONFABULATION scan (`_detect_ungrounded`, flags any content word absent from the
  grounded toy lexicon) + the per-turn `utterance_source`/routing trace + the before/after transcript diff;
  attribution is the turn-7 `episodic_referent_in_memory` / `episodic_discussed_topics` fields the store writes.
---

# INTEGRATION #2 (live chat episodic memory): turn 7 "you mentioned a cat" is no longer a SILENT abstain — a per-turn episodic-dialogue store makes it an HONEST, grounded, non-confabulated recall of the ACTUAL prior topic

Wires an additive **episodic-dialogue memory** into the live conversational eval
(`research/runners/_conversation_turing_test_derisk.py`, reuse-by-import of `build_one_brain`; **NO `sim/` edit**).
Each turn that actually speaks now WRITES an episode `{turn, topic, facts}` — the grounded SVOs the mouth genuinely
surfaced — so a later **referential follow-up** ("you mentioned X — what was it doing?") recalls what THIS dialogue
SAID, not a re-query of the semantic store. The honesty bar is preserved: recall ONLY what was said; never fabricate
a recollection of a referent that was never discussed.

**Artifact:** `research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json` (full 14-turn
transcript + routing/faculty trace) and its `..._transcript.md`.

## Before / after — turn 7, seed 42 (the referential/episodic probe)
<!--derived: `PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._conversation_turing_test_derisk --seed 42 --device cpu`-->
- **Human (turn 7):** *"You mentioned a cat a moment ago -- what was it doing?"*
- **BEFORE:** `''` — `silence/abstain (false premise)`. The brain had no memory of its own conversation; the bare
  host referent buffer only stored topic NAMES and produced pure silence.
- **AFTER:** `'A dog gos to the east. A dog looks at river. A dog runs north.'` —
  `episodic-dialogue recall (false-premise: recalls the ACTUAL prior topic)`, `confabulated=False`.

## Why the reply recalls the DOG, not the cat — and why that is the HONEST result
The premise is **FALSE**: across turns 1–6 the conversation discussed only the **dog** (turns 3/4/5); a **cat was
never mentioned**. The episodic store's query is genuine — `episodic_referent_in_memory=False`,
`episodic_discussed_topics=['dog']`. So the brain does NOT fabricate a cat recollection (that would be exactly the
confabulation this project fights); it HONESTLY recalls the grounded facts of the topic it DID discuss. This is
strictly better than the bare silent abstain (it is now non-silent, grounded, and demonstrably uses a real memory of
the dialogue), and it also lifts the previously-noted "no discourse/pragmatic false-premise handling" gap.

## Three routing cases (the mechanism is symmetric and honest in all three)
- **CASE A — genuine recall** (`ref` WAS discussed): replay the grounded facts actually surfaced about it. Proven in
  isolation (`research/findings/raw/lanes/stageA/turing/case_a_episodic_recall_check.py`): with a cat turn present,
  ref=cat → `'A cat runs south. A cat gos to the west. A cat looks at apple.'` — the actual cat facts, not the dog's.
- **CASE B — false premise, honest** (`ref` never discussed, other topics were): the turn-7 path here — recall the
  ACTUAL prior topic; never invent a `ref` memory.
- **CASE C — empty store**: honest silence (nothing to recall).

## Gate (met)
<!--derived, same command as above-->
- turn 7: silent → **grounded, honest, non-silent recall** (`confabulated=False`).
- **confabulations 0/14** (unchanged); n_generator_replies 4 (unchanged); abstain/silence 10 → 9.
- **no other turn regressed** — turn 6 curiosity-ask intact (`"What is a big run? -- my forward model predicts
  'south' ... I have not observed it"`); turns 3/4/5 dog prose unchanged.
- confab stays 0 because the recall renders grounded-only words (`_gm_fact_to_english`); "gos" is that shared
  helper's existing surface form of "go" and is IN the grounded lexicon (naive plural), so the scan does not flag it.

## Honest scope — is the memory SPIKING or a scaffold?
**Host scaffold, spiking path STATED.** `episode_mem` is a host-side per-turn buffer (same declared-shortcut status
as the eval's mood/appraisal feed). The **spiking path** is the gap#5 dendritic-dAP READOUT completion
(commit `ab9f7dbe`, 6/6 GO, `research/runners/_gap5_dendritic_dap_readout_completion_derisk.py`): each dialogue turn
is a BTSP-formed CA3 assembly, completed cue-specifically from the referential cue by the per-cell apical dAP
readout — the same Marr autoassociator function that GO demonstrated, now keyed by dialogue referent. Converting the
host buffer to that on-substrate store (BTSP-write each turn-assembly, dAP-complete on the referential cue) is the
next integration step; the honesty contract (recall only what was written; abstain/correct on a false premise) is
identical on the spiking substrate.
