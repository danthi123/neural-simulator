---
type: finding
status: contributing
date: 2026-08-09
mechanism: subclausal-posthoc-verify
lane: stageA-generator-mouth
---

# Sub-clausal post-hoc verify catches the discourse-level confabulation the SVO moat was blind to

**One-line:** The generator mouth writes fluent prose but invents ungrounded subordinate/causal clauses
("... because it was looking for water") that the fact-level (SVO) post-hoc moat never checked — it re-parses
only a sentence's FIRST complete SVO. The conversation Turing test (main `300a867b`,
`research/findings/raw/lanes/stageA/turing/`) flagged this on turns 3/4/5. This extends the post-hoc moat to
decompose each generated sentence into ALL its propositions (main clause + every subordinate/causal clause),
map each to a `(subject, relation, object)` the store can be queried on, and DROP any proposition the neural
moat (`comp.query_patient` — the spiking VSA unbind) cannot verify. Additive, **default-off (unchanged code
path; expected unchanged — verified behaviourally, not as a byte hash)**.

## The defect (what the SVO moat missed)

`_gm_posthoc_verify` split the reply per SENTENCE and re-parsed each to ONE SVO in surface order. For
"A dog went to the east because it was looking for water." it recovers `(dog, go, east)` — which verifies —
and the causal tail `because it was looking for water` (proposition `(dog, look, water)`, NOT in the store)
rode through UNCHECKED. The motion triple verified, so the SVO moat reported 0 confabulations while the mouth
asserted causes and details the brain does not know.

## The mechanism

- **Decompose** (`_gm_split_clauses`): split each sentence at a clause-introducing connective
  (`because`/`since`/`so`/`while`/`when`/`although`/`which`/`and then`/... longest-match first) into the main
  clause + each subordinate clause. This is a HOST re-parse of the mouth's surface — the SAME declared
  honest-negative status as the existing SVO re-parse and the host text interface.
- **Map + verify** (`_gm_parse_clause`): each clause is parsed to an SVO (raw → leading-pronoun coref to the
  topic → prepended main-clause subject for an elided subject) and CHECKED against the store via
  `comp.query_patient` — the **NEURAL** moat decision (RF unbind). A clause that reads back no patient, or the
  wrong patient, is a confabulation.
- **Emit** (`_gm_reconstruct_subclausal`): the sentence is rebuilt keeping ONLY verified propositions — the
  main clause plus each verified subordinate re-attached via its connective; an ungrounded `because ...` tail
  is silently dropped. A sentence whose MAIN clause fails to verify is dropped whole (consistent with the
  per-sentence moat).
- **Wiring:** `--subclausal-verify` (default off) threads through `run_multi_turn_loop` →
  `_gm_prose_reply`. Off ⇒ the original per-sentence path runs unchanged (same code branch; expected
  unchanged — the default-off full smoke reproduces the banked GO / MOAT 475/475 / FM4 0-flips, but this is a
  behavioural equality, not a byte-level hash comparison).

## Teeth (single seed 42; command below is the 6-seed)

Reproduce: replay the captured VERBATIM confabulating-turn raw mouth text through the real neural store, BEFORE
(SVO moat) vs AFTER (sub-clausal moat):

```
SIM_BACKEND=numpy python -m research.runners._stageA_full_integration_derisk \
    --subclausal-teeth --seed 42 --no-generator-mouth --no-seam-a --no-seam-c
```

<!--derived-->
| turn | BEFORE (SVO moat, confab) | AFTER (sub-clausal moat) |
|---|---|---|
| 3 | `warmly, gladly A dog went to the east because it was looking for water. The dog looked towards the river because it was south of its current location. The dog ran north because it needed to find shelter or food.` | `warmly, gladly A dog went to the east. The dog looked towards the river. The dog ran north.` |
| 4 | `A dog went to the east because it was looking for water. The dog looked towards the river because it was south of its current location. The dog ran north because it needed to find shelter or food.` | `A dog went to the east. The dog looked towards the river. The dog ran north.` |
| 5 | `warmly, gladly A dog went to the east because it was looking for water. The dog looked towards the river because it was south of its current location. The dog ran north because it needed to find shelter or food.` | `warmly, gladly A dog went to the east. The dog looked towards the river. The dog ran north.` |

- **(a) confab caught:** every ungrounded word (`current`, `find`, `food`, `location`, `needed`, `shelter`,
  `water`) removed on all three turns; 3 subordinate clauses dropped per turn.
- **(b) grounded survives:** the true motion facts (dog → east / river / north) survive on all three turns; no
  over-suppression.
- **(c) matched pair:** `The dog went east because the dog ran north.` — subordinate `(dog, run, north)` IS
  stored → **verified=True**, emitted intact. `The dog went east because it was looking for water.` —
  subordinate maps to no store proposition → **verified=False**, dropped (emits `The dog went east.`). Same
  subordinate position isolates the verify decision.
- **(d) banked invariants intact** (default-off full smoke, `--no-generator-mouth`, verdict GO):
  MOAT `hard_moat_checked=475 hard_moat_abstains=475 false_accepts=0 manufactured=0` (475/475); FM4
  `g_eff_law_abstain_to_assert_flips=0` (`naive_path=15`, `fm4_holds=True`).

Artifacts: `research/findings/raw/lanes/stageA/stageA_full_integration_smoke_subclausal_teeth.json` (teeth) ·
`research/findings/raw/lanes/stageA/stageA_full_integration_smoke.json` (default-off invariants).

## 6-seed command (promote the single-seed smoke)

```
for s in 42 43 44 100 101 102; do
  SIM_BACKEND=numpy python -m research.runners._stageA_full_integration_derisk \
    --subclausal-teeth --seed $s --no-generator-mouth --no-seam-a --no-seam-c \
    --out research/findings/raw/lanes/stageA/stageA_subclausal_teeth_s$s.json
done
```

The captured raw mouth text is seed-42's deterministic output; the teeth logic (decompose → neural verify →
reconstruct) is seed-independent, so the 6-seed run confirms the verify decision + store query are stable
across substrate seeds. The full generator-mouth turn regeneration (GPU) with `--subclausal-verify` is the
follow-on once GPU frees.

## Honest scope / negatives

- The clause split + SVO map is a **host re-parse of surface text** (declared shortcut, same status as the
  existing SVO re-parse). What is BRAIN-BASED is the per-proposition CHECK (`query_patient` RF unbind). The
  requirement met is the deliverable one: **no unsupported proposition reaches the user.**
- This drops confabulated clauses; it does not give the brain a causal faculty. On turn 4 ("why did the dog go
  east?") the honest result is now the grounded motion facts with the invented reasons removed — the brain
  still has no genuine "why". That remains an open faculty, not a confabulation.
- A subordinate clause that maps to no in-vocab SVO at all (e.g. `because it needed to find shelter or food`)
  is treated as unverifiable and dropped — the conservative choice (an assertion the moat cannot confirm does
  not reach the user).
