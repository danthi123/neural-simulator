# R1 — imperfect-English demo: the robust multi-cue parser, made VISIBLE (side-by-side vs order-only)

**Date:** 2026-06-20
**Type:** SHOWCASE build (no new mechanism). Makes the already-validated, CI-guarded robust multi-cue
Competition-Model parser visible: it comprehends IMPERFECT / non-canonical English where the default order-only
parser inverts the roles, with the no-confab moat held in BOTH.
**Status:** DONE — GPU (`SIM_BACKEND=cupy`) PASS, mirrored by a numpy smoke. Reuse-by-import; NO `sim/` edit.

---

## What this is

The robust multi-cue parser is the owner's stated conversational PRIMARY and it is already built + CI-verified
(`BrainConversationalAgent(enable_multicue_competition=True, multicue_verbs=…)` →
`research/runners/multicue_role_parser.py`; guard `tests/test_multicue_competition_agent.py`). The deep-research
gate `2026-06-20-robust-multicue-parser-deep-research.md` recommended **R1**: flip the imperfect-English DEMO to
the validated multi-cue parser — scoped to a demo, NOT the library default (to preserve numpy-CPU portability +
because the validated scope is the 2-noun transitive). This is that demo.

`research/runners/imperfect_english_demo.py` runs the SAME `BrainConversationalAgent` twice on the SAME bridge
config (rf composer + explicit vocab, no cache), differing only in the multi-cue flag:
- **order-only** (`enable_multicue_competition=False`) — the position-by-construction `(position × voice) → role`
  map (`BridgeParser._GT`); a corrupted word order corrupts the role.
- **multi-cue** (`enable_multicue_competition=True`) — the validated SPIKING role-COMPETITION: word order COMPETES
  with animacy + verb-selectional-fit, each weighted by its learned validity, so the surviving content cues carry
  the role assignment when order is degraded.

For each imperfect input (ground truth: ANIMATE agent acts on INANIMATE patient) it prints the role each agent
comprehended, then has each STORE its parse and answer a who/what query — so the divergence shows up behaviorally
too. The no-confab MOAT (an UNKNOWN-subject query → None) is verified in BOTH agents.

## The side-by-side result

The imperfect / non-canonical battery (4 sentences), per-sentence parse:

| Input | Degradation | order-only parse | multi-cue parse |
|---|---|---|---|
| `apple eat dog` | object-fronted (patient first) | agent=**apple**, patient=**dog** — WRONG (inverted) | agent=dog, patient=apple — **CORRECT** |
| `ball kick cat` | object-fronted (patient first) | agent=**ball**, patient=**cat** — WRONG (inverted) | agent=cat, patient=ball — **CORRECT** |
| `bone dog bite` | scrambled + dropped function words (verb last) | agent=**bone**, patient=**bite** — WRONG | agent=dog, patient=bone — **CORRECT** |
| `rock fox push` | scrambled + dropped function words (verb last) | agent=**rock**, patient=**push** — WRONG | agent=fox, patient=rock — **CORRECT** |

**Multi-cue 4/4 correct vs order-only 0/4** on imperfect English. (On the verb-last inputs the order-only parser
even files the VERB as the patient — a vivid "this is genuinely broken" illustration; the position map is the only
cue it has.)

Behavioral readout (content-correct who/what query, stored fact → answer): the multi-cue agent answers correctly
(e.g. `who_does("eat","apple") == "dog"`); the order-only agent abstains (None) on that query because it stored
the INVERTED fact — i.e. the brittleness is visible end-to-end, not just at the parse step.

- **Canonical control** (`wolf carry stick`, native SVO): multi-cue CORRECT — the multi-cue parser does NOT break
  the native word order.
- **No-confab MOAT**: `what_does("stick","carry")` and `who_does("chase","bird")` (unstored subject / unstored
  relation) → **None in BOTH agents**. The robustness win never comes at the cost of confabulation.

**DEMO PASS: multi-cue (4/4) beats order-only (0/4) on imperfect English, canonical control intact, moat HELD.**
GPU (`SIM_BACKEND=cupy`, seed 42) and the numpy smoke agree exactly.

## How to run

```bash
# GPU (production substrate)
SIM_BACKEND=cupy python -m research.runners.imperfect_english_demo --seed 42 \
    --out research/findings/raw/_R1_imperfect_english_demo.json

# tiny smoke (CPU)
SIM_BACKEND=numpy python -m research.runners.imperfect_english_demo --seed 42
```

Also launchable from the webapp (CHAT DEMOS section: `imperfect_english_demo`).

## Scope / honesty

- This builds NO new mechanism — it reuses the validated agent + parser (the capability + its anti-cheat controls
  are CI-guarded by `tests/test_multicue_competition_agent.py`: position-only-collapse contrast, cue-lesion,
  no-learning, permuted-cue, held-out, moat 0 breaches).
- The library default stays order-only (the demo is the scoped flip, per the deep-research R1 recommendation +
  the onebrain-320 precedent), so numpy-CPU portability is preserved.
- Validated scope is the 2-noun transitive (the de-risk scope); the moat keeps a non-decisive input safe.
- NO `sim/` edit; on `main`; PATHSPEC commit.
