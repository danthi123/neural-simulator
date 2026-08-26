---
type: finding
status: live
date: 2026-08-25
mechanism: integrated-conversational-state-rerun-post-reasoning-route-and-da-fix
lane: integration
seeds: [42]
seed-waiver: This is a QUALITATIVE integrated-state diagnostic (a single warm brain over one multi-turn
  dialogue through the real /api/brain-chat), the same single-session method as the original
  2026-08-25-integrated-conversational-state-diagnostic it is compared against — not a seeded
  generalization claim. The underlying faculties each carry their own multi-seed evidence (reasoning-route
  moat 6-seed in 2026-08-25-fhrr-decode-rate-at-scale; the framing faculties in their own de-risks). The
  behaviours reported below are deterministic reads of the live handler, quoted verbatim from the transcript.
instrument: research/findings/raw/_integrated_convo_diagnostic_2/_diag_driver_2.py -- one warm brain reused
  across 24 turns through the REAL webapp /api/brain-chat handler (in-process, SIM_BACKEND=cupy, stub
  renderer per the original's method; couplings fully exercised, only prose fluency stubbed).
artifacts:
  - research/findings/raw/_integrated_convo_diagnostic_2/transcript_2026-08-25.jsonl
  - research/findings/raw/_integrated_convo_diagnostic_2/run_log_2026-08-25.txt
---

# Integrated conversational-state diagnostic #2 — the session's work CONFIRMED live on the production brain

**The spine-test.** The original diagnostic (`2026-08-25-integrated-conversational-state-diagnostic`) FALSIFIED the
north-star: the integrated brain "recalls one fact and dresses it, never reasons," with the entire dopamine axis
silently erroring and curiosity dead. This re-run, on `main` after the reasoning-route merge (`3bb9bfdf`) and the
DA-axis fix (`dcb1c4d9`), confirms every one of this session's landings works TOGETHER, live, through the real
`/api/brain-chat`. All quotes are verbatim from the full transcript
`research/findings/raw/_integrated_convo_diagnostic_2/transcript_2026-08-25.json` (24 turns).

## Per-faculty: BEFORE (original diagnostic) → AFTER (this run)

| Faculty | Original diagnostic | This run (confirmed live) |
| --- | --- | --- |
| **Multi-hop reasoning** | ABSENT — "never derives a new conclusion"; every turn one SVO recall or abstain | **LIVE** — D3 "what does the wolf's prey eat?" → *"I derived this from: wolf hunt deer; deer eat grass. The deer eats grass."* — derives, GENERATED-framed |
| **No-confab moat over chains** | (untested — no chains) | **HOLDS** — E2 unsupported 2nd hop → abstain; E6 ambiguous multi-valued hop (eagle hunts fish AND snake) → abstain; no fabrication |
| **Chain-route load-bearing** | n/a | **YES both ways** — F1 lesion (`BRAIN_CHAIN_ROUTE=0`) → abstain; F2 restore → derivation returns |
| **Dopamine axis** | ERRORING every turn (`da_drives.reason=error:...`); DA-mode + DA-encoding + DA-curiosity all inert | **LIVE** — every turn reads `da=engaged`/`low_engagement`, ZERO `error:`; engaging msg → "— worth going further here." suffix |
| **Curiosity follow-up** | DEAD (`curious:false` on every abstain) | **BACK** — abstains now append *"My curiosity is piqued — I haven't learned about…"* (A2, C2, E2, E6, F1) |
| **Affect lead (#84)** | working | working — C1+ "Gladly — " on the enthusiastic message |
| **Topic-swap lead (#85)** | working | working — C4/D "On dog, then —" / "On wolf, then —" |
| **Continuous wander** | working | working — B4/E4 "(I'd been mulling over cat.)" / "(…bird.)" |
| **Self-initiation** | working | working — D5 empty turn → "Something's been on my mind — cat eat worm. What does cat eat?" |
| **Comprehension-repair** | working | working — D6 "say something on your mind" → "I don't know the words 'say' or 'mind' yet" |
| **Baseline recall / abstain** | working | working — A1 "The dog chases cat. The cat eats fish."; A2 unknown → abstain |

## Honest verdict

The integrated brain now **reasons to a new conclusion over facts it was taught, live and by default, while the
no-confab moat holds and the honesty framing marks the answer as its own inference** — the exact capability the
original diagnostic proved absent. The dopamine axis and curiosity, silently dead before, are restored. No framing
faculty regressed. This is the session's headline landing, confirmed on the integrated production system rather than
in isolation.

## The next wall (what this run newly reveals)

The reasoning route fires on the **possessive two-hop shape** ("what does X's ROLE V?") over **just-taught** facts.
Two boundaries are now the frontier:
1. **Reasoning does not yet reach the brain's OWN 15k knowledge.** As in the original diagnostic, the shipped
   knowledge core still needs exact underscore-token phrasing; natural questions over it don't route. So the brain
   reasons over what you teach it in-session, not over its large stored knowledge. Making the chain route (and
   recall) work over the 15k LTM via natural language is the highest-leverage next frontier.
2. **Only the possessive 2-hop surface parses.** D4 "the wolf hunts the deer and the deer eats grass" (an explicit
   compound-sentence chain) → abstain — the parser handles "X's prey" but not compound sentences, inverse relations
   ("X's predator"), or 3+ hops. Broader compositional coverage is the second frontier.

Plus the banked reasoning-route residuals (audit reqs #7/#11 lemmatizer irregular-table + shard-routing; the
`[unverified render` cosmetic leak on the single-fact non-default path; parser-truncation `task_1a5eaba8`).

## Honest scope

Single warm brain, one session, stub renderer (prose fluency stubbed; all faculty couplings exercised) — the same
method as the original, so the before/after is apples-to-apples. FUNCTIONAL conversational correlates; no phenomenal
claim. The reasoning inference runs over the phasor store (the brain's own bound facts) = legitimate substrate; the
compositional-question parser is a documented host scaffold on the emergence-bar ladder to a learned replacement.
