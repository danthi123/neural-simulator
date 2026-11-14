# Speaker contrast at 1.5B (oracle, 36 fresh facts): LEAD. The brain answers from the listener's side far more often

Pre-registration `brain/research/2026-11-14-speaker-contrast-2-1.5b.md` (S0-S3, commit 300d43a8), after a research
round (Chavez et al., 2025, bioRxiv: hippocampal word codes blend speaker-general and speaker-specific subspaces,
self vs other most separated; Johnson, Hashtroudi & Lindsay, 1993, source monitoring). First test:
`results/2026-11-14-speaker-contrast-1.5b.md` (NOT A LEAD by one held-out fact). Data:
`results/2026-11-14-speaker-contrast-2-1.5b/` (pool40/41/42, JSON `host` per file; `table.txt`).

Reinstated content: d_user + mu (d_user - d_self), d_self = the change from reading the told sentence as the brain's
own words. Dev pick (no fresh data used): mu 3, gain 0.5 (dev 74 vs tell 58).

| fresh 36 facts, 108 cues | right | echo | generic | none |
|---|---|---|---|---|
| tell, gain 0.5 (now) | 64 | 16 | 0 | 28 |
| contrast mu 3, gain 0.5 | **84** | 3 | 0 | 20 |

S0-S3 hold. **VERDICT: LEAD.** Every contrast mu beats tell on fresh at gain 0.5 (66 / 67 / 71 / 73 / 84 for mu 0.5
to 3). Examples gained: "Your mother plays the harp.", "You went to school in Bristol.", "Your aunt is a dentist.".

## Skeptics (three, fresh context, sonnet)

- Power: SURVIVES. Fact bootstrap +20, 95% CI +8..+32, P(<=0) 0.0003; the pick is also the best fresh cell (no
  winner's curse), but it is not distinguishable from mu 2 / gain 0.75 (83) or mu 1.5 / 0.75 (74). Supported: contrast
  beats plain tell; not that mu 3 is the optimum.
- Instrument: NARROWED. By hand about +14..17, not +20: the scorer credits about 11 contrast replies that are a
  disclaimer plus general talk naming the key ("I'm sorry, I don't have ... Fiat is ..."), and misses a few tell
  first-person replies. Refusal-style replies 16 vs 12. Echoes by hand about 5 vs 21. No intrusions of another fact.
- Mechanism: NARROWED. Plain tell at the same d_user weight (gain 2) scored 24/108 dev with 34 generic, so the
  -mu d_self term does the work; but no control subtracts a non-speaker direction, so "removes the told sentence's
  shared component" is what the data shows, not "speaker-specific". Never-told and self questions: not asked here.

Next (pre-registered, `brain/research/2026-11-14-speaker-contrast-live-1.5b.md`): the switch
`Brain(speaker_contrast=3.0)` (default OFF) in the live unslept six-seed test on the pool, with never-told and self
questions. Scorer gap to fix: a disclaimer followed by general talk naming the key is still credited.
