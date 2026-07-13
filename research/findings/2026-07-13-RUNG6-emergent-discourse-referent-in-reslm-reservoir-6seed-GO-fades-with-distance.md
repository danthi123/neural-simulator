# RUNG 6 (cheap-first) — emergent discourse coherence: the reslm generator's OWN reservoir carries the discourse referent (6-seed GO), and fades with distance (the D3-attractor boundary)

**Date:** 2026-07-13
**Runner:** `research/runners/_rung6_reslm_discourse_referent_derisk.py` (reuse-by-import: the reslm ladder `_emerge_reservoir_lm_derisk` + the EMERGE-67 16-word vocab; numpy-CPU; NO `sim/` edit).
**Verdict:** cheap-first GO (6-seed) at short-to-medium range + a characterized fading boundary. Honest scope below.

## The question (the emergence bar, NOT the composer scaffold)
The project already has cross-sentence coherence — but on the **composer/VSA scaffold** (`2026-06-17-cross-sentence-coherence-derisk.md`: a hand-wired referent buffer + pronominalize + `MultiTurnAgentV2.referent_at` spiking unbind). Per the emergence bar that is a scaffold to REPLACE. Rung 6's mission-critical version asks whether discourse coherence EMERGES in the learned reslm GENERATOR: after "SUBJ VERB . [distractor clause] . it", is the antecedent (which subject "it" refers to) **decodable from the reservoir's OWN state** — does the generator itself track who we're talking about, the way an LM learns coreference from text, with no hand-wired discourse module?

## The task + result (2-clause discourse over the 16 A→W words; 6-seed 42/43/44/100/101/102; chance 1/8=0.125)
"SUBJ VERB1 . DSUBJ DVERB . it" — a ridge read-out recovers the antecedent SUBJ from the reservoir state at the final token "it". A `--gap G` inserts G distractor clauses (each a DIFFERENT subject) between clause 1 and "it".

| gap=1 (6-seed) | value |
|---|---|
| **RESERVOIR referent-track** | **0.997** (all seeds ≥0.988) |
| memoryless BAG-of-prefix (same positions, no order) | 0.539 (≈ 2-subject chance — the bag has both subjects, can't tell which is the antecedent) |
| SHUFFLED-antecedent (break the SUBJ↔it link) | 0.125 (= chance, collapses) |
| **GO gate** (res>0.5 ∧ res>bag+0.15 ∧ shuffled<0.30) | **GO — 6/6 seeds** |

**The fade curve (seed 42, gap sweep) — the honest boundary:** reservoir **1.000 → 0.988 → 0.950 → 0.831** at gap 1→2→4→6 (bag ~0.4, shuffled at chance throughout). The reslm's fixed reservoir tracks the referent at short-to-medium range and **FADES with distance** — exactly what the D3 finding (`2026-07-09-D3-language-reference-tracking-GO.md`) established a fixed reservoir must do (EMERGE-83 retention fades; a DISCRETE ATTRACTOR is needed for UNBOUNDED tracking).

## Why this is a genuine emergent result (anti-cheats)
- **Memory is load-bearing:** the memoryless bag (which contains both subjects) can only guess between them (0.539); the reservoir's recurrent state carries the FIRST-mentioned subject forward past the distractor to 0.997. The gap makes the bag confound impossible (at gap=0 the bag trivially wins because only one subject is present — that regime is discarded).
- **Tracks the REAL referent:** shuffling the antecedent label collapses the read to chance (0.125) — the read is tied to the true SUBJ↔it link, not a spurious pattern.

## Honest scope (self-checked, adversarial-verify discipline)
- **The antecedent is always the FIRST-mentioned subject**, so this demonstrates the reservoir **carrying the first referent forward past distractors** (a real fading-memory capability the memoryless bag cannot do) — NOT yet full pronoun-ambiguity resolution (which mention is the topic when several compete). Varying which mention is the topic + genuine ambiguity is the follow-on.
- **Bounded 16-word vocab** (the EMERGE-67 A→W set, shared with Rung 5).
- The read is a linear probe of the reservoir state = the same state the reslm's next-token read-out uses, so "the generator can condition on the referent" is the fair reading.

## ⇒ Ladder status + the next mechanism
Rungs 1–5 GO + **Rung 6 short-range emergent referent-tracking (this, 6-seed GO)**. The generator carries the discourse referent emergently — the composer scaffold is not needed for short-range coherence. The **fading boundary is the next mechanism, not a wall:** wire the D3 discrete-attractor (`_d3_reference_tracking_derisk.py`, spiking K=6 attractor + FS-WTA, 6-seed GO) so the referent is tracked UNBOUNDED (the attractor re-discretizes the fading reservoir estimate each clause) — the reslm supplies the per-clause referent evidence, the attractor holds it against the fade. That is the honest Rung 6 completion.

NEXT CONCRETE ACTION: build `_rung6b_reslm_plus_d3_attractor_derisk.py` — feed the reslm's per-clause referent read into the D3 discrete-attractor; test that referent-track stays high at gap≥8 where the reservoir-alone fades (the reservoir fades to ~0.83 at gap 6; the attractor should hold ~flat). Reuse-by-import; NO `sim/` edit anticipated.

Runner: `research/runners/_rung6_reslm_discourse_referent_derisk.py` (`--gap G`, `--seeds ...`).
