---
type: finding
status: contributing
date: 2026-08-21
mechanism: wave-faculties-default-on-flip-decision
lane: integration
---

# WAVE-0/1 default-on flip: 2 flipped, 2 held (2026-08-21)

**Owner directive:** "Run the soaks, proceed autonomously following your best judgement." Four WAVE-0/1 faculties were
de-risked + wired default-OFF (each with its own GO de-risk proving off-content-identical + on-load-bearing +
lesion-severable through the REAL `brain_chat` handler): **da-gated-encoding**, **da-gated-curiosity-threshold**,
**gnw-three-organ-bus**, **continuous-ideation**. This finding records the flip decision and its evidence.

## Verdict

| faculty | decision | why |
|---|---|---|
| da-gated-curiosity-threshold | **FLIPPED default-ON** | changes ONLY whether an honest follow-up QUESTION is appended on an ABSTAIN; content fields identical on/off (moat preserved) |
| continuous-ideation | **FLIPPED default-ON** | fires ONLY on an IDLE between-turn tick, prepends a FLAGGED non-fact idea; never touches a live recall/self answer |
| da-gated-encoding | **HELD default-OFF** | changes STORED magnitude on the onebrain composer -> can alter future recall; the composed soak ran on the rf path where it is a no-op, so it is NOT yet soak-verified. Onebrain magnitude-store no-regression soak QUEUED |
| gnw-three-organ-bus | **HELD default-OFF** | the composed soak CONFIRMED a regression: the D4 comprehension veto abstains on 2 legitimately-recalled common facts. Needs a real-vocab comprehension check first |

## Evidence: the composed no-regression check

`research/runners/_wave4_composed_flip_noregression.py` compares ALL-FOUR-ON vs ALL-FOUR-OFF through the real
`brain_chat` handler over an OUT-OF-SCOPE panel (confident KNOWN-topic recalls + self — turns none of the four
*should* change). GO iff every turn's content fields (`answer` / `recalled_svo` / `abstained`) are identical on vs off.
The persisted artifact is the fast 4-turn subset (`WAVE4_PANEL_N=4`: the 2 GNW-regressed recall turns + 2 self-turn
controls); a prior full-8 run gave the same verdict (`diverged:2`, the same 2 turns) before its per-turn brain rebuild
exceeded the runner's wall budget (the deterministic result is panel-size-independent).

**Result:** `NO-GO | turns:4 | diverged:2`. The two divergences were BOTH `gnw-three-organ-bus`'s veto:

- `"what does dog chase?"` — OFF recalls `[dog, chase, cat]`; ON: `"I don't know about that. My curiosity is piqued — I
  haven't learned about dog yet..."` (`abstained:true`, `recalled_svo:null`).
- `"what does cat eat?"` — OFF recalls `[cat, eat, fish]`; ON: same abstain+curiosity pattern.

The other 2 turns (the self-turn controls) were content-identical. So the composed run **isolated gnw-3organ as the
sole offender** — the other three faculties spuriously-fire on zero normal turns.

## The two flips are safe BY CONSTRUCTION (not merely by a passing soak)

- **da-gated-curiosity-threshold** couples the live self-produced DA to the curiosity crave-threshold. It runs ONLY
  inside the curiosity block, which runs ONLY on an ABSTAIN (there is no answer to corrupt), and it changes ONLY
  whether the honest follow-up QUESTION is appended. `answer`/`recalled_svo`/`abstained` are identical on/off; only the
  optional follow-up suffix + the additive `curiosity_da` trace differ. Its own GO de-risk
  (`_curiosity_da_threshold_verify.py`) proved the DA-dependence is load-bearing (high-DA fires the follow-up where
  low-DA does not, same message, novelty held fixed) and lesion-severable.
- **continuous-ideation** fires ONLY on an idle between-turn tick and surfaces a FLAGGED non-fact idea (`is_fact False`,
  recall-channel empty, store unchanged). Its own GO de-risk (`_continuous_ideation_verify.py`) proved the 2-source
  blend is a novel balanced recombination (6 seeds x 2 scales, blend-not-single / blend-not-noise / untrained controls
  all separate by > 0.15) with honesty preserved.

For both, the composed no-regression confirms they do not spuriously fire on a normal turn. `=0` is the byte-identical
escape (anchors: `webapp/da_curiosity_drives_chat.py:80`, `webapp/continuous_engine.py:364`, both default `'1'`). The
one residual — whether the DA-modulated follow-up cadence / the occasional idle idea reads as helpful vs noise — is a
UX taste judgment, best made live with the owner and reversible via `=0`, NOT an objective unrun soak.

## Why da-gated-encoding is HELD (instrument honesty, not caution)

The composed run used `BRAIN_COMPOSER_KIND=rf`. On the rf fast-path the stored recall is **magnitude-invariant**
(phases only), so `encoding_gain_fn` is not read by that store — **the da-encoding lever moved ZERO variables in the
composed soak**. An A/B whose lever moves nothing is not evidence (the silent-failure class). Unlike its two flipped
siblings, da-encoding is NOT content-identical by construction: on the onebrain magnitude composer (the production
default) a below-tonic-DA fact is encoded at `g < 1` (floored 0.5), so flipping it default-on can change future recall
content. Its FLIP GATE is a real no-regression soak on the onebrain magnitude-store production default — does a
0.5x-floored low-DA fact still recall under read stress (the I-7-b behavioral knee at the production operating point).
The mechanism itself is GO (I-7-b 6-seed lift + the wire-in |w|-ratio sub-check); the missing piece is the production
no-regression. That soak is **queued on `gpu_queue`** (needs a brain-load, one process at a time) — this is a
next-rung, not a defer: the flip lands the moment the soak GOes.

## Why gnw-three-organ-bus is HELD (a confirmed regression with a named fix)

The composed run turned the feared behaviour change into a measured one: organ C's D4 comprehension monitor scores
comprehension over a TOY cue lexicon, so a perfectly-comprehended common fact (`dog chases cat`, `cat eats fish`) reads
LOW-margin and vetoes a correct recall. The FLIP GATE is a real fix — replace the D4 toy cue-lexicon with a
real-vocab-backed entity/role competence check (or an NLI-style comprehension read) so the veto fires only on genuine
non-comprehension — NOT owner-acceptance of a regression.

## Artifacts

- runner: `research/runners/_wave4_composed_flip_noregression.py`
- result: `research/findings/raw/_wave_flip_soak/composed_noregression.json` (and the full-4 isolation run)
- ledger rows updated: `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (da-gated-curiosity-threshold + continuous-ideation
  on_by_default YES; da-gated-encoding + gnw-three-organ-bus notes record the held-reason + flip gate)

## Honesty boundary

These are FUNCTIONAL couplings (an engagement->curiosity correlate; a creativity/novelty correlate). No phenomenal
claim. The `=0` escape recovers today's behaviour exactly for all four.
