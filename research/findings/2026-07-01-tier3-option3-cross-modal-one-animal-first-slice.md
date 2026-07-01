# Tier-3 Option 3 'cross-modal one animal' — the SHARED hunger drive tightens the CONVERSATIONAL moat

**2026-07-01 (CYCLE 742-744, autonomous night).** The third Tier-3 property: the SAME limbic drive touches BOTH
halves of the one brain. A HUNGRY brain's interoceptive drive raises the shared spiking dopamine, which sharpens the
moat-safe conversational confidence gate — so the hungry brain is measurably more conservative in conversation
(abstains more on uncertain reads, with lower error) than a sated one, while the no-confab moat holds 0 false-accepts
at every hunger level (the gate can only TIGHTEN). **Seed-42 GO — all 6 checks; the full 6-seed is [PENDING — bg
b4gtm8jez; verdict `research/findings/raw/_tier3_cross_modal_one_animal.json`].** Runner:
`research/runners/_tier3_cross_modal_one_animal_derisk.py`. **NO `sim/` edit** (the hunger→DA link is an additive
default-off `from_region_firing` rule appended to the shared dopamine modulator — a runner-layer seam).

## Why this is cheap (largely-done + one arrow)
Three of the four pieces were already validated on the merged one brain: the shared spiking `dopamine` modulator
(nav-critic default-ON), the 6/6-GO spiking hunger drive (`drive_agrp`/`drive_pomc`), and the moat-safe
`_da_confidence_gate` (DA→abstention, 6/6 GO `2026-06-18-DA-composer-precision-derisk`). The genuine residual was
**one missing arrow**: `drive_agrp` is nav-inert (zero out-edges) and fed no modulator, so hunger didn't raise DA.
Closed by the `drive_to_da` link — a `from_region_firing` rule reading `drive_agrp` appended to the shared
`dopamine` modulator's `production_rules` (verified: `from_region_firing` is a shipped rule type used by other
modulators; a modulator's rules are summed per step, so DA = the SNc term + a hunger term). Biology: **O.10 incentive
motivation** (deprivation amplifies the reward value of goal stimuli; Berridge/Toates).

## The de-risk (two composed, each-validated links)
1. **Measure the new link on the REAL merged bridge:** inject the body deficit as interoceptive current into
   `drive_agrp`/`drive_pomc`, run the bridge, read the shared `dopamine` (driven by `drive_agrp` spikes).
2. **Feed the measured DA levels into the VALIDATED DA-gate machinery** (reuse-by-import `da_to_gate` +
   `FHRRCleanupComposer` + `run_condition` from the 6/6-GO precision-gate de-risk) under matched cleanup noise.

## The six gates (seed-42, on the real merged bridge, GPU)
| gate | result | evidence |
|---|---|---|
| **hunger→DA link** | GO | DA sated **0.500** → hungry **0.583** (rises with the deficit, read off the shared modulator) |
| **drive-lesion (clean)** | GO | severing the `drive_agrp`→DA rule at the SAME deficit → DA **0.511 ≈ sated** (network matched; the deficit reaches DA only via the link) |
| **graded (dose-response)** | GO | corr(deficit, DA) ≥ 0.7 over a 5-point sweep (robust to the stochastic spiking read) |
| **gate tightens** | GO | `da_to_gate`(measured DA): g_eff **0.060 → 0.226** |
| **cross-modal behavior** | GO | hungry abstains **0.12 → 0.43** on uncertain reads with **lower** error **0.22 → 0.09** (salience-gated precision) |
| **no-confab MOAT (HARD)** | GO | **0 false-accepts at both hunger levels** — the gate can only tighten (moat-safe by construction) |

## The debug arc (honest — the mechanism held throughout; four DA-measurement methodology fixes)
The link/gate/behavioral/moat gates were GO from smoke-1; the four smoke iterations each fixed a *measurement*
issue, not the mechanism: (1) a composer-vocab `KeyError` reused pattern; (2) **carryover** — sequential DA reads on
the shared bridge didn't reset → a washout to a rest state between measures made them independent; (3) **lesion
confound** — zeroing the drive current changed the network so the SNc term drifted → the clean lesion *severs the
link* at the same deficit (network matched); (4) **brittle monotonicity** — a strict 3-point check flipped run-to-run
on the stochastic read → a corr-based graded check over a 5-point sweep. Each fix is in the runner + committed.

## Honest scope
A one-brain **property demonstration** (one drive touches both halves), not a new life — the scoping's cheap
"Phase-3.1" follow-on to the two closed slices (live-and-remember, develop-with-a-body). The DA→abstention half was
validated 2026-06-18; this adds the new hunger→DA link on the real bridge + composes them. NO `sim/` edit.

## Verdict
Seed-42 is a full 6/6-check GO — the shared hunger drive demonstrably modulates the conversational half of the one
brain, moat intact, no `sim/` edit. **[6-seed robustness PENDING — b4gtm8jez.]** On a 6/6 GO this closes the third
Tier-3 property, giving: lives+remembers (Option 1) · develops-from-lived-experience (Option 2) · one-drive-touches-
both-halves (Option 3). Remaining follow-ons: Option 2B (the 24/7 develop harness, pre-scoped + build-ready) · Option
4 (lived consolidation) · richer-world upgrades.
