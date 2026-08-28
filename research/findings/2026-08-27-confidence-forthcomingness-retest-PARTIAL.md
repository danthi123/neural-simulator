---
type: finding
status: live
mechanism: confidence-caps-forthcomingness
lane: introspection-self-model
integration_faculty: confidence-forthcomingness
date: 2026-08-27
artifacts:
  - research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_FINAL.json
  - research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_recalibrated.json
  - research/findings/raw/_confidence_forthcoming_retest/_sigma_sweep.json
runner: research/findings/raw/_confidence_forthcoming_retest/{verify_real_traffic_recalibrated,verify_real_traffic_recalibrated_part2,verify_real_traffic_recalibrated_part3,_sigma_sweep}.py
---

# Confidence-caps-forthcomingness RETEST: the confidence-saturation bug is CLOSED (issue #181) -- the flip STAYS default-OFF, content-exhaustion is the new, precise blocker (NO-GO on the flip)

**Verdict: the flip stays default-OFF.** The specific reason the 2026-08-27 NOGO
([`2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md`](2026-08-27-confidence-forthcomingness-chain-trace-fix-still-default-OFF-NOGO.md))
gave for staying off -- `mean_role_confidence` saturating at 1.0 on every real turn -- is now CLOSED by issue
#181's root-cause fix (margin-keyed confidence, `da84fde7c`). Retested end-to-end through the real
`/api/brain-chat` handler (`webapp.server.brain_chat`, called in-process): confidence genuinely discriminates
now, the cap MECHANISM is proven sound and lesion-attributable, but on the actual shipped tiny-demo brain's
real, un-overridden production floor, the cap never has anything to trim regardless of confidence -- a
DIFFERENT, more precisely characterized residual (content volume, not confidence) than the one #181 fixed.

## (1) Confidence genuinely discriminates on real traffic now

Through the real handler, unforced: a clean recall of "what does the brain use" reads `mean_role_confidence
= 0.60809` (`metacog.confident = True`, no hedge). The SAME real query against the SAME composer with its store
synaptic-noise-perturbed (sigma=1.3, the identical `_noise()` degradation model issue #181's own verification
used) reads `mean_role_confidence = 0.28363` -- below `ROLE_CONF_LO = 0.30` -- `metacog.confident = False`, and
the E1 honesty hedge genuinely fires in the rendered prose: *"My decision-margin reads this as low-confidence,
so take it as uncertain: The brain uses spikes. ..."* `recalled_svo` is unchanged (`["brain","use","spikes"]`)
-- this is a genuine confidence drop, not a misrecall. Neither value is 1.0; the saturation bug is gone.

**Noise is stochastic, and matters for this test**: sigma=2.2 (the level the recalibration arc's raw
`composer.query_patient` probe called "still answering") makes the FULL rich-answer-composer + VERIFY pipeline
abstain outright here -- a stricter gate than the raw probe exercised. A 3-value sigma sweep (`_sigma_sweep.py`)
found sigma=1.3 answers; even at that sigma, per-seed variance mattered (seeds 4001/4002 abstained, seed 9013
did not) -- seed 9013 was reused for every uncertain condition below for a clean comparison.

## (2) The cap mechanism is sound and lesion-attributable -- WHEN there is content to cap

Using the module's own documented `BRAIN_CONFIDENCE_FORTHCOMING_FLOOR` testing affordance (its docstring:
"useful against a small demo KB whose natural content is already exhausted well below the production floor"),
floor forced to (2,1): the CONFIDENT real turn keeps all 3 gathered facts (`granted=True`, reason
`high_confidence`); the UNCERTAIN real turn (same noised composer, same question) truncates to 2
(`granted=False`, reason `low_confidence_capped`, one elaboration dropped) -- a genuine 3-vs-2 sentence
difference driven by the SAME real evidence differential measured in (1). Both truncations are moat-safe: every
kept sentence's SVO matches the tiny-demo ground-truth fact set and carries `verified=True`.

**LESION** (`BRAIN_METACOG_LESION=1`, the SAME reused lesion the E1 hedge already uses, no separate flag): on
the identical floor=(2,1) conditions, BOTH the confident and the noised turn collapse to `confident=False` /
`kept_sentences=2` -- the 3-vs-2 difference from the paragraph above COLLAPSES to 2-vs-2. The discrimination
rides the spiking confidence margin, not a host shortcut.

## (3) But on the TRUE, un-overridden production floor, real traffic still shows NO visible difference

This is the decisive measurement. With NO floor override -- the actual out-of-the-box floor
(`NEUTRAL_SENTENCES=4`) a real user's turn gets -- the confident turn gathers 3 facts (`brain use spikes`,
`brain store memory`, `brain learn words`) and keeps all 3 (`reason: nothing_to_cap`, since 3 <= 4). The
noised/uncertain turn ALSO gathers and keeps 3 facts, hedge prepended, `reason: nothing_to_cap` again.
`n_sentences` is 3 either way. The cap never engages on this KB's real, unforced traffic, REGARDLESS of
confidence, because this tiny-demo's buffer-tier-only elaboration content (`RichAnswerComposer._chain_facts`
reads only `TieredFactStore`'s buffer, never the routed LTM shard -- a pre-existing, separately-documented
structural residual, unrelated to and unfixed by #181) never exceeds even the un-overridden floor of 4. A user
talking to the shipped brain right now would see IDENTICAL sentence counts whether the brain was confident or
not -- the exact "hollow flip" shape the owner's rule prohibits, just for a newly precise reason.

## (4) Byte-identical off, tested in-process against the current default

`BRAIN_CONFIDENCE_FORTHCOMING` set explicitly to `"0"` (not popped -- guards the pop-as-off staleness trap,
since an unset env var now means whatever `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` says) reproduces the exact
confident-control answer/sentence-count and carries no `confidence_forthcoming` key, in the SAME process as the
ON arm above.

## Decision: stay default-OFF, but the RESIDUAL changes precisely

`_CONFIDENCE_FORTHCOMING_DEFAULT_ON` stays `False`. The prior blocker (confidence never varies) is CLOSED. The
current blocker is content volume: the tiny-demo KB's buffer-tier-only elaboration content structurally cannot
exceed the production floor, so this coupling has nothing to do on real out-of-the-box traffic regardless of
how well confidence now reads. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `confidence-forthcomingness` row is
updated to reflect this (levels unchanged: `on_by_default: NO`; the row's evidence/notes now cite this retest
and name content-exhaustion, not calibration, as the live residual).

## The precise next rung

Closing this for real needs elaboration content to reach past the buffer tier (residual 1 in the ledger's own
`scaffold_retired` note -- `RichAnswerComposer._chain_facts`/`_facts_about`/`_facts_mentioning` would need to
read the routed LTM shard, not just `TieredFactStore.buffer`), OR a richer default-floor test vocabulary whose
natural multi-hop chains genuinely exceed `NEUTRAL_SENTENCES=4`. Neither is attempted here.

Artifact: research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_FINAL.json · runners in
research/findings/raw/_confidence_forthcoming_retest/.
