---
type: finding
status: corrected
date: 2026-08-27
mechanism: confidence-caps-forthcomingness
lane: introspection-self-model
integration_faculty: confidence-forthcomingness
artifacts:
  - research/findings/raw/_confidence_forthcoming_prodflip/verify.json
  - research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.json
  - research/findings/raw/_confidence_forthcoming_prodflip/soak_summary_6seed.json
runner: research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.py
---

# Confidence-caps-forthcomingness: a real structural fix, still HONESTLY default-OFF (NO-GO on the flip)

## Why this doc exists
`2026-08-27-confidence-forthcomingness-production-default-GO.md` flipped `BRAIN_CONFIDENCE_FORTHCOMING`
default-ON the same day, but never checked whether the coupling fires on REAL (unpatched) production
traffic — every check it ran used FORCED evidence. Measured directly, `mean_role_confidence` was
structurally `None` on every real turn, so the coupling was a hollow flip: `confidence_forthcoming` never
even attached to a real response. The owner has a hard rule against exactly this. This doc: (1) fixes the
structural cause, (2) measures real traffic again, (3) reports what it actually found, honestly.

## The structural fix
`RichAnswerComposer._chain_facts` (`research/runners/rich_answer_composer.py`) always probes one hop past
the last successful match, exploring whether the chain continues — that IS the design. But
`OneBrainComposer.query_patient` resets `self.last_trace = None` unconditionally at entry
(`research/runners/one_brain_composer.py:1473-1495`), so the inevitable dead-end probe clobbers the
confidence trace the LAST successful hop left behind — the trace of the fact that actually got said.
Fixed: track the trace off the last hop that genuinely matched (`abstained: False`) and restore it after
the loop. This ONLY touches the composer's `last_trace` side-channel; the returned `facts` list (what the
method decides to say) is untouched by construction.

**Byte-identical verification** (check 1, `verify_chain_trace_fix.json`): on two real production-default
turns, the rendered answer, `recalled_svo`, and `n_sentences` are IDENTICAL before and after the fix
(`"the brain uses the spikes the brain stores the memory the brain learns the words"`, 3 sentences;
`"the dog chases the cat the cat eats the fish"`, 2 sentences). `chain_output_identical: true`.

## The fix works — `mean_role_confidence` is genuinely non-None now
Before this fix, `activity`/`mean_role_confidence` read `None` on essentially every real turn (the residual
disclosed, but not load-bearing-tested, in the superseded finding). After: `activity_is_none: false` and
`mean_role_confidence` returns a real float on BOTH tested real turns (`mean_role_confidence_nonNone_both:
true`). This is a genuine improvement, independent of what follows.

## But real traffic still shows NO variation — the coupling is STILL hollow, for a different reason
Measured with NOTHING patched and NOTHING forced (`BRAIN_CONFIDENCE_FORTHCOMING` and `BRAIN_METACOG` left
unset — their real shipped defaults), through the real `/api/brain-chat` handler:

| question | mean_role_confidence | metacog.confident | confidence_forthcoming.granted |
|---|---|---|---|
| "what does the brain use" | 1.0 | True | False |
| "what does the dog chase" | 1.0 | True | False |

Both real questions decode at a SATURATED confidence of **1.0** — above the metacog organ's calibrated HIGH
band (`role_conf_hi=0.52`) — so `evidence_from_role_conf` clips to 1.0 and `confident` reads `True`
unconditionally. `confidence_genuinely_varies_real_traffic: false`. No real LOW-confidence turn has ever
been observed on this brain to compare against. Separately, on BOTH turns the composer's own natural gather
(2-3 facts) never even reached the mood-coupling's floor (4 sentences with `BRAIN_AFFECT` isolated off), so
`granted` stays `False` regardless — there is nothing to cap either way on this vocabulary's short answers.
`confidence_forthcoming_key_present_real_traffic: true` (the diagnostic key now attaches, an improvement),
but the observable ANSWER TEXT a user would see is unchanged ON vs OFF on real traffic, exactly as before
this fix. Full artifact: `research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.json`.

## Decision: HONEST REVERT (option B)
Per the owner's hard rule (a default-on flag that never drives real conversation is the exact drift the
production-integration gate exists to stop), `_CONFIDENCE_FORTHCOMING_DEFAULT_ON` is reverted to `False` in
`webapp/confidence_forthcoming_chat.py`. `BRAIN_CONFIDENCE_FORTHCOMING=1` remains a genuine, GO-verified
opt-in (checks A-F pass with forced evidence — the MECHANISM works; only the DEFAULT is reverted).
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s `confidence-forthcomingness` row: `on_by_default: NO`.

## What's kept regardless of this decision
Two real bugs, found and fixed en route, independent of the flip's outcome:
1. `TieredFactStore.__setattr__` (`research/runners/tiered_fact_store.py`) — the B3 activity-trace flip
   never reached the buffer composer on the `+LTM` production default since the 2026-08-26 knowledge-core
   flip. Fixed; benefits the ALREADY-default-on E1 metacog hedge too.
2. `RichAnswerComposer._chain_facts` trace preservation (this doc) — benefits the E1 hedge identically,
   whenever a real turn's confidence happens to fall inside the calibrated LOW/HIGH band.

## The precise next rung (not attempted here)
The metacog organ's `ROLE_CONF_LO`/`ROLE_CONF_HI` calibration band (0.35/0.52) was apparently tuned against
a DIFFERENT measurement context than what this fix now genuinely surfaces (raw role-decode confidence on
clean, unambiguous SVO facts reads 1.0, not inside the band). Closing this needs either (a) a richer/more
ambiguous production vocabulary that naturally produces a sub-1.0, sub-0.52 decode, or (b) recalibrating the
band against genuinely-measured (not synthetic) role-decode data across a real question set. Neither is
attempted here — flagged as the honest next rung, not silently deferred.

Artifact: research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.json · runner
research/findings/raw/_confidence_forthcoming_prodflip/verify_chain_trace_fix.py.
