---
type: finding
status: finding
date: 2026-09-04
mechanism: full-live flip-gate confirmation of the linattn mouth AFTER the affect-fix — BRAIN_AFFECT_LESION vary through webapp.server.brain_chat on the exact deployed config (BRAIN_WKV_MOUTH_RECURRENCE=linattn)
lane: language (own-voice mouth / production flip gate)
seeds: [42]
verdict: NO-GO for the flip — affect is STILL NOT load-bearing on the LINATTN mouth live (byte-identical under the affect lesion) despite the affect-fix; the fix worked for ssm but not linattn. The discipline (verify the deployed config, not a proxy) caught it.
artifacts:
  - research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation.json
---

# The linattn flip is a NO-GO: affect is still hollow on the flip target, live

**Status:** NO-GO for the production flip. The affect-fix (`d798b2bf`) made affect load-bearing on the **ssm** recurrence live, but a full-live confirmation on the **linattn** recurrence — the actual flip target — shows affect is **still not load-bearing** there. Flipping would have shipped an affect-hollow mouth. This is the "probes must match the deployed config" discipline catching a proxy-verified false positive.

## Result — the confirmation, on the exact flip config

<!--derived-->
From `research/findings/raw/_affect_wkv_mouth_verify_phase4_linattn_flip_confirmation.json` (BRAIN_WKV_MOUTH_RECURRENCE=linattn, seed42 ckpt, bpe, broad scope, through `webapp.server.brain_chat`, real onebrain composer + spiking affect organ):

| flip-gate question | result |
|---|---|
| Q1 affect load-bearing (raw differs `BRAIN_AFFECT_LESION` 0 vs 1) | **FAIL — byte-identical** |
| Q1 determinism control (lesion0 vs lesion0-repeat identical) | PASS (so the null is real, not noise) |
| Q2 moat holds with the bias active (unknown topic not claimed known) | PASS |
| **FLIP_CONFIRM_GO** | **False** |

<!--derived-->
The priming turn DID establish a real positive mood (organ differential +0.040, valence sign +, appraisal hits thrilled/overjoyed/wonderful) — so this is not a mood-not-established artifact. The linattn mouth simply produced the identical string with affect active and affect lesioned.

## Why it failed (diagnosis)

The affect-fix's coupling is an **additive decode-time logit bias** (boost 5.0 over a Warriner affect lexicon). linattn is a stronger LM than ssm → a **sharper** output distribution → the bias × the realistic live valence magnitude (~0.04) is too small to flip an argmax token. ssm is more sensitive, so the same mechanism moved ssm (verified live in the affect-fix's own phase3) but not linattn. The affect-fix already noted linattn needed boost≥8 to show an effect and that ≥8 collapses it into word-salad — i.e. a fixed larger boost is not the fix; the coupling needs to scale to linattn's logit margins.

## Why this matters (the discipline worked)

The affect-fix verified linattn only at the isolated + answer_turn level (a direct valence *sweep* with strong synthetic valence) and ran the decisive full-live *lesion* test on ssm only. Both gave a false "affect works" impression. The full-live lesion on the exact flip config (linattn) is what exposed the gap. Flipping on the affect-fix's report would have shipped an affect-hollow linattn mouth as the production default.

## Next (no-defer — a method gap, not a wall)

Strengthen the linattn affect coupling to be load-bearing live without breaking fluency or the moat — a **sharpness-aware** bias (scaled to linattn's own logit spread / decision margin) rather than a fixed boost, and/or the larger appraisal-valence signal. Fix in flight (branch `research/linattn-affect-coupling-strength`). The neural coupling (fold into FewSpikeWordRead's Izhikevich gain) remains the deeper burn-down target. The flip stays held until linattn passes the full-live affect + fluency + moat gate.

## Caveat on Q2

Q2 checked only that the unknown topic is not flagged `known` (True) — it did not verify the final post-filtered abstain text. The raw free-gen still fabricates gibberish about the made-up entity; the moat's post_filter is expected to abstain on `known=False`, but that final-output check was not run here. Q2 is a partial pass; the load-bearing blocker is Q1.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -u \
    research/findings/raw/_affect_wkv_mouth_verify/phase4_linattn_flip_confirmation.py
```
