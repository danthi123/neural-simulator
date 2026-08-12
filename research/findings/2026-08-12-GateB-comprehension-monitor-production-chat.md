---
status: live
lane: gap#1
date: 2026-08-12
type: finding
integration_faculty: comprehension-monitor
seed-waiver: production-INTEGRATION verify of an already-6/6-GO faculty (the D4 de-risk `2026-08-12-spiking-comprehension-success-monitor-GO.md`, AUC=1.000 all 6 seeds). This doc verifies the deterministic WIRING glue on the real /api/brain-chat handler (single process, one seed=42 organ), not a new statistical GO; the 6-seed evidence is the cited de-risk. Lesion + flag-off arms are decisive on the single wired seed.
---

# Gate-B (D4): COMPREHENSION MEASUREMENT wired into the production /api/brain-chat turn — the brain honestly abstains ("I didn't follow that") when its role-binding doesn't resolve, default-on, moat-STRENGTHENING, lesion-load-bearing (WIRED)

**Date:** 2026-08-12
**Status:** GO / WIRED (production-integration). Single-process synchronous in-process verify on the real
`/api/brain-chat` handler (`SIM_BACKEND=numpy`, GPU-free stub renderer, `rich=False` single-fact path — the D4 gate
runs BEFORE the rich/single split, so this is the identical D4 code path; composer-independent). All 7 verify checks pass.

## What changed

Before acting on an incoming TRANSITIVE ASSERTION, the brain now reads a genuinely-SPIKING signal of whether its
**role-binding RESOLVED** — the owner's "measurement of understanding of spoken language". On a LOW margin (an
out-of-vocabulary or content-ambiguous input whose thematic roles the substrate could not resolve) the brain honestly
**abstains** — *"My role-binding didn't resolve on that — I couldn't tell which word plays which role, so I didn't
follow it."* — instead of silently ingesting the utterance. This **STRENGTHENS the no-confab moat** (it never weakens
it): the brain refuses to store/answer on content it did not comprehend.

Additive, no `sim/` edit. Two pieces:

- **`research/runners/comprehension_production_organ.py`** (new) — the production-integration glue. It REUSES the
  adversarially-verified D4 faculty (`_spiking_comprehension_monitor_derisk.py`, 6/6 GO, type-2 AUC=1.000,
  lesion→0.500): the on-brain `SpikingRoleCompetition`'s two Wong-Wang accumulator pools (`sel_agent`/`sel_patient`,
  mutual inhibition), driven by the SEMANTIC (animacy+verbfit) cues only, settle to a firing margin
  `|agentEv_0 − agentEv_1|` read off `bridge.cp_firing_states`. HIGH when the content decisively separates the two
  nouns; LOW on ambiguity/OOV. The host `_semantic_contrast` dot-product (the shortcut this replaces) is never called.
  A build-time calibration (a small deterministic battery) sets the well-vs-ill threshold; a **hard per-turn reset**
  from a resting snapshot makes each read history-INDEPENDENT (the NMDA-slow sel pools don't fully quiesce in the
  internal 8-step soft reset). Threshold ≈ **0.249** (min_well ≈ 0.331, max_ill ≈ 0.167 — a clean gap, matching the
  de-risk's AUC=1.000).
- **`webapp/server.py`** `brain_chat` — per turn, BEFORE the rich/single split: judge the message. SCOPE
  (non-regressive by construction): fires ONLY on a competent 3-content-token transitive (fully cue-COVERED — verb in
  `VERB_SELECTS` AND both nouns in `ANIMACY` — OR fully OOV). Questions (patient is the query → 2 content tokens),
  self/identity queries, anaphora, open-ended prompts, feel-queries, and real-but-untabled vocabulary are OUT OF SCOPE
  → byte-identical, unchanged. GUARD: never abstains on an `(agent,action)` the brain KNOWS (`what_does` truthy) — a
  known fact is honored (its patient-mismatch is D2's job). `BRAIN_COMPREHENSION_GATE=0` fully disables (byte-identical
  oracle); `BRAIN_COMPREHENSION_LESION=1` zeroes the learned cue→role synapses for the load-bearing test.

## Verify (SYNCHRONOUS, in-process, real `/api/brain-chat` handler, numpy-CPU, rich=False, 7/7 checks PASS, 77 s)

Artifact: `research/findings/raw/_gateB_comprehension_production_verify.json` (all numbers below). De-risk numbers
(AUC=1.000, lesion 0.500, 6/6 GO) are quoted from `research/findings/2026-08-12-spiking-comprehension-success-monitor-GO.md`. <!--derived-->

Threshold calibrated at build ≈ **0.249** (min_well ≈ 0.331, max_ill ≈ 0.167).

| input | comprehension read | behaviour |
|---|---|---|
| `what does the dog chase` (recall question) | comprehension:null (out of scope — question) | answered `The dog chases cat.` — **unregressed** |
| `what does the dragon do` (untaught) | comprehension:null (out of scope) | abstains `I don't know about that.` — **moat unregressed** |
| `the dog eats the bone` (well-formed transitive) | m=**0.338** ≥ thr → **comprehended=True** | **PASSES** the comprehension gate (no comp-abstain) |
| `wug blicket toma` (OOV) | m=**0.026** < thr → comprehended=False | honest **"my role-binding didn't resolve — I didn't follow it"** abstain |
| `apple push rock` (two-inanimate ambiguous) | m=**0.142** < thr → comprehended=False | honest **didn't-follow** abstain |
| `dog chase cat` (ambiguous BUT a KNOWN fact) | m=**0.088** < thr, **known=True** | **HONORED** → answered `The dog chases cat.` — **no false abstain** (the guard) |

**LESION (`BRAIN_COMPREHENSION_LESION=1`):** zeroing the learned cue→role synapses collapses the margin on the
well-formed input **0.338 → 0.000** (comprehended flips True→False → the comprehensible input now reads as
not-understood). The host cue VALUES are byte-identical with/without the lesion, so the discrimination is caused by
the learned spiking competition — **load-bearing**.

**FLAG-OFF (`BRAIN_COMPREHENSION_GATE=0`):** `comprehension` is null on every input and no comprehension abstain
fires — the OOV `wug blicket toma` is rendered exactly as the pre-wire brain would (`The wug blickets toma.`),
i.e. the byte-identical oracle. This makes visible what the wire STRENGTHENS: with the gate OFF the brain blindly
ingests OOV nonsense; with it ON (default) the brain honestly refuses.

## Honest residuals (declared; each rides an existing burn-down row)

- **CO-RESIDENT:** the comprehension monitor runs on its OWN `SpikingRoleCompetition` bridge, ALONGSIDE the recall
  composer, not merged onto the single recall bridge — the remaining one-brain consolidation step (**burn-down #1**),
  exactly as the affect organ.
- **VOCAB CEILING:** the cue lexicon (`ANIMACY`/`VERB_SELECTS`) is the toy 2-noun transitive scope; a real-but-untabled
  word the brain knows is OUT of the monitor's competence → passed through unchanged (no false abstain). Calibrating on
  a graded/near-threshold battery + a LEARNED cue lexicon is the next rung (the de-risk's mapped residual).
- **STRUCTURAL malformedness** (no verb / wrong arity) is still a host arity/shape check, not the spiking read.
- **INFLECTION** is handled by a light base-form lemmatizer (recognizes only KNOWN cue words; real OOV stays OOV).

## Repro

```
SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf   # POST /api/brain-chat (or in-process brain_chat)
#  comprehensible: "the dog eats the bone"  -> PASS (comprehended, answered)
#  OOV:            "wug blicket toma"        -> honest "I didn't follow that" abstain
#  ambiguous:      "apple push rock"         -> honest "I didn't follow that" abstain
#  known fact:     "dog chase cat"           -> honored (no false abstain)
#  lesion:  BRAIN_COMPREHENSION_LESION=1 (margin -> ~0)   ;   disable: BRAIN_COMPREHENSION_GATE=0 (byte-identical oracle)
```
