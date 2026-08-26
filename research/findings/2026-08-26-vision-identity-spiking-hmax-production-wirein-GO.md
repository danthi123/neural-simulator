---
type: finding
status: live
lane: integration
date: 2026-08-26
---

# Visual object -> category identity ("spiking HMAX"): a default-OFF production consumer for the EMERGE-36 vision GO, verified load-bearing (GO)

**Status: GO — a de-risk / production-integration BUILD. The faculty is DEFAULT-OFF (`BRAIN_VISION_IDENTITY`); it is
NOT yet on by default. The parent flips it default-on after the 6-seed pool soak passes.** The claim here is: the
consumer exists, flag-OFF is byte-identical to today, and the ON path is load-bearing (verified by the faculty's own
lesion oracle).

**Date:** 2026-08-26 (autonomous, production-integration).
**Faculty:** the EMERGE-36 fully-spiking perception->pooler->inference GO (6 seeds) — the owner's canonical "build a
cheap consumer for a vision GO" case.
**Flag:** `BRAIN_VISION_IDENTITY` (default **OFF**; the parent flips default-on after the 6-seed pool soak passes).
**Files:** `research/runners/vision_identity_production_organ.py` (the production organ, reuse-by-import), a guarded
BEGIN/END block in `webapp/server.py::brain_chat` (+ an optional `percept` field on `BrainChatRequest`),
`research/runners/_vision_identity_flip_soak.py` (the soak gate). **NO `sim/` edit.**

## What was wired

A percept image (the ENVIRONMENT's retinal render) now drives the production `/api/brain-chat` turn: on a
'what do you see?'-class message that CARRIES a percept (`req.percept`), the brain SEES the object through the real
`sim.visual_cortex` Gabor/V1 front end -> a spiking Marr-Albus coincidence-column pooler on a real `SimulationBridge`
(`coincidence_weighted_drive`, **NO numpy kWTA**) -> reads the winning self-organized category column block as the
recognized-object identity, and the recognized concept SEEDS the turn's answer: **"I see a &lt;recognized-object&gt;.
It can &lt;property&gt;."** (and writes the recognized concept as the discourse referent so a follow-up can reason
about it). The organ is a reuse-by-import of `_emerge36_spiking_perception_pipeline_derisk.SpikingPerceptionProbe`
(which composes EMERGE-34 Gabor/V1 + EMERGE-35 spiking codon + EMERGE-14 on-bridge inheritance).

**BRAIN-BASED-ONLY boundary** (CLAUDE.md standing standard): host code is legitimate here only for the ENVIRONMENT —
rendering the retinal image the neural retina/V1 receive. Everything between sensation and the recognized identity —
Gabor/V1, the coincidence-column pooler, the codon->property inheritance — is neurons/synapses on the bridge (no
numpy kWTA). Naming the recognized category with an object noun is the environment/body label layer (same status as
the finding's CATPROP tag).

## Flag-OFF is byte-identical (asserted in the data — SHA1 hash-equal through the real handler)

`BRAIN_VISION_IDENTITY` unset -> the wiring block's cheap env read is the ONLY thing that runs; it imports nothing and
returns nothing. The block ALSO only executes when `is_visual_query(msg) AND req.percept` (a short-circuit `and`), so
an ordinary turn has ZERO code-path difference. Asserted **in the data** by a SHA1 hash-compare of the full
`webapp.server.brain_chat` JSON response (stub brain in the cache to isolate this block from the heavy composer boot;
all other default-ON faculties disabled identically in both arms):

- an ORDINARY turn ("what does the cat eat?") — flag-OFF hash == flag-ON hash (**equal**);
- a visual query WITHOUT a percept ("what do you see?") — flag-OFF hash == flag-ON hash (**equal**);
- a visual query WITH a percept but flag-OFF carries no `vision_identity` key (the host path).

The guard predicate is also unit-verified: `is_visual_query` is True for "what do you see?" and **False** for "what
does the dog eat?"/"tell me about cats"; `resolve_percept` returns None for an absent/unknown percept.

## LOAD-BEARING (vary -> the answer differs; lesion -> it vanishes, hash-equal to flag-off)

The coupling is percept -> spiking recognition -> answer content. Verified through the real handler (same stub-brain
byte-compare) AND at the organ boundary the handler returns:

- **VARY the percept** (intact recognizer, seed 42): percept `bird` -> **"I see a bird. It can fly."**
  (`recognized_category=0`); percept `fish` -> **"I see a fish. It can swim."** (`recognized_category=1`). The answers
  DIFFER and each recognizes the shown category. The category is the SPIKING read, not an echo of the input label.
- **LESION the pooler** (`BRAIN_VISION_IDENTITY_LESION=1`, coincidence detection OFF): the codon never charges ->
  `recognize()` returns -1 (ABSTAIN) for every percept -> `answer_percept` returns **None** -> the handler's guarded
  block returns nothing -> the turn FALLS THROUGH to the host path. In the data, the ON+lesion `visual+bird` response
  hash EQUALS the flag-OFF `visual+bird` hash (the same "I don't know about that." host answer) while the intact ON
  response hash DIFFERS — the visual answer **VANISHES** exactly to flag-off bytes. This is the source finding's own
  dAP/pooler lesion oracle applied at the production boundary.
- **PER-IMAGE PIXEL-SCRAMBLE** (the finding's headline lesion): within-category visual similarity destroyed -> within-
  category codon overlap COLLAPSES (mean 0.475 -> 0.105), recognition drops toward chance (mean 1.00 -> 0.556) <!--derived-->
  ; see the 6-seed table.

## Vision readout — 6-seed stability (the pool soak, PART A)

_The table + means below are rounded reads/aggregates of the cited soak artifact
`research/findings/raw/_vision_identity_prodflip/soak_summary_6seed.json`._
<!--derived-->

`SIM_BACKEND=numpy python -m research.runners._vision_identity_flip_soak --seeds 42 43 44 100 101 102`
GO gate (the source finding's methodology): PER SEED intact held-out accuracy >= 0.85 AND the pooler-lesion abstains on
every object (both deterministic); the noisy per-image scramble collapse is keyed on the 6-seed MEAN (margin >= 0.30).

| seed | intact acc | intact codon overlap | scramble acc | scramble codon overlap | pooler-lesion |
|---|---|---|---|---|---|
| 42  | 1.00 | 0.631 | 0.83 | 0.098 | abstains-all |
| 43  | 1.00 | 0.458 | 0.33 | 0.062 | abstains-all |
| 44  | 1.00 | 0.438 | 0.50 | 0.161 | abstains-all |
| 100 | 1.00 | 0.498 | 0.83 | 0.076 | abstains-all |
| 101 | 1.00 | 0.295 | 0.50 | 0.112 | abstains-all |
| 102 | 1.00 | 0.534 | 0.33 | 0.124 | abstains-all |
| **mean** | **1.00** | **0.475** | **0.556** | **0.105** | **floor** |

**GO** (the source finding's own methodology): intact held-out recognition is **1.00 on every one of the 6 seeds** and
the pooler-lesion **floors (abstains on every object) on every seed** — both deterministic per-seed gates pass. The
per-image pixel-scramble is noisy at a single seed (0.33–0.83, the small setup), so its collapse is keyed on the
**6-seed MEAN**: intact 1.00 vs scramble **0.556** (margin **0.444** >= 0.30; and the mean codon overlap collapses
0.475 -> 0.105). This matches the source EMERGE-36 finding's own scramble mean (0.56) and its stated "GO keys on the
multi-seed mean + the deterministic lesion".

## Soak (the gate the parent runs before flipping default-on)

`SIM_BACKEND=numpy python -m research.runners._vision_identity_flip_soak --seeds 42 43 44 100 101 102`
- **PART A** — the vision readout 6-seed stability above (pool-friendly numpy; the core gate).
- **PART B** — chat no-regression: flag-ON == flag-OFF on ordinary + visual-without-percept turns, through the real
  `brain_chat` handler (stub renderer). NIL-by-construction (the block is a guarded short-circuit — an ordinary turn
  never enters it), which the runner also exercises end-to-end when the webapp deps import; it degrades to a reported
  SKIP (never a false NO-GO) on a bare pool node lacking the webapp deps or when the full-handler numpy boot is too
  heavy. PART A is the core gate; run `--vision-only` to skip PART B.

## Honest scope (do NOT overclaim)

- The invariance is over **WELL-POSED SYNTHETIC category sets** (oriented-bar shape classes with within-category
  visual jitter). This is **NOT** natural-image translation-invariance (a separate, checked NO-GO). The recognized
  identity is the taught visual category (the 6-seed-GO codon->inheritance), surfaced as an object noun.
- The percept reaches the brain as an ENVIRONMENT descriptor on the request (`percept`); the neural retina/V1/pooler
  do the recognition. There is no camera/live-vision path — that is the named next rung.
- The Gabor/V1 encode is the rate-reference sensory front end; the pooler codon + inheritance run on the spiking
  bridge (no numpy kWTA) — as in the source finding.
- The discourse-referent seeding lets a follow-up resolve to the seen object only when the noun is in the brain's
  vocabulary; the primary load-bearing coupling (percept -> answer content) holds regardless.

## Provenance

Source GO: `research/findings/2026-07-02-emerge36-spiking-perception-pipeline-GO.md`,
`research/findings/2026-07-11-EMERGENT-fully-spiking-perception-codon-drives-the-ladder-6seed.md`.
Organ reuse-by-import: `research/runners/_emerge36_spiking_perception_pipeline_derisk.py` (+ EMERGE-34/35/14 helpers).
Raw soak: `research/findings/raw/_vision_identity_prodflip/soak_summary_6seed.json`.
