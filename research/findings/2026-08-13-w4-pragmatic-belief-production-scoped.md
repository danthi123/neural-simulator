---
type: finding
status: wired
date: 2026-08-13
lane: D-pragmatics
integration_faculty: pragmatic-implicature
mechanism: W4 depth-2 RSA graded scalar-implicature listener-belief wired as the production belief source (scalar-quantity turn class) in the default /api/brain-chat turn
runner: research/runners/_w4_pragmatic_belief_production_verify.py
seed-waiver: production-INTEGRATION verify of an already-6-seed GO faculty (the W4 graded-implicature belief: `2026-08-13-w4-detector-operating-point-homeostat-GO.md`, seeds 42/43/44/100/101/102 6/6; the faithful graded-vs-onehot belief source `_pragmatic_graded_belief_source_derisk`). This doc verifies the deterministic WIRING glue on the REAL handler (single process, one seed=42 co-resident organ) — not a new scientific GO. The default-graded / byte-identical-off / normalization-lesion / additive arms are decisive on the single wired seed.
artifacts:
  - research/findings/raw/_w4_pragmatic_prod/production_verify.json
builds_on:
  - research/findings/2026-08-13-w4-detector-operating-point-homeostat-GO.md
  - research/runners/_pragmatic_graded_belief_source_derisk.py
---

# Task-#12 — the de-risk-CLOSED W4 GRADED scalar-implicature belief is now WIRED into the production speaking pipeline as the belief source for a scalar-quantity turn class. The pipeline had NO pragmatic-implicature slot before this, so this is the honest SCOPED deliverable: the minimal genuine end-to-end path (one scalar turn class) + the mapped gap.

<!--derived-->
The belief values, the wiring, and the four verify arms below are read from
`research/runners/pragmatic_production_organ.py`, the handler edits in `webapp/server.py`, and the cited
`_w4_pragmatic_belief_production_verify` GO artifact.

## What I found FIRST (investigate-before-wiring)

<!--derived-->

The task asked to wire the graded-implicature belief in "replacing/augmenting the one-hot leg2_v2 belief the composer
uses for scalar-implicature / pragmatic responses". I code-traced the production speaking pipeline
(`webapp/server.py::brain_chat`, `research/runners/brain_chat_tui.py`, the composer) and found **NO
pragmatic-implicature slot at all**: zero references to implicature / scalar / RSA / quantifier / leg2 anywhere in the
production path. `leg2_v2` is a DE-RISK RUNNER (`_pragmatic_success_readback_leg2_v2_derisk`), never wired to
production; the composer forms no belief over interpretations. So there was no one-hot belief in production to
"replace".

Per the brief, I did NOT fabricate a slot. The honest deliverable is (a) the minimal integration point, (b) the
smallest genuine end-to-end path I can verify (one implicature-sensitive turn class), (c) the gap mapped.

## The one-hot vs graded belief (read from the substrate)

<!--derived-->

Verified via `graded_belief_sources` / `onehot_belief_sources` (seed 42, numpy-CPU, real Izhikevich RSA bridge):

| utterance | GRADED (W4, wired) | ONE-HOT (leg2_v2 WTA) | NORMALIZATION-LESION (flat) | analytic Frank-Goodman RSA |
|---|---|---|---|---|
| none | [1, 0, 0] | [1, 0, 0] | [1, 0, 0] | [1, 0, 0] |
| **some** | **[0, 0.731, 0.269]** | [0, 1.0, 0.0] | [0, 0.5, 0.5] | [0, 0.75, 0.25] |
| all | [0, 0, 1] | [0, 0, 1] | [0, 0, 1] | [0, 0, 1] |

States = {none, SBNA, all}. The GRADED belief for "some" carries the real "some -> not all" content (SBNA 0.731
preferred) **while "all" stays 0.269-possible** — matching the analytic RSA; the leg2_v2 ONE-HOT falsely rules "all"
IMPOSSIBLE (0.0). Calibration L1 to the analytic RSA: graded 0.037 vs one-hot 0.500 (~13x better). Under the
normalization-lesion (RSA_FS_EXC_W=0) the graded belief collapses to FLAT — the graded implicature content is the
substrate's FS divisive normalization, NOT host-injected (the moat).

## The wiring (organ + handler; additive; NO sim/ edit)

<!--derived-->

`research/runners/pragmatic_production_organ.py` (reuse-by-import of the W4 graded belief; the surprise-organ pattern):
`pragmatic_enabled()` (default-ON; `BRAIN_PRAGMATIC=0`), `pragmatic_lesioned()` (`BRAIN_PRAGMATIC_LESION=1` =
normalization-lesion), `extract_scalar_utterance(text)` (the host sensory boundary — maps a surface scalar term in a
partitive/probe context to the RSA utterance), a `PragmaticProductionOrgan` (built once + frozen; graded + one-hot
belief cached, lesion twin lazy), `interpret()` (the belief distribution + the enriched reading + the residual "all"
hedge), `pragmatic_notice()` (the honest functional reading).

`webapp/server.py::brain_chat`: the pragmatic block runs BEFORE the comprehension block — the scalar implicature is a
STRUCTURAL property of the quantifier, independent of whether the brain knows the content words — so on a detected
scalar-quantity turn the organ read sets `pragmatic_info` (the `pragmatic` block) + `pragmatic_prefix` (the honest
functional reading), which are surfaced on WHATEVER path the turn takes: the comprehension-repair early-return, the
rich path, and the single-fact path. A startup warm + a `_get_pragmatic_organ()` singleton mirror the other organs.
Default-ON; `BRAIN_PRAGMATIC=0` -> no reading + a null `pragmatic` block (byte-identical).

## Verify — `_w4_pragmatic_belief_production_verify` GO (REAL ChatBrain + REAL brain_chat handler, rf recall, numpy-CPU)

<!--derived-->

GO on all four arms (single process, seed=42 co-resident organ, through the EXACT handler code path;
artifact `research/findings/raw/_w4_pragmatic_prod/production_verify.json`):

- **(A) DEFAULT-ON GRADED belief source:** "I ate some of the cookies." -> `pragmatic.belief`(some) = [0, 0.731,
  0.269], implicature REPRESENTED (margin 0.463), better-calibrated to analytic RSA than one-hot (calib_l1 0.037 <
  0.500), the residual "all"-probability 0.269 retained (one-hot 0.0), the honest reading ("some but not all")
  reaches the surface. NOTE: this turn's content words ("ate"/"cookies") are OOV for the tiny-demo brain, so the
  underlying turn is a D4 comprehension-repair (an abstain) — and the reading is STILL surfaced + the block attached,
  because the scalar implicature is a STRUCTURAL property of the quantifier, independent of lexical knowledge (the
  pragmatic block runs BEFORE comprehension and is carried on the repair path). A nice separability: the brain reads
  "some" pragmatically even while asking what "cookies" refers to.
- **(B) BYTE-IDENTICAL-when-off (real handler):** a 4-turn NON-scalar recall/abstain panel is byte-identical flag-ON
  vs flag-OFF (the `pragmatic` block is null on both); a casual "some" filler ("tell me some facts", no partitive) is
  out-of-scope (no reading) — the detector is moat-safe.
- **(C) LESION-LOAD-BEARING:** `BRAIN_PRAGMATIC_LESION=1` -> belief(some) collapses to flat [0, 0.5, 0.5] (margin ~0)
  -> the reading is SUPPRESSED. The graded implicature is caused by the substrate's FS divisive normalization.
- **(D) ADDITIVE / moat-safe:** on the scalar turn the recall (abstained / recalled_svo / verified) matches flag-off
  exactly; only the reading + the `pragmatic` block differ. The pragmatic reading never manufactures a fact, never
  causes or flips an abstain.

## The honest gap (the rest needs a pragmatic turn-class first)

<!--derived-->

This wires ONE lexical scalar family {none, some, all} in a partitive/probe context. It does NOT parse arbitrary
pragmatic inference. The named next rungs (the mapped gap): (1) a general pragmatic comprehension front-end —
embedded / downward-entailing environments (where "some" does NOT implicate "not all"), non-lexical scalars, the
Q-under-discussion — that would let this belief drive arbitrary pragmatic RESPONSES (not just a prepended reading);
(2) the surface-scalar -> RSA-utterance mapping + scalar-context detection are a host language/sensory boundary (the
same class as the surprise organ's assertion extraction); (3) the graded belief is a BUILD-TIME spiking read at a
fixed operating point cached per process (plasticity off, as the W4 GO specifies) — a live per-turn re-read is
identical because there is no learning; (4) co-resident on its own RSA bridge (rides the one-brain merge, burn-down
#1). A scoped wiring + this honest gap is the deliverable; the W4 MECHANISM is de-risk-CLOSED 6/6.

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate: the brain now FORMS a graded listener-belief over interpretations for a
scalar-quantity utterance and surfaces an honest functional reading. It re-uses the de-risk-CLOSED W4 graded RSA
belief as the production belief source; it does NOT change the recall/moat/abstain, does NOT claim phenomenal access
to another mind (a self-report would be a functional read-out). numpy-CPU real spiking Izhikevich; additive;
`BRAIN_PRAGMATIC=0` byte-identical oracle; NO sim/ edit.

## Sources

- **Frank & Goodman (2012), Science 336(6084):998** — the RSA depth-2 listener posterior the graded belief encodes.
- **Grice (1975)** — scalar implicature ("some" +> "not all").
- **Carandini & Heeger (2012), Nat Rev Neurosci 13:51** — the FS divisive normalization the graded content is
  attributable to (the lesion moat).
- Builds on `2026-08-13-w4-detector-operating-point-homeostat-GO.md` (the W4 arc CLOSED 6/6) and
  `_pragmatic_graded_belief_source_derisk.py` (the faithful graded-vs-onehot belief source + the normalization-lesion moat).

## Reproduce

```
SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._w4_pragmatic_belief_production_verify
```
