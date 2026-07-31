---
type: plan
status: live
date: 2026-05-21
---

# Compositional readout via predictive-coding difference: design

> **For Claude / autonomous continuation:** This design confronts the
> blocker the eight-architecture compositional series localized but
> never addressed. After the cheap-first probe (pre-registered below)
> shows signal-or-no-signal, this proceeds to a full pre-registered
> three-state test mirroring the prior arcs' discipline. If the cheap
> probe shows no signal, that is an honest fast negative and the design
> line reaches terminal closure.

## Why this design exists

Eight architectures have been tested for compositional retrieval --
the capability of cueing a noun and recalling the attribute bound to
it ("apple" -> "red"). All eight failed at biological scale, plateauing
at roughly 0.46 retrieval accuracy where 0.80 is the bar. The eight:
static two-store retrieval, theta-multiplexed acetylcholine gating,
disinhibition-based theta, per-regime metacognitive monitoring,
cue-suppression during retrieval, generative replay plus a prefrontal
frame, aggressive consolidation, and pool-firing readout substitution.

A localization diagnostic (2026-05-20) pinned the failure precisely:
compositional retrieval emits **strong-but-wrong** answers. When cued
with a noun, the substrate produces a high-confidence output, but in
four of five test queries the output is the wrong word. The diagnosed
cause: the cue noun's broad input drive dominates the engram tag's
selective drive to the bound attribute. The readout -- a cosine
similarity over the shared language-output population -- cannot hear
the recalled attribute over the cue itself.

Critically, **all eight arcs shared one fixed readout mechanism**:
drive the cue, measure the cosine of the language-output population
against each candidate word. The arcs varied the substrate dynamics
around that readout (gating, rhythms, replay, priming). None changed
the readout computation. The sixth arc's own design document
(2026-05-20) pre-registered this as the terminal diagnosis: the
blocker is "drive cue + measure language-output cosine" and the fix
"would require ... replace the cosine readout with something more
selective."

This design does exactly that.

## The mechanism: read the deviation, not the raw response

The failure is that the cue and the recalled attribute compete in one
measurement. The fix is to measure them in two and take the
difference.

**Procedure for one compositional query (cue noun N, bound attribute A):**

1. Drive the cue noun N alone (no engram tag stimulation). Record the
   language-output population response. Call this the *baseline* -- it
   is everything the cue produces on its own, including the cue's own
   word signature and any pre-existing trained associations.

2. Drive the same cue noun N **and** stimulate the engram tag for the
   (N, A) binding. Record the language-output response. Call this the
   *bound response*.

3. The compositional readout is the **difference**: bound response
   minus baseline, per word-pattern projection. Rank candidate words by
   this difference. The recalled attribute is whichever word the engram
   tag *adds* over the cue's baseline.

The cue noun N is present in **both** measurements, so its broad drive
cancels in the difference. What survives is precisely the contribution
the engram tag makes -- the recalled attribute. Encoding-specificity
(Tulving 1973) is fully preserved: the cue context is present
throughout; nothing is suppressed. This is the dual mechanism the
cue-suppression arc's own findings concluded was needed -- "keep the
cue present ... while amplifying the engram tag's selective drive" --
achieved by a readout computation rather than a dynamics overlay.

## Why this is biology, not an engineering trick

Reading the deviation from a context-set baseline is a canonical
cortical computation, not a convenience:

- **Predictive coding** (Rao & Ballard 1999): cortex represents the
  *error* between input and prediction. The cue establishes the
  predictive context; the engram tag's contribution is the news. The
  recalled fact *is* the prediction error. A readout that reports the
  tag-induced deviation over the cue baseline is the predictive-coding
  readout.

- **Divisive/normalization computation** (Carandini & Heeger 2012,
  "Normalization as a canonical neural computation"): cortical
  responses are normalized against contextual drive. A raw cosine
  lacks this; the difference readout supplies the contextual
  normalization the language-output region's weak lateral inhibition
  does not provide.

The eight prior arcs are not invalidated -- they remain a genuine
biology-translatable finding (compositional capability does not emerge
from dynamics overlays on a fixed raw-cosine readout). This design
tests whether the missing ingredient is the readout's failure to
normalize against context.

## What is genuinely new vs. the eight prior arcs

| Aspect | Arcs 1-8 | This design |
|--------|----------|-------------|
| Readout | raw cosine over language-output, single measurement | difference of two measurements (bound minus baseline) |
| What varied | substrate dynamics (gating, rhythm, replay, priming) | the readout computation itself |
| Cue handling | present (arcs 6-8) or suppressed (arc 5) | present in both measurements; cancels by subtraction |
| Substrate | unified v14/v16 + hippocampus, unchanged | identical, unchanged -- reused cached checkpoints |

The eighth arc substituted *which population* is read (adjective-pool
firing instead of language-output cosine) but still took one raw
measurement; the cue still contaminated it. This design takes two
measurements and removes the cue by construction. That is the
distinction.

## Pre-registered cheap-first falsification probe

Before any arc machinery (frozen verdict module, adversarial review,
multi-seed decisive run), a cheap single-seed probe establishes
whether the difference readout shows *any* signal. This honours the
falsify-cheaply-first discipline.

**Probe (single seed 42; reuse the cached 200-event unified substrate;
no retraining; pure eval; estimated minutes):**

1. Load the cached unified substrate (seed 42, 200 events/word).
2. Encode 5 (noun, adjective) bindings as engram tags via the reused
   `encode_concept_pair` helper -- byte-unchanged.
3. For each binding, compute three readouts:
   - **raw**: cue + tag, ranked by raw cosine (the arcs-1-8 readout --
     the baseline to beat)
   - **difference**: (cue + tag) minus (cue alone), ranked by deviation
   - record the top word and whether it equals the bound attribute
4. **Controls** (pre-registered, fixed before the run):
   - *ungroundable control*: cue a noun with NO encoded tag. The
     difference readout must produce a near-zero deviation (no tag =
     nothing added). If the difference readout emits a confident word
     here, it is fabricating -- the probe is VOID.
   - *permuted-tag control*: cue noun N but stimulate the tag for a
     DIFFERENT binding (N', A'). The difference readout must point to
     A' (the stimulated tag's attribute), not A. If it still points to
     A, the readout is not actually reading the tag -- VOID.

**Pre-registered decision rule (fixed; never tuned):**

- If the difference readout scores strictly more correct than the raw
  readout on the 5 groundable queries AND both controls behave
  correctly (ungroundable near-zero, permuted points to the permuted
  attribute): the mechanism shows signal. Proceed to the full
  pre-registered three-state arc (frozen verdict module + adversarial
  review + multi-seed decisive run).
- If the difference readout does NOT beat the raw readout, OR a control
  fails: honest fast negative. The readout-computation hypothesis is
  not the missing ingredient. The design line reaches terminal
  closure; the eight-arc-plus-readout convergent ceiling becomes the
  terminal biology-translatable finding (compositional capability
  requires a substrate-level change beyond any readout or dynamics
  overlay on the validated direct-binding substrate).

## Honest ceiling (binding throughout)

- A cheap-probe signal followed by a full-arc PASS would be the first
  architecture to clear the bar -- biology-grounded compositional
  retrieval at small loads. It would still NOT be fluent open-ended
  language; it would be reliable cue-to-attribute recall.
- A cheap-probe negative is an honest fast finding and ends the design
  line. That is not a failure of the project -- the eight-arc
  convergent ceiling plus a localized, mechanistically-precise
  readout-level negative is a genuine biology-translatable result: it
  would say compositional binding cannot be read out of this
  substrate's shared language population by any computation, and the
  substrate's representation itself must change.
- No bar is tuned. The protected module set stays byte-unchanged. The
  no-confabulation moat stays 7/7 byte-identical. The cheap probe and
  any subsequent arc are reuse-by-import only; no protected, frozen, or
  validated module is modified.

## Discipline pins

- The cheap probe's decision rule is fixed above and never changed in
  response to results.
- The ungroundable and permuted-tag controls are mandatory; a probe
  without both behaving correctly is VOID, not a PASS.
- Reuse-by-import only: `encode_concept_pair`, the substrate builder,
  the cached checkpoints, and `measure_pool_firing` are all
  byte-unchanged.
- No automatic differentiation, no external language model.
- Honest propagation of the probe outcome -- positive or negative -- to
  both git remotes.
- If the probe shows signal, the full arc mirrors the prior eight
  arcs' discipline exactly: pre-registered frozen verdict module,
  dedicated adversarial review before the decisive run, controller-only
  multi-seed decisive run, mandatory smell-test, honest propagation.

## Next step

Write the cheap-first probe script
(`research/findings/raw/difference_readout_probe.py`), run it on the
cached substrate, apply the pre-registered decision rule, propagate
the outcome honestly. This design document and the probe outcome are
committed together with the result.
