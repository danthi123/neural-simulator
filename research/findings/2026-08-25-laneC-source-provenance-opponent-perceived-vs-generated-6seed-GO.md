---
type: finding
status: positive
date: 2026-08-25
lane: laneC
board: 129
mechanism: source-provenance-opponent (perceived-vs-generated, learned context-gated)
runner: research/runners/_laneC_source_provenance_opponent_derisk.py
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO (6/6) — runner's own verdict GO=True at the >=5/6 bar
artifacts:
  - research/findings/raw/four_day/_laneC_source_provenance_opponent_6seed.json
  - research/findings/raw/four_day/_laneC_source_provenance_opponent_6seed.json.prov.json
---

# Source monitoring (#129): did the brain SEE a fact or IMAGINE it — a LEARNED, context-gated OPPONENT provenance trace clears the source-attribution margin AND the no-harm control 6/6, surpassing the rate-floor family

<!--derived-->
**Verdict: GO (6/6), runner verdict GO=True at the >=5/6 bar.** On one spiking Izhikevich substrate, a learned
perceived-vs-generated provenance code attributes every recalled item to its correct source (accuracy 1.000) with a
min normalized discriminability of 0.859 (per-seed 0.832-0.894), while normal content recall is byte-unchanged by the
provenance module (delta 0.0). This is a GENUINELY DIFFERENT mechanism from the banked NO-GO family, and it works
because it reads provenance as the SIGN of an opponent comparator, not the absolute rate of one pool.

## The boundary being surpassed

Every prior laneC source variant is banked NO-GO — attractor_competition, attractor_joint, conjunctive_tag,
plastic_source_memory, coresidency v1/v2, popcode+homeostasis. They share ONE family: source is read from the
ABSOLUTE RATE of one source pool among competitors (margin = own_rate - max(rival_rate)). The
2026-08-11 honest negative named the exact residual: "a source whose single-source Hebbian encoding is genuinely too
weak, which no recall-time gain can lift off the f-I ceiling". Because each pool sits near its f-I ceiling and varies
per seed, one source lands below an ABSOLUTE floor; recall-time gain over-drives it, cross-pool competition drains a
rival (the V2 no-harm fail). That is an operating-point wall of the RATE-FLOOR readout, not of the capability.

## The mechanism (a different family: opponent SIGN, not rate floor)

The perceived-vs-generated axis (Johnson-Hashtroudi-Lindsay 1993 reality monitoring; Simons-Schacter medial-aPFC) is
carried on a channel ORTHOGONAL to content. The agency/authorship 1-bit GO (2026-08-01) proved an opponent SIGN
read-out is robust, but only for a real-time comparator; it EXPLICITLY named this as its follow-on: "the content-cued
episodic SOURCE-MEMORY version (Hebbian-bind content->tag at encoding, content-cue the tag at recall)". This runner
builds exactly that.

- Two neuromodulatory ENCODING-CONTEXT lines (`ctx_perceived` = external/high-ACh feedforward encoding mode,
  Hasselmo-Bower; `ctx_generated` = internal-generation mode); exactly one is active per encode.
- A SEPARATE zero-init plastic trace per provenance (`episode -> prov_perceived`, `episode -> prov_generated`). At
  encode the active context DRIVES its prov pool's firing, so the three-factor product (pre=content x post, post gated
  by the context neuromodulator) potentiates ONLY the provenance whose context was on. The rival trace stays ~0.
- OPPONENT read-out (Namburi-Tye biased competition, the agency motif): the two prov pools mutually inhibit via FS
  interneurons. At RECALL the contexts are SILENT; the content cue alone drives the learned trace; the judgment is the
  SIGN of rate(prov_perceived) - rate(prov_generated), reported as a divisively-normalized discriminability
  d = (r_true - r_false)/(r_true + r_false). d is a RATIO -> immune to the common-mode absolute-rate weakness that
  killed the rate-floor family: even a weakly-encoded source reads d~+1 as long as its rival trace is ~0.

<!--derived-->
**The one mechanistic refinement that carried the de-risk was a COINCIDENCE THRESHOLD (sparse coding), not a tune.**
With a high Hebbian weight cap the discrimination was near-linear and min d sat at ~0.4 (a partial-cue leak: the
overlap_k=3 SHARED neurons of a within-pair item drive the rival pool). Lowering the per-synapse cap (HEBB_WMAX 160
-> 60) makes the 3-neuron partial cue SUB-threshold while the 12-neuron full cue fires -> min d jumps to ~0.85. This
was set on 6 NON-canonical calibration seeds (worst calib min d 0.81) and frozen before the canonical run.

## 6-seed result {42 43 44 100 101 102} — GO (all six PASS)

All values are rounded from the cited artifact (`means` + `per_seed`); no per-seed file holds a mean.

<!--derived-->
| metric | mean | per-seed | reads as |
|---|---|---|---|
| **provenance accuracy** | **1.000** | 1.0 x6 | every item's perceived/generated sign correct (chance 0.5) |
| **min normalized d** | **0.859** | 0.832-0.894 | worst item's discriminability, floor 0.50 (ratio r_true:r_false >= 3) |
| **no-harm on content** | **0.000** | 0.0 x6 | content ("what") recall delta, prov module ON vs lesioned |
| context-swap -> flips | 0.000 / 1.000 | 0.0 / 1.0 x6 | encode under opposite context -> provenance flips (vs original / relabelled) |
| learning-off -> silent | 0.000 | 0.0 x6 | no learned trace -> prov pools receive no drive at recall (rate ~0) |
| novel item -> no source | 0.000 | 0.0 x6 | never-encoded pattern confabulates no provenance |
| content decode from prov | 0.000 | 0.0 x6 | pair identity does not decode from prov rates (chance 0.25) |
| winner-drop (diagnostic) | 0.019 | 0.014-0.023 | opponent inhibition barely touches the correct pool |

- **Attribution margin clears (A)**: accuracy 1.000 and min d 0.859 >> 0.50 on every seed, under within-pair CONTENT
  OVERLAP (a perceived fact and an imagined fact SHARE overlap_k=3 of 12 content neurons — the reality-monitoring
  stressor). The prior family could not even hold the SIGN (sources went negative); here d is a clean 0.83-0.89.
- **No-harm holds (B)**: content_readout recall is byte-identical with the provenance module active vs transmission-
  lesioned (delta exactly 0.0, OU off + per-recall state reset) — the provenance channel does not break normal recall.
- **Instrument verified**: breaking the context gate (W_CTX_PROV=0) zeroes the learned trace (prov_l1_after=0.0),
  drops accuracy to chance 0.500, and the gate returns NO-GO/UNDEFINED — it fails in its failing direction. The
  Verdict machinery selftest passes; the substrate is seeded (seed42==seed42 thresholds, seed42!=seed43, verified).

## Why it surpasses the family, in one line

The rate-floor family asked "is the winning pool's absolute rate above 0.15?" — a question one seed always fails. The
opponent code asks "is the winner's trace bigger than the rival's?" — a SIGN/RATIO the sparse coincidence threshold
makes near-1, so common-mode weakness is invisible. A NO-GO on the readout deferred a METHOD, never the capability.

## Scope and scaffolds

<!--derived-->
Scaffolds (unchanged from the family): caller-supplied sparse episode/content activity, innate context routing and
opponent wiring, an externally-timed encode window, host spike-count evaluation, OU noise off. The
context->provenance binding IS learned (zero-init Hebbian). No language, confidence scalar, or speech policy is
claimed. Named next steps (no-defer): (1) self-organize the context routing + opponent wiring rather than wiring them
innate; (2) replace the host-timed encode window with a neuromodulator-driven plasticity gate; (3) integrate the
provenance read-out into the live chat honesty pathway as a functional "I saw this / I inferred this" self-report.

Reproduce:
```
SIM_BACKEND=numpy python -u -m research.runners._laneC_source_provenance_opponent_derisk \
  --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/four_day/_laneC_source_provenance_opponent_6seed.json
```
