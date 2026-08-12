---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes — ARC SUMMARY (gap#4 ALL-IN); READ THIS BEFORE RE-ATTACKING ANY spiking deep-credit / feedback-alignment / "credit collapses at depth" wall
lane: gap#4 ALL-IN (owner-directed 2026-08-11)
verdict: The gap#4 "deep credit on spikes" wall was THREE things wearing one label, and separating them dissolves most of it. (1) On the LIF surrogate net, the 2026-08-02 "chained transport-free FA/KP collapse to majority-class at N>=3" wall is a PER-ARM LEARNING-RATE ARTIFACT (6-seed: a fair per-arm lr enters 6/6 both arms both depths + beats the optimal reservoir; the shared lr sits at the majority floor — a wall that stood ~10 days was a step-size mismatch). (2) Deep credit IS TRACTABLE at de-risk via a LEARNED transport-free rule: Kolen-Pollack learned feedback reaches the 3rd hidden layer on a genuinely-depth-3 tent3 (Telgarsky) fit (majority of the depth-2→depth-3 gap closed, 6/15 ceiling-testable, RATE net; fixed-DFA fails; freezing the feedback collapses it) — VERIFIED artifact. (3) The GENUINE remaining wall is the PRODUCTION Izhikevich bridge, and it is lr-INVARIANT: fixed-DFA 0/6, KP 0/6, DRTP fails, AND a perfect Wᵀ oracle also fails — so it is the few-spike READ regime (CV), not the feedback type or the learning rate. Both routes to a clean spiking depth-necessity TEST are closed (generalisation: Q5 matched-width; fit: the spike-count read gives ~W·T pieces, so depth-2 fits a Telgarsky sawtooth on spikes). Q_C: gap#4 is NOT the single load-bearing blocker on fluent conversation (the working faculties use zero deep credit); OWNER APPROVED (2026-08-11) re-pointing the crux at the BPTT-mouth burn-down + Gate-B delayed-reward credit.
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/2026-08-11-gap4-the-LIF-chained-FAKP-wall-is-a-per-arm-lr-artifact-6seed.md
  - research/findings/2026-08-11-gap4-wave1-verification-corrected-the-FA-KP-wall-is-partly-an-lr-artifact.md
  - research/findings/2026-08-11-gap4-learned-feedback-KP-reaches-the-3rd-hidden-layer-where-fixed-DFA-could-not-smoke-GO.md
  - research/findings/raw/_gap4_perarm_fakp/AGGREGATE.txt
  - research/findings/raw/_gap4_learned_feedback_6valid.json
---

# gap#4 ALL-IN — ARC SUMMARY: a spiking deep-credit "wall" turned out to be a hyperparameter. READ THIS before re-attacking any spiking deep-credit / feedback-alignment / "credit collapses at depth" question.

## Why this doc exists (owner directive, 2026-08-11)

The owner directed an all-in push on gap#4 (deep credit through a deep spiking net with a local, transport-free rule),
then — after the verified reframe below — asked that **this arc be recorded durably so any future wall on a similar
topic surfaces it and we do NOT re-derive it or reach an inaccurate conclusion.** This is that record. If you are about
to attack a "spikes can't carry deep credit / feedback alignment collapses at depth / the deep local rule doesn't enter
the learning regime" wall, **read the four META-LESSONS first — three of them would have saved this arc weeks.**

## The three-substrate verdict (each independently established this session)

1. <!--derived--> **LIF surrogate net (`sim/bptt_snn_gpu`, imported by NO production code): the wall was a PER-ARM-LR ARTIFACT.** The
   2026-08-02 "chained transport-free FA/KP collapse to majority-class at N>=3 on XOR" was measured at ONE shared lr
   (0.05) across arms of very different gradient scale. A 6-seed per-arm-lr sweep
   (`_gap4_perarm_tuned_fakp_baseline_derisk.py`): at a fair per-arm lr (roughly a fifth to a tenth of the shared one)
   the SAME arms leave majority-class + beat the optimal reservoir at N=3 AND N=4, **6/6 both arms**; at the shared lr
   they sit at exactly the majority floor. The
   "degenerate-dynamics fingerprint" was step-size divergence. **The wall stood ~10 days as biology; it was a knob.**
2. <!--derived--> **Rate net: deep credit IS TRACTABLE via a LEARNED transport-free rule.** Kolen-Pollack learned feedback (G_l
   co-adapts toward W_l^T by the same local step, never copied) reaches the 3rd hidden layer on a genuinely-depth-3
   `tent3` (Telgarsky iterated-tent) FIT target: closes **66% of the BP-depth-2→BP-depth-3 gap** (6/15 ceiling-testable
   seeds), where fixed-DFA closes −85% and FREEZING the feedback collapses it to −40% (the win is DUE TO learning G).
   cos(G,Wᵀ) rises from near-zero to 0.826 (co-adapted, transport-free). VERIFIED against the 308KB artifact. Scope: RATE (not spiking),
   partial (reaches ≠ matches oracle). **Fixed feedback (DFA) does NOT get there; learned feedback does.**
3. <!--derived--> **Production Izhikevich bridge: the genuine wall, and it is lr-INVARIANT.** fixed-DFA 0/6, KP 0/6, DRTP (seed-42)
   0.515 < frozen 0.532 < chance — AND a perfect Wᵀ oracle ALSO fails (`2026-08-02-gap4-crux-wall-LOCATED-...`,
   `...FA-convergence-...0of6-izhikevich`). No learning rate moves it. DRTP removing feedback-alignment and still
   failing localizes it to the **eligibility×surrogate credit factor on the post-reset membrane** — the **few-spike
   READ regime (CV)**. This is where a real gap#4 wall lives; the LIF-surrogate and rate results do NOT transfer here.

## The measurability wall (why a clean SPIKING depth-necessity test is not constructible here)

Both routes to "prove credit flows through OBLIGATORY depth-3 on spikes" are closed, for the SAME reason — the
finite-spike READ:
<!--derived-->
- **Generalisation route (Q5, `2026-08-12-gap4-obligatory-depth3-...NEGATIVE`):** on point neurons a matched-width
  depth-2 always matches a depth-3 net on held-out — the finite-spike-read redundancy. Boolean depth separations
  (parity/mux/nestedxor) collapse under a spike-count read.
- **Fit route (Telgarsky sawtooth, adversarially refuted this session):** on a spike-COUNT read over T timesteps each
  hidden unit is a ~T-step staircase, so depth-2@W expresses ~W·T pieces — vastly more than a k-fold tent's 2^(k-1)
  teeth. The O(W)-piece capacity bound that makes Telgarsky depth-necessary on ReLU **does not bind on a spike-count
  read**, so depth-2 fits. (The KP GO in (2) is on a RATE net, where the bound DOES hold — that is why it works there
  and would not be a valid depth test on spikes.)

## THE FOUR META-LESSONS (the durable part — check these BEFORE banking any spiking-credit wall)

1. **A "wall" is a verdict on a METHOD or a HYPERPARAMETER until proven a substrate property. Before banking a
   deep-credit / feedback-alignment wall, run a PER-ARM learning-rate sweep** (each arm tuned independently — FA, KP,
   BPTT have very different gradient scales; one shared lr is an unfair A/B). A single shared lr faked this entire wall
   for ~10 days. The instrument (a fixed-shared-lr A/B) was measuring lr-fairness, not the credit rule.
2. **Verify a "wall" on the PRODUCTION substrate before treating it as real** — the LIF surrogate net is not what the
   brain runs, and its results (both walls AND surpasses) do NOT transfer to the Izhikevich bridge. The genuine wall was
   lr-invariant on Izhikevich (perfect-oracle-fails = the READ regime), which is a DIFFERENT and real thing.
3. **Distinguish "enter the regime" from "deep credit."** Leaving majority-class + beating a reservoir on a
   depth-2-solvable task (XOR) is NOT credit through obligatory depth — local rules AND fair-lr FA/KP AND (probably)
   anything does it. Fixed vs LEARNED feedback is the real axis: fixed-DFA does not reach depth-3; learned KP does (rate).
4. **A clean depth-obligatory TEST is not constructible on a point-neuron spike-count read** (both the generalisation
   and the fit routes collapse under the finite-spike read). If you need to test DEEP credit, do it on a RATE net (where
   Telgarsky/depth-separation holds) or change the READ (more spikes / longer integration / ensemble) — do not spend
   weeks trying to build a depth-obligatory SPIKING task; it is foreclosed for a documented reason.

## Artifacts (the raw records these quoted numbers derive from)

- `research/findings/raw/_gap4_perarm_fakp/AGGREGATE.txt` — the 6-seed per-arm-lr sweep (the LIF wall = lr artifact).
- `research/findings/raw/_gap4_learned_feedback_6valid.json` — the KP learned-feedback rate-net GO (66% gap-close).
- The five cited findings (frontmatter) carry the full per-seed tables + the Izhikevich/Q5 records.

## Where gap#4 actually stands (and the owner's re-prioritization)

Deep credit is **tractable at de-risk via learned feedback (rate), untested on production, and genuinely walled only at
the Izhikevich few-spike read regime.** Per the Q_C reassessment (verified this session): **gap#4 is NOT the single
load-bearing blocker on fluent conversation** — the working faculties (distal-referent WM, register handoff, action
selection, convention) use zero deep credit; the language path gets depth from temporal membrane integration.
**OWNER APPROVED (2026-08-11) re-pointing "the crux" at the BPTT-mouth burn-down + Gate-B delayed-reward credit**, keeping
gap#4 open on its one real residual: porting KP learned feedback to the Izhikevich bridge + a read-CV manipulation.
