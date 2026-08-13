---
type: finding
status: contributing
date: 2026-08-13
mechanism: coincidence-k-threshold-recalibration-readout
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
instrument: per-seed ignition-curve calibration of coincidence_k_threshold (controlled fractional/solo drives, content-independent) + the W4 A/B (onehot [leg2_v2 WTA] vs graded [W4 RSA soft-competition] vs scramble vs lesion, plateau vs linear) on TWO pragmatic-alignment metrics -- M1 the pre-registered intent-averaged magnitude-fidelity to analytic Frank-Goodman RSA, M2 the finding's re-diagnosis-named implicature-recovery cell S[all|some] (per-intent-normalized). A DEFAULT-k44 control arm (same runner) reproduces the 2026-08-13 negative.
artifacts:
  - research/findings/raw/_pragmatic_success/w4_krecal_6seed.json
  - research/findings/raw/_pragmatic_success/w4_krecal_smoke.json
---

# W4 Task#12: recalibrating the coincidence k-threshold MAKES THE DETECTOR READ the graded fractional mass (the 2026-08-13 base-rate wall is surpassed at the present/absent level) -- but the ALL-OR-NONE plateau SATURATES it (overshooting the graded RSA magnitude), so the valid pragmatic-alignment metric still does not move: a 6-seed A/B that RELOCATES the residual from the detector base rate to the detector's magnitude-blindness + the metric aggregation, and names the magnitude-preserving read-out as the next mechanism

<!--derived-->

**One-line verdict:** the 2026-08-13 finding's own NAMED next lever -- recalibrate `coincidence_k_threshold` to
the per-step GRADED coincident drive so the plateau triggers on fractional coincidence and strips the base rate
-- was BUILT and A/B-tested on 6 seeds with a verified-clean instrument. It DOES what it mechanically set out to
do: the recalibrated detector now READS the graded fractional mass (implicature-recovery cell S[all|some]:
graded 0.360 vs onehot 0.136 at the recalibrated threshold, 6/6 seeds, where the DEFAULT k44 washes it, 0.223
vs 0.271). The detector base-rate WALL -- "a 0.27-fraction coincidence is sub-plateau and invisible" -- is
surpassed at the present/absent level. BUT this does NOT move the valid pragmatic-alignment metric: on the
pre-registered intent-averaged magnitude-fidelity (M1) the graded belief still LOSES to the one-hot (mean move
-0.046, 0/6 seeds), and the single implicature cell (M2), while it moves, FAILS its own scramble control
(scramble 0.552 > graded 0.360). The mechanism WHY is now airtight and NEW: the all-or-none coincidence plateau
is a THRESHOLD, not a MAGNITUDE-preserving read -- recalibrating `k` moves the present/absent boundary so 0.27
registers as "present", but the SATURATED output OVERSHOOTS the analytic RSA magnitude (0.360 vs target 0.20),
so the one-hot's true-zero (0.136, closer to 0.20) actually beats it on fidelity. The residual is RELOCATED off
the base rate onto (a) the detector's all-or-none magnitude-blindness and (b) the metric's intent-averaging.
NOT a GO; an honest boundary with the specific, in-engine next mechanism -- the GRADED dendritic-plateau
read-out (a read-out NONLINEARITY, NOT a learning rule). The refuted two-compartment / dendritic / BDSP /
burstprop deep-CREDIT rule -- already tested-NEGATIVE for hidden credit on spikes
(`2026-05-17-dendritic-credit-assignment-NEGATIVE`, `2026-07-22-gap4-real-issue-NOT-dendrites`) -- is NOT
re-proposed here.

## The build (the finding's own named lever, additive, reuse-by-import, NO sim/ edit)

<!--derived-->

- **STEP A -- instrument-verify + calibrate (content-independent).** For each seed, sweep
  `coincidence_k_threshold` over a grid and measure the MATCHED-detector ignition curve with CONTROLLED drives
  (NOT the RSA content): the two-input rate at a fractional belief drive f=0.27 (the graded off-diagonal mass;
  analytic L1(some)[all]=0.25) vs the two SOLO arms (intent-alone, belief-alone). A GENUINE coincidence gate must
  keep BOTH solo (single-afferent) arms SILENT while the fractional two-input IGNITES. Pick, per seed, the kthr
  that maximizes margin=(r_frac - max_solo) SUBJECT TO max_solo < 0.05 and r_frac > 0.06. Grounded in Larkum
  (2013): the plateau has a tunable coinciding-input threshold ("lower the amount of coinciding spikes required
  to initiate a plateau potential").
- **STEP B -- the W4 A/B at the recalibrated kthr**, reuse-by-import of the exact 2026-08-13 belief sources +
  spiking magnitude-fidelity metric (`_pragmatic_spiking_graded_belief_derisk`): onehot/graded/scramble/lesion,
  plateau/linear, PLUS a DEFAULT-k44 control arm (same runner, only kthr differs -- isolates the recalibration).
- `build_success_bridge` already exposes a `kthr` argument, so this is a READOUT-threshold change ONLY: the
  belief is byte-identical to the W4 runner, plasticity off, no `sim/` edit.

## Result -- 6 seeds {42,43,44,100,101,102}, CPU numpy

<!--derived-->

| read-out (6-seed) | onehot | graded | scramble | verdict |
|---|---|---|---|---|
| **Calibration** clean seeds | — | — | — | **5/6** (seed 43 uncalibratable: solo_belief 0.176 -- its assemblies are heterogeneity-hyperexcitable, no k separates a single afferent from the fractional two-input) |
| **M2** implicature-recovery S[all\|some] (RECAL kthr) | **0.136** | **0.360** | **0.552** | graded > onehot 6/6 BUT scramble > graded (single cell is CHEATABLE) |
| **M2** S[all\|some] (DEFAULT k44) | 0.271 | 0.223 | 0.423 | **washed** (both at base rate -- reproduces the 2026-08-13 wall) |
| **M1** avg magnitude-fidelity (RECAL) | **0.727** | **0.681** | 0.250 | move **-0.046 (0/6)** -> FAILS; scramble LOSES (0.25) so M1 is VALID |
| **M1** avg magnitude-fidelity (DEFAULT k44) | 0.669 | 0.623 | 0.311 | move **-0.035 (1/6)** -> reproduces the finding's -0.035 EXACTLY |
| per-intent fidelity, intent="all" only (RECAL) | **0.835** | 0.745 | — | onehot WINS even on the implicature intent (graded OVERSHOOTS) |
| — linear point-soma detector (RECAL, M1) | 0.343 | 0.354 | 0.428 | plateau is load-bearing; scramble beats both (linear cannot separate) |

Belief-side unchanged (only the read-out threshold moved): graded implicature margin +0.506 (lesion +0.006,
98.9% attributable), calib 12x better than one-hot -- identical to the 2026-08-13 belief read.

## Why it does not move -- the mechanism (NEW, airtight, at the single-cell level)

<!--derived-->

The recalibration mechanically WORKS on present/absent. At the DEFAULT k44 the 0.27 fractional coincidence is
sub-plateau, so graded S[all|some]=0.223 sits at the SAME base rate as the one-hot's true-zero (0.271) -- the
finding's wall. At the recalibrated (lower) k, the detector CROSSES on the 0.27 fraction: graded S[all|some]
jumps to 0.360 while the one-hot's genuine-zero drops to 0.136. The detector now DISTINGUISHES a 0.27-fraction
from a true zero (6/6 seeds). **But the all-or-none plateau is a THRESHOLD, not a magnitude read:** once the
0.27 mass crosses, the plateau SATURATES, so graded S[all|some]=0.360 OVERSHOOTS the analytic Frank-Goodman RSA
magnitude (target 0.20). The one-hot's 0.136 (undershoot 0.064) is CLOSER to 0.20 than the graded's 0.360
(overshoot 0.160), so on the intent-"all" fidelity the one-hot WINS (0.835 vs 0.745). Recalibrating `k` moved
the present/absent boundary but could not make the read GRADED -- the magnitude is not preserved.

Two independent controls make the residual unambiguous, and BOTH were load-bearing:
- **M1 (averaged TV fidelity) is the VALID metric but graded loses:** scramble (graded mass on WRONG intents)
  scores 0.250 << onehot 0.727 -- M1 correctly rewards matching the analytic RSA SHAPE, so scramble loses. Yet
  graded does not beat onehot (mean -0.046): M1 averages the ONE implicature-carrying intent row (only intent
  "all" has graded off-diagonal in the analytic) with two one-hot rows (none, SBNA) where the graded belief adds
  spurious off-diagonal mass -- dilution + the saturation overshoot cancel the gain.
- **M2 (single implicature cell) MOVES but is CHEATABLE:** graded 0.360 > onehot 0.136 (6/6), but scramble 0.552
  > graded 0.360 -- a derangement that maps the dominant 0.73 SBNA mass onto the "all" state produces an even
  larger S[all|some]. So the single cell rewards "mass on this cell", not the CORRECT graded structure; its
  scramble control (pre-registered) correctly REFUSED the surpass claim. Reporting M2 as a win would have been a
  goalpost move; the anti-cheat caught it.

So after fixing the BELIEF (W4, 6/6 GO) and the DETECTOR BASE RATE (this recalibration), the residual is
RELOCATED, not eliminated: it is now the detector's all-or-none MAGNITUDE-BLINDNESS (saturation) plus the metric
AGGREGATION -- NOT the base rate, NOT the belief, and NOT a credit-assignment problem.

## The residual + the named next mechanism (a wall on a METHOD, not the capability)

<!--derived-->

The magnitude-sensitive read the graded RSA posterior needs is a plateau whose OUTPUT GRADES with the coincident
input (V(near) > V(mid) > V(far)), not an all-or-none switch that snaps 0/1 at a threshold. This mechanism
already exists in the engine and is NOT the refuted credit rule: the **GRADED dendritic-plateau read-out**
(`enable_graded_dendritic_plateau`, bridge.py "2.3a-ter" block; the SMOOTH, non-saturating sibling of the
all-or-none coincidence switch; de-risk A GO 2026-06-20, `2026-06-20-dendrite-stage1-onbridge-graded-plateau.md`;
Mikulasch & Priesemann analog dendritic read-out). It passes the WEIGHTED coincident drive through a gentle
centered logistic scaled to a regenerative plateau current -- so a 0.27-fraction coincidence yields a
PROPORTIONALLY smaller plateau than a full-mass one, preserving the graded magnitude the RSA metric scores. The
next de-risk is: read S[t,u] through the graded plateau (calibrate its center/slope to the fractional operating
point via the SAME ignition-curve instrument) and re-run this exact A/B on M1; the prediction is that graded's
S[all|some] tracks ~0.20 (not overshoot to 0.36), so the averaged fidelity moves with the scramble control
intact. A metric refinement (localize the fidelity to the implicature-carrying structure / RSA-informativeness
weighting, per Frank & Goodman) is the parallel lever on the aggregation half of the residual. The capability is
NOT abandoned; the residual is isolated + quantified (overshoot +0.16 on the implicature cell; averaged-metric
dilution across 2 of 3 intents).

## External grounding (deep-research gate)

(1) The plateau's tunable coinciding-input threshold that grounds the k-recalibration: **Larkum (2013), Trends
in Neurosciences 36(3):141, "A cellular mechanism for cortical associations"** ("lower the amount of coinciding
spikes required to initiate a plateau potential"); the NMDA-spike / dendritic "coincidence detector" framing is
confirmed in Kandel 6e (Ch. 13, regenerative NMDA-spike depolarization). (2) The RSA objective is a graded
MAGNITUDE (the listener posterior L1(s|u) is a graded distribution; the speaker objective is expected surprisal,
not an argmax), which is WHY a threshold read is insufficient and a magnitude-preserving read is required:
**Frank & Goodman (2012), Science 336(6084):998, "Predicting Pragmatic Reasoning in Language Games"**. Both
external searches are logged in the queue external-searches record (lane d-pragmatics, 2026-08-13).

## Honest scope

<!--derived-->

A FUNCTIONAL pragmatics correlate. This is a READOUT-threshold recalibration ONLY (how the detector READS a
fractional coincidence), NOT a learning rule and NOT a belief change (byte-identical to the W4 belief). The
per-seed kthr is calibrated on the ignition CURVE (controlled fractional/solo drives), NOT on the
graded-vs-onehot A/B -- a detector PROPERTY, independent of the RSA content; the scramble controls (graded mass
on WRONG intents must LOSE) keep it honest, and on M2 the scramble control CAUGHT that the single-cell move was
not the correct content. A lone single-afferent input must NOT ignite (the solo-silence anti-cheat catches
"lower kthr until everything ignites"; seed 43 is honestly reported as uncalibratable, 5/6 clean). Plasticity
off (STDP/Hebbian/homeostasis/STP/structural/OU/NMDA disabled) -- a fixed-operating-point read, as in the
W3/W4/W5/leg2 GOs. Does NOT overturn the 2026-08-13 negative by moving goalposts: the DEFAULT-k44 control arm
reproduces it EXACTLY (move -0.035), and neither candidate metric yields a graded win with its anti-cheat
intact. NOT a claim of phenomenal access to another mind; self-report would be a functional read-out. numpy-CPU
real spiking Izhikevich bridges; additive NEW runner
(`research/runners/_w4_detector_k_recalibration_derisk.py`, reuse-by-import of the W4 A/B + the Leg-1 coincidence
detector); NO `sim/` edit.

Reproducer: `SIM_BACKEND=numpy python -u -m research.runners._w4_detector_k_recalibration_derisk --seeds 42 43
44 100 101 102 --json research/findings/raw/_pragmatic_success/w4_krecal_6seed.json`.
