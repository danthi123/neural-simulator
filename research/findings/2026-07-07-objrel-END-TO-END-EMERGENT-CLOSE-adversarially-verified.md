# objrel — END-TO-END EMERGENT CLOSE, adversarially verified (0/3 refute, 3 independent from-scratch reimplementations): the emergent learned read-out + the answer-independent `gradedtie` tie-break reads objrel-slot0 on all 10 seeds

**Date:** 2026-07-07
**Verdict:** GENUINE end-to-end emergent close — `genuine_endtoend_close=True`, `gradedtie_composes_cleanly=True`, `anticheats_hold=True`, **0/3 adversarial skeptics refute** (each independently reimplemented the mechanism from scratch; one reproduced the stored JSONs bit-for-bit).
**Workflow:** `w999fjt47` (the first clean adversarial-verify pass of the arc, after 5+ that caught + corrected overclaims).
**Runners:** `research/runners/_rungB1c_objrel_emergent_gradedtie_smoke.py` (threads `gradedtie` into the emergent read-out) + `_rungB1c_objrel_dopamine_plasticity_derisk.py` (the emergent read-out) + `_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (`--read gradedtie`). NO `sim/` edit anywhere.

## The result (10-seed, controller-fanned + independently reimplemented)
**The EMERGENT learned read-out + `gradedtie` reads objrel-slot0 = 1.00 on ALL 10 seeds** (canon 1.00 all 10, Dale-legal all 10), honestly decomposed:
- **7/10 GENUINELY EMERGENT** ({42,44,45,46,101,102,103}): pre-learning (0-epoch random Dale init) objrel-slot0 = 0.00 → RISES to 1.00 only via the reward-driven graded-DA delta rule (epoch trajectory e0=0.0→e300=1.0, holds at 500); the no-reward (DA==0, weights byte-unchanged) control stays 0.00 → the plasticity, not the init, does the work. NOT ridge-warm-started. (BPTT-from-scratch was 0/6.)
- **3/10 init-lucky** ({43,100,104}): pre-learning = 1.00 AND no-reward = 1.00 → the random Dale init already reads objrel; honestly labeled + EXCLUDED from the learning claim (the 10/10 is NOT inflated as 10-learned).
- `gradedtie` closes the SINGLE genuine emergent read-tie (**seed 101**: 12/12 slot0 count-ties `[13,0,13]` → RAW argmax defaults AGENT → 0.00 → `gradedtie` → 1.00) and is inert off-tie (the other 9 close RAW).

## Why it's genuine (the decisive adversarial controls)
- **`gradedtie` manufactures nothing on 101:** the LEARNED graded drive favors THEME 12/12 with a POSITIVE margin (+0.039), while the un-learned (pre-learning) AND no-reward drives point AWAY from THEME (NEGATIVE margin −0.059, THEME 0/12) → the THEME answer is carried by the LEARNED weights; `pre-learning+gradedtie = 0.00` and `no-reward+gradedtie = 0.00`. It broke a genuinely-learned saturation tie, not a learning failure.
- **`gradedtie` is answer-independent** (not a minority-THEME prior — the failure mode that killed the earlier `calibrated`/`gainnorm` attempts): the same graded drive gives AGENT on canonical (drive-hist `[12,0,0]`) and THEME on objrel (`[0,0,12]`), drive-acc 1.00/1.00. First-spike LATENCY does NOT substitute (it reads the E-onset transient; the I-path is delayed one step) — `gradedtie` reads the settled graded synaptic drive, a real neural quantity the saturating spike COUNT quantizes away.
- Three independent from-scratch reimplementations (reuse-by-import of only the frozen reservoir/feature-cache; NO `sim/` edit; numpy/CPU) reproduced the identical 7/3 split + the 101 fix; anti-cheats (canon-not-regressed, held-out, Dale-legal, reward-load-bearing on the 7 genuine seeds) all hold.

## The full objrel arc (a week-old blocker → genuinely closed, honestly)
1. **Foundation** — the Dale-legal spike-native objrel read provably EXISTS in weight space (adversarially verified).
2. **Reservoir** — encodes objrel LINEARLY on all 10 seeds (ridge 1.00); no reservoir/capacity problem (an earlier "reservoir info-absence" was a self-caught misread of the spiking-read for the ridge).
3. **Read-out plasticity** — genuinely learns objrel EMERGENTLY (per-role Dale-legal spiking detectors, graded reward-modulated delta rule; 7/10 genuine, BPTT was 0/6).
4. **The last residual** — a spike-count SATURATION tie on the ambiguous slot0 (both AGENT+THEME pools fire at max) — closed by `gradedtie`, an answer-independent graded-drive tie-break (the earlier `calibrated`/`gainnorm` fixes were caught as disguised minority-THEME priors; first-spike-latency shown not to substitute).
5. **The discipline** — the adversarial-verify workflow caught 5+ of my own overclaims across the arc (uniform-meets-bar, salience-red-herring, purely-upstream-residual, no-reservoir-problem-was-a-misread, calibration-is-a-prior) and each was corrected in the record; the final composed close passed clean (0/3 refute).

## Honest remaining scope (cleanly separated follow-ons, none a blocker)
- The 3 init-lucky seeds ({43,100,104}) are a property of the base dopamine runner's random-Dale-init (some inits already read objrel); characterizing/removing that init-luck is a separate read-out-init question, not part of this close.
- `gradedtie` reads the settled graded synaptic drive — a legitimate graded neural read (graded synaptic integration), NOT a pure spike-count/timing read; a spike-pure route (a post-inhibition-settling timing/rate quantity) is a possible future refinement, not required.

## Files
`research/runners/_rungB1c_objrel_emergent_gradedtie_smoke.py`, `_rungB1c_objrel_reservoir_robustness_sweep_derisk.py` (`--read gradedtie`/`latencytie`); `research/findings/raw/_emgt_s{42..104}.json` (10-seed), `_gt10_s*.json` + `_dist_s*.json` (gradedtie recovery + answer-independence); adversarial-verify Workflows `wgxmgy82f` (prior-caught) + `w999fjt47` (clean close).
