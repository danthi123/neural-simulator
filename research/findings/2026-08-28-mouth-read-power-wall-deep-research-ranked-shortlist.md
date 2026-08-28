---
type: finding
status: reference
lane: mouth
board: 80
date: 2026-08-28
mechanism: mouth read-power wall (#80 / gap#4) — deep-research synthesis + ranked de-risk-ready shortlist for the substrate-read that supplies the mouth e-prop error; cross-arc-weighted against the sibling read-fidelity F2 crux
verdict: RESEARCH-SCOPING deliverable (NOT a validated GO) — a ranked build-ahead de-risk shortlist the controller queues when the GPU frees, honest about what is proven vs open
seeds: []
seed-waiver: this is a scoping/synthesis reference doc, not an experimental result — no GO/NO-GO is claimed here; every ranked item carries its own pre-registered 6-seed gate to be run later
artifacts:
  - research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
  - research/runners/_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py
  - research/biology/urbanczik-senn-dendritic-prediction.md
external:
  - Urbanczik & Senn, "Learning by the Dendritic Prediction of Somatic Spiking", Neuron 81:521-528 (2014), PMID 24507189
  - Mikulasch, Rudelt, Wibral & Priesemann, Trends Neurosci 46:45-59 (2023), PMID 36577388
  - Salinas & Abbott, "Vector reconstruction from firing rates", J Comput Neurosci 1:89-107 (1994)
  - Jadi, Behabadi, Poleg-Polsky, Schiller & Mel, Proc IEEE 102(5) (2014), PMID 25554708
  - Poirazi, Brannon & Mel, "Pyramidal Neuron as Two-Layer Neural Network", Neuron 37:989-999 (2003)
  - Polsky, Mel & Schiller, "Computational subunits in thin dendrites of pyramidal cells", Nat Neurosci 7:621-627 (2004)
  - Litwin-Kumar, Harris, Axel, Sompolinsky & Abbott, "Optimal Degrees of Synaptic Connectivity", Neuron 93:1153-1164 (2017)
  - Cayco-Gajic, Clopath & Silver, "Sparse synaptic connectivity is required for decorrelation and pattern separation", Nat Commun 8:1116 (2017)
  - Moreno-Bote, Beck, Kanitscheider, Pitkow, Latham & Pouget, "Information-limiting correlations", Nat Neurosci 17:1410-1417 (2014)
  - Bienenstock, Cooper & Munro (BCM), J Neurosci 2:32-48 (1982); Oja, J Math Biol 15:267-273 (1982)
---

# Mouth read-power wall (#80 / gap#4): deep-research synthesis + a ranked de-risk-ready shortlist

> **HONESTY BANNER.** This is a RESEARCH-SCOPING reference doc, not a result. No GO/NO-GO is claimed here.
> It ranks build-ahead de-risks the controller can queue when the GPU frees. Every item carries its own
> pre-registered 6-seed gate that must actually run before any claim. The ranking is a prior, weighted by
> the banked record — it is not a verdict.
>
> Machine-readable shortlist artifact (this doc's structured output, from the 14-agent deep-research workflow):
> `research/findings/raw/_mouth_read_power_deep_research/ranked_shortlist.json`.

## 1. The wall (do not re-derive)

The mouth read-out e-prop learning FORWARD — the graded-conductance substrate read that supplies the per-output
error to the local three-factor rule — plateaus at `sub_learned_recov_mean` ~0.34-0.37 while the copied-weight
read reaches ~0.98 and a matched host-linear-proxy forward reaches ~0.86-0.90. The read gradient is already
ideal; ~16 internal levers are refuted. Two open questions are LIVE:

- **A PENDING coverage/epochs GPU readout run** (PID 4186250, actively executing at write time: B=48, n-train-pos
  8000->20000, epochs 10->12, ~3x the gradient-step budget, all else held; ~3/6 seeds done, seed44 GO=True at
  last log check) — tests whether a pure DATA/COMPUTE-BUDGET lever closes the NARROW residual (the 3/6-of-6 seeds
  that missed the strict `sub_recov_ratio>=0.85` bar in the 2026-08-28 stale-cache-fix confirmation, all sitting
  in a tight 0.84-0.91 band). Source: `research/findings/2026-08-28-mouth-eprop-readout-tuning-residual-coverage-epochs-lever-STAGED.md`.
- **The DEEPER ~0.34-0.37 plateau** — the read-power wall the dendritic arc and this shortlist target. Source:
  `research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`.

These are DIFFERENT problems (see §5). The pending run adjudicates which one the residual belongs to.

## 2. What is already banked (the negatives that shape the ranking)

- **Ensemble (`--sub-pop`) read: INERT BY CONSTRUCTION** (honest NO-GO, not a bug). Word-pool members are
  DETERMINISTIC CONDUCTANCE REPLICAS of one shared noisy hidden population (OU noise injected as current, pools
  never spike), so pooling + gain-normalizing cancels the common-mode noise EXACTLY (bit-identical
  `sub_learned_recov`=0.4977 at sub-pop 1 and 2, quoted from the ensemble NO-GO finding). <!--derived--> Averaging only helps when members carry INDEPENDENT noise.
  This is the fact that escalated the dendritic (Urbanczik-Senn) contingency from named-in-reserve to active.
- **The dendritic (Urbanczik-Senn) incumbent is BUILT, gated, byte-identical-OFF verified, and smoke-clean at
  the WIRING/anti-cheat level** (`--dendritic` in `_wkv_mouth_readout_eprop_batched_substrate_derisk.py`):
  host_matmul_on_forward=0, the apical substrate read fires every step, apical calibration has the correct sign
  (m_target>m_nontarget), shuffle and freeze-apical lesions collapse learning (the teacher is load-bearing).
  Its EFFICACY is genuinely open — smoke go=false at 12 steps by design; the decisive 6-seed run is STAGED, not
  run, and the runner branch (`research/mouth-read-snr-dendritic`) is not yet merged to the main checkout the
  live gpu_queue runs from.
- **CROSS-ARC EVIDENCE (the decisive input this synthesis adds).** The sibling **read-fidelity F2 crux**
  (surprise->`source_provenance`) shows the IDENTICAL signature: a linear/MLP DECODER separates the distributed
  code 6/6 shuffle-clean, but substrate-faithful spiking reads recover ~0 — a READ-POWER gap, not a signal gap.
  On that crux, FIXED reads have been serially refuted: mean-rate 0/6, first-spike-latency 0/6, ISI/Fano 1/6,
  **popvec/matched-filter template 0/6** (`2026-08-28-read-fidelity-popvec-template-biological-read-NOGO-power-gap-not-signal-gap.md`),
  and **opponent/push-pull 0/6, net WORSE than the single channel** (`2026-08-28-read-fidelity-opponent-pushpull-NOGO-...md`).
  The opponent NO-GO's own diagnosis is load-bearing for our ranking: an unregularized full-power estimator
  ADDS variance without matching signal — "a full-power estimator built naively is not automatically more
  powerful than a well-regularized one." And a **two-layer nonlinear granule-expansion spiking readout did NOT
  lift a linear ceiling** on a related vision crux
  (`2026-08-25-vision-nonlinear-2layer-granule-expansion-readout-does-not-lift-the-c2-linear-ceiling.md`).

**The convergent read of the whole record:** across BOTH read-power arcs, every FIXED read failed, and the two
nonlinear-EXPANSION attempts on record did not lift a ceiling. The least-refuted direction is a **LEARNED,
regularized/variance-weighted local read rule with an INDEPENDENT teaching pathway** — which is exactly the
Urbanczik-Senn apical teacher (rank 1) and its variance-weighted refinement (rank 2). The nonlinear-expansion
family (ranks 4-5, 7) carries an explicit internal countervailing NO-GO and is downweighted accordingly.

## 3. The ranked shortlist (cheapest-first de-risk, 6-seed gate where a GO is claimed)

Ranked by (read-power upside x cheapness x biological faithfulness), re-weighted by §2's banked record.

1. **Urbanczik-Senn two-compartment apical-teacher read (BUILT incumbent).** Independent apical spiking read +
   per-unit local error `sigma(apical)-sigma(basal)` replacing the cross-unit softmax over the noisy basal read;
   local delta update, no weight transport. Grounded in `research/biology/urbanczik-senn-dendritic-prediction.md`
   (Urbanczik & Senn, Neuron 81:521-528, 2014, PMID 24507189; the same rule already shipped on the live bridge
   for the error-SOURCE, `2026-08-19-neural-error-onbridge-GO.md`). **De-risk: DONE at the wiring level; only the
   merge + staged 6-seed GPU run remains.** GO-gate (pre-registered): per-seed `sub_learned_recov >= 0.85 x
   sub_copied_recov` OR `sub_learned_recov >= 0.55`, AND the 4 anti-cheats hold; board GO = >=5/6 seeds. Lane:
   small-GPU-queue. **Caveat:** the binding scopes U-S to the error-source and states it does NOT touch the
   read-regime — applying it as a read-SNR lever is a NEW use whose efficacy is open.
2. **Variance-weighted / learned-gain regularized local read rule.** Replace the fixed mean-difference/1:1
   estimator with a variance-aware local delta rule that down-weights noise-dominated channels (Salinas & Abbott
   1994 optimal-linear-estimator). This is the read-fidelity arc's OWN ranked-#1 residual, promoted by the
   opponent NO-GO. **Cheapest real read-power lever** — numpy replay on already-recorded traces, held-out CV
   A/B. GO-gate: beats the fixed estimator beyond a neuron-identity-permutation null at >=5/6 seeds on held-out
   folds, then a decisive 6-seed GPU run at the rank-1 bar. Lane: CPU-pool -> agent-build -> small-GPU-queue.
3. **Differential-correlation / information-limiting DIAGNOSTIC (a decision gate) + the mandatory
   wall-survives-cache-fix recheck.** Leading eigenvector of the read's noise covariance vs head_w, against a
   permutation null (Moreno-Bote et al. 2014). Zero direct read-power upside but the **cheapest, most decisive
   sequencing gate**: an information-limited verdict predicts ranks 5-6 futile and routes effort to the
   independent-pathway family (ranks 1-2), and it independently EXPLAINS the ensemble INERT NO-GO. Bundles the
   mandatory recheck that the deep-plateau wall survives the stale-COO-cache fix (it was retracted once as a
   cache artifact). **Run this FIRST temporally.** Lane: CPU-pool. No read-power GO claimed (diagnostic).
4. **Nonlinear dendritic-subunit refinement of the LINEAR basal compartment** (Poirazi-Mel 2003; Polsky-Mel-
   Schiller 2004; Jadi et al. 2014 PMID 25554708). Strongest form: K nonlinear subunits INSIDE the incumbent's
   basal compartment feeding the apical error (COMPOSES with rank 1). **Downweighted** by the vision-2layer
   granule-expansion NO-GO. De-risk: numpy reanalysis with a MANDATORY held-out train/test split (so a
   correlation rise is not in-sample overfit to K x subset extra params). GO-gate: beats single-linear beyond
   BOTH the identity-shuffle null AND held-out CV at >=5/6 seeds, then a real nonlinear-conductance build + 6-seed
   run. Lane: CPU-pool -> agent-build -> small-GPU-queue.
5. **Fixed sparse-K random expansion recoding** (Marr-Albus; Litwin-Kumar et al. 2017; Cayco-Gajic et al. 2017;
   candidate-7 decorrelation diagnostic folded into ONE combined sweep). **Doubly caveated:** the vision-2layer
   NO-GO applies with full force AND there is a mandatory cheaper prerequisite — re-confirm the structured-vs-
   random decode wall survives the cache-fix (the cupy structure-characterization wall vanished 0.96 vs 0.95
   once a stale-COO cache was fixed); if it does not re-confirm, this angle is MOOT. GO-gate (post-prerequisite):
   K-expanded decode beats raw-dense-hid beyond the shuffle null at >=5/6 seeds, advantage growing with probe
   structuredness. Lane: CPU-pool -> agent-build -> small-GPU-queue.
6. **Ensemble WIRING FIX** — route each read clone to its own disjoint independently-OU-noised sub-population so
   pooling averages genuinely INDEPENDENT noise (a real sqrt(P) gain, the mechanism behind the 2026-08-13
   few-spike population-coding GO). Directly repairs the common-mode cancellation that made `--sub-pop` inert.
   Cheap wiring change, P={1,2,4} sweep. GO-gate: `sub_learned_recov` rises monotonically with P beyond the
   bit-identical-cancellation null at >=5/6 seeds (P=4 minus P=1 lift >= +0.05 vs the old 0.000). <!--derived--> **Vulnerable**
   to upstream common-mode — sequence AFTER rank 3. Lane: CPU-pool / small-GPU-queue.
7. **Plastic sparse-K expansion (BCM/Oja)** — local unsupervised reorganization of the rank-5 expansion geometry.
   **Non-actionable now** by its own logic (only worth it if rank-5's fixed version first moves the needle) and
   doubly gated. Its motivating 2026 granule-cell-geometry citation is UNVERIFIED — treat as design motivation
   until a binding doc anchors it. Lane: CPU-pool (deferred). Lowest actionability.

## 4. Cost-routing (per the `cost-routing` skill)

Every cheap de-risk in ranks 2-7 is numpy reanalysis of ALREADY-RECORDED traces on the CPU pool (0 GPU, 0 Claude
tokens). Reanalysis/build agents are haiku/sonnet-tier (mechanical/moderate); opus is reserved for the
decisive-run interpretation and the rank-4 nonlinear-conductance design. Rank 1's decisive run is a single
sequential GPU job (the build is sunk). This preserves the one-GPU-brain-at-a-time + minimize-token discipline.

## 5. How the PENDING readout GO-vs-FLAT reshapes the whole shortlist

The pending coverage/epochs run pre-commits to this interpretation, and it re-prices the entire list:

- **GO (>=5/6 seeds hit `ratio>=0.85`):** the narrow residual closes via pure DATA/COMPUTE-BUDGET scaling — a
  GPU-economy artifact of the deliberately cheap 8000/10 fix-confirmation, NOT a read-SNR ceiling. The mouth
  crutch-burndown / larger-vocab retrain then proceeds DIRECTLY on the coverage-tuned substrate-forward with NO
  new read architecture. This whole shortlist is answering the SEPARATE deeper ~0.34-0.37 plateau, so it
  **drops from load-bearing to background build-ahead**: rank 1 stays worth its staged run for the deep target;
  ranks 2-7 become opportunistic idle-lane fillers. The two arcs stay logically separate — the GO does not
  "validate" or "unblock" the dendritic run.
- **FLAT (`go_count` stays <5/6 despite ~3x the gradient-step budget):** falsifies "just needs more data" for
  this residual and MERGES it into the deep-plateau lane. The read-SNR/architecture family becomes the
  **load-bearing next action**, and the wall framing upgrades from "a tuning residual" to a genuine
  architecture-level SNR/credit-assignment ceiling on the single-compartment softmax-over-noisy-basal-read
  design. Then the sequence is: **rank 3 FIRST** (cheapest, adjudicates independent-pathway vs recoding and
  confirms the wall survives the cache-fix) -> **rank 1 merge+queue** as the decisive job -> **rank 2** as the
  highest-value cheap parallel lever (best-supported by the banked negatives) -> ranks 4-6 only as rank 3's
  verdict licenses them.

## 6. Honest residual

If rank 1 (independent apical pathway) AND rank 2 (learned regularized estimator) BOTH fail, the read-fidelity
arc's own honest residual becomes the mouth arc's: the decoder's separability may rely on read power a spiking
substrate cannot realize (unconstrained signed weights + full-population access + proper regularization), and the
fix moves UPSTREAM — how the forward SHAPES the read's target representation (a wiring lever) — rather than the
read itself. That is a hypothesis to hold, not a wall to accept: it would still be a verdict on a METHOD, and the
next method is the upstream shaping.
