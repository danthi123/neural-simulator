---
type: finding
status: negative
date: 2026-09-01
lane: read-fidelity
board: 129
mechanism: learned / covariance-whitened opponent read (shrinkage Fisher LDA + logistic delta-rule directions, rectified into the same Dale's-law push-pull E/I + LIF readout) vs the diagonal mean-difference opponent, on the surprise->source_provenance F2 crux
verdict: >
  NO-GO on the z>=2.0 gate (0 of 6 seeds pass), BUT the read-power residual is now DECISIVELY ISOLATED and it is NOT
  the estimator. iteration 6 of the read-fidelity arc replaces iterations 4/5's DIAGONAL mean-difference template with a
  covariance-aware LEARNED direction (primary: shrinkage Fisher LDA = the covariance-whitened matched filter /
  decorrelating-inhibition read; diagnostic: L2 logistic fit by a local delta rule = the estimator family that gets
  iteration-3's linear decoder to separate the pools on all six seeds), still realized as the SAME biological opponent
  (push-pull E/I) + LIF readout on the SAME captured rasters (clean single-variable A/B). Covariance-whitening LIFTS the
  mean read z above the diagonal opponent (reproduced in-run) by rescuing catastrophic-negative seeds, but NO learned
  direction clears the floor on ANY seed, and the shrinkage sweep never clears it either -- so this is not a
  single-shrinkage artifact. THE DECISIVE DIAGNOSTIC (no prior iteration ran it): direction quality is DECOUPLED from
  read-power. On one seed the fitted directions classify held-out gen-vs-perc NEURONS at near-perfect accuracy, yet all
  three LIF reads on that seed recover a near-null margin; across seeds the holdout-accuracy ranking does NOT predict the
  read-z ranking. A near-perfect linear direction yields a near-null population-collapse read. So the read-power residual
  is NOT the weight-direction / estimator (constraint #3 is now BANKED as helping-but-insufficient -- a better direction
  does not lift this read); it is the READ ARCHITECTURE: collapsing the population into ONE pool-membership-signed scalar
  per time-bin, read as a held-vs-base LIF amplification margin, discards the per-neuron discriminability the decoder
  (and now the learned directions) prove is present, BEFORE the weight direction can matter. NO-DEFER next lever (named,
  ready): a per-neuron matched-filter opponent read (each afferent an individually-learned, generalizing matched-filter
  synaptic weight, opponent E/I sign) that PRESERVES per-neuron discriminability instead of collapsing it -- and/or the
  upstream #129 separate-trace + opponent-ratio wiring (already GO on all six seeds for the faculty), the honest reading
  that a read off ONE shared edge may be intrinsically harder than a separate-trace-encoded read.
seed-waiver: 6-seed run (42/43/44/100/101/102) -- this IS the 6-seed de-risk; the clean 0/6 + the 6/6-valid instrument
  + the direction-quality-vs-read-power decoupling are the result.
artifacts:
  - research/findings/raw/_read_fidelity_learned_opponent_read_derisk_6seed.json
runner: research/runners/_read_fidelity_learned_opponent_read_derisk.py
external:
  - King, Zylberberg & DeWeese, J Neurosci, PMID 23536063 <!--derived--> -- a SEPARATE inhibitory population (Dale's law)
    DECORRELATES the excitatory population via LOCAL plasticity rules; grounds the whitening-by-inhibition opponent read
    (verified via the bio-research PubMed MCP this session)
  - Cayco-Gajic, Clopath & Silver, Nat Commun (2017) -- sparse connectivity / inhibition for decorrelation and pattern
    separation (already in the mouth read-power deep-research shortlist's verified external list)
  - Hirsch, Alonso, Reid & Martinez, J Neurosci, PMID 9801388 <!--derived--> -- cortical simple-cell push-pull opponent
    motif (unchanged from iteration 5)
  - Salinas & Abbott, J Neurosci (1994) -- optimal linear estimator / matched-filter normalization
---

# Read-fidelity iteration 6: the LEARNED / covariance-whitened opponent read is NO-GO 0/6 -- and it PROVES the read-power residual is the READ ARCHITECTURE, not the estimator

<!--derived-->

Artifact: `research/findings/raw/_read_fidelity_learned_opponent_read_derisk_6seed.json` (numpy, 6 seeds; SAME trained
cross-edge + SAME captured rasters iterations 1-5 used -- no retraining confound; the build-twice seed-trap hash is
identical to iterations 4/5, so the substrate is byte-for-byte the same one). Runner:
`research/runners/_read_fidelity_learned_opponent_read_derisk.py`.

## Why this run exists (the opponent read is BANKED -- this is its rank-1 residual, not a re-derivation)

<!--derived-->

The surprise->`source_provenance` F2 crux asks whether a biological spiking read can separate GENERATED from PERCEIVED
off ONE SHARED trained cross-edge. Serially banked NO-GO: mean-rate 0/6, first-spike-latency 0/6, ISI-CV/Fano
dispersion 1/6, popvec/matched-filter template 0/6, and the OPPONENT / PUSH-PULL read 0/6 (net worse than the single
channel; `2026-08-28-read-fidelity-opponent-pushpull-NOGO-...`). A linear+MLP DECODER over the SAME 10-bin per-neuron
profile separates the pools 6/6 shuffle-clean (`2026-08-28-read-fidelity-decoder-SIGNAL-FOUND-...`) -- so the signal is
present; the biological reads cannot reach it (a READ-POWER gap). BOTH the opponent NO-GO and the popvec NO-GO named the
SAME rank-1 next lever: the diagonal mean-difference template is a WEAKER ESTIMATOR than the regularized logistic the
decoder uses; a LEARNED / variance-weighted / regularized read direction is the fix. This run builds exactly that. The
decisive prior fact this run leans on: the decoder arm that recovers the signal is LINEAR (L2 logistic) -- so the signal
is linearly separable and the missing ingredient is ESTIMATOR QUALITY, not nonlinearity (which is why the dendritic
lever is deprioritized here; our own 2026-08-25 vision-2layer NO-GO already showed nonlinear expansion does not reliably
lift a linear ceiling).

## The mechanism (single-variable A/B vs iteration 5: the template-fitting METHOD only)

<!--derived-->

iterations 4/5 fit the 10-bin template as the DIAGONAL mean-difference `(mu_gen-mu_perc)/pooled_per_bin_std`, which
normalizes each time-bin by its own std and IGNORES cross-bin covariance. The 10 bins are temporally correlated, so the
covariance-aware direction can differ substantially from the diagonal one -- and that difference is exactly what the
working decoder exploits. This run replaces the diagonal template with a covariance-aware LEARNED direction, everything
downstream (the Dale's-law opponent E/I rectification, the LIF readout, the stratified neuron-identity CV, the
permutation null, the delta_held_base gate) reused VERBATIM from iteration 5:

- **`meandiff`** = iteration 5's EXACT signed direction (diagonal). Run IN THIS PROCESS so the comparison is a
  same-fold, same-null A/B, not a cross-file number lift. (It reproduces the banked opponent: on seed 42 it is a strong
  negative here, matching the banked opponent's own strong-negative seed 42 -- the instrument and the NO-GO both
  reproduce.)
- **`lda`** (PRIMARY / gating) = shrinkage Fisher LDA `w=(Sigma_pooled+shrink*mean_diag*I)^{-1}(mu_gen-mu_perc)` -- the
  covariance-whitened matched filter. Biologically a matched filter preceded by DECORRELATING inhibition: a separate
  inhibitory population that decorrelates the excitatory code via local plasticity (King, Zylberberg & DeWeese 2013,
  PMID 23536063; Cayco-Gajic, Clopath & Silver 2017). The shrinkage is the homeostatic regularizer the opponent NO-GO's
  rank-1 residual asked for (down-weight the noise-dominated directions), realized as covariance shrinkage.
- **`logistic`** (diagnostic) = L2 logistic fit by a local error-corrective DELTA rule (batch GD on the cross-entropy +
  weight-decay) -- the exact estimator family that gets iteration-3's linear decoder its 6/6, realized through the SAME
  opponent-LIF read.

Each signed direction is rectified into the same opponent (push-pull) E/I pair (`template_E=clip(w,0,None)`,
`template_I=clip(-w,0,None)`) driving ONE LIF readout via net excitatory-minus-inhibitory current. Anti-leakage
(unchanged bars): direction + standardization fit on TRAIN-fold neurons only; readout evaluated on the held-out test
fold's own raw spikes; pool membership (+1/-1) is the structural wiring sign every iteration uses. Instrument VALID: the
neuron-identity permutation null collapses on all 6 seeds on every combo; seed-trap identical; all 3 preconditions OK.

## Result -- GO=False, 0/6, but the residual is isolated

<!--derived-->

Primary gate = `lda` on `delta_held_base` (the cross-edge-attributable component): 0 of 6 seeds PASS (none clears
Z_FLOOR=2.0; none is lesion-attributable -- the near-zero intact margins make the lesion/intact ratio unstable, the same
signature the banked opponent/popvec showed).

Per-method intact-delta read z (the same-process A/B) and per-neuron direction holdout-accuracy (gen-vs-perc
classification, chance 0.5) -- all values rounded from the cited artifact's `z_summary` / `direction_holdout_acc_summary`:

| method | read z per-seed [42,43,44,100,101,102] | read z mean / peak | holdout-acc per-seed | acc mean |
|---|---|---|---|---|
| meandiff (=iteration-5 opponent) | [-1.29, -0.24, 0.19, 0.20, 1.01, 0.86] | 0.121 / 1.007 | [0.567,0.692,1.000,0.518,0.676,0.630] | 0.680 |
| **lda (whitened, gating)** | [0.08, 0.66, -0.19, 0.34, 0.47, 0.61] | **0.330 / 0.663** | [0.652,0.683,0.831,0.580,0.712,0.751] | 0.702 |
| logistic (best-linear diag) | [-0.65, 0.88, 0.20, 1.57, 0.85, 0.33] | 0.530 / 1.573 | [0.731,0.801,0.975,0.561,0.800,0.664] | 0.755 |

Shrinkage-sensitivity (lda intact-delta z at shrink in {0.1, 0.3, 1.0}, per seed): no lam clears the floor on any seed
(largest observed ~1.39 at seed100/lam1.0, ~1.09 at seed102/lam0.1) -- the NO-GO is not a single-shrinkage artifact.

## What this settles (NO-DEFER -- a verdict on the METHOD, and it moves the whole arc)

<!--derived-->

**1. Covariance-whitening HELPS but is insufficient.** `lda` lifts the mean read z above the diagonal opponent (0.330 vs
0.121) chiefly by rescuing the catastrophic-negative seeds (seed42 -1.29 -> 0.08; seed43 -0.24 -> 0.66); the improvement
is real but NON-uniform (lda beats meandiff on 3/6, loses on 3/6) and sub-threshold everywhere. So the diagonal template
WAS leaving estimator power on the table -- but recovering it does not clear the floor.

**2. The residual is DECISIVELY the READ ARCHITECTURE, not the estimator.** Direction quality and read-power are
DECOUPLED. On seed 44 the fitted directions classify held-out gen-vs-perc NEURONS at accuracy 1.000 (meandiff) / 0.975
(logistic) -- a near-perfect linear discriminant -- yet every LIF read on that seed recovers z ~= 0.2 (null). Across
seeds the holdout-accuracy ranking (logistic 0.755 > lda 0.702 > meandiff 0.680) does NOT track the read-z, and
`logistic` (best direction) is also the most VARIABLE read. This is the diagnostic no prior iteration ran, and it
resolves the deep-research shortlist's open ranks-1-vs-2 question: a better read DIRECTION (the estimator lever,
constraint #3 / the "learned-gain" rank-1 residual) is NOT the binding constraint. What is binding is the read's
ARCHITECTURE -- the population is collapsed into ONE pool-membership-signed scalar per time-bin and read as a
held-vs-base LIF amplification margin, which discards the per-neuron discriminability the decoder and the learned
directions both prove is present, BEFORE any weight direction can act on it.

**3. The estimator-quality lever family is now BANKED** (diagonal mean-difference, covariance-whitened LDA, and
regularized logistic, over the full shrinkage sweep) as helping-but-insufficient on this read architecture. This closes
the "just needs a better/regularized/whitened linear estimator" hypothesis for the F2 shared-edge read.

## Honest residual + the next lever (NO-DEFER -- the next method, not a wall)

<!--derived-->

The read-power gap on the F2 crux is real and it lives in the READ ARCHITECTURE. Two ranked next levers, both biological:

1. **A per-neuron matched-filter opponent read** (the direct architectural fix): give EACH afferent an
   individually-learned, generalizing matched-filter synaptic weight (its profile's projection onto the train-fit
   direction), opponent E/I by sign, instead of collapsing the population into a fixed +/-1 pool-membership scalar. This
   PRESERVES the per-neuron discriminability the decoder exploits (the working reader keeps per-neuron resolution; every
   read tried so far collapses it). It must be built leakage-safe (the per-neuron weight computed from a
   calibration/base read, applied to the held read, to avoid a self-energy artifact) -- a careful separate iteration 7,
   named and ready.
2. **The upstream #129 separate-trace + opponent-ratio wiring** (already GO on all six seeds for the FACULTY,
   `2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO`): source monitoring is DELIVERED via
   TWO context-gated SEPARATE Hebbian traces + a divisively-normalized opponent RATIO read. The honest reading of five
   banked read NO-GOs on ONE shared edge is that reading provenance off a SHARED cross-edge may be intrinsically harder
   than a separate-trace-encoded read -- i.e. the fix is UPSTREAM SHAPING (how the edge writes provenance), not the
   read. This is a hypothesis to test next, not a wall to accept.

## Downstream implication

<!--derived-->

This is the SHARED read-power wall behind (a) the mouth spiking generator's read-SNR and (b) learn-through-use recall
(reading a stored association off a shared substrate edge). The MOUTH side was separately re-framed (the "deep
0.34-0.37 plateau" was a stale-cache artifact; post-fix the mouth read sits ~0.85, a tuning residual -- board 2026-08-28
update). So this result is most load-bearing for the LEARN-THROUGH-USE-RECALL side and the general primitive "can a
biological read recover a decoder-provable signal off a shared edge". It does NOT unblock own-mouth generation directly
(that is on a mostly-separate, largely-closed track). It DOES advance the recall/shared-edge read: the estimator lever
is banked, and the next lever (per-neuron matched-filter read, or upstream separate-trace shaping) is named and
quantified. The instrument (`_read_fidelity_learned_opponent_read_derisk.py`, with the direction-quality-vs-read-power
diagnostic) is reusable verbatim for iteration 7.

## Scaffolds / scope

<!--derived-->

DE-RISK ONLY -- new research runner + finding, no `sim/` edit, no production wiring, no default flip, so there is no
default-off flag to assert byte-identical-off on (nothing in the production path changed). The covariance and the
directions are HOST-computed (linear solve / gradient descent), not yet an on-substrate spiking decorrelating
microcircuit -- the whitening is biologically motivated (King 2013 decorrelating inhibition + local rules) but
host-realized here; a spiking anti-Hebbian whitening layer is the on-substrate form if a whitened direction ever proves
load-bearing (it did not, here). Shrinkage/logistic knobs are host-chosen (a shrinkage sweep is reported so the NO-GO is
not a single-lam artifact). The readout trains/tests on neuron-identity folds, not independent trials (this pool family
has none) -- inherited constraint, unchanged.

Reproduce:
```
SIM_BACKEND=numpy python -m research.runners._read_fidelity_learned_opponent_read_derisk \
  --seeds 42,43,44,100,101,102 \
  --out research/findings/raw/_read_fidelity_learned_opponent_read_derisk_6seed.json
```
