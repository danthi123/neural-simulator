---
type: finding
status: positive
date: 2026-09-01
lane: read-fidelity
board: 129
mechanism: read the learn-through-use RECALL (source-provenance) signal off the #129 SEPARATE, context-gated Hebbian traces (episode->prov_perceived / episode->prov_generated) via a POOL-MEAN opponent margin, graded on the read-fidelity gate (perm-null z>=2.0, lesion-attributable, neuron-identity-shuffle-clean)
verdict: >
  GO (6/6). The read-fidelity iteration-6 finding isolated the surprise->source_provenance F2 read-power residual
  to READ ARCHITECTURE, not estimator, and named its rank-2 NO-DEFER next lever: read the recall signal off the
  #129 SEPARATE TRACE (upstream shaping) rather than off ONE SHARED cross-edge. This run applies exactly that. On
  the #129 separate-trace substrate (laneC ProvenanceBrain reused VERBATIM), the learn-through-use RECALL signal
  read as a POOL-MEAN opponent margin -- the SAME population-collapse statistic FORM the shared-edge iterations
  1-6 used, NOT the ratio, no per-neuron matched filter -- clears the SAME read-fidelity gate on ALL 6 seeds:
  permutation-null z 5.60-11.21 (mean 8.59) >> Z_FLOOR=2.0, lesion of the learned trace collapses the read to an
  exact 0.0 on every seed (attribution 1.0, l1_lesion verified 0.0), and the neuron-identity shuffle null stays
  sub-floor 6/6 (frac self-clearing 0.00-0.05 << 0.15). The SHARED-edge read never cleared this floor on ANY seed
  across 6 banked iterations (mean-rate 0/6, latency 0/6, dispersion 1/6, popvec 0/6, opponent 0/6, learned-LDA
  0/6). DECISIVE MECHANISTIC POINT: the population-collapse read was never the problem in itself -- it was fatal
  only when the SHARED-edge encoding hid the signal off the pool-identity axis; the separate-trace ENCODING moves
  it ONTO that axis, so the SAME collapse read recovers it. The read-power residual is a READ-ARCHITECTURE x
  ENCODING interaction, and separate-trace encoding is the biological fix. This closes the read-power residual at
  the de-risk level; it is NOT production integration (the faculty #129 was already GO; this is the read-power
  half of the recall/shared-edge-read primitive).
seed-waiver: 6-seed run (42/43/44/100/101/102) -- this IS the 6-seed de-risk; 6/6 PASS, seed-trap identical/differs verified.
artifacts:
  - research/findings/raw/_read_fidelity_separate_trace_recall_read_6seed.json
runner: research/runners/_read_fidelity_separate_trace_recall_read_derisk.py
external:
  - Johnson, Hashtroudi & Lindsay, Psychol Bull (1993) -- reality monitoring: source (perceived-vs-generated)
    is carried on a channel ORTHOGONAL to content; grounds the separate-trace (dedicated source pool) encoding
  - Namburi et al. (Tye lab), Nature (2015) -- biased-competition opponent motif; grounds the opponent read
  - Hasselmo & Bower, Trends Neurosci (1993) -- ACh sets a feedforward-encoding mode; grounds the context-gated
    encode window that potentiates ONLY the active provenance's trace
---

# Read-fidelity iteration 7: reading the recall signal off the #129 SEPARATE TRACE (upstream shaping) CLOSES the read-power residual the shared edge could not -- GO 6/6

<!--derived-->

Artifact: `research/findings/raw/_read_fidelity_separate_trace_recall_read_6seed.json` (numpy, 6 seeds). Runner:
`research/runners/_read_fidelity_separate_trace_recall_read_derisk.py`. Seed-trap (build-twice at seed 42):
identical=True, differs-across-seed=True, n=352, hash `02a1f0301568` -- a genuine 6-seed de-risk on a seeded
substrate.

## Why this run exists (the named rank-2 lever, not a re-derivation)

<!--derived-->

The read-fidelity iteration-6 finding
(`2026-09-01-read-fidelity-learned-whitened-opponent-read-NOGO-residual-is-read-architecture-not-estimator`)
proved the surprise->`source_provenance` F2 read-power residual is NOT the estimator: a near-perfect linear
decoder direction still gives a null LIF read (direction-quality and read-power are DECOUPLED), because the read
ARCHITECTURE -- collapsing the population into ONE pool-membership-signed scalar per bin -- discards the
per-neuron discriminability the decoder proves is present, BEFORE any weight direction can act. Six banked
shared-edge NO-GOs (mean-rate 0/6, first-spike-latency 0/6, ISI-CV/Fano dispersion 1/6, popvec/matched-filter
0/6, opponent/push-pull 0/6, learned/whitened LDA+logistic 0/6). That finding's own rank-2 NO-DEFER next lever,
verbatim: "the upstream #129 separate-trace + opponent-ratio wiring ... the honest reading that a read off ONE
shared edge may be intrinsically harder than a separate-trace-encoded read -- i.e. the fix is UPSTREAM SHAPING
(how the edge writes provenance), not the read." This run tests exactly that, on the SAME read-fidelity gate.

## The lever (board #129, the separate-trace wiring; already GO 6/6 for the FACULTY)

<!--derived-->

Instead of writing BOTH provenances onto ONE shared edge and decoding which one it was, source monitoring is
delivered by TWO SEPARATE zero-init plastic traces, `episode->prov_perceived` and `episode->prov_generated`, each
gated open at encode ONLY by its own neuromodulatory context line (`ctx_perceived` / `ctx_generated`). The active
context drives its prov pool's postsynaptic firing, so the three-factor Hebbian product potentiates ONLY that
provenance's trace; the rival stays ~0. At recall the contexts are silent and the content cue alone drives the
learned trace: provenance is carried by WHICH POOL FIRES. The substrate is the laneC #129 GO substrate
(`ProvenanceBrain`, `make_paired_patterns`, `_encode_all`) reused VERBATIM (byte-for-byte the faculty's own).

## What this run adds, and why it is not the laneC GO re-badged

<!--derived-->

This run is NOT the laneC GO under a different name. Three things are new and decisive. (1) It is graded on the
READ-FIDELITY gate -- the SAME permutation-null `z>=Z_FLOOR=2.0`, the SAME `F2_LESION_RATIO=0.34` lesion bar, the
SAME neuron-identity shuffle anti-cheat the shared-edge iterations 1-6 were graded by (NOT laneC's own `min_d`
bar) -- so it is an apples-to-apples GATE answer to the read-power finding's open question. (2) The read is a RAW
POOL-MEAN opponent margin `mean(counts[true_pool]) - mean(counts[false_pool])`, the SAME population-collapse
statistic FORM the shared-edge read used (a signed pool-contrast, no per-neuron matched filter, and crucially NOT
the normalized ratio -- so the win is not ratio normalization). (3) It reads per-neuron spike counts over the
recall window -- a genuinely different statistic of the identical `cp_firing_states` stream laneC's pool-rate read
consumes -- so the shuffle anti-cheat and the pool-mean margin come from one spike stream.

## Result -- GO, 6/6

<!--derived-->

Primary gate (per seed): `real_margin > 0` AND permutation-null `z >= 2.0` AND lesion-attributable
(`|real_lesion| < 0.34*|real_intact|`) AND the neuron-identity shuffle null collapses
(`frac_self_clearing <= 0.15`). All values rounded from the cited artifact.

| seed | pool-mean margin | perm-null z | null mean +- std | shuffle frac-clears | min d (non-gating) | lesion margin | PASS |
|---|---|---|---|---|---|---|---|
| 42 | +9.086 | 8.80 | -0.28 +- 1.06 | 0.050 | 0.894 | 0.000 | True |
| 43 | +8.625 | 9.66 | +0.23 +- 0.87 | 0.050 | 0.858 | 0.000 | True |
| 44 | +8.602 | 11.21 | -0.18 +- 0.78 | 0.000 | 0.832 | 0.000 | True |
| 100 | +8.957 | 8.56 | -0.20 +- 1.07 | 0.000 | 0.854 | 0.000 | True |
| 101 | +9.105 | 7.73 | +0.45 +- 1.12 | 0.000 | 0.869 | 0.000 | True |
| 102 | +9.289 | 5.60 | -0.12 +- 1.68 | 0.050 | 0.848 | 0.000 | True |

- **perm-null z 5.60-11.21 (mean 8.59)** -- every seed clears the floor with wide margin; the SHARED edge never
  cleared it on ANY seed across 6 iterations. The null is well-behaved (null_mean ~0, null_std 0.78-1.68 in count
  units, not degenerate), so the z is a real signal-vs-null separation, not a collapsed-null artifact.
- **Lesion is clean and verified.** Zeroing the learned `prov_learn` traces IN PLACE gives `prov_l1_lesion = 0.0`
  on every seed (the "lesion" TERMS condition holds -- the manipulation still holds at measurement) and the read
  collapses to an exact 0.0 (both pools receive no learned drive -> silent), attribution 1.0. The intact margin is
  LARGE (+8.9) and the lesion is a clean 0.0, so the lesion/intact ratio is perfectly stable -- the OPPOSITE of the
  shared edge's near-zero-intact instability.
- **Learned, not pre-wired.** `prov_l1_before = 0.0` -> `prov_l1_after ~98k-103k` on every seed (emergence grew
  from exactly zero). All 4 `tools.verdict` preconditions PASS, so the verdict machinery returns GO (not UNDEFINED).
- **Normalized d (non-gating, laneC faculty metric)**: worst-seed min 0.832, mean 0.913 -- reproduces the #129
  faculty GO's 0.83-0.89 range, confirming the substrate is the faculty's own.

## What this settles (NO-DEFER -- it moves the whole read-power arc)

<!--derived-->

**1. The read-power residual is a READ-ARCHITECTURE x ENCODING interaction, and separate-trace encoding is the
fix.** The primary read here is a POPULATION-COLLAPSE read (a difference of two pool means). On the SHARED edge
that exact form failed 0/6 because the gen-vs-perc signal lived in a per-neuron pattern WITHIN a single pool that
the pool-mean discards. On the SEPARATE-TRACE substrate the SAME collapse read PASSES 6/6, because the encoding
has moved the signal ONTO the pool-identity axis the collapse read preserves. So the population-collapse read was
never the problem in itself -- it was fatal only when the encoding hid the signal off the pool-identity axis.

**2. It resolves iteration 6's honest doubt.** Iteration 6 asked "whether ANY population-collapse read can close a
shared-edge read -- so the separate-trace ENCODING, not a better read, is the lever." Answer: the separate-trace
ENCODING is the lever; once it is in place, a plain population-collapse read closes it. A better read on the shared
edge (the estimator family: mean-difference, whitened LDA, regularized logistic, over the full shrinkage sweep) is
banked as helping-but-insufficient by iteration 6; upstream shaping is what carries it.

**3. The named alternative lever (per-neuron matched-filter opponent read) is now OPTIONAL for this residual.**
Iteration 6 named two next levers: (rank 1) a per-neuron matched-filter read that preserves per-neuron
discriminability, and (rank 2, this run) upstream separate-trace shaping. Rank 2 closes the read-power residual at
the de-risk level 6/6, so rank 1 is not required to close THIS residual (it remains the lever if a future need
demands reading provenance off a genuinely shared edge -- e.g. a substrate where separate traces are not
available).

## Honest scope -- a GATE comparison, not a single-variable A/B; a de-risk, not integration

<!--derived-->

This is a GATE comparison, NOT a controlled single-variable A/B. The separate-trace read differs from the
shared-edge read in BOTH the encoding (two context-gated traces vs one trained cross-edge) AND the substrate (the
pure #129 `ProvenanceBrain` vs the merged `surprise->prov_generated` pool) -- precisely because the ENCODING is the
lever (upstream shaping). What is held IDENTICAL is the GATE: `Z_FLOOR=2.0`, the neuron-identity permutation null,
`F2_LESION_RATIO=0.34`, and the signed pool-contrast (population-collapse) statistic FORM. On that identical gate
the shared-edge read is banked NO-GO 0/6 across 6 iterations and the separate-trace read is GO 6/6 here. This
closes the read-power RESIDUAL (the measurement gap the shared-edge read could not clear); it is a DE-RISK, not a
"closed" capability in the production sense (no `sim/` edit, no production wiring, no default flip). The FACULTY
(source monitoring #129) was already GROUP-A-migrated and GO; this result is most load-bearing for the general
primitive "can a biological read recover a decoder-provable signal" and for the learn-through-use-recall /
shared-edge-read side -- it says: shape the encoding onto a dedicated (separate) trace and a population read
suffices.

## Scaffolds / residuals

<!--derived-->

The read is a HOST spike-count of the substrate's own `cp_firing_states` (the accepted read scaffold used
identically by the whole read-fidelity arc and the laneC faculty GO); the SIGNAL (which pool fires) is computed by
the brain -- the learned traces + context gating drive the pools -- the host only counts spikes and takes the
pool-mean difference. Innate context routing + opponent interneuron wiring are the laneC scaffolds (unchanged); the
context->provenance BINDING is LEARNED (zero-init Hebbian, verified by the emergence + lesion arms). An
externally-timed encode window + caller-supplied sparse episode/content activity are laneC scaffolds (unchanged);
OU noise off (deterministic substrate) -- the read variance is genuine item-to-item + permutation variance, not
injected noise. The 8-item battery is a host-constructed reality-monitoring stressor (within-pair overlap), same
class of scaffold as the read-fidelity arc's ambiguous item. Named next step (no-defer, not blocking): the
per-neuron matched-filter opponent read (iteration-6 rank 1) remains the lever for reading provenance off a
genuinely SHARED edge where separate traces are unavailable.

Reproduce:
```
SIM_BACKEND=numpy python -m research.runners._read_fidelity_separate_trace_recall_read_derisk \
  --seeds 42,43,44,100,101,102 \
  --out research/findings/raw/_read_fidelity_separate_trace_recall_read_6seed.json
```
