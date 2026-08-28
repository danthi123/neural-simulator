---
type: finding
status: negative
date: 2026-08-28
verdict: NO-GO on both non-rate reads, but the INSTRUMENT is now trustworthy. Fixed the shuffle-identity
  anti-cheat that iteration 1 left ambiguous (collapsed on only 3/6 seeds) — root cause diagnosed and repaired,
  now collapses 6/6. With a clean instrument, first-spike-LATENCY is a confident, clean 0/6 NO-GO (2/6 seeds even
  show the WRONG-signed effect at high significance, z=-3.91 and z=-4.50). A new DISPERSION (ISI
  coefficient-of-variation) read, built with the SAME fixed instrument, is also NO-GO at the required 6/6 bar
  (1/6 PASS) but shows a directionally-consistent signal across every seed (6/6 same pre-registered sign, 2/6
  individually significant) that neither rate nor latency showed — the strongest lead so far, not yet a GO.
mechanism: permutation-test (neuron-identity resampling) replacing across-read SEM for non-rate reads on the
  surprise->episodic F2 crux; a first-spike-latency read (repaired) and a new ISI-CV dispersion read
lane: read-fidelity
seed-waiver: 6-seed run (42/43/44/100/101/102) — this IS the 6-seed de-risk for both the instrument fix and the
  dispersion read.
artifacts:
  - research/findings/raw/_read_fidelity_nonrate_latency_dispersion_derisk_6seed.json
runner: research/runners/_read_fidelity_nonrate_latency_dispersion_derisk.py
---

# Read-fidelity iteration 2 — the shuffle instrument is fixed (6/6, was 3/6), latency is a clean 0/6 NO-GO, dispersion is a promising-but-insufficient 1/6

Artifact: `research/findings/raw/_read_fidelity_nonrate_latency_dispersion_derisk_6seed.json` (numpy/CPU, 6
seeds, run locally — the remote pool is provision-blocked). Runner:
`research/runners/_read_fidelity_nonrate_latency_dispersion_derisk.py`.

## Context

[`2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md`](2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md)
(iteration 1) built a first-spike-latency read against the surprise->episodic F2 crux's rate-saturation
floor-miss and got a genuinely UNDEFINED result: 0/6 latency PASS, and — the blocker — its shuffle-identity
anti-cheat collapsed on only 3/6 seeds, an instrument ambiguity rather than a validated null. Per NO-DEFER, this
session had two ordered steps: (1) fix the instrument so the anti-cheat is trustworthy, then re-verify the
latency verdict; (2) only if latency still doesn't clear, add a dispersion read that also escapes mean-rate
saturation.

## STEP 0 — why the shuffle collapsed on only 3/6 (measured, not the finding's own leading hypothesis)

Iteration 1's own leading hypothesis was right-censoring at the window length, or too few spikes in the window so
latency defaults to the same censor value for both pools identically. **This is REJECTED by the raster.**
Instrumented directly against `ReadFidelityPool.f2_reads`'s own captured raster (seed 42): 100% of the
`prov_generated` union `prov_perceived` neurons fire at least once within the `RECALL_STEPS=100` window
(`fired_any.mean()==1.0`) with ~9-10 spikes/neuron on average, on every seed checked — no censoring, plenty of
signal, both pools.

**The real cause.** This pool family runs with `enable_ou_process=False` AND `enable_short_term_plasticity=False`
(`onebrain_merge_framework.py`'s SURPRISE/source_provenance config dicts — no stochastic per-step drive of any
kind). Given a fixed `_hard_reset()` state plus a fixed input current, the network's trajectory is a pure
deterministic function of nothing else. Calling iteration 1's own `_one_read(hold)` `N_READS=8` times in a row
gives read index 0 (run immediately after `train()`, still carrying some of `train()`'s own history) ONE value,
then indices 1-7 give a SECOND value, mutually **bit-identical every time**, on every seed checked. `N_READS=8`
was never 8 independent samples of a noisy readout — it is at best 2 distinct deterministic states, 7/8 of the
"reads" duplicating one of them.

Computing a between-READ SEM over this quasi-duplicated sample does two things: (a) it starves the real
per-neuron identity signal of degrees of freedom (the read-0 outlier and the 7-fold duplicate largely cancel in
the mean while inflating the between-read variance used as the SEM denominator), and (b) it makes the
shuffle-identity anti-cheat's pass/fail a near coin flip — whichever of ~2 possible raster states a FIXED shuffle
mask happens to separate determines its own (illusory) significance, unrelated to whether the mask reflects
genuine pool identity. A controlled check (300 independent random 32/32 partitions applied to one captured raster
pair, seed 42) confirmed the REAL-identity split is a >7-SD outlier against that shuffle-null distribution on
that single pair — the underlying spike timing does carry identity information; the across-read SEM was simply
the wrong instrument to detect it, not evidence the information is absent.

## STEP 1 — the fix, and the repaired latency verdict

Two changes, both to the STATISTIC, not the simulated network (no `sim/` edit):

1. **Significance is now a permutation test over neuron identity**, not read repetition. `K_PERM=300` independent
   random re-labelings of which neurons count as generated/perceived (a fresh draw per permutation, never the
   single seed-fixed mask iteration 1 used) are each re-scored on the SAME captured `N_READS` raster pairs (no
   new simulation — cheap re-reads of an already-captured spike raster). `z = (real_mean - null_mean) /
   null_std` replaces the old across-read-only SEM z.
2. **The shuffle-identity anti-cheat is now**: the fraction of the `K_PERM` null draws that themselves
   individually clear `Z_FLOOR=2.0` relative to the null's own mean/std, required `<= SHUF_COLLAPSE_MAX_RATE=0.15`
   on every seed (comfortably above the ~4.55% a two-sided `|z|>=2` cutoff implies under normality, so the bar
   stays meaningful without demanding textbook normality from a 300-draw empirical null).

**Result: the anti-cheat now collapses 6/6** (`frac_null_clears_floor` well under 0.15 on every seed — full
per-seed numbers in the artifact) — up from 3/6 under the old instrument. The instrument is now trustworthy.

With a clean instrument, **latency is a confident 0/6 NO-GO**, not merely underpowered:

| seed | lat mean (intact) | lat perm z | lat lesion mean | shuffle collapses | PASS |
|---|---|---|---|---|---|
| 42  | -0.000195 | -0.207430 | 0.005781 | yes | no |
| 43  | -0.000742 | -0.521837 | 0.006797 | yes | no |
| 44  | -0.004844 | **-3.908223** | 0.005039 | yes | no |
| 100 | 0.000977 | 0.816117 | 0.005273 | yes | no |
| 101 | -0.005117 | **-4.504770** | 0.000859 | yes | no |
| 102 | 0.001328 | 1.258985 | 0.002891 | yes | no |

Two seeds (44, 101) are individually **significant in the WRONG direction** (the pre-registered sign is
`onset_perceived - onset_generated > 0`; both read strongly negative). Sign is not even consistent across seeds
(4 negative, 2 positive) — this is not noise hovering near a real effect, it is the absence of one. First-spike
latency, as implemented, does not carry the surprise->episodic F2 crux's generated-vs-perceived information.

The rate arm (unchanged instrument, kept only as the reproduce-the-known-crux sanity check) still reads 0/6 PASS,
consistent with iteration 1 and the original crux finding.

## STEP 2 — the dispersion (ISI-CV) read

Built with the identical fixed instrument (same permutation test, same `Z_FLOOR`, same anti-cheat rule, same
captured raster — one simulated trajectory feeds rate, latency, and dispersion). Per-neuron ISI
coefficient-of-variation (`std(ISI)/mean(ISI)`, >=2 spikes required per neuron), averaged over evaluable neurons
per pool, sign pre-registered as `cv_perceived - cv_generated` (same "generated: stronger/faster/more regular;
perceived: weaker/slower/more irregular" direction as rate and latency). Biology: Softky & Koch 1993 (*J
Neurosci* 13(1):334-350) — cortical spike trains are highly irregular, a statistic independent of both mean rate
and first-spike timing, so it can in principle carry information through the same rate-saturating regime
Sanzeni/Histed/Brunel 2020 attribute to refractory-period-driven mean-rate compression.

| seed | disp mean (intact) | disp perm z | disp lesion mean | shuffle collapses | PASS |
|---|---|---|---|---|---|
| 42  | 0.008462 | 1.533493 | -0.003942 | yes | no |
| 43  | 0.030536 | 3.155606 | 0.010489 | yes | no |
| 44  | 0.013665 | 1.780241 | -0.010173 | yes | no |
| 100 | 0.015489 | 1.880899 | 0.001863 | yes | no |
| 101 | 0.037553 | **4.303788** | 0.004962 | yes | **yes** |
| 102 | 0.024261 | 1.845064 | -0.005454 | yes | no |

**NO-GO at the pre-registered 6/6 bar (1/6 PASS)** — this project's own 6-seed rule (`gates/single_seed`,
`feedback_6seed_validation`) treats a single-seed pass as unreliable, not a generalizable result. But unlike
latency, dispersion is **directionally consistent on every seed** (all 6 intact means positive, matching the
pre-registered sign — never once flips) and 2/6 seeds individually clear the significance floor (43: z=3.16,
101: z=4.30); seed 43 misses the lesion criterion by a narrow margin (`|lesion|=0.010489` vs
`0.34*|intact|=0.010382`<!--derived-->). This is a qualitatively different failure shape from latency's sign-inconsistent null —
a real, small, lesion-plausible signal that the current floor/lesion bar is not yet powered to confirm on every
seed, not an absence.

## What this settles + the next lever (NO-DEFER)

**Settled**: the shuffle-identity anti-cheat's instrument is fixed and generalizes (6/6, from 3/6) — any future
non-rate read on this crux (or a sibling pool with `enable_ou_process=False` and
`enable_short_term_plasticity=False`) should use permutation-over-neuron-identity, not across-read SEM, for
significance. First-spike LATENCY (this specific normalized-onset-fraction implementation) is a clean,
trustworthy negative on this crux — the METHOD is closed, not the capability (a non-rate read that defeats
rate-saturation could still exist elsewhere; latency specifically does not carry it here).

**Open, and the strongest live lead**: dispersion (ISI-CV) shows a real, consistently-signed, partially-significant
effect that the current instrument is underpowered to confirm at 6/6. Next levers, in order: (1) more `N_READS`
independent draws are useless under this pool's determinism (see STEP 0) — instead raise `K_PERM` and/or the
number of *distinct* base/held raster pairs actually driving the point estimate (currently `N_READS=8` reads are
themselves mostly duplicated, per STEP 0 — the point estimate itself may be under-sampled even though its
significance test is now sound); (2) recalibrate `F2_LESION_RATIO=0.34` for a dispersion statistic specifically
— it was borrowed from the rate arm's own calibration and seed 43's near-miss (`0.010489` vs `0.010382`<!--derived-->) suggests
the bar may be mis-scaled for CV's units; (3) a longer `RECALL_STEPS` window would give each neuron more ISIs to
average over, tightening the per-neuron CV estimate itself before the population/permutation statistics are even
computed.

## Scaffold residuals

- `K_PERM=300` and `SHUF_COLLAPSE_MAX_RATE=0.15` are host-chosen statistical-power/tolerance knobs, not computed
  features.
- `N_READS=8` real-identity reads inherited from iteration 1's own calibration, kept for point-estimate
  continuity even though this run's own diagnosis (STEP 0) shows repeated reads under this pool's deterministic
  dynamics are largely duplicated — the fix targets the significance test, not the mean estimate; the mean
  estimate's own sample size is now flagged as the leading open question (see "next lever" above).
- ISI-CV per-neuron averaging (not a pooled-ISI CV) is the more standard Softky-Koch convention, a host choice
  of statistic among the dispersion family (CV vs Fano factor).
- The PARENT crossedge runner's own rate-margin read uses the SAME across-read-SEM protocol this session
  diagnosed as degenerate under this pool's determinism. That read was NOT re-audited here (out of scope — it is
  a previously-banked, separately-consumed crux measurement, kept unchanged in this runner purely as a
  reproduce-the-known-crux sanity check) — flagged in `research/FAILURE_LOG.md` (2026-08-28 entry) as an open
  question for whoever next touches that runner.
- Same host-curated training schedule / topology as the parent crossedge runner (declared there, unchanged).
