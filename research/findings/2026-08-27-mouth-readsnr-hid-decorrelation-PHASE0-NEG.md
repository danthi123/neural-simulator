---
type: finding
status: negative
date: 2026-08-27
verdict: PHASE-0 NEGATIVE -- the hidden population (hid+hidinh) is already ~uncorrelated at the mouth's real operating point; the recurrent-inhibition decorrelation lever has nothing to decorrelate
mechanism: mouth read-SNR -- hidden-population noise correlation diagnostic (gap#4 / #80, the panel-ranked hidfb lever)
lane: E-language/mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed42.json
  - research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed43.json
  - research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed44.json
  - research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed42_featscale3.json
runner: research/runners/_wkv_mouth_hid_correlation_diagnostic.py
---

# Mouth read-SNR (#80 continued): PHASE-0 diagnostic gate is NEGATIVE -- the hidden population is already decorrelated, so the recurrent-inhibition (hidfb) lever is MOOT as specified

## Context (do not re-derive)

An exhaustive 6-mechanism adversarially-verified design panel ranked RECURRENT-INHIBITION ACTIVE DECORRELATION of
the mouth's hidden population as the top root-cause lever for the read-SNR wall
(`sub_learned_recov_mean` plateau ~0.34-0.37, see
`research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`). That finding already
closed the ENSEMBLE (`--sub-pop`) lever as inert by construction: the P word-pool "clones" for one word are
DETERMINISTIC CONDUCTANCE REPLICAS of a shared hidden drive (same wiring, no independent per-member noise since
the pools never spike and OU noise enters as current, not conductance) -- summing them cancels the pooling gain
exactly. The panel's hidfb lever proposed a DIFFERENT target: not the word-pool replicas, but the upstream
HIDDEN population (hid+hidinh) itself -- if ITS trial-to-trial noise is genuinely correlated across the neurons a
downstream read pools over, a fast E-to-I-to-E recurrent-inhibition circuit sited on hid/hidinh could decorrelate
it and raise the linearly-decodable information at the read.

This is a PHASE-0 diagnostic, run before writing a single line of the hidfb mechanism, to answer exactly that
premise cheaply: is the hidden population's noise actually correlated?

## RAG precedent engaged and reconciled

`research/findings/2026-06-15-offdiagonal-decorrelation-local-mechanism-deep-research.md` (a different subsystem
-- the cortex whitening arc -- but the general result transfers) already established: (1) off-diagonal
cross-neuron decorrelation is mathematically impossible for any per-feature-local mechanism -- it requires a
genuine cross-neuron recurrent circuit; (2) the recommended biological form for exactly that circuit is a
Dale's-law INHIBITORY-INTERNEURON population (King, Zylberberg, DeWeese, J. Neurosci. 2013) -- the same E-to-I-
to-E shape the panel's hidfb lever proposes; (3) FULL whitening over-whitens and collapses
(<!--derived--> quoted from that finding's own CYCLE-73 probe: centering-only +0.307 -> full-ZCA -0.012), so any
real decorrelation mechanism must stay LOW-RANK/regularized. None of that is in tension with this file's result.
That finding's whole argument is about
what a decorrelation circuit can accomplish WHEN correlation exists to remove; it says nothing about whether
correlation exists at THIS site. This file measures that directly, for the first time, at the mouth's actual
hidden-population read.

## Method (measurement only, no sim/ edit)

Additive runner `research/runners/_wkv_mouth_hid_correlation_diagnostic.py`, reuse-by-import of
`BatchedSubstrateReadout` (`_wkv_mouth_readout_eprop_batched_substrate_derisk.py`) -- its `_build_bridge`/`_wire`
run completely unmodified, so the wiring is byte-for-byte the same class the production forward already uses.
All B block-diagonal copies (normally B independent data positions) are driven with the SAME fixed feature
vector, repurposing them as B independent trial-repeats; N further repeated `read_window` passes are run on top
(fresh accumulated OU evolution each pass, matching exactly how the production forward's repeated `batch_margin`
calls already behave across gradient steps). This gives `n_repeats * B` trial samples per neuron.

hid/hidinh neurons receive ONLY the external drive current (`internal_density=0.0`, zero incoming synapses in
this bridge) -- their own `cp_conductance_g_e`/`g_i` is identically 0 always, so there is no such thing as "the
conductance ON a hid neuron" to read directly. What matters for a downstream reader is the conductance a
hid/hidinh neuron INDUCES downstream, which for a linear conductance-based synapse is proportional to that
neuron's own integrated spike count over the read window. The diagnostic therefore reads each hid/hidinh
neuron's spike count (`cp_firing_states` summed post-settle) as the "per-trial integrated conductance it
contributes downstream" -- the correct proxy for exactly the shared-vs-independent-drive question being asked.

For each of the F=256 feature groups (D=128, dual-nonneg split), it gathers the 2*Hp=8 "clone" neurons (Hp=4 hid
+ Hp=4 hidinh sharing that feature's drive) and computes: mean pairwise Pearson correlation rho across trials;
the pooling gain CV(single)/CV(sum-of-clones) vs the ideal sqrt(2*Hp)=2.83; and
Var(sum)/[2*Hp*Var(single)] (1.0 = independent, 8.0 = full common-mode). Feature groups with near-zero variance
(never fire, or saturate identically every trial) are reported as UNDEFINED, not scored as 0 -- 10-19 of 256 at
the decisive settings, the rest evaluable.

Vocab truncation (`--v-diag`, default 24-30 vs the real V=1000) is a COST-ONLY simplification: `_wire()` wires
all Hn=F*Hp hid/hidinh neurons densely onto V*P word-pool neurons, ~2M edges/block at real V -- expensive for a
CPU diagnostic that never reads wpool. wpool has NO feedback path to hid/hidinh (no recurrent edges anywhere in
this bridge), so its size cannot affect the measured correlation, only the irrelevant wiring cost. `ou_std=40`,
`hid_gain=120`, `sub_read_window=120`, `settle_frac=0.2`, `uniform_thresh=True`, `sub_hid_pop=4` are the mouth's
real operating point -- in fact already the class's own defaults, passed explicitly here for the record. The
fixed feature vector is a real state from the checkpoint's own dynamics (a short random-token walk through
`ro.advance`, read via the same `_host_feat` the production forward uses) -- realistic in scale without needing
a corpus load.

Backend: numpy (CPU), per the task's preference -- the diagnostic ran in 16-18s per seed at the decisive
settings (`--B 16 --n-repeats 50 --v-diag 30`, 800 trials/neuron), no GPU contention with the in-flight dendritic
run. The CLAUDE.md seed trap is checked (`cfg.seed` build-twice hash of `cp_neuron_firing_thresholds`,
SEEDED on every run).

## Result: rho is essentially zero, at three seeds and two drive regimes

| seed | feat scale | n trials/neuron | rho mean | rho median | rho min | rho max | var_ratio mean (1.0=indep, 8.0=full common-mode) | gain actual | gain ideal |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 1.0 (real) | 800 | -0.0009 | -0.0015 | -0.07 | 0.1791 | 0.9959 | 2.3181 | 2.8284 |
| 43 | 1.0 (real) | 800 | -0.0013 | -0.0015 | -0.0703 | 0.0885 | 0.9967 | 2.3725 | 2.8284 |
| 44 | 1.0 (real) | 800 | 0.0017 | -0.0015 | -0.0575 | 0.1342 | 1.0123 | 2.3933 | 2.8284 |
| 42 | 3.0 (near-saturating drive) | 800 | 0.0 | -0.0015 | -0.0858 | 0.1466 | 0.9995 | 2.3311 | 2.8284 |

<!--derived--> (seed42 mean spikes/neuron went from 1.230 at feat scale 1.0 to 3.120 at feat scale 3.0, per the
per-repeat console log of those two runs -- not itself a JSON summary field, hence the near-saturating-drive
label above rather than a cited figure.)

<!--derived--> Every reading sits at or below the BDSP precedent the owner cited as "already decorrelated" (rho
~0.03) -- most are an order of magnitude smaller, and several are slightly negative (sampling noise around zero,
not a systematic anti-correlation). `var_ratio_mean` (0.9959 / 0.9967 / 1.0123 / 0.9995 across the four runs
above) sits close to the independent-noise reference point (1.0), nowhere near the full-common-mode reference
(8.0 = 2*Hp). This holds not only at the real feature magnitude but also at 3x the drive (pushing mean hidden
firing from ~1.2 to ~3.1 spikes/neuron over the read window) -- ruling out a low-activity corner case or a
saturation/refractory-driven synchrony effect as an alternative story.

The seed-42 run at real feature scale was also re-executed after a small runner change (adding the
`--feat-scale` argument used for the 3x check) and reproduced the identical summary numbers, confirming the
measurement is deterministic given `cfg.seed` and unaffected by the added code path.

Artifact: `research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed42.json` (plus seed43/seed44/
seed42_featscale3, all cited in frontmatter) -- each carries the full per-feature-group table (`per_feature`),
the seed-trap hash check, the operating-point record, and provenance (`.prov.json` sidecar, argv + git SHA).

## Mechanistic reading (why this is not surprising, in hindsight)

The word-pool ensemble finding's mechanism was structural: the P pool members share IDENTICAL wiring to the SAME
hidden neurons, so they compute the exact same weighted sum P times -- there was never any independent noise to
average over, by construction. The hidden population is a different circuit: `sim/bridge.py`'s
`_draw_ou_noise_samples` draws ONE global `cp.random.randn(n)` per step over ALL n neurons in the pool by
default (no `per_region_ou_seed`/`per_neuron_ou_seed` opt-in active here), giving every neuron -- including the
same-role neuron in a different feature-group clone -- its own independent N(0,1) increment every step. hid and
hidinh neurons DO spike (unlike the subthreshold wpool), and their spiking is genuinely driven by that
independent per-neuron noise on top of a shared deterministic mean (the feature-group's drive). There was no
structural reason to expect correlation here, and the measurement confirms none is present.

This directly reconciles with the offdiagonal-decorrelation finding's own math: a decorrelating recurrent
circuit removes off-diagonal covariance that exists. When the covariance is already ~diagonal (rho ~0, var_ratio
~1.0), there is nothing for an E-to-I-to-E circuit to remove -- adding one at this site would, at best, be inert,
and at worst introduce new correlated inhibitory drive where none existed before (a risk, not a benefit).

## Verdict and redirect

**PHASE-0 GATE: NEGATIVE.** <!--derived--> rho_baseline (the four rho_mean readings in the table above,
-0.0009/-0.0013/0.0017/0.0) is far below the
owner-specified 0.05-0.10 negative threshold, and below the BDSP 0.03 precedent. Per the pre-registered gate,
this STOPS the hidfb lever here -- no `--hidfb` mechanism, no `hidfb` BrainRegion, no E-to-I-to-E wiring is
built by this session. Building it against a population that already reads as independent noise would spend a
decisive 6-seed GPU run testing a mechanism with nothing to act on, and any apparent lift it produced would need
to be attributed to something OTHER than decorrelation (the exact failure mode `tools.lab.attributable_to` exists
to catch).

**Redirect (the honest, valuable part of a Phase-0 negative):** the mouth read-SNR gap
(`sub_learned_recov_mean` ~0.34-0.37 vs the ~0.86-0.90 host-linear-proxy ceiling) is NOT explained by correlated
noise in the hidden population feeding the read. The design panel's other-ranked mechanisms besides recurrent
decorrelation are divisive-normalization (a homeostatic gain-control process the read currently lacks entirely --
the graded margin has no normalization stage) and a predictive-prior pathway (the apical Urbanczik-Senn
two-pathway lever already in flight on the GPU,
`research/findings/2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`, STEP 2 -- a distinct
mechanism from, and not to be conflated with, the gap#4 deep-credit-assignment two-compartment rule that
`2026-07-22-gap4-real-issue-NOT-dendrites` and `2026-05-17-dendritic-credit-assignment-NEGATIVE` already refuted
for hidden-layer credit assignment on spikes). A further, genuinely-untested direction this diagnostic surfaces
directly: since the hidden population IS already an independent-noise population, the bottleneck is more likely
the ABSOLUTE sample count / firing rate the fixed readout weights sum over (a signal-to-noise property of the
linear combination itself, not a redundancy the population carries) -- i.e. a rate/gain lever (more spikes per
neuron per window, not more neurons or less shared noise) is a promising next diagnostic.

## Files

- `research/runners/_wkv_mouth_hid_correlation_diagnostic.py` -- the Phase-0 diagnostic (additive, no `sim/`
  edit, reuse-by-import of `BatchedSubstrateReadout`'s unmodified `_build_bridge`/`_wire`).
- `research/findings/raw/_wkv_mouth_hid_correlation_diagnostic/seed{42,43,44}.json`,
  `seed42_featscale3.json` -- the four runs this finding reports, each with full per-feature-group data and
  provenance sidecars.
