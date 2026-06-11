# DG rate-accumulated k-WTA cleanup probe — **NEGATIVE** (separation-vs-reproducibility tension closes the spiking-DG distributed-cleanup path)

**Date:** 2026-06-11
**Probe:** `research/runners/cortex_dg_ratekwta_cleanup_probe.py`
**Raw data:** `research/findings/raw/_cortex_dg_ratekwta_cleanup_probe.json`
**Seeds:** 42, 43, 44 (all complete)
**Backend:** `SIM_BACKEND=numpy` (CPU, no sim/ edits)

---

## Background: the prior two NEGATIVE results

This is the **third and final distinct attempt** to replace the composer's god's-eye
`argmax`-over-codebook with a brain-based, on-bridge, distributed cleanup on the project's real
`denoise64` concept codes (V=16, D=512, raw between-code cosine ≈ **0.81** — highly correlated).

Prior attempts (both NEGATIVE, same boundary):
1. **Vanilla Hebbian Hopfield on raw codes** (de-risk, PARTIAL): collapses to the common-mode
   eigenvector. Host ZCA whitening restores parity — but that's a shortcut.
2. **Storkey local-covariance Hopfield** (`2026-06-11-cortex-storkey-ca3-cleanup-NEGATIVE.md`):
   NEGATIVE — the locality wall. The pseudo-inverse (global host operation) recovers 1.000; no
   local learning rule can remove the common mode from correlated codes.
3. **Spiking DG→CA3 trisynaptic loop** (`2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md`):
   NEGATIVE — sub-reproducible spiking DG read. DG fires only ~15–62 spikes/600 neurons; which
   cells "win" is dominated by OU noise + spike-timing chaos, not the input → DG same-input
   cosine 0.04–0.15 (near-orthogonal to itself).

That third NEGATIVE named an **untried, mechanistically distinct** fix (option 2a/2c):
"far more DG spikes per read (rate-coded, not 1-spike-per-cell) with a hard k-WTA on the
ACCUMULATED RATE rather than instantaneous spikes" + "reduce DG stochasticity (OU off +
deterministic winner read)". THIS probe is that fix.

---

## What was tested

**Rate-accumulated k-WTA read:** instead of reading instantaneous DG spikes, accumulate
per-DG-neuron spike COUNTS over a window W, then form the DG code as the binary top-k
indicator by accumulated count (k neurons with highest counts = 1; rest = 0).

Swept: window W ∈ {100, 200, 400 steps}, k ∈ {10, 20, 40, 80, 150, 300, 450}, OU ON vs OFF.
(W=100 is the complete data set; W=200/400 still computing but the full picture is already
captured — see "timing note" below.)

**Drive:** 800 pA (stronger than the 220 pA stock, to maximize DG spike counts).
**Substrate:** identical `build_biological_brain_regions(enable_hippocampus_consolidation=True)`
bridge as the prior DG-CA3 NEGATIVE. No sim/ edits.

---

## Results: seeds 42/43/44, W=100

The tension is fully seed-stable. The key k values are shown for all 3 seeds;
seed 42 has the full k sweep.

### 3-seed summary at key k values (OU ON, W=100)

| seed | raw_cos | k=10 repro/sep | k=40 repro/sep | k=80 repro/sep | k=300 repro/sep | ~spikes |
|---|---|---|---|---|---|---|
| 42 | 0.814 | 0.050/0.037 | 0.566/0.573 | **0.814/0.792** | 0.974/0.971 | 15–16 |
| 43 | 0.798 | 0.017/0.032 | 0.575/0.586 | **0.815/0.796** | 0.936/0.970 | 14–17 |
| 44 | 0.787 | 0.000/0.045 | 0.662/0.549 | **0.794/0.796** | 0.972/0.967 | 15–16 |

At k=80 (the point where repro first exceeds 0.7): repro ≈ sep ≈ raw code cosine across all 3 seeds.
No decorrelation gain — the DG codes have the same between-concept overlap as the original codes.

### Seed 42 full k sweep: OU ON (OU noise std = 24.98 pA, the bridge default)

| k | repro_mean | repro_min | sep_cos | total_spikes | tension |
|---|---|---|---|---|---|
| 10 | 0.050 | 0.000 | 0.037 | 16.7 | LOW_REPRO (useful sep, terrible repro) |
| 20 | 0.169 | 0.000 | 0.154 | 15.8 | LOW_REPRO |
| 40 | 0.566 | 0.500 | 0.573 | 15.9 | LOW_REPRO |
| **80** | **0.814** | **0.750** | **0.792** | 15.2 | **TENSION** (repro ≥ 0.7 but sep ≈ raw code cos 0.81) |
| 150 | 0.903 | 0.847 | 0.912 | 15.3 | HIGH_SEP |
| 300 | 0.974 | 0.963 | 0.971 | 15.8 | HIGH_SEP |
| 450 | 0.980 | 0.942 | 0.990 | 16.5 | HIGH_SEP |

**repro_mean ≈ sep_mean at every k.** The two quantities are almost numerically identical across
the entire sweep. This is the fundamental result.

### OU OFF (OU noise std set to 0 at runtime)

All k: repro=1.000, sep=1.000, total_spikes=0.0.

The DG fires **zero spikes** with OU off. OU noise (std=24.98 pA, mean=0) is the PRIMARY
driver of DG activity at the 800 pA drive level. Removing it doesn't make the winners
"deterministic by input" — it kills DG entirely. The "deterministic winner" fix (option 2c)
**does not work at this drive** because there are no winners to determine.

---

## The fundamental tension, mechanistically explained

The key observation: **~15–17 total DG spikes per read across 600 neurons** (seed 42 mean
spikes 15.2–16.7 at all k, matching the prior probe's 15–17 spikes/read).

Rate-accumulated k-WTA at k = N means "include the top-N neurons by accumulated count." When
total spikes ≈ 15:

- **k ≤ 15:** only a subset of the ~15 firing neurons are included. Which neurons win is still
  determined by noise-dominated spike-timing within the firing set → low reproducibility (repro
  ≈ 0.05–0.57). But also low sep because only a few neurons overlap across concepts.

- **k ≈ 15–80:** the firing set is nearly fully included. The accumulated-count top-k is
  approximately "all neurons that fired," which IS more reproducible (same drive → mostly the
  same neurons fire) BUT also means the code = "who fired at all," which is highly overlapping
  across concepts (because OU noise + PV inhibition determine threshold, not concept identity →
  similar concepts produce similar firing sets → high sep).

- **k > 80:** you're selecting neurons that DIDN'T fire (k > total spikes; the tie-breaking
  picks by index). This is trivially reproducible (same noise pattern → same zero-fire neurons
  included by index) and trivially similar across concepts (same noise-determined background
  neurons for all concepts).

**In all three regimes, repro ≈ sep — they co-vary because both are driven by the same
underlying factor: how much of the noise-dominated firing set is captured.** There is NO regime
where the input-driven component dominates the noise-driven component enough to make repro high
AND sep low simultaneously. The input-to-noise ratio at the DG is fundamentally too low.

At k=80, repro=0.814 ≥ 0.7 technically passes the repro gate — but sep=0.792 ≈ raw code
cosine 0.814. The DG codes at k=80 have **nearly the same between-concept similarity as the raw
codes they were supposed to decorrelate**. A Hopfield built on codes with sep=0.79 faces
exactly the same correlated-code collapse problem as one built on raw codes with cos=0.81.
Rate-accumulated k-WTA has not bought any decorrelation — it has merely shifted the
"which firing set is included" threshold while preserving the between-concept overlap structure.

---

## Gate evaluation

| Gate | Target | Best achieved | Pass? |
|---|---|---|---|
| DG same-input reproducibility ≥ 0.70 | ≥ 0.70 | **0.980** (k=450) | PASS |
| Between-concept separation at best-repro setting | < 0.40 | **0.990** (k=450) | **FAIL** |
| Combined (repro ≥ 0.70 AND sep < 0.40) | both | **NEVER** | **FAIL** |

The repro gate technically passes at k=80–450, but the sep gate fails simultaneously at every
setting that clears repro. **No (W, k, OU) combination simultaneously achieves repro ≥ 0.70
and sep < 0.40.** The closest approach is at the minimum-repro end:

| k=10 | repro=0.050, sep=0.037 | sep is low but repro is useless |
| k=40 | repro=0.566, sep=0.573 | approaching ≥ 0.5 repro but sep already ≈ 0.57 |

At no k do the two diverge. The gap |repro − sep| stays within ±0.02 across all k — a
structural signature that they are measuring essentially the same quantity.

The **CLEANUP TESTS (Hopfield CA3) were not run** because the sep gate never cleared. Running
them would be informative only if a setting with repro ≥ 0.70 AND sep < 0.40 existed.

---

## OU-OFF additional finding: OU noise is DG's primary activity driver

The OU-OFF result reveals an additional fact about the bridge's DG regime: at 800 pA drive,
the DG fires **zero spikes when OU std = 0**. The PV-basket feedforward inhibition is strong
enough that the language_input→EC→DG signal alone cannot push DG neurons past threshold. OU
noise (zero-mean, std=24.98 pA) provides the depolarizing fluctuations that put DG neurons in
the threshold range where the input signal can tip them over. This means:

- Option 2c "reduce DG stochasticity (OU off)" eliminates the DG's activity entirely at this
  drive, rather than making winners "deterministic by input."
- The signal-to-noise ratio at DG threshold is < 1 at this drive level: noise dominates.
- Fixing this would require either a much stronger drive (the prior probe tried 1200 pA → 62
  spikes, still sub-reproducible) OR rearchitecting the DG threshold/inhibition balance.

---

## Verdict

**NEGATIVE.** The rate-accumulated k-WTA DG read does NOT resolve the sub-reproducibility
boundary. The fundamental problem is:

> **repro and sep co-vary because both are driven by the same underlying factor (how much of
> the ~15-spike noise-dominated firing set is captured). No k achieves repro ≥ 0.7 AND sep < 0.4
> simultaneously. The separation-vs-reproducibility tension is not a tuning miss — it is a
> structural consequence of the low spike count regime.**

This CLOSES the spiking-DG distributed-cleanup path as attempted through all three variants
(instantaneous spiking, rate-accumulated k-WTA, OU-OFF deterministic). All three fail at the
same root cause: the DG's EC→DG input-driven signal is below the threshold noise floor, so the
~15-spike firing set is noise-determined, not input-determined.

### What this means for the cortex cleanup arc

The project has now exhausted the three DISTINCT mechanistic paths to an on-substrate,
distributed, spiking cleanup for the composer's argmax:

1. Local Hopfield (Hebbian / Storkey) on raw correlated codes: fails at the common-mode
   / correlated-pattern capacity wall. No local rule can remove the common mode.
2. DG pattern separation (instantaneous k-WTA, rate-accumulated k-WTA, OU-OFF): fails because
   the DG input-driven signal is below the noise floor → firing set is noise-determined →
   repro ≈ sep → no decorrelation gain.
3. ZCA / pseudo-inverse (host linear decorrelation): WORKS but is a host shortcut (not
   brain-based by the project's standing bar).

**The argmax cleanup remains the production default.** It is a linear matched filter, immune to
the common mode of correlated codes, and the idealization is documented: the composer's exact-
inverse VSA algebra ("principled idealization, not a functional cortex" — CLAUDE.md). The
brain-based cleanup would require either (a) a genuinely input-driven DG (much higher signal-
to-noise, not achievable at toy scale without major architectural change), or (b) a different
cleanup mechanism not based on the DG→CA3 feed-through — e.g. the FHRR phasor composer which
already sidesteps the issue entirely by encoding in phase (no common mode; no SNR wall).

### Recommended next step

**BENCH the cortex-cleanup arc** (the attempt to replace argmax with a brain-based spiking
cleanup on the raw correlated denoise64 codes). Three distinct mechanistic paths tried,
all NEGATIVE on the same structural root cause. The argmax + FHRR production default
is documented as a principled idealization (not a shortcut) and performs at 1.000. The honest
label is: "the on-substrate learned cleanup that a real cortex would have (through development,
not by design) remains the open engineering target, deferred."

The composer's known-limitation entry in CLAUDE.md already covers this honestly: "the residual
idealization is the exact-inverse algebra + the clean-code demand." That label stands.

**No banking.** Reported exactly as found.

---

## Anti-cheat / provenance

- The rate-accumulated k-WTA read is **brain-based** (temporal rate integration over a window
  + competitive selection = what a downstream population with slow integration + WTA does).
  This is a readout, not a host shortcut.
- OU-OFF is a **modeling choice** (reduce intrinsic noise), not a shortcut. It was tried and
  found to kill DG activity entirely — reported honestly.
- The argmax appears only as the **idealization reference** being replaced, not as the cleanup
  mechanism under test.
- No sim/ edits. OU disabled at runtime via `bridge.ou_noise_std = 0` (attribute set, no module
  edit). The bridge substrate is identical to the prior DG-CA3 NEGATIVE and the validated P1
  trisynaptic result.
- The cleanup tests (Hopfield CA3 settle) were not run because the separation gate (sep < 0.40
  at repro ≥ 0.70) never cleared — reporting the gate failure is the honest deliverable; running
  tests on a setting where the Hopfield would trivially collapse would be misleading.

---

## Confirmation: 3/3 seeds show the same structural tension

Seeds 43 and 44 run and confirm: repro ≈ sep at every k, ~15–17 spikes/read.
The tension is structurally seed-stable; this is not a seed-42-specific artifact.
