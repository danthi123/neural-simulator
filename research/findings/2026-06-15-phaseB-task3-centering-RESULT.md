# Phase-B Task-3 HARD GATE attempt 2 — brain-based COMMON-MODE REMOVAL (centering) on the spiking SM cortex

**Date:** 2026-06-15. **Status: WALL** (honest NEGATIVE; the controller-decision deliverable). **Discipline:** test-driven, CPU/numpy
(`SIM_BACKEND=numpy`), **NO `sim/` edits** (every ingredient is framework-region / config / runner-level).
Builder + runner changes are all in `research/runners/spiking_sm_cortex.py` (additive, default-off =
Task-1/2 byte-preserved); the gate machinery is unchanged (`tests/test_spiking_sm_cortex.py`,
`_build_synth_64`).

## The corrected diagnosis going in (from the controller)
With the STDP clock fixed (`_step_with_time`) + structural plasticity off + the C1a competitive recipe, the
cortex FIRES and STDP engages (weights 0.05→0.46) — the silence is CURED. But the HARD GATE is PARTIAL:
trained `Pearson(cos(codes), S_true) ≈ −0.07` (anti-correlated), robust across the prior 36-cell grid. The
input carries the structure (rate-level log-input cosine **+0.891**). The corrected diagnosis: the spiking
hub→cortex transform DESTROYS it because the **COMMON MODE is not removed** (the 200 high-frequency common
hubs swamp the cortex drive). The missing op is **CENTERING (common-mode removal)** — exactly what L1 proved
load-bearing (`2026-06-14-L1-learned-cortex-fair-test-GO.md`: the numpy `center_cols(X)=X−X.mean(0)` is THE
op; the spike-robust way is **subtractive-inhibition centering + bounded Hebbian, NO lateral**).

## Decisive localization: the threshold is NOT the destroyer; the common mode is
`research/findings/raw/_phaseB_task3_localize_destroyer.py` reads, on the SAME trained C1a bridge, the cortex
code two ways with plasticity frozen:

| readout | Pearson(cos, S_true) |
|---|---|
| rate-level log-input cosine (the ceiling the input carries) | **+0.891** |
| cortex **spike-count** code (the gate's code) | **−0.074** |
| cortex **g_e (pre-threshold analog conductance)** code | **−0.063** |

**g_e-cos == spike-cos ≈ −0.07.** The common mode survives into the ANALOG excitatory drive — the spiking
threshold is NOT discarding the structure; the common-mode is destroying it upstream. ⇒ **centering
(common-mode removal) is exactly the right target**, confirming the corrected diagnosis and the L1 finding.

### Read-sparsity control (rules out a readout artifact)
`_phaseB_task3_readstrength.py` reads the SAME trained C1a cortex with progressively DENSER codes (longer
window / higher drive). A dense, fully-non-silent readout does NOT recover the structure — it gets *worse*:

| read drive / window | mean spk/neuron | silent_frac | eff_rank | Pearson |
|---|---|---|---|---|
| ds=12, win=40 (the gate's read) | 5.4 | 0.12 | 13.5 | −0.074 |
| ds=12, win=120 | 22.5 | 0.00 | 15.7 | −0.121 |
| ds=30, win=120 | 24.7 | 0.00 | 16.0 | −0.145 |
| ds=60, win=40 | 8.0 | 0.00 | 21.3 | −0.129 |

⇒ the −0.07 is **NOT** a sparse-readout artifact; more cortex firing makes it MORE anti-correlated. The
common mode in the drive is the sole destroyer — it must be removed at the drive, which is what centering
targets.

## The three centering mechanisms tried (cheapest-first), and their best structure Pearson

| Mechanism | What it does | Best structure Pearson | vs centering-OFF baseline (~−0.12) | Verdict |
|---|---|---|---|---|
| **(3) stronger dendritic divisive gain** (smaller `sigma` 0.02→0.005) | down-weights high-EMA common hubs (the existing `/marginal` gain, biting harder) | **−0.089** (sigma 0.01) | ~0 margin | NEGATIVE |
| **(2) `enable_synaptic_scaling`** (Turrigiano per-cortex-neuron renorm) | scales each cortex neuron's incoming weights toward target rate | **−0.092** | ~0 margin | NEGATIVE |
| **(1) feedforward subtractive-inhibition cm pool** (the L1-faithful centering) | all-inhibitory `cm` region tracks the pooled common-mode hub drive, inhibits the cortex = (hub excitation) − (cm inhibition ∝ common mode) | **−0.054** (best fixed; w_hubcm=10, w_cmcx=20) | **+0.064** margin | DIRECTIONAL but INSUFFICIENT |

### The cm pool's directional-but-insufficient band (the load-bearing fixed-cm sweep)
The cm pool DOES center — directionally correct, the **only** mechanism with a non-trivial margin — but
caps an order of magnitude below the +0.30 gate, and stronger cm SILENCES the cortex (a hard ceiling):

| n_cm | hub→cm w | cm→cortex w | cm_bias | ON Pearson | silent | centering-OFF | margin |
|---|---|---|---|---|---|---|---|
| 200 | 5 | 20 | 0 | −0.085 | 0.30 | −0.115 | +0.030 |
| 200 | 5 | 40 | 0 | −0.065 | 0.47 | −0.115 | +0.050 |
| 200 | 10 | 20 | 0 | **−0.054** | 0.33 (alive) | −0.118 | **+0.064** |
| 200 | 10 | 40 | 0 | −0.038 | **0.59 (FAILS collapse-guard)** | −0.118 | +0.080 |
| 200 | (coarse) any | 6–12 | 200–300 | +0.000 | **1.00** | — | (silenced) |

The trade-off is a hard ceiling: as cm inhibition strengthens, the structure improves
(−0.085→−0.054→−0.038) **but `silent_frac` climbs in lock-step** (0.30→0.33→0.59) — the −0.038 "best"
already **fails the NOT-SILENT collapse-guard** (silent 0.59 ≥ 0.5; eff_rank collapses to 6.9). The best
config that KEEPS the cortex alive (silent < 0.5) is ≈ **−0.054**. A tonic `cm_bias` ≥ 200 pA makes cm fire
so hard the cortex goes **fully silent** (Pearson 0.000). There is no Goldilocks zone where the cm pool
subtracts *just* the common mode and leaves the structure — the scalar inhibition removes signal and common
mode together.

A **plastic cm→cortex** (inhibitory STDP, to learn each cortex neuron's own common-mode susceptibility)
**runs away to full silence** (ON & OFF both silent=1.0) — the same spike-fragile recurrent/learned-lateral
failure the L1 Phase-A capstone already found (+0.386 < +0.545; the project DROPPED the recurrent lateral).

### Why (3) and (2) cannot work (mechanistic)
- **(3) divisive ≠ subtractive.** The dendritic gain `g_h=σ/(σ+EMA_h)` DIVIDES each hub's contribution; L1
  specifically found centering = **SUBTRACTION** of the column mean is the op, and that divisive
  normalization does NOT recover the structure (matching the project's prior "gain on/off ≈ same" note).
  Suppressing the common hubs toward 0 also kills any signal they carry and still leaves a structure-less
  residual mean.
- **(2) synaptic scaling is a per-neuron RATE homeostat, not a common-mode subtractor.** It scales ALL of a
  cortex neuron's incoming weights uniformly (common + signal alike) toward a target firing rate — it
  normalizes total drive but cannot SEPARATE the common mode from the signal. No structure recovery.

### The cm pool (1): the framework idiom + what happened
**Framework idiom for the all-inhibitory cm pool (verified, works):** add a `BrainRegion(name="cm",
exc_fraction=0.0, internal_density=0.0)` — the `RegionManager` puts EVERY cm neuron in the inhibitory set
(`(1-exc_fraction)*n`), the bridge's framework path concatenates `inhibitory_indices` across regions and
passes them to `inject_explicit_wiring(output_inhibitory_indices=...)` which flips their trait to inhibitory
(`cp_traits=1`), so a `cm→cortex` `RegionPathway` routes through the **inhibitory** conductance `g_i`
(`bridge.py:5666–5700` splits the E/I matvec by the presynaptic neuron's trait). `hub→cm` is excitatory (hub
neurons are excitatory). So the cortex receives (hub→cortex excitation) − (cm inhibition ∝ common mode) =
the CENTERED drive. **The framework expresses an all-inhibitory source region projecting inhibitorily, with
no `sim/` edit.** (Verified directly: `_phaseB_task3_cm_diag.py` confirms `cm→cortex` raises `g_i` on the
cortex.)

## Tuned params swept
- cm: `n_cm` ∈ {100,200}; `hub_to_cm_weight` ∈ {0.05…10}; `cm_to_cortex_weight` ∈ {6…40};
  `cm_bias_pA` (tonic cm depolarization so cm sits near threshold) ∈ {0,50,100,200,300};
  `drive_scale` ∈ {12,30,60,120}; dendritic gain OFF (`sigma=1e9`) for the cm arms; no-WTA
  (`cortex_exc_fraction=1.0`, `internal_density=0.0`) per the L1 "no lateral" finding; plastic `cm→cortex`
  (inhibitory STDP) tried.

## Outcome: **WALL** (honest NEGATIVE — the decision-relevant deliverable)

Even with brain-based common-mode removal, the spiking SM-cortex does NOT recover the synthetic category
structure: the best of the three centering mechanisms (the feedforward subtractive-inhibition cm pool)
reaches **structure Pearson ≈ −0.054** (margin +0.064 over centering-OFF) — **directionally correct but an
order of magnitude short** of the +0.30 gate, and pushing the cm harder SILENCES the cortex. Synaptic
scaling and the stronger dendritic gain give **no** margin. The gate test stays honestly **`xfail`'d** on
the structure bar (a); its collapse-guard (c), random-projection (b), and permuted (d) checks remain real;
the new g_e-vs-spike localization print documents the diagnosis.

**Why (the mechanism, citable):** L1's load-bearing op is `center_cols(X) = X − X.mean(0)` — a **per-input-
dimension** subtraction performed on the **full-precision analog code BEFORE** the projection. On the bridge,
a single inhibitory `cm` pool can only deliver a **rank-1, ~uniform** inhibition (all cm neurons fire ~
together, inhibiting all cortex neurons ~equally) — a **scalar** common-mode subtraction. But the common
mode's contribution to the cortex drive is **per-cortex-neuron-varying** (each cortex neuron connects to a
random subset of the 200 common hubs), so a uniform inhibition cannot cancel it without also cancelling the
signal (→ silence). This is exactly the deep-research's pre-registered risk #1/#2/#6 = the **Mikulasch-
Priesemann point-neuron limit**: decorrelation / common-mode removal is an **analog / pre-spike** operation
the point-neuron spiking substrate fundamentally cannot do at the per-dimension granularity the algebra
needs. (The same wall the project has hit 5+ times on the conversational whitening / opponency theme.) The
rate-level de-risk could NOT see this — it is visible only with the bridge in the loop, which is precisely
what this HARD GATE was for.

**Controller decision (the fork this result hands up):**
- **(C1b) a guarded, default-off, byte-reviewed `sim/` edit** — a **per-cortex-neuron** centering the
  framework cannot express: either (i) a Diehl-Cook **post-triggered STDP** branch (update only at a post
  spike, using the pre-trace → silence ≠ catastrophic LTD, and the per-synapse rule can shape each cortex
  neuron's own inhibitory subtraction), OR (ii) a **per-postsynaptic-neuron incoming-weight CENTERING**
  step (subtract each cortex neuron's mean incoming weight — the true `x − mean` at the synapse, with
  anti-runaway normalization) — distinct from `synaptic_scaling` (which is a rate homeostat, NOT a mean
  subtractor). Both are small, guarded edits the build plan already budgeted for.
- **(NEGATIVE)** accept the honest finding that the point-neuron spiking substrate cannot realize
  unsupervised competitive similarity-matching with common-mode removal at this scale — a citable result
  under the project's BRAIN-BASED-ONLY standard, mapping the rate→spike wall precisely. (The flat-distinct
  composer and the curated cortex remain the shipped conversational artifacts in parallel.)

The recommendation is **C1b option (ii) [per-postsynaptic-neuron incoming-weight centering] first** — it is
the most direct spiking realization of L1's exact `x − col_mean` op, it is per-dimension (the thing the cm
pool structurally cannot do), and it is a smaller/safer edit than a new STDP rule. If it also under-recovers,
the NEGATIVE is then airtight.

## NO `sim/` edits — confirmed
Every change is in `research/runners/spiking_sm_cortex.py` (the builder + runner: the cm-pool region/pathways,
the synaptic-scaling config flags, the cm-bias drive, all default-off) and `tests/` + `research/findings/`.
`git diff --stat` touches no `sim/` file. The existing default-builder tests pass unchanged (byte-preserved).

## Files
- `research/runners/spiking_sm_cortex.py` — builder + runner (additive centering knobs, default-off).
- `tests/test_spiking_sm_cortex.py` — the HARD GATE (`test_trained_cortex_recovers_structure`).
- `research/findings/raw/_phaseB_task3_localize_destroyer.py` — g_e vs spike-code localization.
- `research/findings/raw/_phaseB_task3_cm_gate.py` — the cm-pool centering gate sweep.
- `research/findings/raw/_phaseB_task3_cm_diag.py` — cm firing diagnostic.
- `research/findings/raw/_phaseB_task3_centering_sweep.py` — mechanisms 2+3 sweep.
- `research/findings/raw/_phaseB_task3_readstrength.py` — read-sparsity control.
