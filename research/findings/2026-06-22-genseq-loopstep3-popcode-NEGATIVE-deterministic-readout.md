# Loop-step 3 de-risk #2 Phase B (population coding, OOM-safe) — NEGATIVE: pop-coding is a NO-OP on the deterministic graded readout; the multi-layer gap is deterministic clip-compression, not averageable noise (2026-06-22)

**One-line verdict:** `popcode: narrow512 blocks=3 GRADED_analog n_per_sweep=[1,2,4] -> cumulative_analog_spearman=0.288 (FLAT in n_per) per_layer=[0.846,0.62,0.288] within_pop_std_max=0.00e+00 -> NEGATIVE` — population coding does **not** lift the multi-layer GRADED consolidation cumulative fidelity (stays 0.288 ≈ Phase A's 0.327 single-neuron), because the graded non-spiking `a_cont` readout is **deterministic** (within-population std measured at exactly 0.00e+00), so the `n_per` copies are clones and population averaging cancels nothing.

## Scope / OOM safety (the prior run OOM'd; this one did not)
`research/runners/_genseq_loopstep3_popcode_derisk.py`, GPU, **NO `sim/` edit** (reuse-by-import of the Phase-A graded runner). The prior Phase B OOM'd at the **full 2048-wide** MLP, n_per=8 (~83K neurons / ~823M synapses / ~26 GB). This run **narrows the cortex to a 512-wide dense MLP slice** (`[66,512,512,512]`, 3 dense blocks) and sweeps **n_per ∈ {1,2,4}**. Pre-flight OOM plan printed + asserted < 16 GB ceiling / 8 GB safe budget before each build; built one config at a time, CuPy pool freed between. **Planned bridge sizes** (32 B/edge estimate): n_per=1 → 2,692 neurons / 854,016 edges / ~0.03 GB; n_per=2 → 5,384 / 3,416,064 / ~0.11 GB; n_per=4 → 10,768 / 13,664,256 / ~0.44 GB. Actual peak VRAM ~2.5 GB. Never near the wall.

## Result — pop-coding is a literal no-op (FLAT trend), with the mechanism pinned
| n_per | n_total | cumulative analog-Spearman | per-block [L0,L1,L2] |
|---|---|---|---|
| 1 | 2,692 | 0.288 | [0.846, 0.620, 0.288] |
| 2 | 5,384 | 0.288 | [0.846, 0.620, 0.288] |
| 4 | 10,768 | 0.288 | [0.846, 0.620, 0.288] |

(Phase A single-neuron baseline = 0.327 at scale=20; this run uses scale=20 throughout.)

**WHY-diagnostic (load-bearing):** the within-feature std of `a_cont` *across the n_per population copies*, measured at every block output, is **exactly 0.00e+00** (mean and max). The copies are deterministic clones → population mean ≡ single-neuron value → n_per is a literal no-op. Cross-check: a spiking+threshold-jitter arm (`non_spiking=False`, jitter 3 mV) gives the **byte-identical** 0.288 at n_per 1 and 4 — the membrane stays far below the (jittered) threshold (`a_cont_mean ≈ 0.18`, spike rate ≈ 0), so no per-neuron divergence is injected and pop-coding still cancels nothing.

## The honest interpretation — a category mismatch, then a deeper wall
The documented pop-code lift (CYCLE 91/95, single-neuron 47% → n_per=8 100% → n_per=32 108% of host) cancels **per-neuron STOCHASTIC read-out noise** — averaging many *noisy spiking-rate* estimates of one quantity. **But the graded `a_cont` readout — the very thing Phase A introduced to escape the spike-rate saturation — is deterministic and noiseless.** There is no read-out noise to average, so population coding cannot apply. The two mitigations are a **category mismatch**: the spike-rate readout had noise (but saturated); the graded readout un-saturates (Phase A's win) but is noiseless (so pop-coding is moot).

The remaining per-layer fidelity loss (`0.846 → 0.620 → 0.288`) is therefore **NOT** read-out noise — it is **deterministic signal compression** through the stacked saturating `clip` nonlinearities (each dense block's analog matmul rank is progressively crushed by the `clip(·,0,1)` at the next layer). Per-char it is not a smooth decay: most chars hold L2 ≈ 0.41–0.53, but a couple collapse (`'t'` → −0.23, `'o'` → 0.18) — the third clip destroys the rank for them. This is a **deeper per-layer wall that population coding structurally cannot address**; extrapolation to a larger (cloud-scale) n_per is **moot** (a flat no-op, not a slow lift).

## What this means for the loop-step-3 ladder
- **Phase A stands** (the single most important finding): the rate-saturation wall IS a readout artifact, surpassed by the in-bridge graded path (cumulative 0.327, NO `sim/` edit).
- **Phase B (this) is NEGATIVE** the way it was posed: population coding is not the lever for the multi-layer graded cumulative, because that readout has no noise to average. The lever the diagnosis points to instead is **reducing per-layer deterministic compression** — e.g. fewer stacked saturating stages, a wider linear-response band per block (the clip is the culprit, not noise), or per-layer rank-preserving normalization — NOT more neurons per feature. This is an honest finding under BRAIN-BASED-ONLY: the substrate transmits the analog signal faithfully for ~1 dense layer (L0 0.85) and degrades it deterministically over the stack.
- **NO `sim/` edit**; not committed.

Raw: `research/findings/raw/_genseq_loopstep3_popcode.json`.
