# De-risk (A): spiking divisive-normalization cleanup — NEGATIVE as a deployable fixed-op readout (not seed-robust) — 2026-06-04/05

**Verdict: NEGATIVE (honest boundary).** A spiking matched-filter + divisive-normalization + temporal-integration
cleanup CAN reach numpy parity (1.000) on the composer's real noisy unbind est *per seed*, but there is **no single
fixed operating point that reaches parity across seeds** — the best worst-case over 60 operating points is **0.844**
(margin −0.156 from numpy's 1.000). Shipping it as the composer's `argmax` replacement would regress the validated
capability matrix from 1.000 to ~0.84 worst-case. The disclosed numpy `argmax` readout **stands** (no sub-parity
ship). The mechanism, and a sharp bridge-mechanics discovery, are real deliverables.

> **UPDATE 2026-06-05 — the deeper TWO-STAGE fix was subsequently BUILT + tested (owner-approved), and is ALSO
> NEGATIVE, though it clearly helps.** Adding a spiking INPUT-layer divisive-normalization circuit (the diagnosed
> scale-invariance fix) lifts the seed-robust worst-case from output-only's **0.844 → 0.911** (mean 0.926, per-seed
> 42:0.956 / 43:0.911 / 44:0.911) — input normalization is genuinely the right lever (validated: it rescued seed-42
> from 0.356 to 0.867 at the gentle weight). But 0.911 is still below numpy's exact **1.000**, and the operating
> point is FRAGILE (seed 43 swings 0.2→0.978 across nearby ops). A 0.911 cleanup would still regress the matrix ~9pp.
> So the disclosed readout stands. See the UPDATE section at the bottom.

This is the de-risk gate doing its job: the seed-42 result *looked* like a clean GO (divnorm 1.000) and the
multi-seed gate caught it as an overfit before any composer rewrite.

## What was built + the key discovery (NO sim/ edits)

`research/findings/raw/_spiking_cleanup_divnorm_probe.py` — concept codes as synaptic receptive fields (matched
filter, ON/OFF channels) + a **divisive-normalization circuit**: an INHIBITORY-TRAIT FS pool that pools the concept
population (concept→FS, E_TO_I) and feeds **conductance-based shunting** back (FS→concept, I_TO_E), so
`response_i ~ drive_i / (σ + Σ_j drive_j)`. σ is realized by `syn_reversal_potential_i`. Temporal integration over the
readout window.

**Bridge-mechanics discovery (reusable):** the bridge routes a synapse's weight to `g_e` vs `g_i` by whether the
**presynaptic neuron carries an inhibitory trait** (`cp_traits ∈ inhibitory_trait_indices`), NOT by the wiring plan's
`conn_type` string (`sim/bridge.py` ~5046-5070). So the *prior* WTA cleanup probe (which left
`enable_inhibitory_neurons=False`) was adding its "I_TO_E" weights to **g_e = excitation** — that is why "WTA hurt"
(0/45): it was lateral *excitation*, not inhibition. The mechanism sanity (`_divnorm_mechanism_sanity.py`) confirms a
true inhibitory-trait FS pool produces genuine divisive (rank-preserving, drive-scaled) shunting.

## The measurement (V=320 production codes, real `_unbind_onoff` est, cue-cos ~0.31)

Per-seed sweeps (`_divnorm_sweep.py`, 60 operating points each) + the robust aggregator
(`_divnorm_robust_agg.py`, max-of-min across seeds 42/43/44):

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| numpy oracle | 1.000 | 1.000 | 1.000 |
| divnorm best (each seed's OWN op) | 1.000 | 1.000 | 1.000 |
| no-divnorm global best | 0.956 | 0.933 | 1.000 |

**Each seed is parity-capable** — but at a DIFFERENT operating point. The multiseed at the seed-42-best op
(`_divnorm_multiseed_800.json`): divnorm 42=1.000 / 43=**0.507** / 44=0.986 (mean 0.831). The robust search over all
60 shared operating points (`_divnorm_robust_agg.json`):

```
[ROBUST BEST] min-across-seeds=0.844 mean=0.904  op={w_match:100, bias:-800, w_cfs:25, w_fs:8, einh:-90}
[VERDICT] robust worst-case 0.844 vs numpy 1.000  margin=-0.156  -> NEGATIVE
```

Divisive normalization **helps** (+9pp mean over the matched-filter-only baseline, +23pp on the hardest seed 43) and
the robust sweep already pushed the *output* pooling to its max (w_cfs up to 25). But the worst-case stays at 0.844.

## Root cause + the deferred deeper fix

The failure mode is **scale-variance of the absolute firing threshold**. Each seed's est has a different magnitude
(the unbind output scale varies seed-to-seed), so the fixed concept-bias threshold that isolates the true concept on
one seed is mis-calibrated on another (too high → collapse, e.g. seed 43 = 0.507; too low → saturation tie). Output
divisive normalization (the concept-layer FS pool) standardizes the *output* population but not the *input* drive
magnitude, so it cannot by itself make the threshold seed-invariant.

The fix is **input-layer normalization** — a spiking divisive-normalization circuit on the input (ON/OFF) population
*before* the matched filter, so the matched-filter drive is contrast/magnitude-normalized and the threshold transfers
across seeds. To do this faithfully (no new numpy op) requires its OWN spiking normalization circuit (a second FS
pool on the input layer). That is a genuine cortical mechanism, but it is **more than a thin `argmax` readout
warrants** — consistent with the prior finding (`2026-06-04-spine-item2-cleanup-noisy-est-wall.md`) that the full
cortical cleanup circuit is a legitimate sub-project, and a partial version regresses the validated matrix.

## Biology-translatable insight

`argmax` cleanup quietly relies on **scale-invariance** (it compares dot-products regardless of the est's magnitude).
A spiking readout earns scale-invariance only with **normalization** — and it needs it at BOTH stages: *output*
divisive normalization (concept-layer gain control, validated here to keep the population responsive) AND *input*
normalization (contrast/magnitude gain control, so the threshold is invariant to the input scale). Output
normalization alone is necessary but not sufficient; a readout that is robust across different input-magnitude
regimes (here, seeds) needs input normalization too. This maps the `argmax` "shortcut" onto a concrete two-stage
cortical normalization architecture.

## Decision

**NEGATIVE for a deployable fixed-operating-point spiking cleanup.** The disclosed numpy `argmax` readout stands (no
sub-parity ship that would regress the matrix from 1.000 to ~0.84). The divisive-normalization mechanism + the
g_e/g_i trait-routing discovery are kept as validated infrastructure. The deeper two-stage (input + output)
normalization circuit is the path to a fully-spiking seed-robust cleanup, deferred as exceeding the thin-readout
value — to be revisited if/when the agent is otherwise complete, or folded into a dedicated cortical-cleanup arc.

Per the owner's full-clear sequencing, the cleanup (A) is the *readout* shortcut; the deeper **(B) memory shortcut**
(the bound fact held as a numpy vector + numpy superposition/opponency) is next.

## Artifacts
- `research/findings/raw/_spiking_cleanup_divnorm_probe.py`, `_divnorm_mechanism_sanity.py`, `_divnorm_sweep.py`,
  `_divnorm_multiseed.py`, `_divnorm_robust_agg.py`
- `_divnorm_sweep_seed{42,43,44}.json`, `_divnorm_multiseed_800.json`, `_divnorm_robust_agg.json`
- Backend: CuPy / RTX 3090.

---

## UPDATE 2026-06-05 — the deeper TWO-STAGE fix (input + output normalization): BUILT, helps, still NEGATIVE

The owner chose to build the deeper fix rather than defer it. `research/findings/raw/_spiking_cleanup_2stage.py` adds
a spiking **INPUT-layer** divisive-normalization circuit — a second inhibitory-trait FS pool that pools the est's
ON/OFF input population and shunts it (input→input_FS E_TO_I, input_FS→input I_TO_E) — *before* the matched filter,
so the matched-filter drive is contrast/magnitude-normalized and the firing threshold transfers across seeds. Plus
the validated concept-layer (output) divisive norm. Both FS pools carry the inhibitory trait.

**Calibration matters enormously (the input-FS pools ~1600 neurons → tiny weights):** the first grid (w_in_cfs
100-200) over-shunted to **0.000** (input signal killed). A single-seed sweep found the gentle sweet spot — input
normalization lifts seed-42 from **0.356 (no input norm) → 0.867** at w_in_cfs=0.5, then collapses again by 4.0. So
the mechanism genuinely works; it is just sharply tuned.

**Multi-seed result (seeds 42/43/44, V=320 real est, the output-norm best region + gentle input norm):**

| approach | robust worst-case (min across seeds) | mean | best op |
|---|---|---|---|
| output-only divisive norm | 0.844 | 0.904 | w_match100 bias-800 w_cfs25 |
| two-stage (gentle, narrow grid) | 0.778 | 0.852 | w_match100 bias-600 w_in_cfs0.5 |
| **two-stage (output-best + input norm)** | **0.911** | **0.926** | w_match100 bias-700 w_in_cfs0.5 w_cfs25 (per-seed 0.956/0.911/0.911) |
| numpy oracle | 1.000 | 1.000 | — |

**Verdict: NEGATIVE (improved but sub-parity + fragile).** Input normalization is the right lever (+6.7pp over
output-only → 0.911 seed-robust), confirming the two-stage diagnosis. But the spiking cleanup plateaus at ~0.91 — it
does NOT reach numpy's exact 1.000 — and the operating point is FRAGILE: seed 43 swings 0.2→0.978 across neighbouring
ops, so 0.911 sits on a knife-edge, not a robust basin. A 0.911 cleanup still regresses the validated matrix ~9pp, so
the disclosed numpy `argmax` readout **stands**.

**Why the gap persists (the honest mechanism):** numpy `argmax` is **infinite-precision and exactly scale-invariant
for free**. The spiking two-stage cleanup earns *approximate* scale-invariance (input normalization) and *finite*
precision (temporal integration), and the residual ~9pp is the precision/robustness the rate-coded readout cannot buy
at a fixed operating point on a 320-way comparison with a cue-cos-0.31 est. Reaching a robust 1.000 would need either
(a) a non-rate readout (e.g. a learned/attractor cleanup that settles to an exact stored code), or (b) far more
integration time / a much larger population — both exceeding what a thin `argmax` readout warrants.

**Status of (A):** the two-stage cortical cleanup circuit is now BUILT + fully characterized (input + output divisive
normalization, the inhibitory-trait shunting mechanism, the gentle-calibration requirement, the 0.844→0.911 lift). It
is the best spiking cleanup attempted and a genuine biology-grounded artifact, but it does not reach deployable
seed-robust numpy parity. The disclosed numpy readout stands. Per the owner's full-clear sequencing, the deeper **(B)
memory shortcut** (the bound fact held as a numpy vector + numpy superposition/opponency) is the higher-value next
piece. Extra two-stage artifacts: `_spiking_cleanup_2stage.py`, `_2stage{,_gentle,_final}.json`.
