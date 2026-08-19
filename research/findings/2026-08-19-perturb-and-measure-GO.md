---
type: finding
status: live
date: 2026-08-19
mechanism: functional-connectivity-perturbation-vs-anatomy
lane: instrument
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
instrument: perturb one region of a running multi-region spiking brain, measure the signed Δ firing-rate of every other region → a FUNCTIONAL connectivity matrix F; compare to the ANATOMICAL signed weight-graph W. GO gate (all must hold, 6 seeds): determinism byte-identical F at fixed cfg.seed; F≠W (spearman_rho_nz<1); ≥1 polysynaptic edge/seed (W==0 but |F|>thresh); sign-agreement on active direct edges ≥0.8; lesion dissociation (cut gpi→thal ⇒ gpi→thal AND polysynaptic cortex→thal collapse <40%/<50% retained, upstream cortex→gpi retained >60%); topographic specificity (cortex_N drives its own str_D1_N, selectivity-index>0.7). Controls: pathway LESION (rebuild without the synapses), STATE-DEPENDENCE (gpi pacemaker 900→300 reweights the effective edge), determinism re-run.
artifact: research/findings/raw/perturb_measure/pm_6seed.json
---

# Perturb-and-measure functional connectivity DIFFERS from the anatomical weight graph — a validation instrument (6-seed GO)

**Verdict: GO (6/6 seeds).** Driving one region of the running navigation basal-ganglia spiking
substrate and measuring where the activity goes builds a FUNCTIONAL connectivity map that is
POSITIVELY correlated with, but clearly DISTINCT from, the anatomical weight graph — because
di-synaptic inhibition creates strong effective edges where there is NO direct synapse, and the
global drive state reweights them. The probe passes a pathway-lesion dissociation, a topographic
specificity test, and byte-identical determinism. This is a reusable validation instrument (the
"instrument is part of the emulation"), complementary to the weight-shuffle `dependency_control`.

## Method (external source — deep-research gate)

Randi, Sharma, Dvali & Leifer, "Neural signal propagation atlas of *Caenorhabditis elegans*",
**Nature 623:406–414 (2023)** (doi:10.1038/s41586-023-06683-4). <!--derived--> Instead of only inspecting the static
wiring diagram, they PERTURBED each neuron in the live animal (optogenetics) and MEASURED the
downstream calcium response, building a *functional* connectivity atlas — and found it DIFFERS from
the anatomical connectome: ongoing state and extrasynaptic/neuromodulatory signalling reweight the
effective connectivity, and many functional edges have weak or absent direct wiring (polysynaptic
propagation). The static graph says what CAN connect; the perturbation says what DOES propagate. This
finding ports that method to our substrate as a validation probe.

## Substrate + operating point (all spiking / all synaptic)

Reuse-by-import of the flagship nav basal ganglia (`build_bg_brain_regions`, cluster-A/E config; the
same 42-region / 758-neuron / 128-pathway substrate `_n8_thal_disinhibition_probe.py` drives). The
perturbation is an external drive current onto a region's neurons (+800 pA); the read-out is those
regions' spikes. NO host arithmetic stands between the perturbation and the measured propagation.
Regions are aggregated to 12 canonical TYPES (cortex, str_D1, str_D2, str_PV_FSI, str_striosome,
gpe, gpe_arky, gpi, stn, snc, thal, motor). All plasticity + noise + heterogeneity are OFF (isolation:
measure propagation on a FIXED graph). `cfg.seed` seeds the substrate (per the seed trap — not
`actual_seed_used`); seeds differ in which random pre→post pairs realise each pathway's density.

Operating point = the BG's own working regime (a strong gpi pacemaker, which is what makes it
function as a selector, so the disinhibition route carries signal) plus moderate cortical/striatal/
motor tone so inhibitory responses register too. This point is NOT tuned to flatter the probe: 8 of
11 canonical sign-checks pass at it and the 3 residuals are weak anatomical edges masked on saturated
targets (see residuals).

## Result 1 — F is correlated with W but NOT equal to it (the scientifically interesting part)

Over the union of cells nonzero in either matrix (the fair set — shared zeros inflate agreement),
pooled across 6 seeds:

| quantity | value | reading |
|---|---|---|
| Spearman ρ(F, W) | **0.6249** (std ≈ 0.01) | positive — F partially tracks anatomy — but well below 1 |
| Pearson r(F, W) | **0.6252** | same |
| sign-agreement on ACTIVE direct edges | **1.0** | every direct edge the probe resolves has the CORRECT sign |
| polysynaptic edges / seed (W==0 but \|F\|>0.008) | 18.5 mean, **17 min** | functional edges with NO direct synapse |

The correlation sits at ρ≈0.62 (all six seeds 0.61–0.64), not ~1, and the gap is mechanistic, not
noise. The probe recovers the sign of every direct edge it resolves (sign-agreement 1.0), so the
sub-unity correlation is driven by the **polysynaptic edges** — activity that propagated through an
intermediate region to a target it has no synapse onto.

**Headline polysynaptic edges (anatomical W = exactly 0.0):**

- `cortex → gpi`: F = **−0.15** (pooled), yet there is NO direct cortico-pallidal synapse. The path is
  cortex →(+) str_D1 →(−) gpi: a di-synaptic inhibition.
- `cortex → thal`: F = **+0.02521** (pooled mean), NO direct cortico-thalamic synapse. The path is
  cortex →(+) str_D1 →(−) gpi →(−) thal: a **double inhibition = net disinhibition**, an emergent
  POSITIVE functional edge with a positive sign the raw weight graph cannot express.
- `cortex → motor`, `str_D1 → thal`, `str_D1 → motor`, `thal → gpi`: further disinhibition/relay edges,
  all with zero direct weight.

So the functional map contains sign-flipped, no-synapse edges the anatomical map does not — the Randi
result reproduced on our substrate.

## Result 2 — the difference has a DRIVER: global-drive state reweights the effective edge

Lowering only the gpi pacemaker (900→300 pA) collapses the `cortex → thal` functional edge from
+0.02521 toward ~0 (0.000 at seed 42; ≤0.003 across seeds). <!--derived--> With little tonic gpi activity there is no
inhibition to remove, so the disinhibition route carries no signal — the *effective* connectivity
depends on the ongoing state, exactly the state-dependence Randi et al. attribute to neuromodulation.
This is the mechanistic driver of F≠W, not merely its measurement.

## Anti-cheat 1 — LESION dissociation (the probe measures propagation, not an artifact)

Rebuild the substrate WITHOUT the gpi→thal pathway (structural lesion, no `sim/` edit) and re-run.
Pooled retained-fraction of each edge (|F_lesion| / |F_intact|):

| edge | intact | lesioned | retained | prediction |
|---|---|---|---|---|
| `gpi → thal` (the cut edge) | −0.024 | 0.000 | **0.0** | collapse (cut) ✓ | <!--derived-->
| `cortex → thal` (POLYSYNAPTIC, routes through gpi→thal) | +0.025 | 0.000 | **0.0** | collapse (route severed) ✓ | <!--derived-->
| `cortex → gpi` (UPSTREAM of the lesion) | −0.15 | −0.19 | **1.2737** | unchanged (upstream) ✓ |

A single anatomical cut zeroes the cut edge AND the polysynaptic edge two hops downstream, while
leaving the upstream edge intact — a textbook triple dissociation, consistent across all 6 seeds. The
functional edges track the ACTUAL pathways, so F is measuring propagation, not a drive artifact.

## Anti-cheat 2 — determinism + specificity

- **Determinism:** F is byte-identical on a re-run at fixed `cfg.seed` (max|ΔF| = 0.0). The seed
  genuinely controls the substrate.
- **Topographic specificity:** driving ONE action channel (cortex_N) drives its OWN downstream pool
  str_D1_N by +0.04892 (pooled) while the sibling channels str_D1_{E,S,W} move −0.0035 (mean, slightly
  NEGATIVE via striatal lateral inhibition). Selectivity index = 1.0 (min 1.0 over seeds); |own|/mean|
  sibling| ≈ 11–18×. Stimulating a population drives its known target, not unrelated ones.

## Honest residuals / scope

- **3 canonical edges read ~0** at this operating point (`thal→cortex`, `str_striosome→gpi`,
  `stn→gpi`): all are weak anatomical edges (low density×weight) whose targets are near saturation or
  ceiling, so the effect is masked. This is a REAL property — functional detectability is
  state-dependent (the same reason Result 2 works) — not a probe bug. It caps sign-agreement coverage
  at the ACTIVE direct edges (~10 of 19 direct type-edges resolve above threshold).
- **W is a magnitude PROXY**: sign(source) × Σ density×weight_mean. Cortico-striatal weights (125) dwarf
  gpi→thal (8), so absolute magnitudes are not comparable across edge classes — one reason a rank
  (Spearman) comparison is primary and the correlation is not expected near 1 even without polysynapse.
- **Type aggregation** collapses the 4 action pools per type; the topographic test is the per-pool check.
- **Isolation**: all learning + noise are OFF; F is the propagation on the fixed graph, not a learning
  read-out.

## Reproduce

```
SIM_BACKEND=numpy python -m research.runners._perturb_and_measure_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/perturb_measure/pm_6seed.json
```

Artifact + provenance sidecar: `research/findings/raw/perturb_measure/pm_6seed.json`. The
functional-vs-anatomical comparison is a reusable numpy-only helper
`tools.lab.functional_vs_anatomical(F, W, labels, edge_thresh)` (returns the correlations, direct-edge
sign-agreement, and the polysynaptic / silent / sign-flip edge lists), complementary to
`tools.lab.dependency_control` (weight-shuffle) — the shuffle asks "does function ride on the ACTUAL
weights?", the perturbation asks "what does the wiring ACTUALLY propagate, given the state?".
