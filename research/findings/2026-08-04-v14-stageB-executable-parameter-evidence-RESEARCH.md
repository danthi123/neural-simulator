---
type: research-finding
status: complete
date: 2026-08-04
mechanism: v14-stageB-snr-executable-parameter-evidence
claim_check: synthesis
---

# V14 Stage B SNr executable-parameter evidence

## Decision

No located source supplies one complete, measured adult-mouse SNr parameter
set. Stage B must therefore fit an ensemble while preserving three distinct
evidence classes:

1. preparation-matched physiology targets;
2. transferred direct channel measurements;
3. published model priors used only to bound unresolved parameters.

The executable packet must retain those labels per parameter. A fitted value
does not become a biological measurement because it reproduces a waveform.

## Best preparation-matched targets

McElvain et al. measured 120 SNr cells from 14 C57BL/6J mice, both sexes,
4-6 weeks old, in 250-300 um coronal slices at 34 C. Picrotoxin (100 uM) and
kynurenic acid (2 mM) blocked fast GABAergic and ionotropic glutamatergic input.
Figure 3A reports:

- spontaneous firing: `21 +/- 15 Hz`;
- ISI CV: `0.10 +/- 0.10`.

These are the primary intrinsic-rate and regularity targets for the first
Stage B screen. They do not supply CV2 or numerical passive, waveform, and AHP
values. Source: [McElvain et al. 2021](https://doi.org/10.1016/j.neuron.2021.03.017),
Figure 3A and STAR Methods.

Sitzia et al. provide mature-adult context from 10-12 month WT mice. The
recording did not block all fast synaptic input, and recording temperature is
not explicit, so these values are held-out context rather than co-fitted as if
they came from the McElvain preparation:

| Measurement | Mean +/- SEM |
|---|---:|
| firing rate | `16.32 +/- 1.90 Hz` |
| reported CV | `15.0 +/- 1.51`, apparently percent |
| interspike voltage | `-51.92 +/- 1.39 mV` |
| threshold | `-38.80 +/- 0.95 mV` |
| AP half-width | `0.53 +/- 0.04 ms` |
| AP amplitude | `62.71 +/- 2.33 mV` |
| reported AHP "amplitude" | `-47.89 +/- 1.95 mV`, apparently absolute voltage |
| AHP duration | `1.90 +/- 0.26 ms` |

Source: [Sitzia et al. 2022](https://doi.org/10.3390/biom12111635),
Methods, Figure 1, and Table 1.

## Transferred direct channel constraints

These measurements are direct SNr biology but come from juvenile rat rather
than adult mouse. Stage B may use them as transfer-bounded targets and must
test temperature/species sensitivity.

- Fast sodium: peak density `148.9 +/- 17.8 pA/pF`; conductance density
  `2481.8 +/- 296.2 pS/pF`; activation Vhalf/slope `-30.2 +/- 0.6 /
  6.2 +/- 0.2 mV`; inactivation Vhalf/slope `-63.3 +/- 1.3 /
  8.1 +/- 0.5 mV`; recovery fast/slow tau `0.59 +/- 0.07 /
  35.1 +/- 6.4 ms`; resurgent current at -40 mV `3.3 +/- 0.5 pA/pF`.
  Source: [Ding, Wei, and Zhou 2011](https://doi.org/10.1152/jn.00305.2011),
  Table 3 and Figures 5-7 and 10-13.
- Kv3-like current: peak composite potassium current at +40 mV
  `772.8 +/- 91.8 pA/pF`; Kv3 fraction at +20 mV `58.8 +/- 5.3%`;
  activation Vhalf/slope `-8.5 +/- 1.6 / 8.9 +/- 0.6 mV`;
  inactivation Vhalf/slope `-49.2 +/- 1.6 / 8.7 +/- 0.6 mV`;
  20-80% rise at +40 mV `0.41 +/- 0.03 ms`; deactivation tau
  `0.82, 1.35, 1.87 ms` at `-60, -50, -40 mV`.
  Source: [Ding, Matta, and Zhou 2011](https://doi.org/10.1152/jn.00707.2010),
  Table 1 and Figures 5, 8, and 9.

No measured SNr fast-Na or Kv3 Q10 was located. Temperature correction values
therefore remain model assumptions and require explicit sensitivity arms.

## Cav2.2-to-SK constraints

Preparation-matched experiments establish the causal mechanism but not its
absolute calcium geometry:

- `100 nM` apamin removed the medium AHP; four of twelve cells entered
  depolarization block, and remaining-cell CV changed `0.054 -> 0.162`.
- `1 uM` omega-conotoxin GVIA removed part of the medium AHP; firing changed
  `11.22 -> 13.97 Hz` and CV changed `0.080 -> 0.140` (`n=6`).

Source: [Atherton and Bevan 2005](https://doi.org/10.1523/JNEUROSCI.1475-05.2005),
Methods and Figures 5-6. These lesion directions are held-out causal gates for
the model; fitting intact rate alone is insufficient.

## Model priors for unresolved calcium handling

Model values below bound experiment arms. They are not direct measurements.

| Prior | Key values | Transfer limitation |
|---|---|---|
| Phillips SNr model | calcium decay `250 ms`; minimum `5e-8 mM`; conversion `1e-8 mM/fC`; dynamic ECa with external calcium `4 mM` | The paper prints SK half-activation `0.4 mM` and `tau_sk=0.1 mS`; both conflict with surrounding text or units and are quarantined pending code/author resolution. |
| Thompson/Snudda adult-mouse fit | shell depth `0.1 um`; calcium relaxation `43 ms`; resting `0.01 uM`; SK half-activation `0.57 uM`; Hill `5.2`; base tau `4.9 ms` | Waveform-fitted model assembled from transferred channel data; geometry and affinity were not measured in SNr. |
| Johnson-McIntyre GPi model | baseline `0.1 uM`; shell `0.2 um`; decay `10 ms`; external calcium `2.4 mM`; SK half-activation `0.741 uM`; tau `6.1 ms` | Mixed-source GPi model, not SNr and not direct calcium measurement. |
| Recombinant SK2 | EC50 `0.62 +/- 0.14 uM`; Hill `3.2 +/- 1.0`; activation `4.1 +/- 0.6 ms` at a 9.5 uM step; deactivation `57.3 +/- 11.1 ms` | Rat SK2 expressed in Xenopus oocytes at room temperature. |

Sources: [Phillips et al. 2020](https://doi.org/10.7554/eLife.55592),
[Thompson et al. 2025](https://doi.org/10.1073/pnas.2528602122),
[Johnson and McIntyre 2008](https://doi.org/10.1152/jn.90372.2008), and
[Hirschberg et al. 1998](https://doi.org/10.1085/jgp.111.4.565).

## Unresolved values

- resting and spike-evoked free calcium in identified adult SNr neurons;
- effective calcium shell depth or accessible volume;
- native Cav2.2-SK2 distance and stoichiometry;
- BAPTA-versus-EGTA coupling in SNr;
- native adult-SNr SK2 affinity and kinetics;
- direct adult-mouse fast-Na and Kv3 kinetics and Q10;
- intrinsic CV2 under the target preparation;
- numerical passive properties and AP base width under complete fast-synaptic
  blockade.

These remain explicit unknowns in the packet. They may be searched as bounded
experiment parameters, but cannot be promoted to measured authority without a
new source or experiment.

## Stage B use

The existing non-executable readiness structure is
`research/specs/v14_snr_stageB_target_packet.json`; this finding tightens its
evidence interpretation before values are promoted into authenticated packets.

The first campaign should fit the preparation-matched intact waveform/rate
targets while retaining parameter ensembles. Confirmation must then hold out:

- NALCN/NaP reduction;
- Cav2.2 block;
- SK block and depolarization-block incidence;
- HCN perturbation during hyperpolarization;
- inhibitory pause and recovery waveforms;
- a second temperature and the mature-adult context.

Only packets that preserve every source locator, evidence class, unit,
uncertainty, and transfer decision may become executable. A successful fit is
evidence for a viable mechanism family, not proof that its unresolved
parameters equal the biological values.
