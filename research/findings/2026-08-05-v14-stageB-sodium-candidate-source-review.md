---
type: source-review
status: reviewed_pending_claim_corrections
date: 2026-08-05
scope: sodium claims only
packet: research/packets/v14-stageB-fast-channel-online-research-v1.json
---

# Stage B sodium candidate source review

## Decision

None of the three sodium claims can be accepted exactly as written. Their main
scientific content is supported, but each needs a bounded correction. The
sodium ranking in `fast-channel-candidate-ranking` is supported as written once
those corrections are applied. This review does not accept any claim or
authorize model implementation, calibration, or promotion.

## Claim review

<!--derived-->

### `na-milescu-model-iii-coverage`: correction required

Supported: neonatal rat medullary raphe neurons were recorded by whole-cell
voltage clamp in slices; the paper spans -120 to +20 mV and 50 microseconds to
5 seconds; all rates use `k_ij(V)=k0_ij*exp(k1_ij*V)`; Figure 2 data were fitted
simultaneously; and Model III consists of two interconnected copies of the
13-state Model II topology. The reported assay coverage and 10-20% local
parameter non-uniqueness are also supported.

Correction: replace "with an extra slow-inactivated state per mode" with
"each 13-state copy contains the additional I13/I26 state, but mode 2 is
parameterized to have no slow inactivation." The paper explicitly says I26 is
included mainly for symmetry. Retain the limitation that the numerical rates
in Supplemental Table 1 were not independently recovered; without them this is
a topology and assay reference, not an executable baseline.

Exact locators: Milescu et al. 2010, Methods, `Voltage-clamp experiments` and
`Kinetic modeling`; Figures 1-3 and their captions; Results,
`Nav channel kinetic model`, especially the Model II/III and parameter
non-uniqueness paragraphs; Discussion paragraph reporting the six-order time
range. PMCID `PMC2945634`, DOI `10.1523/JNEUROSCI.0445-10.2010`.

### `na-khaliq-downloadable-markov`: correction required

Supported: immutable `rsg.mod` defines 13 states (`C1-C5`, `O`, `I1-I6`, and
blocked state `B`), exact voltage-dependent rates, open-channel block, an
Ohmic current proportional to `O`, and no Q10 term. Every listed parameter
value is exact.

Corrections:

- Replace "Model fitted from dissociated cerebellar Purkinje-cell
  sodium-current data" with "Khaliq et al. adapted the Raman and Bean 2001
  Purkinje sodium scheme; most rate constants were retained, coupling factors
  were revised, and the current was tested in a model of acutely dissociated
  P14-P20 mouse Purkinje somata." The 2003 paper says its sodium-current data
  source was Raman et al. 1997, rather than a new sodium-current fit in that
  study.
- Correct the code locators to `rsg.mod` lines 16-38 (parameters), 87-101
  (states), 103-106 (conductance/current), 114-135 (transition graph), and
  156-195 (rate equations). The packet's final range extends beyond the file.

Exact paper locators: Khaliq et al. 2003, Methods, `Preparation`, `Simulations`,
and `Sodium current`; DOI `10.1523/JNEUROSCI.23-12-04899.2003`, PMCID
`PMC6741194`.

### `na-balbi-nav16-six-state`: locator correction required

The substantive claim is supported. The Nav1.6 file has two closed, two open,
and two inactivated states; conductance is proportional to `O1+O2`; Q10 is 3
relative to 20 C; and rates use the stated sigmoid or sums of sigmoids. The
paper/model package covers I-V traces, conductance-voltage, steady-state
availability, and recovery. The limitation is accurate: experimental
deactivation data were available for Nav1.5 and Nav1.7, not Nav1.6, so this
model alone cannot establish the Stage B Nav deactivation endpoint.

Correction: use `Nav16_a.mod` lines 20-86 (parameters), 105 and 118-127 (Q10,
open probability, and current), 108-115 (states), 130-140 (transition graph),
and 143-160 (rate function and equations). The packet's current line ranges
omit parts of the graph and rate equations.

Exact paper locators: Balbi et al. 2017, Methods, `Experimental data`,
`Electrophysiological features`, `Modelling formalism`, and `NEURON code`;
Results, `State diagram of the model` and `Deactivation curves`; Figures 1 and
3. DOI `10.1371/journal.pcbi.1005737`, PMCID `PMC5599066`.

### Sodium portion of `fast-channel-candidate-ranking`: accept as written

The ordering correctly separates a comprehensive but currently non-executable
CNS topology reference from two immutable executable comparators. "Baseline"
means an unmodified prospective source-transfer run under the sealed Stage B
clamps; it does not imply biological acceptance or permission to tune.

## Authenticated sources

<!--derived-->

| Source | Immutable identity | Exact locator |
|---|---|---|
| Khaliq/Raman ModelDB 48332 | commit `c96405173a17d18999d2a8d63d40899a76d02bdf`; `rsg.mod` blob `1992a31e3b0cde7797c88fbd4300629c541eeb88`; file SHA-256 `1a3382714bd0962665ec31f7dfac2aa3a9e403a5e3d23e29851afec232c4543e` | ranges above |
| Balbi ModelDB 230137 | commit `815a1d7762d0cdccc3a3c6e6bed3a678d15888e4`; `Nav16_a.mod` blob `7ce52ff0f71438ec08ef33ac251677ce4e903efc`; file SHA-256 `69931ced1587944070edb3169a865e9e3e2a42f715b19a8b7b57e72e831ba71d` | ranges above |
| Milescu et al. primary paper | Europe PMC PDF snapshot SHA-256 `36551512da6e423d5866183b82f3d7681cb5f8b37f2f5b297cfa872166c50290` | PMCID and sections above |
| Khaliq et al. primary paper | Europe PMC PDF snapshot SHA-256 `17129961939914920d62c3372b6db51530f53e1ce9606fe6d636bf8b7b4bcfec` | PMCID and sections above |
| Balbi et al. primary paper | Europe PMC PDF snapshot SHA-256 `f29c585ef401bf5287a3b56a8b7b842a50751f577fa9ebb89731f773dda689e7` | PMCID and sections above |

PDF hashes identify files fetched on 2026-08-05; the PMCID and DOI remain the
canonical paper identities.

## Ranked executable baseline recommendation

<!--derived-->

1. **Khaliq/Raman 13-state model, unmodified.** Run this first under every
   sealed Stage B sodium command. It has exact rates, coupled activation and
   inactivation, explicit deactivation paths, conventional and blocked-state
   recovery paths, and an open-channel-blocked state relevant to the reported
   SNr resurgent current.
   Its mouse CNS context is closer than the heterologous comparator. Treat
   temperature and preparation mismatch as declared transfer risks.
2. **Balbi Nav1.6 six-state model, unmodified.** Run it as the lower-cost,
   isoform-specific comparator. It is useful for determining whether the
   Stage B failures require the larger coupled/blocking topology, but its
   Nav1.6 deactivation and resurgent behavior lack direct source validation.
3. **Milescu Model III is not presently executable.** Use its complete-current
   protocol coverage and 26-state modal topology to interpret failures. Do not
   reconstruct or fit missing rates from the figures under the source-transfer
   label; recover Supplemental Table 1 or file a separate prospective
   digitization/calibration protocol first.

The first executable comparison should preserve each source model's original
equations, parameters, and temperature convention, apply only the sealed
voltage commands, declare any temperature mismatch, and score complete current
traces with the existing Stage B analyzer. No conductance adjustment can
rescue a kinetic failure.
