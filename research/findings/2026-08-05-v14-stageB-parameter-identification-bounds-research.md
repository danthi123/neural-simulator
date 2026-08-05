---
type: research-finding
status: complete
lane: gateb-v14-source-constrained-identification
date: 2026-08-05
claim_check: synthesis
artifacts:
  - research/specs/v14_snr_stageB_kinetic_identification_partition_v1.json
  - research/specs/v14_snr_stageB_source_model_transfer_v1.json
  - research/specs/v14_snr_stageB_structural_successor_v1.json
  - research/specs/v14_snr_stageB_population_digitization_protocol_v1.json
---

# Stage B source-model parameter-identification bounds: evidence review

**Date:** 2026-08-05
**Status:** prospective research finding; no implementation authority
**Scope:** the currently selected Stage B source-transfer lane in `sim/sodium_source_models.py` and `sim/kv3_source_models.py`. This memo does not authorize fitting, alteration of the source graphs, conductance calibration, or a Stage 2 run.

## Decision

No inspected primary source establishes a biological continuous lower/upper bound, cell-to-cell distribution, or confidence interval for any *microscopic* transition constant currently exposed by the four source models. Therefore, there are presently **no safely tunable microscopic constants with source-derived numeric search bounds**. The source vectors are executable point estimates and must remain fixed in the unmodified-source arms.

Direct SNr recordings do provide quantitative constraints on *macroscopic observables*. They can become a likelihood or acceptance target after the documented population panels are digitized, but neither their SEMs nor an output-level acceptance interval may be inverted into independent per-edge rate bounds. The present gate is correctly blocked pending that digitization and the scope correction recorded in `research/findings/2026-08-05-v14-stageB-kinetic-identification-readiness-BLOCKED.md`.

This is deliberately not a generic percentage-based trust-region proposal. Where an exact bound was not found, the table says so.

## Evidence hierarchy and applicability

| Tier | Evidence and use | What it cannot justify |
|---|---|---|
| Direct SNr evidence | Ding, Wei, and Zhou recorded juvenile-rat SNr neurons using nucleated somatic patches. Their population current summaries are the closest biological constraints for the Stage B endpoints. Sodium assays were generally at 30 C, with a separate higher-bandwidth 25 C preparation; Kv3-like assays were at 30 C. | A unique HH gate exponent, a source-model state graph, a microscopic transition-rate interval, or a subunit-specific Kv3.1/Kv3.3 assignment. |
| Cross-preparation source prior | Khaliq is mouse Purkinje/Nav resurgent-current modeling; Balbi is heterologous human Nav isoform modeling; Labro is Kv3.1b mechanism work; Desai is CHO mouse Kv3.3. They provide the exact point vectors and topology for source-transfer comparisons. | An SNr population distribution or a continuous biological parameter range around the source vector. |
| Methodological cross-preparation evidence | Milescu demonstrates all-edge exponential sodium fitting and local non-identifiability. Linkevicius et al. demonstrate cell-aware hierarchical fitting for rat-channel CHO Kv3.1/Kv3.3 recordings across temperature. | A published numeric ensemble for this Stage B graph, or numeric SNr priors. |
| Unsupported if asserted today | Independent perturbation of source rate constants, the existing Labro 20/22.5/25 C scan as a biological interval, and componentwise minima/maxima across source isoforms. | Any biological claim. Such quantities would be engineering choices only. |

### Direct SNr macroscopic constraints (not microscopic rate bounds)

The values below are reported population summaries in the source's stated recording condition. They are useful later as observed-data summaries, with the paper's acquisition/filtering and junction-potential caveats retained. They are **not** proposed parameter bounds.

| Channel and observable | Direct SNr result | Source locator and condition |
|---|---|---|
| Sodium steady-state activation | `V1/2 = -30.2 +/- 0.6 mV`, slope `6.2 +/- 0.2 mV` (Results; `n = 12`). A locally extracted Table 3 entry reports `+/- 0.2 mV` for the midpoint; preserve this source-internal discrepancy rather than silently selecting one. | Ding et al., Fig. 6A / Results / Table 3; normalized conductance from `g_Na = I_Na/(V - E_rev)`, with `E_rev` approximately `+50 mV`; nominally 30 C. |
| Sodium activation at 0 mV | 10--90% rise `0.085 +/- 0.005 ms`. | Ding et al., Fig. 6A / Results; step from -100 mV; 30 C. |
| Sodium decay at 0 mV | `0.191 +/- 0.012 ms`. | Ding et al., Fig. 6B / Results; 30 C. |
| Sodium deactivation | Tail `tau(-40 mV) = 0.099 +/- 0.0089 ms`; `tau(-100 mV) = 0.0481 +/- 0.0026 ms`. | Ding et al., Fig. 9 / Results; single-exponential tail fits; 30 C. |
| Sodium recovery | Fast `0.59 +/- 0.07 ms`, slow `35.1 +/- 6.4 ms`, fast fraction `52.6 +/- 5.7%` (`n = 5`). | Ding et al., Fig. 7 / Results; biexponential recovery; 30 C. |
| Kv3-like activation | Isolated TEA-sensitive G/V `V1/2 = -8.5 +/- 1.6 mV`, slope `8.9 +/- 0.6 mV` (`n = 12`). | Ding, Matta, and Zhou, Fig. 8 / Results; nucleated SNr patches; 30 C. |
| Kv3-like inactivation | `V1/2 = -49.2 +/- 1.6 mV`, slope `8.7 +/- 0.6 mV` (`n = 5`). | Ding, Matta, and Zhou, Fig. 8 / Results; 30 C. |
| Kv3-like activation at +40 mV | 20--80% rise `0.41 +/- 0.03 ms`. | Ding, Matta, and Zhou, Fig. 8 / Results / Table 1; 30 C. |
| Kv3-like deactivation | Tail `tau(-60 mV) = 0.82 +/- 0.06 ms`, `tau(-50 mV) = 1.35 +/- 0.12 ms`, `tau(-40 mV) = 1.87 +/- 0.16 ms`. | Ding, Matta, and Zhou, Fig. 4 / Results; single-exponential tail fits; 30 C. |

The sodium study explicitly distinguishes a 25 C high-bandwidth data set from the usual 30 C data and notes that finite sampling/filtering and capacitance can slow apparent activation/deactivation. The reported liquid-junction potentials were not corrected (about 4.6--4.7 mV sodium; 3.6--4.2 mV potassium). Those are condition/measurement effects to model or retain in the observation record, not permission to shift microscopic voltage parameters freely.

The Kv3-like SNr conductance was pharmacologically isolated, but the paper does not establish Kv3.1/Kv3.3 subunit identity or stoichiometry. It reports an SNr current phenotype, not direct validation of either heterologous source model.

## Current source-model equations and provenance

The values below are the actual defaults currently carried in the source-model modules. Units are `ms^-1` for rate amplitudes, `mV` for voltage locations/scales, and dimensionless where stated.

### Khaliq/Raman 13-state sodium

In `sim/sodium_source_models.py`, the source transfer is a 13-state graph with closed chain `C1 <-> C2 <-> C3 <-> C4 <-> C5 <-> O`, an open blocked state `B`, a parallel inactivated chain `I1 ... I6`, and closed-to-inactivated links. The open-current state is `O`.

The implementation uses:

```text
a(V) = alpha * exp(V / x1)       b(V) = beta  * exp(V / x2)
g(V) = gamma * exp(V / x3)       d(V) = delta * exp(V / x4)
e(V) = epsilon * exp(V / x5)     z(V) = zeta  * exp(V / x6)
alfac = (oon / con)^(1/4)        btfac = (ooff / coff)^(1/4)
```

The `C1..C5 <-> O` chain uses multiplicities `4,3,2,1` forward and `1,2,3,4` reverse. The parallel inactivated-chain transition factors are derived from `alfac` and `btfac`; the closed/inactivated cross-links are likewise derived. This coupling is part of the source graph, not an optional regularizer.

Primary vector and exact source locator: [Khaliq, Gouwens, and Raman (2003), ModelDB 48332 `rsg.mod`](https://raw.githubusercontent.com/ModelDBRepository/48332/c96405173a17d18999d2a8d63d40899a76d02bdf/rsg.mod), parameters lines 16--38, state declarations lines 87--101, graph lines 114--135, rates lines 156--195. The local source digest is recorded in `research/findings/2026-08-05-v14-stageB-kinetic-parameter-authority-RESEARCH.md`.

### Balbi six-state Nav1.6 sodium

The `BalbiNav16Parameters` implementation uses a `C1 <-> C2`, `C2 <-> O1`, `C2 <-> O2`, `O1 <-> I1`, `I1 <-> C1`, `I1 <-> I2` graph. `O1` and `O2` are conducting. Each elementary component is:

```text
r(V, T) = q10^((T - reference_temperature_c) / 10)
          * b / (1 + exp((V - V0) / k))
```

where `b` is `ms^-1` and `V0`, `k` are `mV`. Three composite reverse/branch rates are source-required sums: `C2->C1 = c2c1_extra + c1c2`, `O1->C2 = o1c2_extra + c2o1`, `O2->C2 = o2c2_first + o2c2_second`, and `O1->I1 = o1i1_first + o1i1_second`.

Primary vector and exact source locator: [Balbi et al. (2017), ModelDB 230137 `Nav16_a.mod`](https://raw.githubusercontent.com/ModelDBRepository/230137/815a1d7762d0cdccc3a3c6e6bed3a678d15888e4/Nav16_a.mod), parameters lines 20--86, state declarations lines 108--115, graph lines 130--140, rates lines 143--160. The primary article is [Balbi et al. (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005737), particularly Methods/model construction and Fig. 7. The article's statement that most modeled *observables* lie within experimental standard deviations is not a confidence interval for its microscopic constants.

### Labro Kv3.1b four-state activation/deactivation model

The source model uses the four-state activation/deactivation graph and no inactivation state. Its implementation form is:

```text
alpha_i(V, T) = alpha0_i * exp(z_i * (V - Vhalf) / VT)
beta_i(V, T)  = beta0_i  * exp(-z_i * (V - Vhalf) / VT)
VT = 1000 * k_B * (T + 273.15) / e
```

`alpha0_i` and `beta0_i` are `ms^-1`, `z_i` is dimensionless, `Vhalf` and `VT` are `mV`; `i = p, l, s`. The source reports room temperature rather than a single exact assay temperature and no Q10 law. The current transfer's `20`, `22.5`, and `25 C` sweep is consequently a prospective computational envelope, not a source-derived temperature distribution.

Primary source: [Labro et al. (2015)](https://www.nature.com/articles/ncomms10173), kinetic scheme and Supplementary Information/Table 1; [official supplement PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fncomms10173/MediaObjects/41467_2015_BFncomms10173_MOESM1717_ESM.pdf). Supplementary sensitivity examples set `beta_l` to `0.4`, `0.6`, and `1.8 ms^-1`; these are three deliberate simulation conditions, not a sample distribution, error bar, or biological lower/upper bound.

### Desai Kv3.3 two-gate model

The code retains the published independent-gate form:

```text
alpha_n(V) = k_alpha_n * exp(eta_alpha_n * V)
beta_n(V)  = k_beta_n  * exp(eta_beta_n  * V)
alpha_p(V) = k_alpha_p * exp(eta_alpha_p * V)
beta_p(V)  = k_beta_p  * exp(eta_beta_p  * V)
Popen = n^3 * (0.23 + 0.77 * p)
```

The `k` values are `ms^-1` and `eta` values are `mV^-1`. The control-condition current weights are a topology/current-law constraint. The paper's PKC comparison changes them to `0.9 + 0.1*p`; it is a distinct experimental condition, not a continuous healthy-control range.

Primary source: [Desai et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2494927/), DOI [10.1074/jbc.M801663200](https://doi.org/10.1074/jbc.M801663200), Experimental Procedures and Fig. 3A--E (activation, inactivation, recovery, and deactivation protocols), plus the numerical simulation description. This is a mouse Kv3.3 construct expressed in CHO cells and has no source Q10 law.

## Prospective disposition for every current constant

**Key:**

- **Fixed source transfer**: retain exact source value in the currently selected unmodified-source arm.
- **Coupled/reparameterize later**: do not independently tune. A later, explicitly authorized identification study may expose the stated transformed group while preserving the source equations/graph; no biological numeric interval has been found.
- **Unresolved**: no source-derived numeric search bound/distribution is available.

### Khaliq sodium defaults

| Constant (current value) | Role and units | Evidence tier | Prospective disposition | Bound/distribution result |
|---|---|---|---|---|
| `alpha_per_ms = 150` | Activation-chain amplitude, `ms^-1` | Cross-preparation | Fixed source transfer; later coupled/reparameterize with `beta`, `x1`, `x2` | Unresolved. One source vector only. |
| `beta_per_ms = 3` | Deactivation-chain amplitude, `ms^-1` | Cross-preparation | Fixed; coupled with activation-chain group | Unresolved. |
| `gamma_per_ms = 150` | `C5 <-> O` forward amplitude, `ms^-1` | Cross-preparation | Fixed; coupled with `delta`, `x3`, `x4` | Unresolved. |
| `delta_per_ms = 40` | `O -> C5` amplitude, `ms^-1` | Cross-preparation | Fixed; coupled with `gamma`, `x3`, `x4` | Unresolved. |
| `epsilon_per_ms = 1.75` | `O -> B` amplitude, `ms^-1` | Cross-preparation | Fixed; coupled with `zeta`, `x5`, `x6` | Unresolved. |
| `zeta_per_ms = 0.03` | `B -> O` amplitude, `ms^-1` | Cross-preparation | Fixed; coupled with `epsilon`, `x5`, `x6` | Unresolved. |
| `con_per_ms = 0.005` | Closed-to-inactivated base link, `ms^-1` | Cross-preparation | Fixed; coupled with `coff`, `oon`, `ooff` through `alfac`/`btfac` | Unresolved. |
| `coff_per_ms = 0.5` | Inactivated-to-closed base link, `ms^-1` | Cross-preparation | Fixed; coupled with `con`, `oon`, `ooff` | Unresolved. |
| `oon_per_ms = 0.75` | `O -> I6` rate, `ms^-1` | Cross-preparation | Fixed; coupled with `ooff`, `con`, `coff` | Unresolved. The paper's intentional `0.75 -> 2.3 ms^-1` sensitivity/disease-style perturbation is not a healthy bound. |
| `ooff_per_ms = 0.005` | `I6 -> O` rate, `ms^-1` | Cross-preparation | Fixed; coupled with `oon`, `con`, `coff` | Unresolved. |
| `x1_mv = 20` | Activation voltage scale, `mV` | Cross-preparation | Fixed; coupled with `alpha` | Unresolved. Positivity of `x1` is required to retain the source activation direction. |
| `x2_mv = -20` | Deactivation voltage scale, `mV` | Cross-preparation | Fixed; coupled with `beta` | Unresolved. Negativity of `x2` is required to retain source direction. |
| `x3_mv = 1e12` | `gamma` voltage scale sentinel, `mV` | Cross-preparation/source encoding | Fixed and reparameterize to exactly zero voltage coefficient, not a fit variable | This is numerical representation of a voltage-independent source rate, not an identifiable 10^12-mV biology. |
| `x4_mv = -1e12` | `delta` voltage scale sentinel, `mV` | Cross-preparation/source encoding | Fixed and reparameterize to exactly zero voltage coefficient | Same result. |
| `x5_mv = 1e12` | `epsilon` voltage scale sentinel, `mV` | Cross-preparation/source encoding | Fixed and reparameterize to exactly zero voltage coefficient | Same result. |
| `x6_mv = -25` | `zeta` voltage scale, `mV` | Cross-preparation | Fixed; coupled with `epsilon`/`zeta` group | Unresolved. Negativity preserves source direction. |

### Balbi Nav1.6 defaults

Each `(b, V0, k)` triple below is a single logistic component in the equation above. None is a three-dimensional confidence ellipsoid. The nine published isoform vectors share a model family but represent different channel isoforms; their componentwise extrema must not be used as a Nav1.6 uncertainty box.

| Constant (current value) | Role and units | Evidence tier | Prospective disposition | Bound/distribution result |
|---|---|---|---|---|
| `q10 = 3`, `reference_temperature_c = 20` | Global temperature law; dimensionless, C | Cross-preparation | Fixed source transfer. Any SNr temperature random effect is unresolved and must be introduced as one global condition variable, not per-edge Q10s. | Exact model assumption, not an SNr-measured Q10/CI. |
| `c1c2 = (14, -8, -10)` | `C1 -> C2`; `ms^-1, mV, mV` | Cross-preparation | Fixed; later coupled logistic component group | Unresolved. Keep `b > 0`, `k < 0`. |
| `c2c1_extra = (2, -38, 9)` | Additive part of `C2 -> C1` | Cross-preparation | Fixed; keep additive relation to `c1c2` | Unresolved. Keep `b > 0`, `k > 0`. |
| `c2o1 = (14, -18, -10)` | `C2 -> O1` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k < 0`. |
| `o1c2_extra = (4, -48, 9)` | Additive part of `O1 -> C2` | Cross-preparation | Fixed; keep additive relation to `c2o1` | Unresolved. Keep `b > 0`, `k > 0`. |
| `c2o2 = (0.0001, -10, -8)` | `C2 -> O2` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k < 0`. |
| `o2c2_first = (0.0001, -55, 10)` | First additive part of `O2 -> C2` | Cross-preparation | Fixed; retain sum with `o2c2_second` | Unresolved. Keep `b > 0`, `k > 0`. |
| `o2c2_second = (0.0001, -20, -5)` | Second additive part of `O2 -> C2` | Cross-preparation | Fixed; retain sum with `o2c2_first` | Unresolved. Keep `b > 0`, `k < 0`. |
| `o1i1_first = (6, -40, 13)` | First additive part of `O1 -> I1` | Cross-preparation | Fixed; retain sum with `o1i1_second` | Unresolved. Keep `b > 0`, `k > 0`. |
| `o1i1_second = (10, 15, -18)` | Second additive part of `O1 -> I1` | Cross-preparation | Fixed; retain sum with `o1i1_first` | Unresolved. Keep `b > 0`, `k < 0`. |
| `i1o1 = (0.00001, -40, 10)` | `I1 -> O1` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k > 0`. |
| `i1c1 = (0.1, -86, 9)` | `I1 -> C1` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k > 0`. |
| `c1i1 = (0.08, -55, -12)` | `C1 -> I1` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k < 0`. |
| `i1i2 = (0.00022, -50, -5)` | `I1 -> I2` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k < 0`. |
| `i2i1 = (0.0018, -90, 30)` | `I2 -> I1` | Cross-preparation | Fixed; coupled logistic component group | Unresolved. Keep `b > 0`, `k > 0`. |

### Labro Kv3.1b defaults

| Constant (current value) | Role and units | Evidence tier | Prospective disposition | Bound/distribution result |
|---|---|---|---|---|
| `alpha0_per_ms = (0.05, 6, 1)` for `(p, l, s)` | Forward amplitudes, `ms^-1` | Cross-preparation | Fixed source transfer; later reparameterize as three positive coupled amplitudes | Unresolved. No numeric cell distribution/CI found. |
| `beta0_per_ms = (0.15, 0.6, 0.8)` for `(p, l, s)` | Reverse amplitudes, `ms^-1` | Cross-preparation | Fixed; later reparameterize as three positive coupled amplitudes | Unresolved. `beta_l = 0.4, 0.6, 1.8 ms^-1` are discrete sensitivity cases, not an interval. |
| `z = (3.5, 0.4, 0.001)` for `(p, l, s)` | Gating charges, dimensionless | Cross-preparation | Fixed; later constrain positive and preserve pairwise alpha/beta sign coupling | Unresolved. |
| `vhalf_mv = 6.2` | Shared voltage reference, `mV` | Cross-preparation | Fixed; only a shared shift if an authorized observation model warrants it | Unresolved. Do not independently shift each edge. |
| Runtime `temperature_c` (current transfer evaluates 20, 22.5, 25 C) | Assay condition, C | Unsupported as interval | Record actual experiment temperature when available; do not infer a source distribution | The three existing values are prospective envelope points only. No source Q10 or exact room-temperature bound was found. |

### Desai Kv3.3 defaults and current-law weights

| Constant (current value) | Role and units | Evidence tier | Prospective disposition | Bound/distribution result |
|---|---|---|---|---|
| `k_alpha_per_ms = (0.039, 0.000045)` for `(n, p)` | Forward rate amplitudes, `ms^-1` | Cross-preparation | Fixed source transfer; later reparameterize positive amplitudes jointly with corresponding `eta` | Unresolved. |
| `eta_alpha_per_mv = (0.0467, -0.18925)` for `(n, p)` | Forward voltage coefficients, `mV^-1` | Cross-preparation | Fixed; preserve signs `(+, -)` | Unresolved. |
| `k_beta_per_ms = (0.0868, 0.00246)` for `(n, p)` | Reverse rate amplitudes, `ms^-1` | Cross-preparation | Fixed; later reparameterize positive amplitudes jointly with corresponding `eta` | Unresolved. |
| `eta_beta_per_mv = (0.0067, 0.01075)` for `(n, p)` | Reverse voltage coefficients, `mV^-1` | Cross-preparation | Fixed; preserve signs `(+, +)` | Unresolved. |
| `n` exponent `3` in `n^3` | Current-law topology | Cross-preparation/source equation | Fixed, never a continuous fit parameter in this lane | No population uncertainty reported for a different exponent. |
| Control weights `(0.23, 0.77)` in `0.23 + 0.77*p` | Current-law mixture weights, dimensionless | Cross-preparation/source equation | Fixed control condition; represent as a two-simplex only if modeling explicitly labeled condition effects | Control and PKC `(0.9, 0.1)` are distinct conditions, not endpoints of a healthy-control interval. |
| `temperature_c = None` / no Q10 | Preparation condition | Cross-preparation | Preserve absence of a temperature correction in the source arm | Unresolved; do not import a Q10 from another Kv3 study. |

## Source-derived constraints that are available, but at the correct level

The direct SNr evidence can constrain future fitted predictions, conditional on voltage/reference/junction-potential handling and the source's fitting definitions. For example, treating the paper's quoted mean plus/minus SEM as an observed summary (not as a hard parameter box) gives sodium activation observations around `-30.2 mV` and `6.2 mV`, and Kv3-like deactivation observations around `0.82`, `1.35`, and `1.87 ms`. The Stage B gate currently records acceptance bands such as `-31.4` to `-29.0 mV` for sodium midpoint and `0.70` to `0.94 ms` for Kv3-like `tau(-60 mV)`; those are endpoint-level gate bands, not sources of independent microscopic bounds.

The source-transfer failures reinforce the distinction. The completed unmodified comparisons missed multiple direct SNr endpoints: for example Khaliq sodium deactivation at -40 mV was `1.650 ms` against the recorded `0.0812--0.1168 ms` target band; Balbi sodium inactivation at 0 mV was `1.834 ms` against `0.167--0.215 ms`; Labro Kv3.1b activation midpoint was approximately `0 mV` against `-11.7` to `-5.3 mV`; and Desai Kv3.3 rise at +40 mV was `4.477 ms` against `0.35--0.47 ms`. A mismatch does not identify which edge to vary, much less a defensible edge-specific interval.

## Reparameterization and hard constraints for a later authorized fit

These are parameterization recommendations, not a proposal to run the presently blocked calibration.

1. **Keep source arm versus fitted arm separate.** The source-transfer arm uses exact constants and source temperature handling. A future fitted arm must have a new spec, source asset list, target-data receipt, identifiability protocol, and holdout definition. It must not overwrite source defaults.
2. **Use log amplitudes.** For every positive rate amplitude use `a = exp(u)` (or a source-centered log contrast `a = a_source * exp(u)`). This protects positivity without arbitrary rate floors or ceilings. Apply it to Khaliq amplitudes, Balbi `b`, Labro `alpha0`/`beta0`, and Desai `k`.
3. **Encode directional voltage dependence by sign, rather than post-hoc clipping.** Use `x = s * exp(u)` for Khaliq finite voltage scales, with source sign `s`; use `k = s * exp(u)` for Balbi logistic scales; use `eta = s * exp(u)` for Desai coefficients; and `z = exp(u)` for Labro gating charges. This preserves positive rates and the source's activation/deactivation directions at all voltages.
4. **Replace Khaliq sentinels with exact zero coefficients in any fitted representation.** Write a rate as `a * exp(c V)`, with `c = 0` for the source voltage-independent `gamma`, `delta`, and `epsilon` paths. Do not let `1e12 mV` behave like a biological parameter or create ill-conditioned gradients.
5. **Retain algebraic and graph coupling.** Recompute Khaliq `alfac`/`btfac` from the four base rates; retain all Khaliq multiplicities and links. Retain Balbi additive component sums rather than fitting a total and its parts independently. Retain Labro's common `Vhalf` and paired plus/minus voltage dependence. Retain Desai's two gates, `n^3`, and control weights summing to one. Do not add states, remove links, alter open-state membership, or change exponents during source-lane fitting.
6. **Do not impose detailed balance without a source-derived thermodynamic model.** These source graphs are kinetic mechanisms with specified directed rates and derived relations, not a documented equilibrium free-energy parameterization. The safe hard constraint is exact preservation of their documented topology and algebra, not an invented cycle-balance condition.
7. **Make condition variables global.** Temperature, voltage offset/junction-potential treatment, and recording bandwidth should enter as explicit assay/observation-condition variables shared by all rates affected by that condition. Do not fit one temperature factor per edge. No source establishes a numeric SNr Q10 hierarchy for this lane.
8. **Use numerical validity checks that are physical, not generic bounds.** On the predeclared voltage/time mesh, reject nonfinite rates, invalid generator rows, nonfinite occupancies, occupancy outside numerical tolerance, and solver failure. Do not replace this with arbitrary maximum-rate caps. Existing source-state initializers and exact graph solvers remain the reference.
9. **Preserve consumer-GPU tractability structurally.** Batch the fixed 13-state, 6-state, 4-state, and 2-gate systems with static array shapes; retain the current uniformization/scaling-squaring path for Labro and analytic independent-gate path for Desai. Fit a low-rank set of source-centered contrasts only after profile analysis supports them. Avoid state expansion, per-edge random effects, or waveform-level particle methods until data identify a smaller hierarchy.

## Defensible route when microscopic bounds are absent

The appropriate output is a hierarchical *strategy*, not invented numeric authority. After authorized digitization/acquisition, use a source-centered formulation such as:

```text
theta_cell = transform_inverse(transform(theta_source) + L z_cell)
z_cell ~ Normal(0, I)
y_cell,protocol = observation_model(current_model(theta_cell, protocol, condition), error)
```

`transform` is the constrained map described above. Initially `L` should be structurally sparse at the macro-kinetic group level (for example, shared activation, deactivation, and inactivation contrasts) and fixed to zero for unidentifiable directions. There is no present evidence for a numeric covariance, rank, or standard deviation; these must be estimated only where repeated-cell data and profiles support them. The direct SNr SEMs belong in the observation/summary likelihood and do not authorize an edge-level Gaussian prior.

First compare the complete source vectors as discrete model/prior choices. That is supported by the evidence. Permit a continuous source-centered contrast only after it improves prespecified held-out population curves while retaining an uncertainty profile that excludes a flat/unbounded direction. Constants remaining flat must be reported as unresolved, not selected from an optimizer's arbitrary box.

### Required experimental/data design before numeric priors

1. Digitize or obtain the exact primary population curves named in the blocked readiness record, with source figure/panel, units, normalization, condition, point extraction method, uncertainty representation, and asset digest. Do not synthesize traces from retired Stage B models or summary fits.
2. Preserve complete voltage-command protocols, junction-potential policy, sampling/filtering, temperature, and ionic conditions. Fit activation, inactivation, recovery, and deactivation jointly; scalar `tau` summaries alone cannot identify a multi-state graph.
3. For a new SNr data set, use repeated cells with the same protocol at predeclared temperatures and independently randomized train/validation/held-out protocols. Measure the direct current phenotype before asserting Nav1.6 or Kv3.1/Kv3.3 molecular composition.
4. Run structural/practical identifiability profiles and posterior predictive checks on population curves and held-out voltage protocols. Report correlated identifiable combinations, not only individual edge estimates.
5. Keep conductance as a separately declared nuisance/measurement scale if it is ever introduced. Do not compensate kinetic errors with conductance fitting.

## RAG workflow receipt and local catalog evidence

Local retrieval was completed before online escalation, following `docs/RESEARCH_ESCALATION_WORKFLOW.md`. The standard worktree-safe command and receipt were:

```text
bash tools/rag/search.sh 'Stage B SNr sodium Nav1.6 parameter bounds' 5 --corpus catalog
date: 2026-08-05
corpus: catalog
index: /home/dant123/Projects/rag_index/llamaindex_full
elapsed: 4.43 s
```

The returned catalog records, in rank order, were Balbi ModelDB Nav1.6 author implementation (`source-5e86fe863279f2dc`), Khaliq author-supplied Raman/Bean resurgent sodium model (`source-43fe19c85f078a9c`), Khaliq paper record (`source-1bf...`), Desai record (`source-5d3...`), and Milescu record (`source-cc...`). The RAG runner reported `LLM disabled MockLLM`; this was an indexed-source discovery receipt, not evidence extraction.

Direct local catalog/source records then read for claims were:

- `references/source-7842bd4596c87642-ding-wei-and-zhou-2011-molecular-and-functional-differences-in-v.md` for direct SNr sodium values and condition caveats.
- `references/source-0e9002067340eb4a-ding-matta-and-zhou-2010-kv3-like-potassium-channels-are-require.md` for direct SNr Kv3-like values and subunit-composition limitation.
- `references/source-5e86fe863279f2dc-balbi-et-al-2017-modeldb-230137-nav1-6-author-implementation.md` and `references/source-43fe19c85f078a9c-khaliq-et-al-2003-author-supplied-raman-bean-resurgent-sodium-mo.md` for immutable source-model locators.

The project findings/specifications read as the active gate record were `research/findings/2026-08-05-v14-stageB-fast-channel-state-family-research-gate.md`, `research/findings/2026-08-05-v14-stageB-kinetic-parameter-authority-RESEARCH.md`, `research/findings/2026-08-05-v14-stageB-kinetic-identification-readiness-BLOCKED.md`, `research/findings/2026-08-05-v14-stageB-fast-channel-local-source-audit.md`, `research/findings/2026-08-05-v14-stageB-source-model-transfer-NO-CANDIDATE.md`, `research/specs/v14_snr_stageB_kinetic_identification_partition_v1.json`, `research/specs/v14_snr_stageB_source_model_transfer_v1.json`, and `research/specs/v14_stageB_source_model_transfer_source_manifest_v1.sha256`.

## Online primary-data and author-artifact check

| Source/artifact | Exact technical finding | Parameter-bound outcome |
|---|---|---|
| [Ding, Wei, and Zhou (2011)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3234097/) | SNr nucleated-patch sodium population kinetics and voltage dependence; source locations above. | Direct output constraints only; no microscopic source-model ensemble. |
| [Ding, Matta, and Zhou (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3059163/) | SNr Kv3-like population kinetics, G/V, and inactivation; source locations above. | Direct phenotype constraints only; no Kv3.1/Kv3.3 microscopic ensemble. |
| [Khaliq, Gouwens, and Raman (2003)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6741194/) and immutable ModelDB module above | Complete point vector and 13-state graph. The paper's selected `O -> I` perturbation is a scenario/sensitivity intervention. | No CI/distribution or healthy numeric interval for edge constants. |
| [Balbi et al. (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005737) and immutable ModelDB module above | Nine isoform-specific fitted vectors sharing a six-state family, heterologous experimental basis, model Q10. | Isoform variation is not a within-Nav1.6 distribution; no Nav1.6 parameter covariance/CI was found. |
| [Labro et al. (2015)](https://www.nature.com/articles/ncomms10173) and official supplement above | One Kv3.1b vector; Supplementary Table 1 sensitivity values including three `beta_l` settings. | No numeric uncertainty/ensemble; sensitivity settings are not bounds. |
| [Desai et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2494927/) | One CHO/mouse-Kv3.3 vector and two condition-specific current-weight choices. | No control-condition microscopic distribution or Q10. |
| [Milescu et al. (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2945634/), DOI [10.1523/JNEUROSCI.0445-10.2010](https://doi.org/10.1523/JNEUROSCI.0445-10.2010) | Fits a coupled 26-state (two 13-state) sodium scheme with `k_ij(V) = k0_ij exp(k1_ij V)` to neonatal-rat medullary-raphe data. The authors report that roughly 10--20% local parameter changes around an optimum can make little difference to fitness. | Evidence of local non-identifiability, not a biological 10--20% interval. Supplementary microscopic table was not recovered as an SNr or current-source-model bound. |
| [Linkevicius et al. (2026)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013078), [data DOI](https://doi.org/10.7488/ds/8052), [author code, immutable commit](https://github.com/dom-linkevicius/SciMLHHModels.jl/tree/26dc63b6cd5de79536731f072cf6c4d28328bb00) | Cell-aware two-gate nonlinear mixed-effects fits for rat Kv3.1/Kv3.3 CHO cells at 15/25/35 C. Table 1 reports Kv3.1 `26` cells (`9/9/8`) and Kv3.3 `49` (`8/26/15`); Eqs. 8--17 use fixed effects, eight random effects, covariance `Omega`, and observation `sigma`. Author repository stores selected fits as Julia serialized artifacts; DataShare supplies raw/derived Channelpedia-derived NWB data (about 9.7 GB archive). | Strong support for a hierarchical *form* and reanalysis route. No inspectable tabulated individual parameters or numeric `Omega` for the Stage B source graphs was found; the binary artifacts/raw data require execution/reanalysis before numbers may be cited. |
| [Ranjan et al. (2019) Channelpedia](https://www.frontiersin.org/journals/cellular-neuroscience/articles/10.3389/fncel.2019.00358/full) | Cell-resolved CHO rat-channel recordings and API/NWB data infrastructure, including large-scale voltage-clamp traces. | An acquisition resource, not direct SNr evidence and not a published bound for the present source graphs. |

The online check found no exact published parameter ensemble, cell-to-cell covariance, or source-derived continuous interval for the listed Khaliq, Balbi Nav1.6, Labro, or Desai constants. That statement is scoped to the inspected technical papers, supplements, ModelDB modules, author repository/artifacts, and Channelpedia-style data sources above; it is not a claim that no such data can exist anywhere.

## Bottom line for the current Stage B lane

- **Fixed now:** every source-model default and source graph/current-law constant listed above, including zero-voltage-dependence sentinels represented as exact zero coefficients in any future fitted representation.
- **Safely tunable now with biological numeric bounds:** none.
- **Coupled/reparameterize only in a future authorized fit:** positive rate amplitudes and sign-constrained voltage terms, while retaining the exact named algebraic couplings and topology.
- **Unresolved:** all continuous microscopic rate/voltage parameter ranges, population covariance, SNr-specific Q10, and SNr Kv3 molecular composition.

The next evidence-generating action is the existing blocked-gate requirement: obtain/digitize the source population panels with provenance and uncertainty, then assess identifiable coupled contrasts against prespecified held-out protocols. Until then, source vector transfer remains the defensible comparison and no numeric microscopic search box should be introduced.
