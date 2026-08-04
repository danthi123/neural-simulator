---
type: research-finding
status: complete
date: 2026-08-04
mechanism: adult-mammalian-gpi-snr-autonomous-pacemaking-biophysical-fallback
---

# GPi/SNr autonomous pacemaking: evidence-bounded biophysical fallback

## Decision

Do not replace the reduced V13 tonic-output current with one generic
"GPi/SNr" conductance model. The best-supported next biological fallback is a
separate SNr cell family built around a NALCN-like inward leak, persistent and
spike-generating sodium currents, and Cav2.2-coupled SK recovery. Ih should be
present for hyperpolarized states but should not be required for baseline SNr
pacemaking. BK is optional until a causal tonic-firing effect is measured.

GPi needs a separate fallback. Direct adult mammalian GPi channel measurements
are too incomplete to claim an exact model. The defensible starting point is an
ensemble based on the two electrophysiological classes of rat entopeduncular
nucleus (EP, the rodent GPi homolog), with published GPi/EP model
densities treated as search priors rather than measured biology. Type I cells
need strong Ih and low-threshold rebound; Type II cells need little Ih and an
A-like delayed return from hyperpolarization.

The current `40-80 Hz` V13 target remains an executable output-substrate
contract. It is compatible with SNr in-vivo reports, but it is not a universal
intrinsic rate: juvenile SNr slices commonly produced about `9-21 Hz`, and
adult mouse SNr cells in later slice experiments often occupied roughly the
`25-40 Hz` range. Age, temperature, extracellular ions, severed inputs, and
recording configuration must therefore be recorded with every comparison.

## Evidence boundary

The local catalog and PBR-160 chapter 9 correctly identify autonomous output
firing, a TTX-sensitive subthreshold sodium current, a TTX-insensitive sodium-
dependent inward current, Cav2.2-to-SK control of the afterhyperpolarization,
and Ih in SNr neurons. They do not supply adult channel densities or justify
using the same mechanism in GPi. The local search also exposed the older TRPC3
interpretation; later genetic evidence supersedes it.

The strongest direct constraints are:

- In juvenile rat SNr slices at `37 C`, fast synaptic blockers did not change
  autonomous firing (`11.35 +/- 3.27` to `12.57 +/- 4.27 Hz`). A TTX-sensitive
  inward current began near `-62.5 mV` and peaked at `-72.02 +/- 32.72 pA` near
  `-50 mV`. AHP minimum was `-68.49 +/- 4.93 mV`. HCN block had no significant
  baseline rate effect. Apamin increased irregularity and sometimes caused
  depolarization block; Cav2.2 block weakened the AHP. These results support
  Na drive plus Cav2.2-SK timing, not an HCN-driven SNr clock
  ([Atherton and Bevan 2005](https://doi.org/10.1523/JNEUROSCI.1475-05.2005)). <!--derived-->
- Juvenile rat SNr GABA neurons had transient sodium peak current density
  `148.9 +/- 17.8 pA/pF`, conductance `2481.8 +/- 296.2 pS/pF`, measured
  `E_Na = +49.8 mV`, activation `Vhalf = -30.2 mV`, slope `6.2 mV`, and
  inactivation `Vhalf = -63.3 mV`, slope `8.1 mV`. Recovery had `0.59 ms` fast
  and `35.1 ms` slow components. Persistent sodium activated near `-60 mV`,
  peaked near `-40 mV` (`185.4 +/- 17.5 pA` per cell), and riluzole silenced
  firing. Resurgent sodium at `-40 mV` was `3.3 +/- 0.5 pA/pF`
  ([Ding, Wei, and Zhou 2011](https://doi.org/10.1152/jn.00305.2011)). <!--derived-->
- In `2-3` week mouse SNr, conditional NALCN loss reduced firing from
  `21.0 +/- 1.3` to `11.9 +/- 0.9 Hz` but did not silence cells. At `4 mM K+`,
  control and knockout rates were `30.2 +/- 3.9` and `16.8 +/- 2.0 Hz`.
  NALCN therefore contributes substantially but is not the only pacemaker.
  TRPC3-null and all-seven-TRPC-null cells still fired, directly opposing the
  earlier TRPC3-required account
  ([Lutas et al. 2016](https://doi.org/10.7554/eLife.15271)). <!--derived-->
- Rat EP slices contained `86/104` Type I and `18/104` Type II classified cells;
  the accessible primary abstract does not report animal age.
  Type I cells had strong Cs-sensitive anomalous rectification, weak adaptation,
  and a strong low-threshold calcium rebound; Type II cells lacked that
  rectification, adapted strongly, and showed an A-like post-inhibitory ramp.
  Both had high-threshold calcium spikes and calcium-activated potassium AHPs
  ([Nakanishi, Kita, and Kitai 1990](https://doi.org/10.1016/0006-8993(90)91063-M)). <!--derived-->
- Adult mouse SNr inhibitory responses vary cell to cell. A direct-pathway-like
  dynamic-clamp IPSG used `1.3 ms` rise, `5 ms` decay, and `2 nS` peak. Pallidal
  inhibition can yield a rate overshoot after release, but phase-response
  measurements show that this is generally phase resetting, not an intrinsic
  rebound burst ([Simmons et al. 2018](https://doi.org/10.1152/jn.00535.2018); <!--derived-->
  [Simmons et al. 2020](https://doi.org/10.1152/jn.00678.2019)). <!--derived-->

No located primary study supplies a complete adult SNr or adult GPi set of
NaP/NALCN, HCN, calcium, SK, BK, and spike-current densities and kinetics in the
same cell. That missing dataset is real, not a catalog-search failure.

## Implementation-ready parameter envelopes

All densities below assume specific capacitance `1 uF/cm2`; therefore
`1 nS/pF = 0.001 S/cm2`. "Measured" means directly constrained in the named <!--derived-->
cell/preparation. "Model" means a published fitted value. "Search" is an
explicit engineering envelope to identify with lesion and waveform data; it
must not be reported as a biological measurement.

### SNr family

| Component | Initial value or range | Evidence and required interpretation |
|---|---:|---|
| `E_Na` | `+45` to `+55 mV`; center `+50` | Measured juvenile rat center `+49.8 mV`. |
| Fast Na density | `0.002-0.035 S/cm2`; start `0.01`, log-spaced | Lower end is somatic voltage-clamp measurement (`0.00248`); `0.035` is a published two-compartment model value. The gap is space-clamp/model compensation, not known biological variability. | <!--derived-->
| Fast Na activation | `Vhalf -30.2 mV`, slope `6.2 mV` | Measured juvenile rat. |
| Fast Na inactivation | `Vhalf -63.3 mV`, slope `8.1 mV`; recovery `0.59/35.1 ms` | Measured juvenile rat; preserve two recovery timescales. |
| Persistent Na density | `0.00010-0.00025 S/cm2`; start `0.000175` | Model/search prior. Direct data establish activation and necessity, but not density. | <!--derived-->
| Persistent Na gating | activation `Vhalf -50 +/- 3 mV`, slope `3-6 mV`; inactivation `Vhalf -57 +/- 4 mV`, slope magnitude `4-8 mV` | Phillips model center plus Ding/Atherton activation range. Treat exact time constants as tunable: activation `<0.2 ms`; inactivation `10-30 ms`. |
| Resurgent Na | `0-0.00008 S/cm2`; target current `2.8-3.8 pA/pF` at `-40 mV` | Current density measured; conductance envelope is search-derived. It is a spike-recovery modifier, not the sole tonic drive. | <!--derived-->
| NALCN-like leak | `0.00001-0.00010 S/cm2`, log-spaced; `E_rev -10` to `+40 mV` | Search prior only. SNr knockout constrains function, not density or reversal. Select by reproducing a `35-55%` rate loss without silencing. Use a voltage-independent ohmic current first; do not label its fitted density as measured NALCN. | <!--derived-->
| Passive leak | `0.00002-0.00008 S/cm2`; `E_leak -65` to `-55 mV` | Published SNr model center `0.00004`, searched for input resistance/AHP fit. Keep separate from NALCN. | <!--derived-->
| HCN/Ih | baseline-search `0-0.00005 S/cm2`; hyperpolarization arm up to `0.0002`; `E_h -35` to `-20 mV` | Ih is present, but acute block did not alter baseline SNr firing. Reject fits that require Ih to meet the intact tonic rate. | <!--derived-->
| Cav2.2-dominant Ca | `0.0005-0.0020 S/cm2`; start `0.0007`; dynamic `E_Ca` | Model/search range bounded by published SNr and GPi/EP models. SNr pharmacology identifies Cav2.2 coupling, not density. Generic model gating seed: activation `Vhalf -27.5 mV`, slope `3 mV`, tau `0.5 ms`; inactivation `Vhalf -52.5 mV`, slope `5.2 mV`, tau `18 ms`. | <!--derived-->
| SK | `0.000005-0.00005 S/cm2`; `E_K -95` to `-85 mV` | Search around GPi/EP model density `0.00001`. Fit calcium coupling to AHP and apamin signature; absolute calcium half-activation cannot be transferred until calcium units and nanodomain coupling are measured. | <!--derived-->
| BK | off initially; optional `0-0.00005 S/cm2` sensitivity arm | Presence/function in SNr remains disputed; available evidence concerns metabolic stress or bursting more than normal tonic timing. Add only if BK block predicts a held-out AP-width or high-rate phenotype ([Bao et al. 2012](https://doi.org/10.1371/journal.pone.0052148)). | <!--derived-->

The published SNr model that supplies several centers used `gNa=0.035`, <!--derived-->
`gNaP=0.000175`, `gCa=0.0007`, and `gLeak=0.00004 S/cm2`, with <!--derived-->
`E_Na=+50`, `E_K=-90`, and `E_leak=-60 mV`. It was tuned near `10 Hz`, had a
shallower AHP than recordings, and used a TRPC3 conductance that later knockout
work made biologically obsolete. It is an equation seed, not closure
([Phillips et al. 2020](https://doi.org/10.7554/eLife.55592)). <!--derived-->

### GPi/EP family

Start with two ensembles rather than one averaged cell. Type I receives the
full HCN and low-threshold calcium/rebound machinery. Type II sets HCN near zero
and adds an A-type potassium current fitted to the delayed post-inhibitory ramp.
Use the observed EP class proportion (`0.70-0.90` Type I as a conservative
ensemble range), then test whether the target species supports the same split.

The only implementation-complete density set located is a computational GPi
model fitted manually to rat EP Type I waveforms, using missing kinetics from
other cell types and temperature scaling to `36 C`. Its densities are useful
search centers, not direct GPi measurements:

| Current | Published center (`S/cm2`) | First identification range |
|---|---:|---:|
| fast Na | `0.020` | `0.010-0.040` | <!--derived-->
| persistent/leak Na | `0.000018` in-vitro; `0.000028` in-vivo fit | `0.000010-0.000050` | <!--derived-->
| HCN, Type I | `0.00020` | `0.00010-0.00040` | <!--derived-->
| Kv3.1 fast rectifier | `0.0040` | `0.0020-0.0080` | <!--derived-->
| Kv2.1 delayed rectifier | `0.0010` | `0.0005-0.0020` | <!--derived-->
| SK | `0.000010` | `0.000005-0.000050` | <!--derived-->
| L-type Ca | `0.00050` | `0.00020-0.00100` | <!--derived-->
| N-type Ca | `0.0020` | `0.0005-0.0040` | <!--derived-->

Source: [Johnson and McIntyre 2008](https://doi.org/10.1152/jn.90372.2008). <!--derived-->
Do not transfer these values to SNr, call them adult-primate measurements, or
use the in-vivo Na-leak adjustment as evidence for one specific channel.

## Contradictions and resolutions

| Question | Evidence in tension | Resolution for implementation |
|---|---|---|
| What is the SNr tonic leak? | TRPC3 antibody/pharmacology nearly silenced juvenile cells ([Zhou et al. 2008](https://doi.org/10.1523/JNEUROSCI.3978-07.2008)); TRPC3 and hepta-TRPC knockouts retained firing, while NALCN loss halved it (Lutas 2016). | Use NALCN-like leak as the named fallback. Keep the old `E_rev=-37 mV`, `g=0.0001 S/cm2` TRPC3 model only as a legacy comparator. | <!--derived-->
| Does Ih drive output pacemaking? | Ih is anatomically/electrophysiologically present; HCN blockers did not change baseline juvenile SNr rate. Adult rat EP Type I cells show strong Cs-sensitive rectification and rebound. | SNr Ih is state-dependent, not required at baseline. GPi/EP Type I Ih is load-bearing for hyperpolarization/recovery. |
| Is tonic rate `40-80 Hz`? | In-vivo SNr summaries report `40-80 Hz`; isolated juvenile and adult slices are often slower. | Preserve `40-80 Hz` as the V13 system contract, but separately fit preparation-matched intrinsic rates. Never tune channel density against mixed preparations. |
| Is post-inhibitory overshoot a rebound? | EP Type I shows a low-threshold calcium rebound. Adult mouse SNr overshoot can be explained by phase resetting without intrinsic rebound. | Require distinct GPi and SNr pause tests. Do not add T-type rebound to SNr solely to reproduce a population overshoot. |
| Is BK part of tonic timing? | Some SNr work reports or infers BK; other work explicitly calls its presence debated, with effects clearest under glucose deprivation/bursting. | SK is required first. BK remains a preregistered optional lesion, not a compensating tuning knob. |
| Are exact densities known? | Direct voltage clamp constrains some sodium currents; complete published models provide very different densities. | Fit ensembles to direct currents and causal lesion ratios. Treat model densities as priors and report posterior non-identifiability. |

## Experiments that resolve missing values

1. **Build preparation-matched datasets.** Record adult mouse SNr and adult
   rat/mouse EP separately at `34-37 C`, first cell-attached or perforated patch,
   then whole-cell. Archive age, sex, location, temperature, ion composition,
   blockers, capacitance, input resistance, baseline rate/CV, AP threshold,
   AHP, and complete voltage traces. Do not pool juvenile values into an adult
   density estimate.
2. **Identify sodium drive.** Use a slow `-80` to `0 mV` ramp with TTX and
   riluzole-sensitive subtraction for NaP; use nucleated-patch steps for fast
   and resurgent sodium. For NALCN, combine conditional knockout with Na-to-NMDG
   substitution and dynamic-clamp rescue. Select leak conductance by the
   knockout rate ratio and current-voltage difference, not intact rate alone.
3. **Identify Ca-to-SK coupling.** Apply `1 uM` omega-conotoxin GVIA and
   `100 nM` apamin separately and together. Fit AHP minimum, AP threshold,
   interspike CV, depolarization-block incidence, and calcium transient. This
   separates calcium density from SK density/coupling better than rate fitting.
4. **Bound HCN and rebound.** Apply ZD7288 or Cs during baseline and during a
   fixed hyperpolarization ladder. In EP, add T-type calcium and A-current
   blockers to classify Type I/II. In SNr, reject T-type rebound if pause
   recovery is predicted by the cell's phase-response curve.
5. **Test inhibition with held-out waveforms.** Use the measured SNr IPSG
   (`2 nS`, `1.3 ms` rise, `5 ms` decay), `50-500 ms` inhibitory steps, and
   pallidal barrages. Score suppression depth, first-spike latency, phase shift,
   overshoot area, and return-to-baseline. Randomizing onset phase distinguishes
   a true rebound current from phase accumulation.
6. **Fit an identifiable ensemble.** Use simulation-based inference or ABC over
   the explicit ranges above. Train on baseline waveform plus pharmacological
   lesion effects; hold out inhibitory recovery and a second temperature. Keep
   parameter sets, not one optimum, whenever multiple combinations pass.
7. **Promote only causal components.** A channel enters the fallback only if
   its lesion changes the preregistered phenotype in the measured direction.
   Rate rescue by compensating another conductance is not evidence of the same
   mechanism.

## Replacement gate recommendation

The reduced V13 current should be considered surpassed only when an explicit
SNr or GPi family, with zero host tonic current, passes all existing tonic,
asynchrony, inhibition, recovery, backend, and checkpoint requirements and
also reproduces preparation-matched AP/AHP waveforms plus at least three held-
out lesion signatures: NALCN/NaP, Cav2.2-SK, and HCN during hyperpolarization.
GPi additionally requires Type I/II separation and a rebound/A-current test.

This gate deliberately permits several parameter sets. Present evidence can
identify a biologically constrained mechanism family; it cannot identify one
exact adult mammalian channel-density vector.

## Local evidence consulted first

- `tools/rag/search.sh` paper, catalog, and finding searches for GPi/SNr tonic
  pacemaking, NaP/NALCN, HCN, Cav, SK/BK, pause, recovery, and rebound.
- `references/textbooks/basal-ganglia-reviews/TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.txt`, chapter 9, via the local catalog.
- `references/feature-catalog.md`, entry A.04, via the local catalog.
- The V13 continuous-selector research gate and tonic-output preregistration.
