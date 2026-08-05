---
type: research-finding
status: complete
date: 2026-08-05
mechanism: v14-stageB-fast-channel-local-source-audit
claim_check: bounded-local-primary-source-audit
web_access: prohibited-not-used
---

# Stage B fast-channel local source audit

## Decision

The locally stored Ding studies are sufficient to define current-level voltage-clamp targets for fast transient sodium and Kv3-like current. They are not sufficient to identify unique channel-state equations.

The papers report command protocols, normalized steady-state curves, current rise or decay measurements, and selected single- or biexponential current fits. They do not report complete voltage-dependent transition rates, a Markov scheme, individual Hodgkin-Huxley gate trajectories, or the gate powers that generated the measured currents. Consequently, another direct conversion of the Ding current time constants into independent `m`, `h`, `n`, or `q` time constants would remain a model prior. The Stage B successor needs either an independently sourced state model constrained against these Ding assays or a deliberately identifiable empirical current-state model whose unmeasured choices remain explicit.

No web source was consulted. This audit read the two locally retrieved primary full texts, their local catalog records, the parameter-research state, current repository equations, and prior findings.

## Local evidence inspected

<!--derived-->

- Ding, Wei, and Zhou, *Molecular and functional differences in voltage-activated sodium currents between GABA projection neurons and dopamine neurons in the substantia nigra*, DOI `10.1152/jn.00305.2011`. Local full text: `/home/dant123/.local/share/neural-simulator/scholarly-fulltext-v14-stageB/content/4d77ff156fc7d7f7bbe48c455d6a5cb05996c1d1b4acc4325a164fb8de369001.html`; catalog record: `/home/dant123/Projects/sim-catalog/references/source-7842bd4596c87642-ding-wei-and-zhou-2011-molecular-and-functional-differences-in-v.md`.
- Ding, Matta, and Zhou, *Kv3-Like Potassium Channels Are Required for Sustained High-Frequency Firing in Basal Ganglia Output Neurons*, DOI `10.1152/jn.00707.2010`. Local full text: `/home/dant123/.local/share/neural-simulator/scholarly-fulltext-v14-stageB/content/bb08674efd0db8fcae5c578ac5e1ba3c07493a6e73bb9e3c87c048751b6067d1.html`; catalog record: `/home/dant123/Projects/sim-catalog/references/source-0e9002067340eb4a-ding-matta-and-zhou-2010-kv3-like-potassium-channels-are-require.md`.
- Retrieval and prior-search record: `research/parameter_research/v14_stageB_fast_channel_primary_state.json`.
- Prior equation and result records: `sim/kernels.py`, `research/findings/2026-08-05-v14-stageB-successor-preregistration-audit-CORRECTION.md`, `research/findings/2026-08-05-v14-stageB-fast-channel-source-transfer-STRUCTURAL-NO-GO.md`, and `research/findings/2026-08-05-v14-stageB-v3-failure-diagnostic-STRUCTURAL-NO-GO.md`.

The catalog entries are discovery records marked pending review and explicitly do not accept scientific claims. Numerical statements below were checked against the locally stored full text.

## What Ding measured

### Fast transient sodium current

Preparation and instrumentation:

- Nucleated somatic patches from SNr GABA neurons in 16- to 24-day-old Sprague-Dawley rats; most recordings at `30 degC`.
- Sodium current was isolated by offline subtraction after `1 uM` TTX. Potassium and calcium currents were blocked; leak was subtracted with P/4.
- Signals were sampled at `50 kHz` and low-pass filtered at `10 kHz`. The authors state that filtering and stray capacitance probably slowed the measured activation and deactivation. Additional `25 degC` recordings used `200 kHz` sampling and `30 kHz` filtering, but these are a separate condition.
- Junction potentials of about `4.6-4.7 mV` were not corrected.

Measured current-level equations and values:

```text
g_Na(V) = I_Na(V) / (V - E_rev), with E_rev estimated near +50 mV

f(V) = 1 / (1 + exp(+(V - Vhalf)/k))   [decreasing/inactivation]
f(V) = 1 / (1 + exp(-(V - Vhalf)/k))   [increasing/activation]
```

The activation fit is a normalized conductance curve, not a directly observed activation gate. The reported SNr values are `Vhalf = -30.2 mV` and `k = 6.2 +/- 0.2 mV`. The Results prose gives `Vhalf = -30.2 +/- 0.6 mV` (`n=12`), while the locally extracted Table 3 gives `-30.2 +/- 0.2 mV`. The repository currently uses `0.6 mV`; the discrepancy must not be silently erased.

From a hold at `-100 mV`, tests ran from `-80` through `+30 mV` in `5 mV` increments. At a step from `-100` to `0 mV`, the measured current had a `10-90%` rise time of `85 +/- 5 us` and a `10-90%` decay time of `191 +/- 12 us`. The local HTML text does not state the activation test-pulse duration; the repository's `20 ms` duration is read from or inherited through the filed protocol, not established by the extracted prose alone.

For deactivation, the patch was held at `-90 mV`, stepped to `-120 mV` for `50 ms`, opened at `0 mV` for `200 us`, and then stepped for `50 ms` to commands from `-100` through `-20 mV` in `10 mV` increments. Each tail current was fitted with one exponential. At `-40 mV`, the SNr current-tail time constant was `99 +/- 8.9 us` (`n=7`); at `-100 mV` it was `48.1 +/- 2.6 us`. The paper provides a voltage plot for the other commands but no tabulated numerical series in the local text.

The empirical current-tail fit can be represented generically as:

```text
I_tail(t; Vcmd) = C(Vcmd) + A(Vcmd) * exp(-t / tau_tail(Vcmd))
```

The paper reports the single-exponential fit and selected `tau_tail` values. It does not specify a channel-state equation that makes `tau_tail` equal to an activation-gate time constant, nor does it publish `A`, `C`, microscopic rates, or a complete `tau(V)` function.

Recovery used a hold at `-90 mV`, `50 ms` at `-120 mV`, `300 ms` at `0 mV`, a variable recovery interval at `-120 mV`, and a `20 ms` test at `0 mV`. Normalized recovery was fitted by a sum of two exponentials. SNr values were fast tau `0.59 +/- 0.07 ms`, slow tau `35.1 +/- 6.4 ms`, and fast contribution `52.6 +/- 5.7%` (`n=5`). The full text does not enumerate the recovery-duration ladder. These measurements establish two current-recovery timescales, but not two independent inactivation particles with a shared voltage-dependent steady state.

### Kv3-like fast delayed rectifier

Preparation and instrumentation:

- Nucleated patches from the same juvenile-rat SNr cell class, recorded at `30 degC`.
- Signals were sampled at `20 kHz` and filtered at `10 kHz`; junction potentials of roughly `3.6-4.2 mV`, depending on solution, were not corrected.
- Calcium was replaced by magnesium, sodium current was blocked with `1 uM` TTX, and the Kv3-like `I_DR-fast` component was obtained by subtracting current recorded with `1 mM` external TEA from control. Recovery after washout was required as a subtraction-quality check.

Measured current-level equations and values:

```text
G_K(V) = I_K(V) / (V - E_K), with E_K estimated near -100 mV

f(V) = 1 / (1 + exp(+(V - Vhalf)/K))   [decreasing/inactivation]
f(V) = 1 / (1 + exp(-(V - Vhalf)/K))   [increasing/activation]
```

For pharmacologically isolated `I_DR-fast`, the normalized activation curve had `Vhalf = -8.5 +/- 1.6 mV`, slope `8.9 +/- 0.6 mV` (`n=12`), and current `20-80%` rise time at `+40 mV` of `0.41 +/- 0.03 ms`. The inactivation curve had `Vhalf = -49.2 +/- 1.6 mV` and slope `8.7 +/- 0.6 mV` (`n=5`). These are current/conductance summaries, not measurements of an `n^4 q` state decomposition.

The isolated-current activation used the protocol described for Figure 5: hold `-100 mV`, `10 mV` command increments, and `100 ms` test pulses in the repository's filed reconstruction. The locally extracted Figure 5 caption states the hold and increment but does not spell out the activation endpoints or pulse duration. The filed `-80` through `+50 mV`, `100 ms` ladder therefore contains figure-read/reconstruction details beyond the prose-extracted text. Steady-state inactivation is textually specified as hold `-90 mV`, `10 s` prepulses from `-110` through `0 mV` in `10 mV` increments, followed by `100 ms` at `+50 mV`.

Kv3-like deactivation is the best specified kinetic protocol in the local record: hold `-90 mV`, step to `+20 mV` for `100 ms`, then step separately to `-30`, `-40`, `-50`, `-60`, or `-70 mV`. The TEA-sensitive tail-current decay was fitted with one exponential. SNr time constants were:

| Command | Measured current-tail tau |
|---:|---:|
| `-60 mV` | `0.82 +/- 0.06 ms` |
| `-50 mV` | `1.35 +/- 0.12 ms` |
| `-40 mV` | `1.87 +/- 0.16 ms` |

The current-level empirical fit has the same generic exponential form shown above. Ding et al. do not provide a voltage-dependent `alpha_n(V)`, `beta_n(V)`, `tau_n(V)`, state power, or Markov transition scheme. They also do not establish whether the TEA-sensitive current is one heteromeric Kv3.1/Kv3.4 population or a sum of homomeric populations; the paper explicitly leaves subunit composition unresolved.

## Equations inferred from current

These transformations follow algebraically only after choosing a model family; they were not measured by Ding et al.:

```text
If I_Na = gbar_Na * m^3 * h * (V - E_Na),
then preserving the measured equilibrium activation curve A_Na requires
m_inf(V) = A_Na(V)^(1/3), provided h is treated separately.

If I_Kv3 = gbar_Kv3 * n^4 * q * (V - E_K),
then preserving the measured equilibrium activation curve A_Kv3 requires
n_inf(V) = A_Kv3(V)^(1/4), provided q is treated separately.
```

For a powered gate relaxing to zero while every other factor is fixed, `I proportional to x^p` gives `tau_gate = p * tau_current`. For a powered gate rising from zero with constant availability, a current crossing interval can likewise be converted to a gate tau analytically. Neither condition holds exactly in the sodium activation assay because activation and inactivation evolve concurrently. The Kv3 tail may also include nonzero equilibrium activation, inactivation, more than one channel population, and subtraction effects. Thus these conversions are conditional current-to-gate inferences, not exact state identification.

## Model priors used by the project

The executed V2 clamp added choices that the Ding papers do not supply:

- `m^3` fast-sodium and `n^4 q` Kv3-like current forms;
- cube- and fourth-root steady-state activation gates;
- two first-order sodium availability particles mixed by the measured recovery fraction;
- current-to-powered-gate time conversion at selected voltage anchors;
- log-linear interpolation between kinetic anchors and nearest-anchor clamping outside them;
- a constant Kv3 inactivation tau;
- ideal, unfiltered simulated current traces;
- the project-defined sodium recovery-duration ladder;
- no species, age, or temperature correction from juvenile rat at about `30 degC` to the intended adult-mouse context.

Each is a legitimate testable prior, but none should be described as a Ding equation or measured parameter.

## Prior related attempts and failures

### Classic Hodgkin-Huxley Stage B packet

The first executable Stage B family reused the simulator's classic squid-axon-style Hodgkin-Huxley rates:

```text
I_Na = g_Na * m^3 * h * (V - E_Na)
I_K  = g_K  * n^4     * (V - E_K)
dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
```

It searched sodium and potassium conductance scales while retaining those fixed rate equations. An initial 512-point GPU screen produced two engineering survivors, but complete authority remained unavailable; a fresh 512-point V3 screen produced zero engineering passes. The subsequent controlled failure diagnostic showed that the single-compartment packet depended completely on NaP for sustained firing and almost never recovered sustained firing after restored fast-sodium availability. The project retired that family as structurally inadequate. This does not isolate classic HH rates as the sole cause, but it shows that further conductance tuning around those non-SNr equations did not solve the Stage B mechanism.

### Structural successor V1

V1 assigned the measured normalized-conductance Boltzmann functions directly to `m_inf` and `n_inf`, then raised them to powers three and four in the currents. This algebraically moved the effective current activation midpoints to about `-21.85 mV` for sodium and `+6.32 mV` for Kv3. It was rejected before execution. This was a specification failure, not experimental evidence.

### Structural successor V2

<!--derived-->

V2 corrected the equilibrium error with activation-gate roots, then converted current timing endpoints into gate anchors and interpolated them. Under the exact Ding-inspired clamps it passed 11 of 18 source gates and failed seven:

| Endpoint | V2 result | Ding target, mean +/- 2 SEM |
|---|---:|---:|
| Na activation midpoint | `-21.559 mV` | `-31.4` to `-29.0 mV` |
| Na activation slope | `8.000 mV` | `5.8` to `6.6 mV` |
| Na rise at `0 mV` | `0.0377 ms` | `0.075` to `0.095 ms` |
| Na deactivation at `-40 mV` | `0.1919 ms` | `0.0812` to `0.1168 ms` |
| Kv3 deactivation at `-60 mV` | `1.1896 ms` | `0.70` to `0.94 ms` |
| Kv3 deactivation at `-50 mV` | `2.2158 ms` | `1.11` to `1.59 ms` |
| Kv3 deactivation at `-40 mV` | `3.5778 ms` | `1.55` to `2.19 ms` |

This is direct evidence that preserving equilibrium conductance curves and converting isolated current timing summaries into first-order powered-gate taus is insufficient for the filed state family. Conductance rescaling cannot repair these normalized kinetic failures.

## Unresolved questions

1. What state topology reproduces sodium activation, inactivation, recovery, deactivation, cumulative inactivation, persistent current, and resurgent current with one parameter set? Ding reports all of these current modes but does not identify their transition scheme.
2. Can the original sodium and Kv3 figure traces be recovered locally as numerical data, or must curves be digitized prospectively with an explicit error model? The current local archive contains HTML and remote image locators, not local numeric trace data.
3. Which full voltage-dependent tau or rate laws are appropriate for juvenile-rat SNr NaV1.1/NaV1.6 with beta1/beta4 and Kv3.1/Kv3.4 mixtures? Ding supplies only sparse current-level anchors.
4. Is the sodium activation midpoint uncertainty `0.6 mV` from Results or `0.2 mV` from the locally extracted Table 3 authoritative? The present Stage B bound uses `0.6 mV` and should record the inconsistency.
5. How much of the fastest sodium timing is recording-filter limited? The authors explicitly warn of underestimation of activation and deactivation speed, and the separate `25 degC`, higher-bandwidth measurements change the kinetic context.
6. What adult-mouse and target-temperature corrections apply? Neither paper reports an SNr-specific Q10 or adult-mouse kinetics.
7. Does Kv3-like current require one heteromeric state model, parallel Kv3.1/Kv3.4 populations, or another mixture? Pharmacology and mRNA establish a Kv3-like functional component but not molecular identity or stoichiometry.
8. What is the exact figure-derived activation command duration and endpoint ladder for each isolated current? These details should be reverified from a locally archived figure before the next preregistration rather than inherited as unmarked prose facts.

## Bounded next-source requirement

The next local-source search should seek a current-level kinetic model for the implicated NaV and Kv3 channel compositions that publishes explicit transition equations and enough rate parameters to replay the complete Ding protocols. Ding should remain the SNr assay authority, but not be stretched into a state-equation authority it is not. Any candidate model must be tested first against the full current waveforms and all reported endpoints, with its species, temperature, expression system, state topology, and unmeasured priors recorded separately.
