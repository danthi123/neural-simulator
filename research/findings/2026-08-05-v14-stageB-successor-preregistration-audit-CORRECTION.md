---
type: research-finding
status: complete
date: 2026-08-05
mechanism: v14-stageB-structural-successor-preregistration-audit
claim_check: correction
---

# Stage B structural-successor preregistration correction

## Decision

Do not execute the V1 structural-successor protocol. It binds measured
normalized-conductance Boltzmann curves directly to activation gates and then
raises those gates to powers three and four. That changes the curves the
papers measured, so V1 cannot simultaneously satisfy its equation and Stage 1
activation gates. A V2 protocol must supersede it before execution.

This was caught before a simulation result or calibration search existed. The
partial kernel implementation is retained only after changing its steady-state
gate equations to the corrected conductance-to-gate transformation.

## Source-grounded correction

Ding, Wei, and Zhou report the fast-Na normalized conductance activation curve
with midpoint `-30.2 +/- 0.6 mV` and positive slope `6.2 +/- 0.2 mV`. Ding,
Matta, and Zhou report the Kv3-like normalized conductance activation curve
with midpoint `-8.5 +/- 1.6 mV` and slope `8.9 +/- 0.6 mV`. These are
conductance curves, not independently observed Hodgkin-Huxley gate curves.

For the filed currents `m^3` and `n^4`, the narrowest algebraic bridge that
preserves the measured curves is:

```text
A_Na(V) = 1 / (1 + exp(-(V + 30.2) / 6.2))
m_inf(V) = A_Na(V)^(1/3)

A_Kv3(V) = 1 / (1 + exp(-(V + 8.5) / 8.9))
n_inf(V) = A_Kv3(V)^(1/4)
```

The roots are equation-derived model transformations, not measurements of
individual gate particles. Leaving V1 unchanged would move the effective
half-activation to approximately `-21.85 mV` for fast Na and `+6.32 mV` for
Kv3-like current.

The inactivation curves can remain direct availability steady states because
the current equation uses availability to the first power.

## Kinetic evidence boundary

The repository artifact carrying the quoted source values, units, uncertainty,
and transfer class is
`research/specs/v14_snr_stageB_structural_successor_v2.json`.

The primary sources identify only a small number of current-level kinetic
endpoints, not complete voltage-dependent gate time-constant functions:

- fast Na: 10-90 rise `0.085 +/- 0.005 ms` and decay
  `0.191 +/- 0.012 ms` at 0 mV; recovery at -120 mV has fast/slow taus
  `0.59 +/- 0.07 / 35.1 +/- 6.4 ms` and fast fraction
  `0.526 +/- 0.057`; deactivation at -40 mV is
  `0.099 +/- 0.0089 ms`;
- Kv3-like: 20-80 rise at +40 mV is `0.41 +/- 0.03 ms`; current-tail
  deactivation taus are `0.82 +/- 0.06`, `1.35 +/- 0.12`, and
  `1.87 +/- 0.16 ms` at -60, -50, and -40 mV.

Converting a powered first-order gate's current rise uses
`t_a-b = tau * log((1-a^(1/p))/(1-b^(1/p)))` only when other gates are fixed.
Converting a zero-equilibrium current-tail tau gives `tau_gate = p*tau_current`
under the same conditional approximation. These are equation-derived priors,
not source-measured gate kinetics. Log-linear interpolation between anchors
and endpoint clamping are also explicit model priors. No Kv3 inactivation tau
was measured.

The Na trace activates and inactivates concurrently, so its measured rise and
decay cannot be promoted into independent exact gate taus. Stage 1 must measure
the composite simulated current and accept a structural NO-GO if the filed
state family cannot reproduce both endpoints.

## Additional V1 omissions

- The recovery protocol says “filed duration ladder” but V1 contains no
  duration list. V2 must enumerate a project-operational ladder and must not
  call the list source-reported.
- The Na recordings were sampled at 50 kHz and filtered at 10 kHz; the authors
  explicitly warn that the fastest activation and deactivation values may be
  instrument-limited. An ideal simulated trace matching them is a transferred
  operational target, not proof of unfiltered channel equivalence.
- Both channel studies used juvenile-rat nucleated patches around 30 C. No
  adult-mouse kinetics or measured Q10 was located.
- The local automated extraction remains `pending_review` and carries
  `scientific_claim: false`; this correction relies on reading the primary
  full text, not on promoting the automated state.

## Primary locators

- Ding, Wei, and Zhou 2011, Methods and Figures 5-9, DOI
  `10.1152/jn.00305.2011`. <!--derived-->
- Ding, Matta, and Zhou 2011, Methods and Figures 5, 8, and 9, DOI
  `10.1152/jn.00707.2010`. <!--derived-->
- Local full text:
  `/home/dant123/.local/share/neural-simulator/scholarly-fulltext-v14-stageB/content/4d77ff156fc7d7f7bbe48c455d6a5cb05996c1d1b4acc4325a164fb8de369001.html`.
- Local full text:
  `/home/dant123/.local/share/neural-simulator/scholarly-fulltext-v14-stageB/content/bb08674efd0db8fcae5c578ac5e1ba3c07493a6e73bb9e3c87c048751b6067d1.html`.

## Consequence

V1 remains in history as a preregistration caught before execution. It earns no
result. V2 must bind this correction, preserve the measured conductance curves
through root-transformed activation gates, enumerate the recovery ladder, and
retain every interpolation and current-to-gate conversion as a model prior.
