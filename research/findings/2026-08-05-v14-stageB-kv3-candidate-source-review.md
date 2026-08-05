---
type: source-review
status: complete
date: 2026-08-05
scope: v14-stageB-kv3-only
packet: research/packets/v14-stageB-fast-channel-online-research-v1.json
claims_reviewed:
  - kv3-linkevicius-four-protocol-models
  - kv3-labro-four-state-relaxation
  - kv3-classical-full-gate-equations
  - kv3-desai-validation-protocols
  - fast-channel-candidate-ranking.kv3
acceptance_action: none
---

# Stage B Kv3 candidate source review

## Decision summary

Two direct claims can be accepted exactly as written after independent source
review:

- `kv3-linkevicius-four-protocol-models`
- `kv3-desai-validation-protocols`

The Labro claim is substantively correct, but its rate equation needs an
implementation note before acceptance: the exponent must be evaluated as
`z * e * (V - Vhalf) / (k_B * T)`, equivalently
`z * (V - Vhalf) / V_T` with volts used consistently. The paper prints the
customary `z(V-Vhalf)/kT` shorthand. Copying that string literally with
millivolts would be dimensionally wrong.

The classical Kv3.3 claim cannot be accepted as written. At immutable commit
`26dc63b6cd5de79536731f072cf6c4d28328bb00`, the fixed baseline calculates a
Q10 factor but never applies it. Its stated condition, "Q10=3 referenced to
22 C," is therefore false for the exact fixed model cited by the packet. The
equations listed in the claim otherwise match the code.

The Kv3 portion of `fast-channel-candidate-ranking` needs correction. Labro's
four-state graph is an executable activation/deactivation candidate, but the
packet does not supply source-defined transitions for the proposed added
inactivation/recovery states. Desai et al. do publish a compact Kv3.3 current
form and rate constants that the ranking omitted. Neither source is a direct
SNr state-equation authority.

## Claim review

<!--derived-->

### `kv3-linkevicius-four-protocol-models`: accept exactly as written

The paper and repository support the claim:

- Paper Table 2 gives exactly these retained sweep counts:

  | Channel | Activation | Deactivation | Inactivation | Recovery |
  |---|---:|---:|---:|---:|
  | Kv3.1 | 2034 | 1286 | 949 | 674 |
  | Kv3.3 | 4104 | 1993 | 1180 | 810 |
  | Kv3.4 | 2430 | 527 | 364 | 100 |

- Paper Methods, equations 8-17, define two independent first-order gates and
  the normalized current. The implementation at
  `src/models/sciml_models.jl:1-59` is:

  ```text
  dm1/dt = (m1_inf(V,T) - m1) * r1(V,T)
  dm2/dt = (m2_inf(V,T) - m2) * r2(V,T)
  E_K(T) = (R * (T + 273.15) / F) * ln(4/151) * 1000
  I_norm = m1 * m2 * (V - E_K) / (80 - E_K)
  ```

  `m_i_inf` uses a sigmoid output and each positive relaxation rate `r_i`
  uses a softplus output. The paper writes the equivalent dynamics as
  `(m_i_inf-m_i)/tau_i`; the committed code's object named `m_i_tau` is used
  multiplicatively and is therefore an inverse time constant in execution.

- `src/data_wrangling/protocol_setups.jl:1-204` defines all four clamp
  families. Root `process_data.jl:59-62` maps Kv3.1, Kv3.2, Kv3.3, and Kv3.4.
  Root `process_data.jl:75-95` shows the four-protocol selection and the
  60/20/20 train/validation/test split. The selected individual artifacts are
  identified in `training_analysis.jl:282-294`: Kv3.1 iteration 270 seed 2,
  Kv3.3 iteration 300 seed 3, and Kv3.4 iteration 300 seed 1. Those serialized
  artifacts exist under `Models_split` at the cited commit.

- Paper Table 3 is explicitly a statistical comparison on the test data set.

This remains an oracle/initialization source. The neural functions are fitted
phenomenology, the data are homomeric channels expressed in CHO cells, and the
serialized Julia/Pumas objects are not a suitable biological end-state or a
lightweight production dependency.

Exact locators: Linkevicius et al. 2026, Methods "Voltage-clamp protocols,"
"Kv channel models," and "Model fitting," equations 8-17, Tables 2-3;
repository commit `26dc63b6cd5de79536731f072cf6c4d28328bb00`, files and lines
listed above.

### `kv3-labro-four-state-relaxation`: correction note required

The topology, reference parameter values, preparation, and limitations are
supported. Figure 7b defines this sequential graph:

```text
Resting (closed) <-> Pre-active (closed) <-> Relaxed-pre-active (closed)
                                      <-> Relaxed-active (open)
transition names:       p                       l                 s
forward rates:          alpha_p                 alpha_l           alpha_s
reverse rates:          beta_p                  beta_l            beta_s
```

The Methods define, for transition `j in {p,l,s}`:

```text
alpha_j(V) = alpha_j0 * exp(+z_j * e * (V - 6.2 mV)/(k_B*T))
beta_j(V)  = beta_j0  * exp(-z_j * e * (V - 6.2 mV)/(k_B*T))
I_K(V,t) is proportional to P_open(t) * (V - (-58 mV))
```

The equivalent voltage-domain implementation is
`exp(+/- z_j*(V-6.2 mV)/V_T)` with `V_T=k_B*T/e` expressed in mV. The packet's
printed shorthand matches the paper, but this unit-explicit form is required
for an executable transfer.

Supplementary Table 1 gives the reference black/pink/green trace parameters,
all rates in `ms^-1`:

| Transition | alpha0 | beta0 | z |
|---|---:|---:|---:|
| p | 0.05 | 0.15 | 3.5 |
| l | 6.0 | 0.6 | 0.4 |
| s | 1.0 | 0.8 | 0.001 |

For Figure 7e only, `beta_l` is changed from `0.6` to `1.8 ms^-1` for the red
trace and to `0.4 ms^-1` for the blue trace. The reference candidate is the
`0.6 ms^-1` value. Figure 7d shows the reference graph reproducing the hooked
tail after a 0.8 ms depolarization and losing it after longer pulses; Figure
7e identifies reverse transition `beta_l` as the tested control of that tail.

One source-level ambiguity must remain recorded. The Methods call the table
entries rates "at 0 mV," but the printed rate law centers the exponent at
`Vhalf=6.2 mV`, where each multiplier equals one. Implement the printed law
and table prospectively; do not silently reinterpret the constants to remove
this internal wording mismatch.

Exact locators: Labro et al. 2015, Figure 7b-e and caption; Methods,
"Simulations"; Supplementary Information, Supplementary Table 1.

### `kv3-classical-full-gate-equations`: do not accept as written

The packet transcribes the fixed equations correctly from
`src/models/baseline_models.jl:649-707`:

```text
am = 7.344 / (1 + exp(-0.0807*(V - 61.6)))
bm = 0.611 / (1 + exp(+0.08625*(V + 33)))
m_inf = am/(am+bm)
tau_m = 1/(am+bm)

h_inf = 0.1 + 0.9/(1 + exp((V + 29.7)/25))
ah = 0.0066/(1 + exp((V - 10)/-6))
bh = 0.01/(1 + exp((V + 20)/8))
tau_h = 1/(ah+bh)

P_open = m^4*h
```

Required correction: remove the fixed-model condition "Q10=3 referenced to
22 C." Lines 666-668 calculate `q10 = 3^((T+eta-22)/10)`, but lines 681-698
do not use it in either rate or time constant. The fixed executable baseline
is temperature invariant apart from the temperature-dependent reversal
potential and normalization.

The random-effects variant at lines 708-780 does divide both time constants
by Q10, but it also computes `m_inf = am*tau_m` after that division. Away from
Q10=1 this incorrectly makes the activation steady state temperature
dependent by a factor `1/Q10`. It should not be used as evidence that the
fixed condition is implemented correctly.

The upstream ModelDB source is now identifiable: repository 231818 commit
`05d3f755e85efd7cffdd31020ec80c82d33c3f63`,
`lib_mech/Kv33.mod:19-74`. It attributes the mechanism to Beining et al.'s
dentate granule-cell model and contains the same fixed `m^4*h` equations with
no Q10. It is a model prior, not a clamp-fitted Kv3.3 state-equation source.

### `kv3-desai-validation-protocols`: accept exactly as written

The CHO-cell experiments and Figure 3 support every stated protocol and the
`131 ms` recovery half-life:

- activation: hold `-70 mV`, 800 ms commands through `+70 mV` in 10 mV steps;
- steady-state inactivation: hold `-75 mV`, 730 ms prepulses from `-100` to
  `+50 mV`, then 250 ms at `+20 mV`;
- recovery: 2000 ms at `+20 mV`, recover at `-100 mV`, then 350 ms at
  `+40 mV`; reported recovery half-life `131 ms`;
- deactivation: hold `-70 mV`, 5 ms at `+50 mV`, then 10 ms commands from
  `-110` through `+40 mV` in 10 mV steps.

Exact locators: Desai et al. 2008, Experimental Procedures,
"Electrophysiological Analysis"; Results, "Biophysical Properties of
Recombinant Kv3.3 Channels in CHO Cells"; Figure 3A-E and legend.

The claim's limitation is also accurate: the paper does not provide a
downloadable channel implementation shown to reproduce all four clamp
families. It does, however, publish more executable structure than the packet
records. Its numerical-simulation section gives:

```text
I_Kv3.3 = g_Kv3.3 * n^3 * (0.23 + 0.77*p) * (V - E_K)  [control]
dx/dt = alpha_x(V)*(1-x) - beta_x(V)*x
alpha_x(V) = k_alpha_x * exp(eta_alpha_x*V)
beta_x(V)  = k_beta_x  * exp(eta_beta_x*V)

k_alpha_n = 0.039 ms^-1       eta_alpha_n =  0.0467 mV^-1
k_beta_n  = 0.0868 ms^-1      eta_beta_n  =  0.0067 mV^-1
k_alpha_p = 0.000045 ms^-1    eta_alpha_p = -0.18925 mV^-1
k_beta_p  = 0.00246 ms^-1     eta_beta_p  =  0.01075 mV^-1
```

The exponential rate convention is the one cited by Desai for the
Hodgkin-Huxley-like variables and is printed explicitly in later Kaczmarek
laboratory model descriptions. This model was used for neuron simulations;
the paper does not demonstrate waveform-level replay of Figure 3A-E with it.

### `fast-channel-candidate-ranking.kv3`: corrections required

Keep the first and third recommendations with tighter boundaries:

- Linkevicius individual fits are waveform oracles and search initializers,
  not biological mechanisms or SNr transfers.
- The compact classical `m^4*h` model is a negative-control comparator, with
  no effective Q10 in its cited fixed implementation.

Replace the second recommendation. "Add explicit inactivation/recovery states
constrained by Desai and Channelpedia" does not define transitions or
parameters and is not yet executable without invention. The source-backed
candidate set is instead:

1. unmodified Labro four-state graph for activation/deactivation structural
   replay;
2. unmodified Desai `n^3*(0.23+0.77p)` model for a compact full-gate Kv3.3
   comparator;
3. Linkevicius fitted Kv3.1/Kv3.3/Kv3.4 models as waveform oracles only.

## Smallest defensible Stage B candidate

<!--derived-->

The smallest biologically defensible **state-family candidate for the failed
Stage B deactivation gates** is the unmodified Labro four-state reference
model. It has four occupancies, six source-defined transition rates, one open
state, and a mechanistic relaxed-pre-active path that specifically changes
deactivation after depolarization. Run it first without calibration under the
sealed Ding activation, rise-time, and deactivation commands. It is a
prospective cross-preparation transfer from human Kv3.1b in oocytes, not an
SNr model.

It is **not** a complete candidate for promotion through every sealed Kv3
gate because it has no inactivation state. The smallest source-backed full
assay comparator is Desai's two-gate Kv3.3 model, but it is an empirical HH
model and its published neuron simulation is not a replay of all four Desai
clamps. Run that definition unmodified as a second comparator; do not tune it
before observing the sealed result.

No located source justifies grafting Desai's availability gate onto Labro's
relaxation graph as an accepted biological mechanism. If neither unmodified
candidate satisfies the complete Ding waveform endpoints, the next bounded
step is to preregister a topology search over explicit inactivated branches
using the Linkevicius/Channelpedia traces as fitting data and Ding as the SNr
promotion authority. Such a fitted hybrid remains a scaffold until its state
transitions receive independent biological support.

## Source retrieval record

<!--derived-->

The review used immutable or content-hashed source copies in `/tmp`; none was
added to the catalog:

- SciMLHHModels.jl commit
  `26dc63b6cd5de79536731f072cf6c4d28328bb00`.
- ModelDB 231818 commit
  `05d3f755e85efd7cffdd31020ec80c82d33c3f63`.
- Labro main PDF SHA-256
  `3ee88c2be52f1f1b2c329836ddd6dd9e2cfe2d445397d04b13e8232eb2bc5b60`.
- Labro supplementary PDF SHA-256
  `d0eb5e8d565d715588543120739fc4d82fff7629f1064b6e5d87d50f9c41a882`.
- Desai PMC HTML snapshot SHA-256
  `4c04b05905e8a976a3590b0d19302e8ec4c03841ff8760a67f07dc39e1da9878`.

No packet claim was accepted or edited by this review.
