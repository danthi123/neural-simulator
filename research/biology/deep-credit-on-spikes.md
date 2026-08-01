---
type: biology
id: deep-credit-on-spikes
mechanism: Deep credit assignment on a spiking substrate via a transport-free local rule (e-prop) plus population coding
status: established
last_verified: 2026-08-01
current_finding: research/findings/2026-08-01-gap4-6seed-bar-RUN-deep-credit-control-shuffleDFA-leaks-forward-learning-real-attribution-not.md
current_status: "FORWARD-LEARNING REAL, DEEP-CREDIT ATTRIBUTION NOT CLOSED AT 6 SEEDS (2026-08-01, corrects this morning's closure claim). The 6-seed bar — named by the prior current_status as the SOLE remaining residual — has now been RUN (noise-OFF, depth-2, K=8 and K=16, seeds 42/43/44/100/101/102), and the runner returns SIGNAL=False on 11/12 runs because the shuffle-DFA DEEP-CREDIT control LEAKS on 4/6 seeds at each K (mean shuffle-DFA 0.438 at K=8, 0.494 at K=16, vs the GO bar chance+0.10=0.433). What IS real and holds at 6 seeds: e-prop trains the forward task on the production Izhikevich bridge, the teacher signal is load-bearing (permuted control clean, mean 0.247 -> chance), and `inherit` climbs monotonically with the population factor K (0.685 at K=8 -> 0.852 at K=16). What is NOT established: that the climb is DEEP CREDIT rather than reservoir expressivity. TWO controls say it is the reservoir, the second decisive: (1) shuffle-DFA leaks (0.33-0.59, not chance 0.333) on 4/6 seeds at each K; (2) the runner's OWN frozen-hidden reservoir_control (reservoir_control_run=True on every run) reports deep_credit_share mean 0.066 at K=8 and 0.005 at K=16 — NEGATIVE on 3/6 seeds at each K. At K=16 e-prop 0.852 vs a FROZEN random hidden reservoir 0.852: training the hidden feedforward pathways adds NOTHING. RETRACTED: 'the K=8 residual is CLOSED / surpasses the reference ceiling' — it read the `eprop_inherit` field (0.85) and never read the `deep_credit_share` field (0.005) the SAME runner computed (silent-failure rules #1 + #7). The √K inherit curve (K=1 0.37 -> K=16 0.85) is real as a reservoir-CAPACITY curve. REAL RESIDUAL (a mechanism, NOT a control to build — the frozen-hidden control already exists and already ran): an operating-point/mechanism where hidden-layer credit actually contributes — the learned instructive signal (arc B, replaces fixed-random DFA), the φ′-vanishing-credit fix (2026-07-24 root cause), or the representable-forward expander (2026-07-25 GO, never combined with the credit runner). Arc-A's shallow atom reached the same place independently."
sources:
  - path: "doi:10.1038/nrn1198 (Destexhe, Rudolph & Pare 2003, the high-conductance state of neocortical neurons in vivo)"
    anchor: "high-conductance state"
    note: "EXTERNAL — recorded for when the reference is added locally. The claim it supports: cortical neurons fire irregularly because they receive INDEPENDENT high-conductance background synaptic bombardment, and that independence is precisely what makes population averaging reduce noise by sqrt(K)."
  - path: research/findings/2026-07-14-deep-credit-population-coding-smoke-INCONCLUSIVE-read-snr-flat-across-K.md
    anchor: "the pooled neurons are CORRELATED because the drive is deterministic"
    note: "our own measurement of the failure mode: read-SNR corr(pooled E, soma_rate) FLAT across K (0.289 at K=1, 0.291 at K=8, 0.277 at K=16) because every neuron in a slice gets an identical constant tonic current and no OU/conductance noise is enabled."
# NO constraints_config DECLARED, DELIBERATELY. The obvious one -- "population coding requires
# enable_ou_process / enable_conductance_noise, because averaging correlated copies averages nothing" -- is
# exactly what the 06:55 finding measured (read-SNR FLAT 0.289 -> 0.291 -> 0.277 across K=1/8/16 under
# deterministic tonic drive). But NO artifact from the K=8 closure records either knob, and the runner
# DEFAULTS both to False. So the requirement is well-motivated and UNCONFIRMED on the runs that matter.
# Declaring it here would assert a config-biology binding the record cannot support -- the precise error
# this file exists to prevent. It goes in when a run records the knobs.
implemented_by:
  - research/runners/_onbridge_eprop_port_derisk.py
findings:
  - research/findings/2026-07-14-deep-credit-spiking-training-wall-research-gate-graded-credit-decisive.md
  - research/findings/2026-07-14-deep-credit-population-coding-smoke-INCONCLUSIVE-read-snr-flat-across-K.md
  - research/findings/2026-07-19-gap4-research-gate-my-7-runs-were-downstream-of-the-KNOWN-apical-decoupled-bug-read-the-record-first.md
---

# Deep credit on spikes — a strong closure LEAD via e-prop + population coding, with its mechanism unverified

**What is measured:** a transport-free biological local rule (e-prop) plus population coding is reported to
train deep compositional credit on the production spiking bridge, K=8 reaching the LIF ceiling at 3 seeds.

**What is NOT established, and matters:** *why* it works. The natural explanation — population coding is a
√K noise-averaging lever, so it needs decorrelated neurons — is well motivated and directly measured as the
FAILURE mode (below). But no artifact from the K=8 runs records `ou_noise` / `cond_noise`, and the runner
defaults both to `False`. So the mechanism is a hypothesis, not a finding, and the closure is a lead that
needs re-running with the knobs recorded.

## Why this entry exists — the record contradicted itself for seventeen days

Two findings dated **2026-07-14** appeared to reach opposite verdicts on the same lever, and neither declared
a `status:`, so nothing adjudicated them:

| | committed | verdict |
|---|---|---|
| population-coding smoke | 06:55 | "REFUTED — pooling ALREADY works but does NOT lift accuracy"; K=1 "cannot even FIT the training set" |
| training-wall research gate | 14:11 | "COMPLETE, POSITIVE CLOSURE" — K=8 reaches the LIF ceiling |

**They do not contradict, because THEY MEASURED DIFFERENT NETS.** This is the whole resolution, and it is
verifiable from source rather than inferred:

| | net class | independent noise | population result |
|---|---|---|---|
| 06:55 smoke | `OnBridgeBDSPNet` | off | read-SNR FLAT across K; pooling gives no √K benefit |
| 14:11 closure | `OnBridge**Eprop**Net` | off | K=1 0.47 → K=4 0.62 → K=8 0.877 |

The 06:55 document root-caused ITS failure by reading `OnBridgeBDSPNet.__init__`: every neuron in a slice
receives an identical constant tonic current (`tonic_h_pA`, `tonic_o_pA`) with no OU or conductance noise, so
the pooled neurons are redundant copies. It named the biology-grounded fix — independent high-conductance
background bombardment.

**That fix was never applied, and it was not what made K=8 work.** `ou_noise` / `cond_noise` are constructor
parameters defaulting to `False`, exposed on NO command line, and passed `True` by NO runner anywhere in the
tree. The e-prop net therefore closed the task on nominally-correlated neurons. So the correlated-pool
diagnosis is real and specific to the **BDSP** net; whatever makes population coding work for the **e-prop**
net is a different and still-unexplained route.

## The cost of not having this entry

On **2026-07-31** a nine-hour, eight-cell GPU crux ran `_gap4_onbridge_spiking_selfpredict_derisk` at
`pool_k=16` and reported that even the idealised transport ceiling could not fit its own training set. That
runner uses **`OnBridgeBDSPNet`** — the very net the 06:55 finding measured — with `tonic_h_pA=450.0` /
`tonic_o_pA=500.0` and no OU or conductance noise. That is precisely the configuration proved to yield no √K
benefit, so the population lever was nominally at 16 and functionally at 1.

So the crux was not measuring whether deep credit works on spikes. It was re-measuring the correlated-pool
failure mode, with the rule that closed the arc (e-prop) not even present in its `--arms`.

This is the third instance of the same class: **2026-07-19** records seven runs re-derived downstream of a
known root-caused bug, its title literally *"read the record first"*.

## What is genuinely open

Three things, in priority order:

1. **Re-run K=8 with `ou_noise` / `cond_noise` RECORDED.** Until then the closure's mechanism is a
   hypothesis and the result itself rests on a config nobody can reconstruct — the "a filename is not
   provenance" defect this project has already paid for once.
2. **Confirm at 6 seeds.** The closure is 3-seed against a 6-seed bar, and two banked eprop artifacts carry
   `PRE-SEEDFIX-CONFOUNDED` in their own names.
3. **Then** test whether K=16 exceeds K=8, which the √K trend predicts only if the pool is decorrelated.
