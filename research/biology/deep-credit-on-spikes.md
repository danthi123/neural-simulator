---
type: biology
id: deep-credit-on-spikes
mechanism: Deep credit assignment on a spiking substrate via a transport-free local rule (e-prop) plus population coding
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-14-deep-credit-spiking-training-wall-research-gate-graded-credit-decisive.md
current_status: "CLOSURE CLAIMED, MECHANISM UNVERIFIED. Transport-free e-prop plus population coding is reported to reach the LIF ceiling on-bridge (inherit 0.877 / train 0.963 at K=8 vs LIF 0.89; sqrt-K trend K=1 0.47 -> K=4 0.62 -> K=8 0.877), 3-seed against a 6-seed bar. WHAT CANNOT BE VERIFIED FROM THE RECORD: whether those runs had independent per-neuron background noise. The e-prop port runner DEFAULTS ou_noise=False and cond_noise=False, and NO artifact records either knob — so the causal link between decorrelation and the closure is UNCONFIRMED, and two banked eprop artifacts are additionally named PRE-SEEDFIX-CONFOUNDED. Treat the closure as a strong lead requiring re-run with the knobs recorded, not as a closed capability."
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

**They do not contradict. The earlier one is the DIAGNOSIS and the later one is the RESOLUTION.** The 06:55
document root-caused the failure by reading the substrate: every neuron in a slice receives an identical
constant tonic current (`tonic_h_pA`, `tonic_o_pA`) with no OU or conductance noise enabled, so the pooled
neurons are redundant copies and read-SNR is flat across K. It then named the biology-grounded fix —
independent high-conductance background bombardment. The 14:11 document then reports a working lever.

The tempting reading is that the fix was applied and that is why K=8 works. **That reading is not supported:**
the K=8 artifacts do not record either noise knob and the runner defaults both to False. What IS established
is the ordering — diagnosis at 06:55, working lever at 14:11 — and that ordering is invisible without
checking commit times, which is why reading either document alone misleads.

## The cost of not having this entry

On **2026-07-31** a nine-hour, eight-cell GPU crux ran `_gap4_onbridge_spiking_selfpredict_derisk` at
`pool_k=16` and reported that even the idealised transport ceiling could not fit its own training set. That
runner sets `tonic_h_pA=450.0` / `tonic_o_pA=500.0` and enables **no** OU or conductance noise — precisely
the configuration the 06:55 finding proved yields no √K benefit. The population lever was nominally at 16
and functionally at 1.

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
