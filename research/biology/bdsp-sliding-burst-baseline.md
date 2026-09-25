---
type: biology
id: bdsp-sliding-burst-baseline
mechanism: The burst-probability baseline Pbar of burst-dependent synaptic plasticity (BDSP) is a SLOW per-neuron moving average of the proportion of the neuron's events that are bursts (time scale ~1-10 s), not a constant and not a fast (~20 ms) trace. It is the BDSP form of the BCM sliding modification threshold, and the source introduces it to keep synaptic weight growth finite. In the engine it is `cfg.bdsp_pbar_ratio_tau_ms` (Pbar = EMA(B_post) / EMA(E)), which replaces the preset constant p0 that let a rectified mean of the burst probability drive every active hidden synapse to the hard +w_max clamp.
status: de-risking
last_verified: 2026-09-25
current_finding: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md
current_status: "PRE-REGISTERED (AMENDMENT 6, 2026-09-25), not yet scored. Built as the companion process of the +-12 BDSP clamp that the 2026-09-24 bound census found load-bearing (one-sided: ff_0/ff_1 synapses pinned at +w_max, none at w_min, in all three hidden-learning arms). Engine unit test: a zero-mean apical credit gives the preset baseline a net-LTP drive of 54.6 (summed E*(P-Pbar) over hidden neurons) and the ratio baseline (tau 200 ms) 3.2. The dev-seed-7 C21 check (4 arms x 3 replicates) is queued; its criteria are in AMENDMENT 6."
sources:
  - path: "doi:10.1101/2020.03.30.015511 (Payeur, Guerguiev, Zenke, Richards & Naud, bioRxiv v1 2020; published Nat Neurosci 24:1010, 2021, doi:10.1038/s41593-021-00857-x)"
    anchor: "To ensure a finite growth of synaptic weights, we set this to a moving average of the proportion of events that are bursts in postsynaptic neuron i"
    note: "EXTERNAL, verified 2026-09-25 in the bioRxiv v1 full text (downloaded HTML, whitespace-normalised). The same sentence continues 'with a slow (~ 1 – 10 s) time scale (see Methods)'. This is the rule's own source naming the moving-average baseline as the process that bounds weight growth -- the role the runner gave a hard clamp."
  - path: "doi:10.1101/2020.03.30.015511 (Payeur et al., Methods, 'Event and burst detection')"
    anchor: "kept track of its time-averaged burst probability by using exponential moving averages of its event train"
    note: "EXTERNAL, verified in the same text: the baseline is the ratio of two exponential moving averages (burst train over event train), 'where tau avg is a slow time constant (~ 1-10 s)'. Values used: tau_avg = 5 s in the XOR learning task (Fig. 4), 15 s in the pairing protocols (Fig. 2b-d). The same Methods add event-rate homeostatic terms (H_i, G_i, active when the running event rate leaves [e_min, e_max], typically 2-10 Hz) that 'help to restrict the activity of neurons to an appropriate range' -- NOT implemented here (see companion_processes)."
  - path: "PMC6564292 (Bienenstock, Cooper & Munro 1982, J Neurosci 2:32, doi:10.1523/JNEUROSCI.02-01-00032.1982)"
    anchor: "depends not only on instantaneous pre- and postsynaptic activities but also on a slowly varying time-averaged value of the postsynaptic activity"
    note: "EXTERNAL, verified in the PubMed abstract. The BCM sliding threshold. Payeur et al. relate their baseline to it explicitly: 'the Bienenstock-Cooper Munro (BCM) model [11] has posited that the switching point between LTD and LTP depends on a nonlinear moving average of past activity.'"
  - path: "PMC6773092 (van Rossum, Bi & Turrigiano 2000, J Neurosci 20:8812, doi:10.1523/JNEUROSCI.20-23-08812.2000)"
    anchor: "strong synapses undergo relatively less potentiation than weak synapses, whereas depression is independent of synaptic strength"
    note: "EXTERNAL, verified in the PubMed abstract. The weight-dependent (soft-bound) alternative, weighed and NOT chosen as the primary lever: it makes the bound an asymptote instead of a wall but does not remove a drive that is one-signed on every synapse of a neuron, so under the measured one-sided drift weights would still pile up just below w_max (the hollow pass the census's near-bound fraction now guards against)."
  - path: "doi:10.1038/nature01530 (Royer & Pare 2003, Nature 422:518)"
    anchor: "little change in total synaptic weight occurs, even though the relative strength of inputs is modified"
    note: "EXTERNAL, verified in the PubMed abstract. Heterosynaptic conservation of total weight, the other alternative weighed and NOT chosen: it is a second process added beside BDSP, whereas the sliding baseline is part of the BDSP rule as published."
  - path: sim/bridge.py
    anchor: "RATIO baseline (Payeur et al. 2020/2021"
    note: "LOCAL. The engine implementation (additive, default-off guarded block in the BDSP step), pinned by tests/test_bdsp_pbar_ratio.py (OFF byte-identical to the pre-edit engine; ON equals the ratio of EMAs and respects the mask; a zero-mean apical integrates to a net-LTP drive under the preset baseline and to a small fraction of it under the ratio baseline)."
constants:
  tau_avg_s_xor_task: 5.0
  tau_avg_s_pairing: 15.0
  tau_avg_s_range: "1-10"
operating_point:
  # Not a machine-checked protocol rule: the flag defaults to 0.0 (OFF, byte-identical), and the checker compares the
  # runner DEFAULT, so a 'gte 1000' check would fire on the required default. It is a requirement WHEN ON.
  - key: pbar_ratio_tau_ms
    requires: ">= 1000 (1 s) when > 0; the source range is ~1-10 s (5 s in the XOR task)"
    why: "The baseline must be SLOW relative to one stimulus presentation, or it averages each teaching transient away inside its own credit phase. The legacy EMA of P at alpha 0.05/step (tau ~20 ms) did exactly that (AMENDMENT 3; frozen-readout training accuracy fell BELOW chance, 0.115-0.142, at C12/C18). One presentation here is 65 ms."
companion_processes:
  - process: "event-rate homeostasis (Payeur's H_i / G_i terms: plasticity pushes a neuron's running event rate back into [e_min, e_max])"
    status: proxied
    proxied_by: "tonic background currents fixed per layer (tonic_h_pA / tonic_o_pA) and the engine's threshold homeostasis; no plasticity-side rate homeostasis"
    why_it_matters: "in the 2026-09-24 census the hidden-learning arms end with the OUTPUT layer nearly silent (held-out mean read 0.001-0.09 vs 0.17-0.19 in the frozen arm), and the credit's phi = E(1-E) factor then vanishes. If the sliding baseline removes the drift but the output still falls silent, this is the next companion (the diagnosis finding's rung 2)."
  - process: "plasticity gated to the teaching period (Payeur's prefactor M: 'in the XOR task ... M = 1 when the teaching signal is present and 0 otherwise')"
    status: not_implemented
    proxied_by: "BDSP runs on every step, including the forward settle (as in every C0-C24 config)"
    why_it_matters: "with a slow baseline the settle steps carry a small BCM-like depression (P at rest below the running mean) on the current example's co-activity. Kept as in C21 so AMENDMENT 6 changes only the baseline."
implemented_by:
  - sim/bridge.py
  - research/runners/_gap4_transport_ceiling_readout_derisk.py
findings:
  - research/findings/2026-09-24-gap4-transport-ceiling-bound-census-clamp-load-bearing-fullsize-UNDEFINED.md
  - research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md
---

# The BDSP baseline is a slow moving average, and that is what keeps the weights finite

**What the rule's own source says.** Payeur et al. write the BDSP update as presynaptic trace x (burst -
baseline x event) and set the baseline to "a moving average of the proportion of events that are bursts in
postsynaptic neuron i, with a slow (~ 1 – 10 s) time scale", giving the reason in the same sentence: "To ensure a
finite growth of synaptic weights". It is the BDSP form of the BCM sliding threshold, which the paper names.

**What the runner had instead.** Two constants. The C21 dev config presets the baseline to p0 = 0.3 for every
neuron (`--pbar-alpha 0`), because the engine's only moving form was an EMA of the instantaneous P at 0.05 per
step (~20 ms), which averaged each teaching transient away (AMENDMENT 3). The engine also clips every BDSP weight
to +-12. The 2026-09-24 census found the clip load-bearing: in the transport-ceiling arm 18-80% of the two hidden
pathways' synapses end at +w_max, none at w_min, and the other two hidden-learning arms saturate the same way.

**Why a constant baseline drives weights to the clamp.** P = sigmoid(beta*scale*(v_apical - E_rest) + logit(p0)).
With p0 = 0.3 the operating point sits on the convex part of the sigmoid, so an apical credit that is zero on
average still raises the mean of P above p0 (Jensen's inequality). With a fixed baseline at p0, that excess is a
net potentiation drive on every active synapse onto the neuron, whatever the sign of its credit. The engine unit
test measures it: under a zero-mean apical (blocks of +60 / -60 / 0) the hidden neurons' summed E*(P - Pbar) is
+54.6 with the preset baseline and +3.2 with the ratio baseline (tau 200 ms). With the ratio of moving averages the
baseline is the event-weighted mean of P, so the time-integral of the drive over about tau is zero and only the
input-specific part of the credit remains.

**Declared shortcuts.** The ratio EMAs run inside the engine step (per-neuron state, like the existing EMA). The
runner chooses which neurons use it (the mask). The credit projection, the error, and the argmax read-out remain
host computations, as declared by the parent pre-registration. The +-12 clip stays in the kernel as a backstop, and
the census measures whether it still binds. Functional read-outs only.
