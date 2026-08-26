---
type: biology
id: spiking-confidence-synaptic-integration
mechanism: A metacognitive confidence read-out is a SPIKING neuronal firing rate that INTEGRATES the first-order decision evidence THROUGH SYNAPSES — not a host dot product injected as a current. A presynaptic ACC/aPFC evidence-&-conflict population projects, via reward-gated synapses, onto an opponent aPFC confidence channel (V+ "was correct" / V- "was error"); confidence = rate(V+) - rate(V-).
status: established
last_verified: 2026-08-26
current_finding: research/findings/2026-08-18-self-organized-metacog-monitor-GO.md
current_status: "The plastic three-factor Hebbian monitor (`--confidence-read plastic_acc`) is a 6-seed mission_go=True GO: the confidence->correctness MAPPING WEIGHTS are learned by a local reward-gated three-factor Hebbian rule (no host optimizer, host_logistic_fit_calls==0). Its OWN named honest residual: the presynaptic ACC features are host RATE READS and the learned synaptic sum w.z is computed by a host np.dot and INJECTED as the meta subpool current -- NOT propagated through installed presynaptic synapses driven by ACC spikes. This binding grounds the next rung (`--confidence-read spiking_acc`): install the learned w_plus/w_minus as an ACC->meta projection and let the V+/V- drive EMERGE from ACC spikes through those synapses, so confidence is a genuine spiking population read of a synaptic integral. Dale's-law E/I split of signed weights is the residual after that."
sources:
  - path: "doi:10.1126/science.1169405 (Kiani & Shadlen 2009, Representation of Confidence Associated with a Decision by Neurons in the Parietal Cortex, Science 324:759-764)"
    anchor: "the same neurons that represent formation of a decision encode certainty about the decision"
    note: "EXTERNAL (WebSearch, 2026-08-26; local kandel/catalog corpus thin on metacognition). The load-bearing fact: decision CONFIDENCE is carried by the FIRING RATE of neurons that INTEGRATE the decision evidence via synaptic input -- confidence is a graded spiking population rate downstream of (and synaptically driven by) the evidence integrator, NOT a quantity computed off-substrate. This is the biological warrant for scoring the confidence read as rate(V+)-rate(V-) where V+/V- receive their drive from ACC spikes THROUGH installed synapses, rather than from a host dot product injected as current."
  - path: "doi:10.1037/rev0000045 (Fleming & Daw 2017, Self-evaluation of decision-making: a general Bayesian framework for metacognitive computation, Psychological Review 124:91-114)"
    anchor: "second-order computation over first-order decision variables"
    note: "EXTERNAL. Metacognition is a SECOND-ORDER read that takes the first-order decision variables as its INPUT -- i.e. the confidence stage receives the first-order competition as afferent input, motivating a presynaptic-population -> confidence-channel projection rather than an off-substrate transform."
  - path: research/findings/2026-08-18-self-organized-metacog-monitor-GO.md
    anchor: "fully-spiking presynaptic-population read is the remaining shortcut to burn down"
    note: "Our own committed GO names this exact residual as the next rung. This binding scopes and grounds it."
# NO constraints_config DECLARED, DELIBERATELY (following deep-credit-on-spikes.md). The mechanism does not REQUIRE
# a specific NUMERIC biophysical config the way Ecker requires dt_ms=0.1: `confidence_read=spiking_acc` is a MODE
# SELECTOR (not a biophysical constant), and `enable_parameter_heterogeneity=True` is already the shared runner
# default for every metacog mode. The graded-rate requirement (heterogeneity) is real biology but it is a property
# of the SUBSTRATE the runner already sets, not a config the biology pins to a value -- declaring it would assert an
# unenforceable config-biology binding, the exact error the gate exists to prevent.
implemented_by:
  - research/runners/_second_order_metacog_monitor_derisk.py
findings:
  - research/findings/2026-08-18-self-organized-metacog-monitor-GO.md
  - research/findings/2026-08-17-wave1-second-order-metacog-6-GO.md
  - research/findings/2026-08-02-laneC-metacog-margin-comparator-PARTIAL-real-signal-not-robust-next-is-symmetric-or-learned-error-monitor.md
---

# Metacognitive confidence as a spiking synaptic integral (not a host dot product)

**What the biology says.** Kiani & Shadlen (2009) recorded LIP neurons while monkeys reported decision
confidence via an opt-out. The *same* neurons that integrate the motion evidence to form the direction
decision also encode the *degree of certainty* in that decision, in their graded firing rate. Confidence is
therefore a **spiking population rate that is synaptically driven by the first-order evidence** — a
second-order read (Fleming & Daw 2017) whose *input* is the first-order decision variables, delivered as
afferent synaptic drive, and whose *output* is a graded rate.

**Why it grounds the `spiking_acc` advance.** The committed plastic three-factor Hebbian monitor
(`plastic_acc`, 6-seed `mission_go=True`) learns the confidence→correctness mapping *weights* by a local
reward-gated rule with no host optimizer — a real closure of the "weights are a host logistic fit" residual.
But it then computes the opponent synaptic sums `w_plus·z`, `w_minus·z` with a **host `np.dot`** and *injects*
them as the V+/V- meta-subpool currents. The synapse `w` is a host array, not an installed projection carrying
spikes. Under the BRAIN-BASED-ONLY standard that is a shortcut: the *brain* is not performing the synaptic
integration, the runner's bookkeeping is.

The `spiking_acc` mode closes that: the learned `w_plus`/`w_minus` are **installed as an ACC→meta
projection**, an ACC presynaptic population is **driven by the (standardized) first-order features encoded as
non-negative rate channels**, and the V+/V- drive **emerges from ACC spikes propagated through those
synapses**. Confidence = rate(V+) − rate(V-) is then a genuine spiking read of a synaptic integral, matching
Kiani-Shadlen. Anti-cheat: `host_dot_on_read_path == 0` — the V+/V- currents are set ONLY by ACC synaptic
input during the report window, never by `monitor.currents_from_features`.

**The residual after this rung.** Signed weights routed onto a single ACC channel violate Dale's law (a
presynaptic neuron cannot be excitatory to one target and inhibitory to another). The faithful E/I split
(signed weights → excitatory channels + an inhibitory interneuron relay) is the residual that follows
`spiking_acc`, exactly as the affect / deep-credit arcs stage their spiking-purity rungs.
