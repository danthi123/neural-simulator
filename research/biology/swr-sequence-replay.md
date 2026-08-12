---
type: biology
id: swr-sequence-replay
mechanism: Sharp-wave/ripple replay -- the waking spike SEQUENCE is re-emitted in the same temporal ORDER, time-compressed into the ripple
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-25-gap5-ecker-nS-recurrent-model-SCAFFOLD-built-dt-fixed-recurrent-transmission-blocker.md
current_status: "ON-SUBSTRATE 6/6: a cue-triggered localized traveling bump on a Gaussian near-diagonal CA3 band of AdEx neurons decodes as a directional trajectory, weighted-corr DECODE_r = 1.000 on all 6 seeds, bump_width 0.8 with width_growth ~ 0. Band REQUIRED (no-band 0.000); forward ASYMMETRY required (symmetric band + middle cue 0.139, width 23 and growing). Adaptation and the PVBC pool are INERT in this regime."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "in the same temporal order during sharp waves of sleep"
    note: "the load-bearing claim: it is the ORDER that is replayed, and only AFTER the experience, not before"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "millisecond intervals at the troughs of the ripple"
    note: "the timescale replay is emitted at -- 5-6 ms between successive cells, at ripple troughs"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "twice as fast as the cell assembly sequences compressed"
    note: "SWR replay is ~2x faster than the same sequence compressed into a theta cycle -- replay is not a replay at behavioural speed"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "retain and replay the information embedded in the synaptic network"
    note: "the mechanism attribution: the sequence comes OUT OF THE WEIGHTS, which is why a learned band reproduces it"
constants:
  ripple_spike_interval_ms_low: 5
  ripple_spike_interval_ms_high: 6
  swr_over_theta_speed_factor: 2
constraints_config:
  - key: dt_ms
    value: 0.1
    why: "The Ecker CA3 pyramidal AdEx parameterization (DeltaT=4.23, V_T=-24.42) is STIFF. Measured 2026-07-25: at dt=0.5 ms the exponential term blows up, V sticks at +45.9 mV past V_peak=-3.25 and never resets, so there is no replay to measure -- the failure looks like a dead mechanism, not a dead integrator. Ecker uses dt=0.1 ms."
implemented_by:
  - research/runners/_gap5_ecker_recurrent_replay.py
findings:
  - research/findings/2026-07-25-gap5-ecker-nS-recurrent-model-SCAFFOLD-built-dt-fixed-recurrent-transmission-blocker.md
  - research/findings/2026-07-25-gap5-learned-band-emergence-STDP-directed-traversal-6seed-GO.md
  - research/findings/2026-07-24-gap5-moving-bump-replay-decode-encode-WIN-replay-BOUNDARY.md
---

# Replay is an ORDER, re-emitted about twice as fast as theta

**The claim the code must respect.** After a wheel-running experience, "[t]he same neurons repeatedly fired in the
same temporal order during sharp waves of sleep immediately following, but not preceding" it. The spikes land at
**5–6 ms intervals at the troughs of the ripple**, making the sharp-wave sequence "**twice as fast as the cell
assembly sequences compressed into a single theta period**." And the sequence is not stored anywhere separate:
sharp waves "retain and replay the information embedded in the synaptic network that gives rise to the event."

Three consequences for our code, all of which the gap#5 arc had to rediscover empirically:
1. **Order is the observable, not rate.** A read-out that scores which cells fired, but not in what order, is not
   measuring replay. (Bayesian weighted-correlation decode is the instrument the arc converged on.)
2. **The sequence lives in the recurrent weights.** So an asymmetric band produces directional replay and a
   symmetric one does not — measured here as DECODE_r 1.000 vs 0.139.
3. **Replay is compressed and fast**, which is why it is a *different dynamical regime* from the waking traversal
   rather than a slow rerun of it.

**Why this tag sits on the gap#5 row alongside `btsp-place-field-formation`.** That entry covers the ENCODE half
(a single plateau writes a place field). This entry covers the READ half the row is actually named for — the
ordered replay read-out. They are different claims, different papers, and different runners.

## Scope of the current status — read this before citing the 1.000

`DECODE_r = 1.000` is a *decode* correlation on a 2000-neuron place-field track, not a behavioural score, and the
Bayesian decoder is a measurement instrument sitting outside the brain. The remaining work named on the ledger row
(a neural reader; merge into the one-brain agent) is not done by this result. Two mechanism attributions were
**refuted by lesion** in the same run and should not be re-derived: the Ecker neg-`a`/large-`b` adaptation and the
PVBC inhibitory pool are both **inert** here — every adapt-lesion and no-PVBC arm still decodes 1.000. What is
load-bearing is the band plus AdEx refractoriness.

⚠️ **Provenance honesty.** Ecker et al. 2022 is **not in the local corpus** and was not read for this entry; the
Buzsáki anchors above are what `biology_check` can actually resolve, and they support the *biology* (ordered,
compressed, weight-borne replay), not the specific Gaussian-band model. The dt constraint is attributed to Ecker
by our own 2026-07-25 finding, and independently earned by the blow-up measured in that run. No separate raw JSON
aggregate is banked for the 6-seed replay run — the finding doc is the artifact of record.
