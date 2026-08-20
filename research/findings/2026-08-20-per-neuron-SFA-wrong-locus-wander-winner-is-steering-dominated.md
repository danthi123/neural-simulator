---
type: finding
status: live
date: 2026-08-20
mechanism: continuous-state-engine
lane: continuous-substrate
seeds: [42]
seed-waiver: A deterministic mechanism-locus diagnosis — is the per-neuron intrinsic-SFA current engaged, and if so why does it not shift the wander winner. The evidence is substrate-code tracing (which fields the step reads / _hard_silence zeros) plus the byte-identical two-arm sequences already banked; not a stochastic effect size, so a seed population measures nothing here.
instrument: substrate trace (sim/bridge.py current sum; _hard_silence; _steered_rest) + the s500/s800 cupy artifacts from 52e673fd
runner: research/runners/_continuous_wander_perneuron_sfa_derisk.py
external: NO-EXTERNAL-NEEDED — a locus diagnosis of an in-repo mechanism against in-repo substrate code; the corrected locus (short-term depression on the steering projection) is a standard synaptic mechanism, not a novel claim.
artifacts:
  - research/findings/raw/_continuous_live_cupy/wander_perneuron_sfa_s800_p0.7.json
---
# The per-neuron INTRINSIC SFA current is the WRONG fatigue locus — the wander winner is STEERING-dominated

Artifact: research/findings/raw/_continuous_live_cupy/wander_perneuron_sfa_s800_p0.7.json (byte-identical arms, from 52e673fd)

**One line.** The faithful burn-down of the wander-IOR (a per-neuron spike-frequency-adaptation CURRENT injected on
`cp_intrinsic_current_pA`) left the SFA arm byte-identical to baseline at production (both strengths, 52e673fd),
banked UNDEFINED pending an injection-engagement check. This RESOLVES it by tracing the substrate: the injection IS
engaged and read — so it is NOT a no-op — but a per-neuron intrinsic hyperpolarizing current is the WRONG locus to
break this wander, because the winner is set by the tonic STEERING drive, not by intrinsic excitability.

## The trace (why the injection is engaged yet the winner does not change)
- **Engaged + read.** `_hard_silence` (`_gap5_spontaneous_reactivation_derisk.py:208`) zeros `cp_external_input_current`
  and the conductances/recovery vars, but does NOT touch `cp_intrinsic_current_pA`; and `sim/bridge.py:7952`
  (`if self.cp_intrinsic_current_pA is not None: dynamics_current = dynamics_current + self.cp_intrinsic_current_pA`)
  adds it into every step's current. `sfa_state` becomes nonzero after wander 1 (normalized own-spike counts), so the
  injection fires on wanders 2+. So the "UNDEFINED — maybe not engaged" hypothesis is FALSIFIED: it is engaged.
- **Wrong locus.** The wander is a winner-take-all competition driven by `_steered_rest`, which sets
  `bridge.cp_external_input_current[:] = bias_dev` EVERY step (`:197`) — `bias_dev` is the CURIOSITY-GAIN steering
  (a large tonic neuromod drive), plus Poisson noise. The just-fired basin ('cat') wins because its STEERING bias is
  large; a per-neuron intrinsic hyperpolarization of a few hundred pA is a small counter-current against that bias, so
  it does not flip the winner (byte-identical winner sequence), even though it does perturb sub-winner firing.

## The corrected faithful locus (the real next mechanism)
The gain-level IOR (2026-08-20, wired live) fatigues the STEERING GAIN itself — and that is why it WORKS: the correct
fatigue target for a steering-dominated competition is the DRIVE, not intrinsic excitability. So the gain-level IOR is
NOT merely a host scaffold for the intrinsic-SFA form — it fatigues the mechanistically-correct locus (the neuromod
steering drive) at the population level. The truly-faithful per-neuron burn-down is therefore NOT an intrinsic-SFA
current; it is **short-term synaptic DEPRESSION on the curiosity→CA3 STEERING projection** to the just-fired basin
(fatigue the DRIVE per-synapse, mirroring the population gain-fatigue the IOR does). That is the named next mechanism;
the intrinsic-SFA-current arm is a NO-GO for this competition (correctly, now that engagement is confirmed).

## Status
Resolves the per-neuron-SFA UNDEFINED (52e673fd): engagement CONFIRMED (not a no-op), locus WRONG (intrinsic current
vs a steering-dominated winner), correct faithful locus NAMED (STD on the steering projection). The host-gain IOR
stays the working live form and is now understood to be the RIGHT fatigue locus, not a placeholder. No production
change; the intrinsic-SFA de-risk runner is retained with its interpretation updated.
