---
type: research-gate
status: complete
date: 2026-08-08
mechanism: pragmatics-neural-communicative-success-coincidence-read-back
lane: conversation (pragmatics wired to speaking)
---

# Wave-2 pragmatics research gate — a NEURAL communicative-success coincidence detector READ BACK into the speaking policy

## Decision (one line)

A genuinely-new mechanism exists and is **buildable now** by reuse-by-import (no `sim/` edit): replace Wave-1's
**intent-gated routing of the belief echo** (a host index into `L1[u][s*]` dressed as a neuron) with a real
**two-input coincidence detector** (threshold-AND / VIP-SST disinhibitory column) whose population rate IS the
overlap `P(listener infers s* | u)`, and **close the reverse coupling** — that coincidence rate drives the
group-scoped DA bus which **trains an intent to utterance selection assembly by three-factor reward-modulated
plasticity**, so the speaker's choice becomes a WTA over a **learned** assembly rather than a host `argmax` over the
imported RSA table. The honest residual (the one that can turn this into deeper research) is whether the naive
three-factor loop converges the policy across seeds without an action-conditioned value baseline — the exact wall the
vocal-credit arc hit (v1 NO-GO, yoked control).

## What Wave-1 actually did (grounded in its own code, not its abstract)

`research/runners/_pragmatics_perlocutionary_reward_derisk.py` (commit `135d96480`) passed 6/6 anti-cheats, but its
own source confirms the task's critique verbatim:

- **The success signal is a host index-multiply, not a coincidence.** Line 122 comment:
  *"the overlap is realized by intent-GATED routing of the belief echo, **not a threshold-AND**"*. The intent one-hot
  selects which belief dimension's echo current drives the `match` population (`BELIEF_GAIN * rate(L1[u][s*])`), so
  `match` **relays a single host-selected table entry** `L1[u][s*]`. One neural input carries information; the other
  (intent) is a routing switch. The neuron adds nothing decision-relevant.
- **The decision is a pure host `argmax` over the imported RSA table** (lines 32-33): the speaker "picks the utterance
  whose perlocutionary+rewarded success is highest (argmax over the small utterance set)". Delete every added neuron
  and the argmax is unchanged.
- **The coupling is one-directional.** Lines 58-59 / residual #1 of the GO: *"the reward bus is instantiated and
  shown contingent … the SHAPING is not yet learned-from-experience"* — the DA signal never updates a synapse that the
  choice reads. Cutting the reward would not change the utterance.

This is the same failure class Wave-1 already REFUTED for options B and D elsewhere (host-argmax proxy; host
index-multiply sold as neural). The measurements are real; the mechanism claim is the overclaim.

## The genuinely-new mechanism (different in kind, three legs)

### Leg 1 — success by NEURAL COINCIDENCE (two spiking inputs), not routing

A `success[k]` detector column per state `k`. It receives TWO independent spiking inputs:
- **belief input**: the listener's inferred-belief population fires with per-state rate `L1[u][k]` (the listener is
  the *social environment* — a real second `SimulationBridge` or the imported RSA listener; either is legitimate as
  the "other agent", but the belief arrives as SPIKES, not a current indexed by intent);
- **intent input**: the speaker's intent population fires on state `s*` (a one-hot goal code — the world/goal
  boundary, delivered as spikes).

The detector is a genuine AND. Two buildable realizations, cheapest first:
1. **Threshold-AND (minimal, point-neuron):** set weights so `belief[k]` alone ≈ 0.5·rheobase and `intent[k]` alone
   ≈ 0.5·rheobase; `success[k]` crosses threshold **only when both are co-active at the same k**. Threshold
   nonlinearity is intrinsic neuronal biophysics; this is a standard spiking coincidence detector. It is *exactly the
   "threshold-AND" Wave-1 explicitly refused to build*.
2. **VIP-SST disinhibitory column (more faithful):** `belief[k]` is normally shunted by an SST guard; `intent[k]`
   recruits a VIP interneuron that disinhibits **only column k**, so the belief echo passes iff the intent selects the
   same state. Pi et al. 2013 (VIP-|SST-|pyramidal) is the anchor; the project already has a validated normally-closed
   disinhibitory cascade (D1-|GPi releases thalamus) with the substrate's safe GABA-weight band (2-20; a weight ~300
   pins the membrane and causes rebound — `2026-06-04-cheat2-genuine-bg-disinhibition-RESOLVED.md`).

Success signal = **Σ_k rate(success[k])** = the neural inner product `⟨belief, intent⟩` computed multiplicatively by
coincidence. Two neural inputs; no host routing; no `L1[u][s*]` index.

### Leg 2 — READ BACK: coincidence rate → DA bus → three-factor plasticity on the utterance policy

Reuse the Wave-1 delivery pattern (volume transmission from a population rate to a group-scoped neuromodulator,
`scope=group:da`, never `all`), but now the DA **trains** an `intent → utterance` selection assembly:

- Each candidate utterance `u` leaves a **local eligibility tag** on the `intent → utterance_u` synapses when that
  utterance is emitted (actor-style eligibility already in the bridge:
  `cfg.reward_eligibility_from_coactivity` — presynaptic trace × postsynaptic event → tag; `bridge.py:8566-8611`).
- The coincidence-driven DA converts eligibility to weight change:
  `Δw = lr · (reward − baseline) · eligibility`, gated per-pathway
  (`cfg.enable_reward_modulation` + `cp_eligibility_trace`, `bridge.py:831/8137/8551+`; deferred-until-reward via
  `cfg.reward_defer_stdp_weight_update`; scoped by `cp_reward_eligibility_synapse_indices` +
  `cp_synapse_plastic_mask` + `cp_plasticity_rate_gain`). Utterance-conditioned credit uses the existing
  `from_action_specific_reward` rule (`neuromodulators.py:148-164`, fires only when
  `last_selected_action == source_action`).

Over trials, the utterance that produces belief==intent coincidence accrues DA and strengthens; the speaker's choice
becomes a **WTA over the learned `intent → utterance` assembly**, not a host argmax over the imported table.

### Leg 3 — the honesty floor stays (reuse Wave-1's no-confab moat as a truth-conditioned veto)

Unchanged from Wave-1: false utterances are removed before selection, so learned pragmatic pressure never trains a lie.

## Why this is DIFFERENT IN KIND (not a parameter tweak)

| leg | Wave-1 (failed) | Wave-2 (this gate) |
|---|---|---|
| success signal | host routing of `L1[u][s*]` (one input carries info, one is a switch) | coincidence AND of two spiking populations (belief ⊗ intent) |
| decision | host `argmax` over imported RSA table (neurons decorative) | WTA over a **learned** intent→utterance assembly (neurons decision-relevant) |
| coupling | one-directional (reward never trains the choice) | closed loop (DA → three-factor plasticity → next-trial policy) |

## Anti-cheats WITH TEETH (each can flip in the failing direction)

1. **Coincidence teeth (load-bearing).** A SHAM detector that **sums** the two inputs linearly (low threshold, no
   supralinearity / no disinhibition) must FAIL to separate match from mismatch **when total input rate is matched**
   across conditions; the real threshold-AND / disinhibitory detector must separate them. Flips: linear-sham success
   ≈ equal for match vs mismatch; real success high for match, ≈0 for mismatch. Proves the multiplicative AND (not
   summation) computes success — this is the control Wave-1 never ran.
2. **Reverse-coupling teeth.** Freeze the `intent → utterance` plasticity (`set_plasticity_gate` → 0), leaving the
   coincidence detector firing normally: the speaker must NOT learn (stays at prior/chance across trials). Intact:
   the speaker converges to the correct utterance. This is the exact property Wave-1 lacked — and it fails
   Wave-1-style (cutting reward there changed nothing because the choice was a host argmax).
3. **Lesion vs matched SHAM.** Lesion the coincidence pop (or the VIP disinhibitor) → DA silent → learning curve flat;
   a matched SHAM lesion (equal-size unrelated inhibitory/excitatory pool) → learning intact. Real flips output; sham
   does not. Read-out is the DOWNSTREAM behavioral learning curve (does the learned policy reach the correct
   utterance + generalize), NOT the success signal itself — avoiding the tautology that sank Wave-1's option B.
4. **Contingency / derangement (reuse Wave-1's fixed control).** Intent-belief deranged (3-cycle, guarantees the
   intended state moves) → coincidence silent → reward 0 → no policy change; matched → reward → policy change.
5. **Default-off byte-identity + seed.** With the Wave-2 regions off, `hash(cp_neuron_firing_thresholds)` at build#1 ==
   build#2 (seed set via `cfg.seed`, verified per CLAUDE.md), and the added regions do not perturb the base draw.

## Buildable-now verification (hooks confirmed in-code today)

- Three-factor reward-modulated plasticity: `sim/bridge.py:831, 8137, 8551-8611` (`enable_reward_modulation`,
  `cp_eligibility_trace`, `fused_eligibility_trace_decay`, `reward_eligibility_tau_ms`,
  `reward_defer_stdp_weight_update`, `reward_eligibility_from_coactivity`).
- Per-pathway scoping/freezing: `cp_reward_eligibility_synapse_indices` (`bridge.py:600/8580`),
  `set_plasticity_gate`/`set_transmission_gate` (`bridge.py:3516/3542`), `cp_synapse_plastic_mask`,
  `cp_plasticity_rate_gain`.
- Utterance-conditioned credit: `from_action_specific_reward` + `from_reward` (`sim/neuromodulators.py:95/148-164/
  647-663`); group-scoped DA neuromodulator delivery = the Wave-1 / affect-state pattern.
- Coincidence detector = threshold nonlinearity of ordinary spiking neurons (no dendrite required for the minimal
  build); disinhibitory variant reuses the validated GABA-disinhibition precedent.
- Reuse-by-import: instantiate the real `SimulationBridge`; import the RSA listener builder
  (`_recursive_tom_rsa_derisk.build_rsa_bridge`) as the social-environment belief source. No `sim/` edit.

## Biological grounding (READ, not skimmed)

- **VIP-SST-pyramidal disinhibition:** Pi et al. 2013 (cortical VIP interneurons inhibit SST, disinhibiting
  pyramidal cells) — the inhibitory-on-inhibitory opening motif (via `2026-08-04-...v12-disinhibitory-boundary`
  research gate, which read Pi 2013, Schneider 2014 motor-collateral→recipient inhibition, Zhang 2021 M2 SOM/PV).
- **Local GABAergic confinement of excitation:** Kandel PNS 6e, Fig 58-5 / pp. 1455-1456
  (`sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt`).
- **Coincidence detection by threshold/NMDA supralinearity:** intrinsic biophysics of the spiking neuron (rheobase);
  the multiplicative AND is what a summator cannot do (the teeth of anti-cheat 1).
- **Reward-modulated three-factor plasticity / actor-critic policy improvement:** catalog C.29/C.30, O.20 (Schultz
  1998; Sutton & Barto actor `H(s,a)←H(s,a)+αδ`); the substrate's `enable_reward_modulation` path implements the
  `(reward−baseline)·eligibility` form.
- **Contingent-vs-yoked reward + the action-conditioned-value requirement:**
  `2026-08-03-neural-vocal-credit-gateB-v1-yoked-NO-GO.md` — raw reward without an action-conditioned RPE created a
  self-reinforcing loop under yoked feedback; the fix was a spiking action-conditioned value critic. This is the
  named risk for Leg 2.

## Builds ON

- `2026-08-08-pragmatics-perlocutionary-reward-wired-to-speaking-6seed-GO.md` (Wave-1 — the method this supersedes;
  its residual #1 IS this gate's target).
- `2026-08-01-W4-recursive-theory-of-mind-...-6seed-GO.md` (RSA listener = the belief source, imported).
- `research/findings/raw/_learned_talkativeness_scoping.md` (the reward→context→policy plasticity composition + hooks).
- `2026-06-27-conv-thinking-research-discourse-pragmatics.md` (the gap map: §6 implicature/RSA, §8 speech acts).

## Honest feasibility

**buildable_now = YES**, reuse-by-import, no `sim/` edit — all plasticity, gating, scoping and neuromodulator hooks
are exposed today. **Honest risk:** Leg 1 (coincidence) is low-risk (standard threshold-AND). Leg 2 (read-back) is
the risk: the vocal-credit arc showed a naive DA→three-factor loop can over-reinforce whichever action was active
early and fail the yoked control, needing an **action-conditioned value baseline** to subtract expected reward. Here
the coincidence success is naturally utterance-contingent (mismatch → no DA), which is better-posed, and the
three-factor rule already carries a running baseline — but whether that suffices to converge the policy across 6 seeds
is the empirical question. If the naive loop fails anti-cheat 2/3 convergence, add the reusable spiking value critic
(still no `sim/` edit) — that is a de-risk step, not a wall. Recommend: preregister Leg-1 coincidence + linear-sham
teeth FIRST (cheap, decisive on whether the neural success signal is real), then Leg-2 read-back with the
reverse-coupling + matched-sham-lesion battery, escalating to the value critic only if the naive loop stalls.
