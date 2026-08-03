---
type: finding
status: contributing
date: 2026-08-03
mechanism: developmental-vocal-convention
runner: research/runners/_developmental_vocal_convention_derisk.py
artifacts:
  - research/findings/raw/developmental_vocal_convention_s42_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s42_t360_full.json.prov.json
  - research/findings/raw/developmental_vocal_convention_s43_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s43_t360_full.json.prov.json
  - research/findings/raw/developmental_vocal_convention_s44_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s44_t360_full.json.prov.json
  - research/findings/raw/developmental_vocal_convention_s100_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s100_t360_full.json.prov.json
  - research/findings/raw/developmental_vocal_convention_s101_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s101_t360_full.json.prov.json
  - research/findings/raw/developmental_vocal_convention_s102_t360_full.json
  - research/findings/raw/developmental_vocal_convention_s102_t360_full.json.prov.json
---

# A shared spiking brain learns a small vocal convention from consequences

<!--derived-->
**Verdict: GO at six GPU seeds for a preverbal two-intent by two-referent convention.** A shared spiking network
learned which raw vocal channels an external listener treats as `request` versus `report` and `apple` versus `river`.
It learned from the listener's contingent response, composed the two intent-referent combinations withheld from
training, and learned a freshly swapped convention in a new brain. This is an early communication-learning result,
not natural language or open-ended conversation.

## Role In The Whole Brain

The previous grounded-speech result established one fixed food request. This experiment asks whether communication
can begin as an action whose meaning is learned through social consequences. Internal need or joint attention selects
an intent route, perception selects a referent route, and the brain may emit a raw two-channel vocal action. The host
listener assigns the external meaning and changes the world only when that action matches its convention.

The learning sequence is:

```text
internal state + perceived object -> raw vocal action -> listener consequence
    -> reward-US spikes -> SNc-like activity -> dopamine-gated local eligibility
    -> changed vocal choice on a later encounter
```

The host never supplies the desired vocal channel during learning. Exploration tries raw channels, and reward arrives
only after the listener consequence.

## Result

Per-seed measurements and commands are preserved in
`research/findings/raw/developmental_vocal_convention_s42_t360_full.json` and its sibling seed artifacts and
provenance sidecars.

Training presented only `request apple` and `report river`. Evaluation also included the untrained cross-combinations
`request river` and `report apple`.

| seed | main joint accuracy | held-out combinations | fresh swapped convention | no consequence | yoked reward | dopamine lesion | changed synapses outside vocal routes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.00 | 2/2 | 1.00 | 0.00 | 0.25 | 0.00 | 0 |
| 43 | 1.00 | 2/2 | 1.00 | 0.00 | 0.00 | 0.00 | 0 |
| 44 | 1.00 | 2/2 | 1.00 | 0.00 | 0.00 | 0.00 | 0 |
| 100 | 1.00 | 2/2 | 1.00 | 0.00 | 0.50 | 0.00 | 0 |
| 101 | 1.00 | 2/2 | 1.00 | 0.00 | 0.00 | 0.00 | 0 |
| 102 | 1.00 | 2/2 | 1.00 | 0.00 | 0.00 | 0.00 | 0 |

All six seeds passed every registered check. Main training produced 44-46 contingent rewards and changed 11,907-12,071
synapses, all within the declared vocal learning routes. The fresh-brain swapped control learned the opposite
raw-channel meanings at all six seeds.

## What The Controls Establish

- **The convention is external and learnable:** a new brain learns when both intent and referent meanings are swapped.
- **The result is compositional within this tiny factorization:** both combinations absent from training are correct.
- **The consequence is necessary:** withholding the listener's response produces zero joint accuracy.
- **Temporal contingency matters:** replaying an unrelated reward schedule produces at most one correct case of four.
- **The dopamine path is necessary:** reward without SNc/dopamine-dependent weight expression produces zero accuracy.
- **State and perception are causal:** context lesion, perception lesion, and the no-reason condition prevent a valid
  action as predicted.
- **Learning is anatomically scoped:** no measured synapse outside the vocal routes changes.

## Honest Boundary And Scaffolds

This runner uses two hand-declared intent channels, two hand-declared referent channels, fixed regional anatomy, direct
sensory and body currents, spike-count action readout, and a host listener/world. A balanced host-injected motor
babbling schedule makes the immature network try every raw action combination; it does not read the target action or
the listener mapping, but it remains a major developmental scaffold. Architecture and operating-point values are
hand-set.

The result does not contain words, syntax, phonology, a speech motor system, unrestricted meanings, continuous human
interaction, or a claim of understanding comparable to a language model. It also does not establish same-brain
adaptation after a listener changes an already learned convention. The current positive eligibility rule can
strengthen a rewarded route but has no validated omission/error-driven process for weakening the obsolete route.

## Next Mechanism

1. Replace injected balanced babbling with intrinsic, brain-generated exploration driven by novelty, uncertainty, and
   social or homeostatic value.
2. Add omission- or prediction-error-dependent depression and pass same-brain convention reversal.
3. Expand the world, needs, percepts, and meanings while retaining held-out composition and causal controls.
4. Connect the learned preverbal message to a brain-native word and sequence-learning path.
5. Bring source memory, confidence, affect, and speech inhibition onto the same continuously operating bridge.
