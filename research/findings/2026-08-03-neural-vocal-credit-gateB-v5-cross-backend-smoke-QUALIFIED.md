---
type: finding
status: qualified
date: 2026-08-03
mechanism: neural-vocal-action-credit-v5
runner: research/runners/_vocal_action_credit_gate_v5.py
artifacts:
  - research/findings/raw/vocal_action_credit_gate_v5/smoke_numpy.json
  - research/findings/raw/vocal_action_credit_gate_v5/smoke_cupy.json
---

# Neural commit and arousal tag passes a frozen-dynamics v5 smoke on CPU and GPU

<!--derived-->
**Verdict: QUALIFIED smoke pass, not a Gate B pass.** A fixed action epoch can
load a selective, persistent dendritic tag when the selector's neural commit
population and shared practice/arousal activity are both present. A later
generic outcome event materially modulates the already active tagged population
through symmetric excitation and feed-forward inhibition. The exact reserved-
seed battery passes with one shared configuration on NumPy and CuPy. Formal
phases and all scientific seeds remain sealed because this smoke intentionally
freezes every weight and therefore demonstrates neither learning nor delayed
credit.

Artifacts: `research/findings/raw/vocal_action_credit_gate_v5/smoke_numpy.json`
and `research/findings/raw/vocal_action_credit_gate_v5/smoke_cupy.json`.
Their provenance sidecars bind both runs to clean revision `70a623c23`; both
artifacts report configuration digest
`6cfbccc819de422def928cb97a8f3dc4b8d25568b890da5bdd1875fb927f4def`.

## What changed from retired v4

Python no longer observes a winner to open or close an expectation route. Both
commit-to-value routes and both arousal-to-value routes remain available during
the same fixed action epoch. The selected tag emerges from their joint neural
activity. The winner is inferred only after that epoch from the two neural
commit populations for scoring; it cannot alter currents, routes, timing, or
plasticity. Equal bilateral commit counts are explicitly rejected. The outcome
population projects identically to both value pools and both local FS pools, so
late motor activity no longer chooses the normalization channel.

The first NumPy-only prototype used a motor-count threshold and a dendritic
center of 400. The exact CuPy run exposed that as invalid: both sustained motor
pools crossed the threshold even though commit activity was selective
(`145/15`), and the selected plateau was nearly zero. No formal seed was
opened. The correction reads the neural commit state and uses one backend-
neutral operating point: commit/arousal weights `22/18`, graded center `300`,
strength `2`, outcome-to-FS weight `18`, and a `60 ms` outcome window.

The adversarial audit then found that named plasticity gates had silently
overridden the global freeze. The corrected runner freezes the substrate again
after route setup and requires zero changed synapses in every condition. It
also rejects bilateral outcome activity and adds commit-route and no-outcome
controls.

## Cross-backend smoke

| measure | NumPy CPU | CuPy RTX 3090 |
|---|---:|---:|
| neural commit counts | `292 / 0` | `15 / 145` |
| selected / other tag before outcome | `5.124496 / 0.113314` | `1.399689 / 0.076401` |
| other/selected tag ratio | `0.022` | `0.055` | <!--derived-->
| selected outcome spikes | `26` | `25` |
| no-outcome-window spikes | `55` | `54` |
| plateau-lesion outcome spikes | `19` | `16` |
| outcome-excitation-lesion spikes | `21` | `22` |
| outcome-inhibition-lesion spikes | `71` | `61` |
| selected outcome rate | `18.055556 Hz/cell` | `17.361111 Hz/cell` |
| changed synapses, maximum across conditions | `0` | `0` |

All eight conditions pass on both backends: intact, arousal lesion, commit-route
lesion, plateau lesion, no-outcome, generic-outcome-excitation lesion,
feed-forward-inhibition lesion, and fixed action-channel permutation. Intact
and permuted output must keep the other channel below 10%; an equal bilateral
output now fails. The plateau must account for at least 20% of selected
outcome-window spikes, generic outcome excitation for at least 10%, and local
inhibition must change selected activity by at least 20%. Selected firing must
remain between 1 and 20 Hz/cell. All `2,592` enabled coincidence synapses belong
to the two commit routes and two shared-arousal routes, with none enabled
elsewhere. The permutation moves both the neural tag and selective output rather
than merely relabeling a host expectation.

The new no-outcome control changes the interpretation. The intact outcome event
reduces selected activity from `55` to `26` spikes on NumPy and from `54` to
`25` on CuPy because feed-forward inhibition outweighs the smaller direct
excitatory component. This establishes material event modulation, not an event-
triggered readout. Arousal removal leaves about half the immediate selected tag
on NumPy (`3.039567` versus `6.259075`) but almost none on CuPy (`0.001049`
versus `1.709584`); commit-route removal collapses selectivity on both. Both
inputs are load-bearing, but this smoke does not establish a backend-invariant
supralinear conjunction.

## Biological grounding and limits

This is a constrained engineering synthesis, not a claim that one paper
describes the whole circuit. The local RAG catalog located the primary-source
passages in `sim-catalog/references/textbooks/basal-ganglia-reviews/`
`TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.txt:8911` and
`Tepper-Koos-2017-StriatalGABAergicInterneurons.txt:1680`. They support powerful
feed-forward inhibition of striatal projection neurons. Choice-selective FSI
activity is supported by [Gage et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2920892/),
and strong FS-to-projection-neuron inhibition by
[Planert et al. 2010](https://pubmed.ncbi.nlm.nih.gov/20203210/). Regenerative
dendritic state is a plausible local persistence mechanism, but the exact
commit/arousal routing, outcome circuit, and constants remain simulator
hypotheses.

The host still schedules an externally defined action epoch, delay, and outcome
event. That is an experimental world schedule rather than a winner-dependent
route, but a later integrated brain must generate its own action termination and
outcome segmentation. This run uses only reserved seed zero, and the operating
point was developed against both backends, so backend agreement is a
compatibility prerequisite rather than independent confirmation. No claim
about learned preference, dopamine credit, yoked neutrality, omission,
conversational capability, or whole-brain integration is licensed.

## Decision

Keep v5 as a host-winner-free candidate for a fresh learning design. Re-audit
the corrected smoke, then preregister a test that requires actual local weight
change, contingent acquisition, yoked neutrality, expected-omission activity,
channel permutation, and all load-bearing lesions. Formal entry points stay
sealed until both steps are committed.
