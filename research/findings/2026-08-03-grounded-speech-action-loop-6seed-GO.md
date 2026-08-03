---
type: finding
status: contributing
date: 2026-08-03
mechanism: grounded-speech-action
runner: research/runners/_grounded_speech_action_loop_derisk.py
artifacts:
  - research/findings/raw/grounded_speech_action_loop_6seed.json
  - research/findings/raw/grounded_speech_action_loop_6seed.json.prov.json
---

# Grounded speech-action loop: hunger, learned perception, request, and satiety GO

<!--derived-->
**One-line verdict.** One shared spiking bridge now learns a food association and combines it with body state to choose
a conceptual request. A hungry brain seeing the learned apple requests it, the world delivers food only after that
request, and the same apple scene produces silence after satiation. The RTX 3090 run passed all causal checks on six
seeds. This is the first preverbal communication rung, not natural language.

## Role In The Whole Brain

Language should be an action caused by what the brain perceives, needs, remembers, and expects to happen next. This
runner tests the smallest complete case. It connects a learned visual association, an AgRP/POMC-inspired hunger and
satiety drive, a request-versus-silence competition, and a body-changing social consequence in one simulation.

The important result is not the word `apple`. It is the closed causal chain:

```text
learned food percept + hunger -> neural request -> food delivery -> satiety -> silence
```

## Mechanism

- A caregiver presentation co-activates the apple visual features and a food-cue population. Local gated Hebbian
  plasticity strengthens only that association, then learning is frozen for evaluation.
- The existing co-resident AgRP-like and POMC-like pools represent hunger and satiety as graded spiking activity.
- Food cue and hunger converge on a request population. Satiety drives a competing silence population. A shared
  fast-spiking inhibitory pool makes the two actions compete.
- Host code renders the visual/body currents, decodes the fixed conceptual action `request apple`, and changes energy
  only if that request occurs. It does not choose whether the brain speaks.
- Hungry-to-sated trials remain continuous. Counterfactual controls restore the same post-training neural state so
  each lesion changes only the named cause.

## Six-Seed Result

Artifact: `research/findings/raw/grounded_speech_action_loop_6seed.json` (CuPy/RTX 3090).

Request margin is request spikes minus silence spikes. Positive means request; negative means silence.

| seed | hungry apple | same scene, sated | wrong object | drive lesion | perception lesion | learned-route lesion | drive correlation |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | +52 | -401 | -71 | -139 | -71 | -71 | 0.982 | <!--derived-->
| 43 | +34 | -466 | -81 | -160 | -81 | -81 | 0.983 | <!--derived-->
| 44 | +95 | -492 | -102 | -132 | -102 | -102 | 0.995 | <!--derived-->
| 100 | +134 | -357 | -40 | -83 | -40 | -40 | 0.981 | <!--derived-->
| 101 | +77 | -381 | -54 | -124 | -54 | -54 | 0.988 | <!--derived-->
| 102 | +156 | -388 | -87 | -113 | -87 | -87 | 0.993 | <!--derived-->

All six seeds learned nonzero association changes. Energy rose from 0.25 to 1.00 only after the brain request. The
logged request preceded delivery on every seed. Without the consequence, energy stayed at 0.25 and the brain requested
again. Untrained association weights also produced silence on every seed.

## What The Controls Establish

- **Need is necessary:** removing hunger activity collapses the request.
- **Perception is necessary:** removing visual input collapses the request.
- **Learning is necessary:** zero or initial association weights collapse the request.
- **The referent is specific:** a learned-irrelevant river percept remains silent while hunger is unchanged.
- **Body consequence matters:** the same scene changes from request to silence only after energy changes.
- **Drive is graded:** request margin tracks energy deficit with Pearson correlation above 0.98 on every seed.

## Failed Route That Should Not Be Repeated

The first implementation read the learned category-concept population. It passed only two of six honest stabilized
seeds: one seed produced no usable concept spikes, and another responded more strongly to the wrong object. That is the
known raw graded-readout wall documented in
`research/findings/2026-06-16-generalization-onsubstrate-convergence.md`.

The promoted route learns directly from the active visual-feature population. It is still local and spiking, and the
wrong object produced zero food-cue spikes in the tested operating point. A future concept-level route must solve the
graded neural read rather than recreate a host-normalized score.

## Honest Boundary And Scaffolds

This is one conceptual action, not fluent speech. The caregiver pairing is explicit. The food-cue, request, silence,
and inhibitory populations are hand-declared. Their major weights, request threshold, and silence bias are set by the
runner. A host decoder maps one neural winner to `request apple`; there is no learned intent choice, word order,
articulation, or user conversation. The body/world belongs on the host, but the fixed semantic decoder is temporary.
Snapshot restore is an evaluation instrument and is not used by the live hungry-to-sated path.

The 267M WKV language circuit is deliberately absent. A renderer may be attached later only after the brain chooses
the intent, referent, certainty, and whether to speak.

## Next Mechanism

1. Learn several need-to-object associations and choose the relevant referent.
2. Learn several intents from contingent outcomes rather than fixing `request`.
3. Replace the host semantic decoder with a brain-native message-to-word path.
4. Add curiosity or uncertainty as a second internal reason to communicate.
5. Bring learned source monitoring onto this same bridge before claims are rendered.
