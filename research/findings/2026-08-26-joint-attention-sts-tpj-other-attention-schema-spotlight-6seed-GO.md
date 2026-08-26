---
type: finding
status: contributing
date: 2026-08-26
mechanism: joint-attention-gaze-following
lane: catalog-derisk
seeds: [42, 43, 44, 100, 101, 102]
instrument: four-way decomposition (intact vs lesion-of-schema-output vs scrambled-partner-gaze vs layout-blind gaze-only) attributes the alignment to the gaze x layout inference, not to a fixed response or a copied coordinate
---

# Joint attention (STS-TPJ other-attention-schema -> spiking one-of-K spotlight) is a 6-seed GO (Stage-0 ToM)

## Claim
An other-attention-schema aligns the agent's attentional spotlight to the object a PARTNER is inferred to be
attending to, from the partner's gaze/biological-motion cue alone. At 6 seeds the runner's OWN verdict is `GO`:
the spotlight tracks the partner's actual target far above chance, and all four built-in anti-cheat controls
collapse to chance as designed. This de-risks Kandel's STS-temporoparietal-junction mentalizing component
(Fig. 62-4), the developmentally-earliest theory-of-mind precursor (Stage-0), as a spiking mechanism.

## Result
`research/runners/_joint_attention_derisk.py --seeds 42 43 44 100 101 102 --n-trials 60` (K=6, n_dir=48, chance
= 1/6 = 0.167), CPU numpy (SIM_BACKEND=numpy). Aggregate artifact
`research/findings/raw/_joint_attention/summary_6seed.json` -> `go: true`, `checks` all true. Anti-cheat band =
chance + 0.10 = 0.267.

Aggregate over 6 seeds:
- `align_acc` = 0.978 <!--derived--> (GATE >= 0.85; rounded mean of artifact aggregate.align_acc = 0.9778333) -- the spotlight tracks the partner's actual attended object.
- `align_acc_lesion` = 0.000  (anti-cheat <= 0.267) -- sever the STS-TPJ read (uniform spotlight drive) and
  alignment collapses BELOW chance to zero: the schema OUTPUT is load-bearing.
- `align_acc_scramble` = 0.175  (anti-cheat <= 0.267) -- replace the partner's gaze with another trial's gaze
  and alignment falls to chance: the answer rides the ACTUAL partner gaze, not a fixed response.
- `align_acc_blind` = 0.178 <!--derived--> (not-a-copy <= 0.267; rounded mean of artifact aggregate.align_acc_blind = 0.1776667) -- a layout-blind decode (fixed angular bin of the gaze,
  ignoring the per-trial object layout) is at chance: gaze alone cannot name the object; the intact success
  needs gaze x layout, so it is an inference, not a copy of a transmitted coordinate.

Per-seed `align_acc` / `lesion` / `scramble` / `blind` (chance 0.167):
- seed 42:  0.967 / 0.000 / 0.250 / 0.100
- seed 43:  0.917 / 0.000 / 0.200 / 0.267
- seed 44:  1.000 / 0.000 / 0.150 / 0.183
- seed 100: 1.000 / 0.000 / 0.183 / 0.183
- seed 101: 0.983 / 0.000 / 0.167 / 0.133
- seed 102: 1.000 / 0.000 / 0.100 / 0.200

Every seed clears `align >= 0.85`. The three near-chance controls are stable across seeds; seed 43's
`blind` = 0.267 sits at the band ceiling as a single-seed sampling value, but the verdict is computed on the
6-seed aggregate (`blind` = 0.178 <= 0.267), which passes. <!--derived--> The 1-seed smoke (seed 42) reproduced exactly.

## Instrument + control
- Instrument: the four-way decomposition above. It attributes the alignment: lesion isolates the schema output,
  scramble isolates dependence on the true partner gaze, and the layout-blind arm isolates that the target is a
  gaze x layout conjunction rather than a gaze-only readout. The lesion arm collapsing to 0.000 (below the 0.167
  chance floor, i.e. a uniform-drive spotlight that never matches) is the discriminating control.
- The object index is decorrelated from angular rank per trial (`make_layout` permutes the angle->identity
  assignment), which is what makes the layout-blind baseline a genuine at-chance control.
- The artifact's `attribution` block (`tools.lab.attributable_to`) records the fraction of the intact alignment
  NOT present in each control: the alignment is entirely absent in the lesion control (fully attributable to the
  STS-TPJ read) and largely absent in the scramble and layout-blind controls, so almost none of it is a fixed
  response or a copied coordinate.

## What this is NOT (honesty boundary; brain-based-only standard)
- NOT fully spiking end-to-end. The gaze-direction ring (Izhikevich) and the one-of-K spotlight competition
  (K Izhikevich attractor pools + shared FS lateral inhibition) ARE spiking, but the STS-TPJ object-cell read
  `s[k] = W_obj[k,:] @ rates` is a HOST synaptic-sum over the gaze-ring spike rates, and the spotlight winner is
  a HOST argmax over accumulated spike counts. Under this project's brain-based-only standard both are documented
  shortcuts (the direction-tuned synaptic read and the read-out), to be converted to on-substrate synapses / a
  neural read-out in a later rung.
- NOT wired / integrated / closed. This is a runner-only, default-off de-risk; joint attention is not reachable
  from the production `/api/brain-chat` path. "Closed" would require the integrated, on-by-default, scaffold-
  retired form per docs/TERMS.md. The correct status here is: de-risked at 6 seeds (GO).

## Biology binding
`research/biology/joint-attention-gaze-following.md` -> Kandel, Principles of Neural Science, Ch. 62, Fig. 62-4:
the mentalizing-system component "in the temporoparietal region of the superior temporal lobe, is known to be
activated by eye gaze and biological motion" (STS-TPJ), and "mutual attention normally appears toward the end of
the first year when signs of mentalizing are still sparse" -- grounding joint attention as a dissociable Stage-0
precursor to false-belief mentalizing (W3 / mPFC). Both anchors resolve in the indexed source.

## Next (no-defer)
The de-risk is GO. The next rungs toward one-brain, in order: (1) replace the host STS-TPJ synaptic-sum with an
on-substrate direction-tuned pathway so the object-cell drive is computed by real synapses on gaze-ring spikes;
(2) replace the host argmax read-out with a neural read of the spotlight winner; (3) drive the schema from a live
partner cue in conversation/scene rather than a synthetic per-trial layout, so joint attention happens through
use. None of these is blocked by the present result.
