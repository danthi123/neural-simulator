# Content-selection Milestone 2 (spiking dlPFC context) — cheap-first CHARACTERIZED (2026-06-03)

**Result: the spiking dlPFC context buffer does not drop in — working-memory PERSISTENCE is a genuine
spiking-tuning problem, cleanly isolated by a cheap-first test before any large build.**

## What was tested

`research/runners/content_selection_spiking.py`: a minimal one-region bridge using the project's
validated dlPFC working-memory configuration (recurrent self-excitation + NMDA bistability;
`g11_bg_runner.py` `dlpfc_wm`). The question (Milestone 2 / Approach 3): can a spiking dlPFC region
hold a fading multi-concept context the way the Milestone-1 structured buffer does? Drive concept
patterns into the region, then read the sustained firing after the drive stops.

## Findings (decisive)

1. **The dlPFC fires strongly DURING the drive** — 1250 spikes over 60 steps for a 50-neuron pattern
   at 2500 pA (and saturating by 8000 pA). So the neurons and the drive are fine.
2. **It goes silent the instant the drive stops** — read-window firing drops to noise (2-12 total
   spikes). The untrained random recurrence (density 0.2) has no concept-specific attractor, so the
   pattern does not re-excite itself: **no persistent activity = no working memory.**
3. **Cheap attractor attempts did not fix it** — enabling plastic recurrence + Hebbian co-activity
   learning and training each pattern 40x, and raising recurrent self-excitation from 2.0 to 6.0,
   both still showed no persistence (read firing ~12, decoded context ~0). Forming a self-sustaining
   spiking attractor needs more than these cheap levers.

## Interpretation (honest)

Spiking working-memory persistence requires the network to sit in a bistable regime where a driven
pattern re-excites itself after the input is removed. A standalone recurrent region with random,
lightly-trained connectivity does not reach that regime here. Notably, the project's PFC working
memory has worked **inside the full circuit** (cortex -> PFC -> cortex loops, the basal-ganglia loop
sustaining activity), not as a standalone self-sustained attractor — which suggests faithful WM
persistence may need the **cortico-PFC loop**, not just a recurrent dlPFC region, and/or a carefully
tuned NMDA/recurrence/inhibition balance plus proper attractor training.

This matches the session-wide pattern: spiking *dynamics* tuning (firing propagation, persistence) is
genuinely hard; the structured / cheap-first versions validate the mechanism, and the spiking-faithful
versions are a separate, harder step.

## Status

- **Milestone 1 (structured Control) stands VALIDATED** (coherent dialogue vs no-control, 5/5 seeds;
  finding 2026-06-03-content-selection-control-milestone1-VALIDATED.md). The content-selection
  mechanism works and is the usable result.
- **Milestone 2 (spiking dlPFC context) is now de-risked + scoped**: it is a dedicated spiking
  working-memory build — get persistent activity first (bistability tuning and/or the cortico-PFC
  loop + attractor training), then it can replace the structured context buffer and re-run the
  Milestone-1 coherence eval. Not a session-tail drop-in.

## Next options (owner-strategic)

1. Dedicated spiking-WM persistence build: tune NMDA/recurrence/inhibition for bistability and/or add
   the cortico-PFC loop, train concept attractors, then verify a fading multi-item context.
2. Strengthen Milestone 1 instead: richer real-association substrate (a tagged bridge) for a larger
   coherence eval, and extend the structured controller toward interactive (query-answering) dialogue.
3. A different conversational capability on the validated substrate.

Cheap-first; reuse-by-import (no protected-module edits); honest negative is the deliverable; both
remotes.
