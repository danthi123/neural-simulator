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

## Follow-up: the cortico-PFC LOOP probe (2026-06-03) -- persistence YES, content NO (untrained)

Tested the standing hypothesis directly with `build_loop_wm_bridge` -- two mutually-exciting NMDA
regions (cortex_ctx <-> dlpfc_wm) forming a reverberating loop:

- **Persistence is recovered by the loop.** Post-drive cortex spikes over a 40-step delay:
  standalone region ~5; loop at coupling weight 4/10/20 = 12 / 16 / **182**. At strong coupling the
  loop clearly self-sustains activity after the drive is removed -- the single region could not.
  This confirms the hypothesis: biological WM persistence comes from the **loop**, not a lone
  recurrent region.
- **But the sustained activity is a generic blob, not the driven pattern.** Pattern-specificity
  ratio = **0.2x** (driven-pattern neurons fire 0.10/neuron vs 0.51/neuron for the rest -- i.e. the
  activity drifted OFF the pattern). With random untrained loop connections, the loop sustains
  *some* activity but not the *specific* concept, so it carries no usable content.

**Mechanism fully characterized (cheap-first, three probes):** (1) standalone region = no
persistence; (2) untrained loop = persistence but no content; (3) => **faithful spiking WM = a
TRAINED cortico-PFC loop** -- the loop connections must be shaped (autoencoder/attractor: cortex
pattern -> dlPFC pattern -> back to the SAME cortex pattern) so the reverberation holds the specific
concept. That is the scoped faithful build: train the loop into pattern-specific attractors, then the
spiking dlPFC context buffer (Milestone 2) and the spiking Control (Milestone 3) follow.

This is a biology-translatable characterization (persistence = loop reverberation; content =
trained loop attractors) obtained for the cost of three small probes -- the cheap-first discipline
locating exactly what the faithful build requires before committing to it.
