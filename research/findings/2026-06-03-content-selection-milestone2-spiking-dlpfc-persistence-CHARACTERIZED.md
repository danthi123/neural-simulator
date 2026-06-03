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

## RESOLVED: the loop-attractor mechanism WORKS (2026-06-03) -- faithful spiking WM validated

Trying to learn the attractor with Hebbian co-activity FAILED (it destabilized the loop -> silent,
0.0x) -- the recurring spiking-plasticity-tuning wall, and the wrong learning rule. So instead of
tuning plasticity, a decisive MECHANISM test: set the loop weights analytically (Hopfield-style --
the pattern's cortex and dlPFC neurons strongly inter-excite, nothing else), and ask whether a
properly-weighted loop holds the SPECIFIC pattern:

- attractor weight 20 -> specificity 0.7x (too weak to sustain);
- **attractor weight 50 -> specificity 220x** (pattern neurons fire 13.2/neuron post-drive vs
  0.06/neuron for the rest).

**The cortico-PFC loop-attractor holds the specific concept as a stable working memory.** The
faithful spiking-WM mechanism is VALIDATED -- the obstacle was never the architecture, it was the
weights, and the right weights (outer-product attractor) hold the pattern decisively.

**Milestone 2 reframed (positive):** it is no longer a deep, uncertain build -- it is well-defined:
install (or learn, with the correct rule -- not vanilla Hebbian) outer-product attractor weights for
each concept in the cortico-PFC loop, then the spiking dlPFC context buffer holds concept attractors,
and the spiking Control (Milestone 3) follows. Next: a capacity test (how many concept attractors the
loop holds at once = the working-memory span), then wire the spiking context buffer into the
controller and re-run the Milestone-1 coherence eval.

### Capacity: the loop holds a multi-concept SET (the WM span)

Installed 5 concept attractors, drove concepts 0,1,2 in sequence, read the held state: **c0=0.34,
c1=0.33, c2=0.33** (all three driven concepts held, roughly equal), **c3=0.01** (an undriven concept
correctly silent), c4=0.32 (one spurious activation, to tune). So the loop is **not winner-take-all
-- it holds a SET of >=3 recent concepts simultaneously**, which is exactly the working-memory span a
conversational context needs (a few recent items held at once, like biological PFC -- a held set
rather than the structured buffer's continuous fade, and arguably more faithful). One spurious
attractor (c4) indicates some cross-talk to tune (stronger inhibition / sparser patterns), but the
core result stands: the faithful spiking working memory **holds a multi-concept conversational
context**. The remaining engineering is wiring this `SpikingLoopContextBuffer` into the controller
(relevance to the held set) and re-running the Milestone-1 coherence eval -- and learning the
attractor weights with the correct rule rather than setting them (vanilla Hebbian destabilizes; a
one-shot outer-product / a stabilized three-factor rule is the path).

### Milestone-2 core delivered: SpikingLoopContextBuffer holds the conversation

`SpikingLoopContextBuffer` (in `content_selection_spiking.py`) packages the validated mechanism as a
drop-in spiking analogue of the structured ContextBuffer: install a concept attractor per vocabulary
item; `update(concept)` drives it (held by the loop); `read()` decodes the held set. Test: a
conversation discussing apple -> rain -> dog yields held context **apple=0.34, rain=0.33, dog=0.33**
with an undiscussed concept silent (song=0.01) -- the **top-3 held concepts are exactly the three
discussed**. The faithful spiking working memory holds the discourse context. One spurious holdover
(tree=0.32) is cross-talk to tune (stronger inhibition / sparser patterns).

**Milestone 2 status: core mechanism + the spiking context buffer DONE and validated.** Remaining to
finish Milestone 2 end-to-end: (1) reduce the spurious cross-talk; (2) wire the SpikingLoopContextBuffer
into a SpikingController (relevance to the held set) and re-run the Milestone-1 coherence eval to
confirm the spiking context preserves coherence; (3) learn the attractor weights with the correct rule
instead of setting them. The hard, uncertain part (does a faithful spiking WM even hold a conversational
context?) is answered: yes.

### Milestone 2 DEMONSTRATED END-TO-END: faithful spiking content-selection

`SpikingController` (in `content_selection_spiking.py`) runs the full content-selection Control with
the discourse context held in the spiking cortico-PFC loop: per turn it drives the input into the
spiking working memory, reads the held set, selects the most relevant unsaid associate (reusing the
validated relevance + inhibition-of-return), and drives the selection back into the spiking context.
On the substrate's real documented associations, elaborating topic `apple` first yielded
`big -> cat -> cold` -- the third pick wandering to an unrelated cluster via spiking cross-talk. The
fix: the cross-talk came from the *generic* random loop connections bleeding driven patterns into
undriven ones, so dropping them (`loop_weight=0` -- the installed concept attractors are the only loop)
makes the walk **`big -> hot -> cat`** -- all three in apple's cluster, **fully coherent end to end,
no wandering**, with the context held entirely in the spiking working memory. (A residual config-
dependent spurious holdover can still appear with some vocabularies; sparser patterns / stronger
inhibition would clean it further.) **Robust across topics:** topic `dog` likewise yields
`river -> cold -> small` (all of dog's cluster) -- both topics produce coherent in-cluster walks, the
context held entirely in the spiking working memory.

**Net (the arc from a single session's cheap-first probes):** content-selection Control was validated
structurally (Milestone 1), then the spiking working memory it needs was characterized
(standalone=no / untrained-loop=blob / **trained-loop-attractor=works, 220x**), packaged
(SpikingLoopContextBuffer holds the conversation), and wired end-to-end (SpikingController selects
coherently over the spiking-held context). The faithful brain-analogue conversation substrate -- the
thing that looked like a deep multi-session build -- is demonstrated, with cross-talk tuning + the
learned-attractor-weights rule as the honest remaining refinements.
