# Biology reference for direction 2 (multi-turn dialog): multiple timescales in hippocampus + PFC; cross-episode binding; persistent parahippocampal spiking (2026-05-24)

Web-search-found references during the (c) decisive GPU run. Saves biology for direction 2 of the post-(c) roadmap (multi-turn dialog) so it's ready when direction 2 dispatches.

## Key biology findings

### Multiple timescales (Tang et al.)
Hippocampal place-cell sequences exist at MULTIPLE timescales:
- Fast theta sequences: 100-200ms within theta oscillation cycles (matches the project's gamma-slot framework: 7 slots × ~14ms per slot = ~100ms theta cycle)
- Slow behavioral sequences: ~seconds (extended-time-scale behavior)

Direct mapping to direction 2:
- Within-turn dialog: fast theta sequence (the (c) loop's per-iteration gamma-slot binding)
- Across-turn dialog: slow behavioral sequence (dlpfc_wm holds dialog state; consolidation cycles transfer recent turns into schema)

### Cross-episode binding (Preston & Eichenbaum 2013; van Kesteren 2010 / Bunsey & Eichenbaum-like)
Both MTL and PFC support:
- RAPID WITHIN-EPISODE BINDING: standard hippocampal episodic encoding (D.14 engram tagging already validated in the project)
- CROSS-EPISODE BINDING: binding information ACROSS multiple distinct episodes for "mnemonic flexibility"

Direct mapping to direction 2:
- Each user query = one episode (engram-tagged via D.14)
- Multi-turn dialog requires CROSS-EPISODE BINDING: current turn's PFC frame references prior turns' engrams
- The PFC infrastructure for cross-episode binding is what direction 2 needs to build (the within-episode mechanism is already validated)

### Persistent parahippocampal spiking
Parahippocampal regions can SUSTAIN representations over many seconds. This supports BOTH working memory maintenance AND episodic encoding into long-term memory. The dual-purpose nature is biology-faithful for direction 2: dlpfc_wm holds the dialog frame across turns AND that frame can be engram-tagged into the hippocampus for cross-turn retrieval.

### Hippocampal-cortical dialog (slow regularity extraction)
The "dialog" between hippocampus and cortex extracts statistical regularities across overlapping episodic events. For a multi-turn conversation, this means: as the dialog accumulates turns, the schema progressively incorporates dialog-context regularities (the conversational equivalent of consolidation extracting domain knowledge from individual episodes).

## Mapping to direction 2 (multi-turn dialog) design

The post-(c) roadmap sketched direction 2 as:
- (c) generative-replay loop runs per user query (within-turn = within-episode binding)
- Each turn's engram tagged (D.14)
- dlpfc_wm holds dialog state across turns
- Replay during between-turn pauses uses prior-turn engrams as context for current-turn predictions
- Phase 1.3 SWR consolidation transfers stable dialog content into the cortical schema

The biology found above directly supports this design:
- Theta sequences at within-turn scale (the (c) loop's gamma-slot per-iteration)
- Behavioral sequences at across-turn scale (dlpfc_wm + consolidation)
- Cross-episode binding via MTL + PFC interaction
- Persistent parahippocampal spiking for the dialog-frame-holding role
- Hippocampal-cortical dialog for schema extraction

## Pre-registered test for direction 2 (sketch; refine post-(c)-result)

3-turn dialog where each turn's partial cue includes a back-reference to a prior turn's bound concepts:
- Turn 1: "remember (apple, red)" → engram-tag the pair
- Turn 2: "remember (banana, yellow)" → engram-tag
- Turn 3: "what colour was the apple?" → partial cue references turn-1; loop should retrieve via prior-turn engram + decode "red"

PASS iff multi-seed-mean back-reference completion accuracy ≥ 0.80 at multi-seed.

## Implementation cost estimate (per post-(c) roadmap)

~1-2 weeks subagent-driven build + ~6-12 hr decisive GPU run. Substantial scope; requires careful design of the across-turn engram retrieval mechanism.

## Sources

WebSearch query "multi-turn dialog working memory hippocampus prefrontal cortex extended timescale episodic binding biology" (2026-05-24); primary references:
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7993991/ — "Multiple time-scales of decision-making in the hippocampus and prefrontal cortex"
- https://www.jneurosci.org/content/30/44/14676 — "Flexible Memories: Differential Roles for Medial Temporal Lobe and Prefrontal Cortex in Cross-Episode Binding"
- https://www.cell.com/neuron/fulltext/S0896-6273(17)30840-1 — "Consolidation Promotes the Emergence of Representational Overlap in the Hippocampus and Medial Prefrontal Cortex"
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4372545/ — "Hippocampal subfield and medial temporal cortical persistent activity during working memory reflects ongoing encoding"

## Files

- This reference doc: `research/findings/2026-05-24-biology-reference-multi-turn-dialog-multiple-timescales-cross-episode-binding.md`
- Companion biology refs:
  - `research/findings/2026-05-24-Schwartenbeck-2023-biology-reference-for-c-generative-replay-three-stage-iterative-refinement.md`
  - `research/findings/2026-05-24-biology-references-PFC-SWR-replay-30-50ms-window-selective-trajectory-encoding.md`
- Post-(c) roadmap: `docs/plans/2026-05-24-post-c-direction-roadmap-multi-turn-and-beyond.md`
- Direction 2 specifics in the roadmap

## Standing constraints (reference research only)

- No code written
- No protected/frozen/moat modified
- No capability claim
- Ready for direction 2 dispatch when (c) decisive validates the (c) substrate
