---
type: biology
id: joint-attention-gaze-following
mechanism: An other-attention-schema in the STS-temporoparietal junction reads a partner's eye-gaze / biological-motion cue and infers WHICH target the partner is attending to; aligning one's own attentional spotlight to that inferred target is joint (mutual) attention, the earliest theory-of-mind precursor (Stage-0), developmentally dissociable from and preceding full mentalizing.
status: established
last_verified: 2026-08-26
current_finding: research/runners/_joint_attention_derisk.py
current_status: "De-risk built: a spiking direction-ring gaze code (STS eye-gaze/biological-motion input) read by direction-tuned STS-TPJ object cells whose per-object drive scores a reused spiking one-of-K attentional spotlight (build_fswta_score_bridge/fswta_drive). The spotlight winner tracks the partner's INFERRED attended object above chance; lesion of the STS-TPJ read collapses it to chance; scrambling the partner gaze collapses it to chance; a layout-blind (gaze-only) baseline is at chance, so the alignment is an inference over the current object layout, not a copy of a transmitted coordinate. Smoke VERDICT pasted; 6-seed pending."
sources:
  # The Kandel "kandel" corpus (Principles of Neural Science, Ch. 62, Autism) that the project's scientific RAG
  # indexes. The on-disk PDF-derived full-book.txt is two-column OCR-interleaved so multi-word phrases are not
  # contiguous there; the RAG's de-interleaved index (docstore.json) is the readable form I READ the passage from,
  # and the anchors below resolve verbatim in it.
  - path: /home/dant123/Projects/rag_index/llamaindex_full/docstore.json
    anchor: "eye gaze and biological"
    note: "Kandel Fig. 62-4 'The mentalizing system of the brain': the SECOND of four mentalizing components, 'in the temporoparietal region of the superior temporal lobe, is known to be activated by EYE GAZE and BIOLOGICAL MOTION' -- the STS-temporoparietal junction, the perceptual front end that reads another agent's direction of attention. Lesions here (left hemisphere) impair the mentalizing task. This is the region my other-attention-schema models: gaze/biological-motion in -> inferred attention target out."
  - path: /home/dant123/Projects/rag_index/llamaindex_full/docstore.json
    anchor: "mutual attention"
    note: "Same chapter: 'mutual attention normally appears toward the end of the first year when signs of mentalizing are still sparse', and 'the absence of preferential attention to social stimuli and mutual attention are widely acknowledged as early signs of ASD'. Joint/mutual attention is DEVELOPMENTALLY DISSOCIABLE from and PRECEDES full mentalizing -- grounds the spec's 'theory-of-mind Stage-0': align-to-inferred-partner-target is a self-contained precursor faculty, not false-belief mentalizing (the mPFC + Sally-Anne register, W3)."
constraints_config:
  # No numeric substrate default is load-bearing here -- the property that matters is STRUCTURAL:
  # (1) the partner transmits only a continuous gaze DIRECTION (+ noise), never the target's identity/coordinate;
  # (2) object angular positions are RANDOMIZED per trial, so a fixed gaze->object map cannot solve it;
  # (3) the inferred target is decoded by a direction-tuned SYNAPTIC read (STS-TPJ object cells) of the SPIKING
  #     gaze-direction population -- an inference combining gaze with the current layout, not a copy.
  # These are enforced by the runner's anti-cheats (scramble, lesion, layout-blind baseline), not by a cfg value.
  {}
implemented_by:
  - research/runners/_joint_attention_derisk.py
findings: []
---

# Joint attention = align your spotlight to the partner's INFERRED attention target (STS-TPJ, Stage-0 ToM)

**The claim the code must respect.** Kandel's mentalizing system (Fig. 62-4) has four components; the second,
"in the temporoparietal region of the superior temporal lobe, is known to be activated by eye gaze and biological
motion." This STS-temporoparietal junction is the perceptual front end that reads ANOTHER agent's direction of
attention from their eyes and body motion. Damage here impairs mentalizing. Developmentally, "mutual attention
normally appears toward the end of the first year when signs of mentalizing are still sparse" -- so aligning to a
partner's attended target (joint attention) is a DISSOCIABLE, EARLIER faculty than full false-belief mentalizing.
Its absence is an early sign of ASD.

**What this makes the mechanism.** An *other-attention-schema*: a population that (a) encodes the partner's gaze
DIRECTION (STS eye-gaze / biological-motion code), (b) infers WHICH of the currently-present objects lies along
that direction (a direction-tuned read over the current layout), and (c) drives the agent's own attentional
spotlight to that inferred object. The inference is load-bearing: the partner never emits the target's identity,
only a noisy direction, and the object layout changes every trial, so the answer exists only as the geometric
conjunction of gaze-with-layout -- exactly what a copy of a visible coordinate cannot supply.

**Why it is Stage-0, not full ToM.** This aligns attention to an INFERRED target; it does not model the partner's
BELIEF (that is W3 / the Sally-Anne false-belief register, mPFC) or FEELING (that is W5 / affective ToM). It is the
perceptual-social precursor those build on, matching the developmental ordering in the source.
