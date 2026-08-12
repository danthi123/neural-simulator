---
type: biology
id: systems-consolidation
mechanism: Systems consolidation -- the hippocampus binds an episode, replay transfers it, and the CORTEX is the long-term repository
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md
current_status: "HIPPO-LESIONED recall of a cortically-stored fact 18/18 (3/3 on 6/6 seeds) vs chance 6/18; 6/6 perfect seeds vs scramble-control 0/6, Fisher exact p = 0.00108. BOTH ingredients necessary: subtractive normalization on the BTSP write AND a read-only read -- each ALONE gives chance (7/18, 7/18, 8/18). IN ISOLATION ONLY: word->pool binding is unbuilt, so pools are cued directly, and the read-only read is a HOST intervention."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "serve as the long-term repository of the separate ele"
    note: "the defining claim -- the CORTICAL regions, not the hippocampus, hold the elements long-term"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "circuit for consolidation and retrieval involving sev"
    note: "consolidation is a DISTRIBUTED circuit with the hippocampus binding associations during encoding AND retrieval -- not a one-way copy"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "could be replayed multiple times, assisting with the consolidation"
    note: "the transport: repeated sharp-wave replay of one episode is what carries it, over a protracted molecular timecourse"
implemented_by:
  - research/runners/nmda_compositional_consolidation.py
  - research/runners/_consol_cortical_store_probe.py
findings:
  - research/findings/2026-07-25-CRITICAL-apical-R-333x-miscalibration-invalidates-consolidation-operating-point.md
  - research/findings/2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md
---

# The cortex is the repository; replay is the transport

**The claim the code must respect.** Kandel: "[t]he cortical regions serve as the long-term repository of the
separate elements of information that constitute a memory," reached through "a distributed circuit for
consolidation and retrieval involving several brain regions, with the hippocampus playing an essential role in the
binding of associations during both encoding and retrieval." Buzsáki supplies the transport: "neuronal
representations of a single episode could be replayed multiple times, assisting with the consolidation process,"
because the molecular machinery of synaptic change runs far slower than the episode did.

Two things follow that this project got wrong at least twice:
- **The test of consolidation is recall with the hippocampus REMOVED.** If the hippocampus is still in the loop,
  a correct answer is evidence of hippocampal retrieval, not of a cortical repository.
- **Encoding and retrieval are different states.** The hippocampus participates in both, but a read that also
  writes is not a read.

## The current status, and its three limits

The capability is **6-seed established**: hippo-lesioned recall of a cortically-stored fact at **18/18 (3/3 on 6/6
seeds)** against chance 6/18, with a scramble-teach control that is *causally* diagnostic rather than merely null
— teach a deranged pool→slot mapping and recall follows the derangement (1/18 against the true mapping, **17/18
against the mapping actually taught**). Both ingredients are necessary and jointly sufficient: each alone returns
chance (7/18, 7/18, 8/18).

**The limits are not optional reading.** (a) The read-only read is a **host intervention** (`--freeze-read`), a
tracked shortcut whose named biologization is SPEAR/Hasselmo ACh encode-vs-retrieve via the existing
`plasticity_gate` neuromodulator target — it is not closed. (b) This is consolidation **in isolation**: word→pool
binding is unbuilt, so concept pools are cued directly by teacher current. (c) Word→pool binding has never worked
above chance on any configuration reproducible today; the old 87.5% baseline depended on deleted cached substrates
and is retired.

## What this entry cannot catch — and it is the exact defect that caused the retraction

**No `constraints_config`, and that is a gate limitation, not a judgement.** The single highest-value constraint
here is that the recall read must be plasticity-frozen. In the code that is `--freeze-read`, an
`action="store_true"` flag with no numeric default. `biology_check --config` matches only numeric defaults
(`default=<number>`, or a `key = <number>` assignment), so a boolean flag is invisible to it: declaring
`freeze_read` here would produce "constrains 'freeze_read' but it was not found in the runner" on a runner that
*does* have it. A live read cost this arc a retraction — the store was being overwritten while being read (weight
drift **+1.28–1.41** live vs **+0.000000** frozen) — and the checker as written cannot prevent a recurrence.

⚠️ **Read the current_finding's header, not its body.** That document is an append-log of six reversals; only its
"READ FIRST — CURRENT STATE" block reflects the final state. Its own frontmatter declares `mechanism:
consolidation`, a slug whose sole `status: live` finding is from 2026-05-10 and no longer current; this entry
deliberately uses the distinct id `systems-consolidation` rather than assert a live-ness it cannot fix from here.
