# PRE-REGISTRATION — the unsupervised stream cortex at 787-concept scale ("given enough training, does emergent structure hold?")

**2026-07-17. Written BEFORE the results land (the run is on the GPU as this is committed) so the gate cannot be
p-hacked post-hoc — per the silent-failure discipline (pre-register the control + the pass bar).**

## Why this test exists (the decided direction)

The learning-rule frontier is closed: supervised deep-credit-on-spikes is PARKED (feedforward e-prop NOT-GO 2026-07-17,
NP retired 2026-07-13, the graded-readout escape hatch does not obviously work 2026-07-17), and the record's own verdict
is **commit to the UNSUPERVISED on-spike stream cortex** — the HTM + committed-BDSP `fused_htm_permanence_update` cortex
that learns representations from a stream WITHOUT supervised global-loss deep credit. The emergence bar's core claim is
literally *"LLM-like conversation GIVEN ENOUGH TRAINING"* — i.e. capability from LEARNING, not hand-building. The
cleanest test of that claim is: **does the emergent structure (learned codes, retention, no-confab) HOLD as the brain
learns HUNDREDS of concepts from a stream, or does it plateau/collapse?** The develop loop's corpus-curriculum path is
the tool; every prior develop run (week1/month1) used the hardcoded ~24-concept demo schedule. This is ~33× that.

## The run (frozen)

`develop_run --corpus-curriculum --brain-npz brain_curriculum_vocab_regen.npz --n-days 40 --seed 42 --root
bridges/developed/scale787` (`SIM_BACKEND=cupy`, RTX 3090). Vocab: **787 concepts** (regenerated high-freq-first from
the TinyStories SVO facts + corpus, `_regen_curriculum_vocab.py`; the E:-wiped original was only a vocab-ordering
source). Curriculum: 24 concepts/day × 33 days to introduce all 787, + margin. 264 corpus SVO facts asserted as the
concepts arrive. Per-day battery: recall / held-out / retain / chain / moat_fa / corr(M,C). develop_D = 128 (default).

## PRE-REGISTERED metrics + pass bars (frozen before results)

**PRIMARY — the emergence claim (D-INDEPENDENT, the load-bearing one):**
- **`corr(M,C)` (StreamCortex code-learning quality) stays ≥ 0.70 at the FINAL vocab (day ~33, 787 concepts), with no
  collapse across the curve.** This is "structure emerges from experience AT SCALE" — the stream cortex keeps learning
  faithful concept codes from co-occurrence as the vocabulary grows 24 → 787. Day-0 baseline was +0.89 @ 24 concepts.
  GO = holds ≥ 0.70 at 787; BOUNDARY = degrades below 0.70 (⇒ the stream-cortex learning itself has a scale limit — a
  genuine finding, launches the next-mechanism search, NOT accepted as a wall).

**SECONDARY — retention (the artificial-life "no catastrophic forgetting" claim at scale):**
- **`retain` ≥ 0.80 across the curve** (old facts still recalled after subsequent new-concept days). Day-0 was 1.00.
  A decline as vocab grows = the interference/capacity signal to characterize.

**SECONDARY — the no-confab moat at scale:**
- **`moat_fa` (false-accepts on never-taught cues) stays low** as vocab grows. Frozen interpretation (per
  `feedback_moat_not_hard_lossy_memory_ok` + the CLAUDE.md 2026-06-15 note): a FEW tail false-accepts on the
  lowest-fidelity concepts are the CODE-FIDELITY cost (the lever is more stream / wider familiarity gap), NOT a
  moat-MECHANISM failure. GO = moat_fa stays O(single digits) and does not blow up monotonically with vocab; the
  reported number is the honest per-scale confabulation rate.

**CHARACTERIZATION (NOT a pass/fail — a curve to report):**
- **`recall` vs vocab is the composer's develop_D=128 FHRR capacity curve.** Day-0 was 0.67 @ 24 concepts / 8 facts.
  As the 264 facts accumulate, recall is EXPECTED to degrade — the √D/M capacity relation (CLAUDE.md: √D/M validated to
  320 concepts at D=2048; at D=128, hundreds of superposed facts exceed capacity). This is a KNOWN, characterizable
  limit whose lever is a bigger `develop_D`, **not** a stream-cortex or emergence failure. Report recall(vocab); if it
  is the binding limit, the pre-registered follow-on is a bigger-`develop_D` re-run (the design's own knowledge-scaling
  knob — `use_multiturn=False` exists precisely to free VRAM for it).

## Anti-cheats / validity guards (frozen)

1. **The primary signal must be the STREAM CORTEX, not the composer.** `corr(M,C)` is read directly (a per-day column),
   independent of the composer's recall — so the D=128 capacity wall CANNOT confound the primary emergence claim. This
   is the whole reason the primary bar is corr(M,C), not recall.
2. **Retention is a genuine hold-out in time** (old facts tested after new-concept days), not a same-day re-read.
3. **Do not lift a headline from a degrading curve** — if recall collapses at D=128, that is the composer-capacity
   characterization, reported AS SUCH, and the emergence verdict rests on corr(M,C)+retain+moat, not on recall.
4. **The seed controls the substrate** (the 2026-07-17 seed fix): the develop loop seeds `cfg.seed`; a re-run at the
   same seed is reproducible.

## What each outcome means (decided in advance, so the verdict is not motivated reasoning)

- **corr(M,C) holds ≥ 0.70 at 787 + retain/moat hold:** GO — the emergent stream cortex keeps learning structure from
  experience at hundreds-of-concepts scale; the "given enough training" thesis holds at this scale; recall is a
  separate, D-tunable composer knob. Next: the bigger-`develop_D` re-run to lift recall, then a richer corpus
  (wikitext103, which covers the 203 dropped concepts) to push vocab further.
- **corr(M,C) degrades below 0.70:** the STREAM-CORTEX LEARNING has a scale limit — a genuine, un-guessed boundary
  (undiscovered mechanism), launching a research gate on WHY (co-occurrence saturation? hub interference? homeostasis?)
  — the honest first-class deliverable, not a wall.
- **retain collapses but corr holds:** the interference/consolidation frontier (the sequence-completing-CA3 lever, ROADMAP §9 #4) is the named next mechanism.

Result → `2026-07-17-stream-cortex-787-concept-scale-test-RESULT.md` (to be written from the per-day log, against THIS
frozen gate).
