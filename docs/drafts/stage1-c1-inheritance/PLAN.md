# Stage-1 C1 (category held-out inheritance): the path to exit attempt 2 (revised, Claude review 2026-10-06)

**This replaces the 2026-10-04 version of this file** (commit c10a74767, a cloud Claude session). That version
proposed building a rung-3c member -> context pathway on the strict route's recall miss. The same build already
existed on `research/stage1-c1-category` (2026-09-30: organ, default-off wiring, prereg, 54-line gate), and its gate
had landed 52 of 54 shards on 10-04/05 without being scored. The 10-04 plan was written without checking that
branch. Its build step is therefore retired; its controls survive in the 09-30 prereg.

## 1. Where C1 stands

- **Stage-1 attempt 1 (production defaults): FAIL**, capability absent 6/6. Diagnosis from the data
  (`research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md`): routing and retrieval worked; the
  category link was never encoded (each fact is an isolated block; the strict route is an exact match), and the
  generative route never proposed. `cat_told` also lost its premise at encoding (`the bird eats the worm` is never
  acquired: the comprehension monitor asks which is the patient).
- **The 09-30 wiring, gate seeds, flag on: PASS 6/6, narrowed** (same finding). Organ worm 0.108-0.126 vs bug 0.0;
  lesion abstains 6/6, reproduced; permuted reads bug 6/6. Caveats: measured on the 09-30 stack and numpy; flip safety
  beyond the C1 script unmeasured through the handler; a bridging member over-generalizes at organ level; host
  admission and a standalone organ.

**Will the registered C1 bar plausibly pass at attempt 2?** Yes for correctness, permutation and the null, if the
flipped stack reproduces the 09-30 numbers. **No for the lesion clause as frozen**: the Stage-1 prereg counts
`lesion_draw`, which does not engage on this route, so under Amendment A1's engagement rule C1 would read UNDEFINED.
The 10-04 plan did not address this; it is step D below.

## 2. The method stays: rung-3c with integrative encoding (no new mechanism needed for the bar)

Biology. When a new episode overlaps stored ones, the hippocampus reactivates them while encoding it, so an
inferential link is written at encoding time (integrative encoding; Shohamy & Wagner 2008; Zeithamova et al. 2012).
The organ does exactly this: member -> frame and frame -> frame synapses learn by on-bridge rate-Hebbian during the
rehearsal of each heard fact, and the canary -> `eat worm` link is written while the canary's own fact is rehearsed
(the read-only lateral lesion keeps worm 5/5; the learning lesion removes it 5/5). One brain-based mechanism, no
host category, no host fact lookup. Its residuals are named in section 5 and are not on the attempt-2 path.

## 3. Steps to attempt 2 (in order)

**A. Land the record (this branch).** Merge of `research/stage1-c1-category` (the organ, wiring, runner, prereg,
tests; default off), the scored aggregate, the RESULT finding, this plan, the flip dev check
(`research/runners/c1_flip_dev_check.py`, tests, bar-sanity record). Hermes commits the 52 untracked gate shards in
the primary checkout (HANDOFF step 1); they are not on this branch because the merge would collide with them.

**B. Flip dev check (Hermes, pool, dev seeds 1-8; HANDOFF steps 2-5).** On the merged stack, through the real
handler: every Stage-1 group whose turns can reach the wiring (cat_disc, cat_told, word, lived, val_novel) with
`BRAIN_CATEGORY_INHERIT=1` vs `0`, plus a REPORTED bridging probe (`cat_bridge`: the C1 context plus a bat that eats
the bug and flies in the sky). 96 jobs (~30 min each). Dev bar (bar-sanity OK: perfect 1.000, every null 0.000, n
1000; `bar_sanity_record_flipdev.json`): per seed, cat_disc flag-on names worm, and reply content is equal on every
other turn of every group, with the only allowed change a flagged worm guess on the canary probes; PASS iff 8/8.
This measures the two things the gate did not: the merged stack, and flip specificity outside the C1 script.

**C. Flip (Claude, after B PASS).** Claude reads B's score and the bridging report and decides the flip (section 4).
On a flip branch: `_CATEGORY_INHERIT_DEFAULT_ON = True`, the explicit `BRAIN_CATEGORY_INHERIT=0` path stays
byte-identical to today's default, tests updated, and the wiring's own C1 gate re-run at the flip revision is NOT
needed (attempt 2 measures it at the gate seeds).

**D. Attempt-2 C1 amendment (Claude, before any attempt-2 data).** Fold
`ATTEMPT2-C1-AMENDMENT-TEXT.md` into the attempt-2 amendment (with the C4 control of RULING stage1-sync): for
`cat_disc` and `cat_told` the counted lesion becomes `lesion_inherit` (`BRAIN_CATEGORY_INHERIT_LESION=1`, engaged
only if the direct-probe read records `lesion: all` and `lesion_held: true`), `lesion_draw` becomes informational,
and the runner (`research/runners/stage1_exit_gate.py`) gains that arm and its engagement check. The bar is unchanged.

**E. Attempt 2** at the flip revision, all groups, per RULING stage1-calendar (no later than the Stage-2 window
11-03..11-10, co-scheduled with it).

## 4. The flip decision (what C weighs)

- **Specificity** (B): any content change outside the canary probes is a STOP, not a judgment call.
- **The bridging member** (B, reported): the expected result, from organ dev, is that `what does the canary swim`
  becomes a flagged guess `lake` in the bridge session. That is a wrong guess through a chain (canary -> fly sky ->
  bat -> eat bug -> trout -> swim lake). It is flagged as a guess, never a fact, and over-extension is normal in an
  infant learner, but a default-on faculty that volunteers it in richer sessions trades abstentions for confident-
  sounding errors. Claude's default ruling: flip with the guess flagged and the over-generalization rate reported in
  the flip commit, and open the hop-attenuation rung (section 5) the same day; the owner can override (see the
  report's owner items).
- **Capacity:** 24 member and 24 frame slots; past that the organ stops encoding and inheritance degrades to
  abstaining. Safe (no false answers); reported.

## 5. Strengthening, in parallel and off the attempt-2 path (no defer: each is a capability, not a footnote)

1. **Hop attenuation (the bridging member).** Ask first what the real system runs alongside the read: feedforward
   inhibition scaled by the cue's own drive, so activity two lateral hops away stays subthreshold. Build it as a
   divisive-inhibition pool on the organ's frame layer, default off, dev seeds 1-8 at organ level with the `bat`
   case: the bar is canary swim abstains 8/8 while the C1 inheritance still reads worm 8/8.
2. **The host pick inside the read.** The floor plus 1.25x margin is a host comparison of spike counts. The spiking
   commit stage (`docs/drafts/spiking-commit/PLAN.md`: exclusive gate + self-holding latch on the same bridge, read
   by persistence with a threshold-invariance bar) is the replacement. Order per RULING spiking-commit-design: after
   A1 step O passes and A2 is written, the C1 read is a candidate A4. Admission by the asked verb's string stays host
   until a spiking relation cue exists.
3. **One substrate.** The organ is a standalone bridge. Next rung: co-locate its member and frame populations on
   the one-brain merged pool with a cross-edge to the composer (non-negotiable 2).
4. **Told inheritance (`cat_told`).** First the acquisition defect (`the bird eats the worm` is never stored), then
   composition of `isa` with the class property (EMERGE-26/EMERGE-30 are the record). Diagnostic only for Stage 1.
5. **The generative route.** `what might the canary eat` never proposes; the plausibility gate could read the organ's
   frame rates as its likelihood. Optional for the bar (either route counts).
