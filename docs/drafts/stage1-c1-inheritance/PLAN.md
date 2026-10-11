# Stage-1 C1 (category held-out inheritance): the path to exit attempt 2 (revised, Claude review 2026-10-06)

**Update 2026-10-06 (RULING c1-hop-first, owner):** the hop-attenuation fix (section 5.1) lands BEFORE the flip.
It is built on `claude-review/c1-hop` (`BRAIN_CATEGORY_INHERIT_HOP`, default off; organ dev 16/16 on dev seeds 1-16;
`research/findings/2026-10-06-c1-hop-attenuation-bridging-member.md`). Steps A2-A3 below are new; B now runs with
the fix on and the bridge in its bar; the C1 gate PASS is re-measured with the fix (A3) before the flip.

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
The bar-sanity sweep confirmed it (`docs/drafts/stage1-exit/bar_sanity_record_s1_c1_registered.json`: UNREACHABLE for this route). It is step D below (RULING stage1-c1-lesion).

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
(`research/runners/c1_flip_dev_check.py`, tests, bar-sanity record), the `lesion_inherit_pathway` patch for A2 (merged with `claude-review/stage-bar-sanity-s1`, whose C1 record is refreshed here). Hermes commits the 52 untracked gate shards in
the primary checkout (HANDOFF step 1); they are not on this branch because the merge would collide with them.

**A2. The hop fix (`claude-review/c1-hop`, Claude).** Hop attenuation in the organ behind
`BRAIN_CATEGORY_INHERIT_HOP` (default off; off is byte-identical): the frame -> frame pathway transmits at 0.5 of its
weight (encoding and read) and the read lasts one 30-step retrieval cycle. Organ dev 16/16 on its own bar
(`bar_sanity_record_hop_dev.json`): the bat case's canary swim abstains in both fact orders, the canary's eat keeps
worm, the controls hold. Section 5.1 has the diagnosis.

**A3. C1 gate re-measure with the fix (Hermes, pool, gate seeds; HANDOFF step 3b).** The 09-30 C1 PASS 6/6 was
measured without the fix, so it does not cover the configuration the flip ships. Prereg section 13, amendment
2026-10-06: the same 54 lines (9 arms x 6 gate seeds) with `BRAIN_CATEGORY_INHERIT_HOP=1`, into `gate_hop/`, scored
by `stage1_c1_category --aggregate ... --hop` (same bars, plus every arm must record the hop). The C1 PASS that the
flip and attempt 2 rely on is the gate_hop verdict; if it is not PASS, the flip is off and the fix is reworked.

**B. Flip dev check (Hermes, pool, dev seeds 1-8; HANDOFF steps 2-5), WITH the fix.** On the merged stack, through the
real handler: every Stage-1 group whose turns can reach the wiring (cat_disc, cat_told, word, lived, val_novel) and
the bridge group `cat_bridge` (the C1 context plus a bat that eats the bug and flies in the sky), with
`BRAIN_CATEGORY_INHERIT=1` vs `0`, and `BRAIN_CATEGORY_INHERIT_HOP=1` on every cell, both sides. 96 jobs (~30 min
each). Dev bar (bar-sanity OK, n 1000; `bar_sanity_record_flipdev.json`, re-run for this change): per seed, cat_disc
flag-on names worm, and reply content is equal on every other turn of every group, with the only allowed changes a
flagged guess `worm` on the canary probes (cat_disc, cat_told, cat_bridge) and a flagged guess `lake` on the bat's
swim (one hop through its own told `eat bug`; reported); `what does the canary swim` and `what does the robin swim`
in the bridge session must not change, and the bridge session's `what does the canary eat` must be the flagged worm
guess. PASS iff 8/8. A cell without the hop env, or an organ read without `hop:
true`, is UNDEFINED. This measures the merged stack, flip specificity outside the C1 script, and the fix through the
handler.

**Amendment B-1 (2026-10-11, Claude; RULING c1-flipdev-B1): the "flag engaged" precondition is replaced.**
Written AFTER the rerun at DEV_SHA 2c2d55952 read UNDEFINED 0/8, and said so here. The old precondition (every
flag-on turn carries the `category_inherit` trace) could not be met by construction: chain, open-ended, episodic,
feel and bare-word turns are answered by endpoints that never call `chat.gate`, so they can never carry the trace,
and no seed of any group with such turns could ever be defined (the rerun: 32 of 48 ON cells, all for that reason).
A precondition no run can meet has no discriminating power. It was meant to prove the wiring was APPLIED, and
step B is a specificity check: whether the flag changes replies it should not.

New precondition, per ON cell: the cell records `BRAIN_CATEGORY_INHERIT=1` and `BRAIN_CATEGORY_INHERIT_HOP=1`; at
least one turn carries the trace (the wiring is installed in that session); every organ read records `hop: true`;
and, in groups that ask it, `p_canary_eat` carries an organ read with `hop: true`. The scorer reports untraced
turns per group (count and labels); they are never silently passed. OFF cells: no trace on any turn, unchanged.

Unchanged: the dev bar above, word for word. Every turn of every group is still compared ON vs OFF on reply
content, and an untraced turn whose reply changes FAILS the seed exactly as before. The rescore uses the committed
rerun cells (`research/findings/raw/_stage1_c1_flipdev_rerun_2c2d55952/`); no new run. A seed defined under B-1 but
failing the bar is a FAIL, not an UNDEFINED. The scorer change re-runs `bar_sanity_record_flipdev.json` in its commit.

**C. Flip (Claude, after A3 PASS and B PASS).** Claude reads both scores and the bridge report and decides the flip
(section 4). On a flip branch: `_CATEGORY_INHERIT_DEFAULT_ON = True` AND the hop option on by default (unset
`BRAIN_CATEGORY_INHERIT_HOP` -> on; `=0` restores the 09-30 organ), the explicit `BRAIN_CATEGORY_INHERIT=0` path stays
byte-identical to today's default, tests updated. No further C1 gate run at the flip revision (A3 measured the
configuration; attempt 2 measures it again at the gate seeds).

**D. Amendment A2, C1 section (Claude, before any attempt-2 data; RULING stage1-c1-lesion).** Fold
`ATTEMPT2-C1-AMENDMENT-TEXT.md` into A2 (with the C4 control of RULING stage1-sync): for `cat_disc` and `cat_told` the
counted lesion becomes `lesion_inherit_pathway` (one name for the arm and the patch it installs: every organ read with
both learned pathways cut; applied = calls > 0; the read trace also records `lesion: all`, `lesion_held: true`), and
`lesion_draw` becomes informational. The bar is unchanged. The patch is already on main with a CPU test; A2 only swaps
the GROUPS entry. A2's `bar-sanity:` line cites `docs/drafts/stage1-exit/bar_sanity_record_s1_c1_fixed_pathway_lesion.json`
(OK; re-run in A2's commit because A2 edits a scorer file; that commit also renames the spec's arm key from
`lesion_pathway` to `lesion_inherit_pathway`). That spec's `pathway` brain is this route as measured: flagged worm on
the direct probe, generative abstains, the pathway lesion abstains, `lesion_draw` changes nothing, permuted names bug.
Attempt 2 runs with the hop option on (the flipped default). No different route, so no new spec.

**E. Attempt 2** at the flip revision, all groups, per RULING stage1-calendar (no later than the Stage-2 window
11-03..11-10, co-scheduled with it).

## 4. The flip decision (what C weighs)

- **Specificity** (B): any content change outside the canary probes is a STOP, not a judgment call.
- **The bridging member** (B, now in the bar): with the fix, `what does the canary swim` and `what does the robin
  swim` (two category hops through the bat) must stay abstentions. The one-hop case stays: `what does the bat swim`
  becomes a flagged guess `lake` at organ level (dev seeds 1-8: 8/8), because the bat's own told `eat bug` is shared with the fish;
  that is the C1 inference itself, not a hop. Reported in the flip commit; whether one shared feature should be
  enough is an owner item (a category-competition / cancellation rung, section 5.6, would refine it).
- **Capacity:** 24 member and 24 frame slots; past that the organ stops encoding and inheritance degrades to
  abstaining. Safe (no false answers); reported.

## 5. Strengthening, in parallel and off the attempt-2 path (no defer: each is a capability, not a footnote)

1. **Hop attenuation (the bridging member): BUILT, `claude-review/c1-hop`.** Diagnosis (dev seeds 1-8): the bridge is
   written at ENCODING, not only spread at the read. While the bat's `fly sky` is rehearsed, `swim lake` is
   reactivated two hops away (0.05-0.06 vs 0.09 for a first-order frame) and the LTP-only rule writes a direct
   `fly sky -> swim lake` link (5.7-9.9) that no member carried; if the canary comes after the bat, its own rehearsal
   also writes `canary -> swim lake` (3.7-6.9), and 54-99% of the lake read survives a read-only lateral cut. At the
   read, activity walks one lateral hop per ~10 steps, so the 60-step window reaches two hops. A divisive feedback
   pool on the frame layer was tried first and scaled every frame alike (lake 0.035 vs worm 0.042, seed 1). The
   built option: the frame -> frame pathway transmits at 0.5 (recurrent associative synapses carry less current than
   the afferent ones: one attenuated hop still reactivates, a second is attenuated again) and the read lasts one
   30-step retrieval cycle. Both were chosen on dev seeds 1-16 (verify-go: a 35-step read passed seeds 1-8 and failed
   fresh seed 9), so the gate seeds of A3 are the held-out test. Residuals: the 30-step cycle is a host timing constant (as the 60 was); an in-substrate cycle terminator
   (theta-paced or GABA_B feedback inhibition) is the next rung.
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
6. **One shared feature is enough to inherit (the bat swims).** With the hop fix, a member that shares one told frame
   with a category still inherits that category's other frames (`bat swim -> lake`, organ 8/8 at dev seeds 1-8), the same inference
   as C1. A member already carrying frames of a second category could hold the inheritance back (category
   competition, the 07-08 cancellation finding's per-member override). Owner item before building.
