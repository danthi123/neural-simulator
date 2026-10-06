---
type: finding
status: live
date: 2026-10-06
lane: integration-first (Stage-1 exit criterion C1; RULING c1-hop-first, owner 2026-10-06)
mechanism: hop attenuation in the category-inheritance organ (BRAIN_CATEGORY_INHERIT_HOP, default off): the learned frame->frame pathway transmits at 0.5 of its weight at encoding and at the read, and the read lasts one 30-step retrieval cycle, so a bridging member no longer carries a property two category hops away
seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
runner: research/runners/category_inherit_organ_dev.py
artifacts:
  - research/findings/raw/_stage1_c1_category/dev/organ_dev_hop_seeds1-16.json
  - research/findings/raw/_stage1_c1_category/dev/organ_dev_hop_seeds1-8.json
  - research/findings/raw/_stage1_c1_category/dev/hop_scratch_reads.json
  - research/findings/raw/_stage1_c1_category/dev/organ_dev_nohop_baseline_seeds1-8.json
  - docs/drafts/stage1-c1-inheritance/bar_sanity_record_hop_dev.json
---

# C1 over-extension through a bridging member: diagnosed, and fixed at organ level by hop attenuation (dev 16/16)

**Scope.** Organ level only (the `CategoryInheritOrgan` fed the Stage-1 C1 facts directly, numpy, CPU), dev seeds
1-16 only, no gate seed. Both parameters were chosen on these same dev seeds, so this is in-sample. This is a dev result and a design record, not a generalization claim: the gate seeds and the
real handler are measured next (prereg 2026-09-30 section 13 amendment; `docs/drafts/stage1-c1-inheritance/HANDOFF.md`).

## The defect

After the C1 facts (robin, sparrow and finch eat the worm and fly in the sky; trout and salmon eat the bug and swim in
the lake), add a bat that eats the bug and flies in the sky, then the canary that flies in the sky. The 09-30 organ
answers `what does the canary swim` with lake on 8 of 8 dev seeds (prereg section 12), in either fact order.

## Diagnosis: the bridge is written at encoding, and the read walks the merged graph

Measured with the 09-30 organ on dev seeds 1-8. The encoding and read-dynamics numbers below are scratch reads of
the organ's own weight audit and spike counts (the scripts drive only the organ's public `observe` / `read` / `infer`
/ `weight_audit`), saved after the fact in `research/findings/raw/_stage1_c1_category/dev/hop_scratch_reads.json`
(blocks `diag_0930_bridge`, `encoding_frame_rates_seed1`, `read_time_course_seed1_bridge`; ranges in
`summary_ranges`, computed from those blocks); the decisions are reproduced by the battery files above.

1. **Encoding writes links between frames no member carried together.** The rule is LTP-only (no decay, no
   depression) and the frame->frame pathway is all-excitatory with no inhibition. While the bat's second fact
   (`fly sky`) is rehearsed, its first fact's frame (`eat bug`, 0.09 spikes/neuron/step) and that frame's associate
   `swim lake` (0.05-0.06, two hops from the heard frame) fire too (seed 1), so a direct `fly sky -> swim lake` link
   grows to 5.688-9.865 (8/8 seeds, both orders) from the 0.05 floor. No member was ever told both.
2. **When the canary comes after the bat, its own rehearsal compiles the chain into its member links.** `canary ->
   swim lake` reaches 3.694-6.853 and `canary -> eat bug` 4.202-8.1; cutting the frame->frame pathway at read time leaves
   54-99% of the canary's lake read standing (8/8). When the canary comes first, those member links stay at the floor
   and the lake read (0.0758-0.0958) disappears entirely under the same cut: it travels the bat-built lateral link.
3. **The read is a wave, and 60 steps is long enough for two hops.** Spike counts in 10-step bins (seed 1): the
   cue's direct frames fire in steps 10-20, then fall silent, and the activity moves one lateral hop per ~10 steps.
   With the bridge, the robin's own read is dominated by the fish frames after step 20 (its organ read answers `eat`
   with bug, `swim` with lake, 8/8; the strict route hides this for told facts, not for untold ones).

So the over-extension is not only "two-step spread at read time": the encoding step turns a two-hop chain into
one-hop weights, and the long read then also reaches two hops.

## The fix and why it is the biology

The wall-reframe question (what does the real system run alongside this that the organ replaced with a constant?):
the organ's associative (recurrent) frame->frame synapses carried the same current per unit weight as the afferent
member->frame drive, and the read had no end short of 60 ms. In hippocampal and cortical associative memory the
recurrent associative input is weaker than the afferent detonator drive and is further suppressed while new input is
being encoded (muscarinic presynaptic suppression of intrinsic but not afferent fibres; Hasselmo & Bower 1992,
Hasselmo & Schnell 1994; the "runaway synaptic modification" these models were built to prevent is this defect), and
retrieval is chunked into short cycles (one theta half-cycle) rather than run until the activity has crossed the
whole graph.

Built, behind `BRAIN_CATEGORY_INHERIT_HOP=1` (default off): the frame->frame pathway transmits at 0.5 of its weight (a
transmission gate on exactly those synapses, at encoding and at the read), and the read lasts 30 steps. Each lateral
hop multiplies the drive by about one half. One hop still reactivates, so integrative encoding still writes
`canary -> eat worm` (2.2-7.0, was 7.2-9.4). A second hop is attenuated again: the bat-built shortcut link is
attenuated, not removed (`fly sky -> swim lake` after the canary, bat first: 0.05-3.452 on dev seeds 1-8 with the
option, at the 0.05 floor only on seed 5, against 9.103-15.787 without), and the read ends before the second hop arrives in full. Off builds no gate and
reads 60 steps.

How the parameters were chosen (all on dev seeds; recorded so the selection is not mistaken for held-out evidence;
every scratch read named here is a block of `hop_scratch_reads.json`).
Levers tried first, none kept: a divisive feedback inhibitory pool on the frame layer (40 interneurons) scaled every
frame alike (lake 0.035 vs worm 0.0417 on seed 1, bat first, still above the 0.02 floor), and encoding-only suppression at 0.7 /
0.5 / 0.35 with the 60-step read lowered lake from 0.0725 (seed 1, no suppression) to 0.043 / 0.040 (seeds 1 / 2,
gain 0.35), never below the floor. With gain 0.5 at encoding
and read, a 35-step read passed seeds 1-8 (`organ_dev_hop_seeds1-8.json`, at 260bc1e59), but two verify-go skeptics
ran it on fresh dev seeds 9-12 and seed 9 failed (bridge canary swim lake 0.0229, bat first). A 16-seed scratch grid
(gain 0.4 / 0.45 / 0.5 x read 25 / 30 / 35 / 40) then found one setting that passes every dev seed 1-16: gain 0.5 with
a 30-step read (35 and 40 fail seeds 9, 13 and 14 on lake; 25 loses the permuted read on seed 5; lower gains lose the
permuted read). No dev seed is left that the choice did not see: the gate seeds (prereg section 13 re-measure) are
the held-out test.

## Result (organ, dev seeds 1-16, `--hop`, at 5040178c0 on the workstation; hop dev bar PASS 16/16)

Rates are spikes per neuron per step over the read (floor 0.02, margin 1.25x). For the option off on seeds 1-8 see
`organ_dev_nohop_baseline_seeds1-8.json`: lake on every bridge read (0.0642-0.0958), trout fly sky on every read, and the
canary's eat tied on 4 (bat first) and 6 (canary first) of 8 seeds.

| seed | C1 canary eat (worm / bug) | permuted (bug / worm) | read lesion | bat first: canary eat / swim | canary first: canary eat / swim | trout fly (bat first / canary first) |
|---|---|---|---|---|---|---|
| 1 | worm (0.0733 / 0.0000) | bug (0.0617 / 0.0000) | abstain, held | worm 0.0567 / abstain 0.0050 | worm 0.0733 / abstain 0.0033 | abstain 0.0150 / abstain 0.0100 |
| 2 | worm (0.0617 / 0.0000) | bug (0.0533 / 0.0000) | abstain, held | worm 0.0667 / abstain 0.0033 | worm 0.0633 / abstain 0.0017 | abstain 0.0133 / abstain 0.0167 |
| 3 | worm (0.0367 / 0.0000) | bug (0.0383 / 0.0000) | abstain, held | worm 0.0683 / abstain 0.0000 | worm 0.0367 / abstain 0.0000 | abstain 0.0133 / abstain 0.0117 |
| 4 | worm (0.0633 / 0.0000) | bug (0.0467 / 0.0000) | abstain, held | worm 0.0617 / abstain 0.0033 | worm 0.0633 / abstain 0.0000 | abstain 0.0167 / abstain 0.0000 |
| 5 | worm (0.0450 / 0.0000) | bug (0.0250 / 0.0000) | abstain, held | worm 0.0600 / abstain 0.0000 | worm 0.0450 / abstain 0.0000 | abstain 0.0000 / abstain 0.0017 |
| 6 | worm (0.0750 / 0.0000) | bug (0.0533 / 0.0000) | abstain, held | worm 0.0833 / abstain 0.0000 | worm 0.0767 / abstain 0.0000 | abstain 0.0133 / abstain 0.0150 |
| 7 | worm (0.0667 / 0.0000) | bug (0.0350 / 0.0000) | abstain, held | worm 0.0850 / abstain 0.0000 | worm 0.0700 / abstain 0.0033 | abstain 0.0000 / abstain 0.0033 |
| 8 | worm (0.0483 / 0.0000) | bug (0.0517 / 0.0000) | abstain, held | worm 0.0550 / abstain 0.0000 | worm 0.0500 / abstain 0.0000 | abstain 0.0083 / abstain 0.0183 |
| 9 | worm (0.0650 / 0.0000) | bug (0.0583 / 0.0000) | abstain, held | worm 0.0700 / abstain 0.0133 | worm 0.0683 / abstain 0.0033 | abstain 0.0183 / abstain 0.0150 |
| 10 | worm (0.0317 / 0.0000) | bug (0.0433 / 0.0000) | abstain, held | worm 0.0583 / abstain 0.0067 | worm 0.0317 / abstain 0.0017 | abstain 0.0100 / abstain 0.0150 |
| 11 | worm (0.0767 / 0.0000) | bug (0.0683 / 0.0000) | abstain, held | worm 0.0667 / abstain 0.0050 | worm 0.0767 / abstain 0.0017 | abstain 0.0167 / abstain 0.0183 |
| 12 | worm (0.0450 / 0.0000) | bug (0.0417 / 0.0000) | abstain, held | worm 0.0483 / abstain 0.0017 | worm 0.0483 / abstain 0.0017 | abstain 0.0117 / abstain 0.0117 |
| 13 | worm (0.0700 / 0.0000) | bug (0.0750 / 0.0000) | abstain, held | worm 0.0667 / abstain 0.0017 | worm 0.0700 / abstain 0.0000 | abstain 0.0183 / abstain 0.0100 |
| 14 | worm (0.0700 / 0.0000) | bug (0.0650 / 0.0000) | abstain, held | worm 0.0517 / abstain 0.0083 | worm 0.0700 / abstain 0.0017 | sky 0.0267 / abstain 0.0050 |
| 15 | worm (0.0617 / 0.0000) | bug (0.0500 / 0.0000) | abstain, held | worm 0.0767 / abstain 0.0083 | worm 0.0650 / abstain 0.0000 | abstain 0.0133 / abstain 0.0133 |
| 16 | worm (0.0683 / 0.0000) | bug (0.0367 / 0.0000) | abstain, held | worm 0.0500 / abstain 0.0067 | worm 0.0700 / abstain 0.0000 | abstain 0.0183 / abstain 0.0017 |

Also with the option on, 16/16: the learning lesion (`lateral_learn`) abstains, same-seed reads are identical, the
C2/C3 moat shapes abstain, every winning worm spike belongs to the learned pathways (attribution 1.0), and every read
records `hop: true`, `query_steps: 30`. The canary's inherited `eat worm` link is still written at encoding (2.2-7.0);
a read-only lateral cut leaves worm at 0.0133-0.05, so 0.31-0.62 of the winning rate rides the lateral read (0.22-0.34
before). The bridge's lake read is at most 0.0133 (0.67x the floor) while the canary's eat read in the bridge sessions
is worm at 0.0317-0.085 in both orders. Scratch reads at the same revision, seeds 1-16 (saved in
`hop_scratch_reads.json`, block `probes_bat_robin_trout_30step_seeds1-16`): `what does the robin
swim` abstains 16/16 (lake 16/16 off), `what does the bat swim` reaches lake 16/16 (0.025-0.0933; off 16/16), and the
robin's and bat's own `eat` reads now follow their told facts (robin worm 16/16, bat bug 14/16; off: bug and worm
16/16, inverted).

The bar (`hop_dev_verdict`; bar-sanity OK, n 1000, every null 0.000, both perfect brains 1.000,
`docs/drafts/stage1-c1-inheritance/bar_sanity_record_hop_dev.json`): per seed, the C1 canary reaches worm, the
permuted canary reaches bug, the read lesion abstains with the cut held, the C2/C3 moat shapes abstain, and in both
bridge orders the canary's swim abstains and its eat reaches worm. PASS iff 16/16 (HOP_DEV_SEEDS).

**Off is unchanged, in the data.** The same runner with the option off (`organ_dev_nohop_baseline_seeds1-8.json`)
equals the recorded 09-30 dev file (`organ_dev_seeds1-8.json`) on all 80 shared row fields (exact compare), and
`tests/test_category_inherit_hop.py` reproduces the recorded seed-3 reads exactly.

## Verify-go (2026-10-06): two independent skeptics; NARROWED, and the configuration was changed

- **Held-out seeds (both skeptics).** The first configuration (35-step read) passed seeds 1-8 and failed fresh dev
  seed 9 (bridge lake 0.0229). It was replaced by the 30-step read chosen on seeds 1-16 (above); the claim is now
  "16/16 in-sample", and the gate seeds are the held-out test.
- **Not a mass artifact.** The fix lowers every read, but not equally: on seeds 9-12 the bridge lake/worm ratio fell
  from 0.59-0.64 to 0.05-0.30, and a structurally new two-hop chain (a gull that eats the worm and swims in the lake;
  a heron that only flies; `what does the heron swim`) abstained on 4/4 fresh seeds (lake 0.0014-0.0143 on, 0.0642-0.085 off). One-hop
  reads with a single supporter sit just above the floor (sparrow swim via the gull 0.02-0.0243).
- **Diagnosis narrowed.** The shortcut link is attenuated, not prevented (wording fixed above).
- **Instrument gaps closed.** The battery now records each read's `hop` and `query_steps`; the flip dev check now
  fails a bridge session whose canary eat ties and requires the canary probe's organ read to carry `hop` in the
  cat_disc and cat_bridge cells (a cell with no organ read no longer passes the hop check).
- **Required lenses.** Engagement: every hop row records `hop: true` and the read lesion `lesion_held: true`.
  Answer rate (organ level only; no handler shards yet): the option removes answers only on over-extensions (lake,
  the trout's sky) and turns the bridge's canary-eat ties into worm. Where ran: every cited battery on the workstation
  (dant123-wk), numpy; the organ reads no file under research/findings/raw/, so validity does not depend on the host.

## What this does not show, and what is left

- **Gate seeds and the real handler are unmeasured with the fix.** The 09-30 C1 PASS 6/6 was measured without it; the
  re-measure is registered (prereg section 13, amendment 2026-10-06) and the flip dev check now runs with the fix and
  the bridge in its bar. Both are Hermes pool steps.
- **Thin margins, in-sample.** With the option the C1 worm read is 0.0317-0.0767 (was 0.1042-0.1192; seed 10 at 1.6x the
  0.02 floor) and the permuted bug read 0.025-0.075 (seed 5 at 1.25x). The lake refusal sits at most 0.67x the floor.
  Every clause has a dev seed within a factor of two of the floor, the parameters were chosen on all 16 dev seeds, and
  one earlier choice (35 steps) failed a fresh seed. The gate re-measure can fail on correctness or on the permuted
  clause, and the flip dev check on the bridge.
- **One shared feature is still enough.** `what does the bat swim` reaches lake 16/16 with the option (0.025-0.0933,
  saved scratch read): the bat's own told `eat bug` is shared with the fish, so this is one hop, the C1 inference itself.
  `what does the robin swim` (two hops) went from lake 16/16 to abstain 16/16. A member that already carries another
  category's frames could hold the inheritance back (category competition; the 07-08 cancellation finding); an owner
  item.
- **One-member support.** `what does the trout fly` (one hop over the bat-built `eat bug -> fly sky` link) abstains on
  31 of 32 reads (sky on seed 14, bat first, 0.0267); off it is sky on every read. These lost answers are
  over-extensions (the trout was never said to fly), not lost correct answers.
- **Host seams unchanged.** The read length is a host timing constant (as the 60 steps were); an
  in-substrate end of the retrieval cycle (theta-paced or slow GABA_B feedback inhibition) is the next rung. Admission
  by the asked verb, the floor and the margin stay host. The organ is still a standalone bridge.
