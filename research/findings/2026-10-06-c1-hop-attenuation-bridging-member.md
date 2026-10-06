---
type: finding
status: live
date: 2026-10-06
lane: integration-first (Stage-1 exit criterion C1; RULING c1-hop-first, owner 2026-10-06)
mechanism: hop attenuation in the category-inheritance organ (BRAIN_CATEGORY_INHERIT_HOP, default off): the learned frame->frame pathway transmits at 0.5 of its weight at encoding and at the read, and the read lasts one 35-step retrieval cycle, so a bridging member no longer carries a property two category hops away
seeds: [1, 2, 3, 4, 5, 6, 7, 8]
runner: research/runners/category_inherit_organ_dev.py
artifacts:
  - research/findings/raw/_stage1_c1_category/dev/organ_dev_hop_seeds1-8.json
  - research/findings/raw/_stage1_c1_category/dev/organ_dev_nohop_baseline_seeds1-8.json
  - docs/drafts/stage1-c1-inheritance/bar_sanity_record_hop_dev.json
---

# C1 over-extension through a bridging member: diagnosed, and fixed at organ level by hop attenuation (dev 8/8)

**Scope.** Organ level only (the `CategoryInheritOrgan` fed the Stage-1 C1 facts directly, numpy, CPU), dev seeds
1-8 only, no gate seed. This is a dev result and a design record, not a generalization claim: the gate seeds and the
real handler are measured next (prereg 2026-09-30 section 13 amendment; `docs/drafts/stage1-c1-inheritance/HANDOFF.md`).

## The defect

After the C1 facts (robin, sparrow and finch eat the worm and fly in the sky; trout and salmon eat the bug and swim in
the lake), add a bat that eats the bug and flies in the sky, then the canary that flies in the sky. The 09-30 organ
answers `what does the canary swim` with lake on 8 of 8 dev seeds (prereg section 12), in either fact order.

## Diagnosis: the bridge is written at encoding, and the read walks the merged graph

Measured with the 09-30 organ on dev seeds 1-8. The encoding and read-dynamics numbers below are scratch reads of
the organ's own weight audit and spike counts (CPU, not saved as artifacts; the scripts drive only the organ's public
`observe` / `read` / `infer` / `weight_audit`); the decisions are reproduced by the two battery files above.

1. **Encoding writes links between frames no member carried together.** The rule is LTP-only (no decay, no
   depression) and the frame->frame pathway is all-excitatory with no inhibition. While the bat's second fact
   (`fly sky`) is rehearsed, its first fact's frame (`eat bug`, 0.09 spikes/neuron/step) and that frame's associate <!--derived-->
   `swim lake` (0.05-0.06, two hops from the heard frame) fire too (seed 1), so a direct `fly sky -> swim lake` link <!--derived-->
   grows to 5.7-9.9 (8/8 seeds, both orders) from the 0.05 floor. No member was ever told both. <!--derived-->
2. **When the canary comes after the bat, its own rehearsal compiles the chain into its member links.** `canary ->
   swim lake` reaches 3.7-6.9 and `canary -> eat bug` 4.2-8.1; cutting the frame->frame pathway at read time leaves
   54-99% of the canary's lake read standing (8/8). When the canary comes first, those member links stay at the floor
   and the lake read (0.076-0.096) disappears entirely under the same cut: it travels the bat-built lateral link. <!--derived-->
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
transmission gate on exactly those synapses, at encoding and at the read), and the read lasts 35 steps. Each lateral
hop then multiplies the drive by about one half: one hop still reactivates (so integrative encoding still writes
`canary -> eat worm`, 2.9-7.0 vs 7.2-9.4 before), two hops stay subthreshold, so no shortcut link is written and the
read ends before the second hop. Off builds no gate and reads 60 steps.

Levers tried on dev seeds before this one (each recorded, none kept): a divisive feedback inhibitory pool on the frame
layer (40 interneurons; it scaled every frame alike: lake 0.035 vs worm 0.042 on seed 1, still above the 0.02 floor), <!--derived-->
and encoding-only suppression at 0.7 / 0.5 / 0.35 with the 60-step read (lake fell from 0.072 to 0.040, never below <!--derived-->
the floor). The read window was then chosen on dev seeds 1-8 among 30 / 35 / 40 steps: 30 left the permuted read at
1.25x the floor, 40 put the bridge lake read at 0.8x it, 35 balances the two. <!--derived-->

## Result (organ, dev seeds 1-8, `--hop`; hop dev bar PASS 8/8)

Rates are spikes per neuron per step over the read (floor 0.02, margin 1.25x); `off` is the same runner with the <!--derived-->
option off (`organ_dev_nohop_baseline_seeds1-8.json`; tie = the organ abstains on a worm/bug tie).

| seed | C1 canary eat (worm / bug) | permuted (bug / worm) | read lesion | bat first: canary eat / swim | canary first: canary eat / swim | trout fly (bat first) |
|---|---|---|---|---|---|---|
| 1 | worm (0.0771 / 0.0000) | bug (0.0600 / 0.0000) | abstain, held | worm 0.0743 / abstain 0.0100 (off: worm / lake 0.0725) | worm 0.0771 / abstain 0.0029 (off: tie / lake 0.0817) | sky 0.0286 (off: sky 0.1075) | <!--derived-->
| 2 | worm (0.0714 / 0.0000) | bug (0.0571 / 0.0000) | abstain, held | worm 0.0729 / abstain 0.0043 (off: worm / lake 0.0775) | worm 0.0714 / abstain 0.0014 (off: tie / lake 0.0842) | abstain 0.0171 (off: sky 0.1108) | <!--derived-->
| 3 | worm (0.0586 / 0.0000) | bug (0.0529 / 0.0000) | abstain, held | worm 0.0657 / abstain 0.0000 (off: tie / lake 0.0725) | worm 0.0600 / abstain 0.0014 (off: tie / lake 0.0858) | sky 0.0229 (off: sky 0.1067) | <!--derived-->
| 4 | worm (0.0771 / 0.0000) | bug (0.0529 / 0.0000) | abstain, held | worm 0.0786 / abstain 0.0043 (off: tie / lake 0.0842) | worm 0.0771 / abstain 0.0000 (off: tie / lake 0.0833) | sky 0.0271 (off: sky 0.1033) | <!--derived-->
| 5 | worm (0.0586 / 0.0000) | bug (0.0286 / 0.0000) | abstain, held | worm 0.0729 / abstain 0.0000 (off: tie / lake 0.0800) | worm 0.0586 / abstain 0.0000 (off: worm / lake 0.0758) | abstain 0.0014 (off: sky 0.1025) | <!--derived-->
| 6 | worm (0.0771 / 0.0000) | bug (0.0614 / 0.0000) | abstain, held | worm 0.0871 / abstain 0.0114 (off: worm / lake 0.0808) | worm 0.0786 / abstain 0.0000 (off: tie / lake 0.0833) | sky 0.0271 (off: sky 0.1000) | <!--derived-->
| 7 | worm (0.0814 / 0.0000) | bug (0.0486 / 0.0000) | abstain, held | worm 0.0829 / abstain 0.0014 (off: worm / lake 0.0700) | worm 0.0814 / abstain 0.0100 (off: worm / lake 0.0958) | abstain 0.0086 (off: sky 0.0933) | <!--derived-->
| 8 | worm (0.0629 / 0.0000) | bug (0.0543 / 0.0000) | abstain, held | worm 0.0671 / abstain 0.0000 (off: tie / lake 0.0642) | worm 0.0657 / abstain 0.0014 (off: tie / lake 0.0800) | abstain 0.0157 (off: sky 0.1042) | <!--derived-->

Also with the option on, 8/8: the learning lesion (`lateral_learn`) abstains, same-seed reads are identical, the
C2/C3 moat shapes abstain, and every winning worm spike belongs to the learned pathways (attribution 1.0); the
canary's inherited `eat worm` link is still written at encoding (2.9-7.0) and a read-only lateral cut leaves worm at
0.020-0.054, so 0.30-0.66 of the winning rate rides the lateral read (0.22-0.34 before). The bridge's lake read fell <!--derived-->
from 0.064-0.096 to at most 0.0114 (below 0.57x the floor) while the canary's eat read in the bridge sessions is worm <!--derived-->
at 0.059-0.087 in both orders, where the 09-30 organ tied on 4 (bat first) and 6 (canary first) of 8 seeds. <!--derived-->

The bar (`hop_dev_verdict`; bar-sanity OK, n 1000, every null 0.000, both perfect brains 1.000, <!--derived-->
`docs/drafts/stage1-c1-inheritance/bar_sanity_record_hop_dev.json`): per seed, the C1 canary reaches worm, the
permuted canary reaches bug, the read lesion abstains with the cut held, the C2/C3 moat shapes abstain, and in both
bridge orders the canary's swim abstains and its eat reaches worm. PASS iff 8/8.

**Off is unchanged, in the data.** The same runner with the option off (`organ_dev_nohop_baseline_seeds1-8.json`)
equals the recorded 09-30 dev file (`organ_dev_seeds1-8.json`) on all 80 shared row fields (exact compare), and
`tests/test_category_inherit_hop.py` reproduces the recorded seed-3 reads exactly.

## What this does not show, and what is left

- **Gate seeds and the real handler are unmeasured with the fix.** The 09-30 C1 PASS 6/6 was measured without it; the
  re-measure is registered (prereg section 13, amendment 2026-10-06) and the flip dev check now runs with the fix and
  the bridge in its bar. Both are Hermes pool steps.
- **Thinner margins.** With the fix the C1 worm read is 0.059-0.081 (was 0.104-0.119) and the permuted bug read <!--derived-->
  0.029-0.061 (seed 5 at 1.43x the 0.02 floor); a gate seed could fall nearer the floor than any dev seed did, and <!--derived-->
  the re-measure's permuted clause is the one most at risk. <!--derived-->
- **One shared feature is still enough.** `what does the bat swim` reaches lake on 8/8 with the fix (0.036-0.077, <!--derived-->
  scratch read at 260bc1e59): the
  bat's own told `eat bug` is shared with the fish, so this is one hop, the C1 inference itself. `what does the robin
  swim` (two hops) went from lake 8/8 to abstain 8/8. A member that already carries another category's frames could
  hold the inheritance back (category competition; the 07-08 cancellation finding); an owner item.
- **One-member support.** `what does the trout fly` (one hop over the bat-built `eat bug -> fly sky` link) went from
  sky 8/8 to abstain on 4 of 8 seeds at 35 steps (8/8 at 30). Not barred: it is a one-hop inference with one
  supporting member.
- **Host seams unchanged, one added.** The read length is a host timing constant (as the 60 steps were); an
  in-substrate end of the retrieval cycle (theta-paced or slow GABA_B feedback inhibition) is the next rung. Admission
  by the asked verb, the floor and the margin stay host. The organ is still a standalone bridge.
