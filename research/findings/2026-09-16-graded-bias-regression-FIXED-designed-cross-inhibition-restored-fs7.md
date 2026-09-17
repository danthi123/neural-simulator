---
type: finding
status: verified
date: 2026-09-16
mechanism: biased-competition WTA designed cross-inhibition (sel_FS_X -> sel_Y!=X) strengthened 5.0 -> 7.0 to
  carry the rival-suppression a trait-init artifact (removed by e7f009a37) had been carrying
integration_faculty: selective-attention-biased-competition
lane: language + focus (scaffold-retirement, owner's #1 metric)
seeds: [42, 43, 44, 100, 101, 102]
verdict: The pre-existing graded-bias CI regression (RED on clean main since e7f009a37, 2026-08-03) is FIXED by a
  principled MECHANISM re-tune -- NOT a tune-to-seed-100. git bisect had root-caused it: e7f009a37 correctly removed
  random inhibitory-trait leakage that every nominally-excitatory region used to transmit, and the biased-competition
  WTA rival-suppression had been co-tuned to LEAN on that leakage; at the deployed fs_to_sel_weight=5.0 the DESIGNED
  sel_FS->sel_Y interneuron cross-inhibition alone lands the extreme-asymmetry cases ~1.27x, just under the 1.3x moat
  -> abstain. A structural sweep across 6 seeds found a STABLE BASIN {6,7,8} all restoring the 2-referent GO-arm to
  6/6 (>=9 destabilises the marginal seed-102 case -- the code's documented "symmetric over-inhibition is unstable"
  regime), centred at 7.0. At the new 7.0 default: the 2 failing tests PASS, production + byte-identity tests
  UNCHANGED, de-risk 2-ref GO-arm 6/6 (was 3/6) with lesion 6/6 + moat 6/6 intact, and the seed-100 fixed-vs-graded
  contrast preserved (FIXED->cat mis-resolve, GRADED->ball). HONEST RESIDUAL: e7f009a37 also degraded the de-risk's
  3-REFERENT scale probe 6/6 -> 4/6; fs 5->7 leaves it 4/6 (orthogonal), a separate deeper scaling weakness for a
  follow-on. Unblocks the biased-competition content_bias_target host-lexicon delete -> RETIRED.
runner: research/runners/_phaseB_biased_competition_graded_derisk.py
artifacts:
  - research/findings/raw/_biased_competition_graded_fs7_fix_6seed.json
external: NO-EXTERNAL-NEEDED -- this is a mechanism re-calibration confirming a git-bisect root cause; the fix
  restores the designed Rutishauser selective-inhibition motif already grounded in research/biology.
builds_on:
  - research/findings/2026-09-16-wirein-flips-biased-competition-gnw-stop-conflict-scaled-DEFAULT-ON-GO.md
---

# Graded-bias regression FIXED — the designed cross-inhibition now carries the suppression (fs_to_sel 5.0 -> 7.0)

## What was broken (root cause, banked in research/FAILURE_LOG.md)

`tests/test_multireferent_graded_bias_agent.py::{test_fixed_bias_default_mis_resolves_seed100_roll,
test_graded_bias_closes_seed100_roll}` were RED on clean committed main. A git bisect (good=c5f70a9ef where the test
was added + green, bad=main) pinpointed **e7f009a37 "add neural vocal action selector gate" (2026-08-03)**. Its
sim/bridge.py trait-init fix correctly made `set_pathway_weights(output_inhibitory_indices=...)` reset nominally-
excitatory regions to PURELY excitatory (before: "a fraction of every nominally excitatory region transmit
inhibition" -- random inhibitory-trait leakage). The biased-competition WTA rival-suppression had been INADVERTENTLY
relying on that leakage; with it gone, at the deployed `fs_to_sel_weight=5.0` the designed sel_FS_X -> sel_Y!=X
interneuron cross-inhibition alone suppresses the rival only enough for a ~1.27x margin -- just under the 1.3x moat --
so the extreme seed-100 case abstains. e7f009a37 is CORRECT (a proper Dale's-law cleanup); the de-risk had been partly
resting on the artifact (the CLAUDE.md "a proxy/artifact carried the result" class).

## The fix (principled mechanism re-tune, NOT tune-to-seed-100)

Strengthen the DESIGNED interneuron circuit so it carries the suppression itself. A structural sweep of
`fs_to_sel_weight` (a MECHANISM param; base_pA/gain/ref/cap bias magnitudes untouched) across all 6 seeds:

| fs_to_sel | 2-ref GO-arm | lesion-breaks | moat |
|-----------|--------------|---------------|------|
| 5.0 (old) | 3/6          | 6/6           | 6/6  |
| 6.0       | 6/6          | 6/6           | 6/6  |
| **7.0**   | **6/6**      | **6/6**       | **6/6** |
| 8.0       | 6/6          | 6/6           | 6/6  |
| 9.0       | 5/6          | 6/6           | 6/6  |
| 10.0      | 5/6          | 6/6           | 6/6  |

{6,7,8} is a stable basin; >=9 destabilises the marginal seed-102 roll case (the code comment's "symmetric
over-inhibition is unstable"). 7.0 is the basin CENTRE -- maximally robust, a modest +40% restoration, chosen for
basin-centering not for closing any one seed. Change: `research/runners/biased_competition_buffer.py` default
5.0 -> 7.0 + rationale comment.

## Verification (all CPU/numpy, local -- GPU untouched during the owner's game)

- `tests/test_multireferent_graded_bias_agent.py`: 5/5 PASS (was 2 failed / 3 passed).
- `tests/test_multireferent_biased_competition.py` (production learned-map path): 5/5, UNCHANGED.
- `tests/test_multi_turn_agent.py` (byte-identity): 3/3, UNCHANGED.
- de-risk 6-seed `research/findings/raw/_biased_competition_graded_fs7_fix_6seed.json`: 2-ref GO-arm 6/6, roll-cases
  6/6, lesion-breaks 6/6, moat-intact 6/6 (`summary.go_arm_seeds`=6, `lesion_breaks_seeds`=6, `moat_intact_seeds`=6).
- seed-100 detail at the operative spec=1.3: FIXED bias -> cat (the documented mis-resolve the graded bias closes),
  GRADED bias -> ball (correct). The fixed-vs-graded contrast -- the mechanism's whole point -- is preserved, so this
  is a genuine restoration, not a collapse of the distinction.

## Honest residual (noticed; NOT-GATEABLE, launches a follow-on)

e7f009a37 also degraded the de-risk's 3-REFERENT scale probe ({cat,ball,river}, eat->cat) from 6/6 to 4/6 (seeds
42,44 abstain against TWO rivals), and fs 5->7 leaves it 4/6 (orthogonal to fs within the stable basin). So the
de-risk runner's FULL GO bar (which requires `three_seeds == n`) still prints NEGATIVE -- the runner's own instrument
honestly flags it. The 2-referent mechanism -- what the CI tests, the production /api/brain-chat path, and the
content_bias_target delete all depend on -- is fully restored. The 3-referent scaling weakness is a separate deeper
residual (pairwise cross-inhibition does not scale to N rivals without destabilising the pair case); it is logged in
`research/FAILURE_LOG.md` as NOT-GATEABLE (no 3-ref CI guard exists) for a follow-on mechanism. A wall defers a
METHOD, not the capability.
