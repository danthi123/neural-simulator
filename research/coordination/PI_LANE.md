# Pi lane (pilot from 2026-10-31; lean method from 2026-11-02)

The supervisor feeds each `- [ ] ` line below to Pi with `tools/pi_driver/LANE_PROMPT.md`. Only Claude adds items.
The next item comes from `HANDOFF.md` or the board, as a build task for the new core (`brain/ARCHITECTURE.md`
section 4 milestones). Until Claude adds one, the lane is open and empty, and Pi waits.

## Archived (#203, frozen synapses; old track, closed by RULING pivot-1102)
Full text and evidence: `git show 6b4ff08e5:research/coordination/PI_LANE.md` (main) and the lane branch
`research/pi-lane` (its copy of this file holds the third item and the flip-smoke record).
- [x] #203 freeze: non-plastic synapses no longer drift, flag `cfg.enforce_plastic_mask_beyond_hebbian`, default OFF.
  Branch `research/pi-203-frozen-synapse-freeze` @ `fd42e0120`, merge_review APPROVED (2026-10-31).
- [x] #203 follow-up: the comprehension organ's frozen drift is the Hebbian loop (`sim/bridge.py` LTP + decay); fix
  on branch `research/pi-203-write-path-attribution`, default OFF, no reply-level change (2026-10-31).
- [x] #203 flip smoke (on `research/pi-lane`): stopped 2026-11-02 18:20: the in-flight GPU cells were killed by Claude
  (RULING overnight-1102). Nothing to land. No flip, no new #203 work.

## Open
- [ ] New core, weekly conversation test upgrade (`brain/eval/weekly.py`, milestone day 14, board #281). Today it
  sums the intact and lesion arms into one total, asks the same fixed questions every week, and credits degenerate
  replies (e.g. "Nova, Nova."). Change it so that: (1) the header prints one score line per arm (intact, hippocampus
  off) and the difference; (2) a third arm runs sleep between the two sessions (`brain/sleep.py`, adapters from
  `brain/plastic_cortex.py`) and then asks with the hippocampus OFF, so the transcript shows what sleep consolidated;
  (3) half the facts are drawn each week from a larger list, seeded by the ISO week, so the score cannot improve by
  memorising fixed questions (keep the other half fixed for week-to-week comparison); (4) a reply counts only if it
  contains the key word AND is not degenerate (repeated token runs, empty, or only the brain's own name). Add unit
  tests that run without the cortex weights (mock `Cortex`); real runs go to W via `run_gpu`. Read first:
  `brain/eval/weekly.py`, `brain/experiments/sleep_probe.py`, `brain/PREREQUISITES.md` (Sleep consolidation).
  Done = branch + tests passing + one real run on W attached to a merge_review checkpoint.
- [x] New core, sleep: varied-replay arm in `brain/experiments/sleep_probe.py` (after the weekly-test item). Six-seed
  0.1B result (`brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md`): sleep recalls the original question
  (3-4/12) but not reworded questions (2.2 vs 1.3 shuffled). Add arm `sleep_varied`: each told fact is replayed in
  three surface forms, the original plus two rewordings of the STATEMENT (e.g. "My dog's name is Pixel." -> "I have
  a dog called Pixel."), same total replay steps as `sleep_true`. Write the rewordings once into a committed file
  `brain/experiments/facts_rewordings.py` (teacher-written text, declared in `brain/SCAFFOLDS.md` like text replay);
  they must not reuse the wording of any `PARA` question (add a unit test that checks no PARA question's content
  words appear as a run of 3+ in any rewording) and must keep the key word. Add `--arms` (default: all four) so a run
  can skip arms. Unit tests run without weights (mock `Cortex`, as the weekly item). Do not run the real probe:
  Claude runs the six seeds. Read first: `brain/experiments/sleep_probe.py`, `brain/sleep.py`,
  `brain/experiments/facts.py`. Done = branch + tests passing + merge_review checkpoint.
  DONE 11-03 (Pi): branch `research/pi-sleep-varied-replay`. Claude's verify checkpoint approved the merge at
  `4b349ae5512cc8e5e2181ae190f73e7541c29fa6` (`research/coordination/checkpoints/2026-11-03-verify-varied-replay-final-sha.VERDICT.md`);
  the only commits after it are the checkpoint verdict docs themselves (c38431c88, cd767e85b) - no code change, so the
  driver may merge either sha. Code commits: c71a96480/1138d251f/2fb6187c4/d75899aa8 (weekly item), e61f80164, then
  f1a7ead39 + 67bfd202d + 6305bf2da (this arm). Stacks on the approved `research/weekly-eval-upgrade` @ e61f80164;
  Claude said no split needed. merge_review APPROVED:
  `research/coordination/checkpoints/2026-11-03-merge_review-sleep-varied-replay-arm.VERDICT.md`. Its 4 conditions:
  (1) rewordings rewritten so none plants a PARA word absent from the statement and its question - done; (2) word-set
  test added (`test_rewordings_do_not_plant_the_paraphrase_question_own_words`, `test_rewordings_share_at_most_one_
  content_word_with_any_para_question`, possessives stripped in `_content`) - done, 3-run and 2-run tests kept; (3) all
  three test files re-run at the final head: 26 passed, 3 cuda-guarded skipped, counts in the commit message; (4) the
  six-seed record must cite this commit (`6305bf2da`) when Claude launches the run. Second opinions: two MiMo passes -
  the first REFUTED on missing data (facts.py not quoted), the second cleared all 7 defect classes and, after the
  rewrite, hand-verified 0 violations of both word-set conditions. No result claimed: the arm is code + tests only.
