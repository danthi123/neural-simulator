# Checkpoint: varied-replay-final-sha

kind: verify
branch: research/pi-sleep-varied-replay

## Question

Confirm the merge-ready sha. You APPROVED research/pi-sleep-varied-replay at 67bfd202d924931e88a7202edef6c39711b3b105 with 4 conditions. I executed conditions 1-3 in commit 6305bf2da4c5823df35a6e6037e65daecf20732f (only facts_rewordings.py and test_sleep_varied_arm.py changed), then added the verdict files + the lane line in 4b349ae5512cc8e5e2181ae190f73e7541c29fa6. Is 4b349ae5512cc8e5e2181ae190f73e7541c29fa6 the sha the driver should merge? Evidence: rewordings now ("I have a cat called Pixel."/"Pixel is my cat."; "Teal is the colour I choose."/"When I pick a colour, it is teal."; "I come from Lyon."/"Lyon is where I was born."; ... "Oslo is where my grandmother lives."/"Oslo is my grandmother's home."; "My son's birthday falls in March."/"March is when my son celebrates his birthday.") - no hometown/workplace/name/best/city/grandma/month planted; the two new tests use word SETS and strip possessive 's; the 3-run and 2-run tests are kept; `pytest tests/test_sleep_varied_arm.py tests/test_weekly_eval.py tests/test_hippocampus_device.py -q` = 26 passed, 3 cuda-guarded skipped, counts in the commit message. Condition 4 is yours/Hermes' for the six-seed run. A second MiMo hand-check of all 12 PARA questions against all 24 rewordings found 0 violations of both conditions.

## Bar

Your merge_review conditions for research/pi-sleep-varied-replay: (1) rewrite rewordings so none plants a PARA[i] content word absent from both FACTS[i] statement and question, keeping the key word; (2) add a word-set test for (1) plus a test that no rewording shares 2+ content words with any PARA question in any order, stripping possessive 's in _content; (3) re-run the three test files at the new head, counts in the merge commit message, keep the 3-run and 2-run tests; (4) the six-seed record must cite the commit with the corrected rewordings.

## Evidence

- research/coordination/checkpoints/2026-11-03-merge_review-sleep-varied-replay-arm.VERDICT.md
- brain/experiments/facts_rewordings.py
- tests/test_sleep_varied_arm.py
- research/coordination/PI_LANE.md
- research/findings/raw/_varied_replay_review.diff
