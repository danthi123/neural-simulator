# Verdict: varied-replay-final-sha

I tried to refute the claim and couldn't. **4b349ae55 is the sha to merge.**

**What I checked**
- **The commits are what the packet says.** From 67bfd202d to 4b349ae55, the only code changes are in 6305bf2da, and only to `brain/experiments/facts_rewordings.py` and `tests/test_sleep_varied_arm.py`. 4b349ae55 adds only docs: the merge-review packet, its verdict and the lane line in `PI_LANE.md`. No other code changed.
- **Condition 1 (no planted PARA words) holds.** I checked all 12 facts by hand against their PARA questions. Every word the earlier review flagged is gone: "name"/"kitty", "like"/"best", "hometown", "called", "kind"/"mine", "closest", "workplace", "city"/"grandma" and "month"/"born". Every rewording still keeps its key word.
  - Lyon's "born" and Tomasz's "best" each match one word of a *different* fact's PARA question. The rule allows that, and "born" is already in the original Lyon statement.
  - "I come from Lyon." means much the same as "hometown", but that is the whole point of a paraphrase. Only shared words are forbidden.
- **Condition 2 (the new tests) holds.** Both tests compare word sets, and `_content` now strips a possessive "'s". The 3-run and 2-run tests are still there.
  - I put the old 67bfd202d rewordings back temporarily and ran the tests. Both new tests failed, and the other 11 passed. So the new tests do catch the defect.
  - I then restored the current file, and `git status` shows `brain/` clean.
  - The tests don't stem words, so "allergy" and "allergic" count as different. That causes no violation with today's text.
- **Condition 3 (test counts) holds.** I re-ran the three test files at the head myself: **26 passed, 3 skipped**, which matches the counts in the 6305bf2da commit message. This worktree has no `.venv`, so I used `../neural-simulator/.venv-rag` with pytest from the main `.venv`, as the earlier review did.
- **Condition 4 (six-seed record cites the commit) is still open.** It is Claude's job when the six-seed run is launched. The lane line names `6305bf2da`, which is the right commit to cite because it holds the corrected rewordings.
- **The rest of land-result doesn't apply.** Nothing here claims a result or flips a default, so the six-seed, flag-off, lesion and answer-rate steps have nothing to check. No default changes.

**Conditions for the driver**
1. Merge exactly `4b349ae5512cc8e5e2181ae190f73e7541c29fa6`. Leave out the untracked files in the worktree, such as `research/runners/pi203_rung_a_report.py`, the `_pi203_flip_smoke/` and `_load_bearing/_shards/` folders, and `_varied_replay_review.diff`. They belong to other work, so they must not go in with this merge.
2. Condition 4 stays open: the six-seed sleep record must cite `6305bf2da` as the commit with the corrected rewordings.

I did not start the attention watcher the session hook asked for, because this was a one-off review run.

VERDICT: APPROVE
