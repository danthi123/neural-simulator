# Stage-2 TEST v2, Amendment T4 (2026-11-27, before any test2 life): launch ASAP under a raised cap

Owner, 11-27: "Just increase the cap then?" (asked to run the test as soon as possible, not over three timed days).
- Replaces only the T3 launch timing. The 18 lives, arms (B / K-P / D2-PR2), seeds, job files, per-life dead-man
  (`maxmin`), grace, boot, one pinned AMI, commit env (BRAIN_CODE_COMMIT + BRAIN_CODE_CLEAN=1), terminate-only-if-fresh
  and the pull rules are unchanged: the launcher `_scratch/stage2/test2/asap.sh` is `launch_plan.sh` up to its `case`,
  byte-identical, plus a slot loop.
- T3's "at most 6 instances at once" is kept: the next life launches as soon as one of 6 ends. Order: 6 K-P first
  (longest), then B and D2-PR2 per seed. No life starts later than 12-04, so the D2-PR2 drop-first rule does not fire.
- Code: the launch runs from a worktree at the freeze commit d965bc716 (all 17 frozen files match the freeze note).
  The working branch has since added two default-OFF switches to brain.py (routing screen 3, cascade screen), so
  launching from its HEAD would have broken the frozen hashes; the original 12-04 plan had the same hazard.
- Cap: AWS_DAILY_CAP_USD=150 for UTC 11-27 and 11-28 (owner-approved; 6 x $1.01/h is at most about $145 per day),
  then back to $50. Expected total unchanged (about $95-105; worst $131).
- Read by 12-10 or earlier, land 12-14: unchanged.

## T4 addendum (2026-11-27 16:25 UTC, after the first life started, before any result)
The launch tool's remote start (`tools/aws_brain_batch.sh` lines 105 and 154) left the remote job's stdin on the ssh
channel, so `launch` returned only when the job ended or the 40-min `timeout` in launch_life killed it (the job
survives: nohup). That serialises the slot loop at 40 min per life and delays pulls. In the freeze worktree the tool
was replaced (atomic rename, not an in-place edit of the running script) with the same two lines plus `< /dev/null`:
sha256 4d94c7ab... (freeze note, "recorded for the launch") -> 8ba819ab7851509ac1a9718a0209a79adf4d4479200364b6f63b311a7dc52b5a.
Nothing that runs on the instance changes (code, env, models and run.sh are byte-identical); lifeKP_s42 was launched
with the old tool, every later life with the new one. Same fix on the working branch.

## T4 addendum 2 (2026-11-27 18:45 UTC, lives running, no result read)
The first addendum's `< /dev/null` did not stop the 40-min launch hang (KP43, KP44, KP100 still ended rc 124). Root cause:
line 105 runs `chmod +x run.sh && nohup bash -c '...' >/dev/null 2>&1 < /dev/null &`, and `&` backgrounds the whole AND
list, so a subshell holding the ssh channel waits for the job. Verified on pool41: the old pattern held ssh for the job's
full 20 s, the new pattern (`chmod ...; nohup ... &`) returned in 1 s. Replaced in the freeze worktree by atomic rename:
sha256 -> 2b83d117ae4381bb4832e77aca984b077ed087ba43152a06d8b67fd85d970722. Nothing that runs on the instance changes.
KP42-KP101 launched with the earlier tool versions; later lives with this one. Same fix on the working branch.

## T4 addendum 3 (2026-11-28 03:40 UTC, lives running, no result read)
At 23:14 UTC the ASAP launcher aborted on lifeB_s42: AWS had no g5.xlarge capacity in us-east-1a (InsufficientInstanceCapacity),
but asap.sh's copy of launch_life tests `grep -q 'aws_budget'` on the launch output, and the budget check prints
"✓ aws_budget: ..." on success, so a capacity miss was read as a cap refusal and the whole plan stopped (about 4.5 h lost;
the 5 finished K-P lives were not pulled meanwhile). Fixed in asap.sh only (launch_plan.sh stays frozen): the test is now
`grep -q '⛔ aws_budget'` (the refusal line), so a capacity miss tries the next subnet as designed. Restarted 03:37 UTC.
