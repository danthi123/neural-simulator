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
