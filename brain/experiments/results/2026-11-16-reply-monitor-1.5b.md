# Result: reply monitor, 1.5B, AWS seeds 1126-1131 — bars pass on ties, no benefit shown: NOT flipped

Prereg `brain/research/2026-11-16-reply-monitor-1.5b.md`. Arms: `brain/eval/open_conv.py` default (ocd) vs
`--reply-monitor 4` (ocm). AWS g5.xlarge batch 20261116-1410 (host ip-172-31-80-250 in every rows file; the
dead-man stopped the batch during ocm_s1131, which was rerun by `tools/aws_brain_batch.sh resume`, see `run.log`).

Scorer (`rehearsal_night_score.py <dir> 1126 ... 1131`, intact day 2):

| arm | collapsed | recall | false | intrusions | needless | echoes |
|---|---|---|---|---|---|---|
| default | 1/6 (1130, one line x10) | 25/36 | 7 | 8 | 17 | 0 |
| reply monitor | 1/6 (1130, the same line) | 25/36 | 7 | 7 | 17 | 0 |

- M1, M2, M3 hold (`R1 True R2 True R3 True`); abstentions on memory questions 0 vs 0; engagement: rows `withheld`
  > 0 on 6/6 seeds (counting day 1), 5/6 on day 2.
- M4 paired blind reading (`r4_blind_reading.json`): default / monitor 1126 3/3, 1127 3/3, 1128 4/5, 1129 4/3,
  1130 1/1, 1131 6/6: monitor >= default on 5/6, four of them ties; readers found 1126, 1127, 1130 and 1131
  word-for-word or nearly identical.

**Decision: not flipped, a stated departure from the prereg's "GO on all -> flip".** Every bar is a no-worse bar, and
an inert switch passes them. The prereg's purpose was fewer collapses; that is not shown: the one collapse (1130,
"You're Leopold, my friend.") is the same in both arms. The monitor withheld drafts on 24 of its turns and every
one ended with the default's reply, because all four redraws were copies too. Most other differences between the arms
are on turns it did not withhold (the redraws advance the sampler, so the conversation diverges). The defect it
targets, echo, did not occur on these seeds in either arm. On the pool's echo seed (report-only, 1121) it turned the
echo into a stored-line loop, and false recall rose 1 -> 4 over 1121/1124/1125. `reply_monitor` stays 0. The
switch adds a host scaffold and up to four extra generations per turn with no measured benefit.

**Skeptic (sonnet, fresh context, lens block a-c): PARTLY REFUTED the flip claim, recommends do_not_flip.**
(a) Engagement is recorded (`withheld`) but inconsequential where it matters. (b) `tools/answer_rate.py` cannot read
this layout; the regex count is 0 vs 0. (c) `tools/where_ran.py` reports unknown=12, since it does not parse these rows
files; the rows' `host` field gives the AWS host in both arms. The pool-vs-AWS contrast (echo present vs absent) is
across host classes and is not isolated. The skeptic counted M4 as 6/6; it is 5/6 (1129: 4 vs 3).

**What this means for the defect.** Output-side levers (habituation, monitor) leave the state's attractor in place:
the cortex re-draws the same copy. The own-reply-gain probe (`2026-11-16-own-gain-probe.md`) located the amplifier in
the write of the brain's own reply into the conversation state, but a flat gain also erases who is speaking. Next:
a state-side method that weakens the self-reinforcement without losing the turn structure (the research round's
method 2, adaptation of recent own-output representations, or a self/other tag on the state write).
