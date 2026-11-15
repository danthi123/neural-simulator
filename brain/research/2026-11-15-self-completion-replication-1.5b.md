# Pre-registration: replication of self-completion self episodes on fresh seeds (1.5B weekly, seeds 300-305, AWS)

Written 2026-11-15 after the registered test and before any run on these seeds. Capability: Self and identity; board
#280. A second registered test of the same switch with the same bars; not a new method.

**Record brief.**
- Registered test (`results/2026-11-15-self-completion-weekly-1.5b.md`, seeds 42-102): NARROWED. Day-1 own name
  6/6 on vs 2/6 off; session 2 equal in both arms; C2 failed by one day-1 cat answer (s43, "My cat is called
  Pixel.", after a conversation that diverged from turn 1).
- Report-only pool seeds 200-205: C0-C3 hold by the same bars (cat 3/6 both, moving both ways).
- The open question is whether the switch costs day-1 told answers, or whether the s43 loss is a different
  conversation path. This test cannot tell those apart on one seed either; it asks whether the cost recurs.
- Prerequisites met; host decision none.

**Method.** One AWS batch (`jobs/2026-11-15-aws-self-completion-replication.txt`, g6.xlarge L4), 12 jobs: the
registered command on seeds 300/301/302/303/304/305, off and on. Scored by `brain/experiments/self_completion_score.py`
as written for the registered test.

**Criteria: C0-C3 exactly as in `brain/research/2026-11-15-self-completion-weekly-1.5b.md`**, on these six seeds.

**Decision.**
- C0-C3 hold here: GO. Flip `self_completion` on by default after the skeptic. The record reports both tests: the
  registered one NARROWED (one day-1 cat answer), this one holding.
- C1 holds and C2 or C3 fails here as well: the cost recurs. No flip; next is a direct test of the day-1 told answer
  with the conversation history held fixed.
- C1 fails: the day-1 effect does not replicate; NOT A LEAD.

**Expected.** C1 holds (6/6 twice so far). C2 is a coin-flip-sized risk at one seed.

**Addendum (before any run): report-only seeds 306-311 on the pool (CPU fp32), off and on**, same pool script as the
registered test's addendum (`jobs/2026-11-15-pool-self-completion-weekly.sh`). Not criteria, not pooled.

**Note (before any result, written while the batch launches).** In the registered test session 2 recalled no self
episode in either arm (the sister telling wins "What's your name?" on 6/6), so session 2 was byte-identical and C3
tested nothing. Expected the same here; C3 is kept as written, and the result will say whether it was vacuous again.
