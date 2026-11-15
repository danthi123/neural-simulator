# Pre-registration: self episodes stored as their completion: does the brain say its own name on day 1? (1.5B weekly, six seeds, AWS)

Written 2026-11-15 before any weekly run with the switch. Capability: Self and identity; board #280 (high-use facts
never wrong); milestone day 14. Method 1 on the day-1 own-name defect (lever count 1).

**Record brief.**
- Day 1, turn 1: own name right on 2/6 registered seeds, though the self episode is recalled on 6/6
  (`results/2026-11-15-weekly-1.5b-milestone.md` addendum).
- Diagnosis (`results/2026-11-15-own-name-day1-probe-1.5b.md`, one brain, 12 samples): at gain 0.5 the full self
  episode names Nova in 5/12; stored as its completion (d(question + answer) - d(question), norm-matched) it names
  Nova in 12/12 at 0.5 and stays clean to gain 4, where the full episode replays its own question.
- The switch: `Brain(self_completion=True)` / `weekly --self-completion` (default OFF). It changes only the three
  permanent self episodes grown at birth ("What's your name?", "Who are you?", "Do you remember my name?"); keys and
  routing are read from the full change, tellings are unchanged, sleep replays the same text
  (`brain/tests/test_episode_switch.py`).
- Smoke: `Brain.hear` at 1.5B on pool40, off vs on, seeds 42-102, before this batch (numbers in the result).
- Prerequisites met; host decision none (no text, no rule; the brain stores what its own answer added).
- Not addressed: "My name is Dan." after the night on some seeds (learned in sleep; the false-recall arc).

**Method.** One AWS batch (`jobs/2026-11-15-aws-self-completion-weekly.txt`, g6.xlarge L4), 12 jobs:
`brain.eval.weekly --model 1.5b --shared-sleep --seed <s>` with and without `--self-completion`, seeds
42/43/44/100/101/102. The off arm is rerun in the same batch (not the banked 11-15 transcripts) so both arms share
hardware and code. Scored by the weekly scorer as written.

**Criteria (on vs off, per seed).**
- C0: every on transcript records "self episodes store their completion" in its header; every off transcript does not.
- C1 (the target): day-1 turn 1 ("Hi! What's your name?") correct on >= 5/6 seeds on, and on >= off + 2 seeds.
- C2 (no cost on told facts): session-2 keyed correct, intact arm, mean on >= mean off - 0.5; and the day-1
  "What is my cat called?" correct on >= off seeds.
- C3 (no cost on the friend's name): "Do you remember my name?" in session 2, intact, correct on >= off seeds.
- Reported, not criteria: session-2 own name per arm (the night's "Dan" binding), the hippocampus-off arm, replies
  that echo a question back.

**Decision.** C0-C3 hold: GO; flip `self_completion` on by default (the brain is born with completion self episodes)
after the skeptic, and the weekly record uses it. C1 fails: NOT A LEAD, next method from the probe (a higher gain
for permanent episodes, a window 0.75-2). C2 or C3 fails with C1 holding: NARROWED, reported, no flip.

**Expected.** C1 holds: 12/12 in the probe at the live gain.
