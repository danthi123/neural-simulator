# Result (NARROWED): the awake teacher is clean on never-told cues (T2: 1/87 false), but its own-cue answers are 0/12 on every seed because the cortex writes those cues to the brain, not in the user's voice (T1 fails). No build; next is a better cue source

Prereg: `brain/research/2026-11-15-teacher-night-gate-1.5b.md` (T1-T3, committed before any 1.5B run). Probe
`brain/experiments/teacher_gate_probe.py --model 1.5b --seed <s> --k 4`, seeds 42/43/44/100/101/102, pool CPU fp32.
Hosts are recorded in each `tg_s<seed>.json`: node-mxl1462drl for 42/43, node-dl4g243 for 44/100, node-dl5d243 for
101/102. No training. Scorer `brain/experiments/teacher_gate_score.py`; output in `table.txt`.

| seed | own_ok (of 12) | never-told false | overlap with test questions |
|---|---|---|---|
| 42 | 0 | 0/13 | none |
| 43 | 0 | 0/13 | none |
| 44 | 0 | 0/14 | none |
| 100 | 0 | 0/17 | none |
| 101 | 0 | 1/16 | "Who is my best friend?" |
| 102 | 0 | 0/14 | none |

- **T1 fails: 0.00/12** (bar >= 8).
- **T2 holds: 1/87 = 1.1%** (bar <= 10%).
- **T3: one overlap,** "Who is my best friend?" on s101. It is a probe test question, so it is listed and would be
  dropped from any method.

## Why T1 is 0

The cues are at fault, not the teacher or the scorer. Of the 72 own cues (12 per seed):
- 54 address the brain ("What's your cat's name?", "What is your favourite colour?").
- 18 address neither the brain nor the user ("What is the capital of France?", "What color is the Volvo?").
- None ask in the user's own voice ("What is my cat called?").

The teacher answers a "your" question as itself. One case recalled the right telling and still replied "My favourite
colour is teal." That is a first-person claim of the user's fact, and the strict scorer correctly marks it wrong
(`correct_answer`: echo / wrong subject).

The recall step is half right: it returned the cue's own telling on 36/72 own cues. The reply names the key on 24 of
those 36, but always in a voice the scorer rejects.

## Decision (from the prereg)

T1 fails with T2 holding: the cues are the weak part. Next is a better cue source before any build. One option is to
ask the cortex for the question the user would ask to check the brain remembers, which gives the user's own voice.
Swapping "your" for "my" by hand would be an English-grammar template and is not allowed.

## Caveats

Skeptic (sonnet, fresh context, lens block verbatim): **NARROWED**; the no-build decision stands. Its points:
- **The own cues are one draw, not six.** OWN is greedy with k=1, so all six seeds have the identical 12 own cues.
  T1 is one cue set repeated, and "0/12 on every seed" means "this cue set is unscorable", not six confirmations.
- 24/72 own cues contain the key word ("What color is the Volvo?", "Do you know where Oslo is?").
  `correct_answer` rejects those by construction, so at most 7/12 could pass on any seed.
- The only near-pass is "What is your favourite colour?" -> "My favourite colour is teal.", rejected as a
  first-person echo. That rejection is defensible.
- `near_miss.py`'s OWN frame is first-person ("I told a friend ... Later I asked them"). It does intend the user's
  voice; the cortex drifts from it. Off-topic cues also appear ("What month is it?", "What is the capital of France?").
- T2's `told_topic` split is a keyword heuristic. Two frame cues whose reply named a told key ("Your name is Tomasz")
  were counted as told-topic, not never-told.
- `where_ran` reports unknown=6. The JSON's own `host` field records the nodes.

- T2's cleanliness is partly a floor effect: the teacher seldom names any told key, so a cleaner never-told rate is
  cheap. It does name a key on 24/72 own cues, so the floor is not total.
- These are pool CPU fp32 runs, one draw per seed.
