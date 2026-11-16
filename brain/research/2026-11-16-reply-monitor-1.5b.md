# Pre-registration: reply monitor against in-context collapse, 1.5B (`Brain(reply_monitor=4)`)

Written 2026-11-16 before any run with the switch.

**Defect (read from whole transcripts).** Day 2 of open conversation collapses in context on some seeds, in
either arm: the brain echoes the user's turn on every later turn (AWS reply-only 1118, default 1120, pool 1121 both
arms), or repeats one line (default 1117, "I collect old typewriters." x13), or one template ("I'm not sure, I don't
know what your X is." on every question, pool reply-only 1124). 3 of 12 AWS day-2 conversations collapse
(`brain/experiments/results/2026-11-16-reply-rehearse-1.5b.md`).

**What we know.** The echo is the pretrained cortex's own behaviour: it is identical with the night and the
hippocampus both off (`2026-11-16-reply-rehearse-pool-replication/echo_diag/`). It is a sampled onset: the cortex
alone over two day-2 scripts under 6 sampling seeds each echoes in 1 of 12 conversations, and that one from turn 1 on
15/16 turns (`brain/experiments/echo_onset_probe.py`). Once a copy is in the conversation state it repeats.

**Record and lever count.** Record check run (`tools/before_you_build.sh`, 11-16): no prior method on the new core
targets the question echo (the 11-14 "echo" arc was first-person fact echoes, a different defect). Levers against
in-context collapse so far: habituation (no-repeat 4-gram, NO-GO: R4 fails, it hides loops as paraphrase). This is
the second; the research gate fires if it fails.

**Method.** `Brain(reply_monitor=N)` (switch, default OFF). Before a reply enters the conversation state it is
compared with the turn it answers and with the brain's own replies since waking; a draft sharing >= 0.75 of its words
(>= 3 words) with one of them is withheld and the cortex speaks again from the same state, up to N = 4 times; if every
draft is a copy the last is kept. Each withheld draft is recorded (`info["withheld"]`, the rows' `withheld` field).
Declared scaffold (`brain/SCAFFOLDS.md`): a word-overlap comparison standing in for self-monitoring of inner speech.
Host decision: the comparison and its 0.75 bar only; no reply text comes from the host.

**Prerequisites.** Met: the 1.5B cortex speaks fluent turns; the onset is a sampled event (so a second draft from the
same state usually differs). Expected to fail where the collapse is not sampled but forced by the state (a stored
line the night over-trained: then every draft repeats it and the last copy is kept).

**Run.** `brain/eval/open_conv.py --model 1.5b` default (ocd) vs `--reply-monitor 4` (ocm), fresh AWS seeds 1126-1131
(decision); pool seeds report-only. Same scorer as addenda 4-5 of `2026-11-16-rehearsal-night-1.5b.md`.

**Bars (all against the default on the same seeds):**
- M1 collapse: collapsed seeds (top repeat > 3, stored-line reuse > 3, or echoes > 3) <= 1/6 and <= the default's.
- M2 recall: intact day-2 recall >= the default's - 2 (of 36).
- M3 false recall: <= the default's + 1.
- M4 blind reading: one sonnet reader per seed, two conversations under shuffled letters, key stored apart; the
  monitor scores >= the default on >= 5/6 seeds.
- Answer rate: abstentions on memory questions <= the default's + 2.
- Engagement: `withheld` > 0 on at least one turn of at least 3 seeds, else UNDEFINED (the switch did not act).
GO on all -> flip `reply_monitor=4` on by default (validated + engagement shown). Any fail -> NO-GO on this method;
the capability stays open and the research gate fires (two levers).

**Report-only extra.** Echo scorer counts exact matches only; also report pronoun-flipped echoes from the rows.
