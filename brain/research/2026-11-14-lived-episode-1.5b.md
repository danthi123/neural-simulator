# Pre-registration: store the lived exchange, not the telling alone (first-person echo), 1.5B oracle

Written 2026-11-14 before the run (a 2-fact dev smoke only checked that the probe runs). Capability: episodic recall
(answering from a memory from the right perspective: "Your cat is called Pixel.", not "My cat is called Pixel.").

**Record brief.** At gain 0.5 the 1.5B brain answers told facts, but many replies restate the user's sentence in the
first person, which the scorer rejects as an echo: unslept S_all named the fact in 54 told replies and was credited
for 35 (`results/2026-11-14-key-default-1.5b.md`, part A). No earlier method targets the echo (record check: echo
appears only as a scorer rule, `strict_scorer.is_echo`, and as counts). Biology: an episode is the whole lived event,
including one's own part in it; the hippocampus binds who said what. Now the brain stores only the user's sentence.

**Prerequisites.** Met: 1.5B cortex, gain switch, the oracle probe (`hippo_gain_refit.py`, deterministic, exact
across pool nodes). Host decision: none. The stored reply is the brain's own greedy reply to the telling, generated
from the same state (no template text).

**Method.** `brain/experiments/episode_format_probe.py`, formats tell (now) and lived ("User: <s>\n\nAssistant:
<own reply>"), gains 0.25 / 0.5 / 0.75 / 1, dev = DEV_FACTS36 (108 cues), test = 12 FACTS (orig + para). Oracle
routing (the right episode). pool40: dev tell; pool41: dev lived; pool42: test both. Table:
`brain/experiments/episode_format_table.py` (tested).

**Criteria.** G*_fmt = the dev gain with the most right replies for that format (ties -> smaller gain).
- E0 instrument: tell at gain 0.5 equals the 11-13 refit exactly (dev 58/108, test told 7, para 7). Else VOID.
- E1 dev right: lived at G* >= tell at G* + 10 (of 108).
- E2 test told: lived at G* >= tell at G* + 2 (of 12), and para lived >= tell.
- E3 dev echoes: lived at G* <= half of tell at G*.
LEAD = E0-E3 -> store the lived exchange in the live Brain behind a switch (default OFF), six-seed live test (pool
part A + AWS mixed day). Not a lead otherwise; the verdict is on this storage format, not on perspective.

**Expected.** Uncertain. The own reply puts a second-person frame ("your hamster") into the episode, which should cut
echoes; it may also dilute the fact at a fixed gain, which is why each format gets its own G*.
