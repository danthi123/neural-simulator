# Result: self episodes keyed by their question win every question (K1 12/12, but K2 0/12, K3 fails): NOT A LEAD. Question-shaped keys group together under S0

Prereg: `brain/research/2026-11-15-self-key-question-1.5b.md` (K1-K3, committed before the run, d81f692c3). Probe:
`brain/experiments/self_key_probe.py`, run on copies of the 12 slept off-arm intact homes (seeds 200-205 and
306-311) with pool CPU fp32. Hosts: pool40 node-mxl1462drl (200/201/306/307), pool41 (202/203/308/309), pool42
(204/205/310/311); each JSON records its `host`. There was no training and no reply. Scorer
`brain/experiments/self_key_score.py`; output in `table.txt`.

- **K1 holds, 12/12.** Re-keyed, "What's your name?" recalls its self episode at 0.998 with margin 0.43. On the
  exchange key the sister telling wins on 12/12.
- **K2 fails, 0/12.** On every home the told questions recall the "What's your name?" self episode, passing. The
  cat (both sessions), sister, car and work questions all do, so told recalls go from 4/5 to 0/5 per home.
- **K3 fails.** "How are you feeling today?" recalls the "Who are you?" self episode, passing, on 12/12 homes; on
  the exchange key it was 0/12.

The homes are near-identical draws: all 12 give the same numbers to four decimals. Within one session-1 path, the
stored set and its keys differ only slightly.

## One home, every cue (s200; S0 score of the winner)

| cue | exchange key | question key |
|---|---|---|
| Hi again! Do you remember my name? | sister telling 0.416 | friend self episode 0.831 |
| What's your name? | sister telling 0.459 | own-name self episode 0.998 |
| What is my cat called? | cat telling 0.490 | own-name self episode 0.666 |
| What is my sister's name? | sister telling 0.572 | own-name self episode 0.799 |
| What car do I drive? | Volvo telling 0.603 | own-name self episode 0.457 |
| What do I do for work? | Volvo telling 0.440 (wrong) | own-name self episode 0.577 |
| How are you feeling today? | Volvo telling 0.427 | "Who are you?" self episode 0.682 |

## What it shows

Under S0 (the centred layer-0 attention delta), a key read from a bare question sits closer to every question cue
than any telling does, and even to "How are you feeling today?". Which cause drives this is untested. It could be
sentence form, shared words ("name"), or the key text's shortness; no statement-form or content-matched control was
run.

The exchange key is imperfect too, by the same read-out. The work telling is stored ("Nice to meet you. I work as a
system administrator..."), yet "What do I do for work?" ranks the Volvo telling first and the work telling is not in
the top three. A question-only key cannot serve under S0 as built.

Skeptic (sonnet, fresh context, lens block verbatim): **NARROWED**. It found no probe bug: the re-key reads the
same text as `store(key_text=)`, and the re-keyed key matches the live cue read to cosine 0.998. The scorer matches
K1-K3. It asked for the form claim to be softened, which is done above. Its claim that no work telling exists is
wrong: the home holds it, with 8 episodes = 3 self + 5 tellings. Its other points: the 12 homes are one draw in
effect; a home born with the switch, with its own session-1 path, is untested; "Who are you?" for "How are you
feeling today?" is arguably defensible, but the prereg counts it as a fail. `where_ran`: unknown=4; each JSON's
`host` field records the node.

## Decision (from the prereg)

K2 and K3 fail: NOT A LEAD as built. The switch stays default OFF.

Next method: the one the self-recall skeptic named. S_all, the meaning key read from every layer, ranks the self
episode first for "What's your name?" on 6/6 slept homes. It keeps the told controls on 5/6, and only its 0.22 bar
fails. A bar refit for S_all on these slept homes needs its own prereg. It should fit on some homes and test on
others, because a bar fit and scored on the same homes would be circular.
