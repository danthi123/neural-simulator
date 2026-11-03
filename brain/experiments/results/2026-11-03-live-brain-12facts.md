# Live-brain 12-facts consolidation test (2026-11-03) -- PRE-REGISTRATION (written before any run)

Branch research/live-brain-12facts-1103 (from research/sleep-in-brain-1103). Probe: `brain/experiments/live_brain_probe.py`;
scorer: `brain/experiments/strict_scorer.py` (unit tests `brain/tests/test_strict_scorer.py`). Model RWKV-7 0.1B, CPU,
sleep 240 steps, kl_weight 1.0, lr 5e-4, rank 8 (Brain defaults). Seeds 42/43/44/100/101/102. Results `2026-11-03-live-brain-12facts/seed<N>.json`.

## Design
Day 1: the live Brain is told the 12 FACTS in conversation, saved, not closed. Restart = a new Brain on the same home.
Fresh Brain per question (no carry-over), replies sampled as in chat, np/torch seeded per question.
Arms: (a) intact + sleep; (b) intact, same day-1 home copied BEFORE the restart, sleep_on_restart=False (the missing control);
(c) hippocampus off after sleep; (d) hippocampus off + consolidated adapters off; (e) day 1 told with answer words swapped
between facts (derangement, as sleep_probe), slept, hippocampus off, scored against the ORIGINAL keys.
Questions: 12 originals, 12 PARA, 4 NEVER_TOLD. Per arm: recall /12, paraphrase /12, false recall /4 (any of the 12 key words
as a word in a never-told answer), echo count (key present but reply first-person), plus lenient substring counts.
Strict hit = key word present as a word start, not in the question, and no first-person word (I/my/me/mine/I'm...) in the reply.

## Pass criteria (fixed now; the instrument is not changed after the first run, else everything is rerun)
- P1 (sleep alone answers): strict recall of (c) > (d) AND > (e) on every one of the six seeds, and mean (c) >= 2/12.
- P2 (sleep adds over the intact control): strict recall of (a) >= (b) on at least 5/6 seeds and mean(a) - mean(b) >= 1.
- P3 (false recall): mean false recall (a) <= mean (b) + 0.5, and mean (c) <= mean (e).
- P4 (paraphrase): mean paraphrase (c) > mean (e) + 1 (reported; not required for the lead).
- Echo counts are reported, never used to excuse a failure.
Verdict rule: P1 and P2 and P3 all hold = "lead" (never GO from one probe); any fails = reported as that criterion failing.
Known limits stated now: single sample per question (sampled replies), 0.1B in-context ceiling 6/12, hear() replies are sampled.
