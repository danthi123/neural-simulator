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

---
# RESULTS (six seeds, 0.1B, CPU; numbers from `2026-11-03-live-brain-12facts/seed*.json`, table by `live_brain_table.py` -> `table.txt`)
Instrument unchanged from the pre-registration (one commit, run once; seed 43 started late because of the memory gate, same code).
Per-arm mean over seeds [per seed 42,43,44,100,101,102]. Strict scorer.

| arm | recall /12 | paraphrase /12 | false recall /4 | echo | lenient recall |
|---|---|---|---|---|---|
| a intact + sleep | 6.50 [5,6,9,6,7,6] | 5.50 [5,5,6,5,6,6] | 3.00 [3,3,3,3,3,3] | 5.5 | 10.2 |
| b intact, NO sleep | 6.33 [6,6,6,8,6,6] | 3.83 [4,3,5,4,3,4] | 2.00 [2,2,2,2,2,2] | 4.5 | 8.8 |
| c hippocampus off after sleep | 5.50 [5,5,5,6,5,7] | 3.33 [3,4,2,4,3,4] | 3.67 [4,4,2,4,4,4] | 5.2 | 8.0 |
| d hippo off + consolidated off | 0.17 [0,0,1,0,0,0] | 0.00 | 0.00 | 0.2 | 0.3 |
| e shuffled sleep, hippo off | 0.00 [0,0,0,0,0,0] | 0.50 [0,0,1,1,0,1] | 3.17 [3,4,2,4,4,2] | 0.3 | 0.2 |

Sleep adds over the no-sleep intact arm (a-b): recall +0.17, paraphrase +1.67, false recall +1.00 (a is higher on every seed).

## Criteria
- P1 PASS: c > d and c > e on every seed (c 5-7/12 vs d 0-1, e 0); mean c 5.5. Consolidated synapses alone answer, the lesion (d) removes it, the shuffled control gives 0 (not a broken test).
- P2 FAIL: a >= b on only 4/6 seeds (seed 42: 5 vs 6; seed 100: 6 vs 8); mean a-b = +0.17 (needed >= 1). On the original questions the intact brain without sleep recalls as much as with sleep, because the hippocampus already answers them.
- P3 FAIL: false recall a 3.0 vs b 2.0 (limit b+0.5); c 3.67 vs e 3.17 (c above shuffled). Sleep adds about one false recall of four never-told questions on every seed, and a no-sleep brain already confabulates 2/4 (hippocampal cue on a similar question).
- P4 PASS (not required): c paraphrase 3.33 vs e 0.50, but note a (5.5) vs b (3.83): the +1.67 paraphrase gain comes with sleep when the hippocampus is on.

## Honest verdict
Not a lead under the pre-registered rule (P2, P3 fail). What is established: (1) with the hippocampus off, sleep leaves a real consolidated trace in the live brain: 5.5/12 strict correct-perspective answers vs 0.17 with consolidation off and 0 for shuffled content, six of six seeds (the in-context ceiling at 0.1B is 6/12). (2) The skeptic's missing control shows sleep adds little on the original questions (+0.17) because the intact no-sleep brain answers 6.3/12 via the hippocampus; its measurable contributions are paraphrase (+1.7) and independence from the hippocampus. (3) Cost: false recall up (+1.0 intact, 3.7/4 with the hippocampus off, no better than shuffled 3.2/4): learned names leak into unrelated answers. (4) Echo is common (4.5-5.5 of 24 answers in intact arms; seed 42 sleep 10-11): strict recall is 2/3 of lenient in the intact arms. Caveats: one sampled reply per question; false recall counts any fact word in a never-told answer.
Next step: put the familiarity bar (false-recall-1103) in front of the consolidated path, then re-run this probe unchanged; and a lesion that changes the reply is shown by d vs c (5.3 answers lost).

## Scorer caveats (added in integration-final, 2026-11-03; the numbers above are unchanged and were NOT re-scored)
1. The strict scorer (`strict_scorer.score`) voids ANY reply containing a first-person word, including ones that are not an echo
   ("I think you drive a Volvo"). It can under-count correct answers; its counts are a lower bound on correct-perspective recall.
2. It accepts tautologies such as "Pixel is called Pixel." and "Marguerite is Marguerite." (the answer word is present, nothing
   was recalled about the thing asked), so it can over-count, most in the intact arms where the hippocampus reinstates the
   episode text. The two errors push in opposite directions; the net effect on the table was not measured.
A corrected scorer (`strict_scorer.correct_answer`: echo = first-person reply restating the told sentence; tautologies
rejected) now exists and is used by the sleep smoke and the weekly transcript. Re-scoring these JSONs with it is a
possible follow-up; until then read the table as: hippocampus-off consolidation 5.5/12 vs 0.17 (lesion) vs 0 (shuffled)
is a real contrast, and absolute counts are approximate.
