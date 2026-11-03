# Night of 2026-11-02/03: what was tested, in one page

For the owner. Every number below is copied from the writeup on the named branch (read with `git show <branch>:<path>`),
not from memory. "Six-seed" = seeds 42/43/44/100/101/102 unless the row says otherwise. Recall counts are told facts the
brain answers with its hippocampus OFF (so the answer must come from what sleep wrote into the cortex), unless stated.
Nothing below is a GO: the best results are leads.

| # | Track | Question | Verdict | Key number | Branch |
|---|---|---|---|---|---|
| 1 | Overnight sleep runs (mini-PCs) | Does sleep get better with a bigger cortex (0.4B), a stronger drift guard (kl 3) or more steps (480)? | lead (0.1B 480 steps: more recall, more false recall); 0.4B under-trained at 0.1B settings | 0.1B 480 steps: recall 4.5/12, paraphrase 3.0/12, but false recall 3.0/4 (shuffled control 2.8/4). 0.4B at 0.1B settings: 2.3/12 (seed 43: 0/12). More consolidation = more recall AND more made-up names. | research/overnight-results-1103 |
| 2 | Sleep inside the live brain | When the brain restarts, does it sleep, and can it then answer from sleep alone? | lead (wiring works; small test, 3 facts) | Hippocampus off: 8/18 told, 8/18 reworded (strict scorer; the first 15/18 was an echo-counting scorer, corrected); consolidation also off: 0/18. False recall 4/6. Only 2 of the 3 facts ever answered cleanly. | research/sleep-in-brain-1103 + scorer fix on research/integration-final-1103 |
| 3 | False recall: feeling of knowing | Can the brain tell a real memory from a made-up one without new training? | lead (question familiarity); NO-GO (answer confidence) | How sure the answer sounds does not separate them (name-token AUC 0.47). How familiar the QUESTION is does: hippocampal score AUC 0.88 (keeps 81-84% of correct, removes 71-72% of made-up); cortex cue familiarity AUC 0.78-0.79. Fitted on one question template. | research/false-recall-1103 |
| 4 | High-use facts, part 1 | Does a fact told 10 times become never-wrong after sleep? | NO-GO (this method, 0.1B) | k=10 facts 4/4 on 0 of 6 seeds (best 3/4); use arm total 4.8/12 vs equal-replay control 3.8/12, worse than control on 2 seeds. | research/high-use-facts-1103 |
| 5 | High-use facts, part 2 (reachable facts, use counted in the hippocampus) | Same, using only facts the cortex can hold, with use counted by re-telling | NO-GO (pre-registered bar 6/6); weak lead for "use helps" | k=10 4/4 on 1 of 6 seeds (bar 6/6). k=10 recall use 2.33 vs equal 1.33 of 4 (4 up, 2 ties, none down). Rarely told facts lose (k=1: 1.33 vs 2.00). False recall ~3/4. | research/high-use-ceiling-1103 |
| 6 | Cortex size | Which cortex size should the brain use? | lead: 1.5B provisionally; 0.4B is the only size with six seeds | 0.4B tuned (lr 1e-3, 480 steps, kl 3): 4.0/12 mean, six seeds (range 1-6), false 2.0/4. 1.5B (lr 5e-4): 7.0/12 on 3 seeds only (3, 7, 11), false 1.7/4; fits the 3090 (~19.8 GB). In-context ceiling 11/12 for both. | research/cortex-size-1103b |
| 7 | Live brain, 12 facts | With 12 facts told in conversation, does sleep help, and does it cost false recall? | NOT PASS (P1 pass, P2 and P3 fail) | Sleep alone answers: hippocampus off 5.50/12 vs consolidation off 0.17 and shuffled sleep 0.00, 6/6 seeds. But sleep adds only +0.17/12 over a no-sleep brain on the same questions (needed +1); paraphrase +1.67; false recall up from 2.00 to 3.00/4. | research/live-brain-12facts-1103 |
| 8 | Meaning-based selection (probe) | Which memory key picks the right memory for a reworded question? | lead | All-layer key (S_all) vs default (S0), heavy rewording, forced choice: 0.4B selection 0.76 vs 0.10, recall 0.66 vs 0.05; 0.1B selection 0.69 vs 0.33. Better on 6/6 seeds at both sizes. | research/meaning-selection-1103 |
| 9 | Meaning-based selection (real hippocampus path, fix) | Does S_all still win through the real recall code, with its own familiarity bar? | lead (all six pre-registered checks pass at both sizes; defaults not flipped) | Refit bar, heavy rewording recall S_all vs S0: 0.14 vs 0.03 (0.1B), 0.12 vs 0.00 (0.4B), 6/6 seeds. Memory per episode 3.58 -> 1.18 MB (0.1B). But the bar drops reworded recall from 0.66 to 0.12 and still fires on 15-22% of never-told questions: the key picks WHICH memory, not WHETHER to recall. | research/meaning-selection-fix-1103 |
| 10 | Mood, iteration 1 | Does an induced mood change the reply to the same question, with no mood words added? | prototype pass, 0.1B (positive side weak) | Lexical tone positive-minus-negative +5.77 per 100 words (random direction +1.38); lesion makes replies byte-identical 6/6. Positive events read positive on only 0.50 and 0.12 on two seeds. | research/mood-state-1103 |
| 11 | Mood, iteration 2 | Fixed positive appraisal: does positive mood now brighten replies? | NOT PASS (6/7 criteria) | Negative mood darkens 6/6 (mean -4.97 vs off); positive brightens only 3/6 (bar 5/6), mean +1.36. Held-out appraisal fixed 5/6. | research/mood-fix-1103 |
| 12 | Mood, iteration 3 (neutral-ish questions; 0.1B and 0.4B) | Does positive mood brighten on a mid-tone register, and at 0.4B? | NOT PASS at both sizes; lead at 0.1B, negative at 0.4B | 0.1B: positive brightens vs random 6/6 (+3.03), negative darkens 6/6 (-4.08); fails only the echo check (4/6, bar 6/6). 0.4B: positive vs random 1/6 (-0.24). Part of "darker" is refusals ("I'm sorry, I can't help"). | research/mood-04b-1103 |
| 13 | Abstention v1 | Can the live brain hold back an answer it does not have? | NOT PASS (both methods) | Gate: false recall 63 -> 34 (hippocampus on), 46 -> 38 (off); target was a cut to 1/4 on 6/6 seeds, met on 0/6. Replayed "I don't know" pairs: false recall 46 -> 7 but told answers 21 -> 3 (a memorised string). | research/abstain-in-brain-1103 |
| 14 | Abstention v2 | Meaning-based familiarity + one taught "I don't know" lesson | NOT PASS (every arm); lead for the taught lesson | False recall 64 -> 44 (needed a cut to 1/4; 0/6 seeds). When the gate fires, learned-name false recall is 0/50 (the signal, not the gate, is the problem). Taught lesson gives "I don't know"-type replies in 27/50 gated cases, but often calls the user "Nova". | research/abstain-v2-1103 |
| 15 | Varied replay | Does replaying 3 wordings per fact help reworded recall? | no result (in progress) | Code and tests only (sleep_varied arm, teacher rewordings); no run yet. | research/varied-replay-1103 |
| 16 | Weekly test dry run | Does the weekly conversation test run, and can its transcript be trusted? | no result (dry run); scorer fixed | 4 h on the loaded CPU (idle estimate 10-20 min, unmeasured). Fixed scorer: told facts intact 3/4, lesion 4/4, clean answers 0/4 and 1/4. The lesion arm is not a hippocampus test (both arms sleep); a hippocampus-on, sleep-off arm is needed. | research/weekly-dry-run-1103 |
| 17 | Live brain 12 facts at 0.4B | Does the live brain do better at 0.4B? | no result | Not finished: 2 seeds had not completed their first sleep after 2 h 45 min on the shared CPU; stopped. Needs the GPU or AWS lane. | research/live-brain-04b-1103 |
| 18 | Pi wait-cancel (tooling) | Stop Pi sitting blocked on GPU jobs that no longer exist | done (tool fix, not an experiment) | Pi had sat blocked 8 h. New cancel_wait tool; stale GPU waiters close after two "job gone" polls; wait time capped at 8 h. | research/pi-wait-cancel-fix-1103 |

## The pattern across the night

1. **Sleep writes real memories into the cortex.** With the hippocampus off, the live brain answers 5.5/12 told facts
   (row 7) and the consolidation lesion takes that to ~0. That part is solid on six seeds.
2. **The main defect is made-up recall.** Every way of adding consolidation (more steps, more use, bigger model) also
   adds learned names answering questions that were never told (rows 1, 5, 7). Two abstention attempts failed for
   the same reason: the "do I know this?" signal does not yet generalise to new question wordings (rows 13, 14).
3. **The fix for reworded questions is found but not yet in the brain.** The all-layer key (S_all) picks the right memory
   far better, especially at 0.4B (rows 8, 9). Deciding whether to recall at all needs a separate signal.
4. **Mood changes tone through the brain's own state** (lesion-proven at 0.1B), negative more reliably than positive;
   at 0.4B the positive side is not yet distinguishable from a random push (rows 10-12).

## What this means for the milestones (`brain/ARCHITECTURE.md` §4)

**2026-11-09 (first weekly conversation test; upgraded test with intact / hippocampus-off / sleep arms).** On track,
with one fix. The weekly test runs end to end and its scorer no longer credits echoes (row 16). The intact-vs-lesion arms
currently measure sleep, not the hippocampus; a hippocampus-on, sleep-off arm must be added before 11-09. Run it on the
GPU or an idle host (4 h on the loaded CPU).

**2026-11-16 (sleep on every restart; high-use facts never wrong; a lesion visibly changes replies; cortex size chosen).**
- Sleep on every restart: built and working on branch (rows 2, 7), not yet on main.
- A lesion visibly changes replies: met on branch (consolidation lesion 5.5 -> 0.17 of 12, row 7; mood lesion, row 10).
- Cortex size: a provisional pick is ready (1.5B at lr 5e-4), but it needs seeds 100/101/102 (~90 min GPU) before it is
  final; fall back to 0.4B if the 1.5B mean drops to 4 or below (row 6).
- **High-use facts never wrong: at risk.** NO-GO twice (rows 4, 5): 10x exposure does not make a fact reliable at 0.1B.
  Next methods to try: more steps or a larger adapter on reachable facts, a replay floor for rarely told facts, and the
  1.5B cortex. This is the milestone most likely to slip.

**2026-11-30 (reworded recall; mood shapes replies, lesion-proven; asks about what is new).**
- Reworded recall: the method exists (S_all, row 9); remaining work is wiring it into the live brain with its own bar
  and a live-brain test.
- Mood: lesion-proven tone change at 0.1B; positive mood at 0.4B and the refusal confound remain; not wired into the brain.
- "Asks about what is new to it" depends on knowing what it does not know, which is still failing (rows 3, 13, 14). The
  next methods are a cue-completion recollection check and storing the "I don't know" lesson in the brain's own voice.
  This item is at risk.
