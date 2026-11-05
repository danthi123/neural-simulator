# Pi lane (pilot from 2026-10-31; lean method from 2026-11-02)

The supervisor feeds each `- [ ] ` line below to Pi with `tools/pi_driver/LANE_PROMPT.md`. Only Claude adds items.
The next item comes from `HANDOFF.md` or the board, as a build task for the new core (`brain/ARCHITECTURE.md`
section 4 milestones). Until Claude adds one, the lane is open and empty, and Pi waits.

## Archived (#203, frozen synapses; old track, closed by RULING pivot-1102)
Full text and evidence: `git show 6b4ff08e5:research/coordination/PI_LANE.md` (main) and the lane branch
`research/pi-lane` (its copy of this file holds the third item and the flip-smoke record).
- [x] #203 freeze: non-plastic synapses no longer drift, flag `cfg.enforce_plastic_mask_beyond_hebbian`, default OFF.
  Branch `research/pi-203-frozen-synapse-freeze` @ `fd42e0120`, merge_review APPROVED (2026-10-31).
- [x] #203 follow-up: the comprehension organ's frozen drift is the Hebbian loop (`sim/bridge.py` LTP + decay); fix
  on branch `research/pi-203-write-path-attribution`, default OFF, no reply-level change (2026-10-31).
- [x] #203 flip smoke (on `research/pi-lane`): stopped 2026-11-02 18:20: the in-flight GPU cells were killed by Claude
  (RULING overnight-1102). Nothing to land. No flip, no new #203 work.

## Open
- [x] New core, weekly conversation test upgrade (`brain/eval/weekly.py`, milestone day 14, board #281). Today it
  sums the intact and lesion arms into one total, asks the same fixed questions every week, and credits degenerate
  replies (e.g. "Nova, Nova."). Change it so that: (1) the header prints one score line per arm (intact, hippocampus
  off) and the difference; (2) a third arm runs sleep between the two sessions (`brain/sleep.py`, adapters from
  `brain/plastic_cortex.py`) and then asks with the hippocampus OFF, so the transcript shows what sleep consolidated;
  (3) half the facts are drawn each week from a larger list, seeded by the ISO week, so the score cannot improve by
  memorising fixed questions (keep the other half fixed for week-to-week comparison); (4) a reply counts only if it
  contains the key word AND is not degenerate (repeated token runs, empty, or only the brain's own name). Add unit
  tests that run without the cortex weights (mock `Cortex`); real runs go to W via `run_gpu`. Read first:
  `brain/eval/weekly.py`, `brain/experiments/sleep_probe.py`, `brain/PREREQUISITES.md` (Sleep consolidation).
  Done = branch + tests passing + one real run on W attached to a merge_review checkpoint.
  CLOSED 2026-11-05 (Claude): done on `research/pi-sleep-varied-replay` (merge_review APPROVE) but NOT merged:
  main's `weekly.py` was rewritten in parallel for 1.5B (both arms sleep, lesion in session 2, `--model`), and
  its CPU-episode design already fixes the device bug. Pi's ISO-week rotation and degenerate-reply flag are kept
  on that branch for a later item on top of main.
- [x] New core, sleep: varied-replay arm in `brain/experiments/sleep_probe.py` (after the weekly-test item). Six-seed
  0.1B result (`brain/experiments/results/2026-11-02-sleep-0.1b-six-seeds.md`): sleep recalls the original question
  (3-4/12) but not reworded questions (2.2 vs 1.3 shuffled). Add arm `sleep_varied`: each told fact is replayed in
  three surface forms, the original plus two rewordings of the STATEMENT (e.g. "My dog's name is Pixel." -> "I have
  a dog called Pixel."), same total replay steps as `sleep_true`. Write the rewordings once into a committed file
  `brain/experiments/facts_rewordings.py` (teacher-written text, declared in `brain/SCAFFOLDS.md` like text replay);
  they must not reuse the wording of any `PARA` question (add a unit test that checks no PARA question's content
  words appear as a run of 3+ in any rewording) and must keep the key word. Add `--arms` (default: all four) so a run
  can skip arms. Unit tests run without weights (mock `Cortex`, as the weekly item). Do not run the real probe:
  Claude runs the six seeds. Read first: `brain/experiments/sleep_probe.py`, `brain/sleep.py`,
  `brain/experiments/facts.py`. Done = branch + tests passing + merge_review checkpoint.
  CLOSED 2026-11-05 (Claude): superseded. A Claude-built varied-replay arm landed on main first (0d4f5d120,
  six seeds at 0.1B: NOT SHOWN). Pi's duplicate on `research/pi-sleep-varied-replay` is not merged (Claude's
  lane bookkeeping error: the item stayed open after the 11-03 steer).
- [x] New core, weekly test: perspective for the two keys seeded at birth (base: a new branch from `main`, where `research/landing-1103` is merged;
  if the weekly-upgrade item above has landed by then, apply this on top of it). `Brain._grow_self` stores the
  brain's own name (Nova) and the user's name (Dan) at birth, so these two keys are not told in the session.
  `weekly.TOLD` gives both `None`, and with `None`, `strict_scorer.is_echo` accepts ANY first-person reply. That is
  right for Nova ("I'm Nova.") and wrong for Dan: "My name is Dan." / "I'm Dan." (the brain says the user's name
  is its own) are marked correct today. In the other direction, "Your name is Nova." (the brain gives its own name
  to the user) is also marked correct. The fix: (1) in `brain/experiments/strict_scorer.py` add
  `wrong_perspective(reply, key, whose)`. Split the reply into clauses as `is_echo` does. For `whose="self"` (a fact
  about the brain), it is wrong when a clause holding the key has a second-person word (`_SECOND`) and no
  first-person word. For `whose="user"` (a fact about the user), it is wrong when a clause holding the key has a
  first-person word and no second-person word. (2) Give `correct_answer` an optional `whose=None` that rejects
  wrong-perspective replies. With `None` the behaviour is exactly today's, and every existing test in
  `brain/tests/test_strict_scorer.py` and `test_weekly_scorer.py` passes unchanged. (3) In `brain/eval/weekly.py`:
  `WHOSE = {"nova": "self", "dan": "user"}` and `SEEDED = {"dan", "nova"}` (comment: stored by `_grow_self`, not told
  in the session); `mark_for` passes `whose`; a seeded key's mark gets " (seeded at birth, not told today)"; the
  printed total counts told-today keys and seeded keys separately ("told today x/4, seeded y/2"). Tests (no
  weights): Nova: "I'm Nova." and "My name is Nova, and you're Dan." are True, "Your name is Nova." and "You're
  Nova." are False. Dan: "Of course, you're Dan, my friend.", "Dan." and "I remember you, Dan." are True, "I'm Dan."
  and "My name is Dan." are False. Add a stub-Brain `run()` test that checks the seeded mark and the split total.
  Do NOT merge `brain/eval/scoring.py` from `research/weekly-dry-run-1103`: its first-person flag is replaced by
  `correct_answer`. Only its "seeded" mark idea is carried over. Read first: `brain/eval/weekly.py`,
  `brain/experiments/strict_scorer.py`, `brain/brain.py` (`_grow_self`), and the audit in
  `git show research/weekly-dry-run-1103:brain/eval/transcripts/dry-run-2026-11-03.md`. No real run is needed.
  Done = branch + tests passing + merge_review checkpoint.
  DONE 2026-11-05 (Pi): built on `research/pi-lane-1105` (f81324fa1). `strict_scorer.wrong_perspective(reply, key,
  whose)` + `correct_answer(..., whose=None)` (with None the earlier behaviour is exact: every existing test in
  brain/tests/test_strict_scorer.py and test_weekly_scorer.py passes unchanged); weekly.py `WHOSE`/`SEEDED`, `mark_for`
  passes `whose`, a seeded key's mark gets " (seeded at birth, not told today)", per-arm total "told today x/4, seeded
  y/2" (keys, last scored ask). Tests without weights (brain/tests/test_seeded_keys.py): the spec's Nova/Dan cases,
  whose=None unchanged, stub-Brain run() for the seeded mark and the split total; 62 brain/tests pass,
  tests/test_brain_adopt_config.py's weekly test passes (17 passed). brain/tests/conftest.py stubs `rwkv` only where
  the package is absent, so the two protected test files run unchanged. Note: `research/weekly-dry-run-1103` no longer
  exists in either repo (the audit is unreachable); only its "seeded" mark idea was needed and it is carried over. No
  real run (spec: none needed). merge_review checkpoint requested (branch + sha f81324fa1).
- [x] New core, live-brain 12 facts: rescore and rerun with the correct-perspective scorer (base: a new branch from `main`;
  `research/landing-1103` is merged into main). The six-seed result (`brain/experiments/results/2026-11-03-live-brain-12facts.md`) was
  scored with `strict_scorer.score`, which voids every first-person reply and credits tautologies ("Pixel is called
  Pixel."); its caveat section says so. `strict_scorer.correct_answer(reply, key, question, told)` fixes both. Its
  pre-registration says a changed instrument means everything is rerun (see the scope cut below). (A) New
  `brain/experiments/live_brain_rescore.py`: read `seed*.json` in a results dir, and for every `told`/`para` row in
  every arm compute `correct_answer` with key and told sentence from `FACTS` by position (told rows and para rows
  are both in `FACTS` order; arm e is scored against the ORIGINAL keys and sentences, as the probe does). Write
  `rescored.json` and `table_correct.txt` next to them; never modify the seed JSONs or `table.txt`. The table is
  `live_brain_table.py`'s P1-P4 logic with the metric names as parameters (refactor `main(d)` to take them; the
  old call must print the same `table.txt` byte for byte, which a test checks). False recall is unchanged. SCOPE CUT 2026-11-05 (Claude): part (A) only. Part (B) is already on main (`live_brain_probe.py` writes
  `recall_correct`/`paraphrase_correct`, used by the 1.5B six seeds), and the 0.1B rerun is dropped: the 1.5B live
  brain (`results/2026-11-03-live-brain-1.5b.md`) is now the decision point. Unit tests without weights: rescore on
  a tiny fixture JSON (echo, tautology, "I believe you drive a Volvo" counted, wrong-subject), plus the
  byte-identical `table.txt` check. Read first: `live_brain_probe.py`, `live_brain_table.py`, `strict_scorer.py`,
  `brain/tests/test_strict_scorer.py`. Pi does not write a verdict; the table prints the P-checks and Claude rules on
  them. Done = branch + tests passing + `rescored.json` + `table_correct.txt` + merge_review checkpoint.
  DONE 2026-11-05 (Pi), part (A) per the scope cut: `brain/experiments/live_brain_rescore.py` rescores every
  told/para row with `correct_answer` (key + told from FACTS by position; arm e against the ORIGINAL keys/sentences;
  false recall unchanged) and writes `rescored.json` + `table_correct.txt` (run on the real six-seed dir, both
  committed; seed JSONs and table.txt untouched, test-pinned). `live_brain_table.py` refactored: `main(d)` takes the
  metric names as parameters and the old call prints the committed `table.txt` byte for byte (test); the CLI keeps
  its "scorer <x>" line via `header`. Tests without weights (brain/tests/test_live_brain_rescore.py: fixture echo /
  tautology / "I believe you drive a Volvo" counted / wrong-subject, byte-identical table.txt, write-only-new-files);
  66 brain/tests pass. Rescored P-checks: P1 True, P2 False, P3 False, P4 True (table_correct.txt) — Pi writes no
  verdict, Claude rules. Note: the rescored recall is far below the strict score (a_intact_sleep 6.50 -> 1.83 mean):
  the strict score's tautologies and wrong-subject hits are gone. merge_review checkpoint requested (sha 20ffe2f9a).
- [x] New core, mood: refusal and paired-echo re-analysis of the existing iteration-3 replies (base: a new branch from `main`;
  `research/landing-1103` is merged into main; no model, no new runs). `brain/experiments/results/2026-11-03-mood3.md` leaves two
  instrument questions open. (i) Part of "negative mood darkens" is refusals ("I'm sorry, but I can't help with
  that."): 0.1B neg 35/144 vs off 9/144. (ii) The 0.1B fail is only C6 echo (s43 +0.104, s101 +0.125), and the off
  arm itself echoes 0.12-0.50. Build: (1) in `brain/experiments/sentiment_instrument.py` add `REFUSAL_PATTERNS`
  (fixed now: "i'm sorry", "i am sorry", "i can't help", "i cannot help", "i can't assist", "i'm unable", "i am
  unable", "as an ai"; case-insensitive, straight or curly apostrophe) and `is_refusal(reply)`. (2) In
  `brain/experiments/mood_v3.py` add a subcommand `reanalyse <dir> <size>` that reads the existing
  `mood3_<size>_s*.json` rows (arms pos, neg, off and the random arms the null uses in `mood_probe.py`, paired per
  (q, k) exactly as `mood_probe.main` builds the `pos_null`/`neg_null` contrasts) and prints per seed: refusal counts
  per arm; lex pos-null and neg-null (a) as stored, (b) recomputed with the word "sorry" removed before
  `si.lexical`, (c) on the (q, k) pairs where neither the mood reply nor any of its null replies is a refusal (n
  pairs shown); and the paired echo: mean over (q, k) of `primary_echo(mood reply) - primary_echo(off reply)`, per
  sign, next to the registered `echo mood-off`. CIs use `mood_probe.boot_ci` with `np.random.default_rng(0)`. Write
  the output to `brain/experiments/results/2026-11-03-mood3/reanalysis_<size>.txt` (new files only). Tests: the
  `is_refusal` cases; and with no filter, (a) reproduces the stored `contrasts.pos_null.lex_diff` and
  `neg_null.lex_diff` to 1e-6 for all 12 JSONs (checks the pairing). BEFORE running it on the real JSONs, commit
  `brain/experiments/results/<date>-mood3-reanalysis.md` with the definitions above, marked "exploratory
  re-analysis of existing replies, not a verdict; no criterion or bar changed". Then add the tables. No PASS/FAIL
  words: any new bar or rerun at 36 rows is Claude's call through a checkpoint. Read first: `mood_v3.py`,
  `mood_probe.py` (contrasts, `boot_ci`), `sentiment_instrument.py`, `brain/research/2026-11-03-mood-state-design.md`
  sections 12-13. Done = branch + tests passing + both size tables in the writeup + merge_review checkpoint.
  DONE 2026-11-05 (Pi): `sentiment_instrument.REFUSAL_PATTERNS`/`is_refusal` (the fixed list) +
  `mood_v3.reanalyse <dir> <size>` built and run on both sizes; definitions writeup committed BEFORE the run
  (`brain/experiments/results/2026-11-05-mood3-reanalysis.md`, d776b8555), tables added after
  (`reanalysis_0.1b.txt`/`reanalysis_0.4b.txt` + writeup, e3b76d9b6). Tests: the is_refusal cases +
  `lex_null_diffs` reproduces the stored `contrasts.pos_null/neg_null.lex_diff` to 1e-6 on all 12 JSONs (pairing
  check); 68 brain/tests pass. Anchors agree with the 11-03 record (0.1B refusals neg 35/144 vs off 9/144; the two
  0.1B registered echo values over 0.10 are s43 +0.104 / s101 +0.125). No PASS/FAIL words anywhere; the numbers' reading
  is Claude's call. Note: `mood_probe.py`'s brain.affect/brain.brain/brain.cortex imports moved into `main()` so
  `boot_ci` imports weight-free (behaviour identical with the model). merge_review checkpoint requested (sha e3b76d9b6).
