# Result: self episodes stored as their completion: own name right on day 1 on 6/6 seeds (2/6 off); session 2 unchanged; one day-1 cat answer lost; session 2 never recalls a self episode, so C3 is untested (NARROWED; C0, C1 hold, C2 fails by one seed)

Prereg: `brain/research/2026-11-15-self-completion-weekly-1.5b.md` (C0-C3, committed before any run).
Run: AWS batch 20261115-1522 (g6.xlarge L4, instance i-078c18933b6c1df27), 12 jobs, all rc 0 (`run.log`);
`brain.eval.weekly --model 1.5b --shared-sleep --seed <s>` with and without `--self-completion`, both arms in one
batch. Transcripts `sc{off,on}_s*.md` (the runner wrote them with a .json name; they are the weekly markdown).
Report-only: pool CPU fp32, seeds 200-205, same command (`pool/`). `tools/where_ran.py`: unknown (the transcripts
record no host; `unknown=25`); provenance is the batch log and instance id (prose only) and the pool job's done.txt.
The pool seeds are CPU fp32 and the registered seeds L4 GPU: the pool is not a like-for-like replication. Scorer
`brain/experiments/self_completion_score.py`; output in `table.txt`.

## Registered seeds (42/43/44/100/101/102)

| | day-1 own name | day-1 cat | session-2 keyed (intact, mean) | told facts (of 4) | "remember my name?" | session-2 own name | hippocampus off keyed |
|---|---|---|---|---|---|---|---|
| off | 2/6 | 5/6 | 5.17 | 3.50 | 6/6 | 4/6 | 5.50 |
| on | **6/6** | 4/6 | 5.17 | 3.50 | 6/6 | 4/6 | 5.50 |

Day-1 turn 1, off: "[name]", "David", "Nova", "Sam", "Bob", "Nova". On: "My name is Nova." on all six, with
"What's yours?" added on s42, s102.

- **C0 holds.** Every on header records "self episodes store their completion"; no off header does.
- **C1 holds.** Day-1 own name 6/6 on vs 2/6 off (bar >= 5/6 and >= off + 2).
- **C2 fails, by one seed.** Session-2 keyed means are equal (5.17 = 5.17, bar >= off - 0.5), but the day-1 "What is
  my cat called?" is right on 4 seeds on vs 5 off. The lost one, s43: "My cat is called Pixel." (first person, scored
  wrong; it recalled the right episode). The off arm said "Pixel."
- **C3 holds by its bar, but tests nothing.** "Do you remember my name?" in session 2 is right 6/6 in both arms;
  session 2 recalls no self episode in either arm (below), so the switch cannot act there. The smoke's "It's Nova."
  on the friend question (a day-1 risk) is not read by C3 at all.
- **Verdict: NARROWED.** Per the prereg: reported, no flip.

## What it says

- The switch does what the probe predicted: the brain names itself on day 1 on every seed.
- Session 2 (after the night) is byte-identical in both arms on all six seeds (`diff` from `## intact` on is empty).
  The night and the told facts are untouched, and so is session-2 self recall: "What's your name?" recalls "My
  sister's name is Marguerite." on all six seeds in both arms, and the self episode never wins after the night. The
  remaining own-name error, "My name is Dan." (s101, s102, both arms), comes with that selection error; whether
  recalling the self episode would fix it is untested.
- The day-1 cat loss is unexplained. The switch changes only the stored self episodes, both s43 arms recalled the
  cat telling, and the s43 conversation diverged from turn 1 on. The failing form "My cat is called Pixel." is an
  existing first-person mode: it also appears in the off arm (s44, pool s201/s203/s205). Over the 12 seeds the day-1
  cat answer is right 8 off vs 7 on and moves both ways, consistent with no systematic cost, not proof of it. It is
  still a registered failure.
- Reported, not criteria: the day-1 reply style shifts. On the registered seeds the on arm asks questions back in
  session 1 on every seed at least as often as off (replies with a "?": off 0/0/5/3/0/1, on 2/5/5/5/5/1; 23 vs 9;
  s43 on asks "What kind of games do you play?" twice), plausibly from "My name is Nova. What's yours?" in the context;
  on the pool seeds 18 vs 16. No reply echoes "What's your name?" back in either arm.
- Engagement: recorded only by the header (C0) and the changed turn-1 replies; no call count of the completion store.
  Answer rate: "I don't know about that." appears in 0 of 24 transcripts (grep; `tools/answer_rate.py` does not read
  this layout).
- Skeptic (sonnet, fresh context, lenses a-c): NARROWED; numbers, scorer and code claim confirmed; the wording on the
  cat loss, the reply style, C3 and the host is its narrowing.

## Report-only seeds 200-205 (pool, CPU)

Day-1 own name 2/6 off, 6/6 on; day-1 cat 3/6 both; session-2 keyed 5.33 both; "remember my name?" 6/6 both;
session-2 own name 4/6 both ("My name is Marguerite." s202, "My name is Dan." s203). By the same bars C0-C3 hold.
Not pooled with the registered seeds.

## Next

A fresh-seed replication with C0-C3 unchanged (`brain/research/2026-11-15-self-completion-replication-1.5b.md`,
seeds 300-305, AWS), pre-registered before it runs; the default flips only if it holds. Separately: session-2 self
recall loses to a told episode after the night on every seed; that selection error is the next lead on the
remaining own-name error.
