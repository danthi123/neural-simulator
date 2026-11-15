# Result: own name on day 1 at 1.5B: the recalled self episode is too weak at gain 0.5, and its stored question is what garbles it at higher gain (D0, D1, D3, D4 hold; D2 fails)

Prereg: `brain/research/2026-11-15-own-name-day1-probe-1.5b.md` (D0-D4, written before the 1.5B run).
Run: pool40 (`host` node-mxl1462drl in the JSON), CPU fp32, `brain.experiments.own_name_probe --model 1.5b
--samples 12`, rc 0. Data: `2026-11-15-own-name-day1-probe-1.5b/on_1.5b.json`. No training, no sleep: a fresh brain's
birth state (asserted equal to the turn-1 state), the brain's own recall, 12 samples per arm with the brain's own
`speak` defaults (seeds 1000-1011). One brain, one node: a diagnosis of the mechanism, not a six-seed claim.

## Live prompt "Hi! What's your name?" (what the brain hears on day 1)

| gain | full delta (as stored now): samples naming Nova /12 | P(Nova after "My name is") | typical reply | answer-only delta: /12 | P(Nova) |
|---|---|---|---|---|---|
| 0 | 0 | 0.0007 | "Hi! I'm ChatGPT, a large language model..." | | |
| 0.25 | 0 | 0.015 | "My name is John." | 3 | 0.050 |
| **0.5 (now)** | **5** | 0.116 | "My name is Nova." 5, "NAME_1", "Brain" | **12** | 0.478 |
| 0.75 | 12 | 0.498 | "My name is Nova." | 12 | 0.827 |
| 1.0 | 12 | 0.653 | "My name is Nova." | 12 | 0.931 |
| 1.5 | 12 | 0.803 | "My name is Nova." | 12 | 0.979 |
| 2.0 | 12 | 0.847 | "My name is Nova." | 12 | 0.991 |
| 3.0 | 2 | 0.950 | "What's your name?" (10 of 12) | 12 | 0.997 |
| 4.0 | 12 (garbled) | 0.987 | "Nova, Nova, Nova." / "Nova, what's your name?" | 12 | 0.998 |

Answer-only = d(question + answer) - d(question), norm-matched to the stored delta.
Bare prompt ("What's your name?"): full 9/12 at 0.5, 12/12 at 0.75-2, 0/12 at 3 ("What's your name?"), answer-only
12/12 at 0.5-4.

## Criteria

- **D0 holds.** Both prompts recall the self episode ("User: What's your name? / Assistant: My name is Nova.",
  similarity 0.48); gain 0 names Nova in 0/12.
- **D1 holds (strength).** Gains 0.75, 1, 1.5 and 2 name Nova in 12/12 with the plain sentence "My name is Nova.";
  0.5 names it in 5/12. Five of 12 matches the weekly's 2/6 seeds.
- **D2 fails (not the frame).** At 0.5 all 12 replies start "My name is"; the name after it is the weak part
  (P 0.12, rank 1).
- **D3 holds, at the bar.** Bare 9/12 vs live 5/12 at 0.5 (+4). The greeting weakens the cue's effect.
- **D4 holds.** Answer-only names Nova in 12/12 at 0.5 (+7) and stays clean at every gain up to 4.

## What it says

- The day-1 own-name error is a strength problem: the self episode is recalled, but at gain 0.5 (fitted on told
  facts) it moves the name only to P 0.12, and the top_p 0.3 sampler then picks a prior name most of the time.
- Raising the gain fixes it only in a window (0.75-2). Above it, the stored delta replays its own question: at 3 the
  brain asks "What's your name?" back; at 4 it loops on "Nova". That is the old "identity traces loop above ~4.5"
  note in `hippocampus.py` explained: the stored self episode holds the question as well as the answer.
- Storing the completion only (what the answer added after the question) carries the name at the current gain 0.5 and
  never garbles up to 4. It matches what recall should return (the completion; the cue is already in the input) and
  needs no new gain number.
- Told facts are plain tellings (no question part), so an answer-only store changes only question/answer episodes:
  the three self episodes, and lived episodes when `episode="lived"` (default off).

## Next

Method test, pre-registered before any run: store the self episodes as their completion (switch default OFF), six
registered seeds of the weekly on AWS with an explicit off arm, criteria on the day-1 own name and no cost on told
facts or "Do you remember my name?". The higher-gain-for-permanent-episodes alternative is not chosen first: it adds
a host number and works only inside a window.
