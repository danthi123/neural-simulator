# M3: retrieval practice in sleep, hippocampal episode as teacher (0.1B, CPU) (2026-11-03)

Plan: `brain/research/2026-11-03-confabulation-next-methods.md`, method M3. Branch `research/confab-m3-1103`.
Capability: Sleep consolidation + Metacognition (confabulation). Code: `brain/sleep_retrieval.py` (mechanism),
`brain/experiments/retrieval_sleep_probe.py` (probe), `tests/test_brain_sleep_retrieval.py` (5 unit tests, mock cortex).

## Deviation from the plan, stated up front
The plan specifies the decisive test at 0.4B on GPU and says "do not run M3 at 0.1B" (half the 12 old facts are
unreachable in context there). This task assigns 0.1B on CPU (GPU busy). Mitigation: the test uses only REACHABLE
facts (the first 12 of `ceiling_probe.POOL` whose original question the 0.1B cortex answers in context,
`2026-11-03-high-use-ceiling/ceiling_0.1b.json`: lyon marguerite biscuit cello peanut oslo zephyr gustav alvarez stamp
maple iceland), so every fact has a correct hippocampal teacher in principle. Even so, a negative here is read against
scale first (`brain/PREREQUISITES.md`): M3 is not called dead until the 0.4B GPU run in the plan (step 4) is done.

## Mechanism as built (plan wording -> implementation)
- "the cortex (hippocampus off) first answers the episode's own question-form cue (sampled by the cortex itself from
  the episode, not a host template)": before sleep, for each episode the cortex poses 4 questions about it
  (`SELFQ_PROMPT`, the already-declared self-question scaffold of the meaning-selection key; 1 greedy + 3 sampled at
  T 1.0, top_p 0.7, deduplicated). In sleep the cortex from the birth state with its slow synapses only answers one of
  these cues greedily.
- "the loss is applied on the answer span against the episode's content": the target is the answer the cortex gives
  to the same cue WITH the episode reinstated (told turn in context, slow synapses at birth): the hippocampal teacher.
  Cross-entropy on the answer span only.
- "weighted by the cortex's error (large when it said a wrong learned name)": per-token weight 1 + (1 - p(target)),
  normalised over the span (moves learning to the wrong tokens without changing the step size relative to
  `sleep_true`); and an episode whose last attempt failed (greedy answer diverged from the teacher's) is replayed 2x as
  often as one that succeeded. The attempt and the loss come from one forward pass (greedy decoding is deterministic,
  so divergence of the free answer = argmax mismatch under teacher forcing; unit-tested).
- "KL guard unchanged": same three ordinary-chat snippets, same weight, as `brain/sleep.py`.
- Host code: the self-question prompt (declared scaffold, `brain/SCAFFOLDS.md`, row added with this commit) and the
  gradient-descent sleep update (existing declared trade). No answer text is written by the host.

## PRE-REGISTRATION (committed BEFORE any of the six-seed runs; instrument fixed)
- Instrument: `retrieval_sleep_probe.py` as committed here. Defaults: 0.1B (`rwkv7-g1d-0.1b-20260129`), 240 steps,
  lr 5e-4, kl 1.0, rank 8, 4 cues, prio 2.0. One smoke run on seed 7 (not one of the six; 20 steps, retrieval arm only)
  checks that the code runs and how long a step takes; it is not read as a result. If anything in the instrument has to
  change after a six-seed run starts, all seeds are rerun.
- Arms (same seed, same 240 steps, same 12 facts): `no_sleep`; `sleep_true` (current restudy sleep, strength 1);
  `sleep_retrieval` (M3); `sleep_retrieval_shuffled` (control: M3 on the told turns with answer words swapped
  between facts, cues and teacher answers generated from the shuffled turns).
- Seeds 42 43 44 100 101 102. Scorer for the bar: key as a lowercase substring of the greedy hippocampus-off answer to
  the ORIGINAL question (the `sleep_probe`/high-use-ceiling scorer; the questions do not contain the key). Also
  reported: `strict_scorer.correct_answer`.
- **LEAD (the plan's bar):** recall(sleep_retrieval) - recall(sleep_true) >= +2/12 on >= 5/6 seeds, AND mean false
  recall on the 4 never-told questions not higher than sleep_true, AND sleep_retrieval_shuffled mean recall <= 0.5/12.
  A pass is a lead, not a GO (needs 0.4B, the live Brain, a fresh question set and a lesion that changes the reply).
- **FALSIFIED at this scale** if the mean recall gain is < +1/12 OR mean false recall (4 never-told) is higher than
  sleep_true. Between the two: inconclusive.
- Reported, not part of the bar: paraphrase recall (transfer: the self-cues may equal the test question, the count of
  exact matches is reported), false recall on the 24 abstain-v2 fresh never-told questions, drift KL, strict recall,
  fraction of failed attempts in the first and last 40 sleep steps, how many facts have the key in any teacher answer
  (a teacher that never names the key cannot teach it: reported per seed, the facts are kept regardless).

## Smoke (seed 7, instrument check only)
Cue generation, teacher answers and the first retrieval step ran without error; stopped after step 0 to free CPU (load
~50 on 20 cores; one sleep step took minutes). Cue quality at 0.1B is poor, as the plan feared for generated cues:
of 48 self-questions, several drift off the episode ("What is a Redux API?", "What's your name?" -> "I'm Gustav."),
some teacher answers name the key in a wrong role. This is the method as specified at this size; the per-seed count of
facts whose key appears in a teacher answer is reported. Jobs run with OMP_NUM_THREADS=3 (CPU shared), which does not
change results beyond float summation order.

## Result (six seeds, 0.1B, read from `s*.json` / `aggregate.txt`)
**Verdict: FALSIFIED at this scale by the pre-registered rule (mean recall gain -0.50/12 < +1).** Not a verdict on M3
in general: the plan's decisive test is 0.4B on GPU, still to run.

| arm (mean of 6) | recall (bar scorer) | strict recall | paraphrase | false recall /4 | false recall /24 fresh | drift KL |
|---|---|---|---|---|---|---|
| no_sleep | 0 | 0 | 0 | 0 | 1.0 | 0 |
| sleep_true | 4.50 | 0.50 | 3.83 | 3.33 | 17.7 | 0.029 |
| sleep_retrieval (M3) | 4.00 | 2.00 | 2.83 | 2.67 | 8.0 | 0.015 |
| sleep_retrieval_shuffled | 0.17 | 0 | 0.33 | 1.50 | 4.5 | 0.022 |

- Bar: recall gain retrieval - true per seed (42 43 44 100 101 102): 0, -3, +2, -1, 0, -1; >= +2 on 1/6 (needed 5/6).
  False recall (4 never-told) 2.67 vs 3.33: not higher (that half of the bar holds). Shuffled control 0.17 (holds).
- Cues: 44-48 per seed, none equal to a test question; 10-12 of 12 facts have the key in at least one teacher answer.
  Failed attempts fall from 27-34/40 (first 40 steps) to 4-13/40 (last 40): the cortex learns to reproduce the
  teacher's answers to its own cues, but that does not transfer to the test questions.

### Not part of the bar (secondary; a lead to pre-register, not a result)
- False recall on the 24 fresh never-told questions roughly halves: 17.7 -> 8.0, lower on 6/6 seeds (10->6, 21->7,
  20->5, 20->9, 20->9, 15->12). Drift halves too (0.029 -> 0.015).
- Strict recall (`correct_answer`: no first-person echo, right subject) 0.5 -> 2.0, up on 5/6 seeds (-1 on s43). The
  bar's substring scorer counts first-person echoes ("I'm Marguerite.", "I'm named Biscuit.") that restudy sleep
  produces a lot of; retrieval practice trains answers in the second/third person, so the two scorers disagree in
  direction here. The pre-registered scorer stands; this is why the next test should pre-register the strict one.
- New failure mode: the teacher's own confabulations are consolidated. Off-episode cues at 0.1B ("What's the name of
  the person you're talking to?" -> "It's a young woman named Lili.") make "Lili" an intrusion after sleep (s43: "What
  is my daughter's name?" -> "She's called Lili."). Garbage cues in, garbage bindings out: cue quality is a 0.1B
  prerequisite for this method.

### What it shows / does not show
- Shows: at 0.1B, swapping restudy for self-tested retrieval with the in-context episode as teacher does not raise
  recall on the original questions; it does reduce schema intrusions on fresh never-told questions and ordinary-text
  drift (secondary, unregistered).
- Does not show: anything at 0.4B/1.5B (where self-cues and teacher answers should be on-topic); M3 combined with
  restudy; any other cue source, priority or step count; the live Brain.

## Next
1. The plan's step 4 at 0.4B on GPU (same probe, `--model` 0.4B, lr 1e-3, 480 steps, kl 3 per the cortex-size
   operating point), with the strict scorer pre-registered as primary and false recall on fresh24 as a co-primary.
2. At 0.1B, if retried: restudy + retrieval practice interleaved (the 24-question false-recall halving is the thing
   worth keeping), and drop cues whose teacher answer does not name anything from the episode (a comparator, M1-style).

## Result (2026-11-03): 0.4B, GPU, six seeds
Data: `04b/s*.json`, `04b/aggregate.txt` (from `retrieval_sleep_summary.py`, unchanged). Config read from the JSONs:
`rwkv7-g1d-0.4b-20260210`, 480 steps, lr 1e-3, kl 3, 4 cues, prio 2.0, seeds 42 43 44 100 101 102. Source run dir:
`_scratch/brainv2/m3-04b/` (it exists in the main checkout's `_scratch`, not in this worktree). The 0.1B files have no
`.prov.json` sidecars, so none were made. No numeric 0.4B amendment exists in this README: the bar scored below is the
plan's M3 bar (plan section M3, same as the 0.1B pre-registration), and the strict scorer and fresh24 false recall
were named in "Next" but given no numeric threshold, so they are reported, not scored.

**Verdict: FALSIFIED at 0.4B by the pre-registered rule.** Mean recall gain is -2.67/12 (needed >= +1 to avoid
falsification, +2 on >= 5/6 seeds for a lead). Two of the three lead conditions also fail.

| seed | true | retrieval | gain | shuffled | strict (true / retr) | false /4 (true / retr) | fresh24 (true / retr) |
|---|---|---|---|---|---|---|---|
| 42 | 3 | 1 | -2 | 1 | 0 / 0 | 1 / 2 | 10 / 9 |
| 43 | 2 | 4 | +2 | 0 | 0 / 1 | 1 / 3 | 2 / 6 |
| 44 | 10 | 3 | -7 | 0 | 0 / 2 | 4 / 0 | 19 / 5 |
| 100 | 7 | 1 | -6 | 1 | 3 / 0 | 3 / 2 | 17 / 7 |
| 101 | 1 | 0 | -1 | 3 | 0 / 0 | 4 / 1 | 19 / 3 |
| 102 | 4 | 2 | -2 | 0 | 0 / 0 | 1 / 3 | 6 / 7 |
| mean | 4.50 | 1.83 | -2.67 | 0.83 | 0.50 / 0.50 | 2.33 / 1.83 | 12.2 / 6.17 |

(no_sleep is 0 on every measure for all seeds. Recall is the bar scorer, out of 12.)

Against the bar:
- Gain >= +2 on >= 5/6 seeds: 1/6 (only s43). FAIL.
- Mean false recall (4 never-told) not higher than sleep_true: 1.83 vs 2.33. HOLDS.
- Shuffled control mean <= 0.5/12: 0.83 (s101 reached 3). FAIL. The instrument's null arm is not clean at 0.4B, so even
  the absolute numbers carry some non-specific recall.
- Falsification: mean gain -2.67 < +1. FALSIFIED. (False recall is not higher, so that clause does not apply.)

Reported, not in the bar:
- Strict recall is a tie, 0.50 vs 0.50 (retrieval higher on 2 seeds, lower on 1, equal on 3). At 0.1B this was 0.5 -> 2.0.
- Fresh24 false recall is lower for retrieval on 4/6 seeds (12.2 -> 6.17 mean; higher on s43 and s102). The 0.1B
  halving partly repeats, less cleanly. Drift KL is higher for retrieval (0.026 vs 0.017; s102 0.061), the reverse of 0.1B.
- Cues: 47-48 per seed, none equal to a test question; 8-12 of 12 facts have the key in a teacher answer (9, 12, 9,
  10, 10, 8 for the six seeds). Failed attempts fall less than at 0.1B: 31->19, 29->13, 34->10, 25->5, 30->18, 33->28.
  s102 barely learns to reproduce its own teacher answers, and has the worst teacher coverage.

### Read against scale (`brain/PREREQUISITES.md`)
The 0.1B excuse (half the facts unreachable in context) no longer applies: the 12 test facts are reachable at 0.4B by
construction (the probe picks the first 12 of the 36-fact pool reachable at 0.4B; the old-12 ceiling under this
protocol is 9/12), and 8-12 of 12 facts have a correct key-bearing teacher answer. The 0.4B fact set differs from the
0.1B one, so cross-scale recall is not directly comparable. So this is not a missing-scale result for the reachable-fact
question. What it is a verdict on is this method: self-generated cues, loss on the cortex's answer to the teacher, one
rank-8 adapter shared by 12 facts, 480 steps. What it does not settle: the capacity explanation (the plan's own
fallback: adapter rank or one adapter per night), cue quality at 1.5B, retrieval practice interleaved with restudy,
and the strict scorer as a primary (no threshold was set, and the result there is a tie). The plan's next test, rank
or per-night adapters, is the one the falsification clause names.

Note the large spread in sleep_true itself (1 to 10 of 12 across seeds), which makes the 6-seed gain noisy. It does not
rescue the bar: the gain is negative on 5/6 seeds.

Host: not recorded in the JSONs (`tools/where_ran.py`: unknown); the logs show cuda fp16 on the W GPU.
