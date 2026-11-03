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

## Result
(not yet run)
