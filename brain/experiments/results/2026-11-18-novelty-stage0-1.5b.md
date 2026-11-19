# Novelty stage 0 at 1.5B: the cortex's surprise marks coined words perfectly; the brain does not act on it (2026-11-18)

Measurement only (`experiments/novelty_stage0.py`, no switch, nothing fitted). Six seeds 42/43/44/100/101/102, W CPU
fp32, 6 threads. 12 frames x {C common, R rare-real, N coined} per seed, each item in a fresh brain (Brain defaults).
Per-seed JSON and readable replies in `2026-11-18-novelty-stage0-1.5b/s<seed>.{json,md}`.

**Signal (cortex prediction error, read from the birth state; deterministic, identical on all seeds):**
- Item surprisal (summed nats) separates N from C and N from R at AUROC 1.00; R vs C 0.99. First token only: N vs C
  0.99, N vs R 0.72. Per token: C 4.3, R 5.0, N 8.0 nats (N items are 3.1 tokens, R 2.5, C 1.0).
- Habituation: the same item in a follow-up frame after the brain heard it and replied drops by C 7.6, R 12.0,
  N 23.6 nats against the cold read; repeat vs first mention for N AUROC 1.00 on 6/6.
- So "new to me" and "rare in general" are not yet separated by a fresh brain's prediction error alone: R is between.
  In a fresh brain both are new; the post-sleep `cue_response` term (design record) is what separates told from untold.

**Behaviour (the unchanged brain's reply, sampled):**
- Any question: C 64/72, R 55/72, N 56/72. A question that names the item: C 46, R 40, N 25.
- Asks what the thing is (rough regex: "what's X", "X?", "never heard of"): C ~8, R ~13, N ~10 of 72. No
  selectivity for novelty.
- It treats coined items as known: "We had frunkets for dinner" -> "What kind of frunkets did you have?";
  "My wife works as a strobbler" -> "A strobbler is a person who works as a secretary or assistant."; "I'm learning to
  play krastle" -> "Krlaste is a fun game to learn...". The rare-real case shows the target behaviour exists in the
  cortex: "We had kedgeree" -> "What's kedgeree?" (2/6 seeds), "netsuke?".

**What it shows.** The prerequisite signal exists and is strong (PREREQUISITES Curiosity: "surprise signal" moves from
missing to measured). The gap is behavioural: nothing turns surprise into asking, and the default reply invents a
meaning (the K2 invention defect on single words). Stage 1 (design record) is justified as written, with the measure
changed: the target is a reply that asks what the new thing is, scored by a blind reader, and a count of invented
meanings (must not rise); "a question naming the item" is not the target (C already does it 46/72).
