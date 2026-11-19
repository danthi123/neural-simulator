# Curiosity stage-1 dev at 1.5B: an "asking" direction added to the state does not make the brain ask; it makes it define (2026-11-18)

Dev only (`experiments/inquiry_dev.py`, dev bank, sampling seed 1143, 2 samples, common random numbers across arms,
pool CPU fp32). The direction: mean matrix-memory trace of the brain's own 51 blind-confirmed "what is X?" replies
minus its 106 other replies to the same 14 statements, read back from the birth state; added before the brain reads
the turn (always on, no novelty gate). Null: a per-layer norm-matched random direction. 432 replies, two blind readers
(Haiku + Sonnet, shuffled, arm and class hidden; agreement asks 0.99, invents 0.98, coherent 0.93). Counts where both
readers agree, n = 24 per cell. Labels: `2026-11-18-inquiry-dev-1.5b/blind_labels.json`.

| arm | asks what it is C / R / N | invents a meaning C / R / N | coherent C / R / N |
|---|---|---|---|
| off | 0 / 5 / 7 | 0 / 4 / 2 | 23 / 24 / 22 |
| ask, gain 2 | 0 / 5 / 8 | 7 / 17 / 17 | 21 / 23 / 20 |
| ask, gain 4 | 3 / 0 / 0 | 10 / 18 / 20 | 12 / 16 / 21 |
| ask, gain 8 | 0 / 0 / 0 | 1 / 7 / 6 | 0 / 1 / 2 |
| random, gain 4 | 0 / 0 / 1 | 0 / 3 / 2 | 7 / 13 / 1 |
| random, gain 8 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

- Asking does not rise at any gain (N: 7 off, 8 at gain 2, 0 at gain 4 and 8).
- Inventing rises sharply (N 2 -> 17 of 24 at gain 2; R 4 -> 17): the direction carries the content of the replies it
  was read from ("Glaciology is the study of glaciers", "A farrier is a person who works with horses"), not the act
  of asking. Coherence collapses at gain 4-8; the random direction breaks it faster.
- Same lesson as mood iterations 4-5: a fixed direction added to the matrix memory primes content at 1.5B; it does not
  set an intent. **NO-GO for the additive asking direction (dev), a verdict on the method.** No test run.

Next (design record branch "stage 2"): a route that changes WHICH memory the brain answers from rather than pushing its
state, e.g. the novelty signal gating the reinstatement of an ordinary episode in which a teacher answered a question
about an unknown word (the brain's own `reinstate_self_belief` path does this for low familiarity), or sleep-learned
asking. Record check first: the self-belief reinstatement's own results.
