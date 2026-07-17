# D5 SETTLED — the "full-backprop LSTM" half of the ceiling claim had **never been run**. It has now been run, and it **CONFIRMS** the claim it was asserting.

**2026-07-16. ~4 min of GPU. The audit found an unsupported claim; the follow-up VINDICATED it.**
**Runner:** `research/runners/_recurrent_lm_ceiling.py` (reuse-by-import, NO `sim/` edit).
**Raw:** `raw/_d5settle_lstm_wikitext_V300_5M.json` (seed 42) + `raw/_d5settle_lstm_V300_s{43,44}.json`.

## The claim, and why it mattered

The months-scale plan's Data cell (and `ROADMAP.md:215`, and the upstream 07-15 gate finding) asserted:

> "even a full transformer **+ full-backprop LSTM** lose to a *tuned/interpolated* n-gram at 5M-tok/V=300"

**That row is load-bearing far beyond bookkeeping: it is what EXCULPATES THE SPIKING SUBSTRATE.** If *every* model
class — including recurrence — is n-gram-bound at our data scale, then "our brain is only n-gram-level" is a property
of the **data regime**, not a defect of the substrate, and the whole fluency gap routes to **DATA**. That is the
plan's central strategic call.

**The anchor-claim audit found the LSTM half was never measured.** `_recurrent_lm_ceiling.py` defaults
`--vocab 2000 --max-tokens 24_000_000`; the string "300" does not occur in it; only three LSTM artifacts existed, all
at TinyStories-23.7M/V=2000 or WikiText-103-60M/V=8000 — **where the LSTM BEATS the bigram at every depth with a
margin that GROWS with context** (+0.494→+1.813; +0.485→+1.201). So the exculpation was **unproven for recurrence —
the one architecture class the spiking substrate actually is** (the CEILING finding itself calls the recurrent net
*"the closest full-gradient analogue of the recurrent spiking substrate"*).

## The settle

Matched to the CEILING transformer's **own defaults** (`_wikitext_transformer_ceiling.py`: `--corpus
data/corpus/wikitext.txt --vocab 300`). Apples-to-apples is asserted by the LSTM runner's own docstring (`:15`):
*"SAME corpus, SAME top-V word vocab, SAME add-1 bigram baseline"* — both build vocab via
`Counter(words).most_common(V-1)`, both compute an add-1 bigram, both report `margin = bigram_ce − model_ce` bucketed
by within-block context depth. LSTM: 2 layers, d=384, block=128, **2.6M params**, 8k steps, cuda.

**Seed 42 — CE by context depth (margin > 0 ⇒ LSTM beats bigram):**

| ctx depth | LSTM CE | bigram CE | margin | |
|---|---|---|---|---|
| 1-1 | 3.507 | 2.716 | **−0.791** | bigram wins |
| 2-2 | 3.612 | 2.637 | **−0.975** | bigram wins |
| 3-3 | 3.929 | 2.738 | **−1.191** | bigram wins |
| 4-8 | 4.064 | 2.733 | **−1.331** | bigram wins |
| 9-16 | 4.144 | 2.736 | **−1.408** | bigram wins |
| 17-128 | 4.133 | 2.741 | **−1.392** | bigram wins |

**The LSTM loses to an add-1 bigram at EVERY depth, and monotonically MORE with depth (−0.79 → −1.41).** That is the
**same signature as the transformer** at this condition (−0.059 → −0.379) — only far stronger. *More context makes the
LSTM relatively worse* — the opposite of "captures long-range". The tell is identical: **train-CE collapses to ~1.5-2.1
(memorization) while held-out CE is 3.5-4.1, worse than a bigram's 2.7.**

**⇒ THE CLAIM IS NOW TRUE AND EVIDENCED.** The exculpation of the spiking substrate holds **for recurrence too**:
at this data scale, a full-backprop recurrent net is n-gram-bound exactly as the transformer is. **The plan's Data-cell
strategic call SURVIVES — and is no longer an assertion.**

## Honest caveats (each of which cuts against over-reading this)

1. **"5M-tok" is a MISNOMER in the plan, the CEILING finding, and this run's own filename.** `wikitext.txt` contains
   **2,045,059 words**, so `--max-tokens 5000000` is **capped by the corpus at ~2M**. Both the transformer and this
   LSTM are capped identically on the same corpus, **so the comparison is valid** — but the label is wrong everywhere
   it appears. (Yet another instance of the day's metric-label class.)
2. **Both models are OVER-PARAMETERIZED for the corpus.** 2.6M params on ~2M words. The CEILING finding says this of
   itself: *"a 2M-param transformer on 1.7M words has far too much capacity for the data: it memorizes train and
   generalizes worse than random"*, and *"the corpus is ~50-100× too small."* ⇒ **this result demonstrates THE DATA
   REGIME, not a recurrence-specific limit.** Which is precisely the DATA thesis — but it means the honest claim is
   *"at this data scale, an over-parameterized model of EITHER class overfits and loses to a bigram"*, **not**
   *"recurrence cannot hold long-range"* (it demonstrably can — at 23.7M+, the same runner's LSTM wins by +1.8).
3. **A smaller LSTM was not tried.** The capacity/data mismatch is the mechanism; a right-sized recurrent model at 2M
   words is unmeasured. This does not threaten the DATA thesis (a model too small to overfit is also too small to
   extract deep structure), but it is not excluded.
4. **Seed 42 is complete; 43/44 were in flight at write-up.** The margin is large (−1.4 nats) and monotone across six
   depth buckets, so seed noise is an implausible explanation — but this is stated as **n=1 → 3**, not 6.

## Process notes (both cost real time today)

- **The first 43/44 attempt DIED SILENTLY at ~115s** — logs stopped mid-training, no traceback, no JSON: the exact
  signature I had previously attributed to an MPS daemon death. **Cause: my own wait-loop command was SIGTERM'd by the
  harness 2-minute timeout (exit 143), and `nohup … & disown` children were still in that command's PROCESS GROUP, so
  the timeout killed them.** ⇒ **never launch background work inside a command that can time out**; use `setsid` to
  detach into a new process group (the relaunch does).
- **zsh, not bash** — an unmatched glob **aborts the whole command**, which silently skipped an `rm` earlier and
  aborted a wait loop here. Use `setopt NULL_GLOB` or quote.

## Corrections landed from this

`docs/plans/2026-07-15-months-scale-plan-...` `:24`, `ROADMAP.md:215`, and
`2026-07-15-beyond-ngram-wall-...-fixed-bind-is-test-A.md:6` (which **contradicted itself two lines apart** — `:6`
said the LSTM loses, `:7` said *"a full-backprop LSTM reaches it"*) were corrected to state the transformer result
only, flag the LSTM as unmeasured, and name this settle. **They should now be updated again to record that the settle
RAN and CONFIRMED the claim.**

**The meta-point worth keeping:** the audit's value here was **not** that the record was wrong — the claim turned out
right. It was that a load-bearing strategic call rested on a measurement **nobody had taken**, and it cost ~4 minutes
to take it. An assertion that happens to be true is still an assertion.
