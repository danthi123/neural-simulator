# mood3 re-analysis: refusals and the paired echo (exploratory)

**Capability:** Affect shaping speech. **Record:** `2026-11-03-mood3.md` (the run) and `brain/research/2026-11-03-mood-state-design.md`
sections 12-13 (the pre-registration).

**Exploratory re-analysis of existing replies, not a verdict; no criterion or bar changed.** Every number below is
computed from the already-recorded `brain/experiments/results/2026-11-03-mood3/mood3_<size>_s*.json` rows. Nothing is
re-run, no reply is regenerated, and no pre-registered criterion (`C1`-`C6`) is re-judged. Any new bar or a rerun at
36 rows is Claude's call through a checkpoint.

## What is open (from the 11-03 record)

(i) Part of "negative mood darkens" is refusals: 0.1B neg 35/144 vs off 9/144 (the 11-03 writeup's refusal reading).
(ii) The 0.1B failure is only `C6_echo_le_0.10` (s43 +0.104, s101 +0.125), and the off arm itself echoes 0.12-0.50.
The instrument questions this re-analysis asks of the recorded replies: how much of the lex pos-null / neg-null
numbers survives without refusals, and what the paired echo per sign reads.

## Definitions (frozen 2026-11-05, before the run)

**Refusals.** `brain/experiments/sentiment_instrument.py:REFUSAL_PATTERNS` (fixed): "i'm sorry", "i am sorry",
"i can't help", "i cannot help", "i can't assist", "i'm unable", "i am unable", "as an ai" — case-insensitive,
straight or curly apostrophe. `is_refusal(reply)` = any pattern occurs in the reply. These strings are a fixed
instrument for this re-analysis only; nothing in the registered run's scoring changes.

**Pairing.** Exactly as `mood_probe.main` builds the `pos_null` / `neg_null` contrasts: rows are parallel across arms
per (question, sample) index `i = q * samples + k`, and the null for sign `s` is the mean over the arms
`rand*_s` present in the JSON's `rows` (`rand_pos`, `rand1_pos` / `rand_neg`, `rand1_neg` at `n_rand = 2`).

**Lex pos-null / neg-null**, per sign `s`, three ways:
(a) **as stored** — the JSON's `contrasts[s_null].lex_diff`, recomputed from the rows (a test checks the two agree to
1e-6 for all 12 JSONs, i.e. the pairing above is the recorded pairing);
(b) **without "sorry"** — the same paired difference recomputed with the word *sorry* removed before
`sentiment_instrument.lexical` (both sides), since "sorry" is a NEG-list word that occurs in refusals;
(c) **on non-refusal pairs** — (a) restricted to the (q, k) pairs where neither the mood reply nor any of its null
replies is a refusal; `n` pairs shown per seed and sign.

**Paired echo, per sign.** Mean over (q, k) of `primary_echo(mood reply) - primary_echo(off reply)` (the revised echo
instrument, `sentiment_instrument.primary_echo`), with a bootstrap CI, printed next to the **registered**
`echo mood-off` = `(pos.echo_rate + neg.echo_rate) / 2 - off.echo_rate` from the stored summary (the `C6` quantity).

**CIs.** `mood_probe.boot_ci` with `np.random.default_rng(0)`, 2000 resamples.

**Outputs.** `reanalyse <dir> <size>` in `brain/experiments/mood_v3.py` writes
`brain/experiments/results/2026-11-03-mood3/reanalysis_<size>.txt` (new files only; the seed JSONs and every other
record are read-only). Per seed: refusal counts per arm; the three lex numbers with CIs; the paired echo per sign
next to the registered `echo mood-off`.

## Tables (added after the run)

(to be filled from `reanalysis_0.1b.txt` / `reanalysis_0.4b.txt`)