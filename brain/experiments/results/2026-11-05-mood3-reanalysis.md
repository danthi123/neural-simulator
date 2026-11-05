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

## Tables (from `reanalysis_0.1b.txt` / `reanalysis_0.4b.txt`, run 2026-11-05 after the definitions above were committed)

Anchors that agree with the 11-03 record's own readings: the 0.1B refusal totals are neg 35/144 vs off 9/144 (the
record's "negative partly a refusal mode" reading), and the 0.1B registered `echo mood-off` exceeds 0.10 exactly at
s43 (+0.104) and s101 (+0.125) (the record's `C6` reading). Everything below is those files' numbers, not a re-judgement.

### 0.1B (GAIN 16, 6 seeds, 24 rows per arm)

Refusals (`is_refusal` / 24 rows per arm):

| seed | pos | neg | off | rand_pos | rand_neg | rand1_pos | rand1_neg |
|---|---|---|---|---|---|---|---|
| 42 | 0 | 5 | 3 | 0 | 3 | 2 | 0 |
| 43 | 0 | 10 | 3 | 2 | 0 | 0 | 11 |
| 44 | 0 | 8 | 2 | 3 | 0 | 0 | 8 |
| 100 | 0 | 3 | 1 | 0 | 0 | 4 | 0 |
| 101 | 0 | 5 | 0 | 1 | 0 | 0 | 1 |
| 102 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| **total** | 0/144 | 35/144 | 9/144 | 6/144 | 3/144 | 6/144 | 20/144 |

Lex pos-null / neg-null ((a) as stored / (b) without "sorry" / (c) non-refusal pairs, n shown) and the echo:

| seed | lex pos-null (a) / (b) / (c) n | lex neg-null (a) / (b) / (c) n | paired echo pos | paired echo neg | registered echo mood-off |
|---|---|---|---|---|---|
| 42 | +4.132 [+1.918, +6.548] / +3.899 [+1.707, +6.179] / +3.564 [+1.425, +5.975] n=22/24 | -2.917 [-4.840, -0.982] / -1.355 [-2.614, -0.097] / -1.169 [-2.664, +0.468] n=19/24 | -0.042 [-0.292, +0.208] | -0.250 [-0.417, -0.083] | -0.146 |
| 43 | +2.586 [+0.182, +5.139] / +2.094 [-0.045, +4.200] / +1.896 [-0.357, +4.387] n=22/24 | -3.896 [-6.316, -1.447] / -0.987 [-1.614, -0.397] / -1.288 [-2.576, +0.000] n=9/24 | +0.083 [-0.083, +0.250] | +0.125 [-0.208, +0.417] | +0.104 |
| 44 | +3.102 [+1.567, +4.950] / +2.512 [+1.278, +3.970] / +1.982 [+0.946, +3.280] n=21/24 | -4.883 [-7.207, -2.602] / -1.846 [-3.001, -0.800] / -1.952 [-3.012, -0.792] n=11/24 | -0.042 [-0.208, +0.125] | +0.000 [-0.250, +0.208] | -0.021 |
| 100 | +2.655 [+0.688, +5.116] / +1.874 [+0.119, +4.070] / +2.014 [+0.022, +4.831] n=20/24 | -2.279 [-4.342, -0.422] / -0.832 [-1.980, +0.301] / -0.775 [-2.055, +0.537] n=21/24 | +0.000 [-0.208, +0.208] | +0.083 [-0.125, +0.292] | +0.042 |
| 101 | +2.633 [+1.204, +4.057] / +2.372 [+1.001, +3.920] / +2.638 [+1.223, +4.149] n=23/24 | -4.304 [-6.480, -2.293] / -2.320 [-3.790, -0.987] / -2.357 [-4.030, -0.980] n=18/24 | +0.042 [-0.125, +0.250] | +0.208 [+0.000, +0.417] | +0.125 |
| 102 | +3.076 [+1.508, +4.692] / +3.076 [+1.543, +4.685] / +3.076 [+1.623, +4.686] n=24/24 | -6.220 [-8.752, -3.947] / -4.600 [-6.883, -2.813] / -4.833 [-7.385, -2.714] n=20/24 | -0.125 [-0.292, +0.042] | -0.083 [-0.292, +0.167] | -0.104 |

### 0.4B (GAIN per dev, 6 seeds, 24 rows per arm)

Refusals (`is_refusal` / 24 rows per arm):

| seed | pos | neg | off | rand_pos | rand_neg | rand1_pos | rand1_neg |
|---|---|---|---|---|---|---|---|
| 42 | 0 | 5 | 4 | 0 | 1 | 1 | 0 |
| 43 | 1 | 10 | 1 | 1 | 0 | 2 | 0 |
| 44 | 0 | 9 | 0 | 2 | 0 | 3 | 7 |
| 100 | 1 | 10 | 1 | 2 | 0 | 0 | 5 |
| 101 | 0 | 7 | 2 | 2 | 0 | 1 | 0 |
| 102 | 0 | 6 | 0 | 1 | 0 | 0 | 0 |
| **total** | 2/144 | 47/144 | 8/144 | 8/144 | 1/144 | 7/144 | 12/144 |

Lex pos-null / neg-null ((a) / (b) / (c) n) and the echo:

| seed | lex pos-null (a) / (b) / (c) n | lex neg-null (a) / (b) / (c) n | paired echo pos | paired echo neg | registered echo mood-off |
|---|---|---|---|---|---|
| 42 | +1.471 [+0.041, +2.789] / +1.410 [-0.005, +2.706] / +1.403 [-0.105, +2.770] n=23/24 | -2.282 [-4.108, -0.628] / -0.898 [-1.872, +0.040] / -0.603 [-1.627, +0.507] n=18/24 | -0.083 [-0.292, +0.125] | -0.083 [-0.250, +0.083] | -0.083 |
| 43 | +0.614 [-1.774, +2.656] / +0.111 [-2.168, +2.037] / +0.114 [-2.460, +2.240] n=22/24 | -7.635 [-11.217, -3.920] / -2.588 [-4.616, -0.642] / -1.140 [-3.649, +1.306] n=14/24 | +0.083 [+0.000, +0.208] | +0.375 [+0.125, +0.583] | +0.229 |
| 44 | +1.322 [-0.633, +3.701] / +0.284 [-1.104, +1.923] / -1.015 [-1.993, -0.231] n=19/24 | -4.398 [-6.373, -2.600] / -1.239 [-2.527, -0.099] / -1.619 [-3.469, -0.052] n=14/24 | +0.000 [+0.000, +0.000] | +0.083 [-0.083, +0.292] | +0.042 |
| 100 | -3.518 [-8.608, -0.216] / -3.778 [-8.541, -0.547] / -4.453 [-10.005, -0.863] n=21/24 | -6.095 [-9.606, -2.879] / -1.983 [-4.308, +0.009] / -1.326 [-3.871, +1.231] n=13/24 | -0.042 [-0.168, +0.083] | +0.292 [+0.083, +0.500] | +0.125 |
| 101 | -0.927 [-3.252, +1.210] / -1.291 [-3.449, +0.695] / -1.577 [-3.962, +0.668] n=21/24 | -3.852 [-5.950, -1.869] / -1.027 [-2.171, +0.060] / -1.350 [-2.881, +0.115] n=17/24 | +0.000 [-0.208, +0.167] | -0.042 [-0.292, +0.208] | -0.021 |
| 102 | -0.389 [-2.017, +1.087] / -0.562 [-1.971, +0.760] / -0.794 [-2.349, +0.520] n=23/24 | -3.306 [-5.090, -1.640] / -1.015 [-2.010, -0.004] / -1.435 [-2.763, -0.129] n=18/24 | +0.000 [-0.167, +0.167] | +0.125 [-0.125, +0.375] | +0.062 |

Reading the numbers is Claude's call (checkpoint); this writeup keeps no verdict.