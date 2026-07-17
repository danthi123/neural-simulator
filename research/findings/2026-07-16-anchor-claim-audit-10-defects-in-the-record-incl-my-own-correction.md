# Anchor-claim audit — 10 defects in the project record, **including the correction I wrote the same morning**

**2026-07-16. Read-only (no runner, no GPU — a 4-arm sweep was in flight). 94 agents / 6 lenses × 3 adversarial
skeptics per candidate. 29 candidates → 14 survived refutation → 10 distinct defects after merging duplicates.
Every survivor re-verified BY ME from the raw artifacts before it was written here.**

## Why this audit ran

Three load-bearing claims were found false earlier the same day (the deep-credit "GO"; the nav sum-vs-mean metric; the
Cluster-K-v2 "NO heuristic"). My prior contamination audit only caught the shape *"cites a `SIGNAL=False` run, claims a
GO"* and returned clean. But it **could not see** the Cluster-K-v2 shape — an absent flag whose *default* leaves the
cheat open. **3 of 3 spot-checked claims had been wrong**, so the anchor docs (`CLAUDE.md` — auto-loaded into every
session — `ROADMAP.md`, and the live months-scale plan) were audited systematically.

## Headline

**The record is substantially sound.** Nothing fabricates a mechanism or a run. Every disputed number is a real
measurement **of something**. **Every defect is at the WRITE-UP layer** — a metric mislabelled, a quantifier
over-reached, a retracted number left standing, a true claim copied onto a config it does not cover. Same failure mode
as the three already corrected: **the instrument keeps working; the prose overrides it.**

**Two error FAMILIES recur and are worth naming as classes:**
1. **Metric-name conflation** — `sum_finalQ` and `mean_distance_overall` are printed *on one line* by the runner
   (`g11_bg_runner.py:8158-8161`) and get quoted interchangeably. **4 of the 10 defects are this one line.**
2. **Quantifier over-reach** — "each", "every", "all figures are": a blanket in a header that is true of 12 of 15
   members, or 7 of 8.

## The defects (ranked by how badly a future session is misled)

| # | where | defect | status |
|---|---|---|---|
| **D1** | `CLAUDE.md:3185-3196` | **the 2026-07-16 sum-vs-mean correction is ITSELF WRONG** — declares "all figures are `sum_finalQ`" one line above a figure that is a MEAN | **FIXED** |
| **D2** | `CLAUDE.md:3298` | A+E "3.31 ± 0.74" is **retracted by its own cited source** as dev-seed selection; cheat ledger **inverted** | **FIXED** |
| **D5** | plan `:24`, `ROADMAP:215` | "transformer **+ full-backprop LSTM** lose to an n-gram at 5M-tok/V=300" — **the LSTM was never run there** | **FIXED** |
| **D6** | plan `:54` | Gate B certified by "111 tok/s" — that is the **PREFILL** rate; measured decode is **19.8** | **FIXED** |
| **D3** | `CLAUDE.md:54` | "EMERGE-15..29 (**each** 6-seed GO)" — EMERGE-21/25/29 have **zero** seed artifacts | **FIXED** |
| **D4** | `CLAUDE.md:79` | "EMERGE-78..85 … **every result** 6-seed" — EMERGE-81 is 3-seed | **FIXED** |
| **D8** | `2026-05-05-step3-64x64-*.md:30-34` | a `sum_finalQ` sitting under a "Mean Manhattan" header | **FIXED** |
| **D10** | `2026-07-15-TEST-A-*.md:14` | linear control "0.31" — the artifact says **0.4166** | **FIXED** |
| **D9** | `2026-07-10-D3-*.md:34` | anti-cheat labelled "(3 dev seeds)" — it is **n=1** (seed 44) | **FIXED** |
| **D7** | plan `:37` | *thinnest — partially EXONERATED*; only the stale CYCLE-1038 citation + a recall/moat-vs-gen conflation are defensible | **deferred, see below** |

### D1 — the correction that certified the conflation it was written to kill

I recomputed all of it myself:

| set | `mean_distance_overall` | `sum_finalQ` |
|---|---|---|
| 32×32 n=6 | **2.5748 ± 0.1138, range 2.42–2.72** | 2.739 ± 0.167, range 2.54–3.04 |
| 16×16 K v2 n=3 | 1.0654 ± 0.0250 | **2.9617 ± 0.1290** |

`CLAUDE.md`'s **2.97 IS** `sum_finalQ`; its **"2.57 ± 0.11, range 2.42–2.72" is `mean_distance_overall`** — matching on
all three statistics, and matching `sum_finalQ` on none. The block's own per-quarter row ([4.33, 2.34, 1.84, 1.79])
**averages to 2.575**, proving it. So "13.3% better" subtracts a mean from a sum: (2.9676−2.5748)/2.9676 = **13.24%**.
Like-for-like it is **7.5%** (4.3% vs n=6), **5/6** seeds (seed 43 = 3.04 loses), and variance is **WIDER** (0.17 vs
0.12), not tighter. **What survives: 32×32 holds a 4× grid at ~equal `sum_finalQ`.**

**The lesson, and it is mine: I fixed the LABEL without re-checking the NUMBER.** A correction is a claim. It needs the
same verification as the claim it corrects — arguably more, because it is read as already-audited. Ranked #1 because
`CLAUDE.md` is auto-loaded into every session, and because a future session reads this block as *settled*.

### D5 — the exculpation of the spiking substrate is unproven for the one class the brain actually is

The plan's Data cell said *"even a full transformer + full-backprop LSTM lose to a tuned/interpolated n-gram at
5M-tok/V=300."* Verified: `_recurrent_lm_ceiling.py` defaults `--vocab 2000 --max-tokens 24_000_000`; **the string
"300" never occurs in it**; only three LSTM artifacts exist (TinyStories-23.7M/V=2000, WikiText-103-60M/V=8000).
**No LSTM has ever been run at 5M-tok/V=300.** And everywhere it *was* run it **BEATS** the bigram at every depth,
with the margin **growing** with context: +0.494 → **+1.813** (tiny), +0.485 → +1.201 (wt103).

**Why load-bearing:** that row is what routes the entire fluency gap to **DATA** and exculpates the spiking substrate —
"every model class is n-gram-bound at our scale." The CEILING finding itself calls the recurrent net *"the closest
full-gradient analogue of the recurrent spiking substrate."* The exculpation is currently unsupported **for recurrence**.

**Honest tempering of the fan-out's framing (which I decline to repeat):** the transformer's loss is on **WikiText
(1.7M words)**, where the CEILING finding says there is *no* vocab regime it wins (V=300 barely trains; V≥2000
catastrophically overfits). The LSTM's wins are at **23.7M/60M** — i.e. *at* the regime the plan already calls the
threshold. **So the plan's strategic call ("signal only real at ~23.7M") is SUPPORTED, not inverted.** The defect is
narrower: it asserts a measurement that does not exist. **Cheap settle (~10 min GPU):**
`_recurrent_lm_ceiling.py --vocab 300 --max-tokens 5000000`. If the LSTM **wins** there, the wall at achievable scale
is the **learning rule**, not the data, and the cell's call flips. If it loses — plausible, V=300 is
function-words-only — the call is confirmed on recurrence too. **Either way the row becomes evidence instead of
assertion.** Queued.

### D2 — a dev-seed-selected headline outliving its own blind-seed refutation (**second instance today**)

`2026-04-29-overnight-FINAL.md`'s own headline: *"A+E n=12 = 3.93 ± 1.55 vs baseline 4.39 ± 1.92 … **not statistically
significant at Welch's t=0.65**. The earlier n=6 '3.31 ± 0.74' was **favorable-seed selection** — adding 6 new seeds
(200-202, 300-302) gave A+E n=6 = 4.56 ± 1.95."* I recomputed the blind seeds from raw: **4.567 ± 1.952** — matching.
`CLAUDE.md` has carried the retracted 3.31 as its biology-grounded flagship for ~2.5 months, with no correction block.

Plus the ledger inversion: 4.08 is the **cheat-CLOSED** flagship (it passes `--cue-reflex-replaces-heuristic`, which
sets `h_strength=0.0`), not "cheats-allowed" (4.41) — and `CLAUDE.md` says so itself 30 lines later. Meanwhile A+E's
own recorded command sets **no** heuristic flag (default 1.0 → 800 pA from direct goal reads) and omits
`--sensed-reward` (restoring the Manhattan reward cheat). **A+E is strictly MORE cheat-laden than the 4.08 it was said
to beat.**

**This is the same defect class as the deep-credit "6-seed GO" that was 3 dev seeds.** Two independent instances in one
day ⇒ **dev-seed selection is a recurring failure in this project, not a one-off.** Standing implication: the standing
"6-seed rule" is necessary but insufficient — *which* six matters, and a blind half is what makes the number a claim.

### D10 — where I caught my own audit's synthesizer

`linear_concat_held` = [0.214, 0.5, 0.071, 0.357, 0.571, 0.786] → **0.4166**, not the table's 0.31. Every other cell
reproduces to 2dp, so it is an isolated transcription slip. **But the synthesizer's own gloss was wrong** and I
corrected it from the artifact: it claimed the blind seeds "reverse (linear 0.464 vs learner 0.500)". Actual —
**dev** [42,43,44]: linear 0.262 vs learner 0.286 (learner marginally ahead); **blind** [100,101,102]: linear **0.571**
vs learner **0.500** — **linear WINS on blind; it does not reverse.** Which *strengthens* "depth buys no advantage over
a plain linear model — both fail." The GO and the headline (fixedbind 0.87 ≫ linear 0.42) are unaffected.

## D7 — deferred, deliberately

The one survivor I did **not** act on. The fan-out alleged the plan's row-4 status is stale; the synthesizer found a
substantial **exonerating path** the fan-out's majority missed (on-bridge 320 *recall+moat* are 3-seed GO, but on-bridge
320 **generalization** was never measured and the numpy-320 ideal tops out at 0.45 — so "GO @64" is the project's own
honest anchor on the axis that matters, not a stale line). Only two narrower things look defensible: the row conflates
recall/moat with generalization, and its **CYCLE-1038 citation is stale and mis-scoped** (that gate belongs to the
EMERGE-44/45 is-a arc and was unlocked at CYCLE 1040/1041, seven days before the plan). **Left alone pending a closer
read** — acting on a thin survivor is how a false accusation enters the record, and today already produced three of
those.

## What this audit could NOT check (blind spots — stated because a clean-looking audit is itself a claim)

1. **Nothing was executed.** Every verdict rests on committed artifacts. A stale artifact would not be visible.
2. **`/mnt/projects` (E:) was not searched.** It holds ~701 untracked artifacts and is the only intact copy; a missing
   seed file (EMERGE-81's blind seeds, D3's readfloor 42/43) could exist there and never have been committed.
3. It hunts **6 defect classes** on **3 anchor docs**. Overclaims that cite no artifact, or live in the other ~1400
   findings, are out of scope.
4. **15 of 29 candidates were KILLED by the skeptics** — the refutation stage did real work, which is the reason to
   trust the 14 that survived.

## The rules this earned (now in the skill)

- **A CORRECTION IS A CLAIM.** It needs the same verification as what it corrects — *more*, because it is read as
  settled. (D1: I fixed a label without re-checking its number, and certified the conflation.)
- **DEV-SEED SELECTION IS THIS PROJECT'S RECURRING FAILURE.** Two independent instances in one day. "6-seed" is not the
  guarantee; **a blind half is.**
- **WHEN TWO METRICS PRINT ON ONE LINE, THEY WILL BE QUOTED INTERCHANGEABLY.** 4 of 10 defects trace to
  `g11_bg_runner.py:8158-8161`. The fix is upstream — never emit two same-shaped numbers on one line without labels
  that travel with them.
- **A BLANKET QUANTIFIER ("each", "every", "all") IS A CLAIM ABOUT EVERY MEMBER.** Enumerate the members or weaken the
  word.
