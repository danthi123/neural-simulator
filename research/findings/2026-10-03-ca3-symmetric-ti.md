---
type: finding
status: live
date: 2026-10-03
lane: prove-who-owns-the-computation
mechanism: symmetric broad CA3-CA3 rule (Mishra 2016) as the sleep-TI STI7 successor, frozen at
  hebbian_learning_rate=0.02 by the section-11 rung selection; row 11 (rev instrument identity) fails on all
  four dev seeds, so every seed reads UNDEFINED(dev) instrument; the row-2 WRITE-ABSENT shape is reported
  as a shape only (Claude verify-go caveat 2026-10-04).
---
# PROVISIONAL finding: CA3 symmetric-TI successor at lr=0.02 reads UNDEFINED(dev): instrument on all four dev seeds (row-2 WRITE-ABSENT shape reported, not read)

> **Claude verify-go caveat (2026-10-04).** The first version of this document headlined WRITE-ABSENT. That is a
> mischaracterization under the table fixed before the run (`docs/drafts/ca3-symmetric-ti/PLAN.md` section 7). Row 11
> fires on all four seeds, and its fixed wording is "no biology reading from **that seed**". The scope is per-seed as
> written, and it is not re-scoped to the `rev` arm after the data are in (RULING `ca3-sym-ti-row11`). So every seed
> reads **UNDEFINED(dev): instrument**. The row-2, row-6 and row-9 entries below are kept as the shapes the records
> show, and none of them is a reading. The sym-arm numbers themselves (DI gain 0 to 8e-4, WAD rise 0.85 to 1.90 pA,
> thousands of co-spikes) are not affected by the `rev` instrument, so they stand as raw values.

**Status: PROVISIONAL — dev seeds only (8, 9, 10, 11).** This document assigns the section-7
interpretation words to dev reads under the table fixed before any run
(`docs/drafts/ca3-symmetric-ti/PLAN.md` section 7, fixed at `31eea95e2`); it makes no capability
claim and no registered-shape outcome.

## Question and method

Does the symmetric broad CA3–CA3 rule (Mishra 2016, RULING `T3-sleep-ti`) repair the sleep-TI
erosion (STI7) while still writing A→D inference, without false inference? The rule was frozen at
`hebbian_learning_rate = 0.02` by the section-11 rung selection (all three rungs met all
four registered rung criteria; largest satisfying value per the registered wording).

- Revision `d4a533ae77186050e3e8bc6156e24bb81af3221b`, draft branch
  `research/draft-ca3-symmetric-ti`, numpy backend, pool placement (RULING compute-placement).
- 4 dev seeds × 5 arms: `wake`, `sym` (symmetric night, STDP off), `wpermc` (within-class weight
  permutation control), `dis` (replay cut 4.2 ms after each SWR start), `rev` (time-reversed
  raster instrument). Part a = sym/wpermc/rev, part b = wake/dis; encode before the night.
- Records: `research/findings/raw/_ca3_symmetric_ti/parts/seed*_merged.json` (4 files, seeds 8–11) — 4/4
  merged, `u7_encode_hash_equal=true`, `error=null` in every record.

## Section-9 readings (one MiMo Pro delegate, numbers from the records only)

Per-seed primary row (first applicable in table order; superseded by row 11, see the caveat at the top):

| seed | primary row | reading | DI(sym)−DI(wake) | WAD rise (pA) | sym cospike_total |
|---|---|---|---|---|---|
| 8 | 2 | WRITE-ABSENT | 0.0 | 0.85 | 8594 |
| 9 | 2 | WRITE-ABSENT | 0.0 | 1.36 | 11315 |
| 10 | 2 | WRITE-ABSENT | 0.00078 | 1.22 | 12498 | <!--derived-->
| 11 | 2 | WRITE-ABSENT | 0.00078 | 1.90 | 15513 | <!--derived-->

Additional rows firing on **all four** seeds:

- **Row 6 CONTROL-DEGENERATE** (U6-analogue, histogram clause alone): 70–73% of the night's |Δw|
  mass sits in classes of size 1 (21056–21496 singleton classes) → **no STI3 structure reading is
  taken from any seed** (row 6 overrides row 5); read as UNDEFINED(dev) for that shape only.
- **Row 9 NO-FALSE-INFERENCE** holds 4/4: foil ≤ 0.05 above `wake` (all foil reads 0.0) and every
  never-encoded C < 0.10 (all C reads 0.0).
- **Row 11 UNDEFINED(dev): instrument** — the `rev` replica identity residual is 34.3–47.2 pA,
  far above the 1e-3 pA expectation (by construction the same-step rule's reversed vs
  un-reversed passes were expected to coincide); pair-mean cross-checks of |W(rev)−W(sym)| are
  0.28–0.42 pA. No biology reading is taken from the `rev` instrument; the residual itself is
  reported as an open instrument question.

Bar readings (dev reference points, met/not-met only):

| shape | met | not-met |
|---|---|---|
| premises learned (STI0) | 8, 9, 10, 11 | — |
| premise retention (STI7) | 8, 9, 10, 11 | — |
| inference written (STI1/STI6) | — | 8, 9, 10, 11 |
| replay necessary (STI2) | — | 8, 9, 10, 11 |
| structure necessary (STI3) | no reading taken (row 6, all four) | — |
| no false inference (STI5) | 8, 9, 10, 11 | — |
| control validity clauses (U6: premises/foil/C) | 8, 9, 10, 11 | — |

`dis` writes nothing measurable (WAD 0.0, |Δw|_between 0.0, lesion held on raster+weights
4/4), and `wake` DI is ~0 everywhere, so the DI comparisons all sit at 0–1e-3 against
thresholds of 0.15/0.05.

## What row 2 says, and what this does NOT show

Row 2 (registered wording): with `cospike_total` ≥ 1 the symmetric write exists but is **too
weak at these constants** — dev evidence for the broad-window stand-in (plan section 13),
Claude's call. It is explicitly not "the rule is never exercised" (U8 does not fire: thousands
of same-step co-spike pairs per night).

What this does NOT show: no inference write occurred, so nothing here speaks to the
replay-necessary or structure-necessary shapes on real data (STI3 has no reading at all);
dev seeds 8–11 only, one lr, one encode; premise retention (STI7) and no-false-inference (STI5)
reading clean 4/4 is a necessary-but-not-sufficient signal; the row-6 histogram and row-11
replica residual are open instrument/design questions, not readings about biology.

## Proposed next step (for the weekly review)

Per row 2's registered pointer: the broad-window stand-in (plan section 13) as the next dev
method, plus a Claude reading of two anomalies the delegate isolated — the singleton-mass
clause firing 4/4 and the 34–47 pA `rev` replica residual. Both need judgment before any
follow-up run.

## Provenance

- Plan and outcome table: `docs/drafts/ca3-symmetric-ti/PLAN.md`; rung freeze:
  `docs/drafts/ca3-symmetric-ti/WORKLOG.md` (section 11, `31eea95e2`).
- Runner: `research/runners/_ca3_symmetric_ti_symrule.py` at `d4a533ae7`.
- Scored by one MiMo Pro delegate applying the fixed section-7 table to the four merged
  records; numbers above are taken only from that scoring output.
