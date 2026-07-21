# The emergence engine's read-out FAMILY (matched filter + biased-competition variant) spans gaps #2/#5 and #3 (6-seed GO)

**2026-07-21 · GO, 6-seed.** The gap-close research gate's core insight — "one competitive read-out unifies the binder,
disambiguation, and completion" — made concrete + HONESTLY CORRECTED. It is not ONE fixed function (the first pass at
that was chance, see below); it is one read-out FAMILY: the **matched filter** (score against the codebook, take the
best) is the shared primitive for binder cleanup (#2) and pattern completion (#5); **biased-competition** (lateral
inhibition over the small held-referent set) is the multi-referent (#3) VARIANT.

## Result (`_gap235_unified_competitive_read_derisk.py`, N=64 codebook, D=128, corr 0.6, 6-seed)

| gap | read | result |
|---|---|---|
| **#2 binder cleanup** (noisy estimate → concept) | matched filter | **1.000** |
| **#5 pattern completion** (half-masked partial cue → full stored pattern) | matched filter | **1.000** (no-overlap cue 0.014 ≈ chance) |
| **#3 multi-referent** (salient of a correlated referent set) | biased-competition | **0.925** vs matched-filter-only 0.621 (competition load-bearing) |

All correlated codes (corr 0.6) — the read RIDES on code overlap (completion needs it), so it WANTS correlated codes,
which kills the self-defeating decorrelation demand. The anti-cheats are clean: the no-overlap completion cue → chance;
the multi-referent biased-competition beats matched-filter-only by +0.30 (the lateral inhibition is load-bearing
exactly where the referents are correlated).

## The silent-failure catch (the overstatement, corrected)

The first pass coded the gate's claim LITERALLY — ONE biased-competition function over the whole codebook for all three
— and it was **chance (0.02-0.04) everywhere.** The result flagged it: two bugs. (1) The cleanup noise was `0.6·randn`
over 128 dims = magnitude 6.8, **6.8× the code** — the estimate was mostly noise (`eta=0.06` → magnitude 0.68 fixes it).
(2) Biased-competition over the FULL N=64 codebook OVER-SUPPRESSES (a 63-term lateral inhibition kills every unit) — it
is specific to a SMALL correlated competing set (the multi-referent case, N≤8), not a universal codebook read. ⇒ the
honest unification is a matched-filter primitive SHARED by cleanup + completion, with biased-competition as the
multi-referent variant — not one monolithic primitive. The failure flagged the overstatement (silent-failure rule:
the anti-cheat/result told me the framing was wrong).

## Read-out

- **⇒ the emergence engine's cognitive read-out is ONE FAMILY** (matched filter + a biased-competition variant),
  spanning three of the five gaps with correlated codes, anti-cheats clean, 6-seed. This is the read the learned binder
  (#2, 2026-07-21 spiking GO), the multi-referent resolver (#3, 2026-07-21 GO), and pattern completion (#5) all use.
- **Honest scope:** rate-level demonstration (numpy) that consolidates the three separate GO de-risks into one
  read-out account; the SPIKING realizations are the gap-by-gap findings (gap#2 spiking binder GO; gap#3 spiking
  phase-cluster WTA + gap#5 CA3 completion = follow-ons). The correction (family, not one function) is the load-bearing
  honesty here.

Runner: `_gap235_unified_competitive_read_derisk.py` (`--N`, `--D`, `--corr`, `--seeds`).
