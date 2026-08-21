---
type: finding
status: contributing
date: 2026-08-21
mechanism: d5-learn-through-use-presaturation-read-window-5of6-monotone
lane: integration
integration_faculty: d5-live-consolidation
---

# D5 learn-through-use knob 2: the pre-saturation read window gives a 5/6 monotone rise (meets the >4/6 bar; seed 44 borderline, run-variable)

**Board #71 — the SECOND blocker to turning learn-through-use on by default.** Knob 1 (the memory separator) is closed
(sep_bias=1000 -> 6/6 disjoint, commit e62113ef). Knob 2 was the "rise to 6/6" residual: the stabilized-read finding
(2026-08-21-d5-stabilized-read-NEGATIVE) showed the surfaced dendritic-depth read rises monotonically on only 4/6
(isolated) / 5/6 (soft) seeds because of a DETERMINISTIC saturating-tail wobble — near the top the plateau-depth read
DROPS while the weight keeps growing. Its named fix: read the rise over the PRE-SATURATION window (fewer use-rounds). This is biologically grounded: the
dendritic plateau the read measures is a REGENERATIVE, ceiling-bounded NMDA event (Bittner, Milstein, Grienberger,
Romani & Magee 2017, *Science* 357(6355):1033–1036 — BTSP plateau potentials drive one-shot CA1 field/memory
formation; the plateau is a large regenerative dendritic event whose amplitude saturates), so a plateau-depth read of
memory strength is linear (monotone in the weight) only BELOW saturation — reading the rise pre-saturation is measuring
in the plateau's linear regime.

## Result — the pre-saturation window (n-turns=2) improves the rise to 5/6, meeting the >4/6 bar

`research/runners/_d5_step6_graded_apical_read_derisk.py --seeds 42 43 44 100 101 102 --n-turns 2` (6-seed, cupy,
weak-usable te-grid per seed). Clean run artifact `research/findings/raw/_d5_step6_knob2final/summary_6seed.json`:

| graded read | monotone-rise (clean re-run) | note |
|---|---|---|
| **depth_rest** | **5/6** | seed 44 NO-GO; go_flags [42✓ 43✓ 44✗ 100✓ 101✓ 102✓] |
| **depth_hold** | **5/6** | same seed 44 the lone miss |
| soft | 4/6 | the bounded sigmoid read |

This MEETS the stabilized-read finding's conversation-visibility bar (>4/6, i.e. the rise must clear the 4/6 baseline)
and IMPROVES on the n-turns=3 baseline (4/6 isolated). On the 5 GO seeds the rise is large and clearly monotone with
every anti-artifact condition holding — MOVES, MONO (strict), **LESION_VAN** (the rise vanishes with consolidation off
-> it is caused by the learning, not a read artifact), **SPECIFIC** (the untouched neighbour stays flat), FAITHFUL,
moat byte-identical. The strength roughly doubles over two uses on the strong seeds (e.g. seed 100 22.5->46.9, seed 101
34.3->55.6).

## HONESTY: it is NOT a stable 6/6 — seed 44 flips run-to-run

A FIRST n-turns=2 run (log `/tmp/claude-1000/knob2_nt2.log`) scored depth_rest 6/6, with seed 44 barely GO on a tiny
rise (21.92->22.92, ~1 mV). This CLEAN re-run scores 5/6 with seed 44 NO-GO. So seed 44 is a BORDERLINE, weakly-
consolidating seed whose ~1 mV rise flips GO<->NO-GO across runs on the seeded cupy substrate — the 6/6 was NOT
reproducible. The reliable figure is **depth_rest >=5/6, with one borderline seed**, not a clean 6/6. (The re-run
existed precisely to check reproducibility; it caught the non-reproducible 6/6 before it was committed.)

## Status: knob 2 substantially improved + meets the >4/6 bar, but not the 6-seed production standard

The pre-saturation read window resolves the saturating-tail wobble for 5/6 seeds and clears the conversation-visibility
bar. It does NOT reach a stable 6/6 (the project's production-flip standard): seed 44 consolidates too weakly for its
rise to reliably clear the margin. Residual (a bounded operating-point choice, NOT a substrate wall): make seed-44-like
weak-consolidators strengthen enough for a reliable rise — e.g. a per-seed store-strength floor, more use-rounds within
the pre-saturation window, or a relative-tolerance margin scaled to the (small) total move. The learn-through-use flip
can either proceed on the >4/6 bar (an owner UX call — a visible rise on 5/6, reversible) or wait for a stable 6/6 via
that residual. NO `sim/` edit; additive; the binary moat gate is unchanged (a faithful spiking read, not a phenomenal
claim).
