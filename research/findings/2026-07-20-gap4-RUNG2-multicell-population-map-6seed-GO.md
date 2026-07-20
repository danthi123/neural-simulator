# gap#4 RUNG 2 — the one-shot BTSP rule COMPOSES: 4 cells learn 4 DISTINCT place fields in ONE lap (6-seed GO)

**2026-07-20.** Rung 1 (same day) showed a *single* CA1 cell learns a localized place field from ONE plateau via
thresholded BTSP. That is one cell. Every downstream use — a population code, a map, anything a later layer could
read — requires N cells to acquire N **different** fields **without interfering**. This is the prerequisite rung
toward gap#4's still-open deep-credit frontier.

## Setup — built so interference is possible

4 CA1 pools share the SAME 20-bin position input. Each pool receives its OWN plateau at its OWN target bin during
**one** induction lap. The inputs are shared, so every cell's potentiation writes onto the same presynaptic pools;
the only thing distinguishing the cells is **which plateau each received**. If per-cell credit smeared, all cells
would converge on one field or on mush.

## Result — 6-seed GO (dev and blind reported separately)

| arm | per-cell acc | distinctness | peaks vs targets |
|---|---|---|---|
| **MAIN** — dev **1.000**, **blind 1.000** | **1.00** | **1.00** | **[4, 8, 12, 16]** vs [5, 9, 13, 17] |
| C1 frozen (`eta=0`) | 0.00 | 0.00 | no fields form |
| C3 no-plateau moat | 0.00 | 0.00 | no fields, `dw = 0` |
| **C2 shuffled targets** (fixed) | **0.00** | 1.00 | fields exist but score ZERO against the wrong target |

**Every cell forms its field at exactly −1 from ITS OWN plateau**, width 3/20, on every seed. `distinctness = 1.00`
(all cell pairs separated by > 2 bins) was gated explicitly because per-cell accuracy alone could be satisfied by every
cell learning the *same* field.

**C2 at 0.00 is the decisive control:** the fields are precisely tied to their own instructive signal, not to the
shared input statistics.

**n=1 trap checked** (it already bit this arc once): seeds build genuinely different networks (weight hashes differ,
`w_sum` 5933.538 vs 5933.531) and `dw` varies across seeds (2341–2347). Field *placement* is identical across seeds —
that is robustness, not degeneracy, since the eligibility gradient determines placement and the controls collapse.

## ⚠️ A flaw in MY OWN control, caught and fixed before reporting

C2's first form shuffled targets by **1**, which scored **0.75** — not because the mechanism failed but because the
targets are 4 bins apart while the scoring window spans 7 (−5..+1), so each peak landed inside the **neighbouring**
target's window. A control-geometry flaw. Shifting by `N_CELL//2` separates them maximally and the control now reads
**0.00**. Reported because a control that passes for the wrong reason is worse than no control.

Also: the first smoke ran at `bin_steps=60`, which breaks the lap/τ ratio the rung-1 result depends on (lap 1200 ms vs
τ 1000 ms, against the calibrated 4000 ms) and produced net *depression* (`dw = −2323`) and no fields. Not a result —
a mis-timed harness. At the proper 200 ms/bin it works.

## What this does and does NOT establish

**Does:** the local, one-shot, plateau-gated rule **composes to a population** — distinct, correctly-placed, non-
interfering fields from a single experience, on shared inputs, 6-seed, blind-clean, with a zero-scoring shuffle control.

**Does NOT:** this is still a *single layer*. Nothing downstream reads the learned code yet. gap#4's deep-credit
frontier — a substrate that learns **deep** representations by a biological rule — remains open exactly where the
three audits left it. The next rung is whether a **downstream layer can learn to read this population code**, which is
the first point at which the word "deep" is earned.
