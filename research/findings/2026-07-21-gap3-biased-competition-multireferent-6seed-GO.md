# gap#3 (multi-referent disambiguation) — BIASED-COMPETITION resolves correlated referents where recency/salience failed (6-seed GO, rate rung)

**2026-07-21 · GO, 6-seed (42/43/44/100/101/102), rate rung.** The gap-close research gate's Rank-4: the mechanism the
two prior NEGATIVEs named but never built — **biased-competition** (Desimone-Duncan 1995, lateral inhibition between
referent attractors). It resolves a bare pronoun among several CORRELATED held referents, closing the
`2026-06-17-multireferent-disambiguation-NEGATIVE` (0/3; "the loop holds the SET, not a ranked salience") + the
salience-boost NEGATIVE.

## The wall + the fix

With N correlated referents in WM (a salience-weighted superposition), reading the max `<WM, ref_r>` is dominated by
the inter-referent CORRELATION, not the salience → it does not track recency/topicality (the prior NEGATIVE). The fix:
biased-competition SUBTRACTS the correlated crosstalk via lateral inhibition (`G[r,r'] = <ref_r, ref_r'>`),
decorrelating the activations so the SALIENCE wins:

    a_r ← relu( <WM, ref_r> − λ · Σ_{r'≠r} G[r,r'] a_{r'} ) ,  winner = argmax a

## Result (`_gap3_biased_competition_multiref_derisk.py`, N=4, D=128, 6-seed × 300 trials, chance 0.250)

| inter-referent corr | OFF (read-max / salience-boost = the prior NEGATIVE) | ON (biased-competition) | permuted-position | equal-salience ctrl |
|---|---|---|---|---|
| 0.6 | 0.601 | **0.928** | 0.925 | 0.245 |
| 0.75 | 0.581 | **0.938** | 0.931 | 0.246 |
| 0.9 | 0.568 | **0.949** | 0.938 | 0.244 |

- **The biased-competition advantage GROWS with correlation** (ON−OFF gap 0.33 → 0.38 as corr 0.6→0.9): read-max
  degrades toward ambiguity while lateral inhibition stays ~0.93-0.95. It is decisively load-bearing exactly where the
  referents are correlated — the regime that broke the prior approaches.
- **Anti-cheats clean, all seeds/corr:** permuted-position ~0.93 (the winner tracks SALIENCE, not position);
  equal-salience control ~0.245 ≈ chance (no spurious winner when there is no salient referent).

## Read-out

- **⇒ gap#3's named-but-unbuilt mechanism WORKS:** biased-competition (lateral inhibition) resolves the salient
  referent among correlated referents (0.93-0.95) where read-max/salience-boost fail (~0.58-0.60). This closes the
  two prior NEGATIVEs at the rate rung — the loop CAN produce a ranked salience if the referents COMPETE.
- **Honest scope:** rate rung (numpy). The gate's full Rank-4 is the SPIKING phase-cluster WTA on the RF substrate
  (the same competitive read that separates multi-binds, gap#2) — the follow-on. The salience signal here is a clean
  recency profile; wiring it to a real discourse WM (the emergent recurrent cortex learns salience) is the emergence
  step above the mechanism.
- **This is the third gap advanced this cycle** — the gate's plan (one competitive-read primitive unifying gaps
  #2/#3/#5) is bearing out: the learned binder (#2, 6-seed GO), and now the multi-referent read (#3, 6-seed GO), both
  ride the same role-keyed / biased-competition read.

Runner: `_gap3_biased_competition_multiref_derisk.py` (`--n-ref`, `--corr`, `--seeds`, `--n-trials`).
