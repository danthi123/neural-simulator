# clause-depth2 ceiling RESOLVED (for flat inner args) — a 4-line decode fix, not a deep rewrite — 2026-06-04

**One line:** The one benchmark category stuck at 0% in both constructed and grounded — clause-depth2 (a clause
whose patient is *itself* a clause, e.g. "dog see (cat chase (bird eat leaf))") — was NOT the SNR/dimension ceiling
it was filed as. The failure was a **decode-policy bug**: at two-plus levels of nesting the flat-vs-attributed
resonator over-triggered AND returned a *wrong* noun, which `resid > conf` selected over the correct cleanup. The
fix is to trust the cleanup (flat) at depth ≥ 2 — clause-depth2 **0% → 100%**, with depth-1 (including attributed
inner args) preserved. **Multi-seed (5 seeds): clause-depth2 15/15 = 100%, and the FULL unified-agent benchmark is
now 195/195 = 100% with NO category below 100%** (the boundary report's "ceilings: none" — up from 92.3% with
clause-depth2 the lone ceiling).

## The prior framing and why it was incomplete

clause-depth2 was parked as a "documented honest ceiling": multi-seed ~0%, and crucially "NOT dimension-budget
(D=4096 still 0/6)". The D-independence was read as "the per-level auto-detection over-triggers (the inner clause's
flat agent gains a spurious attribute)" — true, but the consequence was under-diagnosed: the spurious-attribution
path doesn't just *add* an adjective, it routes the noun through the resonator's factoring, which returns the
WRONG noun at deep nesting.

## The actual failure (seed 42, before the fix)

```
dog see?   got='cat chase soft noun_124 eat fast noun_103'   want='cat chase bird eat leaf'
man watch? got='wolf follow adj_40 noun_2 find adj_25 noun_90' want='wolf follow mouse find bread'
```

The OUTER clause decodes ("cat chase …", "wolf follow …") — depth-1 is perfect (5/5, incl. the attributed inner
arg "cold river"). It's the INNERMOST flat args (bird→noun_124, leaf→noun_103) that corrupt, each with a spurious
adjective. In `_decode_filler`, a terminal arg inside a clause (depth > 0) is classified by `_resonator2`:
`return noun if conf >= resid else f"{adj} {nn}"`. At depth ≥ 2 the two-level bundle crosstalk depresses the flat
cleanup confidence `conf` so the resonator residual `resid` spuriously wins — and the resonator's noun `nn` at that
crosstalk level is itself wrong. The **cleanup** `noun`, however, is correct (the inner clause filler is clean
after the exact agent+action crosstalk subtraction); only the override corrupts it.

## The fix (4 lines, regression-safe for depth-1)

`research/runners/nested_composition_agent.py`, `_decode_filler`: at **depth ≥ 2**, return the cleanup noun
directly (skip the resonator override). Depth-1 terminals keep the full flat-vs-attributed resonator (so the
attributed inner args of depth-1 clauses — "cat chase (cold river)" — still decode). Biologically/architecturally
defensible: an attributed innermost argument inside a clause-in-clause is out of scope, and the resonator is
demonstrably unreliable at that crosstalk level — so trust the cleanup there.

## Result

```
clause-depth2: 0/3 -> 3/3   (seed 42: 'cat chase bird eat leaf', 'wolf follow mouse find bread', 'frog catch duck hold fish')
clause-depth1: 5/5          (preserved -- incl. attributed inner arg 'cold river')
24 nested/recursive/clause tests: PASS (no regression)
```

Full unified-agent benchmark, 5 seeds (42–46), constructed, D=2048:

```
flat 40/40  1-attribute 30/30  2-attribute 25/25  clause-depth1 25/25  clause-depth2 15/15  who 30/30  abstain 30/30
OVERALL: 195/195 = 100.0%   boundary report -> ceilings below 100%: (none)
```

Every category is 100% across all 5 seeds — the benchmark has no remaining ceiling. The fix touches only depth ≥ 2
terminals, so the other six categories are unchanged (no regression); clause-depth2 is the only category that moved
(0% → 100%).

## Honest scope

- This resolves clause-depth2 for **flat innermost args** — the benchmark cases and the common case. Depth-2
  **attributed** innermost args ("dog see (cat chase (big bird))") remain out of scope: they now return the flat
  noun (drop the adjective), because the resonator can't recover an attribute at that crosstalk level anyway. So
  the honest statement is "deep center-embedding with flat innermost arguments now decodes; with attributed
  innermost arguments it gracefully degrades to the flat noun."
- Deep center-embedding past depth ~2-3 is rare in real conversation (humans struggle with it too), so the flat-
  inner case is the one that matters; this lifts the documented ceiling for it with a minimal, regression-safe
  change rather than the feared deep rewrite.

## Files

- `research/runners/nested_composition_agent.py` — `_decode_filler` depth ≥ 2 flat-bias (4 lines).
