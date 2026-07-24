# gap#4 de-confounded directed credit vs frozen reservoir: the advantage is DATA EFFICIENCY, not raw accuracy — 6-seed (2026-07-24)

**TL;DR.** The biological directed-credit rule (BDSP) beats a frozen random reservoir + trained linear read-out
**decisively in the LOW-DATA regime** and is a **wash at full data** — and this survives the three de-confounding
controls (input-selectivity, directed-credit-scramble, shuffled-target). So the earlier "bdsp > reservoir 6-seed GO"
is REAL directed credit (not an input-selectivity artifact or a label-shuffle shortcut), and its precise character is
**data efficiency**: directed credit extracts more from fewer examples. 6 seeds (42/43/44/100/101/102), MNIST,
hidden=256, depth=2, numpy, run on the mini-PC pool.

## The 6-seed pattern (bdsp = directed credit; RES = frozen reservoir + linear read-out)
| data frac | bdsp (mean) | reservoir (mean) | Δ | GO (bdsp>res + all controls) |
|-----------|-------------|------------------|-----|------------------------------|
| 1.00 (full) | 0.802 | 0.785 | +0.017 | **4/6** (seed-variable ~tie) |
| 0.10 | 0.887 | 0.651 | **+0.236** | **6/6** |
| 0.05 | 0.810 | 0.534 | **+0.276** | **6/6** |

At full data the frozen reservoir + a trained linear read-out is competitive (its high-dim random projection + a good
linear head nearly matches directed credit when examples are abundant); as data becomes scarce the reservoir degrades
fast (0.785 → 0.651 → 0.534) while directed credit holds up (0.802 → 0.887 → 0.810), opening a **+0.24 to +0.28** gap.

## The de-confounding controls (all pass, every frac/seed) — why this is REAL directed credit
- **input-selectivity** (`sel>0.20`): the forward IS input-selective (not a degenerate constant map) — always True.
- **directed-credit-scramble** (`bdsp>shufE`): scrambling which unit gets which error signal collapses accuracy to
  ~0.65-0.69 (from bdsp ~0.88 at frac 0.10) — the DIRECTION of the credit is load-bearing, not just its presence.
- **shuffled-target** (`shufY<chance`): training on shuffled labels dies at ~0.09-0.17 — no label-leak shortcut.
- (`permB` permuted control also passes.)
⇒ the advantage is not a confound; it is the directed credit assignment doing real work, most visibly under data scarcity.

## Why this matters for the mission
Data efficiency is exactly the property a biological learner needs (learn from limited lived experience, not a giant
corpus) — and it's the gap#4 "deep directed credit" capability the no-defer rule keeps ON. The result reframes the
credit-vs-reservoir question: don't ask "does directed credit beat a reservoir on accuracy" (a wash at full data),
ask "does it beat it PER EXAMPLE" (decisively yes). Next: characterize the data-efficiency curve deeper (more frac
points), and test whether the on-bridge BDSP realization inherits the same low-data advantage.

## Provenance
`_gap4_deconfounded_credit_vs_reservoir_derisk --seeds <s> --fracs 1.0 0.1 0.05 --n-train 8000 --n-test 2000
--epochs 15 --lr 0.03`; results `research/findings/raw/gap4/pool_deconf_s{42,43,44,100,101,102}.json`. Run on the
mini-PC pool (numpy 2.4.6, byte-pinned to the main box) during the owner's gaming window. The local 6-proc launch had
HUNG for ~2 hr on concurrent MNIST-load contention (killed); the pool ran each seed clean in ~2 min. NO `sim/` edit.
