# The on-bridge spiking BDSP does not train the depth-2 net at the configs tried (6-seed) — and the per-example wall-clock is the binding limit; a block-diagonal batched forward gives 8–15× but has an unresolved global coupling (2026-07-12)

**One-line verdict:** across 6 seeds at a trained-scale config (H96/ep120), the on-bridge spiking depth-2 BDSP net does NOT train the semantic-inheritance task — every credit arm sits BELOW chance (plain-FA 0.179, microcircuit 0.21, burstprop; single-layer floor 0.228, chance 0.333) while the rate oracle solves it (1.0). Scaling epochs 4× at lr 0.25 DESTABILIZED plain-FA (collapsed to 0.037). So the point-neuron spiking deep credit doesn't train a depth-2 net at the configs tried (smoke under-trained; trained-lr unstable) — and finding a training regime needs an lr/config sweep that is intractable per-example (~1–2 hr/run). The block-diagonal batched forward that would make the sweep tractable is de-risked to **8–15× speedup** but has an **unresolved global-op coupling** (correctness fix = the follow-on).

## Context — this closes the pool-k gate's open half
The pool-k gate (`2026-07-12-poolk-population-read-NOT-the-deep-credit-lever-flat-multiseed.md`) resolved that population read is not the lever, but left OPEN whether the "no biological-credit arm margin" was under-training (all arms at the single-layer floor at the smoke H40/ep30). This tests that: a trained-scale run (H96/ep120), 6-seed (42/43/44/100/101/102), K=1 (small bridge → tractable).

## Result — the net does NOT train at H96/ep120 (6-seed)

| seed | plain_fa | microcircuit | burstprop | 1-layer floor | oracle |
|---|---|---|---|---|---|
| 42  | 0.037 | 0.148 | 0.185 | 0.333 | 1.0 |
| 43  | 0.185 | 0.259 | 0.111 | 0.185 | 1.0 |
| 44  | 0.074 | 0.148 | 0.148 | 0.444 | 1.0 |
| 100 | 0.222 | 0.111 | 0.370 | 0.148 | 1.0 |
| 101 | 0.259 | 0.296 | 0.222 | 0.185 | 0.37 |
| 102 | 0.296 | 0.296 | 0.222 | 0.074 | 1.0 |
| **mean** | **0.179** | **0.210** | — | **0.228** | ~0.9 |

**Every arm is below chance (0.333) and near/below the single-layer floor** — the on-bridge spiking net does not learn the depth-2 task, while the rate oracle solves it (1.0, 5/6 seeds). `mc_beats_fa=True` on 4/6 is meaningless noise below the floor (microcircuit 0.21 "beating" a collapsed plain-FA 0.18). More epochs at lr 0.25 destabilized plain-FA (0.037 on seed 42) — the credit DIVERGES at this lr/epoch combination.

## Honest scope (not "BDSP fundamentally can't train")
This is **the configs tried don't train**, NOT a proven "no config can." The smoke (H40/ep30) under-trains; the trained attempt (H96/ep120, lr 0.25) is UNSTABLE (lr likely too high for 4× epochs). A proper **lr × epochs × stability sweep** is needed to conclude — and that is the point: each on-bridge run is ~1–2 hr (per-example spiking forward × epochs), so the sweep is intractable at per-example speed. **The wall-clock is the binding limit on resolving this** — the demonstrated case for a batched forward.

## The batched forward (the enabler) — 8–15× speedup, correctness OPEN
`research/runners/_batched_onbridge_forward_derisk.py`: evaluate M examples as M DISJOINT block-diagonal copies of the net on ONE bridge (M examples advance in ONE `_run_one_simulation_step`, amortizing the per-call overhead M×; the RF composer proved disjoint slices don't cross-talk).
- **SPEEDUP: 8–15× confirmed** (M=8: 79 → 6 ms/example; M=16: 8.9×). The per-call Python overhead of the step pipeline IS the numpy bottleneck, and batching amortizes it — a real, large lever.
- **CORRECTNESS: MISMATCH (max|batched−serial| ≈ 0.17), UNRESOLVED.** Ruled out (each still mismatches): OU background noise (`enable_ou_process=False`), threshold homeostasis (`enable_homeostasis=False`), per-copy init RNG (copy 0 mismatches too, 0.147 — so it is NOT just heterogeneity). ⇒ a genuine per-step GLOBAL operation couples the block-diagonal copies (some op sees all neurons despite no cross-copy synapses). Per systematic-debugging (3 fixes tried → step back), the next diagnostic is to isolate the global op (drive only copy 0 with an idle copy present → does its read shift?) or switch batching strategy (true batch-dim vectorization of the kernels vs block-diagonal copies).

## Next (the concrete chain)
1. **Resolve the batched-forward coupling** (isolate the global step op, or vectorize over a batch dim) → a correct 8–15× forward. This unblocks (2).
2. **lr × epochs × stability sweep** of the on-bridge BDSP depth-2 net (now tractable) → does ANY config train it above the floor? If yes → the boundary was config; if no → the point-neuron spiking deep credit genuinely can't train depth-2 → the dendritic two-compartment substrate (`enable_two_compartment_dap`, finer continuous-error credit — the standing priority) is the mechanism past it.
3. The dendritic surpass (`2026-07-12-dendritic-per-compartment-gain-SURPASSES-...`) already showed the DEVELOPMENTAL normalization works where learned credit doesn't; #2 tests whether learned credit can train at all on this substrate.

**Rigor:** 6-seed (42/43/44/100/101/102) trained-K1; controls intact (oracle ~1.0). Batched-forward de-risk: 3 coupling hypotheses ruled out. NO `sim/` edit. A first-class honest negative + a partially-de-risked enabler, with the exact next diagnostic named.
