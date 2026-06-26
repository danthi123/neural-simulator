# 1454-concept breadth: window sweep — recall peaks at 3K, densifies after 8K; 7K = the working first-chat brain

**Date:** 2026-06-26
**Context:** scaling the foundational-curriculum stream cortex to **1,454 concepts** (the combined TinyStories + full Simple-English-Wikipedia corpus; the ~1–1.5K first-chat target). The full 150K-window run crashed recall to 0.208; CYCLE-606 diagnosed it as **over-training densification** (the co-occurrence weights go fully dense → the *distinguishing* structure washes out into a uniform background → codes blur → recall collapses; offline-confirmed: no normalization of the 150K weights recovers recall). This sweep finds the optimal window count.

## Recall-vs-windows curve (1454c, n_per 10, n_hub 500, combined corpus; moat 0-FA at EVERY point)

| windows | recall | corr(M,C) | note |
|---|---|---|---|
| **3K** | **1.000** (48/48) | 0.853 | peak recall; thinnest rare-concept coverage |
| 5K | 0.917 | 0.863 | |
| **7K** | **0.958** (46/48) | 0.869 | **best balance** — near-peak recall + highest corr + most pre-densification coverage |
| 8K | 0.917 | 0.868 | |
| 16K | 0.667 | 0.868 | densification declining |
| 150K | 0.208 | 0.758 | full crash (the original run) |

## Mechanism

Recall peaks early (3K perfect) and declines monotonically after ~8K — the densification. Note **corr(M,C) plateaus ~0.868 (7–16K) while recall falls**: the crash is the co-occurrence *structure* diluting, not a read-out-fidelity loss. The **no-confab moat held 0 false-accepts across the entire curve** — abstention is robust to the densification (it never fabricates, even as recall degrades).

## The working first-chat brain

- **Recommended: `bridges/firstchat/brain1454_w7000_seed42.npz`** — recall 0.958, moat 0-FA, corr 0.869, 1,454 concepts. Best recall+coverage+corr balance → richest discursive adjacency.
- Alternative: `brain1454_w3000_seed42.npz` — perfect recall (1.000), thinnest coverage (max-recall pick).

## Honest scope

A single bridge has an inherent **recall-vs-coverage tradeoff** (fewer windows → better recall + thinner rare-concept coverage; more windows → coverage but densification crash). The working brain covers the *frequent* concepts well — good for a first chat — but the rarer tail of the 1,454 stays thin. The full-1,454-with-coverage needs the **multi-bridge split** (each bridge a smaller proven-recipe tier at n_per 16–24, like the 320-tier that held recall at 150K) — the next-step follow-on.

The owner's "sweep under 8K" steer found the 3K recall peak (1.000), above the prior 8K (0.917).
