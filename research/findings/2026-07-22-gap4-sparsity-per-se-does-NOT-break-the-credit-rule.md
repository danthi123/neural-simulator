# gap#4 — SPARSITY per se does NOT break the biological credit rule; the on-bridge negative is a narrower burst-credit/op-point issue, not a sparse-code wall

**2026-07-22, CPU/numpy, coexisting with the fluency training.** The MNIST de-risk showed the biological local credit rule
(FA/DFA) beats a reservoir on a proper deep task in RATE (6-seed). The deep-research's remaining gap#4 open piece (cause #3):
the same BDSP algorithm reaches accuracy on numpy graded signals but degenerates ON the sparse spiking bridge (firing
0.04-0.07) -- the sparse spike code may not carry the graded class signal. This isolates the SPARSITY effect cheaply:
`_gap4_sparse_hidden_credit_derisk.py` makes the DendriticMLP hidden a SPARSE-BINARY code (top-k% active, spiking-like) with
a straight-through estimator (forward = binary; credit derivative = the dense sigmoid derivative), and sweeps sparsity.

## Result (seed 42; 43/44 confirming) — FA beats the reservoir at EVERY sparsity, gap GROWS
| hidden sparsity | FA | RESERVOIR | gap |
|-----------------|-----|-----------|-----|
| 100% (dense)    | 0.929 | 0.744 | +0.186 |
| 20%             | 0.885 | 0.685 | +0.200 |
| 10%             | 0.834 | 0.612 | +0.222 |
| 5%              | 0.753 | 0.520 | +0.233 |
| **2% (spiking-like)** | **0.627** | **0.360** | **+0.267** |

Both FA and reservoir accuracy decline with sparsity (a 2% binary code carries less information) — but **FA beats the
reservoir at every sparsity level, and the ADVANTAGE GROWS as the code gets sparser** (a sparse random reservoir is even
more useless, so credit-training the hidden matters MORE). ⇒ **sparsity per se is NOT the blocker for the biological
credit rule.**

## Implication — the gap#4 frontier narrows again
The on-bridge BDSP negative (degenerate at firing 0.04-0.07) is therefore **NOT a fundamental sparse-code limit** — the
credit rule survives a 2% sparse-binary code and still beats a reservoir. So the on-bridge issue is a NARROWER, more
tractable diagnosis: the burst-credit MECHANISM specifically (the stochastic Bernoulli burst sampling `B ~ Binomial(k,p)/k`
vs the graded credit), or the on-bridge OPERATING POINT, or population size -- NOT that a sparse code cannot carry credit.
Combined with the MNIST 6-seed reframe, the gap#4 keystone is much more open + tractable than "credit can't build accuracy":
the RULE works (rate AND sparse); the residual is the specific on-bridge burst-credit realization.

Honest scope: this uses a straight-through estimator (STE) for the credit -- the standard spiking-net surrogate, NOT the
on-bridge burst credit. It decisively rules OUT sparsity-per-se as the blocker; the on-bridge burst-credit-specific test is
the narrowed next step. NO `sim/` edit. `research/findings/raw/gap4/sparse_hidden_{seed42,2seed}.log`.
