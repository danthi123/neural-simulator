# gap#5 OPT-0: the DC-clamp diagnosis is REFUTED by measurement — and my over-read of that measurement is retracted

**Date:** 2026-07-30 · **Status:** measurement landed; one diagnosis refuted, one of my own claims retracted
within the same cycle. Seed 42, GPU (`backend=cupy` recorded in the artifacts).

## 1. What was predicted, and what was measured

The adversarial round derived from `sim/kernels.py:325-345` + the apical ODE that `is_post` would be **pinned at
~34.82 mV for every reader at every position** (density=1.0 ⇒ ~12 coincident inputs per reader per step; `is_post`
is flat for `c_count>=4`). That was arithmetic, never a measurement — this arc's only view of its instructive
signal was a single scalar `apical_max`, which cannot distinguish "on once" from "on always".

Measured (arm A, density=1.0), spread ACROSS READERS:

| arm | circ | dW | apical max | apical min | **std across readers** |
|---|---|---|---|---|---|
| `lr0_btsp` | 0.0359 | 0 | −21.29 | −1024.72 | **305.14** |
| `btsp` | 0.0000 | 2251 | −81.30 | −767.72 | **206.77** |
| `lr0_btsp_wta` | 0.0359 | 0 | −69.12 | −1049.80 | **354.49** |
| `btsp_wta` | 0.0000 | 2251 | −32.47 | −770.50 | **182.15** |

**⇒ THE DC-CLAMP PREDICTION IS REFUTED.** Predicted spread ≈ 0; measured spread is 182-354 mV. `is_post` is not
pinned. This refutation does not depend on which `circ` variant is used, so it survives §3 below.

## 2. What IS true: density=1.0 drives the apical compartment non-physiological

Across the arm-B sweep, apical voltages are physiological at low density and absurd at 1.0:

| density | apical min | apical std | circ (final weights) | dW |
|---|---|---|---|---|
| 0.10 | −71.98 | 5.7 | 0.4459 | 91 |
| 0.25 | −71.87 | 19.3 | 0.1948 | 560 |
| 0.50 | −79.15 | 20.7 | 0.1219 | 1136 |
| 1.00 | **−1024.72** | **305.1** | 0.0000 | 2251 |

−1025 mV is not a membrane voltage. So density=1.0 IS a pathological operating point — but by
divergence/hyperpolarization, **the opposite of the predicted saturation**. Worth understanding on its own; it is
not the mechanism I claimed.

## 3. ⛔ RETRACTION OF MY OWN READING (same cycle, before it propagated)

From the table in §2 I reported that "`lr=0` beats BTSP at every density ⇒ BTSP is actively damaging the field ⇒
the gap#5 GO is in question." **That is WRONG and is withdrawn.** It compares two different quantities:

- The **headline** is `circ(dW)` — computed on the weight CHANGE, i.e. learning-only — scored against a **randset
  null**. Six seeds: `circ(dW)` mean **0.6705**, randset null **0.0822**, difference **0.5883**.
- **My arms measured `circ` on the FINAL weights**, which are dominated by the random initial structure. With
  `dW=0`, the `lr0` arm's final weights simply ARE the random init, and a sparse random vector over 60 place
  indices has a high circular resultant **by construction**. `lr0` scoring high there is expected and carries no
  information about learning.

⇒ **The gap#5 GO is NOT in question.** My arm B was mis-designed for the question it was built to answer: it
should have measured `circ(dW)` against the randset null, matching the headline. As run, it cannot bear on it.

Note what did and did not catch this: the `lr=0` arm worked exactly as intended and produced a real signal — the
error was mine in INTERPRETING that signal against a mismatched baseline. No check in the repo verifies that two
compared numbers are the same quantity, which is why this needed a manual trace to catch.

## 4. A smaller defect that DOES survive: "67% of oracle" is not like-for-like

`0.588 = 67% of the 0.8719 oracle` puts a **null-subtracted difference** (0.6705 − 0.0822) over a **raw,
un-subtracted ceiling**. The two terms are not on the same footing. This is the same shape as the retraction
already recorded in CLAUDE.md, where a MEAN was subtracted from a SUM and reported as an improvement.

This does **not** invalidate the learning result, which carries its own randset control. It means the **"67%"
ratio specifically** should not be quoted until recomputed with both terms treated identically. Also note that
`0.8719` appears as the max `circ` in every gap#5 artifact because it is the σ=5 **oracle row of the metric
validation block** — an input, not an achieved score. The string `0.588` appears in NO gap#5 artifact at all; it
exists only as prose in the findings doc, computed by hand from the two columns above.

## 5. Next

Re-run arm B measuring `circ(dW)` against the randset null, matching the headline exactly — a small change to
what is already built. Until then, the density sweep says nothing about whether sparser wiring helps learning.
