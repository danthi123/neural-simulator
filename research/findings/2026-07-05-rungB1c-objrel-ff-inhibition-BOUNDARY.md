# objrel structural-read surpass via subtractive FF-inhibition — **honest BOUNDARY** (6-seed-blind); launches the learned-signed read

**Date:** 2026-07-05
**Runner:** `research/runners/_rungB1c_objrel_ff_inhibition_derisk.py`
**Test:** `tests/test_rungB1c_objrel_ff_inhibition.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_ff_inhibition.json`
**Research gate:** the objrel-surpass deep-research gate (this session) — verdict "rank-1 additive common-mode; subtractive FF-inhibition (catalog B.06), NOT divisive".

## The boundary being surpassed

The reservoir's comprehension→composition read-out is synaptic + spiking and works for CANONICAL SVO (role == position),
but the OBJECT-RELATIVE construction (objrel `the PAT that the AGT V`: slot0 = THEME, role ≠ position) FAILS on the
spiking WTA (objrel ~0/36) while a LINEAR argmax read gets objrel 100% — the info is present + linearly separable, so it
is NOT the Mikulasch-Priesemann decorrelation wall. Diagnosis (research gate + probe): a RANK-1 ADDITIVE COMMON-MODE
problem — the linear argmax is shift-invariant (ignores a uniform pedestal); the spiking WTA reads TOTAL drive (firing ∝
total incl. the Dale-shift baseline + the `WS_ENS_FLOOR_C2 = 150` floor), so the winner is the highest-TOTAL pool, not
the highest-DIFFERENTIAL pool.

## The mechanism tested (confound-free, biologically-grounded)

SUBTRACT the common mode before the WTA — a SHARED SUBTRACTIVE INHIBITORY POOL (feedforward PV+ FS interneuron, catalog
B.06; subtractive removes the DC pedestal, divisive does not). **Confound-free** (the CYCLE-919 lesson: +40 neurons flips
seed 42 canon 1.00→0.11): it does NOT add a new pool — it REPURPOSES the c2 WTA's EXISTING shared inhibitory pool (which
already has the B.06 topology: E from all 3 ens, I to all 3 ens) by marking its `wta_i2e` synapses GRADED, so its
continuous membrane tracks the mean linearly and subtracts it. The bridge is the BYTE-IDENTICAL c2 build (canon 1.00,
objrel 0.00 baseline reproduced at `w_i2e=0` — the valid baseline the broken `step11_centered_drive.py` never achieved,
since it drove the ens with the host logit as external current, bypassing the real Ws-shifted synapses). The read is the
REAL synaptic read (`run_with_ens`). 6-seed-BLIND (dev 42/43/44 select the graded op point; frozen + tested on 100/101/102).

## The result — **BOUNDARY** (6-seed-blind; NO anti-cheat weakened)

| seed | subtract-ON canon | subtract-ON objrel-slot0 | pedestal-off objrel-slot0 | scramble objrel-slot0 | recov / canon-ok / diff-LB / scr-collapse |
|---|---|---|---|---|---|
| 42 (dev) | 0.333 | 1.000 | 0.000 | 1.000 | T / **F** / T / **F** |
| 43 (dev) | 0.306 | 0.917 | 0.000 | 0.833 | T / **F** / T / **F** |
| 44 (dev) | 0.361 | 0.417 | 0.000 | 0.417 | F / **F** / T / T |
| 100 (blind) | 0.306 | 0.000 | 1.000 | 0.333 | F / **F** / F / T |
| 101 (blind) | 0.333 | 0.833 | 0.917 | 1.000 | F / **F** / F / **F** |
| 102 (blind) | 0.333 | 1.000 | 1.000 | 1.000 | T / **F** / F / **F** |

**VERDICT: BOUNDARY** (objrel recovered on only 3/6 overall, 1/3 blind; need ≥5/6 AND all blind).

**Why it fails — the anti-correlated see-saw.** On EVERY seed, canonical REGRESSES to ~0.33 with the subtraction on
(`canonical_not_regressed = False`, all 6). The graded I→E that lifts objrel-slot0 FLIPS the canonical winner too — the
subtraction SHIFTS the operating point rather than cleanly removing the DC to reveal the true differential. It is not a
"both-high" point; the op-point search MAXIMIZED `min(canon, objrel_slot0)` and the best achievable is canon ~0.33.
Moreover the "lift" is not even cleanly load-bearing (on 3/6 seeds reverting to the c2 spiking WTA did NOT collapse
objrel — a tuning artifact) and the scrambled-label control did NOT collapse on 4/6 (the read is riding a
position/heterogeneity artifact, not a genuine role read). A FIXED graded subtraction cannot resolve the sub-1%
structural margin through the spiking WTA — it lands on the wrong side of the WTA ignition inversion.

## The honest characterization + the next mechanism (this is NOT a wall)

The info is present + linearly separable (linear argmax = objrel 100% every seed), and subtractive FF-inhibition is the
biologically-correct common-mode family that CLEANLY reproduces the c2 baseline — but a FIXED read cannot adapt to each
draw's operating point. This is the project's documented common-mode / rate-code family, NOT the irreducible
Mikulasch-Priesemann wall — it is the **seed-adaptive-read frontier**. Per the boundaries-are-undiscovered-mechanisms
principle, this launches the named next mechanism: the **LEARNED-SIGNED delta read**
(`research/findings/raw/signed_conductance/step8_learned_signed.py`). The delta rule ADAPTS per-draw — it fits THROUGH
the spiking deploy (the f-I nonlinearity + WTA ignition-order are INSIDE the error term), which is exactly what
generalized the CANONICAL positive read 6/6 where every FIXED read was seed-fragile — extended to a SIGNED conductance
delivery so it learns a signed structural read that adapts to each draw's operating point. (De-risk in flight.)

## Honest scope

- The CANONICAL SVO conversational task (the production use case) is position-solvable and already works (the committed
  c3 learned read, 6/6). This gate is about the harder non-local structural (role-from-form) read — a real new
  capability, worth the cheap de-risk, not a blocker on the shipped one-brain turn.
- The de-risk is rigorous + confound-free (byte-identical c2 reservoir; the real synaptic read; 6-seed-blind; no
  anti-cheat weakened). Reuse-by-import; NO `sim/` edit (the graded flag is set runner-side on `cp_graded_synapse_mask`,
  the existing guarded per-step graded block).

## Files
- `research/runners/_rungB1c_objrel_ff_inhibition_derisk.py` — the confound-free graded-subtraction de-risk.
- `tests/test_rungB1c_objrel_ff_inhibition.py` — structural guard (op-point protocol + anti-cheat thresholds).
- `research/findings/raw/_rungB1c_objrel_ff_inhibition.json` — the 6-seed-blind boundary record.
