# Methodology — anti-cheat control validity: prefer INPUT-DESTRUCTION controls; FIXED-RANDOM-CODE controls are unreliable in small setups

**2026-07-02 (autonomous; surfaced while validating the EMERGE-30..34 emergence-from-experience de-risks, confirmed by a read-only adversarial-audit workflow).** A control-validity issue and its fix, applicable to every de-risk that uses a "collapse" control to prove a learned structure is real.

## The issue
A de-risk claims "structure X emerges / is learned" and backs it with controls that should COLLAPSE to chance if the claim is false. Two kinds of collapse-control exist and they are NOT equally reliable:

- **(A) INPUT-DESTRUCTION controls** — permuted-context / permuted-pool / permuted-features / per-image pixel scramble / label shuffle. They keep the FULL mechanism intact and destroy only the category/relational structure IN THE INPUT. **Reliable:** if the emergence is real, the mechanism cannot form the structure from structureless input, so the metric drops to chance. Seed-dependent by construction; collapse is mechanistic, not luck.
- **(B) FIXED-RANDOM-CODE controls** — "no-pooler" / "no-learning" arms that replace the LEARNED codes with random codes. **Unreliable in small setups:** over a small representation space (e.g. 80 columns, K=12 active) where the training set covers most of the space, a FIXED random held-out code can COINCIDENTALLY overlap its category's training union enough to inherit — or not. The result (0.00 or 1.00) is a LUCK artifact of the specific fixed codes, especially when the codes are seed-INDEPENDENT (identical every seed).

## How it showed up
EMERGE-34's first 6-seed had its NO-POOLER control at **1.00** (did NOT collapse) — the random codes were seed-independent and coincidentally aligned. EMERGE-33's NO-POOLER control gave **0.00** — the OPPOSITE luck, an equally coincidental "clean collapse" that a GO gate then leaned on. Both were invalid controls masquerading as strong evidence (one falsely failing, one falsely passing).

## The fix (applied to EMERGE-33 + EMERGE-34; EMERGE-30/32 were already immune)
1. **Gate on an INPUT-DESTRUCTION control, never on a fixed-random-code control.** EMERGE-33 now gates on permuted-features (mixed-pool inputs → no category structure); EMERGE-34 on per-image pixel scramble (destroys visual similarity). Both collapse to ~chance mechanistically.
2. **Make any random-code control SEED-DEPENDENT** (`default_rng(seed*10000 + hash(item))`) so it varies per seed and averages toward chance instead of riding one lucky fixed draw. Keep it only as a reported secondary check.
3. **Use MORE held-out members** (≥3/category) so the accuracy metric is finer than the coarse {0, 0.5, 1.0} of 1-held-out-per-category, and the collapse-control averages cleanly.

**Audit verdicts (read-only adversarial workflow, one agent per runner):** EMERGE-30 **ROBUST** (permuted-context 0.50 vs 1.00, structurally immune — no random codes, codes hard-coded + associations grown from p_init=0); EMERGE-32 **ROBUST** (permuted-pool 0.29 vs 1.00); EMERGE-33 **RESTED ON A COINCIDENTAL CONTROL** → fixed; EMERGE-34 (original) **RESTED ON A COINCIDENTAL CONTROL** → fixed.

## The standing lesson (for future de-risks)
- The load-bearing collapse-control must be **input-destruction** (destroy the structure in the input, keep the mechanism). A `no-learning` / `lesion` mechanism-ABLATION control is also reliable (deterministic mechanism removal), but a **random-code substitution is not** in a small representation space.
- Never let a GO gate's strict condition rest on a fixed-random-code control. Report it if useful, but gate on the input-destruction + mechanism-ablation controls.
- Widen the setup where feasible (more held-out members, >2 categories to lower chance) so the real-vs-chance margin is large and per-seed control noise is small.

## Artifacts
Audit workflow run `wf_97c57fd7-1d1`. Affected runners: `_emerge3{3,4}_*_derisk.py` (fixed), `_emerge3{0,2}_*_derisk.py` (confirmed robust). Prior: `2026-07-02-emerge3{0,2,3,4}-*.md`.
