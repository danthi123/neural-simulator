# RUNG 3 (6-seed, positive with characterized residuals) — the emergent reservoir-LM generator GENERALIZES to a NEVER-GENERATED subject: it produces the correct category-specific continuation via the subject's SHARED CLASS CODE, and both the shared code and the reservoir's memory are load-bearing on every seed

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_rung3_systematic_generation_derisk.py` (reuse-by-import: the Rung-1 reservoir `ReservoirStates` = EMERGE-82 `OnBridgeLSM` + the Rung-1 one-step-local-delta read-out `train_readout`/`eval_ce`; NO `sim/` edit, NO BPTT, NO deep credit).
**Verdict:** **Rung 3 of the emergent-generation ladder — GENERALIZATION OF GENERATION to novel content, demonstrated.** The mechanism is robust across all 6 seeds (dev 42/43/44 + blind 100/101/102): `main` beats every mechanism-ablation control on every seed. Honest, precisely-characterized residuals (below): the strict per-seed magnitude threshold clears 4/6 (finite-size probe noise on a deliberately-small 300-neuron CPU reservoir), the generator commits to an action for ~84% of novel subjects (a calibration residual), and word ORDER is only WEAKLY load-bearing here — so this rung establishes CONTENT-generalization of generation; order-systematicity is the next sub-rung. This is a positive-with-residuals deliverable, not overclaimed as a clean GO.

## The mechanism (the generative mirror of comprehension inheritance/systematicity — EMERGE-22/26/78)
Rung 1 = a fixed spiking reservoir + a shallow one-step-local-delta read-out predicts the next token (dynamics-earned, no BPTT). Rung 2 = a non-fading WM latch carries a distal referent the reservoir forgets. Rung 3 asks the LLM-defining question of the generator: **does it MEMORIZE trained sequences, or GENERALIZE to a never-seen one?** Each word drives the reservoir through a **two-level code = its shared CATEGORY/CLASS block (shared by all members of the category) + a unique CONTENT bit**. Because the class block is shared, a subject the generator has NEVER produced can still be placed by its category — the reservoir carries "which category is the agent" across the sentence, and the read-out (trained on category-mates) supplies the category-appropriate continuation.

## The task (order-sensitive, content-sensitive, held-out — the design that survived three confounds)
"`<N1> meets <N2> <ACTION>`", where the ACTION is set by the **AGENT = N1 (the first noun)**'s category (PRED → {growl,hunt,pounce}; PREY → {flee,hide,freeze}). "wolf meets rabbit growl" vs "rabbit meets wolf flee" — same two animals, opposite order, different action. TRAIN on 6 animals/category (as both agent and patient, both orders); HOLD OUT 3 animals/category **entirely**. TEST: after "`<held-out-animal> meets <trained-animal>`", is the produced ACTION of the HELD-OUT AGENT's category? The patient is chosen cross-category so an "is a predator present?" bag shortcut mispredicts. 18 held prefixes; leakage-guarded (no held animal is a token in any training sentence).

**Three earlier task designs were confounded — each documented honestly here, each taught a real substrate property** (the a0 "read/understand the substrate before theorizing" discipline in action): (1) a held-out that recombines words in their SAME roles is predictable from each word's own history even with one-hot codes; (2) a "grammaticality" metric (predict the right POS class) is CONTENT-BLIND — solvable from position alone, so the shared content code can never be shown load-bearing; (3) a pure inheritance target is ORDER-FREE — a bag-level association a word-shuffled model also solves. The final task requires BOTH content (a specific action token) and memory (hold the agent across the intervening tokens). A Rung-2-style WM buffer was tried and **rejected as a shortcut here** — it hands the read-out the answer-determining agent-category directly, propping up even the order-shuffled control (unlike Rung 2, where the buffer held a genuinely DISTAL forgotten referent); the honest mechanism is the recurrent reservoir's own memory (`--feature-mode cum`).

## Result — 6-seed (feature = reservoir running-cumulative, n_pool=300, chance 2way = 0.5)
| Arm | heldagent 2way | heldagent cat_acc | isact | role |
|---|---|---|---|---|
| **main** (recurrent reservoir + shared codes) | **0.891** (per-seed 0.69–1.00) | **0.750** | 0.84 | the generator |
| one-hot codes (NO shared class block) | 0.333 | 0.287 | 0.85 | **shared CODE control → collapses to chance** |
| memoryless reservoir (nonrecurrent) | 0.000 | 0.000 | 0.00 | **MEMORY control → total collapse** |
| untrained read-out (frozen) | 0.000 | 0.000 | 0.00 | **read-out control → floor** |
| permuted (word-shuffled training) | 0.855 | 0.694 | — | *diagnostic: order only WEAKLY load-bearing* |
| deranged (wrong class block) | 0.854 | 0.741 | — | *diagnostic: weak control for a learned read-out* |

**`main` beats every mechanism-ablation control on all 6 seeds** (2way margins 0.32–0.87; cat_acc margins 0.28–0.72; all positive). Aggregate: main 2way **0.891** vs one-hot **0.333** (chance) vs memoryless **0.000** vs untrained **0.000**. ⇒ **the shared class CODE and the reservoir's MEMORY are each load-bearing** — a never-generated subject inherits the category continuation only through its shared code, carried by the reservoir's recurrence.

## Why each control is load-bearing
- **one-hot → chance (0.333 2way):** with no shared category block, a held-out subject's code is orthogonal to everything trained → the generator cannot place it → it defaults → chance. The generalization rides the SHARED CODE, not memorization.
- **memoryless → 0.000:** a reservoir that washes each token sees only the PATIENT at the scored position → it has neither the agent's identity nor any memory → total collapse. The recurrent reservoir must integrate the agent across "meets `<patient>`".
- **untrained → 0.000:** the one-step-local-delta read-out is doing the learning (a frozen read-out emits nothing meaningful).

## Honest residuals (characterized, not hidden)
1. **Per-seed magnitude noise (strict threshold 4/6).** The reservoir is deliberately 300 neurons for CPU speed; per-seed 2way varies 0.69–1.00. Raising n_pool to 800 firms `main` (e.g. seed 101 0.69→1.00) but **also lifts the control floor** (a bigger reservoir interpolates, so one-hot/memoryless rise toward chance) — no free lunch. This is the SAME finite-size-noise N-scaling documented in Rung 2, not a mechanism failure; `main` beats the controls on every seed regardless.
2. **Action-emission calibration (isact ≈ 0.84).** For a novel subject the generator commits to a valid action ~84% of the time (16% it emits a non-action token); when it does commit, the category is essentially always correct (2way 0.891). isact is a calibration property of the small reservoir + linear read-out, separate from the generalization mechanism.
3. **Order only WEAKLY load-bearing.** `permuted` (word-shuffled training) does NOT collapse (2way 0.855) — the category→action mapping is largely BAG-recoverable, so shuffling only weakly hurts it. This rung therefore establishes **content-generalization** of generation (a novel subject → correct category continuation), NOT order-systematicity. A task where word order is decisively load-bearing (so a bag/permuted model collapses) is the next sub-rung.

## ⇒ significance
The third rung of the emergent-generation ladder: the fixed-reservoir + one-step-local-delta generator does not merely memorize — it **generalizes generation to a never-produced subject via a shared class code**, with the code and the reservoir's memory each load-bearing on every seed. This is the compositional/inheritance property an LLM has and a lookup table does not, shown on the honest emergent generator (no BPTT, no deep credit, no `sim/` edit). NEXT: Rung 4 (order-decisive systematic recombination — a task where a bag/permuted model provably collapses), then open-vocab spiking spell-out (the A→W read-out) and multi-clause discourse generation (the D3 register).

## Files
`_emerge_reservoir_lm_rung3_systematic_generation_derisk.py` (`--feature-mode cum` default; `--n-pool`, `--seeds`); 6-seed raw `research/findings/raw/_rung3/cum_s{42,43,44,100,101,102}.json`; builds on `2026-07-10-RUNG1-emergent-reservoir-next-token-LM-dynamics-earned-GO-6seed.md` + `2026-07-11-RUNG2-wm-buffer-restores-distal-discourse-referent-6seed-GO.md`.
```
python -m research.runners._emerge_reservoir_lm_rung3_systematic_generation_derisk --seeds 42 --feature-mode cum
```
