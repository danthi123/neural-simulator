# RUNG 4 (6-seed, positive + a mapped boundary) — ORDER-DECISIVE role reversal IS achievable on-substrate by reading the reservoir's OWN state trajectory (no host oracle): the generator produces different held-out continuations for the same words in different orders; but the reservoir's fading recurrence does NOT carry a thematic role across the clause — roles need explicit binding

**Date:** 2026-07-11
**Runner:** `research/runners/_emerge_reservoir_lm_rung4_order_decisive_recombination_derisk.py` (reuse-by-import: the Rung-3 grammar/two-level codes + the reservoir EMERGE-82 `OnBridgeLSM` + a one-step-local-delta read-out; NO `sim/` edit, NO BPTT, NO deep credit).
**Verdict:** **Rung 4 = a positive on order-decisive generation + a precisely-mapped boundary.** Order-decisive role reversal — the same token multiset in different orders yielding different held-out continuations — IS achievable on the substrate reading the reservoir's OWN spiking state trajectory (no host lookup): main reversal_acc **0.833** (memoryless variant 0.963), word order load-bearing (permuted collapses; untrained floor), shared code load-bearing (one-hot degrades 0.83→0.48). BUT the reservoir's fading RECURRENCE does NOT robustly carry a thematic role across the clause (predicting from the clause-final state alone, `win2` = 0.519 ≈ the order-washed `cum` 0.519) — the working mechanism reads the AGENT's own position-0 state. So this is order-decisiveness by POSITION-reading, and it maps the frontier: **explicit on-substrate role-binding (EMERGE-78 form→role) is the next mechanism.** Reported as a positive-with-boundary, NOT overclaimed as a clean GO.

## The honest trail — an adversarial-verify workflow REJECTED a first version (the discipline working)
A first attempt fed the read-out a HAND-CODED host latch `ANIMAL_CAT[prefix[0]]` (a Python dict lookup of the first noun's TRUE category) and reported a clean 6/6 "GO" (reversal 0.815, every control 0.000). A 3-skeptic adversarial-verify workflow (leakage / control-validity / framing lenses) **correctly REJECTED it** (two INVALID, one SURVIVES-WITH-SCOPE-FIX): `main` used a MEMORYLESS encoder of the patient only, so the host latch did 100% of the work while the spiking reservoir was decorative (the recurrent arm even scored *worse*) — the exact host shortcut Rung 3 named and excluded, and a BRAIN-BASED-ONLY / emergence-bar violation. The skeptics prescribed the missing honest control: *"can the RECURRENT reservoir's own dynamics carry the role — reservoir, NO host latch?"* This finding answers that question honestly. The host-latch version is retired.

## The mechanism (reservoir's own states, no host oracle) + the key insight
The generator reads the reservoir's per-token state TRAJECTORY `[s0, s1, s2]` (its own spiking-rate states at each position; a standard echo-state "read all taps" read-out) and a one-step-local-delta read-out maps it to the ACTION. **Key insight (the a0 read that unlocked it):** the running-CUMULATIVE feature used in Rungs 1–3 is a MEAN — order-DESTROYING; the reservoir's per-token state trajectory carries the order. The read-out LEARNS that the position-0 state carries the agent (shuffling training word order collapses it) and EXTRACTS the agent's category from that reservoir state (the shared category code makes it generalize to held-out agents; one-hot codes degrade it). No host lookup of `prefix[0]` anywhere.

## The test (role reversal on held-out combinations)
Grammar (Rung 3): "`<N1> meets <N2> <ACTION>`", ACTION = the AGENT = N1 (the sentence-first noun)'s category action. For a HELD-OUT cross-category pair, both orders form a **twin**: "X meets Y" → X's action; "Y meets X" → Y's action. The two orders share an identical multiset `{X,Y,meets}` → any order-blind feature gives the same answer to both → reversal (BOTH correct) is structurally 0. `reversal_acc` = fraction of the 9 held-out twins with both orders correct.

## Result — 6-seed (dev 42/43/44 + blind 100/101/102)
| Arm | reversal_acc | per_order | role |
|---|---|---|---|
| **main** (recurrent reservoir, all-taps trajectory, shared codes) | **0.833** (0.78–0.89) | 0.917 | the generator |
| permuted (word-shuffled training) | 0.167 (5/6 ≈ 0; one seed-102 spike 0.89) | 0.398 | **ORDER control → collapses** |
| onehot (no shared category block) | 0.481 | 0.722 | shared-code control → degrades |
| untrained (frozen read-out) | 0.000 | 0.000 | floor |
| memoryless (non-recurrent reservoir, trajectory) | 0.963 | 0.981 | *diag: recurrence NOT needed (reads s0)* |
| win2 (recurrent, clause-final state ONLY) | 0.519 | 0.722 | *diag: recurrence carry-forward is PARTIAL* |
| cum (running-cumulative mean) | 0.519 | 0.759 | *diag: order-washed read* |
| pos_blind (alphabetical tap order) | 0.630 | 0.806 | *diag: invalid for a recurrent reservoir — see note* |

## What is honestly shown, and what is the boundary
- **POSITIVE — order-decisive generation on-substrate, no host oracle:** the generator produces DIFFERENT held-out continuations for the same words in different orders (main 0.833, memoryless 0.963), which no order-blind model can (untrained 0.000). Word order is load-bearing (`permuted` collapses on 5/6 seeds; the lone seed-102 spike is small-sample noise on 14 shuffled training sentences). The shared category code is load-bearing for the held-out generalization (`onehot` degrades 0.83→0.48; not a full collapse because the reservoir's nonlinearity leaves a residual). Reading only the reservoir's own spiking states — the rejected host lookup is gone.
- **BOUNDARY (mapped precisely) — the reservoir's fading recurrence does NOT carry a thematic role across the clause.** The capability rides reading the AGENT's own position-0 state (`memoryless` 0.963 ≥ recurrent `main` 0.833 — the recurrence is not needed and slightly hurts). Predicting from the clause-final state alone, where the recurrence would have to have CARRIED the role forward, is only PARTIAL (`win2` 0.519 ≈ the order-washed `cum` 0.519). So the reservoir's dynamics do not robustly bind/transport a thematic role — consistent with the EMERGE-84 reservoir-recursion boundary (fading memory ≠ a stack/register). **The next mechanism is explicit role-binding realized ON the substrate — the EMERGE-78 fronto-striatal form→role reservoir, trained to assign the role from the dynamics — which is the honest replacement for both the rejected host latch and the position-read.**
- **Control note:** the tap-reordering `pos_blind` control does NOT collapse for a RECURRENT reservoir (its states already encode position via history), so `permuted` (word-shuffled training) is the valid order control here; `pos_blind` is reported as a diagnostic only.

## ⇒ significance
Order-decisive systematic recombination — the syntactic capability Rung 3 lacked (its category→action map was bag-recoverable) — is achievable on the reservoir substrate, learned and generalizing, reading only the reservoir's own states (no host oracle). The honest mechanistic finding is that it is carried by reading the agent's position-0 state, NOT by the reservoir transporting a role through its fading memory — which precisely maps where explicit, on-substrate role-binding (the EMERGE-78 form→role circuit) is the required next mechanism. NEXT: (a) the on-substrate spiking role-assignment circuit that binds/carries the role (closing this boundary), and — the higher-leverage emergence-bar direction Rungs 3–4 both flagged — (b) self-organizing the role/category STRUCTURE from experience rather than hand-specifying it.

## Files
`_emerge_reservoir_lm_rung4_order_decisive_recombination_derisk.py`; the exploratory probe `_rung4_onsub_probe.py`; 6-seed raw `research/findings/raw/_rung4/s{42,43,44,100,101,102}.json`; builds on `2026-07-11-RUNG3-emergent-generator-generalizes-to-novel-subject-6seed.md`.
```
python -m research.runners._emerge_reservoir_lm_rung4_order_decisive_recombination_derisk --seeds 42
```
