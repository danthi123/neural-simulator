# The banked "feedforward spiking deep credit is ALREADY GO (K=8 0.877)" is **80% a fixed random reservoir + a linear readout** — the frozen-hidden control existed in the code, unused, and the GO gate never included it

**Date:** 2026-07-16
**Runner:** `research/runners/_onbridge_eprop_port_derisk.py` (+ new `--freeze-hidden`). CuPy, banked config (`enable_bdsp=True`, `lr=0`, `--pool-k 8` = exactly what produced 0.877), seeds 42/43, ONE variable.
**Verdict:** the GO's NUMBER **reproduces** (FULL 0.889 vs banked 0.877). Its **mechanistic claim does not**: a fixed random spiking reservoir + a trained linear readout accounts for **80% of the margin above chance**; deep credit adds **+0.111**, and it is **seed-variable** (+0.185 / +0.037). The runner's **own aggregate gate returns `SIGNAL=False`** for both arms.

## The table

| seed | FULL | FROZEN (reservoir+readout) | deep-credit contribution | permuted | shuffle-DFA | chance |
|---|---|---|---|---|---|---|
| 42 | 0.852 | 0.667 | **+0.185** | 0.185 | 0.519 | 0.333 |
| 43 | 0.926 | 0.889 | **+0.037** | 0.370 | 0.556 | 0.333 |
| **mean** | **0.889** | **0.778** | **+0.111** | 0.278 | 0.537 | 0.333 |

**Above chance: reservoir +0.444 · deep credit +0.111 ⇒ the reservoir is 80% of the margin.**

- **FULL** = all FF pathways train — byte-identical to the config behind the banked 0.877 (and it reproduces it: 0.889).
- **FROZEN** = hidden FF pathways frozen at init; ONLY the last FF pathway (the host-side linear softmax readout, which `_accum_grad` already SKIPS from the e-prop/DFA rule, `:389 skip_output`) trains. Realized via the runner's OWN `train_layers` hook (`:153`).
- **The instrument was verified before the run** (the day's lesson): `default -> train_layers=None` (all pathways train, byte-identical) vs `--freeze-hidden -> train_layers={2}` (hidden pathways 0,1 skipped at `:361`). Had the default also frozen, both arms would be reservoirs and the verdict meaningless — while looking perfectly plausible.

## Why nobody saw it: the control existed, unused

`trains_the_task` (`:481`) gates on **chance / permuted / shuffle-DFA — NOT ONE is a frozen-hidden baseline**. So a pure reservoir+readout result passes the gate **unchanged**. And `train_layers` — documented in the file itself as *"None => update all FF pathways; a set => update only those (isolation)"* — appears **only** as its definition (`:153`) and its check (`:361`). **Someone built the isolation hook for exactly this purpose and it was never once invoked.** Today is the first time this control has ever been run.

This is the session's recurring shape: **the machinery to check the claim already existed; nothing invoked it.** (Cf. `_ensure_gate_capacity` guarding 7 sites but not the Hebbian one; a requirements file nothing audited; an install doc telling you to run a tool it never told you to install.)

## Two corroborating signals inside FULL's own numbers (independent of the frozen arm)

1. **shuffle-DFA sits at 0.537 against chance 0.333.** A large slice of performance is **credit-INDEPENDENT** — visible without any new control.
2. **The runner's own aggregate `SIGNAL=False`** for BOTH arms: *"HONEST NEGATIVE — the ported e-prop does NOT cleanly train the task on the bridge ... The exact residual: controls not clean."* Per-seed `trains_the_task` passes; the aggregate fails. **This contradicts the banked "6-seed GO" and needs running down.**

## What this does and does not say

**DOES:** the banked headline substantially **overstates** the mechanism. "Feedforward spiking deep credit is ALREADY GO / is NOT a blocker" should read: *the on-bridge e-prop port reaches ~0.89 held-out inheritance, but ~80% of that margin is a fixed random spiking reservoir + a linear readout; the deep-credit contribution is ~20%, seed-variable (+0.037..+0.185), and the runner's own aggregate control gate does not pass.*

**DOES NOT:** say deep credit is nothing. It is real and positive on both seeds. The 5-lens adversarial verify (wf_5473ce0f-8d5) concluded "the deep hidden credit contributes NOTHING (readout-only 0.630 BEATS full 0.556)" — **not supported here**; their arms used a **mixed config** (`pool_k=1` appears 17× alongside `pool_k=8`; epochs 40/120/150/200/250) and their FULL (0.556) does not reproduce the same-config FULL measured here, so their conclusion is scoped to their run. Their **structural** finding — that the gate has no reservoir control — is the one that mattered, and it is vindicated.

**HONEST SCOPE:** **n=2**, against the project's standing 6-seed rule ⇒ **INDICATIVE, not final**. The seed-variability is itself the story (+0.185 vs +0.037): a 6-seed FULL-vs-FROZEN is required before the record is rewritten. But n=2 is already enough to say the headline is **unverified as a deep-credit claim**, because the load-bearing control was never run at all.

## Consequences

- **Segment (b) of the longest pole** was to co-train "the stream cortex + the deep-credit learner". That learner is **80% reservoir**. Co-training it would mostly test co-residence of a *reservoir*, not of a second *learning rule* — which was the entire point of (b) (rule heterogeneity). **(b) is gated on this.**
- **`docs/plans/2026-07-15-months-scale-plan-...`** §4 opens the unification critical path with *"The learning rule (feedforward deep-credit / BDSP, GO)"*. Already corrected once today (BDSP→e-prop+population-coding); now the **GO itself** needs the 80/20 caveat.
- **The 2026-07-15 gate's** *"feedforward spiking deep credit is SOLVED (not a blocker) — the genuine open frontier is RECURRENT off-diagonal"* is **too strong**: the feedforward side is ~20% mechanism, 80% reservoir. The frontier is wider than the record says.

## Retractions this produced (all mine, all caught before entering the record)

1. **"The clamp SUPPRESSED the deep-credit GO (0.877→1.000)"** — DEAD. `--bdsp-wmax` is ONE config field but TWO functional variables: the clamp is global over `cp_connections.data`, which holds BOTH the spiking FF synapses AND the host-side linear readout (measured 864/1536 readout synapses crushed 500→≤6 per forward). Widening the clip freed a **linear classifier**.
2. **"shuffle-DFA refutes the reservoir hypothesis"** — VOID. `train_batch` shuffles the delta LIST (`:383-384`) and feeds the SAME shuffled `d` to BOTH the hidden DFA credit AND the readout's delta rule; it collapses even with the hidden pathways FROZEN. It is a second wrong-label control, a near-duplicate of permuted.
3. **"ff-moved 6.6M is churn / 798 is the genuine update"** — BACKWARDS. `ff_weight_norm` = `sum(|w|)`; `ff_moved` = `|L1_after − L1_before|` — a one-way norm difference, blind to direction. 798 is the ZERO-INIT readout growing to `sum|W|≈798` over 1536 synapses (mean ~0.5 = a textbook softmax solution): evidence **FOR** the hidden layers not learning.

## Next

1. **6-seed FULL-vs-FROZEN** on the banked config (42/43/44/100/101/102) — the standing rule; this is n=2.
2. **Add the frozen-hidden arm to `trains_the_task`** so a reservoir result can never pass the gate again — the durable fix, and the same shape as `tests/test_plasticity_inertness.py`.
3. **Run down the `SIGNAL=False` vs banked "6-seed GO" discrepancy** — same config, opposite aggregate verdict.
4. Then segment (b), on a learner whose deep-credit share is known.

## Artifacts

`research/findings/raw/_eprop_banked_{FULL,FROZEN}.{json,log}` · `--freeze-hidden` in `_onbridge_eprop_port_derisk.py` (default off = byte-identical) · verify workflow `wf_5473ce0f-8d5`.
