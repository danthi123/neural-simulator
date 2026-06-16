# Sentence-generation de-templating — cheap-first de-risk GO (CYCLE 104, 2026-06-16)

**Status:** the recommended cheap-first de-risk (from the verified deep-research scoping) **PASSED both phases, 6/6 seeds** — a neural **rate-coded competitive-queuing serial-order generator** produces the SVO word **order** on the spiking substrate, beating both anti-cheat controls. The host word-ordering f-string is replaceable by a neural mechanism. The full on-bridge build (Option 1) is viable.

## The question

The last host-code shortcut in the conversational OUTPUT is the word-ordering template — `rf_phasor_composer.py:346` `f"{agent} {action} {patient}"` (and the adjective-noun `" ".join` at `:321`), mirrored in `core_sim_composition.py:392`. The *per-word content* is already substrate-decoded (spiking unbind + cleanup); only the **order** is host code. Per the deep-research scoping (`2026-06-16-sentence-generation-biologization-deep-research.md`), this is a **serial-order production** problem, and the recommended mechanism is **competitive queuing** (Grossberg 1978; Bullock & Rhodes 2003; catalog G.07/H.19): a planning layer holds the slots with a primacy gradient, a choice WTA emits the highest-primacy slot then suppresses it (inhibition-of-return), so the next-strongest wins.

## The re-frame that made it tractable (verified)

A prior **closed-loop** HVC generator on this exact substrate already FAILED (`2026-05-16-generator-G1-songbird-NEGATIVE.md`: `mean_true 0.000`, `mean_reward 0.0000` every epoch, the generator's weights never moved). The verified root cause was NOT the generator: its **self-comprehension judge could not read order back** (encoded-vs-control AUC 0.775), so it gave zero gradient. The fix used here: the stored fact is an **external, unambiguous order teacher** — graded by an exact word-order comparison, not the substrate's own residual judge. This sidesteps the recorded failure mode.

## Results (both phases reuse the pre-registered anti-cheat harness `song_g1_core`)

| Phase | What it tests | True vs permuted-order control | No-learning control | Verdict |
|---|---|---|---|---|
| **A** (numpy core) | does CQ with a role→primacy gradient *learned from the fact-teacher* emit held-out facts in order? | **1.000 vs 0.333** (200% over, 6/6) | untrained primacy: 0.343 < perm 0.880 (fails) | **GO** |
| **B** (spiking substrate) | does the order survive on REAL spikes (primacy = graded current → rate ranking = order)? | **1.000 vs 0.333** (200% over, 6/6) | equal drive: 0.208 < perm 1.000 (fails) | **GO** |

Both controls behave correctly: the **permuted-order control** (same concept multiset, scrambled order) does not beat the true order, and the **no-learning control** (untrained primacy in A; equal drive in B) drops to chance — proving the order is *produced by the learned/graded primacy*, not concept-ignition or pool bias. The host-template baseline (order = 1.0 by construction) is matched.

## Mechanism (what the spiking substrate does)

The planning-layer **primacy gradient** is realized as **graded external current** into the (driven, non-attractor) concept pools of one fact — the highest-primacy role gets the most current. The spiking **rate tracks the drive**, so the per-pool rate **ranking = the emission order** (rate-coded competitive queuing — robust, not delicate first-spike latency). Read each pool's rate, order the fillers by rate, emit. Iterative emission uses inhibition-of-return (the project's `SaidTrace`, spiking). No host loop orders the words.

## Multi-frame extension (CYCLE 106): the seed of syntax

The phase-A/B de-risk validated a single FIXED frame (SVO). A follow-on asked whether the mechanism can learn
DISTINCT orders for DISTINCT frames (the seed of syntax) and keep them separate. `_phaseB_serial_order_multiframe_derisk.py`
gives each frame its own learned primacy gradient `prim[frame][role]` and adds a **cross-frame control** (the
other frame's order on the same fact must NOT match).

| Test | Frames | True vs permuted vs CROSS-frame | No-learning | Verdict |
|---|---|---|---|---|
| multi-frame (numpy) | F0=[agent,action,patient], F1=[patient,agent,action] | **1.000 vs 0.333 vs 0.000** (6/6) | 0.410 < perm 0.833 | **GO** |
| multi-frame (SPIKING substrate) | same | **0.991 vs 0.343 vs 0.005** (6/6) | — | **GO** |

The cross-frame score of **0.000** is decisive: the same fact is ordered *differently* under F0 vs F1, so the
mechanism is genuinely **frame-conditioned** (not one fixed order). ⇒ the substrate can learn frame-dependent
serial order — distinct orders for distinct reply frames (statement vs who/what vs yes-no vs "X and Y associated"),
and the route toward active/passive voice. Wiring `FrameCQ` into the agent's distinct reply frames is the
integration follow-on.

## Honest scope

- **De-risked GO:** the SERIAL-ORDERING step — the core of what the f-string does — runs on neurons (graded current → rate ranking → order), with the anti-cheat controls confirming it is order-from-gradient, validated 6/6 on the spiking substrate. This is sufficient to replace the **fixed SVO frame** the current composer uses.
- **Separately validated:** per-slot word spelling is the A→W read-out primitive (`concept_speak_demo`, 100% multi-seed) — drive an ordered slot's pool → decode its word. The de-risk emits ordered concept indices; A→W spells each.
- **Not yet built (the next step):** wire the spiking CQ serial-order generator + per-slot A→W into the conversational agent, replacing `render_fact`'s f-string end-to-end, and verify the conversational matrix (who/what answers, abstention/no-confab) is preserved.
- **Documented follow-on:** *learning different orders for different frames* (real syntax — SVO vs the "X and Y are associated" frame vs who/what answers). The current de-risk validates a single fixed frame (which is what the host template is); multi-frame order-learning (Pulvermüller sequence detectors, Option 3) layers on top.

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_serial_order_cq_derisk        # phase A (core)
SIM_BACKEND=cupy  python -u -m research.runners._phaseB_serial_order_spiking_derisk   # phase B (substrate)
```

Anti-cheats (all present): the permuted-ORDER control (primary gate), the no-learning control (must fail), held-out-only facts, the host-template baseline reported, the degenerate-tie guard (canonical candidate order), and the FIXED `g1_verdict` bars (margin ≥10%, abs-floor 0.5, never tuned).
