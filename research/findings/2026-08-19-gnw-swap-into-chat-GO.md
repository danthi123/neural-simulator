---
status: live
type: finding
lane: laneC
date: 2026-08-19
integration_faculty: gnw-thought-swap
---

# GNW NEURAL THOUGHT-SWAP reaches the LIVE `/api/brain-chat` brain — a per-session held-TOPIC workspace whose one ignited coalition is the current conversational thought, SWAPPED by the reused 6/6-seed neural swap machinery when a user turn is a genuine TOPIC CHANGE (a salient MISMATCH between the new input and the held content), and HELD on a same-topic follow-up. Verified through the REAL handler: topic-change swap rate 1.00, same-topic/no-topic hold rate 0.00; byte-identical with the flag OFF (no `gnw_swap` key) and answer-unchanged with it ON. DEFAULT-OFF (a reversible flag pending owner review). GO.

**Date:** 2026-08-19 · **Board:** #77 (INTEGRATION-TO-PRODUCTION). **Backend:** CPU (numpy). **Verdict:** **GO** through the real `/api/brain-chat` handler (in-process). **No `sim/` edit** (`git diff sim/` empty).

**Files:** `webapp/gnw_thought_swap.py` (NEW — the per-session `ThoughtSwapWorkspace` + `observe_turn` + the grounded-topic extractor, reuse-by-import), `webapp/server.py` (the guarded observe block + the `_GNW_SWAP_DEFAULT_ON=False` anchor + the additive `gnw_swap` attach on both main return paths), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `gnw-thought-swap`). **Artifact:** `research/findings/raw/_gnw_swap_chat/verify.json`.

**Reproduce:** `SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u research/findings/raw/_gnw_swap_chat/verify_swap.py` — the verification drives `webapp.server.brain_chat(BrainChatRequest(...))` directly (the REAL handler): part (A) a continuous conversation, part (B) an OFF/ON md5 panel under per-turn RNG control. (First turn ~90s = the full tiny-demo brain + organ warmup; later turns ~8s.)

## What this closes
The GNW thought-swap was fully NEURAL end-to-end at the de-risk level — eviction (`2026-08-19-gnw-recurrence-weaken-swap-GO`, Rung-2d short-term depression drains the incumbent's own recurrent loop → self-collapse), admission (`2026-08-19-gnw-neural-vacancy-gate-GO`, a spiking dis-inhibitory vacancy gate admits the challenger with zero host injection), decision (`2026-08-19-gnw-neural-swap-intention-GO`, a spiking mismatch/salience detector fires only for a salient proposal that does NOT match the held content and triggers the swap) — each 6/6-seed GO, but each honest-limit #5 was the same: *not yet reachable from `/api/brain-chat`*.
This finding closes that gap. It also supplies the cross-turn HELD-CONTENT register the earlier live GNW gates lack: `webapp/gnw_deliberation.py` (single-hop) and `webapp/gnw_multistep_deliberation.py` (multi-step) both RESTORE their workspace to a clean snapshot each turn — they do not hold a thought ACROSS turns.
The swap workspace does: one ignited coalition = the current conversational TOPIC, persisting turn-to-turn and swapping on a topic change.

## The mapping (host boundary vs neural decision)
- **TOPIC (host comprehension of the world/teacher input — the SAME declared boundary the SVO question parser occupies):** each turn the user message's topic = the FIRST GROUNDED concept token in the message (a known agent/patient in the brain's store, read off `composer.kb`). An anaphoric / no-new-concept follow-up ("what does it chase?", "tell me more") yields NO new topic → the held thought persists. Restricting the topic to a KNOWN concept filters action verbs and unknown words and keeps the workspace grounded (moat-consistent: only a concept the brain knows can become a held topic).
- **DECISION (the substrate):** given the held-topic slot (incumbent) and the incoming-topic slot (proposal), the reused `run_intention_swap` drives the proposal into the spiking mismatch/salience detector + the vacancy gate. A DIFFERENT incoming topic is a salient MISMATCH → mm fires → the recurrence-depression boost evicts the incumbent → the vacancy gate admits the newcomer (a SWAP). The SAME incoming topic MATCHES → the pred held-content interneuron vetoes mm → no boost → the incumbent holds (NO swap). The swap-vs-hold verdict is the neurons', not a host `if`.

## Result — the DELIVERABLE, through the REAL handler
### (A) A continuous conversation — topic-change SWAPS, same-topic/no-topic HOLDS
One `/api/brain-chat` session (`tiny-demo`, stub renderer), `BRAIN_GNW_SWAP=1`. Each turn's `gnw_swap` read (from the handler's JSON response):

<!--derived-->

(`0.333` = the period-3 ignited plateau, rounded from the artifact's full-precision `new_rate_post`/`held_rate_post` = 0.3333333333333333; the exact per-turn values are in the cited `verify.json`.)

| # | user turn | kind | swapped | held_topic | evicted | n_ignited | new_rate | old_residual |
|---|---|---|---|---|---|---|---|---|
| 0 | what does the dog chase? | establish | False | dog | — | 1 | 0.333 | 0.000 |
| 1 | what does the dog chase? | same-topic | False | dog | — | 1 | 0.333 | 0.333 |
| 2 | what does the brain use? | topic-change | **True** | brain | dog | 1 | 0.333 | 0.000 |
| 3 | what does the brain store? | same-topic | False | brain | — | 1 | 0.333 | 0.333 |
| 4 | what does the cat eat? | topic-change | **True** | cat | brain | 1 | 0.333 | 0.000 |
| 5 | tell me more | no-topic | False | cat | — | — | — | — |
| 6 | what does the dog chase? | topic-change | **True** | dog | cat | 1 | 0.333 | 0.000 |

**Swap rate: topic-change = 1.00 (3/3); same-topic + no-topic = 0.00 (0/3).** On every swap the OLD topic reads at BASELINE (`old_residual` ~0.000), the NEW topic is ignited (`new_rate` 0.333 = the period-3 ignited plateau), and exactly ONE coalition holds (`n_ignited == 1`). A same-topic follow-up (turns 1, 3) is vetoed by pred and the current thought persists; a no-topic follow-up (turn 5, "tell me more") holds without even running the substrate.

### (B) Byte-identical with the flag OFF, answer-unchanged with it ON
A 6-turn panel (recall · recall-after-change · abstain · multi-step chase · self/identity · follow-up) run OFF then ON under identical RNG seeding per turn (some base-system turns — a curiosity-augmented abstain, a no-topic follow-up — sample a follow-up off the process-global RNG and are non-deterministic ACROSS runs regardless of the swap, so the seed is fixed per turn to isolate whether ENABLING the swap changes the answer):

| user turn | class | OFF md5 | ON md5 | ON\`\gnw_swap` md5 | OFF has key | ON has key | ON\`\gnw_swap` == OFF |
|---|---|---|---|---|---|---|---|
| what does the dog chase? | recall | `6ce36728` | `bf32b0ad` | `6ce36728` | no | yes | ✓ IDENTICAL |
| what does the brain use? | recall (post-change) | `c0422668` | `b455abff` | `c0422668` | no | yes | ✓ IDENTICAL |
| what does a unicorn fly? | abstain (moat) | `fd087b8b` | `a92bd6d7` | `fd087b8b` | no | yes | ✓ IDENTICAL |
| what does the cat eat all the way? | multi-step chase | `8cdaafa2` | `4b6a6706` | `8cdaafa2` | no | yes | ✓ IDENTICAL |
| who are you? | self / identity | `793f1351` | `890b252b` | `793f1351` | no | yes | ✓ IDENTICAL |
| tell me more | follow-up (no topic) | `c8c44b9e` | `0e9d9d85` | `c8c44b9e` | no | yes | ✓ IDENTICAL |

- **OFF response never carries a `gnw_swap` key** (the block is fully skipped) → byte-identical to pre-wiring.
- **ON response always carries `gnw_swap`, and ON-minus-`gnw_swap` == OFF (md5) on every panel turn** → enabling the swap changes NOTHING but adds the additive read; recall/abstain/multi-step/self are all answer-unchanged.

## Why the swap does not perturb the rest of the pipeline (the RNG-isolation fix)
The swap substrate's build reseeds `cfg.seed` and its stepping draws OU noise off the SAME process-global RNG the rest of the pipeline shares. Left unguarded, enabling the swap advanced the global RNG and perturbed the downstream RNG-dependent organs (the curiosity follow-up sampler, self-initiation) — the two RNG-dependent panel turns diverged. `ThoughtSwapWorkspace._isolated` runs the swap build + sim on the workspace's OWN private RNG timeline and restores the host process-global RNG (numpy + the sim backend) afterward, so the swap is RNG-neutral to the host pipeline. (This is the documented cupy/numpy-flip / global-RNG footgun, handled the standard way — save/restore around the isolated sim.)

## Anti-cheats
- **Additive + reversible:** `BRAIN_GNW_SWAP` truthy → enabled; unset/0/false/off/no → the observe block is fully skipped (no workspace built, no `gnw_swap` key). `_GNW_SWAP_DEFAULT_ON=False` is the production anchor (DEFAULT-OFF pending owner review). Byte-identical off (part B).
- **Answer-preserving on:** the swap workspace is a held-topic TRACKER; it never touches `answer`/`abstained`/`recalled_svo`/`verified` — only the additive `gnw_swap` read (part B, ON-minus-key == OFF).
- **The decision is neural:** the swap-vs-hold verdict is the spiking mismatch/salience detector's; the de-risk's trigger-lesion (silence mm → a salient input FAILS to swap) holds 6/6 seeds. The host supplies only the held/incoming topic identities (world input) + the `held_slot` bookkeeping label.
- **Grounded/moat-consistent topic:** only a concept the brain KNOWS can become a held topic; an unknown word or an action verb yields no topic → hold.
- **Never crashes a turn:** `observe_turn` is guarded (any error → an inert info dict); a wiring failure degrades to the unchanged turn.

## Honest limits / remaining scaffolds (named, not claimed closed)
1. **DEFAULT-OFF** — the flag is a reversible opt-in pending owner review; the swap workspace is NOT in the default turn. Byte-identical-off + answer-unchanged-on are proven, so a default-on flip is available on owner review.
2. **The mm→boost COUPLING is host arithmetic** (`eff_boost = gain * mm_rate`) — a neuromodulator-like linear read-out of the salience population's firing to the loop's release-probability U. The DECISION (whether/when there is any boost) is fully the mm spikes; there is no engine primitive for "presynaptic firing raises U of other synapses", so the read-out itself is a scaffold to burn down (unchanged from the de-risk). Functional correlate only; NEVER an assertion of phenomenal experience.
3. **Cross-turn CONTINUITY is a host label.** The "which coalition is held" identity persists as a host variable (`held_slot`) and is RE-ESTABLISHED on the substrate each turn via `run_intention_swap(isolate=True)` (restore the clean snapshot → re-ignite the held topic → present the proposal → neural decision). The swap-vs-hold VERDICT is neural every turn; the between-HTTP-turn persistence of the ignition is host bookkeeping. A truly continuous cross-turn ignition (no restore) is the named next rung.
4. **Per-pattern labeled-line topic routing** (N_PATTERNS=3 held-topic slots; beyond that, least-recently-held reuse) — the topic identity is the world's input, not a learned/composed code (inherited from the de-risk).
5. **Topic extraction is host comprehension** of the user message (the declared world/teacher boundary, the same one the SVO parser and the chase-marker detector occupy) — the COGNITION (swap-vs-hold) is the substrate's.

## Files
`webapp/gnw_thought_swap.py`, `webapp/server.py` (observe block + `_GNW_SWAP_DEFAULT_ON` + `gnw_swap` attach), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `gnw-thought-swap`). Artifact: `research/findings/raw/_gnw_swap_chat/verify.json`. Reuse-by-import from `research/runners/_gnw_neural_swap_intention_derisk` (`build`, `run_intention_swap`, `MultiLoopSTD`, `SALIENT_PA`, `N_PATTERNS`); NO `sim/` edit.
