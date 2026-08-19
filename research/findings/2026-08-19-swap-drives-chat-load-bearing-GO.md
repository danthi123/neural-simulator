---
status: live
type: finding
lane: laneC
date: 2026-08-19
integration_faculty: swap-drives-response
---

# SWAP DRIVES THE LIVE RESPONSE — the #77 neural thought-swap verdict made LOAD-BEARING on `/api/brain-chat` (board #85). A topic-change user turn (the spiking mismatch/salience detector fires → the incumbent coalition self-evicts → the vacancy gate ignites the newcomer) makes the reply LEAD with a topic-transition acknowledgment naming the newly-held coalition ("On <newtopic>, then — <answer>"); a same-topic follow-up holds silent. Verified through the REAL handler: (A) the swap tracks the conversation (topic-change swap-rate 1.00, same-topic/no-topic 0.00, transition-lead iff swap); (B) message FIXED, the held-topic CONTEXT changes the reply lead (swap "On dog, then —" vs hold "") with the base sentence + content byte-identical, and the difference VANISHES under the neural mismatch-detector lesion (silence mm → no swap → lead gone → == base); (C) content swap-invariant, byte-identical with the flag OFF. DEFAULT-ON. GO.

**Date:** 2026-08-19 · **Board:** #85 (INTEGRATION-TO-PRODUCTION). **Backend:** CPU (numpy). **Verdict:** **GO** through the real `/api/brain-chat` handler (in-process). **No `sim/` edit** (`git diff sim/` empty).

**Files:** `webapp/swap_drives_chat.py` (NEW — reuses #77's `gnw_thought_swap.observe_turn` for the neural swap verdict + held-topic coalition, maps the verdict → a topic-transition lead), `webapp/gnw_thought_swap.py` (additive `lesion=` kwarg threaded into `run_intention_swap(trigger_lesion=…)`; default False → #77 observer byte-identical), `webapp/server.py` (the `_SWAP_DRIVES_DEFAULT_ON=True` anchor + `_swap_drives_on()` + the single-neural-swap block that supersedes the #77 observer + the `swap_drives_lead` prepend / `swap_drives` attach on both main return paths), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `swap-drives-response`). **Artifact:** `research/findings/raw/_swap_drives_chat/verify.json`.

**Reproduce:** `SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u research/findings/raw/_swap_drives_chat/verify_swap_drives.py` — drives `webapp.server.brain_chat(BrainChatRequest(...))` directly (the REAL handler). The heavy default organs are disabled for a tractable in-process verify (a consistent baseline across ALL arms; swap-drives reads its own #77 workspace regardless, so the isolation cannot change any swap-drives verdict). First turn ~90s (full tiny-demo warmup); each reset-session arm rebuilds the ChatBrain (~40–90s).

## What this closes
Board #77 wired the GNW thought-swap onto the live chat as an OBSERVER: each turn it ran the reused 6/6-seed-GO neural swap machinery and stashed a swap-vs-hold VERDICT + the held-topic coalition as response METADATA (`gnw_swap` key), but it NEVER changed the answer text (`2026-08-19-gnw-swap-into-chat-GO`: topic-change swap-rate 1.00, answer byte-identical on vs off). An observe-only wiring is hollow — the neural faculty computes a verdict nobody consumes. This finding makes the swap verdict CHANGE the surface: the held-topic coalition is now load-bearing on what the brain SAYS, and the change VANISHES when the neural swap is lesioned. It is the anti-hollow-integration counterpart to the observer, mirroring the board-#84 affect-DRIVES path.

## The coupling (neural verdict → surface, host boundary vs neural decision)
- **READ (the #77 neural mechanism, reused-by-import):** each turn the message's grounded TOPIC (the FIRST known agent/patient token — the SAME host-comprehension boundary the SVO parser occupies) is presented to the per-session held-topic swap workspace. `run_intention_swap` drives it into the spiking mismatch/salience detector + the vacancy gate: a DIFFERENT salient topic is a mismatch → mm fires → the recurrence-depression boost evicts the incumbent → the vacancy gate ignites the newcomer (a SWAP); the SAME topic MATCHES → the pred interneuron vetoes mm → no swap. The `swapped` verdict and the winning coalition's identity (`held_topic` after the decision) are the substrate's, NOT a host `if new != old`.
- **DRIVE (the load-bearing surface change):** `swapped == True` → a transition lead `"On <new_topic>, then — "` prepended to the answer OUTERMOST, where `<new_topic>` is READ from the neural winner coalition; `swapped == False` (a hold / no-topic follow-up / first thought / an unmatched no-swap) → NO lead (a continuation needs no announcement — the natural discourse move is to mark SHIFTS, not continuations). The lead is an honest EXPRESSION of the thought-swap the substrate just performed (a discourse-structuring "mouth"), NOT content: the FACT after it is the SAME gate-matched, moat-verified answer.

## Result — the DELIVERABLE, through the REAL handler (artifact `research/findings/raw/_swap_drives_chat/verify.json`, verdict GO)

### (A) The swap tracks the conversation — topic-change SWAPS (+ a transition lead), same-topic/no-topic HOLDS (no lead)
One `/api/brain-chat` session, `BRAIN_SWAP_DRIVES` default-on. Each turn's `swap_drives` read:

| # | user turn | kind | swapped | held_topic | transition lead |
|---|---|---|---|---|---|
| 0 | what does the dog chase? | establish | False | dog | `''` |
| 1 | what does the dog chase? | same-topic | False | dog | `''` |
| 2 | what does the brain use? | topic-change | **True** | brain | `On brain, then — ` |
| 3 | what does the brain store? | same-topic | False | brain | `''` |
| 4 | what does the cat eat? | topic-change | **True** | cat | `On cat, then — ` |
| 5 | tell me more | no-topic | False | cat | `''` |
| 6 | what does the dog chase? | topic-change | **True** | dog | `On dog, then — ` |

**Swap rate: topic-change = 1.00 (3/3); same-topic + no-topic = 0.00 (0/3); the transition lead is present IFF the neural swap fired (lead-iff-swap = True).** The lead names the newly-held coalition and is prepended to the answer only on a swap.

### (B) The swap DRIVES the response — message FIXED, the held-topic context changes the reply lead; the neural lesion collapses it
Same fixed probe `"what does the dog chase?"` (topic `dog`), two held-topic contexts established by a prior turn (CLEAN separate sessions — the #84 session-leak lesson):

- **swap-context** (held `cat` first → `dog` is a MISMATCH → SWAP): `swapped=True`, lead `On dog, then — `, answer `"On dog, then — The dog chases cat."`
- **hold-context** (held `dog` first → `dog` MATCHES → HOLD): `swapped=False`, lead `''`, answer `"The dog chases cat."`

`intact_diff = True` (the answers + leads differ); `base_identical = True` (the fact under the lead == the hold answer, `"The dog chases cat."`); `content_identical = True` (the abstain/recall/verify md5 is identical across the two arms). So the SAME message yields a DIFFERENT reply solely because the neural swap verdict differs — the held-topic coalition is load-bearing on the surface.

**LESION (`BRAIN_SWAP_DRIVES_LESION=1`, the de-risk's own `trigger_lesion` — silence the mismatch detector):** the SAME swap-context `dog` can NO LONGER swap: `swapped=False`, `reason=mismatch_held_no_swap`, the mismatch-detector peak firing collapses to near zero (`lesion_mm_peak` in the cited artifact), lead `''`, answer `"The dog chases cat."` == the base. So the transition lead VANISHES and the answer reverts byte-identically to the no-lead base — the surface change RIDES the SPIKING mismatch read, not a host `if topic_changed`: kill the neural detector and the topic-transition acknowledgment disappears even though the world input (a topic change) is unchanged.

### (C) No-regression — content swap-invariant + byte-identical-off
- **C1 (no-swap byte-identity):** a no-swap panel (establish · hold · unicorn-abstain) run OFF then default-ON under identical per-turn RNG. OFF response never carries a `swap_drives` key; ON always does; **ON-minus-`swap_drives` == OFF (md5) on every turn** (`194897f8`/`194897f8`/`7f15f541`), with an empty lead. Enabling the swap changes NOTHING on a no-swap turn but adds the additive read → byte-identical to pre-wiring.
- **C2 (content-invariance under an ACTIVE swap):** the same fixed probe under {off, swap-context, hold-context} — the CONTENT fields (`abstained`/`recalled_svo`/`verified`) md5 is identical (`350812dd`) across all three; only the transition lead differs. The swap decorates the surface, never a fact.

## Why the swap does not perturb the rest of the pipeline (the RNG-isolation fix, inherited from #77)
The swap substrate's build reseeds `cfg.seed` and its stepping draws OU noise off the SAME process-global RNG the rest of the pipeline shares. `ThoughtSwapWorkspace._isolated` runs the swap build + sim on the workspace's OWN private RNG timeline and restores the host RNG afterward, so the swap is RNG-neutral to the host pipeline — the OTHER response fields stay byte-identical (the #77 footgun, handled the standard save/restore way). The C1 byte-identity confirms it end-to-end.

## Anti-cheats
- **Additive + reversible:** `_SWAP_DRIVES_DEFAULT_ON=True` is the production anchor; `BRAIN_SWAP_DRIVES=0` → the block is fully skipped (no workspace built, no `swap_drives` key, no lead) → byte-identical oracle (C1: OFF carries no key, no-swap ON-minus-key == OFF md5).
- **Load-bearing, not cosmetic:** the transition lead rides the neural `swapped` verdict — the SAME message swaps-or-holds by the held-topic context (B intact), and the neural mm-lesion collapses the swap → the lead vanishes → == base (B lesion). Content byte-identical throughout (B/C).
- **The decision is neural:** the swap-vs-hold verdict is the spiking mismatch/salience detector's; the de-risk's `trigger_lesion` (silence mm) makes a salient topic-change FAIL to swap, 6/6 seeds (`_gnw_neural_swap_intention_derisk`). The host supplies only the topic identity (world input) + the `held_slot` bookkeeping label + the transition STRING (articulation scaffold).
- **Never crashes a turn:** `observe_turn` is guarded (any error → an inert no-lead info dict); a wiring failure degrades to the unchanged turn.

## Honest limits / remaining scaffolds (named, not claimed closed)
1. **The verdict→TRANSITION-STRING map is a HOST conditioned-articulation scaffold** (the discourse "mouth"): the swap that DRIVES it is the neural mismatch/eviction/admit chain (load-bearing — the lesion collapses the lead), but the surface STRING for a swap is a host template, the owner-sanctioned articulation-crutch pattern (scaffold-ok-as-conditioned-articulation IF the faculty is load-bearing on the surface, which the lesion proves). A brain-native discourse-transition mouth (the marker emitted by a spiking sequencing circuit) is the named next rung.
2. **The coupling is SWAP-only** (a topic change announces the shift; a hold stays silent — the natural discourse move + the clean neural-lesion vanish). A bidirectional continuity lead ("Still on <heldtopic> — " on holds) that makes the persistent coalition visible on EVERY follow-up is a possible enrichment; it was kept swap-only so the mm-silencing lesion produces a byte-identical vanish (a hold-lead would survive the lesion and muddy the anti-hollow check).
3. **Topic extraction is host comprehension** of the user message (the declared world/teacher boundary, the SVO-parser boundary) — the DECISION (swap-vs-hold) + the winning coalition's identity are the substrate's.
4. **Inherited #77 residuals** (unchanged): the mm→boost coupling is a host rate read-out (no engine primitive for presynaptic-firing U-modulation), a neuromodulator-like functional correlate — NEVER a phenomenal claim; the cross-turn CONTINUITY of the held thought is a host label (`held_slot`) re-established each turn via `run_intention_swap(isolate=True)` (the swap-vs-hold VERDICT is neural every turn; a truly continuous cross-turn ignition is the named rung); per-pattern labeled-line topic routing (N_PATTERNS=3 slots, LRU reuse).
5. **CO-RESIDENT** on its own swap bridge, run alongside the recall composer — rides the one-brain merge (burn-down #1).

## Files
`webapp/swap_drives_chat.py`, `webapp/gnw_thought_swap.py` (additive `lesion=` kwarg), `webapp/server.py` (anchor + `_swap_drives_on()` + the single-swap block + the `swap_drives_lead` prepend / `swap_drives` attach on both return paths), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `swap-drives-response`). Artifact: `research/findings/raw/_swap_drives_chat/verify.json` (+ `verify_swap_drives.py`). Reuse-by-import from `webapp/gnw_thought_swap` (#77) → `research/runners/_gnw_neural_swap_intention_derisk` (6/6 GO); NO `sim/` edit. de-risk findings `2026-08-19-gnw-neural-swap-intention-GO` / `-gnw-neural-vacancy-gate-GO` / `-gnw-recurrence-weaken-swap-GO`; observer wiring `2026-08-19-gnw-swap-into-chat-GO`.
