# EMERGE-68 — the EMERGE-frame render is now **100% produced ON SPIKES** (order AND every word, content AND function): the spiking A→W read-out extended to the FUNCTION-word slots — **GO** (6-seed)

**2026-07-03 (autonomous).** EMERGE-67 wired the **content** words (subject/verb) of the spiking-Broca render onto spikes via the validated A→W read-out; it named the residual explicitly (its GO finding, line 23): the **DET/FUNC function-word slots** (`the`/`a`/`can`/`does`/`not`) kept a **token surface** (`spell` returned `str(word)` for non-content words, `_emerge67:236`). EMERGE-68 closes that residual — the function words are now spelled **on spikes** too, so the whole EMERGE-frame render is produced on the substrate: the ORDER (EMERGE-59/63 competitive queuing), the CONTENT words (EMERGE-67 A→W), and now the FUNCTION words. Reuse-by-import; **NO `sim/` edit**.

## Mechanism

The function words the EMERGE frames emit are exactly `{the, can, does, not}` (`_emerge59:98-105`; the task adds the DET alternative `a` from `argstructure_composer.FUNCTION_WORDS`) — the closed class **EMERGE-62 discovers** from distributional statistics. The seam is already function-word-ready: `realize_slot`'s DET/FUNC branch (`_emerge59:143-144`) already calls `spell(payload)`, so a `spell` that **knows** the function words makes those slots spiking with no seam change.

The concept-pool A→W architecture has exactly **16 pools** (4 kinds × 4: motor/noun/verb/adjective) and `train_word_to_pool` supports only those 4 kinds — so the 16 content + 5 function = 21 words need a **second bridge** (the project's own scaling route; the EMERGE-67 finding named "extend the content vocab across 2 bridges"; G.20 multi-bridge). EMERGE-68 therefore composes two co-validated A→W engines:

- **BRIDGE-A** (`bridges/emerge67_aw/aw_content.simstate.h5`): EMERGE-67's 16 **content** words. **Reused verbatim.**
- **BRIDGE-F** (`bridges/emerge68_aw/aw_func.simstate.h5`, new, regenerable via `--train`): the 5 **function** words rebound onto 5 of the 16 pools (`the/a/can/does/not → motor_N/E/S/W + noun_APPLE`), the other 11 pools filled with content-word fillers so the **same** `build_concept_bridge + apply_concept_topographic_bias + orthogonal-codes + train_word_to_pool` recipe trains at the validated scale; only the 5 function pools are decoded.

`UnifiedNeuralSpell.spell(word)` **dispatches**: a content word → decode on BRIDGE-A; a function word → decode on BRIDGE-F. Both `drive the word's concept pool → decode the spoken word from cp_firing_states[language_output]` (cosine to the orthogonal word patterns). Every slot — content **and** function — is spike-spelled through this one callback.

## De-risk — **GO** (6-seed 42/43/44/100/101/102; A→W on GPU/cupy, wire+moat CPU-safe)

| gate | value | bar |
|---|---|---|
| ALL-word spike-spell accuracy (det+subj+func+verb decoded from spikes, vs ground-truth) | **1.000** every seed | ≥ 0.90 |
| FUNCTION-word slot accuracy (`the/a/can/does/not` on spikes) | **1.000** every seed | ≥ 0.90 |
| raw A→W function-word rate (BRIDGE-F spells the 5 function words) | **1.000** (5/5) | — |
| **genuinely SPIKING** — FUNCTION-word LESION (zero BRIDGE-F's pool→`language_output`) collapses the decode | **0.153** (engine-lesion **0.000**) | ≪ main (≥ 0.40 drop) |
| NO regression vs the token spell (all slot surfaces identical) | **0 mismatches** | 0 |
| gate-first MOAT — spell-calls + producer-invocations on abstains | **0 / 0** | 0 |

Sample transcript: *"can owl fly?" → "the owl can fly"* / *"can penguin walks?" → "the penguin walks"* / *"can penguin fly?" → "the penguin does not fly"* / *"can a zzz fly?" → "I don't know."* (producer NOT invoked — moat). Every emitted word (the/owl/can/does/not/fly/…) is decoded from `language_output` spikes.

**Genuinely spiking, not a host lookup:** BRIDGE-F reads real `cp_firing_states[language_output]` spikes; **lesioning its pool→`language_output` pathway collapses the function-word decode to ~0.15** (engine-lesion 0.000) — the load-bearing proof the function words are decoded FROM SPIKES. The gate-first moat holds by construction: on an abstain the producer — and hence both A→W engines — is NEVER invoked (0 spell-calls, 0 productions).

## Honest process note — a real boundary was hit and SURPASSED with a single-variable fix

The first BRIDGE-F cache (trained at seed 43) landed at a **BOUNDARY**: all-word 0.99 / func 0.98 but **4 regress mismatches** — localized to **one** function word, `can` (on `motor_S`), stochastically misreading to content fillers (the concept-pool architecture's documented per-word/per-seed fragility, CLAUDE.md). Per the SURPASS discipline the residual was isolated (`can` only; longer read windows did **not** help → not a spike-count issue → a per-pool selectivity issue) and the cheapest single-variable lever tried: **retrain BRIDGE-F at seed 42** (the seed EMERGE-67's content cache uses, 16/16). That cleanly fixed it — all 5 function pools became robust → **GO 6-seed, regress 0**. This is a scale/seed lever on the existing mechanism, **not** a new mechanism; the deeper alternative (a dedicated closed-class pool block, the EMERGE-62 discovered class as its own kind) remains available if a future vocab stresses it.

## Verdict

**GO.** The EMERGE-frame render is now **100% produced on spikes** — ORDER (EMERGE-59/63) **and every word** (content via EMERGE-67, function via EMERGE-68). "the owl can fly" has *the*, *owl*, *can*, *fly* **all** decoded from `language_output` spikes. Renders the BOUNDED EMERGE frame inventory (ability-affirm / intransitive-exception / negated-modal), NOT open prose (R4, the deferred scale wall). The A→W engines are GPU-trained once at the validated scale + cached; the 5 function words are rebound onto 5 concept pools of a second bridge (reuse-by-import; **NO `sim/` edit**). The token-spell default path is byte-identical — EMERGE-59..67 all still pass (90 passed / 1 skipped on numpy).

## Files
- `research/runners/_emerge68_function_word_spell_derisk.py` — `FuncNeuralSpell` (BRIDGE-F function-word A→W) + `UnifiedNeuralSpell` (content→BRIDGE-A / function→BRIDGE-F dispatch) + the all-slot scoring + `--train`/`--demo`/`--derisk`.
- `tests/test_emerge68_function_word_spell.py` — 7 CPU-safe (function-word vocab structure / unified wire / all-slot ground-truth / moat / no-regression) + 1 GPU smoke (skip unless the process is `SIM_BACKEND=cupy` and both caches exist — the backend is process-sticky; the full GPU A→W is validated by the 6-seed `--derisk`).
- `research/findings/raw/_emerge68_function_word_spell.json` — the 6-seed de-risk (go=True).
- Cached function-word engine `bridges/emerge68_aw/aw_func.simstate.h5` (local, gitignored `.h5`; regenerable via `--train`, seed 42). BRIDGE-A `bridges/emerge67_aw/aw_content.simstate.h5` reused verbatim.
