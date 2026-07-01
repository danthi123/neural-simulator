# Fluid conversation — Phase 17 GO: PERSISTENT grown knowledge (the brain remembers across sessions)

**2026-07-01 (autonomous; the natural next frontier after learn-on-demand — the owner's "grow THROUGH experiences").**
The console learned real facts on demand (Wikidata, Phase-15) + by being taught (Phase-5 growth), but it rebuilt from
the curriculum each run, so the grown knowledge was LOST on restart. This closes that: the brain now **remembers** what
it learned across sessions. Reuse-by-import; **NO `sim/` edit**; CPU.

## The mechanism (cheap — composes built pieces)
A learned concept's composer code is generated **deterministically** (`_ensure_concept`: md5(word) → seed → phases),
so it **reproduces** across sessions. Persistence therefore does NOT need to store the composer's complex-weight tensor
— it saves the learned **fact-list** (JSON) and, on load, re-injects each concept's (identical) code + re-stores the
fact → the KB is rebuilt bit-for-bit. `FluidChat.save_state(path)` / `load_state(path)` + a `--persist <path>` flag
(load on start, save on exit). Instances (dog_1) are session discourse state → intentionally not persisted.

## Result — GO
- **De-risk `_fluidconv_phase17_persistence_derisk.py` (3 seeds, bare agent, fast):** ROUND-TRIP 4/4 learned facts
  recalled after save→load into a fresh same-seed brain; COLD-START control 0/4 (a fresh brain that does NOT load
  recalls none — persistence is load-bearing); base survives; MOAT abstains on the unlearned; **DETERMINISTIC-CODE:
  the re-injected codes are BIT-IDENTICAL to the originals** (the round-trip works because codes are reproducible md5,
  not because they're stored).
- **Console, live across TWO processes:** session 1 *"the wolf eats rabbit" + "learn about elephant"* → saves 5 grown
  facts; session 2 (fresh process) → *"remembered 5 fact(s) from a prior session"* → *"what does the wolf eat?" → "the
  wolf eats rabbit"* AND *"tell me about the elephant" → "An elephant is a mammal; it is grey and has trunk and tusk."*
  — both the taught fact AND the Wikidata knowledge recalled from the prior session.
- **CI-guarded** (`tests/test_fluidconv_chat_repl.py`, 4 tests): the save/load plumbing + idempotency (cache-backed
  `learn about elephant`, offline). Both offline demos + discourse-plan + persistence all pass.

## Honest ceiling
- A **JSON re-instate** of the learned fact-list (like the Tier-3 live-and-remember persistence), NOT the raw composer
  complex-weight tensor — sound because the codes are deterministic + the seed is fixed (a lineage fixes it). Base-
  curriculum facts re-load from the curriculum; the grown delta re-loads from the state file.
- Instances (discourse referents) are session state, not persisted (a persisted "the dog I saw" would need the
  engram/hippocampal episodic path — Tier-3 territory).
- Composes with the project's `BridgeLineage` (persistent continuous-learning) pattern; this is the console-level
  equivalent for the grown conversational KB.

## Where this sits — the fluid-conversation console is now a GROWING artifact
learn a real concept on demand (Wikidata) → discuss it as connected grounded prose (Phase-16) → compare/gist across
concepts → **remember it across sessions** (Phase-17). Plus: instance-rep ("which dog?"), multi-turn anaphora, growth,
and the no-confab moat throughout. The owner's "talk to it like an LLM, grounded in its own knowledge/experience,
growing through it" — realized for the fact-grounded core, MINIMIZING the transformer (fluency-only, brain-gated). The
genuine walls stay honest: free single-pass abstractive synthesis + open-world cross-fact inference on a ~21M (routed
around via the grounded discourse plan, not solved).

**Artifacts:** `research/runners/_fluidconv_phase17_persistence_derisk.py`; `_fluidconv_chat_repl.py` save_state/
load_state + `--persist`; result `research/findings/raw/_fluidconv_phase17_persistence.json`.
