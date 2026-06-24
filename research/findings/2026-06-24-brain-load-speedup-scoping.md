# Brain-LOAD latency speedup — scoping (read-only research gate)

**Date:** 2026-06-24
**Type:** SCOPING / research-gate (read-only — no code changed, no commit, no heavy GPU run)
**Owner priority:** `feedback_prioritize_orchestration_overhead` (per-op latency is the real-time wall, not VRAM)
**Companion JSON:** `research/findings/raw/_scoping_brain_load_speedup.json`

## Problem

The interact-console `POST /api/brain-chat` is rough on the **FIRST** turn per
`(session, brain, renderer)`: **~112 s for the tiny-demo (5 facts)** and **~9 min
for the self-knowledge brain (52 facts)**. Warm turns are 0.2–2.8 s. The cost
(`_console_live_debug_fixes.json` `latency_note`) is the **brain-LOAD**: the
Hebbian parser training on the bridge + the per-fact RF resonate windows. Goal:
a brain **LOADS** its saved trained state instead of **RE-TRAINS / RE-RESONATES**.

## Load-cost breakdown (the diagnosis the gate asked for)

Two loops dominate; **both are pure recomputation of state that could be
persisted, and neither is persisted today.**

| Component | Where | Cost | Scales with | Notes |
|---|---|---|---|---|
| **Parser Hebbian training** (FIXED) | `brain_conversational_agent.py:268` `BridgeParser` + `:299` `AttributedBridgeParser` (built on the rf/rate default path the console uses; `enable_attributed` defaults True) | **~75K `_run_one_simulation_step()` calls** (BridgeParser 30 epochs × 6 conj × ~140 ≈ 25.2K; AttributedBridgeParser 24 epochs × ~15 conj × ~140 ≈ 50.4K) | **fact-INDEPENDENT** | **Dominates tiny-demo (112 s).** Each builds a private `SimulationBridge` and runs a long single-threaded `_train()` in `__init__`. |
| **Per-fact RF resonate re-store** | `rf_phasor_composer.py:432` `store` → `:246` `_encode` → 3×`_bind` + 1×`_bundle`, each a `_resonate(period+8=208)` | **~832 RF steps / fact** (5 facts ≈ 4.2K; 52 facts ≈ **43K**) | **per-fact** | **Dominates the 52-fact SK brain** (why SK is "longer"). |
| Bundle/codes deserialize + skeleton | `developed_brain_io` npz/json + agent skeleton + a few `_build_rf_bridge` | < 1 s | — | Negligible. |

**The load-bearing insight:** `load_developed_brain` does **NOT** re-train any
composer — it **re-RESONATES** each fact via `comp.store` (`_restore_facts`,
`developed_brain_io.py:209`). The **parser** is re-**TRAINED** in the agent
constructor. And critically:

- **The parser is exercised at chat time ONLY when the user TEACHES a new fact**
  (`hear()` → `parser.parse`, `brain_conversational_agent.py:428`). A **LOADED**
  brain restores its facts via `comp.store()` directly, **bypassing the parser**.
  So a read-only Q&A session pays the full ~75K-step parser training **and never
  uses the parser.**
- **The per-fact RF resonate output IS just a `[D]` numpy array** — `store()`
  caches it at `composer.kb[i] = (fact, comp)` (`rf_phasor_composer.py:448`,
  `enable_substrate_store=False` default). So persisting `kb` composites trivially
  skips the re-resonate; the composite is deterministic per seed/codes.

### What `save_checkpoint` / `load_checkpoint` already give us

`sim/bridge.py:7592/7767` persist `cp_connections` (the CSR — **which holds the
parser's trained Hebbian weights**), V/conductances/firing/traits/STP/eligibility-
adjacent arrays. It does **NOT** persist firing thresholds / STP recovery /
eligibility (the documented gotcha → a loaded parser self-recovers in ~10 ms of
free-running, fine for inference), **nor** the RF complex weights (`cp_rf_w_re/im`).

→ `save_checkpoint` is the **right** tool to persist the **parser** (its weights
are in `cp_connections`). It is the **wrong** tool for the composer store (the
`kb` composites are numpy, not bridge arrays; per-op RF bridges are ephemeral) —
persist those as the composite arrays themselves.

## Ranked options (cheapest-first)

**① Persist the composer fact-store composites (`kb`) in the bundle → skip per-fact RF resonate.**
Add a `kb_composites.npz` to the `developed_brain_io` bundle (`{i → comp[D]}` from
`composer.kb`); on load set `composer.kb` directly instead of `_restore_facts`
calling `comp.store()`. Byte-identical (the composite *is* the resonate output).
*Reuse* the npz machinery + `extract_facts`; *build* ~30–40 lines, runner-only,
**no `sim/` edit.** **Removes the per-fact term** (~43K RF steps for SK).

**② Lazy parser for a LOADED brain → skip the ~75K-step parser training entirely.** *(recommended single step)*
The parser is only needed on a runtime *teach*; a loaded brain restores facts
without it. Defer `BridgeParser` + `AttributedBridgeParser` construction+training
until the first runtime `hear()`. Add a default-OFF `defer_parser` on
`BrainConversationalAgent`; `load_developed_brain` passes it.
*Reuse* the **already-present** `BridgeParser(defer_train=...)` plumbing; *build* a
lazy wrapper + a load kwarg (~40–60 lines), **no `sim/` edit.** **Removes the
entire fixed ~75K-step cost** for any session that doesn't teach (≈ the dominant
term of tiny-demo's 112 s). Owner usage ("talk to a developed brain") is
overwhelmingly Q&A → near-100% hit-rate.
- **②b persist parser weights** (so even a first-teach loads, not trains): give the
  parser classes `save_state/load_state` via `bridge.save_checkpoint`/`load_checkpoint`
  (trained weights live in `cp_connections`) + a short recovery free-run for the
  unsaved thresholds. Turns first-teach from ~75K steps into an HDF5 load (< 1 s).

**③ Warm the brains at webapp STARTUP (cheap stopgap; hides, doesn't remove).**
Add an `@app.on_event('startup')` (the server **already** has two at
`server.py:1692/1701`) that background-builds + caches the default brains into
`_BRAIN_CHATS` so the owner's first real turn is warm. *Reuse* `_build_chat_brain`
+ `_BRAIN_CHATS`; *build* ~20 lines (threaded), **no `sim/` edit.** Does not reduce
the build cost — complementary to ①/②.

**④ Pickle the whole built ChatBrain — REJECTED.** Needs custom `__reduce__` for a
cupy/CUDA-graph/h5-bound `SimulationBridge`; fragile, GPU-context-bound,
version-brittle; likely slower to deserialize than ①+②. The structured
persistence (① codes/composites + ②b parser checkpoint) is the project's existing
pattern (bundle + HDF5) and is strictly safer.

## Recommendation

- **Single best cheapest-first step: ② (lazy parser for a loaded brain).** Smallest
  change, reuses existing `defer_train` plumbing, no `sim/` edit, byte-identical for
  any never-teaching session, and alone collapses the tiny-demo ~112 s to seconds.
- **Full fix: ② + ①.** Together they kill **both** dominant loops — ② the fixed
  parser cost, ① the per-fact cost — so a 52-fact brain **loads in
  bundle-deserialize time (low seconds)** instead of ~9 min. Add **②b** so even a
  teaching session loads-not-trains, and optionally **③** as a zero-risk stopgap to
  pre-pay the small residual before the owner's first turn.

**`sim/` flags required: NONE.** Every recommended option is reuse-by-import /
runner + webapp only.

### Anti-cheat controls the build needs

- **Round-trip equality:** a loaded brain (lazy parser + persisted composites)
  must answer the full who/what + yes-no + abstain matrix **byte-identically** to a
  freshly-built+retaught brain (assert per-fact `comp` arrays match a fresh
  `_encode` to atol 1e-9 — the persisted composite is the deterministic resonate).
- **Moat preserved:** the no-confab abstention holds (0 false-accepts) on the loaded
  brain — only saved facts are in `kb`; untaught cues abstain.
- **First-teach correctness (②):** after a lazy-loaded brain teaches a NEW fact, the
  parser builds + works and the fact recalls (with ②b, a loaded-parser teach equals
  a trained-parser teach).
- **Existing CI verbatim:** `tests/test_brain_conversational_agent.py`,
  `tests/test_one_brain_composer_agent.py`, and the `developed_brain_io` round-trip
  pass with the new defaults **OFF** on the standalone build path (byte-identical)
  and **ON** on the load path.

## Key code anchors

- Load path: `webapp/server.py:2788` `_build_chat_brain`; `:3084` `brain_chat`;
  `_BRAIN_CHATS` `:2718`; startup hooks `:1692/:1701`.
- Parsers: `brain_conversational_agent.py:268` (BridgeParser) + `:299`
  (AttributedBridgeParser); `hear()` `:407` (parser used only on a runtime teach);
  `BridgeParser` `defer_train` already supported `:96`/`:103`.
- RF per-fact: `rf_phasor_composer.py:432` `store` → `:246` `_encode` → `:219`
  `_bind` / `:230` `_bundle` → `:156` `_resonate(period+8=208)`; kb composite `:448`.
- Bundle I/O: `developed_brain_io.py:105` save / `:234` load / `:209` `_restore_facts`
  (the per-fact re-resonate on load).
- Bridge persist: `sim/bridge.py:7592` `save_checkpoint` / `:7767` `load_checkpoint`.
- Bundle saver: `_longitudinal_develop_loop_gpu.py:501` `save_developed_bundle`
  (persists neither parser nor RF state → a load re-trains + re-resonates).
