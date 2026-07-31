---
type: finding
status: contributing
date: 2026-07-20
---

# Grounded fluent conversation — DE-RISK 5 (one-brain-one-process) GO: the WHOLE turn runs in ONE cupy process, fluency on spikes

**Date:** 2026-07-20 · **Status:** DE-RISK 5 GO — the whole grounded-fluent turn (composer comprehension + retrieval +
gate-first moat + the fully-spiking on-bridge WKV render) CO-EXECUTES in ONE cupy process = the true one-brain-one-
process (the EMERGE-70/71 pattern applied to grounded fluent conversation). NO `sim/` edit.

## The rung

De-risk 3 proved the WKV renders the grounded answer on-bridge (RF-phase / fully-synaptic parity) in ISOLATION; the
console (De-risk 1) used the off-bridge numpy WKV forward (bit-identical, CPU-portable). De-risk 5 closes the honest
"firming follow-on" flagged there: run BOTH the composer (retrieval + gate) AND the on-bridge WKV render on the cupy
substrate, in ONE process — the owner's explicit "everything in one brain, one process."

## GO — the whole turn in one cupy process

`OnBridgeWKVFaculty` (`_wkv_onbridge_faculty.py`) is the fully-spiking renderer: the WKV forward runs ON a
`SimulationBridge` (each token's value delivered through an RF spike's PHASE → the graded `cp_ssm_state` → the SSM's
own trained read-out). Reuse-by-import of the on-bridge builders (`_build_ssm_state_bridge`, `_build_rf_encoder`,
`_build_synaptic_rf_encoder`); the charge/generate loop replicates the De-risk-3-validated generation block. It matches
`FTFaculty.answer(facts_ctx, question)` so it drops into `FluidChat(renderer="wkv_onbridge")`.

**Verify-first (extraction correctness):** the faculty reproduces the De-risk 3 on-bridge outputs EXACTLY (3/3 on
cupy — "the dog eats meat", "the fox chases rabbit", "the bee makes honey").

**One-process co-execution (SIM_BACKEND=cupy):** the console builds its composer + parser + dlPFC on cupy bridges AND
the `OnBridgeWKVFaculty` on a cupy bridge — all in ONE process (`dev=cupy(on-bridge)`, GPU 2.2 GB / 8.8%):
- grounded Q&A rendered ON SPIKES: "what does the dog eat?" → "The dog eats meat"; "what does the fox chase?" → "The
  fox chases rabbit"; "what does the bee make?" → "The bee makes honey".
- growth live: teach "the wolf eats rabbit" → "what does the wolf eat?" → "The wolf eats rabbit" (rendered on-bridge).
- **GATE-FIRST MOAT VERIFIED (not asserted):** 2 grounded / 2 untaught (lion/zzz). On every abstain the on-bridge WKV
  is invoked **0 times** (`n_invocations==0`; the `_answer` `p is None` short-circuit fires before the render) →
  "I don't know." The moat holds; no spiking render on an untaught query.

## Read-out — the north-star as a true one-brain-one-process artifact

- **⇒ the whole grounded-fluent conversation turn — comprehend + retrieve + gate-first moat + render fluent grounded
  prose ON SPIKES — runs in ONE cupy process, one backend.** The EMERGE-70/71 "one brain, one process" bar is met for
  grounded fluent conversation. The `renderer` flag is additive (default `"ft"` byte-identical; `"wkv"` = off-bridge
  numpy CPU-portable; `"wkv_onbridge"` = fully-spiking on cupy).
- **Honest scope:** the composer and the WKV are on SEPARATE cupy bridges in ONE process (one process, one backend,
  fully-spiking render) — not yet SLICES of a SINGLE bridge. The full one-bridge consolidation (composer + WKV as
  disjoint slices of one `SimulationBridge`, the nav+conv-merge pattern) is the deeper follow-on; "one process, one
  backend, spiking render, gate-first moat" is the genuine milestone here.

**⇒ THE NORTH-STAR ARC IS COMPLETE + demonstrated as one-brain-one-process:** De-risk 0 (ceiling) · 2 (format
fine-tune, 6-seed GO) · 1 (wiring + moat) · 3 (fully-spiking on-bridge parity) · 4 (multi-fact wall + per-fact
method) · 5 (one-brain-one-process co-execution). "A brain you COMMUNICATE with," fluency on spikes, ANN scaffold
retired, NO `sim/` edit anywhere in the arc.

Runner: `_wkv_onbridge_faculty.py` (`OnBridgeWKVFaculty`). Console: `_fluidconv_chat_repl.py` (`--renderer wkv_onbridge`,
requires `SIM_BACKEND=cupy`).
