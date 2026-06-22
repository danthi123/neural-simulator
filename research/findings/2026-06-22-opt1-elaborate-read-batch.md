# Opt #1 — batch `elaborate`'s per-concept spiking read (answer-identical) (2026-06-22)

**Scope:** the latency-arc RANK-1 cheapest-first optimization from the megakernel-revisit audit
(`research/findings/2026-06-22-megakernel-revisit-optimization-audit.md`, §6). Replace `elaborate`'s
per-concept device→host read (`for c in vocab: to_host(fs[idx]).sum()` inside each settle step) with ONE
on-device segment reduction + ONE host sync per step. **Runner-side only — NO `sim/` edit.** GPU (`SIM_BACKEND=cupy`).
**Answer-identity is the hard gate; speed is secondary.**

## The hot path
`research/runners/content_selection_spiking.py`. The `elaborate` dialogue-planning call (both
`BrainConversationalAgent.elaborate` and the merged-bridge `MergedNavConvAgent.elaborate`) runs
`SpikingSpreadingController.turn_latency` → `relevance_by_latency` (was lines 389–414). Its inner loop did, for
each of `steps`≈60 settle steps, a **separate** `float(to_host(fs[self.ctx._cpat[c]]).sum())` for **every** concept
`c` in the vocab — i.e. `steps × V` device→host syncs per call. At V=320 that is ~60·320 ≈ 19,200 syncs ≈ the
~10 s the diagnostic measured. The sibling `SpikingLoopContextBuffer.read()` (the rate path) had the identical
anti-pattern (`for c in self.concepts: to_host(fs[idx]).sum()` per step).

## The fix (technique = device segment-sum, mirror of `_gate_coupling_rates_vectorized`)
Build ONE device array concatenating every concept's attractor indices (`self.ctx._cpat[c]`) in vocab order, plus
the `xp.add.reduceat` segment-start offsets (cached). Per step, one `xp.add.reduceat(fs[concat].astype(float64),
starts)` computes ALL concepts' firing-sums at once on-device; sync the resulting V-vector to host **once** per step
(`relevance_by_latency`, which needs the per-step first-crossing test), or accumulate on-device and sync **once at
the end** (`read()`, which only needs the windowed total). Reuse-by-import: `sim.backend` (already held as `self.B`
/ `self.xp`); the reduceat pattern mirrors `SimulationBridge._gate_coupling_rates_vectorized` (`sim/bridge.py:3253`).

- `relevance_by_latency`: `steps × V` syncs → `steps × 1` (at most 60 V-vector transfers; skips the read once all
  concepts have fired — the sim dynamics still run every step, so the result is unchanged).
- `read()`: `window × V` syncs → **1** (on-device accumulation, one transfer at the end).
- Compatibility: the layout for `relevance_by_latency` is cached on the **controller** (keyed by `self.ctx`
  identity), depending only on the primitive `_cpat`/`xp` attributes the original loop already used — so it works for
  every `.ctx` variant, the standalone `SpikingLoopContextBuffer` AND the shared-slice `_SharedDlpfcContext` used by
  the merged (`nav_conv_merged_bridge.py`) and unified (`unified_brain_bridge.py`) `elaborate` paths. No new
  attribute is required on the ctx.

## Answer-identity proof (the hard gate)
For **boolean** firing states, each reduceat segment's sum equals exactly that concept's integer firing count =
the original `to_host(fs[idx]).sum()`. The fraction `count / psize` (psize=50, count ≤ 50 — exact in float64) and
the `> thresh` test are therefore bit-identical per concept per step, so `relevance_by_latency` records the same
first-crossing step for every concept, and `read()` returns the same per-concept rate. Empirically:

| Check | V=6 (small clustered graph) | V=64 (16×4 clustered graph) |
|---|---|---|
| `relevance_by_latency` dicts identical vs baseline | **True** (all probes, diffs []) | **True** (probe0 + 5 probes) |
| `read()` max abs diff vs baseline | **0.0** | **0.0** |

Baselines were captured on the unmodified code and compared to the batched code on the same seed/graph/state.

## Speed
| V | `relevance_by_latency` baseline (per call) | batched (per call) | speedup |
|---|---|---|---|
| 6  | 1.149 s / 6 calls | 1.172 s / 6 calls | ~1× (sync count too small to matter at V=6) |
| 64 | **0.4719 s** | **0.1964 s** | **2.4×** |

The win is the removal of `steps × (V−1)` syncs, so it grows with V: 2.4× at V=64 is a verified lower bound; at the
production V=320 (the audit's ~10 s `elaborate`) the sync fraction is ~5× larger, so the expected win is
substantially higher. (The fixed per-step `_run_one_simulation_step` cost is shared by both versions and bounds the
asymptote.)

## Gates
- **Answer-identical (HARD):** PASS — latency dicts identical + `read()` diff 0.0 at V=6 and V=64 (above).
- **No-regression:** `tests/test_brain_conversational_agent.py` + `tests/test_one_brain_composer_agent.py` (the
  `elaborate` path runs through `relevance_by_latency` / `_SharedDlpfcContext` here) — `<TEST RESULT PENDING>`.
- **`sim/` edit:** NONE (runner-side `content_selection_spiking.py` only).

## Provenance
Edit: `research/runners/content_selection_spiking.py` (`SpikingLoopContextBuffer._segsum_layout` + batched
`read()`; `SpikingSpreadingController._latency_segsum_layout` + batched `relevance_by_latency`). Pattern source:
`sim/bridge.py:3253` (`_gate_coupling_rates_vectorized`). Audit: `2026-06-22-megakernel-revisit-optimization-audit.md`
(opportunity #1). Baselines/timings on RTX 3090, `SIM_BACKEND=cupy`.
