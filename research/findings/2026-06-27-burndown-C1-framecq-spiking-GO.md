# Burndown conversion C1 — numpy FrameCQ → the SPIKING competitive-queuing serial-order renderer — GO (2026-06-27)

**Verdict: GO — CLEAN CONVERSION, wired + default-on-GPU.** The argument-structure composer's verb-frame
word-ordering, previously a numpy rate-coded primacy gradient + `max()`-argmax (`ArgStructureComposer.FrameCQ`,
`argstructure_composer.py:125`), now runs on the VALIDATED SPIKING competitive-queuing read-out
(`NeuralSerialOrderRenderer` — concept pools driven by a primacy CURRENT on a real `SimulationBridge`, the per-pool
spiking RATE ranking = the emission order; the packaged 6/6-GO `_phaseB_serial_order_spiking_derisk`). This is the
FIRST Bucket-A operation-conversion toward the fully-spiking-one-brain end-state. Spec:
`2026-06-27-burndown-bucketA-build-plan.md` (C1 + §4).

**The conversion is a pure substrate swap of one self-contained function:** the numpy FrameCQ orders frame-slot
indices by a numpy primacy vector; the spiking renderer orders the SAME indices by the rate ranking of pools driven
with the same primacy as graded current. Same function, neurons instead of `max()`.

---

## The de-risk (the HARD GATE) — `_burndown_C1_framecq_spiking_derisk.py`, GPU, 6 seeds (42–47)

| Check | Result |
|---|---|
| **PARITY** (spiking emit-order == numpy FrameCQ.emit-order, EXHAUSTIVE over every verb frame × every realized-slot subset) | **6/6 seeds, 22/22 each** |
| Anti-cheat aggregate (primacy-true vs permuted, `g1_verdict` 0.10/0.5 bars) | **PASS, 150% over permuted, 6/6** |
| **(a) EQUAL-DRIVE control FAILS** (flat primacy → no determinate order) | **True — true 0.174 ≪ permuted 0.870** (the neurons serialize, not pool bias) |
| **(b) PERMUTED-order beaten ≥10%** | **True — 150% over** |
| **(c) CROSS-FRAME** (same slots, a different frame's primacy → a DIFFERENT order) | **True — 8/8 differ, every seed** |
| **GPU rate-real** (ranking read from real `cp_firing_states`, not a host argmax) | **True — mean rate gap 0.320** |
| **(d) MOAT 0-FA** (render gated by a stored composite; unstored → None) | **True** |
| **(e) AGRAMMATISM ablation preserved** ("the boy goes to the park" → "boy go park") | **True** |

Falsification bar (no parity ≥5/6, OR equal-drive doesn't fail) NOT hit — parity is exhaustive 6/6 and equal-drive
fails decisively. Raw: `research/findings/raw/_burndown_C1_framecq_spiking.json`.

**Why parity is exact:** the numpy `FrameCQ` teaches each frame the IDENTITY order (canonical slot 0 first; the
frame lexicon already lists content slots in canonical order), so `emit_order(fid, idx)` returns `idx` in canonical
ascending order. `render`'s realized-slot indices arrive in canonical-frame order. The spiking renderer drives the
first (lowest canonical index) with the highest primacy current → its rate ranking returns the same canonical order.
The order is genuinely neural (equal-drive → an anti-canonical/random order, well below permuted).

### One real wrinkle found + fixed during wiring (the 4-slot tie)

The validated SVO de-risk used a **3-level** primacy gradient `(2400, 1700, 1000)`. The `give`/`send` frames have
**4** content slots (agent/action/THEME/RECIPIENT). With only 3 levels, `order()` (which indexes
`primacy_pA[min(i, len-1)]`) gave slots 3 and 4 the **same** current → their rate ranking flipped on neuron
heterogeneity, so "the girl gives **to the dog the ball**" instead of the canonical "**the ball to the dog**". The
fix (build-plan §4(1) — "package the multi-frame primacy into the renderer") is a monotonic, widely-spaced gradient
sized to the largest frame: `(2800, 1900, 1000, 400)` (4 strictly-decreasing levels, Δ=900). Stress-tested
**0/40 non-canonical** sequential calls × 6 seeds for both 3- and 4-slot frames; with it the de-risk anti-cheat
true rose to ~1.000 and the wired 4-slot render matches the numpy oracle exactly. The fix lives in the adapter
(`SpikingFrameCQ._PRIMACY_GRADIENT`); the shared `NeuralSerialOrderRenderer`'s SVO default is untouched.

---

## What shipped (Step 2) — reuse-by-import, additive, NO `sim/` edit

- **`research/runners/argstructure_composer.py`** (additive):
  - new `SpikingFrameCQ` — a drop-in for `FrameCQ.emit_order` reading the order off the spiking
    `NeuralSerialOrderRenderer` (lazily imported + built, so the CPU/numpy path never touches a bridge);
  - `ArgStructureComposer.__init__(use_spiking_cq=None)` + a `_ordering_engine()` selector + a per-call
    `render(..., use_spiking_cq=None)` override. The render path calls the selected engine's `emit_order` instead of
    the hard-coded `self.frame_cq`.
  - **DEFAULT (`use_spiking_cq=None`) = the `consolidated_320` pattern:** SPIKING on GPU (`SIM_BACKEND=cupy`, the
    production substrate — honors the fully-spiking-one-brain end-state) / the numpy FrameCQ ORACLE on CPU
    (`SIM_BACKEND=numpy` — CPU-portable, the retained test oracle; the spiking renderer cannot run GPU-less).
    `True`/`False` force the choice.
- **`research/runners/tense_aspect_composer.py`** (additive): forwards `use_spiking_cq` through `__init__` + adds the
  per-call override to `render_tensed` (the subclass surface is consistent).
- **`tests/test_argstructure_spiking_cq.py`** (new CI guard, GPU-gated, skips off-GPU): pins parity (spiking ==
  numpy, every frame + subset, seeds 42/43), the wired-render-matches-numpy-oracle parity for the headline frames,
  the per-call override, the EQUAL-DRIVE anti-cheat failure, and moat + agrammatism on the spiking path.

### Verification (both backends)

- **`SIM_BACKEND=numpy`** (default resolves to numpy oracle, byte-identical): `test_argstructure_composer.py` +
  `test_tense_aspect_composer.py` + `test_wh_question_parser.py` → **31 passed, 1 skipped**; no bridge built;
  console + routed imports fine.
- **`SIM_BACKEND=cupy`** (default resolves to spiking): the SAME three guard files → **32 passed** (the spiking
  default-on path produces render output identical to the numpy oracle across tense + wh-questions); the new C1
  guard → **7 passed**.

**Scope honored:** `first_chat_console.py` is NOT touched — its `--argstructure` path now gets the spiking ordering
**for free** on GPU (the default flows from the composer), the numpy oracle on CPU. C2/C3 (the flat-SVO
`enable_neural_render` flip + the onebrain console path) remain separate later conversions.

**Honest residual / follow-on:** the spiking renderer builds a `SimulationBridge` per `ArgStructureComposer`
instance (lazy, cached per instance), so a process that constructs many composers on GPU pays a per-instance bridge
build (the GPU guard suite took ~14.5 min for this reason). In production the console builds ONE composer → the
renderer is built once. A shared-bridge / renderer-pool optimization is a bounded follow-on, not on the C1 critical
path. The numpy oracle remains the CPU-portable path + the CI correctness anchor (parity is asserted against it).

NO `sim/` edit anywhere in C1.
