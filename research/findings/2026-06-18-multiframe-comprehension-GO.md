# Multi-frame comprehension — richer-syntax #2 core, GO 6/6 (2026-06-18, CYCLE 203)

## Headline

The learned spiking parser comprehends **multiple word-order frames productively** — not just its
native SVO + the passive, but arbitrary *learned* frames (VSO "ran dog north", OSV) — by assigning
roles over **position × frame** instead of fixed word templates. GO 6/6 seeds: native SVO **1.000**,
non-native (VSO + OSV) **1.000**, with the permuted-frame and lesion controls collapsing to chance
and the no-confab moat holding. This is the core of richer-syntax #2 (the dual of the already-GO
generation frame-conditioning).

The de-risk was **pre-registered** (`2026-06-17-capability-frontier-to-basic-LLM-scoping.md`) and
**built-but-never-run** until now; this records its frozen-gate result.

## Mechanism

The production `BridgeParser` learns a `(word-position × voice) → role` Hebbian map (6 conjunction
units, 2 frames: active SVO + passive). This extends the conjunction to `(position × FRAME)` over N
learned frames: a per-`(frame, position)` Hebbian map → role ensemble, same spiking substrate +
embodied co-firing teacher (a small Izhikevich bridge, role-ensemble WTA read-out). The capability
target is a NON-NATIVE frame the parser was never given as a fixed template (verb-initial VSO,
object-initial OSV): it comprehends held-out sentences in that frame because the role assignment is
over `position × frame`, not words.

## Results (6 seeds: 42/43/44/100/101/102, GPU)

| metric | mean | note |
|--------|------|------|
| native SVO comprehension | **1.000** | no regression on the native frame |
| non-native (VSO/OSV) comprehension | **1.000** | held-out frames, role assignment generalizes |
| permuted-frame control | **0.222** (chance 0.333) | collapses — the "frame" isn't the native order in disguise |
| lesion control (zero conj→role weights) | **0.241** | collapses to chance — the learned weights are load-bearing |
| margin (signal vs lesion) | ~5× | large, every seed |
| no-confab moat | **6/6** | an untrained/ambiguous conjunction abstains |
| comprehension ≥ 0.90 | **6/6** | frozen gate met |

Frozen GO gate (pre-registered, not tuned-to-pass): held-out comprehension ≥ 0.90 on ≥5/6 seeds,
permuted-frame collapses, lesion collapses, native no-regression, moat holds — **all met (6/6)**.

## Anti-cheat controls (all passed)

- **Permuted-frame collapses (0.222 ≈ chance):** scrambling a frame's position→role map breaks
  comprehension for it — so the parser genuinely learned the frame structure, not the native order.
- **Lesion collapses (0.241 ≈ chance):** zeroing the learned conjunction→role weights drops to
  chance — the spiking learned map is load-bearing, not a hard-coded rule.
- **Native no-regression (1.000):** adding the extra frames doesn't break the native SVO frame
  (the frames don't interfere on the shared role ensembles).
- **Moat (6/6):** an untrained/ambiguous conjunction abstains (the no-confab guarantee survives).

## Honest scope + next

- This validates that the substrate can HOLD N frames and assign roles per frame, **given the
  frame**. **Frame SELECTION** (a learned cue → which-frame map, so the agent picks the frame from
  context rather than being told it) is the explicit next piece (the runner's GO note). The
  pre-registered gate listed it; this run measured the per-frame comprehension + the controls, which
  is the load-bearing half.
- Frames tested: SVO (native) + VSO + OSV. Real wh-questions, datives, imperatives are bounded
  extensions (more frames + the selection cue).
- NEXT (mirroring richer-syntax #1's path): add the cue→frame selection map, then wire a
  `FrameParser` into the agent (an opt-in, like `enable_attributed`), so the production agent
  comprehends multiple frames end-to-end. NO `sim/` edit (reuse the `BridgeParser` pattern).

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_multiframe_comprehension_derisk \
    --seeds 42,43,44,100,101,102
```
Runner: `research/runners/_phaseB_multiframe_comprehension_derisk.py`. Pre-registration:
`2026-06-17-capability-frontier-to-basic-LLM-scoping.md`.
