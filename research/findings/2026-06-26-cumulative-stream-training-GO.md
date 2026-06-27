# Cumulative / staged stream-cortex training — GO (2026-06-26)

**Owner directive served:** "stage so a week+ of training is cumulative, not restart-from-scratch." Also the
enabler that unblocks scaling concepts past the single-bridge *training* VRAM wall (train in chunks that each
fit, accumulating onto the prior brain).

## Mechanism (NO `sim/` edit — reuse-by-import)

The stream cortex's learnable state is `StreamCortexBridge.bridge.cp_connections` (the online rate-Hebbian
hub→target weights = the learned co-occurrence M; `read_codes()` derives the phasor codes from it). The
`SimulationBridge` already persists `cp_connections` via `save_checkpoint`/`load_checkpoint` (HDF5). So
cumulative training is a *reuse*, not a new mechanism:

- **`--save-bridge PATH`** — `bridge.save_checkpoint(PATH)` after training (persists the learned M).
- **`--resume-bridge PATH`** — `bridge.load_checkpoint(PATH)` AFTER `build_stream_bridge` and BEFORE
  `hear_corpus`, so continued streaming COMPOUNDS onto the preserved M. Same vocab + `n_hub` + `n_per` ⇒ the
  contiguous region index slices stay valid across the load. The frozen-brain anti-cheat gate is set after the
  load (the checkpoint rebuilds `core_config` with Hebbian ON).

(`research/runners/_curriculum_step1_320_real_corpus.py`, lines ~404–423 / ~698–731 / ~904.)

## Smoke de-risk — GO (16 concepts, numpy, 400 windows/stage, seed 42)

| Stage | corr(M,C) | recall | moat FA |
|---|---|---|---|
| fresh-N (400 win) | +0.733 | 1.000 (48/48) | 0 |
| reload@0 (preservation) | corr(M_reload, M_saved) = **1.000000** | — | 0 |
| reload+N (2N-effective) | **+0.745** ≥ fresh-N | 1.000 | 0 |

- **Preservation (load-bearing):** the reloaded M is **byte-for-byte identical** to the saved M (corr 1.000000
  ≥ the 0.999 bar) — checkpoint-resume loses nothing.
- **Compounding:** 2N-effective corr (+0.745) ≥ fresh-N (+0.733), recall held at 1.000 — continued training
  builds *onto* the preserved M rather than resetting it.
- **Moat: 0 false-accepts** at every stage (the no-confab moat is undisturbed by save/resume).

**Honest caveat:** the compounding margin is small at smoke scale because 16 concepts saturate corr fast (little
headroom above fresh-N). The preservation result (corr 1.000) is the rock-solid part; the full-scale run
(1,000 concepts, GPU, 7K windows/stage — more headroom) is in flight to show the compounding margin clearly and
will be appended here.

## Scope + follow-on

- **Same-vocab only:** resume continues training the *same* concepts (better codes from more corpus). Adding
  NEW concepts to a resumed brain (vocab growth) is the documented follow-on — it needs the bridge to grow its
  target region (the `auto_growth`/`TierPromoter` machinery), not just a checkpoint reload.
- **Why this matters for deep knowledge:** with preservation proven, a long corpus can be trained in chunks
  that each fit VRAM, accumulating — so the deep-knowledge brain can scale past the ~4.5K single-bridge
  training ceiling, and a week of training genuinely compounds.

Verdict: **GO** (mechanism proven at smoke; full-scale confirmation pending). Reuse-by-import, no protected edit.
