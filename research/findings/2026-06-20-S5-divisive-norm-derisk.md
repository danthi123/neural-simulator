# Phase C — S5 (the last host read) CLOSES on-bridge via OPTION 4 (`input_divisive_norm`) — GO, ZERO new sim/ code (2026-06-20)

**Verdict: OPTION-4 GO across the per-query-peak sweep, 3 seeds, CPU + GPU. The residual host read at seam S5 — the
cleanup score's per-query peak-normalization (`rf_phasor_composer.py:270-272` / Phase B's `scores_to_drive`,
`peak=scores.max(); drive=scores/peak`) — is reproduced ON-BRIDGE by the EXISTING `input_divisive_norm` Carandini-
Heeger primitive (`sim/regions.py:240` + `sim/config.py:440` + the guarded per-step block at `sim/bridge.py:6048`),
flagged on a runner-built cleanup-score pool. The host `scores.max()` read is RETIRED for this seam. NO `sim/` edit
(Option 4 uses the primitive already in the codebase, flipped from the runner). The no-confab moat was NEVER weakened
— 0 false-accepts on every absent/cross cue, both backends, every peak.**

This is the cheap-first de-risk the deep research named (`2026-06-19-S5-on-bridge-normalization-deep-research.md`,
Option 4, the "cheapest first probe ... the primitive exists in `sim/`, zero new code"). It confirms the research's
top-line call: **S5 is the point-neuron-feasible DIVISIVE-GAIN half of normalization (Carandini-Heeger), NOT the
off-diagonal whitening / Mikulasch-Priesemann dendritic boundary** — the cleanup score is already rectified and
non-negative, so there is no signed common mode to lose, only a rank-ordered magnitude to rescale.

## Result (3 seeds, D=64, real `OneBrainComposer`; runner `_phaseC_S5_divnorm_derisk.py`)

```
CPU (SIM_BACKEND=numpy):  DIVNORM S5_GO 3/3 (==host+moat+lesion; moat 3/3, FA_total 0)  HOST-NORM S5_GO 3/3
                          RAW fails_sweep 3/3  lesion 3/3  OFF-guard PASS  | decisive(seq diag) DIV 3/3 HOST 3/3
GPU (SIM_BACKEND=cupy):   DIVNORM S5_GO 3/3 (==host+moat+lesion; moat 3/3, FA_total 0)  HOST-NORM S5_GO 3/3
                          RAW fails_sweep 3/3  lesion 3/3  OFF-guard PASS  | decisive(seq diag) DIV 2/3 HOST 2/3
```

Operating point (a FIXED op, NOT per-query): `input_divisive_norm=True`, `enable_input_divisive_norm=True`,
`sigma=1.0`, `gain=0.05`, `input_gain=1.0`. The placed threshold IS the Izhikevich rheobase.

- **DIVNORM (option 4, on-bridge): S5_GO 3/3 both backends** — every decision `==host`, the moat holds 0-FA, lesion
  fails safe, at peak multipliers {0.1, 1.0, 10.0} (the decisive per-query-peak sweep). On every present cue exactly
  ONE word fires per role per block (`lit=[(1,1),(1,1)]`) — the winner; the runner-up stays silent. NO host
  `scores.max()` anywhere in the score path.
- **HOST-NORM (positive control): S5_GO 3/3** — the existing host-peak `scores_to_drive` on the SAME battery passes
  (battery discriminable, harness sound). The on-bridge version MATCHES it, not a broken control.
- **RAW (negative control): fails_sweep 3/3** — the placed threshold on the RAW un-normalized drive lights the whole
  role row (`lit=[(6,6),(7,8)]` etc.), loses `==host`, breaks the moat → reproduces the Task-1 `both_suprathreshold`
  wall (`2026-06-19-phaseC-task1-S5-seam-derisk.md`). Confirms the normalization is LOAD-BEARING — the GO is
  attributable to the divide, not the wiring.
- **lesion-fails-safe 3/3** — severing the score-pool drive → decoded lines silent → abstain.
- **OFF == byte-identical PASS** — a divnorm-OFF score bridge has `cp_input_divisive_mask=None` (the per-step block
  unreached) and steps byte-identically to a second OFF bridge; the ON bridge's mask is not None and the divide
  CHANGES the dynamics (the primitive's guarded-no-op contract; no `sim/` edit was made, this just asserts it).

## Why Option 4 works (the mechanism — and the one tuning subtlety the probe exposed)

The cleanup score is a GRADED matched-filter membrane `Re(c)` per role per word (probed: winner ~peak ~1.0–1.3e6,
runner-up ~0.4–0.48·peak — RELATIVE discrimination). The `input_divisive_norm` primitive divides each flagged pool's
pre-threshold drive by `sigma + gain·mean(pool drive)` — the per-query divisor IS the pool's own total drive (the
"divide pre-threshold input by total pool drive" the research names). Feed the role's score vector as external input
current into one flagged V-neuron pool (one role at a time, so the divisor is THAT role's per-query total), and:

- **Scale-invariance is exact** (the research's claim, verified): in the saturated regime
  (`input_gain·gain·mean >> sigma`) the post-divide value is `peak/(gain·mean)` — a dimensionless RATIO independent of
  `input_gain` and of any per-query peak multiplier. The normalized winner/runner-up separation = the score ratio
  (~2.1–3.9×), preserved across `input_gain` and per-query peaks spanning ≥4 orders of magnitude.
- **THE TUNING SUBTLETY (the empirical fork the research flagged):** the primitive has NO post-divide output gain, so
  the normalized ratio's MAGNITUDE matters. At `gain=0.5–1.0` the saturated ratio is ~6–12 (a number, not a current)
  — far BELOW the Izhikevich rheobase (tens-to-hundreds of pA) → **nothing fires** (a one-line read of this is why a
  blind `gain=1.0` probe shows 0 firing). The fix is to set `gain` SMALL (≈0.05) so the saturated ratio
  `peak/(gain·mean)` lands IN the firing band: `post_winner ≈ 117 pA` (supra-rheobase → fires), `post_runner ≈ 56 pA`
  (sub-rheobase → silent). The Izhikevich rheobase, sitting between them, IS the placed threshold — and because the
  divide removed the per-query scale, that threshold is the SAME across all peaks. This is still the built-in
  primitive (just the right `gain`), no new `sim/` code, no output-gain addition needed.

So the mean-pool divnorm DOES suffice for S5 — the empirical fork (mean-pool vs the NEF-FS pool) resolves in favour of
the cheapest option. (Option 1, the validated NEF input-norm FS pool, remains the fallback and is NOT needed here.)

## `decisive` is a Phase-B sequencer diagnostic, not an S5 gate (the honest GPU note)

The runner also reports a `decisive` check (the Phase-B match pool `m >= 0.20` on a present cue). It is GREEN on CPU
(DIV 3/3) but dips to 2/3 on GPU. This is NOT an S5 failure: it is a property of the SHARED Phase-B sequencer's
`m0/m1` match-pool rate under the cupy float path, and it dips IDENTICALLY in the HOST-NORM control (HOST 2/3 on GPU,
the host-peak read that has nothing to do with divnorm — on GPU seed 44 the host margin m=0.182 is actually LOWER than
the divnorm m=0.234 at pm=1.0/10.0). The load-bearing S5 gates the task gates on — `==host` and the moat-0-FA — are
3/3 on BOTH backends. The `decisive` margin is a separable sequencer-health knob, foregrounded honestly, not relaxed.

## What it closes / scope

Task 1 (`2026-06-19-phaseC-task1-S5-seam-derisk.md`) walled a FIXED `cp_connections` projection (option a) and the
loop proceeded with the host result-read (option b: all computation + control in spikes, one DATA number read at S5).
Option 4 RETIRES that last host read for the cleanup→sequencer seam: the per-query rescaling is the on-bridge divide,
the threshold is the rheobase, the decision is `==host` with the moat intact. The integrated who/what loop can run
with ZERO host round-trips between cleanup and sequencer.

HONEST SCOPE: validated at D=64, K=2 (the integrated loop's scale), 2 facts / 12-word vocab, 3 seeds. The score
battery's per-query peaks span ≥1 order of magnitude by explicit multipliers (the decisive control) AND the natural
4 cleanup vectors span ~1.46× with ratios 2.08–3.94. The closure is for the divisive-GAIN normalization seam; it does
NOT touch (and is categorically distinct from) the off-diagonal whitening dendritic boundary. The GO is the on-bridge
NORMALIZATION; the binding algebra upstream stays the principled FHRR idealization (the separate step-3 frontier).

## NO `sim/` edit — Option 4 used the existing primitive

Per the task's expectation, NO `sim/` edit was made. Option 4 flags `BrainRegion.input_divisive_norm=True` +
`cfg.enable_input_divisive_norm=True` on a runner-built score pool; the guarded per-step divide block
(`sim/bridge.py:6048`) was already in the codebase (built 2026-06-15 for the PPMI per-concept normalization). The
OFF==byte-identical guard confirms every existing run is byte-unchanged.

## Reproduce

```bash
# CPU (the validated path; deterministic):
SIM_BACKEND=numpy python -u -m research.runners._phaseC_S5_divnorm_derisk --seeds 42,43,44 --dim 64 \
    --out research/findings/raw/_phaseC_S5_divnorm_derisk_cpu.json
# GPU confirm (the production backend):
SIM_BACKEND=cupy python -u -m research.runners._phaseC_S5_divnorm_derisk --seeds 42,43,44 --dim 64 \
    --out research/findings/raw/_phaseC_S5_divnorm_derisk_gpu.json
```

Context: `2026-06-19-S5-on-bridge-normalization-deep-research.md` (the verdict + the ranked options, commit 94ca9fb8),
`2026-06-19-phaseC-task1-S5-seam-derisk.md` (why the fixed projection walled, commit 27c6422e). Harness reuses the
real `OneBrainComposer` (`block_cleanup_scores`) + Phase B's sequencer (`build_sequencer_bridge`,
`run_sequencer_with_drive`) by import.

## Follow-on (NOT on this de-risk's path)

- Wire Option 4 into the integrated loop's cleanup→sequencer seam (`_phaseC_task2_wholeturn_loop.py` `LoopComposer`)
  so the deployed who/what turn has zero host round-trips. (Drop-in: the score pool's divnorm-flagged firing replaces
  the host `scores_to_drive`.)
- Scale check at D=2048 / V=320 (the production composer scale) — the per-query peaks grow but the saturated ratio is
  scale-free, so the same `gain≈0.05` operating point should transfer; confirm the firing-band placement holds and the
  moat stays 0-FA at scale. The NEF input-norm FS pool (Option 1, `2026-06-05-composer-cleanup-NEF-GO.md`) is the
  graded-pool fallback if the firing-band basin needs the matched-filter structure at 320 concepts.
```
