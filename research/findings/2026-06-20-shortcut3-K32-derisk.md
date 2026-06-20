# Shortcut #3 — the production conversational sequencer: the K=32 routing-margin gate (R0) — IN FLIGHT (2026-06-20)

**Type:** implementation (Stage S2-finish of the approved #3 deployment plan — the K=32 routing-margin gate ONLY; NOT
the production fold / 320-scale, which the controller dispatches next if K=32 GOes). GPU (`SIM_BACKEND=cupy`).
Runner/config-side only — **NO `sim/` edit**. The no-confab **moat is the HARD gate (0 false-accepts, NEVER weakened)**.

**Plan:** `research/findings/2026-06-20-shortcut3-sequencer-deployment-scoping.md` (the K=32 treatment §3: R0 → R1 → R2,
cheapest-first). #3 retires the production host `_scan` (`one_brain_composer.py:510`); the on-bridge spiking sequencer
is GO to K=16, and the K=32 "NEGATIVE" was the CHEAPEST retreat ONLY (divnorm re-tune, `gain=0.11`). This stage lifts
the K=32 routing margin via the named retreats.

---

## TL;DR (the verdict)

<!-- VERDICT PLACEHOLDER -->

---

## 1. THE DIAGNOSED K=32 FAILURE (from the committed S2 raw — NOT a moat failure)

The committed S2 K=32 NEGATIVE (`research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json`,
`first_break_K=32, k_star=16, gain=0.11`) was extracted at the seed level and confirmed in this stage:

```
seed 43, cue (sun, hop), correct block 4:
  m4 = 0.116          (the CORRECT block's match pool fired)
  m0..m3, m5..m31 = 0.000   (EXACTLY zero on all 31 other blocks -- confirmed: 31/32 pools exactly 0.0)
  match_thresh = 0.15  ->  m4 below threshold  ->  decision = abstain  (OVER-abstention, the SAFE direction)
  cleanup modes (exact/extra/miss) = 64/0/0   (the divnorm cleanup is PERFECT at K=32)
  moat: 3/3, FA_total = 0 at K=32  (the moat held -- the failure is over-abstention, not confabulation)
  seeds 42, 44: eq_all = True, perm = True, moat 0-FA  (clean GO at K=32)
```

**The diagnosis (unambiguous):** NOT a moat failure (moat 0-FA at K=32), NOT a cleanup/leak failure (`64/0/0`, the 31
non-matching blocks fire EXACTLY `0.000`), but a true-match-rate margin squeeze — the K-way priority WTA's larger
inhibitory fabric at K=32 pulls the WINNER's own match pool to `0.116`, just under the `0.15` threshold the K=2
op-point fixed.

## 2. R0 — THE THRESHOLD RE-CALIBRATION (the cheapest, untried fix)

**The premise:** the no-match floor at K=32 is EXACTLY `0.000` (the divnorm killed all cross-block leak), and the
winner fired `0.116`. So the threshold has a full `0.116`-wide no-match margin to drop into. Lowering `match_thresh`
into the open `(0.000, 0.116)` interval (e.g. `0.06`) ADMITS the correct match WHILE leaving the off-target (all at
`0.000`) far below threshold — **zero false-accept risk by construction.** This is the SAFE-direction fix: it removes
the over-abstention without touching the moat.

**The change (runner-side only, NO `sim/` edit):** the committed S2 run fixed `match_thresh` at the K=2 op-point `0.15`
(no CLI knob). This stage adds a `--match-thresh` CLI argument threaded through `run_seed_K` → all four
`run_sequencerK_with_drive` call sites (present / raw-control / lesion / permuted) — a pure plumbing change in
`research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py`. The default stays `0.15` (byte-identical to the
committed run); R0 runs at the lowered threshold.

## 3. THE PER-SEED MATCH-RATE + FA TABLE (R0, K=32, 6 seeds, D=128)

<!-- R0 TABLE PLACEHOLDER -->

## 4. THE ANTI-CHEAT TABLE (the moat FA==0 foregrounded)

<!-- ANTI-CHEAT TABLE PLACEHOLDER -->

## 5. THE EXACT COMMANDS

```bash
# R0: K=32 routing margin via the lowered match threshold (the open (0,0.116) margin), 6 seeds, D=128, GPU.
SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk \
    --seeds 42,43,44,100,101,102 --dim 128 --ks 2,4,8,16,32 --retreat divnorm --gain 0.11 --match-thresh 0.06 \
    --out research/findings/raw/_phaseB_onebrain_sequencerK_k32_R0.json
```

<!-- R1 COMMAND PLACEHOLDER (only if R0 insufficient) -->

---

## 6. SOURCES

- **The shortcut + host op:** `research/runners/one_brain_composer.py` (`_scan`:510; the inlined twins
  `query_patient`/`query_agent`/`ask_yes_no`/`render_fact`).
- **The runner (R0 knob added):** `research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py`
  (`--match-thresh` threaded through `run_seed_K` → `run_sequencerK_with_drive`).
- **The proven sequencer:** `research/runners/_phaseB_onebrain_sequencerK_derisk.py` (the K-way builder + production
  rule); `_phaseB_onebrain_sequencerK_divnorm_derisk.py` (`run_sequencerK_with_drive` with `match_thresh`);
  `_phaseC_S5_divnorm_derisk.py` (the divnorm score bridge).
- **The committed evidence:** `research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json` (the S2 K=32 NEGATIVE
  by retreat-1 ONLY; seed-43 `m4=0.116` < `0.15`, all others `0.000`, moat 0-FA, cleanup `64/0/0`);
  `2026-06-20-shortcut3-sequencer-deployment-scoping.md` (the R0→R1→R2 treatment).
- **The result (R0):** `research/findings/raw/_phaseB_onebrain_sequencerK_k32_R0.json`.

_The no-confab moat is the HARD gate in this stage and is NEVER weakened. The K=32 failure is over-abstention (the SAFE
direction), so R0 (lowering the threshold into the zero-leak no-match margin) closes it with FA preserved at 0 by
construction. Reuse-by-import; NO `sim/` edit._
