---
type: finding
status: contributing
date: 2026-08-26
mechanism: b1-v1-orientation-selforg-onbridge
artifacts:
  - research/findings/raw/_b1_v1_selforg_flip_soak_6seed.json
  - research/findings/raw/_b1_v1_selforg_surpass_probe_s42.json
---

# B1 V1 self-organized RF bank — PRODUCTION WIRE-IN (PARTIAL): organ + BRAIN_V1_SELFORG flag built, OFF byte-identical, 6-seed flip-soak = HOLD-OFF (common-mode BOUNDARY holds; the previously-inert STDP lever is now genuinely exercised and still does not break it)

**Verdict: PARTIAL.** The production wiring is BUILT, additive, and OFF-byte-identical — a self-organized-RF organ
(`research/runners/v1_selforg_production_organ.py`) drop-in for `apply_v1_gabor_weights`, gated by `BRAIN_V1_SELFORG`
(default OFF) at the g11 visual-cortex build site. But the on-bridge self-org realization remains a **COMMON-MODE
BOUNDARY**: at 6 seeds, production scale, `osi_post_frac` mean 0.0092 (min 0.0051) « the 0.5 flip gate, a negligible
lift over the pre-random baseline (mean 0.0037), with `on_minus_off_mean` -0.001001 ~0 (the ON and OFF channels
potentiate to near-identical weights, so the signed RF cancels). **The flag STAYS OFF and MUST NOT be flipped** —
flipping it would replace the working host
Gabor bank with a non-oriented common-mode bank and degrade V1. This is the honest NO-GO the wiring task anticipated.

## What was NEW this cycle (beyond re-confirming the 2026-08-14 BOUNDARY)

1. **A silent instrument bug: `--rule stdp` was INERT in every prior on-bridge run.** The de-risk runner's `develop`
   loop calls `bridge._run_one_simulation_step()`, which does NOT advance `runtime_state.current_time_ms`. With the
   clock frozen at 0.0, every spike shares one timestamp, every STDP Δt is 0, and **every STDP weight update is exactly
   0.0** — the engine's own guard prints `⛔ STDP IS INERT`. So the LTD lever the runner's docstring proposed as the
   fix for the potentiation-only common mode (`:201`, "the input-specific DEPRESSION that potentiation-only Hebbian
   lacks") had **never actually run**. The organ fixes this (advances the clock each step), so a timing rule is now
   genuinely exercised.

2. **With STDP genuinely exercised (clock-fixed), it still does NOT break the common mode** (artifact
   `research/findings/raw/_b1_v1_selforg_surpass_probe_s42.json`, reduced-scale seed-42, n_v1=2048, dev=10000):
   `rule=stdp` → `on_minus_off_mean` -0.000116, `osi_post_frac 0.0054` (≈ pre 0.0044), and the incoming L2 norm
   **collapses** (`l2_mean` 47.25 vs 497.59 for hebbian) — STDP, with its net-depression bias (a_minus > a_plus),
   drives the weights to a low floor rather than building opponent structure. `rule=both` keeps l2 healthy (433.03) but
   `on_minus_off_mean` stays -0.015516 and `osi_post_frac 0.002`. **HONEST CAVEAT** (per
   `2026-07-31-laneD-normalization-arms-are-RATE-CONFOUNDED`): the STDP arm's low OSI is partly a RATE confound (less
   firing → less orientation signal), so this is not a clean refutation of timing-based decorrelation — it is a
   refutation of *this* STDP configuration (net-depression, no plastic inhibition). The clean discriminator remains
   `on_minus_off_mean`, which is ~0 for every rule → common mode is fundamental to a feedforward rule on ON/OFF-split
   full-field gratings, independent of the rate confound.

3. **A DoG center-surround (retinal whitening) front-end does not supply the opponency either** (same artifact):
   `hebbian+dog` → `on_minus_off_mean` -0.000592, `osi_post_frac 0.001`. Removing the input DC does not, by itself,
   make the feedforward rule bind a signed RF — consistent with SAILnet requiring whitened input **and** plastic
   lateral inhibition *together* (2026-08-14). Neither cheap no-`sim`-edit lever is a substitute for the missing
   decorrelation.

## The production wiring (additive, no `sim/` edit)

* **Organ:** `research/runners/v1_selforg_production_organ.py` — reuse-by-import of the on-bridge de-risk mechanism
  (`_b1_v1_selforg_onbridge_derisk`: build_v1_bridge / build_isotropic_support / read_v1_rfs /
  gabor_orientation_tuning / raw_weight_stats). It develops a self-organized RF bank on a minimal internal substrate
  (isotropic local support + random init → the substrate's own rate-Hebbian/STDP + homeostasis, CLOCK-FIXED), freezes
  (critical-period close), and returns the LEARNED relative-index bank. `apply_v1_selforg_weights(bridge, …)` is a
  drop-in for `apply_v1_gabor_weights` — it installs the learned bank on the production `retina→cortex_v1_simple`
  pathway via `set_pathway_weights(add_missing=True)`, exactly as the Gabor path does.
* **Flag:** `BRAIN_V1_SELFORG` (default OFF) at `g11_bg_runner.py`'s visual-cortex build site. OFF → the branch is
  character-identical to the pre-edit `apply_v1_gabor_weights` call (the organ is not even imported). `BRAIN_V1_SELFORG_LESION`
  in {freeze, shuffle} exposes the faculty's own lesion oracle.
* **Placement rationale:** V1 visual cortex is used ONLY by the g11 gridworld path (`--enable-visual-cortex`), not by
  the live-chat server, so "production" for B1 is the g11 V1-build site.

## Verification

* **OFF byte-identical (PROVEN).** Clean minimal retina+cortex_v1_simple bridges at the g11 arch (8×4×16×16, retina
  32): the wired path with the flag UNSET hashes `sha256(V1 weights)` **identical** to a direct `apply_v1_gabor_weights`
  call (`9a4376a776f5239a…`, wsum 735019.2, nnz 529165). The flag ON installs a genuinely different bank (`d17b1c65…`,
  nnz 724227) — the flag is load-bearing at the wiring level.
* **Lesion oracle (the faculty's own; LOAD-BEARING check).** In the 6-seed soak, seed 42: `learn osi_post_frac ≈
  0.0066`, `freeze (no learning) ≈ 0.0026`, `shuffle (orientation-destroyed input) ≈ 0.0061`. Learning
  does **not** clear the freeze/shuffle controls by the +0.15 margin → the coupling is exercised but the output does
  NOT become load-bearingly oriented. This IS the honest BOUNDARY: the tiny OSI signal that exists collapses to the
  no-learning / no-orientation controls, so there is nothing load-bearing to flip on.

## The 6-seed flip-soak (the flip gate) — HOLD-OFF

`research/runners/_b1_v1_selforg_flip_soak.py`, production scale (n_v1=8192, dev_steps=24000), cupy, seeds 42–47.
Gate: all 6 `osi_post_frac ≥ 0.5` AND `≥ pre+0.15` AND `learn ≥ lesion_ctrl+0.15`.

| metric | value |
|---|---|
| flip_decision | **HOLD-OFF** |
| per-seed verdicts | 6× BOUNDARY |
| osi_post_frac mean (min) | 0.0092 (0.0051) |
| osi_pre_frac mean | 0.0037 |
| on_minus_off_mean | -0.001001 |
| lesion (seed 42) freeze / shuffle / learn | 0.0026 / 0.0061 / 0.0066 |

## Banked next mechanism (named, needs a `sim/` edit — flagged, NOT done here)

The residual is unchanged and specific: a feedforward rate/STDP rule on ON/OFF-split full-field input learns a common
mode with no opponency, because **every V1 cell sees the same full-field stimulus and fires for all of them, so all
cells potentiate the same blob** — there is no competition to make a cell fire *selectively* for one orientation. The
fix the literature uses is **LEARNED (plastic) anti-Hebbian recurrent inhibition — the SAILnet/Foldiák decorrelation
rule** (Zylberberg, Murphy & DeWeese 2011), which the FIXED FS pool tried on-bridge cannot substitute for (it gives
uniform gain control, not per-pair decorrelation). This requires **inhibitory-pathway plasticity**, a likely `sim/`
edit — out of scope for this additive-default-OFF wiring rung, and named as the next rung. The organ + flag + soak are
in place so that the moment that mechanism clears OSI≥0.5 at 6 seeds, the flip is a one-line default change with the
gate already written.

## Anti-cheats (all held)

Isotropic RF support (all ON+OFF within radius-4, carries no orientation) — any oriented RF must be learned. Host Gabor
bank never applied to the self-org pathway (random-init then learned; host used only as the scoring reference). Determinism:
`cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed` (per-seed OSI differs). OSI is label-free. The flip gate requires
`learn ≥ lesion_ctrl + 0.15`, so a rate-inflated or support-inflated OSI cannot pass.

## Sources

SAILnet (whitened patches + plastic anti-Hebbian lateral inhibition → oriented Gabor RFs): Zylberberg, Murphy &
DeWeese 2011, PLoS Comput Biol 7(10):e1002250. Common-mode / decorrelation need: `2026-06-04-spine-item2-spiking-cleanup-needs-decorrelation.md`.
Rate-confound caveat: `2026-07-31-laneD-normalization-arms-are-RATE-CONFOUNDED-not-a-clean-refutation.md`. On-bridge
BOUNDARY this extends: `2026-08-14-b1-v1-selforg-onbridge-operating-point-BOUNDARY.md`. Off-bridge numpy ceiling (GO):
`2026-06-21-B1-v1-gabor-selforg-derisk.md`. Host bank: `sim/visual_cortex.py:build_v1_simple_weights`; wiring site
`research/runners/g11_bg_runner.py` (visual-cortex build).
