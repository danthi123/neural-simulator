---
type: finding
status: corrected
date: 2026-07-07
mechanism: microcircuit-credit
---

# D1 microcircuit — two biological deep-credit rules on the spiking substrate: the CLEAN-ERROR credit (M2.6 somatic-rate feedforward = clean-error feedback alignment) clears the depth-2 held-out bar (0.964, 3-seed) AND is batch-robust; raw Burstprop's NOISY burst-fraction credit is worse (~0.79) and batch-fragile. The `sim/` interneuron-cancellation is built (additive/byte-identical-when-off) + validated on-bridge for the burst READOUT — but adversarial-verify shows the depth-2 ACCURACY is carried by the clean-error FF rule, NOT the interneuron cancellation (its accuracy causality is the D2 on-bridge test).

**Date:** 2026-07-07
**Runner:** `research/runners/_gnw_d1_spiking_bdsp_derisk.py` (`--rule microcircuit`) + the additive `enable_bdsp_microcircuit` `sim/` path. Control: `research/runners/_gnw_d1_microcircuit_control_probe.py` (batch sweep). Adversarial-verify: workflow `wjn6hxyuu` (4 skeptic lenses + synthesizer, SURVIVES_WITH_SCOPE_FIXES) — every caveat below is folded in.
**Verdict:** GO on the mechanism + the clean-error-credit result, with a CORRECTED mechanism attribution (below). The `sim/` interneuron-cancellation rule is built + byte-identical-when-off + its burst-readout cancellation validated on a real bridge; the clean-error credit clears the 0.75 bar (0.964) and is batch-robust; the advance over raw Burstprop is the CLEAN-ERROR CREDIT CHANNEL (feedback alignment on the clean apical error), NOT interneuron cancellation per se and NOT any burst quantity.

## What genuinely SURVIVES (panel re-verified against the artifacts, not the summary)
- **Leakage-free split:** `make_task` permutes all 1024 unique bit patterns, cut 665/359; 0 train/test overlap all 3 seeds; permuted-label arm → chance (0.484).
- **Like-for-like:** `MicrocircuitBDSPNet(BDSPNet)` inherits W-init / fixed-random apical feedback `Y` / `Pbar` / optimizer VERBATIM; same seeds; oracle (fenced backprop) 0.958–1.0 (≥0.80, valid ceiling); **no weight transport** (`Y`, `W_PI` never a forward `W/Wᵀ`).
- **All anti-cheats collapse:** single-layer 0.173, apical-lesion 0.521 (~chance), wrong-sign 0.463 (anti-learns), permuted 0.484, no-teaching null held-out 0.521 with **hidden-layer weight-drift EXACTLY 0.0** (the P0 moat), probe-latent 0.93–0.99 vs frozen 0.51.
- **`sim/` diff additive/default-off/byte-identical-when-off:** pure insertion (42 lines, 0 deletions across `config.py`+`bridge.py`); guarded by `enable_bdsp` ∧ `enable_bdsp_microcircuit` ∧ `cp_bdsp_int_drive is not None`; installs `(apical_drive − int_drive)` then RESTORES the raw drive (no cross-step accumulation); determinism+kernel+numpy-integration suites 52 pass; a default bridge runs with all `cp_bdsp_*` still `None`.
- **On-bridge cancellation (Stage-A′′′), component-wise:** the `sim/` code subtracts `cp_bdsp_int_drive` into `cp_v_apical`; the interneuron-cancelled apical burst-probability **P returns to rest p0 (P: 1.0 uncancelled → 0.300 cancelled = p0)**. Validated on a real `SimulationBridge`.

## The decisive control (my own, 3-seed, hidden=128, ep=600, lr=0.3) — held-out vs batch
| batch | Burstprop | microcircuit / clean-error credit |
|---|---|---|
| 32 | **0.924** | **0.963** |
| 128 | 0.788 | 0.964 |
| 512 | 0.600 | 0.865 |
| 665 (full) | 0.513 | 0.615 |

**The clean-error credit is BATCH-ROBUST** (0.96 flat across batch 32–128); **raw Burstprop is BATCH-FRAGILE** (monotone 0.92→0.79→0.60→0.51). This is a genuine, biologically-meaningful advantage: a clean low-variance credit signal `e_k = φ′(E_k)·(Yᵀ@e_{k+1})` (a weighted average over the upper layer via fixed-random `Y`) is stable across hyperparameters where Burstprop's per-unit burst-fraction credit `b = B − Pbar·E` is fragile. **At each rule's best batch, BOTH clear the 0.75 bar** (0.96 vs 0.92) — so raw Burstprop is NOT hard-noise-limited on this task.

## The CORRECTED mechanism attribution (the load-bearing adversarial-verify finding)
The claimed mechanism was "the SST interneuron cancels the predictable top-down so the apical carries a CLEAN ERROR, and THAT clears the Burstprop floor." **The panel's controls refute that the interneuron/cancellation is the ACCURACY driver at depth-2:**
- A **clean-error feedback-alignment net with NO interneuron and NO burst machinery** reproduces **0.964 byte-identically per seed** → the interneuron is not load-bearing for the numpy accuracy.
- **Killing the burst signal in the microcircuit** (`beta=0` / `p0=0.99` / `ema=0`) leaves accuracy **0.964 unchanged**, while the SAME `beta=0` collapses Burstprop to chance → burst `B/P/Pbar` are **dead-code for the microcircuit's weight update** (`B` is never even computed in `MicrocircuitBDSPNet.train_step`).
- A **trivial variance-reduction knob does NOT close the gap** (more epochs on Burstprop → only ~0.906; wider Burstprop REGRESSES to 0.694) → the FF-rule swap is a **real mechanism difference**, not an averaging artifact.

⇒ **The depth-2 accuracy is carried by the FEEDFORWARD rule** — the Urbanczik-Senn **M2.6 somatic-rate difference** `dw = η·acts[k]ᵀ·(φ′(E)·v_api)` descending the clean apical error `v_api = e_upper@Y` via fixed-random feedback, i.e. **clean-error feedback alignment**. The interneuron cancellation is the **closed-form realization of that clean CHANNEL** (in the numpy reference the interneuron is held at the self-predicting fixed point `W_PI = −Y`, so `v_api` IS the cancelled residual) and is load-bearing on the substrate for the **burst READOUT** (Stage-A′′′), NOT for the numpy weight update.

## The other corrections (all folded in from the panel + the control)
1. **Fair like-for-like = Burstprop 0.788 vs microcircuit 0.964 (margin +0.176) at MATCHED ep=600/lr=0.3.** NOT "0.66 vs 0.96": the committed D1 Burstprop JSON is **ep=300** (0.664, cupy) while the microcircuit JSON is **ep=600** (0.964, numpy) — a cross-config gap that over-states the margin. (This is on top of the batch confound the control caught.)
2. **On-bridge scope is the BURST READOUT only.** Per the `sim/` diff + config comment, the M2.6 FF weight update lives runner-side and is untouched by `cp_bdsp_int_drive`; the component-wise-validated on-bridge cancellation is the credit-channel/readout, NOT the FF plasticity that produces the accuracy.
3. **`cancellation_lowers_burst=False`** (B_cancelled 0.1536 > B_uncancelled 0.1499; E rose to offset B). The load-bearing gated signal is **P → p0**, not B — corrected in the runner comment + gate.
4. **The shipped JSON `rule_microcircuit` string was FALSE** (it quoted the Burstprop formula `dw = η·acts[k]ᵀ·(B − Pbar·E)`). Corrected in the runner source + the JSON regenerated; the result is no longer branded "burst-multiplexed Burst-Dependent Plasticity" (its learning uses no burst quantity).
5. **The 0.964 is the NUMPY REFERENCE** of the `sim/` rule on a depth-2 XOR MLP; only the Stage-A/A′/A′′/A′′′ component checks touch a real `SimulationBridge`; the fully-on-bridge 384-width spiking multi-seed net remains the controller's GPU run (not yet demonstrated).
6. **EMERGE-5c propagation risk (flagged):** EMERGE-5c's verdict "ACTIVE CANCELLATION is MORE noise-robust than raw burst-rate estimation" carries the same attribution error (its own HONEST_NOTE says interneuron maintenance "does not affect the within-step FF update"). The noise-robustness is the clean-error credit CHANNEL, not active cancellation. (Addendum appended to `2026-07-02-emerge5c-microcircuit-noise-robust-GO.md`.)

## What this establishes toward the deep lever (honestly)
Two biological deep-credit rules are now BUILT on the spiking substrate (`enable_bdsp` Burstprop + `enable_bdsp_microcircuit` interneuron-cancellation; both additive/default-off/byte-identical-when-off; the interneuron cancellation validated on-bridge for the burst readout). At depth-2 (numpy reference), the **clean-error credit channel** (M2.6 somatic-rate FF = feedback alignment on the clean apical error) clears the bar (0.964) and is batch-robust; raw Burstprop's noisy burst-fraction credit is worse and fragile — reproducing EMERGE-5c's ordering, with the attribution corrected to the credit CHANNEL. **The genuinely-open question D2 tests:** is the on-substrate interneuron cancellation CAUSALLY load-bearing for accuracy at DEPTH (depth-3, where a point-neuron layer cannot carry a clean continuous error without the physical cancellation, and where the FA depth wall lives)? At depth-2 clean-error FA suffices, so the interneuron's accuracy role is not yet demonstrated — that is the D2 on-bridge depth-3 test (spec: `docs/plans/2026-07-07-D2-depth-stability-spiking-build-spec.md`, with a plain-FA baseline arm). NO expensive training touched.

## Discipline note
This finding is *more accurate than the clean-looking builder GO*: my own control probe caught a batch-mismatch confound, and the mandatory 4-lens adversarial-verify (`wjn6hxyuu`) caught the mechanism-attribution overclaim (interneuron-cancellation vs clean-error-FA) + an epoch confound + a false shipped-JSON string — all corrected BEFORE commit. Two overclaims and a documentation defect caught pre-commit is the discipline working.

## Files
`research/runners/_gnw_d1_spiking_bdsp_derisk.py` (`--rule microcircuit`; corrected strings); `research/runners/_gnw_d1_microcircuit_control_probe.py`; `sim/{config,bridge}.py` (the additive `enable_bdsp_microcircuit` path); `research/findings/raw/_gnw_d1_microcircuit_128.json` (regenerated), `_gnw_d1_ctl_seed{42,43,44}.json`. Spec: `docs/plans/2026-07-07-D1-spiking-bdsp-build-spec.md`, `-D2-depth-stability-spiking-build-spec.md`. Prior: `2026-07-07-D1-spiking-bdsp-burstprop-mechanism-ports-accuracy-noise-limited.md`, `2026-07-02-emerge5c-microcircuit-noise-robust-GO.md`.
