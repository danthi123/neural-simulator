# Loop-step 3 RF + DISTILLATION synthesis = GO (0.872): distilling clip-aware weights THROUGH the RF complex accumulator (no `g·(V−E)`) recovers the per-layer clip loss AND HOLDS on the live RF install — the cheap ladder ENDS WITH A WIN, the multi-week `sim/` edit is NOT needed for the consolidation read (2026-06-22)

**One-line verdict:** `rf_distill: narrow512 blocks=3 RF-faithful-clip-aware-distillation INSTALLED-on-live-RF-bridge cumulative_analog_spearman_vs_teacher=0.872 (best arm=armA_unit; trainer-offline=0.872; RF-verbatim-install=0.556) per_block=[0.982, 0.898, 0.872] specificity_margin=0.442 shuffled_control=0.607 -> GO (vs RF-verbatim 0.556 + graded-distill-install 0.444; GO bar 0.80)`

**Scope:** the LAST cheap shot before the multi-week differentiable-bridge `sim/` edit — the synthesis of the two best partials. `research/runners/_genseq_loopstep3_rf_distill_derisk.py`, GPU (`SIM_BACKEND=cupy`). **NO `sim/` edit** (`git diff --stat -- 'sim/*'` EMPTY; the RF path + the clip-aware trainer both already exist, reuse-by-import). On `main`.

## The idea (synthesis of the two best partials)
- **RF-PARTIAL** (`2026-06-22-genseq-loopstep3-rf-PARTIAL-best-cheap.md`): the RF complex accumulator computes `Re(Z)=nsteps·(a@W)` EXACTLY (rank 1.000), with **NO clip, NO `g·(V−E)`, NO ceiling**. Its residual (cumulative 0.556) is ENTIRELY each layer's `a_hat=clip(Re(Z)/scale,0,1)` readout — the per-layer CLIP compresses (W1), even though the linear matvec is rank-faithful (W2 escaped).
- **distill-NEGATIVE** (`2026-06-22-genseq-loopstep3-distill-NEGATIVE-live-bridge-gap.md`): clip-aware distillation recovered the clip loss OFFLINE (0.815) but LOST it on the GRADED install — the live `g·(V−E)` driving-force divergence dragged it 0.815 → 0.444. The `[VERIFY]` `g·(V−E)` gap was the confirmed load-bearing killer.
- **⇒ the RF path has NO `g·(V−E)`** — the EXACT killer of the distillation install. So distil clip-aware weights THROUGH the RF-faithful forward, then install on the REAL RF bridge → the offline recovery SHOULD HOLD (no conductance divergence). **It does.**

## The load-bearing equivalence (why the trainer forward IS the RF install forward)
The RF accumulator gives `signed = Re(Z)/nsteps = a@W` exactly. The per-layer readout is `a_hat = clip(signed·scale,0,1) = clip(a @ (W·scale), 0, 1)` — the per-block scale folds INTO the weight. So training `W'` (scale absorbed) under the PURE clip forward `clip(a@W',0,1)` is EXACTLY training the RF readout chain — with NO conductance term. On the RF install at **unit scale**, the read is `clip(Re(Z)/nsteps,0,1) = clip(a@W',0,1)` = the trainer forward verbatim. No divergence.

## Result — the offline recovery HOLDS on the live RF install (the [VERIFY])
| Path | per-block [L0, L1, L2] vs teacher | cumulative |
|---|---|---|
| RF-verbatim install (baseline) | [0.934, 0.675, 0.556] | **0.556** (reproduced exactly) |
| RF-faithful clip trainer — **OFFLINE** | [0.982, 0.898, 0.872] | **0.872** ✅ math works |
| RF-faithful clip trainer — **INSTALLED (live RF, unit scale, ARM A)** | [0.982, 0.898, 0.872] | **0.872** ✅ **HOLDS — byte-identical to offline** |
| trained weights — INSTALLED (live RF, *calibrated* scale, ARM B) | [0.934, 0.673, 0.591] | 0.591 (scale control, see below) |
| SHUFFLED-target control — INSTALLED vs REAL teacher | [0.766, 0.706, 0.607] | 0.607 (below real by 0.265) |

**The decisive contrast with the graded distill:** that one's offline 0.815 → installed 0.444 (the `g·(V−E)` erased the gains). Here offline 0.872 → installed **0.872** (the RF path has no `g·(V−E)`, so the trained weights install faithfully). The `[VERIFY]` claim is confirmed: **the RF-faithful trainer's weights HOLD on the live RF install.**

## Anti-cheats — both pass cleanly
- **Specificity margin = 0.442** (matched 0.872 vs mismatched 0.430), strongly re-opened (the graded-distill's was 0.065). The trained weights compute each char's SPECIFIC mapping.
- **Shuffled-target control = 0.607 < real 0.872 by 0.265** (> the 0.2 bar). The shuffled trainer's offline-vs-REAL-teacher is `[0.138, 0.411, 0.527]`: its early blocks correctly DIVERGE from the real teacher (L0 0.138 = the wrong char→activation map), and only the final block floats up to 0.527 — the SAME char-correlated-final-reps confound the rf-PARTIAL documented at 0.373 / the distill-NEGATIVE at 0.542 (a property of the teacher's final reps, NOT a leak). The per-block specificity margin (0.442) is the cleaner discriminator and confirms genuine recovery.

## Honest note — ARM A (unit) vs ARM B (calibrated)
ARM B (0.591) re-runs the RF probe's *occupancy-only* per-block scale calibration on the trained weights; it targets a clipped-mean ~0.18 and so **re-clips away** the trained weights' carefully-fitted occupancy, collapsing back toward verbatim. ARM A (unit scale, gain folded into `W'`) is the correct install — the trainer forward IS the install forward. This is exactly the RF-faithful equivalence the synthesis rests on; ARM B is reported as the scale control (rank is scale-invariant on the RF accumulator, so the *rank* still transfers, but the calibration's occupancy target undoes the fit).

## Verdict + what it routes to
**GO.** Installed-on-live-RF-bridge cumulative **0.872 ≥ 0.8**, specificity margin re-opens (0.442 > 0.1), shuffled-control below real (0.607, real − shuffled = 0.265 > 0.2). The cheap ladder — spike-rate NEG(0.009) → graded PARTIAL(0.327) → pop-code NO-OP → distill NEG-on-live(0.444) → RF PARTIAL(0.556) → **RF+distill GO(0.872)** — ENDS WITH A WIN. The multi-week differentiable-bridge `sim/` edit is **NOT needed** for the consolidation per-layer read: distilling clip-aware weights through the RF-faithful forward + installing on the (conductance-free) RF complex-synapse path recovers the per-layer clip loss and holds on the live install. The generator-as-RF-PHASOR consolidation read is the substrate-native escape.

**Honest scope:** validated at the narrow-512 3-block dense MLP slice + the per-layer analog-Spearman-vs-teacher metric (identical to all the loop-step-3 NEGATIVEs). The teacher's final-block reps are char-correlated (the shuffled control's 0.607 floor), so the per-block specificity margin — not the cumulative alone — is the load-bearing discriminator; both clear their bars. Full-width + the end-task (next-token) head are the named follow-ons.

NO `sim/` edit; not committed. Raw: `research/findings/raw/_genseq_loopstep3_rf_distill.json`. Runner: `research/runners/_genseq_loopstep3_rf_distill_derisk.py`.
