---
type: finding
status: contributing
date: 2026-08-14
mechanism: b1-v1-orientation-selforg-onbridge
artifacts:
  - research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_cpu_3seed_fsON.json
  - research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_cpu_3seed_fsOFF.json
  - research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_meansub_diag_s42.json
---

# B1 on-bridge V1 orientation self-org, operating-point-first re-test: the prior 6-seed NEGATIVE was NOT dead-forward — V1 is active-sparse THROUGHOUT development, and the rule hits a genuine COMMON-MODE BOUNDARY (ON/OFF converge, signed RF cancels), robust to fixed FS inhibition and subtractive normalization

<!--derived-->
**One-line verdict.** With the developmental V1 firing fraction now MEASURED (the instrument every prior on-bridge run lacked), the operating point is verified **active-sparse** — dev firing fraction 0.0086 (FS-on) / 0.0096 (FS-off), all 3 seeds in the [0.005, 0.05] band, RISING toward the 0.012 target across the window, no collapse — so the re-test is **op-point-verified, NOT a dead-forward VOID**.
Yet orientation selectivity does not emerge: `osi_post_frac` 0.0019 (FS-on) / 0.0026 (FS-off) vs pre-random 0.0028 and shuffle 0.0026–0.0029 — no lift over either control, on 0/3 seeds. Verdict: **BOUNDARY** (active-sparse but OSI « 0.5), with a precise, banked next mechanism.

**Artifacts.** `research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_cpu_3seed_fsON.json` (FS-on 3-seed), `research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_cpu_3seed_fsOFF.json` (FS-off 3-seed), `research/findings/raw/lanes/perception/b1_v1_selforg_oppoint_meansub_diag_s42.json` (Miller-MacKay mean-subtract 1-seed diagnostic). CPU numpy, `_b1_v1_selforg_onbridge_derisk` with the new developmental firing-fraction probe; no `sim/` edit.

## The instrument refutes the spec's own dead-forward hypothesis (honestly)

<!--derived-->
This rung was posed as a VOID-check: the hypothesis was that the 2026-07-30 on-bridge 6-seed NEGATIVE (`_b1_v1_selforg_onbridge_derisk`, 6/6 NEGATIVE, `v1_firing_rate_mean=0.0008`) was a DEAD-FORWARD artifact — V1 silent during the plastic phase, the rule never exercised — the exact lens `2026-08-02-laneD-...-DEAD-FORWARD-artifact` established (retina fires, `cortex_v1_simple=0` during training ⇒ trace-rule verdict VOID).
The new developmental probe **refutes that for the developmental phase**: V1 fires at ~0.009–0.010 (active-sparse) throughout development at BOTH reduced (n_v1=2048) and production (n_v1=8192, 24k steps: 0.0099, thirds [0.0081, 0.0107, 0.0109]) scale. The prior "0.0008 silent" was the TEST-read-out rate under sparse-bar drive — a weaker, different input — never the developmental rate under full-field gratings. So the prior NEGATIVE is not a silent artifact; it is a genuine BOUNDARY that the missing instrument had merely mis-attributed.

## The diagnosed failure mode: COMMON-MODE CONVERGENCE (not collapse, not silence)

<!--derived-->
The runner's raw-weight diagnostic (added earlier in the arc, never decisive until now) reads **COMMON-MODE CONVERGENCE** on all 6 seeds: incoming L2 norm is healthy (l2_mean ~506–544, `frac_cells_l2_near_zero=0.000`, `frac_cells_all_zero` 0.02–0.13 — NOT the 0.735 all-zero of the old `b1_saturation_test`, so weight collapse is config-specific and absent here), while ON and OFF channels potentiate to nearly identical values (mean-sub diagnostic: on_mean 2.268 vs off_mean 2.2725, `on_minus_off_mean=-0.0045`).
A full-field grating drives ON at bright phase and OFF at dark phase; averaged over random orientation/phase, every ON and OFF synapse sees equal co-activation, so the potentiation-only rate-Hebbian rule drives both to the same weight and the SIGNED ON−OFF receptive field cancels ⇒ OSI ≈ 0. The rule is fully exercised; it simply has no opponency to break.

## Two cheap no-sim-edit levers do NOT remove the common mode

<!--derived-->
| arm | dev firing frac (op point) | osi_post_frac | osi_pre | osi_shuf | diagnosis |
|---|---|---|---|---|---|
| FS-on (n_inh=64, SAILnet fixed lateral inhibition) | 0.0086 ✓ | 0.0019 | 0.0028 | 0.0026 | common-mode |
| FS-off (n_inh=0) | 0.0096 ✓ | 0.0026 | 0.0028 | 0.0029 | common-mode |
| mean-subtract (Miller-MacKay `HEBB_MEAN_SUB=1`, 1 seed) | 0.0090 ✓ | 0.0029 | 0.0044 | 0.0024 | common-mode |

<!--derived-->
A fixed FS-interneuron pool (the SAILnet/Foldiák competition ingredient the prior runs had OFF) does not help — it provides uniform gain control, not per-pair decorrelation, and marginally lowered OSI while silencing more cells (`frac_cells_all_zero` 0.11–0.13 vs 0.02–0.03 off). Subtractive normalization normalizes the postsynaptic total but does not create ON/OFF opponency. The common mode is robust to both.

## Anti-cheats (all held)

<!--derived-->
Isotropic RF support (all ON+OFF within radius-4, carries no orientation) — any oriented RF must be learned. Host Gabor bank never applied to the pathway (random-init then learned; host used only as the RSA scoring reference). No-learning control: pre-random `osi_frac` ~0.003 ≈ 0. Shuffle-stimulus control does NOT raise OSI (0.0026–0.0029 ≈ pre ≈ post).
NEW operating-point precondition: developmental firing measured, active-sparse ⇒ not VOID. Determinism: `cfg.seed=cfg.ou_seed=cfg.heterogeneity_seed=seed` (per-seed OSI/l2 differ; the tightly homeostatically-pinned rate coincides). OSI/RSA label-free. (RSA-to-host 0.70–0.78 is informational at reduced scale, non-discriminating per the numpy de-risk; the discriminating metric is OSI, where all controls collapse together.)

## Banked next mechanism (named, not pursued here)

<!--derived-->
The residual is now specific: the feedforward rate-Hebbian rule, on ON/OFF-split full-field input, learns a common mode with no opponency. **The named next mechanism is LEARNED (plastic) anti-Hebbian recurrent inhibition — the SAILnet inhibitory learning rule (Foldiák), which decorrelates units so neighbors cannot all latch the same common-mode blob** — not the fixed FS pool tried here. This needs a check on whether the substrate supports inhibitory-pathway plasticity (a possible `sim/` edit, flagged THEN, not now).
Complementary, and cheaper to try first: an **ON/OFF-opponent input front-end (retinal/LGN center-surround whitening)** so the input itself carries the opponency the feedforward rule can bind — the fixed FS pool and subtractive norm demonstrably do not supply it. SAILnet's own GO used whitened patches + plastic lateral inhibition together; this rung shows neither ingredient is substitutable by fixed inhibition alone.

## Sources

<!--derived-->
SAILnet: Zylberberg, Murphy & DeWeese 2011, PLoS Comput Biol 7(10):e1002250 (homeostatic threshold + plastic anti-Hebbian lateral inhibition ⇒ oriented Gabor RFs). Divisive normalization: Carandini & Heeger 2012, Nat Rev Neurosci 13:51. Homeostasis: Turrigiano 2008, Cell 135:422. Re-examines `2026-07-30-density1-destroys-learning-structure-and-laneD-6seed-NEGATIVE.md` through the `2026-08-02-laneD-perception-trace-rule-negative-is-a-DEAD-FORWARD-artifact-...` VOID lens; numpy ceiling `2026-06-21-B1-v1-gabor-selforg-derisk.md`.
