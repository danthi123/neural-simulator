# R-iii on-substrate SURPASS (GO, 6-seed, 4 controls): the two-compartment dendritic dAP plateau COMPLETES a partial CA3 cue where the LINEAR point-neuron cannot — held-out completion 0.571 vs LINEAR 0.007, riding the RIGHT ensemble structure (scramble collapses), on the real spiking substrate, NO `sim/` edit. Realized via the research-gated Rung-0/1 (calibrate k_thresh to the per-step coincident drive + raise apical_R into the thin-high-R regime + two-compartment on) — reuse of the existing `enable_two_compartment_dap` machinery. This surpasses the CYCLE-1067 boundary and is CONSISTENT with it (the point-neuron/linear read-out still fails; the DENDRITIC read-out is what rescues it — exactly the CYCLE-1065 minimal-model prediction, now on-substrate).

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_onsubstrate_readout_test.py` (installed-attractor read-out; `--two-comp --apical-R --apical-gc --k-thresh --scramble`). GPU (cupy). NO `sim/` edit (config flags + public-array writes only).
**Verdict:** GO, 6-seed (dev 42/43/44 + blind 100/101/102), adversarially verified (4 controls). Honest scope: this is the READ-OUT surpass on a HAND-INSTALLED attractor; emergent attractor FORMATION is the separate open problem (CYCLE 1066).

## The result — 6-seed, k=6 FROZEN from seed-42 dev
```
seed   PLATEAU  LINEAR  FLAT   SCRAMBLE  non-ens   (c_drive held/non)
 42     0.826   0.014   0.030   0.158    0.044      80.9 / 7.5
 43     0.640   0.000   0.044   0.082    0.099      76.3 / 7.4
 44     0.716   0.030   0.076   0.262    0.145      74.9 / 7.4
100*    0.436   0.000   0.020   0.077    0.056      74.8 / 7.4
101*    0.353   0.000   0.131   0.075    0.075      71.4 / 7.4
102*    0.458   0.000   0.072   0.072    0.097      71.9 / 7.5
AGG     0.571   0.007   0.045   ~0.13    0.086                       (*=blind)
```
The dendritic dAP completes the held-out (non-cued) ensemble members at 0.35-0.83 where the linear point-neuron reaches ~0 — a partial cue reactivates the rest of the stored pattern, the Marr autoassociator function.

## The four adversarial controls (all pass 6-seed — the refutation lenses, built into every run)
1. **LINEAR (coincidence OFF, SAME installed attractor):** 0.007 avg << PLATEAU 0.571 → the dendritic non-linearity is LOAD-BEARING, not more inputs. The point-neuron limit (CYCLE 1067) STANDS; the dendrite is what surpasses it.
2. **SCRAMBLE (same W_HIGH budget on RANDOM ca3->ca3 synapses, wrong structure):** ~0.13 avg, PLATEAU exceeds it by ≥0.22 EVERY seed → completion rides the RIGHT ensemble structure, not merely the presence of strong synapses. (The gold-standard specificity control, stronger than FLAT.)
3. **FLAT (no attractor, all W_LOW):** 0.045 avg << PLATEAU → completion needs the installed attractor.
4. **SPECIFICITY (non-ensemble neurons):** 0.086 avg << PLATEAU → only ensemble members complete; the plateau does not fire indiscriminately.
Plus: **blind seeds 100/101/102 pass at k=6 frozen from seed-42 dev** → not p-hacked; and the trigger window (k=4/5/6 all complete with high held / low non) is robust, not a single-k knife-edge.

## The mechanism (research-gated, reuse-only)
The research gate (`2026-07-08-riii-two-compartment-dap-completion-research-gate.md`) established: the sim's `enable_two_compartment_dap` apical compartment carries a genuine Mg-regenerative NMDA plateau (`fused_coincidence_plateau` on `cp_v_apical`), triggered by a count/weight threshold (Bouhadjar-Diesmann dAP `θ_dAP`; matches CYCLE-1065's abstracted plateau). CYCLE 1067 failed because it ran the DEFAULT `apical_R=0.15` + an uncalibrated `k_thresh=40` that NEVER triggered (per-step coincident drive is ~6-7, a fraction of the 75-81 max because few cue neurons fire per step). The Rung-0/1 fix, NO `sim/` edit:
- **Rung 0 — calibrate `k_thresh` to the PER-STEP coincident drive.** The `_cdrive` probe reports held c_drive ~75-81 (max) vs non ~7.4; the per-step is ~1/10 of max, so k=6 is the window: held-out members (within-ensemble w_high) cross it, non-members (cross w_low) do not.
- **Rung 1 — thin-high-R apical (`apical_R` 0.15 → 50) + two-comp.** The high local input resistance (Humphries-Mellor R²≈0.80 input-resistance↔threshold; a thin dendrite gives large local ΔV per unit plateau current) lets the ~5 clustered recurrent inputs ignite the NMDA plateau on the apical, whose depolarization drives the soma to complete.
The forward-`I_dAP` `sim/` change (research-gate Rung 2) is NOT needed — the passive-coupling ceiling clears the soma once the apical ignites in the thin-high-R regime.

## Consistency with CYCLE 1067 (NOT a flip-flop)
CYCLE 1067 ("point-neuron CA3 doesn't complete even a hand-installed attractor; plateau-as-wired doesn't rescue it") was ACCURATE for the DEFAULT config it ran, and it correctly prescribed "research-gate the coupling regime, do NOT hack configs further." The gate found the regime; this is the workflow succeeding. The point-neuron/LINEAR read-out STILL fails here (0.007) — the boundary is real; the DENDRITIC read-out is the surpass.

## Honest scope + what remains
- This is the READ-OUT surpass: the dendritic dAP completes a HAND-INSTALLED clean attractor (a labeled teaching scaffold). It validates that the on-substrate dendritic non-linearity CAN do Marr completion — the thing CYCLE 1065 predicted in the minimal model and CYCLE 1067 showed the point neuron cannot.
- **The separate open problem: EMERGENT attractor FORMATION** — CYCLE 1066 showed the ca3->ca3 rate-Hebbian does not potentiate a specific within-ensemble attractor (held c_drive ≈ non). The completion mechanism now works GIVEN an attractor; forming the attractor from experience (the right recurrent plasticity: symmetric co-activity / rate-Hebbian with within-ensemble specificity) is the next arc.
- Residuals: the trigger window is narrow (per-step drive ~6-7, so k must be ~4-6) — robust across seeds at k=6 but reflects sparse per-step cue firing; GPU run-to-run variance ~±0.06 (no CUDA-determinism env set), all values clear the gate comfortably. `apical_R=50` is a tuned config (biologically the thin-high-R dendrite), not a `sim/` edit.

## NEXT
(1) The emergent-attractor-formation arc: get the ca3->ca3 recurrents to potentiate a specific within-ensemble attractor from experience (rate-Hebbian symmetric co-activity; the CYCLE-1066 residual), then re-run THIS completion read-out on the LEARNED attractor. (2) Then the SWR generative-replay loop (R-iii's original goal) rides the now-working completion. (3) Optional: widen the trigger window (stronger/synchronous cue drive) for robustness.

## Files
`research/runners/_riii_onsubstrate_readout_test.py`. Prior: `2026-07-08-riii-two-compartment-dap-completion-research-gate.md` (the gate), `-onsubstrate-point-neuron-completion-limit-decisive-with-installed-attractor.md` (1067), `-dendritic-completion-surpass-cheap-first-GO.md` (1065 minimal model), `-onsubstrate-coincidence-wired-but-blocked-by-missing-attractor.md` (1066). Biology: Kandel 6e Ch 13 pp 297-298 (NMDA spike, plateau→soma), Humphries-Mellor PMC7614718 (branch input-resistance↔threshold), Bouhadjar-Diesmann 2022 (dAP). Catalog G.02 (active dendrites), D.13/D.05 (CA3 completion).
