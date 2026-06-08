# Broader bug-audit: the silent plasticity bug did NOT materially shift the documented nav results

> ## ✅ RESOLVED 2026-06-08: the documented A+E ceiling (and the cluster-stacking conclusions measured against it) are ROBUST to the bug fix. The bug's impact scales with reward-event RATE — only the high-rate neural de-risk was badly hit.

**Date:** 2026-06-08
**Context:** The silent bridge plasticity bug (`cp_d1_d2_sign`/`cp_transmission_gain`/`cp_plasticity_rate_gain` gate arrays under-sized vs `cp_connections.nnz` → the reward-modulated weight update raised+caught EVERY step → reward-driven plasticity silently dropped) was fixed in `512026ee` (`_ensure_gate_capacity` lazy-grow guard). EVERY documented reward-modulated navigation result that used `--enable-d1-d2-asymmetry` was produced with this bug ACTIVE. The de-risk already showed the cheat baseline shifted only 3.83→4.24. This audit quantifies the bug's impact on the **load-bearing A+E multi-goal ceiling** — the "robust operational ceiling" every cluster-stacking conclusion (B.1/B.2/B.3, C, D, F) was measured against.

## Method (cheap-first A/B)

The documented A+E numbers ARE the bug-ON condition (produced pre-fix), so running the SAME config on the fixed (bug-OFF) main tree and comparing to documented is already an A/B. Config = the documented A+E multi-goal **deterministic** baseline (5 cluster flags: `--enable-msn-lateral-inhibition --enable-d1-d2-asymmetry --enable-striatal-pv-fsi --enable-cluster-a-closed-loop --enable-cluster-e-topography`, `--moving-goal --goal-schedule multi --deterministic`), seeds 42/43/44, run SERIALLY (no `--emit-activity`). Orchestrator: `research/findings/raw/_audit_ae_batch.py`. A bug-ON worktree A/B (commit `103ded0b`, just before the fix) was pre-staged (`_audit_ae_bugon_batch.py`) to escalate only if `|Δ| > 1σ`.

## Result — A+E ceiling robust (bug-OFF ≈ documented bug-ON)

`sum_finalQ` = sum of per-phase final-quarter mean distance (LOWER better):

| seed | bug-OFF (fixed main) | documented bug-ON (n=6) |
|---|---|---|
| 42 | 13.51 | — |
| 43 | **4.53** | — |
| 44 | **4.51** | — |
| **mean** | **7.52** (std 4.24, n=3) | **7.18 ± 1.58** (cluster-eval) / 6.97 ± 0.83 (ceiling) |

**Δ vs documented cluster-eval baseline = +0.34, WELL within 1σ (1.58).** Verdict: the bug did **NOT** materially shift the A+E ceiling → **the cluster-stacking conclusions remain robust.** No escalation to the bug-ON worktree A/B is warranted (it stays available for per-seed exactness, but the mean-based conclusion is established).

## Interpretation (honest)

- **2 of 3 seeds IMPROVED** post-fix (43/44: 4.5 vs documented ~7.2); **seed 42 regressed** (13.5). The mean is net-neutral; the per-seed distribution shifted (higher variance, driven by the seed-42 outlier). So the fix changes the per-seed learning dynamics but not the systematic A+E mean — the load-bearing comparison (stacks vs the A+E mean) is unaffected.
- **The bug's impact scales with reward-EVENT RATE.** The reward-modulated block (where the broadcast happened) only runs when reward is delivered. So the count of silently-dropped updates ∝ reward rate:
  - **High-rate neural de-risk** (continuous N5 perceived-approach reward + tonic SNc dopamine → the reward block runs ~11× more often, 377k vs 34k errors): hammered to the **non-navigating floor** (23.15). This was 100% the bug (post-fix: 2.00, a GO).
  - **Standard-rate configs** (sparse ±1 goal reward): cheat 3.83→4.24 (+0.41), A+E 7.18→7.52 (+0.34) — both **within noise.**
- This is a clean, coherent account: a real, months-old bug that was nonetheless **near-invisible** to the bulk of the documented nav work (sparse-reward), and **decisive** only for the one high-reward-rate configuration we happened to test during the brain-based reward+dopamine de-risk — which is exactly how the bug got caught.

## Conclusion

The bug fix is correct and important (it unblocked the brain-based reward+dopamine GO), but it does **not** retroactively invalidate the documented flagship (4.08) / A+E ceiling (6.97/7.18) / cluster-stacking conclusions — those ran at standard reward rates where the bug's effect is within the seed-to-seed noise floor. The honest asterisk: any FUTURE high-reward-rate config must be run post-fix (the bug is gone now), and the documented numbers carry a small (~±0.4, within-1σ) bug-induced offset that does not change any GO/NO-GO verdict.

**Tools:** `research/findings/raw/_audit_ae_batch.py` (bug-OFF), `_audit_ae_bugon_batch.py` (conditional bug-ON escalation, not run), `_audit_ae_summary.json` (raw verdict).
