---
type: finding
status: partial
lane: e-mouth-fluency
date: 2026-08-28
mechanism: post-stale-COO-cache-fix tuning of the WKV mouth's e-prop batched-substrate readout (more training-position
  coverage + more epochs) against the strict 6/6 sub_recov_ratio>=0.85 GO bar; diagnosis + a cheap numpy lever-search
  (honestly inconclusive, regime mismatch) + a production-scale (B=48) 6-seed confirmation QUEUED, not yet landed.
seeds: [42]
runner: research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
artifacts:
  - research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json
  - research/findings/raw/_mouth_readout_tuning/numpy_cheap_search_baseline_b2.json
  - research/findings/raw/_mouth_readout_tuning/numpy_cheap_search_epochs6x_b2.json
  - research/findings/raw/_mouth_readout_tuning/numpy_cheap_search_coverage3x_b2.json
seed-waiver: the cheap numpy lever-search below is a single-seed (42), tiny-batch (B=2) DIRECTIONAL probe only,
  explicitly reported as inconclusive due to a regime mismatch (see §2) -- it is not a generalization claim. The
  decisive claim is the queued production 6-seed run in §3, not yet landed at write time.
---

# Mouth eprop readout tuning residual (3/6 -> strict 6/6): diagnosis + a coverage/epochs lever, QUEUED at production scale, not yet landed

## 0. Assignment and scope

The 2026-08-28 full-scale confirmation of the stale-weight-cache training fix
(`2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md`, artifact
`research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json`, B=48, 6 seeds) put the
learned spiking read-out's `sub_recov_ratio_learned_over_copied` at mean 0.8686 (min 0.8399), **go_count 3/6** against
the strict per-seed bar `ratio >= 0.85`, and named this "a tuning residual, not a wall." This finding (1) diagnoses the
residual, (2) reports a cheap off-GPU lever-search (honest about its limits), (3) names + queues the production-scale
lever, and (4) reports the current (staged, not landed) state honestly per the PARTIAL discipline.

## 1. Diagnosis (before touching any lever)

Per-seed `sub_recov_ratio_learned_over_copied` from the cited artifact: seed 42=0.8796(GO) 43=0.8399(NO) 44=0.8845(GO)
100=0.8446(NO) 101=0.8497(NO) 102=0.9132(GO). **This is not bimodal** -- all six seeds sit in a tight 0.84-0.91 band
straddling the 0.85 line; the three misses fall short by 0.4-6.0 points, not catastrophically. `weight_cosine_to_head_diag`
is even tighter across every seed (0.345-0.366) regardless of GO/NO-GO outcome, i.e. the LEARNED direction's quality is <!--derived-->
essentially seed-invariant -- the residual reads as a shared, systematic shortfall (some combination of data exposure
and substrate read noise), not a seed-specific pathology. `w_hat_norm` converges to ~23 in every seed, well below both
`w_target=40` (the synaptic-scaling cap) and `head_w_norm=37.5` -- an equilibrium between the local delta rule's
potentiation and weight decay, not a runaway.

**Prior-art check (`before_you_build.sh` + `git log --all --grep`) surfaced two exclusions that need a caveat.**
`2026-08-19-mouth-substrate-forward-40k-coverage-EXCLUDED-real-credit-limit.md` concluded 5x more training positions
does NOT move substrate-forward recovery off a ~0.34-0.37 plateau, and a companion finding excluded the read-window
lever the same session. **Both predate the 2026-08-27 stale-weight-cache training fix** (`mark_weights_edited()`,
commits d6c375de5/22c05f41a): the exact `BatchedSubstrateReadout.set_weights` code path those 2026-08-19 runs exercised
had no cache-invalidation call, so their per-step substrate forward was FROZEN after the very first gradient step
(the bug `2026-08-27-mouth-stale-coo-training-fix-PARTIAL.md` fixed for the training loop specifically). Under a frozen
forward, more epochs or more positions provably cannot matter -- every step after the first computes its error against
the SAME stale margin. This fully explains why 5x coverage measured NO effect pre-fix, and means that null result does
**not** bind post-fix. This is flagged here, not re-litigated in full (the broader read-SNR/dendritic arc that took the
~0.34-0.37 plateau as its premise is a separate, larger audit, out of this task's scope).

**Independent support that more data-exposure should help now.** The original 2026-08-14 host-linear-proxy-forward GO
(`2026-08-14-fluid-mouth-readout-eprop-learned-GO.md`) used **40k training positions and 30 epochs**, reaching
`hostlin recov 0.93`, `wcos 0.51`. The post-fix substrate-forward confirmation used only **8000 positions and 10
epochs** (a GPU-budget economy, not a finding that more data doesn't help post-fix) and reaches `hostlin recov 0.82`,
`wcos 0.36` -- both lower, in the direction that finding's OWN measured coverage curve predicts (`~200 pos -> 0.07 host-linear
recov, 3.3k -> 0.85, 30k -> 0.95`; 8000 sits partway up that curve).

## 2. Cheap off-GPU lever-search (numpy backend, honestly inconclusive)

Per the ONE-GPU rule, a tiny-scale (B=2, seed 42 only, `sub-read-window 30`, `read-window 40`, `n-sub-demo 6`) numpy
run compared, at MATCHED total gradient-step budget, epochs (repeat the same positions) vs coverage (more distinct
positions):

| variant | n-train-pos | epochs | n_grad_steps | hostlin_recov | wcos (floor) | sub_learned recov | ratio | w_hat_norm |
|---|---|---|---|---|---|---|---|---|
| baseline | 128 | 2 | 128 | 0.3121 | 0.0637 (0.0023) | 0.1810 | 0.2118 | 40.00 (pinned) |
| epochs6x (repeat) | 128 | 6 | 384 | 0.2560 | 0.0626 (0.0023) | 0.4529 | 0.5300 | 40.00 (pinned) |
| coverage3x (distinct) | 384 | 2 | 384 | 0.2417 | 0.0727 (0.0051) | 0.1197 | 0.1401 | 40.00 (pinned) |

**Verdict: inconclusive, and the reason is itself a real diagnostic.** `w_hat_norm` is pinned at `w_target=40` (the cap)
in ALL THREE conditions here, unlike the production B=48 run where it converges to an equilibrium (~23) well below the
cap. At B=2 the per-step gradient error estimate averages over only 2 substrate reads, so the read noise dominates and
the local rule cannot settle into the same converged regime production reaches -- it is in a qualitatively DIFFERENT
(noise-dominated, cap-pinned) operating point. The `wcos` numbers (the most statistically robust metric here, computed
over the full weight matrix rather than a 6-position eval sample) are correspondingly weak and close together across
all three variants (0.063-0.073, all near their own floor ~0.002-0.005 -- a 12-30x separation vs production's ~35-350x), <!--derived-->
too noisy to license a directional call; the `sub_learned`/`ratio` numbers, evaluated over only 6 held-out demo
positions, are dominated by sampling noise (epochs6x's apparent lift and coverage3x's apparent drop both plausibly
reflect which 6 eval positions happened to land well, not a real epochs>coverage effect). **Conclusion: a cheap
small-batch numpy search cannot answer this specific question, because batch size itself controls whether the local
rule is in the converged or noise-dominated regime, and B is exactly the axis this cheap-search recipe must shrink to
stay off the GPU.** This is recorded so a future session does not re-attempt a tiny-B numpy search for this residual.

## 3. The lever chosen + queued (production scale, B=48, 6 seeds) -- NOT YET LANDED

Given §1's grounding (the project's OWN 2026-08-14 coverage curve, both host-arm epochs=30 and the substrate arm's
8000-position economy choice) and §2's inconclusive-but-uninformative-in-the-wrong-direction cheap search, the chosen
lever is a **moderate combined increase in training-position coverage and epochs**, keeping every other config value
identical to the confirmed 3/6 baseline (B=48, w-target 40, sub-read-window 120, read-window 150, n-sub-demo 250,
n-eval-pos 800, n-sentences 40000, lr 0.5, weight-decay 8e-4):

- `--n-train-pos 8000 -> 20000` (2.5x, the stronger-evidenced axis per the host-arm coverage curve)
- `--epochs 10 -> 12` (1.2x, standard SGD-repetition margin)
- combined ~3x the gradient-step budget of the 3/6 baseline (~5000 steps/seed vs 1660)

Both knobs are plain data/compute-budget parameters with no checkpoint-specific tuning, so the lever is directly
transferable to a future larger-vocabulary WKV checkpoint (just scale `--n-train-pos`/`--epochs` to the new corpus).

**Queued** (per the ONE-GPU rule -- never launched directly):
```
cd /home/dant123/Projects/sim && SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._wkv_mouth_readout_eprop_batched_substrate_derisk \
  --seeds 42,43,44,100,101,102 --batch 48 --n-train-pos 20000 --epochs 12 --w-target 40 \
  --sub-read-window 120 --read-window 150 --n-sub-demo 250 --n-eval-pos 800 --n-sentences 40000 \
  --json research/findings/raw/_mouth_readout_tuning/eprop_ntp20000_ep12_6seed.json
```
Landed in `research/queue/gpu.queue` (line 10 of 10 at queue time, `#checked:post-stale-cache-fix coverage+epochs
lever...`). **Not yet run**: 9 long-running `_longitudinal_develop_loop_gpu`/`persistent_living_loop_derisk`/
`_vision_rstdp_readout_derisk` jobs (60-day longitudinal sims, hours each) sit ahead of it in the shared single-GPU
queue at write time, so this job's wall-clock start is not predictable from this session. Estimated cost once running:
~0.53s/substrate-forward step (measured from the 3/6 baseline's `learn_secs`/`n_grad_steps`) x ~5000 steps/seed x 6
seeds, plus shuffle/frozen/lesion/demo overhead -- roughly 6-8 GPU-hours total.

## 4. Verdict (honest, as of this write)

**PARTIAL / STAGED**, not GO or NO-GO. What is settled: the diagnosis (a tight, seed-uniform ~0.84-0.91 band, not a
seed-specific failure; the two prior "coverage/read-window EXCLUDED" verdicts are confound-flagged, not binding
post-fix). What is NOT yet settled: whether the queued coverage+epochs lever actually closes 3/6 -> >=5/6 at production
scale -- that requires the queued run above to land. The cheap numpy search could not substitute for it (§2).

**Next action for whoever picks this up:** once `research/findings/raw/_mouth_readout_tuning/eprop_ntp20000_ep12_6seed.json`
exists, read its `summary.go_count`/`sub_recov_ratio_mean`/`sub_recov_ratio_min` and the per-seed `anticheats_collapse`/
`forward_is_substrate` flags (must stay true -- this lever changes no anti-cheat logic, only data volume/epoch count).
If go_count reaches >=5/6: promote this finding's status to a GO write-up (retitle, drop STAGED) and proceed with the
mouth crutch-burndown / larger-vocab retrain the parent task named. If go_count stays <5/6: report the new per-seed
numbers (does the WHOLE distribution shift up, consistent with a residual-closing lever that just needs a bit more; or
does it stay flat, which would point away from data-exposure and toward the read-SNR/dendritic family the read-SNR arc
already has staged) -- do not re-run the same lever bigger without that read.

## 5. Biology + anti-cheat provenance (unchanged from the runner; no `sim/` edit)

e-prop local three-factor output rule: Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein & Maass, *A solution to the
learning dilemma for recurrent networks of spiking neurons*, Nature Communications 11:3625 (2020), PMID 32681001
(external-search-gate source recorded 2026-08-28T15:38:02Z, lane `e-mouth-fluency`, in the queue directory's
`.external_searches.jsonl` log).
Weight decay = Turrigiano synaptic scaling (the runner's own docstring). No anti-cheat logic changes: no weight
transport, no host gradient, `host_matmul_on_forward==0` required, shuffle-teach must collapse -- all asserted by the
UNCHANGED runner on the queued job.
