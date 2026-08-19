---
status: partial
type: finding
lane: gap#1
date: 2026-08-17
---

# Fluid mouth — the read-out e-prop learning FORWARD run on the ACTUAL spiking substrate (batched), closing the "forward is a host-linear proxy" qualification — PARTIAL (forward-is-substrate GO; recovery below parity, gap quantified)

**Date:** 2026-08-17 · **Type:** de-risk finding (research) · **Lane:** gap#1 / A1 (fluid mouth), emergence-bar burn-down.
**Closes the qualification on:** [`2026-08-14-fluid-mouth-readout-eprop-learned-GO.md`](2026-08-14-fluid-mouth-readout-eprop-learned-GO.md) (commit 6070d79d) — that rung is 6/6 GO on the substantive claim (a local three-factor rule recovers the mouth read-out head), but was QUALIFIED because the gradient-step FORWARD used the host-linear margin `W@h+head_b` as a fast PROXY for the substrate read (a per-step substrate forward over ~40k positions was ~1e6 sims, intractable). Named next lever: a BATCHED SUBSTRATE FORWARD.
**Runner:** [`research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py`](../runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py).
**Artifact:** `research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json`.
**Scope flags:** runner-only, additive, default-off, **NO `sim/` edit**. cfg.seed-controlled. 6 seeds 42/43/44/100/101/102.

## Headline (honest)

<!--derived-->

The e-prop learning FORWARD now runs on the ACTUAL spiking substrate, in BLOCK-DIAGONAL BATCHES (B=48 independent copies of the graded read-out in ONE bridge, each driven by a different position's feature → B substrate margins per ~read-window sim run), so the per-output error that drives the local rule is `softmax(substrate_margin)-onehot` off `cp_conductance_g_e/g_i` — **`host_matmul_on_the_learning_forward == 0`** (the whole point; a falsifiable count confirms every main gradient step is a substrate read). This CLOSES the methodological qualification: the forward is the substrate, not a host-linear proxy. The substrate-forward gradient is REAL (verify-first: one held batch memorised on CPU; the anti-cheats collapse cleanly).

**But the substrate-forward-trained read-out does NOT reach parity with the copied head at the tested scale.** At 8000 held-out training positions × 10 epochs (B=48), the learned `W_hat` read on the substrate recovers **0.37 of the perfect-argmax mass (mean, range 0.34–0.44) vs the copied head's 0.98** (ratio 0.38 mean, 0.35 min — every seed well below the 0.85 integrated bar).
The host-linear recovery of the same `W_hat` is 0.38 (vs the proxy's 0.93) and the weight-cosine to `head_w` is 0.14 (vs the proxy's 0.51) — host-side reads of the matrix itself, so they show the learned read-out is genuinely a WEAK approximation of the target head, not a read artifact. So this is a **PARTIAL** (0/6 on the integrated bar): forward-is-substrate is a clean GO (6/6, `host_matmul==0`), recovery-at-parity is not — the residual is quantified below and the next lever named.

## 1. Mechanism (the batched substrate forward)

<!--derived-->

Per gradient step, a minibatch of B positions with signed host features `h` (r_h·(Wo_sp@state)) drives B block-diagonal copies of the graded read-out; each block reads its own V word-pools' net signed synaptic-current margin off `cp_conductance_g_e/g_i`:

```
margin_sub  = BATCHED_SUBSTRATE_READ(W_hat, h)         # [B,V] off cp_conductance_*, bias-pop silenced (0 host matmul)
logits      = margin_sub / gain  +  head_b             # gain = a physical conductance->logit calibration; head_b a
                                                       #   [V] base-rate-prior VECTOR add (declared residual, NOT a matmul)
err_j       = softmax(logits)_j - 1{ j == target_t }   # DIRECT per-output error, from the SUBSTRATE margin
target_t    = argmax(head_w @ h + head_b)              # the TARGET HEAD's own decision = the teaching label
Delta w_ij  = -lr * err_j * h_i  -  wd * w_ij          # local delta (explicit outer product) + weight decay
W_hat       = scale_to(W_hat, ||W|| <= w_target)       # SYNAPTIC SCALING (Turrigiano): hold ||W|| in the linear range
```

`head_w` feeds ONLY the teaching decision `target_t` (no weight transport); the update is an explicit `np.outer` of `(err, h)` (no host gradient); the FORWARD margin is the substrate read (asserted `host_matmul_on_learning_forward == 0`, and `main_substrate_reads == n_grad_steps`, both per artifact). The block-diagonal batch makes the forward tractable: **~12 ms/position vs the ~150 ms/position single read** (~12×), so 8000×10 learning ran in ~16 min/seed on the substrate.

**Two calibrations, each a physical measurement (not a tuned knob):** (a) the conductance→logit GAIN is fit once per seed by a RANDOM PROBE weight (`margin_sub ≈ G·(Wfull_probe@h)`; measured substrate-vs-ideal-linear corr ~0.945, so the graded read is faithfully linear) — this puts the substrate margin in the read-out's logit units so the proven lr/decay transfer; (b) SYNAPTIC SCALING caps `||W_hat||` at `w_target` (~40, the head_w scale). Without the cap the substrate SATURATES for large `||W||` (the gain, calibrated near `||head_w||`, drops), so the forward UNDER-reads, the softmax never gets confident, the error persists and W runs away — measured `||W||≈970` vs `head_w 37.5`. With the cap `||W||=40` and the read stays in the linear regime.

## 2. An instrument bug found + fixed (a reused-bridge read corruption — load-bearing)

<!--derived-->

The substrate DEMO reads each candidate `W` (learned / copied / shuffle) to compare recoveries. Reusing ONE demo bridge across reads is CORRUPTING: a large-`||W||` read leaves persistent substrate state that poisons the NEXT read — measured, on a reused bridge a `||W||=40` read dropped the SUBSEQUENT copied read from **recov 0.976 → 0.0004**. The prior GO never hit this (its host-linear-learned `W` had `||W||≈20`, benign).
The fix: read every `W` on its OWN FRESH bridge with `cp.random` reseeded to the seed — this isolates each read AND gives learned/copied/shuffle IDENTICAL OU noise (a fair A/B). With the fix, copied reads 0.98 repeatably and a shuffled-teacher `W` collapses to ~0.004 (so the substrate metric is DISCRIMINATIVE here, not the frequency-tie-break confound the prior GO's §3 warned about at its operating point). This is why the earlier un-fixed runs read both learned AND copied at a spurious ~0.44: a read artifact, now removed.

## 3. Results (6 seeds 42/43/44/100/101/102)

<!--derived-->

8000 held-out training positions, B=48, lr 0.5, weight-decay 8e-4, w_target 40, 10 epochs; sub_read_window 120; substrate demo over 250 held-out positions (production graded read, P=4, read_window 150, bias-pop on). Each candidate `W` is read on its OWN fresh reseeded bridge (§2). Rows transcribed from `research/findings/raw/_wkv_readout_eprop_batched_substrate_6seed.json`.

| seed | SUB learned | SUB copied | ratio | 0.85×copied bar | SUB shuffle | hostlin recov | wcos→head_w | hostlin ac floor | fwd matmul | verify(B=48) |
|---|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.4436 | 0.9801 | 0.4526 | 0.8331 | 0.0044 | 0.3931 | 0.1291 | 0.0619 | 0 | Y |
| 43  | 0.3512 | 0.9663 | 0.3634 | 0.8214 | 0.0006 | 0.3593 | 0.1445 | 0.0520 | 0 | N |
| 44  | 0.3731 | 0.9741 | 0.3830 | 0.8280 | 0.0079 | 0.3874 | 0.1384 | 0.0436 | 0 | N |
| 100 | 0.3848 | 0.9856 | 0.3904 | 0.8378 | 0.0004 | 0.3814 | 0.1332 | 0.0568 | 0 | Y |
| 101 | 0.3442 | 0.9859 | 0.3491 | 0.8380 | 0.0008 | 0.3336 | 0.1381 | 0.0553 | 0 | N |
| 102 | 0.3494 | 0.9791 | 0.3569 | 0.8322 | 0.0011 | 0.3960 | 0.1280 | 0.0461 | 0 | N |
| **mean** | **0.3744** | **0.9785** | **0.3826** | **0.8317** | **0.0025** | **0.3751** | **0.1352** | **0.0526** | **0** | **2/6** |

**What IS a clean result (the decisive assertions, 6/6):** `host_matmul_on_the_learning_forward == 0` on all 6 seeds (the forward margin IS the substrate graded-conductance read, confirmed by the falsifiable `main_substrate_reads == n_grad_steps` count = True 6/6).
The anti-cheats COLLAPSE 6/6 on the discriminative channels: shuffle-teach / frozen / lesion-err host-linear recov ≤0.062 (vs learned 0.33–0.40), their weight-cosine ≤0.005 (vs learned 0.13–0.14), and — with the fresh reads of §2 — the substrate shuffle read ≤0.008 (vs learned 0.34–0.44). No weight transport, no host gradient; `host_rng_draws_on_read_path == 0` (6/6); seeded via `cfg.seed` (build-twice threshold hash `1d90c97348ccaf4a` identical); the conductance→logit calibration is stable (gain 9.8–10.2, substrate-vs-ideal-linear corr 0.94–0.95, `||W_hat||=40.0` held by synaptic scaling).
The `verify(B=48)` in-run guard (8 substrate-forward updates on ONE 48-position held batch must drop its CE) passes only 2/6 — it is NOISE-LIMITED at B=48 (8 steps barely move a 48-position batch under substrate noise); the CLEAN gradient-usability proof is the CPU B=6 smoke, where 3 substrate-forward updates MEMORISE a held batch (CE 6.68→0.00, argmax-error 1.00→0.00). The full-run learning (1660 steps) demonstrably learns a real read-out regardless (hostlin 0.37 vs floor 0.05, anti-cheats collapse 6/6).

**What is NOT at parity (the quantified residual, 6/6):** substrate ratio 0.38 mean / 0.35 min (learned 0.37 vs copied 0.98), hostlin 0.38 (vs proxy 0.93), wcos 0.14 (vs proxy 0.51). The host-linear recov and wcos are host-side numpy reads of the SAME `W_hat` (not substrate reads), so they show DIRECTLY that the learned matrix is a REAL but WEAK approximation of the target head — ~38% of the copied head's substrate recovery at this data scale — independent of any read artifact.

## 4. Why the gap, and the next lever (NOT a wall)

<!--derived-->

The residual is a NOISE + COVERAGE cost of a spiking forward, not a wall:
1. **Substrate gradient noise.** Each gradient step's error comes from ONE finite-window substrate read (OU noise + the graded-read nonlinearity), so the gradient is noisier per position than the noise-free host-linear proxy. At the same 8000 positions the substrate forward reaches hostlin 0.38 (mean) where the proxy reaches ~0.87 (its own data-coverage curve: 3.3k→0.85, 30k→0.95). Levers: LONGER `sub_read_window` (cleaner per-step margin), MULTI-READ averaging per step, or LOWER lr with more epochs (SGD noise averaging).
2. **Coverage.** 8000 unique positions vs the proxy's ~40000. More coverage lifts both; the substrate forward lags but tracks.
3. **It is genuine read-out quality, not a read/demo artifact.** The host-linear recov (~0.39) and wcos (~0.13) are host-side numpy reads of `W_hat` itself (no substrate, no OU noise), and they are LOW — so the learned matrix is a weak approximation of `head_w` at this scale, independent of how it is read. (A separate exploratory 'read `W_hat` through the batched forward's own argmax' probe was found to be head_b-dominated — the base-rate prior swamps the gain-divided feature for an ARGMAX read even though it drives the softmax GRADIENT fine — so it is not used here; the host-side wcos/recov are the clean read-out-quality reads.)

The escalation order for a follow-up rung: longer read window + per-step read averaging (noise) → 2–4× coverage (data) → then re-measure parity.

## 5. Grounded biology (resolving in-project anchors)

<!--derived-->

- **local three-factor / e-prop output rule** (direct output error × filtered presynaptic trace; transport-free): Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, *A solution to the learning dilemma for recurrent networks of spiking neurons*, **Nat Commun 11:3625 (2020)**. On-project: [`research/biology/deep-credit-on-spikes.md`](../biology/deep-credit-on-spikes.md).
- **dendritic-prediction delta rule** (`Δw ∝ (target − predicted) × presynaptic`): Urbanczik & Senn, *Learning by the Dendritic Prediction of Somatic Spiking*, **Neuron 81:521–528 (2014)**.
- **synaptic scaling** (homeostatic magnitude control keeping weights in a viable — here substrate-readable — range): Turrigiano's homeostatic plasticity. Here the explicit companion process that holds `||W_hat||` in the graded read's LINEAR regime (the finding's "what else does the real system run alongside this, that we replaced with a constant?" — here a saturation the un-scaled forward walked into).

## 6. Honest residual (what stays a host shortcut AFTER this rung)

<!--derived-->

1. **Recovery is below parity** (§3–4) — the batched substrate forward learns a real read-out but recovers ~38% of the copied head at 8000×10 (0/6 on the integrated bar); parity needs the noise/coverage levers of §4. So "the substrate learns its own read-out end-to-end" is DEMONSTRATED as a MECHANISM (forward-is-substrate GO, 6/6) but not yet at the copied head's quality.
2. **The base-rate prior head_b is added host-side** (a [V] vector, the declared copied residual — same as the prior GO; the production biologization is the tonic bias-pop, and the pipeline already reads head_b as a synapse).
3. **The teaching signal is a supervised label** = the target head's own next-word decision (a host-supplied scaffold; the end-state is AI-teacher-then-human next-token feedback made real).
4. **Only the OUTPUT layer is learned**; `Wo_sp` deep-credit / `Wv` / LN / embedding / the r_h gate stay copied/host. NOT "fully spiking", NOT production-wired. Functional read-outs only; no phenomenal-experience claim.

## 7. Determinism + provenance

<!--derived-->

cfg.seed set through the pipeline builder for every seed (NOT `actual_seed_used`); build-twice threshold hash identical ⇒ seeded. 6 seeds 42/43/44/100/101/102, per-seed ckpt `wkv_ssmU6_v1000_d128_seed{seed}.npz`. `no_transport=True`, `no_host_grad=True`, `host_matmul_on_learning_forward==0`, `host_rng_draws_on_read_path==0` asserted per artifact. Provenance sidecar auto-recorded by `research/runners/__init__.py`. Runner-only, default-off, NO `sim/` edit.
