---
status: qualified
type: finding
lane: gap#1
date: 2026-08-14
---

> ⚠️ **CONFOUND CORRECTION (2026-08-27) — the SUBSTRATE-integration ratio was measured through a STALE weight cache.**
> The 2026-08-27 stale-weight-cache confound sweep found this rung's runner (`_wkv_mouth_readout_eprop_learn_derisk.py::LearnedReadout.set_weights`) REASSIGNS `cp_connections.data`, which breaks the megakernel-v2 transposed-CSR (WT) transmission cache's view-safety — so the substrate read transmitted STALE weights.
> The headline **`sub_recov_ratio_mean ~1.00` ("learned ≈ copied") was CONFOUNDED** (the stale cache inflated it). **RESOLVED 2026-08-28 by the faithful 6-seed re-measurement under the fix** (`--feature substrate --weight-decay 0.0008`, artifact `research/findings/raw/_stale_weight_cache_confound_sweep/eprop_learn_substrate_REMEASURE_6seed.json`): `sub_recov_ratio_mean = 0.9357` (min 0.9138), **6/6 GO**, anti-cheat 6/6. So the true value is **~0.94 of copied** (not the buggy 1.00, and stronger than the ~0.865 single-seed smoke suggested) — the "learned ≈ copied" claim is DOWNGRADED to "recovers ~94% of copied" but the rung is VINDICATED (clears the ≥0.85 bar at 6/6). The `~0.865` smoke figure below is superseded by this 6-seed re-measurement.
> The RULE-RECOVERY half (host-linear recov ~0.93, weight-cosine ~0.51) is UNAFFECTED — it runs on the host-linear path, not the substrate read.
> See [`2026-08-27-stale-weight-cache-confound-sweep-PARTIAL`](2026-08-27-stale-weight-cache-confound-sweep-PARTIAL.md) + [`2026-08-27-mouth-stale-coo-training-fix-PARTIAL`](2026-08-27-mouth-stale-coo-training-fix-PARTIAL.md).

# Fluid mouth — the read-out head LEARNED on the substrate by a local three-factor rule (e-prop / delta), retiring the copied Qwen weights — GO on rule-recovery + substrate integration

**Date:** 2026-08-14 · **Type:** de-risk finding (research) · **Lane:** gap#1 / A1 (fluid mouth), emergence-bar burn-down.
**Scope flags:** runner-only, additive, default-off, **NO `sim/` edit**. cfg.seed-controlled. 6 seeds 42/43/44/100/101/102.
**Runner:** [`research/runners/_wkv_mouth_readout_eprop_learn_derisk.py`](../runners/_wkv_mouth_readout_eprop_learn_derisk.py).
**Artifacts:** `research/findings/raw/_wkv_readout_eprop_learn_host_6seed.json`, `..._substrate_6seed.json`.

## Headline (honest)

The mouth's whole state→logits path is already on the spiking substrate (signed graded-conductance reads, pipeline GO 2026-08-13), but the read-out head `logits = head_w @ h + head_b` weights were **Qwen's — LOADED, not LEARNED** (residual #4 of the pipeline GO). This rung **LEARNS the read-out head `W_hat` by a local three-factor rule** (e-prop specialised to the output layer = the delta rule), recovering the TARGET head's next-word decisions on **held-out** TinyStories context, with **NO weight transport** (the update reads only the presynaptic feature and the per-output error; `head_w` feeds only the teaching label) and **NO host gradient** (an explicit `np.outer`, `err = softmax − onehot`). The learned weights are Dale-split and read out **on the substrate** off `cp_conductance_g_e/g_i`.

**GO (rule-recovery + integration), 6/6 seeds** — on the two artifact-free discriminators:
- **The local rule recovers the target map:** host-linear `recov_argmax` ≥ 0.90 (mean ~0.92), and the learned weights **align with `head_w`** (weight-cosine ~0.5) — while the frozen / lesion-err / shuffle-teach controls collapse to `recov_argmax` ~0.04 and weight-cosine ~0.00.
- **Integration:** the learned `W_hat`, **read on the substrate** (graded-conductance production read), reproduces the copied head's substrate recovery to a ratio of ~1.00 (learned ≈ copied).

Exact host-arm aggregates (`research/findings/raw/_wkv_readout_eprop_learn_host_6seed.json`): host-linear recov mean 0.9321 (min 0.9176), weight-cosine to `head_w` mean 0.5133 (min 0.4991), substrate learned/copied recov ratio mean 1.001 (min 0.9956); anti-cheat host-linear recov floor ≤ 0.0562 and weight-cosine floor ≤ 0.0104.

**What is NOT claimed (measured, load-bearing):** `argmax_agree ≥ 0.90` (a sub-criterion in the original spec) is **not** met and is **unreachable by any finite-data learned OR copied map here** — see §3. Only the **output layer** is learned; the teaching signal is a supervised scaffold; `Wo_sp` deep-credit stays open (§4). Functional read-outs only; no phenomenal-experience claim.

## 1. Mechanism (local three-factor rule, on the substrate read-out)

Per held-out answer position, with the mouth's hidden feature `h = r_h·(Wo_sp@state)` (host arm) or the on-substrate output-projection read (production arm):

```
elig_i(t) = alpha·elig_i(t-1) + h_i(t)               # e-prop forward eligibility (alpha=0 => plain delta)
err_j     = softmax(margin)_j − 1{ j == target_t }    # DIRECT per-output error (no DFA, no W^T backward path)
target_t  = argmax( head_w @ h + head_b )             # the TARGET HEAD's OWN decision = the teaching label
Delta w_ij = −lr · err_j · elig_i  −  wd · w_ij       # local delta + weight decay (synaptic-scaling companion)
Wp = max(W_hat,0);  Wn = max(−W_hat,0)                # Dale-split onto the V word-pools; read off g_e/g_i
```

For a single OUTPUT layer the learning signal is the **direct output error** — there is no `W^T` backward path to transport, so e-prop's transport-free property is exercised only trivially here (the interesting hidden-credit case is the separate, still-open `Wo_sp`/deep-credit rung). `head_w` is read ONLY to form the teaching decision `target_t`, NEVER into the update: asserted `no_transport=True`, `no_host_grad=True` in every artifact; `host_rng_draws_on_read_path == 0`; seeded via `cfg.seed` (NOT `actual_seed_used`) with a build-twice hash of `cp_neuron_firing_thresholds`.

## 2. Two measurements that reshaped the original spec (these ARE the deliverable's honesty)

**(a) Recovering a 1000-way linear map by a local rule is DATA-limited.** The spec's "~200 eval positions × 12 epochs" cannot cover a 1000-word target (most words are never seen); epochs merely repeat the same positions. Measured host-linear `recov_argmax` vs training positions: ~200 → 0.07, 3.3k → 0.85, 30k → 0.95. ~40k held-out training positions are needed to clear 0.90. This is a data-coverage fact, not a rule failure.

**(b) A per-STEP substrate-margin forward at that data volume is intractable, and the RAW substrate margin is bias-pinned for learning.** ~40k positions × ~30 epochs ≈ 10^6 substrate sims (≈150 ms each) is out of budget; and at small `W_hat` the substrate margin's winner is pinned by the `head_b` tonic bias-pop (the base-rate prior), so the softmax self-regulation stalls (the error does not reflect `W_hat`). So the many gradient-step **forward uses the host-linear margin `W_hat@h + head_b`** — a **faithful fast proxy**: the substrate reconstructs this SAME linear map (mouth pipeline GO, recov ~0.95) — and the learned weights are then **DEMONSTRATED on the substrate read** (the decision IS `argmax` over the substrate net-current margin). **Weight decay** is the synaptic-scaling companion process that keeps `||W_hat||` in the substrate-readable regime: without it `||W||` diverges ~20× (to ~730) and the substrate can no longer read the map (recov collapses); with it `||W|| ~ 20` (vs `||head_w|| = 37.5`) and the map reads cleanly.

**The named next lever** to run the FULL learning with the error read off the substrate margin is a **BATCHED substrate forward** (a block-diagonal read-out processing B positions per sim run) + a de-biased (bias-silenced) learning margin — banked, not deferred.

## 3. Why `argmax_agree ≥ 0.90` is unreachable here (and is the wrong bar)

Two independent ceilings, both binding the COPIED head equally (so neither is a learning limit):
- **The graded substrate read caps argmax fidelity.** Read on the substrate, the COPIED `head_w` itself scores `argmax_agree ~0.68` (its `recov_argmax`, mass-weighted, is ~0.88–0.95). The graded-conductance read is not a perfect argmax reproducer — this is the pipeline's own residual.
- **The rare-word tail.** Even at the host-linear data ceiling, `argmax_agree` plateaus ~0.79 (30k positions): the frequent words carry most probability MASS (so `recov_argmax`, which weights by the target head's confidence, reaches ~0.92–0.95) but the long rare-word tail is never covered by finite data.

So `recov_argmax` (mass-weighted) is the meaningful bar and it passes; a 0.90 argmax_agree would require memorising the tail AND a lossless substrate read — neither is a property of the LEARNING.

**A substrate-argmax confound (documented so it is never mis-read as a result).** The vocab is frequency-ordered (`word 0='the'`, `1='and'`, …) and the FS-WTA+OU dynamics give near-flat margins a systematic **low-index bias**, so `np.argmax` tie-breaks toward low-index = frequent pools. A **frozen / zero** read-out therefore scores a spurious `recov_argmax ~0.95` and `argmax_agree ~0.81` on the SUBSTRATE read — as high as the learned map. Hence the substrate argmax metric is a **consistency check, not a discriminative test**; the discriminative, artifact-free channels are the **host-linear recov** and the **weight-cosine to `head_w`**, on which the anti-cheats collapse cleanly.

## 4. Results (6 seeds 42/43/44/100/101/102)

<!--derived-->

The per-seed rows below are transcribed from the two cited artifacts (`research/findings/raw/_wkv_readout_eprop_learn_host_6seed.json` and `..._substrate_6seed.json`); the exact aggregates are checked in the Headline.

**Arm A — ISOLATION (`--feature host`): the local rule recovers the target map.** 40k held-out training positions, lr 0.5, weight-decay 8e-4, 30 epochs; substrate demo over 250 held-out positions. Discriminative channel = host-linear recov + weight-cosine (the frozen/lesion/shuffle floors are `hostlin` recov and wcos, artifact-free — §3). SUB = production substrate read (bias-pop on; a consistency check).

| seed | hostlin recov | hostlin agree | wcos→head_w | anti-cheat recov (frz/les/shf) | wcos floor | SUB learned recov | SUB copied recov | ratio | GO |
|---|---|---|---|---|---|---|---|---|---|
| 42  | 0.9176 | 0.7275 | 0.5150 | 0.047 / 0.042 / 0.004 | 0.010 | 0.8789 | 0.8767 | 1.0025 | ✓ |
| 43  | 0.9343 | 0.7638 | 0.5145 | 0.044 / 0.041 / 0.002 | 0.004 | 0.9020 | 0.9060 | 0.9956 | ✓ |
| 44  | 0.9374 | 0.7850 | 0.5253 | 0.035 / 0.054 / 0.001 | 0.001 | 0.8799 | 0.8829 | 0.9966 | ✓ |
| 100 | 0.9323 | 0.7725 | 0.5209 | 0.056 / 0.052 / 0.001 | 0.003 | 0.9215 | 0.9223 | 0.9991 | ✓ |
| 101 | 0.9376 | 0.7550 | 0.4991 | 0.056 / 0.051 / 0.001 | 0.001 | 0.9143 | 0.9109 | 1.0037 | ✓ |
| 102 | 0.9334 | 0.7625 | 0.5050 | 0.044 / 0.043 / 0.000 | 0.009 | 0.8827 | 0.8751 | 1.0087 | ✓ |
| **mean** | **0.9321** | **0.7611** | **0.5133** | **≤0.056** | **≤0.010** | **0.8965** | **0.8956** | **1.001** | **6/6** |

Rule-recovery: host-linear recov 0.9321 mean (min 0.9176, all ≥0.90) with weight-cosine to `head_w` 0.513 mean (min 0.499) — vs anti-cheat recov ≤0.056 and anti-cheat wcos ≤0.010: a ~17× recov separation and a ~50× cosine separation, 6/6. Integration: the learned read-out on the substrate reproduces the copied head (ratio 1.001 mean, min 0.9956), 6/6.

**Arm B — PRODUCTION (`--feature substrate`): the learned read-out via the full substrate pipeline.** Identical local-rule learning (host-linear forward); the substrate DEMO now reads via the mouth's **on-substrate output-projection feature** (a projection sim per position) → the graded read-out. `hostlin recov` / `wcos` are identical to Arm A (same learning); the differentiator is the production substrate read (`SUB`).

| seed | SUB learned recov | SUB copied recov | ratio | hostlin recov | wcos→head_w | GO |
|---|---|---|---|---|---|---|
| 42  | 0.8691 | 0.8681 | 1.0012 | 0.9176 | 0.5150 | ✓ |
| 43  | 0.8937 | 0.8876 | 1.0069 | 0.9343 | 0.5145 | ✓ |
| 44  | 0.8771 | 0.8778 | 0.9992 | 0.9374 | 0.5253 | ✓ |
| 100 | 0.9006 | 0.9013 | 0.9992 | 0.9323 | 0.5209 | ✓ |
| 101 | 0.9017 | 0.8995 | 1.0024 | 0.9376 | 0.4991 | ✓ |
| 102 | 0.8737 | 0.8722 | 1.0017 | 0.9334 | 0.5050 | ✓ |
| **mean** | **0.8860** | **0.8844** | **1.0018** | **0.9321** | **0.5133** | **6/6** |

Read via the FULL substrate pipeline (projection feature + graded read-out), the learned read-out reproduces the copied head to ratio 1.0018 mean (min 0.9992), 6/6 — with the same rule-recovery + anti-cheat-collapse as Arm A (`research/findings/raw/_wkv_readout_eprop_learn_substrate_6seed.json`).

**Anti-cheats (discriminative channel — each collapses; the substrate-argmax controls are non-discriminative per §3):**
- **shuffle-teach** (deranged target index): host-linear recov ~0.00, weight-cosine ~0.00 — a misaddressed teacher recovers nothing.
- **frozen** (no update): host-linear recov ~0.04 (base-rate floor), weight-cosine ~0.00.
- **lesion-err** (`err≡0`): host-linear recov ~0.04, weight-cosine ~0.00.

## 5. Grounded biology (resolving in-project anchors)

- **B1 — local three-factor / e-prop output rule** (direct output error × filtered presynaptic trace; transport-free): Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, *A solution to the learning dilemma for recurrent networks of spiking neurons*, **Nat Commun 11:3625 (2020)**. Cited in `research/findings/2026-06-20-FHRR-B-learned-binder-scoping.md`; e-prop proven on-project in `research/biology/deep-credit-on-spikes.md`.
- **B2 — dendritic-prediction delta rule** (`Δw ∝ (target − predicted) × presynaptic`, a local rule that learns a target linear map): Urbanczik & Senn, *Learning by the Dendritic Prediction of Somatic Spiking*, **Neuron 81:521–528 (2014)**.
- **Weight decay = synaptic scaling** (a homeostatic companion that keeps synaptic weights in a viable regime) — Turrigiano's homeostatic plasticity; here the explicit stand-in for the competitive/normalising process biology runs alongside potentiation.

## 6. Honest residual (what stays a host shortcut AFTER this rung)

1. **The TEACHING SIGNAL is a supervised label** = the target head's own next-word decision = a host-supplied teaching **scaffold** (flagged). Legitimate under the scaffold rule (innate-reflex-teaches-a-learned-circuit): the end-state is the AI-teacher-then-human next-token feedback, the SAME per-output error made real; the RULE (local three-factor) is the deliverable, the error GENERATION is the scaffold (convert later to a neural error population — subtractive-inhibition `target − actual`).
2. **The gradient-step FORWARD is the host-linear margin**, a faithful fast proxy for the substrate read (they reconstruct the same linear map; the substrate is the pipeline GO). The FINAL read-out is demonstrated on the substrate. Running the full learning with the error read off the substrate margin is the **batched-substrate** lever (§2).
3. **Only the OUTPUT layer is learned.** `head_w` becomes LEARNED; `Wo_sp` (hidden projection — needs DFA / e-prop-with-feedback, the harder deep-credit rung, `research/biology/deep-credit-on-spikes.md`, open at 6 seeds), `Wv` (its own separate diagonal e-prop GO), LN, embedding, and the `r_h` gate stay copied/host.
4. **NOT "fully spiking" / NOT production-wired.** A named copied-weight shortcut (the read-out head) becomes learned by a local rule and read on the production mouth substrate — a real emergence-bar burn-down, honestly bounded. Functional read-outs only; no phenomenal-experience claim.

## 7. Determinism + provenance

cfg.seed set through the pipeline builder for every seed (NOT `actual_seed_used`); build-twice hash of `cp_neuron_firing_thresholds` identical ⇒ seeded. 6 seeds 42/43/44/100/101/102, per-seed ckpt `wkv_ssmU6_v1000_d128_seed{seed}.npz`. `no_transport=True`, `no_host_grad=True`, `host_rng_draws_on_read_path == 0` asserted per artifact. Provenance sidecar auto-recorded by `research/runners/__init__.py`. Runner-only, default-off, NO `sim/` edit.
