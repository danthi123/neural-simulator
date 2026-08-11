---
type: finding
status: contributing
date: 2026-08-11
mechanism: neural-statevalue-critic + homeostatic-readout-gain
lane: D-pragmatics
seeds: [42, 43, 44, 100, 101, 102]
verdict: PARTIAL (6-seed) — homeostat kills the yoked leak structurally (aggregate), strict per-seed count unmoved at 3/6
artifacts:
  - research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_smoke_s44.json
  - research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_6seed.json
instrument: YOKED-CONTROL contingency decomposition (same as the leg2-v3 gate) — FIX (reward contingent on the action) vs YOKED (identical DA-magnitude distribution DECOUPLED from the action). The learned intent->utter weight separation wsep = mean w[success-optimal] - mean w[others] is the readout-free structural measure; contingency is the FIX-minus-YOKED gap, with an untrained arm (near-zero) as the null. A non-contingent source (heterogeneity/DC/lagging-baseline) would move YOKED equally, so a YOKED at/below the null attributes the FIX separation to reward contingency. The homeostatic readout gain is IDENTICAL machinery in both arms (part of the critic readout, not the reward), so any FIX-vs-YOKED difference is not a calibration artifact.
---

# value-critic-neural: the fully-neural per-intent critic is real + already on main, but its strict-gate score is 3/6 (not the finding's headline 6/6); a homeostatic critic-readout gain cleanly kills the dominant YOKED-leak failure mode (1-seed smoke) — PARTIAL, 6-seed adjudicates

## What was already true (build/verify of the existing neural critic)

<!--derived-->

The `--neural-critic` path in `_pragmatic_readback_leg2_v3_statevalue_derisk.py` (landed on `main` 2026-08-10,
commit `2fdfc3e54`) IS a genuine spiking critic, verified by reading the code: `crit[intent]` is a spiking
population, `Vctx[t]=rate(crit[t])` is read from spikes, and it is trained by DA-gated plasticity on the
`intent[t]->crit[t]` diagonal with the advantage `A = success - Vctx` — which is simultaneously the critic's TD
error, a clean advantage actor-critic (Chen 2018: VTA relays the signed error to both Area-X actor and the ventral
critic). No host EMA. So the host-EMA shortcut IS burned down at the mechanism level.

**Correction to the committed headline (a measurement overclaim, not a mechanism error).** The 2026-08-10 finding
states the neural critic "passes contingency 6/6". Re-reading the six committed artifacts
(`research/findings/raw/_pragmatic_success/v3_neural_b04_s{42,43,44,100,101,102}.json`), the 6/6 is the LOOSE
directional read (fix wsep > 0 AND yoked wsep ~0 on all six). On the runner's OWN strict `contingency_pass` gate
(`fix_warg >= 0.60` AND `fix_warg - yok_warg >= 0.20` AND `fix_sep - yok_sep > 0` AND `|yok_sep| < 0.25*fix_sep`),
the committed neural critic passes **3/6** (s42, s100, s101), failing s43 (fix_warg 0.33) and s44/s102 (residual
YOKED leak). The host-EMA baseline passes 5/6 on the same structural read. So the neural critic is real and
contingent-directional 6/6 but noisier on decision-quality — 3/6 strict — which is the residual this de-risk attacks.

## The un-tried lever: a homeostatic critic-readout gain (the companion process the constant proxied)

<!--derived-->

The neural critic's readout scale is fixed by the constant `CRIT_READ_GAIN_V3 = 1.0` (v3 docstring flags it
"calibration-sensitive"). Per the wall-reframe ("what else does the real system run alongside this that we replaced
with a constant?"): the real VTA/critic readout is gain-controlled to the reward set-point (synaptic-scaling /
intrinsic-excitability homeostasis toward a target output). The constant proxies that slow gain-control. If the raw
critic rate cannot reach the success scale under the read drive, `E[A]` stays biased and a net DA non-contingently
potentiates the heterogeneity-favored assembly — the same lagging-baseline leak that beta-centering killed for the
host EMA, but the neural critic has no beta.

`_pragmatic_readback_leg2_v3b_neuralcritic_homeo_derisk.py` (additive; imports v3+v2; NO `sim/` edit; neural critic
always on) restores the companion process: a scalar readout gain `g = succ_bar / crit_bar`, where `succ_bar` (EMA of
the NEURAL success rate = the reward set-point) and `crit_bar` (EMA of the raw NEURAL critic rate) are running means
(`homeo_beta=0.1`). Then `Vctx[t] = g * rate(crit[t])`, forcing `E[Vctx] -> E[success]` (`E[A]->0`, leak killed)
WHILE preserving per-context differentiation (`Vctx[t]` still varies with the learned `rate(crit[t])`). The VALUE
(which intent is worth more) stays 100% neural in the `intent->crit` weights; only the global readout SCALE is
homeostatically calibrated — a companion homeostatic process, not a value shortcut. Distinct from prior homeostat
levers: `_pragmatic_readback_leg2_v2_homeostat` (assembly-CV threshold homeostat, REFUTED, a readout-SNR problem on
the host line) and `_wta_afferent_winner_homeostat` (common-mode remover for the WTA) — neither calibrates the
critic readout gain to the reward scale.

## The 1-seed smoke — seed 44 (a plain-neural strict-gate FAIL, dominated by the YOKED leak)

<!--derived-->

`SIM_BACKEND=numpy ... --seed 44` (n_train=360, both arms; ~300s). Artifact:
`research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_smoke_s44.json`. The homeostat calibrated the
gain 1.0 -> 0.427 (the raw critic rate ran ~2.3x the success scale — the fixed g=1.0 was genuinely mis-scaled).

| arm variant | fix wsep | yoked wsep | fix-yoked | fix_warg | yok_warg | adv sign_acc | strict pass |
|---|---|---|---|---|---|---|---|
| plain-neural (committed) | +0.0920 | +0.0308 | +0.0612 | 0.67 | 0.33 | 0.899 | False |
| homeo-neural (this lever) | +0.0722 | -0.0630 | +0.1352 | 0.33 | 0.00 | 0.936 | False |

**What the lever DID (its design target):** it killed and REVERSED the yoked leak (+0.031 -> -0.063 — the
non-contingent positive drift is gone; a yoked arm at/below the null is the cleanest non-contingency signal), widened
the FIX-minus-YOKED contingency separation 2.2x (0.061 -> 0.135), and raised advantage-sign accuracy (0.899 ->
0.936). `attributable_to` moves from a leaking split to 187% (control opposite the treatment).

**Why it is still a strict FAIL on this seed (honest):** the composite gate also requires `fix_warg >= 0.60`, and on
seed 44 the FIX weight-argmax fell 0.67 -> 0.33 (the average separation stays positive at +0.072, but for only 1/3
intents does success-optimal become the strict argmax at this trial count). The gate's `|yok_sep| < 0.25*fix_sep`
"near-zero" band — designed for a POSITIVE host-EMA leak — also flags the (favorable) negative yoked. So the lever
demonstrably fixes the CONTINGENCY residual but does not, on this seed, lift the DECISION-QUALITY residual; net
composite gate FAIL -> FAIL. This is a PARTIAL: the leak-kill is real and on-target; whether it nets positive on the
strict-gate COUNT across seeds is what the 6-seed sweep decides.

## Verdict + next mechanism

<!--derived-->

**PARTIAL (1-seed).** The fully-neural critic is genuine and already on `main`; its strict-gate score is 3/6 (the
committed 6/6 is the looser directional read — corrected here, not retracted: the mechanism claim stands). The
homeostatic readout-gain lever cleanly removes the dominant YOKED-leak failure mode on the worst failing seed
(contingency separation 2.2x, yoked reversed below null) but trades fix decision-quality on that seed, so it does
not flip the composite gate at 1 seed. **Next mechanism (named, un-run here to keep this a single-lever probe):**
the residual is now decision-quality / differentiation (`fix_warg`), not the leak — the separation is real but
sub-argmax, the same "sub-argmax refinement" class the reward-misspec finding named; address by (a) more training
trials / larger signed-advantage magnitude (the finding's own "separation set by trial count" logic), or (b) a
common-mode-centered critic readout `Vctx = succ_bar + slope*(raw - crit_bar)` that centers WITHOUT shrinking the
per-context spread the multiplicative gain compresses.

## 6-SEED ADJUDICATION (coordinator-run) — PARTIAL confirmed: the leak-kill is structural, the strict count is unmoved

<!--derived-->
The 6-seed sweep was run (`research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_6seed.json`). Result:
**the homeostat does exactly its designed job at the aggregate — it KILLS the yoked leak** (`yoked_mean_weight_sep_vs_succopt
= -0.0007`, i.e. ~0, vs the plain-neural baseline's +0.031 leak; `fix_mean_weight_sep = 0.1222` so contingent ≫ yoked,
`contingency_pass_aggregate = True`) — **but it does NOT lift the per-seed strict-gate count: still 3/6**
(`per_seed_contingency_pass`: 42✓ 43✓ 101✓ / 44✗ 100✗ 102✗), the same count as the plain-neural critic, with
`fix_mean_weight_argmax_vs_succopt = 0.778` and mean homeo gain 0.445. So the lever removes the yoked-leak confound
STRUCTURALLY (the critic's value is now contingent-only at the mean) but the residual is confirmed to be
decision-quality / per-context differentiation on 3 seeds, NOT the leak. **Verdict: PARTIAL (6-seed) — mechanism works
as designed, strict 6/6 not reached.** Next mechanism (unchanged, now 6-seed-motivated): the common-mode-centered
readout `Vctx = succ_bar + slope*(raw - crit_bar)` (centers without compressing per-context spread), and/or more
signed-advantage magnitude — the sub-argmax refinement class the reward-misspec finding named.

Reproduce: `SIM_BACKEND=numpy python -u -m research.runners._pragmatic_readback_leg2_v3b_neuralcritic_homeo_derisk --seeds 42 43 44 100 101 102 --json research/findings/raw/_pragmatic_success/leg2_v3b_neuralcritic_homeo_6seed.json`
