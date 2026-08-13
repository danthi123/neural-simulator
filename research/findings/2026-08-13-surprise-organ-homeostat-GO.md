---
type: finding
status: de-risk-GO-6of6
verdict: GO (6/6 seeds). A per-block HOMEOSTATIC PREDICTION-GAIN companion closes the surprise organ's read-precision residual. The heterogeneous GNW-bus organ de-risk (`2026-08-13-gnw-bus-heterogeneous-organ-GO.md`) named ONE separate-axis residual — the production surprise/familiarity monitor's single-read confirm precision, `het_vote_rate` mean 0.9375 (3 seeds 8/8, 3 seeds 44/100/101 7/8): on ~1/8 marginal edges a genuinely FAMILIAR concept read just above the surprise threshold so the organ WITHHELD its vote. Reading the organ's per-block confirm rates directly, the residual is a PER-BLOCK collapse of the top-down prediction RECALL (seed 101 block 4 recall 0.58 Hz vs 6-12 Hz elsewhere; seed 100 block 2 recall 0.00 Hz) — the FS/PV prediction pool delivers ~no subtractive inhibition to that block so the familiar assertion fires UN-cancelled at contradict level (4-6 Hz). The uniform topographic prediction weight (0.8) is a fixed CONSTANT standing in for the homeostatic gain-control the animal runs alongside predictive coding. Adding a per-block homeostatic prediction-gain equalizer (strengthen a block's cue->patient_expected gain until its CONFIRM error falls to a low target) lifts `het_vote_rate` to 1.000 (8/8) on ALL 6 seeds AND end-to-end parity with host recall (consensus_acc 1.000 == host 1.000), WHILE surprise SPECIFICITY holds perfectly (novel/contradict still register: mean confirm 0.35 Hz < threshold ~2.6 <= contra/novel ~5.1-5.5 Hz; novel_registers=contradict_registers=1.000) and every [N] control is preserved (substrate_combines True; ignite-when-voted / abstain-when-withheld 1.000; het-dropped / disagree / het-organ-lesion / workspace-lesion / single-organ all 0.000; reflex 1.000; discrimination + moat all seeds). Numbers cite research/findings/raw/_surprise_organ_homeostat/summary.json. NOT closed (per docs/TERMS): a default-off de-risk runner, not the shipped organ.
date: 2026-08-13
mechanism: surprise-organ
---

# Surprise organ — a per-block homeostatic prediction-gain companion closes the read-precision residual (6-seed GO)

## The residual being closed (the named #1 separate-axis residual on the heterogeneous GNW-bus finding)
The heterogeneous GNW-bus organ de-risk (`2026-08-13-gnw-bus-heterogeneous-organ-GO.md`, 6/6 GO) wired the production
spiking surprise/familiarity monitor (`surprise_production_organ.SurpriseProductionOrgan`) as a NON-COMPOSER vote in
the GNW workspace bus. Its declared SEPARATE-axis residual: `het_vote_rate` mean **0.9375** (3 seeds 8/8, 3 seeds
44/100/101 7/8). On the 1/8-per-seed marginal edge a genuinely FAMILIAR concept's confirm read lands just ABOVE the
organ's surprise threshold, so organ H WITHHOLDS its vote and the substrate correctly ABSTAINS (a stronger moat, but
end-to-end parity with host recall is lost). That finding named this the surprise organ's OWN precision boundary and
the next mechanism: "the surprise organ's homeostatic-gain / divisive-normalization companion (equalize the
topographic prediction strength across blocks), or an evidence-integrated settled read." This de-risk builds + validates
that companion.

## The root cause (measured on the organ's own reads, not assumed)
Reading the surprise organ's per-block CONFIRM rates directly, the residual is NOT a global gain miss — it is a
PER-BLOCK collapse of the top-down prediction RECALL. On the marginal block the cue->patient_expected recall is
near-silent (seed 101 block 4: recall 0.58 Hz vs 6-12 Hz on the other blocks; seed 100 block 2: recall 0.00 Hz), so
the FS/PV prediction pool delivers almost NO subtractive inhibition to that block, and the block's asserted-patient
excitation fires UN-cancelled at contradict level (4-6 Hz) even though the assertion is FAMILIAR. The uniform
topographic prediction weight (0.8) is a fixed CONSTANT standing in for the homeostatic gain-control the animal runs
alongside predictive coding — the wall reframe ("what else does the real system run alongside this, that we replaced
with a constant?"). The proxy is the per-block prediction gain. (Confirmed a plain E/I inhibitory-weight boost is WEAK
here — with near-silent recall there is little inhibition to amplify: gain 14->60 only moves the outlier 4.34->1.45
Hz — whereas restoring the RECALL is decisive.)

## The companion process (the biology) + why specificity is preserved by construction
Predictive-coding error units are PRECISION-WEIGHTED: the gain of the prediction that cancels an expected input is set
by a homeostatic / divisive-normalization control (inhibitory E/I balancing, Vogels-Sprekeler-Zenke-Ganguli-Gerstner
2011; homeostatic synaptic scaling, Turrigiano 2008; precision as gain-control, Feldman & Friston 2010; Bastos et al.
2012). Here that companion is a PER-BLOCK homeostatic prediction-gain equalizer: for each stored block, if the CONFIRM
error (the surprise pool's spiking rate when the FAMILIAR patient is asserted) exceeds a low target, the top-down
prediction gain (cue->patient_expected) for THAT block is scaled up until the recalled prediction cancels the familiar
assertion; iterate until every block's familiar read is at target. The controller nulls a SPIKING error (confirm
firing) by adjusting a SYNAPTIC gain, then the firing threshold is re-calibrated on the homeostatted circuit.
**Specificity is preserved BY CONSTRUCTION:** the prediction pathway is topographic + block-diagonal (block c's
prediction inhibits ONLY surprise block c), so boosting block c's prediction gain cancels the CONFIRM read for block c
(assert==expected==c) but leaves every CONTRADICT / NOVEL read untouched — those drive a DIFFERENT block j!=c, which
block c's prediction never inhibits. Measured directly: at every gain from 0.8 to 3.0 the contradict/novel rates stay
5-6 Hz UNCHANGED while confirm collapses to ~0.

## Result — 6 seeds 42/43/44/100/101/102 (SIM_BACKEND=numpy). Cites `research/findings/raw/_surprise_organ_homeostat/summary.json`.

**The residual CLOSES on every seed (`all_residual_closed = True`, 6/6):**
- **`het_vote_rate` 0.9375 -> 1.000 (8/8 on all 6 seeds).** Every genuinely-familiar edge is now voted (the 3 seeds
  44/100/101 that were 7/8 are now 8/8; the 3 that were 8/8 stay 8/8).
- **end-to-end parity restored: `consensus_acc` 1.000 == `host_recall_acc` 1.000 (all seeds).** The het organ no
  longer only ADDS abstentions; on these stored edges the bus now matches bare host recall exactly.
- **the fix is minimal + targeted.** Most blocks stay at the base prediction gain 0.80; only the weak block(s) are
  boosted (per-seed pred_gain max 0.80 / 1.11 / 1.31 / 1.72 / 1.41 / 1.12). Seed 43 (already 8/8) is a NO-OP
  (gains 0.80-0.80, 2 reps) — the equalizer intervenes ONLY where the residual exists. confirm_max per seed drops
  1.33->0.17, 0.52->0.35, 3.36->0.41, 5.61->0.23, 3.88->0.46, 1.68->0.46 Hz; the controller converges in 2-10 reps.

**SURPRISE SPECIFICITY holds perfectly (the anti-"vote on everything" cheat; `all_surprise_specificity_ok = True`):**
on a genuinely-NOVEL and a CONTRADICTING assertion the homeostatted organ STILL reads surprised on every stored fact
(`novel_registers` = `contradict_registers` = 1.000, all seeds). Mean confirm 0.35 Hz < threshold ~2.6 <= contradict
~5.1-5.5 / novel ~5.0-5.5 Hz. A homeostat that merely cranked a GLOBAL gain (or inhibited the whole surprise pool)
would suppress contradict/novel too and the organ would vote on everything — this control FAILS that cheat and it
PASSES with a large margin.

**Every [N] substrate + collapse control is preserved (all seeds):** `all_substrate_combines = True`
(ignite-when-voted 1.000, abstain-when-withheld 1.000); single-organ 0.000; het-dropped 0.000; leave-one-out-worst
0.000; DISAGREE 0.000; shuffle-off 0.000; het-ORGAN-lesion 0.000; workspace-lesion 0.000; the composer recall reflex
survives the workspace lesion at 1.000; `all_het_discriminate = True`; `all_moat_ok = True`; `all_seed_go = True`.

## What the anti-cheats establish
- SURPRISE-SPECIFICITY (the new control): the homeostat restores FAMILIAR-read cancellation WITHOUT touching the
  novel/contradict reads — the topographic block-diagonal prediction guarantees the boost is local to the confirm
  block. Establishes the organ did not "learn to vote on everything."
- HET-ORGAN LESION still collapses to 0.000: zeroing the prediction->surprise edges removes the (now-equalized)
  inhibition so CONFIRM fires as high as CONTRADICT -> organ H withholds -> collapse. The vote is still caused by the
  learned SPIKING prediction, not a fixed input artifact.
- DISAGREE still collapses to 0.000: a contradicting assertion still fires the mismatch high -> organ H withholds.
- The full [N] substrate battery (single / dropped / leave-one-out / shuffle-off / workspace-lesion + reflex-survives
  + moat + discrimination) is unchanged at its GO values — the homeostat touches ONLY the organ's per-block
  prediction gain, not the workspace/composer/consensus machinery.

## Honest scope + residuals (this is a de-risk that CLOSES a named rung; it is not a production wire)
1. **The homeostatic controller is a BUILD-TIME calibration loop** (host-orchestrated: measure per-block confirm ->
   strengthen the weak block's prediction gain -> repeat), exactly like the organ's EXISTING build-time threshold
   calibration and its Hebbian `train_expectation` loop. The fully-faithful version is an ONLINE spiking
   inhibitory/homeostatic-plasticity rule (Vogels 2011) driven by the co-active confirm spikes; this rides the SAME
   "learning/calibration is a host-run loop at build" burn-down the production organ already carries — it does not add
   a new shortcut, it equalizes a gain the build already sets by a constant.
2. **The prediction MAPPING is still a topographic prior** with Hebbian-learned + now homeostatically-equalized
   STRENGTH; a fully-learned all-to-all CA3 recall (2026-06-05-D-cue-recall-RESOLVED) is the unchanged separate next
   rung — this de-risk equalizes the STRENGTH, not the which-patient MAPPING.
3. **Co-residency.** The organ runs on ITS OWN circuit bridge alongside the composer + workspace — rides the one-brain
   merge burn-down, exactly as in [N].
4. **NOT "closed"** (per docs/TERMS.md): a default-off de-risk runner, not the shipped organ. Closure = adding the
   per-block homeostatic equalizer to the production `SurpriseProductionOrgan` build (additive to `ensure_built`) and
   the GNW bus production path (`webapp/gnw_bus_shadow.py`). This de-risk STRENGTHENS both the surprise organ (removes
   its precision boundary) AND the GNW-bus heterogeneous vote (lifts end-to-end parity to host).

## Files
- Runner: `research/runners/_surprise_organ_homeostat_derisk.py` (reuse-by-import: `SurpriseProductionOrgan` +
  `_hard_reset`/`_drive_read`/`measure_conditions` from the production organ / `_spiking_expectation_rpe_derisk`;
  [N]'s FULL gate via a monkeypatch of the drop-in `HeterogeneousOrganVote` -> `HomeostaticHeterogeneousOrganVote`;
  `from tools.lab import attributable_to, void_if`; NO `sim/` edit).
- 6-seed artifact: `research/findings/raw/_surprise_organ_homeostat/summary.json` (all_residual_closed True, 6/6).
- The residual this closes: `research/findings/2026-08-13-gnw-bus-heterogeneous-organ-GO.md` — its baseline
  `het_vote_rate` mean 0.9375 lives in `research/findings/raw/_gnw_bus_heterogeneous/summary.json` (the value this
  de-risk lifts to 1.000).
- The reused organ: `research/runners/surprise_production_organ.py` (spiking expectation-violation / familiarity
  monitor, D2 6/6 GO) + `research/runners/_spiking_expectation_rpe_derisk.py` (the mismatch circuit; its own
  wall-discipline note already names the divisive-normalization / gain-match precision companion this de-risk builds).
