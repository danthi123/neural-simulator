---
type: finding
status: negative
date: 2026-08-27
verdict: 6-seed reduced-scale NO-GO for softmax-confidence / read-gain calibration as the fix for the mouth readout's ||W||->cap runaway -- the root cause is NOT a magnitude/confidence miscalibration (fixable by any scalar rescale) but a STRUCTURE-SELECTIVE collapse in the substrate's raw graded-conductance margin -- a random/incoherent weight direction reads with corr ~0.95 against the ideal linear map, but the STRUCTURED target direction (head_w, at ANY scale from 10% to 100% of its magnitude) reads at corr ~0.00, both globally and row-centered; a self-referential adaptive-gain recalibration (an AGC candidate) does not move ||W|| off the 40 cap in 6/6 seeds and makes weight_cosine slightly WORSE, not better
mechanism: mouth read-SNR decoder direction -- softmax read-CONFIDENCE / gain calibration as the fix for the weight-norm runaway (gap#4 / #80, follow-on to the fullscale-substrate-gap + decoder-direction findings)
lane: E-language-mouth-read-snr
artifacts:
  - research/findings/raw/_wkv_softmax_confidence/diag_s42.json
  - research/findings/raw/_wkv_softmax_confidence/diag_safe_s42.json
  - research/findings/raw/_wkv_softmax_confidence/diag_reconcile_s42.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s42.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s43.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s44.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s100.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s101.json
  - research/findings/raw/_wkv_softmax_confidence/shorttrain_s102.json
runner: research/runners/_wkv_mouth_readout_softmax_confidence_derisk.py
---

# Mouth read-SNR (#80): softmax-confidence / read-gain calibration is a 6-seed reduced-scale NO-GO -- the runaway is a structure-selective read-fidelity collapse, not a magnitude/confidence problem

Artifact: `research/findings/raw/_wkv_softmax_confidence/diag_s42.json` (plus the 6-seed `shorttrain_s{42,43,44,100,101,102}.json`, each bundling the diagnosis + the fixed-vs-adaptive-gain short training probe).

## The question (do not re-derive)

<!--derived-->
`2026-08-27-mouth-readsnr-magnitude-knob-NULL-and-fullscale-substrate-gap.md` established: at full data
scale the substrate-forward learned readout runs `||W||` to the 40 cap (wcos 0.136, recov 0.37) while the
host-proxy (exact linear map) converges naturally at `||W||~24` (wcos 0.40, recov 0.86) -- a `||W||->cap`
runaway. This arc's mission: does the substrate's softmax under-read the true margin (an under-CONFIDENCE
problem fixable by rescaling), and does recalibrating read gain/confidence make `||W||` converge near ~24?

## Method: 3 instruments, all reuse-by-import, memory-bounded (B=8, sub-read-window=64)

A first pass at production scale (B=48, sub-read-window=360) completed twice cleanly, but a subsequent
invocation was killed by the machine owner at 31.5GB RSS (shared-machine OOM risk, matching a prior
2026-08-26 crash). All results below are at the REDUCED, memory-safe scale already established as valid for
this wall (`_wkv_mouth_readout_init_scale_sweep_derisk.py`'s B=8/read-window=64 reference). The qualitative
signal (near-zero correlation, scale-independent) was already visible in the two production-scale runs before
the kill and is reproduced identically at the safe scale, so the conclusion is not a scale artifact.

1. **Direction/magnitude decomposition** (`_measure_gain`): the SAME regression `_calibrate_gain` performs
   (substrate margin vs the exact host-linear margin, on a caller-supplied probe weight), run on 5 probes: a
   random probe at the current calibration scale (A), a random probe rescaled to `||head_w||` (B) and to the
   w_target cap (C), head_w itself (D), and head_w's direction rescaled to A's norm (E). Also a row-centered
   correlation (each read subtracts its own per-position mean across V before correlating) -- the quantity a
   softmax-onehot rule actually sees, since softmax is invariant to a per-row additive shift.
2. **Scale sweep along head_w's direction** (0.1x/0.25x/0.5x/0.75x/1.0x, reusing one built network -- cheap):
   localizes whether any degradation is magnitude-dependent (saturation) or present at all scales (structure).
3. **Order control**: probe A repeated at the END of the ~10-probe sequence, to rule out state-accumulation
   across successive reads on one bridge as the cause of a later probe's bad correlation.
4. **Reconciliation**: reproduces the prior gradalign finding's exact quantity (`cos(g_sub, g_host)` at
   W=head_w, `g = (softmax(logits)-onehot)^T @ h / B`) on the SAME batch, plus the norm of the
   `-onehot^T@h/B` term shared identically by `g_sub` and `g_host` regardless of read quality.
5. **Fix test**: a self-referential ADAPTIVE gain -- periodically (every 20 grad steps) re-measure gain using
   the CURRENT `W_hat` as its own probe (no head_w, no labels; a physical re-calibration exactly like the
   existing one-time `_calibrate_gain`, styled after Turrigiano/Carandini-Heeger gain-control homeostasis) --
   trained head-to-head against the FIXED (current, one-time) calibration, 2 epochs, `n_train_pos=1200`.

## Result 1: the structured direction reads as near-zero correlated noise, at every scale, not an order artifact

<!--derived-->
Mean over 6 seeds (42/43/44/100/101/102): `corr(random probe A, ideal map)` = **0.9465** (row-centered
0.9687); `corr(head_w itself, ideal map)` = **-0.0006** (row-centered 0.0102) -- essentially zero, a >900x
collapse relative to the random-direction read. The scale sweep along head_w's OWN direction (0.1x through
1.0x of its magnitude) stays flat near zero at every fraction (seed 42: 0.0292/0.0304/0.0308/0.0298/0.0331) --
the collapse is present already at 10% of the target's magnitude, so it is NOT a saturation-with-scale effect.
The order control (probe A repeated after 10 head_w-related reads) reproduces the FIRST A read almost exactly
(mean 0.9453 vs 0.9465) -- **not** an artifact of read-sequence state accumulation.

## Result 2: the prior "near-ideal gradient" (cos ~0.9928) is inflated by a read-independent shared term <!--derived-->

<!--derived-->
Reproducing the gradalign finding's exact quantity on the SAME batches: `cos(g_sub, g_host)` at W=head_w =
**0.883** mean (range 0.85-0.90 across 6 seeds) -- in the same high-alignment ballpark as the prior finding's
0.975, confirming this is not a methodology bug. But the `-onehot^T@h/B` term -- driven ENTIRELY by the known
label, present identically in `g_sub` and `g_host` regardless of what the substrate read carries -- has norm
**94.9% of `||g_sub||`** on average (93.8-97.4% across seeds). The substrate-read-dependent component is a
small residual riding on a dominant, read-independent, label-driven term. High gradient cosine at head_w is
therefore consistent with -- and now explained by -- a raw read that carries almost no usable information
about the true target, resolving the apparent tension with this finding's Result 1 rather than contradicting it.

## Result 3: adaptive/self-referential gain recalibration does not close the gap -- 6/6 NO-GO

<!--derived-->
| seed | fixed \|\|W\|\| | adaptive \|\|W\|\| | fixed wcos | adaptive wcos | fixed recov | adaptive recov |
|---|---|---|---|---|---|---|
| 42 | 40.0 | 40.0 | 0.1194 | 0.1164 | 0.3457 | 0.3457 |
| 43 | 40.0 | 40.0 | 0.1430 | 0.1330 | 0.3576 | 0.3639 |
| 44 | 40.0 | 40.0 | 0.1302 | 0.1234 | 0.2640 | 0.2462 |
| 100 | 40.0 | 40.0 | 0.1261 | 0.1201 | 0.4019 | 0.3991 |
| 101 | 40.0 | 40.0 | 0.1234 | 0.1171 | 0.3972 | 0.4010 |
| 102 | 40.0 | 40.0 | 0.1249 | 0.1163 | 0.3517 | 0.3516 |
| **mean** | **40.0** | **40.0** | **0.1278** | **0.1211** | **0.3530** | **0.3513** |

`||W||` hits the 40.0 cap in **0/6** seeds under either fixed or adaptive gain -- the adaptive recalibration
never moves the norm toward the ideal ~24 (this is a reduced-budget, 2-epoch/1200-position probe, so absolute
recov numbers are not comparable to the full-scale 0.37/0.86 ceiling pair; the qualitative `||W||`-and-wcos
comparison between the two calibration modes is the decisive signal here). `weight_cosine` is WORSE under
adaptive gain in **6/6** seeds (mean 0.1278 -> 0.1211); `hostlinear_recov` is flat-to-slightly-worse (mean
0.3530 -> 0.3513, better in only 2/6 seeds by <0.004). The adaptive gain itself drifts toward small/negative
values late in training (e.g. seed 42: 10.5 -> -0.19) because it is self-referentially calibrated against a
`W_hat` that is itself in the near-zero-correlation regime -- a feedback loop that cannot bootstrap out of a
correlation floor. `byte-identical-off` holds by construction: the recalibration branch is gated by
`regain_every > 0`, so `mode="fixed"` never enters it and is identical to the production one-time-calibration
path.

## External source (live literature, this arc)

<!--derived-->
Louie, Grattan & Glimcher (2011), J Neurosci 31(29):10627-39, "Reward value-based gain control: divisive
normalization in parietal cortex", https://doi.org/10.1523/JNEUROSCI.1237-11.2011 (PMC3285508) -- the real
biological gain-control/divisive-normalization mechanism the adaptive-gain candidate was modeled after
(neurons re-normalize their response gain against the local population's own activity). It describes gain
control as RENORMALIZING an already-informative signal against context; it does not claim gain control can
recover information from a signal that carries none, which is consistent with this finding's result. Also
consulted: Kock et al. (2022), "Confidence Histograms for Model Reliability Analysis and Temperature
Calibration" (temperature scaling as a scalar post-hoc softmax fix) -- the negative result here follows
directly from temperature/gain scaling being scale-invariant to Pearson correlation, so it cannot repair a
near-zero correlation regardless of how it is calibrated.

## Verdict + redirect

**NO-GO, 6/6 seeds, reduced scale.** The mission's premise -- an under-confident but well-aligned read that a
gain/temperature rescale can fix -- is refuted by direct measurement: the raw substrate margin for the
STRUCTURED target direction is not under-scaled, it is uncorrelated (corr ~0, both globally and row-centered,
at every tested magnitude from 10% to 100%), while incoherent/random directions of the same or larger norm
read with high fidelity (corr ~0.95). No scalar (gain, temperature, AGC) can repair a near-zero correlation,
since Pearson correlation is invariant to any positive rescaling. This is a stronger, more specific
characterization than "under-reads": the wall is a **structure-selective graded-conductance read-fidelity
collapse**, present specifically for the coherent, correlated weight patterns a trained decoder needs (and
absent for incoherent ones), not a global saturation or an under-confidence artifact. Per the CLAUDE.md wall
reframe, the natural next question -- "what does the real system run alongside this that we substituted a
constant for" -- points at **structure-dependent conductance/driving-force interaction** (a real, biologically
motivated nonlinearity: correlated synaptic drive onto shared postsynaptic pools does not summate the way
independent random drive does), not at a confidence/gain homeostat. **Banked. Next lever, per the mission's
own named fallback: the queued dendritic-objective decisive** (`research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic`,
already staged) -- it changes the OBJECTIVE (per-unit sigmoid teacher vs cross-unit softmax) and, per the
2026-08-27 decoder-direction finding, "keeps the same near-ideal basal read, so it can only help via the
objective route, not a read fix" -- exactly what this finding independently confirms is needed. A second
candidate worth naming for the SAME reason: an objective that reads the pre-softmax margin directly (an
MSE/regression teacher matched to the SUBSTRATE's own achievable range) rather than forcing it through a
scale-sensitive softmax-onehot classification, since Result 2 shows softmax's label-driven term is currently
masking, not exploiting, whatever weak signal the read does carry.

## Files

- `research/runners/_wkv_mouth_readout_softmax_confidence_derisk.py` -- diagnosis (direction/magnitude
  decomposition, scale sweep, order control, gradient reconciliation) + fixed-vs-adaptive-gain short-train
  probe. Additive, no `sim/` edit, reuse-by-import of `BatchedSubstrateReadout` / `_calibrate_gain` /
  `_softmax_rows` / `_positions` / `_load_eval` / `WKVReadout` (all unmodified).
