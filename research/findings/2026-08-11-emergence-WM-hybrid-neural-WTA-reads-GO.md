---
type: finding
status: go
date: 2026-08-11
mechanism: NEURAL-WTA CHANNEL READS FOR THE WM+HTM HYBRID — the rung-3b separate-channel hybrid's TWO host np.argmax channel reads (subject_hat = argmax over the WM slot's cp_firing_states; class_hat = argmax over the HTM's clsrd apical drive) are replaced by the rung-2 emergent DOWN-RAMP release-of-inhibition WTA, read from SPIKES. The learned dendritic conjunction then combines the two WTA-selected channel winners. The ENTIRE verb read path — channel SELECTION and combination — is now neural + reads spikes; NO host argmax remains in the read-out.
lane: emergence engine / working memory (rung 3c — the hybrid's last host-argmax read residual, closed)
verdict: 6-SEED GO — replacing BOTH host-argmax channel reads with the emergent neural-WTA PRESERVES the hybrid GO at parity (in fact +0.021): held-out exact HYBRID-WTA 0.995 [min 0.969] (chance 0.125) vs the argmax-reads reference 0.974 [min 0.938], subject PRESERVED 1.000 [min 1.000], class 0.995. It BEATS both single systems (HTM-alone 0.224, WM-alone 0.516) by +0.48, clearing max+0.20 (0.716). All load-bearing teeth of the neural read path bite: lesion-WM-channel → 0.247 (≈ HTM-alone), lesion-HTM-channel → 0.479 (≈ WM-alone), lesion-the-hold → 0.276 (the WTA subject read reads the slot's SPIKING sustain, external input asserted zero), the UNTRAINED conjunction → 0.096 (≈ chance; the bind is LEARNED), subject-shuffle → 0.000 (no leakage). NO host argmax in the verb read path (AST-asserted + grep-confirmed). Honest sub-negative: the WTA self-calibration is NOT load-bearing on this CLEAN WM latch (lesion-the-WTA-selfcalib → 0.995) — a fair fixed cut isolates the clean winner; the self-calibration is load-bearing in the rung-2 BLUR/allocation regime, not the clean-read regime. (aggregate means over 6 seeds; per-arm values in the body + neuralwta_6seed.json) <!--derived-->
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_emerge_wm_hybrid_neuralwta_derisk.py
artifacts:
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/neuralwta_6seed.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_42.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_43.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_44.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_100.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_101.json
  - research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_102.json
instrument: reuse-by-import of the rung-3b separate-channel hybrid (2026-08-11-emergence-WM-hybrid-separate-channel-GO; stream + WM slot + class channel + learned conjunction + the argmax reference arms) and the rung-2 emergent-WTA (2026-08-11-emergent-neural-WTA-slot-allocation-GO; the D3 slow-NMDA attractor bank + the down-ramp release-of-inhibition competition). The ONLY change vs rung-3b is HOW subject_hat / class_hat are read (host np.argmax → emergent neural-WTA read from spikes); the conjunction, lesions and controls are IDENTICAL, so any accuracy difference is attributable to making the reads neural. The WTA is driven by the channel's SPIKE-derived evidence (slot pool rates / clsrd apical drive) via a labelled-line identity projection (the same accepted host-projection scope as the rung-2 barcode→pool + the rung-3b labelled-line wm[]/clsrd[] projections); the WTA SELECTION and the read are neural (spikes). SIM_BACKEND=numpy; NO sim/ edit. Precedent for a host-argmax→spiking-WTA read replacement: 2026-06-20-burndown-1-onebrain-spiking-cleanup (the OneBrainComposer cleanup selection), rungB1b-neural-role-wta-GO.
---
<!--derived-->

# Neural-WTA channel reads — the WM+HTM hybrid's last host-argmax read residual is closed: hybrid 0.995, subject PRESERVED, 6-seed GO

The rung-3b separate-channel hybrid (`2026-08-11-emergence-WM-hybrid-separate-channel-GO`) is a WORKING WM+HTM neural
fusion (held-out 0.974, subject preserved 1.000) with ONE named host residual: the per-channel WINNER READS are host
`np.argmax` — `subject_hat` = argmax over the WM slot's `cp_firing_states` pool rates, `class_hat` = argmax over the
HTM's `clsrd` apical drive. This de-risk replaces BOTH with the rung-2 emergent neural-WTA, so the ENTIRE verb read
path — channel SELECTION and the CONJUNCTION combine — is neural + reads spikes, and it holds the GO.

## The mechanism — the emergent down-ramp WTA reads each channel from spikes (no host argmax)

<!--derived-->
Each host-argmax channel read is replaced by a `NeuralWTARead` over a K-pool D3 slow-NMDA attractor bank. The channel's
SPIKE-derived evidence is injected as graded per-pool external current via a labelled-line identity projection, then a
DOWN-RAMP release-of-inhibition competition (the rung-2 mechanism) resolves a clean one-of-K winner WITHOUT a hand-set
cut: a pooled subtractive inhibition common to all pools starts HIGH (all silent) and is RELEASED step by step; the
first pool to escape (highest evidence) wins; the ramp STOPS the instant exactly one pool is active. The winner is READ
FROM SPIKES — a threshold on the post-competition HOLD rate (drive removed), the single pool the competition left
latched — NEVER `np.argmax`.

- **subject_hat** — evidence = the WM slot's K held-pool firing RATES (after WRITE+HOLD across the fillers, external
  input asserted zero). The WTA winner → deref (`subj_of_slot`) → subject.
- **class_hat** — evidence = the HTM engine's `clsrd` apical-drive vector (`cp_v_apical`, the substrate's own branch
  prediction). The WTA winner → class.
- the **learned dendritic conjunction** bridge (already neural, rung-3b) then combines `{wm[subject_hat] ∪ clsrd[class_hat]}`
  → the verb column with the max apical plateau (the unique double-driven cell).

The read path is argmax-free by construction: `_assert_no_host_argmax_in_read_path()` AST-inspects
`NeuralWTARead.select`, `class_evidence`, `slot_carry_subject_rates`, and `conj_read` (the conjunction reads the
substrate's apical plateau via `np.max` per verb column — a graded dendritic read of the cell that fired, not an argmax
classifier) and RAISES if a host argmax leaks in. The only `np.argmax` in the runner is the `subj_am` REFERENCE arm
(the rung-3b read being surpassed), which is not on the `hybrid_wta_reads` path.

## Result — 6-seed (`research/findings/raw/_emerge_wm_hybrid_neuralwta/neuralwta_6seed.json`; chance 0.125; n_subj=4 n_fill=8 n_cls=2 L=3, held-out NOVEL fillers)

<!--derived-->
(all numbers below are 6-seed aggregate means / mins from `neuralwta_6seed.json`; the derived arithmetic like +0.021 is difference-of-means)

| arm | held-out exact | subject | class |
|---|---|---|---|
| **hybrid_wta_reads (candidate — neural WTA reads)** | **0.995** [min 0.969] | **1.000** [min 1.000] | 0.995 |
| hybrid_argmax_reads (rung-3b reference, re-run) | 0.974 [min 0.938] | 1.000 | 0.974 |
| HTM-alone | 0.224 | — | — |
| WM-alone | 0.516 | — | — |
| n-gram HELD-OUT floor | 0.263 | | |
| chance | 0.125 | | |

**Making the reads neural cost ZERO accuracy — it slightly IMPROVED (+0.021).** The direct per-channel read comparison
shows why: the WTA class read is 0.995 vs the host argmax-class read's 0.964 — on the harder seeds (44/100/101) the
argmax-with-threshold class read (`class_read` returns −1 below its fixed apical floor) missed some classes, while the
WTA's down-ramp resolves a winner regardless of the drive magnitude (it tracks the operating point the fixed threshold
could not). The WTA subject read is 1.000 (= argmax); subject preserved at 1.000.

### Load-bearing teeth (all bite)

| lesion / control | exact | reads |
|---|---|---|
| lesion-WM-channel (subject channel ablated → class only) | 0.247 | ≈ HTM-alone 0.224 |
| lesion-HTM-channel (class channel ablated → subject only) | 0.479 | ≈ WM-alone 0.516 |
| lesion-the-hold (recur=0 slot → subject evidence is noise) | 0.276 | the WTA subject read reads the slot's SPIKES |
| untrained conjunction (the bind is LEARNED, not host wiring) | 0.096 | ≈ chance |
| subject-shuffle (deref permuted → wrong subject) | 0.000 | no positional/topic leakage |
| hold-alive (with external input ASSERTED zero) | 0.088 | zero-input span verified |

- **WTA reads read spikes:** lesion-the-hold collapses the hybrid from 0.995 → 0.276 (the WTA subject read is driven by
  the slot's held-pool rates; killing the recurrence kills the bump → the evidence is noise → the WTA picks garbage).
  The slot sustains the latch with external input ASSERTED identically zero across the hold+read span.
- **The bind is learned, not a host lookup:** the SAME WTA reads on an UNTRAINED conjunction bridge → 0.096 ≈ chance.
- **Selectivity (read from spikes):** subject WTA 1.000, class WTA 0.994 — a clean one-of-K, not a soft argmax over a blur.

### Honest sub-negative — the WTA self-calibration is NOT load-bearing on this clean latch

`lesion-the-WTA-selfcalib` freezes the release ramp at a fair hand-set cut (fixed_inh_frac=0.45, same settle budget,
only the adaptive release removed) → 0.995 (does NOT bite). On this regime the WM latch is CLEAN (slot decode 1.000,
selectivity 1.000), so a fair fixed cut isolates the winner and the per-read self-calibration adds nothing. This is
consistent — and it precisely maps where the self-calibration DOES matter: the rung-2 BLUR/allocation regime (RUNG6e's
noise-picked winner, margin ~0.31), where a hand-set cut cannot serve the varying per-entity margin. The load-bearing
components of the neural read path HERE are lesion-the-hold + the two channel lesions + the learned conjunction, not the
self-calibration. (The WTA still helps over the host argmax read — see the +0.031 class-read gain above — via the
down-ramp resolving a winner where the fixed apical threshold returned −1.)

## Scope (honest)

- **What is now neural:** the two channel SELECTIONS (release-of-inhibition WTA read from spikes) AND the combination
  (learned dendritic conjunction). **NO host argmax in the verb read-out** (AST-asserted). This closes the rung-3b
  hybrid's one named host-read residual.
- **What is still host (unchanged from rung-3b/rung-2, the accepted de-risk scope):** the WTA is driven by the channel's
  spike-derived evidence via a labelled-line identity projection (the same status as the rung-2 barcode→pool projection
  and the rung-3b wm[]/clsrd[] labelled-line projections); the compositional stream + subject↔slot binder are the host
  scaffold/environment; the WM afferent during TRAINING is the true subject (the WM working in development). The
  on-substrate spiking lateral-inhibitory realisation of the WTA is the named next rung (rung-2's carried residual).
- **Not "fully spiking":** the read-out is spiking with no host argmax, but the driving projections are host labelled-lines
  — this is a host-argmax-elimination result, not an end-to-end fully-spiking claim.

## Reproduce

```
# 6-seed decisive (fan one seed per process, then merge)
for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
  research.runners._emerge_wm_hybrid_neuralwta_derisk --seeds $s \
  --out research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_$s.json & done ; wait
SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_neuralwta_derisk \
  --merge-from research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_*.json \
  --out research/findings/raw/_emerge_wm_hybrid_neuralwta/neuralwta_6seed.json
```
