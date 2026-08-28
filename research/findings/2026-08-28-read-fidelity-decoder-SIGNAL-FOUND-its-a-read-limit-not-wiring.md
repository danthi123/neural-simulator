---
type: finding
status: positive
date: 2026-08-28
verdict: The read-fidelity crux is a READ-FIDELITY LIMIT, not a no-signal/wiring problem — SETTLED. A decoder DOES separate generated-vs-perceived on the surprise->source_provenance F2 crux: GO_signal_found=True, n_seeds_any_separable=6/6, n_seeds_all_chance=0, and the neuron-identity shuffle anti-cheat is clean on EVERY combo on all 6 seeds (n_seeds_shuffle_ok_on_every_combo=6). The richest available per-neuron readout (a 10-bin spike-count profile that SUBSUMES rate/latency/dispersion) fed to a linear (logistic) AND a nonlinear (MLP) decoder, cross-validated, recovers the distinction that mean-rate, first-spike-latency, and ISI-CV/Fano dispersion ALL missed. So the signal is genuinely present in the DISTRIBUTED per-neuron pattern; the scalar spiking reads discard it. NEXT LEVER: a biological spiking read over the DISTRIBUTED pattern (population-vector / template-matching / distributed decode), not a scalar summary.
mechanism: linear + nonlinear decoder separability of generated-vs-perceived on the F2 crux (read-problem vs wiring-problem)
lane: read-fidelity
seed-waiver: 6-seed run (42/43/44/100/101/102) — this IS the 6-seed de-risk; GO_signal_found + shuffle-clean-6/6 are the result.
artifacts:
  - research/findings/raw/_read_fidelity_decoder_separability_6seed.json
runner: research/runners/_read_fidelity_decoder_separability_derisk.py
---

# Read-fidelity crux SETTLED as a READ problem: a decoder separates the pools (6/6, shuffle-clean) that rate/latency/dispersion all missed

Artifact: `research/findings/raw/_read_fidelity_decoder_separability_6seed.json` (numpy, 6 seeds; same trained cross-edge + same captured rasters iteration 1/2 used — no retraining confound).

## The question this closes

Three scalar spiking reads failed on this crux — mean-rate (0/6), first-spike-latency (0/6, clean instrument), ISI-CV/Fano dispersion (1/6). The dispersion finding repointed to: is the F2 "below floor" a READ-FIDELITY limit (a signal the reads miss) or a WIRING problem (no separable signal at all)? This de-risk answers it directly with a DECODER over the full per-neuron pattern.

## Result — a decoder separates; it is a READ problem

- **`GO_signal_found = True`**; **`n_seeds_any_separable = 6`** of 6; **`n_seeds_all_chance = 0`**.
- **`n_seeds_shuffle_ok_on_every_combo = 6`** — the neuron-identity permutation anti-cheat is clean on EVERY decoder/feature combo on all 6 seeds (the separability is a genuine identity-dependent signal, not a shuffle artifact).
- Feature: a 10-bin per-neuron spike-count profile (subsumes rate, latency, dispersion). Decoders: L2 logistic (linear) + an 8-hidden MLP (nonlinear), K=5 x 5-repeat CV with train-fold-only standardization (no leakage).

## What this settles + the next lever (NO-DEFER)

The signal IS there: a decoder over the distributed per-neuron pattern recovers generated-vs-perceived on every seed, shuffle-clean, where three scalar spiking reads all failed. So the F2 crux is a genuine READ-FIDELITY LIMIT — the scalar reads (rate/latency/dispersion) discard the distributed pattern that carries the signal. This closes the "is there a signal?" question (yes) and re-focuses the read-fidelity arc: the next lever is a BIOLOGICAL spiking read over the DISTRIBUTED code — a population-vector read (project onto a learned template direction), a matched-filter / template-matching readout, or a small decorrelating read layer — that a spiking substrate can implement, rather than a 4th scalar summary. HONEST RESIDUALS (from the artifact): the 10-bin resolution + the two decoder families + the CV knobs are host-chosen, so this is strong evidence, not proof that a signal the reads miss is ALWAYS recoverable; and the 1-seed smoke had flagged that only ~half the decodable separation vanished under lesion (part is static structural asymmetry between the two hard-wired pools), so the biological read should target the cross-edge-attributable component specifically.
