---
type: finding
status: contributing
date: 2026-08-01
mechanism: deep-credit-on-spikes
artifacts:
  - research/findings/raw/gap4/realspikes/pregate_aggregate.json
---

# gap#4: the degenerate-forward-pass boundary is CLEARED for the movable-plateau substrate — a REAL-spikes forward pass gives input-dependent, reproducible codons (pre-gate 3/3)

<!--derived-->
**One-line verdict:** the first, cheapest-first step of the on-bridge SPIKING port of the unsupervised movable-
plateau rule (gap#4's only positive credit signal). The rate 5/6 result reads the plateau via a **boolean-hold
reset-read** (`_prime_from_winners`: reset soma, hold winner features as booleans, ZERO input current) — a genuine
rate/analytic stand-in. A prior finding (2026-07-10) found the on-bridge spiking forward pass **degenerate** (hidden
fires input-independently, near-silent) and called it the boundary that gates any real-spikes hidden-credit test.
This probe replaces the boolean hold with a **REAL spiking forward pass** — drive the active feature neurons via
`cp_external_input_current`, integrate a real 30-step window, let them SPIKE, propagate through the coincidence
pathway to the columns' real plateau — and the pre-gate **PASSES 3/3 seeds**: input-dependent firing
(feat_spike_std ≈ 2.37, col_margin_std ≈ 11.1 across inputs — columns respond to the input, not tonic-pinned) AND
reproducibility 1.0, at drive = 1200 pA with the **reservoir weights unchanged** (no forward-weight scaling
needed). The boundary is **cleared for this substrate** — a positive correction to the 2026-07-10 framing. No
`sim/` edit.

Artifact: `research/findings/raw/gap4/realspikes/pregate_aggregate.json` (backend numpy/CPU). Runner:
`research/runners/_gap4_realspikes_pregate_probe.py`.

## Result — pre-gate, seeds {42,43,44}, drive 1200 pA, 30-step window

<!--derived-->
| check | value | pass |
|---|---|---|
| (i) input-dependent feature firing (spike-count std across inputs) | 2.37 | yes (> 0) |
| (i) input-dependent column plateau (margin std across inputs) | 11.09 | yes (> 0) |
| (ii) reproducibility (codon correlation, two identical presentations) | 1.000 | yes (≥ 0.8) |
| mean feature rate / step | ~0.074 | genuine spiking (not silent) |
| PRE-GATE | — | **PASS 3/3** |

The reproducibility is 1.0 because this substrate keeps noise off (`enable_ou_process`/`conductance_noise`/
`parameter_heterogeneity` = False, `deterministic_transpose_matvec` = True) — the SAME config as the rate 5/6 result,
so it is honest for the port (the reset-read's 0.07 collapse came from a noisy rate-settle read, which this config
does not use). Still PASS at a lower drive (600 pA): feat_spike_std 1.45, col_margin_std 2.09 — not knife-edge.

## The read is genuinely different from the stand-in (not a re-labeling)

<!--derived-->
The real-spikes column margins (mean 37.9) are not just rescaled versions of the boolean-hold reset-read (mean
11.6): per-input, the two codons are **anti-correlated** (−0.14 to −0.38). So the real spiking forward pass
produces its OWN column representation, distinct from the rate stand-in — which means the port is a real test, and
the rate 5/6 result does **not** transfer by assumption.

## What this settles, and the decisive next test

<!--derived-->
This is **necessary, not sufficient**. It removes the one boundary (of three prior on-bridge spiking findings) that
actually gated the port — the degenerate/near-silent forward pass — for the movable-plateau substrate at the AMPA
operating point. The other two priors contribute lessons, not blockers: the 2026-06-06 whitening boundary was a
different (lateral anti-Hebbian) rule; the 2026-07-08 missing-attractor boundary was a recurrent autoassociator.
What is NOT yet answered: does the UNSUPERVISED movable-plateau covariance rule, trained and read on **real spikes**
(pre-activity = real feature spike counts), still beat a frozen on-bridge reservoir on held-out inheritance — the
real-spikes analogue of the rate 5/6? Because the real-spikes codons are anti-correlated with the stand-in, that
must be measured fresh, not assumed. **Next: the 6-seed real-spikes credit comparison** (credit-trained vs frozen
on-bridge reservoir, deep_credit_share > 0, with the full anti-cheat battery + the load-bearing reproducibility
gate) — the decisive test of whether gap#4's only positive signal survives the port to real spikes.
