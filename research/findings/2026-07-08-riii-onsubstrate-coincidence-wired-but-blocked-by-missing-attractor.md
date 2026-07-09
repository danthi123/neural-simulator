# R-iii on-substrate rung — the dendritic-COINCIDENCE plateau is CORRECTLY WIRED on the real bridge (weighted coincident drive c_drive ≈ 3.0, non-zero), but CA3 completion is blocked ONE LEVEL DEEPER than the plateau: the ca3->ca3 recurrent autoassociator never forms a SPECIFIC within-ensemble attractor — a held-out MEMBER receives the SAME weighted recurrent drive from the cue (2.99) as a random NON-STORED neuron (2.98). There is no learned structure for the plateau to amplify. The surpass is reframed: FORM the attractor first (rate-Hebbian symmetric-co-activity recurrent potentiation — the documented CYCLE-95/96 mechanism; STDP is silent at the Δt≈0 of synchronous co-firing), THEN the plateau (with a calibrated threshold) completes. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_riii_ca3_coincidence_completion_derisk.py` (+ `_riii_coinc_aggregate.py`). numpy-smoke / GPU-real. NO `sim/` edit (flips `coincidence_detector=True` on the returned `ca3->ca3` RegionPathway; enables the guarded `enable_coincidence_detection` + `coincidence_weighted_drive`).
**Verdict:** DIAGNOSTIC (decisive, converged) — the plateau machinery works; the blocker is the missing recurrent attractor, which names the next mechanism. NOT a wall.

## What this rung set out to do
CYCLE 1065 (minimal numpy model) proved a supra-linear dendritic plateau + synaptic clustering COMPLETES a partial cue where a linear point-neuron read-out fails (0.89 vs 0.26). This rung realizes it on the REAL spiking substrate via the project's EXISTING `fused_coincidence_plateau` (a per-neuron supralinear NMDA-spike plateau; `enable_coincidence_detection`, byte-inert when off) — pure reuse, routing the `ca3->ca3` recurrents through the plateau in WEIGHTED-DRIVE mode so a member's LTP-strengthened partners cross the switch where a non-member's weak inputs do not.

## The decisive measurement (ran the mechanism diagnostic to convergence)
On the trained bridge (seed 42, n_ca3=150, density 0.5, 100 train events), the weighted coincident drive the plateau actually reads:
```
c_drive[held-out MEMBER] = 2.99      c_drive[NON-STORED] = 2.98      (identical -> no separation)
held-out completion:  COINC-ON = 0.026  ==  LINEAR-OFF = 0.026  ==  NO-TRAIN = 0.026   (plateau has NO effect)
```
Two facts, in order of importance:
1. **c_drive shows NO member-vs-non-member separation (2.99 ≈ 2.98).** A held-out ensemble MEMBER receives the same weighted recurrent drive from the cue as an arbitrary NON-STORED neuron. The ca3->ca3 recurrent LTP did NOT write a specific within-ensemble attractor. The plateau has nothing specific to amplify -> it would fire members and non-members alike if triggered (indiscriminate spread, not pattern completion).
2. **`k_thresh` was miscalibrated (18 ≫ c_drive≈3):** the plateau switch `sigmoid(gain·(c_drive−k_thresh))` = `sigmoid(2·(3−18))≈0`, so it never fires — which is why COINC-ON == LINEAR-OFF. This was a red herring: calibrating `k_thresh`≈3 would trigger the plateau, but WITHOUT (1) fixed it fires indiscriminately (no specificity). The real blocker is (1).

## Why this reframes (and sharpens) the CYCLE-1064 boundary
CYCLE 1064 established "the recurrents transmit but a partial cue completes nothing across weight×density×drive." The ROOT CAUSE is now pinned: not the point-neuron summation limit per se, but that **the recurrent autoassociator never forms a specific attractor** — the training strengthens weights UNIFORMLY (or not at all), never the member-specific structure a Marr autoassociator needs. This is why more uniform weight (CYCLE-1064's ca3_weight→200) and the dendritic plateau both fail: neither creates the missing STRUCTURE. It also explains the long D.13/SWR history — the "cos 0.748" was always the drive overlap, because no attractor existed to complete.

The mechanism convergence with a KNOWN project finding: the stored ensemble members are driven by the SAME input and fire ~SYNCHRONOUSLY (Δt≈0), and **STDP cannot potentiate symmetric co-occurrence at Δt≈0** (CLAUDE.md CYCLE 95-96: "measured 656k events / 0 weight change at Δt≈0; STDP is the WRONG rule; rate-Hebbian is required"). So the ca3->ca3 recurrent LTP is silent, and no attractor forms.

## The reframed surpass (the next mechanism, cheap-first)
The plateau is NECESSARY (CYCLE 1065) but needs a SPECIFIC attractor to read. The next mechanism is to FORM that attractor:
1. **Rate-Hebbian symmetric-co-activity potentiation of the ca3->ca3 recurrents** (the documented CYCLE-95/96 rule: co-active pairs potentiate regardless of order) -> within-ensemble weights rise above baseline -> c_drive[member] >> c_drive[non-member]. Verify by re-measuring the c_drive separation (the same diagnostic, now expecting a gap).
2. **Then re-test BOTH read-outs:** does the LINEAR point-neuron complete once a real attractor exists (which would show the CYCLE-1064 boundary was the missing-attractor, not the point-neuron limit)? And/or does the plateau (calibrated `k_thresh` between the member/non-member c_drive) complete SPECIFICALLY? Either way the honest story sharpens.

## Files
`research/runners/_riii_ca3_coincidence_completion_derisk.py` (routes ca3->ca3 through the plateau; the direct c_drive mechanism diagnostic), `_riii_coinc_aggregate.py`. Prior: `2026-07-08-riii-dendritic-completion-surpass-cheap-first-GO.md` (1065, minimal model), `2026-07-08-riii-DEFINITIVE-ca3-partial-cue-completion-fails-across-param-space.md` (1064). NEXT: the rate-Hebbian recurrent-attractor fix + re-measure c_drive separation, then re-test linear vs plateau completion.
