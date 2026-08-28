---
type: finding
status: negative
date: 2026-08-28
verdict: UNDEFINED. A NON-RATE first-spike-latency read (normalized onset-fraction; Thorpe 2001, Gollisch-Meister 2008) did NOT lift the surprise->episodic F2 crux above floor — latency arm 0/6 PASS, the same floor-miss shape as the CURRENT mean-rate read (also 0/6), on the SAME trained cross-edge + SAME lesion event + SAME simulated trajectory (no retraining-noise confound). CRUCIALLY the read is NOT yet trustworthy: the shuffle-identity anti-cheat collapses on only 3/6 seeds (should collapse on all 6), so on 3 seeds the "read" survives a neuron-identity shuffle = an instrument ambiguity, not a validated null. Per NO-DEFER this is a verdict on THIS latency method + instrument, NOT on the capability (a non-rate read that defeats rate saturation).
mechanism: non-rate first-spike-latency read vs mean-rate read on the surprise->episodic F2 crux
lane: read-fidelity
seed-waiver: 6-seed run (42/43/44/100/101/102) — this IS the 6-seed de-risk; the negative + the 3/6 anti-cheat ambiguity are the result.
artifacts:
  - research/findings/raw/_read_fidelity_nonrate_latency_derisk_6seed.json
runner: research/runners/_read_fidelity_nonrate_latency_derisk.py
---

# Non-rate first-spike-latency read on the surprise->episodic crux — UNDEFINED (0/6, and the anti-cheat is ambiguous 3/6)

Artifact: `research/findings/raw/_read_fidelity_nonrate_latency_derisk_6seed.json` (numpy/CPU, 6 seeds, run locally — the remote pool could not run it, provision blocked by stranded node files).

## What this attacked

The surprise->episodic F2 crux ([`2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED`](2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md)) saturates: its margin (`rate_generated - rate_perceived`, a mean-firing-rate read) stays flat while the trained cross-edge weight grows 80-155x — attributed to Sanzeni/Histed/Brunel 2020 refractory-period rate compression. The named untested fix: read something other than thresholded spike rate. This rung built a first-spike-latency read (the one non-rate candidate the crux structure supports — no oscillation to phase-lock, exactly 2 populations so rank-order degenerates, sustained hold not a transient) and compared it against the rate read on the SAME trajectory (a subclass of the crossedge runner verbatim — no retraining confound).

## Result — UNDEFINED (a method-negative + an instrument flag)

- **`n_rate_pass = 0/6`** — the rate read reproduces the original crux's own floor-miss (expected; confirms the harness reproduces the crux).
- **`n_latency_pass = 0/6`** — the latency read (z >= Z_FLOOR=2.0, scale-free, AND lesion-attributable) did NOT clear either. First-spike-latency, as implemented, does not lift this crux.
- **`n_shuffle_collapse = 3/6`** — ⛔ the anti-cheat (a seed-fixed shuffle of which neurons count as generated vs perceived, re-read from the identical raster) should COLLAPSE the read on every seed; it collapses on only 3. On the other 3 the read survives a neuron-identity shuffle — the instrument is reading something identity-invariant (a confound or noise), so the 0/6 latency null is **not yet a trustworthy negative**.

## What this settles + the next lever (NO-DEFER)

The METHOD (this normalized-onset-fraction latency read) does not clear the crux, and its instrument is not yet clean (3/6 anti-cheat). The CAPABILITY — a non-rate read that defeats the rate-saturation shared behind BOTH this crux AND the mouth head-degeneracy — remains open. Next levers, in order: (1) FIX THE INSTRUMENT FIRST — diagnose why the shuffle collapses on only 3/6 (right-censoring at window length collapsing latency to a constant? too few spikes in the read window so latency is undefined and defaults to the censor value identically for both pools?) — a read you cannot anti-cheat cleanly cannot be trusted either way (the "verify the instrument before its output" discipline). (2) Only once the shuffle collapses 6/6: if latency still 0/6, the crux signal may not live in first-spike timing — try a spike-COUNT-variance / Fano or an inter-spike-interval-CV read (dispersion codes that also escape mean-rate saturation), or test whether the crux's read WINDOW is simply too short for any timing code (lengthen `RECALL_STEPS`). The saturation is real; the read primitive that beats it is not yet found.
