---
type: finding
status: negative
date: 2026-08-28
verdict: The read-fidelity latency INSTRUMENT is now FIXED — the shuffle-identity anti-cheat collapses 6/6 (was the ambiguous 3/6 that made the prior latency null untrustworthy). With the trustworthy instrument, on the surprise->episodic F2 crux: rate read 0/6 PASS, first-spike-latency 0/6 PASS (now a CLEAN NO-GO, not ambiguous), spike-train DISPERSION (ISI-CV / Fano) 1/6 PASS. So NO non-rate read primitive tried (latency, dispersion) beats the saturating mean-rate read on this crux. This repoints the question: the F2 "saturation" may not be a read-fidelity limit at all but the ABSENCE of a separable generated-vs-perceived signal in the trained cross-edge — a hypothesis the next lever must test directly, rather than trying yet more read primitives.
mechanism: fixed-instrument latency + spike-train dispersion (ISI-CV/Fano) reads vs mean-rate on the surprise->episodic F2 crux
lane: read-fidelity
seed-waiver: 6-seed run (42/43/44/100/101/102) — this IS the 6-seed de-risk; the clean negative + the 6/6 instrument fix are the result.
artifacts:
  - research/findings/raw/_read_fidelity_nonrate_latency_dispersion_derisk_6seed.json
runner: research/runners/_read_fidelity_nonrate_latency_dispersion_derisk.py
---

# Read-fidelity crux: instrument FIXED (shuffle 6/6), but no non-rate read (latency, dispersion) beats rate — repoint to "is there a signal at all?"

Artifact: `research/findings/raw/_read_fidelity_nonrate_latency_dispersion_derisk_6seed.json` (numpy/CPU, 6 seeds; latency + ISI-CV + Fano reads vs mean-rate on the surprise->`source_provenance` F2 crux, all from the SAME trained cross-edge + SAME raster — no retraining confound).

## What this settles

The prior latency de-risk ([`2026-08-28-read-fidelity-nonrate-latency-UNDEFINED`](2026-08-28-read-fidelity-nonrate-latency-UNDEFINED.md)) was UNDEFINED for TWO reasons: latency 0/6, AND the shuffle anti-cheat collapsed on only 3/6 seeds (an untrustworthy instrument — right-censoring / too-few-spikes made first-spike-latency a constant identical for both pools). This rung FIXED the instrument, then added a dispersion read.

- **Instrument FIXED — shuffle anti-cheat collapses 6/6** (was 3/6): the latency read is now genuinely identity-dependent (restricted to neurons that spike in-window / proper censoring), so its verdict is now TRUSTWORTHY either way.
- **`n_rate_pass = 0/6`** — baseline reproduces the crux's floor-miss.
- **`n_latency_pass = 0/6`** — with the CLEAN instrument, first-spike-latency does NOT clear the crux. This is now a CLEAN NO-GO for latency (not an instrument artifact).
- **`n_dispersion_pass = 1/6`** — the spike-train dispersion read (ISI-CV / Fano factor; Softky-Koch irregularity codes) clears on 1 of 6 seeds — better than latency but not a GO.

## The repoint (NO-DEFER — a verdict on the READ HYPOTHESIS, not the capability)

Three read primitives (mean-rate, first-spike-latency, dispersion) all fail to separate generated-vs-perceived on this crux, and the latency instrument is now proven trustworthy. That makes the original framing — "a rate read SATURATES, so read it differently" — LESS likely: if the signal were present but rate-compressed, at least one timing/dispersion code should have recovered some of it (dispersion got 1/6, a whisper). The stronger hypothesis now: the trained surprise->source_provenance cross-edge may not create a SEPARABLE generated-vs-perceived population difference at all — the F2 "below floor" may be genuine absence of signal, not a read-fidelity artifact. NEXT LEVER: test that directly — measure whether ANY linear/nonlinear decoder (not just a spiking read) can separate the two conditions from the full raster; if not, the crux is a WIRING/credit problem (the edge doesn't induce a distinguishable state), not a read problem, and the fix moves upstream to how the cross-edge shapes source_provenance. The instrument built here (`_read_fidelity_nonrate_latency_dispersion_derisk.py`, 6/6-trustworthy) is the reusable tool for that test.
