---
type: finding
status: contributing
date: 2026-08-01
mechanism: recombinative-replay
lane: H-gap5
artifacts:
  - research/findings/raw/gap5_r4/spk_gamma_recomb_6seed_aggregate.json
---

# gap#5 RANK 3 (imagination) — DIRECT spiking gamma-recombination is NOT learned-selective: the SCRAMBLE control exceeds the learned condition on every seed — 6-seed NEGATIVE (0/6)

<!--derived-->
**One-line verdict:** the companion to the already-banked RANK3 finding (which showed the *extracted-matrix
proxy* sits at the geometric chance level). This runs the gamma-WTA phase-organized recombination in the FULL
direct spiking simulation (30 cycles, 6 seeds, five controls) and resolves to **0/6 GO** with a decisive
diagnostic: the network reliably reaches the target B (co-ignition ≈ 1.0, and the no-encoding control is silent,
so the pathway is real), but the recombination readout is **NOT learned-selective** — the SCRAMBLE control (mean
recomb-fraction 0.633) shows MORE "recombination" than the learned MAIN condition (mean 0.165) on **every one of
the 6 seeds** (scramble−main = +0.55/+0.77/+0.27/+0.55/+0.54/+0.13). The recomb events are non-specific exits, not
learned novel recombination. This confirms the RANK3 boundary via full direct simulation, not just the proxy. Ran
concurrently on the GPU while the CPU lanes ran on the pool (parallelism directive).

Artifact: `research/findings/raw/gap5_r4/spk_gamma_recomb_6seed_aggregate.json` (backend cupy/GPU; the per-seed raw
`spk_gamma_recomb_6seed.json` with its provenance sidecar sits beside it).

## Result — 6 seeds {42,43,44,100,101,102}, direct simulation, 30 cycles

<!--derived-->
| read-out (6-seed mean) | value | reading |
|---|---|---|
| MAIN recomb-fraction (learned) | 0.165 | low + highly seed-variable (0.0–0.5) |
| SCRAMBLE recomb-fraction (control) | 0.633 | **higher than MAIN** — the tell |
| scramble − main (per seed, all positive) | +0.13 … +0.77 | not-learned-selective on 6/6 |
| MAIN co-ignition fraction | 1.000 | the network DOES reach B |
| no-encoding co-ignition (control) | 0.000 | pathway is real (removing encoding silences it) |
| GO | 0/6 | — |

## What this settles (and how it strengthens the prior finding)

<!--derived-->
The prior RANK3 finding read the recombination through an *extracted mean transition matrix* proxy and found it at
the geometric chance level. The obvious rejoinder was that the proxy might be lossy — maybe the direct network
recombines even though the extracted-matrix summary washes it out. This run removes that rejoinder: the DIRECT
network, scored trial-by-trial, does not recombine selectively either. The scramble control is the load-bearing
evidence — if the gamma-WTA timing were extracting *learned* transition structure to build a novel path, scrambling
that structure would REDUCE recombination; instead it INCREASES it on every seed, because the "recomb" outcomes are
non-specific branch exits that the scramble (which weakens the dominant learned exit) makes *more* likely, not less.
The mechanism reaches the shared node and co-ignites, but the timing primitive does not convert co-ignition into a
selective novel sequence.

## Honest scope + next
This is a NEGATIVE for the gamma-WTA timing method as a route to RANK3 recombination on this substrate — a verdict
on the METHOD, not the capability. It does not retire "imagination/recombination"; it retires *this* timing
primitive as the recombination engine and sharpens what the next method must do: produce an exit at the shared node
that is SELECTIVE for the learned continuation (scramble must reduce it, not raise it). The RANK3 research gate's
remaining candidates (a learned gating of the branch, or a content-addressable rather than timing-addressed
selection at the node) are the next levers; the timing-only method is now closed direct-and-proxy. No capability
abandoned.
