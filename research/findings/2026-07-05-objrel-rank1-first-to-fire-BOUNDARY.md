# objrel RANK-1 first-to-fire (latency) read — **BOUNDARY (dt-blocked)**; the ranked ladder is exhausted → fresh research gate

**Date:** 2026-07-05
**Runner:** `research/runners/_rungB1c_objrel_first_to_fire_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_first_to_fire.json`
**Research gate:** `2026-07-05-objrel-spiking-wta-read-research-gate.md` (RANK-1 = rank-order / first-to-fire latency coding).
**Prior boundaries:** `2026-07-05-objrel-rank2-divisive-norm-BOUNDARY.md` (division see-saw) + `2026-07-05-rungB1c-objrel-ff-inhibition-BOUNDARY.md` (subtraction see-saw) + the learned-signed negatives.

## The mechanism tested (genuinely different from the see-sawing common-mode families)

The object-relative role read fails through the spiking winner-take-all (WTA) — it reads TOTAL drive, and the role signal
is a sub-1% common-mode-shifted differential. Both common-mode-REMOVAL families (subtraction, division) see-sawed. RANK-1
attacks it differently: read the winner by spike TIMING, not rate — the winning ensemble = the one whose FIRST spike is
EARLIEST (Thorpe-Gautrais rank-order coding, intrinsically intensity/pedestal-invariant). No pedestal removal at all; the
timing code is meant to be invariant to the additive pedestal by construction. Realized by forking the c2 read to record
each ensemble's first-spike step (winner = earliest), with the ens floor swept DOWN toward threshold (at floor 150 the
ensembles saturate and all fire on step 1 → latency ties). Confound-free (byte-identical c2 bridge, real synaptic read,
6-seed-blind, 4 anti-cheats). NO `sim/` edit. Subagent-built, controller-verified rigorous.

## Result — BOUNDARY (dt-blocked)

| seed | base canon (summed@150) | FTF canon | FTF objrel-slot0 | summed@floor objrel-slot0 | scramble | dt-resolvable |
|---|---|---|---|---|---|---|
| 42 (dev) | 0.97 | 0.44 | 0.00 | 0.00 | 0.00 | **0/12** |
| 43 (dev) | 1.00 | 0.67 | 0.00 | 0.00 | 1.00 | **0/12** |
| 44 (dev, floor 25) | 0.64 | 0.03 | 0.17 | 0.00 | 0.00 | **2/12** |
| 100 (blind) | 0.03 | 0.25 | 0.08 | 0.92 | 0.00 | **0/12** |
| 101 (blind) | 0.00 | 0.14 | 0.25 | 1.00 | 0.25 | **0/12** |
| 102 (blind) | 0.00 | — | — | 1.00 | 0.50 | **0/12** |

agg: verdict **BOUNDARY**; objrel recovers 1/6; dt_resolvable_seeds 0; mean objrel-slot0 first-to-fire **0.25** vs summed 0.486; frozen ens floor 150.

**VERDICT: BOUNDARY — dt-blocked.** dt-resolvable_seeds = 0 across the whole run; the correct ensemble's first
spike is NOT strictly earliest — the sub-1% differential does not produce a resolvable (> dt) latency separation. At a high
floor the ensembles saturate and all fire on the first step (ties); at a lowered floor they fall below threshold and don't
fire at all (no first spike). There is no operating point where the differential sets who-crosses-first. Object-relative
stays at 0.00 and canonical also degrades (the timing read is noisier than the summed read even for canonical).

## Why — the honest mechanism

Rank-order coding is invariant to an additive pedestal ONLY when the differential is large enough to order the first
spikes. Here the differential is sub-1% of the drive, well below the single-step (dt) latency resolution of the
point-neuron f-I curve at any floor — so the pedestal-invariance of timing buys nothing: the ensembles tie. This is a
genuine dt/resolution wall for latency coding on this margin, distinct from the see-saw of the common-mode families but the
same underlying cause: **the sub-1% differential is simply not resolvable by ANY single-ensemble spiking read** (rate,
divided-rate, or timing) — while a linear argmax resolves it trivially.

## The exhausted ladder + the deeper issue → fresh research gate (in flight)

The full ranked ladder from the first research gate is now exhausted: subtraction (see-saw), division (see-saw), three
learned-signed reads (position-basin), and timing (dt-blocked). Plus this arc surfaced a DEEPER issue: the c2 canonical
read-out is only 3-seed-validated and SEED-FRAGILE (base canon ≤ 0.03 on the unseen blind seeds). ⇒ the problem is likely
NOT one more read trick on a shared, fragile 3-way WTA, but the read-out ARCHITECTURE + its robustness. A fresh deep
research gate is dispatched, tasked to weigh a **dual-route** design (separate position and form pathways rather than one
WTA competing over both), a **learned nonlinear** read-out, **population/temporal scale**, and **read-out robustness across
seeds** — against yet another read trick. This BOUNDARY launches that search; it is not the end of the question.

## Files
- `research/runners/_rungB1c_objrel_first_to_fire_derisk.py` — the confound-free first-to-fire de-risk (NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_first_to_fire.json` — the 6-seed-blind boundary record.
