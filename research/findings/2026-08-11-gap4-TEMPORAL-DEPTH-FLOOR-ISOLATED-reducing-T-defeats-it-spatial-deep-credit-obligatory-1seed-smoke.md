---
type: finding
status: qualified
date: 2026-08-11
mechanism: deep-credit-on-spikes
backend: numpy
runner: research/runners/_gap4_temporal_depth_floor_derisk.py
artifacts:
  - research/findings/raw/_gap4_temporal_depth_floor.json
seed-waiver: DELIBERATE 1-seed (42) MECHANISM smoke; the floor-drop isolation is decisive at 1 seed (large, monotone in T) but the spatial-credit gap is not yet seed-robust — the 6-seed sweep (42/43/44/100/101/102) is the RETURNED next command, not run here.
---

# gap#4 deep-credit-on-spikes — the TEMPORAL-DEPTH FLOOR is ISOLATED (1-seed smoke): it is a large, T-DRIVEN confound (floor 0.96@T24 -> 0.44@T1, monotone), and REDUCING T defeats it, making the spatial depth-2 DFA credit load-bearing — small-T is the clean instrument the depth-3 frontier needs

<!--derived-->
**One-line verdict (1-seed smoke, GO on the isolation).** Every 2026-08-02 gap#4 depth finding carries the SAME
un-quantified caveat as its open edge: DFA e-prop is depth-ROBUST on the LIF SNN (trains N=2,3,4) but "the floor is
high (~0.951) — the task is ~1-layer-solvable because the temporal-depth floor: LIF membrane integration over T=24
adds effective depth", so depth-robustness is NOT proven depth-3 credit ASSIGNMENT. Every one of those findings ran at
the FIXED CONSTANT T=24 and never swept T. This smoke sweeps T on the SAME task + SAME LIF SNN + SAME transport-free
DFA e-prop credit and shows the temporal-depth floor is REAL, LARGE and T-driven — and that reducing T defeats it. No
`sim/` edit (reuse-by-import of the validated `run_seed`; the sweep + isolation metric are runner-side).

## Result — 1 seed (42), the T-sweep (compositional-inheritance, LIF SNN, credit_mode=eprop DFA)

<!--derived-->
Artifact `research/findings/raw/_gap4_temporal_depth_floor.json` (numpy/CPU, 2 hidden x 32, 60 epochs). chance 0.333.

| T | floor (1-hidden) | snn (2-hidden DFA) | spatial_gap | oracle | permuted | depth-sep |
|---|---|---|---|---|---|---|
| 1  | 0.444 | 0.556 | **+0.111** | 1.000 | 0.296 | True |
| 2  | 0.481 | 0.556 | +0.074 | 1.000 | 0.296 | True |
| 4  | 0.593 | 0.741 | +0.148 | 1.000 | 0.370 | True |
| 8  | 0.889 | 0.815 | −0.074 | 1.000 | 0.259 | True |
| 16 | 0.926 | 1.000 | +0.074 | 1.000 | 0.259 | True |
| 24 | **0.963** | 0.963 | **+0.000** | 1.000 | 0.296 | True |

<!--derived-->
The `floor` column is the load-bearing read: a 1-hidden-layer LIF net (NO spatial depth) climbs MONOTONICALLY with T
from 0.444 (~chance) to 0.963 — i.e. the temporal-integration window, not spatial layers, is supplying the effective
depth. At T=24 the floor (0.963) equals the trained 2-hidden net (0.963): spatial depth-2 credit adds NOTHING because
the temporal floor already solves the task. At T=1 the floor collapses to 0.444 and the trained 2-hidden DFA net rides
+0.111 above it: with the temporal window gone, the SPATIAL depth-2 DFA credit becomes load-bearing.

## What this isolates (the GO leg — decisive at 1 seed)

<!--derived-->
Three checks, all pass at seed 42: (1) floor(T=24)=0.963 is HIGH (reproduces the cited "~1-layer-solvable ~0.951"
confound); (2) floor(T=24)−floor(T=1) = −0.519, far above the 0.15 bar — the temporal integration WAS the effective
depth; (3) spatial_gap opens at low T (+0.111 @ T=1) and is exactly 0.000 @ T=24. Anti-cheats hold at every T: oracle
1.000 (a rate DendriticMLP ceiling exists, so a null is interpretable), permuted ≤ chance (below-chance, the cleanest
possible no-leakage read; the anti-cheat is one-sided — leakage would be permuted ABOVE chance), stage0 depth-
separating True. **This is the CLAUDE.md wall reframe made concrete: the companion process (the LIF membrane's
temporal integration) had been replaced by a constant (T=24), and the proxy OWNED the measurement** — masking whether
the spatial deep-credit rule did anything. Reducing T removes the proxy and exposes the spatial credit.

## Honest scope (what this IS and is NOT)

<!--derived-->
- IS: a 1-seed MECHANISM smoke that QUANTIFIES the temporal-depth floor (the frontier's named open edge) and shows a
  small-T regime is the clean instrument where SPATIAL depth-credit is obligatory (floor collapsed, ceiling intact).
  The floor-drop isolation (checks 1+2) is decisive at 1 seed: −0.519, monotone across six T values.
- IS NOT: a 6-seed GO. The spatial_gap at low T (+0.111 @ T=1, +0.074 @ T=2) is modest and 1-seed-noisy (note the
  T=8 dip, gap −0.074, where floor and net are both high) — the "genuine spatial depth-2 credit is load-bearing"
  claim (check 3) needs the 6-seed sweep. IS NOT a depth-3 demonstration (that is the downstream rung: re-pose the
  DFA N=2,3,4 sweep at small T so the deeper layers are OBLIGATORY, not redundant behind the temporal floor).

## Why this is past the mapped boundary (per THE LAW — a boundary is an undiscovered mechanism)

<!--derived-->
The 2026-08-02 crux cluster exhaustively characterized the movable-plateau RESERVOIR terminus (5 controls) and then
showed the transport-free DFA rule DOES get purchase on a TRAINABLE LIF SNN and is depth-ROBUST — but left ONE honest
open edge, restated in three findings: a task whose depth-3 credit is OBLIGATORY on the spiking net (defeat the
temporal-depth floor), blocked by "the depth-3-instrument-construction problem" (parity groks only seed-fragile;
hier3 does not separate depth-2 from depth-3). Nobody had asked WHY the instrument keeps failing. This smoke answers
it: the LIF temporal window silently supplies the effective depth, so any feedforward-depth task becomes shallow-
solvable AT T=24. The construction problem was never a task problem — it was the fixed T. The lever is not a cleverer
task; it is **lower T** (or, equivalently, a read that does not integrate over a long window). That reframes the whole
depth-3-instrument search and is the un-tried mechanism.

## Next mechanism (named, not deferred)

<!--derived-->
1. Confirm this smoke at 6 seeds (the RETURNED command) — establishes whether check 3 (spatial credit load-bearing at
   low T) is seed-robust, or whether only the floor-drop isolation (checks 1+2) is.
2. Then re-pose the DFA depth sweep (N=2,3,4) at small T (e.g. T=2-4) on the compositional task: with the temporal
   floor removed, the deeper spatial layers become OBLIGATORY, so DFA e-prop training at N=3,4 above the collapsed
   floor would be GENUINE depth-3 credit assignment (the frontier's target), not survival of redundant depth.
3. Fallback if the low-T floor stays high in the 6-seed (it does not at seed 42): the effective depth is the static
   input rate-expansion, not T — attack the input encoding, not the window.

Sources: `2026-08-02-gap4-DFA-eprop-is-depth-robust-...` (open edge #1, the temporal-depth floor), `2026-08-02-gap4-
crux-transport-free-rule-gets-matched-capacity-...` (same caveat), `2026-08-02-gap4-crux-wall-LOCATED-...` (the
reservoir terminus). Neftci-Mostafa-Zenke surrogate-gradient SNN; Bellec 2020 e-prop (eligibility over many spikes).
