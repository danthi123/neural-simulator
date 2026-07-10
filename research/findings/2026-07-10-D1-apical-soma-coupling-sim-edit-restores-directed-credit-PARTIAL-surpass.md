# D1 surpass — a `sim/` edit couples the BDSP apical to the soma: measured bursts now rise with the apical, directed credit ~triples. PARTIAL (the on-bridge P₀ moat is still leaky).

**Date:** 2026-07-10
**`sim/` edit:** `sim/config.py` (2 additive fields) + `sim/bridge.py` (1 guarded current-assembly block). Additive,
default-off, **byte-identical when off (verified exactly)**.
**Verdict:** PARTIAL surpass — the boundary's core (apical decoupled from measured bursts) is fixed and moat-preserving;
the residual (a leaky on-bridge P₀ moat) is named and open.

## The edit
The boundary: the pure `enable_bdsp` path writes `cp_v_apical` only to compute the burst-probability read `P`, so a
top-down apical raises `P` but never depolarizes the soma → the **measured** burst rate `B` is flat → the committed FF
rule `dw ∝ etilde·(B − Pbar·E)` gets no apical-directed credit. The fix routes a scaled electrotonic fraction of the
apical depolarization to the soma in the current-assembly phase (mirroring the two-compartment block's own
`total_input += gc·(v_apical − v_soma)`), behind two new default-off fields:

```
bdsp_apical_couples_soma: bool = False
bdsp_apical_soma_g: float = 0.0     # pA per unit of P-space apical depolarization
# in _run_one_simulation_step (current-assembly phase), guarded:
#   total_input += bdsp_apical_soma_g · bdsp_v_apical_scale · max(v_apical − E_rest, 0)
```
Moat-preserving by construction: at rest `v_apical == E_rest` → zero coupling.

## Deterministic verification (OU noise off — the Stage-A smokes are otherwise ±0.09 noisy)

**Byte-identity (exact):** flag-off ≡ flag-on with g=0 — both `B_rest 0.11711, B_apical 0.11803`, identical. The default
path is unchanged; `tests/test_determinism.py` 7/7 pass.

**The boundary (deterministic):** pure path, `B_rest 0.11711 → B_apical 0.11803` under a 300 pA apical — **B_rises=False**.
Real and reproducible.

**The surpass — B now rises with the apical, moat-preserving:**

| gain g | B_rest | B_apical | B_rises |
|---|---|---|---|
| off | 0.11711 | 0.11803 | False |
| 10 | 0.11711 | 0.16332 | True |
| 40 | 0.11711 | 0.25973 | True |
| 80 | 0.11711 | 0.38026 | True |
| 160 | 0.11711 | 0.48939 | True |

`B_rest` stays **exactly 0.11711 at every gain** — the moat holds (apical=0 → zero coupling → no spurious bursts). The
apical now raises measured bursts, monotone in the gain.

**Directed credit ~triples** (2-region net, plastic input→output; apical-ON = directed credit, apical-OFF = the P₀ moat):

| | credit dw (apical 300) | moat dw (apical 0) | separation |
|---|---|---|---|
| pre-edit (couple off) | 10.83 | 8.16 | 1.33× |
| couple on, g=40 | 22.61 | 8.16 | 2.77× |
| couple on, g=80 | 30.58 | 8.16 | **3.75×** |

The coupling doubles-to-triples the directed credit while the moat dw is untouched (apical=0 → coupling=0), so the
credit/moat **separation improves 1.33× → 3.75×**.

## Honest residual (why this is PARTIAL, not GO)
The on-bridge P₀ moat is **leaky**: moat dw is **8.16, not ≈0**. A clean moat would have apical-OFF move no weights (at
rest `dev = B − Pbar·E ≈ 0`). On-bridge, measured `B_rest = 0.117` does not equal `Pbar·E_rest`, so `dev ≠ 0` and there is
substantial undirected learning regardless of the apical. The coupling fixes the *directed*-credit half (apical now
raises B) but does not clean the moat. So the surpass is real but partial; full learning-to-accuracy on-bridge additionally
needs the P₀-moat calibration on the substrate (align `Pbar·E_rest` to the measured `B_rest`, or the EMERGE-4 resting-burst
calibration the numpy reference folds into the P-bias).

## A self-correction this cycle
The boundary finding I committed hours earlier claimed the moat was **inverted** (`moat_smaller=False`). That was a
**nondeterministic artifact** — the Stage-A smokes never set `ou_std_current_pA=0` and `B` varies ±0.09 run-to-run.
Deterministically the pure path gives credit 10.83 > moat 8.16 (not inverted; just weakly separated with a leaky moat).
The B-decoupling core stands; the inverted-moat framing is retracted (see the boundary finding's correction block).

## ⇒ the claim
A minimal, byte-safe, additive `sim/` edit — routing the BDSP apical through the substrate's own electrotonic
soma-coupling — restores apical→burst coupling (measured `B` rises 0.117→0.49) and roughly triples the directed-credit
separation (1.33× → 3.75×), moat-preserving. The committed rule now gets **directed** credit on-bridge where it got none.
The remaining gap to full on-bridge learning-to-accuracy is the **leaky P₀ moat** (a substrate calibration), now named
and open.

## Next
Calibrate the on-bridge P₀ moat (make `Pbar·E_rest == B_rest` at rest so `dev≈0` → apical-OFF stops moving weights), then
re-run the 2-region learns test (gate: moat dw → ~0 while credit dw stays high) and the
`_d1_onbridge_learn_to_accuracy` runner with the coupling (gate: held-out ≫ chance, wrong-sign anti-learns). Then the
register transition on-bridge.

## Files
`sim/config.py`, `sim/bridge.py` (the guarded coupling block); `research/runners/_d1_apical_soma_coupling_derisk.py`
(deterministic B probe), `_d1_apical_soma_coupling_learns_derisk.py` (directed-credit probe); the boundary it surpasses
`2026-07-10-D1-onbridge-BDSP-apical-decoupled-from-soma-BOUNDARY-root-caused.md`.
