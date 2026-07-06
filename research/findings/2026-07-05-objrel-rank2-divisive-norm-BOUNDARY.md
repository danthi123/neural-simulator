# objrel RANK-2 recurrent divisive normalization — **BOUNDARY** (the see-saw again; refutes shift-invariance-transfers-to-spikes); launches RANK-1 first-to-fire

**Date:** 2026-07-05
**Runner:** `research/runners/_rungB1c_objrel_divisive_norm_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_divisive_norm.json`
**Research gate:** `2026-07-05-objrel-spiking-wta-read-research-gate.md` (RANK-2 = recurrent divisive normalization, Louie-Glimcher).
**Prior:** `2026-07-05-rungB1c-objrel-ff-inhibition-BOUNDARY.md` (subtractive FF-inhibition, also a see-saw) + the learned-signed negatives.

## The mechanism tested (confound-free, the research-gate RANK-2)

The object-relative role read fails through the spiking winner-take-all (WTA): objrel-slot0 (THEME) ~0 while a LINEAR
argmax gets it ~100% (info present + linearly separable). Diagnosis: the WTA reads TOTAL drive but the role signal is a
per-draw common-mode-shifted differential (a sub-1% margin on a `WS_ENS_FLOOR_C2 = 150` pedestal + Dale shift). RANK-2 =
**recurrent divisive normalization** `R_i ∝ V_i/(σ + gain·mean(V_pool))`, whose argmax is shift-invariant
(`argmax (V_i+c)/(σ+gain·(mean+c)) = argmax V_i`). Realized by reusing the sim's EXISTING guarded divisive-norm primitive
(`bridge.py:6236` — exactly that form, recomputed per step) — flag the 3 role ensembles `input_divisive_norm` runner-side
(`cp_input_divisive_mask`), gain-tuned. **NO `sim/` edit.** Confound-free (byte-identical c2 bridge, the REAL synaptic read
`res._drive_and_read`, 6-seed-blind, 4 anti-cheats). Built by a subagent; controller-verified rigorous (no sim edit, real
synaptic read, op-point requires BOTH canon≥0.90 AND objrel-slot0≥0.85 — no weakened anti-cheat).

## Result — BOUNDARY (the see-saw; canon-ok False on all 6)

| seed | base canon (div-off) | DIV-ON canon | DIV-ON objrel-slot0 | DIV-OFF objrel-slot0 | scramble | recov / canon-ok / diff-LB / scr-chance |
|---|---|---|---|---|---|---|
| 42 (dev) | 0.97 | 0.50 | 0.50 | 0.00 | 0.25 | F / **F** / T / T |
| 43 (dev) | 1.00 | 0.58 | 0.50 | 0.00 | 0.75 | F / **F** / T / **F** |
| 44 (dev) | 0.64 | 0.33 | 1.00 | 0.00 | 0.00 | T / **F** / T / T |
| 100 (blind) | 0.03 | 0.14 | 0.92 | 0.92 | 0.00 | T / **F** / **F** / T |
| 101 (blind) | 0.00 | 0.19 | 0.92 | 1.00 | 0.08 | T / **F** / **F** / T |
| 102 (blind) | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | T / **F** / **F** / T |

agg: verdict **BOUNDARY**; objrel_recovers_gate False; canonical_not_regressed_all False; mean objrel-slot0 divisive-on 0.806 vs off 0.486.

**VERDICT: BOUNDARY** — canonical REGRESSES with the divisive norm on, on every dev seed (0.97→0.50, 1.00→0.58,
0.64→0.33). There is no gain that holds BOTH canon and objrel: seed 44 shows the extreme — a strong gain (0.02) gives
objrel-slot0 **1.00 with scramble 0.00** (a genuinely clean role read) but canon collapses to 0.33. The two trade off along
the gain axis.

## Why — the honest mechanism (an informative refutation)

The research gate predicted the divisive norm's **linear-argmax** shift-invariance would preserve canonical. **It does NOT
transfer to the SPIKING read.** The division `(V_i+c)/(σ+gain·mean)` preserves the differential-to-pedestal **RATIO**: it
maps the pedestal 150 → ~1/gain while shrinking the differential proportionally, so the spiking WTA still reads a
near-uniform drive it cannot resolve. Driving the gain hard enough to expose objrel (near-threshold, steep f-I) equally
destabilizes canonical — because canon and objrel are read by the SAME 3-way WTA on the SAME ensembles. **Both common-mode-
removal families (subtraction — the FF-inhibition BOUNDARY — and division — this) see-saw**, for the same reason: removing
the pedestal for a rate-WTA cannot separate the two reads. (The blind seeds 100/101/102 additionally expose a DEEPER,
separate issue: the c2 canonical read is itself SEED-FRAGILE — base canon ≤ 0.03 on 3 of 6 seeds — so its objrel is already
high divisive-off there, making the norm not-load-bearing. That c2 read-out fragility, not this mechanism, is arguably the
first thing to fix; it means the whole objrel-vs-canon comparison is only clean on the ~half of seeds where the base
canonical read works.)

## The next rung (this launches it, per boundaries-are-undiscovered-mechanisms)

The "remove the pedestal before a rate-WTA" approach is exhausted. The next ranked mechanism is a genuinely DIFFERENT read:
**RANK-1 first-to-fire (latency) coding** — the winner = the ensemble whose first spike is earliest. Spike TIMING is
intrinsically intensity/pedestal-invariant (Thorpe-Gautrais rank-order: "less subject to changes in intensity"), so it
does not depend on resolving a rate ratio at all. Requires the ensembles NEAR threshold (not the saturating 150 pedestal)
so the differential sets who-crosses-first — the dt-resolution pre-check the gate specified. If RANK-1 also fails, a fresh
research gate fires for a genuinely-new class (dual-route position+form reads; or a population/temporal read).

## Files
- `research/runners/_rungB1c_objrel_divisive_norm_derisk.py` — the confound-free RANK-2 de-risk (NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_divisive_norm.json` — the 6-seed-blind boundary record.
