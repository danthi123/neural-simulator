# N8 (thalamic tonic drive) — STEP 1 verification + weight-scale de-risk — 2026-06-06

**Probe:** `research/runners/_n8_thal_disinhibition_probe.py` (builds the REAL nav BG via
`build_bg_brain_regions` flagship A+E config, measures GPi/D1/thal/motor firing per action pool under
the tonic cheat vs. genuine-disinhibition drive regimes). Run on GPU (CuPy), seed 42.

## What is wired vs. what is missing (audit confirmed + corrected)

The genuine disinhibition cascade is **ALREADY FULLY WIRED** in `build_bg_brain_regions`. Nothing
structural is missing:

| Pathway | weight | line | status |
|---|---|---|---|
| `cortex_X -> str_D1_X` (corticostriatal) | 25/density = ~125 @ density 0.20 | ~1029 | present, plastic |
| `str_D1_X -> gpi_X` (GABA, inhibit) | **15.0** | ~1116 | present (matches genuine demo `d1_w=15`) |
| `gpi_X -> thal_X` (GABA, inhibit) | **8.0** | ~1171 | present (matches genuine demo `gpi_w=8`) |
| `thal_X -> motor_X` (excite) | 20.0 | ~1224 | present |

So **N8 is purely a DRIVE cheat, not a wiring gap.** The cheat is entirely in the per-step tonic drives
(`g11_bg_runner.py:3328-3336`, mirrored in the one-time setup at 2918-2930):

- `thal_X <- 300 pA` — **N8**: thalamus externally paced; the relay fires from its own injected current,
  not from GPi release.
- `gpi_X <- 110 pA` — **N9**: GPi tonic operating point hard-wired (too weak to silence thal once thal is
  also driven at 300).
- `gpe_X <- 150`, `stn <- 150`, `snc <- 150` — N9 (left unchanged by this conversion; only GPi+thal are N8).

D1→GPi is wired and D1 IS driven by the cortex heuristic (via corticostriatal). The disinhibition chain
exists end-to-end; it is simply **bypassed** by the tonic thal drive.

## TONIC regime (the current cheat), 80-step settle

```
no selection : gpi[N=0.064 E=0.062 S=0.062 W=0.062]  thal[N=0.024 E=0.024 S=0.021 W=0.024]  motor[~0.001]
cortex_N on  : gpi[N=0.016 E=0.075 S=0.075 W=0.075]  thal[N=0.017 E=0.005 S=0.006 W=0.006]  motor[~0.003]
```

The whole BG operating point is anemic (gpi ~0.06, thal ~0.024, motor ~0.001-0.003 spikes/neuron/step).
Action selection in production reads the argmax of motor spike counts over the 30-100ms readout window;
the tonic drive lets all four thal pools fire roughly equally, and the differential comes from the
heuristic cortex drive perturbing one channel. **The thalamus is externally paced, exactly as the audit
states.**

## GENUINE regime (port: gpi pacemaker, thal tonic, NO direct thal selection)

At the demo's scales (gpi_tonic=2200, thal_tonic=600) the disinhibition mechanism **works qualitatively
and is perfectly selective**, but the released thal/motor rate is low. The cheap-first weight sweep
(settle=120, select=N) shows the mechanism is robust across scales and identifies a cleaner operating
point:

```
gpi_tonic thal_tonic cortex |  d1_N   gpi_N(none->sel)   thal_N(none->sel)  motor_N  motor_other
     2200       600    800   | 0.057   0.500->0.237        0.000->0.016       0.001      0.000
     1500       600   1200   | 0.071   0.292->0.057        0.008->0.042       0.016      0.000
     1000       600    800   | 0.054   0.208->0.043        0.015->0.042       0.016      0.000   <= chosen
     1000       600   1200   | 0.072   0.208->0.033        0.017->0.042       0.016      0.000
```

**Every setting:** GPi drops when its D1 fires, thal_<sel> rises, and **all non-selected motors stay at
exactly 0.000** (selectivity is perfect — strictly better than tonic, where all thal fire). Lower GPi
tonic (1000) silences GPi more completely (0.208->0.043 ≈ 80% drop) and releases thal cleanly
(0.015->0.042), with motor_N reaching 0.016 — *higher* than the tonic regime's 0.003.

**Chosen operating point for the flag: `gpi_tonic=1000, thal_tonic=600`, cortex heuristic unchanged (800).**
This keeps the GPe/STN/SNc loop drives identical to the cheat (those are N9, out of scope), changes only
the GPi tonic (110 -> 1000, the genuine pacemaker) and the thal drive (300 direct-paced ->
600 tonic-excitation-expressed-only-when-released).

## STEP 1 verdict

- GPi→thal **is** wired; D1→GPi **is** wired; D1 **is** driven by the cortex heuristic. ✅
- Under TONIC, thalamus is externally paced (the cheat). ✅ confirmed
- Under GENUINE, driving cortex_<sel> silences gpi_<sel>, releases thal_<sel>, and leaves the other three
  thal+motor pools at 0.000 — **genuine disinhibition expresses selection in the nav substrate.** ✅
- The absolute released motor rate is low but **selectively higher than the tonic regime**, so the
  production argmax-over-motor-spikes readout should pick the released action cleanly.

→ Proceed to STEP 2 (implement the opt-in `--genuine-thal-disinhibition` flag) and STEP 3 (single-seed
nav smoke gate, genuine vs tonic on the cheat-5 multi-goal score).
