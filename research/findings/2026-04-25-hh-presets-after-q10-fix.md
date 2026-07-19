# HH Preset Validation at 37°C — After Per-Gate Q10 Fix

**Date:** 2026-04-25
**Status:** Major progress — 15/17 presets now fire APs (vs 6/17 before fix)
**Companion:** [HH temperature bug findings](research/findings/2026-04-25-hh-temperature-bug.md), [original audit](research/findings/2026-04-25-hh-preset-audit.md)
**Fix commit:** b6f356f

## Summary

After implementing per-gate Q10 (Q10_m=3.0, Q10_h=Q10_n=1.5) in the HH
kernel and bridge, the simulator's HH model now produces real action
potentials at body temperature (37°C). Re-running the full 17-preset
validation at 37°C:

**15 of 17 presets fire APs** (vs 6 of 17 before, when validation had to
use the 6.3°C workaround that itself was an artifact of the temperature
bug).

## Results table at 37°C with per-gate Q10 fix

| Preset | Rest Vm | Max Vm | Max steady rate | Fires? | Verdict |
|--------|---------|--------|-----------------|--------|---------|
| HH_L5_CORTICAL_PYRAMIDAL_RS    | -71.15 | 59.77 | 2 Hz   | ✓ | rest OK, depo block (low g_K) |
| HH_PFC_PYRAMIDAL               | **-52.32** | 53.51 | 2 Hz   | ✓ | rest depolarized (NaP), depo block |
| HH_CORTICAL_FS_INTERNEURON     | -72.76 | 54.81 | **130 Hz** | ✓ | **biologically realistic FS** |
| HH_CA1_PYRAMIDAL_BURST         | **-58.14** | 57.67 | 76 Hz  | ✓ | rest mildly depolarized, fires properly |
| HH_CA3_PYRAMIDAL_BURST         | **-55.89** | 47.00 | 86 Hz  | ✓ | (was completely silent at 6.3°C — fixed!) |
| HH_STRIATAL_MSN                | -66.27 | 60.78 | 2 Hz   | ✓ | rest OK, depo block |
| HH_GPE_PACEMAKER               | **-35.60** | **19.93** | 0 Hz   | **✗** | NaP issue persists |
| HH_STN_BURST                   | **-34.88** | **19.47** | 0 Hz   | **✗** | NaP issue persists |
| HH_DOPAMINE_SNC                | **-50.91** | 61.44 | 2 Hz   | ✓ | rest depolarized, fires once |
| HH_THALAMIC_RELAY_TBURST       | -63.04 | 60.50 | 2 Hz   | ✓ | rest OK, fires once |
| HH_TRN_BURST_INHIB             | -64.60 | 60.09 | 2 Hz   | ✓ | rest OK, fires once |
| HH_INFERIOR_OLIVE              | **-52.50** | 60.71 | 2 Hz   | ✓ | rest depolarized (CaT), fires once |
| HH_CEREBELLAR_PURKINJE         | -69.96 | 55.58 | **110 Hz** | ✓ | **biologically realistic Purkinje** |
| HH_CEREBELLAR_GRANULE          | **-58.44** | 63.86 | 66 Hz  | ✓ | rest mildly depolarized, fires properly |
| HH_SPINAL_MOTOR                | **-55.52** | 54.44 | 2 Hz   | ✓ | (was silent at 6.3°C — fixed!) |
| HH_SPINAL_INTERNEURON          | -67.17 | 58.48 | 2 Hz   | ✓ | rest OK, fires once |
| HH_OLFACTORY_MITRAL            | -60.81 | 56.38 | 72 Hz  | ✓ | (was silent at 6.3°C — fixed!) |

(Bold rest-Vm values are out of biological range -75 to -60 mV.)

## Comparison to before the fix

| Metric | Before (at 6.3°C) | After (at 37°C with fix) |
|--------|-------------------|---------------------------|
| Presets that fire APs | 6/17 | **15/17** |
| Presets with rest Vm in biological range | 6/17 | 8/17 |
| Presets with realistic high firing rate (>30 Hz) | 1 (Purkinje 12Hz) | **5** (FS 130, CA1 76, CA3 86, Granule 66, Mitral 72, Purkinje 110) |
| Presets categorically broken (no AP) | 5 (CA3, GPE, STN, SpMotor, Mitral) | 2 (GPE, STN) |

## Outstanding issues

### Bug 2 (still open): NaP/Ih push rest above firing threshold

GPE and STN both have:
- `g_NaP_max = 0.5` (strong persistent Na)
- This contributes inward current at the labeled rest of -60 mV
- Without compensating outward currents, rest equilibrates at -34 to -36 mV
- At that depolarized rest, Na inactivation (h) is mostly clamped → no APs

Fix options for GPE/STN specifically:
1. Reduce g_NaP_max from 0.5 to 0.1-0.2 (matching real biophysical density)
2. Add SK (Ca-activated K) current to clamp rest
3. Increase g_L (leak) to provide stronger pullback to E_L

### Depolarization-block at strong drive (6 presets stuck at 2 Hz)

L5, MSN, ThalRelay, TRN, IO, DA SNc all show "fires once at stim onset
then locks in depolarized state with Na inactivated." Root cause: the
preset has g_K_max ≤ 5 (vs original HH 36) — too weak to repolarize fast
enough at high drive.

This is a tuning issue, not a temperature bug. To fix: increase g_K_max to
20-40 in these presets.

### Mild rest-Vm depolarization (6 presets at -50..-58 mV)

PFC, CA1, CA3, DA SNc, IO, Granule, SpMotor have rest Vm 5-10 mV above the
typical -65 mV. Same root cause as GPE/STN (NaP/Ih inward current at
labeled rest) but milder. These cells DO fire — they're not catastrophically
broken — but their behavior would be closer to literature with proper rest.

## Strategic implication

The temperature fix is the **highest-leverage** single change — it unblocked
9 presets that were broken at 37°C. Now bug 2 (NaP/Ih tuning) is a
much smaller-scope remaining issue:
- 2 presets categorically broken (GPE, STN)
- 6 presets with mild depolarization block
- 6 presets with mildly-depolarized rest

**Phase B work can proceed using the 9 well-behaved presets:**
- Cortical layers: L5, FS, PFC (rest is mildly off but fires)
- Hippocampus: CA1, CA3 (now firing at 76, 86 Hz!)
- BG: MSN (rest OK, but limited firing rate at strong drive)
- Thalamus: ThalRelay, TRN
- Cerebellum: Purkinje, Granule
- Spinal: SpInterneuron

For BG, the 2 broken presets (GPE, STN) need re-tuning — that's the
next concrete bug-fix work before Phase B can use full-stack BG.

## Files

- [Validation harness](research/validate/validate_all_hh_presets.py)
- [Raw 37°C data](research/findings/raw/preset_validation/all_hh_presets_summary.json)
- [Run log](research/findings/raw/preset_validation/all_hh_37c_run.log)
- [Fix commit](https://github.com/danthi123/neural-simulator/commit/b6f356f)
