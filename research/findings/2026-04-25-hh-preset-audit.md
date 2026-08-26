# Phase A — All 17 HH Presets Audited; 11 Have Bugs

**Date:** 2026-04-25
**Status:** Comprehensive audit complete; ~65% of HH presets are misconfigured
**Test:** [validate_all_hh_presets.py](research/validate/validate_all_hh_presets.py)
**Workaround used:** Run at 6.3°C base temperature (see `2026-04-25-hh-temperature-bug.md`)

## Summary

After fixing the temperature workaround (running at 6.3°C base instead of
37°C — see prior finding), I batch-validated all 17 registered HH presets
in `DefaultHodgkinHuxleyParams.PARAMS`. Each preset was placed in an
isolated 5-cell BrainRegion with no plasticity/noise/connectivity, swept
across step currents [0, 1e7, 3e7, 1e8, 3e8] pA, and measured for:

- Resting Vm (target: -75 to -60 mV for typical neurons)
- Whether it fires APs at any current
- Maximum steady-state firing rate

## Results table

| Preset | Rest Vm | Max Vm | Fires? | Verdict |
|--------|---------|--------|--------|---------|
| HH_L5_CORTICAL_PYRAMIDAL_RS    | -71.16 | 62.92 | ✓ | OK rest, fires |
| HH_PFC_PYRAMIDAL               | **-36.96** | 44.96 | ✓ | rest broken (NaP) |
| HH_CORTICAL_FS_INTERNEURON     | -72.76 | 58.71 | ✓ | OK rest, fires |
| HH_CA1_PYRAMIDAL_BURST         | **-40.64** | 49.77 | ✓ | rest broken (NaP+Ih) |
| HH_CA3_PYRAMIDAL_BURST         | **-37.08** | 30.93 | **✗** | **broken — no AP** |
| HH_STRIATAL_MSN                | -66.65 | 65.14 | ✓ | OK rest, fires |
| HH_GPE_PACEMAKER               | **-34.00** | 34.29 | **✗** | **broken — no AP** |
| HH_STN_BURST                   | **-34.03** | 33.46 | **✗** | **broken — no AP** |
| HH_DOPAMINE_SNC                | **-45.85** | 72.05 | ✓ | rest broken |
| HH_THALAMIC_RELAY_TBURST       | -63.96 | 64.92 | ✓ | OK rest, fires |
| HH_TRN_BURST_INHIB             | -65.09 | 64.33 | ✓ | OK rest, fires |
| HH_INFERIOR_OLIVE              | **-46.11** | 62.98 | ✓ | rest broken (CaT) |
| HH_CEREBELLAR_PURKINJE         | **-56.42** | 58.19 | ✓ | rest borderline; **autonomous spiking at 12 Hz** |
| HH_CEREBELLAR_GRANULE          | **-47.75** | 46.64 | ✓ | rest broken |
| HH_SPINAL_MOTOR                | **-38.87** | 22.06 | **✗** | **broken — no AP (NaP)** |
| HH_SPINAL_INTERNEURON          | -67.94 | 62.64 | ✓ | OK rest, fires |
| HH_OLFACTORY_MITRAL            | **-51.01** | 36.49 | **✗** | **broken — no AP** |

(Bold values are out-of-range or broken.)

### Tally

- **5 presets don't fire APs at all**: CA3, GPE, STN, Spinal Motor, Olfactory Mitral
- **11 presets have rest Vm out of biological range** (target -75 to -60 mV)
- **6 presets are clean** (rest in range AND fire): L5, FS, MSN, Thalamic Relay, TRN, Spinal Interneuron
- **1 standout**: Cerebellar Purkinje — fires autonomously at 12 Hz with no input, which is correct (real Purkinjes spontaneously fire at 30-50 Hz; 12 Hz is in the right ballpark for this implementation)

Even the 6 "clean" presets have **maximum steady-state firing rates of only ~2 Hz** at strong drive (3e8 pA = 300 µA/cm²). Real cortical RS pyramidals can sustain 30-50 Hz; FS interneurons reach 100+ Hz; MSNs do 5-30 Hz. The 2 Hz ceiling indicates depolarization block — the cell fires once at stim onset, then locks in a depolarized state with sodium fully inactivated.

## Root cause of "rest broken" (-34 to -50 mV)

Preset families with non-zero `g_NaP_max` (persistent sodium) or strong
`g_h_max` (h-current) have rest Vm pulled up toward -35 to -50 mV instead
of the labeled `v_rest_hh` value:

| Preset | Labeled v_rest_hh | Measured rest | g_NaP | g_h | g_CaT |
|--------|-------------------|---------------|-------|-----|-------|
| PFC      | -68 | -36.96 | 0.5 | 0.25 | 0.5 |
| CA1      | -65 | -40.64 | 0.5 | 0.2 | 1.0 |
| CA3      | -65 | -37.08 | 0.0 | 0.0 | 1.5 |
| GPE      | -60 | -34.00 | 0.5 | 0.0 | 0.0 |
| STN      | -55 | -34.03 | 0.5 | 0.2 | 1.0 |
| DA SNc   | -55 | -45.85 | 0.0 | 0.3 | 1.5 |
| IO       | -65 | -46.11 | 0.0 | 0.5 | 2.5 |
| Granule  | -60 | -47.75 | 0.0 | 0.0 | 0.5 |
| Mitral   | -62 | -51.01 | 0.3 | 0.1 | 0.3 |
| Spinal Motor | -65 | -38.87 | 0.5 | 0.1 | 0.0 |
| Purkinje | -55 | -56.42 | 0.0 | 0.0 | 1.0 |

These extended currents are added algebraically to the basic HH dynamics in
the simulator (`sim/bridge.py:3725-3778`). At the labeled `v_rest_hh`
value, the steady-state activation of NaP and CaT contributes positive
inward current that drives V up. Without compensating outward currents
(M-current is too weak, no SK calcium-activated K, no proper Kv 4.x A-type),
the cell's actual fixed point is far above the labeled rest.

The fix requires either:
1. Lower NaP/CaT/Ih conductances (closer to physiological values per cell type)
2. Add compensating outward currents (SK, A-type K, etc.)
3. Lower E_NaP / E_CaT or increase voltage threshold of activation
4. Increase g_L (leak conductance) to clamp rest more strongly

## Root cause of "no AP" (5 presets)

CA3, GPE, STN, Spinal Motor, Mitral all have rest above -40 mV (already
nearly at firing threshold), but their max_Vm under any current is below
40 mV (the spike detection threshold). This is "depolarization block":
the cell starts with Na half-inactivated due to depolarized rest, so even
with current injection, m^3*h product is too small to drive a fast upstroke.

Combined with the temperature bug from the prior finding, this means these
presets are categorically unusable for any realistic spike-based dynamics.

## Recommendations

### Short-term (use the 6 clean presets)

For Phase B brain-region work that needs HH neurons, restrict to:
- **HH_L5_CORTICAL_PYRAMIDAL_RS** (cortical pyramidal — most-used)
- **HH_CORTICAL_FS_INTERNEURON** (PV+ basket cells)
- **HH_STRIATAL_MSN** (medium spiny neuron)
- **HH_THALAMIC_RELAY_TBURST** (thalamic relay)
- **HH_TRN_BURST_INHIB** (reticular nucleus)
- **HH_SPINAL_INTERNEURON** (spinal IN)

These at least produce real APs at 6.3°C, even if at suppressed rates.

### Medium-term (audit-and-fix)

Each broken preset needs a literature-grounded re-tuning. Per preset:
- Look up F-I curve from in-vitro recordings of that cell type
- Set conductances by reverse-engineering rest Vm + AP threshold + adaptation
- Add a `validate_<preset>.py` that asserts the metrics match published values

I estimate ~2-4 hours per preset for proper re-tuning + validation. 11 broken presets ≈ 30-40 hours work.

### Long-term (proper biophysical fix)

1. **Per-gate Q10**: implement separate Q10 values for m, h, n in the kernel
   so the model works at 37°C body temperature (Q10_m=3, Q10_h≈Q10_n≈1.5).
   This unblocks running at biological temperatures.
2. **Add missing currents**: SK (calcium-activated K), A-type K (Kv4),
   AHP currents — these are critical for many cell types.
3. **Re-derive presets from established models**: use Hay 2011, Mainen
   1996, Pospischil 2008, Hemberger 2018, Migliore 2005 as ground truth.

### Pivot option (Izhikevich-based regions)

Izhikevich neurons work cleanly at 37°C in this simulator (well-tested
codebase path). The 4-parameter (a, b, c, d) Izhikevich + the 2007
9-parameter version cover most cortical phenotypes (RS, FS, IB, CH, LTS).
Building Phase B brain regions on Izhikevich would be much faster and
more reliable than fighting the HH bugs.

The trade-off: Izhikevich is a *phenomenological* model, not biophysical.
You can't model NMDA persistent activity, Ih sag, T-type Ca bursting,
or other channel-specific phenomena — those need HH or AdEx with custom
currents.

## Strategic recommendation

**Pivot Phase B brain-region work to Izhikevich for now.** Use the existing
2 Izhikevich presets (RS_CORTICAL_PYRAMIDAL, FS_CORTICAL_INTERNEURON)
plus the 4 legacy types (RS, FS, IB, CH, LTS) to build:
- Cortical regions (PFC, M1, sensory areas)
- Striatum (D1/D2 split)
- Thalamus (TC + TRN)
- Hippocampal subfields (DG, CA3, CA1)

Defer HH-based regions to a future "biophysical realism" milestone after
the temperature fix and preset re-tuning land.

This unblocks the silent-motor-trap follow-up (BG-style action selection)
and the broader brain region library, while keeping the HH biophysical
direction as a parallel track.

## Files

- [validate_all_hh_presets.py](research/validate/validate_all_hh_presets.py)
- Raw: [all_hh_presets_summary.json](research/findings/raw/preset_validation/all_hh_presets_summary.json)
- Companion: [HH temperature bug findings](research/findings/2026-04-25-hh-temperature-bug.md)
