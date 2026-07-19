# Phase A — Izhikevich Presets Audit

**Date:** 2026-04-25
**Status:** Complete; 2 of 7 presets work directly, 5 silently fall back

## Summary

The simulator's Izhikevich neuron implementation is in better shape than HH
(no temperature bug) but has its own integration issues:

1. **Bridge ignores `cfg.default_neuron_type_izh`** — assigns neurons to
   IZH2007_RS or IZH2007_FS via `traits % num_variants` (=2). To use a
   single type for all neurons, traits must be controlled separately.
2. **Legacy types silently fall back to RS** because the bridge's call to
   `DefaultIzhikevichParamsManager.get_params()` is hardcoded to
   `use_2007_formulation=True`, which returns FALLBACK_2007 (=RS) for any
   legacy type enum.

After working around both issues (set `num_traits=1`, then directly write
the desired params to `bridge.cp_izh_*` arrays post-init), the validation
results:

| Preset | Rest Vm | Max Vm | Thresh (pA) | Max rate (Hz) | Verdict |
|--------|---------|--------|-------------|----------------|---------|
| IZH2007_RS_CORTICAL_PYRAMIDAL  | -60.00 | 34.98 | 100 | 200 | ✓ working |
| IZH2007_FS_CORTICAL_INTERNEURON | -55.00 | 55.62 | 100 | 500 | ✓ working (FS fires faster as expected) |
| RS_EXCITATORY_LEGACY           | -60.00 | 34.98 | 100 | 200 | ✗ silently uses RS_CORTICAL_PYRAMIDAL params |
| FS_INHIBITORY_LEGACY           | -60.00 | 34.98 | 100 | 200 | ✗ same fallback |
| IB_EXCITATORY_LEGACY           | -60.00 | 34.98 | 100 | 200 | ✗ same fallback |
| CH_EXCITATORY_LEGACY           | -60.00 | 34.98 | 100 | 200 | ✗ same fallback |
| LTS_INHIBITORY_LEGACY          | -60.00 | 34.98 | 100 | 200 | ✗ same fallback |

The 2 IZH2007 presets show correct biological behavior:
- RS at moderate drive: produces real spikes, rest at -60mV (matches preset's vr)
- FS at moderate drive: higher max rate (500 Hz vs 200 Hz for RS), rest at -55mV
- Both reach the spike-detection threshold (~30-55 mV peak)

## Root cause

`sim/bridge.py:944-946`:
```python
default_type_enum = NeuronType[cfg.default_neuron_type_izh]
default_params = DefaultIzhikevichParamsManager.get_params(
    default_type_enum, use_2007_formulation=True
)
```

`sim/enums.py:451-454` `get_params(use_2007_formulation=True)` only accepts
2 enum values; everything else returns FALLBACK_2007 = RS params:

```python
if neuron_type_enum in [IZH2007_RS_CORTICAL_PYRAMIDAL,
                        IZH2007_FS_CORTICAL_INTERNEURON]:
    return PARAMS[neuron_type_enum].copy()
print(f"Warning: Requested legacy type ... Using RS_CORTICAL_PYRAMIDAL fallback.")
return FALLBACK_2007.copy()
```

The print warning is silenced or lost in the simulator's normal log output,
so this fallback happens invisibly.

Plus, `sim/bridge.py:917-921` builds `defined_izh2007_types` as `[ntype for
ntype in NeuronType if "IZH2007" in ntype.name and ntype in PARAMS]`, which
is always exactly the 2 IZH2007 types. It then assigns neurons via
`traits % 2`, splitting the population regardless of the configured default.

## Implications

For Phase B brain-region work:

**Available cortical phenotypes from Izhikevich:**
- IZH2007_RS_CORTICAL_PYRAMIDAL (excitatory, regular spiking, mod adaptation)
- IZH2007_FS_CORTICAL_INTERNEURON (inhibitory, fast spiking, no adaptation)

**NOT available (silently broken):**
- IB (intrinsic bursting) — needed for thalamic relay, sensory-cortex L5
- CH (chattering) — gamma oscillation drivers
- LTS (low-threshold spiking) — regular non-FS interneurons

**Workaround for IB/CH/LTS:** add new entries to
`DefaultIzhikevichParamsManager.PARAMS` keyed on new IZH2007_* enum names
(e.g., `IZH2007_BG_MSN`, `IZH2007_THAL_TC`, `IZH2007_PFC_LTS`). The bridge
already supports any IZH2007_* enum value automatically.

## Recommendations

### Immediate

For Phase B brain regions that need cortical RS + FS, the existing 2
IZH2007 presets work cleanly. Use them.

### Short-term (~1 day work)

Extend `DefaultIzhikevichParamsManager.PARAMS` with biology-grounded entries:

| New enum | Use | Source |
|----------|-----|--------|
| IZH2007_STRIATAL_MSN | basal ganglia direct/indirect pathway | Izhikevich 2003 (rebound bursting cells) |
| IZH2007_THALAMIC_RELAY_TC | thalamic projection neurons | Izhikevich 2003 (RS+adaptation) |
| IZH2007_THALAMIC_RETICULAR | thalamic reticular bursting | Izhikevich 2003 (LTS cells) |
| IZH2007_HIPPO_PYRAMIDAL | CA1/CA3 pyramidal | Izhikevich 2003 (IB) |
| IZH2007_DOPAMINE_VTA | DA neurons | Izhikevich 2003 (RS with low rate) |

Each takes ~30 min to define + validate (4-5 hours total).

### Medium-term (fix the simulator)

Two simulator bugs to fix:
1. `get_params(use_2007_formulation=True)` should not silently fall back —
   either use the proper legacy params or raise.
2. `bridge.py:917-921` IZH initialization should respect
   `cfg.default_neuron_type_izh` for single-type populations, not always
   split-by-trait.

These are small fixes (~1-2 hours each).

## Files

- [validate_all_izh_presets.py](research/validate/validate_all_izh_presets.py)
- Raw: [all_izh_presets_summary.json](research/findings/raw/preset_validation/all_izh_presets_summary.json)
