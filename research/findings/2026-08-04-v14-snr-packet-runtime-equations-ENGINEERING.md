---
type: research-finding
status: engineering-substrate
date: 2026-08-04
mechanism: v14-snr-authenticated-packet-runtime-equations
claim_check: synthesis
---

# V14 authenticated SNr packet runtime equations

## Boundary

The authenticated packet can now be converted into a deeply immutable runtime
record, and a separate fused kernel can evaluate the complete SNr channel
equation surface. This is an engineering result. It does not establish an
adult-mouse SNr parameter set or authorize a physiology verdict.

The source interpretation remains the one recorded in
`research/findings/2026-08-04-v14-stageB-executable-parameter-evidence-RESEARCH.md`:
preparation-matched targets, transferred channel measurements, and model priors
remain distinct evidence classes. The non-executable target catalog remains
`research/specs/v14_snr_stageB_target_packet.json`.

## Implemented

- Every one of the 69 authenticated packet leaves is parsed exactly once and
  retained in immutable raw and typed mappings.
- `nS/pF` densities are converted to `mS/cm2` through packet capacitance.
- Q10 factors are derived from each mechanism's tagged reference temperature.
- Calcium current density is converted to `uM/ms` using membrane area,
  accessible volume, current fraction, calcium valence, and Faraday's constant.
- The new fused packet kernel evaluates Phillips-form NaP kinetics, Cav2.2,
  HCN, physical calcium decay/influx, and direction-dependent SK kinetics.
- The existing legacy SNr kernel remains unchanged.

## Verification

The combined focused suite passed `33` NumPy tests with one expected CuPy-only
skip and `34` CuPy tests on the RTX 3090. Kernel outputs were compared against
the independent equations in `sim/snr_channel_parameters.py`, including a
non-default Cav2.2 activation power and unequal SK activation/deactivation time
constants.

## Next action

Build full-length per-neuron parameter arrays from authenticated region
bindings, preserve exact legacy values in mixed simulations, and route both the
standard and direct-output bridge paths through the packet kernel. Packet mode
must remain provenance-only until that integration and dynamic-only checkpoint
regeneration pass CPU/GPU continuation and mixed-region gates.
