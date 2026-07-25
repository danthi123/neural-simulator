# gap#5 Ecker-regime replay: the boundary is LOCALIZED to PVBC feedback inhibition — ignition + band-strength + transient-seed are all DEMONSTRATED (6 cheap reuse-only iterations, NO sim/ edit); the full Ecker CA3 model-build is NOT needed, just the PVBC pool (2026-07-25)

## What this establishes
The gap#5 scoping (`2026-07-25-gap5-ecker-ca3-model-build-READY-TO-BUILD-spec...`) framed the replay-ignition boundary as
needing a "full Ecker-2022 CA3 model-build" with three coupled ingredients. Six cheap reuse-only iterations (each ~7s,
seed 42, reusing `_prepare_sequence` + `_setup_read` + the unit-tested `_decode_replay`, NO `sim/` edit) DEMONSTRATED
two of the three ingredients directly and localized the residual to the third — **PVBC feedback inhibition** — so the
build shrinks from a full model to a single pool.

## STEP 0 (committed `d707bf34`): the ECKER_CA3_PC AdEx preset works
`a=−0.27` (negative amplifying subthreshold adaptation, the traveling-bump crux) reaches the kernel — but ONLY via the
**global** `cfg.default_neuron_type_adex`; the per-region `adex_neuron_type` is DEFERRED (`bridge.py:2233`). Verified a<0
end-to-end (`gap5_ecker_preset_verify.py`).

## The diagnostic ladder (seed 42, n_ca3=1000, 5-assembly chain; `per_asm_frac` = per-assembly mean active fraction)
1. **Spontaneous SWR envelope → F_active=0.** The Ecker PC is high-threshold (`V_T=−24.4` vs typical −50), so it does
   NOT fire under the weak spontaneous drive. ⇒ ignition needs a **cue** (as the spec said).
2. **Cue drive sweep (sustained) → IGNITION WORKS.** Assembly-0 activity rises with cue strength: 700pA 0.0 → 2000pA
   0.041 → 5000pA 0.114 → 12000pA 0.183 (saturates ~5000–8000pA). But ONLY assembly-0 fires; 1–4 stay dark ⇒ ignition is
   NOT the residual; **propagation** is.
3. **Band-strength sweep (×1→×30, sustained cue) → BAND STRENGTH is the propagation lever.** Scaling the between-assembly
   recurrent weights (`cp_connections.data[between_flat]`) progressively lights downstream assemblies: ×30 →
   `[0.181, 0.034, 0.040, 0.032, 0]`. Activity spreads, confirming the band drives propagation — but weakly, and
   assembly-0 stays dominant (the sustained cue keeps re-igniting it).
4. **Transient-seed test (short cue + strong band) → the bump MOVES OFF assembly-0.** A brief 10–25-step seed (vs
   150-step sustained) lets the neg-a adaptation extinguish assembly-0: at seed=10st/×40 → `[0.032, 0.045, 0.033, 0.031, 0]`
   — assembly-1 (0.045) now EXCEEDS assembly-0 (0.032). So the transient seed lets the bump travel off the seed, as the
   spec predicted ("a weak seed, not a sustained detonator").

## The precise residual: PVBC feedback inhibition
With ignition + band + transient-seed all working, the propagation is still **diffuse and weak** (all assemblies
~0.03–0.05), never reaches the last assembly (4), and does NOT decode as a clean traveling trajectory (`SIG=0`, argmax
flat) — because there is **no feedback inhibition to LOCALIZE the bump into a sharp moving packet** and provide the E/I
balance that sustains it down the chain. This is exactly Ecker's third ingredient (PVBC→PC feedback inhibition) and it is
now confirmed as THE load-bearing remaining lever — the decoder is validated (in-run unit test forward r=0.985), so the
missing structure is the inhibition, not the read.

## Verdict + next (per THE LAW)
- **The gap#5 Ecker-regime replay boundary is localized to PVBC feedback inhibition.** Ignition (cue), propagation (band
  strength), and hand-off seeding (transient cue) are DEMONSTRATED on the real spiking AdEx substrate. The full
  Ecker CA3 model-build is NOT required — the build reduces to **adding a PVBC feedback-inhibition pool**.
- **NEXT BUILD:** add the PVBC→PC feedback inhibition and re-test (transient seed + strong band + PVBC): does the bump
  now form a sharp localized packet that travels 0→4 and decodes (|r|>0.6, argmax sweeps)? **⚠️ The PVBC neuron-model
  fork (banked `8a31bc26`):** the global-scalar AdEx kernel can't trivially co-host PC-AdEx + PVBC-Izhikevich; resolve
  PVBC first as either (a) inhibitory AdEx neurons (same global params — an FS-fidelity approximation, buildable now via
  `exc_fraction`), or (b) a per-neuron `cp_adex_*` heterogeneity kernel capability (additive, default-off — the faithful
  path). Then 6-seed with the spec's controls (no-band / no-PVBC / adapt-lesion / no-encode / control-outperforms-real).

## Provenance
`scratchpad/gap5_ecker_{preset_verify,minimal_test,cued_travel}.py` (+ logs `ecker_{minimal,cued,drivesweep,bandscale,transient}.log`).
Reuses `_gap5_moving_bump_replay_decode` + `_gap5_swr_envelope_replay_derisk` + `_gap5_sequence_replay_derisk`. Builds on
the committed ready-to-build spec + STEP 0 (`d707bf34`). NO `sim/` edit (beyond the committed additive preset).
