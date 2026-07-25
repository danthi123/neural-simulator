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

## The PVBC-mechanism probe REFUTES the host/global shortcut → the real spiking PVBC is required
A host WTA-like feedback-inhibition probe (global inhibitory current ∝ recent PC pop firing, injected to all PC; a
mechanism probe, NOT a brain-based build) was swept at the transient-seed + strong-band config. It does **NOT** localize:
as the gain rises (0→20→60→150→400) the activity just **silences uniformly** (F_active 0.0089→0.0050), the spread stays
diffuse (assemblies 0–3 all ~0.02–0.04), never sharpens into a 0→4 sequence, never decodes (SIG=0 throughout).
⇒ **the phenomenological/global-inhibition shortcut is REFUTED** — a global term hits every assembly equally and cannot
provide the LOCAL, activity-specific, correctly-timed inhibition a traveling bump needs. Localization requires the **real
recurrent spiking PVBC** (PC→PVBC→PC with Ecker's connectivity + timing). (Also: assembly-4 never fires in any config —
the bump decays by assembly-3, consistent with the missing E/I balance a real PVBC loop provides.)

## The REAL spiking basket (E%-max) adds activity but does NOT discretize → the residual is theta/gamma TIMING
The existing de Almeida-Idiart-Lisman E%-max ff-basket (`_build`'s `ca3_ff_inhib`/`ca3_ff_n` — a REAL inhibitory basket
region, `exc_fraction=0`, feedforward + feedback arms) was enabled at the transient-seed + strong-band + ECKER config
(weights 2/6/15/40). It RAISES activity (F_active 0.006→0.018, per_asm_frac up) but does **NOT** organize it into a
discrete moving bump: `windows=0` (the event detector finds NO discrete events — the activity is too CONTINUOUS/diffuse),
SIG=0, assembly-4 still never fires. ⇒ during cue-driven replay only the basket's FEEDBACK arm is active (no DG
feedforward volley to set the E%-level), and feedback-inhibition alone does not DISCRETIZE the spread. **The missing
organizing principle is theta/gamma TIMING** — the gamma-paced WTA that chops continuous activity into one-assembly-per-
cycle SEQUENTIAL bumps (`_gap5_gamma_wta_replay_derisk`, validated at RATE level 1.000 forward, must be carried ON-SPIKES:
a theta injector + gamma-phase-gated basket over the CA3 slice). `_build`'s basket is literally tagged "theta-sweep
RANK-2" — the theta/gamma coupling is the un-enabled piece.

## The band was a CONFOUND — and correcting it reveals the real difficulty (the self-sustaining regime)
Iterations 1–9 used the FLAT band (`within_events=6`, no `chain_adjacent_pairs`) — the encode finding's documented
flat-band config, not the 6/6 SHARP band. Re-running on the correct SHARP band (`chain_adjacent_pairs=True`,
`within_events=2`) FLIPS the failure mode: only **asm0 fires, weakly (0.021), with NO propagation** — the sharp band's
WEAK within-attractor can't sustain the Ecker high-V_T PC long enough for the forward links to re-ignite asm1 (band-scale
×1–25 doesn't bridge it). So: flat band → strong DIFFUSE spread (no clean bump); sharp band → asm0 too weak to propagate.
**The self-sustaining traveling regime lives between these** (within-attractor strong enough to sustain + sharp forward
links strong enough to re-ignite the next assembly before adaptation) and is NOT reachable by config sweeps on the
existing band — it needs the Ecker **nS-calibrated** weights + connectivity (the spec's original "full model-build" call
is VALIDATED), or a systematic (within_events × self_regen × band-nS × cue) search.

## Verdict + next (per THE LAW — a precisely-mapped residual, NOT a wall)
- **The gap#5 Ecker-regime replay is characterized to the SELF-SUSTAINING traveling regime** — the coupled balance of
  within-attractor strength ↔ forward-band nS ↔ PC threshold ↔ discretization timing. DEMONSTRATED on the real spiking
  AdEx substrate: ignition (cue), propagation (flat-band spreads), hand-off seeding (transient cue moves the bump off
  asm0), a real spiking inhibitory basket. NOT achieved: a discrete traveling decodable packet — the flat band spreads
  diffusely (no clean bump), the sharp band's weak within can't sustain the bump to propagate, and no cheap sweep of
  {band-scale, basket, gamma-gate, cue} on the existing band found the between-regime.
- **The systematic search is now EXHAUSTED (definitive):** a 12-cell joint grid {within_events (2,4) × self_regen
  (0.15,0.45,0.8) × band_scale (6,20)} on the SHARP band gives the IDENTICAL result in every cell — only asm0 fires weakly
  (0.02), NO propagation. self_regen (even 0.8) does not sustain the assembly; band_scale (even ×20) does not propagate
  it. ⇒ the coincidence-plateau bistability + the chain-encode band do NOT produce a self-sustaining recurrent assembly
  with the high-V_T Ecker PCs — NO cheap knob reaches the traveling regime.
- **NEXT (the honest build, now confirmed necessary):** the spec's Ecker **nS-calibrated recurrent model** — the exact
  CA3 PC→PC connectivity DENSITY + nS weight scale that Ecker 2022 uses to make the recurrent assembly self-sustaining
  and traveling (NOT the existing chain-encode band, which even scaled ×20 dies at asm0), + a tuned spiking PVBC feedback
  loop + on-spikes gamma-WTA discretization. This is a substantial build (a from-scratch Ecker-connectivity CA3), THE
  honest next step. Ruled out this arc: global/host feedback inhibition; feedback-basket-alone; gamma-gating; and the
  full {within × self_regen × band_scale} sweep — all de-risking the full-model build (ignition/propagation/seed/basket
  work; the ONE missing piece is the self-sustaining recurrent connectivity).
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
