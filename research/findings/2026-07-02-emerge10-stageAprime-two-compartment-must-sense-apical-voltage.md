# EMERGE-10 / rung-4 Stage A' — the two-compartment dAP must be a real VOLTAGE compartment the plateau kernel SENSES, not a current-routing shortcut. A minimal "route the plateau current to a leaky apical + attenuate the somatic coupling" edit fails with POSITIVE FEEDBACK (the plateau senses the somatic voltage, so apical->soma coupling closes a runaway loop). This sharpens the two-compartment spec precisely; sim/ reverted to pristine for the proper build.

**2026-07-02 (autonomous; build-informative).** Runner `research/runners/_emerge10_stageA_dap_fire_first_derisk.py` (now has a `--two-compartment`/`--apical-coupling` option). NO net `sim/` change (a scaffolding edit was made, tested, and REVERTED — sim/ is pristine).

## What was tried
Stage A confirmed (risk-1) that a single-compartment coincidence plateau injected as SOMATIC current is binary (primes only when it also fires). The minimal proposed fix: route the plateau current `I_coincidence` into a leaky apical compartment `cp_v_apical = apical_leak*cp_v_apical + I_coincidence` and add only `apical_soma_coupling * cp_v_apical` (an attenuated, sub-threshold fraction) to the soma — a small guarded, default-off (byte-identical) edit at `sim/bridge.py:6441`.

## Result — the shortcut fails with positive feedback (decisive)
- **OFF is byte-identical** (the else-branch is the original line; Stage A risk-1 reproduced exactly).
- **ON gives an even bigger fire-first advantage** (plateau-specific +80 to +100 pA) but `noFF` (plateau alone) STILL fires — **even at apical_soma_coupling = 0.005** (a 200x attenuation). That is impossible for a genuinely attenuated sub-threshold coupling.
- **Root cause (diagnosed):** the plateau kernel `fused_coincidence_plateau` senses the **SOMATIC** voltage (`cp_membrane_potential_v`, `sim/bridge.py:6435`) for its Mg2+-unblock regeneration. Routing the plateau current to the apical and coupling the apical back to the soma closes a POSITIVE-FEEDBACK loop: any somatic depolarization (even from a tiny apical coupling) unblocks more Mg2+ -> larger `I_coincidence` -> larger apical -> larger somatic coupling -> runaway -> fires. The compartments are not actually separated because the regenerative variable still reads the soma.

## The refined spec (the genuine two-compartment neuron)
A faithful two-compartment dAP needs the plateau REGENERATION to happen in the APICAL compartment, sensed by the APICAL voltage — so the dendrite can generate a large plateau/spike while only an electrotonically-attenuated, sub-threshold fraction reaches the soma (no somatic feedback into the regeneration). Concretely:
1. `cp_v_apical` is a real membrane VOLTAGE (its own leaky ODE), not an integrated current.
2. The coincidence-plateau conductance drives `cp_v_apical`, and the plateau kernel's Mg2+-block + driving force read `cp_v_apical` (pass `cp_v_apical` where `cp_membrane_potential_v` is passed today at `sim/bridge.py:6435`) — so the regeneration is APICAL.
3. Coupling is one-directional-ish: `total_soma_current += g_couple * (cp_v_apical - cp_membrane_potential_v)` (electrotonic), with `g_couple` small enough that a full apical plateau is sub-threshold at the soma; optionally a weak soma->apical back-coupling for realism (kept small so no runaway).
4. Guarded by `enable_two_compartment_dap` (default off -> byte-identical); `cp_v_apical` allocated with the coincidence conductance; a byte-identity-when-off test.
This is the ~120-180-line coupled-ODE two-compartment build the scoping originally sized (the `RESONATE_AND_FIRE` guarded-additive `NeuronModel` remains the structural precedent). It is a `sim/` edit — fair game for faithful biology.

## Why revert
A default-off-but-buggy ON path in a protected module is a landmine. The scaffolding (config fields + guarded branch) was reverted so `sim/` stays pristine; the proper coupled-ODE build (plateau senses apical) is the next continuation, now with a precise, de-risked spec (the positive-feedback failure mode is known and its fix identified).

## Status of the rung-3 -> rung-4 arc (unchanged, all committed)
The full unsupervised sequence-learning mechanism is validated + fully spiking in numpy (EMERGE-9b/c/d GOs, scale-validated to 32 overlapping contexts) and rung-4 is scoped + Stage-A de-risked. The ONE remaining substrate piece is this genuine two-compartment dAP neuron; Stages B (per-column WTA + frozen permanences -> 9c parity) and C (three-term kernel -> 9d parity) follow it.

## Artifacts
`research/runners/_emerge10_stageA_dap_fire_first_derisk.py` (+ `--two-compartment`/`--apical-coupling`). Prior: `2026-07-02-emerge10-rung4-stageA-risk1-confirmed-need-two-compartment.md`, `2026-07-02-rung4-sim-two-compartment-tm-port-scoping.md`.
