# EMERGE-10 / rung-4 Stage A — BUILD-INFORMATIVE: the fire-first dAP mechanism is REAL on the real substrate (plateau-specific advantage +40–100 pA, lesion + desync both collapse), but a single-compartment coincidence-plateau injected as SOMATIC current cannot do SUB-THRESHOLD priming (the "predictive != active" invariant) — scoping risk-1 CONFIRMED. The rung-4 port needs a genuine two-compartment `cp_v_apical` NeuronModel. The cheap-first de-risk did its job: it sized the ONE genuinely-new biophysical requirement before the full port.

**2026-07-02 (autonomous).** Runner `research/runners/_emerge10_stageA_dap_fire_first_derisk.py`; results `research/findings/raw/_emerge10_stageA_*.json`. Reuse-by-import; NO `sim/` edit (the plateau kernel exists + is byte-inert when off); CPU numpy-backend `SimulationBridge`; multi-seed 42/43/44.

## What Stage A tests
The rung-4 scoping identified the ONE genuinely-new biophysical behavior of the sim/ HTM-TM port: a distally-PRIMED (dAP-plateau) cell must fire FIRST / at lower feedforward drive than an unprimed cell (the bias that lets predicted cells win the per-column WTA), AND the plateau alone must NOT fire the cell (predictive != active). Stage A wires one column of Izhikevich cells + a `coincidence_detector`-tagged distal `context->column` pathway (the dAP = `fused_coincidence_plateau`) on a real bridge and sweeps: with the same context volley, coincidence ON (plateau) vs OFF (lesion), across a feedforward-drive grid.

## Results — the mechanism works, but not sub-threshold (single compartment)
- **Plateau-specific fire-first is REAL and load-bearing.** With the SAME context AMPA volley, coincidence-ON primes the cell to fire at a LOWER feedforward drive than coincidence-OFF: plateau-specific advantage **+40 to +100 pA**, multi-seed. **dAP-lesion** (coincidence off) collapses it; **desynchronized context** (no synchronous volley -> no coincidence) collapses it to the unprimed baseline. So the coincidence plateau genuinely biases the somatic competition (fire-first), and the coincidence-not-rate property holds.
- **But there is NO sub-threshold-priming regime.** The "predictive != active" invariant (plateau alone, no feedforward, must not fire) FAILS across the entire sweep. Fine plateau-strength sweep (ctx_weight 0.1, all AMPA negligible):

  | plateau scale | plateau-specific advantage | plateau-alone fires? (noFF) |
  |---|---|---|
  | 0.03 | +0 (no priming) | 0.00 (inert) |
  | 0.04 – 0.06 | **+40** (primes) | **1.00 (fires alone)** |
  | 0.08 – 0.5 | +50 – +100 | 1.00 (fires alone) |

  The transition is SHARP and all-or-none: the plateau is either too weak to prime (inert) or strong enough to fire the cell by itself. There is no strength at which it primes (lowers the feedforward threshold) WITHOUT firing the cell alone.

## Verdict: BUILD-INFORMATIVE — scoping risk-1 CONFIRMED (NOT a wall)
A regenerative all-or-none coincidence plateau injected as a SOMATIC current is intrinsically binary — it either crosses the somatic threshold (fires) or does nothing. A single compartment cannot hold a large priming depolarization sub-threshold. So the current-injection dAP cannot reproduce the HTM "predictive != active" distinction. **The rung-4 port needs a genuine two-compartment `cp_v_apical` NeuronModel**, where the plateau lives in a SEPARATE apical compartment that couples sub-threshold to the soma (a large apical dendritic spike biases the soma toward firing without directly firing it). This is exactly the risk the scoping pre-identified (risk 1) and sized (~120–180 lines + a state array, the same magnitude as the Burstprop `TWO_COMPARTMENT_BURST` scope; the `RESONATE_AND_FIRE` guarded-additive `NeuronModel` is the precedent template). Per the master directive, `sim/` edits for faithful biology are fair game — this is the next mechanism to build, not a stop. The cheap-first de-risk worked perfectly: it found the exact substrate requirement in isolation (no `sim/` edit, hours) before the full port.

## Next: rung-4 Stage A' — the guarded two-compartment `cp_v_apical` dAP neuron
Build an additive/default-off/byte-identical-when-off two-compartment `NeuronModel` (template: how `RESONATE_AND_FIRE` was added, `sim/enums.py`/`sim/bridge.py`): a `cp_v_apical` state array + a coupled somatic-apical ODE where the distal coincidence-plateau conductance drives the APICAL compartment, which couples sub-threshold to the Izhikevich soma (the fire-first bias) but does not directly cross the somatic threshold. Re-run Stage A: the plateau-specific fire-first advantage should PERSIST while `noFF` drops to ~0 (predictive != active). Single-variable, gated, multi-seed. Then Stage B (per-column WTA + frozen numpy-learned permanences => EMERGE-9c spiking-inference parity) and Stage C (the three-term permanence kernel => EMERGE-9d parity).

## Honest scope
- Stage A used the existing `fused_coincidence_plateau` (steep all-or-none, the closer match to the act_th-thresholded HTM segment) with NO `sim/` edit; the graded sibling would have the same somatic-binary problem (a somatic current strong enough to prime is close to firing regardless of shape — the separation needs a compartment, not a smoother current).
- The fire-first mechanism + both anti-cheats (lesion, desync) are validated on the real substrate; only the sub-threshold invariant needs the apical compartment.
- A spiking BOUNDARY where the numpy GO'd is itself a real finding (localizes the exact substrate requirement), not a stop.

## Artifacts
`research/runners/_emerge10_stageA_dap_fire_first_derisk.py`, `research/findings/raw/_emerge10_stageA_*.json`. Prior: `2026-07-02-rung4-sim-two-compartment-tm-port-scoping.md`, `2026-07-02-emerge9c-spiking-tm-rung3b-GO.md`.
