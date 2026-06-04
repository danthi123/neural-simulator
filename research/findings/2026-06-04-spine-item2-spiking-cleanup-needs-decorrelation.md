# Spine item 2: the spiking cleanup needs DECORRELATED codes (numpy argmax "cheats" via common-mode invariance) — 2026-06-04

**One line:** A spiking matched-filter + WTA cleanup on the core `SimulationBridge` (concept codes as synaptic
receptive fields; the dot-product match becomes synaptic, the `argmax` becomes lateral inhibition) works **perfectly
on low-correlation codes** (cos ~0: spiking 0.99 vs numpy 1.00) but **collapses on the real captured-code regime**
(cos ~0.80: spiking 0.17 vs numpy 1.00). The reason is the deliverable: the spiking cleanup is **not common-mode
invariant**, where numpy `argmax` is — so the spiking cleanup *requires* decorrelated codes, which is exactly why
the cortex decorrelates.

## What was built (`research/findings/raw/_spiking_cleanup_core_probe.py`)

A cleanup region on the core bridge: input_ON(D) + input_OFF(D) + concept(M). Each concept neuron's incoming
weights ARE that concept's code (`code_c_on` from input_ON, `code_c_off` from input_OFF), so its drive =
`code_c · est` computed by synaptic propagation — the matched filter. Optional concept→concept lateral inhibition
(WTA) for single-winner selection. This removes BOTH numpy steps (the dot-product match AND the `argmax`). Driven by
the cue's (e_on, e_off) via the composer's own `onoff` + `_scale_to_current`. Operating point handled signed codes
via the ON/OFF channels.

## Result (M=32, D=512, spiking recovery vs the numpy `argmax` baseline)

| code regime | cue-cos 1.0 | 0.91 | 0.74 | 0.60 | 0.41 |
|---|---|---|---|---|---|
| **cos ~0** (decorrelated) spiking | 1.00 | 0.99 | 0.97 | 0.93 | 0.75 |
| cos ~0 numpy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **cos ~0.80** (captured regime) spiking | **0.17** | 0.15 | 0.17 | 0.08 | 0.04 |
| cos ~0.80 numpy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- On **decorrelated** codes the spiking matched filter tracks numpy closely (perfect on clean cues; degrades
  gracefully under noise — the anti-cheat). The mechanism is correct.
- On **correlated** (cos-0.80) codes it collapses to 0.17 *even on clean cues*. WTA (w_inh=80) made it **worse**
  (0.02) — uniform inhibition on uniformly-high drive suppresses the winner too.

## Diagnosis — the biology-translatable insight

Correlated codes share a strong common component. The matched filter's drive to concept neuron c is
`code_c · est`; the shared component contributes a large, **nearly equal** amount to every concept, so all concept
neurons fire near-saturation and the small concept-specific residual cannot separate the firing rates → wrong
winner. numpy `argmax` is **invariant** to a constant added to all match scores (it cancels in the comparison), so
it ignores the common mode for free. **Real spiking neurons do not have that invariance** — saturating firing
rates destroy the residual. So the cortex's decorrelation (efficient coding; Atick-Redlich / Olshausen-Field) is not
just an efficiency trick — it is what makes downstream spiking matching *possible*. numpy `argmax` was quietly
relying on a common-mode invariance that biology has to earn by decorrelating.

## Reframe — decorrelation is the linchpin, not an item-3 option

Stage 1.5 flagged decorrelation (ZCA) as an *efficiency* lever (lower D). This result promotes it to a
**prerequisite for the spiking cleanup (item 2)** — and it is the same step that lowers D for bind/unbind. So one
biological move (decorrelate the captured codes) does three things at once:
1. makes the spiking cleanup work (this finding),
2. lowers the dimensional budget D (stage 1.5),
3. is independently biologically grounded (the ventral hierarchy decorrelates).

**Updated path:** decorrelate the captured codebook globally (ZCA) → then (a) the spiking matched-filter cleanup
replaces numpy `argmax` on the core bridge, and (b) bind/unbind run at lower D. The decisive test is integration:
decorrelate the codebook in the composer, swap `argmax` for the spiking cleanup, re-run the capability matrix. A
residual normalization (divisive/subtractive common-mode removal in the cleanup circuit) is the fallback if some
correlation must be tolerated without full decorrelation.

## Files
- `research/findings/raw/_spiking_cleanup_core_probe.py` (matched-filter + WTA cleanup on the core bridge; `--rho`,
  `--w-inh`, numpy baseline)
- raw: `_spiking_cleanup_*` runs (cos-0 and cos-0.80, ±WTA)
