# CANCELLATION ON SPIKES (GO, 6-seed): a member's OWN property OVERRIDES its inherited class ON THE SPIKING SUBSTRATE — EMERGE-54 apical competition on real-corpus-discovered categories. The exception member's identity→exception drive beats the codon→class drive in `cp_v_apical`; with coincidence detection OFF the override vanishes. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_cancellation_spiking_derisk.py` (reuse-by-import: rung-2's `apply_kernel_update` = the committed `sim.kernels.fused_htm_permanence_update` three-term rule, `_prime_from_winners`, `_host`, `build_inputs`, the EMERGE-42 competitive pooler). numpy-backed real `SimulationBridge`, offline. NO `sim/` edit.
**Verdict:** GO — member-specific cancellation realized on the spiking substrate, 6-seed, all controls pass.

## Why this ran (the fully-spiking directive)
The rate cancellation (`2026-07-08-cancellation-member-exception-overrides-inheritance-real-corpus-GO.md`, 6-seed GO) rode the numpy associative memory. The mission's non-negotiable is FULLY SPIKING on one brain, so this realizes cancellation on the SAME spiking substrate as rung-2 inheritance (EMERGE-42 pooler + committed HTM coincidence kernel + apical read from `cp_v_apical`).

## The mechanism (on spikes — EMERGE-54 apical competition)
The rung-2 substrate already wires **member-identity → property cells** alongside pooler-codon → property. Cancellation adds a DEDICATED exception property and binds the exception member's IDENTITY ensemble to it:
- **INHERIT (rung-2 intact):** a held-out member's category = argmax of the codon→class-property apical drive (`cp_v_apical`), via the shared pooler codon.
- **CANCEL (this rung):** the exception member's identity ensemble is bound to the exception property via `apply_kernel_update` (the committed HTM kernel, dAP coincidence), with a **regulated graded drive** — teach passes added until the member's apical argmax flips to the exception property (the on-spikes analog of the rate adaptive weight). Priming the exception member then drives its OWN property ABOVE the codon-driven inherited class → the apical argmax flips. Other held-out members (no identity→exc binding) still inherit.

## The result — 6-seed (42/43/44/100/101/102), TinyStories, K=1024
```
seed 42  pos=0 passes=6 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
seed 43  pos=0 passes=6 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
seed 44  pos=0 passes=7 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
seed 100 pos=0 passes=7 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
seed 101 pos=0 passes=6 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
seed 102 pos=0 passes=7 | inherit_before=True -> CANCEL=True not_class=True | collateral=0/15 | lesion_override=False
CANCEL all=True | no-collateral all=True | lesion-no-override all=True  -> GO
```
- **CANCEL**: the exception member (which inherited its class BEFORE, via codon→class) → the exception property (its own), NOT the inherited class — a genuine causal flip after teaching ONLY the exception.
- **no-collateral (0/15)**: across ALL 15 held-out members (every category), NONE flip when the exception is taught — member-specific, not a generic drift.
- **lesion-no-override (all seeds)**: with coincidence detection OFF, even the full drive budget (12 passes) does not achieve the override — the cancellation genuinely requires the spiking dAP coincidence kernel, NOT a host computation. (The single most load-bearing control — "genuinely on spikes.")
- **passes 6–7**: the codon→class drive was initially higher; the identity→exc drive had to be built up over 6–7 passes to EXCEED it — real apical competition, not a degenerate exception-always-wins.

## Adversarial self-checks (cheapest load-bearing, inline)
- The flip is causally genuine: `inherit_before=True` (the member inherits via codon before any exception) → CANCEL after teaching ONLY the exception; nothing else changed.
- Not a degenerate EXC-always-wins: it needed 6–7 passes (not 1) to override, and the lesion (coincidence off) never overrides even at 12 passes.
- No test-label leakage: the regulated teach uses only the mechanism's own apical argmax (does it flip to EXC) — a homeostatic "teach until it overrides" regulation, capped.

## Honest scope
- numpy-backed real `SimulationBridge` (offline/CPU) — genuinely on-substrate (real bridge, committed HTM kernel, dAP coincidence, `cp_v_apical` read), the rung-2 scale.
- GO at K=1024 (8 categories, the rung-2 GO scale); the exception is one dedicated property for one member (the de-risk); multiple simultaneous exceptions = a bounded extension.
- The exception property is a distinct random tag (like the class properties); the mechanism is the regulated identity→exc apical drive, not a privileged representation.

## What this establishes
The emergent talkable brain's CANCELLATION reasoning — a member's own property overriding its category's inherited one — is now realized ON THE SPIKING SUBSTRATE (EMERGE-54 apical competition on real-corpus-discovered categories), member-specific (zero collateral across all held-out members) and coincidence-dependent (lesion kills it). Combined with rung-2 spiking inheritance, the real-corpus REASONING (inherit + cancel) is now on spikes — advancing the fully-spiking-one-brain directive. Follow-on: multiple simultaneous exceptions; wiring the spiking cancellation decision into the spoken frame; the cupy/GPU scale.

## Files
`research/runners/_realcorpus_cancellation_spiking_derisk.py`; per-seed `research/findings/raw/_cancelspk_s*.json`. Prior: the rate cancellation `2026-07-08-cancellation-member-exception-overrides-inheritance-real-corpus-GO.md`; the rung-2 spiking inheritance `2026-07-08-knowledge-half-inheritance-ON-SPIKES-real-corpus-rung2-GO.md`; EMERGE-42/54.
