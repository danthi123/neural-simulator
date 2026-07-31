---
type: plan
status: live
date: 2026-06-04
---

# Composer full-clear (A+B) — migrate ALL per-query numpy off the conversational composer — design — 2026-06-04

**Goal:** remove every per-query non-core-biological (numpy) step from `research/runners/core_sim_composition.py`'s
production path, so the conversational composer's READOUT and MEMORY are both realized in the spiking substrate, not
numpy — while keeping numpy parity on the validated capability matrix at every step (no capability regression, ever).
Owner-approved 2026-06-04 (full clear, sequenced A→B, de-risk-first).

**Why:** the shortcut audit (`2026-06-04-composer-shortcut-audit.md`) found the bind/unbind COMPUTE is genuinely
spiking (`hadamard_spiking`, validated pillar n=111) but three per-query numpy steps remain — the cleanup readout,
the superposition+opponency, and the fact storage (the bound fact is a numpy vector held in a Python list). The owner
chose to clear the LAST shortcut (not just the named cleanup) before the fully-grounded run.

**Standing constraints:** numpy parity on the capability matrix (flat / one-attr / two-attr / negation), multi-seed,
NO regression at any step; reuse-by-import, NO `sim/` (protected) edits; honest negatives are the deliverable (a
de-risk that fails to reach parity is a real finding, not a thing to force); both git remotes every outcome;
GPU/CuPy for real runs. Plain professional language.

## The two pieces

### (A) Spiking cleanup circuit — replaces `np.argmax([concepts[w] · est])` (the readout)

The cleanup takes the unbind's noisy estimate `est` (a D-vector) and returns the nearest stored concept. The
item-2 cheap-first (`2026-06-04-spine-item2-cleanup-noisy-est-wall.md`) established the mechanism + the wall: ⛔ SUPERSEDED — that doc's own prescribed lever (integration time) topped out at ~0.78; the noisy-est wall was cleared instead by the NEF thresholded cleanup (worst-case 0.978 / mean 0.993 over seeds 42/43/44, `research/findings/2026-06-05-composer-cleanup-NEF-GO.md`), then folded on-bridge in `research/findings/2026-06-18-one-brain-cleanup-onbridge-GO.md`.
- A spiking matched filter (concept codes as synaptic receptive fields on the ON/OFF channels) is PERFECT at M=320
  on a clean cue (recovery 1.00 at cue-cos 0.87) but the composer's REAL unbind est is noisy (cue-cos ~0.35), where
  it plateaus at **~0.78** vs numpy's 1.00.
- The plateau is **saturation** (true concept + competitors all drive past the neuron's saturation → rates tie).
  The fix is the canonical cortical gain control: **divisive normalization** (Carandini-Heeger) — NOT WTA lateral
  inhibition (which the finding showed HURTS: subtractive inhibition + high threshold suppresses everything).
- The full circuit = matched filter (synaptic dot-product) + **decorrelation** (ZCA, already applied to the codebook
  → common-mode invariance) + **temporal integration** (readout window → rate precision) + **divisive
  normalization** (an FS/shunting pool keeps the concept population in its responsive range).

**The mechanism is the cortical cleanup computation; the missing piece is the divisive-normalization circuit.**

### (B) Substrate-held memory — replaces `self.kb` numpy storage + `bon += o` superposition + `onoff(bon−boff)` opponency

The bound fact is currently produced by a numpy superposition (sum across roles) + ON/OFF opponency, then held as a
numpy (ON,OFF) vector in a Python list and re-driven into the substrate on each query. Substrate-held:
- **Superposition in-network:** the roles' binds drive a SHARED memory bank → their rates sum (superposition is rate
  summation on shared neurons).
- **Opponency in-network:** ON/OFF opponent channels with lateral inhibition (ON−OFF via inhibitory coupling /
  conductance-based shunting).
- **Storage in the substrate:** the superposed bound pattern is HELD by the substrate (not a host vector). The
  mechanism choice (engram-tag set, catalog D.14; vs a recurrent attractor that holds the graded pattern; vs a
  one-shot fast-weight Hebbian imprint) is the (B) de-risk — graded bound-pattern fidelity is the hard part. Resolved
  with its own cheap-first when (A) lands.

## Sequencing — A then B, de-risk-first each

1. **(A) cleanup circuit.** Task 1 = the load-bearing DE-RISK (cheap-first, before any composer rewrite): does
   matched-filter + divisive-normalization + temporal-integration reach numpy parity on the composer's REAL noisy est
   (cue-cos ~0.35, V=320, multi-seed)? **HARD GATE — GO build into the composer only if it reaches parity; otherwise
   an honest boundary (the disclosed readout stands; do NOT ship a sub-parity cleanup that regresses the matrix).**
   Then (on GO) build the region into the composer (replace `np.argmax`), validate numpy parity on the full
   capability matrix multi-seed, no regression.
2. **(B) substrate-held memory.** Designed + de-risked after (A) validates (the bound-pattern-holding mechanism
   choice is its own gate). In-network superposition + opponency + substrate-held storage; numpy parity on the
   matrix, multi-seed.

Why A→B: (A) is bounded with a known mechanism + a clean de-risk; (B) is the bigger re-architecture (how facts are
stored). Doing (A) first de-risks the smaller piece and keeps the matrix green throughout.

## Validation (every step)

The bar is numpy parity on the capability matrix at V=320 production scale, multi-seed (42/43/44 minimum). The
existing numpy path is the regression oracle (it holds 1.00). A step that cannot reach parity is surfaced as an
honest boundary, not forced. The unified-bridge end-to-end capstone (`_unified_brain_capstone_demo.py`) is the
integration check.

## Out of scope
- The dlPFC's Python association graph (a hypothetical step 4; the dlPFC dialogue planning is otherwise on-substrate).
- The setup-time numpy (ZCA decorrelation, code projection) — code preparation, not per-query compute.
- Any change to the validated bind/unbind coincidence compute (already spiking; the load-bearing nonlinearity).
