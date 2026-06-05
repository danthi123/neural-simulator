# Conversion Phase 1 (cheat B, cleanup) — TPAM cleanup de-risk → GO at D>=256 — 2026-06-05

First execution step of the conversational-cheat-conversion plan
(`docs/plans/2026-06-05-conversational-cheat-conversion-plan.md`). The RF composer's cleanup is a numpy phase-cosine
`argmax` (cheat B). The biology-grounded replacement (per `2026-06-05-cheat-BC-spiking-phasor-cleanup-memory-research.md`)
is a TPAM = complex Hopfield (Frady-Sommer 2019) = CA3 pattern completion + striatal WTA: vocabulary in `W = S S*/D`,
magnitude-gated phase-preserving transfer (the resonate-and-fire `|Z|>floor` IS the threshold), winner `argmax|S* z|`.

**De-risk GATE: the TPAM cleanup picks the SAME winner as the numpy argmax on the composer's REAL noisy unbinds,
multi-seed.** `research/findings/raw/_phase1_tpam_cleanup_derisk.py` (3 stored facts, 9 role-unbinds, seeds 42/43/44):

| D | TPAM==argmax parity | argmax-correct |
|---|---|---|
| 128 | 21/27 | 27/27 |
| **256** | **27/27** | 27/27 |
| 512 | 27/27 | 27/27 |
| 1024 | 27/27 | 27/27 |

**Verdict: GO at D>=256.** The TPAM matches the numpy argmax EXACTLY at D>=256 (full parity, multi-seed). The D=128
shortfall (21/27) is the **dense-phasor capacity wall** (17 concepts packed into D=128 → spurious attractors); it is
lifted by D — the same capacity dial that lifted two-attribute. Theta-independent (the gate isn't the issue; capacity
is). So the numpy argmax cleanup is REPLACEABLE by the biology-grounded TPAM at the composer's operating D (use D>=256
for the codebook scale; the agent default D=128 should bump to 256 for the cleanup, or the cleanup runs at a higher-D
codebook projection).

## What this de-risk does + does NOT establish
- ESTABLISHES: the biology-grounded cleanup MECHANISM (complex-Hopfield/CA3 TPAM) is a valid drop-in for the numpy
  argmax at D>=256 — the algorithm matches. The cheat is CONVERTIBLE.
- Does NOT yet clear the cheat: this is the numpy TPAM (validating the algorithm). The CHEAT is cleared by the
  ON-BRIDGE spiking realization. Two on-bridge routes (next step): (a) the matched-filter readout — a concept neuron
  per codebook entry firing proportional to the phase-correlation `|S* rec|` + WTA (the NEF-cleanup analogue on
  phasors; simplest, IS the argmax); (b) the full TPAM iterate via the bridge's complex synapse + RF magnitude gate
  (the attractor/pattern-completion version; needs the iterate mapped onto the RF dynamics). Route (a) is the simpler
  first integration (it is literally the validated argmax, realized in spikes); route (b) is the fuller CA3 attractor.
- Honest note: clearing B fully also wants D>=256 (a small default bump) — cheap, consistent with the substrate's
  capacity-dial behavior.

## Next (the integration + the remaining phases)
Integrate route (a) the spiking matched-filter cleanup as an opt-in on the composer; re-validate the agent's full
suite at parity; commit. Then Phase 2 (C-A phasor weight-store), Phase 3 (A grounded codes), Phase 4 (D substrate
association graph) per the plan. The 4 honest boundaries stay deferred + disclosed.

## Artifacts
`research/findings/raw/_phase1_tpam_cleanup_derisk.py` (the de-risk; numpy TPAM vs argmax parity, D-sweep, multi-seed).
NO sim/ edits. The TPAM mechanism is already in `research/runners/resonate_fire_fhrr.py::ResonateFireTPAM` (numpy ref).
