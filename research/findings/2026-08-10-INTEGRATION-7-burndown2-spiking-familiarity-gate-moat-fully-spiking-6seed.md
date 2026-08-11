---
type: finding
status: go
date: 2026-08-10
mechanism: spiking anti-Hebbian familiarity/source-monitor gate (v320) deciding the no-confab moat abstain for plasticity-learned facts in the live chat
lane: E-language / INTEGRATION
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/plasticity_facts_live_chat_spiking_gate_6seed.json
runner: research/runners/_teacher_loop_facts_into_live_chat_derisk.py (--spiking-familiarity-gate)
instrument: INTEGRATION #7's live-chat de-risk with `--spiking-familiarity-gate`, which routes the plasticity-learned facts' abstain decision through `research/runners/_spiking_conjunctive_familiarity_gate.py::SpikingConjunctiveFamiliarityGate` (the standing v320 spiking anti-Hebbian gate over a genuine spike-phasor conjunction) in place of the host numpy `RealAntiHebbianFamiliarity`; SIM_BACKEND=numpy; cfg.seed-controlled.
---

# INTEGRATION #7 burn-down #2 — the no-confab moat is now decided on SPIKES (6/6 GO)

INTEGRATION #7 wired plasticity-LEARNED facts into the live chat with a LEARNED familiarity/source-monitor gate as the
no-confab moat — but that gate (`RealAntiHebbianFamiliarity`) was a HOST numpy anti-Hebbian projector (correct
mechanism, host implementation — a declared burn-down). Burn-down #2 replaces it with the SPIKING realization (the
standing `2026-06-11-familiarity-gate-v320-GO` gate, which matches the host abstention moat at production scale), so the
brain's ABSTAIN for its own plasticity-learned facts is now decided on spikes.

## What was built (additive, runner-side, NO `sim/` edit)

<!--derived-->

The two anti-Hebbian gates are the SAME projector (`N(x)=‖x‖²−xᵀWx`, Gram-Schmidt orthonormal basis); they differ only
in the input render — the host gate takes a raw real cue vector, the v320 gate reads novelty through the
resonate-and-fire I/Q render `[cos 2πφ, sin 2πφ]`. `SpikingConjunctiveFamiliarityGate` is a drop-in for `#7`'s
`ConjunctiveFamiliarityGate` (identical `imprint`/`novelty`/`novelty_settled`/`familiar`/`lesion` interface over
`(env, referent, action)`, same `NOV_GATE=0.5`): the cue is a genuine spike-phasor conjunction (percept → phasor phase
via the complex-projection bridge, per-action phase codes, BOUND by the real resonate-and-fire `phase_sum_neuron` —
reuse-by-import, verified to compute the modular phase-add to ~2e-3), and novelty is read by the v320 spiking
anti-Hebbian pool. An additive `--spiking-familiarity-gate` flag (default OFF → `#7` byte-identical) routes both gate
constructions through a `_make_fam(seed)` factory.

## Result — 6/6 GO (`research/findings/raw/lanes/stageA/plasticity_facts_live_chat_spiking_gate_6seed.json`, verdict GO, n_go 6/6; coordinator ran the sweep + verified from the raw log)

<!--derived-->

Per seed (42/43/44/100/101/102), the FULL #7 gate holds with the SPIKING gate in place:

| metric | all 6 seeds |
|---|---|
| taught-recall base→treat | 0 → 3/3 |
| taught-recall FROZEN readout | 0 |
| moat false-accepts (untaught + OOD) | 0 |
| grounded-reply delta | +1 |
| held-out recall (dax) | 0.90–1.00 |
| **SPIKING-gate lesion (novelty margin)** | **0.66–0.75 → 0.00** |
| gate-OFF false-accepts | 4–5 |
| mispaired-teacher vs main | 0.00–0.01 < 0.94–0.99 |
| byte-identity (flag OFF vs #6 default) | fully identical |

The load-bearing control is the LESION of the SPIKING gate: silencing the spiking anti-Hebbian pool collapses the
novelty margin to 0.00 on every seed (untaught cues would then answer — confabulation returns), so the abstain rides
the LEARNED SPIKING weights, not a host path. The spiking gate's taught-vs-untaught margin (+0.64 in isolation, seed 42)
is slightly tighter than the numpy gate's (+0.79) but comfortably robust — the flagged risk that the `NOV_GATE=0.5`
margin might shrink at chat-vocab scale did NOT materialize.

## Scope / honesty — what is now fully spiking, what remains

<!--derived-->

**Fully spiking now:** the plasticity-learned facts' ACQUISITION (e-prop weight change, #7) AND the moat ABSTAIN
decision (the v320 spiking anti-Hebbian gate over a real resonate-and-fire phasor conjunction). Burn-down #2 of #7 is
CLOSED.

**Declared burn-downs still remaining (per THE LAW):** (1) TWO co-resident bridges — the e-prop acquisition net + the
gate's phasor render + the conversational `build_one_brain` bridge are co-resident Izhikevich substrates in one process
but NOT merged into one bridge (the "one brain" step — burn-down #1, the next arc); (2) the argmax patient read-out
(neural-motor-read-out target); (3) the SMALL-K (3), jointly-taught scale — continual/sequential breadth is the OPEN
scale-up arc (`frac_recalled~1/N`). This is a demo-scale burn-down beside #6's host-stored breadth.

Workflow note: the build agent BUILT the adapter + ran only a 1-seed smoke + RETURNED the exact 6-seed command; the
coordinator ran the 6-seed sweep, verified from the raw log, and merged — the correct division of labor (a build agent
must not run the multi-seed sweep).
