---
type: finding
status: contributing
date: 2026-08-12
mechanism: gap#4 obligatory-depth-3 credit INSTRUMENT — a spiking task where a depth-2 model FAILS held-out, a depth-3 model GENERALISES, and the jump l3-l2 >= 0.15, so any gap#4 lane's "depth-robust" GO becomes FALSIFIABLE at depth-3
lane: gap#4 ALL-IN (Q5, the falsifiability enabler)
verdict: 6-SEED HONEST NEGATIVE — the instrument is NOT CONSTRUCTIBLE on this substrate. Across 5 task families (parity, xorandxor, mux, nestedxor, hier3), 0/6 satisfy the gate `(l2<=chance+0.06) AND (l3>=0.80) AND (l3-l2>=0.15)` at >=5/6 seeds, on the shared RATE oracle AND on a deep SPIKING-parity run (4000 epochs / 200 spiking-epochs, hidden 24). instrument_exists=False, spiking_go=False, seed_control_verified=True. Obligatory-depth-3-as-a-matched-width GENERALISATION gate is not constructible at practical scale — a depth-2 net matches depth-3 (the finite-read redundancy / weak-coupling: depth is never OBLIGATORY). This SHARPENS the crux wall AND REFRAMES the gap#4 assault's success criterion.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_gap4_obligatory_depth3_instrument_derisk.py
artifacts:
  - research/findings/raw/_gap4_depth3_instrument/instrument_6seed.json
instrument: 5 candidate task families surveyed (parity, xorandxor, mux, nestedxor, hier3); each fitted with matched-width depth-2 vs depth-3 oracles (rate) + a deep spiking-parity attempt; the crux runner's `stage0_depth_genuineness` gate reused (l2<=chance+0.06 ∧ l3>=0.80 ∧ jump>=0.15). Coordinator-recovered from a deferred agent (agent built the runner + ran the survey/6-seed; coordinator wrote this finding). SIM_BACKEND=numpy; NO sim/ edit; cfg.seed verified.
---
<!--derived-->

# gap#4 Q5 — an obligatory-depth-3 credit instrument is NOT constructible on this substrate (6-seed); the finite-read redundancy means depth is never OBLIGATORY — this reframes the assault's success criterion

## Why this lane existed

The gap#4 ALL-IN assault (Q1 Forward-Forward, Q4 DECOLLE, Q2 birdsong-tutor) each claims to get a DEEP (N>=3) spiking net
learning where the top-down transport-free rule could not. But the prior gap#4 deep-credit GOs ("DFA e-prop depth-robust
N2/3/4") are on tasks a DEPTH-2 model might already solve on held-out — so "depth-robust" does not PROVE depth-3 credit.
This lane tried to build the falsifiability INSTRUMENT: a task where only genuine depth-3 credit can generalise, so every
lane's GO can be tested against it.

## Result — 6-seed NEGATIVE (`research/findings/raw/_gap4_depth3_instrument/instrument_6seed.json`)

<!--derived-->
Across FIVE task families (parity, xorandxor, mux, nestedxor, hier3), fitted with matched-width depth-2 vs depth-3
oracles, **0/6 satisfy the obligatory-depth-3 gate** `(l2 <= chance+0.06) AND (l3 >= 0.80) AND (l3-l2 >= 0.15)` at the
>=5/6-seed robustness bar — on the shared RATE oracle AND on a deep SPIKING-parity run (4000 epochs, 200 spiking-epochs,
hidden 24). `instrument_exists=False`, `spiking_go=False`, `seed_control_verified=True` (the substrate is genuinely
seeded; this is not a seeding artifact).

The failure mode is structural, not a tuning miss: whenever a depth-3 model generalises (l3 >= 0.80), a **matched-width
depth-2 model also reaches it** (l2 not <= chance+0.06) — so the jump never clears 0.15. On this point-neuron spiking
substrate, **depth is never OBLIGATORY**: the finite-spike read's redundancy (the same "weak-coupling" the SNN locality
literature names, arXiv:2402.01782) lets a shallower net match a deeper one on any constructible matched-width task.

## What this settles for the assault (and the honest reframe)

<!--derived-->
- **The success criterion for the gap#4 lanes must change.** "Obligatory-depth-3 generalisation" is NOT a usable bar —
  it is unconstructible here. The PRIMARY, falsifiable metric for Q1/Q2/Q4 is therefore the FIRST-ORDER one the wall
  findings already used: **does the deep net ENTER the learning regime — leave majority-class and BEAT the frozen
  reservoir** — where the top-down FA/KP rule collapses to majority-class. Depth-obligatoriness is set aside as
  unmeasurable, not as achieved.
- **This SHARPENS the crux wall.** The `2026-08-02` finding located the wall at "no directed credit through the
  finite-spike read." This adds: even the TASK side can't force depth to matter — the substrate's read redundancy makes
  shallow and deep interchangeable on matched-width generalisation. That is consistent with (and evidence toward) the
  "the substrate, not the rule, is the wall" tell: if a point-neuron spiking read cannot make depth obligatory, the
  eventual surpass may be a substrate change (e.g. the ALIF swap), not a credit rule.
- **Honest scope:** this is a NEGATIVE on CONSTRUCTIBILITY of an instrument, not a claim that deep spiking credit is
  impossible — it says we cannot cleanly MEASURE depth-obligatory credit on matched-width generalisation here, so we
  measure enter-the-regime instead. Runner-side; NO sim/ edit.
