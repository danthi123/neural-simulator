# Adoption plan: making use of the landscape-survey findings (2026-08-07)

Owner directive: don't let the survey sit on the sidelines — scope + USE the payloads, esp. Axon/Rubicon.
RAG-grounded against our record (so we adopt what we HAVEN'T tried, not re-derive). Ranked by leverage.

## #1 (HIGHEST) — Axon CaP−CaD rule + CT-predict/pulvinar target → gap#4 (deep credit on spikes)
OUR WALL (RAG-verified): our deep-credit is the BURST-DEPENDENT (BDSP microcircuit, Payeur/Naud `dw=η[B−P·E]·Ẽ`)
family — RATE-GO (held-out 0.961, microcircuit noise-robust; `2026-07-07-deep-lever-research-gate`), the
two-compartment `cp_v_apical` substrate is BUILT on the bridge — BUT the SPIKING PORT is stuck:
`2026-08-01-gap4-sweet-spot-LOCATED` (forward representable + reservoir fails, but credit can't train there),
`2026-08-02-gap4-...-does-not-enter-the-learning-regime`, `2026-08-01-...-unsupervised-rule-does-NOT-survive-port`,
and `2026-06-17-onbridge-neural-error-realization-boundary` (the error POPULATION fires but the full
spiking-error→readout-learning didn't converge).
AXON OFFERS (un-tried by us, validated in a working spiking system): a DIFFERENT local rule class —
`DWt=(CaP−CaD)·Tr·RLRate`, Error = temporal derivative of two calcium integrators (CaMKII-fast/CaP vs
DAPK1-slow/CaD; Jang 2023-validated) × eligibility trace (e-prop). AND — the key differentiator — the error/target
is MANUFACTURED BY ANATOMY: L6-CT predicts onto pulvinar (minus phase), L5-IB drives the actual outcome (plus phase),
the difference reverberates back via bidirectional cortex → local `CaP−CaD` everywhere, ZERO host teacher. This
directly attacks our two stuck points: (a) a rule that ENTERS the learning regime where BDSP degraded; (b) a concrete
NEURAL error-signal source where our on-bridge neural-error boundaried.
DE-RISKS (cheap-first, grounded, like-for-like vs our microcircuit):
- **D1 (rate/numpy, cheapest):** implement CaP−CaD (two Ca-integrator temporal-derivative) + a predictive plus/minus-
  phase target on the SAME deep-credit depth task our microcircuit rate-GO used (`sim/dendritic_mlp.py`-scale). GO =
  matches/beats the microcircuit held-out (~0.96) with the anti-cheats (no weight transport, no settling-phase, error
  is the CaP−CaD derivative not a host label). If it can't even match at rate → don't port.
- **D2 (the real question):** does CaP−CaD ENTER THE LEARNING REGIME on the gap#4 spiking sweet-spot where BDSP does
  NOT? (the load-bearing test; our wall is here.)
- **D3:** the CT-predict/pulvinar target-manufacturing as the neural error source (addresses the neural-error boundary).
NOTE: honest — this is a genuine research thread, not a one-shot. But it's the single most-aligned external mechanism
to our hardest stuck problem, and D1 is cheap + decisive.

## #2 — Rubicon delayed-credit (maintained-goal bridge + VSPatch reward-timing) → Gate B vocal credit
OUR STATE: Gate B credit is a GO (Stage-2j ≥5/6); 730705 is a heterogeneity boundary; our credit uses a specific BG
cascade + a Hammond-ΔP baseline. Rubicon offers: (a) the delay bridged by a MAINTAINED GOAL (PT L5-IB recurrent-NMDA),
credit assigned to the held goal not a decayed trace; (b) VSPatch learns reward TIMING → correctly-placed RPE without
a host TD chain; (c) "omission errors only exist because a goal is actively held" (connects to Lane-B omission-veto,
already 6/6 GO). DE-RISK (lower priority — Gate B already GO'd): does a maintained-goal / VSPatch-timing formulation
reduce the heterogeneity-sensitivity (the 730705 class) vs our cascade? Study `sims/pvlv`/`bgventral`/`pfcmaint` first.

## #3 (secondary) — BORN pieces
- IPL+Insula LEARNED bodily self-model (STDP action→feedback forward model + match detector, passes mirror test) →
  our SELF-SCHEMA lane; benchmark alongside the Han self-credit agent (arXiv:2606.30191, the closest small analogue).
- Spiking-WM multi-compartment neurons → our dendritic/WM lane.
- R-STDP BG decision circuit (`basalganglia.py`+`BDM-SNN`) → cross-check ref for our BG (we're ahead; low priority).

## Execution order
D1 of #1 (Axon CaP−CaD rate de-risk) FIRST — cheap, decisive, attacks the hardest stuck wall. Then D2 (spiking port)
if D1 GOs. #2/#3 are parallel lower-priority lanes. Every mechanism RAG-grounded before build (the 2026-08-07
dendritic/TRN overclaim lesson).

## PROGRESS (2026-08-07, RAG-grounded execution)
- **#1 Axon CaP−CaD rule → gap#4: TESTED, NO-GO** (`b71ff9d8`, rate 6-seed). Axon 0.476 vs our microcircuit 0.942 —
  Axon's 2-phase bidirectional target's credit DECAYS THROUGH DEPTH (the feedback-alignment depth wall our
  SST-microcircuit ALREADY surpasses via interneuron error-cancellation); CaP−CaD read is a secondary degrader. ⇒
  do NOT wholesale-adopt Axon; this VALIDATES our microcircuit as the better rule.
- **RESIDUAL-A (spiking BDSP port) is NOT un-started — RAG-caught before re-building it.** The BDSP substrate
  (`fused_bdsp_update`+burst detector+apical→P) is BUILT in `sim/` (additive/default-off, `2026-07-07`); raw
  Burstprop PORTS to spikes (representation forms, probe 0.92, no transport) but is NOISE-LIMITED at 0.664; the
  microcircuit variant is CPU-rate 6/6 GO. **The REAL gap#4 wall** (`2026-08-01`/`2026-08-02`): at depth ≥3 the
  transport-free local rules (Burstprop/FA/KP/microcircuit) do NOT get a deep SPIKING net into the learning regime
  (collapse to majority-class / degrade below reservoir 5/6). The rate versions GO; the depth-3 spiking port doesn't
  learn. Axon doesn't help (worse at depth) ⇒ this is a genuine deep frontier needing a NEW mechanism, a research
  round not a rule-swap. NOT a build target right now.
- **#2 Rubicon delayed-credit → RUNNING (de-risk).** RAG: PFC-WM (maintained-goal substrate) + N9-TD-cue-shift
  (reward-timing) are BUILT+GO; the delayed-reward value task (B4/R4) is SCOPED-not-built. Genuine un-built test =
  maintained-goal(PFC-WM) bridge + VSPatch/TD timing on B4/R4, vs a decayed-trace control.
- Discipline note: RAG-first caught 3 would-be re-derivations this session (dendritic, TRN, RESIDUAL-A) — the mature
  project's obvious levers are mostly tried; genuine next steps need scoping, not rushed builds.
