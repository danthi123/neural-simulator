# Next-arc grounding: theta-phase multiplexing for conversational multi-item holding -- the owner-preferred biological mechanism that may SIDESTEP this session's separation-vs-reliability boundary, because it separates items in TIME (theta phase slots) not in SPATIAL pattern. Existing sim to adopt-from: Ursino-Cesaretti-Pirazzini 2022 (spiking Lisman-Idiart theta-gamma multi-item WM). Honest caveat recorded: a 2025 Nature Neuroscience result contests strict phase==order. CHEAP-FIRST gate pre-registered before any spiking build.

> ## ⚠️ CORRECTION / DOWNGRADE (2026-05-31, same day, on checking prior in-project work)
> This "NEXT ARC" proposal is SUPERSEDED. theta-gamma multiplexing is NOT an untried direction:
> its ALGEBRA is already validated with decisive controls (2026-05-24 Direction E PERFECT 1.000 at
> loads {2,3,5,7}; 2026-05-23 FHRR "N16 realisable"), and its SPIKING-SUBSTRATE composition application
> already hit a DECISIVE 5-architecture CONVERGENT CEILING (2026-05-20-THETA-GAMMA-decisive-honest-
> negative-...md: GATE=FAIL, per_regime_advantage NEGATIVE at N=5). The algebra was never the
> bottleneck; the substrate composition is the wall. My night-arc strategic synthesis (2026-05-31-
> NIGHT-ARC-strategic-synthesis-...md) had ALREADY pivoted correctly: DG fundamental boundary -> P4
> (advance the VALIDATED conversational stack), explicitly noting the G.20/engram codes re-instantiate
> the oracle shortcut and that grinding more biologization tuning is not worth GPU time once the
> boundary is characterized. So proposing a theta-multiplexing spiking build as "the next arc" was a
> REGRESSION to an already-ceiling'd direction. RESIDUAL value of this grounding pass (kept): the clean
> cross-arc INSIGHT that temporal (phase-slot) separation sidesteps THIS session's SPATIAL DG
> separation-vs-reliability boundary, + the Miller-7-under-jitter recovery (cheap probe). That insight
> does NOT reopen theta-gamma as a substrate direction. ACTIVE direction remains P4 (the in-flight
> multi-hop hub-reuse test + scaling/usability of the working conversational stack). Body preserved
> below as written; read it through this banner. Root cause: I wrote this before reading the extensive
> 2026-05-19..05-24 in-project theta-gamma work (the standing "check existing work first" directive).

**Date:** 2026-05-31
**Status:** Forward-looking design grounding (not a committed build). Written during the GPU-busy window while the P4 multi-hop hub-reuse decisive test runs. Purpose: set up the OWNER-ALIGNED next arc regardless of the hub-reuse verdict, honoring the standing "check existing sims FIRST" directive (MEMORY.md feedback_check_existing_sims_first) and the 2026-05-19 conversational-path reframe (build conversation on theta-multiplexing / theta-gamma mode-unification / generative replay, NOT static two-store retrieval ranking).

## The strategic tension this resolves

This session produced two BIOLOGICAL conversational-mechanism attempts, both ending in clean boundaries:
1. Phase-factored integrated loop (theta-gamma for working-memory EMERGENCE) -> VOID (two horns: STDP selectivity is emergent-but-unstable; engram store is stable-but-not-emergent).
2. DG biologization (hippocampal pattern separation for compositional symbols) -> FUNDAMENTAL BOUNDARY (a single competitive-sparse-coding stage cannot deliver separation AND reliability from overlapping inputs; no DG size threads both; CA3 completion worsens separation).

After those, I pivoted to extending the validated RETRIEVAL stack (multitag / engram / G.20 multi-hop). That stack works, but it is exactly the "static two-store retrieval ranking" the owner DE-PRIORITIZED on 2026-05-19. So the multi-hop arc (and its hub-reuse scrutiny, in flight) is useful for characterizing what the retrieval substrate can do, but it is NOT the owner's stated direction. The owner's direction is biological conversational mechanism.

The question that unblocks the owner's direction: is there a biological mechanism that does NOT hit this session's separation-vs-reliability boundary?

## The insight: temporal separation sidesteps the spatial tradeoff

The boundary I characterized is SPATIAL. It is a property of competitive k-winners-take-all sparse coding: to make two concepts' active-neuron SETS different (low between-concept cosine) you need few, strongly-competitive winners -- but few near-threshold winners are maximally sensitive, so a concept's own two noisy observation-halves pick different winners (low within-concept cosine, unstable). Separation and reliability pull on the SAME sparsity knob in opposite directions. DG size moves along the curve; it does not escape it.

Theta-phase multiplexing separates items along a DIFFERENT axis: TIME. Each item is a gamma-synchronized assembly that fires in its OWN theta-phase slot (~4 Hz theta carries ~7 gamma sub-cycles -> ~7 item slots, the classic Miller capacity). Two items do not need different spatial codes to be non-interfering -- they need different PHASES. A single item can therefore use a DENSE, STABLE spatial code (high within-concept reliability, the easy side of the tradeoff) while still being kept distinct from other held items by temporal segregation. The spatial separation requirement is RELAXED because separation is carried by phase, not by code-orthogonality.

This is the mechanistic reason the owner's preferred mechanism may succeed where the spatial-separation attempts failed: it routes around the exact knob that produced the boundary. It is not a guarantee (see caveat + the cheap-first gate below) -- but it is a principled, falsifiable, biology-grounded hypothesis that directly serves the stated goal.

## Existing sim to adopt-from (the "check existing sims first" discipline)

PRIMARY: Ursino M., Cesaretti N., Pirazzini G. (2022), "A model of working memory for encoding multiple items and ordered sequences exploiting the theta-gamma code," Cognitive Neurodynamics, DOI 10.1007/s11571-022-09836-9 (open: PMC10050512). A SPIKING-network model in which:
- cell assemblies representing individual items fire highly synchronized in the gamma band (>30 Hz);
- their PHASE within a slower theta rhythm (~4 Hz) defines the sequential ORDER of items;
- result: stable, non-interfering simultaneous maintenance of multiple items + their order.
This is precisely a buildable, published spiking instantiation of the Lisman-Idiart code -- the mechanism the owner named. Adopt-from rather than reinvent.

SUPPORTING / context (from the same grounding pass):
- Lisman & Jensen, "The Theta-Gamma Neural Code" (Neuron 2013, PMC3648857) -- the canonical theory statement.
- Pirazzini & Ursino 2024 (Front. Neural Circuits, 10.3389/fncir.2024.1326609) -- theta-gamma coupling extended to sequential memory / imagination / "dreaming" (generative replay flavor -- relevant to the owner's generative-replay pillar).
- eNeuro 2023 (ENEURO.0373-22.2023) -- gamma/beta-band WM spiking model, concurrent vs sequential items.

HONEST CAVEAT (do not bury): Nature Neuroscience 2025, "Phase of firing does not reflect temporal order in sequence memory of humans and recurrent neural networks" (s41593-025-01893-7) -- direct human + RNN evidence that firing phase does NOT cleanly encode temporal ORDER. This contests the strict phase==order reading. Implication for us: theta-phase multiplexing as a multi-ITEM HOLDING / non-interference mechanism is on firmer ground than theta-phase as an ORDER code. Scope the next arc to the holding/segregation claim first (the part that sidesteps the boundary), treat order-coding as a separate, more-contested sub-goal.

## How it maps to THIS substrate (reuse-by-import, no rewrite premise)

Our substrate already has the two ingredients:
- concept pools (G.20 sparse / v16 pools) = the gamma assemblies (a held item = an active concept pool).
- a theta-timing controller already exists from the parked integrated-loop Task 2 (theta-gamma timing controller) -- reusable by import; no new autograd, no new learning rule.
The conversational payoff: a conversation needs to HOLD several things at once (the topic, the current item, the pending response) without them merging -- exactly multi-item WM. Multitag retrieval gives single-shot association; theta-multiplexing would give simultaneous maintenance of multiple associated concepts in distinct phase slots, which is the substrate a genuine conversational turn needs (and which static retrieval ranking, by the owner's framing, does not provide).

## Pre-registered next step (CHEAP-FIRST, before any spiking build)

Per the standing cheap-first-before-spiking discipline (the same gate that correctly killed the integrated-loop and denoiser builds before they cost GPU-days):

CHEAP PROBE (CPU/numpy, stdlib+numpy only, no protected-module import): model N gamma-assembly items as DENSE stable spatial codes (the easy side of the tradeoff) assigned to distinct theta-phase slots. Drive them in a multiplexed theta cycle. Measure: (a) can all N items be read back from their phase slots above an abstention floor (the HOLDING claim); (b) does a single item retain a STABLE within-item code across noisy halves (reliability -- expected YES, since spatial separation is no longer required); (c) the CONTROL that reproduces the failure -- the SAME items with NO phase separation (all driven in one slot) must interfere/collapse, so a phase-separated PASS is a genuine falsification not a trivial pass.

PRE-REGISTERED FIXED BAR (frozen, never tuned by results): N>=4 items, all read back >= 0.90 from their phase slots AND the no-phase control collapses (mutually-exclusive readback < 0.50). Three-state: RESOLVES (build the spiking version) / BOUNDARY (phase multiplexing also fails on this substrate -- a high-value honest negative) / DOES-NOT-RESOLVE. Instrument-validity checked first; malformed -> CANNOT-CONCLUDE not crash.

HARD GATE: spiking build proceeds ONLY on cheap RESOLVES. A cheap BOUNDARY is propagated as an honest finding and the spiking build does NOT start -- same discipline as the integrated-loop and denoiser arcs.

## Disposition + composition with the in-flight hub-reuse verdict

This note is BANKED grounding, owner-aligned, ready to execute regardless of the multi-hop hub-reuse outcome:
- hub-reuse ROBUST -> retrieval multi-hop is a real (if mechanistically-shallow) capability; bank it, then THIS theta-multiplexing arc is the next biological step toward the owner's stated direction.
- hub-reuse DEGRADES/NEGATIVE -> the retrieval extension is bounded; THIS arc becomes the primary next direction immediately.
Either way the next concrete action after the hub-reuse verdict is recorded is: implement + run the cheap-first theta-multiplexing holding probe above.

## Discipline

Grounding/research only; nothing built yet; no protected/frozen/moat/sim/runner module touched. The next-arc build is pre-gated cheap-first with a frozen bar + reproduce-the-failure control + three-state verdict, inheriting the exact de-risking discipline that produced this session's honest negatives. The boundary-escape claim (temporal sidesteps spatial) is a HYPOTHESIS to be falsified by the cheap probe, not a result. The honest 2025 counter-evidence on phase==order is recorded and scopes the claim to holding/non-interference. Existing sim (Ursino 2022) named for adopt-from per the standing "check existing sims first" directive.
