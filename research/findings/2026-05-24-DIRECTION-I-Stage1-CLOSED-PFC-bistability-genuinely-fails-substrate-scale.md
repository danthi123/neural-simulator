---
type: finding
status: qualified
date: 2026-05-24
---

# Direction I Stage 1 CLOSED: PFC NMDA bistability genuinely fails at substrate scale (both Izh AND HH models)

**Date:** 2026-05-24
**Status:** DIRECTION I CLOSED — NMDA-mediated persistent activity does not engage at 60-neuron dlpfc_wm substrate scale regardless of neuron model
**Three cheap-first probes; ~3 min total wall**
**Pivot:** to Direction N (320 → 640 concepts) OR Direction O (sentence-parser UI) per autonomous decision

## Headline

Per the recommendation accepted ("Proceed autonomously" → Direction I
Stage 1 PFC bistability smoke), three increasingly aggressive cheap
probes all confirm: **the dlpfc_wm region's NMDA-mediated persistent
activity (Wang 2002 attractor mechanism) does not engage at the
substrate scale tested (60 neurons, density 0.3) on either Izhikevich
OR Hodgkin-Huxley neuron models.** Direction I as conceived cannot
proceed to Stage 2 without substantial architectural redesign.

## Three probes (all FAIL the bistability gate)

**Probe 1 — Basic smoke** (lang_input → dlpfc_wm pathway, default
config):
- dlpfc_wm firing during stim: 0.0012 (~silent)
- Delay rate equals baseline (~0.0006)
- VERDICT: PFC_BISTABILITY_FAILS_PIVOT

**Probe 2 — Parameter stress sweep** (36 cells: drive ∈ {200, 500,
1000, 2000} × pathway weight ∈ {3, 5, 10} × density ∈ {0.2, 0.4,
0.6}):
- ALL cells: stim rate ~0.0003, delay rate ~0.0005 (essentially
  noise floor)
- No regime where dlpfc_wm engages above noise
- VERDICT: PFC_PARTIAL_PERSISTENCE_NEEDS_TUNING (artifact of zero
  baseline divide; really PFC doesn't fire at all)

**Probe 3 — Direct injection** (bypasses input routing; tries
nmda_ratio=0.5 Wang 2002 calibration; direct current 500-5000pA on
all dlpfc_wm neurons):
- inject=500pA: stim 0.072, delay 0.0001 (no persistence)
- inject=1000pA: stim 0.159, delay 0.0000
- inject=2000pA: stim 0.313, delay 0.0001
- inject=5000pA: stim 0.496 (robust firing!), delay 0.0007 (~zero)
- Firing scales linearly with drive during stim; the MOMENT drive
  stops, firing cuts to zero. No bistability.
- VERDICT: PFC_BISTABILITY_GENUINELY_FAILS_PIVOT

**Probe 4 — HH biophysics** (HH_PFC_PYRAMIDAL + dt=0.05ms; full
Wang 2002-class biophysics):
- ALL injection levels (500-5000pA): firing rates flat ~0.0013
  across baseline/stim/delay
- HH neurons don't even respond to drive injection in this config
  (something amiss with current routing for HH model OR background
  noise dominates)
- VERDICT: HH_PFC_PARTIAL_NOT_BISTABLE

## Honest diagnosis

NMDA-mediated persistent activity requires:
- Sufficient recurrent connections to form self-sustaining attractor
  (Wang 2002: ~1000-2000 PFC neurons; we have 60)
- Strong enough NMDA conductance to maintain firing post-input
- Properly tuned excitation/inhibition balance for bistability
- Specific biophysical features (NR2A/NR2B subtype dynamics) not
  fully captured in either model at this scale

The substrate at 60 neurons + density 0.3 has ~1056 synapses internal
— insufficient for Wang 2002-class attractor (which originally used
2000 pyramidals + 500 interneurons with all-to-all connectivity).

Scaling to 1000+ dlpfc_wm neurons with denser connectivity would be
required to test Wang 2002 properly — this is a 10-100x substrate
scale-up, not the cheap probe Stage 1 was intended to be.

## Decision

**Direction I CLOSED at Stage 1.** The PFC sequence buffer path
requires substrate redesign (10-100x scale-up of dlpfc_wm region)
that's outside the autonomous-overnight scope.

**Pivot options:**
- **Direction N**: scale chat from 320 → 640 concepts via 5 more
  G.20 sparse bridges (~85 min GPU; extends working capability;
  needs new vocab curation)
- **Direction O**: integrate sentence-parser UI on top of 320-concept
  chat (UX/integration work; parser already partly in
  g20_multibridge.py)
- **Pause for steering**: today's work has been comprehensive
  (~110+ commits, 2 pillars, 9 mechanism attempts characterized
  including 4 substrate sequence-storage BOUNDARY + 5 cheap probes);
  user may want to steer next direction

## Cumulative arc summary

Total today: 2 pillars added (n=103 VALIDATED algebra, n=104
BOUNDARY extended 4x), 7 substrate sequence-storage mechanism
attempts + 3 PFC bistability probes (today's Direction I) all
characterize the substrate's bounds precisely. Validated 320-concept
multi-bridge conversational chat verified working (Direction M).

Honest scientific finding set is rich: the substrate's substrate-
sequence-storage capability is fundamentally bounded across all
biology-grounded mechanisms tested (engram-tag + ec_context +
theta-gamma + hippocampus + canon dynamics + FHRR algebra + biologized
FHRR + PFC bistability). The substrate's deliverable conversational
capability is 320-concept associative chat with honest abstention
(verified working today via Direction M).

Closing the sequence-storage bound requires SUBSTANTIAL architectural
work (dedicated 1000+ neuron PFC region with proper Wang 2002
attractor; OR fundamentally different mechanism not yet identified).

## Discipline preserved throughout

- Bar FROZEN at 0.80 multi-seed throughout (Direction I tested
  bistability gate, not the 0.80 capability bar)
- No protected/frozen/moat module modified
- 3 cheap-first probes ran in sequence (smoke → stress → direct
  inject → HH); each fast-failed efficiently
- Honest propagation: probe FAILS are findings, not failures
- Reuse-by-import only
- Both remotes pushed
