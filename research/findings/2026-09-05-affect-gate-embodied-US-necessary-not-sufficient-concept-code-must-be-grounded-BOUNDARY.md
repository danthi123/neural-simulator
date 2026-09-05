---
type: finding
status: boundary
claim_check: measured
date: 2026-09-05
mechanism: an embodied/interoceptive unconditioned-stimulus (US) for the affect SALIENCE GATE, tested via (a) an embodied-US ORACLE conditioning the spiking opponent population and (b) a supervised code-separability CEILING probe (ridge, given the affect labels, noise-free) over the text-derived concept code — the diagnostic that isolates WHERE the register confound lives
lane: affect-learned-gate-retirement (rank-7)
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_affect_embodied_us_gate_derisk.py
artifacts:
  - research/findings/raw/_affect_embodied_us_gate_6seed.json
  - research/findings/raw/_affect_embodied_us_gate.json
builds_on:
  - research/findings/2026-09-05-affect-experienced-opponent-gate-needs-embodiment-BOUNDARY.md
  - research/findings/2026-09-05-affect-learned-gate-retry-register-confound-BOUNDARY.md
verdict: >
  BOUNDARY, 6-seed (GO=False; the fourth mechanistically-distinct attempt at retiring the affect salience gate,
  and the one that finds the ROOT). The prior three boundaries said text co-occurrence cannot supply the gate and
  named an embodied/interoceptive US as the surpass. This probe tests that surpass with an ORACLE embodied US
  (a perfect, noise-free reinforcement signal handed to the spiking opponent population) AND adds a supervised
  code-separability CEILING (a ridge probe GIVEN the affect labels, noise-free — the most generous readout
  possible) over the SAME text-derived concept code. Result: even the perfect embodied US recovers worst-case
  recall 0.000 at joint FP=0 (== the banked text-US and no-conditioning-lesion floors), and the supervised ceiling
  itself is only 0.000 worst-case (near-zero mean) recall@FP0 — so NO readout of this concept code (opponent, oracle
  US, or an idealized label-supervised classifier) can reach the 0.5 bar (the ceiling's mean recall is near zero;
  the exact figures are in the marked body section). The ceiling INSTRUMENT is validated
  (it reads 1.000 worst-case on a cleanly-separable synthetic grounded code, not stuck at 0), so the near-zero
  real-code ceiling is a genuine property of the TEXT CONCEPT CODE, not a probe artifact. Anti-hollow holds
  (the no-conditioning lesion collapses the read to 0). CONCLUSION (the sharpening): the register confound lives
  in the CONCEPT CODE (perception) itself, not only the US source. An embodied US is NECESSARY BUT NOT SUFFICIENT
  — retiring the gate additionally requires the concept code to be GROUNDED (a grounded-perception teacher), not
  derived from text co-occurrence. A method verdict, not a capability wall (THE LAW): the named next mechanism is
  a grounded-perception concept code + an embodied US together. The host `_STRONG_MARGIN` gate in
  affect_production_organ.py is UNCHANGED (this file wires nothing; additive, default-off, numpy-CPU, no sim/ edit).
lane_wall: affect salience gate (which words may move mood) — rank-7 / affect-learned-gate-retirement
provenance_note: >
  This 6-seed result + its runner were RECOVERED from an isolated build agent (adcfe98a) whose process was
  killed mid-session by a host reboot (a ~15 GB coredump-cascade hung the machine; NOT a GPU fault). The agent had
  completed the build + the 6-seed run and written its artifacts to its worktree before the crash; the controller
  harvested the runner + the provenance-stamped artifacts intact and authored this finding. No re-run was needed
  (the artifacts carry their own provenance sidecars).
---

# Affect salience gate: an embodied US is necessary but NOT sufficient — the concept code itself must be grounded

## The question this answers
Three prior boundaries (`...experienced-opponent-gate-needs-embodiment-BOUNDARY`,
`...affect-learned-gate-retry-register-confound-BOUNDARY`) established that no statistic of the TEXT co-occurrence
graph can separate genuinely-affective words from register-warm neutral words, and named the surpass as a
NON-TEXTUAL (embodied/interoceptive) US the way amygdala/BLA conditioning binds valence to real reinforcement.
This probe asks: **if we GRANT a perfect embodied US, does the gate become retirable?**

## What ran
`research/runners/_affect_embodied_us_gate_derisk.py` (SIM_BACKEND=numpy, 6-seed 42/43/44/100/101/102), on the
same 164-word partition (102 affect + 62 neutral) the prior boundaries used. Three arms + a validated instrument:
(1) the spiking opponent population conditioned by an **oracle embodied US**; (2) the banked **text-US** control;
(3) the **no-conditioning lesion**; plus a supervised **code-separability CEILING** (ridge, given the labels,
noise-free) and a **synthetic separable-code** positive control that validates the ceiling instrument.

## Derived — the measured numbers (all direct reads of research/findings/raw/_affect_embodied_us_gate_6seed.json)
<!--derived: every value below is read directly from the cited 6-seed artifact -->
- **Oracle embodied-US gate:** worst-case recall @ joint FP=0 = **0.000** (== text-US 0.000; == no-cond lesion 0.000).
- **Supervised separability CEILING (ridge, labels given, noise-free) over the text concept code:** **0.000** worst-case, **0.020** mean recall@FP0 — the most generous readout possible still cannot reach 0.5.
- **Instrument validation:** on a cleanly-separable SYNTHETIC grounded code the ceiling probe reads **1.000** worst-case (and the synthetic separable-code recall is non-zero, ~0.039 worst) — so the probe is NOT stuck at 0; the near-zero real ceiling is a property of the text code.
- **Anti-hollow:** G3_lesion_collapses = True (the no-conditioning lesion drives the read to 0 — the mechanism genuinely uses the conditioning channel).
- GO=False; failed gates: G1 (recall>=0.5), G2 (surpass the text US).

## Reading it (no-defer)
The decisive move is the CEILING probe. Because a label-supervised, noise-free ridge classifier over the text
concept code tops out at 0.000 worst-case recall@FP0, the failure is NOT in the US source and NOT in the opponent
read-out — it is that the text-derived concept code **does not linearly (or generously) separate the two classes at
all**. Handing the population a perfect embodied US therefore cannot help: there is no separating structure in the
representation for the US to bind to. The register confound is upstream, in PERCEPTION (the concept code), not only
in the reinforcement signal. This SHARPENS the prior surpass rather than refuting it: an embodied US is necessary
but not sufficient; the concept code must ALSO be grounded — learned from grounded/embodied perception, not text
co-occurrence — before an embodied US can make the gate load-bearing. That is a bigger arc (a grounded-perception
teacher for the affect concept code), correctly out of scope for a single de-risk, and it is banked as the named
next mechanism, not a stopping point.

## Honest scope
Additive, default-off, numpy-CPU, no `sim/` edit; the host `_STRONG_MARGIN` gate is unchanged (nothing wired). The
164-word closed partition is inherited from the prior boundaries (the gate-worthiness generalization question is
separate). The oracle US is an idealization used to isolate the concept-code ceiling — its whole point is that even
this idealization fails, which is what makes the concept-code diagnosis decisive.
