---
type: finding
status: contributing
date: 2026-08-20
mechanism: swr-sequence-replay
lane: EPISODIC
seeds: [42]
seed-waiver: A single-seed FEASIBILITY/SCOPING probe of an integration path — the evidence is a within-run presence/absence of four reactivation properties (discrete / specific / recurrence-riding / completes) across three geometries, plus a code-read of the two organs; a seed population measures nothing about a design's feasibility.
instrument: research/runners/_gap5_ecker_reactivates_d5_stored_assembly_derisk.py — an Ecker AdEx CA3 partial-cued under the SWR envelope on a stored vs never-formed assembly subset, measuring held-out completion / discreteness / specificity / recurrence-lesion teeth
runner: research/runners/_gap5_ecker_reactivates_d5_stored_assembly_derisk.py
external: NO-EXTERNAL-NEEDED — an in-repo integration-feasibility read between two in-repo GO mechanisms; the next step is a build, not a literature question.
artifacts:
  - research/findings/raw/_ecker_reactivates_d5/s42_a80_w500_fixed.json
---
# Wiring the emergent SWR replay into the D5 episodic store is FEASIBLE — by COMPOSING D5's latch with Ecker's self-termination, not by replacing either

Artifact: research/findings/raw/_ecker_reactivates_d5/s42_a80_w500_fixed.json

**One line.** The SWR forward-replay is now a working, brain-pure mechanism (Ecker AdEx CA3 + STDP-learned band). Can it
REACTIVATE a memory stored in the production D5 `EpisodicDapMemory` — the original learn-through-use crux? This scopes
the integration and de-risks the load-bearing connection: **YES, feasible — the Ecker AdEx substrate supplies exactly
the discrete, self-terminating, assembly-specific reactivation D5's persistent dendritic-dAP latch could not, which is
what the prior afferent NO-GO's root cause needed; the buildable path COMPOSES the two organs rather than replacing one.**

## The concrete gap (code-read)
D5: Izhikevich TWO-COMPARTMENT dendritic-dAP neurons, emergent DG-selected ~20-cell ~1%-sparse INDEPENDENT attractors,
static within-assembly pattern completion via a PERSISTENT KIR-latched apical UP. Ecker: single-compartment AdEx
(`ADEX_ECKER_CA3_PC`, spike-triggered adaptation), 80-cell dense blocks, DISCRETE self-terminating SEQUENTIAL replay.
The store's ONLY structural signature on the readout bridge is its potentiated within-assembly recurrence — any replay
engine must ride THAT. D5's latch is PERSISTENT, so it physically cannot make the discrete self-terminating SWR
transient — which is precisely the property the Ecker AdEx adaptation supplies (why Ecker is the right unblock).

## The feasibility probe — PARTIAL, residual precisely localized (and NON-blocking for learn-through-use)
<!--derived-->
A new probe (`_gap5_ecker_reactivates_d5_stored_assembly_derisk.py`, reuse-by-import, NO `sim/` edit) sets a STORED
assembly subset to strong within-recurrence + an UNSTORED subset weak (D5's never-formed "cat" control), partial-cues
each under the SWR envelope, and reads held-out completion / discreteness / specificity / recurrence-lesion teeth.
Across three geometries (D5-realistic asm=20 w=100; asm=20 w=1500; Ecker-scale asm=80 w=500):
- **PASSES (the real positive):** reactivation is **DISCRETE + self-terminating** (adaptation), **SPECIFIC** to the
  stored assembly (unstored held-out stays 0.000 at every scale), and **rides the stored recurrence** (weaken-to-unstored
  lesion collapses it). This is exactly what the prior DG/EC-afferent NO-GO's root-cause-1 said the afferent locus could
  NOT achieve. The stored assembly's members co-fire discretely and specifically.
- **FAILS (the localized residual):** full within-assembly HELD-OUT completion on the AdEx soma recurrence is weak
  everywhere (~5.5% peak at asm=80/w=500; ~0 at D5-realistic asm=20/w=100). Cross-checked on the real Ecker sequence
  runner at D5-scale: the STDP band still forms (fwd 15→325, ~29×) and assemblies activate, but multi-assembly chaining
  collapses (n_multi=1 vs ≥6). This is the SAME small-assembly seam D5 itself documented (a ~23-cell set is too small
  for a recurrent bistable attractor) — the reason D5 adopted the per-cell dendritic-dAP latch for its READ.
- **Why it is non-blocking:** learn-through-use does NOT require full held-out completion — it needs the assembly's
  members to CO-FIRE so BTSP can potentiate, and the probe shows co-firing IS discrete, specific, recurrence-carried. So
  the completion weakness is an op-point/geometry issue, not a wall.

## The buildable path (composition) + the ordered plan
Build a SEPARATE additive AdEx-CA3 idle-replay engine that SHARES D5's membership + potentiated within-recurrence (an
exact weight+index map, not new learned structure), chained by an STDP-learned temporal-context band; D5's read path
stays byte-identical. Do NOT give D5's CA3 the AdEx model (it would break the dendritic-dAP READ the recall gate
depends on). Ordered plan: (1) REAL-STORE transfer — build one genuine D5 store, extract its recurrence+membership, map
onto an AdEx bridge, re-run the probe on the real store (closes the proxy→real gap); (2) fix the completion residual by
the RIGHT biology (compose D5's dendritic-dAP latch — which works at 20 cells — GATED by an AdEx/SWR-envelope
self-termination so it is a TRANSIENT not a persistent latch; or the documented `_gap5_swr_envelope_replay_derisk`
tuning band, a 0-token pool sweep); (3) chain topics into a temporal-context band (STDP forward edges between
consecutively-stored topics) so idle SWR has a spontaneous-ignition path; (4) WRITE-BACK = learn-through-use: BTSP during
the reactivation window, copy strengthened weights back to D5's recurrence, verify with D5's own lesion teeth, wire under
`continuous_engine.py` idle tick default-off. **Arc-2 (brain-pure sleep-replay) is a STRICTLY LARGER integration** — the
Ecker store replaces the host `np.mean` hippocampus STORE + supplies reactivation timing, but NOT the cortical teaching
signal (the replay yields a spike pattern, not the e-prop `(X,y)` percept vector); it needs a content-carrying
CA3→CA1→cortex spiking projection. Do arc-1 first. (Agent-scoped + probe-built; parent verified the probe artifacts.)
