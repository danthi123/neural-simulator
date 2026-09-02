---
status: live
type: finding
lane: onebrain-integration-design
date: 2026-09-02
mechanism: onebrain-crossregion-integration-scope
---

# One-brain cross-region INTEGRATION — ranked design of the plastic cross-synapses to build next (scoping, no build)

**This is a DESIGN/SCOPING finding, not a build or a GO.** It ranks the candidate learned spiking cross-synapses
between the co-resident cortical organs — the genuine one-brain INTEGRATION rung that the two-pool MIGRATION gate
(`2026-09-02-onebrain-twopool-merge-organ-read-GO.md`; 6/6 seeds, organ-read byte-identity max delta 0.00e+00,
artifact `research/findings/raw/_onebrain_twopool_merge_organread_6seed.json`) deliberately excluded (a pool with
zero cross-edges is migrated, not integrated). Produced by a 4-agent research fan-out, each held to four bars: BRAIN-BASED (the
cross-influence is carried by a spiking synapse, not a host formula), EMERGENT (the weight grows from experience via
a local rule, not a host schedule), BIOLOGICALLY GROUNDED (a real projection, cited source), and LOAD-BEARING +
LESION-VANISHING. The template is the already-shipped D6-WM→D4-comprehension edge (`onebrain_xedge_production.py`).

## What already exists (verify-first — do not re-derive)

GENUINE learned spiking cross-synapses already built + default-ON: **D6-WM → D4-comprehension** (the template),
**surprise → source_provenance**, **surprise → episodic-ENCODE-decision**, **curiosity → D6-WM**,
**self_schema → metacog**. Host-computed couplings (candidates to convert): the #94 confidence→forthcomingness,
#84 affect→tone, #85 swap→focus. The 4 core organs (D2 surprise, E2 world-model, E1 metacog, D pragmatic) now
co-reside byte-identically (migration GO), and the single-pool flip (`BRAIN_ONEBRAIN_SINGLE_POOL`, just wired)
puts all 4 on ONE substrate so any pair can be spanned. **The surprise↔world-model and surprise↔metacog couplings
are confirmed UN-BUILT.**

## The convergent insight: D2 surprise is the integration HUB

Three of the four independent clusters put **D2 surprise as the SOURCE** of their top edge. The surprise pool's
expectation-violation firing is exactly the third factor a Hebbian window needs — and the project ALREADY ships that
backbone (`onebrain_crossedge_gate.run_gate` + the surprise→episodic-encode edge, GO 6/6, Lisman & Grace 2005). So
one reusable read (surprise's windowed firing gating plasticity) serves multiple high-value edges.

## Ranked cross-edges to build

| rank | edge | function (the turn it changes) | grounding | readiness | de-risk |
|---|---|---|---|---|---|
| **1** | **D2 surprise → E2 world-model** (error-gated forward-model update) | a semantically+affectively surprising turn UPDATES the world-model, so the next "how's this going?" answer shifts — learn-through-use | Rao & Ballard 1999 (error drives generative-model learning); Lisman & Grace 2005 (novelty→DA→plasticity gate, PMID 15924857); Kafkas & Montaldi 2018 (PMID 30053569) | **HIGHEST** — reuses the SHIPPED surprise-gates-plasticity third factor; both organs already co-resident (pool #1); closes E2's own declared "teacher-driven, not self-organized" residual | 6-seed run_gate: emergence + vary/lesion (freeze surprise→gate → the world-model no longer re-learns) + attributable + byte-off |
| **2** | **D2 surprise → E1 metacog** (error → confidence) | a violated prediction LOWERS the confidence read off the substrate (not a host if-statement) → the reply hedges more | same surprise→neuromod backbone; Yu & Dayan 2005 (unexpected uncertainty, NE/ACh reset) | **HIGH** — same surprise source as #1; UNBLOCKED by the single-pool flip (surprise + metacog now co-reside) | 6-seed run_gate: surprising turn drops metacog margin; lesion restores it |
| **3** | **Arousal → D2 surprise** (prediction-gain) | felt-arousal sharpens/broadens the surprise competition → the same input is more/less surprising by affective state | Aston-Jones & Cohen 2005 (LC-NE adaptive gain — Kandel's canonical LC figure IS this circuit) | MEDIUM — different mechanism (neuromodulatory gain, not a plasticity gate); reads the #81/#84 ladder | 6-seed: arousal shifts the surprise threshold; lesion flattens it |
| 4 | **D pragmatic ↔ E1 metacog** | pragmatic context informs confidence / confidence gates hedging | (metacog→pragmatic is the canonical direction; verify the functional story) | CHEAPEST — both already co-resident on the default-ON pool #2 that calls merge_organs; no new seam | 6-seed once the direction is pinned |

## Recommendation

**Build #1 (surprise → world-model) first.** It is the highest-value AND lowest-risk edge: it reuses a shipped
third-factor backbone, both organs are already co-resident, it is the canonical predictive-coding learning arm, and
it closes a residual the world-model organ *itself* declares (its transition is teacher-driven, not self-organized
from conversation). Because #1 and #2 share the surprise SOURCE, building #1's surprise-third-factor read once feeds
#2 (surprise → metacog) next — an efficient sequence. #3 (arousal gain) is a distinct, also-canonical mechanism to
follow. Each is a narrowly-scoped `run_gate` de-risk (emergence + lesion-vanish + byte-off), not a production flip.

## Honest scope

This is the DESIGN; nothing is built. Each edge is a hypothesis until its 6-seed run_gate GO. "Integration" here
means genuine spiking cross-talk that LEARNS — not a host coupling relabeled. Functional read-outs only; no
phenomenal claim. Full per-cluster proposals (all four, with substrate-level synapse computations + full citations)
are in the workflow transcript.

## Files

- Scoping only — no runner. The build target for #1 reuses `research/runners/onebrain_crossedge_gate.py::run_gate`
  + the surprise + world-model production organs. Template: `research/runners/onebrain_xedge_production.py`.
