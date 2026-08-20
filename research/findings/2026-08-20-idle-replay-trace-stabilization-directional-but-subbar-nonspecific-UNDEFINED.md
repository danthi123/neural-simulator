---
type: finding
status: live
date: 2026-08-20
mechanism: idle-tick-replay-stabilization
lane: memory
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_idle_consolidation_stabilization_derisk.py — encode a weak sub-barrier BTSP trace, idle-tick replay (real fused_btsp_update reactivation) vs matched no-replay / no-presynaptic-reactivation / never-encoded controls, recall-after-delay
runner: research/runners/_idle_consolidation_stabilization_derisk.py
external: NO-EXTERNAL-NEEDED — the residual (a replay DOSE strong enough to rescue a real trace also partially writes an unencoded one) is the SAME dosage/specificity wall the local corpus already characterised (2026-05-20 over-consolidation regression; 2026-08-03 replay-cortical-consolidation v1–v6 NO-GO), rooted in BTSP's saturating (w_max−w) update being weakly sensitive to starting weight. A known in-corpus mechanism, not a novel/paradigm claim.
---
# BOUNDARY: idle-tick replay gives a DIRECTIONAL but sub-bar, NON-SPECIFIC recall boost — the learn-through-use rung is not yet met

Artifact: research/findings/raw/_idle_consolidation_stabilization.json

**One line.** The continuous engine's 3rd property (LEARN-THROUGH-USE) wants the brain to LEARN between turns: an
idle-tick replay pass should STRENGTHEN a recent trace so a later recall is better, and skipping the replay should
remove the gain. This 6-seed de-risk (researched the record first; scoped to STABILIZATION, not "consolidation" per
`docs/TERMS.md` — no source-structure lesion tested) shows the CORE effect is real and lesion-attributable BUT does
not clear the rung's bar: the boost is directionally consistent 6/6 yet SMALL (below the pre-registered separation
bar) and NON-SPECIFIC (the same replay dose partially writes a never-encoded pathway). Overall verdict: **UNDEFINED /
BOUNDARY** (the runner's own Verdict machinery refuses to report a GO). Directionally encouraging, not yet met.

## What holds vs what fails (6 seeds, numpy; runner's own verdict, independently reproduced)
- **G1 core (directional, 6/6): PASS.** Idle-tick replay's after-delay recall beats the matched non-replayed control
  on every seed (e.g. seed 42: 0.016 <!--derived--> vs 0.0075 <!--derived-->; ratios 1.5×–2.2×). So replaying a recent trace during idle DOES leave
  it better recalled.
- **G2 lesion (6/6): PASS.** Replaying WITHOUT presynaptic reactivation (apical pulses alone, pre-side BTSP
  eligibility zeroed) does NOT rescue recall — the gain rides the reactivation, not the apical drive alone. So the
  effect is load-bearing on the replay content, not an artifact of extra pulses.
- **Separation magnitude: FAIL.** The replay-vs-noreplay separation (|sep|≈0.006 <!--derived-->) is BELOW the pre-registered bar
  (0.008) — the effect is directional but small at this 72-neuron scale.
- **G3 specificity: FAIL.** The identical replay dose delivered to a NEVER-encoded pathway (`moat_replay`) also gained
  recall (46–67% of the real trace's level, vs the ≤40% bar). Root cause (diagnosed empirically): BTSP's saturating
  `(w_max − w)` update is only weakly sensitive to starting weight when `w_max` is large vs the tag-and-capture
  barrier, so a dose that robustly rescues a real trace also partially writes an unencoded one. Lowering `w_max`
  toward the barrier weakened BOTH signals together rather than separating them.

## The residual + next lever (matches the corpus's own dosage wall)
The two failures are one mechanism: replay AMPLIFIES and FABRICATES with the same knob, because the write rule is
insufficiently starting-weight-gated. This is exactly the 2026-05-20 over-consolidation / 2026-08-03 replay-gate wall
(more replay dose harms retrieval / fabricates). Next lever (in the artifact's `NEXT_RUNG`): make replay CONTENT
emergent — plastic recurrent pre→pre connectivity + untargeted noise so pattern-completion reactivates only the
ENCODED assembly (a real trace re-ignites, an unencoded one does not), instead of host-directed reactivation of the
same cells; and a starting-weight-gated write (a steeper barrier or a metaplastic threshold) so the dose amplifies
without fabricating. Then port tag-and-capture to a `sim/` kernel and wire under the `continuous_engine` idle tick.
Until specificity clears, idle-replay is NOT safe to wire live (it would strengthen phantom associations).

## Scope
This is the FIRST test of idle-tick replay's effect on recall (the 2026-08-12 BTSP-lasting-trace GO explicitly ran
no reactivation path). The core (replay reactivation drives a recall gain, lesion-clean) is established directionally;
the rung is met only once the effect is both above-bar and specific. Honest BOUNDARY, not a stop — the next lever is
named. (Agent-built, research-record-first, independently re-run + reproduced; TERMS.md-checked: "stabilization", not
"consolidation".)
