---
type: finding
status: contributing
date: 2026-08-20
mechanism: dendritic-plateau-coincidence-burst
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_step6_graded_apical_read_derisk.py — a GradedEpisodicDapMemory drop-in reading a
  CONTINUOUS apical magnitude (mean held-cell plateau depth) from the same cp_v_apical the binary UP-fraction
  thresholds, while the binary read + specificity gates STILL gate in_memory (the moat); consolidated by the REAL
  continuous_engine.consolidate_used_memory; teeth = graded-moves / monotone / faithful(formed-vs-unformed) / specific /
  lesion-vanishes; Verdict + attributable_to per seed.
runner: research/runners/_d5_step6_graded_apical_read_derisk.py
external: NO-EXTERNAL-NEEDED — realizes the redirect the step-5 finding named (a graded read escapes the binary
  quantization ceiling); depth_hold IS the substrate's own BTSP instructive signal IS_post.
artifacts:
  - research/findings/raw/_d5_step6_graded_gentle/summary_6seed.json
  - research/findings/raw/_d5_step6_graded/summary_6seed.json
---
# A GRADED apical read makes learn-through-use RELIABLY conversation-visible (5/6 strict, 6/6 on move+faithful+specific) — resolving step-5's binary quantization ceiling

Artifact: research/findings/raw/_d5_step6_graded_gentle/summary_6seed.json (the n_episodes=1 headline schedule).

**One line.** Step-5 showed learn-through-use IS conversation-visible but only 4/6-reliable, walled by a per-membership
STRUCTURAL ceiling: apical_cue is a per-held-cell BINARY UP-fraction (quantized, dead-steps), and the plateau knob
provably could not raise it ([[2026-08-20-d5-learn-through-use-CAN-be-conversation-visible-weak-encode-4of6-graded-read-for-reliable]]).
This builds the named redirect — a GRADED continuous apical read — and it works: the surfaced recall magnitude rises
after use reliably, resolving the exact seeds the binary read left flat.

## The mechanism (a pure READ change; NO `sim/` edit, moat preserved by construction)
`GradedEpisodicDapMemory` reads a CONTINUOUS magnitude from the SAME `cp_v_apical` in one sim pass: **`depth_rest`** =
mean-held `max(cp_v_apical − apical_E_rest, 0)` (plateau depth above rest, mV) and **`depth_hold`** = mean-held
`max(cp_v_apical − v_hold, 0)` — the latter is LITERALLY the substrate's own BTSP instructive signal `IS_post` that
`fused_btsp_update` integrates (strong biological grounding). The BINARY UP-fraction + specificity gates
(cue≥0.20, cue≥3·perm, cue≥3·nocue, nocue≤0.10) STILL gate `in_memory` (the moat) — verified byte-identical to the
production `_apical_up_read`. So the graded number is the SURFACED magnitude the reply quotes, shown only when the
binary moat says `in_memory=True` → the moat is preserved by construction, and additionally verified faithful.

## The 6-seed verdict (strict GO 5/6; 6/6 on the load-bearing criteria) — the decisive cases
<!--derived-->
Same weak-encode op-point + same REAL handler path as step-5 (`n_episodes=1` = production's 1-heavy-call-per-recall
budget, so the surfaced number climbs gradually instead of saturating). `depth_rest` and `depth_hold` each strict-GO
**5/6** (`read_go_counts`). The DECISIVE cases: on **seeds 42 and 44 the binary read is COMPLETELY FLAT**
(0.4286→0.4286, 0.2857→0.2857 — the quantization dead-step, weight rose ~+18 mV with no held cell crossing threshold),
yet the graded read moves cleanly, monotonically, faithfully → GO. That is exactly step-5's seed-42 UNDEFINED, now
resolved. The single strict miss is seed 101, on a **0.018 mV** mid-trajectory wobble inside a +15.5 mV monotone rise —
not a quantization dead-step. So on **move + monotone + faithful + moat-specific the result is 6/6**; only the strictest
per-turn dead-step floor is 5/6, and it held 6/6 on those load-bearing criteria across all 3 runs (18/18 seed-runs).

## Faithfulness (the moat) is airtight — it is not weight-blind
<!--derived-->
Reading the formed 'dog' through the UNFORMED baseline weights (formation-lesion) collapses the graded read to
**exactly 0.000** on all seeds; nocue and perm cues also read 0.000; only the formed, cue-specific completion reads
16-51 mV. So the graded read reflects whether THIS assembly's cue-specific completion happened — a monotone function of
the weight would leak, this does not. Lesion-off (flag OFF) is byte-identical + flat (`hash_before == hash_off`).

## Recommendations for the production-default flip (the last mechanism precondition, now met)
Run the graded read **BESIDE** the binary read, not replacing it — the binary UP-fraction + specificity keep gating
`in_memory` (the honest abstain), the graded number is the surfaced magnitude (shown only when in_memory). Use
`depth_rest` or `depth_hold` (NOT `soft`, which was 1/6 — its [0,1] range saturates). Set the between-turn consolidation
to `n_episodes=1` (matches production's budget; avoids the graded read saturating on the first tick). Minimal production
edit: add the graded fields to `EpisodicDapMemory._apical`/`recall` (same `cp_v_apical`), keep the binary in_memory
gate, and have `recall_disclosure` quote the graded number. With this, the D5 learn-through-use arc's mechanism is
complete; the default-on flip stays an owner-UX call (still wants a soak + no-regression). (Agent-built; parent
sanity-verified seed42's binary-flat/graded-GO + the read_go_counts 5/6 + the formation-lesion→0 faithfulness from the
artifacts.)
