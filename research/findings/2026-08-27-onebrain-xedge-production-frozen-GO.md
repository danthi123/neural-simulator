---
type: finding
status: live
date: 2026-08-27
mechanism: onebrain-xedge-production-frozen
lane: one-brain/integration/production
artifacts:
  - research/findings/raw/_onebrain_xedge_production_frozen_seed42.json
  - research/findings/raw/_onebrain_xedge_production_frozen_s43s44.json
runner: research/runners/onebrain_xedge_production.py
supersedes_diagnosis_of: research/findings/2026-08-27-onebrain-production-integration-SCOPING.md
---

# One-brain PRODUCTION integration PART 1 — the FROZEN R3-v3 cross-edge is now WIRED INTO THE LIVE chat brain and
DRIVES the real comprehension judge lesion-attributably (3-seed) — but the drive is SUB-DECISION (does not yet flip
conversational content); default-OFF, byte-identical-off — GO on the frozen wire-in de-risk

**One-line:** The first learned cross-region synapse is now wired into the LIVE conversational pipeline. The
de-risked R3-v3 cross-edge (d6 multi-referent WM candidate pool -> D4 comprehension `sel_agent`/`sel_patient` role
competition) is grown-once + FROZEN and made load-bearing on the REAL production `judge()`/`repair_target()` reads:
holding the p_agent candidate pool vs the p_patient candidate pool shifts the comprehension read
(`max|dMargin|` up to 0.048 <!--derived-->; the signed net-lean `repair_target` consumes shifts 0.009-0.044
intact across seeds 42/43/44 <!--derived-->), and the shift is EXACTLY 0 when the cross-edge is lesioned (the clean
net-lean instrument) / ~1.5% of intact on the guarded judge path — a genuine, lesion-attributable WM->comprehension
DRIVE, not a co-resident observer. It is DEFAULT-OFF (`BRAIN_ONEBRAIN_XEDGE`), byte-identical when unset, no-regression at the decision
level, and moat-safe. **Declared, not hidden: the drive is SUB-DECISION on the production instrument — it does NOT
flip `comprehended` or the repair role on the in-scope battery (the ~0.05 shift is small vs the ~0.15-0.33 decision
boundaries), so it does not yet change the turn's CONTENT.** Strengthening the drive to decision-flipping, and
closing the semantic referent->pool binding, are the two named next rungs.

Artifacts: `research/findings/raw/_onebrain_xedge_production_frozen_seed42.json`,
`research/findings/raw/_onebrain_xedge_production_frozen_s43s44.json`.

## What was built (the smallest safe live wire-in the scoping named, plus the coupling it under-scoped)

`research/runners/onebrain_xedge_production.py` — a process-shared `XedgeProductionPool` that co-locates the R3-v3
pair on ONE `MergedPool` and loads the FROZEN pre-grown `w{k}->sel_agent/sel_patient` cross-edge:

- **Grow-once + freeze.** The edge is grown by R3-v3's OWN credit-gated training (`R3v3Pool.train()`, which freezes
  the candidate gate the instant training returns) and then never moves — plasticity is OFF for every live turn.
  Growth is IN-PROCESS (not a saved weight file) on purpose: the CROSS-BACKEND SEED TRAP (a numpy-grown weight file
  is not valid for a cupy build — different RNG, different substrate) means growing in whatever backend the process
  runs guarantees the frozen edge matches the substrate it rides. The converged block-mean weights ARE written to a
  sidecar for the record; correctness never depends on loading it. Verified: the edge grows following the per-seed
  RANDOM role assignment (anti-cheat), not a hardcoded pair — seed 42 `w0->A=13.6`/`w2->P=13.2` (p_agent=w0), seed 43
  `w1->A=13.3` (p_agent=w1), seed 44 `w2->A=13.2` (p_agent=w2).
- **Default-OFF, byte-identical-off, degrade-safe.** `BRAIN_ONEBRAIN_XEDGE` gates everything (default OFF; the flip
  to default-ON is a SEPARATE owner-gated step, not taken here). Unset -> every organ builds standalone exactly as
  today. A build failure DEGRADES to standalone (never crashes brain load). `BRAIN_ONEBRAIN_XEDGE_LESION=1` zeroes
  the cross-edge (the load-bearing lesion control).

### The lifecycle decision (the scoping's main open design question)

Comprehension is a PROCESS singleton; d6 is PER-SESSION (its `_SESSION_MULTIREF` codebook must not leak referents
across conversations). A shared pool cannot be both. **Resolution: the pool is PROCESS-shared; comprehension binds
to it as the singleton; d6 keeps its PER-SESSION wrapper (own codebook) but takes the process pool as `shared=`.**
Why this preserves isolation: the shared spiking d6 SLICE carries only transient, per-turn bumps (every `load()`
resets it) — it holds no cross-session MEANING; only the per-session host codebook (referent-string -> register)
carries meaning, and that stays per-session. So referent isolation is preserved AND comprehension keeps its
singleton. (The alternative — a per-session pool — would rebuild + recalibrate comprehension every conversation.)

### The coupling the scoping under-scoped (why "attach `shared=` and done" would have been HOLLOW)

Comprehension's read `_hard_reset`s the WHOLE shared bridge to `pool.snap` before every sel-settle, and d6's
`load()` runs in `read_isolation` (restores every OTHER slice) — so a held d6 bump does NOT survive into the
comprehension read on its own; a naive attach leaves the cross-edge with no presynaptic activity to transmit (a
co-resident-but-not-interacting observer — the exact hollow-integration drift memory #84/#85 gates). The cross-edge
only transmits when the held d6 pool is FIRING during the sel-settle (R3-v3's F2 `amb_read` protocol). So the
wire-in adds a guarded, byte-identical-off co-drive coupling: comprehension's `_read`/`_read_per_noun` (shared+xedge
path, a focus register set) re-establish the held pool's self-sustaining slow-NMDA bump before the cue window; d6's
`load()` publishes the held referent's candidate pool as that focus. Four additive guarded edits total (the two
`get_organ`/attach points the scoping named + the two coupling edits it missed), all byte-identical-off.

## Verification (the drive-couplings discipline; through the REAL production judge/repair)

| criterion | result |
|---|---|
| (a) LOAD-BEARING, lesion-attributable, through the real judge/repair | **PASS** 3/3 seeds |
| (b) NO-REGRESSION (decision level) | **PASS** — comprehended decisions preserved flag-off vs flag-on |
| (c) MOAT (F4) | **PASS** — a well/clear item's `comprehended` is not flipped by WM focus; out-of-scope stays unchanged |
| (d) BYTE-IDENTICAL-OFF | **PASS** — standalone organs untouched; server diff = only the guarded attach |
| (e) decision-FLIP (a biased margin flips comprehended/repair) | **NO** — sub-decision on the production instrument (declared residual) |

- **(a) Load-bearing.** Hold the p_agent candidate pool vs the p_patient candidate pool during the REAL production
  comprehension read (the differential CANCELS the generic "any WM activity perturbs the shared inhibition"
  confound — hold-vs-no-hold is NOT lesion-attributable, but hold-Ag-vs-hold-Pa is). The signed net-lean
  `repair_target` consumes shifts `max|dNet|` = 0.009 / 0.044 / 0.042 <!--derived--> (seeds 42/43/44) INTACT and is
  EXACTLY 0.0000 when the cross-edge is lesioned on every ambiguous item — a clean, fully lesion-attributable drive
  (rounded from the cited `_onebrain_xedge_production_frozen_*` artifacts' `max_abs_dNet_intact`/
  `max_abs_dNet_lesioned`). <!--derived-->
  On the guarded `judge()` `|margin|` path the shift reaches 0.048 intact vs ~0.0007 lesioned <!--derived--> (~98.5%
  attributable; the small residual is the w0-vs-w2 physical-region asymmetry, not the cross-edge — from the verify
  harness, not a committed artifact). <!--derived-->
  This is the same F2 instrument R3-v3 GO'd (`F2_INTACT_FLOOR=0.008` <!--derived-->, restated from R3-v3) — cleared
  on all 3 seeds, with 5x headroom on seeds 43/44.
- **(b) No-regression.** The no-WM-held (focus unset) `comprehended` decisions are IDENTICAL flag-off vs flag-on
  (well items True, ambiguous items False). Flag-on moves the comprehension SUBSTRATE onto the shared pool, so the
  raw margins + self-calibrated threshold recalibrate to the shared operating point (0.34/0.33 vs 0.34/0.25) — this
  is the R3-v3 6-seed migration property (decisions preserved, not raw-margin identity), not a regression.
- **(c) Moat.** With the WM focus set to p_agent vs p_patient, a well/clear transitive's `comprehended` does not
  flip (both stay True); a question returns `None`; an OOV triple stays the existing honest abstain — all invariant
  to the WM focus.
- **(d) Byte-identical-off.** Every edit is inside a `BRAIN_ONEBRAIN_XEDGE` / xedge-pool-marker guard; unset -> the
  standalone path runs untouched. `webapp/server.py` changed by exactly the one guarded `_get_multiref_organ` attach
  (14 insertions, 1 deletion).

## What is DECLARED RESIDUAL (honest — the drive is real but not yet content-changing)

1. **SUB-DECISION magnitude.** The cross-edge drives the read lesion-attributably but the ~0.05 shift never crosses
   the production decision boundaries (comprehended threshold ~0.33; the repair `lean_margin` ~0.15-0.3), so it does
   NOT flip `comprehended` or the repair role on the in-scope battery — the turn's CONTENT is unchanged. The drive
   is genuine and reversible, but content-neutral at the decision level. **Why:** the production judge consumes
   `|a0-a1|` (a per-noun difference that partly cancels a symmetric sel bias) and a net-lean far from its boundary
   on the toy battery; R3-v3's stronger `amb_read` instrument (a single balanced-cue SIGNED read where the WM is the
   ONLY tiebreaker) is not what the live judge reads. **Next rung (not deferred — the method to surpass this):**
   raise the converged cross-edge weight (a DA_SENSITIVITY / episode-count calibration) so the bias crosses the
   boundary, OR route ambiguous in-scope items through the balanced-cue signed read where the WM bias is decisive.
2. **Semantic referent->pool binding is host-directed** (carried UNCHANGED from R3-v3's declared scaffold): the
   candidate topology (w0/w1/w2) is a host-chosen abstract "3 structurally-identical d6 slot pools", so which real
   discourse referent maps to the agent- vs patient-candidate pool is not learned — the live focus is a POSITIONAL
   proxy. Closing that semantic binding is a later rung.

## Verdict

**GO on the frozen wire-in de-risk:** the first learned cross-region synapse is wired into the LIVE chat brain,
frozen, default-OFF, byte-identical-off, no-regression, moat-safe, and it DRIVES the real production comprehension
judge/repair read lesion-attributably on 3 seeds (net-lean shift exactly 0 under lesion). This closes the scoping's
open questions (lifecycle; the co-drive coupling the naive attach missed) and de-risks the live path. It is NOT a
default-ON flip and it does NOT yet change conversational content (sub-decision) — those are separate, named rungs.
Functional read-outs only; no phenomenal-experience claim.

## PART 2 (live-learning) — concrete design + stopping point (NOT built this session)

PART 2 (make the cross-edge GROW from the live substrate's own activity, no frozen pre-grown weight) is a strictly
larger rung and is NOT built here. The blocker the scoping named is real and unmoved: **production has no live
credit signal.** R3-v3 grows the edge from a HOST teacher schedule (`teach_agent`/`teach_patient` drives on
ground-truth-correct episodes) that a live chat turn does not supply. Concrete design for the follow-on:

- **Credit source.** Reuse R3-v3's spiking DA-coincidence detector by IMPORT, but drive `teach_*` from an
  IN-BRAIN signal instead of a host schedule: the most tractable is the D2 SURPRISE organ's expectation-match (a
  confirmed anaphoric resolution = a low-surprise "the discourse held together" pulse) OR the comprehension
  organ's OWN `comprehended` verdict as the self-supervised teacher (a cleanly-comprehended turn credits the WM
  register that was held). Both are already-spiking production organs on adjacent slices.
- **Bounded, gated growth (F3).** Keep `stdp_w_max`/`hebbian_max_weight` as the runaway bound; open the candidate
  gate ONLY during a credited window (three-factor), freeze between turns. Verify the edge rises from ~0.05 over
  turns and never runs away (F3), stays load-bearing (drives the judge), moat-safe, byte-identical-off.
- **Same default-OFF flag** (or a sibling `BRAIN_ONEBRAIN_XEDGE_LEARN`).
- **Stopping point (here):** PART 1 (frozen) is landed + verified. PART 2 needs (i) a chosen in-brain credit
  signal wired to `teach_*`, (ii) a live-turn credited-window schedule, (iii) the F3 runaway + drift re-verify over
  a multi-turn live session. That is a full arc, correctly a separate finding
  (`...live-learning-<GO|NOGO>.md`), not a same-session add-on.
