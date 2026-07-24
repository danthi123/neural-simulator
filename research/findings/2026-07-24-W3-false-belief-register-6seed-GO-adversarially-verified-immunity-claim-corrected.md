# W3 false-belief register (Theory-of-Mind) — 6/6-seed GO, ADVERSARIALLY VERIFIED (a subagent "structural immunity" claim was caught + corrected) (2026-07-24)

## Result
The W3 false-belief register (a witnessing-gated, agent-keyed belief store that predicts another agent's action
from *their* belief, dissociable from reality and from self) closes from a PARTIAL 4/6 to a genuine **6/6-seed GO**
(local CPU, `SIM_BACKEND=numpy`, seeds 42 43 44 100 101 102). All 7 go-components True on every seed
(false_belief, reality_baseline_fails, true_belief_updates, true_belief_agree, other_lesion_collapses,
scramble_collapses, self_other_dissociation). Mean false_belief_acc **1.000** (chance 0.25).

## The fix (runner-only, additive; NO `sim/` edit)
`DRIVE_STEPS 35→50` (longer witnessing/encoding window) + `helper_pa 3000→5000` (stronger witnessed-event drive) in
`research/runners/_false_belief_register_derisk.py` (+28/-9; both old values remain CLI-selectable to reproduce the
prior PARTIAL). Diagnosis from per-trial store rates: both previously-failing seeds were **ignition-failures** on a
witnessed write (all K locations went near-silent 0.02–0.04 instead of the clean 0.33 self-sustaining attractor), NOT
old-attractor-survival — seed 100 in the WORLD store, seed 43 in the BELIEF store. The two levers are exactly the
"encode a witnessed change more reliably" levers named in the roadmap queue. `attractor_weight=30` (the validated
self-schema geometry) left untouched.

## ⚠️ CORRECTED MECHANISTIC CLAIM (this is what the adversarial verification caught)
The build subagent justified the GO with: *"false_belief_acc is STRUCTURALLY IMMUNE to write-strength because
unwitnessed moves hold the transmission gate closed (zero current to the belief store)."* **That claim is FALSE as
stated and is corrected here.** It is only half-true:
- **TRUE half:** the unwitnessed MOVE (`_write_event(end_loc, witnessed_other=0)`) does deliver ZERO current to the
  belief store — `write_belief=False`, the `witness_other` transmission gate carries 0, and the belief-dev drive is
  skipped. So the belief holds its old attractor A across that window by construction.
- **FALSE half:** every false-belief trial ALSO does an always-witnessed PLACEMENT write (`_write_event(start_loc,
  witnessed=1.0)`) that IGNITES belief=A using the SAME `helper_pa`/`drive_steps` levers, and there is no ungated
  world→belief path. So `false_belief_acc` DOES ride the witnessed-placement write strength. Empirically (skeptic
  sweep, `helper_pa∈{1000,3000,5000,8000} × drive_steps∈{35,50,70}`), `false_belief_acc` degrades to ~0.80 (below the
  0.85 bar) at `helper_pa=1000`.

**Why the GO is nonetheless real (not a gate-cheat):** the collapse is confined to `helper_pa=1000`, which is BELOW
both the OLD (3000/35) and deployed (5000/50) configs — both endpoints sit on the FLAT `false_belief_acc=1.000`
plateau. The metric the fix actually lifts is `true_belief_acc` (the witnessed-MOVE update) + reality-baseline
reliability, which is precisely what closed seeds 43/100. ⇒ the deployed fix is a **witnessed-encoding / world-store
reliability lever, NOT a false-belief gate-cheat** — it does not move the false-belief signal within its operating
range. The corrected framing: `false_belief_acc` is **saturated at the operating point**, not *structurally immune*.

## Anti-cheat controls — all collapse under the fix (skeptic-verified, 6/6 seeds)
- **Reality-baseline FAILS** on false trials (world-read predicts B, wrong): mean **0.021** (≤0.20). PASS.
- **Other-lesion collapses** (sever belief self-loop + force witness open → belief mirrors reality):
  false_belief_acc **0.000** every seed (≪0.45 chance bar); predicts-reality-on-false 1.000. PASS.
- **Scramble-witnessing collapses** (permute witnessed flags): mean **0.451** (≤0.70). PASS.
- **Self/other dissociation holds**: self tracks reality 1.000 while belief is false 1.000, every seed. PASS.

## Seeding integrity (verified)
`build_tom_bridge` sets `cfg.seed = int(seed)` (`:169`), NOT `actual_seed_used`; `bridge.py:1260`
`_initialize_rng(cfg.seed)` runs before the per-neuron firing-threshold draw ⇒ the substrate is genuinely seeded, so
the seed-43/100 misses were real heterogeneity the fix legitimately closes (not the `actual_seed_used` confound).

## Adversarial verification (4-skeptic workflow, `verify-w3-false-belief-go`, CPU)
Independent skeptics each tried to REFUTE from a distinct angle. Reproducibility (6/6 fixed reproduced; OLD genuinely
4/6 → lever load-bearing), control-integrity (all 4 collapse under the fix), and seeding (genuine `cfg.seed`) each
PASS at high confidence + NOT refuted. Gate-cheat REFUTED the *immunity claim* (medium confidence) → corrected above.
Synthesis verdict: **commit-with-caveat** — none of the underlying GO facts are in doubt; the issue was a
mischaracterized justification, now fixed in this record.

## Honest scope (what this is / is not)
A **functional mentalizing correlate** — an agent-keyed belief store predicting X's action from X's witnessing-gated
belief, dissociable from reality and from self. NOT a claim of access to another mind; NOT recursive/2nd-order ToM
(that is W4). Only the **change-of-location** variant is validated; the roadmap's **unexpected-contents** variant (a
semantic relabel of the same witnessing-gated store) is a bounded follow-on for the full roadmap GO-gate. The action
read is an argmax over belief-store spike rates (a host readout, consistent with the nav-readout scaffold precedent);
the cognitive step — the false belief held in the gated NMDA attractor + the self/other dissociation — is fully
neural. This is the 4th pivot-core faculty validated (with P0.3 affect, gap#4 CPU-rate credit, P1.2 workspace).
